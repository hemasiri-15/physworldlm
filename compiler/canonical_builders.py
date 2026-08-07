"""
compiler/canonical_builders.py
────────────────────────────────
WorldSpec -> physworldlm.scene_graph.SceneGraph, using the CANONICAL IR.

This replaces scene_compiler.py's inline SceneNode/SceneGraph/Transform/
NodeType definitions and builder logic. It is deliberately scoped to the
node/edge construction only -- NOT a replacement for scene_compiler.py's
CompilationStage/BuilderRegistry/Exporter orchestration machinery, which
is a separate, larger follow-up (see module docstring bottom).

Two open decisions this module deliberately does NOT make unilaterally
(flagged, not resolved, per "never invent functionality"):

1. NodeKind has no members for Terrain/Atmosphere/Weather/Lighting.
   This module represents them as NodeKind.XFORM children of the
   Environment node, distinguished by `tags=["terrain"]` etc. This is a
   safe, non-invasive default (it works today, and migrating to real
   NodeKind members later is a pure win with no data loss -- the tag is
   still there to search on). If NodeKind is extended later, only
   `EnvironmentBuilder.build()` below needs to change.

2. EdgeKind has no members for "magnetic", "gravity", "fluid_drag", or
   generic "constraint" (all valid WorldSpec.Interaction.type values per
   scene_compiler.py's own `_t2_physics_interactions` valid_types set).
   RelationshipBuilder below maps every type it CAN map correctly and
   explicitly SKIPS the rest with a diagnostic, rather than guessing an
   EdgeKind for them. See `_EDGE_KIND_DISPATCH` / `_unmapped_interaction`.

Bug fixes applied here (verified against the originals -- see chat history
for the reproduction of bug #1):
  1. Material mutation: MATERIAL nodes carry ONLY the immutable catalog
     defaults as Attributes with Provenance.ASSUMED. Per-entity overrides
     are attached exclusively to each entity's own PHYSICS_BODY node
     (which was already correct in PhysicsBuilder) -- the shared MATERIAL
     node is never entity-conditional.
  2. Duplicate entity IDs: EntityBuilder tracks resolved ids and skips
     (with a diagnostic) any entity whose id has already produced a node,
     rather than creating a second node with a colliding deterministic id.
  3. Asset resolution: reused verbatim from worldspec_builder.py's fixed
     `_default_asset_resolver` (the version that actually returns None
     for a real not-found reference, unlike scene_compiler.py's version).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from physworldlm.scene_graph.attributes import Provenance
from physworldlm.scene_graph.edges import EdgeKind, SceneEdge
from physworldlm.scene_graph.graph import SceneGraph
from physworldlm.scene_graph.nodes import NodeKind, SceneNode
from physworldlm.scene_graph.transform import Quaternion, Transform, Vec3 as GraphVec3

from world_spec import Entity, Environment, Interaction, MATERIAL_DEFAULTS, WorldSpec


# ─────────────────────────────────────────────
# Diagnostics + statistics
# ─────────────────────────────────────────────

class BuildDiagnostics:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.statistics: dict[str, int] = {
            "entities": 0,
            "materials": 0,
            "relationships": 0,
            "resolved_assets": 0,
            "duplicate_entities_skipped": 0,
            "unmapped_interactions_skipped": 0,
        }

    def warn(self, message: str, *, stat_key: Optional[str] = None) -> None:
        self.warnings.append(message)
        if stat_key is not None:
            self.statistics[stat_key] = self.statistics.get(stat_key, 0) + 1

    def note(self, message: str) -> None:
        self.info.append(message)


# ─────────────────────────────────────────────
# CompilerReport — LLVM-style structured build result
# ─────────────────────────────────────────────

class CompilerReport:
    """Structured result of build_scene_graph(): what got built, what
    went wrong, and how long it took. Replaces bare (graph, diagnostics)
    tuple returns with one typed object, matching the report pattern
    already used by ValidationCompilerPass/WorldCompiler upstream.
    """

    def __init__(self, scene_graph: SceneGraph, diagnostics: BuildDiagnostics, time_ms: float) -> None:
        self.scene_graph = scene_graph
        self.diagnostics = diagnostics
        self.time_ms = time_ms

    @property
    def success(self) -> bool:
        # Nothing in this module raises for recoverable issues today (all
        # failure modes here are warn-and-skip); success is currently
        # always True if build_scene_graph() returned at all. Kept as an
        # explicit property, not a hardcoded True, so a future STRICT mode
        # (raise instead of skip) has a real place to report failure.
        return True

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "scene_id": self.scene_graph.scene_id,
            "time_ms": round(self.time_ms, 3),
            "node_count": self.scene_graph.node_count(),
            "edge_count": self.scene_graph.edge_count(),
            "statistics": dict(self.diagnostics.statistics),
            "warnings": list(self.diagnostics.warnings),
            "info": list(self.diagnostics.info),
        }

    def __str__(self) -> str:
        s = self.diagnostics.statistics
        return (
            f"CompilerReport(scene_id={self.scene_graph.scene_id!r})\n"
            f"  entities={s['entities']}  materials={s['materials']}  "
            f"relationships={s['relationships']}\n"
            f"  warnings={len(self.diagnostics.warnings)}  time_ms={self.time_ms:.2f}"
        )


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class EnvironmentBuilder:
    """Builds the Environment node and its Terrain/Atmosphere/Weather/
    Lighting children as NodeKind.XFORM nodes distinguished by tag (see
    module docstring, open decision #1).
    """

    def build(self, graph: SceneGraph, env: Environment) -> str:
        env_node = SceneNode.create(NodeKind.ENVIRONMENT, "Environment")
        env_node.set_attribute("terrain_type", env.terrain_type, provenance=Provenance.KNOWN, source="world_spec.environment")
        env_node.set_attribute("friction_global", env.friction_global, provenance=Provenance.KNOWN)
        env_node.set_attribute("temperature_K", env.temperature_K, provenance=Provenance.KNOWN)
        env_node.set_attribute("pressure_Pa", env.pressure_Pa, provenance=Provenance.KNOWN)
        env_node.set_attribute("air_density", env.air_density, provenance=Provenance.KNOWN)
        env_node.set_attribute("weather", env.weather, provenance=Provenance.KNOWN)
        env_node.set_attribute("wind_speed_ms", env.wind.speed, provenance=Provenance.KNOWN)
        env_node.set_attribute("wind_direction_rad", env.wind.direction, provenance=Provenance.KNOWN)
        env_node.set_attribute("time_of_day", env.time_of_day, provenance=Provenance.KNOWN)
        graph.add_node(env_node)

        for sub_name, tag in (
            ("Terrain", "terrain"),
            ("Atmosphere", "atmosphere"),
            ("Weather", "weather"),
            ("Lighting", "lighting"),
        ):
            sub_node = SceneNode.create(NodeKind.XFORM, sub_name, tags=[tag])
            graph.add_node(sub_node, parent_id=env_node.id)

        return env_node.id


# ─────────────────────────────────────────────
# Entities
# ─────────────────────────────────────────────

class EntityBuilder:
    """Builds one ENTITY node per WorldSpec entity. Bug fix #2: rejects
    (with a diagnostic) any entity id that already produced a node,
    instead of silently creating a second node with a colliding id.
    """

    def build(
        self, graph: SceneGraph, entities: list[Entity], diagnostics: BuildDiagnostics
    ) -> dict[str, str]:
        """Returns {world_spec_entity_id: scene_graph_node_id}."""
        entity_id_to_node_id: dict[str, str] = {}

        for entity in entities:
            if entity.id in entity_id_to_node_id:
                diagnostics.warn(
                    f"Duplicate entity id '{entity.id}' encountered a second time; "
                    "skipping the duplicate rather than creating a second SceneNode.",
                    stat_key="duplicate_entities_skipped",
                )
                continue

            node = SceneNode.create(NodeKind.ENTITY, entity.label or entity.id, tags=list(entity.tags))
            node.local_transform = self._transform_from_state(entity)
            node.set_attribute("world_spec_id", entity.id, provenance=Provenance.KNOWN)
            node.set_attribute("entity_type", entity.entity_type, provenance=Provenance.KNOWN)
            node.set_attribute("is_static", entity.is_static, provenance=Provenance.KNOWN)

            graph.add_node(node)
            entity_id_to_node_id[entity.id] = node.id
            diagnostics.statistics["entities"] += 1

        return entity_id_to_node_id

    @staticmethod
    def _transform_from_state(entity: Entity) -> Transform:
        pos = entity.state.position
        rot = entity.state.orientation  # Euler radians, per world_spec.py
        bbox = entity.bounding_box
        return Transform(
            translation=GraphVec3(pos.x, pos.y, pos.z),
            rotation=Quaternion.from_euler_xyz(GraphVec3(rot.x, rot.y, rot.z)),
            scale=GraphVec3(bbox.width, bbox.height, bbox.depth),
        )


# ─────────────────────────────────────────────
# Materials — bug fix #1
# ─────────────────────────────────────────────

class MaterialLibrary:
    """Plain, builder-independent lookup of canonical material defaults.

    Extracted from MaterialBuilder so future exporters (PhysX, Bullet,
    MuJoCo) can look up a material's physical defaults without running
    any SceneGraph node-construction logic — just the catalog itself.
    """

    def __init__(self, defaults: dict[str, dict[str, float]] = MATERIAL_DEFAULTS) -> None:
        self._defaults = defaults

    def resolve_name(self, material_name: str) -> str:
        """Return `material_name` if known, else the 'generic' fallback."""
        return material_name if material_name in self._defaults else "generic"

    def get(self, material_name: str) -> dict[str, float]:
        return self._defaults[self.resolve_name(material_name)]


class MaterialBuilder:
    """Builds one, immutable MATERIAL node per unique material name.

    BUG FIX (verified against scene_compiler.py's original, reproduced
    with a script earlier in this review): the material node carries
    ONLY the catalog defaults (Provenance.ASSUMED — they come from the
    static MATERIAL_DEFAULTS table, not any specific entity). It is never
    conditionally overwritten with a particular entity's restitution/
    friction, so processing order can no longer change what a shared
    material means for entities that didn't request an override.
    Per-entity overrides live exclusively on each entity's own
    PHYSICS_BODY node — see PhysicsBuilder below, which was already
    correct.
    """

    def __init__(self, library: Optional[MaterialLibrary] = None) -> None:
        self._library = library or MaterialLibrary()

    def build(
        self,
        graph: SceneGraph,
        entities: list[Entity],
        entity_id_to_node_id: dict[str, str],
        diagnostics: BuildDiagnostics,
    ) -> dict[str, str]:
        """Returns {material_name: scene_graph_node_id}."""
        material_name_to_node_id: dict[str, str] = {}

        for entity in entities:
            mat_name = self._library.resolve_name(entity.material)

            if mat_name not in material_name_to_node_id:
                defaults = self._library.get(mat_name)
                mat_node = SceneNode.create(NodeKind.MATERIAL, mat_name)
                mat_node.set_attribute("density", defaults["density"], provenance=Provenance.ASSUMED, source="MaterialLibrary")
                mat_node.set_attribute("restitution", defaults["restitution"], provenance=Provenance.ASSUMED, source="MaterialLibrary")
                mat_node.set_attribute("friction", defaults["friction"], provenance=Provenance.ASSUMED, source="MaterialLibrary")
                graph.add_node(mat_node)
                material_name_to_node_id[mat_name] = mat_node.id
                diagnostics.statistics["materials"] += 1

            entity_node_id = entity_id_to_node_id.get(entity.id)
            if entity_node_id is not None:
                graph.add_edge(
                    SceneEdge.create(EdgeKind.SPATIAL_RELATION, entity_node_id, material_name_to_node_id[mat_name])
                )

        return material_name_to_node_id


# ─────────────────────────────────────────────
# Physics bodies — per-entity overrides live here (unchanged design,
# this part was already correct in the original)
# ─────────────────────────────────────────────

class PhysicsBuilder:
    def build(self, graph: SceneGraph, entities: list[Entity], entity_id_to_node_id: dict[str, str]) -> None:
        for entity in entities:
            entity_node_id = entity_id_to_node_id.get(entity.id)
            if entity_node_id is None:
                continue

            body_node = SceneNode.create(NodeKind.PHYSICS_BODY, f"{entity.label or entity.id}_physics")
            body_node.set_attribute("body_type", "static" if entity.is_static else "dynamic", provenance=Provenance.KNOWN)
            body_node.set_attribute("mass_kg", entity.mass, provenance=Provenance.KNOWN, source="world_spec.entity.mass")
            # Per-entity overrides: this is exactly what bug #1 was
            # incorrectly ALSO writing onto the shared material node.
            # Here — on the entity's own body — it's correct.
            body_node.set_attribute("restitution", entity.restitution, provenance=Provenance.KNOWN, source="per-entity override")
            body_node.set_attribute("friction", entity.friction, provenance=Provenance.KNOWN, source="per-entity override")
            body_node.set_attribute("forces", entity.forces, provenance=Provenance.KNOWN)
            body_node.set_attribute("constraints", entity.constraints, provenance=Provenance.KNOWN)

            graph.add_node(body_node, parent_id=entity_node_id)


# ─────────────────────────────────────────────
# Relationships — open decision #2: unmappable interaction types are
# skipped with a diagnostic, never force-mapped to the wrong EdgeKind.
# ─────────────────────────────────────────────

def _dispatch_edge_kind(interaction: Interaction) -> Optional[EdgeKind]:
    """Return the correct EdgeKind for `interaction`, or None if there is
    no safe mapping today (see module docstring, open decision #2).
    """
    t = interaction.type
    if t == "contact":
        return EdgeKind.CONTACT
    if t == "friction":
        return EdgeKind.FRICTION
    if t == "collision":
        return EdgeKind.COLLISION
    if t == "spring":
        # PromptParser emits "spring" directly (see the newly-found
        # cross-module mismatch: scene_compiler.py's own valid_types set
        # doesn't include "spring" today — separate bug, noted in review).
        return EdgeKind.JOINT_SPRING
    if t == "joint":
        # Generic "joint" needs a sub-type to pick a specific EdgeKind;
        # dispatch on which parameters are present rather than guessing.
        params = interaction.parameters or {}
        if "k_Nm" in params or "rest_length_m" in params:
            return EdgeKind.JOINT_SPRING
        if "hinge_axis" in params:
            return EdgeKind.JOINT_HINGE
        if "distance_m" in params:
            return EdgeKind.JOINT_DISTANCE
        return EdgeKind.JOINT_FIXED  # generic joint with no distinguishing params
    # "magnetic", "gravity", "fluid_drag", "constraint" -- no safe
    # EdgeKind mapping exists today. Explicitly unmapped, not guessed.
    return None


class RelationshipBuilder:
    def build(
        self,
        graph: SceneGraph,
        interactions: list[Interaction],
        entity_id_to_node_id: dict[str, str],
        env_node_id: str,
        diagnostics: BuildDiagnostics,
    ) -> int:
        built = 0
        for interaction in interactions:
            source_node_id = entity_id_to_node_id.get(interaction.entity_a)
            if source_node_id is None:
                diagnostics.warn(f"Interaction references unknown entity_a '{interaction.entity_a}'.")
                continue

            if interaction.entity_b == "environment":
                target_node_id = env_node_id
            else:
                target_node_id = entity_id_to_node_id.get(interaction.entity_b)
                if target_node_id is None:
                    diagnostics.warn(f"Interaction references unknown entity_b '{interaction.entity_b}'.")
                    continue

            edge_kind = _dispatch_edge_kind(interaction)
            if edge_kind is None:
                diagnostics.warn(
                    f"Interaction type '{interaction.type}' has no EdgeKind mapping today "
                    f"(entity_a='{interaction.entity_a}', entity_b='{interaction.entity_b}'); "
                    "skipped rather than force-mapped. See open decision #2.",
                    stat_key="unmapped_interactions_skipped",
                )
                continue

            edge = SceneEdge.create(edge_kind, source_node_id, target_node_id)
            for key, value in (interaction.parameters or {}).items():
                edge.set_attribute(key, value, provenance=Provenance.KNOWN, source="world_spec.interaction.parameters")
            graph.add_edge(edge)
            built += 1
            diagnostics.statistics["relationships"] += 1

        return built


# ─────────────────────────────────────────────
# Asset resolution — reused fix from worldspec_builder.py, plus a cache
# ─────────────────────────────────────────────

_ASSET_TAG_PREFIX = "asset:"
_REMOTE_ASSET_MARKER = "://"


def default_asset_resolver(ref: str, search_paths: list[Path]) -> Optional[Path]:
    """Verbatim logic from worldspec_builder.py's fixed resolver (bug #3):
    a real not-found local reference returns None instead of always
    returning a Path.
    """
    if _REMOTE_ASSET_MARKER in ref:
        return Path(ref)
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for search_path in search_paths:
        full = search_path / ref
        if full.exists():
            return full
    return candidate if (not search_paths and candidate.exists()) else None


class AssetCache:
    """Caches resolved asset references by (ref, search_paths) so a
    reference shared across many entities — or repeated across benchmark
    iterations — isn't re-probed against the filesystem every time.
    Same rationale as OntologyResolver's LRU cache for repeated entity
    labels.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[Path, ...]], Optional[Path]] = {}

    def resolve(self, ref: str, search_paths: list[Path]) -> Optional[Path]:
        key = (ref, tuple(search_paths))
        if key not in self._cache:
            self._cache[key] = default_asset_resolver(ref, search_paths)
        return self._cache[key]


class AssetBuilder:
    def __init__(self, cache: Optional[AssetCache] = None) -> None:
        self._cache = cache or AssetCache()

    def build(
        self,
        entities: list[Entity],
        entity_id_to_node_id: dict[str, str],
        graph: SceneGraph,
        search_paths: list[Path],
        diagnostics: BuildDiagnostics,
    ) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for entity in entities:
            node_id = entity_id_to_node_id.get(entity.id)
            if node_id is None:
                continue
            node = graph.get_node(node_id)
            for tag in entity.tags:
                if not tag.startswith(_ASSET_TAG_PREFIX):
                    continue
                ref = tag[len(_ASSET_TAG_PREFIX):]
                path = self._cache.resolve(ref, search_paths)
                if path is None:
                    diagnostics.warn(f"Could not resolve asset reference '{ref}' for entity '{entity.id}'.")
                    continue
                resolved[ref] = path
                node.set_attribute(f"asset:{ref}", str(path), provenance=Provenance.KNOWN)
                diagnostics.statistics["resolved_assets"] += 1
        return resolved


# ─────────────────────────────────────────────
# Top-level orchestration (bounded scope — see module docstring)
# ─────────────────────────────────────────────

def build_scene_graph(
    world_spec: WorldSpec, asset_search_paths: Optional[list[Path]] = None
) -> CompilerReport:
    """Lower a WorldSpec into the canonical SceneGraph IR.

    This function is intentionally NOT wired into scene_compiler.py's
    CompilationStage/BuilderRegistry machinery yet — that's the larger
    follow-up (rewriting SceneCompiler.compile() to call this instead of
    its inline builders, and updating USDAsciiExporter to walk
    SceneNode.local_transform.to_matrix4() instead of the old Transform
    struct). This function is the verified-correct core that rewrite
    would call.
    """
    t0 = time.monotonic()
    diagnostics = BuildDiagnostics()
    graph = SceneGraph(scene_id=world_spec.scene_id)
    graph.metadata["description"] = world_spec.description

    env_node_id = EnvironmentBuilder().build(graph, world_spec.environment)

    entity_id_to_node_id = EntityBuilder().build(graph, world_spec.entities, diagnostics)

    # Bug found via testing: MaterialBuilder/PhysicsBuilder/AssetBuilder
    # were still iterating the RAW world_spec.entities list and resolving
    # each entity's id through entity_id_to_node_id — which, for a
    # duplicate id, resolves to the FIRST occurrence's node. That let a
    # duplicate entity's mass/restitution/material data silently attach
    # to the wrong (first) entity's physics body, even though
    # EntityBuilder correctly skipped creating a second SceneNode for it.
    # Fix: every downstream builder operates on the SAME deduplicated
    # entity list — first-occurrence-by-id, matching entity_id_to_node_id
    # exactly — so a skipped duplicate's data can never leak in through
    # any builder, not just EntityBuilder.
    seen_ids: set[str] = set()
    deduplicated_entities: list[Entity] = []
    for entity in world_spec.entities:
        if entity.id in seen_ids:
            continue
        seen_ids.add(entity.id)
        deduplicated_entities.append(entity)

    MaterialBuilder().build(graph, deduplicated_entities, entity_id_to_node_id, diagnostics)
    PhysicsBuilder().build(graph, deduplicated_entities, entity_id_to_node_id)

    AssetBuilder().build(
        deduplicated_entities, entity_id_to_node_id, graph, asset_search_paths or [], diagnostics
    )

    RelationshipBuilder().build(
        graph, world_spec.interactions, entity_id_to_node_id, env_node_id, diagnostics
    )

    time_ms = (time.monotonic() - t0) * 1000.0
    return CompilerReport(scene_graph=graph, diagnostics=diagnostics, time_ms=time_ms)
