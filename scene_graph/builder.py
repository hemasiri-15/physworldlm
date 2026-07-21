"""Lowers a ``WorldSpec`` into the backend-independent ``SceneGraph`` IR.

``WorldSpecBuilder`` is the single translation stage between the
natural-language-derived, physics-resolved ``WorldSpec`` data contract and
the ``SceneGraph`` IR consumed by downstream compilers (USD exporter, PhysX
exporter). It performs no physics computation of its own beyond unit-space
conversions and geometric derivations that are already implied by the
``WorldSpec`` it is given; every value it cannot read directly off the
``WorldSpec`` is tagged with an explicit :class:`Provenance` so that
confidence-calibration and explainability tooling downstream can tell a
literal value apart from a derived or ontology-assumed one.

Construction proceeds in a fixed, dependency-respecting order:

1. the environment node (a global, at-most-one-per-scene node)
2. per-entity subtrees: entity -> physics_body -> collider -> material
3. cross-entity ``SceneEdge`` relationships, built only once every node
   they might reference already exists

so that no edge is ever created against a node id that hasn't been
inserted yet, and the resulting graph is guaranteed to satisfy every rule
enforced by ``SceneGraphValidator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from physworldlm.scene_graph.attributes import Provenance
from physworldlm.scene_graph.edges import EdgeKind, SceneEdge
from physworldlm.scene_graph.graph import SceneGraph
from physworldlm.scene_graph.nodes import NodeKind, SceneNode
from physworldlm.scene_graph.transform import Transform
from physworldlm.scene_graph.transform import Vec3 as SceneVec3

from physworldlm.world_spec import (
    MATERIAL_DEFAULTS,
    Entity,
    Environment,
    Interaction,
    Vec3 as WorldSpecVec3,
    WorldSpec,
)

_JOINT_TYPE_TO_EDGE_KIND: dict[str, EdgeKind] = {
    "spring": EdgeKind.JOINT_SPRING,
    "hinge": EdgeKind.JOINT_HINGE,
    "fixed": EdgeKind.JOINT_FIXED,
    "distance": EdgeKind.JOINT_DISTANCE,
}

_DEFAULT_SPRING_STIFFNESS_NM: float = 1000.0
_KNOWN_CONFIDENCE: float = 1.0
_ASSUMED_CONFIDENCE: float = 0.6


def _ws_vec_to_scene_vec(vector: WorldSpecVec3) -> SceneVec3:
    """Convert a ``world_spec.Vec3`` into the scene-graph IR's ``Vec3`` type.

    The two types are structurally identical but are deliberately distinct
    classes belonging to different layers (``world_spec`` is the upstream
    data contract, ``scene_graph.transform`` is the backend-independent IR
    primitive), so callers must not assume they are interchangeable.
    """
    return SceneVec3(vector.x, vector.y, vector.z)


@dataclass(slots=True)
class _EntityRecord:
    """Bookkeeping for one lowered entity, used to resolve cross-entity edges."""

    entity: Entity
    entity_node_id: str
    physics_body_node_id: str
    collider_node_id: str
    material_node_id: str


class WorldSpecBuilder:
    """Builds a fully populated, validator-passing ``SceneGraph`` from a ``WorldSpec``.

    Each ``_build_*`` method has a single, independently testable
    responsibility (environment, one entity subtree, interactions, declared
    constraints), and interaction-kind resolution is table-driven via
    ``_JOINT_TYPE_TO_EDGE_KIND`` rather than branching, so new joint or
    interaction kinds can be supported by extending data rather than
    modifying existing control flow.
    """

    def build(self, world_spec: WorldSpec) -> SceneGraph:
        """Lower ``world_spec`` into a new, fully populated ``SceneGraph``."""
        graph = SceneGraph(scene_id=world_spec.scene_id)
        graph.metadata.update(world_spec.metadata)
        graph.metadata["description"] = world_spec.description
        graph.metadata["simulation_graph"] = world_spec.simulation_graph.to_dict()
        graph.metadata["builder_warnings"] = []

        environment_node_id = self._build_environment(graph, world_spec.environment)

        records: dict[str, _EntityRecord] = {}
        for entity in world_spec.entities:
            records[entity.id] = self._build_entity(graph, entity)

        self._build_interactions(
            graph, world_spec.interactions, records, environment_node_id
        )
        self._build_declared_constraints(graph, world_spec.entities, records)

        return graph

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #

    def _build_environment(self, graph: SceneGraph, environment: Environment) -> str:
        node = SceneNode.create(NodeKind.ENVIRONMENT, "environment")
        environment_node_id = graph.add_node(node, parent_id=graph.root_id)

        node.set_attribute(
            "gravity_ms2",
            _ws_vec_to_scene_vec(environment.gravity).to_tuple(),
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.gravity",
        )
        node.set_attribute(
            "temperature_K",
            environment.temperature_K,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.temperature_K",
        )
        node.set_attribute(
            "pressure_Pa",
            environment.pressure_Pa,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.pressure_Pa",
        )
        node.set_attribute(
            "air_density_kgm3",
            environment.air_density,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.air_density",
        )
        node.set_attribute(
            "wind_speed_ms",
            environment.wind.speed,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.wind.speed",
        )
        node.set_attribute(
            "wind_direction_rad",
            environment.wind.direction,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.wind.direction",
        )
        node.set_attribute(
            "terrain_type",
            environment.terrain_type,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.terrain_type",
        )
        node.set_attribute(
            "friction_global",
            environment.friction_global,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.friction_global",
        )
        node.set_attribute(
            "time_of_day",
            environment.time_of_day,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.time_of_day",
        )
        node.set_attribute(
            "weather",
            environment.weather,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.environment.weather",
        )
        return environment_node_id

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #

    def _build_entity(self, graph: SceneGraph, entity: Entity) -> _EntityRecord:
        entity_transform = Transform.from_euler(
            translation=_ws_vec_to_scene_vec(entity.state.position),
            euler_radians=_ws_vec_to_scene_vec(entity.state.orientation),
        )
        entity_node = SceneNode.create(
            NodeKind.ENTITY,
            entity.label,
            local_transform=entity_transform,
            tags=list(entity.tags),
        )
        entity_node_id = graph.add_node(entity_node, parent_id=graph.root_id)

        entity_node.set_attribute(
            "source_entity_id",
            entity.id,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.id",
        )
        entity_node.set_attribute(
            "entity_type",
            entity.entity_type,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.entity_type",
        )

        physics_body_node_id = self._build_physics_body(graph, entity, entity_node_id)
        collider_node_id = self._build_collider(graph, entity, physics_body_node_id)
        material_node_id = self._build_material(graph, entity, collider_node_id)

        return _EntityRecord(
            entity=entity,
            entity_node_id=entity_node_id,
            physics_body_node_id=physics_body_node_id,
            collider_node_id=collider_node_id,
            material_node_id=material_node_id,
        )

    def _build_physics_body(
        self, graph: SceneGraph, entity: Entity, entity_node_id: str
    ) -> str:
        body_node = SceneNode.create(
            NodeKind.PHYSICS_BODY,
            f"{entity.label}_body",
            local_transform=Transform.identity(),
        )
        body_node_id = graph.add_node(body_node, parent_id=entity_node_id)

        body_node.set_attribute(
            "mass_kg",
            entity.mass,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.mass",
        )
        body_node.set_attribute(
            "is_static",
            entity.is_static,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.is_static",
        )
        body_node.set_attribute(
            "velocity_ms",
            _ws_vec_to_scene_vec(entity.state.velocity).to_tuple(),
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.state.velocity",
        )
        body_node.set_attribute(
            "acceleration_ms2",
            _ws_vec_to_scene_vec(entity.state.acceleration).to_tuple(),
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.state.acceleration",
        )
        body_node.set_attribute(
            "angular_velocity_rads",
            _ws_vec_to_scene_vec(entity.state.angular_vel).to_tuple(),
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.state.angular_vel",
        )
        if entity.forces:
            body_node.set_attribute(
                "applied_forces",
                list(entity.forces),
                provenance=Provenance.KNOWN,
                confidence=_KNOWN_CONFIDENCE,
                source="world_spec.entity.forces",
            )
        return body_node_id

    def _build_collider(
        self, graph: SceneGraph, entity: Entity, physics_body_node_id: str
    ) -> str:
        collider_node = SceneNode.create(
            NodeKind.COLLIDER,
            f"{entity.label}_collider",
            local_transform=Transform.identity(),
        )
        collider_node_id = graph.add_node(collider_node, parent_id=physics_body_node_id)

        collider_node.set_attribute(
            "shape",
            "box",
            provenance=Provenance.ASSUMED,
            confidence=_ASSUMED_CONFIDENCE,
            source="world_spec only supplies an axis-aligned bounding box",
        )
        half_extents = (
            entity.bounding_box.width / 2.0,
            entity.bounding_box.height / 2.0,
            entity.bounding_box.depth / 2.0,
        )
        collider_node.set_attribute(
            "half_extents_m",
            half_extents,
            provenance=Provenance.DERIVED,
            confidence=_KNOWN_CONFIDENCE,
            source="derived from world_spec.entity.bounding_box",
        )
        return collider_node_id

    def _build_material(
        self, graph: SceneGraph, entity: Entity, collider_node_id: str
    ) -> str:
        material_node = SceneNode.create(
            NodeKind.MATERIAL,
            f"{entity.label}_material_{entity.material}",
            local_transform=Transform.identity(),
        )
        material_node_id = graph.add_node(material_node, parent_id=collider_node_id)

        defaults = MATERIAL_DEFAULTS.get(entity.material, MATERIAL_DEFAULTS["generic"])
        friction_is_default = entity.friction == defaults["friction"]
        restitution_is_default = entity.restitution == defaults["restitution"]

        material_node.set_attribute(
            "friction",
            entity.friction,
            provenance=Provenance.ASSUMED if friction_is_default else Provenance.KNOWN,
            confidence=_ASSUMED_CONFIDENCE if friction_is_default else _KNOWN_CONFIDENCE,
            source=(
                f"MATERIAL_DEFAULTS[{entity.material!r}].friction"
                if friction_is_default
                else "world_spec.entity.friction"
            ),
        )
        material_node.set_attribute(
            "restitution",
            entity.restitution,
            provenance=Provenance.ASSUMED if restitution_is_default else Provenance.KNOWN,
            confidence=_ASSUMED_CONFIDENCE if restitution_is_default else _KNOWN_CONFIDENCE,
            source=(
                f"MATERIAL_DEFAULTS[{entity.material!r}].restitution"
                if restitution_is_default
                else "world_spec.entity.restitution"
            ),
        )
        material_node.set_attribute(
            "density_kgm3",
            defaults["density"],
            provenance=Provenance.ASSUMED,
            confidence=_ASSUMED_CONFIDENCE,
            source=f"MATERIAL_DEFAULTS[{entity.material!r}].density",
        )
        material_node.set_attribute(
            "material_name",
            entity.material,
            provenance=Provenance.KNOWN,
            confidence=_KNOWN_CONFIDENCE,
            source="world_spec.entity.material",
        )
        return material_node_id

    # ------------------------------------------------------------------ #
    # Cross-entity edges
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_endpoint(
        entity_id: str,
        records: dict[str, _EntityRecord],
        environment_node_id: str,
    ) -> Optional[str]:
        """Resolve a ``WorldSpec``-level entity id (or ``"environment"``) to a node id."""
        if entity_id == "environment":
            return environment_node_id
        record = records.get(entity_id)
        return record.physics_body_node_id if record is not None else None

    def _resolve_interaction_edge_kind(
        self,
        interaction: Interaction,
        source_id: str,
        target_id: str,
        environment_node_id: str,
    ) -> tuple[EdgeKind, bool]:
        """Map a ``WorldSpec`` interaction type onto an ``EdgeKind``.

        Returns the resolved kind plus whether it was downgraded from a
        body-bearing joint kind because one endpoint is the environment
        node (``SceneGraphValidator`` requires body-bearing joint endpoints
        to be ``PHYSICS_BODY``/``ENTITY`` nodes, which the environment node
        is not).
        """
        if interaction.type == "collision":
            return EdgeKind.COLLISION, False
        if interaction.type == "contact":
            return EdgeKind.CONTACT, False
        if interaction.type == "joint":
            joint_type = str(interaction.parameters.get("joint_type", "fixed"))
            edge_kind = _JOINT_TYPE_TO_EDGE_KIND.get(joint_type, EdgeKind.JOINT_FIXED)
            touches_environment = environment_node_id in (source_id, target_id)
            if touches_environment:
                return EdgeKind.CONTACT, True
            return edge_kind, False
        # fluid_drag, magnetic, or any other interaction type with no direct
        # EdgeKind counterpart: preserved as a descriptive relation so the
        # information is never silently dropped.
        return EdgeKind.SPATIAL_RELATION, False

    def _build_interactions(
        self,
        graph: SceneGraph,
        interactions: list[Interaction],
        records: dict[str, _EntityRecord],
        environment_node_id: str,
    ) -> None:
        for interaction in interactions:
            source_id = self._resolve_endpoint(
                interaction.entity_a, records, environment_node_id
            )
            target_id = self._resolve_endpoint(
                interaction.entity_b, records, environment_node_id
            )
            if source_id is None or target_id is None:
                graph.metadata["builder_warnings"].append(
                    f"interaction {interaction.type!r} references unresolved "
                    f"entity_a={interaction.entity_a!r} / "
                    f"entity_b={interaction.entity_b!r}; skipped"
                )
                continue
            if source_id == target_id:
                graph.metadata["builder_warnings"].append(
                    f"interaction {interaction.type!r} has identical source "
                    f"and target {source_id!r}; skipped"
                )
                continue

            edge_kind, downgraded = self._resolve_interaction_edge_kind(
                interaction, source_id, target_id, environment_node_id
            )
            edge = SceneEdge.create(edge_kind, source_id, target_id)

            edge.set_attribute(
                "interaction_type",
                interaction.type,
                provenance=Provenance.KNOWN,
                confidence=_KNOWN_CONFIDENCE,
                source="world_spec.interaction.type",
            )
            if interaction.parameters:
                edge.set_attribute(
                    "parameters",
                    dict(interaction.parameters),
                    provenance=Provenance.KNOWN,
                    confidence=_KNOWN_CONFIDENCE,
                    source="world_spec.interaction.parameters",
                )
            if downgraded:
                edge.set_attribute(
                    "downgrade_reason",
                    "body-bearing joint kind requires two non-environment endpoints",
                    provenance=Provenance.DERIVED,
                    confidence=_KNOWN_CONFIDENCE,
                    source="builder",
                )

            if edge_kind is EdgeKind.JOINT_SPRING:
                self._populate_spring_attributes(graph, edge, interaction, source_id, target_id)

            graph.add_edge(edge)

    def _populate_spring_attributes(
        self,
        graph: SceneGraph,
        edge: SceneEdge,
        interaction: Interaction,
        source_id: str,
        target_id: str,
    ) -> None:
        """Ensure a ``JOINT_SPRING`` edge carries the attributes ``SceneGraphValidator`` requires."""
        stiffness = interaction.parameters.get(
            "k_Nm", interaction.parameters.get("stiffness")
        )
        if stiffness is None:
            edge.set_attribute(
                "k_Nm",
                _DEFAULT_SPRING_STIFFNESS_NM,
                provenance=Provenance.ASSUMED,
                confidence=_ASSUMED_CONFIDENCE,
                source="builder default spring stiffness",
            )
        else:
            edge.set_attribute(
                "k_Nm",
                float(stiffness),
                provenance=Provenance.KNOWN,
                confidence=_KNOWN_CONFIDENCE,
                source="world_spec.interaction.parameters",
            )

        rest_length = interaction.parameters.get("rest_length_m")
        if rest_length is not None:
            edge.set_attribute(
                "rest_length_m",
                float(rest_length),
                provenance=Provenance.KNOWN,
                confidence=_KNOWN_CONFIDENCE,
                source="world_spec.interaction.parameters",
            )
        else:
            edge.set_attribute(
                "rest_length_m",
                self._derive_rest_length(graph, source_id, target_id),
                provenance=Provenance.DERIVED,
                confidence=_KNOWN_CONFIDENCE,
                source="derived from world-space distance between joint endpoints",
            )

    @staticmethod
    def _derive_rest_length(graph: SceneGraph, source_id: str, target_id: str) -> float:
        """Derive a spring's rest length from its endpoints' current world-space distance."""
        source_world_position = graph.world_transform(source_id).translation
        target_world_position = graph.world_transform(target_id).translation
        return (target_world_position - source_world_position).magnitude()

    def _build_declared_constraints(
        self,
        graph: SceneGraph,
        entities: list[Entity],
        records: dict[str, _EntityRecord],
    ) -> None:
        """Materialize ``Entity.constraints`` id references as ``CONTACT`` edges.

        ``Entity.constraints`` is a flat list of other entity ids with no
        further typing information, so each declared pair is represented as
        a generic, non-body-bearing ``CONTACT`` edge; duplicate and
        self-referential pairs are skipped.
        """
        seen_pairs: set[frozenset[str]] = set()
        for entity in entities:
            record = records.get(entity.id)
            if record is None:
                continue
            for other_entity_id in entity.constraints:
                other_record = records.get(other_entity_id)
                if other_record is None:
                    graph.metadata["builder_warnings"].append(
                        f"entity {entity.id!r} declares a constraint to unknown "
                        f"entity {other_entity_id!r}; skipped"
                    )
                    continue
                if other_record.physics_body_node_id == record.physics_body_node_id:
                    continue
                pair_key = frozenset(
                    (record.physics_body_node_id, other_record.physics_body_node_id)
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                edge = SceneEdge.create(
                    EdgeKind.CONTACT,
                    record.physics_body_node_id,
                    other_record.physics_body_node_id,
                )
                edge.set_attribute(
                    "declared_via",
                    "world_spec.entity.constraints",
                    provenance=Provenance.KNOWN,
                    confidence=_KNOWN_CONFIDENCE,
                    source="world_spec.entity.constraints",
                )
                graph.add_edge(edge)


def build_scene_graph(world_spec: WorldSpec) -> SceneGraph:
    """Convenience function: lower ``world_spec`` into a new ``SceneGraph``."""
    return WorldSpecBuilder().build(world_spec)
