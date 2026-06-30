"""entity_builder.py

OpenUSD Entity Builder for the PhysWorldLM framework.

This module is part of the Scene Compiler execution layer. It runs
immediately after ``stage_builder.py`` has created the canonical
``/World`` hierarchy and immediately before ``transform_builder.py``
applies spatial information. Its sole responsibility is translating
validated entities from the Scene Graph (produced by
``scene_compiler.py``) into OpenUSD prims under ``/World/Entities``.

Position in the pipeline::

    Prompt -> World Parser -> Ontology -> WorldSpec -> Scene Compiler
           -> Stage Builder -> Entity Builder -> Transform Builder
           -> USD Exporter -> scene.usda

This module does NOT:
    * apply transforms (translation / rotation / scale / pivot)
    * generate meshes or other renderable geometry
    * assign materials
    * configure physics
    * attach sensors
    * resolve assets
    * export USD files to disk
    * perform ontology reasoning or WorldSpec validation

It only creates entity prims, preserves their hierarchy, and attaches
semantic/identity metadata. Those other responsibilities belong to
other builder modules (``transform_builder.py``, future
``material_builder.py`` / ``physics_builder.py`` / etc.).

Scene Graph contract
---------------------
This module does not redefine ``SceneGraph``, ``SceneNode``, or
``WorldSpec``. It accesses the Scene Graph defensively (duck typing,
the same convention used in ``usd_exporter.py``) against the interface
documented there:

    SceneGraph.entities  -> Iterable[SceneNode]   (top-level entities)
    SceneGraph.root       -> SceneNode | None       (fallback if
                                                       'entities' is absent)
    SceneNode:
        id, name, node_type, category, ontology_id, semantic_label,
        visible, enabled, metadata, children

No entity-type-specific logic exists anywhere in this module; any
entity type defined by the ontology (Aircraft, UAV, Tank, Radar,
Building, Human, Sensor, ...) is handled identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger("physworldlm.entity_builder")


# =============================================================================
# Exceptions
# =============================================================================


class EntityBuilderError(Exception):
    """Base class for all errors raised by the Entity Builder."""


class InvalidEntityError(EntityBuilderError):
    """Raised when a Scene Graph entity is missing or malformed."""


class EntityCreationError(EntityBuilderError):
    """Raised when a USD prim cannot be created for a given entity."""


# =============================================================================
# Build report
# =============================================================================


@dataclass
class EntityBuildResult:
    """Diagnostics describing the outcome of an ``EntityBuilder.build()`` call.

    Attributes:
        entity_count: Total number of entity prims created (including
            nested children).
        max_depth: Maximum nesting depth reached, relative to
            ``/World/Entities`` (a top-level entity has depth 1).
        metadata_count: Number of custom ``physworld:*`` attributes
            written.
        warnings: Non-fatal issues encountered while building.
    """

    entity_count: int = 0
    max_depth: int = 0
    metadata_count: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Scene Graph accessors (duck-typed, mirrors usd_exporter.py conventions)
# =============================================================================


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Fetch ``name`` from ``obj``, supporting attributes or mapping keys."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iter(obj: Any) -> Sequence[Any]:
    """Coerce a possibly-``None`` iterable into a concrete sequence."""
    if obj is None:
        return ()
    return tuple(obj)


def _node_children(node: Any) -> Sequence[Any]:
    return _iter(_get(node, "children"))


def _node_metadata(node: Any) -> Mapping[str, Any]:
    metadata = _get(node, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


# =============================================================================
# Entity Builder
# =============================================================================


class EntityBuilder:
    """Converts Scene Graph entities into OpenUSD prims under ``/World/Entities``.

    The Entity Builder assumes the canonical hierarchy already exists
    (created by ``StageBuilder``); it looks up ``/World/Entities``
    rather than creating it.

    Example:
        >>> builder = EntityBuilder()
        >>> result = builder.build(stage, scene_graph)
        >>> result.entity_count
        3
    """

    _ENTITIES_ROOT = "/World/Entities"

    def __init__(self) -> None:
        self._result = EntityBuildResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, stage: Usd.Stage, scene_graph: Any) -> EntityBuildResult:
        """Build entity prims for every entity in ``scene_graph``.

        Args:
            stage: An existing ``Usd.Stage`` that already contains the
                canonical hierarchy produced by ``StageBuilder``.
            scene_graph: The validated Scene Graph produced by
                ``SceneCompiler.compile(...)``.

        Returns:
            An ``EntityBuildResult`` summarizing what was created.

        Raises:
            InvalidEntityError: If ``stage`` or ``scene_graph`` is
                missing, or ``/World/Entities`` does not exist.
            EntityCreationError: If a prim cannot be created for an
                entity.
        """
        self._result = EntityBuildResult()
        entities_prim = self._resolve_entities_root(stage)
        entities = self._collect_top_level_entities(scene_graph)

        if not entities:
            self._result.warnings.append("Scene graph contains no entities to build.")

        for entity in entities:
            logger.info("Creating entity")
            self._build_entity(stage, entities_prim.GetPath(), entity, depth=1)

        logger.info("Finished entity")
        return self._result

    # ------------------------------------------------------------------
    # Validation / setup
    # ------------------------------------------------------------------

    def _resolve_entities_root(self, stage: Usd.Stage) -> Usd.Prim:
        if stage is None:
            raise InvalidEntityError("EntityBuilder requires an existing Usd.Stage, got None.")

        entities_prim = stage.GetPrimAtPath(self._ENTITIES_ROOT)
        if not entities_prim or not entities_prim.IsValid():
            raise InvalidEntityError(
                f"'{self._ENTITIES_ROOT}' does not exist. Run StageBuilder.build(stage) "
                "before EntityBuilder.build(stage, scene_graph)."
            )
        return entities_prim

    def _collect_top_level_entities(self, scene_graph: Any) -> Sequence[Any]:
        if scene_graph is None:
            raise InvalidEntityError("EntityBuilder requires a scene_graph, got None.")

        entities = _iter(_get(scene_graph, "entities"))
        if entities:
            return entities

        # Fallback: some Scene Graph implementations may only expose a
        # generic 'root' whose children are the top-level entities.
        root = _get(scene_graph, "root")
        return _node_children(root) if root is not None else ()

    def _validate_entity(self, node: Any) -> None:
        if node is None:
            raise InvalidEntityError("Encountered a None entity during traversal.")
        if not _get(node, "name") and not _get(node, "id"):
            raise InvalidEntityError(
                "Entity has neither 'name' nor 'id'; cannot derive a USD prim name."
            )

    # ------------------------------------------------------------------
    # Prim construction
    # ------------------------------------------------------------------

    def _sanitize_name(self, name: str, fallback: str = "Entity") -> str:
        """Produce a valid USD prim name from an arbitrary entity name.

        USD prim names must be valid identifiers: alphanumeric and
        underscore, not starting with a digit.
        """
        if not name:
            name = fallback
        cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
        if not cleaned:
            cleaned = fallback
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned

    def _create_prim(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        """Create a single ``UsdGeom.Xform`` entity prim at ``path``."""
        xform = UsdGeom.Xform.Define(stage, path)
        if not xform:
            raise EntityCreationError(f"Failed to create entity prim at '{path}'.")
        return xform.GetPrim()

    def _build_entity(
        self,
        stage: Usd.Stage,
        parent_path: Sdf.Path,
        node: Any,
        depth: int,
    ) -> Usd.Prim:
        """Create a prim for ``node`` and recursively build its children."""
        self._validate_entity(node)

        type_label = _get(node, "node_type") or _get(node, "category") or "entity"
        logger.info("Creating %s", type_label)

        raw_name = _get(node, "name") or _get(node, "id")
        prim_name = self._sanitize_name(raw_name)
        prim_path = parent_path.AppendChild(prim_name)

        prim = self._create_prim(stage, prim_path)
        self._result.entity_count += 1
        self._result.max_depth = max(self._result.max_depth, depth)

        self._set_visibility(prim, node)
        self._attach_metadata(prim, node)

        self._build_children(stage, prim_path, node, depth=depth + 1)

        return prim

    def _build_children(
        self,
        stage: Usd.Stage,
        parent_path: Sdf.Path,
        node: Any,
        depth: int,
    ) -> None:
        """Recursively build all nested entities of ``node``, preserving hierarchy."""
        for child in _node_children(node):
            self._build_entity(stage, parent_path, child, depth=depth)

    def _set_visibility(self, prim: Usd.Prim, node: Any) -> None:
        visible = _get(node, "visible", True)
        try:
            xform = UsdGeom.Xform(prim)
            if visible:
                xform.MakeVisible()
            else:
                xform.MakeInvisible()
        except Exception as exc:
            self._result.warnings.append(
                f"Could not set visibility for '{prim.GetPath()}': {exc}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _attach_metadata(self, prim: Usd.Prim, node: Any) -> None:
        """Attach identity and semantic metadata as ``physworld:*`` attributes."""
        logger.info("Attaching metadata")

        fields: Dict[str, Any] = {
            "id": _get(node, "id"),
            "type": _get(node, "node_type"),
            "category": _get(node, "category"),
            "ontology": _get(node, "ontology_id"),
            "semantic_label": _get(node, "semantic_label"),
            "classification": _get(node, "classification"),
            "enabled": _get(node, "enabled", True),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        self._write_custom_attributes(prim, fields, namespace="physworld")

        extra_metadata = _node_metadata(node)
        if extra_metadata:
            self._write_custom_attributes(prim, extra_metadata, namespace="physworld:meta")

    def _write_custom_attributes(
        self, prim: Usd.Prim, fields: Mapping[str, Any], namespace: str
    ) -> None:
        for key, value in fields.items():
            attr_name = f"{namespace}:{key}"
            value_type, coerced = self._infer_sdf_type(value)
            try:
                attr = prim.CreateAttribute(attr_name, value_type, custom=True)
                attr.Set(coerced)
                self._result.metadata_count += 1
            except Exception as exc:
                self._result.warnings.append(
                    f"Could not write attribute '{attr_name}' on '{prim.GetPath()}': {exc}"
                )

    @staticmethod
    def _infer_sdf_type(value: Any) -> "tuple[Sdf.ValueTypeName, Any]":
        """Map a Python value to a reasonable Sdf attribute type.

        Falls back to a string representation for unrecognized types so
        metadata export never hard-fails due to an unexpected value
        type coming from the Scene Graph.
        """
        if isinstance(value, bool):
            return Sdf.ValueTypeNames.Bool, value
        if isinstance(value, int):
            return Sdf.ValueTypeNames.Int, value
        if isinstance(value, float):
            return Sdf.ValueTypeNames.Float, value
        return Sdf.ValueTypeNames.String, str(value)


# =============================================================================
# Self-contained smoke test / demo
# =============================================================================
#
# Builds a minimal mock Scene Graph (Vehicle -> Turret -> Cannon, plus a
# Radar) satisfying the duck-typed interface, on top of a stage produced
# by StageBuilder, to sanity-check this module before scene_compiler.py
# is available.


def _build_demo_scene_graph() -> Any:
    @dataclass
    class _Node:
        id: str
        name: str
        node_type: str = "entity"
        category: Optional[str] = None
        ontology_id: Optional[str] = None
        semantic_label: Optional[str] = None
        visible: bool = True
        enabled: bool = True
        metadata: Dict[str, Any] = field(default_factory=dict)
        children: List["_Node"] = field(default_factory=list)

    @dataclass
    class _SceneGraph:
        entities: List[_Node] = field(default_factory=list)

    cannon = _Node(id="cannon-1", name="Cannon", node_type="weapon", category="armament")
    turret = _Node(id="turret-1", name="Turret", node_type="component", children=[cannon])
    vehicle = _Node(
        id="vehicle-1",
        name="Vehicle",
        node_type="tank",
        category="ground_vehicle",
        semantic_label="main_battle_tank",
        children=[turret],
    )
    radar = _Node(
        id="radar-1",
        name="Radar",
        node_type="sensor",
        category="sensor",
        semantic_label="ground_radar",
    )

    return _SceneGraph(entities=[vehicle, radar])


if __name__ == "__main__":
    from stage_builder import StageBuilder  # local import: demo-only dependency

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    demo_stage = Usd.Stage.CreateInMemory()
    StageBuilder().build(demo_stage)

    demo_scene_graph = _build_demo_scene_graph()
    build_result = EntityBuilder().build(demo_stage, demo_scene_graph)

    print(build_result)
    for path in (
        "/World/Entities/Vehicle",
        "/World/Entities/Vehicle/Turret",
        "/World/Entities/Vehicle/Turret/Cannon",
        "/World/Entities/Radar",
    ):
        print(path, "->", demo_stage.GetPrimAtPath(path).IsValid())
