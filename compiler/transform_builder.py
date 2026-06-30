"""transform_builder.py

OpenUSD Transform Builder for the PhysWorldLM framework.

This module is part of the Scene Compiler execution layer. It runs
immediately after ``entity_builder.py`` has created the entity prim
hierarchy under ``/World/Entities`` and immediately before
``usd_exporter.py`` writes the stage to disk. Its sole responsibility
is applying spatial transforms (translation, rotation, scale, pivot)
to the prims that ``EntityBuilder`` already created.

Position in the pipeline::

    Prompt -> World Parser -> Ontology -> WorldSpec -> Scene Compiler
           -> Stage Builder -> Entity Builder -> Transform Builder
           -> USD Exporter -> scene.usda

This module does NOT:
    * create entities or modify scene hierarchy
    * create meshes or other renderable geometry
    * assign materials
    * configure physics
    * attach sensors
    * resolve assets
    * export USD files to disk
    * validate the WorldSpec

It only locates already-existing entity prims and sets their
translation, rotation, and scale.

Scene Graph contract
---------------------
This module does not redefine ``SceneGraph``, ``SceneNode``, or
``WorldSpec``. It accesses the Scene Graph defensively (duck typing,
the same convention used in ``usd_exporter.py`` and
``entity_builder.py``):

    SceneGraph.entities  -> Iterable[SceneNode]   (top-level entities)
    SceneGraph.root       -> SceneNode | None       (fallback)
    SceneNode:
        id, name, children, transform (or position/translation/
        rotation/quaternion/scale/pivot directly on the node)

To locate the USD prim for a given SceneNode, this module rebuilds the
same prim path ``EntityBuilder`` used: each node's sanitized name,
appended under its parent's path, starting at ``/World/Entities``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom

logger = logging.getLogger("physworldlm.transform_builder")


# =============================================================================
# Exceptions
# =============================================================================


class TransformBuilderError(Exception):
    """Base class for all errors raised by the Transform Builder."""


class InvalidTransformError(TransformBuilderError):
    """Raised when transform data on a SceneNode is malformed."""


class TransformApplicationError(TransformBuilderError):
    """Raised when a transform cannot be applied to a USD prim."""


# =============================================================================
# Build report
# =============================================================================


@dataclass
class TransformBuildResult:
    """Diagnostics describing the outcome of a ``TransformBuilder.build()`` call.

    Attributes:
        applied_count: Number of prims that received a transform.
        missing_prim_count: Number of entities whose corresponding
            prim could not be found on the stage.
        default_count: Number of entities that had no transform data
            and received the default identity transform.
        warnings: Non-fatal issues encountered while building.
    """

    applied_count: int = 0
    missing_prim_count: int = 0
    default_count: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Scene Graph accessors (duck-typed, mirrors entity_builder.py conventions)
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


def _sanitize_name(name: str, fallback: str = "Entity") -> str:
    """Reproduce EntityBuilder's prim-naming rule so prims can be located.

    USD prim names must be valid identifiers: alphanumeric and
    underscore, not starting with a digit. This must stay identical to
    ``EntityBuilder._sanitize_name`` or prim lookups will fail.
    """
    if not name:
        name = fallback
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


# =============================================================================
# Transform Builder
# =============================================================================


class TransformBuilder:
    """Applies spatial transforms to entity prims created by ``EntityBuilder``.

    Example:
        >>> builder = TransformBuilder()
        >>> result = builder.build(stage, scene_graph)
        >>> result.applied_count
        3
    """

    _ENTITIES_ROOT = "/World/Entities"
    _DEFAULT_TRANSLATION: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _DEFAULT_ROTATION: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _DEFAULT_SCALE: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self) -> None:
        self._result = TransformBuildResult()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, stage: Usd.Stage, scene_graph: Any) -> TransformBuildResult:
        """Apply transforms to every entity prim described by ``scene_graph``.

        Args:
            stage: An existing ``Usd.Stage`` whose ``/World/Entities``
                hierarchy was already populated by ``EntityBuilder``.
            scene_graph: The validated Scene Graph produced by
                ``SceneCompiler.compile(...)``.

        Returns:
            A ``TransformBuildResult`` summarizing what was applied.

        Raises:
            TransformBuilderError: If ``stage`` or ``scene_graph`` is
                missing.
        """
        if stage is None:
            raise TransformBuilderError("TransformBuilder requires an existing Usd.Stage, got None.")
        if scene_graph is None:
            raise TransformBuilderError("TransformBuilder requires a scene_graph, got None.")

        self._result = TransformBuildResult()
        entities = self._top_level_entities(scene_graph)

        self._traverse_scene_graph(stage, Sdf.Path(self._ENTITIES_ROOT), entities)

        logger.info("Completed entity transform pass: %d applied, %d missing prim(s)",
                    self._result.applied_count, self._result.missing_prim_count)
        return self._result

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _top_level_entities(self, scene_graph: Any) -> Sequence[Any]:
        entities = _iter(_get(scene_graph, "entities"))
        if entities:
            return entities
        root = _get(scene_graph, "root")
        return _node_children(root) if root is not None else ()

    def _traverse_scene_graph(
        self,
        stage: Usd.Stage,
        parent_path: Sdf.Path,
        nodes: Sequence[Any],
    ) -> None:
        """Recursively walk the Scene Graph, applying transforms as it goes."""
        for node in nodes:
            if node is None:
                self._result.warnings.append("Encountered a None node during traversal; skipped.")
                continue

            raw_name = _get(node, "name") or _get(node, "id")
            prim_path = parent_path.AppendChild(_sanitize_name(raw_name))

            prim = self._find_entity_prim(stage, prim_path)
            if prim is None:
                self._result.missing_prim_count += 1
                self._result.warnings.append(
                    f"No prim found at '{prim_path}' for entity "
                    f"'{raw_name}'; skipping transform (was it created by EntityBuilder?)."
                )
            else:
                self._apply_transform(prim, node)

            self._traverse_scene_graph(stage, prim_path, _node_children(node))

    def _find_entity_prim(self, stage: Usd.Stage, path: Sdf.Path) -> Optional[Usd.Prim]:
        """Locate the USD prim corresponding to a Scene Graph node."""
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
        return None

    # ------------------------------------------------------------------
    # Transform extraction
    # ------------------------------------------------------------------

    def _extract_transform(self, node: Any) -> Dict[str, Any]:
        """Pull translation/rotation/scale/pivot data from a SceneNode.

        Looks first for a nested ``transform`` object (matching the
        convention used by ``usd_exporter.py``), then falls back to
        flat attributes directly on the node (``position`` /
        ``translation``, ``rotation`` / ``quaternion``, ``scale``,
        ``pivot``). Missing values fall back to the class defaults.

        Raises:
            InvalidTransformError: If a present value cannot be
                interpreted as a 3-component vector.
        """
        transform = _get(node, "transform")

        translation = (
            _get(transform, "translation")
            if transform is not None
            else (_get(node, "translation") or _get(node, "position"))
        )
        rotation = (
            _get(transform, "rotation") if transform is not None else _get(node, "rotation")
        )
        quaternion = (
            _get(transform, "quaternion") if transform is not None else _get(node, "quaternion")
        )
        scale = _get(transform, "scale") if transform is not None else _get(node, "scale")
        pivot = _get(transform, "pivot") if transform is not None else _get(node, "pivot")

        return {
            "translation": self._coerce_vec3(translation, self._DEFAULT_TRANSLATION, "translation"),
            "rotation": self._coerce_vec3(rotation, self._DEFAULT_ROTATION, "rotation"),
            "quaternion": quaternion,
            "scale": self._coerce_vec3(scale, self._DEFAULT_SCALE, "scale"),
            "pivot": self._coerce_vec3(pivot, None, "pivot") if pivot is not None else None,
            "had_data": any(v is not None for v in (translation, rotation, scale, pivot, quaternion)),
        }

    def _coerce_vec3(
        self,
        value: Any,
        default: Optional[Tuple[float, float, float]],
        field_name: str,
    ) -> Optional[Tuple[float, float, float]]:
        if value is None:
            return default
        try:
            components = tuple(float(v) for v in value)
        except (TypeError, ValueError) as exc:
            raise InvalidTransformError(
                f"Could not interpret '{field_name}' value {value!r} as a 3-component vector: {exc}"
            ) from exc
        if len(components) != 3:
            raise InvalidTransformError(
                f"'{field_name}' must have exactly 3 components, got {len(components)}: {value!r}"
            )
        return components  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    def _apply_transform(self, prim: Usd.Prim, node: Any) -> None:
        """Extract and apply a full transform to a single prim."""
        try:
            transform = self._extract_transform(node)
        except InvalidTransformError as exc:
            self._result.warnings.append(f"Invalid transform for '{prim.GetPath()}': {exc}")
            logger.warning("Invalid transform values for '%s': %s", prim.GetPath(), exc)
            transform = {
                "translation": self._DEFAULT_TRANSLATION,
                "rotation": self._DEFAULT_ROTATION,
                "quaternion": None,
                "scale": self._DEFAULT_SCALE,
                "pivot": None,
                "had_data": False,
            }

        if not transform["had_data"]:
            logger.info("Missing transform data for '%s'; applying defaults", prim.GetPath())
            self._result.default_count += 1

        xform = UsdGeom.Xform(prim)
        if not xform:
            raise TransformApplicationError(
                f"Prim at '{prim.GetPath()}' is not a UsdGeom.Xform; cannot apply transform."
            )

        try:
            xform_api = UsdGeom.XformCommonAPI(xform)
            self._apply_translation(xform_api, transform["translation"], prim.GetPath())
            self._apply_rotation(xform_api, transform["rotation"], transform["quaternion"], prim.GetPath())
            self._apply_scale(xform_api, transform["scale"], prim.GetPath())
            if transform["pivot"] is not None:
                xform_api.SetPivot(Gf.Vec3f(*transform["pivot"]))
        except TransformApplicationError:
            raise
        except Exception as exc:
            raise TransformApplicationError(
                f"Failed to apply transform to '{prim.GetPath()}': {exc}"
            ) from exc

        self._result.applied_count += 1

    def _apply_translation(
        self, xform_api: UsdGeom.XformCommonAPI, translation: Tuple[float, float, float], path: Sdf.Path
    ) -> None:
        logger.info("Applying translation to '%s'", path)
        xform_api.SetTranslate(Gf.Vec3d(*translation))

    def _apply_rotation(
        self,
        xform_api: UsdGeom.XformCommonAPI,
        rotation: Tuple[float, float, float],
        quaternion: Optional[Any],
        path: Sdf.Path,
    ) -> None:
        logger.info("Applying rotation to '%s'", path)
        if quaternion is not None:
            euler = self._quaternion_to_euler_xyz(quaternion)
            xform_api.SetRotate(Gf.Vec3f(*euler), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        else:
            xform_api.SetRotate(Gf.Vec3f(*rotation), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    def _apply_scale(
        self, xform_api: UsdGeom.XformCommonAPI, scale: Tuple[float, float, float], path: Sdf.Path
    ) -> None:
        logger.info("Applying scale to '%s'", path)
        xform_api.SetScale(Gf.Vec3f(*scale))

    def _quaternion_to_euler_xyz(self, quaternion: Any) -> Tuple[float, float, float]:
        """Convert a (w, x, y, z) or (x, y, z, w) quaternion to XYZ Euler degrees.

        Accepts any 4-component sequence and assumes ``(w, x, y, z)``
        ordering, the Gf.Quatf/Gf.Quatd convention. Falls back to the
        default rotation on malformed input rather than raising, since
        rotation is best-effort metadata, not safety-critical.
        """
        try:
            w, x, y, z = (float(c) for c in quaternion)
            quat = Gf.Quatf(w, Gf.Vec3f(x, y, z))
            rotation = Gf.Rotation(quat)
            return tuple(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))
        except Exception as exc:
            self._result.warnings.append(f"Could not decode quaternion {quaternion!r}: {exc}")
            return self._DEFAULT_ROTATION


# =============================================================================
# Self-contained smoke test / demo
# =============================================================================


def _build_demo_scene_graph() -> Any:
    @dataclass
    class _Transform:
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @dataclass
    class _Node:
        id: str
        name: str
        transform: Optional[_Transform] = None
        children: List["_Node"] = field(default_factory=list)

    @dataclass
    class _SceneGraph:
        entities: List[_Node] = field(default_factory=list)

    cannon = _Node(id="cannon-1", name="Cannon", transform=_Transform(translation=(0.0, 0.5, 0.0)))
    turret = _Node(id="turret-1", name="Turret", transform=_Transform(rotation=(0.0, 45.0, 0.0)), children=[cannon])
    vehicle = _Node(
        id="vehicle-1",
        name="Vehicle",
        transform=_Transform(translation=(100.0, 0.0, 50.0)),
        children=[turret],
    )
    radar = _Node(id="radar-1", name="Radar")  # no transform -> defaults

    return _SceneGraph(entities=[vehicle, radar])


if __name__ == "__main__":
    from entity_builder import EntityBuilder  # local import: demo-only dependency
    from stage_builder import StageBuilder  # local import: demo-only dependency

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    demo_stage = Usd.Stage.CreateInMemory()
    StageBuilder().build(demo_stage)

    demo_scene_graph = _build_demo_scene_graph()
    EntityBuilder().build(demo_stage, demo_scene_graph)
    transform_result = TransformBuilder().build(demo_stage, demo_scene_graph)

    print(transform_result)
    vehicle_prim = demo_stage.GetPrimAtPath("/World/Entities/Vehicle")
    print("Vehicle xformOpOrder:", vehicle_prim.GetAttribute("xformOpOrder").Get())
