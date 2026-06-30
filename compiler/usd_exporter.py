"""usd_exporter.py

OpenUSD Exporter for the PhysWorldLM framework.

This module is the bridge between the backend-independent Scene Graph
produced by ``scene_compiler.py`` and downstream OpenUSD-consuming
platforms (NVIDIA Omniverse, Isaac Sim, usdview, Blender, Unreal, Unity,
and future robotics / digital-twin ecosystems).

Responsibilities of this module, and only this module:
    * Translate an already-constructed Scene Graph into a standards
      compliant OpenUSD stage.
    * Preserve hierarchy, transforms, metadata, semantic information,
      and relationships present in the Scene Graph.
    * Produce a structured ``ExportResult`` describing the outcome.

This module explicitly does NOT:
    * Parse natural-language prompts.
    * Perform ontology reasoning or WorldSpec validation.
    * Perform scene compilation.
    * Implement rendering, physics, or simulation logic.

Scene Graph contract
---------------------
This module intentionally does **not** redefine ``SceneGraph``,
``SceneNode``, or ``WorldSpec``. Those types are owned by
``scene_compiler.py`` / ``world_spec.py``. Because the real
implementations were not available at the time this module was
written, the exporter accesses the Scene Graph defensively (duck
typing via ``getattr`` / mapping access) against the interface
documented in ``SceneGraphInterface`` below. As soon as the real
``scene_compiler.py`` is dropped in next to this file, no changes
should be required here as long as it exposes attributes with the
same names -- if names differ, only the small accessor helpers in the
"Scene Graph Accessors" section need to be updated.

Expected (duck-typed) interface::

    SceneGraph:
        root            -> SceneNode | None
        environment     -> SceneNode | None
        entities        -> Iterable[SceneNode]
        sensors         -> Iterable[SceneNode]
        relationships   -> Iterable[RelationshipLike]   (optional)
        metadata        -> Mapping[str, Any]             (optional)

    SceneNode:
        id              -> str
        name            -> str
        node_type       -> str   (e.g. "entity", "terrain", "light")
        category        -> str | None
        ontology_id     -> str | None
        semantic_label  -> str | None
        visible         -> bool
        enabled         -> bool
        transform       -> TransformLike | None
        metadata        -> Mapping[str, Any]
        children        -> Iterable[SceneNode]

    TransformLike:
        translation     -> (x, y, z)
        rotation        -> (rx, ry, rz)   (degrees, XYZ order)
        scale           -> (sx, sy, sz)
        pivot           -> (x, y, z) | None

    RelationshipLike:
        kind            -> str  (e.g. "attached_to", "mounted_on")
        source_id       -> str
        target_id       -> str

Author: PhysWorldLM execution layer team.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from pxr import Usd, UsdGeom, Sdf, Gf
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'usd_exporter' module requires Pixar's OpenUSD Python "
        "bindings. Install them with `pip install usd-core` "
        "(or the NVIDIA Omniverse / Isaac Sim provided 'pxr' package)."
    ) from exc


logger = logging.getLogger("physworldlm.usd_exporter")


# =============================================================================
# Enums
# =============================================================================


class ExportFormat(str, Enum):
    """Supported (or future) OpenUSD export formats.

    Only USDA is fully implemented today. The enum exists so the public
    API does not need to change when USDC / USDZ / Nucleus support is
    added.
    """

    USDA = "usda"
    USDC = "usdc"
    USDZ = "usdz"
    NUCLEUS = "nucleus"  # placeholder for future Omniverse Nucleus export


class UpAxis(str, Enum):
    """Stage up-axis options."""

    Y = "Y"
    Z = "Z"


# =============================================================================
# Exceptions
# =============================================================================


class USDExportError(Exception):
    """Base class for all errors raised by the USD exporter."""


class InvalidSceneGraphError(USDExportError):
    """Raised when the supplied Scene Graph is missing or malformed."""


class StageInitializationError(USDExportError):
    """Raised when the USD stage cannot be created or configured."""


class PrimCreationError(USDExportError):
    """Raised when a USD prim cannot be created for a given SceneNode."""


class MetadataExportError(USDExportError):
    """Raised when custom/semantic metadata cannot be written to a prim."""


class FileWriteError(USDExportError):
    """Raised when the stage cannot be saved/written to disk."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExportConfig:
    """Configuration controlling how a Scene Graph is exported.

    Attributes:
        export_format: Target OpenUSD format. Only ``ExportFormat.USDA``
            is implemented; other values raise ``NotImplementedError``.
        overwrite: Whether an existing file at ``output_path`` may be
            overwritten.
        up_axis: Stage up-axis, defaults to Y (OpenUSD convention).
        meters_per_unit: Stage linear unit scale.
        default_prim_name: Name of the root prim ("World").
        time_codes_per_second: Stage ``timeCodesPerSecond``.
        frames_per_second: Stage ``framesPerSecond``.
        start_time_code: Stage ``startTimeCode``.
        end_time_code: Stage ``endTimeCode``.
        generate_metadata: Whether to write ``physworld:*`` custom
            attributes onto exported prims.
        generate_relationships: Whether to export Scene Graph
            relationships into ``/World/Relationships``.
        verbose_logging: Raise the exporter's logger to DEBUG level.
    """

    export_format: ExportFormat = ExportFormat.USDA
    overwrite: bool = True
    up_axis: UpAxis = UpAxis.Y
    meters_per_unit: float = 1.0
    default_prim_name: str = "World"
    time_codes_per_second: float = 24.0
    frames_per_second: float = 24.0
    start_time_code: float = 0.0
    end_time_code: float = 0.0
    generate_metadata: bool = True
    generate_relationships: bool = True
    verbose_logging: bool = False


# =============================================================================
# Export Result / Diagnostics
# =============================================================================


@dataclass
class ExportResult:
    """Structured outcome of an export operation.

    Attributes:
        success: Whether the export completed without fatal errors.
        output_path: Absolute path of the written stage file.
        export_time: Wall-clock duration of the export, in seconds.
        file_size: Size of the written file, in bytes.
        prim_count: Total number of USD prims created.
        hierarchy_depth: Maximum depth of the exported prim hierarchy.
        metadata_count: Number of metadata attributes written.
        relationship_count: Number of relationships exported.
        warnings: Non-fatal issues encountered during export.
        errors: Fatal issues encountered during export (if ``success``
            is False).
    """

    success: bool = False
    output_path: str = ""
    export_time: float = 0.0
    file_size: int = 0
    prim_count: int = 0
    hierarchy_depth: int = 0
    metadata_count: int = 0
    relationship_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Scene Graph Accessors
# =============================================================================
#
# These small helpers isolate every place this module touches the Scene
# Graph's attribute names. They support both attribute-style objects
# (the expected real interface) and plain dict/mapping nodes, so the
# exporter degrades gracefully if scene_compiler.py represents nodes as
# dataclasses, plain objects, or dictionaries.


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


def _sanitize_prim_name(name: str, fallback: str = "Node") -> str:
    """Produce a valid USD prim name from an arbitrary string.

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


# =============================================================================
# USD Exporter
# =============================================================================


class USDExporter:
    """Exports PhysWorldLM Scene Graphs to OpenUSD stages.

    The exporter is intentionally organized as a composition of small,
    single-responsibility private methods rather than one large export
    function, so future format/platform support can be layered in
    without touching the public API.

    Example:
        >>> exporter = USDExporter()
        >>> result = exporter.export(scene_graph, "scene.usda")
        >>> result.success
        True
    """

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self._config = config or ExportConfig()
        if self._config.verbose_logging:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Per-export mutable counters, reset at the start of every export().
        self._prim_count = 0
        self._max_depth = 0
        self._metadata_count = 0
        self._relationship_count = 0
        self._warnings: List[str] = []
        self._node_id_to_path: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, scene_graph: Any, output_path: str = "scene.usda") -> ExportResult:
        """Export a Scene Graph to an OpenUSD stage on disk.

        Args:
            scene_graph: The Scene Graph produced by
                ``SceneCompiler.compile(...)``.
            output_path: Destination path for the exported stage.

        Returns:
            An ``ExportResult`` describing the outcome of the export.
            On fatal failure, ``success`` is False and ``errors``
            contains the failure reason(s) -- this method does not
            raise for ordinary, anticipated failures so callers can
            inspect the report programmatically.
        """
        start_time = time.perf_counter()
        self._reset_counters()

        result = ExportResult(output_path=str(Path(output_path).resolve()))

        try:
            self._validate_scene_graph(scene_graph)
            self._validate_format()

            stage = self._initialize_stage(output_path)
            self._configure_stage(stage)
            world_prim = self._export_world_root(stage)

            self._export_environment(stage, scene_graph, world_prim)
            self._export_entities(stage, scene_graph, world_prim)
            self._export_sensors(stage, scene_graph, world_prim)
            self._export_physics_placeholder(stage, world_prim)
            self._export_materials_placeholder(stage, world_prim)
            self._export_relationships(stage, scene_graph, world_prim)
            self._export_metadata(stage, scene_graph, world_prim)

            self._finalize_stage(stage)
            file_size = self._save_stage(stage, output_path)

            result.success = True
            result.file_size = file_size
            result.prim_count = self._prim_count
            result.hierarchy_depth = self._max_depth
            result.metadata_count = self._metadata_count
            result.relationship_count = self._relationship_count
            result.warnings = list(self._warnings)

            logger.info("Export Complete")

        except USDExportError as exc:
            logger.error("Export failed: %s", exc)
            result.success = False
            result.errors.append(str(exc))
            result.warnings = list(self._warnings)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Unexpected error during export")
            result.success = False
            result.errors.append(f"Unexpected error: {exc}")
            result.warnings = list(self._warnings)

        result.export_time = time.perf_counter() - start_time
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _reset_counters(self) -> None:
        self._prim_count = 0
        self._max_depth = 0
        self._metadata_count = 0
        self._relationship_count = 0
        self._warnings = []
        self._node_id_to_path = {}

    def _validate_scene_graph(self, scene_graph: Any) -> None:
        if scene_graph is None:
            raise InvalidSceneGraphError("scene_graph is None; nothing to export.")
        has_any_section = any(
            _get(scene_graph, attr) is not None
            for attr in ("root", "environment", "entities", "sensors")
        )
        if not has_any_section:
            raise InvalidSceneGraphError(
                "scene_graph does not expose any of 'root', 'environment', "
                "'entities', or 'sensors'; the exporter cannot interpret it."
            )

    def _validate_format(self) -> None:
        if self._config.export_format != ExportFormat.USDA:
            raise USDExportError(
                f"Export format '{self._config.export_format.value}' is not yet "
                "implemented. Only ExportFormat.USDA is currently supported; "
                "USDC/USDZ/Nucleus are reserved for future work and will not "
                "require public API changes when added."
            )

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def _initialize_stage(self, output_path: str) -> Usd.Stage:
        logger.info("Creating Stage")
        path = Path(output_path)
        if path.exists() and not self._config.overwrite:
            raise FileWriteError(
                f"Output path '{path}' already exists and overwrite=False."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                raise FileWriteError(f"Could not remove existing file '{path}': {exc}") from exc

        stage = Usd.Stage.CreateNew(str(path))
        if stage is None:
            raise StageInitializationError(f"Usd.Stage.CreateNew returned None for '{path}'.")
        return stage

    def _configure_stage(self, stage: Usd.Stage) -> None:
        try:
            UsdGeom.SetStageUpAxis(
                stage,
                UsdGeom.Tokens.y if self._config.up_axis == UpAxis.Y else UsdGeom.Tokens.z,
            )
            UsdGeom.SetStageMetersPerUnit(stage, self._config.meters_per_unit)
            stage.SetTimeCodesPerSecond(self._config.time_codes_per_second)
            stage.SetFramesPerSecond(self._config.frames_per_second)
            stage.SetStartTimeCode(self._config.start_time_code)
            stage.SetEndTimeCode(self._config.end_time_code)
        except Exception as exc:
            raise StageInitializationError(f"Failed to configure stage: {exc}") from exc

    def _export_world_root(self, stage: Usd.Stage) -> Usd.Prim:
        logger.info("Creating Prim: /%s", self._config.default_prim_name)
        world_path = Sdf.Path(f"/{self._config.default_prim_name}")
        world_xform = UsdGeom.Xform.Define(stage, world_path)
        if not world_xform:
            raise StageInitializationError("Failed to define the World root Xform.")
        stage.SetDefaultPrim(world_xform.GetPrim())
        self._prim_count += 1
        self._max_depth = max(self._max_depth, 1)
        return world_xform.GetPrim()

    def _finalize_stage(self, stage: Usd.Stage) -> None:
        # Reserved for cross-cutting, end-of-export fix-ups (e.g.
        # validating that referenced relationship targets exist). No-op
        # today, kept as an explicit named stage per the architecture.
        return None

    def _save_stage(self, stage: Usd.Stage, output_path: str) -> int:
        logger.info("Saving File")
        try:
            stage.GetRootLayer().Save()
        except Exception as exc:
            raise FileWriteError(f"Failed to save stage to '{output_path}': {exc}") from exc

        path = Path(output_path)
        if not path.exists():
            raise FileWriteError(f"Stage save reported success but '{path}' does not exist.")
        return path.stat().st_size

    # ------------------------------------------------------------------
    # Section exporters
    # ------------------------------------------------------------------

    def _define_group(self, stage: Usd.Stage, parent_path: Sdf.Path, name: str) -> Usd.Prim:
        """Define a (possibly empty) organizational Xform group."""
        child_path = parent_path.AppendChild(_sanitize_prim_name(name))
        xform = UsdGeom.Xform.Define(stage, child_path)
        if not xform:
            raise PrimCreationError(f"Failed to create group prim at '{child_path}'.")
        self._prim_count += 1
        return xform.GetPrim()

    def _export_environment(self, stage: Usd.Stage, scene_graph: Any, world_prim: Usd.Prim) -> None:
        env_prim = self._define_group(stage, world_prim.GetPath(), "Environment")
        for sub_group in ("Terrain", "Atmosphere", "Weather", "Lighting"):
            self._define_group(stage, env_prim.GetPath(), sub_group)

        environment_node = _get(scene_graph, "environment")
        if environment_node is not None:
            logger.info("Exporting Entity: Environment root")
            self._export_node_recursive(stage, env_prim.GetPath(), environment_node, depth=2)

    def _export_entities(self, stage: Usd.Stage, scene_graph: Any, world_prim: Usd.Prim) -> None:
        entities_prim = self._define_group(stage, world_prim.GetPath(), "Entities")
        entities = _iter(_get(scene_graph, "entities"))
        if not entities:
            root = _get(scene_graph, "root")
            entities = _node_children(root) if root is not None else ()

        for entity in entities:
            logger.info("Exporting Entity: %s", _get(entity, "name", "<unnamed>"))
            self._export_node_recursive(stage, entities_prim.GetPath(), entity, depth=2)

    def _export_sensors(self, stage: Usd.Stage, scene_graph: Any, world_prim: Usd.Prim) -> None:
        sensors_prim = self._define_group(stage, world_prim.GetPath(), "Sensors")
        for sensor in _iter(_get(scene_graph, "sensors")):
            logger.info("Exporting Entity: %s", _get(sensor, "name", "<unnamed sensor>"))
            self._export_node_recursive(stage, sensors_prim.GetPath(), sensor, depth=2)

    def _export_physics_placeholder(self, stage: Usd.Stage, world_prim: Usd.Prim) -> None:
        # No physics logic belongs here; this group exists so downstream
        # tools (e.g. PhysX) have a stable, predictable insertion point.
        self._define_group(stage, world_prim.GetPath(), "Physics")

    def _export_materials_placeholder(self, stage: Usd.Stage, world_prim: Usd.Prim) -> None:
        self._define_group(stage, world_prim.GetPath(), "Materials")

    def _export_relationships(self, stage: Usd.Stage, scene_graph: Any, world_prim: Usd.Prim) -> None:
        rel_prim = self._define_group(stage, world_prim.GetPath(), "Relationships")
        if not self._config.generate_relationships:
            return

        relationships = _iter(_get(scene_graph, "relationships"))
        for index, rel in enumerate(relationships):
            kind = _get(rel, "kind", "related_to")
            source_id = _get(rel, "source_id")
            target_id = _get(rel, "target_id")

            rel_holder_path = rel_prim.GetPath().AppendChild(
                _sanitize_prim_name(f"rel_{index}_{kind}")
            )
            rel_holder = UsdGeom.Xform.Define(stage, rel_holder_path)
            if not rel_holder:
                self._warnings.append(f"Could not create relationship prim for '{kind}' (#{index}).")
                continue
            self._prim_count += 1

            usd_rel = rel_holder.GetPrim().CreateRelationship(f"physworld:{kind}", custom=True)
            source_path = self._node_id_to_path.get(source_id)
            target_path = self._node_id_to_path.get(target_id)
            targets = [p for p in (source_path, target_path) if p]
            if targets:
                usd_rel.SetTargets([Sdf.Path(p) for p in targets])
            else:
                self._warnings.append(
                    f"Relationship '{kind}' (#{index}) references unknown node id(s) "
                    f"source={source_id!r} target={target_id!r}; stored without targets."
                )

            rel_holder.GetPrim().CreateAttribute(
                "physworld:source_id", Sdf.ValueTypeNames.String, custom=True
            ).Set(str(source_id) if source_id is not None else "")
            rel_holder.GetPrim().CreateAttribute(
                "physworld:target_id", Sdf.ValueTypeNames.String, custom=True
            ).Set(str(target_id) if target_id is not None else "")

            self._relationship_count += 1

    def _export_metadata(self, stage: Usd.Stage, scene_graph: Any, world_prim: Usd.Prim) -> None:
        logger.info("Writing Metadata")
        metadata_prim = self._define_group(stage, world_prim.GetPath(), "Metadata")
        if not self._config.generate_metadata:
            return

        graph_metadata = _get(scene_graph, "metadata")
        if isinstance(graph_metadata, Mapping):
            self._write_custom_attributes(metadata_prim, graph_metadata, namespace="physworld")

    # ------------------------------------------------------------------
    # Node traversal
    # ------------------------------------------------------------------

    def _export_node_recursive(
        self,
        stage: Usd.Stage,
        parent_path: Sdf.Path,
        node: Any,
        depth: int,
    ) -> Usd.Prim:
        """Recursively convert one SceneNode (and its children) to USD prims."""
        if node is None:
            raise InvalidSceneGraphError("Encountered a None SceneNode during traversal.")

        name = _sanitize_prim_name(_get(node, "name") or _get(node, "id") or "Node")
        prim_path = parent_path.AppendChild(name)

        logger.info("Creating Prim: %s", prim_path)
        xform = UsdGeom.Xform.Define(stage, prim_path)
        if not xform:
            raise PrimCreationError(f"Failed to create prim at '{prim_path}'.")
        prim = xform.GetPrim()
        self._prim_count += 1
        self._max_depth = max(self._max_depth, depth)

        node_id = _get(node, "id")
        if node_id is not None:
            self._node_id_to_path[str(node_id)] = str(prim_path)

        self._apply_transform(xform, node)
        self._apply_visibility(xform, node)
        if self._config.generate_metadata:
            self._export_node_metadata(prim, node)

        for child in _node_children(node):
            self._export_node_recursive(stage, prim_path, child, depth=depth + 1)

        return prim

    def _apply_transform(self, xform: UsdGeom.Xform, node: Any) -> None:
        transform = _get(node, "transform")
        translation = _get(transform, "translation", (0.0, 0.0, 0.0)) if transform else (0.0, 0.0, 0.0)
        rotation = _get(transform, "rotation", (0.0, 0.0, 0.0)) if transform else (0.0, 0.0, 0.0)
        scale = _get(transform, "scale", (1.0, 1.0, 1.0)) if transform else (1.0, 1.0, 1.0)
        pivot = _get(transform, "pivot") if transform else None

        try:
            xform_api = UsdGeom.XformCommonAPI(xform)
            xform_api.SetTranslate(Gf.Vec3d(*translation))
            xform_api.SetRotate(Gf.Vec3f(*rotation), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            xform_api.SetScale(Gf.Vec3f(*scale))
            if pivot is not None:
                xform_api.SetPivot(Gf.Vec3f(*pivot))
        except Exception as exc:
            self._warnings.append(
                f"Could not fully apply transform for '{xform.GetPath()}': {exc}"
            )

    def _apply_visibility(self, xform: UsdGeom.Xform, node: Any) -> None:
        visible = _get(node, "visible", True)
        try:
            if visible:
                xform.MakeVisible()
            else:
                xform.MakeInvisible()
        except Exception as exc:
            self._warnings.append(
                f"Could not set visibility for '{xform.GetPath()}': {exc}"
            )

    def _export_node_metadata(self, prim: Usd.Prim, node: Any) -> None:
        try:
            fields: Dict[str, Any] = {
                "id": _get(node, "id"),
                "type": _get(node, "node_type"),
                "category": _get(node, "category"),
                "ontology": _get(node, "ontology_id"),
                "semantic_label": _get(node, "semantic_label"),
                "enabled": _get(node, "enabled", True),
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            self._write_custom_attributes(prim, fields, namespace="physworld")

            extra_metadata = _node_metadata(node)
            if extra_metadata:
                self._write_custom_attributes(prim, extra_metadata, namespace="physworld:meta")
        except Exception as exc:
            raise MetadataExportError(
                f"Failed to write metadata for prim '{prim.GetPath()}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Attribute writing
    # ------------------------------------------------------------------

    def _write_custom_attributes(self, prim: Usd.Prim, fields: Mapping[str, Any], namespace: str) -> None:
        for key, value in fields.items():
            attr_name = f"{namespace}:{key}"
            value_type, coerced = self._infer_sdf_type(value)
            try:
                attr = prim.CreateAttribute(attr_name, value_type, custom=True)
                attr.Set(coerced)
                self._metadata_count += 1
            except Exception as exc:
                self._warnings.append(
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
        if isinstance(value, (list, tuple)) and len(value) == 3 and all(
            isinstance(v, (int, float)) for v in value
        ):
            return Sdf.ValueTypeNames.Float3, Gf.Vec3f(*value)
        return Sdf.ValueTypeNames.String, str(value)


# =============================================================================
# Self-contained smoke test / demo
# =============================================================================
#
# This block does not run as part of the public API; it exists so the
# module can be sanity-checked before the real scene_compiler.py is
# available. It builds minimal mock objects that satisfy the duck-typed
# interface documented at the top of this file.


def _build_demo_scene_graph() -> Any:
    @dataclass
    class _Transform:
        translation: tuple = (0.0, 0.0, 0.0)
        rotation: tuple = (0.0, 0.0, 0.0)
        scale: tuple = (1.0, 1.0, 1.0)
        pivot: Optional[tuple] = None

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
        transform: Optional[_Transform] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        children: List["_Node"] = field(default_factory=list)

    @dataclass
    class _Relationship:
        kind: str
        source_id: str
        target_id: str

    @dataclass
    class _SceneGraph:
        root: Optional[_Node] = None
        environment: Optional[_Node] = None
        entities: List[_Node] = field(default_factory=list)
        sensors: List[_Node] = field(default_factory=list)
        relationships: List[_Relationship] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    enemy_aircraft = _Node(
        id="enemy-aircraft-1",
        name="EnemyAircraft",
        node_type="entity",
        category="aircraft",
        semantic_label="hostile_air_target",
        transform=_Transform(translation=(1200.0, 8500.0, 0.0)),
    )
    radar = _Node(
        id="radar-1",
        name="Radar",
        node_type="sensor",
        category="sensor",
        semantic_label="ground_radar",
        transform=_Transform(translation=(0.0, 0.0, 0.0)),
    )
    interceptor = _Node(
        id="interceptor-1",
        name="Interceptor",
        node_type="entity",
        category="missile",
        semantic_label="interceptor_missile",
        transform=_Transform(translation=(0.0, 0.0, 0.0)),
    )

    return _SceneGraph(
        entities=[enemy_aircraft, interceptor],
        sensors=[radar],
        relationships=[_Relationship(kind="tracks", source_id="radar-1", target_id="enemy-aircraft-1")],
        metadata={"scenario": "demo", "version": 1},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    demo_scene_graph = _build_demo_scene_graph()
    exporter = USDExporter(ExportConfig(verbose_logging=True))
    export_result = exporter.export(demo_scene_graph, "scene.usda")

    print(export_result)
