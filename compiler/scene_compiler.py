"""
scene_compiler.py
══════════════════════════════════════════════════════════════════════════
Central orchestration engine of PhysWorldLM.

Pipeline position
------------------
    Natural Language Prompt
            │
            ▼
        World Parser
            │
            ▼
         Ontology
            │
            ▼
      Validated WorldSpec
            │
            ▼
     ┌─────────────────┐
     │  SCENE COMPILER  │   <-- this module
     └─────────────────┘
            │
            ▼
        Scene Graph
            │
            ▼
       OpenUSD Export
            │
            ▼
       NVIDIA PhysX
            │
            ▼
     NVIDIA Omniverse

Scope
-----
This module owns the transformation WorldSpec -> Scene Graph -> OpenUSD
export. It does NOT render, simulate, or visualize anything. It is a pure
orchestration / compilation layer.

The compiler is backend-independent: it builds an internal, USD-agnostic
Scene Graph and then hands that graph to a pluggable exporter. New
execution backends (PhysX, Omniverse, Cesium, Unreal, Unity, Blender,
ROS 2, Isaac Sim, ...) integrate by implementing new `Exporter` /
`Builder` plugins -- never by modifying `SceneCompiler` itself.

Public API
----------
    compiler = SceneCompiler()
    report = compiler.compile(world_spec, output_path="scene.usda")

No other public entry point is exposed.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from world_spec import Entity, Environment, Interaction, WorldSpec

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.scene_compiler")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class CompilationError(Exception):
    """Base exception for all scene-compilation failures."""


class ValidationError(CompilationError):
    """Raised when a WorldSpec fails structural or semantic validation."""


class BuilderError(CompilationError):
    """Raised when a registered builder fails to execute its stage."""

    def __init__(self, builder_name: str, message: str, *, cause: Optional[Exception] = None):
        self.builder_name = builder_name
        self.cause = cause
        super().__init__(f"[{builder_name}] {message}")


class ExportError(CompilationError):
    """Raised when the Scene Graph cannot be exported to OpenUSD."""


class DependencyError(CompilationError):
    """Raised when compilation stages are invoked out of dependency order."""


class AssetResolutionError(CompilationError):
    """Raised when a referenced asset (mesh, texture, material file) cannot be resolved."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class CompilationStage(Enum):
    """Ordered stages of the internal compilation pipeline."""

    VALIDATE_WORLD_SPEC = auto()
    CREATE_CONTEXT = auto()
    INIT_SCENE_GRAPH = auto()
    BUILD_WORLD_ROOT = auto()
    BUILD_ENVIRONMENT = auto()
    BUILD_ENTITIES = auto()
    APPLY_TRANSFORMS = auto()
    RESOLVE_ASSETS = auto()
    ASSIGN_MATERIALS = auto()
    ATTACH_PHYSICS = auto()
    CONFIGURE_SENSORS = auto()
    BUILD_RELATIONSHIPS = auto()
    GENERATE_METADATA = auto()
    EXPORT_USD = auto()
    PRODUCE_REPORT = auto()

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


# Declares, for every stage, the stages that must already be COMPLETE.
# This is the authoritative dependency graph referenced by the
# DependencyManager -- builders are never invoked out of this order.
STAGE_DEPENDENCIES: dict[CompilationStage, tuple[CompilationStage, ...]] = {
    CompilationStage.VALIDATE_WORLD_SPEC: (),
    CompilationStage.CREATE_CONTEXT: (CompilationStage.VALIDATE_WORLD_SPEC,),
    CompilationStage.INIT_SCENE_GRAPH: (CompilationStage.CREATE_CONTEXT,),
    CompilationStage.BUILD_WORLD_ROOT: (CompilationStage.INIT_SCENE_GRAPH,),
    CompilationStage.BUILD_ENVIRONMENT: (CompilationStage.BUILD_WORLD_ROOT,),
    CompilationStage.BUILD_ENTITIES: (CompilationStage.BUILD_ENVIRONMENT,),
    CompilationStage.APPLY_TRANSFORMS: (CompilationStage.BUILD_ENTITIES,),
    CompilationStage.RESOLVE_ASSETS: (CompilationStage.APPLY_TRANSFORMS,),
    CompilationStage.ASSIGN_MATERIALS: (CompilationStage.RESOLVE_ASSETS,),
    CompilationStage.ATTACH_PHYSICS: (CompilationStage.ASSIGN_MATERIALS,),
    CompilationStage.CONFIGURE_SENSORS: (CompilationStage.ATTACH_PHYSICS,),
    CompilationStage.BUILD_RELATIONSHIPS: (CompilationStage.CONFIGURE_SENSORS,),
    CompilationStage.GENERATE_METADATA: (CompilationStage.BUILD_RELATIONSHIPS,),
    CompilationStage.EXPORT_USD: (CompilationStage.GENERATE_METADATA,),
    CompilationStage.PRODUCE_REPORT: (CompilationStage.EXPORT_USD,),
}


class DiagnosticSeverity(Enum):
    """Severity levels for structured compiler diagnostics."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class CompilationStatus(Enum):
    """Final outcome of a compilation run."""

    SUCCESS = auto()
    SUCCESS_WITH_WARNINGS = auto()
    FAILED = auto()


class NodeType(Enum):
    """Semantic category of a Scene Graph node."""

    WORLD = auto()
    ENVIRONMENT = auto()
    TERRAIN = auto()
    ATMOSPHERE = auto()
    WEATHER = auto()
    LIGHTING = auto()
    ENTITIES_GROUP = auto()
    ENTITY = auto()
    SENSORS_GROUP = auto()
    SENSOR = auto()
    MATERIALS_GROUP = auto()
    MATERIAL = auto()
    PHYSICS_GROUP = auto()
    PHYSICS_BODY = auto()
    METADATA = auto()


# ════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Diagnostic:
    """A single structured diagnostic message produced during compilation.

    Attributes:
        stage: The compilation stage active when the diagnostic was raised.
        severity: One of INFO / WARNING / ERROR / CRITICAL.
        message: Human-readable description of the diagnostic.
        source_module: Name of the builder or subsystem that raised it.
        timestamp: UTC time the diagnostic was created.
        entity_ref: Optional entity id this diagnostic concerns.
    """

    stage: CompilationStage
    severity: DiagnosticSeverity
    message: str
    source_module: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entity_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.label,
            "severity": self.severity.name,
            "message": self.message,
            "source_module": self.source_module,
            "timestamp": self.timestamp.isoformat(),
            "entity_ref": self.entity_ref,
        }

    def __str__(self) -> str:
        ref = f" (entity={self.entity_ref})" if self.entity_ref else ""
        return f"[{self.severity.name}] {self.stage.label} :: {self.source_module} :: {self.message}{ref}"


# ════════════════════════════════════════════════════════════════════════
# Scene Graph
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Transform:
    """Minimal USD-agnostic spatial transform used inside the Scene Graph."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "translation": self.translation,
            "rotation_euler_rad": self.rotation_euler_rad,
            "scale": self.scale,
        }


@dataclass
class SceneNode:
    """A single node in the backend-independent Scene Graph.

    The Scene Graph is intentionally decoupled from OpenUSD. It can be
    exported to USD, or to any other backend, without modification.

    Attributes:
        uuid: Stable unique identifier for this node.
        name: Human-readable / USD-prim-safe name.
        node_type: Semantic category (see NodeType).
        parent: Parent node, or None for the root.
        children: Ordered list of child nodes.
        components: Arbitrary structured data attached to this node
            (e.g. physics body data, material data, sensor data). Keyed
            by component name so multiple builders can attach data to
            the same node without collisions.
        metadata: Free-form metadata (provenance, tags, notes).
        transform: Local-space spatial transform.
        visible: Whether the node should be rendered.
        enabled: Whether the node participates in simulation.
    """

    name: str
    node_type: NodeType
    node_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional["SceneNode"] = field(default=None, repr=False)
    children: list["SceneNode"] = field(default_factory=list)
    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    enabled: bool = True

    def add_child(self, child: "SceneNode") -> "SceneNode":
        """Attach `child` to this node, setting back-reference, and return it."""
        child.parent = self
        self.children.append(child)
        return child

    def find(self, predicate: Callable[["SceneNode"], bool]) -> Optional["SceneNode"]:
        """Depth-first search for the first descendant matching `predicate`."""
        if predicate(self):
            return self
        for child in self.children:
            found = child.find(predicate)
            if found is not None:
                return found
        return None

    def find_by_uuid(self, node_uuid: str) -> Optional["SceneNode"]:
        return self.find(lambda n: n.node_uuid == node_uuid)

    def walk(self) -> list["SceneNode"]:
        """Return this node and all descendants, depth-first, pre-order."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    @property
    def path(self) -> str:
        """USD-style forward-slash path from the root to this node."""
        if self.parent is None:
            return f"/{self._safe_name()}"
        return f"{self.parent.path}/{self._safe_name()}"

    def _safe_name(self) -> str:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in self.name)
        if not safe or safe[0].isdigit():
            safe = f"_{safe}"
        return safe

    def to_dict(self) -> dict:
        return {
            "uuid": self.node_uuid,
            "name": self.name,
            "type": self.node_type.name,
            "path": self.path,
            "visible": self.visible,
            "enabled": self.enabled,
            "transform": self.transform.to_dict(),
            "components": list(self.components.keys()),
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class SceneGraph:
    """Root container for the backend-independent scene representation.

    Logical top-level layout (mirrored as children of `root`):

        World
        ├── Environment
        │     ├── Terrain
        │     ├── Atmosphere
        │     ├── Weather
        │     └── Lighting
        ├── Entities
        ├── Sensors
        ├── Materials
        ├── Physics
        └── Metadata
    """

    root: SceneNode = field(
        default_factory=lambda: SceneNode(name="World", node_type=NodeType.WORLD)
    )

    def node_count(self) -> int:
        return len(self.root.walk())

    def nodes_of_type(self, node_type: NodeType) -> list[SceneNode]:
        return [n for n in self.root.walk() if n.node_type == node_type]

    def to_dict(self) -> dict:
        return self.root.to_dict()


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

class ExportFormat(Enum):
    USDA = "usda"   # ASCII OpenUSD
    USDC = "usdc"   # Binary (crate) OpenUSD -- requires `pxr`
    USDZ = "usdz"   # Packaged OpenUSD -- requires `pxr`


class OptimizationLevel(Enum):
    NONE = auto()
    BASIC = auto()
    AGGRESSIVE = auto()


class ValidationMode(Enum):
    STRICT = auto()      # any error aborts compilation
    PERMISSIVE = auto()  # errors are recorded as diagnostics, compilation continues
    DISABLED = auto()    # validation stage is skipped entirely


class CoordinateSystem(Enum):
    Y_UP = "y_up"
    Z_UP = "z_up"


class UnitSystem(Enum):
    SI_METERS = "meters_kilograms_seconds"


@dataclass
class CompilerConfig:
    """User-configurable settings controlling compiler behavior.

    Attributes:
        export_format: Target OpenUSD flavor.
        optimization_level: How aggressively to prune/merge scene nodes.
        validation_mode: STRICT aborts on the first ValidationError;
            PERMISSIVE records errors as diagnostics and proceeds;
            DISABLED skips WorldSpec validation altogether.
        log_level: Python logging level name (e.g. "INFO", "DEBUG").
        overwrite_existing: Whether export may overwrite an existing file.
        asset_search_paths: Directories searched when resolving asset_refs.
        coordinate_system: Up-axis convention used by the exported scene.
        unit_system: Unit convention (PhysWorldLM is SI end-to-end).
        generate_metadata: Whether to emit a metadata node / USD layer
            metadata describing provenance.
        deterministic: If True, node UUIDs are derived deterministically
            from entity ids rather than randomly generated, so repeated
            compilations of the same WorldSpec produce byte-identical
            Scene Graphs (modulo timestamps).
    """

    export_format: ExportFormat = ExportFormat.USDA
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    validation_mode: ValidationMode = ValidationMode.STRICT
    log_level: str = "INFO"
    overwrite_existing: bool = True
    asset_search_paths: list[Path] = field(default_factory=list)
    coordinate_system: CoordinateSystem = CoordinateSystem.Y_UP
    unit_system: UnitSystem = UnitSystem.SI_METERS
    generate_metadata: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))


# ════════════════════════════════════════════════════════════════════════
# Statistics
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CompilationStatistics:
    """Quantitative summary of a single compilation run."""

    compilation_time_s: float = 0.0
    entity_count: int = 0
    relationship_count: int = 0
    asset_count: int = 0
    material_count: int = 0
    sensor_count: int = 0
    environment_object_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    exported_file_size_bytes: int = 0
    success: bool = False
    stage_durations_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "compilation_time_s": round(self.compilation_time_s, 6),
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "asset_count": self.asset_count,
            "material_count": self.material_count,
            "sensor_count": self.sensor_count,
            "environment_object_count": self.environment_object_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "exported_file_size_bytes": self.exported_file_size_bytes,
            "success": self.success,
            "stage_durations_s": {k: round(v, 6) for k, v in self.stage_durations_s.items()},
        }


# ════════════════════════════════════════════════════════════════════════
# Compilation Context
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CompilationContext:
    """Mutable state shared across every compilation stage.

    A single `CompilationContext` instance is created per `compile()` call
    and threaded through every builder. It is the one source of truth for
    "what stage are we in / what has been built so far / what has gone
    wrong".
    """

    world_spec: WorldSpec
    config: CompilerConfig
    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    builder_registry: "BuilderRegistry" = field(default=None)  # type: ignore[assignment]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    statistics: CompilationStatistics = field(default_factory=CompilationStatistics)
    asset_registry: dict[str, Path] = field(default_factory=dict)
    completed_stages: set[CompilationStage] = field(default_factory=set)
    current_stage: Optional[CompilationStage] = None
    # Lookup of WorldSpec entity id -> the SceneNode built for it.
    entity_node_index: dict[str, SceneNode] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    # ── diagnostics helpers ────────────────────────────────────────

    def log_diagnostic(
        self,
        severity: DiagnosticSeverity,
        message: str,
        source_module: str,
        entity_ref: Optional[str] = None,
    ) -> Diagnostic:
        diag = Diagnostic(
            stage=self.current_stage or CompilationStage.VALIDATE_WORLD_SPEC,
            severity=severity,
            message=message,
            source_module=source_module,
            entity_ref=entity_ref,
        )
        self.diagnostics.append(diag)
        if severity is DiagnosticSeverity.WARNING:
            self.statistics.warning_count += 1
            logger.warning(str(diag))
        elif severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL):
            self.statistics.error_count += 1
            logger.error(str(diag))
        else:
            logger.info(str(diag))
        return diag

    def info(self, message: str, source_module: str, entity_ref: Optional[str] = None) -> None:
        self.log_diagnostic(DiagnosticSeverity.INFO, message, source_module, entity_ref)

    def warning(self, message: str, source_module: str, entity_ref: Optional[str] = None) -> None:
        self.log_diagnostic(DiagnosticSeverity.WARNING, message, source_module, entity_ref)

    def error(self, message: str, source_module: str, entity_ref: Optional[str] = None) -> None:
        self.log_diagnostic(DiagnosticSeverity.ERROR, message, source_module, entity_ref)

    def has_errors(self) -> bool:
        return self.statistics.error_count > 0

    # ── stage bookkeeping ───────────────────────────────────────────

    def mark_stage_complete(self, stage: CompilationStage, duration_s: float) -> None:
        self.completed_stages.add(stage)
        self.statistics.stage_durations_s[stage.label] = duration_s

    def assert_dependencies_met(self, stage: CompilationStage) -> None:
        missing = [
            dep for dep in STAGE_DEPENDENCIES.get(stage, ()) if dep not in self.completed_stages
        ]
        if missing:
            raise DependencyError(
                f"Cannot run stage '{stage.label}': missing prerequisite stage(s) "
                f"{[m.label for m in missing]}."
            )


# ════════════════════════════════════════════════════════════════════════
# Compilation Report
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CompilationReport:
    """Final, structured result returned by `SceneCompiler.compile()`."""

    status: CompilationStatus
    scene_id: str
    output_path: Optional[Path]
    statistics: CompilationStatistics
    diagnostics: list[Diagnostic]
    scene_graph: Optional[SceneGraph] = None

    @property
    def success(self) -> bool:
        return self.status is not CompilationStatus.FAILED

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)]

    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is DiagnosticSeverity.WARNING]

    def to_dict(self) -> dict:
        return {
            "status": self.status.name,
            "scene_id": self.scene_id,
            "output_path": str(self.output_path) if self.output_path else None,
            "statistics": self.statistics.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def __str__(self) -> str:
        lines = [
            f"CompilationReport(scene_id={self.scene_id!r}, status={self.status.name})",
            f"  output_path : {self.output_path}",
            f"  entities    : {self.statistics.entity_count}",
            f"  warnings    : {self.statistics.warning_count}",
            f"  errors      : {self.statistics.error_count}",
            f"  time        : {self.statistics.compilation_time_s:.4f}s",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Builder protocol + registry
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Builder(Protocol):
    """Structural contract every builder plugin must satisfy.

    Builders are pure-ish, single-responsibility units invoked by the
    `SceneCompiler` through the `BuilderRegistry`. They contain NO
    orchestration logic -- they read what they need from `context` and
    mutate `context.scene_graph` (or other context state).
    """

    name: str

    def build(self, context: CompilationContext) -> None:
        ...


class BuilderRegistry:
    """Dynamic registry mapping a builder name to a `Builder` instance.

    Builders are discovered by name rather than hard-coded, which is what
    allows new backends/builders to be added without modifying
    `SceneCompiler`.
    """

    def __init__(self) -> None:
        self._builders: dict[str, Builder] = {}

    def register(self, builder: Builder) -> None:
        if not hasattr(builder, "name") or not hasattr(builder, "build"):
            raise BuilderError(
                getattr(builder, "name", builder.__class__.__name__),
                "Builder must expose a `name` attribute and a `build(context)` method.",
            )
        self._builders[builder.name] = builder
        logger.debug("Registered builder: %s", builder.name)

    def unregister(self, name: str) -> None:
        self._builders.pop(name, None)

    def get(self, name: str) -> Builder:
        try:
            return self._builders[name]
        except KeyError as exc:
            raise BuilderError(name, "No builder registered under this name.") from exc

    def has(self, name: str) -> bool:
        return name in self._builders

    def invoke(self, name: str, context: CompilationContext) -> None:
        builder = self.get(name)
        start = time.monotonic()
        try:
            logger.info("Builder start  : %-22s stage=%s", name, context.current_stage.label if context.current_stage else "?")
            builder.build(context)
        except CompilationError:
            raise
        except Exception as exc:  # noqa: BLE001 - intentionally broad, re-wrapped below
            raise BuilderError(name, f"Unhandled exception during build: {exc}", cause=exc) from exc
        finally:
            duration = time.monotonic() - start
            logger.info("Builder finish : %-22s duration=%.4fs", name, duration)


# ════════════════════════════════════════════════════════════════════════
# Concrete builders
# ════════════════════════════════════════════════════════════════════════
# Each builder owns exactly one compilation concern. They are registered
# under well-known names and invoked by SceneCompiler through the
# BuilderRegistry -- never called directly.

class StageBuilder:
    """Creates the root `World` stage node of the Scene Graph."""

    name = "stage_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        root.metadata.update(
            {
                "scene_id": context.world_spec.scene_id,
                "description": context.world_spec.description,
                "coordinate_system": context.config.coordinate_system.value,
                "unit_system": context.config.unit_system.value,
            }
        )
        context.info(f"World root initialized for scene '{context.world_spec.scene_id}'.", self.name)


class EnvironmentBuilder:
    """Builds Environment / Terrain / Atmosphere / Weather / Lighting nodes."""

    name = "environment_builder"

    def build(self, context: CompilationContext) -> None:
        env: Environment = context.world_spec.environment
        root = context.scene_graph.root

        env_node = root.add_child(SceneNode(name="Environment", node_type=NodeType.ENVIRONMENT))
        env_node.components["environment"] = env.to_dict()

        terrain_node = env_node.add_child(SceneNode(name="Terrain", node_type=NodeType.TERRAIN))
        terrain_node.metadata["terrain_type"] = env.terrain_type
        terrain_node.metadata["global_friction"] = env.friction_global

        atmosphere_node = env_node.add_child(SceneNode(name="Atmosphere", node_type=NodeType.ATMOSPHERE))
        atmosphere_node.metadata.update(
            {
                "temperature_K": env.temperature_K,
                "pressure_Pa": env.pressure_Pa,
                "air_density": env.air_density,
            }
        )

        weather_node = env_node.add_child(SceneNode(name="Weather", node_type=NodeType.WEATHER))
        weather_node.metadata.update(
            {
                "weather": env.weather,
                "wind_speed_ms": env.wind.speed,
                "wind_direction_rad": env.wind.direction,
            }
        )

        lighting_node = env_node.add_child(SceneNode(name="Lighting", node_type=NodeType.LIGHTING))
        lighting_node.metadata["time_of_day"] = env.time_of_day

        context.statistics.environment_object_count = len(env_node.children)
        context.info("Environment hierarchy built (terrain/atmosphere/weather/lighting).", self.name)


class EntityBuilder:
    """Builds one SceneNode per WorldSpec Entity, under an `Entities` group."""

    name = "entity_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        entities_group = root.add_child(SceneNode(name="Entities", node_type=NodeType.ENTITIES_GROUP))

        for entity in context.world_spec.entities:
            node = SceneNode(
                name=entity.label or entity.id,
                node_type=NodeType.ENTITY,
                node_uuid=self._stable_uuid(entity.id) if context.config.deterministic else str(uuid.uuid4()),
            )
            node.metadata.update(
                {
                    "world_spec_id": entity.id,
                    "entity_type": entity.entity_type,
                    "is_static": entity.is_static,
                    "tags": list(entity.tags),
                }
            )
            node.components["entity"] = entity.to_dict()
            node.enabled = True
            node.visible = True

            entities_group.add_child(node)
            context.entity_node_index[entity.id] = node

        context.statistics.entity_count = len(context.world_spec.entities)
        context.info(f"Built {context.statistics.entity_count} entity node(s).", self.name)

    @staticmethod
    def _stable_uuid(seed: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"physworldlm://entity/{seed}"))


class TransformBuilder:
    """Applies position / orientation kinematic state onto each entity node."""

    name = "transform_builder"

    def build(self, context: CompilationContext) -> None:
        applied = 0
        for entity in context.world_spec.entities:
            node = context.entity_node_index.get(entity.id)
            if node is None:
                context.warning(
                    f"No scene node found for entity '{entity.id}'; skipping transform.",
                    self.name,
                    entity_ref=entity.id,
                )
                continue
            pos = entity.state.position
            rot = entity.state.orientation
            bbox = entity.bounding_box
            node.transform = Transform(
                translation=(pos.x, pos.y, pos.z),
                rotation_euler_rad=(rot.x, rot.y, rot.z),
                scale=(bbox.width, bbox.height, bbox.depth),
            )
            node.components["kinematics"] = entity.state.to_dict()
            applied += 1
        context.info(f"Applied transforms to {applied} entity node(s).", self.name)


class AssetResolver:
    """Resolves any external asset references declared on entities.

    `WorldSpec.Entity` in this build of PhysWorldLM does not yet carry
    geometry/mesh asset references directly (geometry is implicit via
    `bounding_box` + procedural primitives), so this builder currently
    walks `entity.tags` / `entity.metadata`-style hints defensively and
    is a safe no-op when no asset references are present. It is the
    integration point future geometry-asset support should extend.
    """

    name = "asset_resolver"

    ASSET_TAG_PREFIX = "asset:"

    def build(self, context: CompilationContext) -> None:
        resolved = 0
        for entity in context.world_spec.entities:
            for tag in entity.tags:
                if not tag.startswith(self.ASSET_TAG_PREFIX):
                    continue
                ref = tag[len(self.ASSET_TAG_PREFIX):]
                path = self._resolve(ref, context)
                if path is None:
                    context.warning(
                        f"Could not resolve asset reference '{ref}' for entity '{entity.id}'.",
                        self.name,
                        entity_ref=entity.id,
                    )
                    continue
                context.asset_registry[ref] = path
                node = context.entity_node_index.get(entity.id)
                if node is not None:
                    node.components.setdefault("assets", []).append(str(path))
                resolved += 1

        context.statistics.asset_count = len(context.asset_registry)
        context.info(f"Resolved {resolved} asset reference(s).", self.name)

    def _resolve(self, ref: str, context: CompilationContext) -> Optional[Path]:
        candidate = Path(ref)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for search_path in context.config.asset_search_paths:
            full = search_path / ref
            if full.exists():
                return full
        # Permissive: in absence of search paths / files on disk, record
        # the logical reference itself rather than raising, since asset
        # resolution may legitimately point at a remote/Omniverse Nucleus
        # path that is not locally checkable.
        return candidate


class MaterialBuilder:
    """Builds Material nodes from the static MATERIAL_DEFAULTS table + per-entity overrides."""

    name = "material_builder"

    def build(self, context: CompilationContext) -> None:
        from world_spec import MATERIAL_DEFAULTS

        root = context.scene_graph.root
        materials_group = root.add_child(SceneNode(name="Materials", node_type=NodeType.MATERIALS_GROUP))

        created: dict[str, SceneNode] = {}
        for entity in context.world_spec.entities:
            mat_name = entity.material
            if mat_name not in created:
                defaults = MATERIAL_DEFAULTS.get(mat_name, MATERIAL_DEFAULTS["generic"])
                mat_node = SceneNode(name=mat_name, node_type=NodeType.MATERIAL)
                mat_node.components["material"] = {
                    "name": mat_name,
                    "density": defaults["density"],
                    "restitution": entity.restitution if entity.material == mat_name else defaults["restitution"],
                    "friction": entity.friction if entity.material == mat_name else defaults["friction"],
                }
                materials_group.add_child(mat_node)
                created[mat_name] = mat_node

            entity_node = context.entity_node_index.get(entity.id)
            if entity_node is not None:
                entity_node.components["material_ref"] = created[mat_name].node_uuid

        context.statistics.material_count = len(created)
        context.info(f"Built {len(created)} unique material node(s).", self.name)


class PhysicsBuilder:
    """Attaches physics-body metadata (mass, restitution, friction, forces) to each entity node."""

    name = "physics_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        physics_group = root.add_child(SceneNode(name="Physics", node_type=NodeType.PHYSICS_GROUP))

        for entity in context.world_spec.entities:
            entity_node = context.entity_node_index.get(entity.id)
            if entity_node is None:
                continue

            body_node = SceneNode(
                name=f"{entity_node.name}_physics",
                node_type=NodeType.PHYSICS_BODY,
            )
            body_node.components["physics_body"] = {
                "body_type": "static" if entity.is_static else "dynamic",
                "mass_kg": entity.mass,
                "restitution": entity.restitution,
                "friction": entity.friction,
                "forces": entity.forces,
                "constraints": entity.constraints,
            }
            physics_group.add_child(body_node)
            entity_node.components["physics_ref"] = body_node.node_uuid

            if not entity.is_static and entity.mass <= 0:
                context.warning(
                    f"Dynamic entity '{entity.id}' has non-positive mass ({entity.mass} kg).",
                    self.name,
                    entity_ref=entity.id,
                )

        context.info(f"Attached physics metadata to {len(context.world_spec.entities)} entity node(s).", self.name)


class SensorBuilder:
    """Builds the Sensors group.

    The current `WorldSpec` data contract (world_spec.py) does not define
    per-entity sensor payloads, so this stage is a structurally-present
    no-op: it creates the `Sensors` group node (so downstream exporters
    and future WorldSpec extensions have a stable attachment point) and
    records zero sensors. This is the integration point for future
    SensorSpec support.
    """

    name = "sensor_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        sensors_group = root.add_child(SceneNode(name="Sensors", node_type=NodeType.SENSORS_GROUP))
        context.statistics.sensor_count = len(sensors_group.children)
        context.info("Sensors group initialized (0 sensors -- not present in current WorldSpec contract).", self.name)


class RelationshipBuilder:
    """Encodes WorldSpec `Interaction` entries as edges between entity nodes."""

    name = "relationship_builder"

    def build(self, context: CompilationContext) -> None:
        count = 0
        for interaction in context.world_spec.interactions:
            a_node = context.entity_node_index.get(interaction.entity_a)
            if a_node is None:
                context.warning(
                    f"Interaction references unknown entity_a '{interaction.entity_a}'.",
                    self.name,
                )
                continue

            target_id = interaction.entity_b
            if target_id != "environment" and target_id not in context.entity_node_index:
                context.warning(
                    f"Interaction references unknown entity_b '{target_id}'.",
                    self.name,
                )
                continue

            edge = {
                "type": interaction.type,
                "target": target_id,
                "parameters": interaction.parameters,
            }
            a_node.components.setdefault("relationships", []).append(edge)
            count += 1

        context.statistics.relationship_count = count
        context.info(f"Built {count} relationship edge(s).", self.name)


# ════════════════════════════════════════════════════════════════════════
# USD Export
# ════════════════════════════════════════════════════════════════════════

class Exporter(ABC):
    """Abstract export backend. Implementations turn a SceneGraph into bytes on disk."""

    name: str = "exporter"

    @abstractmethod
    def export(self, scene_graph: SceneGraph, output_path: Path, context: CompilationContext) -> Path:
        """Write `scene_graph` to `output_path` and return the path actually written."""


class USDAsciiExporter(Exporter):
    """Exports the Scene Graph as ASCII OpenUSD (.usda).

    Uses the official `pxr` (OpenUSD) Python bindings when available for
    a fully spec-compliant stage. When `pxr` is not installed in the
    current environment, falls back to hand-emitting valid USDA text --
    the USDA format is a documented, human-readable text format, so this
    fallback produces a real, loadable .usda file rather than a stub.
    """

    name = "usd_ascii_exporter"

    def export(self, scene_graph: SceneGraph, output_path: Path, context: CompilationContext) -> Path:
        try:
            from pxr import Usd, UsdGeom  # type: ignore

            return self._export_with_pxr(scene_graph, output_path, Usd, UsdGeom)
        except ImportError:
            context.warning(
                "`pxr` (OpenUSD) bindings not found in this environment; falling back to a "
                "hand-emitted USDA writer. Install `usd-core` for full OpenUSD fidelity.",
                self.name,
            )
            return self._export_manual(scene_graph, output_path, context)

    # ── pxr-backed path ─────────────────────────────────────────────

    def _export_with_pxr(self, scene_graph: SceneGraph, output_path: Path, Usd, UsdGeom) -> Path:  # noqa: N803
        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        def visit(node: SceneNode, parent_path: str) -> None:
            prim_path = f"{parent_path}/{node._safe_name()}" if parent_path != "/" else f"/{node._safe_name()}"
            xform = UsdGeom.Xform.Define(stage, prim_path)
            t = node.transform
            xform.AddTranslateOp().Set(tuple(t.translation))
            xform.AddRotateXYZOp().Set(tuple(r * 57.29577951308232 for r in t.rotation_euler_rad))
            xform.AddScaleOp().Set(tuple(t.scale))
            prim = xform.GetPrim()
            prim.SetActive(node.enabled)
            UsdGeom.Imageable(prim).MakeVisible() if node.visible else UsdGeom.Imageable(prim).MakeInvisible()
            for key, value in node.metadata.items():
                try:
                    prim.SetCustomDataByKey(key, value)
                except Exception:  # noqa: BLE001 - metadata best-effort only
                    pass
            for child in node.children:
                visit(child, prim_path)

        visit(scene_graph.root, "/")
        stage.GetRootLayer().Save()
        return output_path

    # ── manual fallback path ────────────────────────────────────────

    def _export_manual(self, scene_graph: SceneGraph, output_path: Path, context: CompilationContext) -> Path:
        lines: list[str] = []
        lines.append('#usda 1.0')
        lines.append("(")
        lines.append(f'    doc = "Generated by PhysWorldLM SceneCompiler"')
        lines.append(f'    upAxis = "{("Y" if context.config.coordinate_system is CoordinateSystem.Y_UP else "Z")}"')
        lines.append(")")
        lines.append("")

        def emit(node: SceneNode, indent: int) -> None:
            pad = "    " * indent
            t = node.transform
            lines.append(f'{pad}def Xform "{node._safe_name()}"')
            lines.append(f"{pad}{{")
            inner = "    " * (indent + 1)
            lines.append(f"{inner}double3 xformOp:translate = {tuple(t.translation)}")
            lines.append(f"{inner}double3 xformOp:rotateXYZ = {tuple(t.rotation_euler_rad)}")
            lines.append(f"{inner}double3 xformOp:scale = {tuple(t.scale)}")
            lines.append(
                f"{inner}uniform token[] xformOpOrder = "
                f'["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]'
            )
            lines.append(f"{inner}bool active = {str(node.enabled).lower()}")
            lines.append(f'{inner}token visibility = "{"inherited" if node.visible else "invisible"}"')
            if node.metadata:
                lines.append(f"{inner}customData = {{")
                for key, value in node.metadata.items():
                    lines.append(f'{inner}    string {key} = "{value}"')
                lines.append(f"{inner}}}")
            for child in node.children:
                emit(child, indent + 1)
            lines.append(f"{pad}}}")

        emit(scene_graph.root, 0)

        if context.config.overwrite_existing or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            raise ExportError(f"Output path already exists and overwrite is disabled: {output_path}")

        return output_path


# ════════════════════════════════════════════════════════════════════════
# SceneCompiler
# ════════════════════════════════════════════════════════════════════════

class SceneCompiler:
    """The central orchestration engine of PhysWorldLM.

    `SceneCompiler` transforms a validated `WorldSpec` into an internal,
    backend-independent Scene Graph, then coordinates export of that
    graph to OpenUSD. It performs no rendering or simulation itself --
    it is purely an orchestrator that runs a fixed sequence of
    independent, pluggable builder stages (see `CompilationStage`).

    Example:
        >>> compiler = SceneCompiler()
        >>> report = compiler.compile(world_spec, output_path="scene.usda")
        >>> print(report)
        >>> report.success
        True

    Custom builders or exporters can be supplied at construction time to
    extend or override default behavior without modifying this class::

        >>> compiler = SceneCompiler(
        ...     config=CompilerConfig(validation_mode=ValidationMode.PERMISSIVE),
        ... )
        >>> compiler.registry.register(MyCustomSensorBuilder())
    """

    def __init__(
        self,
        config: Optional[CompilerConfig] = None,
        exporter: Optional[Exporter] = None,
    ) -> None:
        """Initialize the compiler with optional configuration and exporter override.

        Args:
            config: Compiler-wide settings. Defaults to `CompilerConfig()`.
            exporter: The export backend used at the EXPORT_USD stage.
                Defaults to `USDAsciiExporter`. Supplying a different
                `Exporter` implementation is the supported way to target
                a different backend (binary USD, USDZ, a custom Nucleus
                uploader, etc.) without modifying `SceneCompiler`.
        """
        self.config = config or CompilerConfig()
        self.exporter = exporter or USDAsciiExporter()
        self.registry = BuilderRegistry()
        self._register_default_builders()

    # ── public API ───────────────────────────────────────────────────

    def compile(self, world_spec: WorldSpec, output_path: str | Path) -> CompilationReport:
        """Compile `world_spec` into an OpenUSD scene at `output_path`.

        This is the single public entry point of the Scene Compiler.

        Args:
            world_spec: A validated `WorldSpec` instance describing the
                world to compile.
            output_path: Destination path for the exported OpenUSD scene
                (e.g. "scene.usda").

        Returns:
            A `CompilationReport` describing the outcome, statistics, and
            any diagnostics collected during compilation.

        Raises:
            ValidationError: If `world_spec` fails validation under
                `ValidationMode.STRICT`.
            CompilationError: For any other unrecoverable failure during
                compilation (wraps the originating cause where possible).
        """
        output_path = Path(output_path)
        context = CompilationContext(world_spec=world_spec, config=self.config, builder_registry=self.registry)

        logger.info("=" * 72)
        logger.info("Starting compilation of scene '%s'", world_spec.scene_id)
        logger.info("=" * 72)

        try:
            self._run_stage(context, CompilationStage.VALIDATE_WORLD_SPEC, self._stage_validate_world_spec)
            self._run_stage(context, CompilationStage.CREATE_CONTEXT, self._stage_create_context)
            self._run_stage(context, CompilationStage.INIT_SCENE_GRAPH, self._stage_init_scene_graph)
            self._run_stage(context, CompilationStage.BUILD_WORLD_ROOT, self._stage_build_world_root)
            self._run_stage(context, CompilationStage.BUILD_ENVIRONMENT, self._stage_build_environment)
            self._run_stage(context, CompilationStage.BUILD_ENTITIES, self._stage_build_entities)
            self._run_stage(context, CompilationStage.APPLY_TRANSFORMS, self._stage_apply_transforms)
            self._run_stage(context, CompilationStage.RESOLVE_ASSETS, self._stage_resolve_assets)
            self._run_stage(context, CompilationStage.ASSIGN_MATERIALS, self._stage_assign_materials)
            self._run_stage(context, CompilationStage.ATTACH_PHYSICS, self._stage_attach_physics)
            self._run_stage(context, CompilationStage.CONFIGURE_SENSORS, self._stage_configure_sensors)
            self._run_stage(context, CompilationStage.BUILD_RELATIONSHIPS, self._stage_build_relationships)
            self._run_stage(context, CompilationStage.GENERATE_METADATA, self._stage_generate_metadata)
            self._run_stage(
                context,
                CompilationStage.EXPORT_USD,
                lambda ctx: self._stage_export_usd(ctx, output_path),
            )
            report = self._run_stage(
                context,
                CompilationStage.PRODUCE_REPORT,
                lambda ctx: self._stage_produce_report(ctx, output_path),
            )
        except CompilationError as exc:
            context.error(f"Compilation aborted: {exc}", "scene_compiler")
            report = self._build_failure_report(context, output_path)
            logger.error("Compilation FAILED for scene '%s': %s", world_spec.scene_id, exc)
            return report

        logger.info("Compilation finished for scene '%s' -> status=%s", world_spec.scene_id, report.status.name)
        return report

    # ── stage runner ─────────────────────────────────────────────────

    def _run_stage(
        self,
        context: CompilationContext,
        stage: CompilationStage,
        fn: Callable[[CompilationContext], Any],
    ) -> Any:
        context.assert_dependencies_met(stage)
        context.current_stage = stage
        start = time.monotonic()
        logger.info("Stage start    : %s", stage.label)
        result = fn(context)
        duration = time.monotonic() - start
        context.mark_stage_complete(stage, duration)
        logger.info("Stage complete : %-22s duration=%.4fs", stage.label, duration)
        return result

    # ── stage implementations ───────────────────────────────────────
    # Each stage is an independent internal method. Stages call into the
    # BuilderRegistry rather than embedding builder logic directly.

    def _stage_validate_world_spec(self, context: CompilationContext) -> None:
        if context.config.validation_mode is ValidationMode.DISABLED:
            context.info("Validation disabled by configuration; skipping.", "scene_compiler")
            return

        errors: list[str] = []
        ws = context.world_spec

        if not ws.scene_id:
            errors.append("WorldSpec.scene_id must be a non-empty string.")

        seen_ids: set[str] = set()
        for entity in ws.entities:
            if not entity.id:
                errors.append("Entity found with empty id.")
                continue
            if entity.id in seen_ids:
                errors.append(f"Duplicate entity id: '{entity.id}'.")
            seen_ids.add(entity.id)
            if not entity.is_static and entity.mass <= 0:
                errors.append(f"Dynamic entity '{entity.id}' must have mass > 0 (got {entity.mass}).")
            for axis_val, axis_name in (
                (entity.bounding_box.width, "width"),
                (entity.bounding_box.height, "height"),
                (entity.bounding_box.depth, "depth"),
            ):
                if axis_val <= 0:
                    errors.append(f"Entity '{entity.id}' has non-positive bounding_box.{axis_name}.")

        entity_ids = {e.id for e in ws.entities}
        for interaction in ws.interactions:
            if interaction.entity_a not in entity_ids:
                errors.append(f"Interaction references unknown entity_a '{interaction.entity_a}'.")
            if interaction.entity_b != "environment" and interaction.entity_b not in entity_ids:
                errors.append(f"Interaction references unknown entity_b '{interaction.entity_b}'.")

        if ws.simulation_graph.dt <= 0:
            errors.append("SimulationGraph.dt must be > 0.")
        if ws.simulation_graph.duration <= 0:
            errors.append("SimulationGraph.duration must be > 0.")

        if errors:
            if context.config.validation_mode is ValidationMode.STRICT:
                raise ValidationError("; ".join(errors))
            for err in errors:
                context.error(err, "validator")

        context.info(f"Validated WorldSpec '{ws.scene_id}' ({len(ws.entities)} entities).", "validator")

    def _stage_create_context(self, context: CompilationContext) -> None:
        # The context object already exists by construction; this stage
        # exists explicitly (per the pipeline contract) as the seam where
        # future per-run setup (e.g. plugin discovery, telemetry hooks)
        # is attached.
        context.info("Compilation context created.", "scene_compiler")

    def _stage_init_scene_graph(self, context: CompilationContext) -> None:
        context.scene_graph = SceneGraph()
        context.info("Scene Graph initialized.", "scene_compiler")

    def _stage_build_world_root(self, context: CompilationContext) -> None:
        self.registry.invoke(StageBuilder.name, context)

    def _stage_build_environment(self, context: CompilationContext) -> None:
        self.registry.invoke(EnvironmentBuilder.name, context)

    def _stage_build_entities(self, context: CompilationContext) -> None:
        self.registry.invoke(EntityBuilder.name, context)

    def _stage_apply_transforms(self, context: CompilationContext) -> None:
        self.registry.invoke(TransformBuilder.name, context)

    def _stage_resolve_assets(self, context: CompilationContext) -> None:
        self.registry.invoke(AssetResolver.name, context)

    def _stage_assign_materials(self, context: CompilationContext) -> None:
        self.registry.invoke(MaterialBuilder.name, context)

    def _stage_attach_physics(self, context: CompilationContext) -> None:
        self.registry.invoke(PhysicsBuilder.name, context)

    def _stage_configure_sensors(self, context: CompilationContext) -> None:
        self.registry.invoke(SensorBuilder.name, context)

    def _stage_build_relationships(self, context: CompilationContext) -> None:
        self.registry.invoke(RelationshipBuilder.name, context)

    def _stage_generate_metadata(self, context: CompilationContext) -> None:
        if not context.config.generate_metadata:
            context.info("Metadata generation disabled by configuration.", "scene_compiler")
            return
        root = context.scene_graph.root
        metadata_node = root.add_child(SceneNode(name="Metadata", node_type=NodeType.METADATA))
        metadata_node.metadata.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "compiler": "PhysWorldLM.SceneCompiler",
                "scene_node_count": context.scene_graph.node_count(),
                "simulation_graph": context.world_spec.simulation_graph.to_dict(),
            }
        )
        context.info("Metadata node generated.", "scene_compiler")

    def _stage_export_usd(self, context: CompilationContext, output_path: Path) -> None:
        if output_path.exists() and not context.config.overwrite_existing:
            raise ExportError(f"Refusing to overwrite existing file (overwrite disabled): {output_path}")
        try:
            written_path = self.exporter.export(context.scene_graph, output_path, context)
        except ExportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExportError(f"Export failed: {exc}") from exc

        context.statistics.exported_file_size_bytes = written_path.stat().st_size if written_path.exists() else 0
        context.info(f"Exported OpenUSD scene to '{written_path}'.", "exporter")

    def _stage_produce_report(self, context: CompilationContext, output_path: Path) -> CompilationReport:
        context.statistics.compilation_time_s = time.monotonic() - context.started_at
        context.statistics.success = not context.has_errors()

        if context.has_errors():
            status = CompilationStatus.FAILED
        elif context.statistics.warning_count > 0:
            status = CompilationStatus.SUCCESS_WITH_WARNINGS
        else:
            status = CompilationStatus.SUCCESS

        return CompilationReport(
            status=status,
            scene_id=context.world_spec.scene_id,
            output_path=output_path if status is not CompilationStatus.FAILED else None,
            statistics=context.statistics,
            diagnostics=context.diagnostics,
            scene_graph=context.scene_graph,
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _register_default_builders(self) -> None:
        for builder in (
            StageBuilder(),
            EnvironmentBuilder(),
            EntityBuilder(),
            TransformBuilder(),
            AssetResolver(),
            MaterialBuilder(),
            PhysicsBuilder(),
            SensorBuilder(),
            RelationshipBuilder(),
        ):
            self.registry.register(builder)

    def _build_failure_report(self, context: CompilationContext, output_path: Path) -> CompilationReport:
        context.statistics.compilation_time_s = time.monotonic() - context.started_at
        context.statistics.success = False
        return CompilationReport(
            status=CompilationStatus.FAILED,
            scene_id=context.world_spec.scene_id if context.world_spec else "",
            output_path=None,
            statistics=context.statistics,
            diagnostics=context.diagnostics,
            scene_graph=context.scene_graph,
        )


__all__ = [
    "SceneCompiler",
    "CompilerConfig",
    "CompilationReport",
    "CompilationStatistics",
    "CompilationStatus",
    "CompilationStage",
    "CompilationContext",
    "SceneGraph",
    "SceneNode",
    "NodeType",
    "Transform",
    "Diagnostic",
    "DiagnosticSeverity",
    "Builder",
    "BuilderRegistry",
    "Exporter",
    "USDAsciiExporter",
    "ExportFormat",
    "OptimizationLevel",
    "ValidationMode",
    "CoordinateSystem",
    "UnitSystem",
    "CompilationError",
    "ValidationError",
    "BuilderError",
    "ExportError",
    "DependencyError",
    "AssetResolutionError",
]
