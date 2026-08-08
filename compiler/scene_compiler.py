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

logger = logging.getLogger("physworldlm.scene_compiler")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class CompilationError(Exception):
    pass


class ValidationError(CompilationError):
    pass


class BuilderError(CompilationError):
    def __init__(self, builder_name: str, message: str, *, cause: Optional[Exception] = None):
        self.builder_name = builder_name
        self.cause = cause
        super().__init__(f"[{builder_name}] {message}")


class ExportError(CompilationError):
    pass


class DependencyError(CompilationError):
    pass


class AssetResolutionError(CompilationError):
    pass


class CompilationStage(Enum):
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


STAGE_DEPENDENCIES: dict = {
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
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class CompilationStatus(Enum):
    SUCCESS = auto()
    SUCCESS_WITH_WARNINGS = auto()
    FAILED = auto()


class NodeType(Enum):
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


@dataclass
class Diagnostic:
    stage: CompilationStage
    severity: DiagnosticSeverity
    message: str
    source_module: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entity_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {"stage": self.stage.label, "severity": self.severity.name, "message": self.message, "source_module": self.source_module, "timestamp": self.timestamp.isoformat(), "entity_ref": self.entity_ref}

    def __str__(self) -> str:
        ref = f" (entity={self.entity_ref})" if self.entity_ref else ""
        return f"[{self.severity.name}] {self.stage.label} :: {self.source_module} :: {self.message}{ref}"


@dataclass
class Transform:
    translation: tuple = (0.0, 0.0, 0.0)
    rotation_euler_rad: tuple = (0.0, 0.0, 0.0)
    scale: tuple = (1.0, 1.0, 1.0)

    def to_dict(self) -> dict:
        return {"translation": list(self.translation), "rotation_euler_rad": list(self.rotation_euler_rad), "scale": list(self.scale)}


@dataclass
class SceneNode:
    name: str
    node_type: NodeType
    node_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional["SceneNode"] = field(default=None, repr=False)
    children: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    enabled: bool = True

    def add_child(self, child: "SceneNode") -> "SceneNode":
        child.parent = self
        self.children.append(child)
        return child

    def find(self, predicate):
        if predicate(self):
            return self
        for child in self.children:
            found = child.find(predicate)
            if found is not None:
                return found
        return None

    def find_by_uuid(self, node_uuid: str):
        return self.find(lambda n: n.node_uuid == node_uuid)

    def walk(self):
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    @property
    def path(self) -> str:
        if self.parent is None:
            return f"/{self._safe_name()}"
        return f"{self.parent.path}/{self._safe_name()}"

    def _safe_name(self) -> str:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in self.name)
        if not safe or safe[0].isdigit():
            safe = f"_{safe}"
        return safe

    def to_dict(self) -> dict:
        return {"uuid": self.node_uuid, "name": self.name, "type": self.node_type.name, "path": self.path, "visible": self.visible, "enabled": self.enabled, "transform": self.transform.to_dict(), "components": list(self.components.keys()), "metadata": self.metadata, "children": [c.to_dict() for c in self.children]}


@dataclass
class SceneGraph:
    root: SceneNode = field(default_factory=lambda: SceneNode(name="World", node_type=NodeType.WORLD))

    def node_count(self) -> int:
        return len(self.root.walk())

    def nodes_of_type(self, node_type: NodeType):
        return [n for n in self.root.walk() if n.node_type == node_type]

    def to_dict(self) -> dict:
        return self.root.to_dict()


class ExportFormat(Enum):
    USDA = "usda"
    USDC = "usdc"
    USDZ = "usdz"


class OptimizationLevel(Enum):
    NONE = auto()
    BASIC = auto()
    AGGRESSIVE = auto()


class ValidationMode(Enum):
    STRICT = auto()
    PERMISSIVE = auto()
    DISABLED = auto()


class CoordinateSystem(Enum):
    Y_UP = "y_up"
    Z_UP = "z_up"


class UnitSystem(Enum):
    SI_METERS = "meters_kilograms_seconds"


@dataclass
class CompilerConfig:
    export_format: ExportFormat = ExportFormat.USDA
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    validation_mode: ValidationMode = ValidationMode.STRICT
    log_level: str = "INFO"
    overwrite_existing: bool = True
    asset_search_paths: list = field(default_factory=list)
    coordinate_system: CoordinateSystem = CoordinateSystem.Y_UP
    unit_system: UnitSystem = UnitSystem.SI_METERS
    generate_metadata: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))


@dataclass
class CompilationStatistics:
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
    stage_durations_s: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"compilation_time_s": round(self.compilation_time_s, 6), "entity_count": self.entity_count, "relationship_count": self.relationship_count, "asset_count": self.asset_count, "material_count": self.material_count, "sensor_count": self.sensor_count, "environment_object_count": self.environment_object_count, "warning_count": self.warning_count, "error_count": self.error_count, "exported_file_size_bytes": self.exported_file_size_bytes, "success": self.success, "stage_durations_s": {k: round(v, 6) for k, v in self.stage_durations_s.items()}}


@dataclass
class CompilationContext:
    world_spec: WorldSpec
    config: CompilerConfig
    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    builder_registry: "BuilderRegistry" = field(default=None)
    diagnostics: list = field(default_factory=list)
    statistics: CompilationStatistics = field(default_factory=CompilationStatistics)
    asset_registry: dict = field(default_factory=dict)
    completed_stages: set = field(default_factory=set)
    current_stage: Optional[CompilationStage] = None
    entity_node_index: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    sensor_manager: Optional[Any] = None

    def log_diagnostic(self, severity, message, source_module, entity_ref=None):
        diag = Diagnostic(stage=self.current_stage or CompilationStage.VALIDATE_WORLD_SPEC, severity=severity, message=message, source_module=source_module, entity_ref=entity_ref)
        self.diagnostics.append(diag)
        if severity is DiagnosticSeverity.WARNING:
            self.statistics.warning_count += 1
        elif severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL):
            self.statistics.error_count += 1
        return diag

    def info(self, message, source_module, entity_ref=None):
        self.log_diagnostic(DiagnosticSeverity.INFO, message, source_module, entity_ref)

    def warning(self, message, source_module, entity_ref=None):
        self.log_diagnostic(DiagnosticSeverity.WARNING, message, source_module, entity_ref)

    def error(self, message, source_module, entity_ref=None):
        self.log_diagnostic(DiagnosticSeverity.ERROR, message, source_module, entity_ref)

    def has_errors(self) -> bool:
        return self.statistics.error_count > 0

    def mark_stage_complete(self, stage, duration_s):
        self.completed_stages.add(stage)
        self.statistics.stage_durations_s[stage.label] = duration_s

    def assert_dependencies_met(self, stage):
        missing = [dep for dep in STAGE_DEPENDENCIES.get(stage, ()) if dep not in self.completed_stages]
        if missing:
            raise DependencyError(f"Cannot run stage '{stage.label}': missing prerequisite stage(s) {[m.label for m in missing]}.")


@dataclass
class CompilationReport:
    status: CompilationStatus
    scene_id: str
    output_path: Optional[Path]
    statistics: CompilationStatistics
    diagnostics: list
    scene_graph: Optional[SceneGraph] = None
    sensor_manager: Optional[Any] = None

    @property
    def success(self) -> bool:
        return self.status is not CompilationStatus.FAILED

    def errors(self):
        return [d for d in self.diagnostics if d.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)]

    def warnings(self):
        return [d for d in self.diagnostics if d.severity is DiagnosticSeverity.WARNING]


@runtime_checkable
class Builder(Protocol):
    name: str

    def build(self, context: CompilationContext) -> None: ...


class BuilderRegistry:
    def __init__(self) -> None:
        self._builders: dict = {}

    def register(self, builder) -> None:
        self._builders[builder.name] = builder

    def get(self, name: str):
        return self._builders[name]

    def invoke(self, name: str, context) -> None:
        builder = self.get(name)
        try:
            builder.build(context)
        except CompilationError:
            raise
        except Exception as exc:
            raise BuilderError(name, f"Unhandled exception during build: {exc}", cause=exc) from exc


class StageBuilder:
    name = "stage_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        root.metadata.update({"scene_id": context.world_spec.scene_id, "description": context.world_spec.description, "coordinate_system": context.config.coordinate_system.value, "unit_system": context.config.unit_system.value})


class EnvironmentBuilder:
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
        atmosphere_node.metadata.update({"temperature_K": env.temperature_K, "pressure_Pa": env.pressure_Pa, "air_density": env.air_density})
        weather_node = env_node.add_child(SceneNode(name="Weather", node_type=NodeType.WEATHER))
        weather_node.metadata.update({"weather": env.weather, "wind_speed_ms": env.wind.speed, "wind_direction_rad": env.wind.direction})
        lighting_node = env_node.add_child(SceneNode(name="Lighting", node_type=NodeType.LIGHTING))
        lighting_node.metadata["time_of_day"] = env.time_of_day
        context.statistics.environment_object_count = len(env_node.children)


class EntityBuilder:
    name = "entity_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        entities_group = root.add_child(SceneNode(name="Entities", node_type=NodeType.ENTITIES_GROUP))
        for entity in context.world_spec.entities:
            node = SceneNode(name=entity.label or entity.id, node_type=NodeType.ENTITY, node_uuid=self._stable_uuid(entity.id) if context.config.deterministic else str(uuid.uuid4()))
            node.metadata.update({"world_spec_id": entity.id, "entity_type": entity.entity_type, "is_static": entity.is_static, "tags": list(entity.tags)})
            node.components["entity"] = entity.to_dict()
            entities_group.add_child(node)
            context.entity_node_index[entity.id] = node
        context.statistics.entity_count = len(context.world_spec.entities)

    @staticmethod
    def _stable_uuid(seed: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"physworldlm://entity/{seed}"))


class TransformBuilder:
    name = "transform_builder"

    def build(self, context: CompilationContext) -> None:
        for entity in context.world_spec.entities:
            node = context.entity_node_index.get(entity.id)
            if node is None:
                continue
            pos = entity.state.position
            rot = entity.state.orientation
            bbox = entity.bounding_box
            node.transform = Transform(translation=(pos.x, pos.y, pos.z), rotation_euler_rad=(rot.x, rot.y, rot.z), scale=(bbox.width, bbox.height, bbox.depth))
            node.components["kinematics"] = entity.state.to_dict()


class AssetResolver:
    name = "asset_resolver"
    ASSET_TAG_PREFIX = "asset:"

    def build(self, context: CompilationContext) -> None:
        for entity in context.world_spec.entities:
            for tag in entity.tags:
                if not tag.startswith(self.ASSET_TAG_PREFIX):
                    continue
                ref = tag[len(self.ASSET_TAG_PREFIX):]
                path = self._resolve(ref, context)
                context.asset_registry[ref] = path
        context.statistics.asset_count = len(context.asset_registry)

    def _resolve(self, ref, context):
        candidate = Path(ref)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for search_path in context.config.asset_search_paths:
            full = search_path / ref
            if full.exists():
                return full
        return candidate


class MaterialBuilder:
    name = "material_builder"

    def build(self, context: CompilationContext) -> None:
        from world_spec import MATERIAL_DEFAULTS
        root = context.scene_graph.root
        materials_group = root.add_child(SceneNode(name="Materials", node_type=NodeType.MATERIALS_GROUP))
        created: dict = {}
        for entity in context.world_spec.entities:
            mat_name = entity.material
            if mat_name not in created:
                defaults = MATERIAL_DEFAULTS.get(mat_name, MATERIAL_DEFAULTS["generic"])
                mat_node = SceneNode(name=mat_name, node_type=NodeType.MATERIAL)
                mat_node.components["material"] = {"name": mat_name, "density": defaults["density"], "restitution": entity.restitution if entity.material == mat_name else defaults["restitution"], "friction": entity.friction if entity.material == mat_name else defaults["friction"]}
                materials_group.add_child(mat_node)
                created[mat_name] = mat_node
            entity_node = context.entity_node_index.get(entity.id)
            if entity_node is not None:
                entity_node.components["material_ref"] = created[mat_name].node_uuid
        context.statistics.material_count = len(created)


class PhysicsBuilder:
    name = "physics_builder"

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        physics_group = root.add_child(SceneNode(name="Physics", node_type=NodeType.PHYSICS_GROUP))
        for entity in context.world_spec.entities:
            entity_node = context.entity_node_index.get(entity.id)
            if entity_node is None:
                continue
            body_node = SceneNode(name=f"{entity_node.name}_physics", node_type=NodeType.PHYSICS_BODY)
            body_node.components["physics_body"] = {"body_type": "static" if entity.is_static else "dynamic", "mass_kg": entity.mass, "restitution": entity.restitution, "friction": entity.friction, "forces": entity.forces, "constraints": entity.constraints}
            physics_group.add_child(body_node)
            entity_node.components["physics_ref"] = body_node.node_uuid


class SensorBuilder:
    """Instantiates every sensor declared in WorldSpec via the sensors/
    framework's SensorManager, validates each one, and attaches a SENSOR
    SceneNode per sensor as a CHILD of its owning entity's node.

    Design constraints honored (see prior review round for the full
    rationale — kept brief here):
      - No duplicated sensor logic: delegated 100% to SensorManager/Sensor.
      - Backend-independent: only serialize()d plain-dict data enters the
        SceneGraph, never a live Sensor object.
      - The live SensorManager itself is NOT discarded when this builder
        finishes — it's attached to `context.sensor_manager`, which
        SceneCompiler.compile() carries through to CompilationReport, so
        a caller can look up the SAME live Sensor objects (by the
        `sensor_id` now stored in each node's metadata) after
        compilation finishes, e.g. for a later capture()/simulation
        stage. Deliberately NOT owned by SceneCompiler.__init__ and
        reused across multiple compile() calls on the same instance —
        that would leak one scene's sensors into another scene's
        manager on repeated compile() calls. One manager per compile(),
        surfaced on the report instead of thrown away.
    """

    name = "sensor_builder"

    _RESERVED_PARAM_KEYS = frozenset({"sensor_id", "mount_transform", "enabled"})

    def build(self, context: CompilationContext) -> None:
        root = context.scene_graph.root
        sensors_group = root.add_child(SceneNode(name="Sensors", node_type=NodeType.SENSORS_GROUP))

        # Import fallback: the sensors/ package's own files (base_sensor.py,
        # camera.py, ...) use bare imports ("from base_sensor import
        # Sensor"), implying sensors/ is meant to sit directly on
        # sys.path rather than be imported as a "sensors.X" package. Bare
        # import first (matches that established convention); fall back
        # to a package-style import for callers who DO have it installed
        # as a proper package, so this doesn't hard-fail either way.
        SensorManager = None
        Transform6DoF = None
        import_error = None
        for import_style in ("bare", "package", "relative"):
            try:
                if import_style == "bare":
                    from sensor_manager import SensorManager as _SM
                    from sensor_types import Transform6DoF as _T6
                elif import_style == "package":
                    from sensors.sensor_manager import SensorManager as _SM
                    from sensors.sensor_types import Transform6DoF as _T6
                else:
                    from .sensor_manager import SensorManager as _SM  # type: ignore[import]
                    from .sensor_types import Transform6DoF as _T6  # type: ignore[import]
                SensorManager, Transform6DoF = _SM, _T6
                break
            except ImportError as exc:
                import_error = exc
                continue

        if SensorManager is None:
            context.warning(
                f"sensors/ framework not importable under any known layout "
                f"(bare / sensors.* / relative) — last error: {import_error}. "
                "Sensors group left empty.",
                self.name,
            )
            context.statistics.sensor_count = 0
            return

        manager = SensorManager()
        context.sensor_manager = manager  # survives past this compile() via CompilationReport
        built = 0

        print("ENTERED SENSOR BUILDER")
        print("Sensor list:", context.world_spec.all_sensors())

        seen_keys: set = set()  # (entity_id, sensor_name) — duplicate detection

        for entity, spec in context.world_spec.all_sensors():
            print("LOOP:", entity.id, spec.sensor_type, spec.name)
            dedup_key = (entity.id, spec.name)
            if dedup_key in seen_keys:
                context.warning(
                    f"Duplicate sensor declaration '{spec.name}' on entity '{entity.id}'; "
                    "skipping the duplicate.",
                    self.name,
                    entity_ref=entity.id,
                )
                continue
            seen_keys.add(dedup_key)

            entity_node = context.entity_node_index.get(entity.id)
            if entity_node is None:
                context.warning(
                    f"Sensor '{spec.name}' declared on unknown entity '{entity.id}'; skipping.",
                    self.name,
                    entity_ref=entity.id,
                )
                continue

            params = dict(spec.params or {})
            colliding_keys = self._RESERVED_PARAM_KEYS & params.keys()
            for key in colliding_keys:
                context.warning(
                    f"Sensor '{spec.name}' on entity '{entity.id}': params contained reserved "
                    f"key '{key}'; the SensorSpec-level value takes precedence.",
                    self.name,
                    entity_ref=entity.id,
                )
                params.pop(key)
                print("LOOP:", entity.id, spec.sensor_type, spec.name)
                print("BEFORE CREATE")

            try:
                sensor = manager.create(
                    spec.sensor_type,
                    spec.name,
                    sensor_id=spec.sensor_id,
                    enabled=spec.enabled,
                    **params,
                )

                print("AFTER CREATE")
                print("CREATED:", sensor.sensor_id)

            except Exception as exc:  # noqa: BLE001
                # DELIBERATELY broad, not narrowed to guessed exception
                # types: sensor_types.py's real exception hierarchy is
                # unknown (not available at the time this was written),
                # and a bad WorldSpec-declared sensor spec (unknown type,
                # invalid kwarg, whatever the concrete Sensor subclass
                # itself raises) must be a reportable diagnostic, not a
                # reason to abort the entire scene compilation. If/when
                # sensor_types.py's real exceptions are confirmed, narrow
                # this to that hierarchy specifically — but narrowing
                # against a GUESSED hierarchy would risk silently letting
                # a real construction failure crash compilation instead
                # of being caught here.

                print("CREATE FAILED:", repr(exc))
                raise

                context.warning(
                    f"Could not construct sensor '{spec.name}' (type='{spec.sensor_type}') "
                    f"for entity '{entity.id}': {exc}",
                    self.name,
                    entity_ref=entity.id,
                )
                continue

            mount_transform = None
            if spec.mount_transform:
                if isinstance(spec.mount_transform, dict):
                    mount_transform = Transform6DoF.from_dict(spec.mount_transform)
                else:
                    context.warning(
                        f"Sensor '{spec.name}' on entity '{entity.id}': mount_transform must be a "
                        f"dict (got {type(spec.mount_transform).__name__}); ignoring, using identity mount.",
                        self.name,
                        entity_ref=entity.id,
                    )
            manager.attach_sensor(sensor.sensor_id, entity.id, mount_transform)

            try:
                problems = sensor.validate()
            except Exception as exc:  # noqa: BLE001 — validate() itself misbehaving
                # shouldn't be able to crash compilation either.
                problems = [f"validate() raised {type(exc).__name__}: {exc}"]
            for problem in problems:
                context.warning(
                    f"Sensor '{spec.name}' on entity '{entity.id}': {problem}",
                    self.name,
                    entity_ref=entity.id,
                )

            sensor_node = SceneNode(name=spec.name, node_type=NodeType.SENSOR)
            if context.config.deterministic:
                sensor_node.node_uuid = self._stable_uuid(entity.id, spec.name)
            # Gap found while implementing USD export of sensors (this is
            # what "preserve mount transforms" in the exporter actually
            # reads): the SceneNode's own .transform was never populated
            # from spec.mount_transform, only the live Sensor object's
            # mount_transform was (via manager.attach_sensor above). USD
            # export walks node.transform, so without this the sensor
            # prim would always be authored at identity. Only
            # `translation` is extracted here — Transform6DoF's rotation
            # representation is unknown (sensor_types.py unavailable), so
            # rotation is left at identity rather than guessed; extend
            # once that file's real shape is confirmed.
            if spec.mount_transform and isinstance(spec.mount_transform, dict):
                t = spec.mount_transform.get("translation")
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    sensor_node.transform = Transform(translation=tuple(float(v) for v in t))
            sensor_node.components["sensor"] = sensor.serialize()
            sensor_node.metadata.update({
                "sensor_id": sensor.sensor_id,   # NEW — explicit SceneNode -> live Sensor lookup key,
                                                   # not buried inside components["sensor"]
                "sensor_type": spec.sensor_type,
                "owning_entity_id": entity.id,
            })
            sensor_node.enabled = spec.enabled
            entity_node.add_child(sensor_node)
            sensors_group.components.setdefault("sensor_refs", []).append(sensor_node.node_uuid)
            built += 1

        context.statistics.sensor_count = built
        context.info(f"Instantiated and attached {built} sensor(s) via SensorManager.", self.name)

    @staticmethod
    def _stable_uuid(entity_id: str, sensor_name: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"physworldlm://sensor/{entity_id}/{sensor_name}"))


class RelationshipBuilder:
    name = "relationship_builder"

    def build(self, context: CompilationContext) -> None:
        count = 0
        for interaction in context.world_spec.interactions:
            a_node = context.entity_node_index.get(interaction.entity_a)
            if a_node is None:
                continue
            target_id = interaction.entity_b
            if target_id != "environment" and target_id not in context.entity_node_index:
                continue
            edge = {"type": interaction.type, "target": target_id, "parameters": interaction.parameters}
            a_node.components.setdefault("relationships", []).append(edge)
            count += 1
        context.statistics.relationship_count = count


class Exporter(ABC):
    name: str = "exporter"

    @abstractmethod
    def export(self, scene_graph, output_path, context): ...


class USDAsciiExporter(Exporter):
    name = "usd_ascii_exporter"

    def export(self, scene_graph, output_path, context):
        try:
            from pxr import Usd, UsdGeom
            return self._export_with_pxr(scene_graph, output_path, Usd, UsdGeom)
        except ImportError:
            return self._export_manual(scene_graph, output_path, context)

    # ── NEW: sensor component flattening (shared by both export paths) ──
    #
    # SensorBuilder already attaches the sensor's full, backend-agnostic
    # config to node.components["sensor"] (whatever sensor.serialize()
    # returned — resolution, fov_deg, range, noise model, calibration,
    # whatever fields exist there, unknown to this exporter and not
    # duplicated here). This helper only FLATTENS that existing dict into
    # USD-attribute-safe string values; it never re-derives or
    # re-instantiates anything sensor-specific — no new sensor logic, no
    # Isaac Sim/PhysX APIs, just USD custom-attribute authoring.
    #
    # Nested dict/list values (e.g. a noise model sub-dict) are
    # JSON-encoded into a single string attribute rather than guessed at
    # field-by-field, since this exporter has no knowledge of what shape
    # sensor.serialize() actually returns for any given sensor type.
    @staticmethod
    def _sensor_custom_data(node) -> dict:
        if node.node_type is not NodeType.SENSOR:
            return {}
        sensor_data = node.components.get("sensor")
        if not isinstance(sensor_data, dict):
            return {}
        flattened = {}
        for key, value in sensor_data.items():
            if isinstance(value, (dict, list)):
                import json as _json
                flattened[f"sensor:{key}"] = _json.dumps(value)
            else:
                flattened[f"sensor:{key}"] = value
        return flattened

    def _export_with_pxr(self, scene_graph, output_path, Usd, UsdGeom):
        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        def visit(node, parent_path):
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
            # NEW: sensor component data, same best-effort custom-data
            # mechanism already used for node.metadata above.
            for key, value in self._sensor_custom_data(node).items():
                try:
                    prim.SetCustomDataByKey(key, value)
                except Exception:  # noqa: BLE001
                    pass
            for child in node.children:
                visit(child, prim_path)

        visit(scene_graph.root, "/")
        stage.GetRootLayer().Save()
        return output_path

    def _export_manual(self, scene_graph, output_path, context):
        lines = ['#usda 1.0', "(", '    doc = "Generated by PhysWorldLM SceneCompiler"', ")", ""]

        def emit(node, indent):
            pad = "    " * indent
            t = node.transform
            lines.append(f'{pad}def Xform "{node._safe_name()}"')
            lines.append(f"{pad}{{")
            inner = "    " * (indent + 1)
            # NEW: combine metadata + sensor component data into one
            # customData block, same mechanism, so a sensor node's data
            # shows up right alongside every other node's metadata.
            combined_data = dict(node.metadata)
            combined_data.update(self._sensor_custom_data(node))
            if combined_data:
                lines.append(f"{inner}customData = {{")
                for key, value in combined_data.items():
                    escaped = str(value).replace('"', '\\"')
                    lines.append(f'{inner}    string {key} = "{escaped}"')
                lines.append(f"{inner}}}")
            for child in node.children:
                emit(child, indent + 1)
            lines.append(f"{pad}}}")

        emit(scene_graph.root, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


class SceneCompiler:
    def __init__(self, config=None, exporter=None) -> None:
        self.config = config or CompilerConfig()
        self.exporter = exporter or USDAsciiExporter()
        self.registry = BuilderRegistry()
        self._register_default_builders()

    def compile(self, world_spec: WorldSpec, output_path) -> CompilationReport:
        output_path = Path(output_path)
        context = CompilationContext(world_spec=world_spec, config=self.config, builder_registry=self.registry)

        print("===== BEFORE SENSOR BUILDER =====")
        print("WorldSpec id:", id(context.world_spec))
        print("Sensors:", context.world_spec.all_sensors())
        print("===============================")

        try:
            self._run_stage(context, CompilationStage.VALIDATE_WORLD_SPEC, self._stage_validate_world_spec)
            self._run_stage(context, CompilationStage.CREATE_CONTEXT, lambda ctx: None)
            self._run_stage(context, CompilationStage.INIT_SCENE_GRAPH, self._stage_init_scene_graph)
            self._run_stage(context, CompilationStage.BUILD_WORLD_ROOT, lambda ctx: self.registry.invoke(StageBuilder.name, ctx))
            self._run_stage(context, CompilationStage.BUILD_ENVIRONMENT, lambda ctx: self.registry.invoke(EnvironmentBuilder.name, ctx))
            self._run_stage(context, CompilationStage.BUILD_ENTITIES, lambda ctx: self.registry.invoke(EntityBuilder.name, ctx))
            self._run_stage(context, CompilationStage.APPLY_TRANSFORMS, lambda ctx: self.registry.invoke(TransformBuilder.name, ctx))
            self._run_stage(context, CompilationStage.RESOLVE_ASSETS, lambda ctx: self.registry.invoke(AssetResolver.name, ctx))
            self._run_stage(context, CompilationStage.ASSIGN_MATERIALS, lambda ctx: self.registry.invoke(MaterialBuilder.name, ctx))
            self._run_stage(context, CompilationStage.ATTACH_PHYSICS, lambda ctx: self.registry.invoke(PhysicsBuilder.name, ctx))
            self._run_stage(context, CompilationStage.CONFIGURE_SENSORS, lambda ctx: self.registry.invoke(SensorBuilder.name, ctx))
            self._run_stage(context, CompilationStage.BUILD_RELATIONSHIPS, lambda ctx: self.registry.invoke(RelationshipBuilder.name, ctx))
            self._run_stage(context, CompilationStage.GENERATE_METADATA, self._stage_generate_metadata)
            self._run_stage(context, CompilationStage.EXPORT_USD, lambda ctx: self._stage_export_usd(ctx, output_path))
            report = self._run_stage(context, CompilationStage.PRODUCE_REPORT, lambda ctx: self._stage_produce_report(ctx, output_path))
        except CompilationError as exc:
            context.error(f"Compilation aborted: {exc}", "scene_compiler")
            return self._build_failure_report(context, output_path)
        return report

    def _run_stage(self, context, stage, fn):
        context.assert_dependencies_met(stage)
        context.current_stage = stage
        start = time.monotonic()
        result = fn(context)
        context.mark_stage_complete(stage, time.monotonic() - start)
        return result

    def _stage_validate_world_spec(self, context) -> None:
        ws = context.world_spec
        errors = []
        if not ws.scene_id:
            errors.append("scene_id empty")
        if errors and context.config.validation_mode is ValidationMode.STRICT:
            raise ValidationError("; ".join(errors))

    def _stage_init_scene_graph(self, context) -> None:
        context.scene_graph = SceneGraph()

    def _stage_generate_metadata(self, context) -> None:
        root = context.scene_graph.root
        metadata_node = root.add_child(SceneNode(name="Metadata", node_type=NodeType.METADATA))
        metadata_node.metadata.update({"generated_at": datetime.now(timezone.utc).isoformat(), "compiler": "PhysWorldLM.SceneCompiler", "scene_node_count": context.scene_graph.node_count()})

    def _stage_export_usd(self, context, output_path) -> None:
        written_path = self.exporter.export(context.scene_graph, output_path, context)
        context.statistics.exported_file_size_bytes = written_path.stat().st_size if written_path.exists() else 0

    def _stage_produce_report(self, context, output_path) -> CompilationReport:
        context.statistics.compilation_time_s = time.monotonic() - context.started_at
        context.statistics.success = not context.has_errors()
        status = CompilationStatus.FAILED if context.has_errors() else (CompilationStatus.SUCCESS_WITH_WARNINGS if context.statistics.warning_count > 0 else CompilationStatus.SUCCESS)
        return CompilationReport(status=status, scene_id=context.world_spec.scene_id, output_path=output_path if status is not CompilationStatus.FAILED else None, statistics=context.statistics, diagnostics=context.diagnostics, scene_graph=context.scene_graph, sensor_manager=context.sensor_manager)

    def _register_default_builders(self) -> None:
        for builder in (StageBuilder(), EnvironmentBuilder(), EntityBuilder(), TransformBuilder(), AssetResolver(), MaterialBuilder(), PhysicsBuilder(), SensorBuilder(), RelationshipBuilder()):
            self.registry.register(builder)

    def _build_failure_report(self, context, output_path) -> CompilationReport:
        context.statistics.compilation_time_s = time.monotonic() - context.started_at
        return CompilationReport(status=CompilationStatus.FAILED, scene_id=context.world_spec.scene_id if context.world_spec else "", output_path=None, statistics=context.statistics, diagnostics=context.diagnostics, scene_graph=context.scene_graph)
