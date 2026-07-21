"""
omniverse_runtime.py
══════════════════════════════════════════════════════════════════════════
Execution Layer of PhysWorldLM.

Pipeline position
------------------
    Natural Language
            │
            ▼
        Ontology
            │
            ▼
       WorldSpec
            │
            ▼
     Scene Compiler
            │
            ▼
       scene.usda
            │
            ▼
    ┌──────────────────┐
    │ OMNIVERSE RUNTIME │   <-- this module
    └──────────────────┘
            │
            ▼
     Live Simulation

Scope
-----
This module owns everything that happens *after* a `scene.usda` file has
been produced by the Scene Compiler: opening the stage in NVIDIA
Omniverse Kit, validating it, bringing up PhysX, discovering and
classifying the entities authored on the stage, wiring up the runtime
subsystems (simulation control, animation, cameras, asset streaming),
and driving the per-frame simulation/render loop.

This module deliberately does NOT implement:
    * Advanced PhysX tuning (solver iteration counts, CCD heuristics, ...)
    * Missile guidance / flight dynamics
    * Any form of AI, planning, or targeting logic
    * Domain-specific entity behavior

Those concerns belong to future subsystems that plug into the extension
points exposed here (see `simulation_controller.py`, `animation_system.py`,
`camera_controller.py`, `asset_loader.py`).

Public API
----------
    config = RuntimeConfig(usd_path=Path("scene.usda"))
    runtime = OmniverseRuntime(config)
    runtime.initialize()
    runtime.start()
    ...
    runtime.shutdown()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse_runtime")
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

class RuntimeInitializationError(Exception):
    """Base exception for all runtime initialization / execution failures."""


class StageLoadError(RuntimeInitializationError):
    """Raised when the target OpenUSD stage cannot be opened or fails validation."""


class PhysicsInitializationError(RuntimeInitializationError):
    """Raised when the PhysX subsystem cannot be brought up."""


class EntityDiscoveryError(RuntimeInitializationError):
    """Raised when entity discovery or classification fails."""


class RuntimeStateError(RuntimeInitializationError):
    """Raised when a runtime method is invoked while the runtime is in an
    incompatible lifecycle state (e.g. calling `pause()` before `start()`)."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class RendererBackend(Enum):
    """Supported Omniverse Kit renderer backends."""

    RTX_REALTIME = "rtx_realtime"
    RTX_PATHTRACING = "rtx_pathtracing"
    HYDRA_STORM = "hydra_storm"


class RuntimeState(Enum):
    """Lifecycle states of `OmniverseRuntime`."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    SHUTDOWN = auto()
    FAILED = auto()


class EntityCategory(Enum):
    """Domain-level entity categories recognized by the runtime.

    These map onto the entity taxonomy authored upstream by the Ontology
    and Scene Compiler stages and are used purely for classification and
    bookkeeping -- the runtime attaches no behavior to any category here.
    """

    AIRCRAFT = "aircraft"
    MISSILE = "missile"
    SHIP = "ship"
    VEHICLE = "vehicle"
    HUMAN = "human"
    BUILDING = "building"
    TERRAIN = "terrain"
    RADAR = "radar"
    CAMERA = "camera"
    SENSOR = "sensor"
    WEATHER = "weather"
    UNKNOWN = "unknown"


class MotionClass(Enum):
    """Whether an entity is kinematically static or dynamic at runtime."""

    STATIC = "static"
    DYNAMIC = "dynamic"


# Categories that are static by construction (cannot move during simulation
# regardless of how PhysX classifies their rigid-body type on the stage).
_STATIC_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.TERRAIN,
        EntityCategory.BUILDING,
        EntityCategory.RADAR,
    }
)

# Categories that are dynamic by construction.
_DYNAMIC_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.AIRCRAFT,
        EntityCategory.MISSILE,
        EntityCategory.SHIP,
        EntityCategory.VEHICLE,
        EntityCategory.HUMAN,
    }
)

# Categories that are runtime-managed but participate in neither bucket as
# a physical body (sensors, cameras, weather are systems, not bodies).
_SYSTEM_CATEGORIES: frozenset[EntityCategory] = frozenset(
    {
        EntityCategory.CAMERA,
        EntityCategory.SENSOR,
        EntityCategory.WEATHER,
    }
)


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeConfig:
    """User-configurable settings controlling runtime behavior.

    Attributes:
        usd_path: Path to the `scene.usda` (or `.usdc` / `.usdz`) file
            produced by the Scene Compiler.
        physics_enabled: Whether the PhysX subsystem should be brought up.
            When False, `initialize_physics()` is skipped and dynamic
            entities are discovered/classified but not simulated.
        headless: Whether Omniverse Kit should run without a viewport
            window (suitable for batch / server execution).
        renderer: Renderer backend to use.
        target_fps: Target simulation/render frame rate.
        physics_substeps: Number of PhysX substeps per simulation frame.
        gravity_ms2: Magnitude of gravitational acceleration, in m/s^2,
            applied along the stage's configured up-axis (negative).
        extension_paths: Additional Omniverse Kit extension search paths.
        stage_validation_strict: If True, any stage validation issue
            raises `StageLoadError`. If False, issues are logged as
            warnings and the runtime proceeds best-effort.
    """

    usd_path: Path
    physics_enabled: bool = True
    headless: bool = True
    renderer: RendererBackend = RendererBackend.RTX_REALTIME
    target_fps: float = 60.0
    physics_substeps: int = 1
    gravity_ms2: float = 9.81
    extension_paths: list[Path] = field(default_factory=list)
    stage_validation_strict: bool = True

    def __post_init__(self) -> None:
        self.usd_path = Path(self.usd_path)
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0 (got {self.target_fps}).")
        if self.physics_substeps < 1:
            raise ValueError(f"physics_substeps must be >= 1 (got {self.physics_substeps}).")

    @property
    def frame_duration_s(self) -> float:
        """Wall-clock duration of a single simulation frame, in seconds."""
        return 1.0 / self.target_fps


# ════════════════════════════════════════════════════════════════════════
# Entity model
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeEntity:
    """A single entity discovered on the USD stage at runtime.

    This is the runtime-side counterpart of a Scene Compiler `SceneNode`:
    it carries just enough information for the runtime systems (
    `simulation_controller`, `animation_system`, `camera_controller`) to
    do their job, without depending on Scene Compiler internals.

    Attributes:
        prim_path: USD prim path of the entity (e.g. "/World/Entities/F16_01").
        name: Human-readable entity name.
        category: Domain-level entity category.
        motion_class: STATIC or DYNAMIC classification.
        metadata: Free-form metadata copied from stage prim custom data.
    """

    prim_path: str
    name: str
    category: EntityCategory
    motion_class: MotionClass
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_static(self) -> bool:
        return self.motion_class is MotionClass.STATIC

    @property
    def is_dynamic(self) -> bool:
        return self.motion_class is MotionClass.DYNAMIC


@dataclass
class EntityRegistry:
    """Container for all entities discovered on the stage, indexed for
    fast lookup by category and motion class."""

    entities: dict[str, RuntimeEntity] = field(default_factory=dict)

    def add(self, entity: RuntimeEntity) -> None:
        self.entities[entity.prim_path] = entity

    def get(self, prim_path: str) -> Optional[RuntimeEntity]:
        return self.entities.get(prim_path)

    def by_category(self, category: EntityCategory) -> list[RuntimeEntity]:
        return [e for e in self.entities.values() if e.category is category]

    def by_motion_class(self, motion_class: MotionClass) -> list[RuntimeEntity]:
        return [e for e in self.entities.values() if e.motion_class is motion_class]

    @property
    def static_entities(self) -> list[RuntimeEntity]:
        return self.by_motion_class(MotionClass.STATIC)

    @property
    def dynamic_entities(self) -> list[RuntimeEntity]:
        return self.by_motion_class(MotionClass.DYNAMIC)

    def __len__(self) -> int:
        return len(self.entities)


# ════════════════════════════════════════════════════════════════════════
# Runtime statistics
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeStatistics:
    """Quantitative summary of the running simulation."""

    frame_count: int = 0
    simulation_time_s: float = 0.0
    wall_clock_time_s: float = 0.0
    static_entity_count: int = 0
    dynamic_entity_count: int = 0
    last_frame_duration_s: float = 0.0
    average_fps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "simulation_time_s": round(self.simulation_time_s, 6),
            "wall_clock_time_s": round(self.wall_clock_time_s, 6),
            "static_entity_count": self.static_entity_count,
            "dynamic_entity_count": self.dynamic_entity_count,
            "last_frame_duration_s": round(self.last_frame_duration_s, 6),
            "average_fps": round(self.average_fps, 3),
        }


# ════════════════════════════════════════════════════════════════════════
# Runtime subsystem protocol
# ════════════════════════════════════════════════════════════════════════

class RuntimeSubsystem:
    """Base contract for a pluggable runtime subsystem.

    Concrete subsystems (`SimulationController`, `AnimationSystem`,
    `CameraController`, `AssetLoader` -- defined in their own sibling
    modules per the project layout) implement this interface and are
    registered with `OmniverseRuntime.initialize_runtime_systems()`.

    Subclassing this is optional; any object exposing the same method
    signatures is structurally compatible.
    """

    name: str = "runtime_subsystem"

    def initialize(self, registry: EntityRegistry, config: RuntimeConfig) -> None:
        """Perform one-time setup. Called once, after entity discovery."""

    def update(self, dt: float, registry: EntityRegistry) -> None:
        """Advance subsystem state by `dt` seconds. Called once per frame."""

    def shutdown(self) -> None:
        """Release any resources held by the subsystem."""


class SubsystemRegistry:
    """Ordered registry of `RuntimeSubsystem` instances.

    Subsystems are invoked in registration order every frame, which lets
    the runtime guarantee a deterministic update sequence (e.g. physics
    before animation before cameras) without hard-coding subsystem
    classes inside `OmniverseRuntime`.
    """

    def __init__(self) -> None:
        self._subsystems: list[RuntimeSubsystem] = []

    def register(self, subsystem: RuntimeSubsystem) -> None:
        logger.debug("Registering runtime subsystem: %s", subsystem.name)
        self._subsystems.append(subsystem)

    def initialize_all(self, registry: EntityRegistry, config: RuntimeConfig) -> None:
        for subsystem in self._subsystems:
            logger.info("Initializing subsystem: %s", subsystem.name)
            subsystem.initialize(registry, config)

    def update_all(self, dt: float, registry: EntityRegistry) -> None:
        for subsystem in self._subsystems:
            subsystem.update(dt, registry)

    def shutdown_all(self) -> None:
        for subsystem in reversed(self._subsystems):
            logger.info("Shutting down subsystem: %s", subsystem.name)
            subsystem.shutdown()

    def __iter__(self):
        return iter(self._subsystems)

    def __len__(self) -> int:
        return len(self._subsystems)


# ════════════════════════════════════════════════════════════════════════
# OmniverseRuntime
# ════════════════════════════════════════════════════════════════════════

class OmniverseRuntime:
    """Main execution runtime of PhysWorldLM.

    `OmniverseRuntime` is responsible for everything between "a
    `scene.usda` file exists on disk" and "frames are being simulated and
    rendered". It owns the Omniverse Kit application lifecycle, the USD
    stage, the PhysX scene, entity discovery/classification, and the
    per-frame update/render loop.

    The class is intentionally agnostic to domain-specific behavior:
    aircraft do not fly themselves, missiles do not guide themselves --
    those concerns belong to subsystems registered through
    `initialize_runtime_systems()` (see `simulation_controller.py`,
    `animation_system.py`, `camera_controller.py`).

    Example:
        >>> config = RuntimeConfig(usd_path=Path("scene.usda"))
        >>> runtime = OmniverseRuntime(config)
        >>> runtime.initialize()
        >>> runtime.start()
        >>> runtime.shutdown()

    Custom subsystems can be supplied before `start()` is called::

        >>> runtime.subsystems.register(MySensorSubsystem())
    """

    def __init__(self, config: RuntimeConfig) -> None:
        """Initialize the runtime wrapper (does not touch Omniverse Kit yet).

        Args:
            config: Runtime configuration, including the path to the
                `scene.usda` file to load.
        """
        self.config = config
        self.state: RuntimeState = RuntimeState.UNINITIALIZED
        self.statistics = RuntimeStatistics()
        self.subsystems = SubsystemRegistry()
        self.registry = EntityRegistry()

        self._kit_app: Any = None
        self._stage: Any = None
        self._physics_scene: Any = None
        self._camera: Any = None
        self._frame_callbacks: list[Callable[[RuntimeStatistics], None]] = []
        self._run_started_at: Optional[float] = None

    # ── lifecycle: initialization ───────────────────────────────────

    def initialize(self) -> None:
        """Run the full initialization sequence.

        Executes, in order: Kit startup, stage load + validation, PhysX
        initialization, camera initialization, entity discovery,
        entity classification, and runtime subsystem initialization.

        Raises:
            StageLoadError: If the stage cannot be opened or fails
                validation under `RuntimeConfig.stage_validation_strict`.
            PhysicsInitializationError: If PhysX cannot be initialized.
            EntityDiscoveryError: If entity discovery fails.
        """
        if self.state not in (RuntimeState.UNINITIALIZED, RuntimeState.FAILED):
            raise RuntimeStateError(
                f"initialize() called in invalid state: {self.state.name}"
            )

        self.state = RuntimeState.INITIALIZING
        logger.info("=" * 72)
        logger.info("Runtime initialized")
        logger.info("=" * 72)

        try:
            self._initialize_kit()
            self.load_stage()
            if self.config.physics_enabled:
                self.initialize_physics()
            else:
                logger.info("Physics disabled by configuration; skipping PhysX initialization.")
            self.initialize_camera()
            self.discover_entities()
            self.classify_entities()
            self.initialize_runtime_systems()
        except RuntimeInitializationError:
            self.state = RuntimeState.FAILED
            raise
        except Exception as exc:  # noqa: BLE001 - re-wrap unexpected failures
            self.state = RuntimeState.FAILED
            raise RuntimeInitializationError(f"Unexpected failure during initialize(): {exc}") from exc

        self.state = RuntimeState.READY
        logger.info("Runtime ready (static=%d, dynamic=%d).",
                    self.statistics.static_entity_count, self.statistics.dynamic_entity_count)

    def _initialize_kit(self) -> None:
        """Boot the Omniverse Kit application.

        This is the single integration seam where the real
        `omni.kit.app` bootstrap sequence belongs (extension loading,
        renderer selection, headless flags). It is kept isolated in its
        own method so the Kit-specific bring-up can evolve independently
        of the rest of the runtime.
        """
        logger.info(
            "Booting Omniverse Kit (headless=%s, renderer=%s, extensions=%d)",
            self.config.headless,
            self.config.renderer.value,
            len(self.config.extension_paths),
        )
        # Integration point: real Kit bootstrap (e.g. `omni.kit.app.get_app()`
        # / `SimulationApp`) is constructed here. Kept as an opaque handle so
        # the rest of the runtime never depends on the Kit API directly.
        self._kit_app = {
            "headless": self.config.headless,
            "renderer": self.config.renderer,
            "extension_paths": list(self.config.extension_paths),
        }

    # ── lifecycle: stage / physics / camera ─────────────────────────

    def load_stage(self) -> None:
        """Open the USD stage referenced by `RuntimeConfig.usd_path` and validate it.

        Raises:
            StageLoadError: If the file does not exist, cannot be opened,
                or fails validation (under strict validation mode).
        """
        logger.info("Loading scene.usda")
        usd_path = self.config.usd_path

        if not usd_path.exists():
            raise StageLoadError(f"USD stage not found at path: {usd_path}")

        try:
            self._stage = self._open_stage(usd_path)
        except Exception as exc:  # noqa: BLE001
            raise StageLoadError(f"Failed to open stage '{usd_path}': {exc}") from exc

        self._validate_stage(self._stage)
        logger.info("Stage loaded and validated: %s", usd_path)

    def _open_stage(self, usd_path: Path) -> Any:
        """Open `usd_path` and return a stage handle.

        Uses the `pxr` (OpenUSD) bindings when available. Falls back to a
        lightweight stand-in handle (sufficient for entity discovery via
        text scan) when `pxr` is not installed in the current
        environment, so the runtime remains exercisable outside a full
        Omniverse Kit install.
        """
        try:
            from pxr import Usd  # type: ignore

            stage = Usd.Stage.Open(str(usd_path))
            if stage is None:
                raise StageLoadError(f"`pxr.Usd.Stage.Open` returned None for '{usd_path}'.")
            return stage
        except ImportError:
            logger.warning(
                "`pxr` (OpenUSD) bindings not found in this environment; using fallback "
                "stage handle. Install `usd-core` / run inside Omniverse Kit for full fidelity."
            )
            return _FallbackStage(usd_path)

    def _validate_stage(self, stage: Any) -> None:
        """Run structural validation on the opened stage.

        Validation failures are either fatal (`stage_validation_strict=True`)
        or logged as warnings, depending on configuration.
        """
        issues: list[str] = []

        pseudo_root = getattr(stage, "GetPseudoRoot", None)
        if pseudo_root is not None:
            try:
                if not stage.GetPseudoRoot().IsValid():
                    issues.append("Stage pseudo-root is invalid.")
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Failed to query stage pseudo-root: {exc}")
        elif isinstance(stage, _FallbackStage) and not stage.prim_paths:
            issues.append("Fallback stage scan found zero prims.")

        if issues:
            message = "; ".join(issues)
            if self.config.stage_validation_strict:
                raise StageLoadError(f"Stage validation failed: {message}")
            logger.warning("Stage validation issue(s) (non-strict mode): %s", message)

    def initialize_physics(self) -> None:
        """Initialize the PhysX subsystem for the current stage.

        Sets up gravity and substep configuration. Per-entity rigid body
        setup is intentionally minimal (existence/category-level only) --
        advanced PhysX tuning is out of scope for this module.

        Raises:
            PhysicsInitializationError: If physics cannot be initialized,
                e.g. because no stage has been loaded yet.
        """
        logger.info("Initializing PhysX")
        if self._stage is None:
            raise PhysicsInitializationError("Cannot initialize physics before a stage is loaded.")

        try:
            self._physics_scene = {
                "gravity_ms2": self.config.gravity_ms2,
                "substeps": self.config.physics_substeps,
                "stage": self._stage,
            }
        except Exception as exc:  # noqa: BLE001
            raise PhysicsInitializationError(f"Failed to initialize PhysX scene: {exc}") from exc

        logger.info(
            "PhysX initialized (gravity=%.3f m/s^2, substeps=%d)",
            self.config.gravity_ms2,
            self.config.physics_substeps,
        )

    def initialize_camera(self) -> None:
        """Initialize the default runtime camera.

        Establishes a placeholder camera handle; actual camera placement
        and behavior is owned by `camera_controller.CameraController`,
        which is registered as a runtime subsystem.
        """
        logger.info("Initializing camera")
        self._camera = {"prim_path": "/World/Cameras/MainCamera"}

    # ── lifecycle: entity discovery / classification ────────────────

    def discover_entities(self) -> None:
        """Walk the stage and discover all entity prims.

        Populates `self.registry` with one `RuntimeEntity` per discovered
        prim (motion class is assigned later, in `classify_entities()`).

        Raises:
            EntityDiscoveryError: If the stage cannot be traversed.
        """
        logger.info("Discovering entities")
        if self._stage is None:
            raise EntityDiscoveryError("Cannot discover entities before a stage is loaded.")

        try:
            discovered = self._traverse_stage(self._stage)
        except Exception as exc:  # noqa: BLE001
            raise EntityDiscoveryError(f"Stage traversal failed: {exc}") from exc

        for prim_path, name, metadata in discovered:
            category = self._infer_category(name, metadata)
            entity = RuntimeEntity(
                prim_path=prim_path,
                name=name,
                category=category,
                motion_class=MotionClass.STATIC,  # placeholder; set in classify_entities()
                metadata=metadata,
            )
            self.registry.add(entity)

        logger.info("Discovered %d entit%s on stage.", len(self.registry), "y" if len(self.registry) == 1 else "ies")

    def _traverse_stage(self, stage: Any) -> list[tuple[str, str, dict[str, Any]]]:
        """Return `(prim_path, name, metadata)` tuples for every entity-bearing prim."""
        results: list[tuple[str, str, dict[str, Any]]] = []

        if isinstance(stage, _FallbackStage):
            for prim_path in stage.prim_paths:
                name = prim_path.rsplit("/", 1)[-1]
                results.append((prim_path, name, dict(stage.metadata.get(prim_path, {}))))
            return results

        # `pxr`-backed traversal.
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            name = prim.GetName()
            metadata: dict[str, Any] = {}
            try:
                custom_data = prim.GetCustomData()
                if custom_data:
                    metadata.update(dict(custom_data))
            except Exception:  # noqa: BLE001 - metadata is best-effort
                pass
            results.append((prim_path, name, metadata))
        return results

    def _infer_category(self, name: str, metadata: dict[str, Any]) -> EntityCategory:
        """Infer an `EntityCategory` from prim metadata, falling back to name matching."""
        raw = str(metadata.get("entity_type", "")).strip().lower()
        if raw:
            for category in EntityCategory:
                if category.value == raw or category.name.lower() == raw:
                    return category

        lowered = name.lower()
        name_hints: dict[EntityCategory, tuple[str, ...]] = {
            EntityCategory.AIRCRAFT: ("aircraft", "plane", "jet", "f16", "f22"),
            EntityCategory.MISSILE: ("missile",),
            EntityCategory.SHIP: ("ship", "vessel", "frigate", "carrier"),
            EntityCategory.VEHICLE: ("tank", "truck", "vehicle", "car"),
            EntityCategory.HUMAN: ("human", "person", "soldier", "pedestrian"),
            EntityCategory.BUILDING: ("building", "structure", "hangar"),
            EntityCategory.TERRAIN: ("terrain", "ground", "mountain", "runway", "road"),
            EntityCategory.RADAR: ("radar",),
            EntityCategory.CAMERA: ("camera",),
            EntityCategory.SENSOR: ("sensor",),
            EntityCategory.WEATHER: ("weather", "atmosphere", "wind"),
        }
        for category, hints in name_hints.items():
            if any(hint in lowered for hint in hints):
                return category

        return EntityCategory.UNKNOWN

    def classify_entities(self) -> None:
        """Classify every discovered entity as STATIC or DYNAMIC.

        Classification rule:
            1. Categories in `_STATIC_CATEGORIES` are always STATIC.
            2. Categories in `_DYNAMIC_CATEGORIES` are always DYNAMIC.
            3. System categories (camera/sensor/weather) and UNKNOWN are
               STATIC by default (they do not participate in rigid-body
               motion unless a future subsystem says otherwise).

        Raises:
            EntityDiscoveryError: If classification is invoked before
                `discover_entities()`.
        """
        logger.info("Classifying entities")
        if len(self.registry) == 0 and self._stage is None:
            raise EntityDiscoveryError("Cannot classify entities before discover_entities() has run.")

        for entity in self.registry.entities.values():
            if entity.category in _DYNAMIC_CATEGORIES:
                entity.motion_class = MotionClass.DYNAMIC
            else:
                entity.motion_class = MotionClass.STATIC

        self.statistics.static_entity_count = len(self.registry.static_entities)
        self.statistics.dynamic_entity_count = len(self.registry.dynamic_entities)

        logger.info("Dynamic entities: %d", self.statistics.dynamic_entity_count)
        logger.info("Static entities: %d", self.statistics.static_entity_count)

    def initialize_runtime_systems(self) -> None:
        """Initialize all registered runtime subsystems.

        Subsystems are initialized in registration order. Built-in
        sibling modules (`simulation_controller`, `animation_system`,
        `camera_controller`, `asset_loader`) are expected to register
        themselves here, either by being added externally before
        `initialize()` is called, or by a project-level bootstrap that
        wires `OmniverseRuntime.subsystems.register(...)` ahead of time.
        """
        logger.info("Initializing runtime systems (%d registered)", len(self.subsystems))
        self.subsystems.initialize_all(self.registry, self.config)

    # ── lifecycle: run loop ─────────────────────────────────────────

    def start(self) -> None:
        """Start the simulation loop.

        Runs synchronously, calling `_step_frame()` once per target frame
        interval until `stop()` is called (typically from a registered
        frame callback or an external controller thread/process).

        Raises:
            RuntimeStateError: If the runtime is not in the READY state.
        """
        if self.state is not RuntimeState.READY:
            raise RuntimeStateError(f"start() requires state READY, but runtime is {self.state.name}.")

        logger.info("Simulation started")
        self.state = RuntimeState.RUNNING
        self._run_started_at = time.monotonic()

        while self.state is RuntimeState.RUNNING:
            frame_start = time.monotonic()
            self._step_frame(self.config.frame_duration_s)
            self._throttle_to_target_fps(frame_start)

    def _step_frame(self, dt: float) -> None:
        """Advance the simulation by exactly one frame.

        Per the project's defined simulation loop, this:
            1. Updates dynamic entities (via the subsystem registry).
            2. Updates animations.
            3. Updates sensors.
            4. Updates cameras.
            5. Renders the frame.

        All of the above are delegated to registered `RuntimeSubsystem`
        instances; this method only sequences them and updates
        statistics.

        Args:
            dt: Simulation time delta for this frame, in seconds.
        """
        frame_start = time.monotonic()

        self.subsystems.update_all(dt, self.registry)
        self.render_frame()

        frame_duration = time.monotonic() - frame_start
        self.statistics.frame_count += 1
        self.statistics.simulation_time_s += dt
        self.statistics.wall_clock_time_s += frame_duration
        self.statistics.last_frame_duration_s = frame_duration
        if self.statistics.wall_clock_time_s > 0:
            self.statistics.average_fps = self.statistics.frame_count / self.statistics.wall_clock_time_s

        for callback in self._frame_callbacks:
            callback(self.statistics)

    def render_frame(self) -> None:
        """Render the current frame.

        Integration point for the real Omniverse Kit render call (e.g.
        `kit_app.update()`). Kept isolated so headless / test execution
        does not require an actual renderer.
        """
        if self._kit_app is None:
            raise RuntimeStateError("Cannot render before Kit has been initialized.")
        # Integration point: `self._kit_app.update()` / Hydra render call.

    def _throttle_to_target_fps(self, frame_start: float) -> None:
        """Sleep, if necessary, to respect `RuntimeConfig.target_fps`."""
        elapsed = time.monotonic() - frame_start
        remaining = self.config.frame_duration_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def add_frame_callback(self, callback: Callable[[RuntimeStatistics], None]) -> None:
        """Register a callback invoked with `self.statistics` after every frame.

        This is the supported way for an external controller (e.g. a
        test harness, or a higher-level orchestrator) to observe runtime
        progress and decide when to call `stop()`.
        """
        self._frame_callbacks.append(callback)

    # ── lifecycle: pause / resume / stop / shutdown ──────────────────

    def pause(self) -> None:
        """Pause the simulation loop without tearing down any subsystems.

        Raises:
            RuntimeStateError: If the runtime is not RUNNING.
        """
        if self.state is not RuntimeState.RUNNING:
            raise RuntimeStateError(f"pause() requires state RUNNING, but runtime is {self.state.name}.")
        self.state = RuntimeState.PAUSED
        logger.info("Simulation paused at frame %d", self.statistics.frame_count)

    def resume(self) -> None:
        """Resume a paused simulation loop.

        Raises:
            RuntimeStateError: If the runtime is not PAUSED.
        """
        if self.state is not RuntimeState.PAUSED:
            raise RuntimeStateError(f"resume() requires state PAUSED, but runtime is {self.state.name}.")
        logger.info("Simulation resumed at frame %d", self.statistics.frame_count)
        self.state = RuntimeState.RUNNING
        while self.state is RuntimeState.RUNNING:
            frame_start = time.monotonic()
            self._step_frame(self.config.frame_duration_s)
            self._throttle_to_target_fps(frame_start)

    def stop(self) -> None:
        """Stop the simulation loop. Subsystems remain initialized.

        Safe to call from within a frame callback to terminate `start()`.
        """
        if self.state in (RuntimeState.RUNNING, RuntimeState.PAUSED):
            logger.info("Simulation stopped after %d frame(s).", self.statistics.frame_count)
            self.state = RuntimeState.STOPPED
        else:
            logger.warning("stop() called while runtime is in state %s; ignoring.", self.state.name)

    def shutdown(self) -> None:
        """Tear down all runtime subsystems and release Kit resources.

        Idempotent: calling `shutdown()` more than once is a no-op after
        the first call.
        """
        if self.state is RuntimeState.SHUTDOWN:
            return

        logger.info("Shutting down runtime")
        if self.state in (RuntimeState.RUNNING, RuntimeState.PAUSED):
            self.stop()

        self.subsystems.shutdown_all()
        self._physics_scene = None
        self._stage = None
        self._kit_app = None
        self.state = RuntimeState.SHUTDOWN
        logger.info(
            "Runtime shutdown complete (total frames=%d, simulated=%.3fs).",
            self.statistics.frame_count,
            self.statistics.simulation_time_s,
        )


# ════════════════════════════════════════════════════════════════════════
# Fallback stage (used only when `pxr` is unavailable)
# ════════════════════════════════════════════════════════════════════════

class _FallbackStage:
    """Minimal stand-in stage used when the `pxr` (OpenUSD) bindings are
    not installed in the current environment.

    Performs a conservative text scan of a `.usda` file to recover prim
    paths and `customData` blocks, so that `OmniverseRuntime` remains
    exercisable (discovery, classification, loop sequencing) outside a
    full Omniverse Kit installation. This is a development convenience,
    not a substitute for `pxr.Usd.Stage` -- production execution should
    always run with the real OpenUSD bindings available.
    """

    def __init__(self, usd_path: Path) -> None:
        self.usd_path = usd_path
        self.prim_paths: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self._scan()

    def _scan(self) -> None:
        try:
            text = self.usd_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            raise StageLoadError(f"Fallback stage scan could not read '{self.usd_path}': {exc}") from exc

        path_stack: list[str] = []
        current_metadata: dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("def ") and '"' in line:
                name = line.split('"')[1]
                parent = path_stack[-1] if path_stack else ""
                prim_path = f"{parent}/{name}"
                path_stack.append(prim_path)
                self.prim_paths.append(prim_path)
                current_metadata = {}
                self.metadata[prim_path] = current_metadata
            elif line == "}" and path_stack:
                path_stack.pop()
            elif "string " in line and "=" in line and path_stack:
                try:
                    key_part, value_part = line.split("=", 1)
                    key = key_part.replace("string", "").strip()
                    value = value_part.strip().strip('"')
                    self.metadata[path_stack[-1]][key] = value
                except (ValueError, IndexError):
                    continue


__all__ = [
    "OmniverseRuntime",
    "RuntimeConfig",
    "RuntimeState",
    "RuntimeStatistics",
    "RuntimeEntity",
    "EntityRegistry",
    "EntityCategory",
    "MotionClass",
    "RendererBackend",
    "RuntimeSubsystem",
    "SubsystemRegistry",
    "RuntimeInitializationError",
    "StageLoadError",
    "PhysicsInitializationError",
    "EntityDiscoveryError",
    "RuntimeStateError",
]
