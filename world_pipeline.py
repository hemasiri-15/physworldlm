"""
world_pipeline.py
══════════════════════════════════════════════════════════════════════════
Top-level orchestration layer of PhysWorldLM.

Pipeline position
------------------
        Natural Language Prompt
                │
                ▼
     ┌────────────────────────────────────────────────────────────────┐
     │                          WORLD PIPELINE                         │
     │                                                                  │
     │   PluginManager        EventBus            AssetRegistry         │
     │        │             (logger / debugger /       │                │
     │        │              profiler / telemetry)      │                │
     │        ▼                    ▲                    ▼                │
     │   ExecutionGraph ───────────┴──────────── ExecutionContext        │
     │        │                                         │                │
     │        ▼                                         ▼                │
     │   TaskQueue ──────────────────────────────► Workers               │
     │        │                                                          │
     └────────┼──────────────────────────────────────────────────────────┘
              ▼
   PromptParser → EntityEncoder → PhysicsOntology → WorldSpec
              │
              ▼
   AssetRegistry (resolves `asset:` refs)
              │
              ▼
   WorldSpecBuilder (WorldSpec → SceneGraph IR)   +   SceneCompiler (WorldSpec → SceneGraph → USD)
              │
              ▼
   SimulationBackend.load_stage() → SimulationBackend.start_simulation()
              │
              ▼
          Simulation

Scope
-----
`WorldPipeline` is the *only* public entry point PhysWorldLM users are
expected to touch::

    pipeline = WorldPipeline(parser=..., encoder=..., ontology=..., backends=[...])
    report = pipeline.generate("a red rubber ball bouncing on wet grass")

Everything else happens automatically, behind one call, orchestrated as an
`ExecutionGraph` of independently cacheable, independently profilable
nodes, executed through a `TaskQueue`/worker seam, with every phase
transition and diagnostic broadcast on an `EventBus` that loggers,
debuggers, profilers, visualizers, and telemetry sinks can attach to
*without this module changing*.

`WorldPipeline` owns **orchestration only**. It does not parse natural
language, encode entities, resolve ontology, lower a `WorldSpec` into a
`SceneGraph`, compile/export OpenUSD, or step a simulation -- those
responsibilities are injected as collaborators (`PromptParser`,
`EntityEncoder`, `PhysicsOntology`, `worldspec_builder.WorldSpecBuilder`,
`scene_compiler.SceneCompiler`, `SimulationBackend`) and invoked through
narrow protocols, exactly mirroring the dependency-injection style
already used by those two modules (`id_factory`/`asset_resolver`,
`BuilderRegistry`/`Exporter`).

Architectural building blocks (all defined in this module)
------------------------------------------------------------
    EventBus         -- pub/sub broadcast of every lifecycle event; the
                         seam loggers/debuggers/profilers/visualizers/
                         telemetry attach to.
    PluginManager     -- generalized `register_backend()`: backends,
                         sensors, physics plugins, terrain generators,
                         and AI agents, registered dynamically by kind
                         and name, optionally loaded from an import path.
    AssetRegistry     -- WorldSpec → Asset Registry → Asset Resolver → USD:
                         a pre-flight cache of resolved `asset:` refs
                         shared by both downstream compiler stages.
    TaskQueue         -- Pipeline → Task Queue → Workers: every graph
                         node executes through this seam so a future
                         remote/GPU/HPC `TaskExecutor` can be substituted
                         without touching orchestration logic.
    ExecutionGraph    -- the pipeline as an actual DAG object (nodes
                         declare inputs/outputs/dependencies/cacheability)
                         instead of a hard-coded `step1 → step2 → step3`
                         call chain: visualizable, cacheable, replayable,
                         profilable without changing this class.
    ExecutionContext  -- immutable-ish per-run context (config, event
                         bus, plugin manager, asset registry, task queue,
                         cancellation token) threaded through every node.
    ExecutionState    -- the mutable per-run state a graph execution
                         accumulates (node outputs, diagnostics, stats).

Public API
----------
    with WorldPipeline(parser, encoder, ontology, backends=[...]) as pipeline:
        report = pipeline.generate(prompt)
        report = pipeline.compile(prompt)
        report = pipeline.export(prompt, output_path=...)
        report = pipeline.simulate(usd_path=..., backend="mujoco")
        bench  = pipeline.run(prompt, mode=ExecutionMode.BENCHMARK, iterations=20)

        pipeline.event_bus.subscribe(my_listener)      # logger/debugger/telemetry
        profiler = pipeline.attach_profiler()           # profiler, zero pipeline changes
        graph = pipeline.graph(ExecutionMode.FULL)       # inspect/visualize the DAG

        pipeline.register_backend(my_backend)
        pipeline.register_plugin(PluginKind.SENSOR, my_lidar_plugin)

        pipeline.statistics(); pipeline.diagnostics(); pipeline.health_check()
        pipeline.save_worldspec(report.world_spec, "world.json")
        pipeline.save_scenegraph(report.scene_graph, "scene_graph.json")
        pipeline.save_compiled_scene(report.compilation_report, "compiled.json")
        pipeline.save_usd(report.compilation_report, "final.usda")
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import random
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    runtime_checkable,
)

from scene_compiler import (
    AssetResolver as _SceneCompilerAssetResolver,
    CompilationError as SceneCompilationError,
    CompilationReport,
    CompilationStatus,
    CompilerConfig,
    SceneCompiler,
    SceneGraph,
    ValidationMode,
)
from worldspec_builder import (
    BuildReport,
    BuildStatus,
    Severity,
    ValidationPolicy,
    WorldSpecBuilder,
    WorldSpecBuilderConfig,
    WorldSpecBuilderError,
)

if TYPE_CHECKING:
    from world_spec import WorldSpec


# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.world_pipeline")
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

# `asset:`-tag convention already established by both downstream stages;
# reused here rather than redefined, so AssetRegistry's notion of an
# "asset reference" never drifts from what WorldSpecBuilder/SceneCompiler
# themselves recognize.
_ASSET_TAG_PREFIX: str = _SceneCompilerAssetResolver.ASSET_TAG_PREFIX


# ════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ════════════════════════════════════════════════════════════════════════

class WorldPipelineError(Exception):
    """Base exception for all `WorldPipeline` failures."""


class PipelineLifecycleError(WorldPipelineError):
    """Raised when public API methods are invoked out of lifecycle order."""


class PipelineConfigurationError(WorldPipelineError):
    """Raised when the pipeline is misconfigured for the requested operation."""


class PipelineCancelledError(WorldPipelineError):
    """Raised internally when a run is cancelled; never escapes `run()`."""


class StageExecutionError(WorldPipelineError):
    """Raised when an injected collaborator or graph node fails.

    Wraps the originating exception so pipeline-level diagnostics stay
    structured even though the collaborator's own error types are
    unknown to this module.
    """

    def __init__(self, node_name: str, collaborator: str, message: str, *, cause: Optional[Exception] = None):
        self.node_name = node_name
        self.collaborator = collaborator
        self.cause = cause
        super().__init__(f"[{node_name}:{collaborator}] {message}")


class BackendUnavailableError(WorldPipelineError):
    """Raised when the requested simulation backend is not registered or unavailable."""


class GraphCycleError(WorldPipelineError):
    """Raised when an `ExecutionGraph` contains a dependency cycle."""


class GraphValidationError(WorldPipelineError):
    """Raised when an `ExecutionGraph` references an unknown node dependency."""


class PluginError(WorldPipelineError):
    """Raised for plugin registration/lookup/dynamic-load failures."""


class PipelineAssetError(WorldPipelineError):
    """Raised (best-effort, non-fatal by default) when an asset reference cannot be resolved."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    """Controls which subgraph of the pipeline a `run()` call executes."""

    FULL = auto()             # prompt -> ... -> running simulation
    DRY_RUN = auto()          # every generation stage runs/validates; nothing is written, no backend
    COMPILE_ONLY = auto()     # prompt -> WorldSpec -> SceneGraph (WorldSpecBuilder) only
    EXPORT_ONLY = auto()      # prompt -> ... -> compiled + exported USD; no backend
    SIMULATION_ONLY = auto()  # skip generation; load an existing USD file and simulate
    BENCHMARK = auto()        # run FULL N times, collecting aggregate timing stats

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


class PipelinePhase(Enum):
    """Ordered phases of the end-to-end pipeline; one per `ExecutionGraph` node."""

    UNINITIALIZED = 0
    INITIALIZED = 1
    PROMPT_PARSED = 2
    ENTITIES_ENCODED = 3
    ONTOLOGY_RESOLVED = 4
    ASSETS_REGISTERED = 5
    WORLDSPEC_BUILT = 6      # WorldSpecBuilder: WorldSpec -> SceneGraph IR
    SCENE_COMPILED = 7       # SceneCompiler: WorldSpec -> SceneGraph -> OpenUSD
    BACKEND_STAGE_LOADED = 8
    SIMULATION_STARTED = 9
    COMPLETE = 10
    SHUTDOWN = 99

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


class PipelineStatus(Enum):
    """Final outcome of a `run()`/`generate()`/etc. call."""

    SUCCESS = auto()
    SUCCESS_WITH_WARNINGS = auto()
    FAILED = auto()
    CANCELLED = auto()


class ComponentHealth(Enum):
    """Per-collaborator health status reported by `health_check()`."""

    OK = auto()
    DEGRADED = auto()
    UNAVAILABLE = auto()
    NOT_CONFIGURED = auto()


class EventType(Enum):
    """Every event `EventBus` can broadcast."""

    RUN_STARTED = auto()
    RUN_COMPLETED = auto()
    RUN_FAILED = auto()
    RUN_CANCELLED = auto()
    NODE_STARTED = auto()
    NODE_COMPLETED = auto()
    NODE_FAILED = auto()
    NODE_CACHE_HIT = auto()
    NODE_CACHE_MISS = auto()
    DIAGNOSTIC = auto()
    BACKEND_REGISTERED = auto()
    PLUGIN_REGISTERED = auto()


class TaskStatus(Enum):
    """Lifecycle status of a `TaskQueue` task."""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()


class PluginKind(Enum):
    """Categories a `PluginManager` can hold, mirroring PhysWorldLM's
    intended extension points: backends today, sensors/physics/terrain/
    AI agents as the WorldSpec contract grows to describe them.
    """

    BACKEND = auto()
    SENSOR = auto()
    PHYSICS = auto()
    TERRAIN = auto()
    AI_AGENT = auto()


# ════════════════════════════════════════════════════════════════════════
# Collaborator protocols (dependency injection seams)
# ════════════════════════════════════════════════════════════════════════
# None of these are implemented in this module -- WorldPipeline only knows
# each collaborator's shape and invokes it. Concrete implementations (a
# transformer PromptParser, a MiniLM EntityEncoder, an ontology resolver,
# an Omniverse/Gazebo/MuJoCo/Unity/Unreal backend, ...) live elsewhere.

@runtime_checkable
class PromptParser(Protocol):
    """Stage 1: natural language prompt -> parsed prompt representation."""

    name: str

    def parse(self, prompt: str) -> Any: ...


@runtime_checkable
class EntityEncoder(Protocol):
    """Stage 2: parsed prompt -> encoded entity representation (e.g. MiniLM embeddings)."""

    name: str

    def encode(self, parsed_prompt: Any) -> Any: ...


@runtime_checkable
class PhysicsOntology(Protocol):
    """Stage 3: encoded entities -> a validated `world_spec.WorldSpec`."""

    name: str

    def resolve(self, encoded_entities: Any) -> "WorldSpec": ...


@runtime_checkable
class SimulationBackend(Protocol):
    """A pluggable execution backend (Omniverse, Gazebo, MuJoCo, Unity, Unreal, ...).

    `WorldPipeline` never assumes anything about a backend's internals --
    it only calls these in order: `is_available()` -> `load_stage()` ->
    `start_simulation()` -> (later) `stop_simulation()`.
    """

    name: str

    def is_available(self) -> bool: ...

    def load_stage(self, usd_path: Path) -> Any: ...

    def start_simulation(self, stage_handle: Any, *, deterministic: bool, seed: Optional[int]) -> Any: ...

    def stop_simulation(self, run_handle: Any) -> None: ...


@runtime_checkable
class AssetResolver(Protocol):
    """Pluggable asset-reference resolution strategy for `AssetRegistry`."""

    name: str

    def resolve(self, ref: str, search_paths: list[Path]) -> Optional[Path]: ...


@runtime_checkable
class SensorPlugin(Protocol):
    """Extension point: sensor models (lidar, camera, IMU, ...).

    Not yet invoked by `WorldPipeline` -- the current `WorldSpec` contract
    has no per-entity sensor payloads (see `scene_compiler.SensorBuilder`)
    -- but registrable today via `register_plugin(PluginKind.SENSOR, ...)`
    so backends/tooling can discover configured sensors ahead of that.
    """

    name: str


@runtime_checkable
class PhysicsPlugin(Protocol):
    """Extension point: alternate/augmented physics solvers or materials."""

    name: str


@runtime_checkable
class TerrainPlugin(Protocol):
    """Extension point: procedural terrain generators."""

    name: str


@runtime_checkable
class AIAgentPlugin(Protocol):
    """Extension point: embodied/scripted agents dropped into a scene."""

    name: str


@runtime_checkable
class TaskExecutor(Protocol):
    """Seam a `TaskQueue` delegates actual execution to.

    `LocalThreadTaskQueue` (below) is the in-process default; a future
    remote/GPU/HPC scheduler integrates by implementing this and nothing
    else in this module changes.
    """

    def submit(self, fn: Callable[[], Any]) -> Future: ...

    def shutdown(self, wait: bool = True) -> None: ...


ProgressCallback = Callable[[PipelinePhase, float, str], None]
"""`(phase, fraction_complete, message) -> None`. Kept for backward-compatible,
low-ceremony progress reporting; internally implemented as a temporary
`EventBus` subscription -- see `WorldPipeline._install_progress_bridge()`."""


_PLUGIN_KIND_TO_PROTOCOL: dict[PluginKind, type] = {
    PluginKind.BACKEND: SimulationBackend,
    PluginKind.SENSOR: SensorPlugin,
    PluginKind.PHYSICS: PhysicsPlugin,
    PluginKind.TERRAIN: TerrainPlugin,
    PluginKind.AI_AGENT: AIAgentPlugin,
}


# ════════════════════════════════════════════════════════════════════════
# Event Bus
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Event:
    """A single immutable event broadcast on the `EventBus`."""

    type: EventType
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "payload": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in self.payload.items()},
        }


EventListener = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe hub every pipeline lifecycle event flows through.

    `WorldPipeline` publishes to this bus at every node/run transition; it
    never calls a logger, debugger, profiler, visualizer, or telemetry
    sink directly. Attaching any of those is exactly: subscribe a
    listener. No pipeline code changes when a new listener is added.

    Listener exceptions are caught and logged, never allowed to break the
    run they were observing (the same "best-effort side channel" contract
    `WorldSpecBuilder`/`SceneCompiler` diagnostics use).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: dict[Optional[EventType], dict[str, EventListener]] = {}

    def subscribe(self, listener: EventListener, event_type: Optional[EventType] = None) -> str:
        """Register `listener` for `event_type` (or every event type if `None`).

        Returns:
            An opaque subscription token; pass it to `unsubscribe()`.
        """
        token = str(uuid.uuid4())
        with self._lock:
            self._listeners.setdefault(event_type, {})[token] = listener
        return token

    def unsubscribe(self, token: str) -> None:
        """Remove a listener previously returned by `subscribe()`."""
        with self._lock:
            for listeners in self._listeners.values():
                listeners.pop(token, None)

    def publish(self, event_type: EventType, **payload: Any) -> Event:
        """Construct and broadcast an `Event`, and return it."""
        event = Event(type=event_type, timestamp=datetime.now(timezone.utc), payload=payload)
        with self._lock:
            targeted = list(self._listeners.get(event_type, {}).values())
            wildcard = list(self._listeners.get(None, {}).values())
        for listener in (*targeted, *wildcard):
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - a broken listener must never break the pipeline
                logger.exception("EventBus listener raised for event %s; ignoring.", event_type.name)
        return event


def _default_logging_listener(event: Event) -> None:
    """Default `EventBus` listener: mirrors events into the standard logger.

    Attached automatically by every `WorldPipeline` so log output exists
    with zero configuration; additional listeners (profiler, telemetry,
    a UI visualizer) are purely additive.
    """
    if event.type is EventType.DIAGNOSTIC:
        logger.log(event.payload.get("log_level", logging.INFO), "%s", event.payload.get("message", ""))
    elif event.type in (EventType.NODE_STARTED,):
        logger.info("Node start     : %s", event.payload.get("node_name"))
    elif event.type in (EventType.NODE_COMPLETED,):
        logger.info(
            "Node complete  : %-24s duration=%.4fs cached=%s",
            event.payload.get("node_name"),
            event.payload.get("duration_s", 0.0),
            event.payload.get("cached", False),
        )
    elif event.type is EventType.NODE_FAILED:
        logger.error("Node failed    : %s :: %s", event.payload.get("node_name"), event.payload.get("error"))
    elif event.type is EventType.RUN_STARTED:
        logger.info("Run started    : mode=%s", event.payload.get("mode"))
    elif event.type is EventType.RUN_COMPLETED:
        logger.info("Run completed  : status=%s", event.payload.get("status"))


class GraphProfiler:
    """Attachable `EventBus` listener that aggregates per-node durations.

    Created and wired up entirely through `EventBus.subscribe()` --
    `WorldPipeline` requires no special-casing to support this; it is the
    reference example of "attach a profiler without changing the pipeline".
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._lock = threading.RLock()
        self._durations: dict[str, list[float]] = {}
        self._token = event_bus.subscribe(self._on_event, EventType.NODE_COMPLETED)
        self._event_bus = event_bus

    def _on_event(self, event: Event) -> None:
        node_name = event.payload.get("node_name", "unknown")
        duration = float(event.payload.get("duration_s", 0.0))
        with self._lock:
            self._durations.setdefault(node_name, []).append(duration)

    def detach(self) -> None:
        self._event_bus.unsubscribe(self._token)

    def report(self) -> dict[str, dict[str, float]]:
        """Return `{node_name: {count, total_s, mean_s, max_s}}`."""
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for node_name, samples in self._durations.items():
                out[node_name] = {
                    "count": len(samples),
                    "total_s": round(sum(samples), 6),
                    "mean_s": round(sum(samples) / len(samples), 6) if samples else 0.0,
                    "max_s": round(max(samples), 6) if samples else 0.0,
                }
            return out


# ════════════════════════════════════════════════════════════════════════
# Plugin Manager
# ════════════════════════════════════════════════════════════════════════

class PluginManager:
    """Generalized collaborator registry: backends today, sensors/physics/
    terrain/AI agents as PhysWorldLM's WorldSpec contract grows to cover
    them, all registered dynamically by `(kind, name)` rather than
    hard-coded onto `WorldPipeline`.

    Thread-safe; publishes `EventType.PLUGIN_REGISTERED` /
    `EventType.BACKEND_REGISTERED` on the shared `EventBus` when provided,
    so registrations are observable the same way node execution is.
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._lock = threading.RLock()
        self._plugins: dict[PluginKind, dict[str, Any]] = {kind: {} for kind in PluginKind}
        self._event_bus = event_bus

    def register(self, plugin: Any, kind: PluginKind, *, name: Optional[str] = None) -> None:
        """Register `plugin` under `kind`, keyed by `name` (or `plugin.name`).

        Raises:
            PluginError: If `plugin` exposes no usable name.
        """
        resolved_name = name or getattr(plugin, "name", None)
        if not resolved_name:
            raise PluginError(f"Plugin of kind {kind.name} must expose a `name` attribute or an explicit `name=`.")
        with self._lock:
            self._plugins[kind][resolved_name] = plugin
        logger.debug("Registered plugin: kind=%s name=%s", kind.name, resolved_name)
        if self._event_bus is not None:
            event_type = EventType.BACKEND_REGISTERED if kind is PluginKind.BACKEND else EventType.PLUGIN_REGISTERED
            self._event_bus.publish(event_type, kind=kind.name, name=resolved_name)

    def unregister(self, kind: PluginKind, name: str) -> None:
        """Remove a previously registered plugin, if present."""
        with self._lock:
            self._plugins[kind].pop(name, None)

    def get(self, kind: PluginKind, name: str) -> Any:
        """Fetch a registered plugin.

        Raises:
            PluginError: If no plugin is registered under `(kind, name)`.
        """
        with self._lock:
            try:
                return self._plugins[kind][name]
            except KeyError as exc:
                raise PluginError(f"No plugin registered for kind={kind.name} name='{name}'.") from exc

    def list(self, kind: PluginKind) -> tuple[str, ...]:
        """Return the names of every plugin registered under `kind`."""
        with self._lock:
            return tuple(sorted(self._plugins[kind]))

    def load_from_path(self, kind: PluginKind, import_path: str, *, name: Optional[str] = None, **init_kwargs: Any) -> Any:
        """Dynamically import and instantiate a plugin.

        Args:
            kind: The `PluginKind` to register the loaded plugin under.
            import_path: `"package.module:ClassName"`. The class is
                imported and instantiated with `**init_kwargs`.
            name: Registration name override. Defaults to the
                instantiated object's `.name` attribute.
            **init_kwargs: Forwarded to the plugin class constructor.

        Returns:
            The instantiated, registered plugin.

        Raises:
            PluginError: If the module/class cannot be imported or
                instantiated.
        """
        if ":" not in import_path:
            raise PluginError(f"import_path must be 'module.path:ClassName', got '{import_path}'.")
        module_path, _, class_name = import_path.partition(":")
        try:
            module = importlib.import_module(module_path)
            plugin_cls = getattr(module, class_name)
            plugin = plugin_cls(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise PluginError(f"Failed to load plugin '{import_path}': {exc}") from exc
        self.register(plugin, kind, name=name)
        return plugin

    def to_dict(self) -> dict:
        with self._lock:
            return {kind.name: sorted(names) for kind, names in self._plugins.items()}


# ════════════════════════════════════════════════════════════════════════
# Asset Registry
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AssetRecord:
    """A single resolved (or unresolved) asset reference."""

    ref: str
    resolved_path: Optional[Path]
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "resolved_path": str(self.resolved_path) if self.resolved_path else None,
            "resolved": self.resolved_path is not None,
            "resolved_at": self.resolved_at.isoformat(),
        }


def _default_asset_resolver(ref: str, search_paths: list[Path]) -> Optional[Path]:
    """Minimal generic filesystem probe: absolute-path check, then each
    search path in order. A remote/URI-scheme reference (`scheme://...`)
    is returned as-is, unverified, matching the same convention
    `worldspec_builder._default_asset_resolver` documents.
    """
    if "://" in ref:
        return Path(ref)
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for search_path in search_paths:
        full = search_path / ref
        if full.exists():
            return full
    return None


class AssetRegistry:
    """Pre-flight cache of resolved asset references, shared by every
    downstream stage: `WorldSpec` → **Asset Registry** → Asset Resolver → USD.

    `WorldSpecBuilder` and `SceneCompiler` each still perform their own
    (authoritative, exporter-time) asset resolution -- this registry does
    not replace that. It exists one level above both: a single place to
    pre-resolve and cache `asset:` references once per `WorldSpec`, so
    repeated compiles/benchmark iterations don't re-probe the filesystem,
    and so future asset-heavy backends have one lookup surface instead of
    two independent ones.

    Thread-safe; publishes no events on its own (resolution is cheap and
    frequent -- callers interested in asset resolution timing should wrap
    `resolve_all()` as a graph node, which *is* profiled).
    """

    def __init__(
        self,
        search_paths: Optional[Sequence[Path]] = None,
        resolver: Optional[AssetResolver | Callable[[str, list[Path]], Optional[Path]]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._search_paths: list[Path] = list(search_paths or [])
        self._resolve_fn: Callable[[str, list[Path]], Optional[Path]] = (
            resolver.resolve if hasattr(resolver, "resolve") else (resolver or _default_asset_resolver)
        )
        self._records: dict[str, AssetRecord] = {}

    @property
    def search_paths(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(self._search_paths)

    def add_search_path(self, path: Path | str) -> None:
        with self._lock:
            self._search_paths.append(Path(path))

    def resolve(self, ref: str) -> Optional[Path]:
        """Resolve (and cache) a single asset reference."""
        with self._lock:
            cached = self._records.get(ref)
            if cached is not None:
                return cached.resolved_path
            resolved = self._resolve_fn(ref, list(self._search_paths))
            self._records[ref] = AssetRecord(ref=ref, resolved_path=resolved)
            return resolved

    def resolve_all(self, refs: Iterable[str]) -> dict[str, Optional[Path]]:
        """Resolve every ref in `refs`, returning `{ref: resolved_path_or_None}`."""
        return {ref: self.resolve(ref) for ref in refs}

    def get(self, ref: str) -> Optional[AssetRecord]:
        with self._lock:
            return self._records.get(ref)

    def unresolved(self) -> tuple[str, ...]:
        """Refs that were looked up but could not be resolved."""
        with self._lock:
            return tuple(ref for ref, rec in self._records.items() if rec.resolved_path is None)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "search_paths": [str(p) for p in self._search_paths],
                "records": {ref: rec.to_dict() for ref, rec in self._records.items()},
            }


def _extract_asset_refs(world_spec: "WorldSpec") -> tuple[str, ...]:
    """Collect every `asset:`-tagged reference across a `WorldSpec`'s entities."""
    refs: list[str] = []
    for entity in getattr(world_spec, "entities", ()):
        for tag in getattr(entity, "tags", ()):
            if tag.startswith(_ASSET_TAG_PREFIX):
                refs.append(tag[len(_ASSET_TAG_PREFIX):])
    return tuple(refs)


# ════════════════════════════════════════════════════════════════════════
# Task Queue
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TaskResult:
    """Outcome of a single `TaskQueue` task."""

    task_id: str
    name: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0


class LocalThreadTaskQueue:
    """Default, in-process `TaskExecutor`: Pipeline → Task Queue → Workers.

    Every `ExecutionGraph` node runs through this queue rather than being
    called inline, so a future remote/GPU/HPC scheduler integrates by
    implementing `TaskExecutor` (`submit()` / `shutdown()`) and being
    passed to `WorldPipeline(task_queue=...)` -- no orchestration code
    changes. `max_workers=1` (the default) makes this behave exactly like
    a direct call, so existing single-threaded behavior is unaffected
    unless a caller explicitly asks for more workers.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._lock = threading.RLock()
        self._history: list[TaskResult] = []

    def submit(self, fn: Callable[[], Any]) -> Future:
        return self._executor.submit(fn)

    def run_sync(self, fn: Callable[[], Any], name: str) -> TaskResult:
        """Submit `fn`, block for its result, and record a `TaskResult`."""
        task_id = str(uuid.uuid4())
        start = time.monotonic()
        future = self.submit(fn)
        try:
            result = future.result()
            outcome = TaskResult(
                task_id=task_id, name=name, status=TaskStatus.SUCCESS, result=result,
                duration_s=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised after bookkeeping
            outcome = TaskResult(
                task_id=task_id, name=name, status=TaskStatus.FAILED, error=str(exc),
                duration_s=time.monotonic() - start,
            )
            with self._lock:
                self._history.append(outcome)
            raise
        with self._lock:
            self._history.append(outcome)
        return outcome

    def history(self) -> tuple[TaskResult, ...]:
        with self._lock:
            return tuple(self._history)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# ════════════════════════════════════════════════════════════════════════
# Handles
# ════════════════════════════════════════════════════════════════════════

@dataclass
class StageHandle:
    """Opaque-to-us reference to a backend-loaded USD stage."""

    backend_name: str
    native_handle: Any
    usd_path: Path


@dataclass
class SimulationHandle:
    """Opaque-to-us reference to a running backend simulation."""

    backend_name: str
    native_handle: Any
    stage_handle: StageHandle
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ════════════════════════════════════════════════════════════════════════
# Diagnostics & statistics
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipelineDiagnostic:
    """A single, immutable structured diagnostic at the pipeline level.

    Reuses `worldspec_builder.Severity` rather than redefining a third
    severity enum. Sub-stage diagnostics remain attached to their own
    `BuildReport`/`CompilationReport`, reachable from `PipelineReport`.
    """

    phase: PipelinePhase
    severity: Severity
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.label,
            "severity": self.severity.name,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.phase.label} :: {self.source} :: {self.message}"


@dataclass
class PipelineStatistics:
    """Quantitative summary of a single `run()` call."""

    total_time_s: float = 0.0
    phase_durations_s: dict[str, float] = field(default_factory=dict)
    warning_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cancelled: bool = False
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "total_time_s": round(self.total_time_s, 6),
            "phase_durations_s": {k: round(v, 6) for k, v in self.phase_durations_s.items()},
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cancelled": self.cancelled,
            "success": self.success,
        }


@dataclass
class BenchmarkStatistics:
    """Aggregate statistics across the iterations of a `BENCHMARK` run."""

    iterations: int = 0
    successes: int = 0
    failures: int = 0
    total_time_s: float = 0.0
    mean_time_s: float = 0.0
    min_time_s: float = 0.0
    max_time_s: float = 0.0
    per_iteration_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "total_time_s": round(self.total_time_s, 6),
            "mean_time_s": round(self.mean_time_s, 6),
            "min_time_s": round(self.min_time_s, 6),
            "max_time_s": round(self.max_time_s, 6),
            "per_iteration_s": [round(t, 6) for t in self.per_iteration_s],
        }


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class WorldPipelineConfig:
    """User-configurable settings controlling `WorldPipeline` behavior.

    Attributes:
        default_mode: `ExecutionMode` used by `run()` when unspecified.
        default_backend: Backend name used when unspecified. Must match
            a name registered via `register_backend()`.
        output_dir: Base directory for relative USD/checkpoint paths.
        usd_filename_template: `str.format(scene_id=...)` template for
            the default USD export path.
        deterministic: Threaded into `WorldSpecBuilderConfig.deterministic`,
            `CompilerConfig.deterministic`, and `SimulationBackend.start_simulation()`.
        random_seed: If set, seeds Python's `random` module at the start
            of every run for reproducibility.
        validation_policy: Forwarded to `WorldSpecBuilderConfig`.
        validation_mode: Forwarded to `CompilerConfig`.
        checkpoint_dir: If set, intermediate artifacts are persisted
            after the node that produces them completes.
        enable_profiling: If True, every node's duration is recorded.
        enable_node_caching: If True, cacheable nodes (prompt parsing
            through SceneCompiler) are memoized by `(node, prompt,
            deterministic, seed)`, enabling cheap replay/benchmark reruns.
        max_workers: Worker count for the default `LocalThreadTaskQueue`.
            Ignored if a `task_queue` is injected directly.
        asset_search_paths: Seed search paths for the pipeline's
            `AssetRegistry`.
        log_level: Python logging level name.
        progress_callback: Optional default progress callback, overridable
            per-call; implemented as a temporary `EventBus` subscription.
    """

    default_mode: ExecutionMode = ExecutionMode.FULL
    default_backend: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("./physworldlm_out"))
    usd_filename_template: str = "{scene_id}.usda"
    deterministic: bool = True
    random_seed: Optional[int] = None
    validation_policy: ValidationPolicy = ValidationPolicy.STRICT
    validation_mode: ValidationMode = ValidationMode.STRICT
    checkpoint_dir: Optional[Path] = None
    enable_profiling: bool = True
    enable_node_caching: bool = True
    max_workers: int = 1
    asset_search_paths: list[Path] = field(default_factory=list)
    log_level: str = "INFO"
    progress_callback: Optional[ProgressCallback] = None

    def __post_init__(self) -> None:
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        self.output_dir = Path(self.output_dir)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.asset_search_paths = [Path(p) for p in self.asset_search_paths]


# ════════════════════════════════════════════════════════════════════════
# Execution Graph
# ════════════════════════════════════════════════════════════════════════

@dataclass
class NodeSpec:
    """One node of an `ExecutionGraph`.

    Attributes:
        name: Unique node id within its graph (e.g. `"ontology"`).
        phase: The `PipelinePhase` this node corresponds to, for
            statistics/events/checkpointing.
        fn: `(ExecutionContext) -> Any`. Reads its inputs from
            `context.state.node_outputs[dep]` for each `dep` in
            `dependencies`; returns this node's output.
        dependencies: Names of nodes that must complete first.
        cacheable: Whether this node's output may be memoized/replayed.
            Side-effecting nodes (anything touching a live backend) must
            be `False`.
    """

    name: str
    phase: PipelinePhase
    fn: Callable[["ExecutionContext"], Any]
    dependencies: tuple[str, ...] = ()
    cacheable: bool = True


class ExecutionGraph:
    """The pipeline as an actual DAG object.

    Each node knows its inputs (`dependencies`), its output (whatever
    `fn` returns, stored in `ExecutionState.node_outputs[name]`), and
    whether it may be cached. `WorldPipeline` builds one `ExecutionGraph`
    per `ExecutionMode` (see `WorldPipeline.graph()`) and executes it
    generically -- visualizing, caching, replaying, or profiling any
    node requires no change to the pipeline itself, only to what
    subscribes to the `EventBus` or reads `ExecutionState`.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeSpec] = {}

    def add_node(self, spec: NodeSpec) -> None:
        self._nodes[spec.name] = spec

    def node(self, name: str) -> NodeSpec:
        return self._nodes[name]

    def nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(self._nodes.values())

    def validate(self) -> None:
        """Check every dependency resolves to a known node.

        Raises:
            GraphValidationError: If a node depends on an unknown node.
        """
        for spec in self._nodes.values():
            for dep in spec.dependencies:
                if dep not in self._nodes:
                    raise GraphValidationError(f"Node '{spec.name}' depends on unknown node '{dep}'.")

    def topological_order(self) -> tuple[str, ...]:
        """Return node names ordered so every node follows its dependencies.

        Raises:
            GraphCycleError: If the graph contains a dependency cycle.
        """
        self.validate()
        in_degree = {name: 0 for name in self._nodes}
        dependents: dict[str, list[str]] = {name: [] for name in self._nodes}
        for name, spec in self._nodes.items():
            for dep in spec.dependencies:
                in_degree[name] += 1
                dependents[dep].append(name)

        ready = sorted(name for name, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while ready:
            ready.sort()
            current = ready.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        if len(order) != len(self._nodes):
            remaining = sorted(set(self._nodes) - set(order))
            raise GraphCycleError(f"Execution graph has a cycle involving node(s): {remaining}.")
        return tuple(order)

    def to_dict(self) -> dict:
        """Structural view of the graph (nodes + edges) for visualization/export."""
        return {
            "nodes": [
                {"name": s.name, "phase": s.phase.label, "cacheable": s.cacheable, "dependencies": list(s.dependencies)}
                for s in self._nodes.values()
            ],
        }

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid flowchart for quick visualization."""
        lines = ["flowchart TD"]
        for spec in self._nodes.values():
            lines.append(f'    {spec.name}["{spec.name}\\n({spec.phase.label})"]')
        for spec in self._nodes.values():
            for dep in spec.dependencies:
                lines.append(f"    {dep} --> {spec.name}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Execution Context & State
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionState:
    """Mutable state one `ExecutionGraph` run accumulates.

    Node outputs are stored generically in `node_outputs`; the typed
    properties below (`world_spec`, `build_report`, ...) are thin, named
    views over well-known node names so the rest of `WorldPipeline` (and
    `PipelineReport`) can keep using familiar attribute access without
    this class hard-coding pipeline-specific fields twice.
    """

    mode: ExecutionMode
    node_outputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[PipelineDiagnostic] = field(default_factory=list)
    statistics: PipelineStatistics = field(default_factory=PipelineStatistics)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def world_spec(self) -> Optional["WorldSpec"]:
        return self.node_outputs.get("ontology")

    @property
    def build_report(self) -> Optional[BuildReport]:
        return self.node_outputs.get("worldspec_builder")

    @property
    def compilation_report(self) -> Optional[CompilationReport]:
        return self.node_outputs.get("scene_compiler")

    @property
    def usd_path(self) -> Optional[Path]:
        report = self.compilation_report
        if report is not None:
            return report.output_path
        return self.node_outputs.get("usd_path_override")

    @property
    def stage_handle(self) -> Optional[StageHandle]:
        return self.node_outputs.get("backend_stage_load")

    @property
    def simulation_handle(self) -> Optional[SimulationHandle]:
        return self.node_outputs.get("backend_simulate")

    @property
    def scene_id(self) -> str:
        return getattr(self.world_spec, "scene_id", "") or ""


@dataclass
class ExecutionContext:
    """Immutable-ish per-run context threaded through every graph node.

    Node functions read whatever they need from here plus
    `context.state.node_outputs` for upstream results; they never reach
    back into `WorldPipeline` directly (aside from the bound-method
    closures that construct them), keeping node logic testable in
    isolation.
    """

    prompt: Optional[str]
    mode: ExecutionMode
    config: WorldPipelineConfig
    event_bus: EventBus
    plugin_manager: PluginManager
    asset_registry: AssetRegistry
    task_queue: TaskExecutor
    state: ExecutionState
    backend_name: Optional[str] = None
    usd_output_path: Optional[Path] = None
    cancellation_event: Optional[threading.Event] = None


# ════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineReport:
    """Final, structured result returned by every `WorldPipeline` entry point."""

    status: PipelineStatus
    mode: ExecutionMode
    scene_id: str
    statistics: PipelineStatistics
    diagnostics: list[PipelineDiagnostic]
    world_spec: Optional["WorldSpec"] = None
    build_report: Optional[BuildReport] = None
    compilation_report: Optional[CompilationReport] = None
    scene_graph: Optional[SceneGraph] = None
    usd_path: Optional[Path] = None
    stage_handle: Optional[StageHandle] = None
    simulation_handle: Optional[SimulationHandle] = None
    benchmark: Optional[BenchmarkStatistics] = None

    @property
    def success(self) -> bool:
        return self.status in (PipelineStatus.SUCCESS, PipelineStatus.SUCCESS_WITH_WARNINGS)

    def errors(self) -> list[PipelineDiagnostic]:
        return [d for d in self.diagnostics if d.severity in (Severity.ERROR, Severity.CRITICAL)]

    def warnings(self) -> list[PipelineDiagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    def to_dict(self) -> dict:
        return {
            "status": self.status.name,
            "mode": self.mode.name,
            "scene_id": self.scene_id,
            "statistics": self.statistics.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "build_report": self.build_report.to_dict() if self.build_report else None,
            "compilation_report": self.compilation_report.to_dict() if self.compilation_report else None,
            "usd_path": str(self.usd_path) if self.usd_path else None,
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
        }

    def __str__(self) -> str:
        lines = [
            f"PipelineReport(scene_id={self.scene_id!r}, mode={self.mode.name}, status={self.status.name})",
            f"  warnings : {self.statistics.warning_count}",
            f"  errors   : {self.statistics.error_count}",
            f"  time     : {self.statistics.total_time_s:.4f}s",
        ]
        if self.usd_path:
            lines.append(f"  usd_path : {self.usd_path}")
        return "\n".join(lines)


@dataclass
class HealthCheckReport:
    """Result of `WorldPipeline.health_check()`."""

    generated_at: datetime
    components: dict[str, ComponentHealth]

    @property
    def healthy(self) -> bool:
        return all(h in (ComponentHealth.OK, ComponentHealth.NOT_CONFIGURED) for h in self.components.values())

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "healthy": self.healthy,
            "components": {k: v.name for k, v in self.components.items()},
        }


def _status_from(statistics: PipelineStatistics) -> PipelineStatus:
    """Derive a `PipelineStatus` from accumulated diagnostics, following the
    same error/warning-count-to-status convention `BuildReport`/
    `CompilationReport` use, applied one level up.
    """
    if statistics.error_count > 0:
        return PipelineStatus.FAILED
    if statistics.warning_count > 0:
        return PipelineStatus.SUCCESS_WITH_WARNINGS
    return PipelineStatus.SUCCESS


# Checkpoint artifacts to persist immediately after a given node completes.
_CHECKPOINT_HOOKS: dict[str, tuple[str, ...]] = {
    "ontology": ("worldspec",),
    "worldspec_builder": ("scenegraph",),
    "scene_compiler": ("compiled", "usd"),
}


# ════════════════════════════════════════════════════════════════════════
# WorldPipeline
# ════════════════════════════════════════════════════════════════════════

class WorldPipeline:
    """Single public entry point for PhysWorldLM.

    `WorldPipeline` builds an `ExecutionGraph` appropriate to the
    requested `ExecutionMode`, then executes it node-by-node through a
    `TaskQueue`, broadcasting every transition on an `EventBus` and
    consulting a `PluginManager` for backends (and, going forward,
    sensors/physics/terrain/AI agents) and an `AssetRegistry` for
    `asset:` references -- without implementing prompt parsing, entity
    encoding, ontology resolution, WorldSpec→SceneGraph lowering,
    SceneGraph→USD compilation, or backend simulation itself.

    Thread-safety: a single instance guards all mutable state behind one
    re-entrant lock. A given *run* is inherently sequential and should be
    driven by one logical caller at a time; concurrent `run()` calls on
    the same instance are serialized rather than interleaved.

    Example:
        >>> pipeline = WorldPipeline(
        ...     parser=MyPromptParser(), encoder=MyEntityEncoder(), ontology=MyPhysicsOntology(),
        ...     backends=[MyMuJoCoBackend()], config=WorldPipelineConfig(default_backend="mujoco"),
        ... )
        >>> with pipeline:
        ...     report = pipeline.generate("a steel ball rolling down a ramp")
        >>> report.success
        True
    """

    def __init__(
        self,
        parser: Optional[PromptParser] = None,
        encoder: Optional[EntityEncoder] = None,
        ontology: Optional[PhysicsOntology] = None,
        backends: Optional[Sequence[SimulationBackend]] = None,
        config: Optional[WorldPipelineConfig] = None,
        worldspec_builder_config: Optional[WorldSpecBuilderConfig] = None,
        compiler_config: Optional[CompilerConfig] = None,
        event_bus: Optional[EventBus] = None,
        plugin_manager: Optional[PluginManager] = None,
        asset_registry: Optional[AssetRegistry] = None,
        task_queue: Optional[TaskExecutor] = None,
    ) -> None:
        """Initialize the pipeline with its injected collaborators.

        Args:
            parser: Stage-1 collaborator: prompt -> parsed prompt.
            encoder: Stage-2 collaborator: parsed prompt -> encoded entities.
            ontology: Stage-3 collaborator: encoded entities -> `WorldSpec`.
            backends: Zero or more `SimulationBackend`s, registered into
                the `PluginManager` under `PluginKind.BACKEND`.
            config: Pipeline-wide settings.
            worldspec_builder_config: Forwarded to the internal
                `WorldSpecBuilder`. Derived from `config` if omitted.
            compiler_config: Forwarded to the internal `SceneCompiler`.
                Derived from `config` if omitted.
            event_bus: Injectable `EventBus`. A new one is created if
                omitted; a default logging listener is always attached.
            plugin_manager: Injectable `PluginManager`. A new one is
                created if omitted.
            asset_registry: Injectable `AssetRegistry`. A new one is
                created (seeded from `config.asset_search_paths`) if omitted.
            task_queue: Injectable `TaskExecutor`. A `LocalThreadTaskQueue`
                (sized by `config.max_workers`) is created if omitted --
                this is the seam a remote/GPU/HPC executor replaces.
        """
        self._parser = parser
        self._encoder = encoder
        self._ontology = ontology
        self._config = config or WorldPipelineConfig()

        self.event_bus = event_bus or EventBus()
        self.event_bus.subscribe(_default_logging_listener)

        self.plugin_manager = plugin_manager or PluginManager(event_bus=self.event_bus)
        for backend in backends or ():
            self.plugin_manager.register(backend, PluginKind.BACKEND)

        self.asset_registry = asset_registry or AssetRegistry(search_paths=self._config.asset_search_paths)
        self.task_queue = task_queue or LocalThreadTaskQueue(max_workers=self._config.max_workers)

        self._worldspec_builder_config = worldspec_builder_config or WorldSpecBuilderConfig(
            validation_policy=self._config.validation_policy,
            deterministic=self._config.deterministic,
            asset_search_paths=list(self.asset_registry.search_paths),
        )
        self._compiler_config = compiler_config or CompilerConfig(
            validation_mode=self._config.validation_mode,
            deterministic=self._config.deterministic,
            asset_search_paths=list(self.asset_registry.search_paths),
        )

        self._builder = WorldSpecBuilder(self._worldspec_builder_config)
        self._compiler = SceneCompiler(self._compiler_config)

        self._lock = threading.RLock()
        self._phase: PipelinePhase = PipelinePhase.UNINITIALIZED
        self._last_report: Optional[PipelineReport] = None
        self._node_cache: dict[str, Any] = {}

    # ── context manager support ────────────────────────────────────────

    def __enter__(self) -> "WorldPipeline":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ── lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Prepare the pipeline for use.

        Raises:
            PipelineLifecycleError: If called while already active.
        """
        with self._lock:
            if self._phase not in (PipelinePhase.UNINITIALIZED, PipelinePhase.SHUTDOWN, PipelinePhase.COMPLETE):
                raise PipelineLifecycleError(
                    f"Cannot initialize(): pipeline is already active in phase '{self._phase.label}'."
                )
            self._phase = PipelinePhase.INITIALIZED
            logger.info("WorldPipeline initialized (backends=%s).", self.plugin_manager.list(PluginKind.BACKEND))

    def shutdown(self) -> None:
        """Release pipeline state, stop the last run's simulation (if any),
        and shut down the task queue.

        Idempotent.
        """
        with self._lock:
            if self._phase is PipelinePhase.SHUTDOWN:
                return
            if self._last_report is not None and self._last_report.simulation_handle is not None:
                self._safe_stop_simulation(self._last_report.simulation_handle)
            self.task_queue.shutdown(wait=True)
            self._phase = PipelinePhase.SHUTDOWN
            logger.info("WorldPipeline shut down.")

    def _safe_stop_simulation(self, handle: SimulationHandle) -> None:
        try:
            backend = self.plugin_manager.get(PluginKind.BACKEND, handle.backend_name)
        except PluginError:
            return
        try:
            backend.stop_simulation(handle.native_handle)
        except Exception:  # noqa: BLE001 - best-effort cleanup on shutdown
            logger.exception("Error stopping simulation on backend '%s' during shutdown.", handle.backend_name)

    # ── plugin / backend registration (thin PluginManager wrappers) ──

    def register_backend(self, backend: SimulationBackend) -> None:
        """Register a `SimulationBackend`. Equivalent to
        `plugin_manager.register(backend, PluginKind.BACKEND)`.
        """
        self.plugin_manager.register(backend, PluginKind.BACKEND)

    def unregister_backend(self, name: str) -> None:
        self.plugin_manager.unregister(PluginKind.BACKEND, name)

    def available_backends(self) -> tuple[str, ...]:
        return self.plugin_manager.list(PluginKind.BACKEND)

    def register_plugin(self, kind: PluginKind, plugin: Any, *, name: Optional[str] = None) -> None:
        """Register any extension-point plugin (sensor/physics/terrain/AI agent/backend)."""
        self.plugin_manager.register(plugin, kind, name=name)

    def _resolve_backend(self, requested: Optional[str]) -> SimulationBackend:
        name = requested or self._config.default_backend
        if name is None:
            raise PipelineConfigurationError(
                f"No backend specified and no `default_backend` configured. "
                f"Registered backends: {self.plugin_manager.list(PluginKind.BACKEND)}."
            )
        try:
            return self.plugin_manager.get(PluginKind.BACKEND, name)
        except PluginError as exc:
            raise BackendUnavailableError(str(exc)) from exc

    # ── observability seams ────────────────────────────────────────────

    def attach_profiler(self) -> GraphProfiler:
        """Attach and return a `GraphProfiler`, without modifying pipeline logic."""
        return GraphProfiler(self.event_bus)

    def attach_telemetry(self, sink: Callable[[dict], None]) -> str:
        """Attach a telemetry sink receiving every event as a dict.

        Returns:
            A subscription token for `event_bus.unsubscribe()`.
        """
        return self.event_bus.subscribe(lambda event: sink(event.to_dict()))

    # ── execution graph construction ──────────────────────────────────

    def graph(self, mode: ExecutionMode) -> ExecutionGraph:
        """Build (without executing) the `ExecutionGraph` for `mode`.

        Useful for visualization (`graph.to_mermaid()`), static
        inspection (`graph.to_dict()`), or dependency validation ahead
        of a real run.
        """
        g = ExecutionGraph()
        needs_generation = mode is not ExecutionMode.SIMULATION_ONLY
        needs_export = mode in (ExecutionMode.FULL, ExecutionMode.EXPORT_ONLY, ExecutionMode.BENCHMARK)
        needs_build = mode in (
            ExecutionMode.FULL, ExecutionMode.DRY_RUN, ExecutionMode.COMPILE_ONLY,
            ExecutionMode.EXPORT_ONLY, ExecutionMode.BENCHMARK,
        )
        needs_backend = mode in (ExecutionMode.FULL, ExecutionMode.SIMULATION_ONLY, ExecutionMode.BENCHMARK)

        if needs_generation:
            g.add_node(NodeSpec("prompt_parser", PipelinePhase.PROMPT_PARSED, self._node_prompt_parser))
            g.add_node(NodeSpec("entity_encoder", PipelinePhase.ENTITIES_ENCODED, self._node_entity_encoder, ("prompt_parser",)))
            g.add_node(NodeSpec("ontology", PipelinePhase.ONTOLOGY_RESOLVED, self._node_ontology, ("entity_encoder",)))
            g.add_node(NodeSpec("asset_registry", PipelinePhase.ASSETS_REGISTERED, self._node_asset_registry, ("ontology",)))
            if needs_build:
                g.add_node(
                    NodeSpec("worldspec_builder", PipelinePhase.WORLDSPEC_BUILT, self._node_worldspec_builder, ("asset_registry",))
                )
            if needs_export:
                g.add_node(
                    NodeSpec("scene_compiler", PipelinePhase.SCENE_COMPILED, self._node_scene_compiler, ("asset_registry",))
                )

        if needs_backend:
            load_deps = ("scene_compiler",) if needs_export else ()
            g.add_node(
                NodeSpec("backend_stage_load", PipelinePhase.BACKEND_STAGE_LOADED, self._node_backend_stage_load, load_deps, cacheable=False)
            )
            g.add_node(
                NodeSpec(
                    "backend_simulate", PipelinePhase.SIMULATION_STARTED, self._node_backend_simulate,
                    ("backend_stage_load",), cacheable=False,
                )
            )

        g.validate()
        return g

    # ── node implementations ────────────────────────────────────────
    # Each node reads its declared dependencies from `context.state`
    # and returns its own output; none of them implement collaborator
    # logic themselves, only invoke the injected collaborator.

    def _node_prompt_parser(self, ctx: ExecutionContext) -> Any:
        if self._parser is None:
            raise PipelineConfigurationError("This mode requires a `parser` to be injected at construction time.")
        try:
            return self._parser.parse(ctx.prompt)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError("prompt_parser", self._parser.name, str(exc), cause=exc) from exc

    def _node_entity_encoder(self, ctx: ExecutionContext) -> Any:
        if self._encoder is None:
            raise PipelineConfigurationError("This mode requires an `encoder` to be injected at construction time.")
        parsed = ctx.state.node_outputs["prompt_parser"]
        try:
            return self._encoder.encode(parsed)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError("entity_encoder", self._encoder.name, str(exc), cause=exc) from exc

    def _node_ontology(self, ctx: ExecutionContext) -> "WorldSpec":
        if self._ontology is None:
            raise PipelineConfigurationError("This mode requires an `ontology` to be injected at construction time.")
        encoded = ctx.state.node_outputs["entity_encoder"]
        try:
            world_spec = self._ontology.resolve(encoded)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError("ontology", self._ontology.name, str(exc), cause=exc) from exc
        ctx.event_bus.publish(
            EventType.DIAGNOSTIC,
            log_level=logging.INFO,
            message=f"WorldSpec '{getattr(world_spec, 'scene_id', '?')}' generated from prompt.",
        )
        return world_spec

    def _node_asset_registry(self, ctx: ExecutionContext) -> dict[str, Optional[Path]]:
        world_spec = ctx.state.node_outputs["ontology"]
        refs = _extract_asset_refs(world_spec)
        resolved = ctx.asset_registry.resolve_all(refs)
        unresolved = [ref for ref, path in resolved.items() if path is None]
        if unresolved:
            ctx.event_bus.publish(
                EventType.DIAGNOSTIC,
                log_level=logging.WARNING,
                message=f"{len(unresolved)} asset reference(s) could not be pre-resolved: {unresolved}.",
            )
        return resolved

    def _node_worldspec_builder(self, ctx: ExecutionContext) -> BuildReport:
        world_spec = ctx.state.node_outputs["ontology"]
        with self._builder as builder:
            report = builder.build(world_spec)
        if report.status is BuildStatus.FAILED:
            raise StageExecutionError(
                "worldspec_builder", "WorldSpecBuilder", f"failed: {[str(d) for d in report.errors()]}"
            )
        if report.status is BuildStatus.SUCCESS_WITH_WARNINGS:
            ctx.event_bus.publish(
                EventType.DIAGNOSTIC, log_level=logging.WARNING,
                message="WorldSpecBuilder completed with warnings; see build_report.diagnostics.",
            )
        return report

    def _node_scene_compiler(self, ctx: ExecutionContext) -> CompilationReport:
        world_spec = ctx.state.node_outputs["ontology"]
        output_path = ctx.usd_output_path or self._default_usd_path(ctx.state.scene_id)
        report = self._compiler.compile(world_spec, output_path)
        if report.status is CompilationStatus.FAILED:
            raise StageExecutionError(
                "scene_compiler", "SceneCompiler", f"failed: {[str(d) for d in report.errors()]}"
            )
        if report.status is CompilationStatus.SUCCESS_WITH_WARNINGS:
            ctx.event_bus.publish(
                EventType.DIAGNOSTIC, log_level=logging.WARNING,
                message="SceneCompiler completed with warnings; see compilation_report.diagnostics.",
            )
        return report

    def _node_backend_stage_load(self, ctx: ExecutionContext) -> StageHandle:
        backend = self._resolve_backend(ctx.backend_name)
        try:
            available = backend.is_available()
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailableError(f"Backend '{backend.name}' health check raised: {exc}") from exc
        if not available:
            raise BackendUnavailableError(f"Backend '{backend.name}' reported itself unavailable.")

        if ctx.mode is ExecutionMode.SIMULATION_ONLY:
            usd_path = ctx.usd_output_path
        else:
            usd_path = ctx.state.compilation_report.output_path

        try:
            native = backend.load_stage(usd_path)
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError("backend_stage_load", backend.name, str(exc), cause=exc) from exc
        return StageHandle(backend_name=backend.name, native_handle=native, usd_path=usd_path)

    def _node_backend_simulate(self, ctx: ExecutionContext) -> SimulationHandle:
        stage_handle: StageHandle = ctx.state.node_outputs["backend_stage_load"]
        backend = self.plugin_manager.get(PluginKind.BACKEND, stage_handle.backend_name)
        try:
            native = backend.start_simulation(
                stage_handle.native_handle, deterministic=ctx.config.deterministic, seed=ctx.config.random_seed
            )
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError("backend_simulate", backend.name, str(exc), cause=exc) from exc
        return SimulationHandle(backend_name=backend.name, native_handle=native, stage_handle=stage_handle)

    # ── graph execution ─────────────────────────────────────────────

    def _cache_key(self, node_name: str, ctx: ExecutionContext) -> str:
        raw = f"{node_name}|{ctx.prompt}|{ctx.config.deterministic}|{ctx.config.random_seed}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _execute_graph(self, g: ExecutionGraph, ctx: ExecutionContext) -> None:
        state = ctx.state
        ctx.event_bus.publish(EventType.RUN_STARTED, mode=ctx.mode.name)
        for node_name in g.topological_order():
            self._check_cancelled(ctx.cancellation_event)
            spec = g.node(node_name)
            ctx.event_bus.publish(EventType.NODE_STARTED, node_name=node_name)

            cache_key = self._cache_key(node_name, ctx) if (spec.cacheable and self._config.enable_node_caching) else None
            cache_slot = f"{node_name}:{cache_key}" if cache_key else None

            start = time.monotonic()
            cached = False
            if cache_slot is not None and cache_slot in self._node_cache:
                output = self._node_cache[cache_slot]
                cached = True
                state.statistics.cache_hits += 1
                ctx.event_bus.publish(EventType.NODE_CACHE_HIT, node_name=node_name)
            else:
                if cache_slot is not None:
                    state.statistics.cache_misses += 1
                    ctx.event_bus.publish(EventType.NODE_CACHE_MISS, node_name=node_name)
                try:
                    task_result = self.task_queue.run_sync(lambda: spec.fn(ctx), name=node_name)
                    output = task_result.result
                except Exception as exc:  # noqa: BLE001
                    duration = time.monotonic() - start
                    ctx.event_bus.publish(EventType.NODE_FAILED, node_name=node_name, error=str(exc), duration_s=duration)
                    raise
                if cache_slot is not None:
                    self._node_cache[cache_slot] = output

            duration = time.monotonic() - start
            state.node_outputs[node_name] = output
            if self._config.enable_profiling:
                state.statistics.phase_durations_s[spec.phase.label] = duration
            ctx.event_bus.publish(EventType.NODE_COMPLETED, node_name=node_name, duration_s=duration, cached=cached)

            self._checkpoint_node(node_name, state)

    @staticmethod
    def _check_cancelled(cancellation_event: Optional[threading.Event]) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise PipelineCancelledError("Run cancelled via cancellation_event.")

    def clear_node_cache(self) -> None:
        """Discard every memoized node output (see `config.enable_node_caching`)."""
        with self._lock:
            self._node_cache.clear()

    # ── checkpointing ────────────────────────────────────────────────

    def _checkpoint_node(self, node_name: str, state: ExecutionState) -> None:
        artifacts = _CHECKPOINT_HOOKS.get(node_name)
        if not artifacts or self._config.checkpoint_dir is None:
            return
        scene_id = state.scene_id or "unnamed_scene"
        directory = self._config.checkpoint_dir / scene_id
        try:
            if "worldspec" in artifacts and state.world_spec is not None:
                self.save_worldspec(state.world_spec, directory / "worldspec.json")
            if "scenegraph" in artifacts and state.build_report is not None and state.build_report.scene_graph is not None:
                self.save_scenegraph(state.build_report.scene_graph, directory / "scenegraph.json")
            if "compiled" in artifacts and state.compilation_report is not None:
                self.save_compiled_scene(state.compilation_report, directory / "compiled_scene.json")
            if "usd" in artifacts and state.compilation_report is not None:
                self.save_usd(state.compilation_report, directory / "scene.usda")
        except Exception:  # noqa: BLE001 - checkpointing is best-effort, never fails the run
            logger.exception("Checkpointing failed for scene '%s'; continuing.", scene_id)

    def _default_usd_path(self, scene_id: str) -> Path:
        filename = self._config.usd_filename_template.format(scene_id=scene_id or "scene")
        return self._config.output_dir / filename

    # ── public API: run() -- the general orchestrator ────────────────

    def run(
        self,
        prompt: Optional[str] = None,
        *,
        mode: Optional[ExecutionMode] = None,
        usd_path: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None,
        backend: Optional[str] = None,
        iterations: int = 1,
        cancellation_event: Optional[threading.Event] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineReport:
        """Execute the pipeline according to `mode`.

        This is the single, general-purpose orchestrator every
        convenience method delegates to. Every code path -- including an
        unexpected exception raised deep inside an injected collaborator
        or graph node -- returns a `PipelineReport` rather than
        propagating, mirroring the `build()`/`compile()` contract of the
        two stages this module orchestrates.

        Args:
            prompt: Natural-language prompt. Required for every mode
                except `SIMULATION_ONLY`.
            mode: Which subgraph to execute. Defaults to `config.default_mode`.
            usd_path: For `SIMULATION_ONLY`, the existing USD file to load.
            output_path: Explicit USD export destination for generation modes.
            backend: Backend name for the simulation stage. Defaults to
                `config.default_backend`.
            iterations: Repeat count; only meaningful for `ExecutionMode.BENCHMARK`.
            cancellation_event: Optional cooperative-cancellation token,
                checked at every node boundary.
            progress_callback: Optional per-call override of
                `config.progress_callback`; bridged onto `event_bus` for
                the duration of this call.

        Returns:
            A `PipelineReport`. For `BENCHMARK`, `.benchmark` is populated
            and the report otherwise reflects the *last* iteration.

        Raises:
            PipelineLifecycleError: If called before `initialize()` or
                after `shutdown()`.
        """
        with self._lock:
            if self._phase in (PipelinePhase.UNINITIALIZED, PipelinePhase.SHUTDOWN):
                raise PipelineLifecycleError("WorldPipeline must be initialize()'d before run().")

            resolved_mode = mode or self._config.default_mode
            bridge_token = self._install_progress_bridge(progress_callback)
            try:
                if resolved_mode is ExecutionMode.BENCHMARK:
                    return self._run_benchmark(
                        prompt, iterations=max(1, iterations), backend=backend,
                        cancellation_event=cancellation_event,
                    )

                if self._config.random_seed is not None:
                    random.seed(self._config.random_seed)

                report = self._run_once(
                    prompt, mode=resolved_mode,
                    usd_path=Path(usd_path) if usd_path is not None else None,
                    output_path=Path(output_path) if output_path is not None else None,
                    backend=backend, cancellation_event=cancellation_event,
                )
                self._last_report = report
                return report
            finally:
                if bridge_token is not None:
                    self.event_bus.unsubscribe(bridge_token)

    def _install_progress_bridge(self, progress_callback: Optional[ProgressCallback]) -> Optional[str]:
        """Bridge the legacy `(phase, fraction, message)` callback style onto
        `EventBus.subscribe()`, so `ProgressCallback` keeps working without
        `run()`/node code needing to know it exists.
        """
        cb = progress_callback or self._config.progress_callback
        if cb is None:
            return None

        # Rough, monotonic fraction-complete estimate keyed by phase index;
        # good enough for a progress bar without coupling to graph shape.
        max_phase_value = max(p.value for p in PipelinePhase if p not in (PipelinePhase.SHUTDOWN,))

        def _bridge(event: Event) -> None:
            if event.type is not EventType.NODE_COMPLETED:
                return
            node_name = event.payload.get("node_name", "")
            phase = _NODE_NAME_TO_PHASE.get(node_name)
            if phase is None:
                return
            fraction = phase.value / max_phase_value
            try:
                cb(phase, fraction, f"{phase.label} complete.")
            except Exception:  # noqa: BLE001
                logger.exception("progress_callback raised; ignoring.")

        return self.event_bus.subscribe(_bridge, EventType.NODE_COMPLETED)

    def _run_benchmark(
        self,
        prompt: Optional[str],
        *,
        iterations: int,
        backend: Optional[str],
        cancellation_event: Optional[threading.Event],
    ) -> PipelineReport:
        per_iteration: list[float] = []
        successes = 0
        failures = 0
        last_report: Optional[PipelineReport] = None
        bench_start = time.monotonic()

        for i in range(iterations):
            self._check_cancelled(cancellation_event)
            if self._config.random_seed is not None:
                random.seed(self._config.random_seed + i)
            iter_start = time.monotonic()
            report = self._run_once(
                prompt, mode=ExecutionMode.FULL, usd_path=None, output_path=None,
                backend=backend, cancellation_event=cancellation_event,
            )
            per_iteration.append(time.monotonic() - iter_start)
            if report.success:
                successes += 1
            else:
                failures += 1
            last_report = report
            if report.simulation_handle is not None:
                self._safe_stop_simulation(report.simulation_handle)

        total = time.monotonic() - bench_start
        stats = BenchmarkStatistics(
            iterations=iterations, successes=successes, failures=failures, total_time_s=total,
            mean_time_s=(sum(per_iteration) / len(per_iteration)) if per_iteration else 0.0,
            min_time_s=min(per_iteration) if per_iteration else 0.0,
            max_time_s=max(per_iteration) if per_iteration else 0.0,
            per_iteration_s=per_iteration,
        )
        assert last_report is not None  # iterations >= 1 guaranteed by caller
        final_report = replace(last_report, mode=ExecutionMode.BENCHMARK, benchmark=stats)
        self._last_report = final_report
        return final_report

    def _run_once(
        self,
        prompt: Optional[str],
        *,
        mode: ExecutionMode,
        usd_path: Optional[Path],
        output_path: Optional[Path],
        backend: Optional[str],
        cancellation_event: Optional[threading.Event],
    ) -> PipelineReport:
        if mode is not ExecutionMode.SIMULATION_ONLY and not prompt:
            state = ExecutionState(mode=mode)
            self._log(state, PipelinePhase.INITIALIZED, Severity.CRITICAL, f"mode={mode.name} requires a non-empty prompt.", "pipeline")
            return self._finalize(state, PipelineStatus.FAILED, mode)
        if mode is ExecutionMode.SIMULATION_ONLY and usd_path is None:
            state = ExecutionState(mode=mode)
            self._log(state, PipelinePhase.INITIALIZED, Severity.CRITICAL, "SIMULATION_ONLY mode requires `usd_path`.", "pipeline")
            return self._finalize(state, PipelineStatus.FAILED, mode)

        state = ExecutionState(mode=mode)
        ctx = ExecutionContext(
            prompt=prompt, mode=mode, config=self._config, event_bus=self.event_bus,
            plugin_manager=self.plugin_manager, asset_registry=self.asset_registry,
            task_queue=self.task_queue, state=state, backend_name=backend,
            usd_output_path=usd_path or output_path, cancellation_event=cancellation_event,
        )

        try:
            g = self.graph(mode)
            self._execute_graph(g, ctx)
            status = _status_from(state.statistics)
            ctx.event_bus.publish(EventType.RUN_COMPLETED, status=status.name)
            return self._finalize(state, status, mode)
        except PipelineCancelledError as exc:
            state.statistics.cancelled = True
            self._log(state, PipelinePhase.INITIALIZED, Severity.WARNING, str(exc), "pipeline")
            ctx.event_bus.publish(EventType.RUN_CANCELLED)
            return self._finalize(state, PipelineStatus.CANCELLED, mode)
        except (WorldPipelineError, WorldSpecBuilderError, SceneCompilationError) as exc:
            self._log(state, PipelinePhase.INITIALIZED, Severity.CRITICAL, str(exc), "pipeline")
            ctx.event_bus.publish(EventType.RUN_FAILED, error=str(exc))
            return self._finalize(state, PipelineStatus.FAILED, mode)
        except Exception as exc:  # noqa: BLE001 - never let a collaborator bug escape run()
            logger.exception("Unexpected error during run().")
            self._log(state, PipelinePhase.INITIALIZED, Severity.CRITICAL, f"Unexpected {type(exc).__name__}: {exc}", "pipeline")
            ctx.event_bus.publish(EventType.RUN_FAILED, error=str(exc))
            return self._finalize(state, PipelineStatus.FAILED, mode)

    def _log(self, state: ExecutionState, phase: PipelinePhase, severity: Severity, message: str, source: str) -> None:
        diag = PipelineDiagnostic(phase=phase, severity=severity, message=message, source=source)
        state.diagnostics.append(diag)
        if severity is Severity.WARNING:
            state.statistics.warning_count += 1
        elif severity in (Severity.ERROR, Severity.CRITICAL):
            state.statistics.error_count += 1
        self.event_bus.publish(
            EventType.DIAGNOSTIC,
            log_level={Severity.WARNING: logging.WARNING, Severity.ERROR: logging.ERROR, Severity.CRITICAL: logging.CRITICAL}.get(severity, logging.INFO),
            message=str(diag),
        )

    def _finalize(self, state: ExecutionState, status: PipelineStatus, mode: ExecutionMode) -> PipelineReport:
        state.statistics.total_time_s = time.monotonic() - state.started_at
        state.statistics.success = status in (PipelineStatus.SUCCESS, PipelineStatus.SUCCESS_WITH_WARNINGS)
        report = PipelineReport(
            status=status, mode=mode, scene_id=state.scene_id, statistics=state.statistics,
            diagnostics=list(state.diagnostics), world_spec=state.world_spec, build_report=state.build_report,
            compilation_report=state.compilation_report,
            scene_graph=state.build_report.scene_graph if state.build_report else None,
            usd_path=state.usd_path, stage_handle=state.stage_handle, simulation_handle=state.simulation_handle,
        )
        logger.info("Run finished for scene '%s' -> status=%s", report.scene_id, report.status.name)
        return report

    # ── public API: convenience wrappers around run() ─────────────────

    def generate(
        self, prompt: str, *, backend: Optional[str] = None, output_path: Optional[Path | str] = None,
        cancellation_event: Optional[threading.Event] = None, progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineReport:
        """Run the full pipeline: prompt -> ... -> loaded + running simulation."""
        return self.run(
            prompt, mode=ExecutionMode.FULL, backend=backend, output_path=output_path,
            cancellation_event=cancellation_event, progress_callback=progress_callback,
        )

    def compile(
        self, prompt: str, *, cancellation_event: Optional[threading.Event] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineReport:
        """Run prompt -> WorldSpec -> SceneGraph (via `WorldSpecBuilder`) only."""
        return self.run(
            prompt, mode=ExecutionMode.COMPILE_ONLY,
            cancellation_event=cancellation_event, progress_callback=progress_callback,
        )

    def export(
        self, prompt: str, *, output_path: Optional[Path | str] = None,
        cancellation_event: Optional[threading.Event] = None, progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineReport:
        """Run prompt -> ... -> compiled and exported OpenUSD file. No backend is touched."""
        return self.run(
            prompt, mode=ExecutionMode.EXPORT_ONLY, output_path=output_path,
            cancellation_event=cancellation_event, progress_callback=progress_callback,
        )

    def simulate(
        self, usd_path: Path | str, *, backend: Optional[str] = None,
        cancellation_event: Optional[threading.Event] = None, progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineReport:
        """Load an existing USD file into a backend and start simulation,
        skipping prompt parsing / WorldSpec generation / compilation entirely.
        """
        return self.run(
            usd_path=usd_path, mode=ExecutionMode.SIMULATION_ONLY, backend=backend,
            cancellation_event=cancellation_event, progress_callback=progress_callback,
        )

    # ── public API: introspection ──────────────────────────────────────

    def statistics(self) -> Optional[PipelineStatistics]:
        """Return the `PipelineStatistics` of the most recently completed run, if any."""
        with self._lock:
            return self._last_report.statistics if self._last_report else None

    def diagnostics(self) -> tuple[PipelineDiagnostic, ...]:
        """Return the diagnostics of the most recently completed run, if any."""
        with self._lock:
            return tuple(self._last_report.diagnostics) if self._last_report else ()

    def health_check(self) -> HealthCheckReport:
        """Check the availability of every injected collaborator and registered plugin."""
        with self._lock:
            components: dict[str, ComponentHealth] = {
                "parser": ComponentHealth.OK if self._parser is not None else ComponentHealth.NOT_CONFIGURED,
                "encoder": ComponentHealth.OK if self._encoder is not None else ComponentHealth.NOT_CONFIGURED,
                "ontology": ComponentHealth.OK if self._ontology is not None else ComponentHealth.NOT_CONFIGURED,
                "worldspec_builder": ComponentHealth.OK,
                "scene_compiler": ComponentHealth.OK,
            }
            backend_names = self.plugin_manager.list(PluginKind.BACKEND)
            if not backend_names:
                components["backend:*"] = ComponentHealth.NOT_CONFIGURED
            for name in backend_names:
                backend = self.plugin_manager.get(PluginKind.BACKEND, name)
                try:
                    components[f"backend:{name}"] = ComponentHealth.OK if backend.is_available() else ComponentHealth.UNAVAILABLE
                except Exception:  # noqa: BLE001
                    components[f"backend:{name}"] = ComponentHealth.DEGRADED
            for kind in (PluginKind.SENSOR, PluginKind.PHYSICS, PluginKind.TERRAIN, PluginKind.AI_AGENT):
                names = self.plugin_manager.list(kind)
                components[f"{kind.name.lower()}:*"] = ComponentHealth.OK if names else ComponentHealth.NOT_CONFIGURED
            return HealthCheckReport(generated_at=datetime.now(timezone.utc), components=components)

    # ── public API: persistence ──────────────────────────────────────

    @staticmethod
    def _write_json(payload: dict, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def save_worldspec(self, world_spec: "WorldSpec", path: Path | str) -> Path:
        """Serialize a `WorldSpec` to JSON at `path` (via `.to_dict()` if available)."""
        payload = world_spec.to_dict() if hasattr(world_spec, "to_dict") else vars(world_spec)
        written = self._write_json(payload, path)
        logger.info("WorldSpec saved to '%s'.", written)
        return written

    def save_scenegraph(self, scene_graph: SceneGraph, path: Path | str) -> Path:
        """Serialize a `SceneGraph` (from either `WorldSpecBuilder` or `SceneCompiler`) to JSON."""
        written = self._write_json(scene_graph.to_dict(), path)
        logger.info("SceneGraph saved to '%s'.", written)
        return written

    def save_compiled_scene(self, report: CompilationReport, path: Path | str) -> Path:
        """Serialize a `CompilationReport` to JSON. Does not re-export USD; see `save_usd()`."""
        written = self._write_json(report.to_dict(), path)
        logger.info("Compiled scene report saved to '%s'.", written)
        return written

    def save_usd(self, report: CompilationReport, path: Path | str) -> Path:
        """Copy the OpenUSD file already produced by `SceneCompiler.compile()` to `path`.

        Never re-runs USD export logic (that belongs solely to
        `scene_compiler.Exporter` implementations) -- copies the file
        `report.output_path` already points at.

        Raises:
            WorldPipelineError: If `report` has no exported file to copy.
        """
        if report.output_path is None or not Path(report.output_path).exists():
            raise WorldPipelineError("CompilationReport has no exported USD file to copy (compilation may have failed).")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report.output_path, destination)
        logger.info("USD scene copied from '%s' to '%s'.", report.output_path, destination)
        return destination


_NODE_NAME_TO_PHASE: dict[str, PipelinePhase] = {
    "prompt_parser": PipelinePhase.PROMPT_PARSED,
    "entity_encoder": PipelinePhase.ENTITIES_ENCODED,
    "ontology": PipelinePhase.ONTOLOGY_RESOLVED,
    "asset_registry": PipelinePhase.ASSETS_REGISTERED,
    "worldspec_builder": PipelinePhase.WORLDSPEC_BUILT,
    "scene_compiler": PipelinePhase.SCENE_COMPILED,
    "backend_stage_load": PipelinePhase.BACKEND_STAGE_LOADED,
    "backend_simulate": PipelinePhase.SIMULATION_STARTED,
}


__all__ = [
    "WorldPipeline",
    "WorldPipelineConfig",
    "PipelineReport",
    "PipelineStatistics",
    "BenchmarkStatistics",
    "PipelineStatus",
    "PipelinePhase",
    "PipelineDiagnostic",
    "ExecutionMode",
    "ComponentHealth",
    "HealthCheckReport",
    "StageHandle",
    "SimulationHandle",
    "ProgressCallback",
    "PromptParser",
    "EntityEncoder",
    "PhysicsOntology",
    "SimulationBackend",
    "AssetResolver",
    "SensorPlugin",
    "PhysicsPlugin",
    "TerrainPlugin",
    "AIAgentPlugin",
    "TaskExecutor",
    "Event",
    "EventType",
    "EventListener",
    "EventBus",
    "GraphProfiler",
    "PluginKind",
    "PluginManager",
    "AssetRecord",
    "AssetRegistry",
    "TaskStatus",
    "TaskResult",
    "LocalThreadTaskQueue",
    "NodeSpec",
    "ExecutionGraph",
    "ExecutionContext",
    "ExecutionState",
    "WorldPipelineError",
    "PipelineLifecycleError",
    "PipelineConfigurationError",
    "PipelineCancelledError",
    "StageExecutionError",
    "BackendUnavailableError",
    "GraphCycleError",
    "GraphValidationError",
    "PluginError",
    "PipelineAssetError",
]
