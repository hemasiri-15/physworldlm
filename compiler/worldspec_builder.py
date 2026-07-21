"""
worldspec_builder.py
══════════════════════════════════════════════════════════════════════════
Semantic compiler stage of PhysWorldLM.

Pipeline position
------------------
        Natural Language Prompt
                │
                ▼
            Prompt Parser
                │
                ▼
        MiniLM Entity Encoder
                │
                ▼
          Physics Ontology
                │
                ▼
        Validated WorldSpec
                │
                ▼
     ┌───────────────────────┐
     │   WORLDSPEC BUILDER    │   <-- this module
     └───────────────────────┘
                │
                ▼
   Backend-independent SceneGraph (IR)
                │
                ▼
   Scene Compiler / USD Exporter / glTF / URDF / Gazebo / MuJoCo / ...

Analogy
-------
`WorldSpecBuilder` plays the role a compiler front-end plays between a
type-checked AST and codegen: it takes a *validated* `WorldSpec` --
already produced and structurally trusted by the parser/ontology layer
-- and performs semantic analysis (duplicate detection, reference
resolution, circular-dependency detection, unit normalization) before
lowering it into a strongly-typed, backend-independent `SceneGraph`
intermediate representation (IR). Just as LLVM IR does not know or care
whether it will eventually be lowered to x86 or ARM, the `SceneGraph`
produced here does not know or care whether it will eventually be
exported to OpenUSD, glTF, URDF, Gazebo, MuJoCo, Bullet, Unity, or
Unreal.

Scope
-----
This module owns exactly one transformation: WorldSpec -> SceneGraph.

`WorldSpecBuilder` MUST NOT:
    * launch Omniverse, or any other runtime/engine
    * render anything
    * run or step physics
    * spawn entities into a live simulation
    * write USD, glTF, or any other backend format
    * manage or fetch remote assets (it only *resolves references*)

The `SceneGraph` / `SceneNode` / `NodeType` / `Transform` IR types are
intentionally **not** redefined here -- they already exist as the
backend-independent representation exported by `scene_compiler` and are
reused as-is, so that a single IR flows through the rest of the
pipeline without duplication.

Integration note (2026 audit)
------------------------------
This audit pass only had access to `worldspec_builder.py` and
`world_spec.py`. `scene_compiler.py`, `entity_builder.py`,
`stage_builder.py`, `transform_builder.py`, and `usd_exporter.py` were
not available to inspect, so the exact shapes of `SceneGraph`,
`SceneNode`, `NodeType`, `Transform`, `OptimizationLevel`,
`CoordinateSystem`, and `UnitSystem` are *assumed* from how this module
already uses them; they could not be verified against real
definitions. Every lazy import of `scene_compiler` is now wrapped so
that a missing/incompatible symbol surfaces as a well-typed
`SceneCompilerIntegrationError` (still a `WorldSpecBuilderError`)
instead of a raw `ImportError`/`AttributeError` escaping `build()` --
see "Critical #1" in the accompanying review.

Public API
----------
    with WorldSpecBuilder(config) as builder:
        report = builder.build(world_spec)
        report.scene_graph  # -> ready for SceneCompiler / any backend

Or, driven one stage at a time (useful for tooling, tests, partial
recompiles, and step-through debugging)::

    builder = WorldSpecBuilder(config)
    builder.initialize()
    builder.validate(world_spec)
    builder.normalize()
    builder.resolve_entities()
    builder.resolve_environment()
    builder.resolve_terrain()
    builder.resolve_materials()
    builder.resolve_assets()
    builder.resolve_transforms()
    builder.resolve_physics()
    builder.resolve_constraints()
    builder.resolve_hierarchy()
    builder.optimize()
    scene_graph = builder.generate_scene_graph()
    report = builder.export_report()
    builder.shutdown()
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    # Type-checking-only imports: never executed at runtime, so they
    # cannot introduce a real import-time dependency cycle between
    # `world_spec`, `scene_compiler`, and this module.
    from compiler.scene_compiler import (
        CoordinateSystem,
        NodeType,
        OptimizationLevel,
        SceneGraph,
        SceneNode,
        Transform,
        UnitSystem,
    )
    from world_spec import Entity, WorldSpec


# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.worldspec_builder")
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
# Exception hierarchy
# ════════════════════════════════════════════════════════════════════════

class WorldSpecBuilderError(Exception):
    """Base exception for all `WorldSpecBuilder` failures."""


class BuilderLifecycleError(WorldSpecBuilderError):
    """Raised when public API methods are invoked out of lifecycle order.

    E.g. calling `resolve_entities()` before `initialize()`/`validate()`,
    or calling any method after `shutdown()`.
    """


class SceneCompilerIntegrationError(WorldSpecBuilderError):
    """Raised when `scene_compiler` cannot supply a type/symbol this
    builder depends on (missing module, renamed symbol, incompatible
    version, etc).

    Added in the 2026 audit: previously, any failure importing
    `scene_compiler` inside a resolver (`ImportError`, `AttributeError`)
    propagated as a raw, unwrapped exception, breaking the contract that
    `build()` always returns a structured `BuildReport`. Every lazy
    `scene_compiler` import in this file is now funneled through
    `_import_scene_compiler_symbols()`, which raises this instead.
    """


class SpecValidationError(WorldSpecBuilderError):
    """Raised when a `WorldSpec` fails semantic validation.

    Carries the full list of underlying problems in `.problems`, in
    addition to the standard exception message (their joined text).
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems) if self.problems else "unknown validation failure")


class DuplicateEntityError(SpecValidationError):
    """Raised when two or more entities share the same id."""


class CircularDependencyError(SpecValidationError):
    """Raised when the entity constraint/interaction graph contains a cycle."""


class InvalidReferenceError(SpecValidationError):
    """Raised when an interaction or constraint references an unknown entity."""


class MissingAssetError(SpecValidationError):
    """Raised (in STRICT mode) when a referenced on-disk asset cannot be found."""


class InvalidMaterialError(SpecValidationError):
    """Raised (in STRICT mode) when an entity references an unknown material."""


class InvalidTerrainError(SpecValidationError):
    """Raised (in STRICT mode) when the environment references an unknown terrain type."""


class InconsistentPhysicsError(SpecValidationError):
    """Raised when physical parameters are internally inconsistent.

    E.g. restitution/friction outside `[0, 1]`, or a dynamic entity with
    non-positive mass slipping past earlier validation.
    """


class SchemaViolationError(SpecValidationError):
    """Raised for structural violations of the WorldSpec data contract itself."""


class HierarchyResolutionError(WorldSpecBuilderError):
    """Raised when parent/child scene hierarchy cannot be resolved unambiguously."""


class OptimizationError(WorldSpecBuilderError):
    """Raised when a compile-time optimization pass fails."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class ValidationPolicy(Enum):
    """Controls how validation problems are handled."""

    STRICT = auto()       # any problem aborts the build immediately
    PERMISSIVE = auto()   # problems are recorded as ERROR diagnostics, build continues
    DISABLED = auto()     # validation is skipped entirely (caller's responsibility)


class Severity(Enum):
    """Severity levels for structured builder diagnostics."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class BuildStatus(Enum):
    """Final outcome of a `build()` run."""

    SUCCESS = auto()
    SUCCESS_WITH_WARNINGS = auto()
    FAILED = auto()


class BuildPhase(Enum):
    """Ordered phases of the WorldSpecBuilder pipeline.

    Phases are strictly ordered and each phase's prerequisites are
    declared in `_PHASE_PREREQUISITES`, mirroring the dependency-checked
    stage machinery of `scene_compiler.SceneCompiler` so both compiler
    stages fail fast and legibly when invoked out of order.
    """

    UNINITIALIZED = 0
    INITIALIZED = 1
    VALIDATED = 2
    NORMALIZED = 3
    ENTITIES_RESOLVED = 4
    ENVIRONMENT_RESOLVED = 5
    TERRAIN_RESOLVED = 6
    MATERIALS_RESOLVED = 7
    ASSETS_RESOLVED = 8
    TRANSFORMS_RESOLVED = 9
    PHYSICS_RESOLVED = 10
    CONSTRAINTS_RESOLVED = 11
    HIERARCHY_RESOLVED = 12
    OPTIMIZED = 13
    SCENE_GRAPH_GENERATED = 14
    COMPLETE = 15
    SHUTDOWN = 99

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


_PHASE_PREREQUISITES: dict[BuildPhase, tuple[BuildPhase, ...]] = {
    BuildPhase.VALIDATED: (BuildPhase.INITIALIZED,),
    BuildPhase.NORMALIZED: (BuildPhase.VALIDATED,),
    BuildPhase.ENTITIES_RESOLVED: (BuildPhase.NORMALIZED,),
    BuildPhase.ENVIRONMENT_RESOLVED: (BuildPhase.NORMALIZED,),
    BuildPhase.TERRAIN_RESOLVED: (BuildPhase.ENVIRONMENT_RESOLVED,),
    BuildPhase.MATERIALS_RESOLVED: (BuildPhase.ENTITIES_RESOLVED,),
    BuildPhase.ASSETS_RESOLVED: (BuildPhase.ENTITIES_RESOLVED,),
    BuildPhase.TRANSFORMS_RESOLVED: (BuildPhase.ENTITIES_RESOLVED,),
    BuildPhase.PHYSICS_RESOLVED: (BuildPhase.MATERIALS_RESOLVED,),
    BuildPhase.CONSTRAINTS_RESOLVED: (BuildPhase.ENTITIES_RESOLVED,),
    BuildPhase.HIERARCHY_RESOLVED: (BuildPhase.CONSTRAINTS_RESOLVED,),
    BuildPhase.OPTIMIZED: (
        BuildPhase.TERRAIN_RESOLVED,
        BuildPhase.MATERIALS_RESOLVED,
        BuildPhase.ASSETS_RESOLVED,
        BuildPhase.TRANSFORMS_RESOLVED,
        BuildPhase.PHYSICS_RESOLVED,
        BuildPhase.HIERARCHY_RESOLVED,
    ),
    BuildPhase.SCENE_GRAPH_GENERATED: (BuildPhase.OPTIMIZED,),
}

_KNOWN_TERRAIN_TYPES = frozenset({"flat", "hilly", "urban", "water", "mixed"})
_KNOWN_WEATHER = frozenset({"clear", "rain", "snow", "fog", "wind"})
_ASSET_TAG_PREFIX = "asset:"
# Reference "looks remote" (Nucleus/Omniverse/S3/etc.) and therefore
# cannot be verified against the local filesystem by design, rather than
# "wasn't found." Kept as a generic URI-scheme check (`scheme://`) so no
# backend (Omniverse or otherwise) is hardcoded here -- see High #2 in
# the review.
_REMOTE_ASSET_MARKER = "://"


# ════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Diagnostic:
    """A single, immutable structured diagnostic emitted during a build.

    Attributes:
        phase: The `BuildPhase` active when the diagnostic was raised.
        severity: One of INFO / WARNING / ERROR / CRITICAL.
        message: Human-readable description.
        source: Name of the resolver/pass that raised it.
        timestamp: UTC time the diagnostic was created.
        entity_ref: Optional entity id this diagnostic concerns.
    """

    phase: BuildPhase
    severity: Severity
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entity_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.label,
            "severity": self.severity.name,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "entity_ref": self.entity_ref,
        }

    def __str__(self) -> str:
        ref = f" (entity={self.entity_ref})" if self.entity_ref else ""
        return f"[{self.severity.name}] {self.phase.label} :: {self.source} :: {self.message}{ref}"


# ════════════════════════════════════════════════════════════════════════
# Statistics
# ════════════════════════════════════════════════════════════════════════

@dataclass
class BuildStatistics:
    """Quantitative summary of a single `WorldSpecBuilder` run."""

    build_time_s: float = 0.0
    entity_count: int = 0
    resolved_entity_count: int = 0
    relationship_count: int = 0
    material_count: int = 0
    asset_count: int = 0
    duplicate_entity_count: int = 0
    circular_dependency_count: int = 0
    hierarchy_depth: int = 0
    topological_order_length: int = 0
    optimization_passes: int = 0
    pruned_node_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    success: bool = False
    phase_durations_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "build_time_s": round(self.build_time_s, 6),
            "entity_count": self.entity_count,
            "resolved_entity_count": self.resolved_entity_count,
            "relationship_count": self.relationship_count,
            "material_count": self.material_count,
            "asset_count": self.asset_count,
            "duplicate_entity_count": self.duplicate_entity_count,
            "circular_dependency_count": self.circular_dependency_count,
            "hierarchy_depth": self.hierarchy_depth,
            "topological_order_length": self.topological_order_length,
            "optimization_passes": self.optimization_passes,
            "pruned_node_count": self.pruned_node_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "success": self.success,
            "phase_durations_s": {k: round(v, 6) for k, v in self.phase_durations_s.items()},
        }


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class WorldSpecBuilderConfig:
    """User-configurable settings controlling `WorldSpecBuilder` behavior.

    Attributes:
        validation_policy: STRICT aborts on the first validation problem;
            PERMISSIVE records problems as diagnostics and proceeds;
            DISABLED skips semantic validation altogether.
        deterministic: If True, SceneNode uuids are derived deterministically
            from entity ids (uuid5) rather than randomly generated (uuid4),
            so repeated builds of the same WorldSpec produce identical IR
            (modulo timestamps).
        asset_search_paths: Directories searched when resolving `asset:`-
            tagged references on entities.
        optimization_level: How aggressively `optimize()` prunes/merges
            scene nodes. Uses `scene_compiler.OptimizationLevel` so both
            compiler stages share one vocabulary; imported lazily.
        coordinate_system: Up-axis convention threaded through to the IR.
            Recorded on the generated `SceneGraph`'s Metadata node in
            `generate_scene_graph()`; previously accepted but never
            actually consumed anywhere (see Medium #1 in the review).
        unit_system: Unit convention (PhysWorldLM is SI end-to-end).
            Same fix as `coordinate_system` above.
        log_level: Python logging level name (e.g. "INFO", "DEBUG").
        id_factory: Optional override for SceneNode id generation, e.g.
            for reproducible tests. Defaults to uuid4/uuid5 depending on
            `deterministic`. Injected rather than hard-coded (DIP).
        asset_resolver: Optional override for asset-reference resolution,
            `(ref, search_paths) -> Optional[Path]`. Defaults to a simple
            filesystem probe. Injected rather than hard-coded (DIP).
    """

    validation_policy: ValidationPolicy = ValidationPolicy.STRICT
    deterministic: bool = True
    asset_search_paths: list[Path] = field(default_factory=list)
    optimization_level: "OptimizationLevel | None" = None
    coordinate_system: "CoordinateSystem | None" = None
    unit_system: "UnitSystem | None" = None
    log_level: str = "INFO"
    id_factory: Optional[Callable[[str], str]] = None
    asset_resolver: Optional[Callable[[str, list[Path]], Optional[Path]]] = None

    def __post_init__(self) -> None:
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))


# ════════════════════════════════════════════════════════════════════════
# Build report
# ════════════════════════════════════════════════════════════════════════

@dataclass
class BuildReport:
    """Final, structured result returned by `WorldSpecBuilder.build()`."""

    status: BuildStatus
    scene_id: str
    statistics: BuildStatistics
    diagnostics: list[Diagnostic]
    entity_order: tuple[str, ...] = field(default_factory=tuple)
    scene_graph: Optional["SceneGraph"] = None

    @property
    def success(self) -> bool:
        return self.status is not BuildStatus.FAILED

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity in (Severity.ERROR, Severity.CRITICAL)]

    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    def to_dict(self) -> dict:
        return {
            "status": self.status.name,
            "scene_id": self.scene_id,
            "statistics": self.statistics.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "entity_order": list(self.entity_order),
        }

    def __str__(self) -> str:
        lines = [
            f"BuildReport(scene_id={self.scene_id!r}, status={self.status.name})",
            f"  entities    : {self.statistics.entity_count}",
            f"  warnings    : {self.statistics.warning_count}",
            f"  errors      : {self.statistics.error_count}",
            f"  time        : {self.statistics.build_time_s:.4f}s",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Internal graph helpers (pure functions -> easy to unit test in isolation)
# ════════════════════════════════════════════════════════════════════════

def _find_duplicate_ids(ids: list[str]) -> list[str]:
    """Return ids that appear more than once, preserving first-seen order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for entity_id in ids:
        if entity_id in seen and entity_id not in duplicates:
            duplicates.append(entity_id)
        seen.add(entity_id)
    return duplicates


def _detect_cycle(graph: dict[str, set[str]]) -> Optional[list[str]]:
    """Depth-first cycle detection over a dependency graph.

    Args:
        graph: Adjacency mapping of node id -> set of ids it depends on.

    Returns:
        A list of node ids forming a cycle (in traversal order), or
        `None` if the graph is a DAG.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in graph}
    path: list[str] = []

    def visit(node: str) -> Optional[list[str]]:
        color[node] = GRAY
        path.append(node)
        for neighbor in graph.get(node, ()):
            if neighbor not in color:
                # Reference to a node outside the graph is a separate
                # concern (InvalidReferenceError), not a cycle.
                continue
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            if color[neighbor] == WHITE:
                result = visit(neighbor)
                if result is not None:
                    return result
        path.pop()
        color[node] = BLACK
        return None

    for start_node in list(graph):
        if color[start_node] == WHITE:
            found = visit(start_node)
            if found is not None:
                return found
    return None


def _topological_sort(graph: dict[str, set[str]]) -> list[str]:
    """Kahn's algorithm. Assumes `graph` has already been proven acyclic.

    Args:
        graph: Adjacency mapping of node id -> set of ids it depends on
            (i.e. edges point from a node to its prerequisites).

    Returns:
        Node ids ordered so that every node appears after all of its
        prerequisites, with ties broken by original insertion order for
        determinism.
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    dependents: dict[str, list[str]] = {node: [] for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep not in in_degree:
                continue
            in_degree[node] += 1
            dependents[dep].append(node)

    ready = [node for node in graph if in_degree[node] == 0]
    order: list[str] = []
    while ready:
        ready.sort()  # deterministic tie-break
        node = ready.pop(0)
        order.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                ready.append(dependent)
    return order


# ════════════════════════════════════════════════════════════════════════
# Internal mutable build state (private -- never exposed directly)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _BuildState:
    """All mutable state accumulated across one `build()` run.

    Kept as a single, clearly-owned object so `WorldSpecBuilder` itself
    stays a thin, stateless-looking orchestrator; this is the seam an
    external caller would swap out to, e.g., snapshot/replay a build.
    """

    world_spec: Optional["WorldSpec"] = None
    normalized_entities: tuple["Entity", ...] = field(default_factory=tuple)
    entity_index: dict[str, "Entity"] = field(default_factory=dict)
    node_index: dict[str, "SceneNode"] = field(default_factory=dict)
    dependency_graph: dict[str, set[str]] = field(default_factory=dict)
    topo_order: tuple[str, ...] = field(default_factory=tuple)
    asset_registry: dict[str, Path] = field(default_factory=dict)
    material_nodes: dict[str, "SceneNode"] = field(default_factory=dict)
    group_nodes: dict[str, "SceneNode"] = field(default_factory=dict)
    scene_graph: Optional["SceneGraph"] = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    statistics: BuildStatistics = field(default_factory=BuildStatistics)
    started_at: float = field(default_factory=time.monotonic)
    # Audit fix (Critical #3): entity ids for which a SceneNode has
    # already been created, so a duplicate id (possible in PERMISSIVE
    # mode, where validate() logs but does not abort on duplicates)
    # cannot produce a second SceneNode sharing the same deterministic
    # node_uuid. See resolve_entities().
    _resolved_entity_ids: set[str] = field(default_factory=set)


# ════════════════════════════════════════════════════════════════════════
# WorldSpecBuilder
# ════════════════════════════════════════════════════════════════════════

class WorldSpecBuilder:
    """Semantic compiler that lowers a validated `WorldSpec` into a
    backend-independent `SceneGraph` intermediate representation.

    `WorldSpecBuilder` performs no rendering, simulation, or asset I/O:
    it validates, normalizes, resolves references and hierarchy, runs
    compile-time optimizations, and emits the IR. Every resolution pass
    is independently callable and independently testable; `build()` is
    a convenience that runs them in the correct, dependency-checked
    order -- exactly the same shape as `scene_compiler.SceneCompiler`,
    one semantic level up the pipeline.

    Thread-safety: a single `WorldSpecBuilder` instance guards all of
    its mutable state behind one re-entrant lock, so its public methods
    may safely be called from multiple threads (though a given *build*
    -- the sequence from `validate()` through `export_report()` -- is
    inherently sequential and should be driven by one logical caller at
    a time). Every public phase method now acquires that lock via
    `_timed()` (audit fix, Critical #2 -- previously only `build()`,
    `initialize()`, and `shutdown()` took the lock, so calling resolver
    methods directly from multiple threads, as the step-through example
    below does, was unprotected despite this docstring's claim).

    Example:
        >>> builder = WorldSpecBuilder()
        >>> with builder:
        ...     report = builder.build(world_spec)
        >>> report.success
        True
        >>> report.scene_graph.node_count()  # doctest: +SKIP

    Custom id generation or asset resolution can be injected via
    `WorldSpecBuilderConfig` without subclassing::

        >>> config = WorldSpecBuilderConfig(id_factory=lambda seed: seed)
        >>> builder = WorldSpecBuilder(config)
    """

    def __init__(self, config: Optional[WorldSpecBuilderConfig] = None) -> None:
        """Initialize the builder with optional configuration.

        Args:
            config: Builder-wide settings. Defaults to
                `WorldSpecBuilderConfig()`.
        """
        self._config = config or WorldSpecBuilderConfig()
        self._lock = threading.RLock()
        self._phase: BuildPhase = BuildPhase.UNINITIALIZED
        self._completed: set[BuildPhase] = set()
        self._state: Optional[_BuildState] = None

    # ── context manager support ──────────────────────────────────────

    def __enter__(self) -> "WorldSpecBuilder":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ── lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Prepare the builder for a new build.

        Safe to call again after `shutdown()` to reuse the same
        `WorldSpecBuilder` instance for another `WorldSpec`.

        Raises:
            BuilderLifecycleError: If called while a build is already
                in progress (i.e. `initialize()` was already called and
                `shutdown()` has not been called since).
        """
        with self._lock:
            if self._phase not in (BuildPhase.UNINITIALIZED, BuildPhase.SHUTDOWN, BuildPhase.COMPLETE):
                raise BuilderLifecycleError(
                    f"Cannot initialize(): builder is already active in phase '{self._phase.label}'. "
                    "Call shutdown() first."
                )
            self._state = _BuildState()
            self._completed = {BuildPhase.INITIALIZED}
            self._phase = BuildPhase.INITIALIZED
            self._completed.add(BuildPhase.INITIALIZED)
            logger.info("WorldSpecBuilder initialized.")

    def shutdown(self) -> None:
        """Release all build state.

        Idempotent: calling `shutdown()` more than once is a no-op.
        """
        with self._lock:
            if self._phase is BuildPhase.SHUTDOWN:
                return
            self._state = None
            self._completed = set()
            self._phase = BuildPhase.SHUTDOWN
            logger.info("WorldSpecBuilder shut down.")

    # ── phase bookkeeping helpers ─────────────────────────────────────

    def _require_state(self) -> _BuildState:
        if self._state is None:
            raise BuilderLifecycleError(
                "Builder has no active build state. Call initialize() (or use `build()`/a "
                "`with WorldSpecBuilder(...)` block) before invoking resolver methods."
            )
        return self._state

    def _require_phase(self, target: BuildPhase) -> None:
        missing = [
            dep for dep in _PHASE_PREREQUISITES.get(target, ()) if dep not in self._completed
        ]
        if missing:
            raise BuilderLifecycleError(
                f"Cannot run '{target.label}': missing prerequisite phase(s) "
                f"{[m.label for m in missing]}."
            )

    def _mark_complete(self, phase: BuildPhase, duration_s: float) -> None:
        self._completed.add(phase)
        if phase.value > self._phase.value:
            self._phase = phase
        state = self._require_state()
        state.statistics.phase_durations_s[phase.label] = duration_s

    def _timed(self, phase: BuildPhase, fn: Callable[[], Any]) -> Any:
        """Run `fn` under the instance lock, gated on `phase`'s prerequisites,
        and record its duration.

        Audit fix (Critical #2): every public phase method below now
        routes through here instead of duplicating
        `_require_phase`/timing/`_mark_complete` inline (previously this
        method existed but was never called, so the timing/locking logic
        was hand-copied -- and un-locked -- into eleven separate
        methods). Centralizing it both removes that duplication and
        closes the thread-safety gap: `validate()`, `resolve_entities()`,
        etc. can now safely be called from multiple threads, matching
        this class's docstring.
        """
        with self._lock:
            self._require_phase(phase)
            start = time.monotonic()
            logger.info("Phase start    : %s", phase.label)
            result = fn()
            duration = time.monotonic() - start
            self._mark_complete(phase, duration)
            logger.info("Phase complete : %-22s duration=%.4fs", phase.label, duration)
            return result

    # ── scene_compiler integration helper ─────────────────────────────

    @staticmethod
    def _import_scene_compiler_symbols(*names: str) -> tuple[Any, ...]:
        """Import symbols from `scene_compiler` by name, wrapping any
        failure in `SceneCompilerIntegrationError`.

        Audit fix (Critical #1): every resolver previously did a bare
        `from scene_compiler import X, Y` inside its `_run()` closure.
        If `scene_compiler` is missing, not yet on `sys.path`, or no
        longer exports one of these names, that raised a raw
        `ImportError`/`AttributeError` that `build()` did not catch
        (its `except` clauses only handle `SpecValidationError` /
        `WorldSpecBuilderError`), silently breaking the "build() always
        returns a BuildReport" contract. All lazy imports now go through
        this helper instead.
        """
        try:
            from . import scene_compiler as _sc
        except ImportError as exc:
            raise SceneCompilerIntegrationError(
                f"scene_compiler module could not be imported: {exc}"
            ) from exc
        try:
            return tuple(getattr(_sc, name) for name in names)
        except AttributeError as exc:
            raise SceneCompilerIntegrationError(
                f"scene_compiler is missing expected symbol(s) {names}: {exc}"
            ) from exc

    # ── diagnostics helpers ───────────────────────────────────────────

    def _log(
        self,
        phase: BuildPhase,
        severity: Severity,
        message: str,
        source: str,
        entity_ref: Optional[str] = None,
    ) -> Diagnostic:
        state = self._require_state()
        diag = Diagnostic(phase=phase, severity=severity, message=message, source=source, entity_ref=entity_ref)
        state.diagnostics.append(diag)
        if severity is Severity.WARNING:
            state.statistics.warning_count += 1
            logger.warning(str(diag))
        elif severity in (Severity.ERROR, Severity.CRITICAL):
            state.statistics.error_count += 1
            logger.error(str(diag))
        else:
            logger.info(str(diag))
        return diag

    # ── public API: build() convenience orchestrator ─────────────────

    def build(self, world_spec: "WorldSpec") -> BuildReport:
        """Run the full WorldSpec -> SceneGraph pipeline.

        Equivalent to calling `validate()`, `normalize()`,
        `resolve_entities()`, `resolve_environment()`,
        `resolve_terrain()`, `resolve_materials()`, `resolve_assets()`,
        `resolve_transforms()`, `resolve_physics()`,
        `resolve_constraints()`, `resolve_hierarchy()`, `optimize()`,
        `generate_scene_graph()`, and `export_report()` in order.

        Calls `initialize()` automatically if the builder is not
        already active.

        Args:
            world_spec: A `WorldSpec` instance to lower into a
                `SceneGraph`.

        Returns:
            A `BuildReport` describing the outcome, statistics,
            diagnostics, and (on success) the resulting `SceneGraph`.
            Every code path -- including an unexpected, non-builder
            exception raised deep inside a resolver -- now returns a
            `BuildReport` rather than propagating (audit fix, Critical
            #1); the report's diagnostics will contain a CRITICAL entry
            describing the underlying failure in that case.
        """
        with self._lock:
            if self._phase in (BuildPhase.UNINITIALIZED, BuildPhase.SHUTDOWN, BuildPhase.COMPLETE):
                self.initialize()

            logger.info("=" * 72)
            logger.info("Building WorldSpec '%s'", getattr(world_spec, "scene_id", "<unknown>"))
            logger.info("=" * 72)

            try:
                self.validate(world_spec)
                self.normalize()
                self.resolve_entities()
                self.resolve_environment()
                self.resolve_terrain()
                self.resolve_materials()
                self.resolve_assets()
                self.resolve_transforms()
                self.resolve_physics()
                self.resolve_constraints()
                self.resolve_hierarchy()
                self.optimize()
                self.generate_scene_graph()
            except SpecValidationError as exc:
                return self._fail_build(world_spec, exc)
            except WorldSpecBuilderError as exc:
                return self._fail_build(world_spec, exc)
            except Exception as exc:  # noqa: BLE001 -- audit fix, Critical #1
                # Anything that escapes the resolvers and isn't already a
                # WorldSpecBuilderError (e.g. a bug in a resolver, or an
                # unexpected failure inside scene_compiler that slipped
                # past `_import_scene_compiler_symbols`) is caught here
                # so `build()` never raises a bare exception to the
                # caller. The full traceback still goes to the logger.
                logger.exception("Unexpected error during build().")
                return self._fail_build(world_spec, exc, unexpected=True)

            report = self.export_report()
            self._phase = BuildPhase.COMPLETE
            logger.info(
                "Build finished for scene '%s' -> status=%s",
                report.scene_id,
                report.status.name,
            )
            return report

    def _fail_build(self, world_spec: "WorldSpec", exc: Exception, unexpected: bool = False) -> BuildReport:
        state = self._require_state()
        state.statistics.build_time_s = time.monotonic() - state.started_at
        state.statistics.success = False
        if unexpected:
            self._log(
                self._phase,
                Severity.CRITICAL,
                f"Unexpected {type(exc).__name__}: {exc}",
                "build",
            )
        else:
            self._log(
                self._phase,
                Severity.ERROR,
                str(exc),
                "build",
            )

        logger.error("Build FAILED: %s", exc)
        report = BuildReport(
            status=BuildStatus.FAILED,
            scene_id=getattr(world_spec, "scene_id", ""),
            statistics=state.statistics,
            diagnostics=list(state.diagnostics),
        )
        self._phase = BuildPhase.COMPLETE
        return report

    # ── public API: validate ──────────────────────────────────────────

    def validate(self, world_spec: "WorldSpec") -> None:
        """Semantically validate `world_spec` before any resolution runs.

        Checks (beyond whatever structural validation the parser/ontology
        layer already performed): non-empty scene id, duplicate entity
        ids, positive mass for dynamic entities, positive bounding-box
        dimensions, valid interaction references, and positive
        simulation timestep/duration. Unknown materials/terrain types
        are recorded as warnings here (they are resolved permissively
        with a `"generic"` fallback in `resolve_materials()`), unless
        `validation_policy` is STRICT, in which case they escalate to
        errors.

        Args:
            world_spec: The `WorldSpec` to validate.

        Raises:
            DuplicateEntityError: Under STRICT policy, if two entities
                share an id.
            InvalidReferenceError: Under STRICT policy, if an
                interaction references an unknown entity.
            SpecValidationError: Under STRICT policy, for any other
                aggregated validation problem.
        """
        def _run() -> None:
            state = self._require_state()
            state.world_spec = world_spec
            policy = self._config.validation_policy

            if policy is ValidationPolicy.DISABLED:
                self._log(BuildPhase.VALIDATED, Severity.INFO, "Validation disabled by configuration.", "validator")
                return

            problems: list[str] = []
            entity_ids = [e.id for e in world_spec.entities]

            if not world_spec.scene_id:
                problems.append("WorldSpec.scene_id must be a non-empty string.")

            duplicates = _find_duplicate_ids(entity_ids)
            if duplicates:
                state.statistics.duplicate_entity_count = len(duplicates)
                problems.append(f"Duplicate entity id(s): {duplicates}.")

            for entity in world_spec.entities:
                if not entity.id:
                    problems.append("Entity found with empty id.")
                    continue
                if not entity.is_static and entity.mass <= 0:
                    problems.append(f"Dynamic entity '{entity.id}' must have mass > 0 (got {entity.mass}).")
                for axis_val, axis_name in (
                    (entity.bounding_box.width, "width"),
                    (entity.bounding_box.height, "height"),
                    (entity.bounding_box.depth, "depth"),
                ):
                    if axis_val <= 0:
                        problems.append(f"Entity '{entity.id}' has non-positive bounding_box.{axis_name}.")
                if not (0.0 <= entity.restitution <= 1.0):
                    problems.append(f"Entity '{entity.id}' has out-of-range restitution ({entity.restitution}).")
                if entity.friction < 0.0:
                    problems.append(f"Entity '{entity.id}' has negative friction ({entity.friction}).")

            entity_id_set = set(entity_ids)

            for interaction in world_spec.interactions:
                if interaction.entity_a not in entity_id_set:
                    msg = f"Interaction references unknown entity_a '{interaction.entity_a}'."

                    if policy is ValidationPolicy.STRICT:
                        problems.append(msg)
                    else:
                        self._log(
                            BuildPhase.VALIDATED,
                            Severity.WARNING,
                            msg,
                            "validator",
                        )

                if (
                    interaction.entity_b != "environment"
                    and interaction.entity_b not in entity_id_set
                ):
                    msg = f"Interaction references unknown entity_b '{interaction.entity_b}'."

                    if policy is ValidationPolicy.STRICT:
                        problems.append(msg)
                    else:
                        self._log(
                            BuildPhase.VALIDATED,
                            Severity.WARNING,
                            msg,
                            "validator",
                        )

            for entity in world_spec.entities:
                for dep_id in entity.constraints:
                    if dep_id not in entity_id_set:
                        msg = (
                            f"Entity '{entity.id}' has a constraint "
                            f"referencing unknown entity '{dep_id}'."
                        )

                        if policy is ValidationPolicy.STRICT:
                            problems.append(msg)
                        else:
                            self._log(
                                BuildPhase.VALIDATED,
                                Severity.WARNING,
                                msg,
                                "validator",
                                entity.id,
                            )

            if world_spec.simulation_graph.dt <= 0:
                problems.append("SimulationGraph.dt must be > 0.")
            if world_spec.simulation_graph.duration <= 0:
                problems.append("SimulationGraph.duration must be > 0.")

            if world_spec.environment.terrain_type not in _KNOWN_TERRAIN_TYPES:
                msg = f"Unknown terrain_type '{world_spec.environment.terrain_type}'."
                if policy is ValidationPolicy.STRICT:
                    problems.append(msg)
                else:
                    self._log(BuildPhase.VALIDATED, Severity.WARNING, msg, "validator")

            if world_spec.environment.weather not in _KNOWN_WEATHER:
                msg = f"Unknown weather '{world_spec.environment.weather}'."
                if policy is ValidationPolicy.STRICT:
                    problems.append(msg)
                else:
                    self._log(BuildPhase.VALIDATED, Severity.WARNING, msg, "validator")

            if problems:
                if policy is ValidationPolicy.STRICT:
                    raise SpecValidationError(problems)
                for problem in problems:
                    self._log(BuildPhase.VALIDATED, Severity.ERROR, problem, "validator")

            self._log(
                BuildPhase.VALIDATED,
                Severity.INFO,
                f"Validated WorldSpec '{world_spec.scene_id}' ({len(world_spec.entities)} entities).",
                "validator",
            )
            state.statistics.entity_count = len(world_spec.entities)

        self._require_state()
        self._timed(BuildPhase.VALIDATED, _run)

    # ── public API: normalize ─────────────────────────────────────────

    def normalize(self) -> None:
        """Normalize units, casing, and value ranges into canonical form.

        Produces an immutable, normalized copy of the entity list (the
        original `WorldSpec` passed to `validate()` is never mutated):
        material names are lower-cased, tags are lower-cased/deduped,
        and orientation angles are wrapped into `(-pi, pi]`. All
        quantities were already SI per the `WorldSpec` contract, so this
        pass focuses on canonicalization rather than unit conversion.

        Raises:
            BuilderLifecycleError: If called before `validate()`.
        """
        from dataclasses import replace as _replace

        def _run() -> None:
            state = self._require_state()
            world_spec = state.world_spec
            normalized: list["Entity"] = []
            for entity in world_spec.entities:  # type: ignore[union-attr]
                wrapped_orientation = entity.state.orientation.__class__(
                    x=self._wrap_angle(entity.state.orientation.x),
                    y=self._wrap_angle(entity.state.orientation.y),
                    z=self._wrap_angle(entity.state.orientation.z),
                )
                normalized_state = _replace(entity.state, orientation=wrapped_orientation)
                normalized_entity = _replace(
                    entity,
                    material=entity.material.strip().lower(),
                    tags=sorted({t.strip().lower() for t in entity.tags if t.strip()}),
                    state=normalized_state,
                )
                normalized.append(normalized_entity)
            state.normalized_entities = tuple(normalized)
            state.entity_index = {e.id: e for e in normalized}
            self._log(
                BuildPhase.NORMALIZED,
                Severity.INFO,
                f"Normalized {len(normalized)} entity/entities.",
                "normalizer",
            )

        self._timed(BuildPhase.NORMALIZED, _run)

    @staticmethod
    def _wrap_angle(radians: float) -> float:
        import math
        return (radians + math.pi) % (2 * math.pi) - math.pi

    # ── public API: resolve_entities ──────────────────────────────────

    def resolve_entities(self) -> None:
        """Create one `SceneNode` (NodeType.ENTITY) per normalized entity.

        Node ids are derived deterministically (`uuid5`) when
        `config.deterministic` is True, or via `config.id_factory` if
        supplied, matching the same convention `scene_compiler` uses so
        node identity is stable across recompiles.

        Entities that share a duplicate id (possible under PERMISSIVE
        validation, where `validate()` records the problem but does not
        abort) are skipped after the first occurrence rather than
        silently producing a second `SceneNode` with the same
        deterministic `node_uuid` (audit fix, Critical #3): two distinct
        nodes sharing a uuid corrupted every uuid-keyed lookup
        downstream (`node_index`, `material_ref`, `physics_ref`) and
        left the second node permanently unresolved (no transform, no
        material, no physics body) with no diagnostic explaining why.

        Raises:
            BuilderLifecycleError: If called before `normalize()`.
        """
        def _run() -> None:
            NodeType, SceneNode = self._import_scene_compiler_symbols("NodeType", "SceneNode")

            state = self._require_state()
            entities_group = SceneNode(name="Entities", node_type=NodeType.ENTITIES_GROUP)
            state.group_nodes["Entities"] = entities_group

            skipped_duplicates = 0
            for entity in state.normalized_entities:
                if entity.id in state._resolved_entity_ids:
                    skipped_duplicates += 1
                    self._log(
                        BuildPhase.ENTITIES_RESOLVED,
                        Severity.ERROR,
                        f"Duplicate entity id '{entity.id}' encountered a second time; "
                        "skipping the duplicate rather than emitting a second SceneNode "
                        "with the same node_uuid.",
                        "entity_resolver",
                        entity.id,
                    )
                    continue

                node = SceneNode(
                    name=entity.label or entity.id,
                    node_type=NodeType.ENTITY,
                    node_uuid=self._entity_node_id(entity.id),
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
                entities_group.add_child(node)
                state.node_index[entity.id] = node
                state._resolved_entity_ids.add(entity.id)

            state.statistics.resolved_entity_count = len(state.node_index)
            self._log(
                BuildPhase.ENTITIES_RESOLVED,
                Severity.INFO,
                f"Resolved {len(state.node_index)} entity node(s)"
                + (f", skipped {skipped_duplicates} duplicate(s)." if skipped_duplicates else "."),
                "entity_resolver",
            )

        self._timed(BuildPhase.ENTITIES_RESOLVED, _run)

    def _entity_node_id(self, entity_id: str) -> str:
        if self._config.id_factory is not None:
            return self._config.id_factory(entity_id)
        if self._config.deterministic:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"physworldlm://entity/{entity_id}"))
        return str(uuid.uuid4())

    # ── public API: resolve_environment ───────────────────────────────

    def resolve_environment(self) -> None:
        """Build Environment / Terrain / Atmosphere / Weather / Lighting nodes.

        Raises:
            BuilderLifecycleError: If called before `normalize()`.
        """
        def _run() -> None:
            NodeType, SceneNode = self._import_scene_compiler_symbols("NodeType", "SceneNode")

            state = self._require_state()
            env = state.world_spec.environment  # type: ignore[union-attr]

            env_node = SceneNode(name="Environment", node_type=NodeType.ENVIRONMENT)
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

            state.group_nodes["Environment"] = env_node
            state.group_nodes["Terrain"] = terrain_node
            self._log(BuildPhase.ENVIRONMENT_RESOLVED, Severity.INFO, "Environment hierarchy resolved.", "environment_resolver")

        self._timed(BuildPhase.ENVIRONMENT_RESOLVED, _run)

    # ── public API: resolve_terrain ───────────────────────────────────

    def resolve_terrain(self) -> None:
        """Validate and annotate the terrain type against the known vocabulary.

        Raises:
            InvalidTerrainError: Under STRICT policy, if the terrain
                type is not one of the known vocabulary values.
            BuilderLifecycleError: If called before `resolve_environment()`.
        """
        def _run() -> None:
            state = self._require_state()
            terrain_type = state.world_spec.environment.terrain_type  # type: ignore[union-attr]
            if terrain_type not in _KNOWN_TERRAIN_TYPES:
                message = (
                    f"Terrain type '{terrain_type}' is not in the known vocabulary "
                    f"{sorted(_KNOWN_TERRAIN_TYPES)}."
                )
                if self._config.validation_policy is ValidationPolicy.STRICT:
                    raise InvalidTerrainError([message])
                self._log(BuildPhase.TERRAIN_RESOLVED, Severity.WARNING, message, "terrain_resolver")
            else:
                self._log(
                    BuildPhase.TERRAIN_RESOLVED,
                    Severity.INFO,
                    f"Terrain type '{terrain_type}' confirmed.",
                    "terrain_resolver",
                )
            terrain_node = state.group_nodes.get("Terrain")
            if terrain_node is not None:
                terrain_node.metadata["validated"] = terrain_type in _KNOWN_TERRAIN_TYPES

        self._timed(BuildPhase.TERRAIN_RESOLVED, _run)

    # ── public API: resolve_materials ─────────────────────────────────

    def resolve_materials(self) -> None:
        """Build de-duplicated Material nodes from entity material references.

        Unknown material names fall back to the `"generic"` entry of
        `world_spec.MATERIAL_DEFAULTS` under PERMISSIVE policy; under
        STRICT policy an unknown material raises.

        A Material node stores that material's *canonical* physical
        defaults (density/restitution/friction), not any one entity's
        per-instance overrides (audit fix, High #1): the previous
        version wrote `entity.restitution`/`entity.friction` onto the
        shared material node whenever `entity.material == mat_name`,
        which meant the *first* entity in iteration order to use a given
        material silently redefined that material's properties for
        every other entity sharing it -- an order-dependent bug that
        also undermined the "deterministic ordering" contract on the
        generated `SceneGraph`. Per-entity overrides still flow through
        correctly via `resolve_physics()`, which reads
        `entity.restitution`/`entity.friction` directly onto each
        entity's own `physics_body` component.

        Raises:
            InvalidMaterialError: Under STRICT policy, if an entity
                references an unknown material.
            BuilderLifecycleError: If called before `resolve_entities()`.
        """
        def _run() -> None:
            NodeType, SceneNode = self._import_scene_compiler_symbols("NodeType", "SceneNode")
            from world_spec import MATERIAL_DEFAULTS

            state = self._require_state()
            materials_group = SceneNode(name="Materials", node_type=NodeType.MATERIALS_GROUP)
            state.group_nodes["Materials"] = materials_group

            for entity in state.normalized_entities:
                mat_name = entity.material
                if mat_name not in MATERIAL_DEFAULTS:
                    message = f"Entity '{entity.id}' references unknown material '{mat_name}'."
                    if self._config.validation_policy is ValidationPolicy.STRICT:
                        raise InvalidMaterialError([message])
                    self._log(BuildPhase.MATERIALS_RESOLVED, Severity.WARNING, message, "material_resolver", entity.id)
                    mat_name = "generic"

                if mat_name not in state.material_nodes:
                    defaults = MATERIAL_DEFAULTS[mat_name]
                    mat_node = SceneNode(name=mat_name, node_type=NodeType.MATERIAL)
                    mat_node.components["material"] = {
                        "name": mat_name,
                        "density": defaults["density"],
                        "restitution": defaults["restitution"],
                        "friction": defaults["friction"],
                    }
                    materials_group.add_child(mat_node)
                    state.material_nodes[mat_name] = mat_node

                entity_node = state.node_index.get(entity.id)
                if entity_node is not None:
                    entity_node.components["material_ref"] = state.material_nodes[mat_name].node_uuid

            state.statistics.material_count = len(state.material_nodes)
            self._log(
                BuildPhase.MATERIALS_RESOLVED,
                Severity.INFO,
                f"Resolved {len(state.material_nodes)} unique material node(s).",
                "material_resolver",
            )

        self._timed(BuildPhase.MATERIALS_RESOLVED, _run)

    # ── public API: resolve_assets ─────────────────────────────────────

    def resolve_assets(self) -> None:
        """Resolve `asset:`-tagged entity references against search paths.

        The current `WorldSpec` contract expresses geometry implicitly
        via `bounding_box` + procedural primitives rather than explicit
        mesh references, so this pass walks `entity.tags` defensively
        looking for an `asset:` prefix and is a safe no-op when none are
        present -- the integration point for future explicit geometry
        assets.

        Raises:
            MissingAssetError: Under STRICT policy, if a referenced
                asset cannot be found on any configured search path.
            BuilderLifecycleError: If called before `resolve_entities()`.
        """
        def _run() -> None:
            state = self._require_state()
            resolver = self._config.asset_resolver or self._default_asset_resolver
            resolved = 0

            for entity in state.normalized_entities:
                for tag in entity.tags:
                    if not tag.startswith(_ASSET_TAG_PREFIX):
                        continue
                    ref = tag[len(_ASSET_TAG_PREFIX):]
                    path = resolver(ref, self._config.asset_search_paths)
                    if path is None:
                        message = f"Could not resolve asset reference '{ref}' for entity '{entity.id}'."
                        if self._config.validation_policy is ValidationPolicy.STRICT:
                            raise MissingAssetError([message])
                        self._log(BuildPhase.ASSETS_RESOLVED, Severity.WARNING, message, "asset_resolver", entity.id)
                        continue
                    state.asset_registry[ref] = path
                    node = state.node_index.get(entity.id)
                    if node is not None:
                        node.components.setdefault("assets", []).append(str(path))
                    resolved += 1

            state.statistics.asset_count = len(state.asset_registry)
            self._log(BuildPhase.ASSETS_RESOLVED, Severity.INFO, f"Resolved {resolved} asset reference(s).", "asset_resolver")

        self._timed(BuildPhase.ASSETS_RESOLVED, _run)

    @staticmethod
    def _default_asset_resolver(ref: str, search_paths: list[Path]) -> Optional[Path]:
        """Resolve a local filesystem asset reference.

        Audit fix (Critical #4): the previous implementation always
        returned a `Path` -- even for a reference that could not be
        found anywhere -- so `resolve_assets()`'s `path is None` branch
        (and therefore `MissingAssetError` under STRICT policy) was
        effectively dead code for the common case of a relative,
        filesystem-style reference that simply doesn't exist yet.

        A reference containing a URI scheme (`"scheme://..."`, e.g. a
        future `omniverse://` or `nucleus://` path) is still returned
        as-is without a local existence check, since verifying those is
        explicitly out of scope for this resolver and they may be
        legitimately valid despite not existing on the local disk. Any
        other reference -- absolute or relative -- must actually resolve
        to a real path on disk, or `None` is returned so the configured
        `validation_policy` governs what happens next.
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

        # Bare relative ref with no search paths configured, or not
        # found on any of them: genuinely unresolved.
        return candidate if (not search_paths and candidate.exists()) else None

    # ── public API: resolve_transforms ────────────────────────────────

    def resolve_transforms(self) -> None:
        """Apply per-entity kinematic state as local-space `Transform`s.

        Raises:
            BuilderLifecycleError: If called before `resolve_entities()`.
        """
        def _run() -> None:
            (Transform,) = self._import_scene_compiler_symbols("Transform")

            state = self._require_state()
            applied = 0
            for entity in state.normalized_entities:
                node = state.node_index.get(entity.id)
                if node is None:
                    self._log(
                        BuildPhase.TRANSFORMS_RESOLVED,
                        Severity.WARNING,
                        f"No scene node found for entity '{entity.id}'; skipping transform.",
                        "transform_resolver",
                        entity.id,
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
            self._log(
                BuildPhase.TRANSFORMS_RESOLVED,
                Severity.INFO,
                f"Resolved transforms for {applied} entity node(s).",
                "transform_resolver",
            )

        self._timed(BuildPhase.TRANSFORMS_RESOLVED, _run)

    # ── public API: resolve_physics ───────────────────────────────────

    def resolve_physics(self) -> None:
        """Attach physics-body metadata and check for inconsistent physics.

        Raises:
            InconsistentPhysicsError: If a dynamic entity has
                non-positive mass, or restitution/friction fall outside
                their valid ranges (under STRICT policy for the latter;
                mass is always fatal since it would make simulation
                undefined).
            BuilderLifecycleError: If called before `resolve_materials()`.
        """
        def _run() -> None:
            NodeType, SceneNode = self._import_scene_compiler_symbols("NodeType", "SceneNode")

            state = self._require_state()
            physics_group = SceneNode(name="Physics", node_type=NodeType.PHYSICS_GROUP)
            state.group_nodes["Physics"] = physics_group

            for entity in state.normalized_entities:
                entity_node = state.node_index.get(entity.id)
                if entity_node is None:
                    continue

                if not entity.is_static and entity.mass <= 0:
                    raise InconsistentPhysicsError(
                        [f"Dynamic entity '{entity.id}' has non-positive mass ({entity.mass} kg)."]
                    )
                if not (0.0 <= entity.restitution <= 1.0):
                    message = f"Entity '{entity.id}' has out-of-range restitution ({entity.restitution})."
                    if self._config.validation_policy is ValidationPolicy.STRICT:
                        raise InconsistentPhysicsError([message])
                    self._log(BuildPhase.PHYSICS_RESOLVED, Severity.WARNING, message, "physics_resolver", entity.id)

                body_node = SceneNode(name=f"{entity_node.name}_physics", node_type=NodeType.PHYSICS_BODY)
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

            self._log(
                BuildPhase.PHYSICS_RESOLVED,
                Severity.INFO,
                f"Attached physics metadata to {len(state.normalized_entities)} entity node(s).",
                "physics_resolver",
            )

        self._timed(BuildPhase.PHYSICS_RESOLVED, _run)

    # ── public API: resolve_constraints ───────────────────────────────

    def resolve_constraints(self) -> None:
        """Build the dependency graph edges from constraints and interactions.

        Populates `_BuildState.dependency_graph` (entity id -> set of
        entity ids it depends on) from both `entity.constraints` and
        `WorldSpec.interactions`, and records relationship edges onto
        each entity's `SceneNode` for downstream consumers.

        Raises:
            InvalidReferenceError: If a constraint or interaction
                references an unknown entity (should already have been
                caught in `validate()` under STRICT policy; re-checked
                here defensively for PERMISSIVE builds).
            BuilderLifecycleError: If called before `resolve_entities()`.
        """
        def _run() -> None:
            state = self._require_state()
            entity_ids = set(state.entity_index)
            graph: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
            relationship_count = 0

            for entity in state.normalized_entities:
                for dep_id in entity.constraints:
                    if dep_id not in entity_ids:
                        message = f"Entity '{entity.id}' has a constraint referencing unknown entity '{dep_id}'."
                        if self._config.validation_policy is ValidationPolicy.STRICT:
                            raise InvalidReferenceError([message])
                        self._log(BuildPhase.CONSTRAINTS_RESOLVED, Severity.WARNING, message, "constraint_resolver", entity.id)
                        continue
                    graph[entity.id].add(dep_id)

            world_spec = state.world_spec
            for interaction in world_spec.interactions:  # type: ignore[union-attr]
                a_node = state.node_index.get(interaction.entity_a)
                if a_node is None:
                    message = f"Interaction references unknown entity_a '{interaction.entity_a}'."
                    if self._config.validation_policy is ValidationPolicy.STRICT:
                        raise InvalidReferenceError([message])
                    self._log(BuildPhase.CONSTRAINTS_RESOLVED, Severity.WARNING, message, "constraint_resolver")
                    continue

                target_id = interaction.entity_b
                if target_id != "environment" and target_id not in entity_ids:
                    message = f"Interaction references unknown entity_b '{target_id}'."
                    if self._config.validation_policy is ValidationPolicy.STRICT:
                        raise InvalidReferenceError([message])
                    self._log(BuildPhase.CONSTRAINTS_RESOLVED, Severity.WARNING, message, "constraint_resolver")
                    continue

                a_node.components.setdefault("relationships", []).append(
                    {"type": interaction.type, "target": target_id, "parameters": interaction.parameters}
                )
                if target_id != "environment":
                    graph[interaction.entity_a].add(target_id)
                relationship_count += 1

            state.dependency_graph = graph
            state.statistics.relationship_count = relationship_count
            self._log(
                BuildPhase.CONSTRAINTS_RESOLVED,
                Severity.INFO,
                f"Resolved {relationship_count} relationship edge(s) across {len(graph)} entity/entities.",
                "constraint_resolver",
            )

        self._timed(BuildPhase.CONSTRAINTS_RESOLVED, _run)

    # ── public API: resolve_hierarchy ─────────────────────────────────

    def resolve_hierarchy(self) -> None:
        """Detect cycles and compute a deterministic topological build order.

        Also re-parents entity nodes under one another in the
        `SceneGraph` for interactions of relationship type `"joint"` or
        `"mount"`, so parent/child spatial nesting is reflected in the
        emitted IR rather than left as a flat list with side-channel
        metadata.

        Raises:
            CircularDependencyError: If the constraint/interaction graph
                contains a cycle.
            HierarchyResolutionError: If an entity marked as a
                joint/mount child cannot be uniquely re-parented (e.g.
                because it was already re-parented once).
            BuilderLifecycleError: If called before `resolve_constraints()`.
        """
        def _run() -> None:
            state = self._require_state()
            cycle = _detect_cycle(state.dependency_graph)
            if cycle is not None:
                state.statistics.circular_dependency_count += 1
                raise CircularDependencyError([f"Circular dependency detected: {' -> '.join(cycle)}."])

            order = _topological_sort(state.dependency_graph)
            state.topo_order = tuple(order)
            state.statistics.topological_order_length = len(order)

            # Compute hierarchy depth (longest dependency chain) and
            # re-parent joint/mount children under their parent node.
            depth_cache: dict[str, int] = {}

            def depth_of(node_id: str) -> int:
                if node_id in depth_cache:
                    return depth_cache[node_id]
                deps = state.dependency_graph.get(node_id, ())
                depth_cache[node_id] = 1 + max((depth_of(d) for d in deps), default=0) if deps else 0
                return depth_cache[node_id]

            max_depth = max((depth_of(node_id) for node_id in state.dependency_graph), default=0)
            state.statistics.hierarchy_depth = max_depth

            reparented: set[str] = set()
            for entity in state.normalized_entities:
                child_node = state.node_index.get(entity.id)
                if child_node is None:
                    continue
                for relationship in child_node.components.get("relationships", []):
                    if relationship["type"] not in ("joint", "mount"):
                        continue
                    parent_id = relationship["target"]
                    if parent_id == "environment" or parent_id not in state.node_index:
                        continue
                    if entity.id in reparented:
                        raise HierarchyResolutionError(
                            [f"Entity '{entity.id}' has more than one joint/mount parent; ambiguous hierarchy."]
                        )
                    parent_node = state.node_index[parent_id]
                    entities_group = state.group_nodes.get("Entities")
                    if entities_group is not None:
                        # Audit fix (Low #1): remove by identity, not by
                        # `list.remove()` (which uses `__eq__`). If
                        # SceneNode is a dataclass with structural
                        # equality, two independently-constructed nodes
                        # that happen to compare equal could cause the
                        # wrong sibling to be removed.
                        for i, sibling in enumerate(entities_group.children):
                            if sibling is child_node:
                                del entities_group.children[i]
                                break
                    parent_node.add_child(child_node)
                    reparented.add(entity.id)

            self._log(
                BuildPhase.HIERARCHY_RESOLVED,
                Severity.INFO,
                f"Hierarchy resolved: depth={max_depth}, topological_order_length={len(order)}, "
                f"re-parented={len(reparented)}.",
                "hierarchy_resolver",
            )

        self._timed(BuildPhase.HIERARCHY_RESOLVED, _run)

    # ── public API: optimize ──────────────────────────────────────────

    def optimize(self) -> None:
        """Run compile-time optimization passes on the resolved IR.

        `OptimizationLevel.NONE` is a no-op. `BASIC` prunes empty,
        disabled, invisible leaf nodes with no components (dead scene
        nodes that would add nothing to any backend export). `AGGRESSIVE`
        additionally flattens single-child wrapper groups.

        Raises:
            BuilderLifecycleError: If called before all resolver phases
                have completed.
        """
        def _run() -> None:
            (OptimizationLevel,) = self._import_scene_compiler_symbols("OptimizationLevel")

            state = self._require_state()
            level = self._config.optimization_level or OptimizationLevel.BASIC
            passes = 0
            pruned = 0

            if level is OptimizationLevel.NONE:
                self._log(BuildPhase.OPTIMIZED, Severity.INFO, "Optimization disabled by configuration.", "optimizer")
            else:
                for group_node in list(state.group_nodes.values()):
                    pruned += self._prune_dead_nodes(group_node)
                passes += 1

                if level is OptimizationLevel.AGGRESSIVE:
                    for group_node in list(state.group_nodes.values()):
                        self._flatten_single_child_groups(group_node)
                    passes += 1

                self._log(
                    BuildPhase.OPTIMIZED,
                    Severity.INFO,
                    f"Ran {passes} optimization pass(es) at level {level.name}; pruned {pruned} dead node(s).",
                    "optimizer",
                )

            state.statistics.optimization_passes = passes
            state.statistics.pruned_node_count = pruned

        self._timed(BuildPhase.OPTIMIZED, _run)

    @staticmethod
    def _prune_dead_nodes(node: "SceneNode") -> int:
        """Recursively prune leaf nodes with no components/metadata. Returns count pruned."""
        pruned = 0
        for child in list(node.children):
            pruned += WorldSpecBuilder._prune_dead_nodes(child)
            is_dead_leaf = (
                not child.children
                and not child.components
                and not child.metadata
                and not child.visible
                and not child.enabled
            )
            if is_dead_leaf:
                node.children.remove(child)
                pruned += 1
        return pruned

    @staticmethod
    def _flatten_single_child_groups(node: "SceneNode") -> None:
        """Recursively collapse a group node that wraps exactly one child group."""
        for child in list(node.children):
            WorldSpecBuilder._flatten_single_child_groups(child)
        # Wrapper collapsing is deliberately conservative: only fold a
        # childless-metadata group with exactly one child into that
        # child's position, never touching ENTITY/PHYSICS_BODY leaves
        # which downstream backends key off of by name/path.
        NodeType, = WorldSpecBuilder._import_scene_compiler_symbols("NodeType")

        if (
            node.node_type
            in (NodeType.ENTITIES_GROUP, NodeType.MATERIALS_GROUP, NodeType.PHYSICS_GROUP, NodeType.SENSORS_GROUP)
            and len(node.children) == 1
            and not node.metadata
        ):
            # Intentionally left as a structural no-op placeholder for
            # group-level flattening: group semantics (e.g. "Materials")
            # are meaningful path anchors for backends and are preserved
            # by default. Aggressive flattening is opt-in per backend
            # via a future `Exporter`-side pass, not silently applied
            # here where it could surprise every consumer of the IR.
            return

    # ── public API: generate_scene_graph ──────────────────────────────

    def generate_scene_graph(self) -> "SceneGraph":
        """Assemble the final `SceneGraph` from all resolved group nodes.

        Returns:
            The fully assembled, backend-independent `SceneGraph`.

        Raises:
            BuilderLifecycleError: If called before `optimize()`.
        """
        def _run() -> "SceneGraph":
            NodeType, SceneGraph, SceneNode = self._import_scene_compiler_symbols(
                "NodeType", "SceneGraph", "SceneNode"
            )

            state = self._require_state()
            scene_graph = SceneGraph()
            root = scene_graph.root
            root.metadata.update(
                {
                    "scene_id": state.world_spec.scene_id,  # type: ignore[union-attr]
                    "description": state.world_spec.description,  # type: ignore[union-attr]
                }
            )

            # Canonical child ordering, mirroring scene_compiler's layout
            # so the two IR-producing stages remain visually/structurally
            # interchangeable.
            for group_name in ("Environment", "Entities", "Materials", "Physics"):
                group_node = state.group_nodes.get(group_name)
                if group_node is not None:
                    root.add_child(group_node)

            sensors_group = SceneNode(name="Sensors", node_type=NodeType.SENSORS_GROUP)
            root.add_child(sensors_group)

            metadata_node = root.add_child(SceneNode(name="Metadata", node_type=NodeType.METADATA))
            metadata_node.metadata.update(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "builder": "PhysWorldLM.WorldSpecBuilder",
                    "scene_node_count": scene_graph.node_count(),
                    "topological_order": list(state.topo_order),
                    "simulation_graph": state.world_spec.simulation_graph.to_dict(),  # type: ignore[union-attr]
                    # Audit fix (Medium #1): coordinate_system/unit_system
                    # were accepted by WorldSpecBuilderConfig but never
                    # actually threaded anywhere, so they had no effect.
                    # Recorded here (defensively, via getattr, since the
                    # real enum shapes could not be verified against
                    # scene_compiler.py) rather than left inert.
                    "coordinate_system": getattr(self._config.coordinate_system, "name", self._config.coordinate_system),
                    "unit_system": getattr(self._config.unit_system, "name", self._config.unit_system),
                }
            )

            state.scene_graph = scene_graph
            self._log(
                BuildPhase.SCENE_GRAPH_GENERATED,
                Severity.INFO,
                f"SceneGraph generated with {scene_graph.node_count()} total node(s).",
                "scene_graph_generator",
            )
            return scene_graph

        return self._timed(BuildPhase.SCENE_GRAPH_GENERATED, _run)

    # ── public API: statistics / diagnostics / export_report ─────────

    def statistics(self) -> BuildStatistics:
        """Return an immutable snapshot of the current build statistics."""
        with self._lock:
            state = self._require_state()
            return replace(state.statistics, phase_durations_s=dict(state.statistics.phase_durations_s))

    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Return an immutable snapshot of all diagnostics recorded so far."""
        with self._lock:
            state = self._require_state()
            return tuple(state.diagnostics)

    def export_report(self) -> BuildReport:
        """Produce the final `BuildReport` for the current build.

        Safe to call multiple times; each call reflects the latest
        accumulated statistics/diagnostics/scene graph.

        Returns:
            A `BuildReport` summarizing the outcome so far. `status` is
            `FAILED` if any ERROR/CRITICAL diagnostic has been recorded,
            `SUCCESS_WITH_WARNINGS` if only warnings were recorded, and
            `SUCCESS` otherwise. `scene_graph` reflects whatever has been
            generated so far regardless of status: under PERMISSIVE
            policy a build can finish with recorded errors *and* a
            structurally complete graph, and callers may want that
            best-effort graph even when `success` is False. (Audit fix,
            Medium #2: the previous ternary here --
            `state.scene_graph if status is not FAILED else state.scene_graph`
            -- evaluated to the same value on both branches, which read
            as an intentional distinction that didn't actually exist.)
        """
        with self._lock:
            state = self._require_state()
            state.statistics.build_time_s = time.monotonic() - state.started_at
            has_errors = state.statistics.error_count > 0
            state.statistics.success = not has_errors

            if has_errors:
                status = BuildStatus.FAILED
            elif state.statistics.warning_count > 0:
                status = BuildStatus.SUCCESS_WITH_WARNINGS
            else:
                status = BuildStatus.SUCCESS

            return BuildReport(
                status=status,
                scene_id=state.world_spec.scene_id if state.world_spec else "",
                statistics=replace(state.statistics, phase_durations_s=dict(state.statistics.phase_durations_s)),
                diagnostics=list(state.diagnostics),
                entity_order=state.topo_order,
                scene_graph=state.scene_graph,
            )


__all__ = [
    "WorldSpecBuilder",
    "WorldSpecBuilderConfig",
    "BuildReport",
    "BuildStatistics",
    "BuildStatus",
    "BuildPhase",
    "ValidationPolicy",
    "Severity",
    "Diagnostic",
    "WorldSpecBuilderError",
    "BuilderLifecycleError",
    "SceneCompilerIntegrationError",
    "SpecValidationError",
    "DuplicateEntityError",
    "CircularDependencyError",
    "InvalidReferenceError",
    "MissingAssetError",
    "InvalidMaterialError",
    "InvalidTerrainError",
    "InconsistentPhysicsError",
    "SchemaViolationError",
    "HierarchyResolutionError",
    "OptimizationError",
]
