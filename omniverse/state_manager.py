"""
stage_manager.py
══════════════════════════════════════════════════════════════════════════
USD Stage lifecycle manager for the Omniverse Connector layer of
PhysWorldLM.

Pipeline position
------------------
    Natural Language
            │
            ▼
      Prompt Parser
            │
            ▼
    MiniLM Entity Encoder
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
       USD Exporter
            │
            ▼
    ┌────────────────────┐
    │ OmniverseLauncher   │
    └─────────┬───────────┘
              ▼
    ┌────────────────────┐
    │ ExtensionManager    │
    └─────────┬───────────┘
              ▼
    ┌────────────────────┐
    │   StageManager      │  <-- this module
    └─────────┬───────────┘
              ▼
    ┌────────────────────┐
    │   PhysicsScene      │
    └─────────┬───────────┘
              ▼
          Renderer
              │
              ▼
         Simulation

Scope
-----
This module owns exactly one concern: the lifecycle of a USD *Stage* --
creating, opening, closing, saving, backing up, reloading, resetting,
clearing, validating, and repairing it; managing its root/session layer
stack, default prim, stage-level metadata (up axis, units, time codes,
FPS), its canonical root-prim hierarchy (``/World``, ``/World/Physics``,
``/World/Sensors``, ...); and reporting statistics/diagnostics about it.

This module explicitly does NOT:
    * launch Omniverse Kit / Isaac Sim (``app_launcher.OmniverseLauncher``
      owns process lifecycle)
    * manage Kit *extensions* (``extension_manager.ExtensionManager``
      owns enabling/disabling/dependency resolution of extensions)
    * create or configure PhysX / physics simulation (``physics_scene.py``)
    * spawn entities, sensors, robots, or terrain content (those
      components only receive a *root prim path* from this module to
      author underneath)
    * render frames (``renderer.py``)
    * parse natural language, ontologies, or ``WorldSpec`` objects, or
      compile/export USD content (those are upstream compiler stages)

Those concerns belong to ``app_launcher.py``, ``extension_manager.py``,
``physics_scene.py``, ``renderer.py``, ``timeline_controller.py``, and
the domain-specific compiler/exporter stages upstream of Omniverse.
This module is imported by none of them, and imports none of them --
callers hand this manager a live USD-context-like handle (typically
obtained from ``OmniverseLauncher.get_context()``) and consume its
results; it never reaches "up" or "down" the pipeline on its own.

Design constraints
-------------------
    * No ``omni``/``pxr`` import happens at module load time. Every such
      import is deferred to the call site that actually needs it, behind
      :func:`_lazy_import`, so this module loads successfully on a
      machine with no Omniverse installation at all.
    * This module never launches an Omniverse process itself and never
      constructs a USD context. When no live handle is supplied to
      :meth:`StageManager.initialize`, it may *attach* to an
      already-running context (via ``omni.usd.get_context()``) but will
      raise rather than create a Kit process.
    * All failure modes raise a documented, specific
      :class:`StageManagerError` subclass. Nothing lets a raw
      ``ImportError``, ``AttributeError``, or opaque USD stack trace
      escape uncaught.
    * No global mutable state. Every piece of runtime state (the bound
      context handle, the cached stage reference, the current stage
      identifier, dirty/backup bookkeeping, ...) lives on the
      :class:`StageManager` instance, guarded by an internal lock, so
      multiple independent managers (e.g. in tests) never interfere
      with one another.
    * Root-prim creation only ever *defines* prims (``Xform``
      containers) at well-known paths -- it never authors geometry,
      physics schemas, sensors, or any other domain payload underneath
      them. Populating those roots is the job of downstream components.

Public API
----------
    manager = StageManager()
    manager.initialize(context)               # bind to a live USD context
    manager.create_stage()
    manager.create_world_root()
    manager.save_as("scene.usda")
    manager.shutdown()

Or, as a context manager::

    with StageManager() as manager:
        manager.initialize(context)
        manager.create_stage()
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.stage_manager")
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

class StageManagerError(Exception):
    """Base class for all errors raised by :class:`StageManager`."""


class NotInitializedError(StageManagerError):
    """Raised when an operation requires :meth:`StageManager.initialize`
    to have been called successfully first."""


class AlreadyInitializedError(StageManagerError):
    """Raised when :meth:`StageManager.initialize` is called while
    already bound to a live handle."""


class InvalidHandleError(StageManagerError):
    """Raised when the object passed to :meth:`StageManager.initialize`
    does not expose a usable USD-context interface."""


class KitImportError(StageManagerError):
    """Raised when a required ``omni.*``/``pxr.*`` module can't be imported.

    Distinct from a bare ``ImportError`` so callers can catch exactly
    "USD/Omniverse isn't available here" without accidentally
    swallowing an unrelated import bug in their own code.
    """


class StageNotOpenError(StageManagerError):
    """Raised when an operation requires an open stage but none is bound.

    Distinct from :class:`NotInitializedError`: the manager may be
    perfectly initialized (bound to a live context) while no stage has
    been created/opened on that context yet.
    """


class StageCreationError(StageManagerError):
    """Raised when creating a new stage fails."""


class StageOpenError(StageManagerError):
    """Raised when opening an existing stage fails."""


class StageCloseError(StageManagerError):
    """Raised when closing the current stage fails."""


class StageSaveError(StageManagerError):
    """Raised when saving the current stage fails."""


class StageBackupError(StageManagerError):
    """Raised when creating a stage backup fails."""


class StageReloadError(StageManagerError):
    """Raised when reloading the current stage from disk fails."""


class StageResetError(StageManagerError):
    """Raised when resetting the current stage fails."""


class StageClearError(StageManagerError):
    """Raised when clearing all prims from the current stage fails."""


class StageValidationError(StageManagerError):
    """Raised when validation itself cannot be performed (not when the
    stage is merely found invalid -- that is reported via
    :class:`StageValidationReport` instead)."""


class StageRepairError(StageManagerError):
    """Raised when an attempted stage repair fails."""


class StageMetadataError(StageManagerError):
    """Raised when setting stage-level metadata (units, up axis, time
    codes, FPS) fails."""


class PrimCreationError(StageManagerError):
    """Raised when defining a root prim fails."""


class DefaultPrimError(StageManagerError):
    """Raised when getting/setting the default prim fails."""


class LayerError(StageManagerError):
    """Base class for layer-related failures."""


class LayerImportError(LayerError):
    """Raised when importing (sublayering) an external layer fails."""


class LayerExportError(LayerError):
    """Raised when exporting a layer to disk fails."""


class UndoRedoError(StageManagerError):
    """Raised when an undo/redo operation fails."""


class StageDiffError(StageManagerError):
    """Raised when diffing the current stage against another fails."""


class StageReportError(StageManagerError):
    """Raised when exporting a combined stage report fails."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class ManagerState(str, Enum):
    """Lifecycle state of the :class:`StageManager` instance itself.

    Transitions (happy path)::

        UNINITIALIZED -> INITIALIZING -> READY -> SHUTDOWN

    ``ERROR`` is reachable from ``INITIALIZING`` and is terminal until
    :meth:`StageManager.shutdown` resets the manager.
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class UpAxis(str, Enum):
    """Stage up-axis convention.

    USD itself defaults to ``Y``; PhysWorldLM defaults to ``Z``
    (:data:`_DEFAULT_UP_AXIS`) to match the robotics/physics-simulation
    convention used elsewhere in the pipeline (``PhysicsScene``,
    terrain, sensors). Either is a fully valid USD stage.
    """

    Y = "Y"
    Z = "Z"


class RootPrimKind(str, Enum):
    """Canonical top-level content categories PhysWorldLM organizes a
    stage's prim hierarchy around.

    Each kind maps to a well-known prim path (see
    :func:`_default_root_prim_paths`) that downstream components
    (``PhysicsScene``, ``Renderer``, sensor simulation, the ROS2
    bridge, a future ``Planner``) author their own content underneath.
    This manager only ever *defines* the container prim itself -- it
    never populates it.
    """

    WORLD = "world"
    ENVIRONMENT = "environment"
    TERRAIN = "terrain"
    LIGHTING = "lighting"
    PHYSICS = "physics"
    SENSOR = "sensor"
    ROBOT = "robot"
    MISSION = "mission"
    ASSET = "asset"
    TIMELINE = "timeline"
    MATERIALS = "materials"
    NAVIGATION = "navigation"
    SEMANTIC = "semantic"


class ValidationSeverity(str, Enum):
    """Severity of a single :class:`ValidationIssue`."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════

#: Prim path of the top-level world container every other root prim is
#: nested underneath.
_DEFAULT_WORLD_PATH = "/World"

#: PhysWorldLM's default up-axis convention (see :class:`UpAxis`).
_DEFAULT_UP_AXIS = UpAxis.Z

#: Default stage scale: 1 stage unit == 1 meter.
_DEFAULT_METERS_PER_UNIT = 1.0

#: Default stage playback rate.
_DEFAULT_FPS = 24.0

#: Default simulation/animation time-code range.
_DEFAULT_START_TIME_CODE = 0.0
_DEFAULT_END_TIME_CODE = 100.0

#: Directory name (relative to a stage's own directory) backups are
#: written to when no explicit destination is given to :meth:`backup`.
_BACKUP_DIRECTORY_NAME = "_backups"

#: Timestamp format embedded in auto-generated backup filenames.
_BACKUP_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

#: File suffix used for anonymous/in-memory stages that have never been
#: saved, purely for display/report purposes.
_ANONYMOUS_STAGE_LABEL = "<anonymous>"


def _default_root_prim_paths(world_path: str) -> dict[RootPrimKind, str]:
    """Build the canonical ``RootPrimKind`` -> prim-path mapping.

    Every non-world root is nested under ``world_path`` so a stage's
    entire authored hierarchy is reachable from a single default prim,
    which is a strong (though not required) USD convention for
    composed, referenceable stages.
    """
    return {
        RootPrimKind.WORLD: world_path,
        RootPrimKind.ENVIRONMENT: f"{world_path}/Environment",
        RootPrimKind.TERRAIN: f"{world_path}/Environment/Terrain",
        RootPrimKind.LIGHTING: f"{world_path}/Environment/Lighting",
        RootPrimKind.PHYSICS: f"{world_path}/Physics",
        RootPrimKind.SENSOR: f"{world_path}/Sensors",
        RootPrimKind.ROBOT: f"{world_path}/Robots",
        RootPrimKind.MISSION: f"{world_path}/Mission",
        RootPrimKind.ASSET: f"{world_path}/Assets",
        RootPrimKind.TIMELINE: f"{world_path}/Timeline",
        RootPrimKind.MATERIALS: f"{world_path}/Materials",
        RootPrimKind.NAVIGATION: f"{world_path}/Navigation",
        RootPrimKind.SEMANTIC: f"{world_path}/Semantics",
    }


# ════════════════════════════════════════════════════════════════════════
# Lazy import helper
# ════════════════════════════════════════════════════════════════════════

def _lazy_import(module_name: str, *, hint: str = "") -> Any:
    """Import ``module_name``, raising :class:`KitImportError` on failure.

    Every ``omni``/``pxr`` import used by this module goes through this
    function so that (a) importing ``stage_manager`` itself never
    requires Omniverse/USD to be installed, and (b) a missing
    dependency surfaces as one clear, catchable exception instead of a
    raw ``ImportError``.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = f"Failed to import '{module_name}'."
        if hint:
            message = f"{message} {hint}"
        raise KitImportError(message) from exc


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LayerInfo:
    """Descriptive snapshot of a single USD layer in the stage's layer stack.

    Attributes:
        identifier: The layer's identifier (path or anonymous tag).
        real_path: Resolved filesystem path, or ``""`` for an anonymous
            (in-memory) layer.
        is_anonymous: Whether this layer has never been assigned a
            real, on-disk identifier.
        is_dirty: Whether this layer has unsaved edits.
        sublayer_paths: Sublayer identifiers this layer directly composes,
            in strength order (strongest first).
        role: ``"root"``, ``"session"``, or ``"sublayer"``.
    """

    identifier: str
    real_path: str
    is_anonymous: bool
    is_dirty: bool
    sublayer_paths: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class ValidationIssue:
    """A single finding from :meth:`StageManager.validate`.

    Attributes:
        severity: How serious the finding is.
        message: Human-readable description.
        prim_path: The prim the issue concerns, if any.
    """

    severity: ValidationSeverity
    message: str
    prim_path: Optional[str] = None


@dataclass
class StageValidationReport:
    """Aggregate result of :meth:`StageManager.validate` / :meth:`repair`.

    Attributes:
        issues: All findings, in the order they were discovered.
        generated_at: Wall-clock timestamp (``time.time()``) this
            report was produced.
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """True if no issue at :attr:`ValidationSeverity.ERROR` was found."""
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this report."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "generated_at": self.generated_at,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "prim_path": issue.prim_path,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class StageStatistics:
    """Snapshot of structural statistics about the current stage.

    Attributes:
        current_identifier: The stage's current on-disk identifier, or
            ``None`` for an anonymous/unsaved stage.
        total_prim_count: Total number of prims reachable via traversal.
        active_prim_count: Number of those prims that are active.
        defined_prim_count: Number of those prims with a defining
            (non-``over``) specifier.
        prim_type_counts: Mapping of USD type name (e.g. ``"Xform"``,
            ``""`` for typeless) to occurrence count.
        layer_count: Total number of layers in the layer stack
            (root + session + all sublayers, recursively).
        has_default_prim: Whether the stage currently has a default prim set.
        default_prim_path: Path of the default prim, if any.
        up_axis: The stage's up-axis, if determinable.
        meters_per_unit: The stage's linear scale, if determinable.
        frames_per_second: The stage's playback rate, if determinable.
        start_time_code: The stage's start time code, if determinable.
        end_time_code: The stage's end time code, if determinable.
        is_dirty: Whether this manager considers the stage to have
            unsaved changes.
    """

    current_identifier: Optional[str]
    total_prim_count: int
    active_prim_count: int
    defined_prim_count: int
    prim_type_counts: dict[str, int]
    layer_count: int
    has_default_prim: bool
    default_prim_path: Optional[str]
    up_axis: Optional[str]
    meters_per_unit: Optional[float]
    frames_per_second: Optional[float]
    start_time_code: Optional[float]
    end_time_code: Optional[float]
    is_dirty: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of these statistics."""
        return {
            "current_identifier": self.current_identifier or _ANONYMOUS_STAGE_LABEL,
            "total_prim_count": self.total_prim_count,
            "active_prim_count": self.active_prim_count,
            "defined_prim_count": self.defined_prim_count,
            "prim_type_counts": dict(self.prim_type_counts),
            "layer_count": self.layer_count,
            "has_default_prim": self.has_default_prim,
            "default_prim_path": self.default_prim_path,
            "up_axis": self.up_axis,
            "meters_per_unit": self.meters_per_unit,
            "frames_per_second": self.frames_per_second,
            "start_time_code": self.start_time_code,
            "end_time_code": self.end_time_code,
            "is_dirty": self.is_dirty,
        }


@dataclass(frozen=True)
class BackupRecord:
    """Record of a single backup created by :meth:`StageManager.backup`.

    Attributes:
        path: Destination path the backup was written to.
        source_identifier: The stage identifier the backup was taken from.
        created_at: Wall-clock timestamp (``time.time()``) of creation.
    """

    path: Path
    source_identifier: Optional[str]
    created_at: float


# ════════════════════════════════════════════════════════════════════════
# StageManager
# ════════════════════════════════════════════════════════════════════════

class StageManager:
    """Manages the lifecycle of a USD Stage for PhysWorldLM.

    ``StageManager`` sits between :class:`extension_manager.ExtensionManager`
    (which ensures the extensions a stage's content depends on are
    enabled) and every downstream component that needs a live, valid
    USD stage to author into (``PhysicsScene``, ``Renderer``,
    ``TimelineController``, sensor simulation, the ROS2 bridge, a
    future ``Planner``). It contains no domain logic of its own: it
    does not decide what a physics scene, sensor, or robot *is*, only
    where each one's content should live in the prim hierarchy.

    A manager instance never creates a USD context. Callers obtain a
    live context handle from ``OmniverseLauncher.get_context()`` (or an
    equivalent object exposing ``get_stage()`` / ``new_stage()`` /
    ``open_stage()`` / ``save_stage()``, i.e. the ``omni.usd.UsdContext``
    interface) and pass it to :meth:`initialize`.

    Thread-safety: all state-mutating operations are guarded by an
    internal lock, so this manager is safe to call from multiple
    threads (e.g. autosave running alongside interactive edits).

    Example:
        >>> manager = StageManager()
        >>> manager.initialize(context)                # context from OmniverseLauncher.get_context()
        >>> manager.create_stage()
        >>> manager.create_physics_root()
        >>> manager.save_as("scenes/demo.usda")
        >>> manager.shutdown()
    """

    def __init__(
        self,
        *,
        world_prim_path: str = _DEFAULT_WORLD_PATH,
        root_prim_paths: Optional[dict[RootPrimKind, str]] = None,
    ) -> None:
        """Create a manager. Does not touch USD/Omniverse yet.

        Args:
            world_prim_path: Prim path used as the top-level world
                container (default ``"/World"``). Every other root
                prim is nested underneath this path unless
                ``root_prim_paths`` overrides it explicitly.
            root_prim_paths: Optional override/extension of the
                built-in :class:`RootPrimKind` -> prim-path mapping.
                When ``None``, the default mapping rooted at
                ``world_prim_path`` is used.
        """
        self._lock = threading.RLock()
        self._state: ManagerState = ManagerState.UNINITIALIZED
        self._context: Optional[Any] = None
        self._stage: Optional[Any] = None
        self._current_identifier: Optional[str] = None
        self._dirty: bool = False
        self._backups: list[BackupRecord] = []
        self._last_error: Optional[BaseException] = None
        self._world_prim_path = world_prim_path
        self._root_prim_paths: dict[RootPrimKind, str] = dict(
            root_prim_paths if root_prim_paths is not None else _default_root_prim_paths(world_prim_path)
        )

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> ManagerState:
        """Current lifecycle state of this manager."""
        with self._lock:
            return self._state

    def is_initialized(self) -> bool:
        """Whether this manager is bound to a live USD-context handle."""
        return self.state is ManagerState.READY

    @property
    def is_dirty(self) -> bool:
        """Whether this manager considers the current stage to have
        unsaved changes, per its own bookkeeping.

        This is a best-effort local flag (set on every mutating
        operation this manager performs and cleared on
        save/save_as/open/reload) rather than a query of USD's own
        layer-dirty bits, so it stays accurate even for anonymous
        stages that have no layer to query.
        """
        with self._lock:
            return self._dirty

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that most recently drove this manager into ``ERROR``, if any."""
        with self._lock:
            return self._last_error

    @property
    def current_identifier(self) -> Optional[str]:
        """Identifier (path/URL) of the currently open stage, or ``None``
        for a new/anonymous stage that has never been saved."""
        with self._lock:
            return self._current_identifier

    @property
    def backups(self) -> list[BackupRecord]:
        """All backups created by this manager instance, oldest first."""
        with self._lock:
            return list(self._backups)

    def _mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def _mark_clean(self) -> None:
        with self._lock:
            self._dirty = False

    # ------------------------------------------------------------------
    # Lifecycle: initialize / shutdown
    # ------------------------------------------------------------------

    def initialize(
        self,
        context: Optional[Any] = None,
        *,
        refresh_stage: bool = True,
        attach_to_running_context: bool = False,
    ) -> None:
        """Bind this manager to a live USD-context interface.

        This method never creates a USD context. It either:
            1. Binds directly to ``context`` if it already looks like a
               USD-context interface (has ``get_stage``/``new_stage``), or
            2. Calls ``context.get_context()`` if it looks like an
               ``OmniverseLauncher``-style object instead, or
            3. If ``context`` is ``None`` and ``attach_to_running_context``
               is True, *attaches* to an already-active USD context via
               ``omni.usd.get_context()`` -- this still does not create a
               new context; it fails with :class:`InvalidHandleError` if
               none exists in this interpreter.

        Args:
            context: A USD-context-like object, an object exposing
                ``get_context()``, or ``None``.
            refresh_stage: If True (default), immediately attempt to
                cache whatever stage (if any) is already open on the
                bound context.
            attach_to_running_context: See above. Defaults to False so
                a caller that forgets to pass a handle gets a clear
                error rather than this manager silently reaching for
                global USD state.

        Raises:
            AlreadyInitializedError: If already bound to a handle.
            InvalidHandleError: If no usable USD-context interface can
                be obtained from the arguments given.
        """
        with self._lock:
            if self._state is ManagerState.READY:
                raise AlreadyInitializedError(
                    "StageManager is already initialized; call shutdown() first."
                )
            self._state = ManagerState.INITIALIZING
            self._last_error = None

        logger.info("Initializing StageManager.")
        try:
            handle = self._resolve_handle(context, attach_to_running_context)
            with self._lock:
                self._context = handle
                self._state = ManagerState.READY

            if refresh_stage:
                self._refresh_stage_handle(raise_on_error=False)

        except StageManagerError as exc:
            with self._lock:
                self._state = ManagerState.ERROR
                self._last_error = exc
            logger.error("StageManager initialization failed: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - never leak an opaque USD traceback
            wrapped = InvalidHandleError(f"Unexpected failure during initialize(): {exc}")
            with self._lock:
                self._state = ManagerState.ERROR
                self._last_error = wrapped
            logger.error("StageManager initialization failed: %s", wrapped)
            raise wrapped from exc

        logger.info("StageManager ready (stage_open=%s).", self._stage is not None)

    def _resolve_handle(self, candidate: Optional[Any], attach_to_running_context: bool) -> Any:
        """Resolve ``candidate`` (or, optionally, the active context) to a
        USD-context interface exposing ``get_stage``/``new_stage``."""
        if candidate is None:
            if not attach_to_running_context:
                raise InvalidHandleError(
                    "initialize() requires a USD context handle (e.g. from "
                    "OmniverseLauncher.get_context()), or attach_to_running_context=True."
                )
            omni_usd_module = _lazy_import(
                "omni.usd",
                hint="No active USD context appears to be available in this interpreter.",
            )
            try:
                candidate = omni_usd_module.get_context()
            except Exception as exc:  # noqa: BLE001
                raise InvalidHandleError(f"Failed to attach to an active USD context: {exc}") from exc
            if candidate is None:
                raise InvalidHandleError("omni.usd.get_context() returned None.")

        if hasattr(candidate, "get_stage") and hasattr(candidate, "new_stage"):
            return candidate

        get_context = getattr(candidate, "get_context", None)
        if callable(get_context):
            try:
                context = get_context()
            except Exception as exc:  # noqa: BLE001
                raise InvalidHandleError(
                    f"'{type(candidate).__name__}.get_context()' failed: {exc}"
                ) from exc
            if context is None:
                raise InvalidHandleError(f"'{type(candidate).__name__}.get_context()' returned None.")
            return context

        raise InvalidHandleError(
            f"Object of type '{type(candidate).__name__}' does not look like a USD "
            "context (no 'get_stage'/'new_stage' methods) or a launcher exposing "
            "'get_context()'."
        )

    def shutdown(self) -> None:
        """Release the bound handle and reset the manager to a fresh state.

        Idempotent: calling ``shutdown()`` when not initialized logs and
        returns rather than raising. This never closes the underlying
        USD context or stage (it does not own the context) -- only this
        manager's own binding and cache. Callers that want the stage
        itself closed first should call :meth:`close_stage` explicitly.
        """
        with self._lock:
            if self._state in (ManagerState.UNINITIALIZED, ManagerState.SHUTDOWN):
                logger.info("shutdown() called with nothing initialized; nothing to do.")
                return
            self._context = None
            self._stage = None
            self._current_identifier = None
            self._dirty = False
            self._state = ManagerState.SHUTDOWN
        logger.info("StageManager shut down.")

    def __enter__(self) -> "StageManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.shutdown()

    def _require_ready(self) -> None:
        if self._state is not ManagerState.READY:
            raise NotInitializedError(
                f"StageManager is not initialized (state='{self._state.value}'). "
                "Call initialize() with a live USD context handle first."
            )

    def _require_stage(self) -> Any:
        self._require_ready()
        with self._lock:
            stage = self._stage
        if stage is None:
            raise StageNotOpenError(
                "No stage is currently open on this manager. Call create_stage() "
                "or open_stage() first."
            )
        return stage

    # ------------------------------------------------------------------
    # Stage handle refresh
    # ------------------------------------------------------------------

    def _refresh_stage_handle(self, *, raise_on_error: bool) -> Optional[Any]:
        """Re-query the bound context for its current stage and cache it."""
        context = self._context
        if context is None:
            if raise_on_error:
                raise NotInitializedError("No context bound; cannot refresh stage handle.")
            return None
        try:
            stage = context.get_stage()
        except Exception as exc:  # noqa: BLE001
            wrapped = StageManagerError(f"Failed to query the current stage from the bound context: {exc}")
            if raise_on_error:
                raise wrapped from exc
            logger.warning("%s (continuing with a stale/empty stage handle).", wrapped)
            return None
        with self._lock:
            self._stage = stage
        return stage

    # ------------------------------------------------------------------
    # Creation / opening / closing
    # ------------------------------------------------------------------

    def new_stage(self) -> Any:
        """Create a brand-new, blank, anonymous (in-memory) stage.

        Discards any previously cached identifier -- the resulting
        stage has not been saved anywhere yet. For a stage that also
        has PhysWorldLM's default hierarchy/metadata applied, use
        :meth:`create_stage` instead.

        Returns:
            The newly created stage handle.

        Raises:
            NotInitializedError: If not bound to a live context.
            StageCreationError: If the underlying context call fails.
        """
        self._require_ready()
        new_stage_fn = getattr(self._context, "new_stage", None)
        if not callable(new_stage_fn):
            raise StageCreationError("Bound context has no 'new_stage' method.")

        logger.info("Creating a new, anonymous stage.")
        try:
            new_stage_fn()
        except Exception as exc:  # noqa: BLE001
            raise StageCreationError(f"Failed to create a new stage: {exc}") from exc

        stage = self._refresh_stage_handle(raise_on_error=True)
        if stage is None:
            raise StageCreationError("new_stage() succeeded but no stage could be retrieved afterward.")

        with self._lock:
            self._current_identifier = None
        self._mark_dirty()
        logger.info("New anonymous stage created.")
        return stage

    def create_temporary_stage(self) -> Any:
        """Create a new anonymous, in-memory stage never intended for
        direct :meth:`save`.

        Functionally identical to :meth:`new_stage`; kept as a distinct,
        explicit entry point for call sites (e.g. offline validation,
        scratch composition) where "temporary" better conveys intent
        than "new". Use :meth:`save_as` if the result should ultimately
        be persisted.
        """
        return self.new_stage()

    def create_stage(self, *, apply_defaults: bool = True) -> Any:
        """Create a new stage and apply PhysWorldLM's default conventions.

        This is the high-level entry point most callers should use: it
        creates a blank stage via :meth:`new_stage` and then, if
        ``apply_defaults`` is True, defines the ``/World`` root prim,
        sets it as the default prim, and applies PhysWorldLM's default
        up-axis, unit scale, FPS, and time-code range.

        Args:
            apply_defaults: If True (default), apply the default
                hierarchy/metadata described above. If False, an
                entirely blank stage is returned (equivalent to
                :meth:`new_stage`).

        Returns:
            The newly created stage handle.

        Raises:
            NotInitializedError: If not bound to a live context.
            StageCreationError: If stage creation itself fails.
        """
        stage = self.new_stage()
        if apply_defaults:
            world_prim = self.create_world_root()
            self.set_default_prim(self._world_prim_path)
            self.set_up_axis(_DEFAULT_UP_AXIS)
            self.set_stage_units(_DEFAULT_METERS_PER_UNIT)
            self.set_fps(_DEFAULT_FPS)
            self.set_timecodes(_DEFAULT_START_TIME_CODE, _DEFAULT_END_TIME_CODE)
            logger.info("Applied default conventions to new stage (default_prim=%s).", world_prim)
        return stage

    def open_stage(self, path: "str | Path") -> Any:
        """Open an existing stage from ``path``.

        Args:
            path: Path or URL identifying the stage to open.

        Returns:
            The opened stage handle.

        Raises:
            NotInitializedError: If not bound to a live context.
            StageOpenError: If the stage cannot be opened.
        """
        self._require_ready()
        identifier = str(path)
        open_stage_fn = getattr(self._context, "open_stage", None)
        if not callable(open_stage_fn):
            raise StageOpenError("Bound context has no 'open_stage' method.")

        logger.info("Opening stage '%s'.", identifier)
        try:
            result = open_stage_fn(identifier)
        except Exception as exc:  # noqa: BLE001
            raise StageOpenError(f"Failed to open stage '{identifier}': {exc}") from exc
        if result is False:
            raise StageOpenError(f"Context reported failure opening stage '{identifier}'.")

        stage = self._refresh_stage_handle(raise_on_error=True)
        if stage is None:
            raise StageOpenError(f"open_stage() succeeded but no stage could be retrieved for '{identifier}'.")

        with self._lock:
            self._current_identifier = identifier
        self._mark_clean()
        logger.info("Stage '%s' opened.", identifier)
        return stage

    def close_stage(self, *, save: bool = False) -> None:
        """Close the currently open stage.

        Args:
            save: If True, :meth:`save` is called before closing.

        Raises:
            NotInitializedError: If not bound to a live context.
            StageCloseError: If the underlying context call fails.
        """
        self._require_ready()
        if save:
            self.save()

        close_stage_fn = getattr(self._context, "close_stage", None)
        logger.info("Closing current stage ('%s').", self._current_identifier or _ANONYMOUS_STAGE_LABEL)
        if callable(close_stage_fn):
            try:
                close_stage_fn()
            except Exception as exc:  # noqa: BLE001
                raise StageCloseError(f"Failed to close the current stage: {exc}") from exc

        with self._lock:
            self._stage = None
            self._current_identifier = None
            self._dirty = False
        logger.info("Stage closed.")

    # ------------------------------------------------------------------
    # Save / backup / reload
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save the current stage to its existing identifier.

        Raises:
            StageNotOpenError: If no stage is open.
            StageSaveError: If the stage has no identifier yet (never
                saved), or the underlying save call fails.
        """
        self._require_stage()
        if self._current_identifier is None:
            raise StageSaveError(
                "The current stage has no on-disk identifier yet; call save_as() first."
            )

        save_stage_fn = getattr(self._context, "save_stage", None)
        logger.info("Saving stage '%s'.", self._current_identifier)
        try:
            if callable(save_stage_fn):
                save_stage_fn()
            else:
                self._stage.GetRootLayer().Save()
        except Exception as exc:  # noqa: BLE001
            raise StageSaveError(f"Failed to save stage '{self._current_identifier}': {exc}") from exc

        self._mark_clean()
        logger.info("Stage saved.")

    def save_as(self, path: "str | Path") -> None:
        """Save the current stage to a new identifier, and adopt it.

        Args:
            path: Destination path or URL.

        Raises:
            StageNotOpenError: If no stage is open.
            StageSaveError: If the underlying save call fails.
        """
        stage = self._require_stage()
        identifier = str(path)

        save_as_stage_fn = getattr(self._context, "save_as_stage", None)
        logger.info("Saving stage as '%s'.", identifier)
        try:
            if callable(save_as_stage_fn):
                save_as_stage_fn(identifier)
            else:
                stage.GetRootLayer().Export(identifier)
        except Exception as exc:  # noqa: BLE001
            raise StageSaveError(f"Failed to save stage as '{identifier}': {exc}") from exc

        with self._lock:
            self._current_identifier = identifier
        self._mark_clean()
        logger.info("Stage saved as '%s'.", identifier)

    def backup(self, path: Optional["str | Path"] = None) -> Path:
        """Create a point-in-time backup (checkpoint) of the current stage.

        Args:
            path: Explicit destination path. When omitted, a
                timestamped filename is generated next to the current
                stage's identifier, under a ``_backups/`` subdirectory.

        Returns:
            The path the backup was written to.

        Raises:
            StageNotOpenError: If no stage is open.
            StageBackupError: If no destination can be determined (no
                explicit ``path`` and the stage is anonymous), or the
                backup itself fails.
        """
        stage = self._require_stage()

        if path is not None:
            destination = Path(path)
        else:
            if self._current_identifier is None:
                raise StageBackupError(
                    "Cannot auto-generate a backup path for an anonymous stage; "
                    "pass an explicit 'path'."
                )
            source_path = Path(self._current_identifier)
            timestamp = datetime.now().strftime(_BACKUP_TIMESTAMP_FORMAT)
            backup_name = f"{source_path.stem}.{timestamp}{source_path.suffix or '.usda'}"
            destination = source_path.parent / _BACKUP_DIRECTORY_NAME / backup_name

        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Creating stage backup at '%s'.", destination)
        try:
            root_layer = stage.GetRootLayer()
            exported = False
            export_fn = getattr(root_layer, "Export", None)
            if callable(export_fn):
                export_fn(str(destination))
                exported = True
            elif self._current_identifier is not None and Path(self._current_identifier).exists():
                shutil.copy2(self._current_identifier, destination)
                exported = True
            if not exported:
                raise StageBackupError(
                    "Neither the root layer nor the current identifier could be used "
                    "to produce a backup."
                )
        except StageBackupError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StageBackupError(f"Failed to create backup at '{destination}': {exc}") from exc

        record = BackupRecord(
            path=destination, source_identifier=self._current_identifier, created_at=time.time()
        )
        with self._lock:
            self._backups.append(record)
        logger.info("Backup created at '%s'.", destination)
        return destination

    def reload(self) -> Any:
        """Discard in-memory changes and reload the stage from its identifier.

        Returns:
            The reloaded stage handle.

        Raises:
            StageNotOpenError: If no stage is open.
            StageReloadError: If the stage has no on-disk identifier, or
                the underlying reload fails.
        """
        self._require_stage()
        if self._current_identifier is None:
            raise StageReloadError("Cannot reload an anonymous stage that was never saved/opened from disk.")

        logger.info("Reloading stage '%s'.", self._current_identifier)
        reload_fn = getattr(self._context, "reload_stage", None)
        try:
            if callable(reload_fn):
                reload_fn()
                stage = self._refresh_stage_handle(raise_on_error=True)
            else:
                identifier = self._current_identifier
                self.close_stage(save=False)
                stage = self.open_stage(identifier)
                return stage
        except StageManagerError as exc:
            raise StageReloadError(f"Failed to reload stage: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise StageReloadError(f"Failed to reload stage: {exc}") from exc

        self._mark_clean()
        logger.info("Stage reloaded.")
        return stage

    # ------------------------------------------------------------------
    # Reset / clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove every prim from the current stage, keeping the stage
        itself (its identifier, metadata, and layer stack) intact.

        Distinct from :meth:`reset`: ``clear()`` empties the prim
        hierarchy but leaves up-axis/units/FPS/time codes and the
        current identifier untouched.

        Raises:
            StageNotOpenError: If no stage is open.
            StageClearError: If prim removal fails.
        """
        stage = self._require_stage()
        logger.info("Clearing all prims from the current stage.")
        try:
            pseudo_root = stage.GetPseudoRoot()
            for child in list(pseudo_root.GetChildren()):
                stage.RemovePrim(child.GetPath())
        except Exception as exc:  # noqa: BLE001
            raise StageClearError(f"Failed to clear stage prims: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage cleared.")

    def reset(self, *, apply_defaults: bool = True) -> None:
        """Reset the current stage to PhysWorldLM's default blank state.

        Distinct from :meth:`clear`: ``reset()`` also reapplies the
        default root hierarchy and stage metadata (equivalent to what
        :meth:`create_stage` applies to a brand-new stage), while
        keeping the same on-disk identifier so a subsequent
        :meth:`save` overwrites it in place.

        Args:
            apply_defaults: If True (default), reapply the default
                ``/World`` root, default prim, up axis, units, FPS, and
                time-code range after clearing.

        Raises:
            StageNotOpenError: If no stage is open.
            StageResetError: If clearing or reapplying defaults fails.
        """
        self._require_stage()
        logger.info("Resetting current stage to default state.")
        try:
            self.clear()
            if apply_defaults:
                self.create_world_root()
                self.set_default_prim(self._world_prim_path)
                self.set_up_axis(_DEFAULT_UP_AXIS)
                self.set_stage_units(_DEFAULT_METERS_PER_UNIT)
                self.set_fps(_DEFAULT_FPS)
                self.set_timecodes(_DEFAULT_START_TIME_CODE, _DEFAULT_END_TIME_CODE)
        except StageManagerError as exc:
            raise StageResetError(f"Failed to reset stage: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage reset complete.")

    # ------------------------------------------------------------------
    # Layer accessors
    # ------------------------------------------------------------------

    def get_stage(self) -> Any:
        """Return the currently bound stage handle.

        Raises:
            StageNotOpenError: If no stage is open.
        """
        return self._require_stage()

    def get_root_layer(self) -> Any:
        """Return the current stage's root layer.

        Raises:
            StageNotOpenError: If no stage is open.
        """
        stage = self._require_stage()
        return stage.GetRootLayer()

    def get_session_layer(self) -> Any:
        """Return the current stage's session layer.

        Raises:
            StageNotOpenError: If no stage is open.
        """
        stage = self._require_stage()
        return stage.GetSessionLayer()

    def get_default_prim(self) -> Any:
        """Return the current stage's default prim.

        Raises:
            StageNotOpenError: If no stage is open.
            DefaultPrimError: If no default prim is set.
        """
        stage = self._require_stage()
        try:
            prim = stage.GetDefaultPrim()
        except Exception as exc:  # noqa: BLE001
            raise DefaultPrimError(f"Failed to query the default prim: {exc}") from exc
        if prim is None or not getattr(prim, "IsValid", lambda: True)():
            raise DefaultPrimError("Stage has no default prim set.")
        return prim

    def set_default_prim(self, prim_path: str) -> None:
        """Set the stage's default prim to the prim at ``prim_path``.

        Args:
            prim_path: Path of an already-defined prim.

        Raises:
            StageNotOpenError: If no stage is open.
            DefaultPrimError: If no prim exists at ``prim_path``, or the
                underlying call fails.
        """
        stage = self._require_stage()
        try:
            prim = stage.GetPrimAtPath(prim_path)
        except Exception as exc:  # noqa: BLE001
            raise DefaultPrimError(f"Failed to look up prim at '{prim_path}': {exc}") from exc
        if prim is None or not getattr(prim, "IsValid", lambda: False)():
            raise DefaultPrimError(f"Cannot set default prim: no valid prim exists at '{prim_path}'.")

        try:
            stage.SetDefaultPrim(prim)
        except Exception as exc:  # noqa: BLE001
            raise DefaultPrimError(f"Failed to set default prim to '{prim_path}': {exc}") from exc

        self._mark_dirty()
        logger.info("Default prim set to '%s'.", prim_path)

    def lock_root_layer(self) -> None:
        """Mark the root layer as not permitted to be edited.

        A defensive measure against accidental authoring into a stage
        meant to be treated as read-only once handed off to downstream
        consumers (e.g. after being exported for distribution).

        Raises:
            StageNotOpenError: If no stage is open.
            LayerError: If the underlying call fails or is unsupported.
        """
        root_layer = self.get_root_layer()
        set_permission_fn = getattr(root_layer, "SetPermissionToEdit", None)
        if not callable(set_permission_fn):
            raise LayerError("Root layer does not support 'SetPermissionToEdit'.")
        try:
            set_permission_fn(False)
        except Exception as exc:  # noqa: BLE001
            raise LayerError(f"Failed to lock root layer: {exc}") from exc
        logger.info("Root layer locked against edits.")

    def unlock_root_layer(self) -> None:
        """Reverse :meth:`lock_root_layer`.

        Raises:
            StageNotOpenError: If no stage is open.
            LayerError: If the underlying call fails or is unsupported.
        """
        root_layer = self.get_root_layer()
        set_permission_fn = getattr(root_layer, "SetPermissionToEdit", None)
        if not callable(set_permission_fn):
            raise LayerError("Root layer does not support 'SetPermissionToEdit'.")
        try:
            set_permission_fn(True)
        except Exception as exc:  # noqa: BLE001
            raise LayerError(f"Failed to unlock root layer: {exc}") from exc
        logger.info("Root layer unlocked for edits.")

    # ------------------------------------------------------------------
    # Root prim creation
    # ------------------------------------------------------------------

    def _create_root_prim(self, kind: RootPrimKind, *, type_name: str = "Xform") -> Any:
        """Define the container prim for ``kind`` at its configured path.

        USD's ``DefinePrim`` implicitly creates any missing ancestor
        prims along the way, but every non-world root additionally
        ensures ``/World`` itself is explicitly defined first (rather
        than left as an untyped ancestor) for a clean, predictable
        hierarchy.

        Raises:
            StageNotOpenError: If no stage is open.
            PrimCreationError: If no path is configured for ``kind``, or
                the underlying define call fails.
        """
        stage = self._require_stage()
        path = self._root_prim_paths.get(kind)
        if not path:
            raise PrimCreationError(f"No configured prim path for root kind '{kind.value}'.")

        try:
            prim = stage.DefinePrim(path, type_name)
        except Exception as exc:  # noqa: BLE001
            raise PrimCreationError(f"Failed to define '{kind.value}' root prim at '{path}': {exc}") from exc
        if prim is None or not getattr(prim, "IsValid", lambda: False)():
            raise PrimCreationError(f"DefinePrim('{path}') did not return a valid prim.")

        self._mark_dirty()
        logger.info("Defined '%s' root prim at '%s'.", kind.value, path)
        return prim

    def create_world_root(self) -> Any:
        """Define the top-level ``/World`` (or configured equivalent) prim."""
        return self._create_root_prim(RootPrimKind.WORLD)

    def create_environment_root(self) -> Any:
        """Define the environment root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.ENVIRONMENT)

    def create_physics_root(self) -> Any:
        """Define the physics root prim (nested under the world root).

        ``PhysicsScene`` authors its scene/collision/rigid-body schemas
        underneath this prim; this manager only reserves the location.
        """
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.PHYSICS)

    def create_sensor_root(self) -> Any:
        """Define the sensor root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.SENSOR)

    def create_material_root(self) -> Any:
        """Define the materials root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.MATERIALS)

    def create_navigation_root(self) -> Any:
        """Define the navigation root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.NAVIGATION)

    def create_robot_root(self) -> Any:
        """Define the robot root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.ROBOT)

    def create_timeline_root(self) -> Any:
        """Define the timeline root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.TIMELINE)

    def create_asset_root(self) -> Any:
        """Define the asset root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.ASSET)

    def create_terrain_root(self) -> Any:
        """Define the terrain root prim (nested under the environment root)."""
        self.create_environment_root()
        return self._create_root_prim(RootPrimKind.TERRAIN)

    def create_lighting_root(self) -> Any:
        """Define the lighting root prim (nested under the environment root)."""
        self.create_environment_root()
        return self._create_root_prim(RootPrimKind.LIGHTING)

    def create_mission_root(self) -> Any:
        """Define the mission root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.MISSION)

    def create_semantic_root(self) -> Any:
        """Define the semantics root prim (nested under the world root)."""
        self.create_world_root()
        return self._create_root_prim(RootPrimKind.SEMANTIC)

    def root_prim_path(self, kind: RootPrimKind) -> str:
        """Return the configured prim path for ``kind`` without defining it."""
        path = self._root_prim_paths.get(kind)
        if not path:
            raise PrimCreationError(f"No configured prim path for root kind '{kind.value}'.")
        return path

    # ------------------------------------------------------------------
    # Stage-level metadata
    # ------------------------------------------------------------------

    def set_stage_units(self, meters_per_unit: float) -> None:
        """Set the stage's linear unit scale.

        Args:
            meters_per_unit: Must be strictly positive (e.g. ``1.0`` for
                meters, ``0.01`` for centimeters).

        Raises:
            StageNotOpenError: If no stage is open.
            StageMetadataError: If ``meters_per_unit`` is not positive,
                or the underlying call fails.
        """
        if meters_per_unit <= 0:
            raise StageMetadataError(f"meters_per_unit must be > 0 (got {meters_per_unit}).")
        stage = self._require_stage()
        try:
            usd_geom = _lazy_import("pxr.UsdGeom", hint="Required to set stage linear units.")
            usd_geom.SetStageMetersPerUnit(stage, meters_per_unit)
        except KitImportError:
            try:
                stage.SetMetadata("metersPerUnit", meters_per_unit)
            except Exception as exc:  # noqa: BLE001
                raise StageMetadataError(f"Failed to set stage units: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise StageMetadataError(f"Failed to set stage units: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage meters-per-unit set to %.6f.", meters_per_unit)

    def set_up_axis(self, axis: UpAxis) -> None:
        """Set the stage's up-axis convention.

        Args:
            axis: Either :attr:`UpAxis.Y` or :attr:`UpAxis.Z`.

        Raises:
            StageNotOpenError: If no stage is open.
            StageMetadataError: If the underlying call fails.
        """
        stage = self._require_stage()
        try:
            usd_geom = _lazy_import("pxr.UsdGeom", hint="Required to set the stage up-axis.")
            usd_geom.SetStageUpAxis(stage, axis.value)
        except KitImportError:
            try:
                stage.SetMetadata("upAxis", axis.value)
            except Exception as exc:  # noqa: BLE001
                raise StageMetadataError(f"Failed to set up-axis: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise StageMetadataError(f"Failed to set up-axis: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage up-axis set to '%s'.", axis.value)

    def set_timecodes(self, start: float, end: float) -> None:
        """Set the stage's start/end time-code range.

        Args:
            start: Start time code.
            end: End time code; must be ``>= start``.

        Raises:
            StageNotOpenError: If no stage is open.
            StageMetadataError: If ``end < start``, or the underlying
                call fails.
        """
        if end < start:
            raise StageMetadataError(f"end time code ({end}) must be >= start time code ({start}).")
        stage = self._require_stage()
        try:
            stage.SetStartTimeCode(start)
            stage.SetEndTimeCode(end)
        except Exception as exc:  # noqa: BLE001
            raise StageMetadataError(f"Failed to set time-code range: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage time-code range set to [%.2f, %.2f].", start, end)

    def set_fps(self, fps: float) -> None:
        """Set the stage's playback rate (frames per second).

        Args:
            fps: Must be strictly positive.

        Raises:
            StageNotOpenError: If no stage is open.
            StageMetadataError: If ``fps`` is not positive, or the
                underlying call fails.
        """
        if fps <= 0:
            raise StageMetadataError(f"fps must be > 0 (got {fps}).")
        stage = self._require_stage()
        try:
            stage.SetFramesPerSecond(fps)
            set_tcps = getattr(stage, "SetTimeCodesPerSecond", None)
            if callable(set_tcps):
                set_tcps(fps)
        except Exception as exc:  # noqa: BLE001
            raise StageMetadataError(f"Failed to set FPS: {exc}") from exc

        self._mark_dirty()
        logger.info("Stage FPS set to %.3f.", fps)

    # ------------------------------------------------------------------
    # Layer stack introspection
    # ------------------------------------------------------------------

    def _layer_info(self, layer: Any, role: str) -> LayerInfo:
        identifier = str(getattr(layer, "identifier", "") or "")
        real_path = str(getattr(layer, "realPath", "") or "")
        is_anonymous_fn = getattr(layer, "IsAnonymous", None)
        is_anonymous = bool(is_anonymous_fn()) if callable(is_anonymous_fn) else not real_path
        is_dirty_fn = getattr(layer, "IsDirty", None)
        is_dirty = bool(is_dirty_fn()) if callable(is_dirty_fn) else False
        sublayer_paths = tuple(getattr(layer, "subLayerPaths", ()) or ())
        return LayerInfo(
            identifier=identifier,
            real_path=real_path,
            is_anonymous=is_anonymous,
            is_dirty=is_dirty,
            sublayer_paths=sublayer_paths,
            role=role,
        )

    def list_layers(self) -> list[LayerInfo]:
        """Return descriptive info for every layer in the stage's layer stack.

        Includes the root layer, session layer, and every sublayer
        reachable transitively from either, each visited at most once.

        Raises:
            StageNotOpenError: If no stage is open.
        """
        stage = self._require_stage()
        root_layer = stage.GetRootLayer()
        session_layer = stage.GetSessionLayer()

        results: list[LayerInfo] = []
        seen: set[str] = set()

        def _visit(layer: Any, role: str) -> None:
            if layer is None:
                return
            identifier = str(getattr(layer, "identifier", "") or id(layer))
            if identifier in seen:
                return
            seen.add(identifier)
            results.append(self._layer_info(layer, role))

            find_layer_fn = None
            try:
                sdf_module = _lazy_import("pxr.Sdf")
                find_layer_fn = sdf_module.Layer.Find
            except KitImportError:
                pass

            for sublayer_path in getattr(layer, "subLayerPaths", ()) or ():
                sublayer = None
                if callable(find_layer_fn):
                    try:
                        sublayer = find_layer_fn(sublayer_path)
                    except Exception:  # noqa: BLE001 - best effort only
                        sublayer = None
                if sublayer is not None:
                    _visit(sublayer, "sublayer")

        _visit(root_layer, "root")
        _visit(session_layer, "session")
        return results

    def list_prims(self, root_path: Optional[str] = None) -> list[str]:
        """Return the paths of every prim reachable from ``root_path``.

        Args:
            root_path: Prim path to start traversal from. Defaults to
                the stage's pseudo-root (i.e. every prim in the stage).

        Raises:
            StageNotOpenError: If no stage is open.
            StageManagerError: If ``root_path`` does not reference a
                valid prim.
        """
        stage = self._require_stage()
        if root_path is None:
            start_prim = stage.GetPseudoRoot()
        else:
            start_prim = stage.GetPrimAtPath(root_path)
            if start_prim is None or not getattr(start_prim, "IsValid", lambda: False)():
                raise StageManagerError(f"No valid prim exists at '{root_path}'.")

        paths: list[str] = []

        def _walk(prim: Any) -> None:
            for child in prim.GetChildren():
                paths.append(str(child.GetPath()))
                _walk(child)

        _walk(start_prim)
        return paths

    # ------------------------------------------------------------------
    # Layer import / export
    # ------------------------------------------------------------------

    def import_layer(self, path: "str | Path", *, strongest: bool = True) -> None:
        """Add an external layer as a sublayer of the current root layer.

        Args:
            path: Path or URL of the layer to sublayer in.
            strongest: If True (default), inserted at the front of the
                root layer's sublayer list (highest strength). If
                False, appended (lowest strength).

        Raises:
            StageNotOpenError: If no stage is open.
            LayerImportError: If the underlying call fails.
        """
        root_layer = self.get_root_layer()
        identifier = str(path)
        try:
            sublayer_paths = root_layer.subLayerPaths
            index = 0 if strongest else len(sublayer_paths)
            sublayer_paths.insert(index, identifier)
        except Exception as exc:  # noqa: BLE001
            raise LayerImportError(f"Failed to import layer '{identifier}': {exc}") from exc

        self._mark_dirty()
        logger.info("Imported layer '%s' as a %s sublayer.", identifier, "strongest" if strongest else "weakest")

    def export_layer(self, path: "str | Path", *, session: bool = False) -> Path:
        """Export the root (or session) layer to ``path`` without
        changing the current stage's identifier.

        Args:
            path: Destination path.
            session: If True, export the session layer instead of the
                root layer.

        Returns:
            The destination path.

        Raises:
            StageNotOpenError: If no stage is open.
            LayerExportError: If the underlying export call fails.
        """
        layer = self.get_session_layer() if session else self.get_root_layer()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            layer.Export(str(destination))
        except Exception as exc:  # noqa: BLE001
            raise LayerExportError(f"Failed to export layer to '{destination}': {exc}") from exc

        logger.info("Exported %s layer to '%s'.", "session" if session else "root", destination)
        return destination

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the most recent authoring command, via Kit's undo stack.

        Raises:
            NotInitializedError: If not bound to a live context.
            UndoRedoError: If no undo stack is available, or the
                underlying call fails.
        """
        self._require_ready()
        try:
            undo_module = _lazy_import(
                "omni.kit.undo", hint="Undo support requires the 'omni.kit.undo' extension."
            )
            undo_module.undo()
        except KitImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UndoRedoError(f"Undo failed: {exc}") from exc
        logger.info("Undo executed.")

    def redo(self) -> None:
        """Redo the most recently undone authoring command.

        Raises:
            NotInitializedError: If not bound to a live context.
            UndoRedoError: If no undo stack is available, or the
                underlying call fails.
        """
        self._require_ready()
        try:
            undo_module = _lazy_import(
                "omni.kit.undo", hint="Redo support requires the 'omni.kit.undo' extension."
            )
            undo_module.redo()
        except KitImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UndoRedoError(f"Redo failed: {exc}") from exc
        logger.info("Redo executed.")

    # ------------------------------------------------------------------
    # Validation / repair
    # ------------------------------------------------------------------

    def validate(self) -> StageValidationReport:
        """Run compliance/consistency checks against the current stage.

        Uses ``pxr.UsdUtils.ComplianceChecker`` when available for
        USD-compliance diagnostics, and always additionally runs a
        small set of PhysWorldLM-specific structural checks (default
        prim present, up axis is one of the supported values, root
        layer reachable).

        Returns:
            A :class:`StageValidationReport` describing every finding.
            Note this method itself only raises if validation *cannot
            be attempted at all* -- an invalid stage is a normal,
            non-raising result reported via the returned report.

        Raises:
            StageNotOpenError: If no stage is open.
            StageValidationError: If validation cannot be attempted.
        """
        stage = self._require_stage()
        issues: list[ValidationIssue] = []

        try:
            usd_utils = _lazy_import("pxr.UsdUtils")
            checker_cls = getattr(usd_utils, "ComplianceChecker", None)
            if checker_cls is not None:
                checker = checker_cls()
                checker.CheckCompliance(stage)
                for error in checker.GetErrors():
                    issues.append(ValidationIssue(severity=ValidationSeverity.ERROR, message=str(error)))
                for warning in checker.GetFailedChecks():
                    issues.append(ValidationIssue(severity=ValidationSeverity.WARNING, message=str(warning)))
        except KitImportError:
            logger.debug("pxr.UsdUtils unavailable; skipping USD-compliance checks.")
        except Exception as exc:  # noqa: BLE001
            raise StageValidationError(f"USD-compliance validation failed to run: {exc}") from exc

        try:
            default_prim = stage.GetDefaultPrim()
            if default_prim is None or not getattr(default_prim, "IsValid", lambda: False)():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Stage has no default prim set.",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise StageValidationError(f"Failed to inspect the default prim: {exc}") from exc

        try:
            up_axis = self._read_up_axis(stage)
            if up_axis is not None and up_axis not in {member.value for member in UpAxis}:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Stage up-axis '{up_axis}' is not one of {[m.value for m in UpAxis]}.",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise StageValidationError(f"Failed to inspect the up-axis: {exc}") from exc

        try:
            root_layer = stage.GetRootLayer()
            if root_layer is None:
                issues.append(
                    ValidationIssue(severity=ValidationSeverity.ERROR, message="Stage has no root layer.")
                )
        except Exception as exc:  # noqa: BLE001
            raise StageValidationError(f"Failed to inspect the root layer: {exc}") from exc

        report = StageValidationReport(issues=issues)
        logger.info(
            "Validation complete (errors=%d, warnings=%d).", report.error_count, report.warning_count
        )
        return report

    def repair(self) -> StageValidationReport:
        """Attempt to automatically fix repairable issues, then re-validate.

        Currently repairs:
            * Missing default prim -- set to the configured world root
              prim, if one is already defined.
            * Missing/invalid up-axis metadata -- set to PhysWorldLM's
                default (:data:`_DEFAULT_UP_AXIS`).

        Issues that cannot be safely auto-repaired (e.g. deep USD
        compliance errors) are left for the caller to address and will
        still appear in the returned report.

        Returns:
            A fresh :class:`StageValidationReport`, generated after any
            repairs were attempted.

        Raises:
            StageNotOpenError: If no stage is open.
            StageRepairError: If a repair attempt itself fails.
        """
        stage = self._require_stage()
        logger.info("Attempting automatic stage repair.")

        try:
            default_prim = stage.GetDefaultPrim()
            has_default_prim = default_prim is not None and getattr(default_prim, "IsValid", lambda: False)()
        except Exception as exc:  # noqa: BLE001
            raise StageRepairError(f"Failed to inspect the default prim during repair: {exc}") from exc

        if not has_default_prim:
            world_path = self._root_prim_paths.get(RootPrimKind.WORLD, self._world_prim_path)
            try:
                world_prim = stage.GetPrimAtPath(world_path)
                if world_prim is not None and getattr(world_prim, "IsValid", lambda: False)():
                    self.set_default_prim(world_path)
                    logger.info("Repaired missing default prim (set to '%s').", world_path)
            except StageManagerError as exc:
                raise StageRepairError(f"Failed to repair default prim: {exc}") from exc

        try:
            up_axis = self._read_up_axis(stage)
            if up_axis not in {member.value for member in UpAxis}:
                self.set_up_axis(_DEFAULT_UP_AXIS)
                logger.info("Repaired missing/invalid up-axis (set to '%s').", _DEFAULT_UP_AXIS.value)
        except StageManagerError as exc:
            raise StageRepairError(f"Failed to repair up-axis: {exc}") from exc

        return self.validate()

    def _read_up_axis(self, stage: Any) -> Optional[str]:
        try:
            usd_geom = _lazy_import("pxr.UsdGeom")
            return usd_geom.GetStageUpAxis(stage)
        except KitImportError:
            get_metadata_fn = getattr(stage, "GetMetadata", None)
            if callable(get_metadata_fn):
                try:
                    return get_metadata_fn("upAxis")
                except Exception:  # noqa: BLE001
                    return None
            return None

    # ------------------------------------------------------------------
    # Statistics / reporting / diff
    # ------------------------------------------------------------------

    def export_statistics(self) -> StageStatistics:
        """Compute and return structural statistics about the current stage.

        Raises:
            StageNotOpenError: If no stage is open.
            StageManagerError: If traversal itself fails.
        """
        stage = self._require_stage()

        total_prim_count = 0
        active_prim_count = 0
        defined_prim_count = 0
        prim_type_counts: dict[str, int] = {}

        try:
            for prim in stage.Traverse():
                total_prim_count += 1
                if prim.IsActive():
                    active_prim_count += 1
                if prim.IsDefined():
                    defined_prim_count += 1
                type_name = str(prim.GetTypeName()) or "<untyped>"
                prim_type_counts[type_name] = prim_type_counts.get(type_name, 0) + 1
        except Exception as exc:  # noqa: BLE001
            raise StageManagerError(f"Failed to traverse stage for statistics: {exc}") from exc

        try:
            default_prim = stage.GetDefaultPrim()
            has_default_prim = default_prim is not None and getattr(default_prim, "IsValid", lambda: False)()
            default_prim_path = str(default_prim.GetPath()) if has_default_prim else None
        except Exception:  # noqa: BLE001
            has_default_prim = False
            default_prim_path = None

        layer_count = len(self.list_layers())
        up_axis = self._read_up_axis(stage)

        try:
            meters_per_unit = float(stage.GetMetadata("metersPerUnit"))
        except Exception:  # noqa: BLE001
            meters_per_unit = None
        try:
            fps = float(stage.GetFramesPerSecond())
        except Exception:  # noqa: BLE001
            fps = None
        try:
            start_time_code = float(stage.GetStartTimeCode())
        except Exception:  # noqa: BLE001
            start_time_code = None
        try:
            end_time_code = float(stage.GetEndTimeCode())
        except Exception:  # noqa: BLE001
            end_time_code = None

        statistics = StageStatistics(
            current_identifier=self._current_identifier,
            total_prim_count=total_prim_count,
            active_prim_count=active_prim_count,
            defined_prim_count=defined_prim_count,
            prim_type_counts=prim_type_counts,
            layer_count=layer_count,
            has_default_prim=has_default_prim,
            default_prim_path=default_prim_path,
            up_axis=up_axis,
            meters_per_unit=meters_per_unit,
            frames_per_second=fps,
            start_time_code=start_time_code,
            end_time_code=end_time_code,
            is_dirty=self.is_dirty,
        )
        logger.info("Statistics computed (%d prim(s), %d layer(s)).", total_prim_count, layer_count)
        return statistics

    def export_stage_report(self, path: "str | Path") -> Path:
        """Write a combined validation + statistics + layer-stack report to disk.

        Args:
            path: Destination JSON file path.

        Returns:
            The destination path.

        Raises:
            StageNotOpenError: If no stage is open.
            StageReportError: If the report cannot be written.
        """
        validation = self.validate()
        statistics = self.export_statistics()
        layers = self.list_layers()

        report = {
            "generated_at": time.time(),
            "current_identifier": self._current_identifier or _ANONYMOUS_STAGE_LABEL,
            "validation": validation.to_dict(),
            "statistics": statistics.to_dict(),
            "layers": [
                {
                    "identifier": layer.identifier or _ANONYMOUS_STAGE_LABEL,
                    "real_path": layer.real_path,
                    "is_anonymous": layer.is_anonymous,
                    "is_dirty": layer.is_dirty,
                    "role": layer.role,
                    "sublayer_paths": list(layer.sublayer_paths),
                }
                for layer in layers
            ],
        }

        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            raise StageReportError(f"Failed to write stage report to '{destination}': {exc}") from exc

        logger.info("Stage report written to '%s'.", destination)
        return destination

    def stage_diff(self, other_path: "str | Path") -> dict[str, Any]:
        """Compare the current stage's prim set against another stage on disk.

        A lightweight, read-only structural diff -- it opens
        ``other_path`` independently (never touching this manager's
        bound context) and compares prim-path sets. It does not diff
        authored attribute values.

        Args:
            other_path: Path to the stage to diff against.

        Returns:
            A dict with ``"added"`` (paths present here but not in
            ``other_path``), ``"removed"`` (paths present in
            ``other_path`` but not here), and ``"common_count"``.

        Raises:
            StageNotOpenError: If no stage is open.
            StageDiffError: If the other stage cannot be opened/read.
        """
        current_paths = set(self.list_prims())

        try:
            usd_module = _lazy_import("pxr.Usd", hint="Required to open a stage for diffing.")
            other_stage = usd_module.Stage.Open(str(other_path))
        except KitImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StageDiffError(f"Failed to open '{other_path}' for diffing: {exc}") from exc

        if other_stage is None:
            raise StageDiffError(f"Could not open '{other_path}' for diffing.")

        try:
            other_paths = {str(prim.GetPath()) for prim in other_stage.Traverse()}
        except Exception as exc:  # noqa: BLE001
            raise StageDiffError(f"Failed to traverse '{other_path}' for diffing: {exc}") from exc

        added = sorted(current_paths - other_paths)
        removed = sorted(other_paths - current_paths)
        common_count = len(current_paths & other_paths)

        logger.info(
            "Stage diff computed against '%s' (added=%d, removed=%d, common=%d).",
            other_path, len(added), len(removed), common_count,
        )
        return {"added": added, "removed": removed, "common_count": common_count}


__all__ = [
    "StageManager",
    "LayerInfo",
    "ValidationIssue",
    "StageValidationReport",
    "StageStatistics",
    "BackupRecord",
    "ManagerState",
    "UpAxis",
    "RootPrimKind",
    "ValidationSeverity",
    "StageManagerError",
    "NotInitializedError",
    "AlreadyInitializedError",
    "InvalidHandleError",
    "KitImportError",
    "StageNotOpenError",
    "StageCreationError",
    "StageOpenError",
    "StageCloseError",
    "StageSaveError",
    "StageBackupError",
    "StageReloadError",
    "StageResetError",
    "StageClearError",
    "StageValidationError",
    "StageRepairError",
    "StageMetadataError",
    "PrimCreationError",
    "DefaultPrimError",
    "LayerError",
    "LayerImportError",
    "LayerExportError",
    "UndoRedoError",
    "StageDiffError",
    "StageReportError",
]
