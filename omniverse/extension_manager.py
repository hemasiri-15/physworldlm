"""
extension_manager.py
══════════════════════════════════════════════════════════════════════════
Omniverse Kit extension lifecycle manager for the Omniverse Connector
layer of PhysWorldLM.

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
    ┌───────────────────┐
    │  OmniverseLauncher │
    └─────────┬──────────┘
              ▼
    ┌───────────────────┐
    │  ExtensionManager  │  <-- this module
    └─────────┬──────────┘
              ▼
    ┌───────────────────┐
    │   StageManager     │
    └─────────┬──────────┘
              ▼
        PhysicsScene
              │
              ▼
          Renderer
              │
              ▼
         Simulation

Scope
-----
This module owns exactly one concern: the lifecycle of Omniverse Kit
*extensions* -- discovering what is installed, enabling/disabling
extensions (individually or in dependency-respecting batches), resolving
and verifying dependency graphs, detecting missing extensions and
version conflicts, loading/exporting extension configuration manifests,
hot-reloading, and reporting extension status. It never boots Kit,
never touches USD, never touches physics, and never renders a frame.

This module explicitly does NOT:
    * launch Omniverse Kit / Isaac Sim (``app_launcher.OmniverseLauncher``
      owns process lifecycle; this module only *attaches to* an
      already-launched application's extension subsystem)
    * parse natural language, ontologies, or ``WorldSpec`` objects
    * compile a Scene Graph or export USD content
    * create, open, or manage a USD stage (``stage_manager.py``)
    * implement or configure PhysX / physics simulation
      (``physics_scene.py``)
    * render frames (``renderer.py``)
    * spawn entities, sensors, or terrain

Those concerns belong to ``app_launcher.py``, ``stage_manager.py``,
``physics_scene.py``, ``renderer.py``, ``timeline_controller.py``, and
the domain-specific compiler/exporter stages upstream of Omniverse.
This module is imported by none of them, and imports none of them --
callers hand this manager a live application/extension-manager handle
(typically obtained from ``OmniverseLauncher.get_app()``) and consume
its results; it never reaches "up" or "down" the pipeline on its own.

Design constraints
-------------------
    * No ``omni``/``pxr`` import happens at module load time. Every such
      import is deferred to the call site that actually needs it, behind
      :func:`_lazy_import`, so this module loads successfully on a
      machine with no Omniverse installation at all.
    * This module never launches an Omniverse process itself. When no
      live handle is supplied to :meth:`ExtensionManager.initialize`,
      it may *attach* to an already-running Kit process (via
      ``omni.kit.app.get_app()``) but will raise rather than start one.
    * All failure modes raise a documented, specific
      :class:`ExtensionManagerError` subclass. Nothing lets a raw
      ``ImportError``, ``AttributeError``, or opaque Kit stack trace
      escape uncaught.
    * No global mutable state. Every piece of runtime state (the bound
      extension-manager handle, the metadata cache, the known-extension
      registry, ...) lives on the :class:`ExtensionManager` instance,
      guarded by an internal lock, so multiple independent managers
      (e.g. in tests) never interfere with one another.
    * Filesystem-based discovery (parsing ``extension.toml`` files) is
      supported so extensions can be inventoried, and manifests
      validated, without any live Kit process at all -- useful for
      unit tests and offline tooling.

Public API
----------
    manager = ExtensionManager()
    manager.initialize(app)                 # bind to a live Kit app
    manager.enable_extension("omni.physx")
    manager.list_extensions()
    manager.export_manifest("manifest.json")
    manager.shutdown()

Or, as a context manager::

    with ExtensionManager() as manager:
        manager.initialize(app)
        manager.enable_multiple(["omni.physx", "omni.replicator.core"])
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.extension_manager")
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

class ExtensionManagerError(Exception):
    """Base class for all errors raised by :class:`ExtensionManager`."""


class NotInitializedError(ExtensionManagerError):
    """Raised when an operation requires :meth:`ExtensionManager.initialize`
    to have been called successfully first."""


class AlreadyInitializedError(ExtensionManagerError):
    """Raised when :meth:`ExtensionManager.initialize` is called while
    already bound to a live handle."""


class InvalidHandleError(ExtensionManagerError):
    """Raised when the object passed to :meth:`ExtensionManager.initialize`
    does not expose a usable extension-manager interface."""


class KitImportError(ExtensionManagerError):
    """Raised when a required ``omni.*`` module can't be imported.

    Distinct from a bare ``ImportError`` so callers can catch exactly
    "Omniverse isn't available here" without accidentally swallowing an
    unrelated import bug in their own code.
    """


class ExtensionDiscoveryError(ExtensionManagerError):
    """Raised when discovering installed/available extensions fails."""


class ExtensionNotFoundError(ExtensionManagerError):
    """Raised when an operation references an extension id this manager
    has no knowledge of (neither discovered nor known-registry)."""


class ExtensionEnableError(ExtensionManagerError):
    """Raised when enabling an extension fails."""


class ExtensionDisableError(ExtensionManagerError):
    """Raised when disabling an extension fails."""


class ExtensionReloadError(ExtensionManagerError):
    """Raised when hot-reloading an extension fails."""


class MissingDependencyError(ExtensionManagerError):
    """Raised when an extension depends on another extension that is
    neither installed/discoverable nor present in the known registry."""


class DependencyCycleError(ExtensionManagerError):
    """Raised when dependency resolution detects a circular dependency."""


class VersionConflictError(ExtensionManagerError):
    """Raised when two extensions declare incompatible version
    constraints on a shared dependency, or an installed version is
    below a declared minimum."""


class ManifestError(ExtensionManagerError):
    """Base class for manifest import/export failures."""


class ManifestExportError(ManifestError):
    """Raised when a manifest cannot be serialized or written to disk."""


class ManifestImportError(ManifestError):
    """Raised when a manifest cannot be read, parsed, or applied."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class ExtensionCategory(str, Enum):
    """Functional grouping of a Kit extension.

    Purely descriptive/bookkeeping -- this manager attaches no behavior
    to a category, it only uses it for filtering
    (:meth:`ExtensionManager.list_extensions`) and reporting.
    """

    RTX = "rtx"
    PHYSX = "physx"
    USD = "usd"
    REPLICATOR = "replicator"
    ROS2 = "ros2"
    NAVIGATION = "navigation"
    NUCLEUS = "nucleus"
    LIVESTREAM = "livestream"
    SYNTHETIC_DATA = "synthetic_data"
    SENSOR_SIMULATION = "sensor_simulation"
    CUSTOM_PHYSWORLDLM = "custom_physworldlm"
    OTHER = "other"


class ExtensionState(str, Enum):
    """Observed lifecycle state of a single extension.

    ``UNKNOWN``   -- referenced (e.g. as a dependency) but never discovered.
    ``DISCOVERED``-- known to be installed/available, not yet enabled.
    ``ENABLED``   -- currently enabled in the live application.
    ``DISABLED``  -- installed/available but explicitly disabled.
    ``MISSING``   -- required (as a dependency, or via the known registry)
                     but not found during discovery.
    ``ERROR``     -- the most recent enable/disable/reload operation on
                     this extension failed.
    """

    UNKNOWN = "unknown"
    DISCOVERED = "discovered"
    ENABLED = "enabled"
    DISABLED = "disabled"
    MISSING = "missing"
    ERROR = "error"


class ManagerState(str, Enum):
    """Lifecycle state of the :class:`ExtensionManager` instance itself.

    Transitions (happy path)::

        UNINITIALIZED -> INITIALIZING -> READY -> SHUTDOWN

    ``ERROR`` is reachable from ``INITIALIZING`` and is terminal until
    :meth:`ExtensionManager.shutdown` resets the manager.
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTDOWN = "shutdown"
    ERROR = "error"


# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════

#: Filename this manager looks for when scanning a directory tree for
#: extension metadata without a live Kit process (offline discovery).
_EXTENSION_TOML_FILENAME = "extension.toml"

#: Maximum directory depth walked below a search root while looking for
#: ``extension.toml`` files, to keep offline discovery bounded on large
#: install trees.
_MAX_DISCOVERY_DEPTH = 6

#: Known-today built-in registry of extension ids PhysWorldLM cares
#: about, grouped by the product categories this module is asked to
#: support. This is *metadata only* (category, human description,
#: declared dependencies, minimum version) -- it is never treated as
#: proof an extension is actually installed. Live/offline discovery is
#: always the source of truth for "is it here"; this registry only
#: fills in descriptive fields discovery can't know and gives
#: dependency-resolution something to reason about before an extension
#: has ever been discovered.
def _default_known_extensions() -> dict[str, "ExtensionDescriptor"]:
    entries = (
        ("omni.rtx.settings.core", ExtensionCategory.RTX, "RTX renderer settings.", ()),
        ("omni.kit.viewport.rtx", ExtensionCategory.RTX, "RTX viewport integration.", ("omni.rtx.settings.core",)),
        ("omni.physx", ExtensionCategory.PHYSX, "PhysX simulation core.", ()),
        ("omni.physx.ui", ExtensionCategory.PHYSX, "PhysX UI/authoring tools.", ("omni.physx",)),
        ("omni.usd", ExtensionCategory.USD, "Core USD stage/context APIs.", ()),
        ("omni.usd.libs", ExtensionCategory.USD, "Bundled OpenUSD libraries.", ()),
        ("omni.replicator.core", ExtensionCategory.REPLICATOR, "Synthetic data replication core.", ("omni.usd",)),
        ("omni.replicator.composer", ExtensionCategory.SYNTHETIC_DATA, "Replicator scenario composition.", ("omni.replicator.core",)),
        ("omni.isaac.ros2_bridge", ExtensionCategory.ROS2, "ROS2 bridge for Isaac Sim.", ("omni.usd",)),
        ("omni.isaac.navigation_mesh", ExtensionCategory.NAVIGATION, "Navigation mesh generation/queries.", ("omni.usd",)),
        ("omni.kit.livestream.native", ExtensionCategory.LIVESTREAM, "Native Kit livestreaming.", ()),
        ("omni.client", ExtensionCategory.NUCLEUS, "Nucleus/Omniverse Client Library connectivity.", ()),
        ("omni.isaac.sensor", ExtensionCategory.SENSOR_SIMULATION, "Isaac Sim sensor simulation (lidar, camera, IMU, ...).", ("omni.usd", "omni.physx")),
        ("physworldlm.core", ExtensionCategory.CUSTOM_PHYSWORLDLM, "PhysWorldLM custom Kit extension core.", ("omni.usd",)),
    )
    return {
        ext_id: ExtensionDescriptor(
            extension_id=ext_id,
            category=category,
            description=description,
            dependencies=tuple(deps),
        )
        for ext_id, category, description, deps in entries
    }


# ════════════════════════════════════════════════════════════════════════
# Lazy import helper
# ════════════════════════════════════════════════════════════════════════

def _lazy_import(module_name: str, *, hint: str = "") -> Any:
    """Import ``module_name``, raising :class:`KitImportError` on failure.

    Every ``omni``/third-party import used by this module goes through
    this function so that (a) importing ``extension_manager`` itself
    never requires Omniverse to be installed, and (b) a missing
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
# Version helpers (small, local -- deliberately not shared with
# ``config.py`` so this module carries no cross-module coupling)
# ════════════════════════════════════════════════════════════════════════

def _parse_version(raw: str) -> Optional[tuple[int, ...]]:
    """Parse a dotted-integer version string, e.g. ``"1.4.2"`` -> ``(1, 4, 2)``.

    Returns ``None`` if ``raw`` has no parsable leading dotted-integer
    prefix (empty string, non-numeric, etc.).
    """
    if not raw:
        return None
    digits = ""
    parts: list[int] = []
    for ch in raw:
        if ch.isdigit():
            digits += ch
        elif ch == "." and digits:
            parts.append(int(digits))
            digits = ""
        elif parts or digits:
            break
    if digits:
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _version_at_least(version: Optional[tuple[int, ...]], minimum: tuple[int, ...]) -> bool:
    """Return True if ``version >= minimum``. Unknown versions never satisfy a minimum."""
    if not version:
        return False
    padded = version + (0,) * max(0, len(minimum) - len(version))
    return padded[: len(minimum)] >= minimum


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExtensionDescriptor:
    """Static, descriptive metadata about an extension known to PhysWorldLM.

    Distinct from :class:`ExtensionMetadata`: a descriptor is *declared*
    knowledge (from the built-in registry, a loaded configuration file,
    or a parsed ``extension.toml``) and may exist for extensions that
    turn out not to be installed at all. Metadata is *observed* fact
    from discovery.

    Attributes:
        extension_id: Canonical Kit extension id (e.g. ``"omni.physx"``).
        category: Functional grouping, for filtering/reporting.
        description: Short human-readable summary.
        dependencies: Extension ids this extension declares as required
            dependencies.
        min_version: Minimum acceptable version, if this PhysWorldLM
            deployment requires one, as a dotted-integer tuple.
        enabled_by_default: Whether PhysWorldLM's default profile wants
            this extension enabled when present. Advisory only -- this
            manager never auto-enables anything on its own.
    """

    extension_id: str
    category: ExtensionCategory = ExtensionCategory.OTHER
    description: str = ""
    dependencies: tuple[str, ...] = ()
    min_version: Optional[tuple[int, ...]] = None
    enabled_by_default: bool = False


@dataclass
class ExtensionMetadata:
    """Observed, runtime metadata about a single extension.

    This is what :meth:`ExtensionManager.list_extensions` returns and
    what the internal cache stores -- a merge of whatever discovery
    (live-handle or filesystem) actually found with the descriptive
    fields from the known registry, when available.

    Attributes:
        extension_id: Canonical Kit extension id.
        version: Installed version, as a dotted-integer tuple, if known.
        state: Current :class:`ExtensionState`.
        category: Functional grouping (from the registry, if known).
        description: Short human-readable summary (from the registry).
        dependencies: Declared dependency extension ids (from discovery
            when available, else the registry).
        path: Filesystem path the extension was discovered at, if any.
        source: ``"live"`` if discovered via a bound Kit application,
            ``"filesystem"`` if discovered via ``extension.toml`` scan,
            ``"registry"`` if only known descriptively (never actually
            discovered).
        last_error: The most recent error message associated with this
            extension (enable/disable/reload failures), if any.
        discovered_at: Wall-clock timestamp (``time.time()``) this
            record was last (re)populated.
    """

    extension_id: str
    version: Optional[tuple[int, ...]] = None
    state: ExtensionState = ExtensionState.UNKNOWN
    category: ExtensionCategory = ExtensionCategory.OTHER
    description: str = ""
    dependencies: tuple[str, ...] = ()
    path: Optional[Path] = None
    source: str = "registry"
    last_error: Optional[str] = None
    discovered_at: float = field(default_factory=time.time)

    @property
    def version_str(self) -> str:
        """Dotted-string rendering of :attr:`version`, or ``"unknown"``."""
        return ".".join(str(part) for part in self.version) if self.version else "unknown"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this record."""
        return {
            "extension_id": self.extension_id,
            "version": self.version_str,
            "state": self.state.value,
            "category": self.category.value,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "last_error": self.last_error,
        }


# ════════════════════════════════════════════════════════════════════════
# ExtensionManager
# ════════════════════════════════════════════════════════════════════════

class ExtensionManager:
    """Manages the lifecycle of Omniverse Kit extensions for PhysWorldLM.

    ``ExtensionManager`` sits between :class:`app_launcher.OmniverseLauncher`
    (which owns the Kit *process*) and every downstream component that
    needs a specific extension enabled (``StageManager``,
    ``PhysicsScene``, ``Renderer``, ``TimelineController``,
    ``Replicator``, the ROS2 bridge, a future ``Planner``). It contains
    no business logic of its own: it does not decide *when* a physics
    scene should exist, only whether ``omni.physx`` is available and
    enabled when asked.

    A manager instance never launches Kit. Callers obtain a live
    application handle from ``OmniverseLauncher.get_app()`` (or an
    equivalent object exposing ``get_extension_manager()``, or the
    ``omni.kit.app`` extension-manager interface directly) and pass it
    to :meth:`initialize`. Without such a handle, the manager can still
    perform filesystem-based discovery and manifest validation, but
    ``enable_extension`` / ``disable_extension`` / ``reload`` require a
    live binding and raise :class:`NotInitializedError` otherwise.

    Thread-safety: all state-mutating operations are guarded by an
    internal lock, so this manager is safe to call from multiple
    threads (e.g. several subsystems enabling extensions concurrently
    during startup).

    Example:
        >>> manager = ExtensionManager()
        >>> manager.initialize(app)                    # app from OmniverseLauncher.get_app()
        >>> manager.enable_extension("omni.physx")
        >>> manager.list_extensions(category=ExtensionCategory.PHYSX)
        >>> manager.shutdown()
    """

    def __init__(
        self,
        *,
        known_extensions: Optional[dict[str, ExtensionDescriptor]] = None,
    ) -> None:
        """Create a manager. Does not touch Omniverse Kit yet.

        Args:
            known_extensions: Optional override/extension of the
                built-in registry of :class:`ExtensionDescriptor`
                entries (keyed by extension id), used to fill in
                descriptive metadata and to seed dependency resolution
                for extensions that have not yet been discovered. When
                ``None``, PhysWorldLM's default registry (covering RTX,
                PhysX, USD, Replicator, ROS2, Navigation, Nucleus,
                Livestream, Synthetic Data, Sensor Simulation, and
                custom PhysWorldLM extensions) is used.
        """
        self._lock = threading.RLock()
        self._state: ManagerState = ManagerState.UNINITIALIZED
        self._handle: Optional[Any] = None
        self._known: dict[str, ExtensionDescriptor] = dict(
            known_extensions if known_extensions is not None else _default_known_extensions()
        )
        self._cache: dict[str, ExtensionMetadata] = {}
        self._cache_populated_at: Optional[float] = None
        self._last_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> ManagerState:
        """Current lifecycle state of this manager."""
        with self._lock:
            return self._state

    def is_initialized(self) -> bool:
        """Whether this manager is bound to a live application handle."""
        return self.state is ManagerState.READY

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that most recently drove this manager into ``ERROR``, if any."""
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------------
    # Lifecycle: initialize / shutdown / reload
    # ------------------------------------------------------------------

    def initialize(
        self,
        app_or_extension_manager: Optional[Any] = None,
        *,
        refresh_cache: bool = True,
        attach_to_running_app: bool = False,
    ) -> None:
        """Bind this manager to a live Kit extension-manager interface.

        This method never launches Omniverse. It either:
            1. Binds directly to ``app_or_extension_manager`` if it
               already looks like an extension-manager interface (has
               ``set_extension_enabled_immediate`` / an equivalent), or
            2. Calls ``app_or_extension_manager.get_extension_manager()``
               if it looks like a Kit ``IApp`` handle instead, or
            3. If ``app_or_extension_manager`` is ``None`` and
               ``attach_to_running_app`` is True, *attaches* to an
               already-running Kit process via ``omni.kit.app.get_app()``
               -- this still does not start a new process; it fails
               with :class:`InvalidHandleError` if no Kit process is
               currently running in this interpreter.

        Args:
            app_or_extension_manager: A Kit ``IApp``-like object, a Kit
                extension-manager interface, or ``None``.
            refresh_cache: If True (default), immediately run discovery
                against the newly bound handle to populate the metadata
                cache.
            attach_to_running_app: See above. Defaults to False so a
                caller that forgets to pass a handle gets a clear error
                rather than this manager silently reaching for global
                Kit state.

        Raises:
            AlreadyInitializedError: If already bound to a handle.
            InvalidHandleError: If no usable extension-manager interface
                can be obtained from the arguments given.
        """
        with self._lock:
            if self._state is ManagerState.READY:
                raise AlreadyInitializedError(
                    "ExtensionManager is already initialized; call shutdown() first."
                )
            self._state = ManagerState.INITIALIZING
            self._last_error = None

        logger.info("Initializing ExtensionManager.")
        try:
            handle = self._resolve_handle(app_or_extension_manager, attach_to_running_app)
            with self._lock:
                self._handle = handle
                self._state = ManagerState.READY

            if refresh_cache:
                self._discover_live(raise_on_error=False)

        except ExtensionManagerError as exc:
            with self._lock:
                self._state = ManagerState.ERROR
                self._last_error = exc
            logger.error("ExtensionManager initialization failed: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - never leak an opaque Kit traceback
            wrapped = InvalidHandleError(f"Unexpected failure during initialize(): {exc}")
            with self._lock:
                self._state = ManagerState.ERROR
                self._last_error = wrapped
            logger.error("ExtensionManager initialization failed: %s", wrapped)
            raise wrapped from exc

        logger.info("ExtensionManager ready (%d extension(s) cached).", len(self._cache))

    def _resolve_handle(
        self, candidate: Optional[Any], attach_to_running_app: bool
    ) -> Any:
        """Resolve ``candidate`` (or, optionally, the running app) to an
        extension-manager interface exposing ``set_extension_enabled_immediate``."""
        if candidate is None:
            if not attach_to_running_app:
                raise InvalidHandleError(
                    "initialize() requires an application or extension-manager handle "
                    "(e.g. from OmniverseLauncher.get_app()), or attach_to_running_app=True."
                )
            kit_app_module = _lazy_import(
                "omni.kit.app",
                hint="No Kit process appears to be running in this interpreter.",
            )
            try:
                candidate = kit_app_module.get_app()
            except Exception as exc:  # noqa: BLE001
                raise InvalidHandleError(
                    f"Failed to attach to a running Kit application: {exc}"
                ) from exc
            if candidate is None:
                raise InvalidHandleError("omni.kit.app.get_app() returned None.")

        if hasattr(candidate, "set_extension_enabled_immediate"):
            return candidate

        get_extension_manager = getattr(candidate, "get_extension_manager", None)
        if callable(get_extension_manager):
            try:
                extension_manager = get_extension_manager()
            except Exception as exc:  # noqa: BLE001
                raise InvalidHandleError(
                    f"'{type(candidate).__name__}.get_extension_manager()' failed: {exc}"
                ) from exc
            if extension_manager is None:
                raise InvalidHandleError(
                    f"'{type(candidate).__name__}.get_extension_manager()' returned None."
                )
            return extension_manager

        raise InvalidHandleError(
            f"Object of type '{type(candidate).__name__}' does not look like a Kit "
            "IApp or extension-manager interface (no 'set_extension_enabled_immediate' "
            "or 'get_extension_manager' method)."
        )

    def shutdown(self) -> None:
        """Release the bound handle and reset the manager to a fresh state.

        Idempotent: calling ``shutdown()`` when not initialized logs and
        returns rather than raising. This never touches the underlying
        Kit process (it does not own it) -- only this manager's own
        binding and cache.
        """
        with self._lock:
            if self._state in (ManagerState.UNINITIALIZED, ManagerState.SHUTDOWN):
                logger.info("shutdown() called with nothing initialized; nothing to do.")
                return
            self._handle = None
            self._cache.clear()
            self._cache_populated_at = None
            self._state = ManagerState.SHUTDOWN
        logger.info("ExtensionManager shut down.")

    def reload(self) -> None:
        """Re-run discovery against the bound handle, refreshing the cache.

        This refreshes this manager's *view* of what's installed and
        enabled -- for hot-reloading a single already-enabled extension's
        code, see :meth:`reload_extension`.

        Raises:
            NotInitializedError: If not currently bound to a live handle.
        """
        self._require_ready()
        logger.info("Reloading extension metadata cache.")
        self._discover_live(raise_on_error=True)
        logger.info("Reload complete (%d extension(s) cached).", len(self._cache))

    def __enter__(self) -> "ExtensionManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.shutdown()

    def _require_ready(self) -> None:
        if self._state is not ManagerState.READY:
            raise NotInitializedError(
                f"ExtensionManager is not initialized (state='{self._state.value}'). "
                "Call initialize() with a live application handle first."
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_live(self, *, raise_on_error: bool) -> None:
        """Populate the cache from the bound live extension-manager handle."""
        handle = self._handle
        if handle is None:
            if raise_on_error:
                raise NotInitializedError("No live handle bound; cannot discover extensions.")
            return

        try:
            summaries = self._fetch_extension_summaries(handle)
        except Exception as exc:  # noqa: BLE001
            wrapped = ExtensionDiscoveryError(f"Live extension discovery failed: {exc}")
            if raise_on_error:
                raise wrapped from exc
            logger.warning("%s (continuing with an empty/stale cache).", wrapped)
            return

        with self._lock:
            for ext_id, version, enabled, dependencies, path in summaries:
                known = self._known.get(ext_id)
                self._cache[ext_id] = ExtensionMetadata(
                    extension_id=ext_id,
                    version=version,
                    state=ExtensionState.ENABLED if enabled else ExtensionState.DISCOVERED,
                    category=known.category if known else ExtensionCategory.OTHER,
                    description=known.description if known else "",
                    dependencies=dependencies or (known.dependencies if known else ()),
                    path=path,
                    source="live",
                )
            self._mark_missing_known_extensions()
            self._cache_populated_at = time.time()

    def _fetch_extension_summaries(
        self, handle: Any
    ) -> list[tuple[str, Optional[tuple[int, ...]], bool, tuple[str, ...], Optional[Path]]]:
        """Translate the live handle's own extension-summary API into a
        uniform list of ``(id, version, enabled, dependencies, path)`` tuples.

        Kit's extension-manager interface exposes
        ``get_extension_summary_list()`` returning objects with an
        ``ext_id`` (``"name-version"``) field and enabled/path
        attributes; this best-effort adapter tolerates minor API
        variation across Kit versions rather than hard-coding one exact
        shape.
        """
        get_summaries = getattr(handle, "get_extension_summary_list", None)
        if not callable(get_summaries):
            raise ExtensionDiscoveryError(
                "Bound handle has no 'get_extension_summary_list' method; "
                "cannot enumerate installed extensions."
            )

        raw_summaries = get_summaries()
        results: list[tuple[str, Optional[tuple[int, ...]], bool, tuple[str, ...], Optional[Path]]] = []
        for summary in raw_summaries or ():
            ext_id_full = getattr(summary, "ext_id", None) or getattr(summary, "id", None)
            if not ext_id_full:
                continue
            ext_id, version = self._split_ext_id(str(ext_id_full))

            enabled = bool(getattr(summary, "enabled", False))
            if not enabled:
                is_enabled_fn = getattr(handle, "is_extension_enabled", None)
                if callable(is_enabled_fn):
                    try:
                        enabled = bool(is_enabled_fn(ext_id))
                    except Exception:  # noqa: BLE001 - best effort only
                        pass

            raw_path = getattr(summary, "path", None) or getattr(summary, "root", None)
            path = Path(raw_path) if raw_path else None

            raw_deps = getattr(summary, "dependencies", None) or ()
            dependencies = tuple(
                self._split_ext_id(str(dep))[0] for dep in raw_deps
            )

            results.append((ext_id, version, enabled, dependencies, path))
        return results

    @staticmethod
    def _split_ext_id(raw: str) -> tuple[str, Optional[tuple[int, ...]]]:
        """Split a Kit ``"name-1.2.3"`` style identifier into ``(name, version)``.

        Extension ids with no embedded version (e.g. plain ``"omni.physx"``)
        return ``(raw, None)`` unchanged.
        """
        if "-" not in raw:
            return raw, None
        name, _, version_part = raw.rpartition("-")
        version = _parse_version(version_part)
        return (name, version) if version is not None else (raw, None)

    def discover_installed_extensions(
        self,
        search_paths: Optional[Iterable["str | Path"]] = None,
        *,
        prefer_live: bool = True,
    ) -> list[ExtensionMetadata]:
        """Discover installed extensions and refresh the cache.

        Args:
            search_paths: Directories to scan for ``extension.toml``
                files (filesystem-based discovery). Ignored when
                ``prefer_live`` is True and a live handle is bound.
            prefer_live: If True (default) and this manager is bound to
                a live handle, discovery uses that handle. Otherwise --
                or if no handle is bound -- filesystem discovery over
                ``search_paths`` is used instead. At least one source
                must be available or :class:`ExtensionDiscoveryError`
                is raised.

        Returns:
            The full, refreshed list of cached :class:`ExtensionMetadata`.

        Raises:
            ExtensionDiscoveryError: If neither a live handle nor usable
                ``search_paths`` are available, or discovery itself fails.
        """
        used_live = False
        with self._lock:
            can_use_live = prefer_live and self._state is ManagerState.READY and self._handle is not None

        if can_use_live:
            self._discover_live(raise_on_error=True)
            used_live = True

        if search_paths is not None:
            self._discover_filesystem(search_paths)
        elif not used_live:
            raise ExtensionDiscoveryError(
                "No live handle bound and no search_paths given; nothing to discover from."
            )

        logger.info(
            "Discovery complete (%d extension(s) cached, live=%s, filesystem_paths=%s).",
            len(self._cache), used_live, bool(search_paths),
        )
        return self.list_extensions()

    def _discover_filesystem(self, search_paths: Iterable["str | Path"]) -> None:
        """Scan ``search_paths`` for ``extension.toml`` files, up to
        :data:`_MAX_DISCOVERY_DEPTH` directories deep, and populate the cache."""
        toml_module = self._load_toml_module()

        found_any = False
        for raw_root in search_paths:
            root = Path(raw_root)
            if not root.exists() or not root.is_dir():
                logger.debug("Filesystem discovery: search path does not exist, skipping: %s", root)
                continue
            for toml_path in self._iter_extension_toml_files(root):
                try:
                    self._ingest_extension_toml(toml_path, toml_module)
                    found_any = True
                except ExtensionDiscoveryError as exc:
                    logger.warning("Skipping unreadable extension descriptor '%s': %s", toml_path, exc)

        if not found_any:
            logger.info("Filesystem discovery found no '%s' files under given search paths.", _EXTENSION_TOML_FILENAME)

        with self._lock:
            self._mark_missing_known_extensions()
            self._cache_populated_at = time.time()

    @staticmethod
    def _load_toml_module() -> Any:
        """Return a TOML-parsing module, preferring the stdlib ``tomllib``
        (Python 3.11+) and falling back to the third-party ``toml`` package."""
        try:
            return importlib.import_module("tomllib")
        except ImportError:
            return _lazy_import(
                "toml",
                hint="Install 'toml' (pip install toml) for Python < 3.11, "
                     "or run under Python 3.11+ for built-in 'tomllib' support.",
            )

    def _iter_extension_toml_files(self, root: Path) -> Iterable[Path]:
        root_depth = len(root.parts)
        for path in root.rglob(_EXTENSION_TOML_FILENAME):
            if len(path.parts) - root_depth > _MAX_DISCOVERY_DEPTH:
                continue
            yield path

    def _ingest_extension_toml(self, toml_path: Path, toml_module: Any) -> None:
        try:
            if hasattr(toml_module, "load") and "b" in getattr(toml_module, "__name__", ""):
                pass  # unreachable; kept for clarity of intent below
            if toml_module.__name__ == "tomllib":
                with toml_path.open("rb") as fh:
                    data = toml_module.load(fh)
            else:
                data = toml_module.loads(toml_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ExtensionDiscoveryError(f"Failed to parse '{toml_path}': {exc}") from exc

        package = data.get("package", {}) if isinstance(data, dict) else {}
        ext_id = package.get("name") or toml_path.parent.name
        version = _parse_version(str(package.get("version", "")))

        deps_section = data.get("dependencies", {}) if isinstance(data, dict) else {}
        dependencies = tuple(deps_section.keys()) if isinstance(deps_section, dict) else ()

        known = self._known.get(ext_id)
        with self._lock:
            self._cache[ext_id] = ExtensionMetadata(
                extension_id=ext_id,
                version=version,
                state=ExtensionState.DISCOVERED,
                category=known.category if known else ExtensionCategory.OTHER,
                description=str(package.get("description", "")) or (known.description if known else ""),
                dependencies=dependencies or (known.dependencies if known else ()),
                path=toml_path.parent,
                source="filesystem",
            )

    def _mark_missing_known_extensions(self) -> None:
        """After a discovery pass, mark any known-registry extension that
        was not actually found as ``MISSING`` (unless already cached)."""
        for ext_id, descriptor in self._known.items():
            if ext_id not in self._cache:
                self._cache[ext_id] = ExtensionMetadata(
                    extension_id=ext_id,
                    state=ExtensionState.MISSING,
                    category=descriptor.category,
                    description=descriptor.description,
                    dependencies=descriptor.dependencies,
                    source="registry",
                )

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_extension(self, extension_id: str, *, resolve_deps: bool = True) -> None:
        """Enable a single extension.

        Args:
            extension_id: The extension to enable.
            resolve_deps: If True (default), dependencies are resolved
                via :meth:`resolve_dependencies` and enabled first, in
                dependency order.

        Raises:
            NotInitializedError: If not bound to a live handle.
            ExtensionNotFoundError: If ``extension_id`` is unknown to
                both discovery and the known registry.
            MissingDependencyError: If ``resolve_deps`` is True and a
                required dependency cannot be found.
            DependencyCycleError: If ``resolve_deps`` is True and a
                circular dependency is detected.
            ExtensionEnableError: If the underlying Kit call fails.
        """
        self._require_ready()
        order = self.resolve_dependencies(extension_id) if resolve_deps else [extension_id]
        for ext_id in order:
            self._set_enabled(ext_id, True)

    def disable_extension(self, extension_id: str) -> None:
        """Disable a single extension.

        Unlike :meth:`enable_extension`, this never cascades to
        dependents -- disabling ``omni.physx`` does not automatically
        disable extensions that depend on it. Kit itself is left to
        enforce (or not) its own runtime dependency invariants; this
        manager only issues the requested toggle and reports the result.

        Raises:
            NotInitializedError: If not bound to a live handle.
            ExtensionNotFoundError: If ``extension_id`` is unknown.
            ExtensionDisableError: If the underlying Kit call fails.
        """
        self._require_ready()
        self._set_enabled(extension_id, False)

    def enable_multiple(
        self,
        extension_ids: Iterable[str],
        *,
        resolve_deps: bool = True,
        continue_on_error: bool = False,
    ) -> dict[str, Optional[str]]:
        """Enable several extensions.

        Args:
            extension_ids: Extension ids to enable.
            resolve_deps: If True (default), each extension's
                dependencies are resolved and enabled first. The
                combined, de-duplicated dependency order across all
                requested extensions is used so a shared dependency is
                only enabled once.
            continue_on_error: If True, a failure enabling one
                extension does not stop the rest -- every requested
                extension is attempted and the per-extension outcome is
                reported. If False (default), the first failure raises
                immediately.

        Returns:
            Mapping of each *requested* extension id (not intermediate
            dependencies) to ``None`` on success or an error message on
            failure.

        Raises:
            NotInitializedError: If not bound to a live handle.
            ExtensionManagerError: The first encountered failure, if
                ``continue_on_error`` is False.
        """
        self._require_ready()
        requested = list(dict.fromkeys(extension_ids))  # de-duplicate, preserve order
        results: dict[str, Optional[str]] = {}

        if resolve_deps:
            try:
                combined_order = self._resolve_combined_order(requested)
            except ExtensionManagerError as exc:
                if continue_on_error:
                    for ext_id in requested:
                        results[ext_id] = str(exc)
                    return results
                raise
            enable_plan = combined_order
        else:
            enable_plan = requested

        succeeded: set[str] = set()
        first_error: Optional[str] = None
        for ext_id in enable_plan:
            try:
                self._set_enabled(ext_id, True)
                succeeded.add(ext_id)
            except ExtensionManagerError as exc:
                message = str(exc)
                if ext_id in requested:
                    results[ext_id] = message
                if first_error is None:
                    first_error = message
                if not continue_on_error:
                    raise

        for ext_id in requested:
            if ext_id not in results:
                results[ext_id] = None if ext_id in succeeded else (first_error or "not attempted")

        return results

    def disable_multiple(
        self,
        extension_ids: Iterable[str],
        *,
        continue_on_error: bool = False,
    ) -> dict[str, Optional[str]]:
        """Disable several extensions independently (no dependency cascade).

        Args:
            extension_ids: Extension ids to disable.
            continue_on_error: If True, failures are collected and every
                extension is still attempted; if False (default), the
                first failure raises immediately.

        Returns:
            Mapping of each extension id to ``None`` on success or an
            error message on failure.

        Raises:
            NotInitializedError: If not bound to a live handle.
            ExtensionManagerError: The first encountered failure, if
                ``continue_on_error`` is False.
        """
        self._require_ready()
        results: dict[str, Optional[str]] = {}
        for ext_id in dict.fromkeys(extension_ids):
            try:
                self._set_enabled(ext_id, False)
                results[ext_id] = None
            except ExtensionManagerError as exc:
                results[ext_id] = str(exc)
                if not continue_on_error:
                    raise
        return results

    def _set_enabled(self, extension_id: str, enabled: bool) -> None:
        """Issue the actual enable/disable call and update the cache."""
        if extension_id not in self._known and extension_id not in self._cache:
            raise ExtensionNotFoundError(
                f"Extension '{extension_id}' is not known to this manager "
                "(not discovered and not in the known registry)."
            )

        handle = self._handle
        set_enabled_fn = getattr(handle, "set_extension_enabled_immediate", None)
        if not callable(set_enabled_fn):
            raise (ExtensionEnableError if enabled else ExtensionDisableError)(
                "Bound handle has no 'set_extension_enabled_immediate' method."
            )

        action = "enable" if enabled else "disable"
        logger.info("Attempting to %s extension '%s'.", action, extension_id)
        try:
            set_enabled_fn(extension_id, enabled)
        except Exception as exc:  # noqa: BLE001
            error_cls = ExtensionEnableError if enabled else ExtensionDisableError
            wrapped = error_cls(f"Failed to {action} extension '{extension_id}': {exc}")
            with self._lock:
                existing = self._cache.get(extension_id)
                if existing is not None:
                    existing.state = ExtensionState.ERROR
                    existing.last_error = str(wrapped)
            logger.error(str(wrapped))
            raise wrapped from exc

        with self._lock:
            existing = self._cache.get(extension_id)
            new_state = ExtensionState.ENABLED if enabled else ExtensionState.DISABLED
            if existing is not None:
                existing.state = new_state
                existing.last_error = None
                existing.discovered_at = time.time()
            else:
                known = self._known.get(extension_id)
                self._cache[extension_id] = ExtensionMetadata(
                    extension_id=extension_id,
                    state=new_state,
                    category=known.category if known else ExtensionCategory.OTHER,
                    description=known.description if known else "",
                    dependencies=known.dependencies if known else (),
                    source="live",
                )
        logger.info("Extension '%s' %sd.", extension_id, action)

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def reload_extension(self, extension_id: str) -> None:
        """Hot-reload a single extension's code without a full Kit restart.

        Uses the bound handle's native reload API when available
        (Kit exposes this on extension-manager interfaces as e.g.
        ``reload_extension`` / ``reload_extensions``); when the handle
        exposes no such API, falls back to a disable-then-enable cycle,
        which is a reasonable approximation for extensions that don't
        require a hard process reload.

        Raises:
            NotInitializedError: If not bound to a live handle.
            ExtensionNotFoundError: If ``extension_id`` is unknown.
            ExtensionReloadError: If the reload (or fallback) fails.
        """
        self._require_ready()
        if extension_id not in self._known and extension_id not in self._cache:
            raise ExtensionNotFoundError(
                f"Extension '{extension_id}' is not known to this manager."
            )

        native_reload = getattr(self._handle, "reload_extension", None) or getattr(
            self._handle, "reload_extensions", None
        )
        logger.info("Hot-reloading extension '%s'.", extension_id)
        try:
            if callable(native_reload):
                native_reload(extension_id)
            else:
                logger.debug(
                    "Bound handle exposes no native reload API for '%s'; "
                    "falling back to disable-then-enable.", extension_id,
                )
                self._set_enabled(extension_id, False)
                self._set_enabled(extension_id, True)
        except ExtensionManagerError:
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = ExtensionReloadError(f"Failed to reload extension '{extension_id}': {exc}")
            with self._lock:
                existing = self._cache.get(extension_id)
                if existing is not None:
                    existing.state = ExtensionState.ERROR
                    existing.last_error = str(wrapped)
            logger.error(str(wrapped))
            raise wrapped from exc

        logger.info("Extension '%s' hot-reloaded.", extension_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_extensions(
        self,
        *,
        category: Optional[ExtensionCategory] = None,
        state: Optional[ExtensionState] = None,
    ) -> list[ExtensionMetadata]:
        """Return cached extension metadata, optionally filtered.

        Does not trigger discovery -- call :meth:`reload` or
        :meth:`discover_installed_extensions` first if the cache may be
        stale. Returned records are deep copies; mutating them does not
        affect this manager's internal state.

        Args:
            category: If given, only extensions in this category.
            state: If given, only extensions in this state.

        Returns:
            A list of :class:`ExtensionMetadata`, sorted by extension id.
        """
        with self._lock:
            records = list(self._cache.values())

        if category is not None:
            records = [r for r in records if r.category is category]
        if state is not None:
            records = [r for r in records if r.state is state]

        records.sort(key=lambda r: r.extension_id)
        return [copy.deepcopy(r) for r in records]

    def extension_exists(self, extension_id: str) -> bool:
        """Whether ``extension_id`` has been discovered or is in the known registry.

        This does *not* mean the extension is enabled -- it means this
        manager has some record of it (see :meth:`extension_enabled` for
        the enabled check). Extensions marked :attr:`ExtensionState.MISSING`
        still return True here (they are *known*, just not found).
        """
        with self._lock:
            return extension_id in self._cache or extension_id in self._known

    def extension_enabled(self, extension_id: str) -> bool:
        """Whether ``extension_id`` is currently enabled, per the cache.

        Raises:
            ExtensionNotFoundError: If ``extension_id`` has never been
                discovered or registered.
        """
        with self._lock:
            record = self._cache.get(extension_id)
        if record is None:
            if extension_id in self._known:
                return False
            raise ExtensionNotFoundError(f"Extension '{extension_id}' is not known to this manager.")
        return record.state is ExtensionState.ENABLED

    # ------------------------------------------------------------------
    # Dependency resolution / verification
    # ------------------------------------------------------------------

    def verify_dependencies(self, extension_id: str) -> list[str]:
        """Return the ids of ``extension_id``'s dependencies that are missing.

        "Missing" means neither discovered nor present in the known
        registry at all (an entry with :attr:`ExtensionState.MISSING`
        counts as missing; a merely-not-yet-enabled but discovered
        entry does not).

        Args:
            extension_id: The extension whose direct dependencies (not
                transitive) should be checked.

        Returns:
            A list of missing dependency extension ids (empty if all
            direct dependencies are satisfiable).

        Raises:
            ExtensionNotFoundError: If ``extension_id`` itself is unknown.
        """
        dependencies = self._dependencies_of(extension_id)
        missing = []
        with self._lock:
            for dep_id in dependencies:
                record = self._cache.get(dep_id)
                is_missing = dep_id not in self._known and (record is None)
                is_missing = is_missing or (record is not None and record.state is ExtensionState.MISSING and dep_id not in self._known)
                if dep_id not in self._known and (record is None or record.state is ExtensionState.MISSING):
                    missing.append(dep_id)
        return missing

    def resolve_dependencies(self, extension_id: str) -> list[str]:
        """Topologically resolve ``extension_id``'s full dependency chain.

        Args:
            extension_id: The extension to resolve (included as the
                final entry of the returned order).

        Returns:
            An ordered list of extension ids ending in ``extension_id``,
            such that enabling them in this order satisfies every
            transitive dependency before the extension that needs it.

        Raises:
            ExtensionNotFoundError: If ``extension_id`` is unknown.
            MissingDependencyError: If a transitive dependency is
                neither discovered nor in the known registry.
            DependencyCycleError: If the dependency graph contains a cycle.
        """
        return self._resolve_combined_order([extension_id])

    def _resolve_combined_order(self, extension_ids: list[str]) -> list[str]:
        """Topologically sort the union of the transitive dependencies of
        every id in ``extension_ids``, via iterative DFS (cycle-safe)."""
        order: list[str] = []
        visited: set[str] = set()
        in_progress: set[str] = set()

        def visit(ext_id: str, chain: tuple[str, ...]) -> None:
            if ext_id in visited:
                return
            if ext_id in in_progress:
                cycle = " -> ".join((*chain, ext_id))
                raise DependencyCycleError(f"Circular extension dependency detected: {cycle}")
            if not self.extension_exists(ext_id):
                raise ExtensionNotFoundError(
                    f"Extension '{ext_id}' is not known to this manager "
                    "(not discovered and not in the known registry)."
                )

            in_progress.add(ext_id)
            for dep_id in self._dependencies_of(ext_id):
                if not self.extension_exists(dep_id):
                    raise MissingDependencyError(
                        f"Extension '{ext_id}' depends on '{dep_id}', which is neither "
                        "discovered nor present in the known registry."
                    )
                visit(dep_id, (*chain, ext_id))
            in_progress.discard(ext_id)
            visited.add(ext_id)
            order.append(ext_id)

        for requested in extension_ids:
            visit(requested, ())
        return order

    def _dependencies_of(self, extension_id: str) -> tuple[str, ...]:
        with self._lock:
            record = self._cache.get(extension_id)
            if record is not None and record.dependencies:
                return record.dependencies
            known = self._known.get(extension_id)
            if known is not None:
                return known.dependencies
            if record is None:
                raise ExtensionNotFoundError(
                    f"Extension '{extension_id}' is not known to this manager."
                )
            return ()

    # ------------------------------------------------------------------
    # Compatibility / version-conflict detection
    # ------------------------------------------------------------------

    def validate_compatibility(self, extension_id: str) -> list[str]:
        """Check a single extension's observed version against known constraints.

        Currently checks the extension's declared :attr:`ExtensionDescriptor.min_version`
        (if this manager's known registry declares one) against the
        version discovery actually observed.

        Returns:
            A list of human-readable issue descriptions (empty if the
            extension is compatible or has no declared constraint).

        Raises:
            ExtensionNotFoundError: If ``extension_id`` is unknown.
        """
        with self._lock:
            record = self._cache.get(extension_id)
            known = self._known.get(extension_id)

        if record is None and known is None:
            raise ExtensionNotFoundError(f"Extension '{extension_id}' is not known to this manager.")

        issues: list[str] = []
        if known is not None and known.min_version is not None:
            observed = record.version if record is not None else None
            if not _version_at_least(observed, known.min_version):
                observed_str = ".".join(map(str, observed)) if observed else "unknown"
                required_str = ".".join(map(str, known.min_version))
                issues.append(
                    f"'{extension_id}' version {observed_str} is below the required "
                    f"minimum {required_str}."
                )
        return issues

    def detect_version_conflicts(self) -> list[str]:
        """Scan the current cache for version-related problems.

        Two kinds of conflicts are detected:
            1. Any cached extension whose observed version is below its
               known-registry :attr:`ExtensionDescriptor.min_version`.
            2. Any extension marked :attr:`ExtensionState.MISSING` that
               at least one discovered, enabled extension declares as a
               dependency (a "phantom" dependency -- present in the
               dependency graph but absent from the environment).

        Returns:
            A list of human-readable conflict descriptions. Empty if
            none are found.
        """
        conflicts: list[str] = []
        with self._lock:
            cache_snapshot = dict(self._cache)

        for ext_id in cache_snapshot:
            conflicts.extend(self.validate_compatibility(ext_id))

        for ext_id, record in cache_snapshot.items():
            if record.state is not ExtensionState.ENABLED:
                continue
            for dep_id in record.dependencies:
                dep_record = cache_snapshot.get(dep_id)
                if dep_record is not None and dep_record.state is ExtensionState.MISSING:
                    conflicts.append(
                        f"'{ext_id}' is enabled but its dependency '{dep_id}' is missing."
                    )

        return conflicts

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def load_extension_config(self, source: "str | Path | dict[str, Any]") -> None:
        """Merge additional :class:`ExtensionDescriptor` entries into the
        known registry from a YAML/JSON file or an in-memory mapping.

        Expected shape (YAML or JSON)::

            extensions:
              omni.my.extension:
                category: custom_physworldlm
                description: "..."
                dependencies: ["omni.usd"]
                min_version: "1.2.0"
                enabled_by_default: false

        Unknown keys within an entry are ignored (forward-compatible);
        unknown top-level keys are ignored with a warning.

        Args:
            source: A path to a ``.yaml``/``.yml``/``.json`` file, or an
                already-parsed mapping with the shape above.

        Raises:
            ManifestImportError: If the source cannot be read/parsed, or
                an entry declares an invalid category.
        """
        if isinstance(source, dict):
            data = source
        else:
            path = Path(source)
            if not path.exists():
                raise ManifestImportError(f"Extension config file not found: '{path}'")
            try:
                if path.suffix.lower() in (".yaml", ".yml"):
                    yaml_module = _lazy_import(
                        "yaml",
                        hint="Install PyYAML ('pip install pyyaml') to load YAML extension configs.",
                    )
                    data = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
                else:
                    data = json.loads(path.read_text(encoding="utf-8"))
            except KitImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ManifestImportError(f"Failed to parse extension config '{path}': {exc}") from exc

        if not isinstance(data, dict):
            raise ManifestImportError("Extension config must be a mapping at the top level.")

        entries = data.get("extensions", {})
        if not isinstance(entries, dict):
            raise ManifestImportError("Extension config 'extensions' key must be a mapping.")

        added = 0
        with self._lock:
            for ext_id, raw in entries.items():
                if not isinstance(raw, dict):
                    logger.warning("Ignoring malformed extension config entry for '%s'.", ext_id)
                    continue
                try:
                    category = ExtensionCategory(raw.get("category", ExtensionCategory.OTHER.value))
                except ValueError:
                    raise ManifestImportError(
                        f"Extension config entry '{ext_id}' has an invalid category: "
                        f"'{raw.get('category')}'."
                    )
                min_version_raw = raw.get("min_version")
                self._known[ext_id] = ExtensionDescriptor(
                    extension_id=ext_id,
                    category=category,
                    description=str(raw.get("description", "")),
                    dependencies=tuple(raw.get("dependencies", ()) or ()),
                    min_version=_parse_version(str(min_version_raw)) if min_version_raw else None,
                    enabled_by_default=bool(raw.get("enabled_by_default", False)),
                )
                added += 1

        logger.info("Loaded %d extension descriptor(s) into the known registry.", added)

    # ------------------------------------------------------------------
    # Manifest export / import
    # ------------------------------------------------------------------

    def export_manifest(self, path: Optional["str | Path"] = None) -> dict[str, Any]:
        """Serialize the current cache to a manifest dict, optionally to disk.

        The manifest records, per extension: id, version, state,
        category, dependencies, and source -- enough to reproduce the
        same enable/disable configuration elsewhere via
        :meth:`import_manifest`.

        Args:
            path: If given, the manifest is also written to this path as
                JSON.

        Returns:
            The manifest as a plain, JSON-serializable dict.

        Raises:
            ManifestExportError: If ``path`` is given but cannot be written.
        """
        with self._lock:
            records = list(self._cache.values())

        manifest = {
            "physworldlm_extension_manifest_version": "1.0.0",
            "generated_at": time.time(),
            "extensions": {r.extension_id: r.to_dict() for r in sorted(records, key=lambda r: r.extension_id)},
        }

        if path is not None:
            out_path = Path(path)
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            except OSError as exc:
                raise ManifestExportError(f"Failed to write manifest to '{out_path}': {exc}") from exc
            logger.info("Exported extension manifest to '%s' (%d entries).", out_path, len(records))
        else:
            logger.info("Exported in-memory extension manifest (%d entries).", len(records))

        return manifest

    def import_manifest(
        self,
        source: "str | Path | dict[str, Any]",
        *,
        apply: bool = True,
        continue_on_error: bool = False,
    ) -> dict[str, Optional[str]]:
        """Load a manifest (from :meth:`export_manifest`) and optionally apply it.

        Applying a manifest means enabling every extension recorded with
        state ``"enabled"`` and disabling every extension recorded with
        state ``"disabled"``; entries in other states are recorded into
        the known registry (for dependency resolution) but left
        untouched.

        Args:
            source: A path to a JSON manifest file, or an already-parsed
                manifest dict.
            apply: If True (default), enable/disable calls are actually
                issued against the bound live handle. If False, the
                manifest is only parsed and validated (useful for
                offline inspection without a live Kit process).
            continue_on_error: Passed through to the underlying
                :meth:`enable_multiple` / :meth:`disable_multiple` calls
                when ``apply`` is True.

        Returns:
            Mapping of extension id to ``None`` on success or an error
            message on failure, for every extension the manifest
            attempted to apply. Empty if ``apply`` is False.

        Raises:
            NotInitializedError: If ``apply`` is True but this manager is
                not bound to a live handle.
            ManifestImportError: If the manifest cannot be read/parsed or
                has an invalid shape.
        """
        if isinstance(source, dict):
            manifest = source
        else:
            path = Path(source)
            if not path.exists():
                raise ManifestImportError(f"Manifest file not found: '{path}'")
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise ManifestImportError(f"Failed to parse manifest '{path}': {exc}") from exc

        if not isinstance(manifest, dict) or "extensions" not in manifest:
            raise ManifestImportError("Manifest must be a mapping containing an 'extensions' key.")

        entries = manifest["extensions"]
        if not isinstance(entries, dict):
            raise ManifestImportError("Manifest 'extensions' key must be a mapping.")

        to_enable: list[str] = []
        to_disable: list[str] = []
        with self._lock:
            for ext_id, raw in entries.items():
                if not isinstance(raw, dict):
                    logger.warning("Ignoring malformed manifest entry for '%s'.", ext_id)
                    continue
                try:
                    category = ExtensionCategory(raw.get("category", ExtensionCategory.OTHER.value))
                except ValueError:
                    category = ExtensionCategory.OTHER
                self._known.setdefault(
                    ext_id,
                    ExtensionDescriptor(
                        extension_id=ext_id,
                        category=category,
                        description=str(raw.get("description", "")),
                        dependencies=tuple(raw.get("dependencies", ()) or ()),
                    ),
                )
                state_value = raw.get("state")
                if state_value == ExtensionState.ENABLED.value:
                    to_enable.append(ext_id)
                elif state_value == ExtensionState.DISABLED.value:
                    to_disable.append(ext_id)

        logger.info(
            "Parsed manifest with %d entries (%d to enable, %d to disable).",
            len(entries), len(to_enable), len(to_disable),
        )

        results: dict[str, Optional[str]] = {}
        if apply:
            self._require_ready()
            if to_enable:
                results.update(
                    self.enable_multiple(to_enable, resolve_deps=True, continue_on_error=continue_on_error)
                )
            if to_disable:
                results.update(
                    self.disable_multiple(to_disable, continue_on_error=continue_on_error)
                )

        return results


__all__ = [
    "ExtensionManager",
    "ExtensionDescriptor",
    "ExtensionMetadata",
    "ExtensionCategory",
    "ExtensionState",
    "ManagerState",
    "ExtensionManagerError",
    "NotInitializedError",
    "AlreadyInitializedError",
    "InvalidHandleError",
    "KitImportError",
    "ExtensionDiscoveryError",
    "ExtensionNotFoundError",
    "ExtensionEnableError",
    "ExtensionDisableError",
    "ExtensionReloadError",
    "MissingDependencyError",
    "DependencyCycleError",
    "VersionConflictError",
    "ManifestError",
    "ManifestExportError",
    "ManifestImportError",
]
