"""
app_launcher.py
══════════════════════════════════════════════════════════════════════════
Omniverse / Isaac Sim application lifecycle manager for the Omniverse
Connector layer of PhysWorldLM.

Pipeline position
------------------
    Natural Language → Ontology → WorldSpec → Scene Compiler → scene.usda
                                                                      │
                                                                      ▼
                                                        ┌───────────────────┐
                                                        │ omniverse/config  │
                                                        └─────────┬─────────┘
                                                                  ▼
                                                        ┌───────────────────┐
                                                        │ app_launcher.py   │  <-- this module
                                                        └─────────┬─────────┘
                                                                  ▼
                                              stage_manager.py / timeline / renderer / ...

Scope
-----
This module owns exactly one concern: the lifecycle of an embedded
Omniverse Kit / Isaac Sim application process --  discovering an
installation, launching it (GUI or headless), bringing up the handful
of subsystems every downstream component depends on (USD context,
extension system, renderer, physics, event loop, timeline, asset
resolver, stage), and tearing it all down again.

This module explicitly does NOT:
    * parse natural language or ontologies
    * build, validate, or compile a ``WorldSpec``
    * construct a Scene Graph
    * export or author USD content
    * spawn entities, sensors, or terrain
    * implement physics, rendering, planning, or ROS2 logic

Those concerns belong to ``scene_compiler.py``, ``usd_exporter.py``,
``stage_builder.py``, and the (future) ``StageManager`` /
``PhysicsScene`` / ``TimelineController`` / ``Renderer`` / ``Planner`` /
sensor-simulation / Replicator / ROS2 components. This module only
hands those components a live, ready application to attach to via
``get_app()`` / ``get_context()`` / ``get_stage()`` and a small
lifecycle-hook registry -- it never imports or references any of them
by name.

Design constraints
-------------------
    * No ``omni``/``pxr``/Isaac Sim import happens at module load time.
      Every such import is deferred to the call site that actually
      needs it, behind :func:`_lazy_import`, so this module -- and
      anything that merely imports it -- loads successfully on a
      machine with no Omniverse installation at all.
    * All failure modes raise a documented, specific
      :class:`LauncherError` subclass. Nothing lets a raw
      ``ImportError`` or an opaque Omniverse/Kit stack trace escape.
    * No global mutable state. Every piece of runtime state (the
      Kit/Isaac app handle, the USD context, the event-loop thread,
      the lifecycle-hook registry, ...) lives on the
      :class:`OmniverseLauncher` instance.
    * The launcher reuses :class:`omniverse.config.OmniverseConfig` as
      its single source of truth for paths, timesteps, and feature
      toggles -- it never re-implements detection or env-var parsing.

Public API
----------
    launcher = OmniverseLauncher()             # config auto-loaded
    launcher.launch()                          # blocks until ready
    app = launcher.get_app()
    stage = launcher.get_stage()
    ...
    launcher.shutdown()

Or, as a context manager::

    with OmniverseLauncher() as launcher:
        stage = launcher.get_stage()
        ...
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    ConfigValidationError,
    OmniverseConfig,
    OmniverseConfigError,
)

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.app_launcher")
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

class LauncherError(Exception):
    """Base class for all errors raised by :class:`OmniverseLauncher`."""


class LauncherConfigurationError(LauncherError):
    """Raised when the launcher has no valid, loaded configuration."""


class InstallationNotFoundError(LauncherError):
    """Raised when the requested backend (Kit / Isaac Sim) is not installed."""


class OmniverseImportError(LauncherError):
    """Raised when a required ``omni``/``pxr``/Isaac Sim module can't be imported.

    Distinct from a bare ``ImportError`` so callers can catch exactly
    "Omniverse isn't installed here" without accidentally swallowing an
    unrelated import bug in their own code.
    """


class LaunchError(LauncherError):
    """Raised when the underlying application process fails to start."""


class InitializationError(LauncherError):
    """Raised when a required subsystem fails to initialize after launch."""


class ShutdownError(LauncherError):
    """Raised when the application fails to shut down cleanly."""


class AlreadyRunningError(LauncherError):
    """Raised when :meth:`OmniverseLauncher.launch` is called while running."""


class NotRunningError(LauncherError):
    """Raised when a live-app accessor is called before/without a running app."""


class ReadyTimeoutError(LauncherError):
    """Raised by :meth:`OmniverseLauncher.wait_until_ready` in strict mode."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class LauncherState(str, Enum):
    """Lifecycle states of an :class:`OmniverseLauncher`.

    Transitions (happy path)::

        UNINITIALIZED -> CONFIGURED -> LAUNCHING -> INITIALIZING
            -> READY -> RUNNING -> SHUTTING_DOWN -> STOPPED

    ``ERROR`` is reachable from ``LAUNCHING`` or ``INITIALIZING`` and
    is terminal until :meth:`OmniverseLauncher.shutdown` (or
    :meth:`OmniverseLauncher.restart`) resets the launcher.
    """

    UNINITIALIZED = "uninitialized"
    CONFIGURED = "configured"
    LAUNCHING = "launching"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


class AppBackend(str, Enum):
    """Which embedded application backend this launcher started.

    Both backends are, in practice, an embedded Kit process created via
    the Isaac Sim ``SimulationApp`` bootstrap, differing only in which
    Kit "experience" file is loaded. They are modeled as distinct enum
    values because they resolve against different roots
    (``kit_root`` vs. ``isaac_root``), have different minimum
    versions, and may diverge further as Anthropic/NVIDIA's own launch
    tooling evolves.
    """

    KIT = "kit"
    ISAAC_SIM = "isaac_sim"


class LifecycleEvent(str, Enum):
    """Hook points future components may attach behavior to.

    ``OmniverseLauncher`` never imports or knows about ``StageManager``,
    ``PhysicsScene``, ``TimelineController``, ``Renderer``, ``Planner``,
    sensor simulation, Replicator, or ROS2 -- those components instead
    register callbacks against these events via
    :meth:`OmniverseLauncher.register_hook`, keeping the launcher fully
    decoupled from anything that consumes it.
    """

    BEFORE_LAUNCH = "before_launch"
    AFTER_LAUNCH = "after_launch"
    READY = "ready"
    BEFORE_SHUTDOWN = "before_shutdown"
    AFTER_SHUTDOWN = "after_shutdown"
    ERROR = "error"


# ════════════════════════════════════════════════════════════════════════
# Constants (no magic numbers scattered through the class body)
# ════════════════════════════════════════════════════════════════════════

#: Fallback wait, in seconds, for wait_until_ready() when no timeout is given.
_DEFAULT_READY_TIMEOUT_SECONDS = 60.0

#: Poll interval, in seconds, used while waiting for readiness.
_READY_POLL_INTERVAL_SECONDS = 0.05

#: Kit "experience" files to look for under a Kit install root, in
#: priority order. Only used as a hint -- if none are found the
#: SimulationApp bootstrap's own default experience is used instead.
_KIT_EXPERIENCE_CANDIDATES: tuple[str, ...] = (
    "omni.create.kit",
    "omni.kit.base.editor.kit",
)

#: Isaac Sim experience files to look for under an Isaac Sim install root.
_ISAAC_EXPERIENCE_CANDIDATES: tuple[str, ...] = (
    "isaac-sim.python.kit",
    "omni.isaac.sim.python.kit",
)

#: Maps an ``OmniverseConfig.renderer`` value to the SimulationApp/Kit
#: launch-config renderer token. Best-effort: unknown values are passed
#: through as-is so a newer Kit renderer name never hard-fails a launch.
_RENDERER_LAUNCH_TOKENS: dict[str, str] = {
    "rtx_realtime": "RaytracedLighting",
    "rtx_pathtracing": "PathTracing",
    "hydra_storm": "iray",
}

#: Config boolean field -> Kit extension id, for optional extensions that
#: are simply enabled/disabled (not owned or configured by this module).
_OPTIONAL_EXTENSIONS: dict[str, str] = {
    "enable_replicator": "omni.replicator.core",
    "enable_ros2": "omni.isaac.ros2_bridge",
    "enable_livestream": "omni.kit.livestream.native",
}


# ════════════════════════════════════════════════════════════════════════
# Lazy import helper
# ════════════════════════════════════════════════════════════════════════

def _lazy_import(module_name: str, *, hint: str = "") -> Any:
    """Import ``module_name``, raising :class:`OmniverseImportError` on failure.

    Every ``omni``/``pxr``/Isaac Sim import in this module goes through
    this function so that (a) importing ``app_launcher`` itself never
    requires Omniverse to be installed, and (b) a missing dependency
    surfaces as one clear, catchable exception instead of a raw
    ``ImportError`` with an Omniverse-specific stack trace.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = (
            f"Failed to import '{module_name}'. This operation requires "
            "running inside (or alongside) an installed Omniverse Kit / "
            "Isaac Sim Python environment."
        )
        if hint:
            message = f"{message} {hint}"
        raise OmniverseImportError(message) from exc


# ════════════════════════════════════════════════════════════════════════
# Internal runtime handles
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _AppHandles:
    """Container for every live handle this launcher manages.

    Deliberately a plain, private, mutable bag of ``Any`` -- the real
    types (``omni.isaac.kit.SimulationApp``, ``omni.kit.app.IApp``,
    ``omni.usd.UsdContext``, ``pxr.Usd.Stage``, ...) are never imported
    at module scope, so this dataclass cannot be meaningfully typed any
    more tightly without defeating the lazy-import design constraint.
    """

    backend: Optional[AppBackend] = None
    simulation_app: Optional[Any] = None
    kit_app: Optional[Any] = None
    usd_context: Optional[Any] = None
    stage: Optional[Any] = None
    extension_manager: Optional[Any] = None
    physics_interface: Optional[Any] = None
    timeline_interface: Optional[Any] = None
    asset_resolver: Optional[Any] = None
    settings_interface: Optional[Any] = None

    def clear(self) -> None:
        """Reset every handle to its unset state."""
        self.backend = None
        self.simulation_app = None
        self.kit_app = None
        self.usd_context = None
        self.stage = None
        self.extension_manager = None
        self.physics_interface = None
        self.timeline_interface = None
        self.asset_resolver = None
        self.settings_interface = None


@dataclass
class _EventLoopPump:
    """Background thread that periodically drives ``app.update()``.

    Embedded Kit applications do not run their own event loop the way
    a standalone Kit executable does -- something has to call
    ``app.update()`` repeatedly to advance rendering, physics, and
    extension callbacks. This small helper owns exactly that, so the
    rest of the process (and anything future components do with
    ``get_app()``) can treat the application as "just running" without
    needing to know how it's pumped.
    """

    interval_seconds: float
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self, update_fn: Callable[[], None]) -> None:
        """Start pumping ``update_fn`` on a daemon thread, if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _run() -> None:
            while not self._stop_event.is_set():
                try:
                    update_fn()
                except Exception:  # noqa: BLE001 - never let the pump die silently
                    logger.exception("Event loop pump: update() raised; continuing.")
                self._stop_event.wait(self.interval_seconds)

        self._thread = threading.Thread(
            target=_run, name="physworldlm-omniverse-event-loop", daemon=True
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        """Signal the pump to stop and wait briefly for it to exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=join_timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ════════════════════════════════════════════════════════════════════════
# OmniverseLauncher
# ════════════════════════════════════════════════════════════════════════

class OmniverseLauncher:
    """Manages the lifecycle of an embedded Omniverse Kit / Isaac Sim app.

    ``OmniverseLauncher`` is the single entry point every other
    Omniverse Connector component (and, later, ``StageManager``,
    ``PhysicsScene``, ``TimelineController``, ``Renderer``, ``Planner``,
    sensor simulation, Replicator, and the ROS2 bridge) launches and
    attaches to. It contains no business logic: it does not compile
    ``WorldSpec`` objects, does not build or export USD, and does not
    spawn entities -- see ``scene_compiler.py`` / ``usd_exporter.py`` /
    ``stage_builder.py`` for those concerns.

    Thread-safety: all state transitions are guarded by an internal
    lock, so ``launch()`` / ``shutdown()`` / ``restart()`` are safe to
    call from multiple threads (though only one will ever win a given
    transition -- the others will observe the resulting state/errors).

    Example:
        >>> launcher = OmniverseLauncher()
        >>> launcher.launch(headless=True)
        >>> stage = launcher.get_stage()
        >>> launcher.shutdown()
    """

    def __init__(
        self,
        config: Optional[OmniverseConfig] = None,
        *,
        auto_load_config: bool = True,
    ) -> None:
        """Create a launcher.

        Args:
            config: An explicit :class:`OmniverseConfig` to use. If
                ``None`` and ``auto_load_config`` is True, a config is
                built via ``OmniverseConfig.load_from_env()`` the
                first time it's needed.
            auto_load_config: Whether to lazily auto-load a config
                (via env-var overrides on top of auto-detection) when
                none has been supplied or loaded yet. When False,
                :meth:`load_config` must be called explicitly before
                :meth:`launch`.
        """
        self._lock = threading.RLock()
        self._state: LauncherState = LauncherState.UNINITIALIZED
        self._config: Optional[OmniverseConfig] = None
        self._config_source: Optional["str | Path | OmniverseConfig"] = None
        self._auto_load_config = auto_load_config
        self._handles = _AppHandles()
        self._event_loop_pump: Optional[_EventLoopPump] = None
        self._ready_event = threading.Event()
        self._hooks: dict[LifecycleEvent, list[Callable[["OmniverseLauncher"], None]]] = {
            event: [] for event in LifecycleEvent
        }
        self._last_error: Optional[BaseException] = None

        if config is not None:
            self.load_config(config)

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> LauncherState:
        """Current lifecycle state."""
        with self._lock:
            return self._state

    def is_initialized(self) -> bool:
        """Whether the app and its required subsystems have been brought up.

        True for ``READY`` and ``RUNNING``; False in every other state,
        including ``ERROR`` (a failed launch is not a usable one).
        """
        return self.state in (LauncherState.READY, LauncherState.RUNNING)

    def is_running(self) -> bool:
        """Whether the application process is live and considered usable.

        Alias-level distinction from :meth:`is_initialized`: kept
        separate so future callers that care specifically about "is the
        event loop actively being pumped" (``RUNNING``) rather than
        merely "is it initialized and idle" (``READY``) have a stable
        method to depend on, without this module needing to guess which
        one they mean today.
        """
        return self.state in (LauncherState.READY, LauncherState.RUNNING)

    def wait_until_ready(
        self,
        timeout: Optional[float] = None,
        *,
        raise_on_timeout: bool = False,
    ) -> bool:
        """Block until the launcher reaches ``READY``/``RUNNING`` or times out.

        Args:
            timeout: Seconds to wait. Defaults to
                ``_DEFAULT_READY_TIMEOUT_SECONDS`` if not given.
            raise_on_timeout: If True, raise :class:`ReadyTimeoutError`
                instead of returning False on timeout.

        Returns:
            True if the launcher became ready before the timeout
            elapsed, False otherwise (unless ``raise_on_timeout``).
        """
        effective_timeout = (
            _DEFAULT_READY_TIMEOUT_SECONDS if timeout is None else timeout
        )
        logger.info("Waiting for Omniverse application to become ready (timeout=%.1fs).", effective_timeout)
        became_ready = self._ready_event.wait(effective_timeout)
        if became_ready:
            logger.info("Omniverse application is ready.")
        else:
            logger.warning("Timed out after %.1fs waiting for readiness.", effective_timeout)
            if raise_on_timeout:
                raise ReadyTimeoutError(
                    f"Launcher did not become ready within {effective_timeout:.1f}s."
                )
        return became_ready

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_config(
        self, source: Optional["str | Path | OmniverseConfig"] = None
    ) -> OmniverseConfig:
        """Load (or set) this launcher's :class:`OmniverseConfig`.

        Args:
            source: One of:
                * ``None`` -- build via ``OmniverseConfig.load_from_env()``
                  (auto-detection layered with any ``PWLM_*`` env vars).
                * a path (``str``/``Path``) to a YAML config file --
                  loaded via ``OmniverseConfig.load_from_yaml()``.
                * an already-constructed ``OmniverseConfig`` -- used
                  as-is.

        Returns:
            The loaded/assigned :class:`OmniverseConfig`.

        Raises:
            LauncherConfigurationError: If the config cannot be built
                or loaded.
        """
        logger.info("Loading configuration.")
        try:
            if source is None:
                config = OmniverseConfig.load_from_env()
            elif isinstance(source, OmniverseConfig):
                config = source
            else:
                config = OmniverseConfig.load_from_yaml(source)
        except OmniverseConfigError as exc:
            raise LauncherConfigurationError(f"Failed to load configuration: {exc}") from exc

        with self._lock:
            self._config = config
            self._config_source = source
            if self._state == LauncherState.UNINITIALIZED:
                self._state = LauncherState.CONFIGURED

        logger.info(
            "Configuration loaded (platform=%s, kit_root=%s, isaac_root=%s).",
            config.platform, config.kit_root, config.isaac_root,
        )
        return config

    def reload_config(self) -> OmniverseConfig:
        """Reload configuration from the same source used last time.

        If the launcher is currently initialized/running, the new
        configuration is stored but does **not** retroactively change
        the live application -- call :meth:`restart` to apply it.

        Raises:
            LauncherConfigurationError: If no source has been recorded
                yet (i.e. :meth:`load_config` was never called and
                auto-loading has not happened either).
        """
        with self._lock:
            source = self._config_source
            was_running = self.is_running()

        if source is None and not self._auto_load_config:
            raise LauncherConfigurationError(
                "reload_config() called with no prior configuration source; "
                "call load_config() at least once first."
            )

        config = self.load_config(source)
        if was_running:
            logger.warning(
                "Configuration reloaded while the application is running; "
                "call restart() to apply the new configuration."
            )
        return config

    def _ensure_config(self) -> OmniverseConfig:
        """Return the active config, auto-loading one if permitted."""
        with self._lock:
            if self._config is not None:
                return self._config
        if not self._auto_load_config:
            raise LauncherConfigurationError(
                "No configuration loaded. Call load_config() first, or "
                "construct OmniverseLauncher(auto_load_config=True)."
            )
        return self.load_config(None)

    # ------------------------------------------------------------------
    # Lifecycle hooks (the launcher's only extension mechanism)
    # ------------------------------------------------------------------

    def register_hook(
        self, event: LifecycleEvent, callback: Callable[["OmniverseLauncher"], None]
    ) -> None:
        """Register ``callback`` to run when ``event`` occurs.

        This is the *only* extension point this module exposes.
        Future components (``StageManager``, ``PhysicsScene``,
        ``TimelineController``, ``Renderer``, ``Planner``, sensor
        simulation, Replicator, ROS2, ...) hook into launcher
        lifecycle exclusively through this method rather than the
        launcher importing or knowing about any of them.

        Callbacks are invoked synchronously, in registration order,
        with this launcher instance as their only argument. A
        callback that raises is logged and does not prevent remaining
        callbacks (or the lifecycle transition itself) from proceeding.
        """
        with self._lock:
            self._hooks[event].append(callback)

    def unregister_hook(
        self, event: LifecycleEvent, callback: Callable[["OmniverseLauncher"], None]
    ) -> bool:
        """Remove a previously registered hook. Returns True if it was present."""
        with self._lock:
            try:
                self._hooks[event].remove(callback)
                return True
            except ValueError:
                return False

    def _run_hooks(self, event: LifecycleEvent) -> None:
        with self._lock:
            callbacks = list(self._hooks[event])
        for callback in callbacks:
            try:
                callback(self)
            except Exception:  # noqa: BLE001 - a hook's failure is not the launcher's
                logger.exception("Lifecycle hook for '%s' raised an exception.", event.value)

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def launch(
        self,
        *,
        headless: Optional[bool] = None,
        use_isaac: Optional[bool] = None,
        wait_for_ready: bool = True,
        timeout: Optional[float] = None,
        run_event_loop: bool = True,
    ) -> None:
        """Launch the Omniverse application and bring up core subsystems.

        Args:
            headless: Override ``config.headless`` for this launch
                (supports both GUI and headless modes). Defaults to
                the loaded config's value.
            use_isaac: Force the Isaac Sim backend (True) or the plain
                Kit backend (False). Defaults to auto-selecting Isaac
                Sim when ``config.is_isaac_available`` is True,
                otherwise Kit.
            wait_for_ready: If True (default), block until the
                launcher reports ready (or ``timeout`` elapses) before
                returning.
            timeout: Seconds to wait for readiness; see
                :meth:`wait_until_ready`.
            run_event_loop: If True (default), start the background
                pump thread that keeps the application's event loop
                advancing (``app.update()``) after subsystem
                initialization completes.

        Raises:
            AlreadyRunningError: If already initialized/running.
            LauncherConfigurationError: If configuration is invalid.
            InstallationNotFoundError: If the selected backend is not
                installed.
            LaunchError: If the application process fails to start.
            InitializationError: If a required subsystem fails to
                initialize.
        """
        with self._lock:
            if self.is_running():
                raise AlreadyRunningError(
                    f"Launcher is already in state '{self._state.value}'; "
                    "call shutdown() (or restart()) first."
                )
            self._state = LauncherState.LAUNCHING
            self._ready_event.clear()
            self._last_error = None

        logger.info("Initializing launch sequence.")
        self._run_hooks(LifecycleEvent.BEFORE_LAUNCH)

        try:
            config = self._ensure_config()
            self._validate_config_for_launch(config)

            effective_headless = config.headless if headless is None else headless
            backend = self._resolve_backend(config, use_isaac)

            logger.info(
                "Launching backend=%s headless=%s renderer=%s",
                backend.value, effective_headless, config.renderer,
            )
            self._create_app(config, backend=backend, headless=effective_headless)

            with self._lock:
                self._state = LauncherState.INITIALIZING
            logger.info("Initializing subsystems.")
            self._initialize_subsystems(config)

            if run_event_loop:
                self._start_event_loop(config)
                with self._lock:
                    self._state = LauncherState.RUNNING
            else:
                with self._lock:
                    self._state = LauncherState.READY

            self._ready_event.set()
            logger.info("Omniverse application ready (backend=%s).", backend.value)
            self._run_hooks(LifecycleEvent.READY)
            self._run_hooks(LifecycleEvent.AFTER_LAUNCH)

        except LauncherError as exc:
            self._enter_error_state(exc)
            raise
        except Exception as exc:  # noqa: BLE001 - never leak an opaque Kit traceback
            wrapped = LaunchError(f"Unexpected failure while launching: {exc}")
            self._enter_error_state(wrapped)
            raise wrapped from exc

        if wait_for_ready:
            self.wait_until_ready(timeout)

    def shutdown(self, *, cleanup: bool = True) -> None:
        """Gracefully shut down the application and release all handles.

        Idempotent: calling ``shutdown()`` when nothing is running logs
        a message and returns rather than raising.

        Args:
            cleanup: If True (default), release internal handles and
                reset the launcher to a fresh, re-launchable state
                after shutdown. If False, the underlying process is
                stopped but handles are left in place for inspection
                (mainly useful for diagnosing a failed shutdown).

        Raises:
            ShutdownError: If the underlying application fails to
                close and ``cleanup`` was requested.
        """
        with self._lock:
            if self._state in (LauncherState.UNINITIALIZED, LauncherState.CONFIGURED,
                                LauncherState.STOPPED):
                logger.info("shutdown() called with nothing running; nothing to do.")
                return
            self._state = LauncherState.SHUTTING_DOWN

        logger.info("Shutting down Omniverse application.")
        self._run_hooks(LifecycleEvent.BEFORE_SHUTDOWN)

        self._stop_event_loop()

        close_error: Optional[Exception] = None
        simulation_app = self._handles.simulation_app
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:  # noqa: BLE001
                close_error = exc
                logger.exception("Error while closing the underlying application.")

        if cleanup:
            logger.info("Cleaning up launcher state.")
            self._handles.clear()
            self._ready_event.clear()

        with self._lock:
            self._state = LauncherState.STOPPED

        logger.info("Shutdown complete.")
        self._run_hooks(LifecycleEvent.AFTER_SHUTDOWN)

        if close_error is not None:
            raise ShutdownError(
                f"Application did not close cleanly: {close_error}"
            ) from close_error

    def restart(self, **launch_kwargs: Any) -> None:
        """Shut down (if running) and relaunch with the given ``launch()`` kwargs."""
        logger.info("Restarting Omniverse application.")
        if self.is_running() or self.state == LauncherState.ERROR:
            self.shutdown()
        self.launch(**launch_kwargs)

    # ------------------------------------------------------------------
    # Accessors for downstream components
    # ------------------------------------------------------------------

    def get_app(self) -> Any:
        """Return the live Kit application object (``omni.kit.app.IApp``).

        Raises:
            NotRunningError: If the launcher is not initialized.
        """
        self._require_running()
        return self._handles.kit_app

    def get_context(self) -> Any:
        """Return the live USD context (``omni.usd.UsdContext``).

        Raises:
            NotRunningError: If the launcher is not initialized.
        """
        self._require_running()
        return self._handles.usd_context

    def get_stage(self) -> Any:
        """Return the current USD stage (``pxr.Usd.Stage``) held by the context.

        Raises:
            NotRunningError: If the launcher is not initialized.
        """
        self._require_running()
        return self._handles.stage

    def get_config(self) -> OmniverseConfig:
        """Return the configuration currently associated with this launcher."""
        return self._ensure_config()

    def _require_running(self) -> None:
        if not self.is_initialized():
            raise NotRunningError(
                f"Omniverse application is not running (state='{self.state.value}'). "
                "Call launch() first."
            )

    # ------------------------------------------------------------------
    # Context-manager convenience
    # ------------------------------------------------------------------

    def __enter__(self) -> "OmniverseLauncher":
        self.launch()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal: error handling
    # ------------------------------------------------------------------

    def _enter_error_state(self, exc: BaseException) -> None:
        logger.error("Launch sequence failed: %s", exc)
        with self._lock:
            self._state = LauncherState.ERROR
            self._last_error = exc
        self._ready_event.clear()
        self._run_hooks(LifecycleEvent.ERROR)
        # Best-effort teardown of anything partially created, so a
        # failed launch never leaves an orphaned process behind.
        self._stop_event_loop()
        if self._handles.simulation_app is not None:
            try:
                self._handles.simulation_app.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing application after a failed launch.")
        self._handles.clear()

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that most recently drove the launcher into ``ERROR``, if any."""
        with self._lock:
            return self._last_error

    # ------------------------------------------------------------------
    # Internal: configuration validation
    # ------------------------------------------------------------------

    def _validate_config_for_launch(self, config: OmniverseConfig) -> None:
        try:
            config.validate(require_kit=False, require_isaac=False)
        except ConfigValidationError as exc:
            raise LauncherConfigurationError(f"Invalid configuration: {exc}") from exc

    def _resolve_backend(
        self, config: OmniverseConfig, use_isaac: Optional[bool]
    ) -> AppBackend:
        if use_isaac is None:
            use_isaac = config.is_isaac_available

        if use_isaac:
            if not config.is_isaac_available:
                raise InstallationNotFoundError(
                    "Isaac Sim backend requested but no Isaac Sim installation "
                    f"was found or configured (isaac_root={config.isaac_root})."
                )
            return AppBackend.ISAAC_SIM

        if not config.is_kit_available:
            raise InstallationNotFoundError(
                "Kit backend requested but no Omniverse Kit installation was "
                f"found or configured (kit_root={config.kit_root})."
            )
        return AppBackend.KIT

    # ------------------------------------------------------------------
    # Internal: application creation
    # ------------------------------------------------------------------

    def _resolve_experience_file(
        self, root: Optional[Path], candidates: tuple[str, ...]
    ) -> Optional[str]:
        """Best-effort lookup of a known experience (``.kit``) file under ``root``.

        Returns ``None`` (letting the SimulationApp bootstrap fall back
        to its own default experience) if ``root`` is unset or none of
        the known candidates exist -- this is a hint, not a requirement.
        """
        if root is None:
            return None
        apps_dir = root / "apps"
        search_dirs = (apps_dir, root) if apps_dir.exists() else (root,)
        for directory in search_dirs:
            for candidate in candidates:
                candidate_path = directory / candidate
                if candidate_path.exists():
                    return str(candidate_path)
        return None

    def _build_launch_config(self, config: OmniverseConfig, headless: bool) -> dict[str, Any]:
        """Translate ``OmniverseConfig`` into a SimulationApp ``launch_config`` dict.

        Only forwards the small, stable subset of settings this module
        is responsible for (headless mode and renderer choice); every
        other subsystem (physics, extensions, timeline, ...) is brought
        up explicitly in :meth:`_initialize_subsystems` instead of via
        opaque launch-config flags, so this module's behavior stays
        legible and testable.
        """
        renderer_token = _RENDERER_LAUNCH_TOKENS.get(config.renderer, config.renderer)
        launch_config: dict[str, Any] = {
            "headless": headless,
            "renderer": renderer_token,
        }
        if config.gpu_device is not None:
            launch_config["active_gpu"] = config.gpu_device
        return launch_config

    def _create_app(
        self, config: OmniverseConfig, *, backend: AppBackend, headless: bool
    ) -> None:
        """Create and store the embedded application via the Isaac Sim bootstrap.

        Both backends embed through ``omni.isaac.kit.SimulationApp`` --
        the standard mechanism for hosting a full Kit process inside a
        Python interpreter -- selecting a different Kit "experience"
        file for a plain-Kit versus an Isaac Sim session.

        Raises:
            OmniverseImportError: If the SimulationApp bootstrap can't
                be imported (Omniverse/Isaac Sim not installed here).
            LaunchError: If the application process fails to start.
        """
        isaac_kit_module = _lazy_import(
            "omni.isaac.kit",
            hint="Install Isaac Sim, or ensure this process is running inside a Kit environment.",
        )
        simulation_app_cls = getattr(isaac_kit_module, "SimulationApp", None)
        if simulation_app_cls is None:
            raise OmniverseImportError(
                "'omni.isaac.kit' was imported but does not expose 'SimulationApp'; "
                "this Omniverse/Isaac Sim installation may be incompatible."
            )

        if backend is AppBackend.ISAAC_SIM:
            experience = self._resolve_experience_file(config.isaac_root, _ISAAC_EXPERIENCE_CANDIDATES)
        else:
            experience = self._resolve_experience_file(config.kit_root, _KIT_EXPERIENCE_CANDIDATES)

        launch_config = self._build_launch_config(config, headless)
        logger.info("Launching application (experience=%s).", experience or "<default>")

        try:
            if experience:
                simulation_app = simulation_app_cls(launch_config, experience=experience)
            else:
                simulation_app = simulation_app_cls(launch_config)
        except Exception as exc:  # noqa: BLE001 - Kit's own exceptions are opaque
            raise LaunchError(
                f"Failed to launch the embedded application (backend={backend.value}): {exc}"
            ) from exc

        kit_app_module = _lazy_import("omni.kit.app")
        try:
            kit_app = kit_app_module.get_app()
        except Exception as exc:  # noqa: BLE001
            raise LaunchError(f"Application launched but 'omni.kit.app.get_app()' failed: {exc}") from exc

        self._handles.backend = backend
        self._handles.simulation_app = simulation_app
        self._handles.kit_app = kit_app

    # ------------------------------------------------------------------
    # Internal: subsystem initialization
    # ------------------------------------------------------------------

    def _initialize_subsystems(self, config: OmniverseConfig) -> None:
        """Run every required (and configured-optional) subsystem init step.

        Each step is isolated so a single subsystem's failure produces
        a specific, attributable :class:`InitializationError` rather
        than an undifferentiated launch failure.
        """
        steps: tuple[tuple[str, Callable[[OmniverseConfig], None]], ...] = (
            ("logging bridge", self._init_logging_bridge),
            ("USD context", self._init_usd_context),
            ("extension system", self._init_extension_system),
            ("renderer", self._init_renderer),
            ("physics", self._init_physics),
            ("timeline", self._init_timeline),
            ("asset resolver", self._init_asset_resolver),
            ("stage manager", self._init_stage_manager),
            ("optional extensions", self._init_optional_extensions),
        )
        for step_name, step_fn in steps:
            logger.info("Initializing %s.", step_name)
            try:
                step_fn(config)
            except OmniverseImportError as exc:
                raise InitializationError(
                    f"Could not initialize '{step_name}': {exc}"
                ) from exc
            except InitializationError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise InitializationError(
                    f"Unexpected failure initializing '{step_name}': {exc}"
                ) from exc

    def _init_logging_bridge(self, config: OmniverseConfig) -> None:
        """Best-effort bridge of Kit's internal logging into this module's logger.

        Non-fatal by design: a bridging failure should never prevent
        the rest of the launch sequence from proceeding.
        """
        try:
            carb_module = importlib.import_module("carb")
            carb_module.log_info("PhysWorldLM Omniverse launcher: logging bridge attached.")
        except Exception:  # noqa: BLE001
            logger.debug("carb logging bridge unavailable; continuing without it.", exc_info=True)

    def _init_usd_context(self, config: OmniverseConfig) -> None:
        omni_usd = _lazy_import("omni.usd")
        context = omni_usd.get_context()
        if context is None:
            raise InitializationError("omni.usd.get_context() returned None.")

        stage = context.get_stage()
        if stage is None:
            try:
                context.new_stage()
            except Exception as exc:  # noqa: BLE001
                raise InitializationError(f"Failed to create a new USD stage: {exc}") from exc
            stage = context.get_stage()

        self._handles.usd_context = context
        self._handles.stage = stage

    def _init_extension_system(self, config: OmniverseConfig) -> None:
        if self._handles.kit_app is None:
            raise InitializationError("Kit application handle is missing; cannot access extension manager.")
        try:
            extension_manager = self._handles.kit_app.get_extension_manager()
        except Exception as exc:  # noqa: BLE001
            raise InitializationError(f"Failed to obtain the extension manager: {exc}") from exc
        self._handles.extension_manager = extension_manager

    def _init_renderer(self, config: OmniverseConfig) -> None:
        # Renderer selection happens at app-creation time via
        # _build_launch_config(); this step only applies the small set
        # of post-launch renderer toggles OmniverseConfig exposes.
        try:
            carb_settings = importlib.import_module("carb.settings")
            settings = carb_settings.get_settings()
        except Exception:  # noqa: BLE001
            logger.warning("carb.settings unavailable; skipping renderer toggle application.")
            return

        self._handles.settings_interface = settings
        if config.enable_dlss:
            if not config.enable_rtx:
                logger.warning("enable_dlss is set but enable_rtx is False; DLSS requires RTX.")
            else:
                try:
                    settings.set("/rtx/post/dlss/enabled", True)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to enable DLSS setting; continuing without it.")

    def _init_physics(self, config: OmniverseConfig) -> None:
        if not config.enable_physics:
            logger.info("Physics disabled by configuration; skipping.")
            return
        physx_module = _lazy_import(
            "omni.physx",
            hint="PhysX extension not available in this Omniverse installation.",
        )
        try:
            self._handles.physics_interface = physx_module.get_physx_interface()
        except Exception as exc:  # noqa: BLE001
            raise InitializationError(f"Failed to acquire the PhysX interface: {exc}") from exc

    def _init_timeline(self, config: OmniverseConfig) -> None:
        timeline_module = _lazy_import("omni.timeline")
        try:
            timeline = timeline_module.get_timeline_interface()
        except Exception as exc:  # noqa: BLE001
            raise InitializationError(f"Failed to acquire the timeline interface: {exc}") from exc
        self._handles.timeline_interface = timeline

    def _init_asset_resolver(self, config: OmniverseConfig) -> None:
        try:
            client_module = importlib.import_module("omni.client")
        except ImportError:
            logger.warning("omni.client unavailable; asset resolver step is a no-op.")
            return
        self._handles.asset_resolver = client_module
        if config.nucleus_server:
            logger.info("Nucleus server configured: %s", config.nucleus_server)

    def _init_stage_manager(self, config: OmniverseConfig) -> None:
        # This refers only to Kit's built-in stage/session bookkeeping
        # (already brought up in _init_usd_context) -- not the future
        # PhysWorldLM StageManager component, which this module never
        # imports or constructs. This step just confirms the invariant
        # every downstream component relies on: a valid, open stage.
        if self._handles.stage is None:
            raise InitializationError("No USD stage is open after USD context initialization.")

    def _init_optional_extensions(self, config: OmniverseConfig) -> None:
        extension_manager = self._handles.extension_manager
        if extension_manager is None:
            logger.warning("No extension manager available; skipping optional extensions.")
            return

        for config_flag, extension_id in _OPTIONAL_EXTENSIONS.items():
            if not getattr(config, config_flag, False):
                continue
            try:
                extension_manager.set_extension_enabled_immediate(extension_id, True)
                logger.info("Enabled optional extension '%s' (%s).", extension_id, config_flag)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not enable optional extension '%s' (%s); continuing without it.",
                    extension_id, config_flag,
                )

    # ------------------------------------------------------------------
    # Internal: event loop pump
    # ------------------------------------------------------------------

    def _start_event_loop(self, config: OmniverseConfig) -> None:
        if self._handles.simulation_app is None:
            raise InitializationError("Cannot start the event loop: no application handle.")

        def _update() -> None:
            self._handles.simulation_app.update()

        self._event_loop_pump = _EventLoopPump(interval_seconds=config.render_dt)
        self._event_loop_pump.start(_update)
        logger.info("Event loop pump started (interval=%.4fs).", config.render_dt)

    def _stop_event_loop(self) -> None:
        if self._event_loop_pump is not None:
            self._event_loop_pump.stop()
            self._event_loop_pump = None


__all__ = [
    "OmniverseLauncher",
    "LauncherState",
    "AppBackend",
    "LifecycleEvent",
    "LauncherError",
    "LauncherConfigurationError",
    "InstallationNotFoundError",
    "OmniverseImportError",
    "LaunchError",
    "InitializationError",
    "ShutdownError",
    "AlreadyRunningError",
    "NotRunningError",
    "ReadyTimeoutError",
]
