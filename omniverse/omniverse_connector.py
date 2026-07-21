"""
omniverse_connector.py
══════════════════════════════════════════════════════════════════════════
The single public facade for the Omniverse backend of PhysWorldLM.

Pipeline position
------------------
    Natural Language → Prompt Parser → MiniLM Entity Encoder
                                              │
                                              ▼
                                     Physics Ontology → WorldSpec
                                              │
                                              ▼
                                   Scene Compiler → USD Exporter
                                              │
                                              ▼
                                          scene.usd
                                              │
                                              ▼
                                ┌────────────────────────────┐
                                │      OmniverseConnector     │  <-- this module
                                └────────────────────────────┘
                                              │
                                              ▼
                                      Omniverse Runtime

Scope
-----
``OmniverseConnector`` is the ONLY public interface to the Omniverse
backend. Nothing outside the ``omniverse`` package should directly
instantiate ``AppLauncher``, ``ExtensionManager``, ``StageManager``,
``AssetServer``, ``USDLoader``, ``PhysicsScene``, ``TimelineController``,
or ``Renderer`` -- the connector owns one instance of each internally
(dependency injection for tests, lazy construction for production) and
coordinates them through a single, simulator-agnostic API.

This module implements NO physics, USD generation, asset management,
renderer internals, timeline internals, or stage internals itself --
every one of those concerns is fully delegated to its owning manager.
The connector is purely an orchestration/facade layer.

**The connector is a pure backend runtime and knows nothing about
``WorldSpec`` or scene compilation.** It accepts only a path to an
already-compiled ``.usd*`` stage file (``load_stage(path)``). Turning a
``WorldSpec`` into USD is the responsibility of the upstream Scene
Compiler / USD Exporter stages of the PhysWorldLM pipeline, which live
outside this package entirely -- keeping this facade, and by extension
every future simulator backend built the same way (Gazebo, MuJoCo,
Bullet, Unreal, Unity), fully independent of PhysWorldLM Core.

Assumptions about undocumented collaborators
---------------------------------------------
``config.py``, ``timeline_controller.py``, and ``renderer.py`` are the
only sibling modules whose source this file was written against
directly -- their real classes (``OmniverseConfig``, ``TimelineController``,
``Renderer``) are imported and used as-is, matching their actual public
APIs.

``AppLauncher``, ``ExtensionManager``, ``StageManager``, ``AssetServer``,
``USDLoader``, and ``PhysicsScene`` were **not** available to inspect
when this file was written. Each is therefore represented here by a
minimal, structural :class:`typing.Protocol` capturing only the methods
this connector needs to call, inferred from the lifecycle and public-API
naming given in the design brief (e.g. ``launch()``, ``enable_extensions()``,
``create_stage()``, ``load_assets()``, ``load()``/``export()``,
``initialize()``/``reset()``). The real sibling modules are imported
lazily -- only inside the small ``_build_*()`` factory method for each
manager, only the first time ``initialize()`` actually needs one -- so
that:

    1. Importing this module never requires every sibling module to
       already exist or be import-clean.
    2. If a real manager's method names differ from the Protocol here,
       only that one Protocol and its ``_build_*()`` factory need
       updating -- nothing else in this facade changes.
    3. Every manager can be replaced with a test fake via constructor
       injection, making the whole connector unit-testable without a
       running Omniverse Kit process.

Design constraints
-------------------
    * All mutable state lives on the instance, guarded by a single
      re-entrant lock. There is no module-level mutable state.
    * Every owned component follows the same "injected-or-lazily-built"
      pattern already used by ``TimelineController``/``Renderer``:
      supply your own instance via the constructor (tests, custom
      backends), or let the connector build a default one lazily inside
      ``initialize()``.
    * ``restart()`` rebuilds every non-injected component from scratch,
      because ``TimelineController``/``Renderer`` (and, by the same
      contract, the other managers) are intentionally single-shot: their
      own ``initialize()`` is a no-op once they've been shut down.

Public API
----------
    connector = OmniverseConnector(config=OmniverseConfig.default())
    with connector:
        connector.load_stage("scene.usd")
        connector.play()
        connector.step()
        connector.render_frame()
        image = connector.capture_image("output/frame_0001.png")
        health = connector.health_check()

Changelog
---------
    * Initial implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union, runtime_checkable

from .config import OmniverseConfig
from .renderer import CaptureResult, RenderStatistics, Renderer
from .timeline_controller import TimelineController, TimelineStatistics

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.omniverse_connector")
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

class OmniverseConnectorError(Exception):
    """Base class for all OmniverseConnector errors."""


class ConnectorStateError(OmniverseConnectorError):
    """Raised when an operation is invalid for the current connector state."""


class ConnectorInitializationError(OmniverseConnectorError):
    """Raised when the runtime (or one of its owned managers) fails to initialize."""


class ConnectorHealthError(OmniverseConnectorError):
    """Raised by strict health checks that choose to fail loudly rather than report."""


class StageLoadError(OmniverseConnectorError):
    """Raised when loading, reloading, unloading, or exporting a stage fails."""


# ════════════════════════════════════════════════════════════════════════
# Connector lifecycle state
# ════════════════════════════════════════════════════════════════════════

class ConnectorState(Enum):
    """Lifecycle state of an :class:`OmniverseConnector`."""

    UNINITIALIZED = "uninitialized"
    LAUNCHING = "launching"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"


_OPERATIONAL_STATES = (ConnectorState.READY, ConnectorState.DEGRADED)


# ════════════════════════════════════════════════════════════════════════
# Structural protocols for owned managers not available to inspect
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class AppLauncherProtocol(Protocol):
    """Structural interface assumed for ``app_launcher.AppLauncher``."""

    def launch(self) -> None: ...

    def shutdown(self) -> None: ...

    def is_running(self) -> bool: ...


@runtime_checkable
class ExtensionManagerProtocol(Protocol):
    """Structural interface assumed for ``extension_manager.ExtensionManager``."""

    def enable_extensions(self) -> None: ...

    def disable_extensions(self) -> None: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class StageManagerProtocol(Protocol):
    """Structural interface assumed for ``stage_manager.StageManager``."""

    def create_stage(self) -> None: ...

    def open_stage(self, path: str) -> None: ...

    def close_stage(self) -> None: ...

    def save_stage(self, path: str) -> None: ...

    def is_stage_open(self) -> bool: ...


@runtime_checkable
class AssetServerProtocol(Protocol):
    """Structural interface assumed for ``asset_server.AssetServer``."""

    def load_assets(self) -> None: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class USDLoaderProtocol(Protocol):
    """Structural interface assumed for ``usd_loader.USDLoader``."""

    def load(self, usd_path: str) -> None: ...

    def export(self, output_path: str) -> Path: ...


@runtime_checkable
class PhysicsSceneProtocol(Protocol):
    """Structural interface assumed for ``physics_scene.PhysicsScene``."""

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def reset(self) -> None: ...

    def statistics(self) -> dict[str, Any]: ...


# ════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectorStatistics:
    """Aggregated, point-in-time statistics for the whole runtime.

    Attributes:
        state: Current :class:`ConnectorState`.
        uptime_seconds: Wall-clock seconds since the most recent
            successful ``initialize()``.
        stages_loaded: Cumulative number of ``load_stage()`` calls that
            completed successfully.
        restart_count: Cumulative number of ``restart()`` calls.
        current_stage: The currently loaded USD path, or ``None``.
        timeline: The owned :class:`~.timeline_controller.TimelineStatistics`,
            or ``None`` if the timeline isn't initialized yet.
        renderer: The owned :class:`~.renderer.RenderStatistics`, or
            ``None`` if the renderer isn't initialized yet.
    """

    state: ConnectorState
    uptime_seconds: float
    stages_loaded: int
    restart_count: int
    current_stage: Optional[str]
    timeline: Optional[TimelineStatistics]
    renderer: Optional[RenderStatistics]


@dataclass
class HealthStatus:
    """Result of :meth:`OmniverseConnector.health_check`.

    Attributes:
        healthy: ``True`` iff the connector is ``READY`` and every
            reachable component reported healthy.
        state: Current :class:`ConnectorState`.
        component_status: ``{component_name: is_healthy}`` for every
            component that was checked.
        issues: Human-readable descriptions of anything unhealthy.
        checked_at: Unix timestamp the check was performed at.
    """

    healthy: bool
    state: ConnectorState
    component_status: dict[str, bool]
    issues: list[str]
    checked_at: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════════════
# OmniverseConnector
# ════════════════════════════════════════════════════════════════════════

class OmniverseConnector:
    """Facade owning and orchestrating the full Omniverse runtime.

    Thread safety: every public method acquires an internal
    :class:`threading.RLock`. Owned components are only ever
    constructed, torn down, or swapped while holding this lock.

    No component is imported or constructed until :meth:`initialize` is
    called (lazy initialization) unless it was explicitly injected via
    the constructor, which is the primary seam for unit testing this
    facade without a running Omniverse Kit process.

    Attributes:
        config: The shared :class:`~.config.OmniverseConfig` for this
            runtime, passed to every owned component that accepts one.
    """

    def __init__(
        self,
        config: Optional[OmniverseConfig] = None,
        *,
        app_launcher: Optional[AppLauncherProtocol] = None,
        extension_manager: Optional[ExtensionManagerProtocol] = None,
        stage_manager: Optional[StageManagerProtocol] = None,
        asset_server: Optional[AssetServerProtocol] = None,
        usd_loader: Optional[USDLoaderProtocol] = None,
        physics_scene: Optional[PhysicsSceneProtocol] = None,
        timeline_controller: Optional[TimelineController] = None,
        renderer: Optional[Renderer] = None,
    ) -> None:
        """Construct a connector. Performs no I/O and constructs nothing eagerly.

        Args:
            config: Shared configuration, passed through to every owned
                component. Defaults to ``OmniverseConfig.default()`` if
                omitted.
            app_launcher: Optional pre-built launcher (test fake or
                custom implementation). Built lazily via
                ``.app_launcher.AppLauncher(config=...)`` if omitted.
            extension_manager: Optional pre-built extension manager.
                Built lazily via ``.extension_manager.ExtensionManager``
                if omitted.
            stage_manager: Optional pre-built stage manager. Built
                lazily via ``.stage_manager.StageManager`` if omitted.
            asset_server: Optional pre-built asset server. Built lazily
                via ``.asset_server.AssetServer`` if omitted.
            usd_loader: Optional pre-built USD loader. Built lazily via
                ``.usd_loader.USDLoader`` if omitted.
            physics_scene: Optional pre-built physics scene. Built
                lazily via ``.physics_scene.PhysicsScene`` if omitted.
            timeline_controller: Optional pre-built
                :class:`~.timeline_controller.TimelineController`. Built
                lazily (using ``config``) if omitted.
            renderer: Optional pre-built :class:`~.renderer.Renderer`.
                Built lazily (using ``config``) if omitted.
        """
        self.config = config or OmniverseConfig.default()
        self._lock = threading.RLock()

        # Each owned manager: keep the original injected value (possibly
        # None) separately from the live instance, so restart() knows
        # exactly which components it's allowed to rebuild from scratch.
        self._injected_app_launcher = app_launcher
        self._injected_extension_manager = extension_manager
        self._injected_stage_manager = stage_manager
        self._injected_asset_server = asset_server
        self._injected_usd_loader = usd_loader
        self._injected_physics_scene = physics_scene
        self._injected_timeline = timeline_controller
        self._injected_renderer = renderer

        self._app_launcher: Optional[AppLauncherProtocol] = None
        self._extension_manager: Optional[ExtensionManagerProtocol] = None
        self._stage_manager: Optional[StageManagerProtocol] = None
        self._asset_server: Optional[AssetServerProtocol] = None
        self._usd_loader: Optional[USDLoaderProtocol] = None
        self._physics_scene: Optional[PhysicsSceneProtocol] = None
        self._timeline: Optional[TimelineController] = None
        self._renderer: Optional[Renderer] = None

        self._state = ConnectorState.UNINITIALIZED
        self._init_wall_time: Optional[float] = None
        self._stages_loaded = 0
        self._restart_count = 0
        self._current_usd_path: Optional[Path] = None

        logger.debug("OmniverseConnector constructed.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OmniverseConnector":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        with self._lock:
            return (
                f"OmniverseConnector(state={self._state.value}, "
                f"stage={self._current_usd_path}, stages_loaded={self._stages_loaded})"
            )

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _require_state(self, *allowed: ConnectorState, action: str) -> None:
        if self._state not in allowed:
            raise ConnectorStateError(
                f"Cannot {action} while in state {self._state.value}; "
                f"expected one of {[s.value for s in allowed]}."
            )

    # ------------------------------------------------------------------
    # Lazy factories -- each imports its sibling module only when called
    # ------------------------------------------------------------------

    def _build_app_launcher(self) -> AppLauncherProtocol:
        try:
            from .app_launcher import AppLauncher
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.app_launcher.AppLauncher'. Either that module "
                "isn't available yet, or its class name differs -- inject an "
                "app_launcher explicitly via the OmniverseConnector constructor "
                "in the meantime."
            ) from exc
        return AppLauncher(config=self.config)

    def _build_extension_manager(self) -> ExtensionManagerProtocol:
        try:
            from .extension_manager import ExtensionManager
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.extension_manager.ExtensionManager'. Inject "
                "an extension_manager explicitly via the constructor in the meantime."
            ) from exc
        return ExtensionManager(config=self.config)

    def _build_stage_manager(self) -> StageManagerProtocol:
        try:
            from .stage_manager import StageManager
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.stage_manager.StageManager'. Inject a "
                "stage_manager explicitly via the constructor in the meantime."
            ) from exc
        return StageManager(config=self.config)

    def _build_asset_server(self) -> AssetServerProtocol:
        try:
            from .asset_server import AssetServer
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.asset_server.AssetServer'. Inject an "
                "asset_server explicitly via the constructor in the meantime."
            ) from exc
        return AssetServer(config=self.config)

    def _build_usd_loader(self) -> USDLoaderProtocol:
        try:
            from .usd_loader import USDLoader
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.usd_loader.USDLoader'. Inject a usd_loader "
                "explicitly via the constructor in the meantime."
            ) from exc
        return USDLoader(config=self.config)

    def _build_physics_scene(self) -> PhysicsSceneProtocol:
        try:
            from .physics_scene import PhysicsScene
        except ImportError as exc:
            raise ConnectorInitializationError(
                "Could not import '.physics_scene.PhysicsScene'. Inject a "
                "physics_scene explicitly via the constructor in the meantime."
            ) from exc
        return PhysicsScene(config=self.config)

    # ------------------------------------------------------------------
    # Lifecycle: initialize / shutdown / restart
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Stand up the complete Omniverse runtime.

        Executes, in order: launch the app, enable extensions, create
        an empty stage, load asset search paths, initialize physics,
        initialize the renderer, initialize the timeline. Idempotent --
        calling this again while already ``READY``/``DEGRADED`` is a
        no-op.

        Raises:
            ConnectorInitializationError: If any stage of startup fails.
                Whatever was already brought up is left in place (not
                automatically torn down) so the caller can inspect
                :meth:`diagnostics` before deciding whether to retry or
                call :meth:`shutdown` explicitly.
        """
        with self._lock:
            if self._state in _OPERATIONAL_STATES:
                logger.debug("initialize() called in state %s; ignoring.", self._state.value)
                return

            self._state = ConnectorState.LAUNCHING
            logger.info("Initializing OmniverseConnector runtime...")

            try:
                self._app_launcher = self._injected_app_launcher or self._build_app_launcher()
                self._app_launcher.launch()
                logger.debug("AppLauncher launched.")

                self._extension_manager = self._injected_extension_manager or self._build_extension_manager()
                self._extension_manager.enable_extensions()
                logger.debug("Extensions enabled.")

                self._stage_manager = self._injected_stage_manager or self._build_stage_manager()
                self._stage_manager.create_stage()
                logger.debug("Stage created.")

                self._asset_server = self._injected_asset_server or self._build_asset_server()
                self._asset_server.load_assets()
                logger.debug("Asset search paths loaded.")

                self._physics_scene = self._injected_physics_scene or self._build_physics_scene()
                self._physics_scene.initialize()
                logger.debug("Physics scene initialized.")

                self._renderer = self._injected_renderer or Renderer(config=self.config)
                self._renderer.initialize()
                logger.debug("Renderer initialized.")

                self._timeline = self._injected_timeline or TimelineController(config=self.config)
                self._timeline.initialize()
                logger.debug("Timeline initialized.")

            except OmniverseConnectorError:
                self._state = ConnectorState.DEGRADED
                raise
            except Exception as exc:  # noqa: BLE001
                self._state = ConnectorState.DEGRADED
                raise ConnectorInitializationError(f"Runtime initialization failed: {exc}") from exc

            self._state = ConnectorState.READY
            self._init_wall_time = time.monotonic()
            logger.info("OmniverseConnector runtime is READY.")

    def shutdown(self) -> None:
        """Tear down the complete Omniverse runtime, in reverse dependency order.

        Every component's shutdown is attempted independently -- one
        component raising is logged as a warning and does not prevent
        the rest from also being torn down (graceful shutdown even
        under partial failure). Idempotent; safe to call even if
        ``initialize()`` was never called or already failed.
        """
        with self._lock:
            if self._state in (ConnectorState.UNINITIALIZED, ConnectorState.SHUTDOWN):
                self._state = ConnectorState.SHUTDOWN
                return

            logger.info("Shutting down OmniverseConnector runtime...")
            teardown_steps: list[tuple[str, Any]] = [
                ("renderer", self._renderer),
                ("timeline", self._timeline),
                ("physics_scene", self._physics_scene),
                ("asset_server", self._asset_server),
                ("stage_manager", self._stage_manager),
                ("extension_manager", self._extension_manager),
                ("app_launcher", self._app_launcher),
            ]
            for name, component in teardown_steps:
                if component is None:
                    continue
                try:
                    if name == "stage_manager":
                        component.close_stage()
                    else:
                        component.shutdown()
                except Exception:  # noqa: BLE001
                    logger.warning("Component '%s' raised during shutdown; continuing.", name)

            self._state = ConnectorState.SHUTDOWN
            logger.info("OmniverseConnector runtime shut down.")

    def restart(self) -> None:
        """Shut down and fully reinitialize the runtime.

        Every owned component that was NOT explicitly injected at
        construction time is discarded and rebuilt fresh (mirroring
        the fact that ``TimelineController``/``Renderer`` and, by the
        same contract, every other manager are single-shot: their own
        ``initialize()`` is a no-op after ``shutdown()``). Injected
        components are assumed to be caller-managed and are reused
        as-is.

        Raises:
            ConnectorInitializationError: If reinitialization fails.
        """
        with self._lock:
            logger.warning("Restarting OmniverseConnector runtime (restart #%d).", self._restart_count + 1)
            self.shutdown()

            self._app_launcher = None
            self._extension_manager = None
            self._stage_manager = None
            self._asset_server = None
            self._usd_loader = None
            self._physics_scene = None
            self._timeline = None
            self._renderer = None

            self._restart_count += 1
            self._state = ConnectorState.UNINITIALIZED
            self.initialize()

    def is_initialized(self) -> bool:
        """Return whether the runtime has been successfully initialized and not shut down."""
        with self._lock:
            return self._state in _OPERATIONAL_STATES

    def is_running(self) -> bool:
        """Return whether the simulation timeline is currently playing.

        Returns ``False`` (rather than raising) if the runtime isn't
        initialized yet or the timeline is unreachable, since this is a
        query method callers may poll opportunistically.
        """
        with self._lock:
            if self._state not in _OPERATIONAL_STATES or self._timeline is None:
                return False
            try:
                return self._timeline.is_playing()
            except Exception:  # noqa: BLE001
                return False

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def load_stage(self, path: Union[str, Path]) -> Path:
        """Load an already-compiled ``.usd*`` stage file into the running runtime.

        This connector is a pure backend runtime: it accepts only a
        path to USD that has already been produced by the upstream
        Scene Compiler / USD Exporter stages of the PhysWorldLM
        pipeline. It has no knowledge of ``WorldSpec`` or how a scene
        is compiled -- that responsibility belongs entirely outside
        this package.

        Args:
            path: Path to the ``.usd*`` file to load.

        Returns:
            The USD path that was loaded (as a :class:`Path`).

        Raises:
            ConnectorStateError: If called before ``initialize()``.
            StageLoadError: If loading fails.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="load_stage()")

            usd_path = Path(path)
            self._usd_loader = self._injected_usd_loader or self._usd_loader or self._build_usd_loader()
            try:
                self._usd_loader.load(str(usd_path))
            except Exception as exc:  # noqa: BLE001
                raise StageLoadError(f"Failed to load USD '{usd_path}': {exc}") from exc

            # Best-effort re-sync of components that may hold per-stage
            # state. Any manager without a reset() hook is simply skipped
            # rather than treated as an error.
            for component in (self._physics_scene, self._timeline):
                if component is not None and hasattr(component, "reset"):
                    try:
                        component.reset()
                    except Exception:  # noqa: BLE001
                        logger.debug("Component '%s' reset() failed after stage load; continuing.", component)

            self._current_usd_path = usd_path
            self._stages_loaded += 1
            logger.info("Loaded stage '%s' (total stages loaded: %d).", usd_path, self._stages_loaded)
            return usd_path

    def reload_stage(self) -> Path:
        """Reload the currently loaded stage from scratch.

        Returns:
            The USD path that was reloaded.

        Raises:
            StageLoadError: If no stage has been loaded yet, or the
                reload itself fails.
        """
        with self._lock:
            if self._current_usd_path is None:
                raise StageLoadError("reload_stage() called but no stage has been loaded yet.")
            return self.load_stage(self._current_usd_path)

    def unload_stage(self) -> None:
        """Unload the current stage, resetting to an empty stage.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
            StageLoadError: If clearing the stage fails.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="unload_stage()")
            if self._stage_manager is None:
                raise StageLoadError("unload_stage() called but no stage_manager is attached.")
            try:
                self._stage_manager.close_stage()
                self._stage_manager.create_stage()
            except Exception as exc:  # noqa: BLE001
                raise StageLoadError(f"Failed to unload stage: {exc}") from exc
            self._current_usd_path = None
            logger.info("Stage unloaded; reset to empty.")

    def export_usd(self, output_path: Union[str, Path]) -> Path:
        """Export the current stage's contents to a ``.usd*`` file.

        Args:
            output_path: Destination path for the exported file.

        Returns:
            The path exported to.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
            StageLoadError: If no exporting component is attached or
                export fails.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="export_usd()")
            try:
                if self._usd_loader is not None and hasattr(self._usd_loader, "export"):
                    result_path = self._usd_loader.export(str(output_path))
                elif self._stage_manager is not None:
                    self._stage_manager.save_stage(str(output_path))
                    result_path = Path(output_path)
                else:
                    raise StageLoadError("No usd_loader or stage_manager available to export the stage.")
            except StageLoadError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise StageLoadError(f"Failed to export stage to '{output_path}': {exc}") from exc
            logger.info("Exported current stage to '%s'.", result_path)
            return Path(result_path)

    def open_stage(self, path: Union[str, Path]) -> None:
        """Open an existing stage file directly.

        Functionally equivalent to :meth:`load_stage` at the
        stage-manager level; kept as a distinct method because it maps
        directly onto ``StageManager.open_stage()`` without going
        through the USD loader (useful when a stage should be opened
        for editing/inspection without re-triggering asset/physics
        re-sync).

        Args:
            path: Path to the ``.usd*`` stage file to open.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
            StageLoadError: If opening the stage fails.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="open_stage()")
            if self._stage_manager is None:
                raise StageLoadError("open_stage() called but no stage_manager is attached.")
            try:
                self._stage_manager.open_stage(str(path))
            except Exception as exc:  # noqa: BLE001
                raise StageLoadError(f"Failed to open stage '{path}': {exc}") from exc
            self._current_usd_path = Path(path)
            logger.info("Opened stage '%s'.", path)

    def close_stage(self) -> None:
        """Close the currently open stage.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
            StageLoadError: If closing the stage fails.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="close_stage()")
            if self._stage_manager is None:
                raise StageLoadError("close_stage() called but no stage_manager is attached.")
            try:
                self._stage_manager.close_stage()
            except Exception as exc:  # noqa: BLE001
                raise StageLoadError(f"Failed to close stage: {exc}") from exc
            self._current_usd_path = None
            logger.info("Stage closed.")

    def save_stage(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the current stage, defaulting to its currently loaded path.

        Args:
            path: Destination path. Defaults to the currently loaded
                USD path if omitted.

        Raises:
            StageLoadError: If no path is available and none is loaded,
                or saving fails.
        """
        target = Path(path) if path is not None else self._current_usd_path
        if target is None:
            raise StageLoadError("save_stage() requires a path (no stage is currently loaded).")
        return self.export_usd(target)

    # ------------------------------------------------------------------
    # Simulation lifecycle (delegates to TimelineController)
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Begin/resume simulation playback. See ``TimelineController.play()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="play()")
            self._timeline.play()  # type: ignore[union-attr]

    def pause(self) -> None:
        """Pause simulation playback. See ``TimelineController.pause()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="pause()")
            self._timeline.pause()  # type: ignore[union-attr]

    def resume(self) -> None:
        """Resume simulation playback after a pause. See ``TimelineController.resume()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="resume()")
            self._timeline.resume()  # type: ignore[union-attr]

    def stop(self) -> None:
        """Stop simulation playback. See ``TimelineController.stop()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="stop()")
            self._timeline.stop()  # type: ignore[union-attr]

    def reset(self) -> None:
        """Stop playback and reset simulation time to zero. See ``TimelineController.reset()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="reset()")
            self._timeline.reset()  # type: ignore[union-attr]

    def step(self, count: int = 1) -> TimelineStatistics:
        """Advance the simulation by ``count`` frame(s). See ``TimelineController.step_frames()``.

        Args:
            count: Number of frames to advance. Defaults to 1.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="step()")
            return self._timeline.step_frames(count)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Rendering lifecycle (delegates to Renderer)
    # ------------------------------------------------------------------

    def render_frame(self) -> RenderStatistics:
        """Render exactly one frame. See ``Renderer.render_frame()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="render_frame()")
            return self._renderer.render_frame()  # type: ignore[union-attr]

    def render_sequence(self, frame_count: int) -> RenderStatistics:
        """Render ``frame_count`` consecutive frames. See ``Renderer.render_sequence()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="render_sequence()")
            return self._renderer.render_sequence(frame_count)  # type: ignore[union-attr]

    def capture_image(self, path: Optional[str] = None) -> CaptureResult:
        """Capture the current frame's RGB image. See ``Renderer.capture_image()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="capture_image()")
            return self._renderer.capture_image(path)  # type: ignore[union-attr]

    def capture_video(self, path: str, num_frames: int, *, fps: Optional[float] = None) -> CaptureResult:
        """Render and capture a video clip. See ``Renderer.capture_video()``.

        Raises:
            ConnectorStateError: If called before ``initialize()``.
        """
        with self._lock:
            self._require_state(*_OPERATIONAL_STATES, action="capture_video()")
            return self._renderer.capture_video(path, num_frames, fps=fps)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Diagnostics / statistics
    # ------------------------------------------------------------------

    def statistics(self) -> ConnectorStatistics:
        """Return aggregated runtime statistics, including timeline and renderer stats."""
        with self._lock:
            uptime = time.monotonic() - self._init_wall_time if self._init_wall_time is not None else 0.0
            timeline_stats = self._timeline.timeline_statistics() if self._timeline is not None else None
            renderer_stats = self._renderer.statistics() if self._renderer is not None else None
            return ConnectorStatistics(
                state=self._state,
                uptime_seconds=uptime,
                stages_loaded=self._stages_loaded,
                restart_count=self._restart_count,
                current_stage=str(self._current_usd_path) if self._current_usd_path else None,
                timeline=timeline_stats,
                renderer=renderer_stats,
            )

    def diagnostics(self) -> dict[str, Any]:
        """Return low-level diagnostic information for the whole runtime.

        Aggregates connector-level state with each owned component's
        own ``diagnostics()`` (when available), so a single call gives
        a full picture during debugging.
        """
        with self._lock:
            def _safe_diagnostics(component: Any) -> Any:
                if component is None or not hasattr(component, "diagnostics"):
                    return None
                try:
                    return component.diagnostics()
                except Exception:  # noqa: BLE001
                    return None

            return {
                "state": self._state.value,
                "current_stage": str(self._current_usd_path) if self._current_usd_path else None,
                "stages_loaded": self._stages_loaded,
                "restart_count": self._restart_count,
                "components_attached": {
                    "app_launcher": self._app_launcher is not None,
                    "extension_manager": self._extension_manager is not None,
                    "stage_manager": self._stage_manager is not None,
                    "asset_server": self._asset_server is not None,
                    "usd_loader": self._usd_loader is not None,
                    "physics_scene": self._physics_scene is not None,
                    "timeline": self._timeline is not None,
                    "renderer": self._renderer is not None,
                },
                "timeline": _safe_diagnostics(self._timeline),
                "renderer": _safe_diagnostics(self._renderer),
                "physics_scene": _safe_diagnostics(self._physics_scene),
            }

    def health_check(self) -> HealthStatus:
        """Probe every reachable component and report an aggregated health status.

        Never raises for an individual component failing its probe --
        each probe is wrapped defensively and contributes to
        ``component_status``/``issues`` instead. Use
        :class:`ConnectorHealthError` yourself if your caller wants a
        raise-on-unhealthy policy: ``if not connector.health_check().healthy: raise ...``.
        """
        with self._lock:
            component_status: dict[str, bool] = {}
            issues: list[str] = []

            def _probe(name: str, fn: Optional[Any]) -> None:
                if fn is None:
                    component_status[name] = False
                    issues.append(f"{name} is not attached.")
                    return
                try:
                    component_status[name] = bool(fn())
                    if not component_status[name]:
                        issues.append(f"{name} reported an unhealthy status.")
                except Exception as exc:  # noqa: BLE001
                    component_status[name] = False
                    issues.append(f"{name} health probe raised: {exc}")

            _probe("app_launcher", self._app_launcher.is_running if self._app_launcher else None)
            _probe("stage_manager", self._stage_manager.is_stage_open if self._stage_manager else None)
            _probe("timeline", (lambda: self._timeline is not None) if self._timeline is not None else None)
            _probe("renderer", (lambda: self._renderer is not None) if self._renderer is not None else None)

            if self._state not in _OPERATIONAL_STATES:
                issues.append(f"Connector state is '{self._state.value}', not READY.")

            healthy = self._state == ConnectorState.READY and not issues
            return HealthStatus(
                healthy=healthy, state=self._state, component_status=component_status, issues=issues,
            )

    def performance_report(self) -> dict[str, Any]:
        """Return a report focused on frame-timing/throughput performance."""
        with self._lock:
            renderer_stats = self._renderer.statistics() if self._renderer is not None else None
            timeline_stats = self._timeline.timeline_statistics() if self._timeline is not None else None
            return {
                "renderer_fps": renderer_stats.current_fps if renderer_stats else None,
                "average_frame_time_ms": renderer_stats.average_frame_time_ms if renderer_stats else None,
                "frames_rendered": renderer_stats.frames_rendered if renderer_stats else None,
                "simulation_time": timeline_stats.current_time if timeline_stats else None,
                "simulation_frame": timeline_stats.current_frame if timeline_stats else None,
                "real_time_elapsed": timeline_stats.real_time_elapsed if timeline_stats else None,
            }

    def memory_report(self) -> dict[str, Any]:
        """Return a best-effort process memory usage report.

        Uses ``psutil`` if available; returns an empty dict (with a
        debug log) if it is not installed, rather than failing.
        """
        with self._lock:
            try:
                import psutil  # type: ignore
            except ImportError:
                logger.debug("psutil not installed; memory_report() returning empty dict.")
                return {}
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_bytes": mem_info.rss,
                "vms_bytes": mem_info.vms,
                "percent": process.memory_percent(),
            }

    def gpu_report(self) -> dict[str, Any]:
        """Return a best-effort GPU utilization/memory report.

        Prefers the renderer's own backend-reported performance stats
        (see ``Renderer.diagnostics()["backend_performance"]``); falls
        back to an empty dict if the renderer isn't attached or reports
        nothing.
        """
        with self._lock:
            if self._renderer is None:
                return {}
            try:
                diag = self._renderer.diagnostics()
            except Exception:  # noqa: BLE001
                return {}
            return {
                "gpu_devices": diag.get("gpu_devices", []),
                "backend_performance": diag.get("backend_performance", {}),
            }

    def __deepcopy__(self, memo: dict[int, Any]) -> "OmniverseConnector":
        """Deep-copying a live connector is not supported.

        A connector holds a lock and live handles to every owned
        component; use :meth:`statistics` / :meth:`diagnostics` to
        inspect state, and construct a new ``OmniverseConnector`` with
        the desired configuration instead.
        """
        raise OmniverseConnectorError(
            "OmniverseConnector cannot be deep-copied; construct a new "
            "OmniverseConnector with the desired configuration instead."
        )


__all__ = [
    "OmniverseConnector",
    "ConnectorState",
    "ConnectorStatistics",
    "HealthStatus",
    "AppLauncherProtocol",
    "ExtensionManagerProtocol",
    "StageManagerProtocol",
    "AssetServerProtocol",
    "USDLoaderProtocol",
    "PhysicsSceneProtocol",
    "OmniverseConnectorError",
    "ConnectorStateError",
    "ConnectorInitializationError",
    "ConnectorHealthError",
    "StageLoadError",
]
