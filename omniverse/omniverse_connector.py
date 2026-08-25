"""
omniverse_connector.py
══════════════════════════════════════════════════════════════════════════
The single public bridge between PhysWorldLM and NVIDIA Omniverse Kit.

Pipeline position
------------------
    WorldSpec → SceneCompiler → scene.usda
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │    OmniverseConnector    │  <-- this module
                        └────────────────────────┘
                                     │
                                     ▼
                     Omniverse Kit (external OS process)

Scope
-----
This connector treats Omniverse Kit as an **external process**, launched
via ``subprocess.Popen`` and pointed at a ``.usd*`` file on disk. It does
NOT run inside a Kit Python interpreter, does NOT import ``carb`` or
``omni.*``, and does NOT talk to any Kit Service/RPC endpoint. That is a
deliberate scope cut: PhysWorldLM's job ends at producing a valid USD
stage (see ``scene_compiler.py``); handing that stage to Kit and keeping
exactly one Kit process alive is *all* this module does.

Because there is no in-process hook into Kit, "reloading" a stage means
terminating the current Kit process and relaunching it against the new
(or updated) file. This is simpler and far more robust than depending on
Kit's live-sync/USD-notice machinery from outside the Kit process, and it
matches the actual requirement: at most one Kit instance, always showing
the latest compiled stage.

Extensions
----------
Kit needs to know where PhysWorldLM's own extensions/app live before it
can find them, typically via one or more ``--ext-folder <path>`` CLI
arguments. Rather than burying that in ``extra_kit_args``, it's a
first-class ``ext_folders`` constructor parameter -- pass the paths, the
connector builds the flags.

Output capture
--------------
Kit's stdout/stderr are captured to a log file per launch (default
``~/.cache/physworldlm/logs/kit_<launch_count>.log``) rather than
discarded, so a failed or misbehaving launch is debuggable. Pass
``capture_output=False`` to inherit the parent process's stdout/stderr
directly instead (useful when running the connector interactively).

Public API
----------
    connector = OmniverseConnector()
    connector.initialize()          # locate Kit, do not launch yet
    connector.show_stage(usd_path)  # launch Kit (or reload it) with usd_path
    ...
    connector.is_running()
    connector.reload_stage(new_usd_path)
    connector.shutdown()

Or, as a context manager::

    with OmniverseConnector() as connector:
        connector.show_stage("scene.usda")
        ...

Single-instance guarantee
--------------------------
Within one connector object, launching while already ``RUNNING`` raises
rather than silently spawning a second Kit process. Across independent
processes (e.g. multiple FastAPI workers), an optional filesystem lock
(``lock_path``) makes the same guarantee machine-wide; pass
``lock_path=None`` to disable it for tests or single-worker deployments.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

from .kit_locator import KitDiscoveryError, KitInstallation, KitLocator

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
    """Base class for all connector errors."""


class ConnectorStateError(OmniverseConnectorError):
    """Raised when an operation is invalid for the connector's current state."""


class KitNotFoundError(OmniverseConnectorError):
    """Raised when no Kit installation could be located."""


class KitLaunchError(OmniverseConnectorError):
    """Raised when the Kit subprocess fails to start or dies immediately."""


class KitAlreadyRunningError(OmniverseConnectorError):
    """Raised by launch() when a Kit instance is already running (this connector, or another process)."""


class StageLoadError(OmniverseConnectorError):
    """Raised when a stage path is missing or otherwise cannot be shown."""


# ════════════════════════════════════════════════════════════════════════
# State
# ════════════════════════════════════════════════════════════════════════

class ConnectorState(Enum):
    """Lifecycle state of an :class:`OmniverseConnector`."""

    UNINITIALIZED = "uninitialized"  # Kit not yet located
    READY = "ready"                  # Kit located, not launched
    LAUNCHING = "launching"          # subprocess.Popen called, startup grace period in progress
    RUNNING = "running"              # Kit process alive
    STOPPED = "stopped"              # Kit was running, now terminated (by us or by exiting on its own)
    FAILED = "failed"                # discovery or launch failed


@dataclass(frozen=True)
class ConnectorStatistics:
    """Point-in-time snapshot of connector state, for health/debug endpoints."""

    state: ConnectorState
    pid: Optional[int]
    current_stage: Optional[str]
    kit_executable: Optional[str]
    kit_version: Optional[str]
    launch_count: int
    uptime_seconds: float
    log_file: Optional[str]


# ════════════════════════════════════════════════════════════════════════
# Cross-process single-instance guard
# ════════════════════════════════════════════════════════════════════════

class _SingleInstanceGuard:
    """Best-effort, PID-file-based lock ensuring only one Kit process is launched machine-wide.

    Not a substitute for the in-process state check (which is authoritative
    for a single connector instance) -- this exists purely to catch the
    case of two separate PhysWorldLM processes (e.g. two FastAPI workers)
    each independently deciding to launch Kit.
    """

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path

    def acquire(self, pid: int) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            holder_pid = self._read_pid()
            if holder_pid == pid:
                return  # we already hold it (e.g. reload_stage() re-launching)
            if holder_pid is not None and self._pid_alive(holder_pid):
                raise KitAlreadyRunningError(
                    f"Another process already holds the Omniverse Kit lock "
                    f"(pid={holder_pid}, lock file={self.lock_path}). Only one "
                    "Kit instance is allowed at a time."
                )
            logger.warning("Removing stale Kit lock file at '%s'.", self.lock_path)
            self.lock_path.unlink(missing_ok=True)
        self.lock_path.write_text(str(pid), encoding="utf-8")

    def release(self) -> None:
        self.lock_path.unlink(missing_ok=True)

    def _read_pid(self) -> Optional[int]:
        try:
            return int(self.lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, owned by someone else
        return True


_DEFAULT_LOCK_PATH = Path.home() / ".cache" / "physworldlm" / "omniverse_kit.lock"
_DEFAULT_LOG_DIR = Path.home() / ".cache" / "physworldlm" / "logs"


# ════════════════════════════════════════════════════════════════════════
# OmniverseConnector
# ════════════════════════════════════════════════════════════════════════

class OmniverseConnector:
    """Launches, tracks, and feeds USD stages into a single external Kit process.

    Thread safety: every public method acquires an internal
    :class:`threading.RLock`, so one connector instance is safe to share
    across threads within one process (e.g. FastAPI's threadpool).

    Attributes:
        kit_executable: Explicit path override for the Kit executable, or
            ``None`` to auto-discover via :class:`KitLocator`.
    """

    def __init__(
        self,
        kit_executable: Optional[Union[str, Path]] = None,
        *,
        ext_folders: Optional[Sequence[Union[str, Path]]] = None,
        extra_kit_args: Optional[Sequence[str]] = None,
        launch_grace_period_s: float = 30.0,
        poll_interval_s: float = 0.5,
        shutdown_timeout_s: float = 10.0,
        lock_path: Optional[Union[str, Path]] = _DEFAULT_LOCK_PATH,
        capture_output: bool = True,
        log_dir: Optional[Union[str, Path]] = _DEFAULT_LOG_DIR,
        locator: Optional[KitLocator] = None,
    ) -> None:
        """Construct a connector. Performs no filesystem search or process I/O.

        Args:
            kit_executable: Explicit path to a Kit executable or its
                containing directory. Overrides auto-discovery.
            ext_folders: Directories passed to Kit as ``--ext-folder``
                arguments (one flag pair per entry), so it can find
                PhysWorldLM's own extensions/app. Order is preserved.
            extra_kit_args: Any further raw CLI arguments, appended after
                the ``--ext-folder`` flags on every launch.
            launch_grace_period_s: How long to watch the freshly spawned
                process for an immediate crash (missing library, bad
                arguments, etc.) before declaring the launch successful.
                This is NOT a wait for Kit's full UI startup -- there is
                no in-process signal available for that without
                ``omni.*`` hooks. Some Kit builds take 20-40s to reach a
                usable UI on shared/DGX hardware; the default here only
                needs to outlast the *crash-on-startup* window, not full
                boot, but is set generously to avoid false negatives.
            poll_interval_s: Polling interval used while waiting out the
                grace period and while checking liveness.
            shutdown_timeout_s: How long to wait for a graceful SIGTERM
                exit before escalating to SIGKILL.
            lock_path: Filesystem path used for the cross-process
                single-instance guard. Pass ``None`` to disable it.
            capture_output: If ``True`` (default), Kit's stdout/stderr
                are redirected to a per-launch file under ``log_dir``
                instead of being discarded, so failures are debuggable.
                If ``False``, Kit inherits this process's stdout/stderr
                directly.
            log_dir: Directory for per-launch log files when
                ``capture_output`` is ``True``.
            locator: Optional pre-built :class:`KitLocator` (mainly for
                tests); a default one is created otherwise.
        """
        self._explicit_kit_executable = Path(kit_executable) if kit_executable else None
        self._ext_folders: tuple[Path, ...] = tuple(Path(p) for p in (ext_folders or ()))
        self._extra_kit_args = tuple(extra_kit_args or ())
        self._launch_grace_period_s = launch_grace_period_s
        self._poll_interval_s = poll_interval_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._locator = locator or KitLocator()
        self._guard = _SingleInstanceGuard(Path(lock_path)) if lock_path is not None else None
        self._capture_output = capture_output
        self._log_dir = Path(log_dir) if log_dir is not None else None

        self._lock = threading.RLock()
        self._installation: Optional[KitInstallation] = None
        self._process: Optional[subprocess.Popen] = None
        self._log_file_handle = None
        self._current_log_path: Optional[Path] = None
        self._state = ConnectorState.UNINITIALIZED
        self._current_stage: Optional[Path] = None
        self._launch_count = 0
        self._launched_at: Optional[float] = None

        logger.debug("OmniverseConnector constructed.")

    # ── context manager ─────────────────────────────────────────────

    def __enter__(self) -> "OmniverseConnector":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.shutdown()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        with self._lock:
            return (
                f"OmniverseConnector(state={self._state.value}, "
                f"stage={self._current_stage}, pid={self._process.pid if self._process else None})"
            )

    # ── lifecycle ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Locate a Kit installation. Does not launch a process.

        Idempotent: calling this again after a successful ``initialize()``
        is a no-op.

        Raises:
            KitNotFoundError: If no Kit installation can be found.
        """
        with self._lock:
            if self._state is not ConnectorState.UNINITIALIZED:
                return
            try:
                self._installation = self._locator.locate(self._explicit_kit_executable)
            except KitDiscoveryError as exc:
                self._state = ConnectorState.FAILED
                raise KitNotFoundError(str(exc)) from exc
            self._state = ConnectorState.READY
            logger.info(
                "OmniverseConnector initialized (kit=%s, version=%s).",
                self._installation.executable, self._installation.version,
            )

    def launch(self, stage_path: Optional[Union[str, Path]] = None) -> None:
        """Spawn the single Kit process, optionally opening a stage immediately.

        Args:
            stage_path: A ``.usd*`` file to pass to Kit on the command
                line. May be omitted to launch Kit with no stage.

        Raises:
            ConnectorStateError: If called before ``initialize()``, or
                while already ``LAUNCHING``.
            KitAlreadyRunningError: If this connector (or, when the
                filesystem lock is enabled, another process) already
                has a Kit instance running.
            StageLoadError: If ``stage_path`` was given but does not
                exist.
            KitLaunchError: If the process fails to start or exits
                immediately.
        """
        with self._lock:
            if self._state is ConnectorState.RUNNING:
                raise KitAlreadyRunningError("Kit is already running; call reload_stage() or shutdown() first.")
            if self._state not in (ConnectorState.READY, ConnectorState.STOPPED):
                raise ConnectorStateError(f"Cannot launch() from state '{self._state.value}'; call initialize() first.")
            assert self._installation is not None

            resolved_stage: Optional[Path] = None
            if stage_path is not None:
                resolved_stage = Path(stage_path).resolve()
                if not resolved_stage.exists():
                    raise StageLoadError(f"Stage file does not exist: {resolved_stage}")

            if self._guard is not None:
                self._guard.acquire(pid=os.getpid())

            #args = [str(self._installation.executable)]

            #app_config = self._installation.app_config
            #if app_config is not None:
                #args.append(str(app_config))

            #for folder in self._ext_folders:
                #args.extend(["--ext-folder", str(folder)])

            #args.extend(self._extra_kit_args)

            #if resolved_stage is not None:
                #args.append(f"--/app/content/usdFile={resolved_stage}")

            args = [str(self._installation.executable)]

            app_config = self._installation.app_config
            if app_config is not None:
                args.append(str(app_config))

            for folder in self._ext_folders:
                args.extend(["--ext-folder", str(folder)])

            args.extend(self._extra_kit_args)

            if resolved_stage is not None:
                args.append(f"--/app/content/usdFile={resolved_stage}")

            stdout_target, stderr_target, log_path = self._open_output_targets()

            self._state = ConnectorState.LAUNCHING
            logger.info("Launching Omniverse Kit: %s", " ".join(args))
            if log_path is not None:
                logger.info("Kit stdout/stderr will be captured to '%s'.", log_path)
            try:
                self._process = subprocess.Popen(
                    args,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    start_new_session=True,
                )
            except OSError as exc:
                self._state = ConnectorState.FAILED
                if self._guard is not None:
                    self._guard.release()
                self._close_output_targets()
                raise KitLaunchError(f"Failed to launch Kit executable '{args[0]}': {exc}") from exc

            if not self._survive_grace_period():
                exit_code = self._process.returncode
                self._process = None
                self._state = ConnectorState.FAILED
                if self._guard is not None:
                    self._guard.release()
                self._close_output_targets()
                log_hint = f" See '{log_path}' for Kit's output." if log_path is not None else ""
                raise KitLaunchError(
                    f"Kit process exited during the {self._launch_grace_period_s}s startup grace "
                    f"period (exit code={exit_code}).{log_hint}"
                )

            self._current_stage = resolved_stage
            self._current_log_path = log_path
            self._launch_count += 1
            self._launched_at = time.monotonic()
            self._state = ConnectorState.RUNNING
            logger.info("Kit is running (pid=%d, stage=%s).", self._process.pid, resolved_stage)

    def _open_output_targets(self):
        """Return (stdout_target, stderr_target, log_path) for the next Popen call."""
        if not self._capture_output:
            return None, None, None
        if self._log_dir is None:
            return subprocess.DEVNULL, subprocess.DEVNULL, None
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"kit_launch_{self._launch_count + 1}_{int(time.time())}.log"
        handle = open(log_path, "wb")  # noqa: SIM115 - lifetime tied to the subprocess, closed explicitly
        self._log_file_handle = handle
        return handle, subprocess.STDOUT, log_path

    def _close_output_targets(self) -> None:
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            self._log_file_handle = None

    def _survive_grace_period(self) -> bool:
        """Poll the freshly spawned process; return False if it exits within the grace period."""
        assert self._process is not None
        deadline = time.monotonic() + self._launch_grace_period_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            time.sleep(self._poll_interval_s)
        return self._process.poll() is None

    def is_running(self) -> bool:
        """Return whether the Kit process is currently alive.

        Also reconciles internal state to ``STOPPED`` if the process was
        previously ``RUNNING`` but has since exited on its own (crash,
        user closed the window, etc.).
        """
        with self._lock:
            if self._process is None:
                return False
            alive = self._process.poll() is None
            if not alive and self._state is ConnectorState.RUNNING:
                logger.warning("Kit process (pid=%s) exited unexpectedly.", self._process.pid)
                self._state = ConnectorState.STOPPED
                self._process = None
                self._close_output_targets()
                if self._guard is not None:
                    self._guard.release()
            return alive

    def show_stage(self, path: Union[str, Path]) -> None:
        """Display `path` in Kit: launches Kit if it isn't running, else reloads it.

        This is the one method most PhysWorldLM callers need: hand it the
        ``.usda`` file just produced by ``SceneCompiler.compile()`` and it
        does the right thing whether or not Kit is already up.

        Args:
            path: The ``.usd*`` stage file to show.

        Raises:
            StageLoadError: If ``path`` does not exist.
            KitNotFoundError: If ``initialize()`` has not been called and
                Kit cannot be auto-discovered.
            KitLaunchError: If (re)launching Kit fails.
        """
        with self._lock:
            resolved = Path(path).resolve()
            if not resolved.exists():
                raise StageLoadError(f"Stage file does not exist: {resolved}")

            if self._state is ConnectorState.UNINITIALIZED:
                self.initialize()

            if self.is_running():
                self.reload_stage(resolved)
            else:
                self.launch(stage_path=resolved)

    def reload_stage(self, path: Optional[Union[str, Path]] = None) -> None:
        """Terminate and relaunch Kit against a (possibly updated) stage file.

        There is no in-process live-reload hook available in this design
        (see module docstring), so "reload" is implemented as a clean
        restart of the single Kit process -- this is what keeps the
        "exactly one Kit instance" guarantee simple and robust.

        Args:
            path: New stage path. Defaults to the currently shown stage
                (useful when the same file was overwritten in place by a
                new ``SceneCompiler.compile()`` run).

        Raises:
            StageLoadError: If no path is given and none has been shown
                yet, or the resolved path does not exist.
            KitLaunchError: If relaunching Kit fails.
        """
        with self._lock:
            target = Path(path).resolve() if path is not None else self._current_stage
            if target is None:
                raise StageLoadError("reload_stage() requires a path; no stage has been shown yet.")
            if not target.exists():
                raise StageLoadError(f"Stage file does not exist: {target}")

            logger.info("Reloading stage '%s' (relaunching the single Kit instance).", target)
            if self.is_running():
                self._terminate_process()
            self._state = ConnectorState.STOPPED
            self.launch(stage_path=target)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Terminate the Kit process, if any. Idempotent and safe to call at any time.

        Args:
            timeout: Override the configured ``shutdown_timeout_s`` for
                this call only.
        """
        with self._lock:
            if self._process is not None:
                self._terminate_process(timeout=timeout)
            if self._guard is not None:
                self._guard.release()
            if self._state is not ConnectorState.UNINITIALIZED:
                self._state = ConnectorState.STOPPED
            logger.info("OmniverseConnector shut down.")

    def _terminate_process(self, timeout: Optional[float] = None) -> None:
        assert self._process is not None
        effective_timeout = timeout if timeout is not None else self._shutdown_timeout_s
        pid = self._process.pid
        if self._process.poll() is None:
            self._signal_process_group(pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Kit (pid=%d) did not exit within %.1fs; sending SIGKILL.", pid, effective_timeout)
                self._signal_process_group(pid, signal.SIGKILL)
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.error("Kit (pid=%d) did not respond to SIGKILL.", pid)
        self._process = None
        self._close_output_targets()

    @staticmethod
    def _signal_process_group(pid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            pass

    # ── diagnostics ──────────────────────────────────────────────────

    def current_stage(self) -> Optional[Path]:
        """Return the currently shown stage path, or ``None`` if none has been shown yet."""
        with self._lock:
            return self._current_stage

    def statistics(self) -> ConnectorStatistics:
        """Return a point-in-time snapshot of connector state."""
        with self._lock:
            uptime = time.monotonic() - self._launched_at if self._launched_at is not None else 0.0
            return ConnectorStatistics(
                state=self._state,
                pid=self._process.pid if self._process else None,
                current_stage=str(self._current_stage) if self._current_stage else None,
                kit_executable=str(self._installation.executable) if self._installation else None,
                kit_version=self._installation.version if self._installation else None,
                launch_count=self._launch_count,
                uptime_seconds=uptime,
                log_file=str(self._current_log_path) if self._current_log_path else None,
            )


__all__ = [
    "OmniverseConnector",
    "ConnectorState",
    "ConnectorStatistics",
    "OmniverseConnectorError",
    "ConnectorStateError",
    "KitNotFoundError",
    "KitLaunchError",
    "KitAlreadyRunningError",
    "StageLoadError",
]
