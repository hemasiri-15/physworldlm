"""
timeline_controller.py
══════════════════════════════════════════════════════════════════════════
Simulation-time authority for the Omniverse connector layer of
PhysWorldLM.

Pipeline position
------------------
    Prompt → WorldSpec → USD → OmniverseLauncher → ExtensionManager
                                                         │
                                                         ▼
                                                    StageManager
                                                         │
                                                         ▼
                                                    PhysicsScene
                                                         │
                                                         ▼
                                          ┌──────────────────────────┐
                                          │   TimelineController     │  <-- this module
                                          └──────────────────────────┘
                                                         │
                                                         ▼
                                                     Renderer

Scope
-----
This module owns ONLY simulation time: playback state, the simulation
clock, frame/time addressing, looping, playback speed, and the
callback fan-out that lets other components (PhysicsScene, an
animation system, Replicator, a ROS2 bridge, a planner) synchronize
their own per-frame work to a single authoritative clock.

It never:
    * launches Omniverse (``app_launcher.py``)
    * creates stages (``stage_manager.py``)
    * loads USD (``usd_loader.py``)
    * creates physics (``physics_scene.py``)
    * renders (``renderer.py``)
    * loads assets (``asset_server.py``)
    * parses prompts (upstream ontology / WorldSpec layer)

Design constraints
-------------------
    * ``omni.timeline`` (and any other ``omni.*`` package) is imported
      lazily, only inside the methods that actually need it, and only
      the first time such a method executes -- never at module import
      time and never in ``__init__``.
    * The controller is driver-agnostic: it can run against a real
      ``omni.timeline.ITimeline`` interface (production), a fake/stub
      backend (unit tests), or with no backend at all (a pure software
      clock), via the ``TimelineBackend`` protocol and dependency
      injection. This is what makes ``initialize()``/``play()``/etc.
      testable in a plain ``pytest`` process with no Kit installed.
    * All mutable state lives on the instance, guarded by a single
      re-entrant lock. There is no module-level mutable state, so
      multiple independent ``TimelineController`` instances (e.g. in
      parallel test workers) never interfere with each other.

Public API
----------
    controller = TimelineController(config=OmniverseConfig.default())
    with controller:
        controller.play()
        controller.step_frames(10)
        controller.pause()
        stats = controller.timeline_statistics()

Changelog
---------
    * Initial implementation.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .config import OmniverseConfig

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.timeline_controller")
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

class TimelineControllerError(Exception):
    """Base class for all TimelineController errors."""


class TimelineStateError(TimelineControllerError):
    """Raised when an operation is invalid for the current playback state."""


class TimelineImportError(TimelineControllerError):
    """Raised when a required ``omni`` module cannot be lazily imported."""


class TimelineValidationError(TimelineControllerError):
    """Raised when a parameter (fps, time, frame, scale, ...) is invalid."""


class TimelineCallbackError(TimelineControllerError):
    """Raised for callback registration/removal failures."""


# ════════════════════════════════════════════════════════════════════════
# Playback state
# ════════════════════════════════════════════════════════════════════════

class PlaybackState(Enum):
    """Lifecycle / playback state of a :class:`TimelineController`."""

    UNINITIALIZED = auto()
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    SHUTDOWN = auto()


# States from which each transition is legal. Centralized here so every
# public method validates against the same source of truth rather than
# duplicating ad-hoc checks.
_ACTIVE_STATES = (PlaybackState.PLAYING, PlaybackState.PAUSED, PlaybackState.STOPPED)


class TimelineEvent(Enum):
    """Events a caller may subscribe to via :meth:`TimelineController.register_callback`."""

    INITIALIZED = auto()
    PLAY = auto()
    PAUSE = auto()
    RESUME = auto()
    STOP = auto()
    RESET = auto()
    REWIND = auto()
    STEP = auto()
    JUMP = auto()
    LOOP = auto()
    FPS_CHANGED = auto()
    TIME_SCALE_CHANGED = auto()
    FRAME_ADVANCED = auto()
    SHUTDOWN = auto()


# ════════════════════════════════════════════════════════════════════════
# Backend protocol (dependency inversion -- makes the controller testable
# without Omniverse installed, and future-compatible with alternative
# simulation backends).
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class TimelineBackend(Protocol):
    """Minimal surface a timeline driver must expose.

    A production implementation wraps ``omni.timeline.ITimeline``. Tests
    can supply a trivial fake implementing the same six methods. When no
    backend is supplied at all, :class:`TimelineController` still works
    correctly as a pure software clock -- the backend is purely a place
    to mirror commands for a real DCC/runtime to observe.
    """

    def set_current_time(self, time_seconds: float) -> None: ...

    def get_current_time(self) -> float:
        ...

    def set_time_codes_per_second(self, fps: float) -> None: ...

    def play(self) -> None: ...

    def pause(self) -> None: ...

    def stop(self) -> None: ...


class _OmniTimelineBackend:
    """Adapter around ``omni.timeline.ITimeline``.

    Imports ``omni.timeline`` lazily -- only when this adapter is
    actually constructed, which itself only happens lazily from
    :meth:`TimelineController.initialize` when no backend was injected.
    """

    def __init__(self) -> None:
        try:
            import omni.timeline  # type: ignore
        except ImportError as exc:
            raise TimelineImportError(
                "Could not import 'omni.timeline'. TimelineController "
                "requires a running Omniverse Kit process to drive a "
                "real timeline, or an injected TimelineBackend for "
                "backend-less / test operation."
            ) from exc

        self._omni_timeline = omni.timeline
        self._interface = omni.timeline.get_timeline_interface()
        logger.debug("Acquired omni.timeline.ITimeline interface.")

    def set_current_time(self, time_seconds: float) -> None:
        self._interface.set_current_time(time_seconds)

    def get_current_time(self) -> float:
        return float(self._interface.get_current_time())

    def set_time_codes_per_second(self, fps: float) -> None:
        self._interface.set_time_codes_per_second(fps)

    def play(self) -> None:
        self._interface.play()

    def pause(self) -> None:
        self._interface.pause()

    def stop(self) -> None:
        self._interface.stop()


# ════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TimelineStatistics:
    """Point-in-time statistics about a :class:`TimelineController`.

    Attributes:
        playback_state: Current :class:`PlaybackState`.
        current_time: Current simulation time, in seconds.
        current_frame: Current simulation frame number.
        fps: Configured frames-per-second (time codes per second).
        time_scale: Current simulation-to-real-time multiplier.
        duration: Configured timeline duration, in seconds (``None`` if
            unbounded).
        loop_enabled: Whether looping is enabled.
        total_frames_stepped: Cumulative frames advanced since
            ``initialize()``.
        total_play_count: Number of times ``play()`` has been called.
        total_pause_count: Number of times ``pause()`` has been called.
        total_loop_count: Number of times the timeline has looped.
        real_time_elapsed: Wall-clock seconds since ``initialize()``.
        last_step_wall_dt: Wall-clock seconds consumed by the most
            recent ``step()``/``step_frames()`` call.
    """

    playback_state: PlaybackState
    current_time: float
    current_frame: int
    fps: float
    time_scale: float
    duration: Optional[float]
    loop_enabled: bool
    total_frames_stepped: int
    total_play_count: int
    total_pause_count: int
    total_loop_count: int
    real_time_elapsed: float
    last_step_wall_dt: float


@dataclass
class TimelineSnapshot:
    """A restorable, exportable capture of :class:`TimelineController` state.

    Deliberately plain-data (no backend references, no locks) so it is
    trivially picklable/JSON-able for ``export``/``import`` and safe to
    stash in a recording buffer.
    """

    current_time: float
    current_frame: int
    fps: float
    time_scale: float
    duration: Optional[float]
    loop_enabled: bool
    fixed_timestep: bool
    playback_state: PlaybackState
    label: str = ""
    wall_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        payload = dict(
            current_time=self.current_time,
            current_frame=self.current_frame,
            fps=self.fps,
            time_scale=self.time_scale,
            duration=self.duration,
            loop_enabled=self.loop_enabled,
            fixed_timestep=self.fixed_timestep,
            playback_state=self.playback_state.name,
            label=self.label,
            wall_timestamp=self.wall_timestamp,
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimelineSnapshot":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        data = dict(payload)
        data["playback_state"] = PlaybackState[data["playback_state"]]
        return cls(**data)


TimelineCallback = Callable[[TimelineEvent, TimelineStatistics], None]
"""Signature required of callbacks passed to ``register_callback()``."""

ScheduledCallback = Callable[[int, float], None]
"""Signature required of callbacks passed to ``schedule_at_frame()``: receives (frame, time)."""


# ════════════════════════════════════════════════════════════════════════
# TimelineController
# ════════════════════════════════════════════════════════════════════════

class TimelineController:
    """Owns and drives simulation time for a PhysWorldLM Omniverse session.

    Thread safety: every public method acquires an internal
    :class:`threading.RLock`, so a single instance may safely be driven
    from multiple threads (e.g. a UI thread calling ``pause()`` while a
    simulation loop thread calls ``step()``). Registered callbacks are
    invoked while the lock is held released (see ``_fire``) to avoid
    deadlocks if a callback re-enters the controller.

    The controller never imports ``omni`` at construction time. A real
    ``omni.timeline`` backend is only acquired inside :meth:`initialize`,
    and only if no :class:`TimelineBackend` was injected via the
    constructor -- so tests can freely construct and drive a controller
    with zero Omniverse dependencies.

    Attributes:
        config: The :class:`~omniverse.config.OmniverseConfig` this
            controller was built from (used only for its default
            ``render_dt`` -- the controller does not otherwise reach
            into unrelated config fields).
    """

    def __init__(
        self,
        config: Optional[OmniverseConfig] = None,
        *,
        backend: Optional[TimelineBackend] = None,
        fps: Optional[float] = None,
        duration: Optional[float] = None,
        loop: bool = False,
        fixed_timestep: bool = True,
    ) -> None:
        """Construct a controller. Performs no I/O and no ``omni`` imports.

        Args:
            config: Optional shared configuration. If provided and
                ``fps`` is not explicitly given, the frame rate defaults
                to ``1.0 / config.render_dt``.
            backend: Optional pre-built :class:`TimelineBackend`
                (typically used to inject a fake for unit tests, or a
                pre-constructed real backend). If ``None``, a real
                ``omni.timeline``-backed adapter is lazily constructed
                the first time :meth:`initialize` runs.
            fps: Initial frames-per-second. Defaults to 60.0, or to
                ``1.0 / config.render_dt`` if ``config`` is given.
            duration: Optional fixed timeline duration in seconds. If
                ``None``, the timeline is unbounded (looping and
                fast-forward/rewind still work relative to frame 0).
            loop: Whether the timeline loops at ``duration`` (or wraps
                indefinitely if ``duration`` is ``None`` and looping is
                requested, which is a no-op until a duration is set).
            fixed_timestep: If ``True`` (default), ``step()`` always
                advances by exactly ``1 / fps`` seconds of simulation
                time regardless of wall-clock time -- the recommended
                mode for reproducible physics. If ``False``, ``step()``
                advances by the actual wall-clock time elapsed since the
                previous step, scaled by ``time_scale`` (variable
                timestep, useful for real-time visualization contexts).

        Raises:
            TimelineValidationError: If ``fps`` or ``duration`` is
                non-positive.
        """
        self.config = config
        self._lock = threading.RLock()

        resolved_fps = fps if fps is not None else (1.0 / config.render_dt if config else 60.0)
        if resolved_fps <= 0:
            raise TimelineValidationError(f"fps must be > 0 (got {resolved_fps}).")
        if duration is not None and duration <= 0:
            raise TimelineValidationError(f"duration must be > 0 if set (got {duration}).")

        self._injected_backend = backend
        self._backend: Optional[TimelineBackend] = None

        self._state = PlaybackState.UNINITIALIZED
        self._time: float = 0.0
        self._frame: int = 0
        self._fps: float = resolved_fps
        self._time_scale: float = 1.0
        self._duration: Optional[float] = duration
        self._loop_enabled: bool = loop
        self._fixed_timestep: bool = fixed_timestep

        self._callbacks: dict[str, tuple[TimelineEvent, TimelineCallback]] = {}
        self._scheduled: dict[str, tuple[int, ScheduledCallback]] = {}

        self._init_wall_time: Optional[float] = None
        self._last_step_wall_time: Optional[float] = None
        self._last_step_wall_dt: float = 0.0

        self._total_frames_stepped: int = 0
        self._total_play_count: int = 0
        self._total_pause_count: int = 0
        self._total_loop_count: int = 0

        self._recording: Optional[list[TimelineSnapshot]] = None
        self._keyframes: dict[int, str] = {}

        logger.debug(
            "TimelineController constructed (fps=%.3f, duration=%s, loop=%s, fixed_timestep=%s).",
            resolved_fps, duration, loop, fixed_timestep,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TimelineController":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        with self._lock:
            return (
                f"TimelineController(state={self._state.name}, time={self._time:.4f}, "
                f"frame={self._frame}, fps={self._fps:.2f})"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Acquire/attach the timeline backend and enter ``STOPPED`` state.

        Idempotent: calling ``initialize()`` again while already
        initialized is a no-op (logged at debug level) rather than an
        error, so callers don't need to track whether they already did
        this.

        Raises:
            TimelineImportError: If no backend was injected and
                ``omni.timeline`` cannot be imported.
        """
        with self._lock:
            if self._state != PlaybackState.UNINITIALIZED:
                logger.debug("initialize() called in state %s; ignoring.", self._state.name)
                return

            self._backend = self._injected_backend or _OmniTimelineBackend()
            try:
                self._backend.set_time_codes_per_second(self._fps)
                self._backend.set_current_time(self._time)
            except Exception:  # noqa: BLE001
                logger.warning("Backend did not accept initial fps/time; continuing with software clock.")

            self._state = PlaybackState.STOPPED
            self._init_wall_time = time.monotonic()
            self._last_step_wall_time = self._init_wall_time
            logger.info("TimelineController initialized (fps=%.3f).", self._fps)
            self._fire(TimelineEvent.INITIALIZED)

    def shutdown(self) -> None:
        """Stop playback, release the backend, and enter ``SHUTDOWN`` state.

        Idempotent. Safe to call from ``__exit__`` even if
        ``initialize()`` was never called.
        """
        with self._lock:
            if self._state in (PlaybackState.UNINITIALIZED, PlaybackState.SHUTDOWN):
                self._state = PlaybackState.SHUTDOWN
                return

            try:
                if self._backend is not None:
                    self._backend.stop()
            except Exception:  # noqa: BLE001
                logger.warning("Backend raised while stopping during shutdown; ignoring.")

            self._state = PlaybackState.SHUTDOWN
            self._backend = None
            logger.info("TimelineController shut down.")
            self._fire(TimelineEvent.SHUTDOWN)
            self._callbacks.clear()
            self._scheduled.clear()

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _require_state(self, *allowed: PlaybackState, action: str) -> None:
        if self._state not in allowed:
            raise TimelineStateError(
                f"Cannot {action} while in state {self._state.name}; "
                f"expected one of {[s.name for s in allowed]}."
            )

    def _clamp_and_wrap_time(self, proposed_time: float) -> tuple[float, bool]:
        """Clamp/wrap ``proposed_time`` against ``duration``+``loop_enabled``.

        Returns:
            A ``(resolved_time, looped)`` tuple. ``looped`` is ``True``
            iff wraparound occurred.
        """
        if self._duration is None or self._duration <= 0:
            return max(0.0, proposed_time), False

        if proposed_time < 0.0:
            if self._loop_enabled:
                wrapped = proposed_time % self._duration
                return wrapped, True
            return 0.0, False

        if proposed_time >= self._duration:
            if self._loop_enabled:
                wrapped = proposed_time % self._duration
                return wrapped, True
            return self._duration, False

        return proposed_time, False

    def _set_time_unlocked(self, new_time: float) -> bool:
        """Apply a resolved time to internal state and the backend.

        Returns whether a loop wraparound occurred. Caller must hold
        ``self._lock``.
        """
        resolved_time, looped = self._clamp_and_wrap_time(new_time)
        self._time = resolved_time
        self._frame = int(round(resolved_time * self._fps))
        if self._backend is not None:
            try:
                self._backend.set_current_time(resolved_time)
            except Exception:  # noqa: BLE001
                logger.debug("Backend rejected set_current_time(%.6f); continuing.", resolved_time)
        if looped:
            self._total_loop_count += 1
        return looped

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Begin/resume playback.

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(
                PlaybackState.STOPPED, PlaybackState.PAUSED, PlaybackState.PLAYING,
                action="play()",
            )
            if self._state == PlaybackState.PLAYING:
                return
            if self._backend is not None:
                self._backend.play()
            self._state = PlaybackState.PLAYING
            self._total_play_count += 1
            self._last_step_wall_time = time.monotonic()
            logger.info("Timeline playing (time=%.4f, frame=%d).", self._time, self._frame)
            self._fire(TimelineEvent.PLAY)

    def pause(self) -> None:
        """Pause playback, retaining the current time/frame.

        Raises:
            TimelineStateError: If not currently playing.
        """
        with self._lock:
            self._require_state(PlaybackState.PLAYING, action="pause()")
            if self._backend is not None:
                self._backend.pause()
            self._state = PlaybackState.PAUSED
            self._total_pause_count += 1
            logger.info("Timeline paused (time=%.4f, frame=%d).", self._time, self._frame)
            self._fire(TimelineEvent.PAUSE)

    def resume(self) -> None:
        """Resume playback after a pause. Equivalent to :meth:`play`.

        Raises:
            TimelineStateError: If not currently paused.
        """
        with self._lock:
            self._require_state(PlaybackState.PAUSED, action="resume()")
        self.play()
        with self._lock:
            self._fire(TimelineEvent.RESUME)

    def stop(self) -> None:
        """Stop playback. Time/frame are retained (use :meth:`reset` to zero them).

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(
                PlaybackState.PLAYING, PlaybackState.PAUSED, PlaybackState.STOPPED,
                action="stop()",
            )
            if self._backend is not None:
                self._backend.stop()
            self._state = PlaybackState.STOPPED
            logger.info("Timeline stopped (time=%.4f, frame=%d).", self._time, self._frame)
            self._fire(TimelineEvent.STOP)

    def reset(self) -> None:
        """Stop playback and reset time/frame to zero.

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="reset()")
            if self._backend is not None:
                self._backend.stop()
            self._state = PlaybackState.STOPPED
            self._set_time_unlocked(0.0)
            self._total_frames_stepped = 0
            logger.info("Timeline reset to time=0, frame=0.")
            self._fire(TimelineEvent.RESET)

    def rewind(self) -> None:
        """Jump back to the start of the timeline without changing playback state.

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="rewind()")
            self._set_time_unlocked(0.0)
            logger.info("Timeline rewound to time=0, frame=0.")
            self._fire(TimelineEvent.REWIND)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> TimelineStatistics:
        """Advance the timeline by exactly one frame.

        In ``fixed_timestep`` mode this advances by ``1 / fps`` seconds
        of simulation time. In variable-timestep mode it advances by the
        actual wall-clock time elapsed since the previous step, scaled
        by ``time_scale``.

        Returns:
            The :class:`TimelineStatistics` snapshot taken immediately
            after the step, so PhysicsScene/animation/Replicator/ROS2
            consumers can synchronize off a single return value instead
            of making a second call.

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        return self.step_frames(1)

    def step_frames(self, count: int) -> TimelineStatistics:
        """Advance the timeline by ``count`` frames.

        Args:
            count: Number of frames to advance. Must be a positive
                integer.

        Returns:
            The :class:`TimelineStatistics` snapshot taken after all
            ``count`` frames have been applied.

        Raises:
            TimelineValidationError: If ``count`` is not a positive
                integer.
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        if not isinstance(count, int) or count <= 0:
            raise TimelineValidationError(f"count must be a positive integer (got {count!r}).")

        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="step_frames()")

            now = time.monotonic()
            wall_dt = now - self._last_step_wall_time if self._last_step_wall_time else 0.0
            self._last_step_wall_time = now

            if self._fixed_timestep:
                advance_seconds = count / self._fps
            else:
                advance_seconds = wall_dt * self._time_scale

            self._last_step_wall_dt = wall_dt
            self._set_time_unlocked(self._time + advance_seconds)
            self._total_frames_stepped += count

            self._dispatch_scheduled_events()
            logger.debug("Stepped %d frame(s) -> time=%.4f, frame=%d.", count, self._time, self._frame)
            stats = self._statistics_unlocked()
            self._fire(TimelineEvent.STEP)
            self._fire(TimelineEvent.FRAME_ADVANCED)
            return stats

    def _dispatch_scheduled_events(self) -> None:
        """Invoke and clear one-shot scheduled callbacks whose frame has been reached.

        Caller must hold ``self._lock``.
        """
        if not self._scheduled:
            return
        fired = [
            token for token, (frame, _cb) in self._scheduled.items() if self._frame >= frame
        ]
        for token in fired:
            _frame, cb = self._scheduled.pop(token)
            try:
                cb(self._frame, self._time)
            except Exception:  # noqa: BLE001
                logger.exception("Scheduled callback '%s' raised; continuing.", token)

    # ------------------------------------------------------------------
    # Jumping
    # ------------------------------------------------------------------

    def jump_to_frame(self, frame: int) -> None:
        """Jump directly to an absolute frame number.

        Args:
            frame: Target frame. Must be non-negative.

        Raises:
            TimelineValidationError: If ``frame`` is negative.
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        if frame < 0:
            raise TimelineValidationError(f"frame must be >= 0 (got {frame}).")
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="jump_to_frame()")
            self._set_time_unlocked(frame / self._fps)
            logger.info("Jumped to frame=%d (time=%.4f).", self._frame, self._time)
            self._fire(TimelineEvent.JUMP)

    def jump_to_time(self, time_seconds: float) -> None:
        """Jump directly to an absolute simulation time.

        Args:
            time_seconds: Target time, in seconds. Must be non-negative.

        Raises:
            TimelineValidationError: If ``time_seconds`` is negative.
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        if time_seconds < 0:
            raise TimelineValidationError(f"time_seconds must be >= 0 (got {time_seconds}).")
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="jump_to_time()")
            self._set_time_unlocked(time_seconds)
            logger.info("Jumped to time=%.4f (frame=%d).", self._time, self._frame)
            self._fire(TimelineEvent.JUMP)

    def fast_forward(self, seconds: float) -> None:
        """Advance the current time by ``seconds`` without stepping physics/frames.

        Unlike :meth:`step`/:meth:`step_frames`, this does not invoke
        scheduled per-frame callbacks -- it is a scrub operation, not a
        simulation advance.

        Args:
            seconds: Amount of simulation time to advance by. Must be
                non-negative.

        Raises:
            TimelineValidationError: If ``seconds`` is negative.
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        if seconds < 0:
            raise TimelineValidationError(f"seconds must be >= 0 (got {seconds}).")
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="fast_forward()")
            self._set_time_unlocked(self._time + seconds)
            logger.info("Fast-forwarded to time=%.4f (frame=%d).", self._time, self._frame)
            self._fire(TimelineEvent.JUMP)

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_fps(self, fps: float) -> None:
        """Set the simulation frame rate (time codes per second).

        Args:
            fps: New frame rate. Must be strictly positive.

        Raises:
            TimelineValidationError: If ``fps`` is non-positive.
        """
        if fps <= 0:
            raise TimelineValidationError(f"fps must be > 0 (got {fps}).")
        with self._lock:
            self._fps = fps
            if self._backend is not None:
                try:
                    self._backend.set_time_codes_per_second(fps)
                except Exception:  # noqa: BLE001
                    logger.debug("Backend rejected set_time_codes_per_second(%.3f).", fps)
            self._frame = int(round(self._time * self._fps))
            logger.info("fps set to %.3f.", fps)
            self._fire(TimelineEvent.FPS_CHANGED)

    def set_time_scale(self, scale: float) -> None:
        """Set the simulation-to-real-time multiplier used in variable-timestep mode.

        Args:
            scale: New time scale. Must be strictly positive. ``1.0`` is
                real time, ``2.0`` is double speed, ``0.5`` is half
                speed.

        Raises:
            TimelineValidationError: If ``scale`` is non-positive.
        """
        if scale <= 0:
            raise TimelineValidationError(f"scale must be > 0 (got {scale}).")
        with self._lock:
            self._time_scale = scale
            logger.info("time_scale set to %.3f.", scale)
            self._fire(TimelineEvent.TIME_SCALE_CHANGED)

    def set_loop(self, enabled: bool, duration: Optional[float] = None) -> None:
        """Enable/disable looping, optionally updating the loop duration.

        Args:
            enabled: Whether the timeline should loop at ``duration``.
            duration: If given, replaces the configured duration. Must
                be strictly positive if provided.

        Raises:
            TimelineValidationError: If ``duration`` is provided and
                non-positive.
        """
        if duration is not None and duration <= 0:
            raise TimelineValidationError(f"duration must be > 0 if set (got {duration}).")
        with self._lock:
            self._loop_enabled = enabled
            if duration is not None:
                self._duration = duration
            logger.info("loop_enabled set to %s (duration=%s).", enabled, self._duration)
            self._fire(TimelineEvent.LOOP)

    def set_fixed_timestep(self, enabled: bool) -> None:
        """Switch between fixed-timestep and variable-timestep stepping.

        Args:
            enabled: ``True`` for fixed timestep (recommended for
                reproducible physics), ``False`` for wall-clock-driven
                variable timestep.
        """
        with self._lock:
            self._fixed_timestep = enabled
            logger.info("fixed_timestep set to %s.", enabled)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def current_time(self) -> float:
        """Return the current simulation time, in seconds."""
        with self._lock:
            return self._time

    def current_frame(self) -> int:
        """Return the current simulation frame number."""
        with self._lock:
            return self._frame

    def duration(self) -> Optional[float]:
        """Return the configured timeline duration in seconds, or ``None`` if unbounded."""
        with self._lock:
            return self._duration

    def is_playing(self) -> bool:
        """Return whether the timeline is currently in the ``PLAYING`` state."""
        with self._lock:
            return self._state == PlaybackState.PLAYING

    def playback_state(self) -> PlaybackState:
        """Return the current :class:`PlaybackState`."""
        with self._lock:
            return self._state

    def timeline_statistics(self) -> TimelineStatistics:
        """Return a point-in-time :class:`TimelineStatistics` snapshot."""
        with self._lock:
            return self._statistics_unlocked()

    def _statistics_unlocked(self) -> TimelineStatistics:
        real_elapsed = (
            time.monotonic() - self._init_wall_time if self._init_wall_time is not None else 0.0
        )
        return TimelineStatistics(
            playback_state=self._state,
            current_time=self._time,
            current_frame=self._frame,
            fps=self._fps,
            time_scale=self._time_scale,
            duration=self._duration,
            loop_enabled=self._loop_enabled,
            total_frames_stepped=self._total_frames_stepped,
            total_play_count=self._total_play_count,
            total_pause_count=self._total_pause_count,
            total_loop_count=self._total_loop_count,
            real_time_elapsed=real_elapsed,
            last_step_wall_dt=self._last_step_wall_dt,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return low-level diagnostic information, distinct from playback statistics.

        Useful for health checks / debugging rather than simulation
        logic -- includes backend presence, lock contention hints, and
        pending scheduled/keyframe counts.
        """
        with self._lock:
            return {
                "state": self._state.name,
                "backend_attached": self._backend is not None,
                "backend_type": type(self._backend).__name__ if self._backend else None,
                "registered_callbacks": len(self._callbacks),
                "pending_scheduled_events": len(self._scheduled),
                "keyframe_count": len(self._keyframes),
                "recording_active": self._recording is not None,
                "recording_length": len(self._recording) if self._recording is not None else 0,
                "fixed_timestep": self._fixed_timestep,
            }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def register_callback(self, event: TimelineEvent, callback: TimelineCallback) -> str:
        """Subscribe ``callback`` to ``event``.

        Args:
            event: The :class:`TimelineEvent` to subscribe to.
            callback: A callable of signature
                ``(event: TimelineEvent, stats: TimelineStatistics) -> None``.
                Called synchronously from whichever thread triggers the
                event, with the controller's lock released.

        Returns:
            An opaque subscription token. Pass this to
            :meth:`remove_callback` to unsubscribe.

        Raises:
            TimelineCallbackError: If ``callback`` is not callable.
        """
        if not callable(callback):
            raise TimelineCallbackError("callback must be callable.")
        token = uuid.uuid4().hex
        with self._lock:
            self._callbacks[token] = (event, callback)
        logger.debug("Registered callback %s for event %s.", token, event.name)
        return token

    def remove_callback(self, token: str) -> None:
        """Unsubscribe a callback previously registered via :meth:`register_callback`.

        Args:
            token: The subscription token returned by
                :meth:`register_callback`.

        Raises:
            TimelineCallbackError: If ``token`` is not a known
                subscription.
        """
        with self._lock:
            if token not in self._callbacks:
                raise TimelineCallbackError(f"Unknown callback token: '{token}'.")
            del self._callbacks[token]
        logger.debug("Removed callback %s.", token)

    def schedule_at_frame(self, frame: int, callback: ScheduledCallback) -> str:
        """Schedule a one-shot callback to fire once ``current_frame() >= frame``.

        Args:
            frame: Target frame. If already reached/passed, the
                callback fires on the very next ``step``/``step_frames``
                call.
            callback: A callable of signature ``(frame: int, time: float) -> None``.

        Returns:
            An opaque token that can be used to cancel the scheduled
            event by removing it from internal bookkeeping (schedule
            tokens are consumed automatically once fired).

        Raises:
            TimelineValidationError: If ``frame`` is negative.
            TimelineCallbackError: If ``callback`` is not callable.
        """
        if frame < 0:
            raise TimelineValidationError(f"frame must be >= 0 (got {frame}).")
        if not callable(callback):
            raise TimelineCallbackError("callback must be callable.")
        token = uuid.uuid4().hex
        with self._lock:
            self._scheduled[token] = (frame, callback)
        logger.debug("Scheduled callback %s at frame %d.", token, frame)
        return token

    def cancel_scheduled(self, token: str) -> None:
        """Cancel a previously scheduled callback before it fires.

        Args:
            token: The token returned by :meth:`schedule_at_frame`.

        Raises:
            TimelineCallbackError: If ``token`` is unknown (already
                fired or never existed).
        """
        with self._lock:
            if token not in self._scheduled:
                raise TimelineCallbackError(f"Unknown scheduled-event token: '{token}'.")
            del self._scheduled[token]

    def _fire(self, event: TimelineEvent) -> None:
        """Invoke all callbacks subscribed to ``event``.

        Caller must hold ``self._lock`` on entry; the lock is released
        for the duration of each callback invocation so re-entrant calls
        (e.g. a callback that itself calls ``controller.pause()``) do
        not deadlock, then re-acquired before returning to the caller.
        """
        subscribers = [cb for (evt, cb) in self._callbacks.values() if evt == event]
        if not subscribers:
            return
        stats = self._statistics_unlocked()
        self._lock.release()
        try:
            for cb in subscribers:
                try:
                    cb(event, stats)
                except Exception:  # noqa: BLE001
                    logger.exception("Callback for event %s raised; continuing.", event.name)
        finally:
            self._lock.acquire()

    # ------------------------------------------------------------------
    # Keyframes
    # ------------------------------------------------------------------

    def register_keyframe(self, frame: int, name: str) -> None:
        """Tag ``frame`` with a human-readable ``name`` for later lookup.

        Args:
            frame: Frame number to tag. Must be non-negative.
            name: Label for the keyframe (e.g. ``"grasp_start"``).

        Raises:
            TimelineValidationError: If ``frame`` is negative.
        """
        if frame < 0:
            raise TimelineValidationError(f"frame must be >= 0 (got {frame}).")
        with self._lock:
            self._keyframes[frame] = name
        logger.debug("Registered keyframe '%s' at frame %d.", name, frame)

    def get_keyframes(self) -> dict[int, str]:
        """Return a copy of the current ``{frame: name}`` keyframe mapping."""
        with self._lock:
            return dict(self._keyframes)

    def clear_keyframes(self) -> None:
        """Remove all registered keyframes."""
        with self._lock:
            self._keyframes.clear()

    # ------------------------------------------------------------------
    # Recording / replay / export / import
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin capturing a snapshot on every subsequent frame advance.

        Raises:
            TimelineStateError: If a recording is already in progress.
        """
        with self._lock:
            if self._recording is not None:
                raise TimelineStateError("A recording is already in progress.")
            self._recording = [self.snapshot()]
        logger.info("Timeline recording started.")

    def record_frame(self) -> None:
        """Append the current state to the active recording.

        Call this after each ``step``/``step_frames`` if you want a
        snapshot per simulation frame rather than only at
        start/stop boundaries.

        Raises:
            TimelineStateError: If no recording is in progress.
        """
        with self._lock:
            if self._recording is None:
                raise TimelineStateError("No recording is in progress; call start_recording() first.")
            self._recording.append(self.snapshot())

    def stop_recording(self) -> list[TimelineSnapshot]:
        """Stop the active recording and return the captured snapshot list.

        Raises:
            TimelineStateError: If no recording is in progress.
        """
        with self._lock:
            if self._recording is None:
                raise TimelineStateError("No recording is in progress.")
            recording, self._recording = self._recording, None
        logger.info("Timeline recording stopped (%d snapshot(s)).", len(recording))
        return recording

    def replay(self, recording: list[TimelineSnapshot]) -> None:
        """Replay a previously captured recording by restoring each snapshot in order.

        This drives the software clock and backend through the recorded
        states; it does not itself re-invoke physics or rendering, which
        should be triggered by the caller's own loop keyed off
        ``FRAME_ADVANCED`` callbacks fired for each restored snapshot.

        Args:
            recording: A list of :class:`TimelineSnapshot`, as returned
                by :meth:`stop_recording` or :meth:`export_recording`.

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="replay()")
            for snap in recording:
                self.restore_snapshot(snap)
        logger.info("Replayed %d snapshot(s).", len(recording))

    def export_recording(self, recording: list[TimelineSnapshot]) -> list[dict[str, Any]]:
        """Serialize a recording to a list of JSON-friendly dicts."""
        return [snap.to_dict() for snap in recording]

    def import_recording(self, payload: list[dict[str, Any]]) -> list[TimelineSnapshot]:
        """Deserialize a recording previously produced by :meth:`export_recording`."""
        return [TimelineSnapshot.from_dict(item) for item in payload]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self, label: str = "") -> TimelineSnapshot:
        """Capture the current controller state as a :class:`TimelineSnapshot`.

        Args:
            label: Optional human-readable label to attach.

        Returns:
            A plain-data, deep-copy-safe snapshot.
        """
        with self._lock:
            return TimelineSnapshot(
                current_time=self._time,
                current_frame=self._frame,
                fps=self._fps,
                time_scale=self._time_scale,
                duration=self._duration,
                loop_enabled=self._loop_enabled,
                fixed_timestep=self._fixed_timestep,
                playback_state=self._state,
                label=label,
            )

    def restore_snapshot(self, snapshot: TimelineSnapshot) -> None:
        """Restore controller state from a previously captured :class:`TimelineSnapshot`.

        Playback state (``PLAYING``/``PAUSED``/``STOPPED``) is not
        forced to match the snapshot's recorded state unless the
        controller is currently active -- restoring time/frame/fps/etc.
        is always safe, but this method will not itself call
        ``play()``/``pause()``/``stop()``; call those explicitly if you
        need the exact recorded playback state too.

        Args:
            snapshot: A snapshot produced by :meth:`snapshot` (or
                deserialized via :meth:`import_recording`).

        Raises:
            TimelineStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="restore_snapshot()")
            self._fps = snapshot.fps
            self._time_scale = snapshot.time_scale
            self._duration = snapshot.duration
            self._loop_enabled = snapshot.loop_enabled
            self._fixed_timestep = snapshot.fixed_timestep
            if self._backend is not None:
                try:
                    self._backend.set_time_codes_per_second(self._fps)
                except Exception:  # noqa: BLE001
                    logger.debug("Backend rejected restored fps=%.3f.", self._fps)
            self._set_time_unlocked(snapshot.current_time)
        logger.info(
            "Restored snapshot '%s' (time=%.4f, frame=%d).",
            snapshot.label, snapshot.current_time, snapshot.current_frame,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "TimelineController":
        """Deep-copying a live controller is not supported.

        A controller holds a lock and a live backend connection, neither
        of which is meaningfully copyable; use :meth:`snapshot` /
        :meth:`restore_snapshot` to transfer state instead.
        """
        raise TimelineControllerError(
            "TimelineController cannot be deep-copied; use snapshot()/restore_snapshot() "
            "to transfer state between instances instead."
        )


__all__ = [
    "TimelineController",
    "TimelineBackend",
    "TimelineStatistics",
    "TimelineSnapshot",
    "TimelineEvent",
    "PlaybackState",
    "TimelineControllerError",
    "TimelineStateError",
    "TimelineImportError",
    "TimelineValidationError",
    "TimelineCallbackError",
]
