"""
timeline.py
══════════════════════════════════════════════════════════════════════════
Simulation clock for the PhysWorldLM Runtime Engine.

Scope
-----
`Timeline` owns exactly one responsibility: turning wall-clock progress
(or, in headless/benchmark mode, raw step requests) into a sequence of
deterministic simulation-time advances. It knows nothing about entities,
components, sensors, or backends -- `SimulationController` asks it
"how much sim time should the next step cover?" and reports back "that
step consumed N seconds of sim time"; nothing else.

This module has no simulator-specific imports and no dependency on any
other runtime module -- it is usable standalone (e.g. by a unit test
that never touches `EntityManager` or a `BackendAdapter`).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from runtime.runtime_context import RuntimeEventType, TimelineError


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class TimeMode(Enum):
    """Stepping discipline a `Timeline` operates under.

    FIXED: every `advance()` call reports a whole number of
        `fixed_timestep_s`-sized substeps (accumulator pattern), giving
        deterministic, backend-agnostic physics stepping.
    VARIABLE: every `advance()` call reports exactly one substep sized
        to however much wall-clock (or synthetic) time actually elapsed.
        Useful for pure visualization/telemetry timelines that don't
        drive physics.
    """

    FIXED = auto()
    VARIABLE = auto()


class TimelineState(Enum):
    """Lifecycle state of a `Timeline`, kept independent of
    `runtime_context.RuntimePhase` so a `Timeline` can be unit-tested
    without a full `SimulationController`.
    """

    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


# ════════════════════════════════════════════════════════════════════════
# Clock snapshot
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClockSnapshot:
    """Immutable point-in-time read of a `Timeline`, safe to hand to
    sensors, checkpoints, or telemetry without holding the timeline's lock.
    """

    sim_time_s: float
    frame_count: int
    time_scale: float
    mode: TimeMode
    state: TimelineState

    def to_dict(self) -> dict:
        return {
            "sim_time_s": round(self.sim_time_s, 6),
            "frame_count": self.frame_count,
            "time_scale": self.time_scale,
            "mode": self.mode.name,
            "state": self.state.name,
        }


@dataclass(frozen=True)
class AdvanceResult:
    """What a single `Timeline.advance()` call produced.

    Attributes:
        substep_count: Number of `fixed_timestep_s`-sized substeps the
            caller should step the backend through (>= 0). Always `1`
            for `TimeMode.VARIABLE`.
        substep_duration_s: Duration of each substep, in sim seconds.
            Equal to `fixed_timestep_s` for `TimeMode.FIXED`; equal to
            the single variable substep's duration otherwise.
        dropped_substeps: How many substeps beyond `max_substeps` were
            discarded this call (spiral-of-death protection).
        sim_time_s: Sim time *after* this advance.
        frame_count: Frame count *after* this advance.
    """

    substep_count: int
    substep_duration_s: float
    dropped_substeps: int
    sim_time_s: float
    frame_count: int


# ════════════════════════════════════════════════════════════════════════
# Timeline
# ════════════════════════════════════════════════════════════════════════

class Timeline:
    """Deterministic simulation clock.

    Thread-safe; every public method acquires an internal `RLock`. A
    `Timeline` never sleeps or blocks on its own -- pacing against
    wall-clock time (`real_time=True`) is implemented by comparing
    `time.monotonic()` deltas against `time_scale`, never by calling
    `time.sleep()`, so `SimulationController` remains free to interleave
    other work (event dispatch, sensor capture) between `advance()` calls.

    Example:
        >>> tl = Timeline(fixed_timestep_s=1 / 60, mode=TimeMode.FIXED)
        >>> tl.start()
        >>> result = tl.advance(elapsed_wall_s=1 / 30)   # two 60Hz substeps
        >>> result.substep_count
        2
    """

    def __init__(
        self,
        fixed_timestep_s: float = 1.0 / 60.0,
        *,
        mode: TimeMode = TimeMode.FIXED,
        max_substeps: int = 8,
        time_scale: float = 1.0,
        event_bus: Optional["EventBus"] = None,  # noqa: F821 - see TYPE_CHECKING note below
    ) -> None:
        """Initialize the timeline.

        Args:
            fixed_timestep_s: Substep size used in `TimeMode.FIXED`.
            mode: Stepping discipline; see `TimeMode`.
            max_substeps: Upper bound on substeps a single `advance()`
                call will report before dropping the remainder.
            time_scale: Multiplier applied to wall-clock elapsed time
                before it is converted into sim-time advancement.
            event_bus: Optional `world_pipeline.EventBus`. If provided,
                `start()`/`pause()`/`resume()`/`stop()`/`seek()` publish
                `RuntimeEventType` diagnostics; entirely optional so a
                bare `Timeline` remains usable in isolation.

        Raises:
            TimelineError: If `fixed_timestep_s <= 0` or `max_substeps < 1`.
        """
        if fixed_timestep_s <= 0:
            raise TimelineError("fixed_timestep_s must be > 0.")
        if max_substeps < 1:
            raise TimelineError("max_substeps must be >= 1.")

        self._lock = threading.RLock()
        self._fixed_timestep_s = fixed_timestep_s
        self._mode = mode
        self._max_substeps = max_substeps
        self._time_scale = time_scale
        self._event_bus = event_bus

        self._state = TimelineState.STOPPED
        self._sim_time_s = 0.0
        self._frame_count = 0
        self._accumulator_s = 0.0
        self._last_wall_time: Optional[float] = None

    # ── properties ──────────────────────────────────────────────────

    @property
    def mode(self) -> TimeMode:
        with self._lock:
            return self._mode

    @property
    def state(self) -> TimelineState:
        with self._lock:
            return self._state

    @property
    def fixed_timestep_s(self) -> float:
        return self._fixed_timestep_s

    @property
    def time_scale(self) -> float:
        with self._lock:
            return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        if value < 0:
            raise TimelineError("time_scale must be >= 0.")
        with self._lock:
            self._time_scale = value

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin (or restart) the timeline from its current sim time."""
        with self._lock:
            self._state = TimelineState.RUNNING
            self._last_wall_time = time.monotonic()
            self._publish(RuntimeEventType.PLAY_STARTED)

    def pause(self) -> None:
        """Pause the timeline; `advance()` becomes a no-op until `resume()`."""
        with self._lock:
            if self._state is TimelineState.RUNNING:
                self._state = TimelineState.PAUSED
                self._publish(RuntimeEventType.PAUSED)

    def resume(self) -> None:
        """Resume a paused timeline without discontinuity in `sim_time_s`."""
        with self._lock:
            if self._state is TimelineState.PAUSED:
                self._state = TimelineState.RUNNING
                self._last_wall_time = time.monotonic()
                self._publish(RuntimeEventType.RESUMED)

    def stop(self) -> None:
        """Stop the timeline. `sim_time_s`/`frame_count` are preserved;
        call `reset()` to zero them.
        """
        with self._lock:
            self._state = TimelineState.STOPPED
            self._last_wall_time = None
            self._publish(RuntimeEventType.STOPPED)

    def reset(self) -> None:
        """Stop the timeline and zero `sim_time_s`, `frame_count`, and
        the fixed-step accumulator.
        """
        with self._lock:
            self._state = TimelineState.STOPPED
            self._sim_time_s = 0.0
            self._frame_count = 0
            self._accumulator_s = 0.0
            self._last_wall_time = None
            self._publish(RuntimeEventType.RESET)

    def seek(self, sim_time_s: float) -> None:
        """Jump `sim_time_s` to an arbitrary value (e.g. replay/scrubbing).

        Clears the fixed-step accumulator so the next `advance()` starts
        cleanly rather than immediately reporting leftover substeps.

        Raises:
            TimelineError: If `sim_time_s < 0`.
        """
        if sim_time_s < 0:
            raise TimelineError("sim_time_s must be >= 0.")
        with self._lock:
            self._sim_time_s = sim_time_s
            self._accumulator_s = 0.0
            self._publish(RuntimeEventType.RESET, sim_time_s=sim_time_s, seek=True)

    # ── stepping ──────────────────────────────────────────────────────

    def advance(self, elapsed_wall_s: Optional[float] = None) -> AdvanceResult:
        """Advance the timeline by one wall-clock tick and report how
        many backend substeps that tick represents.

        Args:
            elapsed_wall_s: Wall-clock seconds elapsed since the previous
                `advance()`. If `None`, computed from `time.monotonic()`
                deltas since the last call (typical interactive use). For
                headless/benchmark stepping, pass a fixed synthetic value
                (e.g. `fixed_timestep_s`) so runs are wall-clock independent.

        Returns:
            An `AdvanceResult` describing substeps to execute this call.
            If the timeline is not `RUNNING`, returns a zero-substep
            result without mutating state.
        """
        with self._lock:
            if self._state is not TimelineState.RUNNING:
                return AdvanceResult(0, self._fixed_timestep_s, 0, self._sim_time_s, self._frame_count)

            now = time.monotonic()
            if elapsed_wall_s is None:
                last = self._last_wall_time if self._last_wall_time is not None else now
                elapsed_wall_s = max(0.0, now - last)
            self._last_wall_time = now

            scaled_elapsed = elapsed_wall_s * self._time_scale

            if self._mode is TimeMode.VARIABLE:
                self._sim_time_s += scaled_elapsed
                self._frame_count += 1
                return AdvanceResult(1, scaled_elapsed, 0, self._sim_time_s, self._frame_count)

            # TimeMode.FIXED -- accumulator pattern.
            self._accumulator_s += scaled_elapsed
            substeps = 0
            while self._accumulator_s >= self._fixed_timestep_s and substeps < self._max_substeps:
                self._accumulator_s -= self._fixed_timestep_s
                self._sim_time_s += self._fixed_timestep_s
                self._frame_count += 1
                substeps += 1

            dropped = 0
            if self._accumulator_s >= self._fixed_timestep_s:
                # Spiral-of-death protection: too much time accumulated
                # to catch up this call. Drop the remainder rather than
                # stepping unboundedly.
                dropped = int(self._accumulator_s // self._fixed_timestep_s)
                self._accumulator_s = 0.0

            return AdvanceResult(substeps, self._fixed_timestep_s, dropped, self._sim_time_s, self._frame_count)

    def step_once(self) -> AdvanceResult:
        """Force exactly one fixed-size substep regardless of wall-clock
        time or accumulator state. Used by `SimulationController.step()`
        for single-stepping / deterministic replay.
        """
        with self._lock:
            self._sim_time_s += self._fixed_timestep_s
            self._frame_count += 1
            self._publish(RuntimeEventType.STEP_COMPLETED, frame=self._frame_count)
            return AdvanceResult(1, self._fixed_timestep_s, 0, self._sim_time_s, self._frame_count)

    # ── introspection ─────────────────────────────────────────────────

    def snapshot(self) -> ClockSnapshot:
        """Return an immutable point-in-time read of this timeline."""
        with self._lock:
            return ClockSnapshot(
                sim_time_s=self._sim_time_s,
                frame_count=self._frame_count,
                time_scale=self._time_scale,
                mode=self._mode,
                state=self._state,
            )

    def to_dict(self) -> dict:
        return self.snapshot().to_dict()

    # ── internal ──────────────────────────────────────────────────────

    def _publish(self, event_type: RuntimeEventType, **payload) -> None:
        if self._event_bus is None:
            return
        payload = {**payload, "runtime_event": event_type.name}
        from world_pipeline import EventType as _PipelineEventType  # local import avoids a hard module-load cycle
        self._event_bus.publish(_PipelineEventType.DIAGNOSTIC, **payload)


__all__ = [
    "TimeMode",
    "TimelineState",
    "ClockSnapshot",
    "AdvanceResult",
    "Timeline",
]
