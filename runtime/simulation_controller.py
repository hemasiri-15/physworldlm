"""
simulation_controller.py
══════════════════════════════════════════════════════════════════════════
Runtime lifecycle orchestrator for the PhysWorldLM Runtime Engine.

Pipeline position
------------------
        CompilationReport (scene_compiler.SceneCompiler)
                        │
                        ▼
        ┌─────────────────────────────────────────────────┐
        │             SimulationController                │
        │                                                 │
        │   RuntimeContext ◄───────────────┐              │
        │        │                         │              │
        │        ▼                         │              │
        │   Timeline ──────► advance() ────┤              │
        │        │                         │              │
        │        ▼                         │              │
        │   EntityManager ──► state sync ──┤              │
        │        │                         │              │
        │        ▼                         │              │
        │   BackendAdapter.step()          │              │
        └──────────────────────┬──────────────────────────┘
                               ▼
                Omniverse / Gazebo / MuJoCo / Unity / Unreal

Scope
-----
`SimulationController` owns **orchestration only** -- exactly the same
posture `world_pipeline.WorldPipeline` takes one layer up. It does not
implement physics, rendering, sensing, or scene compilation; those
responsibilities are injected as collaborators (`Timeline`,
`EntityManager`, a `BackendAdapter`, an optional `SensorManagerProtocol`)
and invoked through the narrow protocols defined in `runtime_context.py`.

`SimulationController` never imports Omniverse, Gazebo, MuJoCo, Unity,
Unreal, or ROS -- every one of those is reached, if at all, through the
caller-supplied `BackendAdapter`.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from runtime.entity_manager import EntityManager
from runtime.runtime_context import (
    BackendAdapterError,
    BenchmarkRuntimeStatistics,
    EntitySnapshotBatch,
    RuntimeCancelledError,
    RuntimeConfig,
    RuntimeContext,
    RuntimeEventType,
    RuntimeLifecycleError,
    RuntimePhase,
    RuntimeStatistics,
    SnapshotError,
    new_runtime_context,
)
from runtime.timeline import AdvanceResult, TimeMode, Timeline, TimelineState

logger = logging.getLogger("physworldlm.runtime.simulation_controller")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════
# Reports
# ════════════════════════════════════════════════════════════════════════

@dataclass
class StepReport:
    """Result of a single `SimulationController.step()` call."""

    frame: int
    sim_time_s: float
    substeps_executed: int
    dropped_substeps: int
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "sim_time_s": round(self.sim_time_s, 6),
            "substeps_executed": self.substeps_executed,
            "dropped_substeps": self.dropped_substeps,
            "duration_s": round(self.duration_s, 6),
        }


@dataclass
class RuntimeReport:
    """Final, structured result returned by `play()`/`run_benchmark()`,
    mirroring the shape of `world_pipeline.PipelineReport` so both halves
    of PhysWorldLM hand callers a consistent report contract.
    """

    phase: RuntimePhase
    scene_id: str
    statistics: RuntimeStatistics
    frame_count: int
    sim_time_s: float
    cancelled: bool = False
    error: Optional[str] = None
    benchmark: Optional[BenchmarkRuntimeStatistics] = None

    @property
    def success(self) -> bool:
        return self.error is None and not self.cancelled

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.label,
            "scene_id": self.scene_id,
            "statistics": self.statistics.to_dict(),
            "frame_count": self.frame_count,
            "sim_time_s": round(self.sim_time_s, 6),
            "cancelled": self.cancelled,
            "error": self.error,
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A serializable point-in-time capture of the whole runtime, used
    for checkpointing and `restore()`.
    """

    scene_id: str
    frame_count: int
    sim_time_s: float
    entities: tuple[dict, ...]
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "frame_count": self.frame_count,
            "sim_time_s": round(self.sim_time_s, 6),
            "entities": list(self.entities),
            "captured_at": self.captured_at.isoformat(),
        }


# ════════════════════════════════════════════════════════════════════════
# SimulationController
# ════════════════════════════════════════════════════════════════════════

class SimulationController:
    """Single orchestration entry point for the Runtime Engine.

    Owns the runtime lifecycle (`initialize` → `play`/`step` →
    `pause`/`resume` → `stop` → `shutdown`) and coordinates a `Timeline`,
    an `EntityManager`, and an injected `BackendAdapter` on every step,
    broadcasting every transition on the shared `RuntimeContext.event_bus`.

    Thread-safety: a single instance guards its lifecycle behind one
    re-entrant lock. As with `world_pipeline.WorldPipeline`, a given
    *run* is inherently sequential; concurrent `play()`/`step()` calls on
    the same instance are serialized rather than interleaved.

    Example:
        >>> context = new_runtime_context(config=RuntimeConfig(real_time=False))
        >>> controller = SimulationController(context=context, backend=my_backend)
        >>> with controller:
        ...     controller.entity_manager.create_entity("ball", kind="rigid_body")
        ...     report = controller.play(max_steps=120)
        >>> report.success
        True
    """

    def __init__(
        self,
        context: Optional[RuntimeContext] = None,
        *,
        backend: Optional["BackendAdapter"] = None,  # noqa: F821 - see runtime_context.BackendAdapter
        entity_manager: Optional[EntityManager] = None,
        timeline: Optional[Timeline] = None,
    ) -> None:
        """Initialize the controller with its injected collaborators.

        Args:
            context: Shared `RuntimeContext`. A default one is created
                (via `new_runtime_context()`) if omitted.
            backend: `BackendAdapter` to drive on each step. May also be
                bound later via `bind_backend()`; `play()`/`step()`
                require one to be bound before they run.
            entity_manager: Injectable `EntityManager`. A new one bound
                to `context` is created if omitted.
            timeline: Injectable `Timeline`, configured from
                `context.config` if omitted.
        """
        self._context = context or new_runtime_context()
        if backend is not None:
            self._context.backend = backend

        self.entity_manager = entity_manager or EntityManager(context=self._context)
        self.timeline = timeline or Timeline(
            fixed_timestep_s=self._context.config.fixed_timestep_s,
            mode=TimeMode.FIXED,
            max_substeps=self._context.config.max_substeps,
            time_scale=self._context.config.time_scale,
            event_bus=self._context.event_bus,
        )

        self._lock = threading.RLock()
        self._phase = RuntimePhase.UNINITIALIZED
        self._last_report: Optional[RuntimeReport] = None

    # ── context manager support ─────────────────────────────────────

    def __enter__(self) -> "SimulationController":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # ── properties ──────────────────────────────────────────────────

    @property
    def context(self) -> RuntimeContext:
        return self._context

    @property
    def phase(self) -> RuntimePhase:
        with self._lock:
            return self._phase

    # ── backend binding ──────────────────────────────────────────────

    def bind_backend(self, backend: "BackendAdapter") -> None:  # noqa: F821
        """Bind (or replace) the `BackendAdapter` this controller drives."""
        with self._lock:
            self._context.backend = backend

    def _require_backend(self) -> "BackendAdapter":  # noqa: F821
        backend = self._context.backend
        if backend is None:
            raise RuntimeLifecycleError("No BackendAdapter is bound. Call bind_backend() or pass backend=... at construction.")
        return backend

    # ── lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Prepare the runtime for use: verifies the backend (if bound)
        is available and seeds determinism.

        Raises:
            RuntimeLifecycleError: If called while already active.
            BackendAdapterError: If a bound backend reports itself unavailable.
        """
        with self._lock:
            if self._phase not in (RuntimePhase.UNINITIALIZED, RuntimePhase.SHUTDOWN, RuntimePhase.STOPPED):
                raise RuntimeLifecycleError(f"Cannot initialize(): runtime is already active in phase '{self._phase.label}'.")

            if self._context.config.random_seed is not None:
                random.seed(self._context.config.random_seed)

            backend = self._context.backend
            if backend is not None:
                try:
                    available = backend.is_available()
                except Exception as exc:  # noqa: BLE001
                    raise BackendAdapterError(f"Backend '{backend.name}' health check raised: {exc}") from exc
                if not available:
                    raise BackendAdapterError(f"Backend '{backend.name}' reported itself unavailable.")

            self._phase = RuntimePhase.INITIALIZED
            self._context.publish(RuntimeEventType.RUNTIME_INITIALIZED, scene_id=self._context.scene_id)
            logger.info("SimulationController initialized (scene_id=%s).", self._context.scene_id)

    def shutdown(self) -> None:
        """Stop the timeline and release runtime state. Idempotent."""
        with self._lock:
            if self._phase is RuntimePhase.SHUTDOWN:
                return
            self.timeline.stop()
            self._phase = RuntimePhase.SHUTDOWN
            self._context.publish(RuntimeEventType.RUNTIME_SHUTDOWN, scene_id=self._context.scene_id)
            logger.info("SimulationController shut down.")

    def play(
        self,
        *,
        max_steps: Optional[int] = None,
        cancellation_event: Optional[threading.Event] = None,
    ) -> RuntimeReport:
        """Run the timeline forward until `max_steps` (or
        `context.config.max_steps`) is reached, or cancellation fires.

        Args:
            max_steps: Overrides `context.config.max_steps` for this call.
                `None` on both means the caller must externally `pause()`/
                `stop()` (e.g. from another thread or an interactive UI).
            cancellation_event: Cooperative-cancellation token checked at
                every step boundary.

        Returns:
            A `RuntimeReport` summarizing the run. Never raises for
            backend/step failures -- those are captured in the report,
            mirroring `world_pipeline.WorldPipeline.run()`'s contract.

        Raises:
            RuntimeLifecycleError: If called before `initialize()`.
        """
        with self._lock:
            if self._phase in (RuntimePhase.UNINITIALIZED, RuntimePhase.SHUTDOWN):
                raise RuntimeLifecycleError("SimulationController must be initialize()'d before play().")

            step_cap = max_steps if max_steps is not None else self._context.config.max_steps
            self._context.cancellation_event = cancellation_event or self._context.cancellation_event

            self.timeline.start()
            self._phase = RuntimePhase.PLAYING
            wall_start = time.monotonic()
            error: Optional[str] = None
            cancelled = False

            try:
                while True:
                    if self._phase is not RuntimePhase.PLAYING:
                        break  # pause()/stop() called from another thread
                    if self._context.is_cancelled():
                        cancelled = True
                        raise RuntimeCancelledError("Run cancelled via cancellation_event.")
                    if step_cap is not None and self.timeline.snapshot().frame_count >= step_cap:
                        break

                    elapsed = self._context.config.fixed_timestep_s if not self._context.config.real_time else None
                    self._advance_and_step(elapsed_wall_s=elapsed)

            except RuntimeCancelledError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 - never let a backend bug escape play()
                logger.exception("Unexpected error during play().")
                error = f"Unexpected {type(exc).__name__}: {exc}"
                self._context.statistics.error_count += 1
                self._context.publish(RuntimeEventType.RUNTIME_ERROR, error=str(exc))
            finally:
                if self._phase is RuntimePhase.PLAYING:
                    self._phase = RuntimePhase.STOPPED
                self.timeline.stop()
                self._context.statistics.total_wall_time_s += time.monotonic() - wall_start

            report = self._build_report(cancelled=cancelled, error=error)
            self._last_report = report
            return report

    def pause(self) -> None:
        """Pause a running simulation; `play()`'s loop will return control."""
        with self._lock:
            if self._phase is RuntimePhase.PLAYING:
                self.timeline.pause()
                self._phase = RuntimePhase.PAUSED

    def resume(self) -> RuntimeReport:
        """Resume a paused simulation by re-entering `play()`'s loop."""
        with self._lock:
            if self._phase is not RuntimePhase.PAUSED:
                raise RuntimeLifecycleError(f"Cannot resume(): runtime is in phase '{self._phase.label}', not PAUSED.")
            self.timeline.resume()
            self._phase = RuntimePhase.PLAYING
        return self.play()

    def stop(self) -> None:
        """Stop the simulation. Sim time/entities are preserved; use
        `reset()` to clear them.
        """
        with self._lock:
            self.timeline.stop()
            self._phase = RuntimePhase.STOPPED

    def reset(self) -> None:
        """Stop the simulation, clear the timeline, and destroy every entity."""
        with self._lock:
            self.timeline.reset()
            self.entity_manager.clear()
            self._phase = RuntimePhase.INITIALIZED

    def step(self) -> StepReport:
        """Advance the simulation by exactly one fixed-size substep,
        regardless of the timeline's running state -- used for
        deterministic single-stepping / manual replay control.

        Raises:
            RuntimeLifecycleError: If called before `initialize()`.
        """
        with self._lock:
            if self._phase in (RuntimePhase.UNINITIALIZED, RuntimePhase.SHUTDOWN):
                raise RuntimeLifecycleError("SimulationController must be initialize()'d before step().")
            start = time.monotonic()
            result = self.timeline.step_once()
            self._sync_backend(result)
            duration = time.monotonic() - start
            if self._context.config.enable_profiling:
                self._context.statistics.record_step(duration)
            self._context.statistics.total_sim_time_s = result.sim_time_s
            self._maybe_checkpoint(result.frame_count)
            return StepReport(
                frame=result.frame_count, sim_time_s=result.sim_time_s,
                substeps_executed=1, dropped_substeps=0, duration_s=duration,
            )

    # ── benchmark mode ────────────────────────────────────────────────

    def run_benchmark(
        self,
        *,
        iterations: int = 10,
        steps_per_iteration: int = 300,
        cancellation_event: Optional[threading.Event] = None,
    ) -> RuntimeReport:
        """Run `iterations` independent `play()` passes of
        `steps_per_iteration` steps each, resetting between iterations,
        and report aggregate timing -- mirroring
        `world_pipeline.WorldPipeline.run(mode=ExecutionMode.BENCHMARK)`.

        Args:
            iterations: Number of independent passes.
            steps_per_iteration: `max_steps` for each pass.
            cancellation_event: Checked at every step boundary across all
                iterations.

        Returns:
            The final iteration's `RuntimeReport`, with `.benchmark`
            populated with aggregate statistics.
        """
        with self._lock:
            if self._phase in (RuntimePhase.UNINITIALIZED, RuntimePhase.SHUTDOWN):
                raise RuntimeLifecycleError("SimulationController must be initialize()'d before run_benchmark().")

            per_iteration: list[float] = []
            successes = 0
            failures = 0
            last_report: Optional[RuntimeReport] = None
            bench_start = time.monotonic()

            for i in range(max(1, iterations)):
                if self._context.config.random_seed is not None:
                    random.seed(self._context.config.random_seed + i)
                self.reset()
                self.initialize()
                iter_start = time.monotonic()
                report = self.play(max_steps=steps_per_iteration, cancellation_event=cancellation_event)
                per_iteration.append(time.monotonic() - iter_start)
                if report.success:
                    successes += 1
                else:
                    failures += 1
                last_report = report
                self._context.publish(RuntimeEventType.BENCHMARK_ITERATION, iteration=i, success=report.success)

            total = time.monotonic() - bench_start
            stats = BenchmarkRuntimeStatistics(
                iterations=iterations, successes=successes, failures=failures, total_time_s=total,
                mean_time_s=(sum(per_iteration) / len(per_iteration)) if per_iteration else 0.0,
                min_time_s=min(per_iteration) if per_iteration else 0.0,
                max_time_s=max(per_iteration) if per_iteration else 0.0,
                per_iteration_s=per_iteration,
            )
            assert last_report is not None
            final = replace(last_report, benchmark=stats)
            self._last_report = final
            return final

    # ── snapshot / checkpointing ────────────────────────────────────

    def snapshot(self) -> RuntimeSnapshot:
        """Capture a serializable point-in-time snapshot of the runtime."""
        with self._lock:
            clock = self.timeline.snapshot()
            entities = tuple(rec.to_dict() for rec in self.entity_manager.all())
            return RuntimeSnapshot(
                scene_id=self._context.scene_id, frame_count=clock.frame_count,
                sim_time_s=clock.sim_time_s, entities=entities,
            )

    def save_snapshot(self, path: Path | str) -> Path:
        """Serialize `snapshot()` to JSON at `path`.

        Raises:
            SnapshotError: If writing fails.
        """
        snap = self.snapshot()
        try:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(snap.to_dict(), indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            raise SnapshotError(f"Failed to write snapshot to '{path}': {exc}") from exc
        self._context.statistics.snapshots_saved += 1
        self._context.publish(RuntimeEventType.SNAPSHOT_SAVED, path=str(destination))
        return destination

    def load_snapshot(self, path: Path | str) -> None:
        """Restore entity bookkeeping and clock position from a JSON
        snapshot written by `save_snapshot()`.

        Note: only `EntityManager`/`Timeline` bookkeeping is restored --
        actual physics/render state restoration is the bound
        `BackendAdapter`'s responsibility (typically via its own
        `sync_entity_state()` call immediately after this).

        Raises:
            SnapshotError: If the file is missing or malformed.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"Failed to load snapshot from '{path}': {exc}") from exc

        with self._lock:
            self.entity_manager.clear()
            self.timeline.reset()
            self.timeline.seek(float(payload.get("sim_time_s", 0.0)))
            id_map: dict[str, Any] = {}
            for entity_dict in payload.get("entities", ()):
                parent_id = entity_dict.get("parent")
                parent_handle = id_map.get(parent_id)
                handle = self.entity_manager.create_entity(
                    name=entity_dict.get("name", "entity"), kind=entity_dict.get("kind", "generic"),
                    parent=parent_handle, owner=entity_dict.get("owner", "scene"),
                    source_id=entity_dict.get("source_id"), entity_id=entity_dict.get("entity_id"),
                )
                id_map[entity_dict.get("entity_id")] = handle
            self._context.statistics.snapshots_restored += 1
            self._context.publish(RuntimeEventType.SNAPSHOT_RESTORED, path=str(path))

    def _maybe_checkpoint(self, frame_count: int) -> None:
        cfg = self._context.config
        if not cfg.enable_checkpointing or cfg.checkpoint_dir is None:
            return
        if frame_count % max(1, cfg.checkpoint_interval_steps) != 0:
            return
        try:
            path = cfg.checkpoint_dir / self._context.scene_id / f"frame_{frame_count:08d}.json"
            self.save_snapshot(path)
        except SnapshotError:
            logger.exception("Checkpoint at frame %d failed; continuing.", frame_count)

    # ── internal stepping ─────────────────────────────────────────────

    def _advance_and_step(self, *, elapsed_wall_s: Optional[float]) -> None:
        start = time.monotonic()
        result = self.timeline.advance(elapsed_wall_s=elapsed_wall_s)
        if result.substep_count > 0:
            self._sync_backend(result)
        duration = time.monotonic() - start
        if self._context.config.enable_profiling and result.substep_count > 0:
            self._context.statistics.record_step(duration)
        self._context.statistics.total_sim_time_s = result.sim_time_s
        self._context.statistics.dropped_substeps += result.dropped_substeps
        if result.substep_count > 0:
            self._maybe_checkpoint(result.frame_count)
            self._context.publish(RuntimeEventType.STEP_COMPLETED, frame=result.frame_count, duration_s=duration)

    def _sync_backend(self, result: AdvanceResult) -> None:
        backend = self._context.backend
        if backend is None:
            return  # entity-only / headless bookkeeping mode -- valid for tests and pure scripting.

        outgoing = EntitySnapshotBatch(
            frame=result.frame_count,
            payload={rec.handle.entity_id: rec.components for rec in self.entity_manager.all() if rec.active},
        )
        try:
            backend.sync_entity_state(outgoing)
            for _ in range(result.substep_count):
                backend.step(result.substep_duration_s)
            incoming = backend.read_back_state()
        except Exception as exc:  # noqa: BLE001
            raise BackendAdapterError(f"Backend '{backend.name}' failed during step: {exc}") from exc

        for entity_id, state in incoming.payload.items():
            try:
                from runtime.entity_manager import EntityHandle as _EH
                if _EH(entity_id) in self.entity_manager:
                    self.entity_manager.set_component(_EH(entity_id), "state", state)
            except Exception:  # noqa: BLE001 - a stray/unknown id from the backend must never break stepping
                logger.debug("Ignoring backend state for unknown entity id '%s'.", entity_id)

    # ── report building ─────────────────────────────────────────────

    def _build_report(self, *, cancelled: bool, error: Optional[str]) -> RuntimeReport:
        clock = self.timeline.snapshot()
        return RuntimeReport(
            phase=self._phase, scene_id=self._context.scene_id, statistics=self._context.statistics,
            frame_count=clock.frame_count, sim_time_s=clock.sim_time_s, cancelled=cancelled, error=error,
        )

    def statistics(self) -> RuntimeStatistics:
        """Return the live `RuntimeStatistics` for the current session."""
        return self._context.statistics

    def last_report(self) -> Optional[RuntimeReport]:
        """Return the most recently completed `play()`/`run_benchmark()` report, if any."""
        with self._lock:
            return self._last_report


__all__ = [
    "StepReport",
    "RuntimeReport",
    "RuntimeSnapshot",
    "SimulationController",
]
