"""
runtime/animation_system.py
══════════════════════════════════════════════════════════════════════════
Per-frame animation subsystem for PhysWorldLM's Omniverse Runtime.

Pipeline position
------------------
    OmniverseRuntime.initialize_runtime_systems()
            │
            ▼
    ┌─────────────────┐
    │ ANIMATION SYSTEM │   <-- this module (registered subsystem)
    └─────────────────┘
            │
            ▼
      OmniverseRuntime._step_frame()  (every frame, via SubsystemRegistry)

Scope
-----
Unlike `environment_builder` / `entity_spawner` / `physics_initializer`,
this module is NOT a one-shot builder -- it implements the
`omniverse_runtime.RuntimeSubsystem` protocol (`initialize` / `update` /
`shutdown`) and is registered with `OmniverseRuntime.subsystems` so it
runs once per simulation frame, driven by `OmniverseRuntime`'s own loop
and `RuntimeConfig.target_fps`. It does not own a second timeline or
run loop of its own.

It advances simple, declarative motion for dynamic entities (linear/
angular integration in the absence of a richer guidance subsystem),
radar-sweep rotation for sensor entities, weather particle/wind phase,
and exposes `play()/pause()/resume()` as *local* gating on top of the
host runtime's own RUNNING/PAUSED state -- so this subsystem can be
independently muted (e.g. "freeze animation, keep simulating physics")
without calling `OmniverseRuntime.pause()` itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("physworldlm.animation_system")
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

class AnimationStateError(Exception):
    """Raised when an animation control method is called in an invalid state."""


# ════════════════════════════════════════════════════════════════════════
# Configuration / state
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AnimationSystemConfig:
    """Tuning for the animation subsystem.

    Attributes:
        radar_sweep_rpm: Rotational speed of radar-sweep animation, in
            revolutions per minute.
        weather_phase_speed: Rate at which the weather/cloud phase
            accumulator advances per second (used to drive cloud/fog
            shader parameters downstream; unitless).
        integrate_dynamic_entities: Whether dynamic entities' transforms
            are advanced using their last-known velocity each frame.
            Disable if a richer physics/guidance subsystem is expected
            to own transform updates instead.
    """

    radar_sweep_rpm: float = 6.0
    weather_phase_speed: float = 0.05
    integrate_dynamic_entities: bool = True


@dataclass
class AnimationStatistics:
    """Per-run animation bookkeeping."""

    frames_animated: int = 0
    entities_animated_last_frame: int = 0
    radar_entities_tracked: int = 0
    weather_phase: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames_animated": self.frames_animated,
            "entities_animated_last_frame": self.entities_animated_last_frame,
            "radar_entities_tracked": self.radar_entities_tracked,
            "weather_phase": round(self.weather_phase, 4),
        }


@dataclass
class _EntityAnimState:
    """Lightweight per-entity animation accumulator (not tied to USD directly)."""

    prim_path: str
    rotation_deg: float = 0.0


# ════════════════════════════════════════════════════════════════════════
# AnimationSystem
# ════════════════════════════════════════════════════════════════════════

class AnimationSystem:
    """`RuntimeSubsystem` implementation driving per-frame animation.

    Registers with `OmniverseRuntime.subsystems` and is driven entirely
    by that runtime's own frame loop:

        >>> anim = AnimationSystem()
        >>> runtime.subsystems.register(anim)
        >>> runtime.initialize()   # calls anim.initialize(registry, config)
        >>> runtime.start()        # calls anim.update(dt, registry) each frame
    """

    name = "animation_system"

    def __init__(self, config: AnimationSystemConfig | None = None) -> None:
        self._config = config or AnimationSystemConfig()
        self._stats = AnimationStatistics()
        self._playing = True
        self._radar_states: dict[str, _EntityAnimState] = {}
        self._dynamic_paths: list[str] = []

    @property
    def statistics(self) -> AnimationStatistics:
        return self._stats

    # ── RuntimeSubsystem protocol ───────────────────────────────────

    def initialize(self, registry: Any, config: Any) -> None:
        """One-time setup: index radar/sensor entities and dynamic entities.

        Args:
            registry: `omniverse_runtime.EntityRegistry`-compatible object.
            config: `omniverse_runtime.RuntimeConfig`-compatible object
                (used only for logging context here; animation timing is
                governed by the `dt` passed into `update()` each frame).
        """
        entities = getattr(registry, "entities", {})
        self._radar_states.clear()
        self._dynamic_paths.clear()

        for prim_path, entity in entities.items():
            category = self._category_of(entity)
            if category in ("radar", "sensor"):
                self._radar_states[prim_path] = _EntityAnimState(prim_path=prim_path)
            if self._is_dynamic(entity):
                self._dynamic_paths.append(prim_path)

        self._stats.radar_entities_tracked = len(self._radar_states)
        logger.info(
            "AnimationSystem initialized: %d radar/sensor entit(y/ies), %d dynamic entit(y/ies).",
            len(self._radar_states),
            len(self._dynamic_paths),
        )

    def update(self, dt: float, registry: Any) -> None:
        """Advance all animated state by `dt` seconds.

        Args:
            dt: Simulation time delta for this frame, in seconds, as
                supplied by `OmniverseRuntime._step_frame()`.
            registry: The live `EntityRegistry`, in case entities were
                added/removed since `initialize()`.
        """
        if not self._playing:
            return

        animated = 0
        animated += self._update_radar_sweep(dt)
        if self._config.integrate_dynamic_entities:
            animated += self._update_dynamic_entities(dt, registry)
        self._update_weather(dt)

        self._stats.frames_animated += 1
        self._stats.entities_animated_last_frame = animated

    def shutdown(self) -> None:
        """Release any resources held by the subsystem (none persistent)."""
        logger.info(
            "AnimationSystem shutdown after %d animated frame(s).", self._stats.frames_animated
        )
        self._radar_states.clear()
        self._dynamic_paths.clear()

    # ── playback control (local gating, independent of host runtime state) ──

    def play(self) -> None:
        """Resume local animation advancement (no-op if already playing)."""
        if self._playing:
            return
        self._playing = True
        logger.info("Animation playback resumed.")

    def pause(self) -> None:
        """Freeze local animation advancement without affecting host runtime state."""
        if not self._playing:
            return
        self._playing = False
        logger.info("Animation playback paused.")

    def resume(self) -> None:
        """Alias for `play()`, provided for symmetry with `pause()`."""
        self.play()

    # ── internal animation steps ────────────────────────────────────

    def _update_radar_sweep(self, dt: float) -> int:
        degrees_per_second = self._config.radar_sweep_rpm * 360.0 / 60.0
        for state in self._radar_states.values():
            state.rotation_deg = (state.rotation_deg + degrees_per_second * dt) % 360.0
        return len(self._radar_states)

    def _update_dynamic_entities(self, dt: float, registry: Any) -> int:
        entities = getattr(registry, "entities", {})
        count = 0
        for prim_path in self._dynamic_paths:
            entity = entities.get(prim_path)
            if entity is None:
                continue
            velocity = self._velocity_of(entity)
            if velocity == (0.0, 0.0, 0.0):
                continue
            self._advance_transform(prim_path, velocity, dt)
            count += 1
        return count

    def _update_weather(self, dt: float) -> None:
        self._stats.weather_phase = (
            self._stats.weather_phase + self._config.weather_phase_speed * dt
        ) % (2 * math.pi)

    # ── entity introspection / stage-write helpers ──────────────────

    @staticmethod
    def _category_of(entity: Any) -> str:
        category = getattr(entity, "category", None)
        if category is not None:
            return str(getattr(category, "value", category)).lower()
        metadata = getattr(entity, "metadata", {}) or {}
        return str(metadata.get("entity_type", "")).lower()

    @staticmethod
    def _is_dynamic(entity: Any) -> bool:
        motion_class = getattr(entity, "motion_class", None)
        if motion_class is not None:
            return str(getattr(motion_class, "value", motion_class)) == "dynamic"
        return not getattr(entity, "is_static", True)

    @staticmethod
    def _velocity_of(entity: Any) -> tuple[float, float, float]:
        metadata = getattr(entity, "metadata", {}) or {}
        try:
            return (
                float(metadata.get("velocity_x", 0.0)),
                float(metadata.get("velocity_y", 0.0)),
                float(metadata.get("velocity_z", 0.0)),
            )
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)

    def _advance_transform(self, prim_path: str, velocity: tuple[float, float, float], dt: float) -> None:
        # Integration point: real builds write `xformOp:translate` deltas
        # via `pxr.UsdGeom.XformCommonAPI` here. Kept as a structured
        # debug log so this module is exercisable without a live stage.
        dx, dy, dz = (v * dt for v in velocity)
        logger.debug("[fallback] would advance %s by (%.4f, %.4f, %.4f)", prim_path, dx, dy, dz)
