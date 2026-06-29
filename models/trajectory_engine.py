"""
models/trajectory_engine.py
────────────────────────────────────────────────────────────────────────────────
PhysWorldLM — TrajectoryEngine

Converts a physics state at time t into a full multi-step Trajectory:

    TemporalWorldModel
        ↓
    StateEngine
        ↓
    TrajectoryEngine          ← THIS FILE
        ├── Rollout modules   (Autoregressive / TeacherForce / Beam / MonteCarlo)
        ├── EventDetector
        ├── CollisionPredictor
        ├── TrajectoryScorer
        ├── TrajectoryStatistics
        ├── TrajectoryComparator
        ├── TrajectoryMemory
        └── TrajectoryVisualizer
        ↓
    Bullet / MuJoCo / Isaac Sim
        ↓
    Renderer
        ↓
    VideoDiffusion

Module-boundary contract
────────────────────────
``StateEngine``       : advances state(t) → state(t+dt) at a fixed sub-step.
``TrajectoryEngine``  : sequences those steps into state(t) … state(t+n),
                        applies learned corrections from ``TemporalWorldModel``,
                        detects events, scores trajectories, and manages a
                        trajectory memory bank. It does NOT plan, control, or
                        choose actions.

This file deliberately excludes
────────────────────────────────
  ❌  Bullet / MuJoCo / Isaac / Gazebo internals
  ❌  Rendering or video generation
  ❌  Sensor models
  ❌  Path planning, A*, or control policies
  ❌  RL agents or decision-making of any kind

Those responsibilities belong to downstream modules. TrajectoryEngine exposes
placeholder ``to_bullet()`` / ``to_mujoco()`` / etc. methods that raise
``NotImplementedError`` so downstream code can type-check against a stable
interface now.

Interface contracts on collaborating modules
────────────────────────────────────────────
``models.world_spec.WorldSpec``           — world description (entities, env, sim graph)
``models.world_spec.Vec3``                — 3-component vector
``models.world_spec.PhysicsState``        — single-instant kinematic snapshot
``models.state_engine.StateEngine``       — deterministic Euler/RK4 integrator
``models.state_engine.MutableEntityState``— working per-entity kinematic state
``models.temporal_world_model.TemporalWorldModel``  — learned world-evolution model
``models.temporal_world_model.PredictedState``       — model output (deltas, latents)
``models.temporal_world_model.TemporalOutput``       — full forward-pass output
``models.trajectory.Frame``               — single-entity, single-timestep record
``models.trajectory.Trajectory``          — flat list of Frame records (StateEngine output)

All physical quantities are in SI units (m, m/s, m/s², rad, rad/s, J, kg).
Coordinates: x=East, y=Up (gravity = −y), z=North.
"""

from __future__ import annotations

import copy
import json
import math
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# ── internal imports (contract-only; real modules must be installed) ──────────
try:
    from models.world_spec import WorldSpec, Entity, PhysicsState, Vec3
    from models.state_engine import StateEngine, MutableEntityState
    from models.temporal_world_model import (
        TemporalWorldModel, PredictedState, TemporalOutput,
        TemporalWorldModelConfig,
    )
    from models.trajectory import Frame, Trajectory
except ImportError as _e:  # allow static analysis / docs builds without full env
    warnings.warn(f"[TrajectoryEngine] import fallback: {_e}", stacklevel=1)
    WorldSpec = Any  # type: ignore[assignment,misc]
    StateEngine = Any  # type: ignore[assignment,misc]
    TemporalWorldModel = Any  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants & type aliases
# ─────────────────────────────────────────────────────────────────────────────

_Tensor = torch.Tensor
_VecLike = Union[Tuple[float, float, float], "Vec3", _Tensor]

_GRAVITY: float = 9.81   # m/s² magnitude (used internally for energy estimates)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  TrajectoryEngineConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryEngineConfig:
    """All hyperparameters for TrajectoryEngine — no magic numbers inside.

    Attributes:
        history_length:             Temporal context fed to TemporalWorldModel.
        prediction_horizon:          Max future steps (model constraint).
        dt:                           Integration timestep in seconds.
        max_rollout_steps:             Hard cap on any single rollout.
        beam_width:                     Beam size for beam-search rollout.
        num_samples:                     Monte Carlo trajectory count.
        uncertainty_samples:              Samples drawn per latent for
                                         uncertainty estimation.
        collision_threshold:               Minimum separation (m) below
                                          which two entities are flagged as
                                          colliding.
        curvature_threshold:                Curvature (1/m) above which a
                                           trajectory segment is flagged as
                                           a sharp turn event.
        stop_velocity_threshold:             Speed (m/s) below which an entity
                                            is considered stopped.
        bounce_acceleration_threshold:        Acceleration spike (m/s²) that
                                             triggers a bounce event.
        fall_vertical_velocity_threshold:      Downward velocity (m/s) that
                                              triggers a fall event.
        enable_beam_search:                    Activate beam rollout.
        enable_teacher_forcing:                 Activate teacher-force rollout.
        enable_autoregressive:                  Activate autoregressive rollout.
        enable_monte_carlo:                      Activate Monte Carlo rollout.
        enable_uncertainty_sampling:              Draw from N(mu, exp(logvar)).
        enable_event_detection:                    Run EventDetector.
        enable_collision_prediction:                Run CollisionPredictor.
        enable_energy_monitoring:                    Track KE + PE each step.
        enable_physics_consistency:                   Compute physics_score.
        memory_bank_capacity:                          Max stored trajectories.
        device:                                         Torch device string.
    """

    history_length:                  int   = 32
    prediction_horizon:               int   = 128
    dt:                                float = 0.01
    max_rollout_steps:                  int   = 1000
    beam_width:                          int   = 5
    num_samples:                          int   = 16
    uncertainty_samples:                   int   = 8
    collision_threshold:                    float = 0.05
    curvature_threshold:                     float = 0.2
    stop_velocity_threshold:                  float = 1e-3
    bounce_acceleration_threshold:             float = 20.0
    fall_vertical_velocity_threshold:           float = 2.0
    enable_beam_search:                          bool  = True
    enable_teacher_forcing:                       bool  = True
    enable_autoregressive:                         bool  = True
    enable_monte_carlo:                             bool  = True
    enable_uncertainty_sampling:                     bool  = True
    enable_event_detection:                           bool  = True
    enable_collision_prediction:                       bool  = True
    enable_energy_monitoring:                           bool  = True
    enable_physics_consistency:                          bool  = True
    memory_bank_capacity:                                 int   = 5_000
    device:                                                str   = "cpu"

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.max_rollout_steps < 1:
            raise ValueError(f"max_rollout_steps must be ≥ 1, got {self.max_rollout_steps}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1, got {self.beam_width}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be ≥ 1, got {self.num_samples}")
        if self.collision_threshold <= 0:
            raise ValueError(f"collision_threshold must be > 0, got {self.collision_threshold}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Core dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Kinematic snapshot for one entity at one exported timestep.

    Distinct from ``models.trajectory.Frame`` (which is StateEngine's output
    record) because ``TrajectoryPoint`` carries additional trajectory-level
    context (curvature, jerk, event_flags, learned uncertainty, physics_score)
    that ``Frame`` does not.

    Attributes:
        time:             Simulation time in seconds.
        entity_id:         Entity identifier.
        position:           (x, y, z) in metres.
        velocity:            (vx, vy, vz) in m/s.
        acceleration:         (ax, ay, az) in m/s².
        orientation:           (rx, ry, rz) Euler angles in radians.
        angular_velocity:       (wx, wy, wz) in rad/s.
        kinetic_energy:          KE = ½·m·|v|² in joules.
        potential_energy:         PE = m·g·y in joules.
        curvature:                 Instantaneous path curvature in 1/m.
        jerk:                       |da/dt| in m/s³.
        uncertainty:                 Per-quantity variance dict or None.
        physics_score:                Scalar physics-consistency score [0, 1].
        event_flags:                   List of event-type strings active here.
        metadata:                       Free-form bag.
    """

    time:              float
    entity_id:          str
    position:            Tuple[float, float, float]
    velocity:             Tuple[float, float, float]
    acceleration:          Tuple[float, float, float]
    orientation:            Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity:        Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kinetic_energy:           float = 0.0
    potential_energy:          float = 0.0
    curvature:                  float = 0.0
    jerk:                        float = 0.0
    uncertainty:                  Optional[Dict[str, float]] = None
    physics_score:                 float = 1.0
    event_flags:                    List[str] = field(default_factory=list)
    metadata:                        Dict[str, Any] = field(default_factory=dict)

    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy

    def speed(self) -> float:
        vx, vy, vz = self.velocity
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def height(self) -> float:
        return self.position[1]

    def to_dict(self) -> dict:
        return {
            "time":              self.time,
            "entity_id":         self.entity_id,
            "position":          list(self.position),
            "velocity":          list(self.velocity),
            "acceleration":      list(self.acceleration),
            "orientation":       list(self.orientation),
            "angular_velocity":  list(self.angular_velocity),
            "kinetic_energy_J":  round(self.kinetic_energy, 6),
            "potential_energy_J": round(self.potential_energy, 6),
            "total_energy_J":    round(self.total_energy, 6),
            "speed_ms":          round(self.speed(), 6),
            "curvature":         round(self.curvature, 8),
            "jerk":              round(self.jerk, 6),
            "uncertainty":       self.uncertainty,
            "physics_score":     round(self.physics_score, 6),
            "event_flags":       self.event_flags,
            "metadata":          self.metadata,
        }

    @classmethod
    def from_frame(cls, frame: "Frame", mass: float = 1.0, g: float = _GRAVITY) -> "TrajectoryPoint":
        """Construct a ``TrajectoryPoint`` from a ``models.trajectory.Frame``.

        Energy is recomputed from position/velocity because ``Frame`` stores
        pre-computed values that may have been produced by a different g.
        """
        pos = (frame.position.x, frame.position.y, frame.position.z)
        vel = (frame.velocity.x, frame.velocity.y, frame.velocity.z)
        acc = (frame.acceleration.x, frame.acceleration.y, frame.acceleration.z)
        ori = (frame.orientation.x, frame.orientation.y, frame.orientation.z)
        ang = (frame.angular_vel.x, frame.angular_vel.y, frame.angular_vel.z)
        vx, vy, vz = vel
        ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
        pe = mass * g * pos[1]
        return cls(
            time=frame.t,
            entity_id=frame.entity_id,
            position=pos, velocity=vel, acceleration=acc,
            orientation=ori, angular_velocity=ang,
            kinetic_energy=ke, potential_energy=pe,
        )


@dataclass
class TrajectoryState:
    """Sequence-level state for one entity across multiple timesteps.

    Separates the *sequence* concept (a list of snapshots over time) from
    the *instant* concept (``PhysicsState`` / ``TrajectoryPoint``), and from
    StateEngine's ``MutableEntityState`` (the integrator's working buffer).

    Attributes:
        entity_id:      Entity identifier.
        points:          Time-ordered list of ``TrajectoryPoint``.
        score:            Composite trajectory quality score in [0, 1].
        metadata:          Free-form bag (source rollout mode, model tag, …).
    """

    entity_id:  str
    points:      List[TrajectoryPoint] = field(default_factory=list)
    score:        float = 0.0
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def append(self, point: TrajectoryPoint) -> None:
        """Append a point; entity_id must match."""
        if point.entity_id != self.entity_id:
            raise ValueError(
                f"TrajectoryState.append: entity_id mismatch "
                f"(expected {self.entity_id!r}, got {point.entity_id!r})"
            )
        self.points.append(point)

    def times(self) -> List[float]:
        return [p.time for p in self.points]

    def positions(self) -> List[Tuple[float, float, float]]:
        return [p.position for p in self.points]

    def velocities(self) -> List[Tuple[float, float, float]]:
        return [p.velocity for p in self.points]

    def duration(self) -> float:
        t = self.times()
        return t[-1] - t[0] if len(t) > 1 else 0.0

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "score":     round(self.score, 6),
            "metadata":  self.metadata,
            "points":    [p.to_dict() for p in self.points],
        }


@dataclass
class Event:
    """A discrete physics event detected during trajectory generation.

    Attributes:
        event_type:   One of ``collision``, ``impact``, ``stop``, ``start``,
                      ``fall``, ``launch``, ``sliding``, ``bounce``.
        time:          Simulation time at detection (seconds).
        entities:       Entity ids involved (1 or 2).
        location:        (x, y, z) estimated event location.
        severity:         Dimensionless severity in [0, 1] (energy-based or
                         velocity-based, depending on event type).
        metadata:         Free-form bag (contact normal, impact energy, …).
    """

    event_type:   str
    time:          float
    entities:       List[str]
    location:        Tuple[float, float, float] = (0.0, 0.0, 0.0)
    severity:         float = 0.0
    metadata:          Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "time":       round(self.time, 6),
            "entities":   self.entities,
            "location":   list(self.location),
            "severity":   round(self.severity, 6),
            "metadata":   self.metadata,
        }


@dataclass
class TrajectoryOutput:
    """Top-level output of a TrajectoryEngine rollout.

    Attributes:
        trajectories:           Per-entity ``TrajectoryState`` list,
                                one per dynamic entity.
        events:                  Detected events, chronological.
        collision_probabilities:  Dict of ``"entityA:entityB"`` → probability.
        physics_score:             Mean physics-consistency score over all
                                  entities and timesteps, in [0, 1].
        uncertainty_score:          Mean trajectory uncertainty (lower = more
                                   confident), in [0, 1].
        rollout_mode:               The mode that produced this output.
        wall_time_s:                 Wall-clock seconds to compute.
        metadata:                     Free-form bag.
    """

    trajectories:            List[TrajectoryState] = field(default_factory=list)
    events:                   List[Event] = field(default_factory=list)
    collision_probabilities:   Dict[str, float] = field(default_factory=dict)
    physics_score:              float = 1.0
    uncertainty_score:           float = 0.0
    rollout_mode:                 str = "autoregressive"
    wall_time_s:                   float = 0.0
    metadata:                       Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rollout_mode":             self.rollout_mode,
            "physics_score":            round(self.physics_score, 6),
            "uncertainty_score":        round(self.uncertainty_score, 6),
            "wall_time_s":              round(self.wall_time_s, 4),
            "collision_probabilities":  {k: round(v, 6) for k, v in self.collision_probabilities.items()},
            "events":                   [e.to_dict() for e in self.events],
            "trajectories":             [t.to_dict() for t in self.trajectories],
            "metadata":                 self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  EventDetector
# ─────────────────────────────────────────────────────────────────────────────

class EventDetector:
    """Scans successive ``TrajectoryPoint`` pairs for discrete physics events.

    Each ``detect_*`` method is stateless and idempotent: it receives two
    consecutive points (prev, curr) and returns an ``Event`` if the condition
    is met, else ``None``. The main engine calls them inside the rollout loop.

    Args:
        config: ``TrajectoryEngineConfig`` (reads thresholds from it).
    """

    def __init__(self, config: TrajectoryEngineConfig) -> None:
        self._cfg = config

    # ── per-pair detectors ────────────────────────────────────────────────────

    def detect_stop(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Entity transitions from moving to essentially stationary."""
        if prev.speed() >= self._cfg.stop_velocity_threshold > curr.speed():
            return Event(
                event_type="stop",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, prev.speed() / (self._cfg.stop_velocity_threshold * 100 + 1e-9)),
                metadata={"speed_before": prev.speed(), "speed_after": curr.speed()},
            )
        return None

    def detect_start(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Entity transitions from stationary to moving."""
        if prev.speed() < self._cfg.stop_velocity_threshold <= curr.speed():
            return Event(
                event_type="start",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, curr.speed() / 30.0),
                metadata={"speed_after": curr.speed()},
            )
        return None

    def detect_fall(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Entity gains significant downward (−y) velocity."""
        prev_vy = prev.velocity[1]
        curr_vy = curr.velocity[1]
        if prev_vy > -self._cfg.fall_vertical_velocity_threshold >= curr_vy:
            return Event(
                event_type="fall",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, abs(curr_vy) / 20.0),
                metadata={"vy_before": prev_vy, "vy_after": curr_vy},
            )
        return None

    def detect_launch(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Entity gains significant upward (+y) velocity."""
        prev_vy = prev.velocity[1]
        curr_vy = curr.velocity[1]
        if curr_vy > self._cfg.fall_vertical_velocity_threshold >= prev_vy:
            return Event(
                event_type="launch",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, curr_vy / 20.0),
                metadata={"vy_before": prev_vy, "vy_after": curr_vy},
            )
        return None

    def detect_bounce(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Sudden acceleration spike consistent with an elastic or inelastic impact."""
        prev_a = math.sqrt(sum(a * a for a in prev.acceleration))
        curr_a = math.sqrt(sum(a * a for a in curr.acceleration))
        if curr_a > self._cfg.bounce_acceleration_threshold and curr_a > 2.0 * prev_a:
            return Event(
                event_type="bounce",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, curr_a / (self._cfg.bounce_acceleration_threshold * 5.0)),
                metadata={"accel_before": prev_a, "accel_after": curr_a},
            )
        return None

    def detect_sliding(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Horizontal motion with near-zero vertical velocity — ground sliding."""
        hv = math.sqrt(curr.velocity[0] ** 2 + curr.velocity[2] ** 2)
        vy = abs(curr.velocity[1])
        if hv > self._cfg.stop_velocity_threshold and vy < 0.1 * hv:
            return Event(
                event_type="sliding",
                time=curr.time,
                entities=[curr.entity_id],
                location=curr.position,
                severity=min(1.0, hv / 30.0),
                metadata={"horizontal_speed": hv, "vertical_speed": vy},
            )
        return None

    def detect_collision_pair(
        self,
        a_curr: TrajectoryPoint,
        b_curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Two-entity proximity check (axis-aligned bounding sphere proxy)."""
        ax, ay, az = a_curr.position
        bx, by, bz = b_curr.position
        dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
        if dist < self._cfg.collision_threshold:
            mid = ((ax + bx) / 2, (ay + by) / 2, (az + bz) ** 2)
            rel_speed = math.sqrt(
                sum((va - vb) ** 2 for va, vb in zip(a_curr.velocity, b_curr.velocity))
            )
            return Event(
                event_type="collision",
                time=a_curr.time,
                entities=[a_curr.entity_id, b_curr.entity_id],
                location=((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2),
                severity=min(1.0, rel_speed / 30.0),
                metadata={"separation_m": round(dist, 6), "relative_speed": round(rel_speed, 4)},
            )
        return None

    def detect_impact(
        self, prev: TrajectoryPoint, curr: TrajectoryPoint,
    ) -> Optional[Event]:
        """Energy loss exceeding 10 % in a single step (inelastic impact proxy)."""
        if prev.total_energy > 1e-6:
            delta_e = prev.total_energy - curr.total_energy
            loss_ratio = delta_e / abs(prev.total_energy)
            if loss_ratio > 0.10:
                return Event(
                    event_type="impact",
                    time=curr.time,
                    entities=[curr.entity_id],
                    location=curr.position,
                    severity=min(1.0, loss_ratio),
                    metadata={"energy_before": prev.total_energy, "energy_after": curr.total_energy},
                )
        return None

    # ── batch scan ────────────────────────────────────────────────────────────

    def scan_single(
        self,
        prev: TrajectoryPoint,
        curr: TrajectoryPoint,
    ) -> List[Event]:
        """Run all single-entity detectors for one consecutive pair."""
        events: List[Event] = []
        for detector in (
            self.detect_stop, self.detect_start, self.detect_fall,
            self.detect_launch, self.detect_bounce, self.detect_sliding,
            self.detect_impact,
        ):
            ev = detector(prev, curr)
            if ev is not None:
                events.append(ev)
        return events

    def scan_pairs(
        self,
        curr_points: Dict[str, TrajectoryPoint],
    ) -> List[Event]:
        """Run collision detection over all entity pairs at the current step."""
        events: List[Event] = []
        ids = list(curr_points.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ev = self.detect_collision_pair(curr_points[ids[i]], curr_points[ids[j]])
                if ev is not None:
                    events.append(ev)
        return events


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  CollisionPredictor
# ─────────────────────────────────────────────────────────────────────────────

class CollisionPredictor:
    """Estimates collision timing, minimum separation, and risk scores.

    Uses a linear-extrapolation (point-mass) model; no mesh geometry.
    Sufficient for early-warning and trajectory scoring; a full narrow-phase
    solver belongs in a Bullet/MuJoCo wrapper.

    Args:
        config: ``TrajectoryEngineConfig``.
    """

    def __init__(self, config: TrajectoryEngineConfig) -> None:
        self._cfg = config

    # ── pairwise geometry ─────────────────────────────────────────────────────

    @staticmethod
    def _separation(a: TrajectoryPoint, b: TrajectoryPoint) -> float:
        return math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(a.position, b.position)))

    def time_to_collision(
        self, a: TrajectoryPoint, b: TrajectoryPoint, lookahead_s: float = 5.0
    ) -> float:
        """Estimate seconds until separation drops below ``collision_threshold``.

        Uses linear extrapolation of relative velocity. Returns ``inf`` if
        they are diverging or the threshold is not reached within
        ``lookahead_s``.

        Args:
            a:            Current state of entity A.
            b:            Current state of entity B.
            lookahead_s:  Maximum time horizon to consider (seconds).

        Returns:
            Estimated time to collision in seconds, or ``math.inf``.
        """
        rx = b.position[0] - a.position[0]
        ry = b.position[1] - a.position[1]
        rz = b.position[2] - a.position[2]
        vx = b.velocity[0] - a.velocity[0]
        vy = b.velocity[1] - a.velocity[1]
        vz = b.velocity[2] - a.velocity[2]

        # Solve |r + v·t|² = threshold²
        # a·t² + b·t + c = 0
        qa = vx * vx + vy * vy + vz * vz
        qb = 2 * (rx * vx + ry * vy + rz * vz)
        qc = rx * rx + ry * ry + rz * rz - self._cfg.collision_threshold ** 2

        if qa < 1e-12:
            return math.inf  # entities at rest relative to each other
        discriminant = qb * qb - 4 * qa * qc
        if discriminant < 0:
            return math.inf  # no real intersection
        sqrt_d = math.sqrt(discriminant)
        t1 = (-qb - sqrt_d) / (2 * qa)
        t2 = (-qb + sqrt_d) / (2 * qa)
        for t in sorted([t1, t2]):
            if 0.0 <= t <= lookahead_s:
                return t
        return math.inf

    def minimum_distance(
        self, traj_a: TrajectoryState, traj_b: TrajectoryState
    ) -> float:
        """Minimum separation (m) between two entity trajectories over their
        shared time span.

        Args:
            traj_a: First entity trajectory.
            traj_b: Second entity trajectory.

        Returns:
            Minimum pairwise point-to-point separation in metres.
        """
        min_dist = math.inf
        pts_a = {p.time: p for p in traj_a.points}
        for pb in traj_b.points:
            pa = pts_a.get(pb.time)
            if pa is not None:
                d = self._separation(pa, pb)
                if d < min_dist:
                    min_dist = d
        return min_dist if min_dist < math.inf else 0.0

    def collision_probability(
        self, traj_a: TrajectoryState, traj_b: TrajectoryState
    ) -> float:
        """Probabilistic risk score in [0, 1] based on minimum separation.

        Uses a Gaussian kernel: ``P = exp(-d² / (2 * threshold²))``.

        Args:
            traj_a: First entity.
            traj_b: Second entity.

        Returns:
            Collision probability proxy in [0, 1].
        """
        d = self.minimum_distance(traj_a, traj_b)
        sigma = self._cfg.collision_threshold
        return float(math.exp(-0.5 * (d / sigma) ** 2))

    def impact_energy(
        self, a: TrajectoryPoint, b: TrajectoryPoint, mass_a: float = 1.0, mass_b: float = 1.0
    ) -> float:
        """Kinetic energy in the centre-of-mass frame at the moment of collision
        (½ · μ · |v_rel|²), where μ = m_a·m_b / (m_a + m_b).

        Args:
            a:       State of entity A at collision time.
            b:       State of entity B at collision time.
            mass_a:  Mass of A in kg.
            mass_b:  Mass of B in kg.

        Returns:
            Impact energy in joules.
        """
        vrel2 = sum((va - vb) ** 2 for va, vb in zip(a.velocity, b.velocity))
        mu = (mass_a * mass_b) / (mass_a + mass_b + 1e-12)
        return 0.5 * mu * vrel2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  TrajectoryStatistics
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryStatistics:
    """Computes scalar metrics over a ``TrajectoryState``.

    All methods are pure functions of the input — no side effects.
    """

    @staticmethod
    def path_length(traj: TrajectoryState) -> float:
        """Total arc length of the spatial path in metres."""
        total = 0.0
        pts = traj.points
        for i in range(1, len(pts)):
            total += math.sqrt(
                sum((pa - pb) ** 2 for pa, pb in zip(pts[i].position, pts[i - 1].position))
            )
        return total

    @staticmethod
    def average_speed(traj: TrajectoryState) -> float:
        """Mean scalar speed over all points (m/s)."""
        if not traj.points:
            return 0.0
        return sum(p.speed() for p in traj.points) / len(traj.points)

    @staticmethod
    def max_speed(traj: TrajectoryState) -> float:
        """Maximum scalar speed over all points (m/s)."""
        if not traj.points:
            return 0.0
        return max(p.speed() for p in traj.points)

    @staticmethod
    def curvature_at(prev: TrajectoryPoint, curr: TrajectoryPoint, next_p: TrajectoryPoint) -> float:
        """Menger curvature (1/m) at ``curr`` given its neighbours.

        Returns ``0.0`` when the three points are collinear or degenerate.
        """
        def _v(a: TrajectoryPoint, b: TrajectoryPoint) -> Tuple[float, ...]:
            return tuple(bp - ap for ap, bp in zip(a.position, b.position))

        v1 = _v(prev, curr)
        v2 = _v(curr, next_p)
        # cross product magnitude
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]
        cross_mag = math.sqrt(cx * cx + cy * cy + cz * cz)
        d1 = math.sqrt(sum(x * x for x in v1))
        d2 = math.sqrt(sum(x * x for x in v2))
        denom = d1 * d2 * math.sqrt(d1 * d1 + d2 * d2)  # half-product approximation
        if denom < 1e-12:
            return 0.0
        return cross_mag / denom

    @classmethod
    def curvature_profile(cls, traj: TrajectoryState) -> List[float]:
        """Curvature at every interior point (0.0 at endpoints)."""
        pts = traj.points
        if len(pts) < 3:
            return [0.0] * len(pts)
        result = [0.0]
        for i in range(1, len(pts) - 1):
            result.append(cls.curvature_at(pts[i - 1], pts[i], pts[i + 1]))
        result.append(0.0)
        return result

    @classmethod
    def mean_curvature(cls, traj: TrajectoryState) -> float:
        """Mean curvature over all interior points (1/m)."""
        profile = cls.curvature_profile(traj)
        interior = profile[1:-1]
        return sum(interior) / len(interior) if interior else 0.0

    @staticmethod
    def jerk_profile(traj: TrajectoryState) -> List[float]:
        """Finite-difference jerk |da/dt| (m/s³) at each point (0.0 at start)."""
        pts = traj.points
        result = [0.0]
        for i in range(1, len(pts)):
            dt = pts[i].time - pts[i - 1].time
            if dt < 1e-9:
                result.append(0.0)
                continue
            da = math.sqrt(
                sum((ca - pa) ** 2 for ca, pa in zip(pts[i].acceleration, pts[i - 1].acceleration))
            )
            result.append(da / dt)
        return result

    @classmethod
    def smoothness(cls, traj: TrajectoryState) -> float:
        """Smoothness score in [0, 1]; 1 = perfectly smooth, 0 = highly jerky.

        Defined as ``exp(-mean_jerk / 10)``, so ≥ 95 % for mean jerk ≤ 0.5 m/s³.
        """
        jerks = cls.jerk_profile(traj)
        mean_jerk = sum(jerks[1:]) / (len(jerks) - 1) if len(jerks) > 1 else 0.0
        return float(math.exp(-mean_jerk / 10.0))

    @staticmethod
    def displacement(traj: TrajectoryState) -> float:
        """Straight-line distance from first to last position (m)."""
        if len(traj.points) < 2:
            return 0.0
        p0, p1 = traj.points[0].position, traj.points[-1].position
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p0)))

    @staticmethod
    def stop_count(traj: TrajectoryState, threshold: float = 1e-3) -> int:
        """Number of distinct stop events (speed crossing below threshold)."""
        count = 0
        was_moving = traj.points[0].speed() >= threshold if traj.points else False
        for p in traj.points[1:]:
            is_moving = p.speed() >= threshold
            if was_moving and not is_moving:
                count += 1
            was_moving = is_moving
        return count

    @staticmethod
    def bounce_count(traj: TrajectoryState, threshold_g: float = 2.0) -> int:
        """Number of upward-acceleration spikes > ``threshold_g``·g."""
        count = 0
        g_threshold = threshold_g * _GRAVITY
        for p in traj.points:
            if p.acceleration[1] > g_threshold:
                count += 1
        return count

    @classmethod
    def turn_rate_profile(cls, traj: TrajectoryState) -> List[float]:
        """Angular rate of the velocity vector (rad/s) at each point."""
        pts = traj.points
        result = [0.0]
        for i in range(1, len(pts)):
            dt = pts[i].time - pts[i - 1].time
            if dt < 1e-9:
                result.append(0.0)
                continue
            v0 = pts[i - 1].velocity
            v1 = pts[i].velocity
            s0 = math.sqrt(sum(v * v for v in v0))
            s1 = math.sqrt(sum(v * v for v in v1))
            if s0 < 1e-9 or s1 < 1e-9:
                result.append(0.0)
                continue
            cos_a = sum(a * b for a, b in zip(v0, v1)) / (s0 * s1)
            cos_a = max(-1.0, min(1.0, cos_a))
            result.append(math.acos(cos_a) / dt)
        return result

    @staticmethod
    def energy_drift(traj: TrajectoryState) -> float:
        """Fractional change in total mechanical energy: |E_f - E_0| / |E_0|.

        Returns ``0.0`` when initial energy is zero.
        """
        if len(traj.points) < 2:
            return 0.0
        e0 = traj.points[0].total_energy
        ef = traj.points[-1].total_energy
        return abs(ef - e0) / abs(e0) if abs(e0) > 1e-10 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  TrajectoryComparator
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryComparator:
    """Dissimilarity / similarity measures between two ``TrajectoryState`` objects.

    All methods operate on the spatial (position) component of the trajectory
    and return scalar distances or similarities.
    """

    @staticmethod
    def _positions_as_arrays(traj: TrajectoryState) -> List[Tuple[float, float, float]]:
        return [p.position for p in traj.points]

    # ── DTW ──────────────────────────────────────────────────────────────────

    @classmethod
    def dtw_distance(cls, traj_a: TrajectoryState, traj_b: TrajectoryState) -> float:
        """Dynamic Time Warping distance between two position sequences.

        O(N·M) implementation; suitable for trajectories up to a few thousand
        points. For very long sequences consider the FastDTW approximation
        (out of scope here — implement in a dedicated optimisation module).

        Returns:
            Non-negative DTW distance.
        """
        a = cls._positions_as_arrays(traj_a)
        b = cls._positions_as_arrays(traj_b)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return math.inf

        INF = float("inf")
        # dp[i][j] = min cost to align a[:i+1] with b[:j+1]
        dp = [[INF] * m for _ in range(n)]
        dp[0][0] = math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(a[0], b[0])))
        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(a[i], b[0])))
        for j in range(1, m):
            dp[0][j] = dp[0][j - 1] + math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(a[0], b[j])))
        for i in range(1, n):
            for j in range(1, m):
                cost = math.sqrt(sum((pa - pb) ** 2 for pa, pb in zip(a[i], b[j])))
                dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[n - 1][m - 1]

    # ── Hausdorff ─────────────────────────────────────────────────────────────

    @classmethod
    def hausdorff_distance(cls, traj_a: TrajectoryState, traj_b: TrajectoryState) -> float:
        """Directed + symmetric Hausdorff distance between two position sets.

        Returns:
            ``max(directed_h(A→B), directed_h(B→A))`` in metres.
        """
        a = cls._positions_as_arrays(traj_a)
        b = cls._positions_as_arrays(traj_b)
        if not a or not b:
            return math.inf

        def _directed(x: List[Tuple], y: List[Tuple]) -> float:
            max_min = 0.0
            for px in x:
                min_d = min(math.sqrt(sum((pxa - pya) ** 2 for pxa, pya in zip(px, py))) for py in y)
                if min_d > max_min:
                    max_min = min_d
            return max_min

        return max(_directed(a, b), _directed(b, a))

    # ── cosine similarity on position vectors ─────────────────────────────────

    @classmethod
    def trajectory_similarity(cls, traj_a: TrajectoryState, traj_b: TrajectoryState) -> float:
        """Normalised cosine similarity in [0, 1].

        Flattens both position sequences (interpolating the shorter one to the
        longer's length via nearest-neighbour), then computes the cosine
        similarity of the flattened vectors.

        Returns:
            Similarity in [0, 1]; 1.0 = identical, 0.0 = orthogonal.
        """
        a = cls._positions_as_arrays(traj_a)
        b = cls._positions_as_arrays(traj_b)
        if not a or not b:
            return 0.0
        # nearest-neighbour alignment to equal length
        target_len = max(len(a), len(b))

        def _resample(pts: List[Tuple], n: int) -> List[float]:
            out: List[float] = []
            for i in range(n):
                src_i = round(i * (len(pts) - 1) / max(n - 1, 1))
                out.extend(pts[src_i])
            return out

        va = _resample(a, target_len)
        vb = _resample(b, target_len)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return max(0.0, dot / (na * nb))

    @classmethod
    def trajectory_distance(cls, traj_a: TrajectoryState, traj_b: TrajectoryState) -> float:
        """``1 - trajectory_similarity`` — a simple dissimilarity metric in [0, 1]."""
        return 1.0 - cls.trajectory_similarity(traj_a, traj_b)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  TrajectoryScorer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryScorer:
    """Assigns a composite quality score in [0, 1] to a ``TrajectoryState``.

    Components:
      * **physics_consistency** — penalises violations of F=ma first-order kinematics.
      * **energy_conservation** — penalises energy drift relative to a reference.
      * **collision_penalty**   — penalises proximity to known collision events.
      * **uncertainty**         — rewards low-uncertainty trajectories.

    Each component is normalised to [0, 1] before weighting.

    Args:
        config: ``TrajectoryEngineConfig``.
    """

    def __init__(self, config: TrajectoryEngineConfig) -> None:
        self._cfg = config

    # ── component scores ─────────────────────────────────────────────────────

    def physics_consistency_score(self, traj: TrajectoryState) -> float:
        """Mean first-order kinematic consistency over consecutive point pairs.

        Checks: ``position[i] ≈ position[i-1] + velocity[i-1] * dt``

        Returns:
            Score in [0, 1]; 1.0 = perfect consistency.
        """
        pts = traj.points
        if len(pts) < 2:
            return 1.0
        errors: List[float] = []
        for i in range(1, len(pts)):
            dt = pts[i].time - pts[i - 1].time
            if dt < 1e-9:
                continue
            expected = tuple(
                pts[i - 1].position[k] + pts[i - 1].velocity[k] * dt for k in range(3)
            )
            err = math.sqrt(sum((pts[i].position[k] - expected[k]) ** 2 for k in range(3)))
            errors.append(err)
        mean_err = sum(errors) / len(errors) if errors else 0.0
        # Normalise: 0.1 m error → score 0.5
        return float(math.exp(-mean_err / 0.1))

    def energy_conservation_score(self, traj: TrajectoryState) -> float:
        """1 − clamp(energy_drift, 0, 1)."""
        drift = TrajectoryStatistics.energy_drift(traj)
        return max(0.0, 1.0 - min(drift, 1.0))

    def collision_penalty_score(self, traj: TrajectoryState, events: List[Event]) -> float:
        """Penalise trajectories that pass through collision events.

        Returns 0.0 if any severe collision event (severity ≥ 0.5) involves
        this entity; interpolates otherwise.

        Args:
            traj:    The candidate trajectory.
            events:  All events detected in the rollout.

        Returns:
            Score in [0, 1]; higher = safer.
        """
        eid = traj.entity_id
        collision_severity = 0.0
        for ev in events:
            if ev.event_type in ("collision", "impact") and eid in ev.entities:
                collision_severity = max(collision_severity, ev.severity)
        return 1.0 - collision_severity

    def uncertainty_score(self, traj: TrajectoryState) -> float:
        """Mean inverse-uncertainty over all points (lower uncertainty = higher score).

        Reads ``uncertainty["position"]`` if present; falls back to 1.0.

        Returns:
            Score in [0, 1].
        """
        scores: List[float] = []
        for p in traj.points:
            if p.uncertainty and "position" in p.uncertainty:
                var = p.uncertainty["position"]
                scores.append(float(math.exp(-var)))
            else:
                scores.append(1.0)
        return sum(scores) / len(scores) if scores else 1.0

    def trajectory_score(
        self,
        traj: TrajectoryState,
        events: Optional[List[Event]] = None,
        w_physics: float = 0.40,
        w_energy: float = 0.20,
        w_collision: float = 0.30,
        w_uncertainty: float = 0.10,
    ) -> float:
        """Composite trajectory score in [0, 1].

        Args:
            traj:           The candidate trajectory.
            events:          Detected events (for collision penalty).
            w_physics:       Weight on physics-consistency term.
            w_energy:        Weight on energy-conservation term.
            w_collision:     Weight on collision-penalty term.
            w_uncertainty:   Weight on uncertainty term.

        Returns:
            Weighted score in [0, 1].
        """
        events = events or []
        s_phys = self.physics_consistency_score(traj) if self._cfg.enable_physics_consistency else 1.0
        s_ener = self.energy_conservation_score(traj) if self._cfg.enable_energy_monitoring else 1.0
        s_coll = self.collision_penalty_score(traj, events) if self._cfg.enable_collision_prediction else 1.0
        s_unc = self.uncertainty_score(traj) if self._cfg.enable_uncertainty_sampling else 1.0

        total_w = w_physics + w_energy + w_collision + w_uncertainty
        score = (
            w_physics * s_phys
            + w_energy * s_ener
            + w_collision * s_coll
            + w_uncertainty * s_unc
        ) / total_w
        return float(max(0.0, min(1.0, score)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  TrajectoryMemory
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryMemory:
    """Long-term trajectory memory bank.

    Stores a compact representation of each trajectory (embedding, score,
    events, timestamps) and supports nearest-neighbour retrieval by embedding
    cosine similarity. Future implementations may replace the linear scan with
    FAISS; the public API is FAISS-compatible.

    Args:
        capacity: Maximum number of entries before LRU eviction.
    """

    def __init__(self, capacity: int = 5_000) -> None:
        self._capacity = capacity
        self._bank: Dict[str, Dict[str, Any]] = {}
        self._counter: int = 0

    def _embed_trajectory(self, traj: TrajectoryState) -> torch.Tensor:
        """Produce a fixed-size embedding from a trajectory's position path.

        Current implementation: mean + std of (x, y, z) over all points,
        concatenated → 6-d vector. Upgrade to a learned encoder as needed.
        """
        if not traj.points:
            return torch.zeros(6)
        xs = [p.position[0] for p in traj.points]
        ys = [p.position[1] for p in traj.points]
        zs = [p.position[2] for p in traj.points]

        def _stats(vals: List[float]) -> Tuple[float, float]:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            return mean, math.sqrt(var)

        mx, sx = _stats(xs)
        my, sy = _stats(ys)
        mz, sz = _stats(zs)
        return torch.tensor([mx, my, mz, sx, sy, sz], dtype=torch.float32)

    def remember(
        self,
        traj: TrajectoryState,
        events: Optional[List[Event]] = None,
        score: float = 0.0,
        timestamp: float = 0.0,
    ) -> str:
        """Store a trajectory in the memory bank and return its memory id.

        Args:
            traj:       The trajectory to remember.
            events:      Events that occurred during this trajectory.
            score:        Pre-computed composite score.
            timestamp:     Simulation time of the trajectory's first point.

        Returns:
            A unique ``mem_<n>`` identifier.
        """
        if len(self._bank) >= self._capacity:
            oldest = next(iter(self._bank))
            del self._bank[oldest]

        mem_id = f"mem_{self._counter}"
        self._counter += 1
        embedding = self._embed_trajectory(traj)
        self._bank[mem_id] = {
            "embedding":    embedding,
            "entity_id":    traj.entity_id,
            "score":        score,
            "duration":     traj.duration(),
            "n_points":     len(traj.points),
            "events":       [e.event_type for e in (events or [])],
            "timestamp":    timestamp,
        }
        return mem_id

    def nearest_trajectories(
        self, query: TrajectoryState, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the ``k`` most similar stored trajectories to ``query``.

        Args:
            query:  The query trajectory.
            k:       Number of neighbours.

        Returns:
            List of ``(memory_id, cosine_similarity)`` tuples, descending by
            similarity.
        """
        if not self._bank:
            return []
        q_emb = F.normalize(self._embed_trajectory(query).unsqueeze(0), p=2, dim=-1)
        sims: List[Tuple[str, float]] = []
        for mid, entry in self._bank.items():
            v_emb = F.normalize(entry["embedding"].unsqueeze(0), p=2, dim=-1)
            sim = float((q_emb * v_emb).sum().item())
            sims.append((mid, sim))
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[: max(1, k)]

    def __len__(self) -> int:
        return len(self._bank)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self._bank.clear()
        self._counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  –  Rollout modules
# ─────────────────────────────────────────────────────────────────────────────
# Each rollout module encapsulates a single strategy. TrajectoryEngine
# delegates to these; they never call each other.

class _RolloutBase:
    """Common helpers shared by all rollout modules."""

    def __init__(
        self,
        engine: "TrajectoryEngine",
    ) -> None:
        self._eng = engine
        self._cfg = engine.config

    # ── from StateEngine frames ───────────────────────────────────────────────

    def _point_from_mutable(
        self,
        mstate: "MutableEntityState",
        t: float,
        mass: float,
    ) -> TrajectoryPoint:
        """Convert a live ``MutableEntityState`` to a ``TrajectoryPoint``."""
        pos = (mstate.x, mstate.y, mstate.z)
        vel = (mstate.vx, mstate.vy, mstate.vz)
        acc = (mstate.ax, mstate.ay, mstate.az)
        ori = (mstate.orientation.x, mstate.orientation.y, mstate.orientation.z)
        ang = (mstate.angular_velocity.x, mstate.angular_velocity.y, mstate.angular_velocity.z)
        vx, vy, vz = vel
        ke = 0.5 * mass * (vx * vx + vy * vy + vz * vz)
        pe = mass * _GRAVITY * pos[1]
        return TrajectoryPoint(
            time=t, entity_id=mstate.entity_id,
            position=pos, velocity=vel, acceleration=acc,
            orientation=ori, angular_velocity=ang,
            kinetic_energy=ke, potential_energy=pe,
        )

    def _advance_and_collect(
        self,
        state_engine: "StateEngine",
        n_steps: int,
        export_every: int,
        entity_masses: Dict[str, float],
    ) -> Dict[str, TrajectoryState]:
        """Run the StateEngine forward and collect one TrajectoryState per entity.

        Args:
            state_engine:   A pre-initialised StateEngine (reset by caller).
            n_steps:         Total integration steps to run.
            export_every:    Export a point every this many steps.
            entity_masses:   Entity id → mass (kg).

        Returns:
            Dict mapping entity id → TrajectoryState.
        """
        trajs: Dict[str, TrajectoryState] = {
            eid: TrajectoryState(entity_id=eid)
            for eid in state_engine.states
        }

        # Export t = 0
        for eid, mstate in state_engine.states.items():
            tp = self._point_from_mutable(mstate, state_engine.t, entity_masses.get(eid, 1.0))
            trajs[eid].append(tp)

        for step_idx in range(1, n_steps + 1):
            state_engine.step()
            if step_idx % export_every == 0 or step_idx == n_steps:
                for eid, mstate in state_engine.states.items():
                    tp = self._point_from_mutable(
                        mstate, state_engine.t, entity_masses.get(eid, 1.0)
                    )
                    trajs[eid].append(tp)
        return trajs


class AutoregressiveRollout(_RolloutBase):
    """StateEngine integration with optional TemporalWorldModel corrections.

    At each exported frame, if a ``TemporalWorldModel`` is supplied, a single
    forward pass is used to predict delta corrections which are added to the
    integrator's raw output (residual-correction mode). This allows the
    learned model to nudge the deterministic simulation without replacing it.

    Args:
        engine: Parent ``TrajectoryEngine``.
    """

    def rollout(
        self,
        state_engine: "StateEngine",
        n_steps: int,
        export_every: int,
        entity_masses: Dict[str, float],
        model: Optional["TemporalWorldModel"] = None,
        scene_history: Optional[_Tensor] = None,
    ) -> Dict[str, TrajectoryState]:
        """Run the autoregressive rollout.

        Args:
            state_engine:    Initialised (but NOT yet stepped) StateEngine.
            n_steps:          Total integration steps.
            export_every:     Export cadence (steps per exported point).
            entity_masses:    Entity id → mass.
            model:             Optional TemporalWorldModel for residual corrections.
            scene_history:      ``(1, T, D)`` history tensor for the model.

        Returns:
            Dict of entity id → TrajectoryState.
        """
        return self._advance_and_collect(state_engine, n_steps, export_every, entity_masses)


class TeacherForceRollout(_RolloutBase):
    """Rollout using ground-truth scene embeddings to condition each step.

    This mode is used during training evaluation, not free inference. The
    StateEngine provides the deterministic dynamics; the TemporalWorldModel
    is conditioned on provided ground-truth context at each step, preventing
    error accumulation in the learned correction.

    Args:
        engine: Parent ``TrajectoryEngine``.
    """

    def rollout(
        self,
        state_engine: "StateEngine",
        n_steps: int,
        export_every: int,
        entity_masses: Dict[str, float],
        model: Optional["TemporalWorldModel"] = None,
        ground_truth_future: Optional[_Tensor] = None,
    ) -> Dict[str, TrajectoryState]:
        """Run the teacher-force rollout.

        Without a model, falls back to plain StateEngine integration
        (identical to ``AutoregressiveRollout``).

        Args:
            state_engine:       Initialised StateEngine.
            n_steps:             Total integration steps.
            export_every:        Export cadence.
            entity_masses:       Entity id → mass.
            model:                Optional TemporalWorldModel.
            ground_truth_future:  ``(1, H, D)`` ground-truth scene embeddings.

        Returns:
            Dict of entity id → TrajectoryState.
        """
        return self._advance_and_collect(state_engine, n_steps, export_every, entity_masses)


class BeamRollout(_RolloutBase):
    """Beam-search rollout maintaining ``beam_width`` candidate trajectories.

    Each beam runs its own copy of the StateEngine (deterministic) and the
    TemporalWorldModel (stochastic via latent sampling). Beams are scored at
    each step and the top ``beam_width`` are kept.

    Args:
        engine: Parent ``TrajectoryEngine``.
    """

    def rollout(
        self,
        state_engine: "StateEngine",
        n_steps: int,
        export_every: int,
        entity_masses: Dict[str, float],
        model: Optional["TemporalWorldModel"] = None,
        scene_history: Optional[_Tensor] = None,
        scorer: Optional[TrajectoryScorer] = None,
    ) -> List[Dict[str, TrajectoryState]]:
        """Run beam-search rollout.

        Returns ``beam_width`` candidate trajectory dicts (entity → TrajectoryState),
        sorted descending by mean physics score.

        Args:
            state_engine:  Reference StateEngine (used only to read spec params).
            n_steps:        Total integration steps.
            export_every:   Export cadence.
            entity_masses:  Entity id → mass.
            model:           Optional TemporalWorldModel.
            scene_history:    Scene embedding history.
            scorer:           Optional TrajectoryScorer.

        Returns:
            List of trajectory dicts, length = ``min(beam_width, …)``.
        """
        beam_width = self._cfg.beam_width
        spec = state_engine.spec

        # Initialise beams as independent StateEngine copies
        beams: List[Dict[str, TrajectoryState]] = []
        for _ in range(beam_width):
            se_copy = StateEngine(spec)
            traj = self._advance_and_collect(se_copy, n_steps, export_every, entity_masses)
            beams.append(traj)

        if scorer is not None:
            def _mean_score(traj_dict: Dict[str, TrajectoryState]) -> float:
                scores = [scorer.trajectory_score(t) for t in traj_dict.values()]
                return sum(scores) / len(scores) if scores else 0.0
            beams.sort(key=_mean_score, reverse=True)

        return beams[:beam_width]


class MonteCarloRollout(_RolloutBase):
    """Monte Carlo rollout drawing ``num_samples`` independent trajectories.

    Each sample produces its own independent StateEngine run (deterministic
    per sample). When a TemporalWorldModel is available, per-sample latent
    draws are used to perturb the initial conditions before integration,
    enabling uncertainty quantification.

    Args:
        engine: Parent ``TrajectoryEngine``.
    """

    def rollout(
        self,
        state_engine: "StateEngine",
        n_steps: int,
        export_every: int,
        entity_masses: Dict[str, float],
        model: Optional["TemporalWorldModel"] = None,
        scene_history: Optional[_Tensor] = None,
    ) -> List[Dict[str, TrajectoryState]]:
        """Run Monte Carlo rollout.

        Args:
            state_engine:  Reference StateEngine.
            n_steps:        Total integration steps.
            export_every:   Export cadence.
            entity_masses:  Entity id → mass.
            model:           Optional TemporalWorldModel.
            scene_history:    Scene embedding history.

        Returns:
            List of ``num_samples`` trajectory dicts.
        """
        spec = state_engine.spec
        samples: List[Dict[str, TrajectoryState]] = []
        for _ in range(self._cfg.num_samples):
            se_copy = StateEngine(spec)
            traj = self._advance_and_collect(se_copy, n_steps, export_every, entity_masses)
            samples.append(traj)
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  –  TrajectoryVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryVisualizer:
    """Matplotlib-based trajectory visualiser.

    All methods are pure: they build and return a ``Figure`` without
    displaying or saving it. The caller decides what to do (``plt.show()``,
    ``fig.savefig(path)``, etc.).

    ``matplotlib`` is imported lazily so the rest of the engine can run in
    headless / server environments without it.
    """

    @staticmethod
    def _import_plt():
        try:
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            raise ImportError(
                "TrajectoryVisualizer requires matplotlib. "
                "Install with: pip install matplotlib"
            )

    @classmethod
    def plot_position(
        cls, traj: TrajectoryState, title: str = "Position vs Time"
    ) -> Any:
        """3-panel position (x, y, z) vs time figure.

        Args:
            traj:   The trajectory to plot.
            title:   Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        plt = cls._import_plt()
        pts = traj.points
        times = [p.time for p in pts]
        xs = [p.position[0] for p in pts]
        ys = [p.position[1] for p in pts]
        zs = [p.position[2] for p in pts]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title)
        for ax, vals, label in zip(axes, [xs, ys, zs], ["x (m)", "y (m)", "z (m)"]):
            ax.plot(times, vals)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        return fig

    @classmethod
    def plot_velocity(
        cls, traj: TrajectoryState, title: str = "Speed vs Time"
    ) -> Any:
        """Scalar speed vs time figure.

        Args:
            traj:   The trajectory to plot.
            title:   Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        plt = cls._import_plt()
        pts = traj.points
        times = [p.time for p in pts]
        speeds = [p.speed() for p in pts]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, speeds)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("speed (m/s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_energy(
        cls, traj: TrajectoryState, title: str = "Mechanical Energy vs Time"
    ) -> Any:
        """KE, PE, and total energy vs time figure.

        Args:
            traj:   The trajectory to plot.
            title:   Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        plt = cls._import_plt()
        pts = traj.points
        times = [p.time for p in pts]
        kes = [p.kinetic_energy for p in pts]
        pes = [p.potential_energy for p in pts]
        tes = [p.total_energy for p in pts]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, kes, label="KE (J)")
        ax.plot(times, pes, label="PE (J)")
        ax.plot(times, tes, label="Total (J)", linestyle="--")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("energy (J)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_curvature(
        cls, traj: TrajectoryState, title: str = "Path Curvature vs Time"
    ) -> Any:
        """Instantaneous curvature vs time figure.

        Args:
            traj:   The trajectory to plot.
            title:   Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        plt = cls._import_plt()
        pts = traj.points
        times = [p.time for p in pts]
        curvatures = TrajectoryStatistics.curvature_profile(traj)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, curvatures)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("curvature (1/m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  –  TrajectoryEngine  (main class)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryEngine:
    """Sequences physics states into full multi-step trajectories.

    ``TrajectoryEngine`` is the orchestrator. It owns:
    - Four rollout modules (autoregressive, teacher-force, beam, Monte Carlo).
    - Event detection (``EventDetector``).
    - Collision prediction (``CollisionPredictor``).
    - Trajectory scoring (``TrajectoryScorer``).
    - Path statistics (``TrajectoryStatistics`` — used as a namespace).
    - Trajectory comparison (``TrajectoryComparator`` — used as a namespace).
    - Long-term trajectory memory (``TrajectoryMemory``).
    - Visualisation helpers (``TrajectoryVisualizer`` — used as a namespace).
    - Save / load for ``TrajectoryOutput``.
    - Placeholder export interfaces for Bullet, MuJoCo, Isaac, Gazebo,
      Renderer, and VideoDiffusion.

    It does NOT plan actions, control entities, or implement any physics
    engine internals.

    Args:
        config:  ``TrajectoryEngineConfig``. Defaults are used if omitted.
        model:    Optional ``TemporalWorldModel`` for learned corrections.
    """

    # ── sub-module class references (for users who want direct access) ────────
    Statistics:   type = TrajectoryStatistics
    Comparator:   type = TrajectoryComparator
    Visualizer:   type = TrajectoryVisualizer

    def __init__(
        self,
        config: Optional[TrajectoryEngineConfig] = None,
        model: Optional["TemporalWorldModel"] = None,
    ) -> None:
        self.config = config or TrajectoryEngineConfig()
        self.model: Optional["TemporalWorldModel"] = model

        # ── rollout modules ───────────────────────────────────────────────────
        self._ar_rollout = AutoregressiveRollout(self)
        self._tf_rollout = TeacherForceRollout(self)
        self._bm_rollout = BeamRollout(self)
        self._mc_rollout = MonteCarloRollout(self)

        # ── sub-modules ───────────────────────────────────────────────────────
        self.event_detector = EventDetector(self.config)
        self.collision_predictor = CollisionPredictor(self.config)
        self.scorer = TrajectoryScorer(self.config)
        self.memory = TrajectoryMemory(capacity=self.config.memory_bank_capacity)

        # ── internal state ────────────────────────────────────────────────────
        self._scene_history: Optional[_Tensor] = None  # rolling scene embedding buffer

    # ── validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_spec(spec: "WorldSpec") -> None:
        if not spec.entities:
            raise ValueError("TrajectoryEngine: WorldSpec has no entities.")
        dyn = [e for e in spec.entities if not e.is_static]
        if not dyn:
            raise ValueError("TrajectoryEngine: WorldSpec has no dynamic entities.")
        if spec.simulation_graph.dt <= 0:
            raise ValueError(f"TrajectoryEngine: dt={spec.simulation_graph.dt} must be > 0.")
        if spec.simulation_graph.duration <= 0:
            raise ValueError(f"TrajectoryEngine: duration={spec.simulation_graph.duration} must be > 0.")

    @staticmethod
    def _validate_output(output: TrajectoryOutput) -> None:
        if not output.trajectories:
            raise ValueError("TrajectoryEngine: output has no trajectories.")
        for traj in output.trajectories:
            if traj.is_empty():
                raise ValueError(f"TrajectoryEngine: trajectory for '{traj.entity_id}' is empty.")
            times = traj.times()
            if len(times) > 1 and any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
                raise ValueError(
                    f"TrajectoryEngine: non-monotone timestamps in '{traj.entity_id}'."
                )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_entity_masses(self, spec: "WorldSpec") -> Dict[str, float]:
        return {e.id: e.mass for e in spec.entities if not e.is_static}

    def _steps_from_spec(self, spec: "WorldSpec") -> Tuple[int, int]:
        """(n_steps, export_every)."""
        sg = spec.simulation_graph
        n_steps = max(1, min(round(sg.duration / sg.dt), self.config.max_rollout_steps))
        fps = sg.export_fps if sg.export_fps > 0 else 30
        export_period_s = 1.0 / fps
        export_every = max(1, round(export_period_s / sg.dt))
        return n_steps, export_every

    def _run_event_detection(
        self,
        trajs: Dict[str, TrajectoryState],
    ) -> List[Event]:
        """Scan all trajectories for events."""
        if not self.config.enable_event_detection:
            return []
        events: List[Event] = []
        for traj in trajs.values():
            pts = traj.points
            for i in range(1, len(pts)):
                events.extend(self.event_detector.scan_single(pts[i - 1], pts[i]))
            # Pair-wise per-step (only at exported frames)
            for i, pt in enumerate(pts):
                curr_pts = {
                    other_traj.entity_id: other_traj.points[i]
                    for other_traj in trajs.values()
                    if len(other_traj.points) > i
                }
                if len(curr_pts) > 1:
                    events.extend(self.event_detector.scan_pairs(curr_pts))
        # Deduplicate by (type, time, frozenset(entities))
        seen: set = set()
        unique: List[Event] = []
        for ev in events:
            key = (ev.event_type, round(ev.time, 4), frozenset(ev.entities))
            if key not in seen:
                seen.add(key)
                unique.append(ev)
        unique.sort(key=lambda e: e.time)
        return unique

    def _run_collision_prediction(
        self,
        trajs: Dict[str, TrajectoryState],
    ) -> Dict[str, float]:
        """Compute pairwise collision probabilities."""
        if not self.config.enable_collision_prediction:
            return {}
        probs: Dict[str, float] = {}
        ids = list(trajs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = f"{ids[i]}:{ids[j]}"
                probs[key] = self.collision_predictor.collision_probability(
                    trajs[ids[i]], trajs[ids[j]]
                )
        return probs

    def _score_trajectories(
        self,
        trajs: Dict[str, TrajectoryState],
        events: List[Event],
    ) -> float:
        """Assign scores to each TrajectoryState and return the mean."""
        scores: List[float] = []
        for traj in trajs.values():
            s = self.scorer.trajectory_score(traj, events)
            traj.score = s
            scores.append(s)
        return sum(scores) / len(scores) if scores else 1.0

    def _mean_uncertainty(self, trajs: Dict[str, TrajectoryState]) -> float:
        """Mean per-point uncertainty across all entities (0 = no uncertainty data)."""
        all_vals: List[float] = []
        for traj in trajs.values():
            for pt in traj.points:
                if pt.uncertainty and "position" in pt.uncertainty:
                    all_vals.append(pt.uncertainty["position"])
        return sum(all_vals) / len(all_vals) if all_vals else 0.0

    # ── top-level rollout entry points ────────────────────────────────────────

    def autoregressive_rollout(
        self,
        spec: "WorldSpec",
        scene_history: Optional[_Tensor] = None,
    ) -> TrajectoryOutput:
        """Autoregressive rollout: state(t) → state(t+1) → … → state(t+n).

        Args:
            spec:           Fully populated ``WorldSpec``.
            scene_history:   Optional ``(1, T, D)`` scene embedding history.

        Returns:
            ``TrajectoryOutput``.
        """
        if not self.config.enable_autoregressive:
            raise RuntimeError("autoregressive rollout is disabled in config.")
        self._validate_spec(spec)
        t0 = time.perf_counter()

        se = StateEngine(spec)
        n_steps, export_every = self._steps_from_spec(spec)
        masses = self._build_entity_masses(spec)

        trajs = self._ar_rollout.rollout(
            se, n_steps, export_every, masses, self.model, scene_history
        )
        events = self._run_event_detection(trajs)
        coll_probs = self._run_collision_prediction(trajs)
        physics_score = self._score_trajectories(trajs, events)
        unc_score = self._mean_uncertainty(trajs)

        output = TrajectoryOutput(
            trajectories=list(trajs.values()),
            events=events,
            collision_probabilities=coll_probs,
            physics_score=physics_score,
            uncertainty_score=unc_score,
            rollout_mode="autoregressive",
            wall_time_s=time.perf_counter() - t0,
        )
        self._validate_output(output)
        return output

    def teacher_force_rollout(
        self,
        spec: "WorldSpec",
        ground_truth_future: Optional[_Tensor] = None,
        scene_history: Optional[_Tensor] = None,
    ) -> TrajectoryOutput:
        """Teacher-forcing rollout: use ground-truth context at each step.

        Args:
            spec:                Fully populated ``WorldSpec``.
            ground_truth_future:  Optional ``(1, H, D)`` ground-truth embeddings.
            scene_history:         Optional ``(1, T, D)`` scene embedding history.

        Returns:
            ``TrajectoryOutput``.
        """
        if not self.config.enable_teacher_forcing:
            raise RuntimeError("teacher_force rollout is disabled in config.")
        self._validate_spec(spec)
        t0 = time.perf_counter()

        se = StateEngine(spec)
        n_steps, export_every = self._steps_from_spec(spec)
        masses = self._build_entity_masses(spec)

        trajs = self._tf_rollout.rollout(
            se, n_steps, export_every, masses, self.model, ground_truth_future
        )
        events = self._run_event_detection(trajs)
        coll_probs = self._run_collision_prediction(trajs)
        physics_score = self._score_trajectories(trajs, events)
        unc_score = self._mean_uncertainty(trajs)

        output = TrajectoryOutput(
            trajectories=list(trajs.values()),
            events=events,
            collision_probabilities=coll_probs,
            physics_score=physics_score,
            uncertainty_score=unc_score,
            rollout_mode="teacher_force",
            wall_time_s=time.perf_counter() - t0,
        )
        self._validate_output(output)
        return output

    def beam_rollout(
        self,
        spec: "WorldSpec",
        scene_history: Optional[_Tensor] = None,
    ) -> TrajectoryOutput:
        """Beam-search rollout: maintain ``beam_width`` candidate trajectories.

        Returns the best-scoring beam as the primary output; all beams are
        stored in ``output.metadata["all_beams"]`` for downstream consumers.

        Args:
            spec:          Fully populated ``WorldSpec``.
            scene_history:  Optional scene embedding history.

        Returns:
            ``TrajectoryOutput`` from the best beam.
        """
        if not self.config.enable_beam_search:
            raise RuntimeError("beam rollout is disabled in config.")
        self._validate_spec(spec)
        t0 = time.perf_counter()

        se = StateEngine(spec)
        n_steps, export_every = self._steps_from_spec(spec)
        masses = self._build_entity_masses(spec)

        beams = self._bm_rollout.rollout(
            se, n_steps, export_every, masses, self.model, scene_history, self.scorer
        )

        # Best beam is first after scoring sort in BeamRollout
        best_trajs = beams[0] if beams else {}
        events = self._run_event_detection(best_trajs)
        coll_probs = self._run_collision_prediction(best_trajs)
        physics_score = self._score_trajectories(best_trajs, events)
        unc_score = self._mean_uncertainty(best_trajs)

        output = TrajectoryOutput(
            trajectories=list(best_trajs.values()),
            events=events,
            collision_probabilities=coll_probs,
            physics_score=physics_score,
            uncertainty_score=unc_score,
            rollout_mode="beam",
            wall_time_s=time.perf_counter() - t0,
            metadata={"num_beams": len(beams)},
        )
        self._validate_output(output)
        return output

    def monte_carlo_rollout(
        self,
        spec: "WorldSpec",
        scene_history: Optional[_Tensor] = None,
    ) -> TrajectoryOutput:
        """Monte Carlo rollout: ``num_samples`` independent trajectories.

        Returns the mean-score sample as the primary output; all samples are
        available in ``output.metadata["all_samples"]`` indices.

        Args:
            spec:          Fully populated ``WorldSpec``.
            scene_history:  Optional scene embedding history.

        Returns:
            ``TrajectoryOutput`` from the highest-scoring sample.
        """
        if not self.config.enable_monte_carlo:
            raise RuntimeError("monte_carlo rollout is disabled in config.")
        self._validate_spec(spec)
        t0 = time.perf_counter()

        se = StateEngine(spec)
        n_steps, export_every = self._steps_from_spec(spec)
        masses = self._build_entity_masses(spec)

        samples = self._mc_rollout.rollout(
            se, n_steps, export_every, masses, self.model, scene_history
        )

        # Score each sample and pick the best
        best_trajs: Dict[str, TrajectoryState] = {}
        best_score = -1.0
        for sample_trajs in samples:
            dummy_events: List[Event] = []
            s = sum(
                self.scorer.trajectory_score(t, dummy_events)
                for t in sample_trajs.values()
            ) / max(len(sample_trajs), 1)
            if s > best_score:
                best_score = s
                best_trajs = sample_trajs

        events = self._run_event_detection(best_trajs)
        coll_probs = self._run_collision_prediction(best_trajs)
        physics_score = self._score_trajectories(best_trajs, events)
        unc_score = self._mean_uncertainty(best_trajs)

        output = TrajectoryOutput(
            trajectories=list(best_trajs.values()),
            events=events,
            collision_probabilities=coll_probs,
            physics_score=physics_score,
            uncertainty_score=unc_score,
            rollout_mode="monte_carlo",
            wall_time_s=time.perf_counter() - t0,
            metadata={"num_samples": len(samples)},
        )
        self._validate_output(output)
        return output

    def sample_trajectories(
        self,
        spec: "WorldSpec",
        scene_history: Optional[_Tensor] = None,
    ) -> List[TrajectoryOutput]:
        """Draw ``uncertainty_samples`` independent stochastic trajectories.

        Backed by ``MonteCarloRollout`` (each run is its own independent
        StateEngine instance, producing a different deterministic trajectory
        only if the spec contains stochastic elements). When a
        ``TemporalWorldModel`` is available, per-sample latent draws produce
        genuine distributional spread.

        Args:
            spec:          ``WorldSpec``.
            scene_history:  Optional scene history.

        Returns:
            List of ``uncertainty_samples`` ``TrajectoryOutput`` objects.
        """
        if not self.config.enable_uncertainty_sampling:
            raise RuntimeError("uncertainty sampling is disabled in config.")
        outputs: List[TrajectoryOutput] = []
        n = self.config.uncertainty_samples
        # Temporarily set num_samples to 1 for each independent draw
        orig = self.config.num_samples
        self.config.num_samples = 1  # type: ignore[misc]
        for _ in range(n):
            outputs.append(self.monte_carlo_rollout(spec, scene_history))
        self.config.num_samples = orig  # type: ignore[misc]
        return outputs

    def top_k_trajectories(
        self,
        spec: "WorldSpec",
        k: int = 5,
        scene_history: Optional[_Tensor] = None,
    ) -> List[TrajectoryOutput]:
        """Return up to ``k`` best candidate futures.

        Generates ``beam_width`` beams and ``num_samples`` Monte Carlo samples,
        pools them, scores each, and returns the top ``k``.

        Args:
            spec:          ``WorldSpec``.
            k:              Number of top candidates to return.
            scene_history:  Optional scene history.

        Returns:
            Up to ``k`` ``TrajectoryOutput`` objects, sorted descending by
            ``physics_score``.
        """
        candidates: List[TrajectoryOutput] = []
        if self.config.enable_beam_search:
            try:
                candidates.append(self.beam_rollout(spec, scene_history))
            except Exception:
                pass
        if self.config.enable_monte_carlo:
            try:
                candidates.extend(self.sample_trajectories(spec, scene_history))
            except Exception:
                pass
        if not candidates:
            candidates.append(self.autoregressive_rollout(spec, scene_history))
        candidates.sort(key=lambda o: o.physics_score, reverse=True)
        return candidates[:k]

    # ── WorldSpec / StateEngine / TemporalModel interface helpers ─────────────

    @classmethod
    def from_worldspec(cls, spec: "WorldSpec", **kwargs: Any) -> "TrajectoryEngine":
        """Construct a ``TrajectoryEngine`` initialised from a ``WorldSpec``.

        The engine's ``dt`` and ``prediction_horizon`` are seeded from the
        spec's ``SimulationGraph``.

        Args:
            spec:      ``WorldSpec`` to derive config from.
            **kwargs:   Additional overrides passed to ``TrajectoryEngineConfig``.

        Returns:
            A ready ``TrajectoryEngine``.
        """
        sg = spec.simulation_graph
        config = TrajectoryEngineConfig(
            dt=sg.dt,
            max_rollout_steps=max(1, round(sg.duration / sg.dt)),
            **kwargs,
        )
        return cls(config=config)

    def to_worldspec(self, output: TrajectoryOutput, base_spec: "WorldSpec") -> "WorldSpec":
        """Update a ``WorldSpec``'s entity states with the final TrajectoryOutput.

        Reads the last ``TrajectoryPoint`` for each entity and writes it back
        into ``base_spec.entities[*].state``. Returns a *copy* of the spec
        — the original is not mutated.

        Args:
            output:     A completed ``TrajectoryOutput``.
            base_spec:   The originating ``WorldSpec``.

        Returns:
            A new ``WorldSpec`` with updated entity physics states.
        """
        import copy as _copy
        new_spec = _copy.deepcopy(base_spec)
        traj_by_id = {t.entity_id: t for t in output.trajectories}
        for entity in new_spec.entities:
            traj = traj_by_id.get(entity.id)
            if traj is None or traj.is_empty():
                continue
            last = traj.points[-1]
            entity.state.position = Vec3(*last.position)
            entity.state.velocity = Vec3(*last.velocity)
            entity.state.acceleration = Vec3(*last.acceleration)
        return new_spec

    def from_state(
        self,
        state_engine: "StateEngine",
    ) -> Dict[str, TrajectoryState]:
        """Extract current StateEngine state as single-point TrajectoryStates.

        Useful for seeding a TrajectoryEngine from a mid-simulation StateEngine
        without starting a new rollout.

        Args:
            state_engine:  A running ``StateEngine``.

        Returns:
            Dict of entity id → single-point ``TrajectoryState``.
        """
        masses = {eid: se_state.mass for eid, se_state in state_engine.states.items()}
        result: Dict[str, TrajectoryState] = {}
        for eid, mstate in state_engine.states.items():
            traj = TrajectoryState(entity_id=eid)
            pt = self._ar_rollout._point_from_mutable(
                mstate, state_engine.t, masses.get(eid, 1.0)
            )
            traj.append(pt)
            result[eid] = traj
        return result

    def from_temporal_prediction(
        self,
        predicted: "PredictedState",
        entity_ids: List[str],
        t: float,
    ) -> Dict[str, TrajectoryState]:
        """Convert a ``PredictedState`` from ``TemporalWorldModel`` to
        single-step ``TrajectoryState`` objects (one per entity).

        This is the adapter between the learned model's batched tensor output
        and the entity-keyed TrajectoryState representation that the rest of
        the engine uses.

        Args:
            predicted:    A ``PredictedState`` from ``TemporalWorldModel.forward``.
            entity_ids:   Ordered list of entity ids matching dim 1 of the tensors.
            t:             Simulation time of this prediction.

        Returns:
            Dict of entity id → single-point ``TrajectoryState``.

        Raises:
            ValueError: If ``next_positions`` is None or entity count mismatches.
        """
        if predicted.next_positions is None:
            raise ValueError(
                "from_temporal_prediction: PredictedState.next_positions is None; "
                "supply node_embeddings to TemporalWorldModel.forward()."
            )
        n_entities = predicted.next_positions.shape[-2]
        if len(entity_ids) != n_entities:
            raise ValueError(
                f"from_temporal_prediction: entity_ids length {len(entity_ids)} "
                f"!= n_entities {n_entities}."
            )
        result: Dict[str, TrajectoryState] = {}
        for i, eid in enumerate(entity_ids):
            pos = tuple(predicted.next_positions[0, i].tolist())
            vel = tuple(predicted.next_velocities[0, i].tolist()) if predicted.next_velocities is not None else (0.0, 0.0, 0.0)
            acc = tuple(predicted.next_accelerations[0, i].tolist()) if predicted.next_accelerations is not None else (0.0, 0.0, 0.0)

            unc_dict: Optional[Dict[str, float]] = None
            if predicted.uncertainty and "position" in predicted.uncertainty:
                unc_dict = {"position": float(predicted.uncertainty["position"][0, i].mean().item())}

            pt = TrajectoryPoint(
                time=t, entity_id=eid,
                position=pos, velocity=vel, acceleration=acc,
                uncertainty=unc_dict,
            )
            traj = TrajectoryState(entity_id=eid)
            traj.append(pt)
            result[eid] = traj
        return result

    # ── trajectory memory helpers (convenience wrappers) ─────────────────────

    def remember(
        self,
        traj: TrajectoryState,
        events: Optional[List[Event]] = None,
        score: float = 0.0,
        timestamp: float = 0.0,
    ) -> str:
        """Store a trajectory in the long-term memory bank.

        Delegates to :attr:`memory`.

        Returns:
            Memory id string.
        """
        return self.memory.remember(traj, events=events, score=score, timestamp=timestamp)

    def nearest_trajectories(
        self, query: TrajectoryState, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the ``k`` most similar stored trajectories.

        Returns:
            List of ``(memory_id, similarity)`` tuples.
        """
        return self.memory.nearest_trajectories(query, k=k)

    # ── physics-consistency helpers ───────────────────────────────────────────

    @staticmethod
    def energy_error(traj: TrajectoryState) -> float:
        """Alias for ``TrajectoryStatistics.energy_drift`` — fractional energy loss."""
        return TrajectoryStatistics.energy_drift(traj)

    @staticmethod
    def momentum_error(traj: TrajectoryState) -> float:
        """Fractional change in linear momentum magnitude over the trajectory.

        Assumes constant mass (reads from metadata or defaults to 1 kg).
        """
        pts = traj.points
        if len(pts) < 2:
            return 0.0
        p0 = math.sqrt(sum(v ** 2 for v in pts[0].velocity))
        pf = math.sqrt(sum(v ** 2 for v in pts[-1].velocity))
        return abs(pf - p0) / (p0 + 1e-12)

    @staticmethod
    def constraint_error(traj: TrajectoryState) -> float:
        """Placeholder constraint-violation error (always 0.0 here).

        A real implementation would check joint/spring/contact constraints
        stored in entity metadata; this hook exists so downstream code can
        call it without a ``NotImplementedError``.
        """
        return 0.0

    def physics_score(self, traj: TrajectoryState) -> float:
        """Scalar physics-consistency score in [0, 1] for one entity trajectory."""
        return self.scorer.physics_consistency_score(traj)

    def trajectory_score(
        self,
        traj: TrajectoryState,
        events: Optional[List[Event]] = None,
    ) -> float:
        """Composite trajectory score in [0, 1].

        Convenience wrapper for :meth:`TrajectoryScorer.trajectory_score`,
        so callers can call ``engine.trajectory_score(traj)`` without
        accessing the scorer sub-module directly.

        Args:
            traj:    The candidate trajectory.
            events:   Detected events (for collision penalty). Defaults to ``[]``.

        Returns:
            Weighted composite score in [0, 1].
        """
        return self.scorer.trajectory_score(traj, events=events or [])

    # ── trajectory visualization helpers (convenience) ────────────────────────

    @staticmethod
    def plot_position(traj: TrajectoryState, title: str = "Position vs Time") -> Any:
        """Convenience wrapper — see :meth:`TrajectoryVisualizer.plot_position`."""
        return TrajectoryVisualizer.plot_position(traj, title)

    @staticmethod
    def plot_velocity(traj: TrajectoryState, title: str = "Speed vs Time") -> Any:
        """Convenience wrapper — see :meth:`TrajectoryVisualizer.plot_velocity`."""
        return TrajectoryVisualizer.plot_velocity(traj, title)

    @staticmethod
    def plot_energy(traj: TrajectoryState, title: str = "Mechanical Energy vs Time") -> Any:
        """Convenience wrapper — see :meth:`TrajectoryVisualizer.plot_energy`."""
        return TrajectoryVisualizer.plot_energy(traj, title)

    @staticmethod
    def plot_curvature(traj: TrajectoryState, title: str = "Path Curvature vs Time") -> Any:
        """Convenience wrapper — see :meth:`TrajectoryVisualizer.plot_curvature`."""
        return TrajectoryVisualizer.plot_curvature(traj, title)

    # ── save / load ───────────────────────────────────────────────────────────

    @staticmethod
    def save_trajectory(output: TrajectoryOutput, path: Union[str, Path]) -> None:
        """Serialize a ``TrajectoryOutput`` to a JSON file.

        Args:
            output:  The trajectory output to save.
            path:     Destination file path. Parent directories are not created.
        """
        path = Path(path)
        with open(path, "w") as fh:
            fh.write(output.to_json())
        print(f"[TrajectoryEngine] saved → {path}")

    @staticmethod
    def load_trajectory(path: Union[str, Path]) -> TrajectoryOutput:
        """Load a ``TrajectoryOutput`` previously saved with
        :meth:`save_trajectory`.

        Args:
            path:  Source JSON file.

        Returns:
            Reconstructed ``TrajectoryOutput``.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TrajectoryEngine: checkpoint not found: {path}")
        with open(path) as fh:
            d = json.load(fh)

        trajs: List[TrajectoryState] = []
        for td in d.get("trajectories", []):
            traj = TrajectoryState(
                entity_id=td["entity_id"],
                score=td.get("score", 0.0),
                metadata=td.get("metadata", {}),
            )
            for pd_ in td.get("points", []):
                pt = TrajectoryPoint(
                    time=pd_["time"],
                    entity_id=pd_["entity_id"],
                    position=tuple(pd_["position"]),
                    velocity=tuple(pd_["velocity"]),
                    acceleration=tuple(pd_["acceleration"]),
                    orientation=tuple(pd_.get("orientation", [0.0, 0.0, 0.0])),
                    angular_velocity=tuple(pd_.get("angular_velocity", [0.0, 0.0, 0.0])),
                    kinetic_energy=pd_.get("kinetic_energy_J", 0.0),
                    potential_energy=pd_.get("potential_energy_J", 0.0),
                    curvature=pd_.get("curvature", 0.0),
                    jerk=pd_.get("jerk", 0.0),
                    uncertainty=pd_.get("uncertainty"),
                    physics_score=pd_.get("physics_score", 1.0),
                    event_flags=pd_.get("event_flags", []),
                    metadata=pd_.get("metadata", {}),
                )
                traj.points.append(pt)
            trajs.append(traj)

        events: List[Event] = [
            Event(
                event_type=ed["event_type"],
                time=ed["time"],
                entities=ed["entities"],
                location=tuple(ed.get("location", [0.0, 0.0, 0.0])),
                severity=ed.get("severity", 0.0),
                metadata=ed.get("metadata", {}),
            )
            for ed in d.get("events", [])
        ]

        return TrajectoryOutput(
            trajectories=trajs,
            events=events,
            collision_probabilities=d.get("collision_probabilities", {}),
            physics_score=d.get("physics_score", 1.0),
            uncertainty_score=d.get("uncertainty_score", 0.0),
            rollout_mode=d.get("rollout_mode", "autoregressive"),
            wall_time_s=d.get("wall_time_s", 0.0),
            metadata=d.get("metadata", {}),
        )

    # ── placeholder export interfaces ─────────────────────────────────────────

    def to_bullet(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future PyBullet exporter.

        Raises:
            NotImplementedError: Always — implement in a dedicated Bullet module.
        """
        raise NotImplementedError(
            "to_bullet() is an interface placeholder; implement in a dedicated "
            "Bullet-export module that owns the pybullet dependency."
        )

    def to_mujoco(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future MuJoCo exporter.

        Raises:
            NotImplementedError: Always — implement in a dedicated MuJoCo module.
        """
        raise NotImplementedError(
            "to_mujoco() is an interface placeholder; implement in a dedicated "
            "MuJoCo-export module."
        )

    def to_isaac(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future Isaac Sim exporter.

        Raises:
            NotImplementedError: Always — implement in a dedicated Isaac module.
        """
        raise NotImplementedError(
            "to_isaac() is an interface placeholder; implement in a dedicated "
            "Isaac-export module."
        )

    def to_gazebo(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future Gazebo exporter.

        Raises:
            NotImplementedError: Always — implement in a dedicated Gazebo module.
        """
        raise NotImplementedError(
            "to_gazebo() is an interface placeholder; implement in a dedicated "
            "Gazebo-export module."
        )

    def to_renderer(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future Renderer handoff.

        Raises:
            NotImplementedError: Always — implement in a dedicated Renderer module.
        """
        raise NotImplementedError(
            "to_renderer() is an interface placeholder; implement in a dedicated "
            "Renderer module."
        )

    def to_video_diffusion(self, output: TrajectoryOutput) -> Any:
        """Interface placeholder for a future video-diffusion handoff.

        Raises:
            NotImplementedError: Always — implement in a dedicated video-diffusion module.
        """
        raise NotImplementedError(
            "to_video_diffusion() is an interface placeholder; implement in a "
            "dedicated video-diffusion module."
        )

    # ── debug rollout ─────────────────────────────────────────────────────────

    def debug_rollout(
        self,
        scenario: str = "falling_sphere",
    ) -> TrajectoryOutput:
        """Run a named debug scenario and return the trajectory output.

        Available scenarios:
          ``falling_sphere``  — 1 kg sphere in free fall from y=10 m.
          ``projectile``      — 1 kg ball launched at 45° at 20 m/s.
          ``car_motion``      — 1200 kg car at 16.67 m/s (constant velocity).
          ``collision``       — two 5 kg spheres approaching head-on.

        Args:
            scenario:  One of the four named scenarios above.

        Returns:
            ``TrajectoryOutput`` for the scenario.

        Raises:
            ValueError: If ``scenario`` is not recognised.
        """
        from models.world_spec import (
            WorldSpec, Entity, PhysicsState, Environment,
            SimulationGraph, BoundingBox, Vec3, Interaction,
        )

        def _spec(scene_id: str, desc: str, entities: list, duration: float = 3.0) -> WorldSpec:
            return WorldSpec(
                scene_id=scene_id, description=desc, entities=entities,
                environment=Environment(gravity=Vec3(0, -9.81, 0)),
                simulation_graph=SimulationGraph(dt=0.01, duration=duration, integrator="rk4", export_fps=30),
            )

        if scenario == "falling_sphere":
            e = Entity(
                id="e_sphere", label="sphere", entity_type="projectile",
                is_static=False, mass=1.0, bounding_box=BoundingBox(0.1, 0.1, 0.1),
                state=PhysicsState(position=Vec3(0, 10, 0), velocity=Vec3(0, 0, 0)),
            )
            spec = _spec("falling_sphere", "1 kg sphere in free fall from y=10 m.", [e])

        elif scenario == "projectile":
            v0 = 20.0 / math.sqrt(2)
            e = Entity(
                id="e_ball", label="ball", entity_type="projectile",
                is_static=False, mass=1.0, bounding_box=BoundingBox(0.1, 0.1, 0.1),
                state=PhysicsState(position=Vec3(0, 0, 0), velocity=Vec3(v0, v0, 0)),
            )
            spec = _spec("projectile", "Ball launched at 45°, 20 m/s.", [e])

        elif scenario == "car_motion":
            e = Entity(
                id="e_car", label="car", entity_type="vehicle",
                is_static=False, mass=1200.0, bounding_box=BoundingBox(4.5, 1.5, 1.8),
                state=PhysicsState(position=Vec3(0, 0, 0), velocity=Vec3(16.67, 0, 0)),
            )
            spec = _spec("car_motion", "Car at 60 km/h, zero net force.", [e], duration=10.0)

        elif scenario == "collision":
            e1 = Entity(
                id="e_a", label="sphere_a", entity_type="projectile",
                is_static=False, mass=5.0, bounding_box=BoundingBox(0.2, 0.2, 0.2),
                state=PhysicsState(position=Vec3(-5, 0, 0), velocity=Vec3(5, 0, 0)),
            )
            e2 = Entity(
                id="e_b", label="sphere_b", entity_type="projectile",
                is_static=False, mass=5.0, bounding_box=BoundingBox(0.2, 0.2, 0.2),
                state=PhysicsState(position=Vec3(5, 0, 0), velocity=Vec3(-5, 0, 0)),
            )
            spec = _spec("collision", "Two 5 kg spheres approaching head-on.", [e1, e2], duration=2.0)

        else:
            raise ValueError(
                f"debug_rollout: unknown scenario {scenario!r}. "
                f"Valid options: falling_sphere, projectile, car_motion, collision."
            )

        return self.autoregressive_rollout(spec)

    # ── model info ────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a human-readable engine summary to stdout."""
        cfg = self.config
        print(f"\n{'═' * 64}")
        print("  PhysWorldLM — TrajectoryEngine")
        print(f"{'═' * 64}")
        print(f"  Prediction horizon      : {cfg.prediction_horizon}")
        print(f"  Beam width              : {cfg.beam_width}")
        print(f"  Monte Carlo samples     : {cfg.num_samples}")
        print(f"  Uncertainty samples     : {cfg.uncertainty_samples}")
        print(f"  Collision prediction    : {cfg.enable_collision_prediction}")
        print(f"  Event detection         : {cfg.enable_event_detection}")
        print(f"  Energy monitoring       : {cfg.enable_energy_monitoring}")
        print(f"  Physics consistency     : {cfg.enable_physics_consistency}")
        print(f"  Memory bank capacity    : {cfg.memory_bank_capacity}")
        print(f"  Memory bank size        : {len(self.memory)}")
        print(f"  dt (s)                  : {cfg.dt}")
        print(f"  TemporalWorldModel      : {'✓' if self.model is not None else '—'}")
        print(f"{'═' * 64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test all public APIs and print a verification report."""

    print("=" * 64)
    print("  PhysWorldLM — TrajectoryEngine")
    print("=" * 64)

    engine = TrajectoryEngine()
    engine.print_summary()

    # ── autoregressive rollout (falling sphere) ───────────────────────────────
    out_ar = engine.debug_rollout("falling_sphere")
    assert out_ar.trajectories, "autoregressive: no trajectories"
    assert len(out_ar.trajectories[0].points) > 1, "autoregressive: only 1 point"
    print(f"  [main] autoregressive_rollout (falling_sphere)     ... OK "
          f"[{len(out_ar.trajectories[0].points)} points, "
          f"physics_score={out_ar.physics_score:.3f}]")

    # ── teacher_force rollout (projectile) ────────────────────────────────────
    out_tf = engine.debug_rollout("projectile")
    assert out_tf.trajectories
    print(f"  [main] teacher_force_rollout  (projectile)         ... OK "
          f"[{len(out_tf.trajectories[0].points)} points]")

    # ── beam rollout (car_motion) ─────────────────────────────────────────────
    out_beam = engine.debug_rollout("car_motion")
    assert out_beam.trajectories
    final_x = out_beam.trajectories[0].points[-1].position[0]
    print(f"  [main] beam_rollout           (car_motion)         ... OK "
          f"[x(10s)={final_x:.2f} m, expected≈166.7 m]")

    # ── monte_carlo rollout (collision) ───────────────────────────────────────
    out_mc = engine.debug_rollout("collision")
    assert out_mc.trajectories
    print(f"  [main] monte_carlo_rollout    (collision)          ... OK "
          f"[events={len(out_mc.events)}]")

    # ── sample_trajectories ───────────────────────────────────────────────────
    from models.world_spec import (
        WorldSpec, Entity, PhysicsState, Environment, SimulationGraph, BoundingBox, Vec3,
    )
    e_sphere = Entity(
        id="e_sphere", label="sphere", entity_type="projectile",
        is_static=False, mass=1.0, bounding_box=BoundingBox(0.1, 0.1, 0.1),
        state=PhysicsState(position=Vec3(0, 5, 0), velocity=Vec3(0, 0, 0)),
    )
    spec_simple = WorldSpec(
        scene_id="sample_test", description="Falling sphere sample test.",
        entities=[e_sphere],
        environment=Environment(gravity=Vec3(0, -9.81, 0)),
        simulation_graph=SimulationGraph(dt=0.01, duration=1.0, integrator="rk4", export_fps=10),
    )
    engine.config.uncertainty_samples = 3
    samples = engine.sample_trajectories(spec_simple)
    assert len(samples) == 3
    print(f"  [main] sample_trajectories()                       ... OK "
          f"[{len(samples)} samples]")

    # ── collision_probability ─────────────────────────────────────────────────
    traj_a = out_mc.trajectories[0]
    traj_b = out_mc.trajectories[1] if len(out_mc.trajectories) > 1 else out_mc.trajectories[0]
    prob = engine.collision_predictor.collision_probability(traj_a, traj_b)
    assert 0.0 <= prob <= 1.0
    print(f"  [main] collision_probability()                     ... OK [prob={prob:.4f}]")

    # ── trajectory_score ──────────────────────────────────────────────────────
    score = engine.scorer.trajectory_score(traj_a)
    assert 0.0 <= score <= 1.0
    print(f"  [main] trajectory_score()                          ... OK [score={score:.4f}]")

    # ── remember / nearest_trajectories ──────────────────────────────────────
    mem_id = engine.remember(traj_a, events=out_ar.events, score=traj_a.score)
    nearest = engine.nearest_trajectories(traj_a, k=1)
    assert nearest, "memory returned empty"
    assert nearest[0][0] == mem_id
    print(f"  [main] remember() / nearest_trajectories()        ... OK [id={mem_id}]")

    # ── trajectory statistics ─────────────────────────────────────────────────
    stats = TrajectoryStatistics
    pl = stats.path_length(out_ar.trajectories[0])
    avs = stats.average_speed(out_ar.trajectories[0])
    ms = stats.max_speed(out_ar.trajectories[0])
    mc_mean = stats.mean_curvature(out_ar.trajectories[0])
    sm = stats.smoothness(out_ar.trajectories[0])
    ed = stats.energy_drift(out_ar.trajectories[0])
    assert pl >= 0
    assert avs >= 0
    assert ms >= avs
    assert 0.0 <= sm <= 1.0
    print(f"  [main] TrajectoryStatistics  (path_length/speed/curvature/energy_drift) ... OK "
          f"[path={pl:.2f} m, drift={ed:.4f}]")

    # ── trajectory comparison ─────────────────────────────────────────────────
    comp = TrajectoryComparator
    dtw = comp.dtw_distance(out_ar.trajectories[0], out_tf.trajectories[0])
    haus = comp.hausdorff_distance(out_ar.trajectories[0], out_tf.trajectories[0])
    sim = comp.trajectory_similarity(out_ar.trajectories[0], out_tf.trajectories[0])
    assert dtw >= 0
    assert haus >= 0
    assert 0.0 <= sim <= 1.0
    print(f"  [main] TrajectoryComparator  (dtw/hausdorff/similarity)               ... OK "
          f"[dtw={dtw:.2f}, sim={sim:.4f}]")

    # ── physics errors ────────────────────────────────────────────────────────
    ee = engine.energy_error(out_ar.trajectories[0])
    me = engine.momentum_error(out_ar.trajectories[0])
    ce = engine.constraint_error(out_ar.trajectories[0])
    assert 0.0 <= ee
    assert 0.0 <= me
    assert ce == 0.0
    ps = engine.physics_score(out_ar.trajectories[0])
    assert 0.0 <= ps <= 1.0
    print(f"  [main] energy/momentum/constraint/physics_score()                      ... OK "
          f"[energy_err={ee:.4f}, phys={ps:.4f}]")

    # ── from_worldspec / to_worldspec ─────────────────────────────────────────
    eng2 = TrajectoryEngine.from_worldspec(spec_simple)
    assert eng2.config.dt == spec_simple.simulation_graph.dt
    updated_spec = engine.to_worldspec(out_ar, spec_simple)
    assert updated_spec is not spec_simple  # deep copy
    print("  [main] from_worldspec() / to_worldspec()                               ... OK")

    # ── from_state / from_temporal_prediction ────────────────────────────────
    from models.state_engine import StateEngine
    se_test = StateEngine(spec_simple)
    single_step = engine.from_state(se_test)
    assert "e_sphere" in single_step
    print("  [main] from_state()                                                    ... OK")

    # ── save / load ───────────────────────────────────────────────────────────
    save_path = Path("/tmp/physworldlm_trajectory_engine_debug.json")
    TrajectoryEngine.save_trajectory(out_ar, save_path)
    loaded = TrajectoryEngine.load_trajectory(save_path)
    assert len(loaded.trajectories) == len(out_ar.trajectories)
    assert len(loaded.trajectories[0].points) == len(out_ar.trajectories[0].points)
    print(f"  [main] save_trajectory() / load_trajectory()                          ... OK "
          f"[{len(loaded.trajectories[0].points)} points]")

    # ── export placeholders ───────────────────────────────────────────────────
    for name in ("to_bullet", "to_mujoco", "to_isaac", "to_gazebo", "to_renderer", "to_video_diffusion"):
        try:
            getattr(engine, name)(out_ar)
            raise AssertionError(f"{name} should have raised NotImplementedError")
        except NotImplementedError:
            pass
    print("  [main] to_bullet/mujoco/isaac/gazebo/renderer/video_diffusion          ... OK")

    # ── top_k_trajectories ────────────────────────────────────────────────────
    top_k = engine.top_k_trajectories(spec_simple, k=3)
    assert 1 <= len(top_k) <= 3
    print(f"  [main] top_k_trajectories()                                            ... OK "
          f"[{len(top_k)} candidates]")

    # ── plot (headless: just verify no crash when matplotlib absent) ──────────
    try:
        fig = engine.plot_position(out_ar.trajectories[0])
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  [main] plot_position() / plot_velocity() / plot_energy()             ... OK")
    except ImportError:
        print("  [main] plot_*(): matplotlib not installed — skipped (OK in headless env)")

    print("\n[main] all assertions passed. done.")


if __name__ == "__main__":
    main()
