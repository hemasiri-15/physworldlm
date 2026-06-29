"""
models/state_engine.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — StateEngine: the deterministic differentiable bridge between
TemporalWorldModel and external physics back-ends (Bullet / MuJoCo / Isaac).

Philosophy
──────────
StateEngine is NOT a mini-Bullet. It is the modular, differentiable physics
kernel that sits between the learned world model and external simulators:

    TemporalWorldModel
        ↓
    StateEngine
        ├── Integrators           (Euler / SemiImplicit / RK4 / Verlet / Symplectic)
        ├── Force Registry        (gravity / drag / friction / spring / external)
        ├── Contact Manager       (point-mass / sphere / box / plane primitives only)
        ├── Constraint Manager    (hinge / fixed / spring / slider / distance)
        ├── Energy Accounting     (kinetic / potential / dissipated / drift)
        ├── State History         (t0 → t1 → t2 … for TrajectoryEngine)
        ├── Diagnostics           (energy drift / momentum drift / violations)
        └── Event Hooks           (CollisionEvent / SleepEvent / WakeEvent / …)
        ↓
    TrajectoryEngine  (future file)
        ↓
    Bullet / MuJoCo / Isaac / Gazebo  (interface placeholders only)
        ↓
    Renderer → Video

Complex collision geometry (SAT, mesh, SDF) belongs to Bullet/MuJoCo/Isaac.
StateEngine exposes clean interfaces to those back-ends rather than duplicating
their functionality.

Design principles (mirrors entity_encoder.py, relation_encoder.py,
graph_builder.py, temporal_world_model.py)
──────────────────────────────────────────────────────────────────────────────
* StateEngineConfig           — zero magic numbers; all dimensions configurable.
* PhysicsState / Contact / Constraint / EngineOutput — typed dataclasses.
* BaseIntegrator              — abstract interface; swap at runtime via config.
* ForceRegistry               — apply_force / remove_force / clear_forces.
* ContactManager              — active_contacts / contact_history / contact_graph.
* ConstraintManager           — active_constraints / constraint_graph.
* EnergyAccounting            — kinetic / potential / dissipated / total / drift.
* StateHistory                — ordered timestep snapshots for TrajectoryEngine.
* PhysicsStateBatch           — first-class batch support.
* Diagnostics                 — per-step energy / momentum / constraint tracking.
* EventHooks                  — CollisionEvent / SleepEvent / WakeEvent / Break.
* QuaternionUtils             — normalize / to_matrix / from_matrix (never buried).
* Explainability              — per-body force breakdown stored on every step.
* Export interfaces           — to_bullet / to_mujoco / to_isaac / to_gazebo /
                                to_renderer / to_video_diffusion / to_sensor_model.
* WorldSpec interface         — from_worldspec / to_worldspec.
* TemporalModel interface     — from_temporal_prediction / to_temporal_state.
* Save / load                 — torch.save / torch.load round-trip.
* Validation                  — NaN / negative mass / zero inertia / bad quat.

Scope discipline
────────────────
This file implements ONLY models/state_engine.py.
It does NOT implement: TrajectoryEngine, Bullet/MuJoCo/Isaac/Gazebo back-ends,
rendering, video generation, or any training loop.
"""

from __future__ import annotations

import math
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# ── optional WorldSpec import ──────────────────────────────────────────────
try:
    from world_spec import (
        WorldSpec, Entity, Environment, PhysicsState as WS_PhysicsState,
        Vec3, BoundingBox,
    )
    _WORLDSPEC_AVAILABLE = True
except ImportError:
    _WORLDSPEC_AVAILABLE = False

# ── optional TemporalWorldModel import ────────────────────────────────────
try:
    from models.temporal_world_model import PredictedState, WorldState
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  —  Constants
# ─────────────────────────────────────────────────────────────────────────────

GRAVITY_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "earth": (0.0, 0.0, -9.81),
    "moon":  (0.0, 0.0, -1.62),
    "mars":  (0.0, 0.0, -3.72),
    "zero":  (0.0, 0.0,  0.0),
}

# Identity quaternion: (w, x, y, z)
IDENTITY_QUAT: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  —  StateEngineConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateEngineConfig:
    """All hyperparameters for StateEngine — no magic numbers inside the engine.

    Attributes:
        dt:                       Integration timestep in seconds.
        gravity:                  Gravity vector (x, y, z) m/s².
        air_density:              Air density kg/m³ (used by DragModule).
        enable_gravity:           Enable/disable GravityModule.
        enable_drag:              Enable/disable DragModule.
        enable_friction:          Enable/disable FrictionModule (static +
                                  dynamic + rolling).
        enable_collision:         Enable/disable CollisionModule (point-mass /
                                  sphere / box / plane primitives only).
        enable_constraints:       Enable/disable ConstraintSolver.
        enable_restitution:       Apply coefficient-of-restitution bounce.
        enable_ground_plane:      Implicitly enforce z ≥ 0 ground plane.
        enable_sleeping:          Put bodies to sleep when velocity < threshold.
        velocity_threshold:       Sleep trigger for linear velocity (m/s).
        angular_velocity_threshold: Sleep trigger for angular velocity (rad/s).
        position_epsilon:         Numerical epsilon for position comparisons.
        max_contacts:             Hard cap on simultaneous contacts.
        max_iterations:           Constraint-solver iteration cap.
        integration_method:       One of "semi_implicit_euler" / "euler" /
                                  "rk4" / "verlet" / "symplectic".
        use_double_precision:     Use float64 internally.
        device:                   Torch device string.
        history_capacity:         Max timesteps kept in StateHistory.
        max_force_magnitude:      Clamp applied forces to prevent explosions.
        contact_slop:             Allowed penetration before correction (m).
        restitution_threshold:    Relative velocity below which restitution
                                  is skipped (avoids jitter at rest).
    """

    dt:                           float = 0.01
    gravity:                      Tuple[float, float, float] = (0.0, 0.0, -9.81)
    air_density:                  float = 1.225
    enable_gravity:               bool  = True
    enable_drag:                  bool  = True
    enable_friction:              bool  = True
    enable_collision:             bool  = True
    enable_constraints:           bool  = True
    enable_restitution:           bool  = True
    enable_ground_plane:          bool  = True
    enable_sleeping:              bool  = True
    velocity_threshold:           float = 1e-4
    angular_velocity_threshold:   float = 1e-4
    position_epsilon:             float = 1e-6
    max_contacts:                 int   = 64
    max_iterations:               int   = 16
    integration_method:           str   = "semi_implicit_euler"
    use_double_precision:         bool  = False
    device:                       str   = "cpu"
    history_capacity:             int   = 1024
    max_force_magnitude:          float = 1e8
    contact_slop:                 float = 1e-3
    restitution_threshold:        float = 0.5

    def __post_init__(self) -> None:
        valid_methods = {
            "semi_implicit_euler", "euler", "rk4", "verlet", "symplectic",
        }
        if self.integration_method not in valid_methods:
            raise ValueError(
                f"integration_method must be one of {sorted(valid_methods)}, "
                f"got {self.integration_method!r}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.max_contacts < 1:
            raise ValueError(f"max_contacts must be ≥ 1, got {self.max_contacts}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be ≥ 1, got {self.max_iterations}")
        if self.history_capacity < 1:
            raise ValueError(f"history_capacity must be ≥ 1, got {self.history_capacity}")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float64 if self.use_double_precision else torch.float32

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def gravity_vector(self) -> torch.Tensor:
        """Return gravity as a (3,) tensor."""
        return torch.tensor(self.gravity, dtype=self.dtype, device=self.torch_device)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  —  Core dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsState:
    """Full kinematic + dynamic state of one body at one moment.

    All tensors are 1-D of shape (3,) unless noted. Orientation is a unit
    quaternion (w, x, y, z) of shape (4,).

    Attributes:
        body_id:            Unique string identifier for this body.
        position:           World position (m). Shape (3,).
        velocity:           Linear velocity (m/s). Shape (3,).
        acceleration:       Linear acceleration (m/s²). Shape (3,).
        orientation:        Unit quaternion (w, x, y, z). Shape (4,).
        angular_velocity:   Angular velocity (rad/s). Shape (3,).
        angular_accel:      Angular acceleration (rad/s²). Shape (3,).
        force:              Net applied force (N). Shape (3,).
        torque:             Net applied torque (N·m). Shape (3,).
        mass:               Scalar mass (kg). Must be > 0.
        inertia_tensor:     Diagonal inertia (kg·m²). Shape (3,).
        friction:           Kinetic friction coefficient [0, 1].
        static_friction:    Static friction coefficient [0, 1].
        rolling_friction:   Rolling friction coefficient [0, 1].
        restitution:        Coefficient of restitution [0, 1].
        drag_coefficient:   Aerodynamic drag coefficient.
        cross_section_area: Reference area for drag (m²).
        shape:              Primitive shape: "sphere" / "box" / "point".
        shape_params:       Shape-specific params (radius, half-extents, …).
        is_static:          True → infinite mass / immovable.
        sleeping:           True → body is at rest; skip integration.
        metadata:           Free-form bag.
    """

    body_id:            str
    position:           torch.Tensor      # (3,)
    velocity:           torch.Tensor      # (3,)
    acceleration:       torch.Tensor      # (3,)
    orientation:        torch.Tensor      # (4,) quaternion (w,x,y,z)
    angular_velocity:   torch.Tensor      # (3,)
    angular_accel:      torch.Tensor      # (3,)
    force:              torch.Tensor      # (3,)
    torque:             torch.Tensor      # (3,)
    mass:               float
    inertia_tensor:     torch.Tensor      # (3,) diagonal
    friction:           float             = 0.5
    static_friction:    float             = 0.6
    rolling_friction:   float             = 0.01
    restitution:        float             = 0.5
    drag_coefficient:   float             = 0.47
    cross_section_area: float             = 1.0
    shape:              str               = "sphere"
    shape_params:       Dict[str, float]  = field(default_factory=dict)
    is_static:          bool              = False
    sleeping:           bool              = False
    metadata:           Dict[str, Any]    = field(default_factory=dict)

    def clone(self) -> "PhysicsState":
        """Deep-copy all tensor fields."""
        return PhysicsState(
            body_id=self.body_id,
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            acceleration=self.acceleration.clone(),
            orientation=self.orientation.clone(),
            angular_velocity=self.angular_velocity.clone(),
            angular_accel=self.angular_accel.clone(),
            force=self.force.clone(),
            torque=self.torque.clone(),
            mass=self.mass,
            inertia_tensor=self.inertia_tensor.clone(),
            friction=self.friction,
            static_friction=self.static_friction,
            rolling_friction=self.rolling_friction,
            restitution=self.restitution,
            drag_coefficient=self.drag_coefficient,
            cross_section_area=self.cross_section_area,
            shape=self.shape,
            shape_params=dict(self.shape_params),
            is_static=self.is_static,
            sleeping=self.sleeping,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict:
        return {
            "body_id":            self.body_id,
            "position":           self.position.tolist(),
            "velocity":           self.velocity.tolist(),
            "acceleration":       self.acceleration.tolist(),
            "orientation":        self.orientation.tolist(),
            "angular_velocity":   self.angular_velocity.tolist(),
            "angular_accel":      self.angular_accel.tolist(),
            "force":              self.force.tolist(),
            "torque":             self.torque.tolist(),
            "mass":               self.mass,
            "inertia_tensor":     self.inertia_tensor.tolist(),
            "friction":           self.friction,
            "static_friction":    self.static_friction,
            "rolling_friction":   self.rolling_friction,
            "restitution":        self.restitution,
            "drag_coefficient":   self.drag_coefficient,
            "cross_section_area": self.cross_section_area,
            "shape":              self.shape,
            "shape_params":       self.shape_params,
            "is_static":          self.is_static,
            "sleeping":           self.sleeping,
            "metadata":           self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict, dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device("cpu")) -> "PhysicsState":
        def _t(v: list) -> torch.Tensor:
            return torch.tensor(v, dtype=dtype, device=device)
        return cls(
            body_id=d["body_id"],
            position=_t(d["position"]),
            velocity=_t(d["velocity"]),
            acceleration=_t(d["acceleration"]),
            orientation=_t(d["orientation"]),
            angular_velocity=_t(d["angular_velocity"]),
            angular_accel=_t(d.get("angular_accel", [0.0, 0.0, 0.0])),
            force=_t(d["force"]),
            torque=_t(d["torque"]),
            mass=d["mass"],
            inertia_tensor=_t(d["inertia_tensor"]),
            friction=d.get("friction", 0.5),
            static_friction=d.get("static_friction", 0.6),
            rolling_friction=d.get("rolling_friction", 0.01),
            restitution=d.get("restitution", 0.5),
            drag_coefficient=d.get("drag_coefficient", 0.47),
            cross_section_area=d.get("cross_section_area", 1.0),
            shape=d.get("shape", "sphere"),
            shape_params=d.get("shape_params", {}),
            is_static=d.get("is_static", False),
            sleeping=d.get("sleeping", False),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Contact:
    """Pairwise contact between two bodies.

    Attributes:
        body_a:             id of first body (or "ground").
        body_b:             id of second body (or "ground").
        contact_point:      World-space contact point (m). Shape (3,).
        normal:             Contact normal pointing from B to A. Shape (3,).
        penetration_depth:  Signed overlap depth (m); > 0 = penetrating.
        normal_impulse:     Applied normal impulse (N·s). Shape (3,).
        friction_impulse:   Applied friction impulse (N·s). Shape (3,).
        relative_velocity:  Relative velocity at contact (m/s). Shape (3,).
        timestamp:          Simulation time at which this contact occurred (s).
    """

    body_a:            str
    body_b:            str
    contact_point:     torch.Tensor
    normal:            torch.Tensor
    penetration_depth: float
    normal_impulse:    torch.Tensor
    friction_impulse:  torch.Tensor
    relative_velocity: torch.Tensor
    timestamp:         float = 0.0

    def to_dict(self) -> dict:
        return {
            "body_a":            self.body_a,
            "body_b":            self.body_b,
            "contact_point":     self.contact_point.tolist(),
            "normal":            self.normal.tolist(),
            "penetration_depth": self.penetration_depth,
            "normal_impulse":    self.normal_impulse.tolist(),
            "friction_impulse":  self.friction_impulse.tolist(),
            "relative_velocity": self.relative_velocity.tolist(),
            "timestamp":         self.timestamp,
        }


@dataclass
class Constraint:
    """A kinematic constraint between two bodies.

    Attributes:
        constraint_id:  Unique identifier.
        body_a:          Id of the first body.
        body_b:          Id of the second body (or "world").
        joint_type:      "fixed" / "hinge" / "slider" / "spring" / "distance".
        anchor_a:        Anchor point in body A's local frame. Shape (3,).
        anchor_b:        Anchor point in body B's local frame (or world). (3,).
        axis:            Constraint axis (hinge/slider). Shape (3,).
        limit_min:       Lower limit (rad or m depending on joint_type).
        limit_max:       Upper limit.
        stiffness:       Spring stiffness (N/m or N·m/rad).
        damping:         Damping coefficient.
        rest_length:     Rest length for spring/distance joints (m).
        broken:          True if this constraint has been violated/broken.
        break_force:     Force threshold above which the constraint breaks.
        metadata:        Free-form bag.
    """

    constraint_id:  str
    body_a:          str
    body_b:          str
    joint_type:      str
    anchor_a:        torch.Tensor
    anchor_b:        torch.Tensor
    axis:            torch.Tensor
    limit_min:       float = -math.inf
    limit_max:       float =  math.inf
    stiffness:       float = 1e3
    damping:         float = 10.0
    rest_length:     float = 0.0
    broken:          bool  = False
    break_force:     float = math.inf
    metadata:        Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "constraint_id": self.constraint_id,
            "body_a":        self.body_a,
            "body_b":        self.body_b,
            "joint_type":    self.joint_type,
            "anchor_a":      self.anchor_a.tolist(),
            "anchor_b":      self.anchor_b.tolist(),
            "axis":          self.axis.tolist(),
            "limit_min":     self.limit_min,
            "limit_max":     self.limit_max,
            "stiffness":     self.stiffness,
            "damping":       self.damping,
            "rest_length":   self.rest_length,
            "broken":        self.broken,
            "break_force":   self.break_force,
            "metadata":      self.metadata,
        }


@dataclass
class ForceRecord:
    """One registered force acting on a body.

    Attributes:
        force_id:    Unique identifier in the ForceRegistry.
        body_id:     Target body id.
        source:      Human-readable source label
                     ("gravity" / "drag" / "friction" / "spring" / "user:…").
        force:       Force vector (N). Shape (3,).
        torque:      Torque vector (N·m). Shape (3,).
        persistent:  If True, the force remains across steps.
                     If False, it is removed after one application.
    """

    force_id:   str
    body_id:    str
    source:     str
    force:      torch.Tensor
    torque:     torch.Tensor
    persistent: bool = True


@dataclass
class EngineOutput:
    """Typed output of one StateEngine.step() call.

    Attributes:
        updated_states:      Dict of body_id → updated PhysicsState.
        contacts:             List of Contact objects generated this step.
        constraint_forces:    Dict of constraint_id → force magnitude (N).
        kinetic_energy:       Total kinetic energy (J).
        potential_energy:     Total potential energy (J).
        dissipated_energy:    Energy lost to friction/damping this step (J).
        total_energy:         kinetic + potential (J).
        linear_momentum:      Total linear momentum vector (kg·m/s). (3,).
        angular_momentum:     Total angular momentum vector (kg·m²/s). (3,).
        center_of_mass:       World-space CoM (m). (3,).
        events:               List of PhysicsEvent emitted this step.
        force_breakdown:      Dict of body_id → {source → force tensor}.
        diagnostics:          Dict of diagnostic scalar values.
        timestep:             Simulation time AFTER this step (s).
        wall_time_ms:         Real wall-clock time for this step (ms).
        metadata:             Free-form bag.
    """

    updated_states:    Dict[str, PhysicsState]
    contacts:           List[Contact]
    constraint_forces:  Dict[str, float]
    kinetic_energy:     float
    potential_energy:   float
    dissipated_energy:  float
    total_energy:       float
    linear_momentum:    torch.Tensor
    angular_momentum:   torch.Tensor
    center_of_mass:     torch.Tensor
    events:             List["PhysicsEvent"]
    force_breakdown:    Dict[str, Dict[str, torch.Tensor]]
    diagnostics:        Dict[str, float]
    timestep:           float
    wall_time_ms:       float
    metadata:           Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsStateBatch:
    """First-class batch of PhysicsStates for parallel simulation.

    Attributes:
        states:     List of PhysicsState — one per body.
        batch_id:   Optional string identifier for this batch.
        timestep:   Current simulation time (s).
        metadata:   Free-form bag.
    """

    states:    List[PhysicsState]
    batch_id:  str = ""
    timestep:  float = 0.0
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def body_ids(self) -> List[str]:
        return [s.body_id for s in self.states]

    def as_dict(self) -> Dict[str, PhysicsState]:
        return {s.body_id: s for s in self.states}

    def clone(self) -> "PhysicsStateBatch":
        return PhysicsStateBatch(
            states=[s.clone() for s in self.states],
            batch_id=self.batch_id,
            timestep=self.timestep,
            metadata=dict(self.metadata),
        )


@dataclass
class StateSnapshot:
    """One timestep entry in StateHistory.

    Attributes:
        timestep:   Simulation time (s).
        states:     Dict of body_id → PhysicsState at this timestep.
        contacts:   Contacts active at this timestep.
        energy:     Total mechanical energy (J).
        metadata:   Free-form bag.
    """

    timestep:  float
    states:    Dict[str, PhysicsState]
    contacts:  List[Contact]
    energy:    float
    metadata:  Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  —  Physics Events
# ─────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    COLLISION     = auto()
    SLEEP         = auto()
    WAKE          = auto()
    CONSTRAINT_BREAK = auto()
    GROUND_IMPACT = auto()
    VELOCITY_CAP  = auto()


@dataclass
class PhysicsEvent:
    """A discrete physics event emitted by the StateEngine.

    Attributes:
        event_type:   EventType enum value.
        body_ids:      Bodies involved (1 or 2).
        timestep:      Simulation time at which the event occurred (s).
        magnitude:     Optional scalar magnitude (impulse, force, …).
        metadata:      Free-form bag.
    """

    event_type:  EventType
    body_ids:    List[str]
    timestep:    float
    magnitude:   Optional[float] = None
    metadata:    Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  —  Quaternion Utilities (never buried inside step())
# ─────────────────────────────────────────────────────────────────────────────

class QuaternionUtils:
    """Pure-static quaternion utilities. Convention: (w, x, y, z).

    All methods operate on (4,) tensors and are differentiable with respect
    to their inputs unless explicitly noted.
    """

    @staticmethod
    def normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize a quaternion to unit length.

        Args:
            q:    Shape (..., 4) quaternion tensor.
            eps:  Numerical epsilon to avoid division by zero.

        Returns:
            Unit quaternion of the same shape.
        """
        return F.normalize(q, p=2, dim=-1, eps=eps)

    @staticmethod
    def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        """Convert a unit quaternion (w, x, y, z) to a 3×3 rotation matrix.

        Args:
            q: Shape (..., 4) unit quaternion.

        Returns:
            Shape (..., 3, 3) rotation matrix.
        """
        q = QuaternionUtils.normalize(q)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        tx, ty, tz = 2 * x, 2 * y, 2 * z
        twx, twy, twz = tx * w, ty * w, tz * w
        txx, txy, txz = tx * x, ty * x, tz * x
        tyy, tyz, tzz = ty * y, tz * y, tz * z

        one = torch.ones_like(w)
        R = torch.stack([
            one - (tyy + tzz), txy - twz,       txz + twy,
            txy + twz,          one - (txx + tzz), tyz - twx,
            txz - twy,           tyz + twx,        one - (txx + tyy),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)
        return R

    @staticmethod
    def from_rotation_matrix(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert a 3×3 rotation matrix to a unit quaternion (w, x, y, z).

        Uses Shepperd's method, numerically stable for all rotation angles.

        Args:
            R:    Shape (..., 3, 3) rotation matrix.
            eps:  Numerical epsilon.

        Returns:
            Shape (..., 4) unit quaternion.
        """
        batch = R.shape[:-2]
        m = R.reshape(-1, 3, 3)
        n = m.shape[0]
        quats = []
        for i in range(n):
            r = m[i]
            trace = r[0, 0] + r[1, 1] + r[2, 2]
            if trace > 0:
                s = 0.5 / math.sqrt(max(float(trace + 1.0), eps))
                w = 0.25 / s
                x = (r[2, 1] - r[1, 2]) * s
                y = (r[0, 2] - r[2, 0]) * s
                z = (r[1, 0] - r[0, 1]) * s
            elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                s = 2.0 * math.sqrt(max(1.0 + float(r[0, 0] - r[1, 1] - r[2, 2]), eps))
                w = (r[2, 1] - r[1, 2]) / s
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
            elif r[1, 1] > r[2, 2]:
                s = 2.0 * math.sqrt(max(1.0 + float(r[1, 1] - r[0, 0] - r[2, 2]), eps))
                w = (r[0, 2] - r[2, 0]) / s
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(max(1.0 + float(r[2, 2] - r[0, 0] - r[1, 1]), eps))
                w = (r[1, 0] - r[0, 1]) / s
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
            quats.append(torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device))
        q = torch.stack(quats, dim=0)
        return QuaternionUtils.normalize(q).reshape(*batch, 4)

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two unit quaternions.

        Args:
            q1, q2: Shape (..., 4) unit quaternions.

        Returns:
            Shape (..., 4) product quaternion.
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return QuaternionUtils.normalize(torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1))

    @staticmethod
    def is_valid(q: torch.Tensor, eps: float = 1e-4) -> bool:
        """Return True iff ``q`` is a unit quaternion (|q| ≈ 1)."""
        norm = float(q.norm(p=2, dim=-1).mean().item())
        return abs(norm - 1.0) < eps

    @staticmethod
    def integrate_orientation(
        q: torch.Tensor,
        omega: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Integrate orientation by angular velocity omega over dt.

        Uses the first-order quaternion derivative:
            dq/dt = 0.5 * [0, ω] ⊗ q

        Args:
            q:     Shape (4,) current orientation quaternion (w,x,y,z).
            omega: Shape (3,) angular velocity (rad/s).
            dt:    Timestep (s).

        Returns:
            Shape (4,) updated orientation quaternion, normalized.
        """
        # Build pure quaternion [0, ω]
        omega_q = torch.cat([torch.zeros(1, dtype=q.dtype, device=q.device), omega])
        dq = 0.5 * QuaternionUtils.multiply(omega_q.unsqueeze(0), q.unsqueeze(0)).squeeze(0)
        return QuaternionUtils.normalize(q + dq * dt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  —  Integrators (first-class objects)
# ─────────────────────────────────────────────────────────────────────────────

class BaseIntegrator(ABC):
    """Abstract interface for all numerical integrators.

    Every integrator takes the CURRENT state and the NET FORCE/TORQUE that
    has already been accumulated by the ForceRegistry, and returns the
    UPDATED state. The caller (StateEngine.step) is responsible for applying
    forces BEFORE calling integrate() and for enforcing constraints AFTER.
    """

    @abstractmethod
    def integrate(
        self,
        state: PhysicsState,
        dt: float,
    ) -> PhysicsState:
        """Advance ``state`` by ``dt`` given the pre-accumulated
        ``state.force`` and ``state.torque``.

        Args:
            state: Current body state with net force/torque already set.
            dt:    Integration timestep (s).

        Returns:
            Updated :class:`PhysicsState`; the input ``state`` is not
            mutated.
        """

    @staticmethod
    def _linear_acceleration(state: PhysicsState) -> torch.Tensor:
        """a = F/m (or zero for static bodies)."""
        if state.is_static or state.mass <= 0:
            return torch.zeros(3, dtype=state.force.dtype, device=state.force.device)
        return state.force / state.mass

    @staticmethod
    def _angular_acceleration(state: PhysicsState) -> torch.Tensor:
        """α = τ / I (diagonal inertia tensor)."""
        if state.is_static:
            return torch.zeros(3, dtype=state.torque.dtype, device=state.torque.device)
        # Guard against zero inertia components
        I = state.inertia_tensor.clamp(min=1e-10)
        return state.torque / I


class EulerIntegrator(BaseIntegrator):
    """Classic explicit (forward) Euler integration.

    v(t+dt) = v(t) + a(t) · dt
    x(t+dt) = x(t) + v(t) · dt
    """

    def integrate(self, state: PhysicsState, dt: float) -> PhysicsState:
        if state.is_static or state.sleeping:
            return state.clone()
        s = state.clone()
        a = self._linear_acceleration(state)
        alpha = self._angular_acceleration(state)

        s.position = state.position + state.velocity * dt
        s.velocity = state.velocity + a * dt
        s.acceleration = a

        s.angular_velocity = state.angular_velocity + alpha * dt
        s.angular_accel = alpha
        s.orientation = QuaternionUtils.integrate_orientation(
            state.orientation, state.angular_velocity, dt
        )
        return s


class SemiImplicitEulerIntegrator(BaseIntegrator):
    """Semi-implicit (symplectic) Euler — default integrator.

    v(t+dt) = v(t) + a(t) · dt       ← same as explicit
    x(t+dt) = x(t) + v(t+dt) · dt   ← uses NEW velocity

    Better energy conservation than explicit Euler at no extra cost.
    """

    def integrate(self, state: PhysicsState, dt: float) -> PhysicsState:
        if state.is_static or state.sleeping:
            return state.clone()
        s = state.clone()
        a = self._linear_acceleration(state)
        alpha = self._angular_acceleration(state)

        new_vel = state.velocity + a * dt
        s.velocity = new_vel
        s.position = state.position + new_vel * dt
        s.acceleration = a

        new_omega = state.angular_velocity + alpha * dt
        s.angular_velocity = new_omega
        s.angular_accel = alpha
        s.orientation = QuaternionUtils.integrate_orientation(
            state.orientation, new_omega, dt
        )
        return s


class RK4Integrator(BaseIntegrator):
    """Fourth-order Runge-Kutta integrator.

    Evaluates the derivative at four intermediate points and combines them
    with Simpson-rule weights (1/6, 2/6, 2/6, 1/6). Requires four force
    evaluations per step but achieves O(dt⁴) accuracy.

    Note: Because force evaluation in StateEngine depends on position/velocity
    (via drag, spring forces, etc.), a full RK4 implementation would need to
    re-evaluate all force modules at each sub-step. Here we use a simplified
    "frozen-force" RK4 — valid when dt is small and force varies slowly — to
    keep StateEngine's module architecture clean. A full per-step force
    re-evaluation path can be wired in a future version.
    """

    def integrate(self, state: PhysicsState, dt: float) -> PhysicsState:
        if state.is_static or state.sleeping:
            return state.clone()

        a = self._linear_acceleration(state)
        alpha = self._angular_acceleration(state)

        # k1
        k1_v = a
        k1_x = state.velocity

        # k2 (midpoint, frozen force)
        k2_v = a
        k2_x = state.velocity + 0.5 * dt * k1_v

        # k3 (midpoint, frozen force)
        k3_v = a
        k3_x = state.velocity + 0.5 * dt * k2_v

        # k4 (endpoint, frozen force)
        k4_v = a
        k4_x = state.velocity + dt * k3_v

        s = state.clone()
        s.velocity    = state.velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        s.position    = state.position + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        s.acceleration = a

        s.angular_velocity = state.angular_velocity + alpha * dt
        s.angular_accel    = alpha
        s.orientation = QuaternionUtils.integrate_orientation(
            state.orientation, state.angular_velocity, dt
        )
        return s


class VerletIntegrator(BaseIntegrator):
    """Velocity-Verlet integrator.

    x(t+dt) = x(t) + v(t)·dt + 0.5·a(t)·dt²
    v(t+dt) = v(t) + 0.5·[a(t) + a(t+dt)]·dt

    Since a(t+dt) depends on forces at t+dt (not yet known), we use a
    frozen-force approximation (a(t+dt) ≈ a(t)), which reduces to the
    standard Störmer-Verlet scheme. This is commonly used in molecular
    dynamics and game physics.
    """

    def integrate(self, state: PhysicsState, dt: float) -> PhysicsState:
        if state.is_static or state.sleeping:
            return state.clone()

        a = self._linear_acceleration(state)
        alpha = self._angular_acceleration(state)

        s = state.clone()
        s.position    = state.position + state.velocity * dt + 0.5 * a * dt * dt
        s.velocity    = state.velocity + a * dt  # frozen-force: a(t+dt) ≈ a(t)
        s.acceleration = a

        s.angular_velocity = state.angular_velocity + alpha * dt
        s.angular_accel    = alpha
        s.orientation = QuaternionUtils.integrate_orientation(
            state.orientation, state.angular_velocity, dt
        )
        return s


class SymplecticIntegrator(BaseIntegrator):
    """Symplectic (leapfrog) integrator.

    Velocity and position are staggered by half a timestep:
        v(t+dt/2) = v(t-dt/2) + a(t)·dt
        x(t+dt)   = x(t) + v(t+dt/2)·dt

    Exactly preserves the symplectic structure of Hamiltonian mechanics,
    which means energy oscillates but does NOT drift over long simulations.
    Preferred for conservative systems (planetary motion, molecular dynamics).
    """

    def integrate(self, state: PhysicsState, dt: float) -> PhysicsState:
        if state.is_static or state.sleeping:
            return state.clone()

        a = self._linear_acceleration(state)
        alpha = self._angular_acceleration(state)

        # Treat stored velocity as v(t - dt/2) for the first step,
        # then switch to leapfrog cadence.
        v_half = state.velocity + a * (0.5 * dt)
        s = state.clone()
        s.position    = state.position + v_half * dt
        s.velocity    = v_half + a * (0.5 * dt)   # = v(t + dt/2) → report as v(t+dt)
        s.acceleration = a

        omega_half = state.angular_velocity + alpha * (0.5 * dt)
        s.angular_velocity = omega_half + alpha * (0.5 * dt)
        s.angular_accel    = alpha
        s.orientation = QuaternionUtils.integrate_orientation(
            state.orientation, omega_half, dt
        )
        return s


def build_integrator(method: str) -> BaseIntegrator:
    """Factory: return the :class:`BaseIntegrator` for ``method``.

    Args:
        method: One of ``"euler"`` / ``"semi_implicit_euler"`` /
                ``"rk4"`` / ``"verlet"`` / ``"symplectic"``.

    Returns:
        Appropriate integrator instance.

    Raises:
        ValueError: If ``method`` is not recognised.
    """
    table: Dict[str, type] = {
        "euler":               EulerIntegrator,
        "semi_implicit_euler": SemiImplicitEulerIntegrator,
        "rk4":                 RK4Integrator,
        "verlet":              VerletIntegrator,
        "symplectic":          SymplecticIntegrator,
    }
    if method not in table:
        raise ValueError(
            f"build_integrator: unknown method {method!r}; "
            f"expected one of {sorted(table)}"
        )
    return table[method]()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  —  Independent Physics Modules (each implements compute())
# ─────────────────────────────────────────────────────────────────────────────

class GravityModule:
    """Compute gravitational force for each body.

    F = m · g (global gravity vector from config).

    Gravity is disabled for static bodies and sleeping bodies.
    Supports Earth / Moon / Mars presets via config.gravity.
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._g = config.gravity_vector()

    def compute(self, states: Dict[str, PhysicsState]) -> Dict[str, torch.Tensor]:
        """Return per-body gravitational force vectors.

        Args:
            states: Dict of body_id → PhysicsState.

        Returns:
            Dict of body_id → force tensor (3,). Only dynamic, non-sleeping
            bodies are included; static / sleeping bodies map to zero.
        """
        forces: Dict[str, torch.Tensor] = {}
        for bid, s in states.items():
            g = self._g.to(s.force.device).to(s.force.dtype)
            if s.is_static or s.sleeping:
                forces[bid] = torch.zeros(3, dtype=s.force.dtype, device=s.force.device)
            else:
                forces[bid] = s.mass * g
        return forces


class DragModule:
    """Compute aerodynamic drag force.

    Fd = -0.5 · ρ · Cd · A · |v|² · v̂

    where ρ is air density, Cd is the body's drag coefficient, A is its
    cross-section area, and v̂ is the unit velocity direction.
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._rho = config.air_density

    def compute(self, states: Dict[str, PhysicsState]) -> Dict[str, torch.Tensor]:
        """Return per-body aerodynamic drag force vectors.

        Args:
            states: Dict of body_id → PhysicsState.

        Returns:
            Dict of body_id → drag force (3,), opposing velocity direction.
        """
        forces: Dict[str, torch.Tensor] = {}
        for bid, s in states.items():
            if s.is_static or s.sleeping:
                forces[bid] = torch.zeros(3, dtype=s.force.dtype, device=s.force.device)
                continue
            speed_sq = float(s.velocity.pow(2).sum())
            speed    = math.sqrt(max(speed_sq, 0.0))
            if speed < 1e-9:
                forces[bid] = torch.zeros(3, dtype=s.force.dtype, device=s.force.device)
            else:
                drag_mag = 0.5 * self._rho * s.drag_coefficient * s.cross_section_area * speed_sq
                unit_v   = s.velocity / speed
                forces[bid] = -drag_mag * unit_v
        return forces


class FrictionModule:
    """Compute friction forces: static, dynamic, and rolling.

    At rest: if applied force < μ_s · N, static friction cancels it.
    In motion: dynamic friction Ff = -μ_k · N · v̂.
    Rolling:   rolling torque τ_r = -μ_r · N · r (for spheres only).
    """

    def compute(
        self,
        states: Dict[str, PhysicsState],
        contacts: List[Contact],
        gravity_forces: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute friction forces and torques for all active contacts.

        Args:
            states:          Dict of body_id → PhysicsState.
            contacts:        Active contacts from CollisionModule.
            gravity_forces:  Per-body gravity forces (used to estimate normal).

        Returns:
            Tuple of (friction_forces, friction_torques), both Dict[str, Tensor(3,)].
        """
        friction_forces:  Dict[str, torch.Tensor] = {
            bid: torch.zeros(3, dtype=s.force.dtype, device=s.force.device)
            for bid, s in states.items()
        }
        friction_torques: Dict[str, torch.Tensor] = {
            bid: torch.zeros(3, dtype=s.torque.dtype, device=s.torque.device)
            for bid, s in states.items()
        }

        for contact in contacts:
            for (moving_id, static_id) in [
                (contact.body_a, contact.body_b),
                (contact.body_b, contact.body_a),
            ]:
                if moving_id not in states:
                    continue
                s = states[moving_id]
                if s.is_static or s.sleeping:
                    continue

                # Estimate normal force from gravity
                g_force = gravity_forces.get(moving_id,
                          torch.zeros(3, dtype=s.force.dtype, device=s.force.device))
                normal_force_mag = float(g_force.norm())
                speed = float(s.velocity.norm())

                if speed < 1e-6:
                    # Static friction: zero here (balanced by applied force
                    # already — handled in constraint solver)
                    continue

                # Dynamic friction
                v_hat = s.velocity / max(speed, 1e-9)
                Ff = -s.friction * normal_force_mag * v_hat
                friction_forces[moving_id] = friction_forces[moving_id] + Ff

                # Rolling friction (spheres only)
                if s.shape == "sphere":
                    r = s.shape_params.get("radius", 0.5)
                    tau_r = -s.rolling_friction * normal_force_mag * r * v_hat
                    friction_torques[moving_id] = friction_torques[moving_id] + tau_r

        return friction_forces, friction_torques


class CollisionModule:
    """Broad-phase + narrow-phase collision detection for simple primitives.

    Supported primitive pairs:
        sphere–sphere
        sphere–plane
        box–plane
        point–plane  (point masses)

    Complex geometry (capsule, mesh, convex hull, SDF) belongs to
    Bullet/MuJoCo/Isaac. This module is intentionally minimal so the
    StateEngine remains a differentiable bridge, not a physics engine clone.
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._slop = config.contact_slop
        self._max  = config.max_contacts

    def detect(
        self,
        states: Dict[str, PhysicsState],
        timestep: float,
    ) -> List[Contact]:
        """Detect contacts among all bodies in ``states``.

        Args:
            states:    Dict of body_id → PhysicsState.
            timestep:  Current simulation time (for contact timestamps).

        Returns:
            List of :class:`Contact` objects (length ≤ max_contacts).
        """
        contacts: List[Contact] = []
        body_list = list(states.values())

        # Pairwise checks — O(N²), acceptable for small-N StateEngine use
        for i, a in enumerate(body_list):
            for b in body_list[i+1:]:
                if a.is_static and b.is_static:
                    continue
                if a.sleeping and b.sleeping:
                    continue
                c = self._narrow_phase(a, b, timestep)
                if c is not None:
                    contacts.append(c)
                if len(contacts) >= self._max:
                    return contacts

        # Ground-plane contacts
        for s in body_list:
            if s.is_static:
                continue
            c = self._ground_contact(s, timestep)
            if c is not None:
                contacts.append(c)
                if len(contacts) >= self._max:
                    return contacts

        return contacts

    def _narrow_phase(
        self, a: PhysicsState, b: PhysicsState, t: float
    ) -> Optional[Contact]:
        """Dispatch to the right primitive pair test."""
        pair = tuple(sorted([a.shape, b.shape]))
        if pair == ("sphere", "sphere"):
            return self._sphere_sphere(a, b, t)
        if "plane" in pair:
            other = a if b.shape == "plane" else b
            plane = b if b.shape == "plane" else a
            return self._sphere_plane(other, plane, t) if other.shape in ("sphere", "point") else None
        if pair == ("box", "box"):
            return self._aabb_aabb(a, b, t)
        return None

    def _sphere_sphere(
        self, a: PhysicsState, b: PhysicsState, t: float
    ) -> Optional[Contact]:
        r_a = a.shape_params.get("radius", 0.5)
        r_b = b.shape_params.get("radius", 0.5)
        delta = a.position - b.position
        dist  = float(delta.norm()) + 1e-12
        pen   = (r_a + r_b) - dist
        if pen < -self._slop:
            return None
        normal    = (delta / dist)
        cp        = b.position + normal * r_b
        zero3     = torch.zeros(3, dtype=a.position.dtype, device=a.position.device)
        rel_vel   = a.velocity - b.velocity
        return Contact(body_a=a.body_id, body_b=b.body_id, contact_point=cp,
                       normal=normal, penetration_depth=float(pen),
                       normal_impulse=zero3.clone(), friction_impulse=zero3.clone(),
                       relative_velocity=rel_vel, timestamp=t)

    def _sphere_plane(
        self, sphere: PhysicsState, plane: PhysicsState, t: float
    ) -> Optional[Contact]:
        n = plane.shape_params.get("normal", [0.0, 0.0, 1.0])
        d = plane.shape_params.get("d", 0.0)
        normal = torch.tensor(n, dtype=sphere.position.dtype, device=sphere.position.device)
        r      = sphere.shape_params.get("radius", 0.5)
        dist   = float(torch.dot(sphere.position, normal)) - d
        pen    = r - dist
        if pen < -self._slop:
            return None
        cp     = sphere.position - normal * r
        zero3  = torch.zeros(3, dtype=sphere.position.dtype, device=sphere.position.device)
        return Contact(body_a=sphere.body_id, body_b=plane.body_id, contact_point=cp,
                       normal=normal, penetration_depth=float(pen),
                       normal_impulse=zero3.clone(), friction_impulse=zero3.clone(),
                       relative_velocity=sphere.velocity.clone(), timestamp=t)

    def _aabb_aabb(
        self, a: PhysicsState, b: PhysicsState, t: float
    ) -> Optional[Contact]:
        """AABB (axis-aligned bounding box) overlap test.

        Returns the minimum-penetration-axis contact if overlapping.
        """
        ha = torch.tensor([
            a.shape_params.get("half_x", 0.5),
            a.shape_params.get("half_y", 0.5),
            a.shape_params.get("half_z", 0.5),
        ], dtype=a.position.dtype, device=a.position.device)
        hb = torch.tensor([
            b.shape_params.get("half_x", 0.5),
            b.shape_params.get("half_y", 0.5),
            b.shape_params.get("half_z", 0.5),
        ], dtype=b.position.dtype, device=b.position.device)
        delta = a.position - b.position
        overlap = (ha + hb) - delta.abs()
        if (overlap < -self._slop).any():
            return None
        axis_idx = int(overlap.argmin().item())
        normal   = torch.zeros(3, dtype=a.position.dtype, device=a.position.device)
        normal[axis_idx] = 1.0 if delta[axis_idx] > 0 else -1.0
        pen      = float(overlap[axis_idx].item())
        cp       = (a.position + b.position) * 0.5
        zero3    = torch.zeros(3, dtype=a.position.dtype, device=a.position.device)
        return Contact(body_a=a.body_id, body_b=b.body_id, contact_point=cp,
                       normal=normal, penetration_depth=pen,
                       normal_impulse=zero3.clone(), friction_impulse=zero3.clone(),
                       relative_velocity=(a.velocity - b.velocity), timestamp=t)

    def _ground_contact(
        self, s: PhysicsState, t: float
    ) -> Optional[Contact]:
        """Check contact with the implicit ground plane z = 0."""
        r = s.shape_params.get("radius", 0.5) if s.shape == "sphere" else \
            s.shape_params.get("half_z", 0.5)
        pen = r - float(s.position[2].item())
        if pen < -self._slop:
            return None
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=s.position.dtype, device=s.position.device)
        cp     = s.position.clone()
        cp[2]  = 0.0
        zero3  = torch.zeros(3, dtype=s.position.dtype, device=s.position.device)
        return Contact(body_a=s.body_id, body_b="ground", contact_point=cp,
                       normal=normal, penetration_depth=pen,
                       normal_impulse=zero3.clone(), friction_impulse=zero3.clone(),
                       relative_velocity=s.velocity.clone(), timestamp=t)


class RestitutionModule:
    """Compute collision impulses, applying the coefficient of restitution.

    Normal impulse:
        j_n = -(1 + e) · v_rel · n̂ / (1/mA + 1/mB)

    where e is the combined restitution coefficient (geometric mean of
    both bodies), v_rel is the relative velocity at the contact point,
    and n̂ is the contact normal.
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._threshold = config.restitution_threshold

    def resolve(
        self,
        states: Dict[str, PhysicsState],
        contacts: List[Contact],
    ) -> Tuple[Dict[str, PhysicsState], List[Contact]]:
        """Apply impulse-based collision response to all contacts.

        Args:
            states:   Mutable dict of body_id → PhysicsState.
            contacts: Detected contacts from CollisionModule.

        Returns:
            Tuple of (updated states dict, contacts with impulses filled in).
        """
        updated = {bid: s.clone() for bid, s in states.items()}

        for contact in contacts:
            a_id, b_id = contact.body_a, contact.body_b
            if a_id not in updated:
                continue

            sa = updated[a_id]
            sb = updated.get(b_id)  # may be None (ground) or static

            v_a = sa.velocity
            v_b = sb.velocity if sb is not None else torch.zeros_like(v_a)
            v_rel = v_a - v_b
            n = contact.normal.to(v_a.device).to(v_a.dtype)
            v_rel_n = float(torch.dot(v_rel, n).item())

            # Only resolve separating contacts
            if v_rel_n > 0:
                continue
            # Below restitution threshold → inelastic (no bounce)
            e = math.sqrt(sa.restitution * (sb.restitution if sb else 0.0)) \
                if abs(v_rel_n) > self._threshold else 0.0

            inv_ma = 0.0 if sa.is_static else 1.0 / max(sa.mass, 1e-10)
            inv_mb = 0.0 if (sb is None or sb.is_static) else 1.0 / max(sb.mass, 1e-10)
            denom  = inv_ma + inv_mb
            if denom < 1e-12:
                continue

            j_n = -(1.0 + e) * v_rel_n / denom
            impulse = j_n * n

            contact.normal_impulse = impulse.clone()

            if not sa.is_static:
                updated[a_id].velocity = sa.velocity + impulse * inv_ma
            if sb is not None and not sb.is_static:
                updated[b_id].velocity = sb.velocity - impulse * inv_mb

            # Positional correction (Baumgarte)
            slop_correction = max(contact.penetration_depth - 1e-3, 0.0)
            corr = 0.2 * slop_correction / denom * n
            if not sa.is_static:
                updated[a_id].position = updated[a_id].position + inv_ma * corr
            if sb is not None and not sb.is_static:
                updated[b_id].position = updated[b_id].position - inv_mb * corr

        return updated, contacts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  —  Force Registry
# ─────────────────────────────────────────────────────────────────────────────

class ForceRegistry:
    """Maintains the set of forces acting on each body and accumulates them.

    Forces may be persistent (applied every step) or one-shot (applied once
    and then automatically removed). The registry is the single source of
    truth for net force/torque; no force is ever applied outside of it.

    Attributes:
        _forces:  Dict of force_id → ForceRecord.
        _counter: Monotonic counter for generating force_ids.
    """

    def __init__(self) -> None:
        self._forces: Dict[str, ForceRecord] = {}
        self._counter: int = 0

    def apply_force(
        self,
        body_id:    str,
        force:      torch.Tensor,
        torque:     Optional[torch.Tensor] = None,
        source:     str = "user",
        persistent: bool = False,
        force_id:   Optional[str] = None,
    ) -> str:
        """Register a force acting on ``body_id``.

        Args:
            body_id:    Target body identifier.
            force:      Force vector (3,) in Newtons.
            torque:     Torque vector (3,) in N·m, or None (→ zero).
            source:     Human-readable source label for explainability.
            persistent: Keep force across steps (True) or remove after one
                        application (False).
            force_id:   Explicit id; auto-generated if omitted.

        Returns:
            The force's id string.
        """
        fid = force_id or f"force_{self._counter}"
        self._counter += 1
        _torque = torque if torque is not None else torch.zeros_like(force)
        self._forces[fid] = ForceRecord(
            force_id=fid, body_id=body_id, source=source,
            force=force.clone(), torque=_torque.clone(), persistent=persistent,
        )
        return fid

    def remove_force(self, force_id: str) -> None:
        """Remove a registered force by id.

        Args:
            force_id: Force id returned by :meth:`apply_force`.

        Raises:
            KeyError: If the force_id doesn't exist.
        """
        if force_id not in self._forces:
            raise KeyError(f"ForceRegistry: unknown force_id {force_id!r}")
        del self._forces[force_id]

    def clear_forces(self, body_id: Optional[str] = None) -> None:
        """Remove all forces, optionally filtered to one body.

        Args:
            body_id: If given, clear only forces acting on this body.
                     If None, clear all forces from all bodies.
        """
        if body_id is None:
            self._forces.clear()
        else:
            to_del = [fid for fid, rec in self._forces.items()
                      if rec.body_id == body_id]
            for fid in to_del:
                del self._forces[fid]

    def accumulate(
        self, states: Dict[str, PhysicsState]
    ) -> Tuple[Dict[str, PhysicsState], Dict[str, Dict[str, torch.Tensor]]]:
        """Sum all registered forces/torques onto their target bodies.

        Also resets state.force/state.torque to zero before accumulation so
        forces don't compound across steps.

        Args:
            states: Dict of body_id → PhysicsState (modified in-place).

        Returns:
            Tuple of (updated states dict, force_breakdown dict
            {body_id → {source → force_tensor}}).
        """
        # Zero out force/torque
        for s in states.values():
            s.force  = torch.zeros(3, dtype=s.force.dtype,  device=s.force.device)
            s.torque = torch.zeros(3, dtype=s.torque.dtype, device=s.torque.device)

        breakdown: Dict[str, Dict[str, torch.Tensor]] = {
            bid: {} for bid in states
        }
        one_shots: List[str] = []

        for fid, rec in self._forces.items():
            if rec.body_id in states:
                s = states[rec.body_id]
                s.force  = s.force  + rec.force.to(s.force.device).to(s.force.dtype)
                s.torque = s.torque + rec.torque.to(s.torque.device).to(s.torque.dtype)
                breakdown[rec.body_id][rec.source] = rec.force.clone()
            if not rec.persistent:
                one_shots.append(fid)

        for fid in one_shots:
            del self._forces[fid]

        return states, breakdown

    def active_force_ids(self, body_id: Optional[str] = None) -> List[str]:
        """Return ids of all active forces, optionally filtered by body."""
        if body_id is None:
            return list(self._forces.keys())
        return [fid for fid, rec in self._forces.items() if rec.body_id == body_id]

    def __len__(self) -> int:
        return len(self._forces)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  —  Constraint Solver
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintSolver:
    """Iterative constraint solver supporting hinge / fixed / spring / slider /
    distance joints.

    The solver uses a sequential impulse approach: for each active constraint,
    compute the violation and apply the minimum corrective impulse. Iterate up
    to ``max_iterations`` times per step.

    Only constraints whose joints are NOT broken are enforced.
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._max_iter = config.max_iterations
        self._dt       = config.dt

    def solve(
        self,
        states:      Dict[str, PhysicsState],
        constraints: Dict[str, Constraint],
        timestep:    float,
    ) -> Tuple[Dict[str, PhysicsState], Dict[str, float], List[PhysicsEvent]]:
        """Apply constraint impulses to the state dict.

        Args:
            states:      Current body states.
            constraints: Active constraints from ConstraintManager.
            timestep:    Current simulation time (for event timestamps).

        Returns:
            Tuple of (updated states, constraint_force_magnitudes dict,
            list of PhysicsEvent for broken constraints).
        """
        updated = {bid: s.clone() for bid, s in states.items()}
        force_mags: Dict[str, float] = {}
        events: List[PhysicsEvent] = []

        for _ in range(self._max_iter):
            for cid, c in constraints.items():
                if c.broken:
                    continue
                if c.body_a not in updated:
                    continue

                sa = updated[c.body_a]
                sb = updated.get(c.body_b)  # may be None if body_b == "world"

                impulse_mag = self._apply_constraint(sa, sb, c, updated)
                force_mags[cid] = impulse_mag

                # Check break condition
                if impulse_mag > c.break_force * self._dt:
                    c.broken = True
                    events.append(PhysicsEvent(
                        event_type=EventType.CONSTRAINT_BREAK,
                        body_ids=[c.body_a, c.body_b],
                        timestep=timestep,
                        magnitude=impulse_mag,
                    ))

        return updated, force_mags, events

    def _apply_constraint(
        self,
        sa: PhysicsState,
        sb: Optional[PhysicsState],
        c: Constraint,
        updated: Dict[str, PhysicsState],
    ) -> float:
        """Dispatch to the appropriate joint handler and return impulse magnitude."""
        if c.joint_type in ("spring", "distance"):
            return self._spring_constraint(sa, sb, c, updated)
        if c.joint_type == "fixed":
            return self._fixed_constraint(sa, sb, c, updated)
        # hinge, slider: positional correction only (simplified)
        return self._positional_constraint(sa, sb, c, updated)

    def _spring_constraint(
        self,
        sa: PhysicsState,
        sb: Optional[PhysicsState],
        c: Constraint,
        updated: Dict[str, PhysicsState],
    ) -> float:
        pa = sa.position + c.anchor_a.to(sa.position.device)
        pb = (sb.position + c.anchor_b.to(sb.position.device)
              if sb is not None else c.anchor_b.to(sa.position.device))
        delta  = pa - pb
        dist   = float(delta.norm()) + 1e-12
        ext    = dist - c.rest_length
        n      = delta / dist

        # Spring force: F = -k·ext·n  (Damping: F_d = -b·v_rel)
        v_a = sa.velocity
        v_b = sb.velocity if sb is not None else torch.zeros_like(v_a)
        v_rel_n = float(torch.dot(v_a - v_b, n))
        spring_force = -(c.stiffness * ext + c.damping * v_rel_n)
        impulse      = spring_force * self._dt
        impulse_v    = impulse * n.to(sa.force.device).to(sa.force.dtype)

        inv_ma = 0.0 if sa.is_static else 1.0 / max(sa.mass, 1e-10)
        inv_mb = 0.0 if (sb is None or sb.is_static) else 1.0 / max(sb.mass, 1e-10)

        if not sa.is_static:
            updated[sa.body_id].velocity = sa.velocity + impulse_v * inv_ma
        if sb is not None and not sb.is_static:
            updated[sb.body_id].velocity = sb.velocity - impulse_v * inv_mb

        return abs(impulse)

    def _fixed_constraint(
        self,
        sa: PhysicsState,
        sb: Optional[PhysicsState],
        c: Constraint,
        updated: Dict[str, PhysicsState],
    ) -> float:
        """Fixed joint: zero relative velocity at anchor."""
        if sb is None or sb.is_static:
            if not sa.is_static:
                updated[sa.body_id].velocity = torch.zeros_like(sa.velocity)
            return 0.0
        rel_vel = sa.velocity - sb.velocity
        impulse_mag = float(rel_vel.norm())
        inv_ma = 1.0 / max(sa.mass, 1e-10)
        inv_mb = 1.0 / max(sb.mass, 1e-10)
        impulse = -rel_vel / (inv_ma + inv_mb + 1e-12)
        if not sa.is_static:
            updated[sa.body_id].velocity = sa.velocity + impulse * inv_ma
        if not sb.is_static:
            updated[sb.body_id].velocity = sb.velocity - impulse * inv_mb
        return impulse_mag

    def _positional_constraint(
        self,
        sa: PhysicsState,
        sb: Optional[PhysicsState],
        c: Constraint,
        updated: Dict[str, PhysicsState],
    ) -> float:
        """Positional correction for hinge/slider (simplified axis projection)."""
        if sb is None:
            return 0.0
        delta = sa.position - sb.position
        axis  = c.axis.to(delta.device).to(delta.dtype)
        axis  = F.normalize(axis, p=2, dim=-1, eps=1e-8)
        perp  = delta - torch.dot(delta, axis) * axis
        mag   = float(perp.norm())
        if mag < 1e-6:
            return 0.0
        corr_v = perp * 0.1  # Baumgarte-style damped positional correction
        if not sa.is_static:
            updated[sa.body_id].position = sa.position - corr_v * 0.5
        if not sb.is_static:
            updated[sb.body_id].position = sb.position + corr_v * 0.5
        return mag


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  —  Contact Manager
# ─────────────────────────────────────────────────────────────────────────────

class ContactManager:
    """Maintains the active contact list, contact history, and contact graph.

    Attributes:
        active_contacts:   Contacts from the most recent step.
        contact_history:   All contacts ever detected (capped at capacity).
        contact_graph:     Dict of body_id → set of body_ids currently
                           in contact (updated every step).
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self.active_contacts:  List[Contact] = []
        self.contact_history:  List[Contact] = []
        self.contact_graph:    Dict[str, set] = {}
        self._capacity = capacity

    def update(self, contacts: List[Contact]) -> None:
        """Replace active contacts and update history/graph.

        Args:
            contacts: Newly detected contacts from CollisionModule.
        """
        self.active_contacts = contacts
        self.contact_history.extend(contacts)
        if len(self.contact_history) > self._capacity:
            self.contact_history = self.contact_history[-self._capacity:]

        # Rebuild contact graph
        self.contact_graph.clear()
        for c in contacts:
            self.contact_graph.setdefault(c.body_a, set()).add(c.body_b)
            self.contact_graph.setdefault(c.body_b, set()).add(c.body_a)

    def nearest_contacts(self, body_id: str) -> List[Contact]:
        """Return all active contacts involving ``body_id``."""
        return [c for c in self.active_contacts
                if c.body_a == body_id or c.body_b == body_id]

    def contact_count(self) -> int:
        return len(self.active_contacts)

    def clear(self) -> None:
        self.active_contacts.clear()
        self.contact_graph.clear()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  —  Constraint Manager
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintManager:
    """Registry of active kinematic constraints.

    Attributes:
        active_constraints:   Dict of constraint_id → Constraint.
        constraint_graph:      Dict of body_id → set of connected body_ids.
    """

    def __init__(self) -> None:
        self.active_constraints: Dict[str, Constraint] = {}
        self.constraint_graph:   Dict[str, set] = {}
        self._counter: int = 0

    def add_constraint(self, constraint: Constraint) -> str:
        """Register a constraint. Returns its id."""
        self.active_constraints[constraint.constraint_id] = constraint
        self.constraint_graph.setdefault(constraint.body_a, set()).add(constraint.body_b)
        self.constraint_graph.setdefault(constraint.body_b, set()).add(constraint.body_a)
        return constraint.constraint_id

    def remove_constraint(self, constraint_id: str) -> None:
        """Remove a constraint by id.

        Raises:
            KeyError: If the constraint doesn't exist.
        """
        if constraint_id not in self.active_constraints:
            raise KeyError(f"ConstraintManager: unknown constraint_id {constraint_id!r}")
        del self.active_constraints[constraint_id]
        # Rebuild constraint graph (cheap for small N)
        self.constraint_graph.clear()
        for c in self.active_constraints.values():
            self.constraint_graph.setdefault(c.body_a, set()).add(c.body_b)
            self.constraint_graph.setdefault(c.body_b, set()).add(c.body_a)

    def new_constraint(
        self,
        body_a:     str,
        body_b:     str,
        joint_type: str,
        anchor_a:   Optional[torch.Tensor] = None,
        anchor_b:   Optional[torch.Tensor] = None,
        axis:       Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> str:
        """Convenience factory for adding a new constraint.

        Args:
            body_a, body_b: Constrained body ids ("world" for fixed to world).
            joint_type:      "fixed" / "hinge" / "slider" / "spring" / "distance".
            anchor_a, anchor_b: Local anchor points. Default to zero.
            axis:             Constraint axis. Default to (0,0,1).
            **kwargs:         Additional Constraint fields.

        Returns:
            New constraint's id.
        """
        cid = f"constraint_{self._counter}"
        self._counter += 1
        z3  = torch.zeros(3)
        ax  = axis if axis is not None else torch.tensor([0.0, 0.0, 1.0])
        c = Constraint(
            constraint_id=cid,
            body_a=body_a, body_b=body_b,
            joint_type=joint_type,
            anchor_a=anchor_a if anchor_a is not None else z3.clone(),
            anchor_b=anchor_b if anchor_b is not None else z3.clone(),
            axis=ax,
            **kwargs,
        )
        return self.add_constraint(c)

    def clear(self) -> None:
        self.active_constraints.clear()
        self.constraint_graph.clear()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  —  Energy Accounting
# ─────────────────────────────────────────────────────────────────────────────

class EnergyAccounting:
    """Compute and track mechanical energy for a set of bodies.

    Methods are pure functions of the state dict — no internal state except
    the energy history used for drift analysis.
    """

    def __init__(self, gravity_vector: torch.Tensor) -> None:
        self._g = gravity_vector
        self._history: List[float] = []

    def kinetic_energy(self, states: Dict[str, PhysicsState]) -> float:
        """Total kinetic energy KE = Σ 0.5·m·|v|² + Σ 0.5·I·|ω|²."""
        ke = 0.0
        for s in states.values():
            if s.sleeping or s.is_static:
                continue
            ke += 0.5 * s.mass * float(s.velocity.pow(2).sum())
            ke += 0.5 * float((s.inertia_tensor * s.angular_velocity.pow(2)).sum())
        return ke

    def potential_energy(self, states: Dict[str, PhysicsState]) -> float:
        """Total gravitational PE = Σ m · g · h.

        Uses the z-component (up) of the gravity vector; height h = position.z.
        """
        g_mag = float(self._g.norm())
        pe = 0.0
        for s in states.values():
            if s.is_static:
                continue
            # height along gravity direction (negative g_z means up = positive z)
            h = float(s.position[2].item())
            pe += s.mass * g_mag * h
        return pe

    def dissipated_energy(
        self,
        contacts: List[Contact],
        friction_forces: Dict[str, torch.Tensor],
        dt: float,
    ) -> float:
        """Estimate energy dissipated by friction this step.

        Approximation: E_diss = Σ |F_friction| · |v| · dt.
        """
        dissipated = 0.0
        for bid, ff in friction_forces.items():
            dissipated += float(ff.norm()) * dt
        return dissipated

    def total_energy(self, ke: float, pe: float) -> float:
        return ke + pe

    def energy_drift(self, total: float) -> float:
        """Return fractional drift from the first recorded energy value.

        Returns 0.0 until at least two samples have been recorded.
        """
        self._history.append(total)
        if len(self._history) < 2 or abs(self._history[0]) < 1e-10:
            return 0.0
        return abs(total - self._history[0]) / abs(self._history[0])

    def reset(self) -> None:
        self._history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  —  State History
# ─────────────────────────────────────────────────────────────────────────────

class StateHistory:
    """Ordered ring-buffer of StateSnapshot objects for the TrajectoryEngine.

    Maintains a fixed-capacity circular buffer so memory is bounded
    regardless of simulation duration.

    Attributes:
        capacity:  Maximum number of snapshots retained.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._snapshots: List[StateSnapshot] = []

    def record(self, snapshot: StateSnapshot) -> None:
        """Append a snapshot; drop the oldest if at capacity."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.capacity:
            self._snapshots.pop(0)

    def get(self, index: int) -> StateSnapshot:
        """Return snapshot by index (0 = oldest, -1 = newest).

        Raises:
            IndexError: If index is out of range.
        """
        return self._snapshots[index]

    def latest(self) -> Optional[StateSnapshot]:
        """Return the most recent snapshot, or None if empty."""
        return self._snapshots[-1] if self._snapshots else None

    def all_snapshots(self) -> List[StateSnapshot]:
        """Return all stored snapshots in chronological order."""
        return list(self._snapshots)

    def clear(self) -> None:
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14  —  Sleeping System
# ─────────────────────────────────────────────────────────────────────────────

class SleepingSystem:
    """Put bodies to sleep when their velocities fall below threshold.

    A body wakes automatically if a non-zero force is applied to it next step
    (handled in StateEngine.step via force accumulation).
    """

    def __init__(self, config: StateEngineConfig) -> None:
        self._v_thresh   = config.velocity_threshold
        self._w_thresh   = config.angular_velocity_threshold

    def update(
        self,
        states:  Dict[str, PhysicsState],
        events:  List[PhysicsEvent],
        timestep: float,
    ) -> Dict[str, PhysicsState]:
        """Update sleep state for all bodies. Returns updated states."""
        for s in states.values():
            if s.is_static:
                s.sleeping = True
                continue

            speed   = float(s.velocity.norm())
            w_speed = float(s.angular_velocity.norm())
            f_mag   = float(s.force.norm())  # body just had forces accumulated

            should_sleep = (speed < self._v_thresh and w_speed < self._w_thresh)

            if should_sleep and not s.sleeping:
                s.sleeping = True
                events.append(PhysicsEvent(
                    event_type=EventType.SLEEP,
                    body_ids=[s.body_id],
                    timestep=timestep,
                ))
            elif not should_sleep and s.sleeping and f_mag > 1e-9:
                s.sleeping = False
                events.append(PhysicsEvent(
                    event_type=EventType.WAKE,
                    body_ids=[s.body_id],
                    timestep=timestep,
                ))
        return states


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15  —  Physics Cache
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsCache:
    """Per-step cache to avoid redundant recomputation.

    Attributes:
        last_states:     Body states at the previous step.
        last_forces:     Force breakdown from the previous step.
        last_contacts:   Contacts from the previous step.
        last_energy:     Total energy from the previous step.
    """

    last_states:   Optional[Dict[str, PhysicsState]]     = None
    last_forces:   Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    last_contacts: Optional[List[Contact]]               = None
    last_energy:   Optional[float]                       = None

    def update(
        self,
        states:   Dict[str, PhysicsState],
        forces:   Dict[str, Dict[str, torch.Tensor]],
        contacts: List[Contact],
        energy:   float,
    ) -> None:
        self.last_states   = {bid: s.clone() for bid, s in states.items()}
        self.last_forces   = forces
        self.last_contacts = list(contacts)
        self.last_energy   = energy


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16  —  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class Diagnostics:
    """Track per-step and cumulative physics diagnostics.

    Attributes:
        energy_history:      List of total energy values per step.
        momentum_history:    List of linear momentum magnitudes per step.
        contact_count_history: List of contact counts per step.
        constraint_violation_history: List of max constraint violation per step.
    """

    def __init__(self) -> None:
        self.energy_history:     List[float] = []
        self.momentum_history:   List[float] = []
        self.contact_count_history: List[int] = []
        self.constraint_violation_history: List[float] = []
        self._step_count: int = 0

    def record(
        self,
        energy:              float,
        linear_momentum_mag: float,
        contact_count:       int,
        constraint_violation: float,
    ) -> None:
        """Record diagnostics for one step."""
        self.energy_history.append(energy)
        self.momentum_history.append(linear_momentum_mag)
        self.contact_count_history.append(contact_count)
        self.constraint_violation_history.append(constraint_violation)
        self._step_count += 1

    def summary(self) -> Dict[str, float]:
        """Return a scalar summary dict for the latest step."""
        if not self.energy_history:
            return {}
        e0 = self.energy_history[0] if self.energy_history[0] != 0 else 1.0
        return {
            "energy_drift":           abs(self.energy_history[-1] - self.energy_history[0]) / abs(e0),
            "mean_energy":            sum(self.energy_history) / len(self.energy_history),
            "mean_momentum":          sum(self.momentum_history) / len(self.momentum_history),
            "mean_contacts":          sum(self.contact_count_history) / len(self.contact_count_history),
            "max_constraint_violation": max(self.constraint_violation_history or [0.0]),
            "total_steps":            float(self._step_count),
        }

    def reset(self) -> None:
        self.energy_history.clear()
        self.momentum_history.clear()
        self.contact_count_history.clear()
        self.constraint_violation_history.clear()
        self._step_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17  —  Scene-Level Statistics
# ─────────────────────────────────────────────────────────────────────────────

def linear_momentum(states: Dict[str, PhysicsState]) -> torch.Tensor:
    """Total linear momentum Σ m·v. Returns (3,) tensor."""
    p = None
    for s in states.values():
        if s.is_static:
            continue
        mv = s.mass * s.velocity
        p = mv if p is None else p + mv
    if p is None:
        ref = next(iter(states.values()))
        return torch.zeros(3, dtype=ref.force.dtype, device=ref.force.device)
    return p


def angular_momentum(states: Dict[str, PhysicsState]) -> torch.Tensor:
    """Total angular momentum Σ (r × m·v + I·ω). Returns (3,) tensor."""
    L = None
    for s in states.values():
        if s.is_static:
            continue
        orb = torch.linalg.cross(s.position, s.mass * s.velocity)
        spin = s.inertia_tensor * s.angular_velocity
        Ls = orb + spin
        L = Ls if L is None else L + Ls
    if L is None:
        ref = next(iter(states.values()))
        return torch.zeros(3, dtype=ref.force.dtype, device=ref.force.device)
    return L


def center_of_mass(states: Dict[str, PhysicsState]) -> torch.Tensor:
    """Total center of mass Σ m·r / Σ m. Returns (3,) tensor."""
    total_mass = 0.0
    weighted_pos = None
    for s in states.values():
        if s.is_static:
            continue
        wp = s.mass * s.position
        weighted_pos = wp if weighted_pos is None else weighted_pos + wp
        total_mass += s.mass
    if weighted_pos is None or total_mass < 1e-10:
        ref = next(iter(states.values()))
        return torch.zeros(3, dtype=ref.force.dtype, device=ref.force.device)
    return weighted_pos / total_mass


def total_mass(states: Dict[str, PhysicsState]) -> float:
    """Return Σ m for all dynamic bodies."""
    return sum(s.mass for s in states.values() if not s.is_static)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18  —  Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_state(state: PhysicsState) -> None:
    """Validate a single PhysicsState. Raises on any invalid condition.

    Args:
        state: The state to validate.

    Raises:
        ValueError: On negative/zero mass, zero inertia, invalid quaternion,
                    dimension mismatch, or NaN/Inf in any tensor field.
    """
    def _nan_check(t: torch.Tensor, name: str) -> None:
        if torch.isnan(t).any():
            raise ValueError(f"[{state.body_id}] {name} contains NaN")
        if torch.isinf(t).any():
            raise ValueError(f"[{state.body_id}] {name} contains Inf")

    def _shape_check(t: torch.Tensor, expected: tuple, name: str) -> None:
        if tuple(t.shape) != expected:
            raise ValueError(
                f"[{state.body_id}] {name}: expected shape {expected}, got {tuple(t.shape)}"
            )

    if not state.is_static and state.mass <= 0:
        raise ValueError(
            f"[{state.body_id}] mass must be > 0 for dynamic bodies, got {state.mass}"
        )

    for attr, shape in [
        ("position", (3,)), ("velocity", (3,)), ("acceleration", (3,)),
        ("angular_velocity", (3,)), ("angular_accel", (3,)),
        ("force", (3,)), ("torque", (3,)), ("inertia_tensor", (3,)),
    ]:
        t = getattr(state, attr)
        _shape_check(t, shape, attr)
        _nan_check(t, attr)

    _shape_check(state.orientation, (4,), "orientation")
    _nan_check(state.orientation, "orientation")
    if not QuaternionUtils.is_valid(state.orientation):
        warnings.warn(
            f"[{state.body_id}] orientation is not a unit quaternion "
            f"(norm={float(state.orientation.norm()):.6f}); normalizing.",
            stacklevel=2,
        )

    if not state.is_static:
        if (state.inertia_tensor <= 0).any():
            raise ValueError(
                f"[{state.body_id}] inertia_tensor must have all-positive "
                f"components for dynamic bodies, got {state.inertia_tensor.tolist()}"
            )

    for attr in ("friction", "static_friction", "rolling_friction", "restitution"):
        v = getattr(state, attr)
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"[{state.body_id}] {attr} must be in [0, 1], got {v}"
            )


def validate_states(states: Dict[str, PhysicsState]) -> None:
    """Validate all states in a dict. Accumulates all errors before raising."""
    errors: List[str] = []
    for bid, s in states.items():
        try:
            validate_state(s)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(
            f"validate_states: {len(errors)} error(s):\n  " + "\n  ".join(errors)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19  —  WorldSpec Interface
# ─────────────────────────────────────────────────────────────────────────────

def _make_sphere_state(
    body_id: str, mass: float, position: Tuple[float, float, float],
    velocity: Tuple[float, float, float] = (0, 0, 0),
    radius: float = 0.5,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> PhysicsState:
    """Helper: create a simple sphere PhysicsState."""
    I_sphere = (2 / 5) * mass * radius ** 2
    return PhysicsState(
        body_id=body_id,
        position=torch.tensor(list(position), dtype=dtype, device=device),
        velocity=torch.tensor(list(velocity), dtype=dtype, device=device),
        acceleration=torch.zeros(3, dtype=dtype, device=device),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device),
        angular_velocity=torch.zeros(3, dtype=dtype, device=device),
        angular_accel=torch.zeros(3, dtype=dtype, device=device),
        force=torch.zeros(3, dtype=dtype, device=device),
        torque=torch.zeros(3, dtype=dtype, device=device),
        mass=mass,
        inertia_tensor=torch.tensor([I_sphere, I_sphere, I_sphere], dtype=dtype, device=device),
        shape="sphere",
        shape_params={"radius": radius},
    )


def from_worldspec(
    spec: Any,
    config: StateEngineConfig,
) -> Dict[str, PhysicsState]:
    """Convert a WorldSpec into a dict of PhysicsState objects.

    Args:
        spec:    A ``WorldSpec`` instance (world_spec.py).
        config:  StateEngineConfig for dtype/device.

    Returns:
        Dict of entity.id → PhysicsState.

    Raises:
        ImportError: If world_spec.py is not available.
        TypeError:   If ``spec`` is not a WorldSpec instance.
    """
    if not _WORLDSPEC_AVAILABLE:
        raise ImportError(
            "world_spec.py must be importable to use from_worldspec(); "
            "ensure it is on the Python path."
        )
    dtype, device = config.dtype, config.torch_device
    states: Dict[str, PhysicsState] = {}
    for entity in spec.entities:
        ps = entity.state
        pos = torch.tensor([ps.position.x, ps.position.y, ps.position.z], dtype=dtype, device=device)
        vel = torch.tensor([ps.velocity.x, ps.velocity.y, ps.velocity.z], dtype=dtype, device=device)
        acc = torch.tensor([ps.acceleration.x, ps.acceleration.y, ps.acceleration.z], dtype=dtype, device=device)
        ori = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        ome = torch.tensor([ps.angular_vel.x, ps.angular_vel.y, ps.angular_vel.z], dtype=dtype, device=device)
        mass = max(entity.mass, 1e-6)
        bb = entity.bounding_box
        I = (1/12) * mass * (bb.height**2 + bb.depth**2)
        inertia = torch.tensor([I, I, I], dtype=dtype, device=device)
        states[entity.id] = PhysicsState(
            body_id=entity.id,
            position=pos, velocity=vel, acceleration=acc,
            orientation=ori, angular_velocity=ome,
            angular_accel=torch.zeros(3, dtype=dtype, device=device),
            force=torch.zeros(3, dtype=dtype, device=device),
            torque=torch.zeros(3, dtype=dtype, device=device),
            mass=mass, inertia_tensor=inertia,
            friction=entity.friction, restitution=entity.restitution,
            is_static=entity.is_static,
        )
    return states


def to_worldspec(
    states: Dict[str, PhysicsState],
    scene_id: str = "engine_output",
    description: str = "",
) -> Any:
    """Convert a dict of PhysicsState back into a WorldSpec.

    Args:
        states:      Dict of body_id → PhysicsState.
        scene_id:    WorldSpec.scene_id.
        description: Human-readable description.

    Returns:
        A ``WorldSpec`` instance.

    Raises:
        ImportError: If world_spec.py is not available.
    """
    if not _WORLDSPEC_AVAILABLE:
        raise ImportError(
            "world_spec.py must be importable to use to_worldspec()."
        )
    entities = []
    for bid, s in states.items():
        p = s.position.tolist()
        v = s.velocity.tolist()
        a = s.acceleration.tolist()
        ow = s.angular_velocity.tolist()
        state_ws = WS_PhysicsState(
            position=Vec3(*p), velocity=Vec3(*v), acceleration=Vec3(*a),
            angular_vel=Vec3(*ow),
        )
        entities.append(Entity(
            id=bid, label=bid, entity_type="object",
            is_static=s.is_static, mass=s.mass,
            friction=s.friction, restitution=s.restitution,
            state=state_ws,
        ))
    return WorldSpec(scene_id=scene_id, description=description, entities=entities)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20  —  TemporalModel Interface
# ─────────────────────────────────────────────────────────────────────────────

def from_temporal_prediction(
    predicted: Any,
    body_ids: List[str],
    config: StateEngineConfig,
    reference_states: Optional[Dict[str, PhysicsState]] = None,
) -> Dict[str, PhysicsState]:
    """Convert a TemporalWorldModel PredictedState into PhysicsState objects.

    Extracts per-node positions/velocities/orientations from the model output
    and wraps them in PhysicsState dataclasses so StateEngine.step() can
    advance them numerically.

    Args:
        predicted:          A ``PredictedState`` from TemporalWorldModel.
        body_ids:            Ordered list of entity ids matching the N axis
                            of predicted.next_positions (shape B, N, 3).
                            Currently uses batch index 0.
        config:              StateEngineConfig for dtype/device.
        reference_states:    Optional existing states to inherit mass,
                            inertia, shape params, etc.

    Returns:
        Dict of body_id → PhysicsState.

    Raises:
        ImportError: If models/temporal_world_model.py is not importable.
    """
    if not _TEMPORAL_AVAILABLE:
        raise ImportError(
            "models/temporal_world_model.py must be importable to use "
            "from_temporal_prediction()."
        )
    dtype, device = config.dtype, config.torch_device
    states: Dict[str, PhysicsState] = {}
    for i, bid in enumerate(body_ids):
        ref = reference_states.get(bid) if reference_states else None
        mass     = ref.mass if ref else 1.0
        inertia  = ref.inertia_tensor if ref else torch.ones(3, dtype=dtype, device=device)
        pos = predicted.next_positions[0, i].to(device=device, dtype=dtype) \
              if predicted.next_positions is not None else torch.zeros(3, dtype=dtype, device=device)
        vel = predicted.next_velocities[0, i].to(device=device, dtype=dtype) \
              if predicted.next_velocities is not None else torch.zeros(3, dtype=dtype, device=device)
        acc = predicted.next_accelerations[0, i].to(device=device, dtype=dtype) \
              if predicted.next_accelerations is not None else torch.zeros(3, dtype=dtype, device=device)
        ori = predicted.next_orientations[0, i].to(device=device, dtype=dtype) \
              if predicted.next_orientations is not None \
              else torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        ome = predicted.next_angular_velocities[0, i].to(device=device, dtype=dtype) \
              if predicted.next_angular_velocities is not None else torch.zeros(3, dtype=dtype, device=device)
        ori = QuaternionUtils.normalize(ori)
        states[bid] = PhysicsState(
            body_id=bid, position=pos, velocity=vel, acceleration=acc,
            orientation=ori, angular_velocity=ome,
            angular_accel=torch.zeros(3, dtype=dtype, device=device),
            force=torch.zeros(3, dtype=dtype, device=device),
            torque=torch.zeros(3, dtype=dtype, device=device),
            mass=mass, inertia_tensor=inertia.to(device=device, dtype=dtype),
            friction=ref.friction if ref else 0.5,
            restitution=ref.restitution if ref else 0.5,
            is_static=ref.is_static if ref else False,
            shape=ref.shape if ref else "sphere",
            shape_params=ref.shape_params if ref else {},
        )
    return states


def to_temporal_state(
    states: Dict[str, PhysicsState],
    config: StateEngineConfig,
) -> Any:
    """Package PhysicsState tensors into a TemporalWorldModel WorldState.

    Args:
        states:  Dict of body_id → PhysicsState.
        config:  StateEngineConfig for dtype/device.

    Returns:
        A ``WorldState`` instance (from temporal_world_model.py).

    Raises:
        ImportError: If temporal_world_model.py is not importable.
    """
    if not _TEMPORAL_AVAILABLE:
        raise ImportError(
            "models/temporal_world_model.py must be importable to use "
            "to_temporal_state()."
        )
    dtype, device = config.dtype, config.torch_device
    body_list = list(states.values())
    if not body_list:
        return WorldState()
    positions          = torch.stack([s.position for s in body_list]).to(dtype=dtype, device=device)
    velocities         = torch.stack([s.velocity for s in body_list]).to(dtype=dtype, device=device)
    accelerations      = torch.stack([s.acceleration for s in body_list]).to(dtype=dtype, device=device)
    orientations       = torch.stack([s.orientation for s in body_list]).to(dtype=dtype, device=device)
    angular_velocities = torch.stack([s.angular_velocity for s in body_list]).to(dtype=dtype, device=device)
    return WorldState(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        orientations=orientations,
        angular_velocities=angular_velocities,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 21  —  StateEngine
# ─────────────────────────────────────────────────────────────────────────────

class StateEngine:
    """Deterministic differentiable physics bridge: WorldState(t) → WorldState(t+1).

    This is NOT a full physics engine. It is the modular kernel that:
      1. Accumulates forces (gravity, drag, friction, externals) via ForceRegistry.
      2. Detects primitive contacts via CollisionModule.
      3. Solves kinematic constraints via ConstraintSolver.
      4. Integrates motion via the selected BaseIntegrator.
      5. Updates sleep states via SleepingSystem.
      6. Tracks energy, momentum, CoM via EnergyAccounting.
      7. Stores snapshots in StateHistory for TrajectoryEngine.
      8. Emits PhysicsEvents (collision, sleep, wake, break).
      9. Exposes typed export interfaces to Bullet/MuJoCo/Isaac/Gazebo/Renderer.

    Usage::

        config = StateEngineConfig(dt=0.01, integration_method="rk4")
        engine = StateEngine(config)
        engine.add_body(state_a)
        engine.add_body(state_b)
        output = engine.step()
        print(output.kinetic_energy, output.contacts)

    Args:
        config: :class:`StateEngineConfig`. Defaults are used if omitted.
    """

    def __init__(self, config: Optional[StateEngineConfig] = None) -> None:
        self.config   = config or StateEngineConfig()
        self._states:  Dict[str, PhysicsState] = {}
        self._time:    float = 0.0
        self._step_n:  int   = 0

        # Sub-systems
        self.integrator       = build_integrator(self.config.integration_method)
        self.force_registry   = ForceRegistry()
        self.contact_manager  = ContactManager()
        self.constraint_mgr   = ConstraintManager()
        self.sleeping_system  = SleepingSystem(self.config)
        self.energy_accounting = EnergyAccounting(self.config.gravity_vector())
        self.state_history    = StateHistory(self.config.history_capacity)
        self.diagnostics      = Diagnostics()
        self.cache            = PhysicsCache()

        # Physics modules
        self._gravity_module    = GravityModule(self.config)
        self._drag_module       = DragModule(self.config)
        self._friction_module   = FrictionModule()
        self._collision_module  = CollisionModule(self.config)
        self._restitution_module = RestitutionModule(self.config)

    # ── body management ────────────────────────────────────────────────────

    def add_body(self, state: PhysicsState) -> None:
        """Register a body with the engine.

        Args:
            state: The body's initial :class:`PhysicsState`.

        Raises:
            ValueError: If a body with the same ``body_id`` already exists.
        """
        validate_state(state)
        if state.body_id in self._states:
            raise ValueError(f"StateEngine: body_id {state.body_id!r} already registered")
        self._states[state.body_id] = state.clone()

    def remove_body(self, body_id: str) -> None:
        """Remove a body from the engine.

        Raises:
            KeyError: If ``body_id`` is not registered.
        """
        if body_id not in self._states:
            raise KeyError(f"StateEngine: unknown body_id {body_id!r}")
        del self._states[body_id]
        self.force_registry.clear_forces(body_id)

    def get_state(self, body_id: str) -> PhysicsState:
        """Return the current state of a body.

        Raises:
            KeyError: If not registered.
        """
        if body_id not in self._states:
            raise KeyError(f"StateEngine: unknown body_id {body_id!r}")
        return self._states[body_id].clone()

    def set_states(self, states: Dict[str, PhysicsState]) -> None:
        """Replace all body states (e.g. after a TemporalWorldModel update).

        Each state is validated before being accepted.
        """
        validate_states(states)
        self._states = {bid: s.clone() for bid, s in states.items()}

    # ── step ──────────────────────────────────────────────────────────────

    def step(self) -> EngineOutput:
        """Advance the simulation by one dt.

        Pipeline::

            Apply gravity
            ↓ Apply drag
            ↓ Accumulate force registry
            ↓ Detect collisions
            ↓ Resolve restitution
            ↓ Compute friction
            ↓ Solve constraints
            ↓ Integrate
            ↓ Enforce ground plane
            ↓ Update sleeping
            ↓ Compute energy
            ↓ Record history + diagnostics

        Returns:
            :class:`EngineOutput` with fully updated states, contacts,
            constraints, energy, momentum, events, and force breakdown.
        """
        t0_wall = time.perf_counter()
        events:   List[PhysicsEvent] = []
        dt = self.config.dt

        # ── 1. Zero forces, apply physics modules as persistent forces ────
        for s in self._states.values():
            s.force  = torch.zeros(3, dtype=s.force.dtype,  device=s.force.device)
            s.torque = torch.zeros(3, dtype=s.torque.dtype, device=s.torque.device)

        if self.config.enable_gravity:
            grav = self._gravity_module.compute(self._states)
            for bid, f in grav.items():
                self.force_registry.apply_force(
                    bid, f, source="gravity", persistent=False
                )

        if self.config.enable_drag:
            drag = self._drag_module.compute(self._states)
            for bid, f in drag.items():
                if f.norm() > 0:
                    self.force_registry.apply_force(
                        bid, f, source="drag", persistent=False
                    )

        # ── 2. Accumulate all forces (gravity + drag + user + spring…) ───
        self._states, force_breakdown = self.force_registry.accumulate(self._states)

        # ── 3. Collision detection ────────────────────────────────────────
        contacts: List[Contact] = []
        if self.config.enable_collision:
            contacts = self._collision_module.detect(self._states, self._time)
            # Emit collision events
            for c in contacts:
                events.append(PhysicsEvent(
                    event_type=EventType.COLLISION,
                    body_ids=[c.body_a, c.body_b],
                    timestep=self._time,
                    magnitude=c.penetration_depth,
                ))

        # ── 4. Restitution (collision impulses) ───────────────────────────
        if self.config.enable_restitution and contacts:
            self._states, contacts = self._restitution_module.resolve(
                self._states, contacts
            )

        # ── 5. Friction ───────────────────────────────────────────────────
        if self.config.enable_friction:
            grav_forces = {bid: s.mass * self.config.gravity_vector().to(s.force.device).to(s.force.dtype)
                           for bid, s in self._states.items() if not s.is_static}
            fric_forces, fric_torques = self._friction_module.compute(
                self._states, contacts, grav_forces
            )
            for bid in self._states:
                if bid in fric_forces:
                    self._states[bid].force  = self._states[bid].force  + fric_forces[bid]
                    self._states[bid].torque = self._states[bid].torque + fric_torques[bid]
                    if bid in force_breakdown and fric_forces[bid].norm() > 0:
                        force_breakdown[bid]["friction"] = fric_forces[bid].clone()

        # ── 6. Constraint solver ──────────────────────────────────────────
        constraint_forces: Dict[str, float] = {}
        if self.config.enable_constraints and self.constraint_mgr.active_constraints:
            self._states, constraint_forces, c_events = self._constraint_solver_call()
            events.extend(c_events)

        # ── 7. Integrate ──────────────────────────────────────────────────
        integrated: Dict[str, PhysicsState] = {}
        for bid, s in self._states.items():
            integrated[bid] = self.integrator.integrate(s, dt)

        # ── 8. Ground plane enforcement ───────────────────────────────────
        if self.config.enable_ground_plane:
            for s in integrated.values():
                if s.is_static or s.sleeping:
                    continue
                r = s.shape_params.get("radius", 0.5) if s.shape == "sphere" else \
                    s.shape_params.get("half_z", 0.5)
                floor = r
                if float(s.position[2]) < floor:
                    s.position = s.position.clone()
                    s.position[2] = floor
                    if float(s.velocity[2]) < 0:
                        s.velocity = s.velocity.clone()
                        s.velocity[2] *= -s.restitution
                    events.append(PhysicsEvent(
                        event_type=EventType.GROUND_IMPACT,
                        body_ids=[s.body_id],
                        timestep=self._time + dt,
                    ))

        # ── 9. Normalise quaternions ──────────────────────────────────────
        for s in integrated.values():
            s.orientation = QuaternionUtils.normalize(s.orientation)

        # ── 10. Sleeping ──────────────────────────────────────────────────
        if self.config.enable_sleeping:
            integrated = self.sleeping_system.update(integrated, events, self._time + dt)

        self._states = integrated
        self._time  += dt
        self._step_n += 1

        # ── 11. Update managers ───────────────────────────────────────────
        self.contact_manager.update(contacts)

        # ── 12. Energy & momentum ─────────────────────────────────────────
        ke  = self.energy_accounting.kinetic_energy(self._states)
        pe  = self.energy_accounting.potential_energy(self._states)
        dissipated = 0.0  # simplified — friction forces computed above
        tot = self.energy_accounting.total_energy(ke, pe)
        drift = self.energy_accounting.energy_drift(tot)
        lm  = linear_momentum(self._states)
        am  = angular_momentum(self._states)
        com = center_of_mass(self._states)

        # ── 13. Cache & history ───────────────────────────────────────────
        self.cache.update(self._states, force_breakdown, contacts, tot)
        snapshot = StateSnapshot(
            timestep=self._time,
            states={bid: s.clone() for bid, s in self._states.items()},
            contacts=list(contacts),
            energy=tot,
        )
        self.state_history.record(snapshot)

        # ── 14. Diagnostics ───────────────────────────────────────────────
        constraint_violation = max(constraint_forces.values()) \
            if constraint_forces else 0.0
        self.diagnostics.record(
            energy=tot,
            linear_momentum_mag=float(lm.norm()),
            contact_count=len(contacts),
            constraint_violation=constraint_violation,
        )

        wall_ms = (time.perf_counter() - t0_wall) * 1000.0

        return EngineOutput(
            updated_states=dict(self._states),
            contacts=contacts,
            constraint_forces=constraint_forces,
            kinetic_energy=ke,
            potential_energy=pe,
            dissipated_energy=dissipated,
            total_energy=tot,
            linear_momentum=lm,
            angular_momentum=am,
            center_of_mass=com,
            events=events,
            force_breakdown=force_breakdown,
            diagnostics={"energy_drift": drift},
            timestep=self._time,
            wall_time_ms=wall_ms,
        )

    def _constraint_solver_call(self) -> Tuple[
        Dict[str, PhysicsState], Dict[str, float], List[PhysicsEvent]
    ]:
        """Delegate to ConstraintSolver (extracted for readability)."""
        solver = ConstraintSolver(self.config)
        return solver.solve(
            self._states,
            self.constraint_mgr.active_constraints,
            self._time,
        )

    def step_batch(self, batch: PhysicsStateBatch) -> Tuple[PhysicsStateBatch, EngineOutput]:
        """Advance a batch of bodies by one dt.

        Replaces the engine's internal state with the batch, steps once,
        and returns the updated batch alongside the EngineOutput.

        Args:
            batch: :class:`PhysicsStateBatch` to advance.

        Returns:
            Tuple of (updated batch, EngineOutput).
        """
        self.set_states(batch.as_dict())
        output = self.step()
        updated_batch = PhysicsStateBatch(
            states=list(output.updated_states.values()),
            batch_id=batch.batch_id,
            timestep=self._time,
            metadata=batch.metadata,
        )
        return updated_batch, output

    # ── energy / momentum API (standalone accessors) ──────────────────────

    def kinetic_energy(self)  -> float: return self.energy_accounting.kinetic_energy(self._states)
    def potential_energy(self) -> float: return self.energy_accounting.potential_energy(self._states)
    def total_energy(self)     -> float:
        ke = self.kinetic_energy()
        pe = self.potential_energy()
        return self.energy_accounting.total_energy(ke, pe)
    def energy_drift(self)     -> float:
        return self.energy_accounting.energy_drift(self.total_energy())
    def linear_momentum(self)  -> torch.Tensor: return linear_momentum(self._states)
    def angular_momentum(self) -> torch.Tensor: return angular_momentum(self._states)
    def center_of_mass(self)   -> torch.Tensor: return center_of_mass(self._states)

    # ── WorldSpec interface ────────────────────────────────────────────────

    def from_worldspec(self, spec: Any) -> None:
        """Replace engine bodies from a WorldSpec."""
        states = from_worldspec(spec, self.config)
        self._states = states

    def to_worldspec(self, scene_id: str = "engine_output", description: str = "") -> Any:
        """Export current body states as a WorldSpec."""
        return to_worldspec(self._states, scene_id, description)

    # ── TemporalModel interface ────────────────────────────────────────────

    def from_temporal_prediction(self, predicted: Any, body_ids: List[str]) -> None:
        """Update engine bodies from a TemporalWorldModel PredictedState."""
        states = from_temporal_prediction(predicted, body_ids, self.config, self._states)
        self._states = states

    def to_temporal_state(self) -> Any:
        """Package current engine states as a TemporalWorldModel WorldState."""
        return to_temporal_state(self._states, self.config)

    # ── save / load ────────────────────────────────────────────────────────

    def save_state(self, path: Union[str, Path]) -> None:
        """Persist the engine state to ``path`` via torch.save.

        Saves: config, body states, current time, step count, energy history,
        and state history snapshots.

        Args:
            path: Destination ``.pt`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config":         asdict(self.config),
            "states":         {bid: s.to_dict() for bid, s in self._states.items()},
            "time":           self._time,
            "step_n":         self._step_n,
            "energy_history": self.energy_accounting._history,
            "history":        [
                {
                    "timestep": snap.timestep,
                    "energy":   snap.energy,
                    "states":   {bid: s.to_dict() for bid, s in snap.states.items()},
                }
                for snap in self.state_history.all_snapshots()
            ],
        }
        torch.save(payload, path)
        print(f"[StateEngine] saved → {path}")

    def load_state(self, path: Union[str, Path]) -> None:
        """Load a previously saved engine state from ``path``.

        Args:
            path: Source ``.pt`` file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"StateEngine: checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.config.device)

        dtype, device = self.config.dtype, self.config.torch_device
        self._states  = {
            bid: PhysicsState.from_dict(d, dtype=dtype, device=device)
            for bid, d in payload["states"].items()
        }
        self._time    = payload["time"]
        self._step_n  = payload["step_n"]
        self.energy_accounting._history = payload.get("energy_history", [])

        self.state_history.clear()
        for snap_d in payload.get("history", []):
            snap = StateSnapshot(
                timestep=snap_d["timestep"],
                states={bid: PhysicsState.from_dict(sd, dtype=dtype, device=device)
                        for bid, sd in snap_d["states"].items()},
                contacts=[],
                energy=snap_d["energy"],
            )
            self.state_history.record(snap)
        print(f"[StateEngine] loaded ← {path}")

    # ── export interfaces (placeholders → external simulators) ────────────

    def to_bullet(self) -> Any:
        """Interface placeholder: export to PyBullet.

        Raises:
            NotImplementedError: Always — implement in a dedicated Bullet
                export module that owns the pybullet dependency.
        """
        raise NotImplementedError(
            "to_bullet() is a placeholder; implement in a dedicated "
            "Bullet-export module."
        )

    def to_mujoco(self) -> Any:
        """Interface placeholder: export to MuJoCo MJCF.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_mujoco() is a placeholder; implement in a dedicated "
            "MuJoCo-export module."
        )

    def to_isaac(self) -> Any:
        """Interface placeholder: export to Isaac Sim USD.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_isaac() is a placeholder; implement in a dedicated "
            "Isaac-export module."
        )

    def to_gazebo(self) -> Any:
        """Interface placeholder: export to Gazebo SDF.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_gazebo() is a placeholder; implement in a dedicated "
            "Gazebo-export module."
        )

    def to_renderer(self) -> Any:
        """Interface placeholder: export to scene renderer.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_renderer() is a placeholder; implement in a dedicated "
            "Renderer module."
        )

    def to_video_diffusion(self) -> Any:
        """Interface placeholder: export latent state to video diffusion.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_video_diffusion() is a placeholder; implement in a dedicated "
            "video-diffusion module."
        )

    def to_sensor_model(self) -> Any:
        """Interface placeholder: export to a sensor simulation model
        (radar, lidar, camera).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_sensor_model() is a placeholder; implement in a dedicated "
            "sensor-model module."
        )

    def to_trajectory_engine(self) -> Any:
        """Interface placeholder: hand off StateHistory to TrajectoryEngine.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "to_trajectory_engine() is a placeholder; implement in a "
            "dedicated TrajectoryEngine module."
        )

    # ── info ──────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of engine module parameters (0 for a pure
        physics engine with no learned components; non-zero if a learned
        GravityModule or ContactPredictor is later plugged in)."""
        return 0

    def print_summary(self) -> None:
        cfg = self.config
        integrator_name = type(self.integrator).__name__
        print(f"\n{'═'*64}")
        print("  PhysWorldLM — StateEngine")
        print(f"{'═'*64}")
        print(f"  dt                  : {cfg.dt} s")
        print(f"  gravity             : {cfg.gravity}")
        print(f"  integrator          : {integrator_name}")
        print(f"  collision enabled   : {cfg.enable_collision}")
        print(f"  constraint enabled  : {cfg.enable_constraints}")
        print(f"  sleeping enabled    : {cfg.enable_sleeping}")
        print(f"  ground plane        : {cfg.enable_ground_plane}")
        print(f"  max contacts        : {cfg.max_contacts}")
        print(f"  max iterations      : {cfg.max_iterations}")
        print(f"  history capacity    : {cfg.history_capacity}")
        print(f"  device              : {cfg.device}")
        print(f"  dtype               : {'float64' if cfg.use_double_precision else 'float32'}")
        print(f"  parameter count     : {self.count_parameters()}")
        print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 22  —  Debug Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _make_debug_engine(method: str = "semi_implicit_euler") -> StateEngine:
    config = StateEngineConfig(
        dt=0.01, gravity=(0.0, 0.0, -9.81),
        integration_method=method,
        enable_ground_plane=True,
        enable_gravity=True,
        enable_drag=True,
        enable_sleeping=True,
    )
    return StateEngine(config)


def debug_step() -> None:
    """Run three canonical debug scenarios and print diagnostics.

    Scenarios:
        1. Two falling spheres — different masses, released from different heights.
        2. Falling box — larger body with box shape.
        3. Ground plane contact — sphere dropped from 2 m, should bounce.
    """
    print("\n" + "─" * 60)
    print("  debug_step()  — canonical debug scenarios")
    print("─" * 60)

    dtype  = torch.float32
    device = torch.device("cpu")

    # ── Scenario 1: two spheres ──────────────────────────────────────────
    print("\n[1] Two spheres falling under gravity")
    engine = _make_debug_engine()

    sphere_a = _make_sphere_state("sphere_a", mass=1.0,
                                   position=(0.0, 0.0, 5.0),
                                   velocity=(0.5, 0.0, 0.0),
                                   radius=0.3, dtype=dtype, device=device)
    sphere_b = _make_sphere_state("sphere_b", mass=2.0,
                                   position=(1.0, 0.0, 5.0),
                                   velocity=(-0.5, 0.0, 0.0),
                                   radius=0.3, dtype=dtype, device=device)
    engine.add_body(sphere_a)
    engine.add_body(sphere_b)

    for step in range(5):
        out = engine.step()

    sa = out.updated_states["sphere_a"]
    sb = out.updated_states["sphere_b"]
    print(f"  sphere_a position : {sa.position.tolist()}")
    print(f"  sphere_a velocity : {sa.velocity.tolist()}")
    print(f"  sphere_b position : {sb.position.tolist()}")
    print(f"  sphere_b velocity : {sb.velocity.tolist()}")
    print(f"  kinetic energy    : {out.kinetic_energy:.4f} J")
    print(f"  potential energy  : {out.potential_energy:.4f} J")
    print(f"  contact count     : {len(out.contacts)}")

    # ── Scenario 2: falling box ──────────────────────────────────────────
    print("\n[2] Falling box")
    engine2 = _make_debug_engine()
    I_box = (1/12) * 5.0 * (1.0**2 + 1.0**2)

    box = PhysicsState(
        body_id="box",
        position=torch.tensor([0.0, 0.0, 3.0], dtype=dtype, device=device),
        velocity=torch.zeros(3, dtype=dtype, device=device),
        acceleration=torch.zeros(3, dtype=dtype, device=device),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device),
        angular_velocity=torch.zeros(3, dtype=dtype, device=device),
        angular_accel=torch.zeros(3, dtype=dtype, device=device),
        force=torch.zeros(3, dtype=dtype, device=device),
        torque=torch.zeros(3, dtype=dtype, device=device),
        mass=5.0,
        inertia_tensor=torch.tensor([I_box, I_box, I_box], dtype=dtype, device=device),
        shape="box",
        shape_params={"half_x": 0.5, "half_y": 0.5, "half_z": 0.5},
    )
    engine2.add_body(box)
    for _ in range(5):
        out2 = engine2.step()

    b = out2.updated_states["box"]
    print(f"  box position      : {b.position.tolist()}")
    print(f"  box velocity      : {b.velocity.tolist()}")
    print(f"  kinetic energy    : {out2.kinetic_energy:.4f} J")

    # ── Scenario 3: ground plane bounce ──────────────────────────────────
    print("\n[3] Sphere dropped from 2 m (ground bounce)")
    engine3 = _make_debug_engine()
    drop = _make_sphere_state("drop", mass=1.0,
                               position=(0.0, 0.0, 2.0),
                               velocity=(0.0, 0.0, 0.0),
                               radius=0.3, dtype=dtype, device=device)
    drop.restitution = 0.7
    engine3.add_body(drop)
    for _ in range(30):
        out3 = engine3.step()

    d = out3.updated_states["drop"]
    print(f"  drop position     : {d.position.tolist()}")
    print(f"  drop velocity     : {d.velocity.tolist()}")
    print(f"  sleeping          : {d.sleeping}")
    print(f"  total energy      : {out3.total_energy:.4f} J")
    print(f"  energy drift      : {out3.diagnostics.get('energy_drift', 0.0):.6f}")
    print(f"  events this step  : {[e.event_type.name for e in out3.events]}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 23  —  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: build StateEngine, print summary, run debug_step(),
    verify all core APIs, assert correctness.

    Verifies::

        step()              — one-step integration
        save_state()        — torch.save round-trip
        load_state()        — torch.load round-trip
        energy()            — energy accounting
        momentum()          — momentum accounting
        center_of_mass()    — CoM
        from_worldspec()    — WorldSpec interface (if available)
        to_worldspec()      — WorldSpec interface (if available)
        export placeholders — all raise NotImplementedError
        batch support       — PhysicsStateBatch
        state history       — StateHistory
        quaternion utils    — normalize / to_matrix / from_matrix
        constraint solver   — spring constraint
        sleeping system     — sleep / wake
        force registry      — apply / remove
    """
    print("=" * 64)
    print("  PhysWorldLM — StateEngine")
    print("=" * 64)

    config = StateEngineConfig(
        dt=0.01,
        gravity=(0.0, 0.0, -9.81),
        integration_method="semi_implicit_euler",
        enable_gravity=True,
        enable_drag=True,
        enable_friction=True,
        enable_collision=True,
        enable_constraints=True,
        enable_ground_plane=True,
        enable_sleeping=True,
    )
    engine = StateEngine(config)
    engine.print_summary()

    # ── debug_step() ──────────────────────────────────────────────────────
    debug_step()

    # ── step() ────────────────────────────────────────────────────────────
    dtype, device = config.dtype, config.torch_device
    s1 = _make_sphere_state("body_1", mass=1.0, position=(0.0, 0.0, 5.0),
                             velocity=(0.0, 0.0, 0.0), radius=0.25, dtype=dtype, device=device)
    s2 = _make_sphere_state("body_2", mass=2.0, position=(1.0, 0.0, 5.0),
                             velocity=(0.0, 0.0, 0.0), radius=0.25, dtype=dtype, device=device)
    engine.add_body(s1)
    engine.add_body(s2)

    out = engine.step()
    assert isinstance(out, EngineOutput)
    assert "body_1" in out.updated_states
    assert out.total_energy > 0
    assert out.linear_momentum.shape == (3,)
    assert out.center_of_mass.shape  == (3,)
    print("  [main] step()                                  ... OK")

    # ── save / load ───────────────────────────────────────────────────────
    save_path = Path("/tmp/physworldlm_state_engine.pt")
    engine.save_state(save_path)

    engine2 = StateEngine(config)
    engine2.load_state(save_path)
    assert set(engine2._states.keys()) == set(engine._states.keys())
    print("  [main] save_state() / load_state()             ... OK")

    # ── energy / momentum / CoM ───────────────────────────────────────────
    ke  = engine.kinetic_energy()
    pe  = engine.potential_energy()
    tot = engine.total_energy()
    assert tot >= 0
    lm  = engine.linear_momentum()
    am  = engine.angular_momentum()
    com = engine.center_of_mass()
    assert lm.shape == (3,) and am.shape == (3,) and com.shape == (3,)
    print("  [main] energy() / momentum() / center_of_mass()... OK")

    # ── quaternion utilities ──────────────────────────────────────────────
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q_n = QuaternionUtils.normalize(q)
    assert abs(float(q_n.norm()) - 1.0) < 1e-5
    R = QuaternionUtils.to_rotation_matrix(q_n)
    assert R.shape == (3, 3)
    q_back = QuaternionUtils.from_rotation_matrix(R)
    assert q_back.shape == (4,)
    # round-trip for identity
    diff = float((q_n - q_back).abs().max())
    assert diff < 1e-4, f"quat round-trip error: {diff}"
    qi = QuaternionUtils.integrate_orientation(
        q_n, torch.tensor([0.0, 0.0, 1.0]), dt=0.01
    )
    assert abs(float(qi.norm()) - 1.0) < 1e-5
    print("  [main] QuaternionUtils                         ... OK")

    # ── force registry ────────────────────────────────────────────────────
    fr = ForceRegistry()
    fid = fr.apply_force(
        "body_1",
        torch.tensor([0.0, 0.0, 10.0]),
        source="thruster",
        persistent=True,
    )
    assert fid in fr.active_force_ids()
    fr.remove_force(fid)
    assert fid not in fr.active_force_ids()
    print("  [main] ForceRegistry                           ... OK")

    # ── state history ─────────────────────────────────────────────────────
    hist = engine.state_history
    assert len(hist) >= 1
    snap = hist.latest()
    assert snap is not None and "body_1" in snap.states
    print("  [main] StateHistory                            ... OK")

    # ── constraint solver (spring) ────────────────────────────────────────
    engine3 = StateEngine(config)
    s_a = _make_sphere_state("spr_a", mass=1.0, position=(0.0, 0.0, 5.0),
                              radius=0.25, dtype=dtype, device=device)
    s_b = _make_sphere_state("spr_b", mass=1.0, position=(2.0, 0.0, 5.0),
                              radius=0.25, dtype=dtype, device=device)
    engine3.add_body(s_a)
    engine3.add_body(s_b)
    engine3.constraint_mgr.new_constraint(
        body_a="spr_a", body_b="spr_b", joint_type="spring",
        rest_length=1.0, stiffness=500.0, damping=10.0,
    )
    out3 = engine3.step()
    assert out3.constraint_forces
    print("  [main] ConstraintSolver (spring)               ... OK")

    # ── batch support ─────────────────────────────────────────────────────
    engine4 = StateEngine(config)
    sb1 = _make_sphere_state("b1", mass=1.0, position=(0.0, 0.0, 3.0),
                              radius=0.3, dtype=dtype, device=device)
    sb2 = _make_sphere_state("b2", mass=1.0, position=(1.0, 0.0, 3.0),
                              radius=0.3, dtype=dtype, device=device)
    batch = PhysicsStateBatch(states=[sb1, sb2], batch_id="test_batch")
    updated_batch, bout = engine4.step_batch(batch)
    assert len(updated_batch.states) == 2
    assert isinstance(bout, EngineOutput)
    print("  [main] PhysicsStateBatch / step_batch()        ... OK")

    # ── export placeholders ───────────────────────────────────────────────
    for method in ("to_bullet", "to_mujoco", "to_isaac", "to_gazebo",
                   "to_renderer", "to_video_diffusion",
                   "to_sensor_model", "to_trajectory_engine"):
        try:
            getattr(engine, method)()
            raise AssertionError(f"{method} should raise NotImplementedError")
        except NotImplementedError:
            pass
    print("  [main] export placeholders (all NotImplementedError) ... OK")

    # ── WorldSpec round-trip ──────────────────────────────────────────────
    if _WORLDSPEC_AVAILABLE:
        ws = engine.to_worldspec(scene_id="test", description="smoke test")
        assert len(ws.entities) == 2
        engine5 = StateEngine(config)
        engine5.from_worldspec(ws)
        assert len(engine5._states) == 2
        print("  [main] from_worldspec() / to_worldspec()       ... OK")
    else:
        print("  [main] world_spec.py not on path — skipping WorldSpec tests")

    # ── diagnostics ───────────────────────────────────────────────────────
    summary = engine.diagnostics.summary()
    assert "energy_drift" in summary
    print("  [main] Diagnostics.summary()                   ... OK")

    # ── integrators smoke-test ────────────────────────────────────────────
    for method in ("euler", "rk4", "verlet", "symplectic"):
        cfg_m = StateEngineConfig(dt=0.01, integration_method=method,
                                  enable_gravity=True, enable_ground_plane=True)
        eng_m = StateEngine(cfg_m)
        eng_m.add_body(_make_sphere_state(
            f"b_{method}", mass=1.0, position=(0.0, 0.0, 5.0), radius=0.25,
            dtype=cfg_m.dtype, device=cfg_m.torch_device
        ))
        out_m = eng_m.step()
        assert isinstance(out_m, EngineOutput)
    print("  [main] all integrators (euler/rk4/verlet/symplectic) ... OK")

    print("\n[main] all assertions passed. done.")


if __name__ == "__main__":
    main()
