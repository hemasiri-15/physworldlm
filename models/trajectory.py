"""
trajectory.py
─────────────
Stores the time-evolution output of StateEngine as a sequence of per-entity
PhysicsState snapshots, one per exported frame.

Key design choices (publication-relevant):
  - Frame-level granularity  : export_fps controls storage density,
      while the integrator runs at full dt resolution internally.
  - Per-entity layout        : easy to slice trajectories by entity_id
      for downstream metrics (MSE vs analytic, energy plots, etc.)
  - Energy tracking          : KE + PE stored per frame for conservation
      diagnostics (essential for RK4 vs Euler comparison tables).
  - Round-trip serialization : to_dict / from_dict / to_json / save / load
      are all provided so dataset_gen can write HuggingFace-compatible records.

Coordinate conventions (inherited from WorldSpec):
  x = East, y = Up (gravity = -y), z = North
  All SI units: m, m/s, m/s², rad, rad/s
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import Optional
from models.world_spec import Vec3

# ─────────────────────────────────────────────
# Frame  (one timestep snapshot for one entity)
# ─────────────────────────────────────────────

@dataclass
class Frame:
    """State of a single entity at a single exported timestep."""
    t:             float          # simulation time, s
    entity_id:     str
    position:      tuple[float, float, float]   # (x, y, z) m
    velocity:      tuple[float, float, float]   # (x, y, z) m/s
    acceleration:  tuple[float, float, float]   # (x, y, z) m/s²
    orientation:   tuple[float, float, float]   # (rx, ry, rz) rad
    angular_vel:   tuple[float, float, float]   # (wx, wy, wz) rad/s
    kinetic_energy:   float = 0.0   # J
    potential_energy: float = 0.0   # J (gravitational, y-axis)

    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy

    def speed(self) -> float:
        return math.sqrt(
            self.velocity.x**2 +
            self.velocity.y**2 +
            self.velocity.z**2
        )

    def height(self) -> float:
        return self.position.y

    def to_dict(self) -> dict:
        return {
            "t":                  self.t,
            "entity_id":          self.entity_id,
            "position":           [self.position.x, self.position.y, self.position.z],
            "velocity":           [self.velocity.x, self.velocity.y, self.velocity.z],
            "acceleration":       [self.acceleration.x, self.acceleration.y, self.acceleration.z],
            "orientation":        [self.orientation.x, self.orientation.y, self.orientation.z],
            "angular_vel":        [self.angular_vel.x, self.angular_vel.y, self.angular_vel.z],
            "kinetic_energy_J":   round(self.kinetic_energy, 6),
            "potential_energy_J": round(self.potential_energy, 6),
            "total_energy_J":     round(self.total_energy, 6),
            "speed_ms":           round(self.speed(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            t=d["t"],
            entity_id=d["entity_id"],

            position=Vec3(*d["position"]),
            velocity=Vec3(*d["velocity"]),
            acceleration=Vec3(*d["acceleration"]),

            orientation=Vec3(
                *d.get("orientation", [0.0, 0.0, 0.0])
            ),

            angular_vel=Vec3(
                *d.get("angular_vel", [0.0, 0.0, 0.0])
            ),

            kinetic_energy=d.get("kinetic_energy_J", 0.0),
            potential_energy=d.get("potential_energy_J", 0.0),
        )

# ─────────────────────────────────────────────
# Trajectory  (full simulation output)
# ─────────────────────────────────────────────

@dataclass
class Trajectory:
    """
    Complete simulation output for one WorldSpec.

    Attributes
    ----------
    scene_id       : mirrors WorldSpec.scene_id
    description    : mirrors WorldSpec.description
    frames         : flat list of Frame objects, all entities interleaved
    entity_ids     : ordered list of simulated entity ids
    dt             : integration step used (s)
    export_fps     : frames exported per second
    integrator     : 'rk4' | 'euler' | 'verlet'
    total_steps    : total integrator steps taken
    wall_time_s    : wall-clock seconds to simulate
    metadata       : free-form dict (parser model, warnings, etc.)
    """
    scene_id:      str
    description:   str
    frames:        list[Frame]   = field(default_factory=list)
    entity_ids:    list[str]     = field(default_factory=list)
    dt:            float         = 0.01
    export_fps:    int           = 30
    integrator:    str           = "rk4"
    total_steps:   int           = 0
    wall_time_s:   float         = 0.0
    metadata:      dict          = field(default_factory=dict)

    # ── derived helpers ──────────────────────

    def add_frame(self, frame: Frame) -> None:
        """Append a frame to the trajectory."""
        self.frames.append(frame)

    def frames_for(self, entity_id: str) -> list[Frame]:
        """All frames for a specific entity, in time order."""
        return [f for f in self.frames if f.entity_id == entity_id]

    def times(self) -> list[float]:
        """Unique exported timestamps (deduplicated, sorted)."""
        seen, out = set(), []
        for f in self.frames:
            if f.t not in seen:
                seen.add(f.t); out.append(f.t)
        return sorted(out)

    def duration(self) -> float:
        ts = self.times()
        return ts[-1] - ts[0] if len(ts) > 1 else 0.0

    def final_frame(self, entity_id: str) -> Optional[Frame]:
        ef = self.frames_for(entity_id)
        return ef[-1] if ef else None

    # ── energy diagnostics ───────────────────

    def energy_drift(self, entity_id: str) -> float:
        """
        |E_final - E_initial| / E_initial  (dimensionless, ideally < 1e-4 for RK4).
        Returns 0.0 if initial energy is zero (free-fall from rest, etc.).
        """
        ef = self.frames_for(entity_id)
        if len(ef) < 2:
            return 0.0
        e0, e1 = ef[0].total_energy, ef[-1].total_energy
        return abs(e1 - e0) / abs(e0) if abs(e0) > 1e-10 else 0.0

    def max_speed(self, entity_id: str) -> float:
        ef = self.frames_for(entity_id)
        return max((f.speed() for f in ef), default=0.0)

    def displacement(self, entity_id: str) -> float:
        """Euclidean distance from initial to final position."""
        ef = self.frames_for(entity_id)

        if len(ef) < 2:
            return 0.0

        p0 = ef[0].position
        p1 = ef[-1].position

        dx = p1.x - p0.x
        dy = p1.y - p0.y
        dz = p1.z - p0.z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # ── serialization ────────────────────────

    def to_dict(self) -> dict:
        return {
            "scene_id":    self.scene_id,
            "description": self.description,
            "entity_ids":  self.entity_ids,
            "dt_s":        self.dt,
            "export_fps":  self.export_fps,
            "integrator":  self.integrator,
            "total_steps": self.total_steps,
            "wall_time_s": round(self.wall_time_s, 4),
            "metadata":    self.metadata,
            "frames":      [f.to_dict() for f in self.frames],
            "summary": {
                eid: {
                    "displacement_m":  round(self.displacement(eid), 4),
                    "max_speed_ms":    round(self.max_speed(eid), 4),
                    "energy_drift":    round(self.energy_drift(eid), 8),
                }
                for eid in self.entity_ids
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json())
        print(f"[Trajectory] saved → {path}")

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            scene_id=    d["scene_id"],
            description= d["description"],
            entity_ids=  d.get("entity_ids", []),
            dt=          d.get("dt_s", 0.01),
            export_fps=  d.get("export_fps", 30),
            integrator=  d.get("integrator", "rk4"),
            total_steps= d.get("total_steps", 0),
            wall_time_s= d.get("wall_time_s", 0.0),
            metadata=    d.get("metadata", {}),
            frames=      [Frame.from_dict(f) for f in d.get("frames", [])],
        )

    @classmethod
    def from_json(cls, path: str) -> "Trajectory":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
