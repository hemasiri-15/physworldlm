"""
world_spec.py
─────────────
Canonical data contract for PhysWorldLM.
Every component (parser, encoder, state engine, dataset gen) imports from here.
All quantities are stored in SI units (kg, m, m/s, rad, Pa, K).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import math


# ─────────────────────────────────────────────
# SI unit conversion helpers
# ─────────────────────────────────────────────

def kmh_to_ms(v: float) -> float:
    return v / 3.6

def mph_to_ms(v: float) -> float:
    return v * 0.44704

def deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0

def celsius_to_kelvin(c: float) -> float:
    return c + 273.15

def fahrenheit_to_kelvin(f: float) -> float:
    return (f - 32) * 5/9 + 273.15


# ─────────────────────────────────────────────
# Material library  (static lookup)
# ─────────────────────────────────────────────

MATERIAL_DEFAULTS = {
    "steel":    {"density": 7850.0, "restitution": 0.6, "friction": 0.6},
    "rubber":   {"density":  950.0, "restitution": 0.8, "friction": 0.9},
    "wood":     {"density":  700.0, "restitution": 0.4, "friction": 0.5},
    "concrete": {"density": 2300.0, "restitution": 0.1, "friction": 0.7},
    "water":    {"density": 1000.0, "restitution": 0.0, "friction": 0.0},
    "glass":    {"density": 2500.0, "restitution": 0.5, "friction": 0.4},
    "flesh":    {"density":  985.0, "restitution": 0.3, "friction": 0.6},
    "plastic":  {"density": 1200.0, "restitution": 0.5, "friction": 0.5},
    "air":      {"density":    1.2, "restitution": 0.0, "friction": 0.0},
    "generic":  {"density":  500.0, "restitution": 0.5, "friction": 0.5},
}

# ─────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in metres."""
    width: float   = 1.0   # x
    height: float  = 1.0   # y
    depth: float   = 1.0   # z

    def volume(self) -> float:
        return self.width * self.height * self.depth

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


# ─────────────────────────────────────────────
# PhysicsState  (per-entity, per-timestep)
# ─────────────────────────────────────────────

@dataclass
class PhysicsState:
    """Full kinematic + dynamic state of one entity at one moment."""
    position:     Vec3  = field(default_factory=Vec3)   # m
    velocity:     Vec3  = field(default_factory=Vec3)   # m/s
    acceleration: Vec3  = field(default_factory=Vec3)   # m/s²
    orientation:  Vec3  = field(default_factory=Vec3)   # Euler angles, rad
    angular_vel:  Vec3  = field(default_factory=Vec3)   # rad/s

    def to_dict(self) -> dict:
        return {
            "position":     self.position.to_dict(),
            "velocity":     self.velocity.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "orientation":  self.orientation.to_dict(),
            "angular_vel":  self.angular_vel.to_dict(),
        }


# ─────────────────────────────────────────────
# Entity
# ─────────────────────────────────────────────

@dataclass
class Entity:
    """
    One physical object in the world.
    is_static=True  → immovable (terrain, walls, ground)
    is_static=False → dynamic (vehicles, projectiles, agents)
    """
    id:            str
    label:         str                         # human-readable name
    entity_type:   str                         # vehicle / projectile / fluid / agent / structure / terrain
    is_static:     bool          = False
    mass:          float         = 1.0         # kg
    material:      str           = "generic"
    restitution:   float         = 0.5         # coefficient of restitution [0,1]
    friction:      float         = 0.5         # kinetic friction coefficient
    bounding_box:  BoundingBox   = field(default_factory=BoundingBox)
    state:         PhysicsState  = field(default_factory=PhysicsState)
    forces:        list[dict]    = field(default_factory=list)  # applied force vectors
    constraints:   list[str]     = field(default_factory=list)  # references to other entity ids
    tags:          list[str]     = field(default_factory=list)  # semantic tags

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "label":        self.label,
            "entity_type":  self.entity_type,
            "is_static":    self.is_static,
            "mass_kg":      self.mass,
            "material":     self.material,
            "restitution":  self.restitution,
            "friction":     self.friction,
            "bounding_box": self.bounding_box.to_dict(),
            "state":        self.state.to_dict(),
            "forces":       self.forces,
            "constraints":  self.constraints,
            "tags":         self.tags,
        }


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

@dataclass
class Wind:
    speed:     float = 0.0   # m/s
    direction: float = 0.0   # rad from north

    def to_dict(self) -> dict:
        return {"speed_ms": self.speed, "direction_rad": self.direction}


@dataclass
class Environment:
    gravity:          Vec3   = field(default_factory=lambda: Vec3(0, -9.81, 0))  # m/s²
    temperature_K:    float  = 293.15    # 20 °C
    pressure_Pa:      float  = 101325.0  # sea-level
    air_density:      float  = 1.225     # kg/m³
    wind:             Wind   = field(default_factory=Wind)
    terrain_type:     str    = "flat"    # flat / hilly / urban / water / mixed
    friction_global:  float  = 0.5       # default ground friction
    time_of_day:      str    = "day"
    weather:          str    = "clear"   # clear / rain / snow / fog / wind

    def to_dict(self) -> dict:
        return {
            "gravity":         self.gravity.to_dict(),
            "temperature_K":   self.temperature_K,
            "pressure_Pa":     self.pressure_Pa,
            "air_density_kgm3": self.air_density,
            "wind":            self.wind.to_dict(),
            "terrain_type":    self.terrain_type,
            "friction_global": self.friction_global,
            "time_of_day":     self.time_of_day,
            "weather":         self.weather,
        }


# ─────────────────────────────────────────────
# Interaction / Constraint
# ─────────────────────────────────────────────

@dataclass
class Interaction:
    """A declared physics interaction between two entities."""
    type:       str        # collision / joint / contact / fluid_drag / magnetic
    entity_a:   str        # entity id
    entity_b:   str        # entity id or "environment"
    parameters: dict       = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type":       self.type,
            "entity_a":   self.entity_a,
            "entity_b":   self.entity_b,
            "parameters": self.parameters,
        }


# ─────────────────────────────────────────────
# SimulationGraph  (time-evolution specification)
# ─────────────────────────────────────────────

@dataclass
class SimulationGraph:
    """
    Describes how the world evolves over time.
    dt and duration drive the state engine's RK4 loop.
    events are discrete triggers (e.g. collision at t=1.2s).
    """
    dt:            float        = 0.01    # seconds per integration step
    duration:      float        = 10.0   # total simulation time, seconds
    integrator:    str          = "rk4"  # rk4 / euler / verlet
    events:        list[dict]   = field(default_factory=list)
    export_fps:    int          = 30      # frames to export for animation

    def to_dict(self) -> dict:
        return {
            "dt_s":        self.dt,
            "duration_s":  self.duration,
            "integrator":  self.integrator,
            "export_fps":  self.export_fps,
            "events":      self.events,
        }


# ─────────────────────────────────────────────
# WorldSpec  (top-level output)
# ─────────────────────────────────────────────

@dataclass
class WorldSpec:
    """
    The complete, physics-ready world representation.
    This is the canonical output of the Prompt → WorldSpec pipeline.
    Feed this directly into StateEngine or export to any physics engine.
    """
    scene_id:         str
    description:      str                    # original natural-language prompt
    entities:         list[Entity]           = field(default_factory=list)
    environment:      Environment            = field(default_factory=Environment)
    interactions:     list[Interaction]      = field(default_factory=list)
    simulation_graph: SimulationGraph        = field(default_factory=SimulationGraph)
    metadata:         dict                   = field(default_factory=dict)

    # ── derived helpers ──────────────────────

    def get_entity(self, eid: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def dynamic_entities(self) -> list[Entity]:
        return [e for e in self.entities if not e.is_static]

    def static_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.is_static]

    # ── serialisation ────────────────────────

    def to_dict(self) -> dict:
        return {
            "scene_id":         self.scene_id,
            "description":      self.description,
            "entities":         [e.to_dict() for e in self.entities],
            "environment":      self.environment.to_dict(),
            "interactions":     [i.to_dict() for i in self.interactions],
            "simulation_graph": self.simulation_graph.to_dict(),
            "metadata":         self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[WorldSpec] saved → {path}")

    @classmethod
    def from_dict(cls, d: dict) -> "WorldSpec":
        """Reconstruct a WorldSpec from a plain dict (e.g. loaded from JSON)."""
        entities = []
        for ed in d.get("entities", []):
            ps = ed.get("state", {})
            state = PhysicsState(
                position=     Vec3(**ps.get("position", {})),
                velocity=     Vec3(**ps.get("velocity", {})),
                acceleration= Vec3(**ps.get("acceleration", {})),
                orientation=  Vec3(**ps.get("orientation", {})),
                angular_vel=  Vec3(**ps.get("angular_vel", {})),
            )
            bb = BoundingBox(**ed.get("bounding_box", {}))
            entities.append(Entity(
                id=           ed["id"],
                label=        ed["label"],
                entity_type=  ed["entity_type"],
                is_static=    ed.get("is_static", False),
                mass=         ed.get("mass_kg", 1.0),
                material=     ed.get("material", "generic"),
                restitution=  ed.get("restitution", 0.5),
                friction=     ed.get("friction", 0.5),
                bounding_box= bb,
                state=        state,
                forces=       ed.get("forces", []),
                constraints=  ed.get("constraints", []),
                tags=         ed.get("tags", []),
            ))
        env_d  = d.get("environment", {})
        env    = Environment(
            gravity=         Vec3(**env_d.get("gravity", {"x": 0, "y": -9.81, "z": 0})),
            temperature_K=   env_d.get("temperature_K", 293.15),
            pressure_Pa=     env_d.get("pressure_Pa", 101325.0),
            air_density=     env_d.get("air_density_kgm3", 1.225),
            wind=            Wind(**{k: v for k, v in env_d.get("wind", {}).items()
                                     if k in ("speed_ms","direction_rad")}
                                  ) if env_d.get("wind") else Wind(),
            terrain_type=    env_d.get("terrain_type", "flat"),
            friction_global= env_d.get("friction_global", 0.5),
            time_of_day=     env_d.get("time_of_day", "day"),
            weather=         env_d.get("weather", "clear"),
        )
        sg_d = d.get("simulation_graph", {})
        sg   = SimulationGraph(
            dt=          sg_d.get("dt_s", 0.01),
            duration=    sg_d.get("duration_s", 10.0),
            integrator=  sg_d.get("integrator", "rk4"),
            export_fps=  sg_d.get("export_fps", 30),
            events=      sg_d.get("events", []),
        )
        interactions = [
            Interaction(**{k: v for k, v in i.items() if k in
                           ("type","entity_a","entity_b","parameters")})
            for i in d.get("interactions", [])
        ]
        return cls(
            scene_id=         d.get("scene_id", ""),
            description=      d.get("description", ""),
            entities=         entities,
            environment=      env,
            interactions=     interactions,
            simulation_graph= sg,
            metadata=         d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "WorldSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))
