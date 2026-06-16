"""
validator.py
────────────
PhysWorldLM — Phase 2: Validation Layer

Three-tier validation pipeline:

  Tier 1 — Schema Completeness
    Required fields present, ids non-empty, type strings valid.

  Tier 2 — Physics Consistency
    Mass, density, friction, restitution in physical bounds.
    Velocity and acceleration finite and plausible.
    Forces dimensionally consistent (Newtons = kg·m/s²).
    Kinematic energy checks.
    CFL condition: dt small enough for the fastest entity.

  Tier 3 — Semantic Consistency
    Impossible scenes (object underground, flying terrain).
    Contradictory interactions (static body with velocity ≠ 0).
    Orphaned interaction references.
    Physically impossible combinations (icy road + dry weather).
    Ground-contact completeness for surface entities.

Each issue carries a severity:
  ERROR   → simulation will fail or produce nonsense. Block.
  WARNING → physically unusual but technically runnable. Flag.
  INFO    → advisory note for the researcher.

Usage:
    from models.validator import PhysicsValidator
    result = PhysicsValidator().validate(spec)
    print(result.report())
    if not result.is_valid:
        raise ValueError("WorldSpec failed validation")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

# ── import the live schema ────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.world_spec import (
    WorldSpec, Entity, Environment, Interaction,
    SimulationGraph, Vec3, MATERIAL_DEFAULTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Issue dataclass
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


@dataclass
class Issue:
    severity:  Severity
    code:      str        # e.g.  "PHYS_MASS_ZERO"
    message:   str
    entity_id: str = ""   # which entity, if applicable

    def __str__(self) -> str:
        loc = f"[{self.entity_id}] " if self.entity_id else ""
        return f"  {self.severity.value:<7}  {self.code:<30}  {loc}{self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    issues: List[Issue] = field(default_factory=list)

    # ── convenience accessors ────────────────────────────────────────────────

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    # ── internal helpers ─────────────────────────────────────────────────────

    def _add(self, severity: Severity, code: str,
             message: str, entity_id: str = "") -> None:
        self.issues.append(Issue(severity, code, message, entity_id))

    def error(self, code: str, message: str, entity_id: str = "") -> None:
        self._add(Severity.ERROR, code, message, entity_id)

    def warn(self, code: str, message: str, entity_id: str = "") -> None:
        self._add(Severity.WARNING, code, message, entity_id)

    def info(self, code: str, message: str, entity_id: str = "") -> None:
        self._add(Severity.INFO, code, message, entity_id)

    # ── report ───────────────────────────────────────────────────────────────

    def report(self) -> str:
        header = (
            f"{'✓ VALID' if self.is_valid else '✗ INVALID'}"
            f"  —  {len(self.errors)} error(s)"
            f"  {len(self.warnings)} warning(s)"
            f"  {len(self.infos)} info(s)"
        )
        lines = [header, "─" * 72]
        for issue in self.issues:
            lines.append(str(issue))
        if not self.issues:
            lines.append("  (no issues)")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        """Machine-readable summary for logging / evaluation pipelines."""
        return {
            "is_valid":      self.is_valid,
            "error_count":   len(self.errors),
            "warning_count": len(self.warnings),
            "info_count":    len(self.infos),
            "error_codes":   [i.code for i in self.errors],
            "warning_codes": [i.code for i in self.warnings],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants & plausibility bounds
# ─────────────────────────────────────────────────────────────────────────────

class _Bounds:
    # Mass
    MASS_MIN_KG         = 1e-9       # nanogram — smallest physically meaningful object
    MASS_MAX_KG         = 1e15       # ~small asteroid
    MASS_WARN_LOW_KG    = 1e-3       # 1 gram
    MASS_WARN_HIGH_KG   = 1e8        # 100,000 tonnes

    # Velocity
    SPEED_MAX_MS        = 3e8        # speed of light — hard limit
    SPEED_WARN_MS       = 3000.0     # Mach 9 — warn above this

    # Acceleration
    ACCEL_MAX_MS2       = 1e8        # hard limit
    ACCEL_WARN_MS2      = 1000.0     # ~100g — warn above this

    # Friction coefficient
    FRICTION_MIN        = 0.0
    FRICTION_MAX        = 10.0       # theoretical max for specialised materials
    FRICTION_WARN_HIGH  = 2.0        # unusual but not impossible

    # Coefficient of restitution
    RESTITUTION_MIN     = 0.0
    RESTITUTION_MAX     = 1.0        # perfectly elastic — hard limit

    # Temperature (Kelvin)
    TEMP_MIN_K          = 0.0        # absolute zero — hard limit (< 0 is error)
    TEMP_WARN_LOW_K     = 4.0        # liquid helium
    TEMP_WARN_HIGH_K    = 6000.0     # surface of the Sun

    # Pressure
    PRESSURE_MIN_PA     = 0.0        # vacuum — allow 0 for space
    PRESSURE_WARN_LOW   = 1.0        # near-vacuum
    PRESSURE_MAX_PA     = 1e11       # interior of neutron star (block)

    # Air density
    AIR_DENSITY_MIN     = 0.0        # vacuum
    AIR_DENSITY_MAX     = 2000.0     # super-dense atmosphere (block)

    # Gravity
    GRAVITY_WARN_HIGH   = 300.0      # m/s²  — Jupiter surface ≈ 25

    # Geometry
    DIM_MIN_M           = 1e-6       # 1 micron
    DIM_MAX_M           = 1e6        # 1000 km

    # Simulation
    DT_MIN_S            = 1e-9       # 1 ns
    DT_MAX_S            = 1.0        # 1 second
    DURATION_MAX_S      = 86400.0    # 24 hours

    # CFL safety factor
    CFL_SAFETY          = 10.0       # entity must not move more than
                                     # (entity_size / CFL_SAFETY) per timestep

    # Ground level tolerance
    GROUND_TOLERANCE_M  = 0.01       # entity base within 1 cm of ground = OK


# ─────────────────────────────────────────────────────────────────────────────
# Known-bad semantic combinations
# ─────────────────────────────────────────────────────────────────────────────

# (weather, friction_range) pairs that are physically contradictory
_WEATHER_FRICTION_CONTRADICTIONS = [
    ("clear",  0.0, 0.15, "Clear weather but friction ≤ 0.15 implies ice/water — contradictory"),
    ("ice",    0.4, 10.0, "Icy weather but friction ≥ 0.4 is too high for ice"),
    ("snow",   0.5, 10.0, "Snowy weather but friction ≥ 0.5 is inconsistent with snow"),
]

# entity_type → expected is_static value (None = either is fine)
_TYPE_STATIC_EXPECTATIONS = {
    "terrain":   True,
    "structure": True,
    "fluid":     None,   # can be static (lake) or dynamic (flow)
    "vehicle":   False,
    "projectile":False,
    "agent":     False,
}

# Grounded entity types — must have a ground-contact interaction
_GROUNDED_TYPES = {"vehicle", "agent", "structure"}


# ─────────────────────────────────────────────────────────────────────────────
# PhysicsValidator
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsValidator:
    """
    Three-tier validator for WorldSpec objects.

    Call  validate(spec)  → ValidationResult
    """

    B = _Bounds  # shorthand

    def validate(self, spec: WorldSpec) -> ValidationResult:
        res = ValidationResult()

        # Tier 1 — Schema
        self._t1_schema(spec, res)

        # Tier 2 — Physics
        self._t2_physics_entities(spec, res)
        self._t2_physics_environment(spec, res)
        self._t2_physics_interactions(spec, res)
        self._t2_physics_simgraph(spec, res)
        self._t2_cfl_condition(spec, res)
        self._t2_energy_sanity(spec, res)

        # Tier 3 — Semantic
        self._t3_static_vs_velocity(spec, res)
        self._t3_ground_positioning(spec, res)
        self._t3_ground_contact_completeness(spec, res)
        self._t3_interaction_ref_integrity(spec, res)
        self._t3_weather_friction_consistency(spec, res)
        self._t3_type_static_consistency(spec, res)
        self._t3_force_dimensional_consistency(spec, res)
        self._t3_impossible_combinations(spec, res)

        return res

    # =========================================================================
    # Tier 1 — Schema Completeness
    # =========================================================================

    def _t1_schema(self, spec: WorldSpec, res: ValidationResult) -> None:
        """Required fields present; type strings valid."""

        if not spec.scene_id:
            res.error("SCHEMA_SCENE_ID_EMPTY", "scene_id is empty")
        if not spec.description:
            res.error("SCHEMA_DESCRIPTION_EMPTY", "description is empty")
        if not spec.entities:
            res.warn("SCHEMA_NO_ENTITIES", "WorldSpec has no entities")

        valid_types = {
            "vehicle", "projectile", "fluid", "agent",
            "structure", "terrain", "object",
        }
        valid_integrators = {"rk4", "euler", "verlet", "rk45"}

        seen_ids: set[str] = set()
        for e in spec.entities:
            eid = e.id or "(missing id)"
            if not e.id:
                res.error("SCHEMA_ENTITY_NO_ID", "Entity has no id", eid)
            if e.id in seen_ids:
                res.error("SCHEMA_DUPLICATE_ID", f"Duplicate entity id '{e.id}'", e.id)
            seen_ids.add(e.id)

            if not e.label:
                res.warn("SCHEMA_ENTITY_NO_LABEL", "Entity has no label", eid)
            if e.entity_type not in valid_types:
                res.warn(
                    "SCHEMA_UNKNOWN_ENTITY_TYPE",
                    f"entity_type '{e.entity_type}' not in known set {valid_types}",
                    eid,
                )
            if e.material not in MATERIAL_DEFAULTS:
                res.warn(
                    "SCHEMA_UNKNOWN_MATERIAL",
                    f"material '{e.material}' not in material library",
                    eid,
                )

        sg = spec.simulation_graph
        if sg.integrator not in valid_integrators:
            res.warn(
                "SCHEMA_UNKNOWN_INTEGRATOR",
                f"integrator '{sg.integrator}' not in {valid_integrators}",
            )

    # =========================================================================
    # Tier 2 — Physics Consistency
    # =========================================================================

    def _t2_physics_entities(self, spec: WorldSpec, res: ValidationResult) -> None:
        B = self.B
        for e in spec.entities:
            eid = e.id

            # ── Mass ──────────────────────────────────────────────────────────
            if not e.is_static:
                if e.mass <= 0:
                    res.error("PHYS_MASS_NONPOSITIVE",
                              f"mass = {e.mass} kg — must be > 0", eid)
                elif e.mass < B.MASS_MIN_KG:
                    res.error("PHYS_MASS_BELOW_PHYSICAL_LIMIT",
                              f"mass = {e.mass:.2e} kg — below nanogram threshold", eid)
                elif e.mass > B.MASS_MAX_KG:
                    res.error("PHYS_MASS_EXCEEDS_LIMIT",
                              f"mass = {e.mass:.2e} kg — exceeds asteroid threshold", eid)
                elif e.mass < B.MASS_WARN_LOW_KG:
                    res.warn("PHYS_MASS_VERY_SMALL",
                             f"mass = {e.mass:.2e} kg — unusually small", eid)
                elif e.mass > B.MASS_WARN_HIGH_KG:
                    res.warn("PHYS_MASS_VERY_LARGE",
                             f"mass = {e.mass:.2e} kg — unusually large", eid)

            # ── Friction ──────────────────────────────────────────────────────
            if e.friction < B.FRICTION_MIN:
                res.error("PHYS_FRICTION_NEGATIVE",
                          f"friction = {e.friction} — must be ≥ 0", eid)
            elif e.friction > B.FRICTION_MAX:
                res.error("PHYS_FRICTION_EXCEEDS_LIMIT",
                          f"friction = {e.friction} — exceeds physical maximum", eid)
            elif e.friction > B.FRICTION_WARN_HIGH:
                res.warn("PHYS_FRICTION_HIGH",
                         f"friction = {e.friction} — unusually high", eid)

            # ── Restitution ───────────────────────────────────────────────────
            if e.restitution < B.RESTITUTION_MIN:
                res.error("PHYS_RESTITUTION_NEGATIVE",
                          f"restitution = {e.restitution} — must be ≥ 0", eid)
            if e.restitution > B.RESTITUTION_MAX:
                res.error("PHYS_RESTITUTION_SUPERELASTIC",
                          f"restitution = {e.restitution} — exceeds 1.0 (violates energy conservation)", eid)

            # ── Velocity ──────────────────────────────────────────────────────
            spd = e.state.velocity.magnitude()
            if not math.isfinite(spd):
                res.error("PHYS_VELOCITY_NONFINITE",
                          "velocity contains NaN or Inf", eid)
            elif spd > B.SPEED_MAX_MS:
                res.error("PHYS_VELOCITY_SUPERLUMINAL",
                          f"speed = {spd:.3e} m/s — exceeds speed of light", eid)
            elif spd > B.SPEED_WARN_MS:
                res.warn("PHYS_VELOCITY_HIGH",
                         f"speed = {spd:.1f} m/s — above Mach 9", eid)

            # ── Acceleration ──────────────────────────────────────────────────
            acc = e.state.acceleration.magnitude()
            if not math.isfinite(acc):
                res.error("PHYS_ACCEL_NONFINITE",
                          "acceleration contains NaN or Inf", eid)
            elif acc > B.ACCEL_MAX_MS2:
                res.error("PHYS_ACCEL_EXCEEDS_LIMIT",
                          f"acceleration = {acc:.2e} m/s² — not physically plausible", eid)
            elif acc > B.ACCEL_WARN_MS2:
                res.warn("PHYS_ACCEL_HIGH",
                         f"acceleration = {acc:.1f} m/s² — above 100g", eid)

            # ── Bounding box ──────────────────────────────────────────────────
            bb = e.bounding_box
            for dim_name, dim_val in [
                ("width", bb.width), ("height", bb.height), ("depth", bb.depth)
            ]:
                if dim_val <= 0:
                    res.error("PHYS_DIM_NONPOSITIVE",
                              f"bounding_box.{dim_name} = {dim_val} — must be > 0", eid)
                elif dim_val < B.DIM_MIN_M:
                    res.warn("PHYS_DIM_VERY_SMALL",
                             f"bounding_box.{dim_name} = {dim_val} m — smaller than 1 micron", eid)
                elif dim_val > B.DIM_MAX_M:
                    res.warn("PHYS_DIM_VERY_LARGE",
                             f"bounding_box.{dim_name} = {dim_val} m — larger than 1000 km", eid)

            # ── Density cross-check ───────────────────────────────────────────
            if not e.is_static and e.mass > 0:
                vol = bb.volume()
                if vol > 0:
                    implied_density = e.mass / vol
                    mat_density     = MATERIAL_DEFAULTS.get(
                        e.material, MATERIAL_DEFAULTS["generic"]
                    )["density"]
                    ratio = implied_density / mat_density if mat_density > 0 else float("inf")
                    if ratio < 0.01 or ratio > 100:
                        res.warn(
                            "PHYS_DENSITY_INCONSISTENT",
                            f"implied density {implied_density:.1f} kg/m³ vs material "
                            f"'{e.material}' default {mat_density:.1f} kg/m³ "
                            f"(ratio {ratio:.2f}x) — mass or dimensions may be wrong",
                            eid,
                        )

    def _t2_physics_environment(self, spec: WorldSpec, res: ValidationResult) -> None:
        B   = self.B
        env = spec.environment

        # Temperature
        if env.temperature_K < 0:
            res.error("PHYS_TEMP_BELOW_ZERO_K",
                      f"temperature_K = {env.temperature_K} — below absolute zero")
        elif env.temperature_K == 0:
            res.warn("PHYS_TEMP_ABSOLUTE_ZERO",
                     "temperature_K = 0 — absolute zero; is this intentional?")
        elif env.temperature_K < B.TEMP_WARN_LOW_K:
            res.warn("PHYS_TEMP_VERY_LOW",
                     f"temperature_K = {env.temperature_K} K — near liquid helium")
        elif env.temperature_K > B.TEMP_WARN_HIGH_K:
            res.warn("PHYS_TEMP_VERY_HIGH",
                     f"temperature_K = {env.temperature_K} K — above surface of Sun")

        # Pressure
        if env.pressure_Pa < 0:
            res.error("PHYS_PRESSURE_NEGATIVE",
                      f"pressure_Pa = {env.pressure_Pa} — cannot be negative")
        elif env.pressure_Pa > B.PRESSURE_MAX_PA:
            res.error("PHYS_PRESSURE_EXCEEDS_LIMIT",
                      f"pressure_Pa = {env.pressure_Pa:.2e} — unrealistically high")
        elif 0 < env.pressure_Pa < B.PRESSURE_WARN_LOW:
            res.warn("PHYS_PRESSURE_NEAR_VACUUM",
                     f"pressure_Pa = {env.pressure_Pa} Pa — near vacuum")

        # Air density
        if env.air_density < 0:
            res.error("PHYS_AIR_DENSITY_NEGATIVE",
                      f"air_density = {env.air_density} — cannot be negative")
        elif env.air_density > B.AIR_DENSITY_MAX:
            res.error("PHYS_AIR_DENSITY_EXCEEDS_LIMIT",
                      f"air_density = {env.air_density} kg/m³ — not physical")

        # Gravity
        grav = env.gravity.magnitude()
        if not math.isfinite(grav):
            res.error("PHYS_GRAVITY_NONFINITE", "gravity vector contains NaN or Inf")
        elif grav == 0:
            res.info("PHYS_GRAVITY_ZERO",
                     "Gravity is zero — valid for space; confirm this is intentional")
        elif grav > B.GRAVITY_WARN_HIGH:
            res.warn("PHYS_GRAVITY_HIGH",
                     f"gravity magnitude = {grav:.1f} m/s² — very high (Jupiter = 24.8)")

        # Wind
        wind_spd = env.wind.speed
        if wind_spd < 0:
            res.error("PHYS_WIND_SPEED_NEGATIVE",
                      f"wind.speed = {wind_spd} m/s — must be ≥ 0")
        elif wind_spd > 150:
            res.warn("PHYS_WIND_SPEED_HIGH",
                     f"wind.speed = {wind_spd} m/s — above Category 5 hurricane (83 m/s)")

        if not math.isfinite(env.wind.direction):
            res.error("PHYS_WIND_DIR_NONFINITE", "wind.direction is NaN or Inf")

        # Global friction
        if env.friction_global < 0:
            res.error("PHYS_FRICTION_GLOBAL_NEGATIVE",
                      f"friction_global = {env.friction_global} — must be ≥ 0")
        elif env.friction_global > _Bounds.FRICTION_MAX:
            res.error("PHYS_FRICTION_GLOBAL_EXCEEDS_LIMIT",
                      f"friction_global = {env.friction_global} — exceeds physical max")

    def _t2_physics_interactions(self, spec: WorldSpec, res: ValidationResult) -> None:
        valid_types = {
            "collision", "joint", "contact", "friction",
            "fluid_drag", "magnetic", "gravity", "constraint",
        }
        for i, itr in enumerate(spec.interactions):
            if itr.type not in valid_types:
                res.warn("PHYS_INTERACTION_UNKNOWN_TYPE",
                         f"Interaction[{i}] type '{itr.type}' not in {valid_types}")

            # Friction coefficient in parameters
            mu = itr.parameters.get("mu_k") or itr.parameters.get("mu")
            if mu is not None:
                if mu < 0:
                    res.error("PHYS_INTERACTION_FRICTION_NEGATIVE",
                              f"Interaction[{i}] mu = {mu} — must be ≥ 0")
                elif mu > _Bounds.FRICTION_MAX:
                    res.error("PHYS_INTERACTION_FRICTION_EXCEEDS_LIMIT",
                              f"Interaction[{i}] mu = {mu} — exceeds physical max")

            # Drag coefficient sanity
            cd = itr.parameters.get("cd")
            if cd is not None:
                if cd < 0:
                    res.error("PHYS_DRAG_CD_NEGATIVE",
                              f"Interaction[{i}] cd = {cd} — must be ≥ 0")
                elif cd > 10:
                    res.warn("PHYS_DRAG_CD_HIGH",
                             f"Interaction[{i}] cd = {cd} — unusual for most shapes")

    def _t2_physics_simgraph(self, spec: WorldSpec, res: ValidationResult) -> None:
        B  = self.B
        sg = spec.simulation_graph

        if sg.dt <= 0:
            res.error("PHYS_SG_DT_NONPOSITIVE", f"dt = {sg.dt} s — must be > 0")
        elif sg.dt < B.DT_MIN_S:
            res.warn("PHYS_SG_DT_VERY_SMALL", f"dt = {sg.dt} s — below 1 ns")
        elif sg.dt > B.DT_MAX_S:
            res.warn("PHYS_SG_DT_VERY_LARGE", f"dt = {sg.dt} s — above 1 second")

        if sg.duration <= 0:
            res.error("PHYS_SG_DURATION_NONPOSITIVE",
                      f"duration = {sg.duration} s — must be > 0")
        elif sg.duration > B.DURATION_MAX_S:
            res.warn("PHYS_SG_DURATION_VERY_LONG",
                     f"duration = {sg.duration} s — longer than 24 hours")

        if sg.dt >= sg.duration:
            res.error("PHYS_SG_DT_EXCEEDS_DURATION",
                      f"dt ({sg.dt} s) ≥ duration ({sg.duration} s)")

        if sg.export_fps <= 0:
            res.warn("PHYS_SG_FPS_NONPOSITIVE", f"export_fps = {sg.export_fps}")
        elif sg.export_fps > 240:
            res.info("PHYS_SG_FPS_HIGH", f"export_fps = {sg.export_fps} — very high")

    def _t2_cfl_condition(self, spec: WorldSpec, res: ValidationResult) -> None:
        """
        Courant–Friedrichs–Lewy (CFL) condition.
        For stability: dt ≤ min_dim / (speed × CFL_SAFETY)
        If violated, the simulation will be numerically unstable.
        """
        dt = spec.simulation_graph.dt
        for e in spec.entities:
            if e.is_static:
                continue
            spd = e.state.velocity.magnitude()
            if spd == 0:
                continue
            min_dim = min(
                e.bounding_box.width,
                e.bounding_box.height,
                e.bounding_box.depth,
            )
            cfl_limit = min_dim / (spd * self.B.CFL_SAFETY)
            if dt > cfl_limit:
                res.warn(
                    "PHYS_CFL_VIOLATION",
                    f"dt={dt}s exceeds CFL limit {cfl_limit:.4f}s for this entity "
                    f"(speed={spd:.1f} m/s, min_dim={min_dim:.2f} m). "
                    "Reduce dt to avoid numerical instability.",
                    e.id,
                )

    def _t2_energy_sanity(self, spec: WorldSpec, res: ValidationResult) -> None:
        """
        Kinetic energy of dynamic entities should not suggest an impossible source.
        This is a soft check — just flags very high KE for review.
        """
        WARN_KE_JOULES = 1e12   # 1 TJ — roughly a tactical nuclear weapon
        total_ke = 0.0
        for e in spec.entities:
            if e.is_static:
                continue
            spd = e.state.velocity.magnitude()
            ke  = 0.5 * e.mass * spd ** 2
            total_ke += ke
            if ke > WARN_KE_JOULES:
                res.warn(
                    "PHYS_KE_VERY_HIGH",
                    f"kinetic energy = {ke:.2e} J — very high for a single entity",
                    e.id,
                )
        if total_ke > 1e14:
            res.warn("PHYS_TOTAL_KE_VERY_HIGH",
                     f"total kinetic energy = {total_ke:.2e} J across all entities")

    # =========================================================================
    # Tier 3 — Semantic Consistency
    # =========================================================================

    def _t3_static_vs_velocity(self, spec: WorldSpec, res: ValidationResult) -> None:
        """A static entity must have velocity = 0."""
        for e in spec.entities:
            if not e.is_static:
                continue
            spd = e.state.velocity.magnitude()
            if spd > 1e-6:
                res.error(
                    "SEM_STATIC_HAS_VELOCITY",
                    f"static entity has velocity magnitude {spd:.4f} m/s — "
                    "static bodies cannot move",
                    e.id,
                )

    def _t3_ground_positioning(self, spec: WorldSpec, res: ValidationResult) -> None:
        """
        Dynamic surface entities (vehicles, agents) must not start below ground.
        Assumes ground plane y = 0; entity base = position.y - height/2.
        """
        GROUND_Y = 0.0
        GROUND_TYPES = {"vehicle", "agent", "structure", "object"}
        for e in spec.entities:
            if e.is_static or e.entity_type not in GROUND_TYPES:
                continue
            base_y = e.state.position.y - e.bounding_box.height / 2.0
            if base_y < GROUND_Y - self.B.GROUND_TOLERANCE_M:
                res.error(
                    "SEM_ENTITY_UNDERGROUND",
                    f"entity base_y = {base_y:.3f} m is below ground (y=0). "
                    "Check position or bounding box height.",
                    e.id,
                )

    def _t3_ground_contact_completeness(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """
        Every grounded dynamic entity (vehicle, agent) should have at least
        one ground-contact or friction interaction, or be airborne (y > threshold).
        """
        GROUNDED_TYPES = {"vehicle", "agent"}
        AIRBORNE_THRESHOLD_M = 1.0   # if position.y > this, airborne — skip

        entity_has_ground_interaction: set[str] = set()
        for itr in spec.interactions:
            if itr.type in ("contact", "friction") and (
                itr.entity_b in {"environment", "e_road", "e_ground"}
                or any(
                    kw in itr.entity_b
                    for kw in ("road", "ground", "terrain", "surface", "floor")
                )
            ):
                entity_has_ground_interaction.add(itr.entity_a)

        for e in spec.entities:
            if e.is_static or e.entity_type not in GROUNDED_TYPES:
                continue
            if e.state.position.y > AIRBORNE_THRESHOLD_M:
                continue   # it's flying — no ground contact expected
            if e.id not in entity_has_ground_interaction:
                res.warn(
                    "SEM_MISSING_GROUND_CONTACT",
                    "grounded entity has no contact/friction interaction with ground. "
                    "Add a ground-contact interaction for physically correct simulation.",
                    e.id,
                )

    def _t3_interaction_ref_integrity(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """All entity ids referenced in interactions must exist."""
        ids = {e.id for e in spec.entities}
        for i, itr in enumerate(spec.interactions):
            if itr.entity_a not in ids:
                res.error(
                    "SEM_INTERACTION_BAD_REF_A",
                    f"Interaction[{i}] entity_a '{itr.entity_a}' not in entities",
                )
            if itr.entity_b not in ids and itr.entity_b != "environment":
                res.error(
                    "SEM_INTERACTION_BAD_REF_B",
                    f"Interaction[{i}] entity_b '{itr.entity_b}' not in entities",
                )

    def _t3_weather_friction_consistency(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """Catch weather/friction contradictions."""
        weather = spec.environment.weather
        mu      = spec.environment.friction_global

        for w_cond, mu_lo, mu_hi, msg in _WEATHER_FRICTION_CONTRADICTIONS:
            if weather == w_cond and mu_lo <= mu <= mu_hi:
                res.warn("SEM_WEATHER_FRICTION_CONTRADICTION", msg)

        # Rain should reduce friction below dry threshold (0.6)
        if weather == "rain" and mu > 0.6:
            res.warn(
                "SEM_RAIN_FRICTION_HIGH",
                f"weather='rain' but friction_global={mu:.2f} — "
                "rain typically reduces friction below 0.6",
            )

        # Snow should be very low friction
        if weather == "snow" and mu > 0.4:
            res.warn(
                "SEM_SNOW_FRICTION_HIGH",
                f"weather='snow' but friction_global={mu:.2f} — "
                "snow typically gives friction below 0.3",
            )

    def _t3_type_static_consistency(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """entity_type should match is_static expectation."""
        for e in spec.entities:
            expected = _TYPE_STATIC_EXPECTATIONS.get(e.entity_type)
            if expected is None:
                continue   # either value is fine
            if e.is_static != expected:
                res.warn(
                    "SEM_TYPE_STATIC_MISMATCH",
                    f"entity_type='{e.entity_type}' normally has is_static={expected} "
                    f"but got is_static={e.is_static}",
                    e.id,
                )

    def _t3_force_dimensional_consistency(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """
        Forces declared on entities should pass a basic dimensional check.
        Forces that are NOT per_unit_mass should have magnitude ≤ mass × 1000g.
        Per-unit-mass forces (accelerations) should be ≤ 1000 m/s².
        """
        for e in spec.entities:
            for f in e.forces:
                vec = f.get("vector_N", {})
                if not isinstance(vec, dict):
                    continue
                fx, fy, fz = (
                    vec.get("x", 0), vec.get("y", 0), vec.get("z", 0)
                )
                mag = math.sqrt(fx**2 + fy**2 + fz**2)

                if f.get("per_unit_mass"):
                    # This is really an acceleration (m/s²)
                    if mag > self.B.ACCEL_WARN_MS2:
                        res.warn(
                            "PHYS_FORCE_ACCEL_HIGH",
                            f"force '{f.get('label','')}' per-unit-mass magnitude "
                            f"{mag:.1f} m/s² — above 100g",
                            e.id,
                        )
                else:
                    # Absolute force (N)
                    max_plausible = e.mass * 10000  # 1000g × mass
                    if mag > max_plausible and e.mass > 0:
                        res.warn(
                            "PHYS_FORCE_TOO_LARGE",
                            f"force '{f.get('label','')}' = {mag:.2e} N on "
                            f"{e.mass} kg entity — implies {mag/e.mass:.0f} m/s² (>{10000} m/s²)",
                            e.id,
                        )

    def _t3_impossible_combinations(
        self, spec: WorldSpec, res: ValidationResult
    ) -> None:
        """Catch physically impossible or contradictory global combinations."""
        env = spec.environment

        # Liquid water at sub-zero temperature without pressure compensation
        if env.temperature_K < 273.15 and env.weather == "rain":
            res.warn(
                "SEM_RAIN_BELOW_FREEZING",
                f"weather='rain' but temperature_K={env.temperature_K:.1f} K "
                f"({env.temperature_K - 273.15:.1f} °C) — water would be ice",
            )

        # Fire/heat keywords in description but cold environment
        desc_lower = spec.description.lower()
        heat_words = {"fire", "flame", "explosion", "burning", "molten", "lava"}
        if any(w in desc_lower for w in heat_words) and env.temperature_K < 500:
            res.info(
                "SEM_HEAT_EVENT_COLD_ENV",
                "Scene description implies heat/fire but environment temperature is "
                f"{env.temperature_K:.1f} K — consider increasing temperature or "
                "adding a localised heat source entity",
            )

        # Space scene (gravity ≈ 0) but has weather
        grav = env.gravity.magnitude()
        if grav < 0.1 and env.weather != "clear":
            res.warn(
                "SEM_SPACE_SCENE_HAS_WEATHER",
                f"Gravity ≈ 0 (space scene) but weather='{env.weather}' — "
                "atmosphere-based weather is not possible in space",
            )

        # Underwater scene should have high pressure
        water_words = {"underwater", "submarine", "submerged", "ocean floor", "deep sea"}
        if any(w in desc_lower for w in water_words) and env.pressure_Pa < 200000:
            res.warn(
                "SEM_UNDERWATER_LOW_PRESSURE",
                f"Scene implies underwater but pressure_Pa={env.pressure_Pa:.0f} Pa — "
                "underwater pressure should be much higher than atmospheric",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible alias  (old code used WorldSpecValidator)
# ─────────────────────────────────────────────────────────────────────────────
WorldSpecValidator = PhysicsValidator
