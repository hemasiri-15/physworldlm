"""
validator.py
────────────
Validates a WorldSpec for:
  1. Schema completeness (required fields present)
  2. Physics plausibility (mass > 0, velocities ≤ speed of light, etc.)
  3. Internal consistency (interaction entity refs exist, no orphaned ids)
  4. SI unit sanity (temperature > 0 K, pressure > 0 Pa, etc.)

Usage:
    from models.validator import WorldSpecValidator
    result = WorldSpecValidator().validate(spec)
    print(result.report())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from models.world_spec import WorldSpec


@dataclass
class ValidationResult:
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def report(self) -> str:
        lines = [f"{'✓ VALID' if self.is_valid else '✗ INVALID'}"]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        return "\n".join(lines)


class WorldSpecValidator:

    # Physical plausibility limits
    MAX_SPEED_MS       = 3e8      # c
    MAX_MASS_KG        = 1e12     # asteroid
    MIN_MASS_KG        = 1e-6     # dust grain
    MAX_TEMP_K         = 1e7
    MIN_TEMP_K         = 0.001
    MIN_PRESSURE_PA    = 0.001

    def validate(self, spec: WorldSpec) -> ValidationResult:
        res = ValidationResult()
        self._check_schema(spec, res)
        self._check_entities(spec, res)
        self._check_environment(spec, res)
        self._check_interactions(spec, res)
        self._check_simgraph(spec, res)
        return res

    def _check_schema(self, spec: WorldSpec, res: ValidationResult):
        if not spec.scene_id:
            res.errors.append("scene_id is empty")
        if not spec.description:
            res.errors.append("description is empty")
        if not spec.entities:
            res.warnings.append("No entities defined")

    def _check_entities(self, spec: WorldSpec, res: ValidationResult):
        ids = set()
        for e in spec.entities:
            if not e.id:
                res.errors.append("Entity missing id")
                continue
            if e.id in ids:
                res.errors.append(f"Duplicate entity id: {e.id}")
            ids.add(e.id)

            if not e.is_static:
                if e.mass <= 0:
                    res.errors.append(f"[{e.id}] mass must be > 0, got {e.mass}")
                elif e.mass < self.MIN_MASS_KG:
                    res.warnings.append(f"[{e.id}] mass {e.mass} kg seems very small")
                elif e.mass > self.MAX_MASS_KG:
                    res.warnings.append(f"[{e.id}] mass {e.mass} kg seems very large")

            spd = e.state.velocity.magnitude()
            if spd > self.MAX_SPEED_MS:
                res.errors.append(f"[{e.id}] velocity {spd:.1f} m/s exceeds speed of light")
            elif spd > 1000:
                res.warnings.append(f"[{e.id}] velocity {spd:.1f} m/s seems very high")

            bb = e.bounding_box
            if bb.width <= 0 or bb.height <= 0 or bb.depth <= 0:
                res.errors.append(f"[{e.id}] bounding box dimensions must be > 0")

            if not e.entity_type:
                res.warnings.append(f"[{e.id}] entity_type is empty")

    def _check_environment(self, spec: WorldSpec, res: ValidationResult):
        env = spec.environment
        if env.temperature_K < self.MIN_TEMP_K:
            res.errors.append(f"temperature_K {env.temperature_K} is below absolute zero")
        elif env.temperature_K > self.MAX_TEMP_K:
            res.warnings.append(f"temperature_K {env.temperature_K} is extremely high")

        if env.pressure_Pa < self.MIN_PRESSURE_PA:
            res.warnings.append(f"pressure_Pa {env.pressure_Pa} is extremely low (near vacuum)")

        if env.air_density < 0:
            res.errors.append(f"air_density {env.air_density} must be ≥ 0")

        grav_mag = env.gravity.magnitude()
        if grav_mag == 0:
            res.warnings.append("Gravity is zero — is this a space scene?")
        elif grav_mag > 300:
            res.warnings.append(f"Gravity magnitude {grav_mag:.1f} m/s² seems very large")

    def _check_interactions(self, spec: WorldSpec, res: ValidationResult):
        ids = {e.id for e in spec.entities}
        for i, itr in enumerate(spec.interactions):
            if itr.entity_a not in ids:
                res.errors.append(
                    f"Interaction[{i}] entity_a '{itr.entity_a}' not in entities"
                )
            if itr.entity_b not in ids and itr.entity_b != "environment":
                res.errors.append(
                    f"Interaction[{i}] entity_b '{itr.entity_b}' not in entities"
                )
            if not itr.type:
                res.warnings.append(f"Interaction[{i}] has no type")

    def _check_simgraph(self, spec: WorldSpec, res: ValidationResult):
        sg = spec.simulation_graph
        if sg.dt <= 0:
            res.errors.append(f"SimulationGraph dt {sg.dt} must be > 0")
        if sg.duration <= 0:
            res.errors.append(f"SimulationGraph duration {sg.duration} must be > 0")
        if sg.dt >= sg.duration:
            res.errors.append("SimulationGraph dt must be less than duration")
        if sg.integrator not in ("rk4", "euler", "verlet"):
            res.warnings.append(f"Unknown integrator: {sg.integrator}")
        if sg.export_fps <= 0:
            res.warnings.append(f"export_fps {sg.export_fps} should be > 0")
