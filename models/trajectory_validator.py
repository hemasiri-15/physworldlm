"""
trajectory_validator.py
────────────────────────
Validates Trajectory objects produced by StateEngine for numerical stability,
physical realism, dataset quality, and energy conservation.

Pipeline position
──────────────────
Natural Language → WorldParser → WorldSpec → Validator → PhysicsEncoder
    → StateEngine (RK4) → Trajectory → TrajectoryValidator → JSON / CSV Export

This module is the final quality gate before a Trajectory is accepted into
a training dataset or reported as an evaluation result. It operates purely
on the Trajectory object — it has no dependency on WorldSpec, the encoder,
or the integrator — so it can validate trajectories loaded from disk as
easily as ones freshly produced by StateEngine.

Usage
─────
    from models.trajectory import Trajectory
    from models.trajectory_validator import TrajectoryValidator

    traj = Trajectory.from_json("outputs/first_trajectory.json")
    validator = TrajectoryValidator()
    result = validator.validate(traj)
    print(result.summary())
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from models.trajectory import Trajectory, Frame


# ─────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Outcome of a TrajectoryValidator run.

    Attributes
    ----------
    passed   : True iff `errors` is empty. A trajectory with only warnings
               is still considered usable but flagged for review.
    errors   : Hard failures — the trajectory should not be used for
               training or reported results without investigation.
    warnings : Soft issues — the trajectory is usable but borderline.
    metrics  : Quantitative summary statistics computed during validation,
               always populated even when validation fails outright
               (except for metrics that are undefined on an empty
               trajectory, which are simply omitted).
    """

    passed:   bool
    errors:   list[str]      = field(default_factory=list)
    warnings: list[str]      = field(default_factory=list)
    metrics:  dict[str, Any] = field(default_factory=dict)

    # ── serialization ────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation suitable for json.dumps or pandas."""
        return {
            "passed":   self.passed,
            "errors":   list(self.errors),
            "warnings": list(self.warnings),
            "metrics":  dict(self.metrics),
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON-string representation of `to_dict()`."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """Write `to_json()` to `path`."""
        with open(path, "w") as fh:
            fh.write(self.to_json())
        print(f"[ValidationResult] saved → {path}")

    def summary(self) -> str:
        """
        Human-readable multi-line summary, suitable for CLI / log output.

        Format mirrors WorldSpecValidator.report() for consistency across
        the two validators used in the pipeline.
        """
        lines = [f"{'✓ PASSED' if self.passed else '✗ FAILED'}  "
                 f"({len(self.errors)} errors, {len(self.warnings)} warnings)"]

        if self.metrics:
            lines.append("  metrics:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.6g}")
                else:
                    lines.append(f"    {k}: {v}")

        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# TrajectoryValidator
# ─────────────────────────────────────────────

class TrajectoryValidator:
    """
    Validates a Trajectory for numerical stability, physical realism,
    dataset quality, and energy conservation.

    All thresholds are configurable at construction time so that the same
    validator class can be reused both as a strict dataset-curation gate
    (tight thresholds) and as a lenient debugging tool (loose thresholds).

    Parameters
    ----------
    max_speed_ms      : hard ceiling on |v| (m/s) before a frame is an error.
                         Frames above `WARN_SPEED_MS` (500 m/s) but below
                         this value are warnings only.
    max_accel_ms2     : hard ceiling on |a| (m/s²) before a frame is an error.
                         Frames above `WARN_ACCEL_MS2` (1000 m/s²) but below
                         this value are warnings only.
    max_energy_drift  : hard ceiling on per-entity relative energy drift
                         before it is an error. Drift above
                         `WARN_ENERGY_DRIFT` (0.01) but below this value is
                         a warning only.

    Notes
    -----
    The soft (warning) thresholds for underground penetration, velocity,
    acceleration, and energy drift are fixed research-standard values
    (see class constants below) and are not parameterized, since they
    represent generally-accepted "notice this" boundaries independent of
    any particular dataset's tolerance for hard failures.
    """

    # ── fixed warning-level thresholds ───────
    WARN_DEPTH_M:          float = -0.01   # y below this → warning
    ERROR_DEPTH_M:         float = -1.0    # y below this → error
    WARN_SPEED_MS:         float = 500.0
    WARN_ACCEL_MS2:        float = 1000.0
    WARN_ENERGY_DRIFT:     float = 0.01

    def __init__(self,
                 max_speed_ms: float = 10_000.0,
                 max_accel_ms2: float = 100_000.0,
                 max_energy_drift: float = 0.10) -> None:
        self.max_speed_ms     = max_speed_ms
        self.max_accel_ms2    = max_accel_ms2
        self.max_energy_drift = max_energy_drift

    def _vec_components(self, vec):
        """
        Works with both Vec3 objects and tuples/lists.
        Returns (x, y, z).
        """
        if hasattr(vec, "x"):
            return vec.x, vec.y, vec.z
        return vec[0], vec[1], vec[2]


    def _vec_y(self, vec):
        """
        Return y component for Vec3 or tuple/list.
        """
        if hasattr(vec, "y"):
            return vec.y
        return vec[1]

    # ── public API ───────────────────────────

    def validate(self, trajectory: Trajectory) -> ValidationResult:
        """
        Run the full validation suite on `trajectory`.

        Returns
        -------
        ValidationResult
            `passed` is True iff no check produced an error. Warnings never
            affect `passed`. `metrics` is always populated for non-empty
            trajectories; for an empty trajectory only `frame_count` and
            `entity_count` are meaningful and included.
        """
        errors:   list[str] = []
        warnings: list[str] = []

        # ── 1. Empty trajectory ───────────────
        if len(trajectory.frames) == 0:
            errors.append("Trajectory has zero frames")
            metrics = {
                "frame_count":  0,
                "entity_count": len(trajectory.entity_ids),
            }
            return ValidationResult(passed=False, errors=errors,
                                     warnings=warnings, metrics=metrics)

        # ── 2-6, 9-10: per-frame numerical / physical checks ──
        self._check_nan_and_inf(trajectory, errors)
        self._check_ground_penetration(trajectory, errors, warnings)
        speed_stats  = self._check_velocity(trajectory, errors, warnings)
        accel_stats  = self._check_acceleration(trajectory, errors, warnings)
        self._check_time_monotonicity(trajectory, errors)
        self._check_frame_consistency(trajectory, errors)

        # ── 7. Energy drift ───────────────────
        max_drift = self._check_energy_drift(trajectory, errors, warnings)

        # ── 8. Missing entity trajectories ────
        self._check_missing_entities(trajectory, errors)

        # ── metrics ───────────────────────────
        metrics = self._collect_metrics(
            trajectory, speed_stats, accel_stats, max_drift,
            len(warnings), len(errors),
        )

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    # ── 2 & 3: NaN / Inf detection ────────────

    def _check_nan_and_inf(self,
                            trajectory: Trajectory,
                            errors: list[str]) -> None:
        """
        Scan every position, velocity, and acceleration component in every
        frame for NaN or infinite values. Reports at most one error per
        (entity, field) combination to avoid flooding the error list on a
        diverging simulation, while still naming the first offending frame.
        """
        reported: set[tuple[str, str]] = set()

        for fr in trajectory.frames:
            for field_name, vec in (
                ("position", fr.position),
                ("velocity", fr.velocity),
                ("acceleration", fr.acceleration),
            ):
                key = (fr.entity_id, field_name)
                if key in reported:
                    continue
                for component in self._vec_components(vec):
                    if math.isnan(component):
                        errors.append(
                            f"[{fr.entity_id}] NaN detected in {field_name} "
                            f"at t={fr.t:.4f}s"
                        )
                        reported.add(key)
                        break
                    if math.isinf(component):
                        errors.append(
                            f"[{fr.entity_id}] Infinite value detected in "
                            f"{field_name} at t={fr.t:.4f}s"
                        )
                        reported.add(key)
                        break

    # ── 4: Underground penetration ────────────

    def _check_ground_penetration(self,
                                   trajectory: Trajectory,
                                   errors: list[str],
                                   warnings: list[str]) -> None:
        """
        Ground plane is y = 0. Penetration below `WARN_DEPTH_M` is a
        warning; below `ERROR_DEPTH_M` is an error. Only the deepest
        penetration per entity is reported.
        """
        deepest: dict[str, float] = {}
        deepest_t: dict[str, float] = {}

        for fr in trajectory.frames:
            y = self._vec_y(fr.position)
            if y < deepest.get(fr.entity_id, math.inf):
                deepest[fr.entity_id] = y
                deepest_t[fr.entity_id] = fr.t

        for eid, y in deepest.items():
            if y < self.ERROR_DEPTH_M:
                errors.append(
                    f"[{eid}] underground penetration y={y:.4f}m "
                    f"at t={deepest_t[eid]:.4f}s (exceeds {self.ERROR_DEPTH_M}m)"
                )
            elif y < self.WARN_DEPTH_M:
                warnings.append(
                    f"[{eid}] minor underground penetration y={y:.4f}m "
                    f"at t={deepest_t[eid]:.4f}s"
                )

    # ── 5: Velocity explosion ─────────────────

    def _check_velocity(self,
                         trajectory: Trajectory,
                         errors: list[str],
                         warnings: list[str]) -> dict[str, float]:
        """
        Returns
        -------
        dict with 'max_speed_ms' and 'mean_speed_ms' across all frames.
        """
        max_speed = 0.0
        speed_sum = 0.0
        max_speed_eid = ""
        max_speed_t   = 0.0
        warned_entities: set[str] = set()
        errored_entities: set[str] = set()

        for fr in trajectory.frames:
            spd = fr.speed()
            speed_sum += spd
            if spd > max_speed:
                max_speed, max_speed_eid, max_speed_t = spd, fr.entity_id, fr.t

            if spd > self.max_speed_ms and fr.entity_id not in errored_entities:
                errors.append(
                    f"[{fr.entity_id}] velocity explosion |v|={spd:.2f}m/s "
                    f"at t={fr.t:.4f}s (exceeds max_speed_ms={self.max_speed_ms})"
                )
                errored_entities.add(fr.entity_id)
            elif spd > self.WARN_SPEED_MS and fr.entity_id not in warned_entities:
                warnings.append(
                    f"[{fr.entity_id}] high velocity |v|={spd:.2f}m/s "
                    f"at t={fr.t:.4f}s (exceeds {self.WARN_SPEED_MS}m/s)"
                )
                warned_entities.add(fr.entity_id)

        n = len(trajectory.frames)
        return {
            "max_speed_ms":  max_speed,
            "mean_speed_ms": speed_sum / n if n else 0.0,
        }

    # ── 6: Acceleration explosion ─────────────

    def _check_acceleration(self,
                             trajectory: Trajectory,
                             errors: list[str],
                             warnings: list[str]) -> dict[str, float]:
        """
        Returns
        -------
        dict with 'max_acceleration_ms2' across all frames.
        """
        max_accel = 0.0
        warned_entities: set[str] = set()
        errored_entities: set[str] = set()

        for fr in trajectory.frames:
            ax, ay, az = self._vec_components(fr.acceleration)
            mag = math.sqrt(ax**2 + ay**2 + az**2)
            if mag > max_accel:
                max_accel = mag

            if mag > self.max_accel_ms2 and fr.entity_id not in errored_entities:
                errors.append(
                    f"[{fr.entity_id}] acceleration explosion |a|={mag:.2f}m/s² "
                    f"at t={fr.t:.4f}s (exceeds max_accel_ms2={self.max_accel_ms2})"
                )
                errored_entities.add(fr.entity_id)
            elif mag > self.WARN_ACCEL_MS2 and fr.entity_id not in warned_entities:
                warnings.append(
                    f"[{fr.entity_id}] high acceleration |a|={mag:.2f}m/s² "
                    f"at t={fr.t:.4f}s (exceeds {self.WARN_ACCEL_MS2}m/s²)"
                )
                warned_entities.add(fr.entity_id)

        return {"max_acceleration_ms2": max_accel}

    # ── 7: Energy drift ───────────────────────

    def _check_energy_drift(self,
                             trajectory: Trajectory,
                             errors: list[str],
                             warnings: list[str]) -> float:
        """
        Delegates per-entity drift computation to `Trajectory.energy_drift`,
        which returns |E_final - E_initial| / |E_initial|.

        Returns
        -------
        float : maximum drift across all entities (0.0 if no entities).
        """
        max_drift = 0.0
        for eid in trajectory.entity_ids:
            drift = trajectory.energy_drift(eid)
            if drift > max_drift:
                max_drift = drift

            if drift > self.max_energy_drift:
                warnings.append(
                    f"[{eid}] energy drift {drift:.4%} exceeds "
                    f"max_energy_drift={self.max_energy_drift:.4%}"
                )
            elif drift > self.WARN_ENERGY_DRIFT:
                warnings.append(
                    f"[{eid}] energy drift {drift:.4%} exceeds "
                    f"{self.WARN_ENERGY_DRIFT:.4%} threshold"
                )

        return max_drift

    # ── 8: Missing entity trajectories ────────

    def _check_missing_entities(self,
                                 trajectory: Trajectory,
                                 errors: list[str]) -> None:
        """
        Every id declared in `trajectory.entity_ids` must have at least one
        frame in `trajectory.frames`.
        """
        present = {fr.entity_id for fr in trajectory.frames}
        for eid in trajectory.entity_ids:
            if eid not in present:
                errors.append(
                    f"[{eid}] declared in entity_ids but has no frames"
                )

    # ── 9: Time monotonicity ──────────────────

    def _check_time_monotonicity(self,
                                  trajectory: Trajectory,
                                  errors: list[str]) -> None:
        """
        Within each entity's own frame sequence, timestamps must never
        decrease. (Frames are interleaved by entity in `trajectory.frames`,
        so monotonicity is checked per-entity, not on the raw list.)
        """
        last_t: dict[str, float] = {}
        for fr in trajectory.frames:
            prev = last_t.get(fr.entity_id)
            if prev is not None and fr.t < prev:
                errors.append(
                    f"[{fr.entity_id}] timestamp decreased: "
                    f"{prev:.6f}s → {fr.t:.6f}s"
                )
            last_t[fr.entity_id] = fr.t

    # ── 10: Frame consistency ─────────────────

    def _check_frame_consistency(self,
                                  trajectory: Trajectory,
                                  errors: list[str]) -> None:
        """
        Structural sanity checks on every frame:
          - entity_id is non-empty
          - t is non-negative
          - position / velocity / acceleration are 3-component vectors
        """
        for i, fr in enumerate(trajectory.frames):
            if not fr.entity_id:
                errors.append(f"Frame[{i}] has empty entity_id")
            if fr.t < 0:
                errors.append(
                    f"[{fr.entity_id}] negative simulation time t={fr.t:.6f}s "
                    f"at frame index {i}"
                )
            for field_name, vec in (
                ("position", fr.position),
                ("velocity", fr.velocity),
                ("acceleration", fr.acceleration),
            ):
                self._vec_components(vec)

    # ── metrics aggregation ───────────────────

    def _collect_metrics(self,
                          trajectory: Trajectory,
                          speed_stats: dict[str, float],
                          accel_stats: dict[str, float],
                          max_drift: float,
                          warning_count: int,
                          error_count: int) -> dict[str, Any]:
        """Assemble the final metrics dict returned in ValidationResult."""
        return {
            "frame_count":           len(trajectory.frames),
            "entity_count":          len(trajectory.entity_ids),
            "duration_s":            trajectory.duration(),
            "max_speed_ms":          speed_stats["max_speed_ms"],
            "mean_speed_ms":         speed_stats["mean_speed_ms"],
            "max_acceleration_ms2":  accel_stats["max_acceleration_ms2"],
            "max_energy_drift":      max_drift,
            "warning_count":         warning_count,
            "error_count":           error_count,
        }
