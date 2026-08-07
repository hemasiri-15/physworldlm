"""
models/repair.py
──────────────────
Bounded repair layer for the Validator → Repair stage.

Deliberately narrow: each RepairStrategy fixes exactly ONE Issue.code it
knows is safe to auto-fix, using a conservative, explainable substitution —
never a guess dressed up as a fix. Any error code without a registered
strategy is left for the caller to handle.

DESIGN NOTE — explicit list, not a registry:
  Considered a decorator-based RepairRegistry (register strategies via
  @RepairRegistry.register). Deliberately NOT adopted yet: with three
  strategies, an explicit list is equally extensible and more readable than
  a registry, and nothing in this codebase currently loads strategies
  dynamically or from third parties. A registry earns its complexity when
  that need actually exists. Revisit if/when plugin-style repair strategies
  become a real requirement.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from models.pipeline_interfaces import CompilerPass, PassResult, RepairStrategy
from models.validator import PhysicsValidator, ValidationResult, Severity
from world_spec import WorldSpec, Vec3, MATERIAL_DEFAULTS


_DEFAULT_MASS_BY_TYPE = {
    "vehicle":    1200.0,
    "projectile": 0.5,
    "agent":      70.0,
    "fluid":      1000.0,
    "structure":  500.0,
    "terrain":    0.0,
    "object":     1.0,
}


class DefaultMassRepair:
    handles_code = "PHYS_MASS_NONPOSITIVE"

    def repair(self, spec: WorldSpec, entity_id: str) -> bool:
        e = spec.get_entity(entity_id)
        if e is None:
            return False
        # NOTE: found via integration test — _DEFAULT_MASS_BY_TYPE["terrain"]
        # is 0.0, which is only valid for STATIC terrain. If a non-static
        # entity somehow has entity_type="terrain" (e.g. a misclassification
        # upstream), applying the type default would re-introduce the exact
        # error this strategy is meant to fix. Non-static entities always
        # get a strictly positive fallback regardless of type.
        default = _DEFAULT_MASS_BY_TYPE.get(e.entity_type, 1.0)
        if not e.is_static and default <= 0:
            default = _DEFAULT_MASS_BY_TYPE["object"]  # 1.0 kg fallback
        e.mass = default
        return True


class ZeroStaticVelocityRepair:
    handles_code = "SEM_STATIC_HAS_VELOCITY"

    def repair(self, spec: WorldSpec, entity_id: str) -> bool:
        e = spec.get_entity(entity_id)
        if e is None:
            return False
        e.state.velocity = Vec3(0.0, 0.0, 0.0)
        return True


class AssignMissingIdRepair:
    """Handles SCHEMA_ENTITY_NO_ID. entity_id passed in is the placeholder
    "(missing id)" from the validator's Issue — locates by identity (first
    entity with an empty id), not by id lookup.
    """

    handles_code = "SCHEMA_ENTITY_NO_ID"

    def repair(self, spec: WorldSpec, entity_id: str) -> bool:
        for e in spec.entities:
            if not e.id:
                e.id = f"e_repaired_{uuid.uuid4().hex[:8]}"
                return True
        return False


_DEFAULT_STRATEGIES: List[RepairStrategy] = [
    DefaultMassRepair(),
    ZeroStaticVelocityRepair(),
    AssignMissingIdRepair(),
]


# ─────────────────────────────────────────────
# RepairReport
# ─────────────────────────────────────────────

@dataclass
class RepairReport:
    """Structured outcome of a repair pass — feeds benchmark metrics
    directly (e.g. expected_metrics.json's repair-effectiveness numbers).
    """

    repaired: int = 0
    failed: int = 0
    skipped: int = 0  # errors with no registered strategy at all
    log: List[str] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "repaired": self.repaired,
            "failed": self.failed,
            "skipped": self.skipped,
            "log": self.log,
        }


class WorldSpecRepairer:
    """Applies registered RepairStrategy instances to a WorldSpec's errors,
    then re-validates to confirm the repair actually cleared the issue.
    """

    def __init__(self, strategies: List[RepairStrategy] | None = None) -> None:
        self._by_code: Dict[str, RepairStrategy] = {
            s.handles_code: s for s in (strategies or _DEFAULT_STRATEGIES)
        }

    def repair(self, spec: WorldSpec, result: ValidationResult) -> tuple[WorldSpec, RepairReport]:
        report = RepairReport()
        for issue in result.errors:
            strategy = self._by_code.get(issue.code)
            if strategy is None:
                report.skipped += 1
                report.log.append(f"NO STRATEGY  {issue.code}  [{issue.entity_id}]")
                continue
            fixed = strategy.repair(spec, issue.entity_id)
            if fixed:
                report.repaired += 1
                report.log.append(f"REPAIRED  {issue.code}  [{issue.entity_id}]")
            else:
                report.failed += 1
                report.log.append(f"FAILED  {issue.code}  [{issue.entity_id}]")
        return spec, report


@dataclass
class ValidationPassPayload:
    """PassResult.payload for ValidationCompilerPass — bundles the
    (post-repair) ValidationResult with the RepairReport describing what
    the repair step actually did, per the review's request.
    """

    validation: ValidationResult
    repair: RepairReport


class ValidationCompilerPass:
    """Wraps PhysicsValidator + WorldSpecRepairer as a single named
    CompilerPass: validate → repair-if-possible → re-validate.
    """

    name = "validation"

    def __init__(self) -> None:
        self.validator = PhysicsValidator()
        self.repairer = WorldSpecRepairer()

    def run(self, spec: WorldSpec) -> PassResult:
        result = self.validator.validate(spec)
        report = RepairReport()

        if result.errors:
            spec, report = self.repairer.repair(spec, result)
            result = self.validator.validate(spec)  # re-check after repair

        payload = ValidationPassPayload(validation=result, repair=report)
        notes = "; ".join(report.log) if report.log else "no errors — repair not invoked"
        return PassResult(ok=result.is_valid, payload=payload, notes=notes)
