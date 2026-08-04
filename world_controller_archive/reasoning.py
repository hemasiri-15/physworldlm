"""
reasoning.py
════════════════════════════════════════════════════════════════════════
The Reasoning Layer of PhysWorldLM's World Controller.

Nothing in this module ever mutates the live `WorldSpec` owned by the
controller. Every command is trial-run against a disposable deep copy,
validated, optionally self-repaired (by mutating the *command's own*
parameters, never someone else's data), and re-trialed — only once a
command is judged safe does `ReasoningPipeline.execute()` return an
approved `ValidationResult` for the (possibly repaired) command. The
`WorldController` is the only thing that ever applies a command to the
real `WorldSpec`; this module only ever advises it.

Pipeline
--------
    Command
        │
        ▼
    Semantic Validation ──┐
    Physics Validation    │
    Relationship Val.     ├──► issues
    Environment Val.      │
    Constraint Solving    │
    Collision Detection ──┘
        │
        ▼
    Automatic Repair (mutates the command's OWN parameters only)
        │
        ▼
    Approve / Reject  ──►  ValidationResult

Scoping rule (why an unrelated pre-existing problem never blocks you)
-----------------------------------------------------------------------
Validators are run twice: once against the WorldSpec as it stood BEFORE
the command, once against a scratch copy AFTER the command has been
dry-run-executed against it. Only issues that are *new* — i.e. did not
already exist before this command ran — are treated as blocking. A scene
that already contains, say, an entity with friction slightly above 1.0
from some earlier, already-approved edit will not suddenly start
rejecting an unrelated `MoveEntityCommand`; only issues this command
itself introduces are ever grounds for rejection or repair.

Repairability rule
-------------------
A `ValidationIssue` is only ever marked `repairable=True` when the
offending value is reachable by a dotted attribute path rooted at the
`Command` object itself (its own public fields, or, for structural
commands, the `Entity` object the command owns before it has been
inserted into the WorldSpec — e.g. `AddEntityCommand._entity`). Repair
strategies therefore only ever rewrite the command's own intent, never
another entity's pre-existing state; re-running the (possibly repaired)
command reproduces exactly what was validated.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

from world_spec import Entity, MATERIAL_DEFAULTS, Vec3, WorldSpec

from world_controller.commands import (
    Command,
    CommandContext,
    resolve_material,
)
from world_controller.enums import ChangeEventType
from world_controller.events import EventBus
from world_controller.exceptions import WorldControllerError
from world_controller.indexes import (
    BODY_MODE_TAG_PREFIX,
    EntityIndex,
    GROUP_TAG_PREFIX,
    PARENT_TAG_PREFIX,
)

logger = logging.getLogger("physworldlm.world_controller.reasoning")


# ════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════

class DiagnosticSeverity(Enum):
    """Severity of a single `ValidationIssue`.

    INFO / WARNING never block approval. ERROR / CRITICAL block approval
    unless a `RepairStrategy` successfully resolves them.
    """

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    @property
    def blocking(self) -> bool:
        return self in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)


@dataclass
class ValidationIssue:
    """A single structured finding produced by a validator, solver, or detector.

    Attributes:
        severity: How serious the issue is.
        code: A stable, machine-readable identifier (e.g. "NEGATIVE_MASS").
            Repair strategies dispatch on this.
        message: Human-readable explanation.
        source: Name of the validator/solver/detector that raised it.
        entity_ref: Entity id this issue concerns, if any.
        repairable: Whether a `RepairStrategy` claims it can fix this.
            Set by the pipeline after `_find_command_attribute_path`
            confirms the offending value is reachable from the command's
            own parameters — never set true by a validator directly.
        repair_hint: Free-form data a matching `RepairStrategy` needs to
            perform the fix. Validators populate at minimum a "field"
            key naming the logical attribute at fault (e.g. "mass").
            The pipeline augments this with "attribute_path" once
            repairability has been confirmed.
        timestamp: UTC time the issue was raised.
    """

    severity: DiagnosticSeverity
    code: str
    message: str
    source: str
    entity_ref: Optional[str] = None
    repairable: bool = False
    repair_hint: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def key(self) -> tuple[str, Optional[str]]:
        """Identity used for before/after (pre-existing vs. newly-introduced) diffing."""
        return (self.code, self.entity_ref)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.name,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "entity_ref": self.entity_ref,
            "repairable": self.repairable,
            "repair_hint": {k: v for k, v in self.repair_hint.items() if k != "attribute_path"},
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        ref = f" (entity={self.entity_ref})" if self.entity_ref else ""
        return f"[{self.severity.name}] {self.source} :: {self.code} :: {self.message}{ref}"


@dataclass
class RepairAction:
    """Record of a single automatic repair applied to a command's parameters."""

    strategy_name: str
    issue_code: str
    attribute_path: str
    old_value: Any
    new_value: Any
    entity_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "issue_code": self.issue_code,
            "attribute_path": self.attribute_path,
            "old_value": _safe(self.old_value),
            "new_value": _safe(self.new_value),
            "entity_ref": self.entity_ref,
        }

    def __str__(self) -> str:
        return (
            f"{self.strategy_name}: {self.attribute_path} "
            f"{self.old_value!r} → {self.new_value!r} (fixes {self.issue_code})"
        )


@dataclass
class ValidationResult:
    """Final verdict returned by `ReasoningPipeline.execute()`.

    Attributes:
        approved: Whether the (possibly repaired) command is safe to
            apply to the real WorldSpec.
        command: The command instance as it will be executed — identical
            to the one passed in unless repairs mutated its parameters.
        issues: Every issue observed on the final validation pass
            (both blocking and informational).
        repairs_applied: Ordered record of every automatic repair made
            across every retry attempt.
        attempts: How many validate → repair cycles were run.
    """

    approved: bool
    command: Command
    issues: list[ValidationIssue] = field(default_factory=list)
    repairs_applied: list[RepairAction] = field(default_factory=list)
    attempts: int = 1

    def blocking_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity.blocking]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is DiagnosticSeverity.WARNING]

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "command": self.command.__class__.__name__,
            "attempts": self.attempts,
            "issues": [i.to_dict() for i in self.issues],
            "repairs_applied": [r.to_dict() for r in self.repairs_applied],
        }

    def format(self) -> str:
        """Human-readable multi-line summary, suitable for logs or a CLI."""
        lines = [
            f"ValidationResult<{self.command.__class__.__name__}> "
            f"{'APPROVED' if self.approved else 'REJECTED'} "
            f"after {self.attempts} attempt(s)"
        ]
        if self.repairs_applied:
            lines.append("  repairs:")
            lines.extend(f"    - {r}" for r in self.repairs_applied)
        blocking = self.blocking_issues()
        if blocking:
            lines.append("  unresolved blocking issues:")
            lines.extend(f"    - {i}" for i in blocking)
        infos = [i for i in self.issues if not i.severity.blocking]
        if infos:
            lines.append("  notes:")
            lines.extend(f"    - {i}" for i in infos)
        return "\n".join(lines)


def _safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


# ════════════════════════════════════════════════════════════════════════
# Dotted attribute path helpers (repair machinery)
# ════════════════════════════════════════════════════════════════════════

def _path_get(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _path_set(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _find_command_attribute_path(
    command: Command, entity_id: Optional[str], field_name: str
) -> Optional[str]:
    """Locate a dotted attribute path, rooted at `command`, that holds `field_name`
    for `entity_id`, if the command owns that value directly.

    Returns `None` when the value is not reachable from the command's own
    parameters — in which case the corresponding issue must not be marked
    repairable (see module docstring: "Repairability rule").
    """
    # Field-edit style commands (SetMassCommand, SetFrictionCommand, ...)
    # store the pending value on `new_value` and record which field they
    # touch on the private `_field_label`.
    if (
        getattr(command, "entity_id", None) == entity_id
        and hasattr(command, "new_value")
        and getattr(command, "_field_label", None) == field_name
    ):
        return "new_value"

    # AssignMaterialCommand exposes `material_name` for Entity.material.
    if (
        field_name == "material"
        and getattr(command, "entity_id", None) == entity_id
        and hasattr(command, "material_name")
    ):
        return "material_name"

    # AddEntityCommand owns a not-yet-inserted Entity on `_entity`.
    owned_entity: Optional[Entity] = getattr(command, "_entity", None)
    if owned_entity is not None and owned_entity.id == entity_id:
        return f"_entity.{field_name}"

    # DuplicateEntityCommand owns its freshly-cloned Entity on `_created_entity`.
    created_entity: Optional[Entity] = getattr(command, "_created_entity", None)
    if created_entity is not None and created_entity.id == entity_id:
        return f"_created_entity.{field_name}"

    return None


def _owned_entity(command: Command, entity_id: Optional[str]) -> Optional[Entity]:
    """Return the `Entity` object owned outright by `command` for `entity_id`,
    if any — the structural counterpart of `_find_command_attribute_path`,
    used by repair strategies that need to mutate a list field (tags,
    constraints) rather than a single scalar.
    """
    for attr in ("_entity", "_created_entity"):
        candidate: Optional[Entity] = getattr(command, attr, None)
        if candidate is not None and candidate.id == entity_id:
            return candidate
    return None


# ════════════════════════════════════════════════════════════════════════
# Shared domain constants
# ════════════════════════════════════════════════════════════════════════

KNOWN_ENTITY_TYPES = frozenset(
    {"vehicle", "projectile", "fluid", "agent", "structure", "terrain"}
)
RESERVED_TAG_PREFIXES = (GROUP_TAG_PREFIX, PARENT_TAG_PREFIX, BODY_MODE_TAG_PREFIX)
KNOWN_BODY_MODES = frozenset({"dynamic", "static", "kinematic", "sensor"})
KNOWN_INTEGRATORS = frozenset({"rk4", "euler", "verlet"})
KNOWN_TERRAIN_TYPES = frozenset({"flat", "hilly", "urban", "water", "mixed"})
KNOWN_WEATHER = frozenset({"clear", "rain", "snow", "fog", "wind"})
KNOWN_TIME_OF_DAY = frozenset({"day", "night", "dawn", "dusk"})
ALLOWED_METADATA_KEYS = frozenset(
    {"author", "created_at", "updated_at", "description", "version", "source_prompt", "tags"}
)
MAX_PLAUSIBLE_SPEED_MS = 3_000.0          # ~ Mach 9, generous upper bound
MAX_PLAUSIBLE_ACCEL_MS2 = 50_000.0        # generous upper bound for game/sim content
MAX_PLAUSIBLE_ANGULAR_VEL = 1_000.0       # rad/s
MAX_PLAUSIBLE_GRAVITY_MS2 = 200.0


def _is_finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _vec3_finite(v: Vec3) -> bool:
    return _is_finite(v.x) and _is_finite(v.y) and _is_finite(v.z)


def _body_mode_of(entity: Entity) -> str:
    """Resolve an entity's body mode from its `__body_mode__:` tag, falling
    back to the `is_static` flag when no explicit tag is present.
    """
    for tag in entity.tags:
        if tag.startswith(BODY_MODE_TAG_PREFIX):
            return tag[len(BODY_MODE_TAG_PREFIX):]
    return "static" if entity.is_static else "dynamic"


def _entity_index(world: WorldSpec) -> EntityIndex:
    idx = EntityIndex()
    idx.rebuild(world)
    return idx


# ════════════════════════════════════════════════════════════════════════
# Validators
# ════════════════════════════════════════════════════════════════════════

class Validator(ABC):
    """Base class for all reasoning-layer validators.

    A validator inspects a *single* WorldSpec snapshot (never a diff) and
    reports every issue it finds in that snapshot. Delta filtering (only
    keeping issues that are new relative to the pre-command world) is the
    `ReasoningPipeline`'s responsibility, not the validator's — this keeps
    each validator simple, stateless, and independently testable.
    """

    name: str = "Validator"

    @abstractmethod
    def validate(self, world: WorldSpec, index: EntityIndex) -> list[ValidationIssue]:
        """Return every issue observed in `world`. Must not mutate `world`."""
        raise NotImplementedError


class SemanticValidator(Validator):
    """Validates identity, taxonomy, and ontology-level correctness:
    unknown entity types, unknown materials, duplicate ids, malformed or
    reserved tags, unknown group references, unknown parent references,
    unknown metadata keys, and invalid ontology references in general.
    """

    name = "SemanticValidator"

    def validate(self, world: WorldSpec, index: EntityIndex) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        seen_ids: set[str] = set()
        known_ids = {e.id for e in world.entities}
        known_groups = set(index.group_names())

        for entity in world.entities:
            if entity.id in seen_ids:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.CRITICAL,
                        code="DUPLICATE_ID",
                        message=f"Entity id {entity.id!r} is used by more than one entity.",
                        source=self.name,
                        entity_ref=entity.id,
                    )
                )
            seen_ids.add(entity.id)

            if entity.entity_type not in KNOWN_ENTITY_TYPES:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.ERROR,
                        code="UNKNOWN_ENTITY_TYPE",
                        message=(
                            f"Entity {entity.id!r} has unknown entity_type "
                            f"{entity.entity_type!r}; expected one of "
                            f"{sorted(KNOWN_ENTITY_TYPES)}."
                        ),
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": "entity_type"},
                    )
                )

            if entity.material not in MATERIAL_DEFAULTS:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.ERROR,
                        code="UNKNOWN_MATERIAL",
                        message=(
                            f"Entity {entity.id!r} references unknown material "
                            f"{entity.material!r}."
                        ),
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": "material"},
                    )
                )

            for tag in entity.tags:
                if tag.startswith(GROUP_TAG_PREFIX):
                    group_name = tag[len(GROUP_TAG_PREFIX):]
                    if not group_name:
                        issues.append(
                            ValidationIssue(
                                severity=DiagnosticSeverity.ERROR,
                                code="RESERVED_TAG_MALFORMED",
                                message=f"Entity {entity.id!r} has an empty group tag.",
                                source=self.name,
                                entity_ref=entity.id,
                                repair_hint={"field": "tags"},
                            )
                        )
                elif tag.startswith(PARENT_TAG_PREFIX):
                    parent_id = tag[len(PARENT_TAG_PREFIX):]
                    if not parent_id or parent_id not in known_ids:
                        issues.append(
                            ValidationIssue(
                                severity=DiagnosticSeverity.ERROR,
                                code="UNKNOWN_PARENT",
                                message=(
                                    f"Entity {entity.id!r} references parent "
                                    f"{parent_id!r} which does not exist."
                                ),
                                source=self.name,
                                entity_ref=entity.id,
                                repair_hint={"field": "tags"},
                            )
                        )
                elif tag.startswith(BODY_MODE_TAG_PREFIX):
                    mode = tag[len(BODY_MODE_TAG_PREFIX):]
                    if mode not in KNOWN_BODY_MODES:
                        issues.append(
                            ValidationIssue(
                                severity=DiagnosticSeverity.ERROR,
                                code="UNKNOWN_BODY_MODE",
                                message=(
                                    f"Entity {entity.id!r} declares unknown body mode "
                                    f"{mode!r}; expected one of {sorted(KNOWN_BODY_MODES)}."
                                ),
                                source=self.name,
                                entity_ref=entity.id,
                                repair_hint={"field": "tags"},
                            )
                        )

            for constraint_ref in entity.constraints:
                if constraint_ref.startswith("group:"):
                    group_name = constraint_ref[len("group:"):]
                    if group_name not in known_groups:
                        issues.append(
                            ValidationIssue(
                                severity=DiagnosticSeverity.ERROR,
                                code="UNKNOWN_GROUP",
                                message=(
                                    f"Entity {entity.id!r} references unknown group "
                                    f"{group_name!r}."
                                ),
                                source=self.name,
                                entity_ref=entity.id,
                                repair_hint={"field": "constraints"},
                            )
                        )
                elif constraint_ref not in known_ids:
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.ERROR,
                            code="INVALID_ONTOLOGY_REFERENCE",
                            message=(
                                f"Entity {entity.id!r} has a constraint referencing "
                                f"unknown entity {constraint_ref!r}."
                            ),
                            source=self.name,
                            entity_ref=entity.id,
                            repair_hint={"field": "constraints"},
                        )
                    )

        for key in world.metadata:
            if key not in ALLOWED_METADATA_KEYS:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.WARNING,
                        code="UNKNOWN_METADATA_KEY",
                        message=f"WorldSpec.metadata has unrecognized key {key!r}.",
                        source=self.name,
                        entity_ref=None,
                        repair_hint={"field": "metadata"},
                    )
                )

        return issues


class PhysicsValidator(Validator):
    """Validates per-entity physical plausibility: mass, density, bounding
    box, restitution, friction, kinematic quantities, and rules specific
    to each body mode (dynamic / static / kinematic / sensor), plus the
    scene-wide simulation timestep, integrator, and gravity.
    """

    name = "PhysicsValidator"

    def validate(self, world: WorldSpec, index: EntityIndex) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for entity in world.entities:
            mode = _body_mode_of(entity)
            issues.extend(self._validate_mass_and_density(entity, mode))
            issues.extend(self._validate_bounding_box(entity))
            issues.extend(self._validate_material_coeffs(entity))
            issues.extend(self._validate_kinematics(entity))
            issues.extend(self._validate_body_mode_rules(entity, mode))

        issues.extend(self._validate_sim_graph(world))
        issues.extend(self._validate_gravity(world))
        return issues

    def _validate_mass_and_density(self, entity: Entity, mode: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if entity.mass < 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="NEGATIVE_MASS",
                    message=f"Entity {entity.id!r} has negative mass {entity.mass}.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "mass"},
                )
            )
        elif entity.mass == 0 and mode == "dynamic":
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="ZERO_MASS",
                    message=f"Dynamic entity {entity.id!r} has zero mass.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "mass"},
                )
            )

        volume = entity.bounding_box.volume()
        if volume > 0 and entity.mass > 0:
            implied_density = entity.mass / volume
            if implied_density > 0 and (implied_density < 0.01 or implied_density > 25_000.0):
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.WARNING,
                        code="IMPLAUSIBLE_DENSITY",
                        message=(
                            f"Entity {entity.id!r} implies a density of "
                            f"{implied_density:.2f} kg/m^3 from its mass and bounding box, "
                            f"which is outside plausible material ranges."
                        ),
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": "mass"},
                    )
                )
        return issues

    def _validate_bounding_box(self, entity: Entity) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        bb = entity.bounding_box
        for axis_name, size in (("width", bb.width), ("height", bb.height), ("depth", bb.depth)):
            if not _is_finite(size) or size <= 0:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.ERROR,
                        code="INVALID_BOUNDING_BOX",
                        message=(
                            f"Entity {entity.id!r} has non-positive/non-finite "
                            f"bounding box {axis_name}={size}."
                        ),
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": "bounding_box"},
                    )
                )
        return issues

    def _validate_material_coeffs(self, entity: Entity) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not _is_finite(entity.restitution) or entity.restitution < 0 or entity.restitution > 1:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="INVALID_RESTITUTION",
                    message=(
                        f"Entity {entity.id!r} has restitution {entity.restitution} "
                        f"outside the valid range [0, 1]."
                    ),
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "restitution"},
                )
            )
        if not _is_finite(entity.friction) or entity.friction < 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="NEGATIVE_FRICTION",
                    message=f"Entity {entity.id!r} has negative friction {entity.friction}.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "friction"},
                )
            )
        return issues

    def _validate_kinematics(self, entity: Entity) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        state = entity.state

        for label, vec, cap in (
            ("velocity", state.velocity, MAX_PLAUSIBLE_SPEED_MS),
            ("acceleration", state.acceleration, MAX_PLAUSIBLE_ACCEL_MS2),
            ("angular_vel", state.angular_vel, MAX_PLAUSIBLE_ANGULAR_VEL),
        ):
            if not _vec3_finite(vec):
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.CRITICAL,
                        code=f"NON_FINITE_{label.upper()}",
                        message=f"Entity {entity.id!r} has a non-finite {label} vector.",
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": label},
                    )
                )
            elif vec.magnitude() > cap:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.WARNING,
                        code=f"EXCESSIVE_{label.upper()}",
                        message=(
                            f"Entity {entity.id!r} has {label} magnitude "
                            f"{vec.magnitude():.2f} exceeding plausible cap {cap}."
                        ),
                        source=self.name,
                        entity_ref=entity.id,
                        repair_hint={"field": label},
                    )
                )

        if not _vec3_finite(state.position):
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="NON_FINITE_POSITION",
                    message=f"Entity {entity.id!r} has a non-finite position vector.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "position"},
                )
            )
        return issues

    def _validate_body_mode_rules(self, entity: Entity, mode: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        state = entity.state
        moving = state.velocity.magnitude() > 1e-9 or state.angular_vel.magnitude() > 1e-9

        if mode == "static" and moving:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="STATIC_BODY_HAS_VELOCITY",
                    message=f"Static entity {entity.id!r} has nonzero velocity/angular velocity.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "velocity"},
                )
            )
        if mode == "dynamic" and entity.mass <= 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="DYNAMIC_BODY_INVALID_MASS",
                    message=f"Dynamic entity {entity.id!r} must have positive mass.",
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "mass"},
                )
            )
        if mode == "kinematic" and entity.mass not in (0.0,) and entity.forces:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.WARNING,
                    code="KINEMATIC_BODY_HAS_FORCES",
                    message=(
                        f"Kinematic entity {entity.id!r} has applied forces, which will "
                        f"be ignored — kinematic bodies are driven by velocity, not force."
                    ),
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "forces"},
                )
            )
        if mode == "sensor" and entity.restitution != 0.0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.INFO,
                    code="SENSOR_BODY_HAS_RESTITUTION",
                    message=(
                        f"Sensor entity {entity.id!r} has nonzero restitution, which has "
                        f"no effect — sensors do not generate collision response."
                    ),
                    source=self.name,
                    entity_ref=entity.id,
                    repair_hint={"field": "restitution"},
                )
            )
        return issues

    def _validate_sim_graph(self, world: WorldSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        sg = world.simulation_graph
        if not _is_finite(sg.dt) or sg.dt <= 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="INVALID_TIMESTEP",
                    message=f"Simulation timestep dt={sg.dt} must be a positive finite number.",
                    source=self.name,
                    repair_hint={"field": "dt"},
                )
            )
        if not _is_finite(sg.duration) or sg.duration <= 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="INVALID_DURATION",
                    message=f"Simulation duration={sg.duration} must be a positive finite number.",
                    source=self.name,
                    repair_hint={"field": "duration"},
                )
            )
        if sg.integrator not in KNOWN_INTEGRATORS:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="UNKNOWN_INTEGRATOR",
                    message=(
                        f"Unknown integrator {sg.integrator!r}; expected one of "
                        f"{sorted(KNOWN_INTEGRATORS)}."
                    ),
                    source=self.name,
                    repair_hint={"field": "integrator"},
                )
            )
        return issues

    def _validate_gravity(self, world: WorldSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        g = world.environment.gravity
        if not _vec3_finite(g):
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="NON_FINITE_GRAVITY",
                    message="Environment gravity vector is non-finite.",
                    source=self.name,
                    repair_hint={"field": "gravity"},
                )
            )
        elif g.magnitude() > MAX_PLAUSIBLE_GRAVITY_MS2:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.WARNING,
                    code="EXCESSIVE_GRAVITY",
                    message=f"Gravity magnitude {g.magnitude():.2f} m/s^2 is implausibly large.",
                    source=self.name,
                    repair_hint={"field": "gravity"},
                )
            )
        return issues


class RelationshipValidator(Validator):
    """Validates the entity relationship graph: broken parent references,
    broken/dangling constraints, duplicate relationships, parent-chain
    cycles, and invalid interaction endpoints.
    """

    name = "RelationshipValidator"

    def validate(self, world: WorldSpec, index: EntityIndex) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        known_ids = {e.id for e in world.entities}

        issues.extend(self._validate_parents(world, known_ids))
        issues.extend(self._validate_constraints(world, known_ids))
        issues.extend(self._validate_interactions(world, known_ids))
        issues.extend(self._validate_parent_cycles(world))
        return issues

    def _validate_parents(self, world: WorldSpec, known_ids: set[str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for entity in world.entities:
            for tag in entity.tags:
                if tag.startswith(PARENT_TAG_PREFIX):
                    parent_id = tag[len(PARENT_TAG_PREFIX):]
                    if parent_id not in known_ids:
                        issues.append(
                            ValidationIssue(
                                severity=DiagnosticSeverity.ERROR,
                                code="BROKEN_PARENT_REFERENCE",
                                message=(
                                    f"Entity {entity.id!r} has a parent reference to "
                                    f"nonexistent entity {parent_id!r}."
                                ),
                                source=self.name,
                                entity_ref=entity.id,
                                repair_hint={"field": "tags"},
                            )
                        )
        return issues

    def _validate_constraints(self, world: WorldSpec, known_ids: set[str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for entity in world.entities:
            seen: set[str] = set()
            for ref in entity.constraints:
                if ref in seen:
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.WARNING,
                            code="DUPLICATE_RELATIONSHIP",
                            message=(
                                f"Entity {entity.id!r} lists constraint {ref!r} more than once."
                            ),
                            source=self.name,
                            entity_ref=entity.id,
                            repair_hint={"field": "constraints"},
                        )
                    )
                seen.add(ref)
                if not ref.startswith("group:") and ref not in known_ids and ref != entity.id:
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.ERROR,
                            code="BROKEN_CONSTRAINT",
                            message=(
                                f"Entity {entity.id!r} has a constraint referencing "
                                f"nonexistent entity {ref!r}."
                            ),
                            source=self.name,
                            entity_ref=entity.id,
                            repair_hint={"field": "constraints"},
                        )
                    )
                if ref == entity.id:
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.ERROR,
                            code="SELF_REFERENTIAL_CONSTRAINT",
                            message=f"Entity {entity.id!r} has a constraint referencing itself.",
                            source=self.name,
                            entity_ref=entity.id,
                            repair_hint={"field": "constraints"},
                        )
                    )
        return issues

    def _validate_interactions(self, world: WorldSpec, known_ids: set[str]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for interaction in world.interactions:
            for endpoint_name, endpoint_id in (
                ("entity_a", interaction.entity_a),
                ("entity_b", interaction.entity_b),
            ):
                if endpoint_id != "environment" and endpoint_id not in known_ids:
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.ERROR,
                            code="INVALID_INTERACTION_ENDPOINT",
                            message=(
                                f"Interaction of type {interaction.type!r} has "
                                f"{endpoint_name}={endpoint_id!r}, which does not "
                                f"reference a known entity or 'environment'."
                            ),
                            source=self.name,
                            entity_ref=None,
                            repair_hint={"field": "interactions"},
                        )
                    )
        return issues

    def _validate_parent_cycles(self, world: WorldSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        parent_of: dict[str, str] = {}
        for entity in world.entities:
            for tag in entity.tags:
                if tag.startswith(PARENT_TAG_PREFIX):
                    parent_of[entity.id] = tag[len(PARENT_TAG_PREFIX):]

        visited_global: set[str] = set()
        for start in parent_of:
            if start in visited_global:
                continue
            path: list[str] = []
            current = start
            local_seen: set[str] = set()
            while current in parent_of:
                if current in local_seen:
                    cycle_start_idx = path.index(current) if current in path else 0
                    cycle_members = path[cycle_start_idx:] + [current]
                    issues.append(
                        ValidationIssue(
                            severity=DiagnosticSeverity.CRITICAL,
                            code="PARENT_CYCLE",
                            message=(
                                f"Parent chain forms a cycle: "
                                f"{' -> '.join(cycle_members)}."
                            ),
                            source=self.name,
                            entity_ref=start,
                            repair_hint={"field": "tags"},
                        )
                    )
                    break
                local_seen.add(current)
                path.append(current)
                current = parent_of[current]
            visited_global |= local_seen

        return issues


class EnvironmentValidator(Validator):
    """Validates scene-wide environmental parameters: gravity, pressure,
    temperature, wind, terrain, weather, global friction, and time of day.
    """

    name = "EnvironmentValidator"

    def validate(self, world: WorldSpec, index: EntityIndex) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        env = world.environment

        if not _vec3_finite(env.gravity):
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="INVALID_ENVIRONMENT_GRAVITY",
                    message="Environment gravity vector is non-finite.",
                    source=self.name,
                    repair_hint={"field": "gravity"},
                )
            )

        if not _is_finite(env.pressure_Pa) or env.pressure_Pa <= 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="INVALID_PRESSURE",
                    message=f"Environment pressure {env.pressure_Pa} Pa must be positive.",
                    source=self.name,
                    repair_hint={"field": "pressure_Pa"},
                )
            )

        if not _is_finite(env.temperature_K) or env.temperature_K <= 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="INVALID_TEMPERATURE",
                    message=(
                        f"Environment temperature {env.temperature_K} K must be a positive "
                        f"finite number (temperatures below absolute zero are impossible)."
                    ),
                    source=self.name,
                    repair_hint={"field": "temperature_K"},
                )
            )

        if not _is_finite(env.wind.speed) or env.wind.speed < 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="INVALID_WIND_SPEED",
                    message=f"Wind speed {env.wind.speed} must be non-negative.",
                    source=self.name,
                    repair_hint={"field": "wind"},
                )
            )

        if env.terrain_type not in KNOWN_TERRAIN_TYPES:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="UNKNOWN_TERRAIN_TYPE",
                    message=(
                        f"Unknown terrain_type {env.terrain_type!r}; expected one of "
                        f"{sorted(KNOWN_TERRAIN_TYPES)}."
                    ),
                    source=self.name,
                    repair_hint={"field": "terrain_type"},
                )
            )

        if env.weather not in KNOWN_WEATHER:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="UNKNOWN_WEATHER",
                    message=(
                        f"Unknown weather {env.weather!r}; expected one of "
                        f"{sorted(KNOWN_WEATHER)}."
                    ),
                    source=self.name,
                    repair_hint={"field": "weather"},
                )
            )

        if env.time_of_day not in KNOWN_TIME_OF_DAY:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.WARNING,
                    code="UNKNOWN_TIME_OF_DAY",
                    message=(
                        f"Unknown time_of_day {env.time_of_day!r}; expected one of "
                        f"{sorted(KNOWN_TIME_OF_DAY)}."
                    ),
                    source=self.name,
                    repair_hint={"field": "time_of_day"},
                )
            )

        if not _is_finite(env.friction_global) or env.friction_global < 0:
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.ERROR,
                    code="INVALID_GLOBAL_FRICTION",
                    message=f"Global friction {env.friction_global} must be non-negative.",
                    source=self.name,
                    repair_hint={"field": "friction_global"},
                )
            )

        return issues


ALL_VALIDATORS: tuple[type[Validator], ...] = (
    SemanticValidator,
    PhysicsValidator,
    RelationshipValidator,
    EnvironmentValidator,
)


# ════════════════════════════════════════════════════════════════════════
# ConstraintSolver
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DependencyGraph:
    """Directed graph of "X depends on Y" edges derived from parent tags,
    group membership, and declared constraints. Used by `ConstraintSolver`
    to detect cycles and to scope incremental solving to only the entities
    reachable from whatever the current command touched.
    """

    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, set()).add(dst)

    def neighbors(self, node: str) -> set[str]:
        return self.edges.get(node, set())

    def reachable_from(self, roots: set[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.neighbors(node) - seen)
        return seen

    def find_cycle(self) -> Optional[list[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in self.edges}
        stack_path: list[str] = []

        def visit(node: str) -> Optional[list[str]]:
            color[node] = GRAY
            stack_path.append(node)
            for nxt in self.neighbors(node):
                if color.get(nxt, WHITE) == WHITE:
                    result = visit(nxt)
                    if result is not None:
                        return result
                elif color.get(nxt) == GRAY:
                    idx = stack_path.index(nxt)
                    return stack_path[idx:] + [nxt]
            stack_path.pop()
            color[node] = BLACK
            return None

        for node in list(color):
            if color[node] == WHITE:
                cycle = visit(node)
                if cycle is not None:
                    return cycle
        return None


class ConstraintSolver:
    """Reasons over the constraint system as a whole: parent constraints,
    group constraints, ontology constraints, physics constraints, and
    interaction constraints. Builds a `DependencyGraph` and performs
    incremental solving scoped to the entities the current command
    actually touched, rather than re-deriving the entire scene graph on
    every command.
    """

    name = "ConstraintSolver"

    def build_graph(self, world: WorldSpec) -> DependencyGraph:
        graph = DependencyGraph()
        known_ids = {e.id for e in world.entities}
        for entity in world.entities:
            graph.edges.setdefault(entity.id, set())
            for tag in entity.tags:
                if tag.startswith(PARENT_TAG_PREFIX):
                    parent_id = tag[len(PARENT_TAG_PREFIX):]
                    if parent_id in known_ids:
                        graph.add_edge(entity.id, parent_id)
            for ref in entity.constraints:
                if not ref.startswith("group:") and ref in known_ids:
                    graph.add_edge(entity.id, ref)
        for interaction in world.interactions:
            if interaction.entity_a in known_ids and interaction.entity_b in known_ids:
                graph.add_edge(interaction.entity_a, interaction.entity_b)
        return graph

    def solve(
        self, world: WorldSpec, touched_entity_ids: set[str]
    ) -> list[ValidationIssue]:
        """Incrementally solve the constraint system, restricting expensive
        graph-wide checks (e.g. cycle detection) to the subgraph reachable
        from the entities the current command touched, plus a scene-wide
        cheap cycle scan since cycles can, in principle, be introduced
        without touching every member of the cycle.
        """
        issues: list[ValidationIssue] = []
        graph = self.build_graph(world)

        cycle = graph.find_cycle()
        if cycle is not None and (set(cycle) & touched_entity_ids):
            issues.append(
                ValidationIssue(
                    severity=DiagnosticSeverity.CRITICAL,
                    code="CONSTRAINT_CYCLE",
                    message=f"Constraint graph contains a cycle: {' -> '.join(cycle)}.",
                    source=self.name,
                    entity_ref=cycle[0] if cycle else None,
                    repair_hint={"field": "constraints"},
                )
            )

        scoped = graph.reachable_from(touched_entity_ids)
        known_ids = {e.id for e in world.entities}
        for node in scoped:
            if node not in known_ids and node in touched_entity_ids:
                issues.append(
                    ValidationIssue(
                        severity=DiagnosticSeverity.ERROR,
                        code="DANGLING_DEPENDENCY",
                        message=f"Entity {node!r} depends on entities outside the known set.",
                        source=self.name,
                        entity_ref=node,
                        repair_hint={"field": "constraints"},
                    )
                )

        return issues


# ════════════════════════════════════════════════════════════════════════
# CollisionDetector
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CollisionReport:
    """Result of an AABB overlap between two entities."""

    entity_a: str
    entity_b: str
    overlap: Vec3  # positive overlap extent on each axis

    def to_dict(self) -> dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "overlap": self.overlap.to_dict(),
        }


class CollisionDetector:
    """Pure-Python broad-phase AABB collision detection over a WorldSpec
    snapshot. No external physics libraries are used, per design mandate.
    """

    name = "CollisionDetector"

    @staticmethod
    def _aabb(entity: Entity) -> tuple[Vec3, Vec3]:
        pos = entity.state.position
        bb = entity.bounding_box
        half = Vec3(bb.width / 2.0, bb.height / 2.0, bb.depth / 2.0)
        lo = Vec3(pos.x - half.x, pos.y - half.y, pos.z - half.z)
        hi = Vec3(pos.x + half.x, pos.y + half.y, pos.z + half.z)
        return lo, hi

    @staticmethod
    def _overlap(lo_a: Vec3, hi_a: Vec3, lo_b: Vec3, hi_b: Vec3) -> Optional[Vec3]:
        ox = min(hi_a.x, hi_b.x) - max(lo_a.x, lo_b.x)
        oy = min(hi_a.y, hi_b.y) - max(lo_a.y, lo_b.y)
        oz = min(hi_a.z, hi_b.z) - max(lo_a.z, lo_b.z)
        if ox > 0 and oy > 0 and oz > 0:
            return Vec3(ox, oy, oz)
        return None

    def broad_phase_pairs(self, entities: list[Entity]) -> list[tuple[Entity, Entity]]:
        """Sweep-and-prune along the X axis: sort by AABB min.x, then only
        test pairs whose X extents actually overlap before doing the full
        3-axis test. O(n log n) sort + roughly O(n) active-list sweep for
        scenes without dense X-axis clustering, versus naive O(n^2).
        """
        boxes = [(e, *self._aabb(e)) for e in entities]
        boxes.sort(key=lambda t: t[1].x)

        candidates: list[tuple[Entity, Entity]] = []
        active: list[tuple[Entity, Vec3, Vec3]] = []
        for entity, lo, hi in boxes:
            active = [(e, l, h) for (e, l, h) in active if h.x >= lo.x]
            for other, other_lo, other_hi in active:
                candidates.append((other, entity))
            active.append((entity, lo, hi))
        return candidates

    def scan(self, world: WorldSpec, entity_ids: Optional[set[str]] = None) -> list[CollisionReport]:
        """Scene-wide (or, if `entity_ids` is given, scoped) collision scan.
        Static-vs-static overlaps are ignored (immovable geometry is
        assumed to have been placed intentionally by scene authors).
        """
        entities = world.entities if entity_ids is None else [
            e for e in world.entities if e.id in entity_ids
        ]
        all_entities = world.entities
        reports: list[CollisionReport] = []

        pairs = self.broad_phase_pairs(all_entities) if entity_ids is None else [
            (a, b)
            for a, b in self.broad_phase_pairs(all_entities)
            if a.id in {e.id for e in entities} or b.id in {e.id for e in entities}
        ]

        for a, b in pairs:
            if a.is_static and b.is_static:
                continue
            lo_a, hi_a = self._aabb(a)
            lo_b, hi_b = self._aabb(b)
            overlap = self._overlap(lo_a, hi_a, lo_b, hi_b)
            if overlap is not None:
                reports.append(CollisionReport(entity_a=a.id, entity_b=b.id, overlap=overlap))
        return reports

    def nearest_free_position(
        self,
        world: WorldSpec,
        entity: Entity,
        max_radius: float = 50.0,
        step: float = 0.5,
    ) -> Optional[Vec3]:
        """Search outward from `entity`'s current position along an
        expanding ring of axis-aligned offsets for the nearest position at
        which `entity`'s AABB overlaps no other entity's AABB. Returns
        `None` if no free position is found within `max_radius`.
        """
        others = [e for e in world.entities if e.id != entity.id]
        origin = entity.state.position
        directions = (
            Vec3(1, 0, 0), Vec3(-1, 0, 0),
            Vec3(0, 1, 0), Vec3(0, -1, 0),
            Vec3(0, 0, 1), Vec3(0, 0, -1),
        )

        def is_free(candidate: Vec3) -> bool:
            probe = copy.deepcopy(entity)
            probe.state.position = candidate
            lo_p, hi_p = self._aabb(probe)
            for other in others:
                lo_o, hi_o = self._aabb(other)
                if self._overlap(lo_p, hi_p, lo_o, hi_o) is not None:
                    return False
            return True

        if is_free(origin):
            return Vec3(origin.x, origin.y, origin.z)

        radius = step
        while radius <= max_radius:
            for d in directions:
                candidate = Vec3(
                    origin.x + d.x * radius,
                    origin.y + d.y * radius,
                    origin.z + d.z * radius,
                )
                if is_free(candidate):
                    return candidate
            radius += step
        return None


# ════════════════════════════════════════════════════════════════════════
# RepairEngine
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class RepairStrategy(Protocol):
    """A strategy that can repair one category of `ValidationIssue` by
    mutating the offending `Command`'s own parameters — never anything
    already stored in the live `WorldSpec`.
    """

    name: str

    def supports(self, issue: ValidationIssue) -> bool:
        """Whether this strategy knows how to repair `issue`."""
        ...

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        """Apply the repair in place on `command`. Returns the `RepairAction`
        performed, or `None` if the value was not reachable on `command`
        (in which case the caller must leave `issue` unresolved).
        """
        ...


class _ScalarClampStrategy:
    """Generic strategy for issues whose fix is clamping a single scalar
    field owned by the command into a valid range.
    """

    def __init__(self, name: str, codes: frozenset[str], clamp) -> None:
        self.name = name
        self._codes = codes
        self._clamp = clamp

    def supports(self, issue: ValidationIssue) -> bool:
        return issue.code in self._codes

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        field_name = issue.repair_hint.get("field")
        if field_name is None:
            return None
        path = _find_command_attribute_path(command, issue.entity_ref, field_name)
        if path is None:
            return None
        try:
            old_value = _path_get(command, path)
        except AttributeError:
            return None
        new_value = self._clamp(old_value)
        if new_value == old_value:
            return None
        _path_set(command, path, new_value)
        return RepairAction(
            strategy_name=self.name,
            issue_code=issue.code,
            attribute_path=path,
            old_value=old_value,
            new_value=new_value,
            entity_ref=issue.entity_ref,
        )


class ClampNegativeMassStrategy(_ScalarClampStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="ClampNegativeMassStrategy",
            codes=frozenset({"NEGATIVE_MASS", "ZERO_MASS", "DYNAMIC_BODY_INVALID_MASS"}),
            clamp=lambda v: max(v, 0.001),
        )


class ClampNegativeFrictionStrategy(_ScalarClampStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="ClampNegativeFrictionStrategy",
            codes=frozenset({"NEGATIVE_FRICTION", "INVALID_GLOBAL_FRICTION"}),
            clamp=lambda v: max(v, 0.0),
        )


class ClampRestitutionStrategy(_ScalarClampStrategy):
    def __init__(self) -> None:
        super().__init__(
            name="ClampRestitutionStrategy",
            codes=frozenset({"INVALID_RESTITUTION"}),
            clamp=lambda v: min(max(v, 0.0), 1.0),
        )


class UnknownMaterialStrategy:
    """Rewrites an unknown `material` reference to the safe fallback
    'generic' material, which is guaranteed to exist in `MATERIAL_DEFAULTS`.
    """

    name = "UnknownMaterialStrategy"

    def supports(self, issue: ValidationIssue) -> bool:
        return issue.code == "UNKNOWN_MATERIAL"

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        path = _find_command_attribute_path(command, issue.entity_ref, "material")
        if path is None:
            return None
        try:
            old_value = _path_get(command, path)
        except AttributeError:
            return None
        if old_value == "generic":
            return None
        _path_set(command, path, "generic")
        return RepairAction(
            strategy_name=self.name,
            issue_code=issue.code,
            attribute_path=path,
            old_value=old_value,
            new_value="generic",
            entity_ref=issue.entity_ref,
        )


class BrokenParentStrategy:
    """Detaches an entity from a broken/unknown/cyclic parent reference by
    removing the offending `__parent__:` tag from the command-owned entity.
    """

    name = "BrokenParentStrategy"
    _CODES = frozenset({"BROKEN_PARENT_REFERENCE", "UNKNOWN_PARENT", "PARENT_CYCLE"})

    def supports(self, issue: ValidationIssue) -> bool:
        return issue.code in self._CODES

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        entity = _owned_entity(command, issue.entity_ref)
        if entity is None:
            return None
        old_tags = list(entity.tags)
        new_tags = [t for t in entity.tags if not t.startswith(PARENT_TAG_PREFIX)]
        if new_tags == old_tags:
            return None
        entity.tags = new_tags
        return RepairAction(
            strategy_name=self.name,
            issue_code=issue.code,
            attribute_path=(
                "_entity.tags" if getattr(command, "_entity", None) is entity
                else "_created_entity.tags"
            ),
            old_value=old_tags,
            new_value=new_tags,
            entity_ref=issue.entity_ref,
        )


class InvalidConstraintStrategy:
    """Removes a broken, dangling, duplicate, or self-referential constraint
    from the command-owned entity's `constraints` list.
    """

    name = "InvalidConstraintStrategy"
    _CODES = frozenset(
        {
            "BROKEN_CONSTRAINT",
            "UNKNOWN_GROUP",
            "INVALID_ONTOLOGY_REFERENCE",
            "SELF_REFERENTIAL_CONSTRAINT",
            "DUPLICATE_RELATIONSHIP",
            "DANGLING_DEPENDENCY",
        }
    )

    def supports(self, issue: ValidationIssue) -> bool:
        return issue.code in self._CODES

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        entity = _owned_entity(command, issue.entity_ref)
        if entity is None:
            return None
        old_constraints = list(entity.constraints)
        seen: set[str] = set()
        new_constraints: list[str] = []
        known_ids = set()
        for ref in old_constraints:
            if ref == entity.id or ref in seen:
                continue
            seen.add(ref)
            new_constraints.append(ref)
        if new_constraints == old_constraints:
            return None
        entity.constraints = new_constraints
        return RepairAction(
            strategy_name=self.name,
            issue_code=issue.code,
            attribute_path=(
                "_entity.constraints" if getattr(command, "_entity", None) is entity
                else "_created_entity.constraints"
            ),
            old_value=old_constraints,
            new_value=new_constraints,
            entity_ref=issue.entity_ref,
        )


class CollisionRepositionStrategy:
    """Repairs a `COLLISION_DETECTED` issue by relocating the command-owned
    entity to the nearest collision-free position, using `CollisionDetector`.
    """

    name = "CollisionRepositionStrategy"

    def __init__(self, detector: CollisionDetector, world_provider) -> None:
        self._detector = detector
        # Callable[[], WorldSpec] — returns the post-command scratch world
        # the current repair attempt is being validated against, so the
        # search for a free position accounts for every other entity.
        self._world_provider = world_provider

    def supports(self, issue: ValidationIssue) -> bool:
        return issue.code == "COLLISION_DETECTED"

    def repair(self, command: Command, issue: ValidationIssue) -> Optional[RepairAction]:
        entity = _owned_entity(command, issue.entity_ref)
        if entity is None:
            return None
        world = self._world_provider()
        live_entity = world.get_entity(entity.id) or entity
        free_position = self._detector.nearest_free_position(world, live_entity)
        if free_position is None:
            return None
        old_value = Vec3(entity.state.position.x, entity.state.position.y, entity.state.position.z)
        if (old_value.x, old_value.y, old_value.z) == (free_position.x, free_position.y, free_position.z):
            return None
        entity.state.position = free_position
        return RepairAction(
            strategy_name=self.name,
            issue_code=issue.code,
            attribute_path=(
                "_entity.state.position" if getattr(command, "_entity", None) is entity
                else "_created_entity.state.position"
            ),
            old_value=old_value,
            new_value=free_position,
            entity_ref=issue.entity_ref,
        )


class RepairEngine:
    """Owns every `RepairStrategy` and dispatches each blocking
    `ValidationIssue` to the first strategy that supports it.
    """

    name = "RepairEngine"

    def __init__(self, collision_detector: Optional[CollisionDetector] = None) -> None:
        self._collision_detector = collision_detector or CollisionDetector()
        self._static_strategies: list[RepairStrategy] = [
            ClampNegativeMassStrategy(),
            ClampNegativeFrictionStrategy(),
            ClampRestitutionStrategy(),
            UnknownMaterialStrategy(),
            BrokenParentStrategy(),
            InvalidConstraintStrategy(),
        ]

    def strategies_for(self, world_provider) -> list[RepairStrategy]:
        """Return the full strategy list, including the collision strategy
        bound to a specific `world_provider` callback for this attempt.
        """
        return [
            *self._static_strategies,
            CollisionRepositionStrategy(self._collision_detector, world_provider),
        ]

    def repair(
        self,
        command: Command,
        issues: list[ValidationIssue],
        world_provider,
    ) -> list[RepairAction]:
        """Attempt to repair every repairable, blocking issue in `issues`.
        Returns the list of `RepairAction`s actually applied. Issues that
        no strategy supports, or that a strategy declines to fix (returns
        `None`), are left untouched — they will surface again on the next
        validation pass if truly unresolved.
        """
        actions: list[RepairAction] = []
        strategies = self.strategies_for(world_provider)
        for issue in issues:
            if not issue.severity.blocking or not issue.repairable:
                continue
            for strategy in strategies:
                if strategy.supports(issue):
                    action = strategy.repair(command, issue)
                    if action is not None:
                        actions.append(action)
                        logger.debug("Repair applied: %s", action)
                    break
        return actions


# ════════════════════════════════════════════════════════════════════════
# ReasoningPipeline
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningPipelineConfig:
    """Tunables for `ReasoningPipeline`."""

    max_repair_attempts: int = 5
    run_collision_detection: bool = True


class ReasoningPipeline:
    """The reasoning layer's main entry point. Sits between `Command`
    creation and `WorldController.execute()`: validates, solves, detects
    collisions, and self-repairs a command against disposable deep copies
    of the live `WorldSpec`, never touching the live `WorldSpec` itself.

    Thread-safety: a single `ReasoningPipeline` instance may be shared
    across threads. All state mutated during `execute()` is either local
    to the call or protected by an internal `RLock`; the `WorldSpec`
    passed in is only ever read, then deep-copied before any dry-run
    execution occurs.
    """

    def __init__(
        self,
        config: Optional[ReasoningPipelineConfig] = None,
        event_bus: Optional[EventBus] = None,
        validators: Optional[list[Validator]] = None,
    ) -> None:
        self.config = config or ReasoningPipelineConfig()
        self._event_bus = event_bus
        self._validators: list[Validator] = validators or [v() for v in ALL_VALIDATORS]
        self._solver = ConstraintSolver()
        self._collision_detector = CollisionDetector()
        self._repair_engine = RepairEngine(self._collision_detector)
        self._lock = threading.RLock()

    # ── public API ──────────────────────────────────────────────────

    def execute(
        self,
        command: Command,
        world_spec: WorldSpec,
        context: Optional[CommandContext] = None,
    ) -> ValidationResult:
        """Validate (and, where possible, repair) `command` against
        `world_spec` without ever mutating `world_spec`. Returns a
        `ValidationResult` describing whether `WorldController` should
        proceed to actually apply the (possibly repaired) command.
        """
        with self._lock:
            baseline_issues = self._run_all_validators(world_spec)
            baseline_keys = {i.key() for i in baseline_issues}

            attempts = 0
            all_repairs: list[RepairAction] = []
            last_new_issues: list[ValidationIssue] = []
            last_working_world: WorldSpec = world_spec

            while attempts < self.config.max_repair_attempts:
                attempts += 1
                working_world = copy.deepcopy(world_spec)

                touched_ids = self._apply_command_dry_run(command, working_world, context)

                new_issues = self._diff_new_issues(
                    self._run_all_validators(working_world), baseline_keys
                )
                new_issues.extend(self._solver.solve(working_world, touched_ids))

                if self.config.run_collision_detection:
                    new_issues.extend(
                        self._collision_issues(working_world, touched_ids, baseline_keys)
                    )

                self._mark_repairability(new_issues, command)
                last_new_issues = new_issues
                last_working_world = working_world

                blocking = [i for i in new_issues if i.severity.blocking]
                if not blocking:
                    result = ValidationResult(
                        approved=True,
                        command=command,
                        issues=new_issues,
                        repairs_applied=all_repairs,
                        attempts=attempts,
                    )
                    self._emit(command, result)
                    return result

                repairable_blocking = [i for i in blocking if i.repairable]
                if not repairable_blocking:
                    # Nothing left we're able to fix automatically — give up.
                    break

                world_snapshot = working_world  # captured for the closure below

                def _provider(_w: WorldSpec = world_snapshot) -> WorldSpec:
                    return _w

                repairs = self._repair_engine.repair(command, blocking, _provider)
                if not repairs:
                    # Every repairable issue's strategy declined to act —
                    # further attempts would loop with no progress.
                    break
                all_repairs.extend(repairs)

            result = ValidationResult(
                approved=False,
                command=command,
                issues=last_new_issues,
                repairs_applied=all_repairs,
                attempts=attempts,
            )
            self._emit(command, result)
            return result

    # ── internals ───────────────────────────────────────────────────

    def _run_all_validators(self, world: WorldSpec) -> list[ValidationIssue]:
        index = _entity_index(world)
        issues: list[ValidationIssue] = []
        for validator in self._validators:
            issues.extend(validator.validate(world, index))
        return issues

    @staticmethod
    def _diff_new_issues(
        candidate_issues: list[ValidationIssue], baseline_keys: set[tuple[str, Optional[str]]]
    ) -> list[ValidationIssue]:
        """Only issues absent from the pre-command baseline are "new" —
        pre-existing, unrelated problems are never grounds for rejection.
        """
        return [issue for issue in candidate_issues if issue.key() not in baseline_keys]

    def _apply_command_dry_run(
        self, command: Command, world: WorldSpec, context: Optional[CommandContext]
    ) -> set[str]:
        """Execute `command` against `world` (a scratch deep copy) and
        return the set of entity ids the command touched, for scoping
        constraint solving and collision detection. Supports either an
        `apply(world, context)` or `execute(world, context)` command
        convention, matching whichever `commands.py` implements.
        """
        before_ids = {e.id for e in world.entities}
        try:
            if hasattr(command, "apply"):
                command.apply(world, context)
            elif hasattr(command, "execute"):
                command.execute(world, context)
            else:
                raise WorldControllerError(
                    f"{command.__class__.__name__} exposes neither apply() nor execute()."
                )
        except WorldControllerError:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad: dry-run isolation
            raise WorldControllerError(
                f"Dry-run execution of {command.__class__.__name__} failed: {exc}"
            ) from exc

        after_ids = {e.id for e in world.entities}
        touched: set[str] = after_ids ^ before_ids  # created/removed
        explicit_id = getattr(command, "entity_id", None)
        if explicit_id:
            touched.add(explicit_id)
        owned = getattr(command, "_entity", None) or getattr(command, "_created_entity", None)
        if owned is not None:
            touched.add(owned.id)
        return touched

    def _collision_issues(
        self,
        world: WorldSpec,
        touched_ids: set[str],
        baseline_keys: set[tuple[str, Optional[str]]],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for report in self._collision_detector.scan(world, touched_ids or None):
            issue = ValidationIssue(
                severity=DiagnosticSeverity.ERROR,
                code="COLLISION_DETECTED",
                message=(
                    f"Entity {report.entity_a!r} overlaps entity {report.entity_b!r} "
                    f"by {report.overlap.to_dict()}."
                ),
                source=self._collision_detector.name,
                entity_ref=report.entity_a if report.entity_a in touched_ids else report.entity_b,
                repair_hint={"field": "position", "report": report.to_dict()},
            )
            if issue.key() in baseline_keys:
                continue
            issues.append(issue)
        return issues

    @staticmethod
    def _mark_repairability(issues: list[ValidationIssue], command: Command) -> None:
        for issue in issues:
            field_name = issue.repair_hint.get("field")
            if field_name is None:
                continue
            path = _find_command_attribute_path(command, issue.entity_ref, field_name)
            if path is not None:
                issue.repairable = True
                issue.repair_hint["attribute_path"] = path
            elif _owned_entity(command, issue.entity_ref) is not None:
                # Structural fields (tags/constraints/position) are repaired
                # by mutating the owned Entity directly rather than via a
                # single scalar path — still repairable, just not through
                # `_path_set`.
                issue.repairable = True

    def _emit(self, command: Command, result: ValidationResult) -> None:
        if self._event_bus is None:
            return
        try:
            event_type = (
                ChangeEventType.VALIDATION_APPROVED
                if result.approved
                else ChangeEventType.VALIDATION_REJECTED
            )
            self._event_bus.publish(event_type, {"command": command, "result": result.to_dict()})
        except Exception:  # noqa: BLE001 - telemetry must never break reasoning
            logger.exception("Failed to publish reasoning pipeline event.")
        logger.info(result.format())
