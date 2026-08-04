"""
world_engine.constraints
────────────────────────────
SemanticConstraintEngine — domain-specific, instance-level reasoning that
goes beyond generic ontology checks.

`OntologyRegistry` (ontology.py) answers "does this entity look right in
isolation" (a structure should be static, a boat should be made of a
plausible hull material). `SemanticConstraintEngine` answers "does this
entity make sense *given everything else in the world*" — e.g. a boat
entity is only sane if there is a body of water nearby for it to float on.

A `SemanticConstraint` is a predicate over `(engine, entity)` that
consults the live `WorldEngine` (its entities, `EntityGraph` relations,
and `SpatialIndex`) rather than the entity in isolation. This file ships a
small built-in rule set covering common LLM-authored-scene failure modes;
`ConstraintRegistry.register` is the extension point for
domain plugins (traffic, robotics, ...) to add their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .engine import WorldEngine
    from world_spec import Entity


@dataclass(frozen=True)
class SemanticConstraint:
    """
    One domain rule.

    Attributes:
        name: unique identifier, e.g. "boat_requires_water".
        applies_to: entity_type this rule is checked against ("*" = all).
        check: callable(engine, entity) -> bool; True means the constraint
            is SATISFIED. Receives the live WorldEngine so it can query
            neighbors, relations, and spatial proximity.
        message: human-readable explanation used when the check fails.
        severity: "ERROR" (hard — can be raised as SemanticConstraintError)
            or "WARNING" (advisory — always just collected).
    """
    name: str
    applies_to: str
    check: Callable[["WorldEngine", "Entity"], bool]
    message: str
    severity: str = "WARNING"


@dataclass
class ConstraintViolation:
    constraint_name: str
    entity_id: str
    message: str
    severity: str


class ConstraintRegistry:
    """Holds the active set of `SemanticConstraint`s, keyed by name."""

    def __init__(self) -> None:
        self._constraints: dict[str, SemanticConstraint] = {}

    def register(self, constraint: SemanticConstraint) -> None:
        self._constraints[constraint.name] = constraint

    def unregister(self, name: str) -> None:
        self._constraints.pop(name, None)

    def for_type(self, entity_type: str) -> list[SemanticConstraint]:
        return [c for c in self._constraints.values()
                if c.applies_to == "*" or c.applies_to == entity_type]

    def all(self) -> list[SemanticConstraint]:
        return list(self._constraints.values())


class SemanticConstraintEngine:
    """
    Evaluates `SemanticConstraint`s against a live `WorldEngine`.

    Usage:
        engine = SemanticConstraintEngine()
        engine.registry.register(BOAT_REQUIRES_WATER)
        violations = engine.evaluate(world_engine)
    """

    def __init__(self, registry: ConstraintRegistry | None = None,
                 install_defaults: bool = True) -> None:
        self.registry = registry or ConstraintRegistry()
        if install_defaults:
            for c in default_constraints():
                self.registry.register(c)

    def evaluate(self, world_engine: "WorldEngine") -> list[ConstraintViolation]:
        """
        Purpose:
            Run every applicable constraint against every entity.
        Complexity:
            O(E * C_avg) where C_avg is the average number of constraints
            applicable per entity_type; each individual `check` call's cost
            depends on what it queries (typically O(log n) via the spatial
            index, or O(deg) via the entity graph).
        """
        violations: list[ConstraintViolation] = []
        for entity in world_engine.list_entities():
            for constraint in self.registry.for_type(entity.entity_type):
                try:
                    ok = constraint.check(world_engine, entity)
                except Exception as exc:  # a broken plugin rule should not crash validation
                    violations.append(ConstraintViolation(
                        constraint.name, entity.id,
                        f"constraint raised {type(exc).__name__}: {exc}", "WARNING",
                    ))
                    continue
                if not ok:
                    violations.append(ConstraintViolation(
                        constraint.name, entity.id, constraint.message, constraint.severity,
                    ))
        return violations


# ─────────────────────────────────────────────────────────────────────────
# Built-in domain rules
# ─────────────────────────────────────────────────────────────────────────

def _near_material(world_engine: "WorldEngine", entity: "Entity",
                    material: str, radius: float) -> bool:
    """Shared helper: is there an entity of `material` within `radius` of `entity`?"""
    nearby = world_engine.spatial_index.within_radius(entity.state.position, radius)
    for eid, _dist in nearby:
        if eid == entity.id:
            continue
        other = world_engine.spec.get_entity(eid)
        if other is not None and other.material == material:
            return True
    # Fall back to explicit symbolic relations (e.g. "on_water", "floats_on")
    # in case the scene author declared the relation without spatial proximity.
    return bool(world_engine.entity_graph.relations_from(entity.id, "floats_on")
                or world_engine.entity_graph.relations_from(entity.id, "on_water"))


def default_constraints() -> list[SemanticConstraint]:
    """The built-in rule set covering common LLM-scene failure modes."""
    return [
        SemanticConstraint(
            name="boat_requires_water",
            applies_to="vehicle",
            check=lambda eng, e: (
                "boat" not in (e.tags or []) and "ship" not in (e.tags or [])
            ) or _near_material(eng, e, "water", radius=25.0),
            message="a boat/ship must be near a 'water' material entity or have a "
                    "floats_on/on_water relation — none found within 25 m",
            severity="WARNING",
        ),
        SemanticConstraint(
            name="fluid_needs_container_or_terrain",
            applies_to="fluid",
            check=lambda eng, e: e.is_static or bool(
                eng.entity_graph.relations_from(e.id, "contained_by")
                or eng.entity_graph.relations_from(e.id, "flows_over")
            ),
            message="a dynamic fluid entity has no 'contained_by' or 'flows_over' "
                    "relation to a container/terrain entity",
            severity="WARNING",
        ),
        SemanticConstraint(
            name="agent_not_inside_structure_wall",
            applies_to="agent",
            check=lambda eng, e: not any(
                other.entity_type == "structure"
                and _bbox_contains(other, e.state.position)
                for other in eng.list_entities()
            ),
            message="an agent's position falls fully inside a structure's bounding "
                    "box — likely embedded in a wall rather than inside/outside it",
            severity="WARNING",
        ),
    ]


def _bbox_contains(structure: "Entity", point) -> bool:
    """Rough point-in-bounding-box test using the structure's own position as center."""
    c = structure.state.position
    bb = structure.bounding_box
    return (
        abs(point.x - c.x) < bb.width / 2
        and abs(point.y - c.y) < bb.height / 2
        and abs(point.z - c.z) < bb.depth / 2
    )
