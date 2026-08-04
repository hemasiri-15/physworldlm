"""
world_engine.ontology
──────────────────────
Open-world symbolic ontology for entity *categories*: what `entity_type`s
exist, whether they're normally static, whether they need ground contact,
and which materials make sense for them.

This is deliberately category-level and *not* instance-level. Instance-level
domain reasoning ("this specific boat must be near this specific body of
water") lives in `world_engine.constraints.SemanticConstraintEngine`, which
is a strictly more expressive layer built on top of the entity graph and
spatial index. Keep the split: ontology = "what kind of thing is this and
does it look right in isolation", constraints = "does this thing make sense
given everything else in the world".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from world_spec import Entity


class EntityCategory(str, Enum):
    """Canonical top-level entity categories known to the symbolic layer."""
    VEHICLE = "vehicle"
    PROJECTILE = "projectile"
    FLUID = "fluid"
    AGENT = "agent"
    STRUCTURE = "structure"
    TERRAIN = "terrain"
    OBJECT = "object"


@dataclass(frozen=True)
class OntologyRule:
    """
    Symbolic expectations for one entity category.

    Attributes:
        category: The `entity_type` string this rule governs.
        expected_static: Expected `Entity.is_static`, or `None` if either
            value is valid (e.g. fluids: a lake is static, a river is not).
        requires_ground_contact: Whether this category is expected to
            participate in a ground/contact interaction unless airborne.
        compatible_materials: Materials that are ontologically sane for
            this category. Empty frozenset means "no restriction".
        description: Human-readable rationale, surfaced in diagnostics.
    """
    category: str
    expected_static: Optional[bool]
    requires_ground_contact: bool
    compatible_materials: frozenset[str]
    description: str


class OntologyRegistry:
    """
    Open-world registry of `OntologyRule`s, queried on every structural
    edit (`WorldEngine.add_entity` / `update_entity`).

    Future extension notes:
        - Swap `_rules` for a query against an RDF/OWL store or knowledge
          graph service to support sub-typing (e.g. "vehicle" ->
          "sedan"/"truck"/"drone" with inherited constraints).
        - `register_type` is the seam for LLM-discovered categories to be
          promoted into the closed ontology at runtime.
    """

    def __init__(self) -> None:
        self._rules: dict[str, OntologyRule] = self._default_rules()

    @staticmethod
    def _default_rules() -> dict[str, OntologyRule]:
        no_restriction: frozenset[str] = frozenset()
        return {
            EntityCategory.VEHICLE: OntologyRule(
                EntityCategory.VEHICLE, expected_static=False,
                requires_ground_contact=True,
                compatible_materials=frozenset({"steel", "plastic", "rubber", "generic"}),
                description="Vehicles are dynamic, grounded, and typically metal/plastic bodied.",
            ),
            EntityCategory.PROJECTILE: OntologyRule(
                EntityCategory.PROJECTILE, expected_static=False,
                requires_ground_contact=False,
                compatible_materials=no_restriction,
                description="Projectiles are dynamic and airborne for at least part of their lifetime.",
            ),
            EntityCategory.FLUID: OntologyRule(
                EntityCategory.FLUID, expected_static=None,
                requires_ground_contact=False,
                compatible_materials=frozenset({"water", "generic", "air"}),
                description="Fluids may be static (a lake) or dynamic (a flow/current).",
            ),
            EntityCategory.AGENT: OntologyRule(
                EntityCategory.AGENT, expected_static=False,
                requires_ground_contact=True,
                compatible_materials=frozenset({"flesh", "steel", "plastic", "generic"}),
                description="Agents (people, animals, robots) are dynamic and normally grounded.",
            ),
            EntityCategory.STRUCTURE: OntologyRule(
                EntityCategory.STRUCTURE, expected_static=True,
                requires_ground_contact=True,
                compatible_materials=frozenset({"concrete", "steel", "wood", "glass", "generic"}),
                description="Structures are static and grounded (buildings, walls, bridges).",
            ),
            EntityCategory.TERRAIN: OntologyRule(
                EntityCategory.TERRAIN, expected_static=True,
                requires_ground_contact=False,
                compatible_materials=no_restriction,
                description="Terrain is static and IS the ground reference, not contacting it.",
            ),
            EntityCategory.OBJECT: OntologyRule(
                EntityCategory.OBJECT, expected_static=None,
                requires_ground_contact=False,
                compatible_materials=no_restriction,
                description="Generic catch-all category with no strong structural expectation.",
            ),
        }

    def register_type(self, rule: OntologyRule) -> None:
        """Install or override the rule for an entity category at runtime. O(1)."""
        self._rules[rule.category] = rule

    def get_rule(self, entity_type: str) -> Optional[OntologyRule]:
        """Return the OntologyRule for `entity_type`, or None if unknown (open-world)."""
        return self._rules.get(entity_type)

    def is_known_type(self, entity_type: str) -> bool:
        return entity_type in self._rules

    def check_material_compatibility(self, entity_type: str, material: str) -> bool:
        rule = self._rules.get(entity_type)
        if rule is None or not rule.compatible_materials:
            return True
        return material in rule.compatible_materials

    def requires_ground_contact(self, entity_type: str) -> bool:
        rule = self._rules.get(entity_type)
        return bool(rule and rule.requires_ground_contact)

    def evaluate_entity(self, entity: Entity) -> list[str]:
        """
        Purpose:
            Produce human-readable, advisory-only ontology messages for one
            entity (never raises — the caller decides what becomes fatal).
        Inputs:
            entity: Entity to evaluate.
        Outputs:
            list[str] of zero or more warning messages.
        Complexity:
            O(1).
        """
        messages: list[str] = []
        rule = self._rules.get(entity.entity_type)
        if rule is None:
            messages.append(
                f"entity_type '{entity.entity_type}' is not in the known ontology "
                "(open-world: allowed, but unverified)."
            )
            return messages

        if rule.expected_static is not None and entity.is_static != rule.expected_static:
            messages.append(
                f"entity_type '{entity.entity_type}' is normally "
                f"is_static={rule.expected_static}, got {entity.is_static}."
            )
        if not self.check_material_compatibility(entity.entity_type, entity.material):
            messages.append(
                f"material '{entity.material}' is unusual for entity_type "
                f"'{entity.entity_type}' ({rule.description})."
            )
        return messages
