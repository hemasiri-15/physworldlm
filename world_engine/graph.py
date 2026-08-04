"""
world_engine.graph
─────────────────────
EntityGraph — the knowledge-graph layer over a world's entities.

`WorldSpec.interactions` already models pairwise *physics* couplings
(collision, contact, joint, ...). `EntityGraph` models pairwise *symbolic*
relations: "e_car on_road e_road", "e_pedestrian near_crosswalk e_light",
"e_trailer part_of e_truck". These are first-class, queryable, directed,
labeled edges — the substrate that makes `WorldEngine.query()` ("find every
object touching water", "find the nearest pedestrian") and
`SemanticConstraintEngine` ("a boat must be near water") possible.

Relations are intentionally NOT stored inside `WorldSpec` (which stays a
clean physics/geometry contract consumed by SimulationEngine); EntityGraph
is WorldEngine-side symbolic metadata. If a relation needs to survive into
OpenUSD/PhysX, `SimulationEngine` is expected to translate specific
relation types into USD relationships or PhysX joints as appropriate — that
translation is out of scope for this file.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .commands import GraphRelationCommand


@dataclass(frozen=True)
class Relation:
    """One directed, labeled edge: `subject --predicate--> object`."""
    subject: str
    predicate: str
    obj: str
    attributes: dict[str, Any] = field(default_factory=dict, compare=False)

    def __str__(self) -> str:
        return f"{self.subject} -{self.predicate}-> {self.obj}"


class EntityGraph:
    """
    Directed, labeled multigraph of symbolic relations between entity ids.

    Mutation goes through `WorldEngine.add_relation` /
    `WorldEngine.remove_relation`, which wrap `GraphRelationCommand`s so
    relation edits participate in the same undo/redo history as every
    other world mutation. The `_add`/`_remove` methods here are the
    package-private mutation primitives the command calls into — do not
    call them directly from outside `WorldEngine` or history will desync.

    Future extension notes:
        - Add edge validity windows (`valid_from_t`, `valid_to_t`) once
          WorldEngine models continuous time rather than discrete ticks.
        - Back `_by_subject`/`_by_object` indices with a proper graph
          library (networkx, or a Rust/C++ graph core) once relation
          counts grow large enough for pathfinding queries to matter.
    """

    def __init__(self) -> None:
        self._relations: set[Relation] = set()
        self._by_subject: dict[str, set[Relation]] = {}
        self._by_object: dict[str, set[Relation]] = {}
        self._by_predicate: dict[str, set[Relation]] = {}

    # ── package-private mutation primitives (called only by GraphRelationCommand) ──

    def _add(self, subject: str, predicate: str, obj: str, attributes: dict[str, Any]) -> None:
        rel = Relation(subject, predicate, obj, dict(attributes))
        if rel in self._relations:
            return
        self._relations.add(rel)
        self._by_subject.setdefault(subject, set()).add(rel)
        self._by_object.setdefault(obj, set()).add(rel)
        self._by_predicate.setdefault(predicate, set()).add(rel)

    def _remove(self, subject: str, predicate: str, obj: str) -> None:
        target = next(
            (r for r in self._by_subject.get(subject, ())
             if r.predicate == predicate and r.obj == obj),
            None,
        )
        if target is None:
            return
        self._relations.discard(target)
        self._by_subject.get(subject, set()).discard(target)
        self._by_object.get(obj, set()).discard(target)
        self._by_predicate.get(predicate, set()).discard(target)

    def make_add_command(self, subject: str, predicate: str, obj: str,
                          attributes: Optional[dict[str, Any]] = None) -> GraphRelationCommand:
        return GraphRelationCommand(self, subject, predicate, obj, attributes or {}, adding=True)

    def make_remove_command(self, subject: str, predicate: str, obj: str) -> GraphRelationCommand:
        return GraphRelationCommand(self, subject, predicate, obj, {}, adding=False)

    # ── read-only query surface ─────────────────────────────────────────

    def relations_from(self, subject: str, predicate: Optional[str] = None) -> list[Relation]:
        """Outgoing edges from `subject`, optionally filtered by predicate. O(deg(subject))."""
        edges = self._by_subject.get(subject, set())
        return [r for r in edges if predicate is None or r.predicate == predicate]

    def relations_to(self, obj: str, predicate: Optional[str] = None) -> list[Relation]:
        """Incoming edges to `obj`, optionally filtered by predicate. O(deg(obj))."""
        edges = self._by_object.get(obj, set())
        return [r for r in edges if predicate is None or r.predicate == predicate]

    def relations_by_predicate(self, predicate: str) -> list[Relation]:
        """All edges with a given predicate, world-wide. O(|edges of that predicate|)."""
        return list(self._by_predicate.get(predicate, set()))

    def neighbors(self, entity_id: str, predicate: Optional[str] = None) -> set[str]:
        """All entities reachable in one hop from `entity_id` (either direction)."""
        out = {r.obj for r in self.relations_from(entity_id, predicate)}
        inn = {r.subject for r in self.relations_to(entity_id, predicate)}
        return out | inn

    def has_relation(self, subject: str, predicate: str, obj: str) -> bool:
        return any(r.predicate == predicate and r.obj == obj for r in self._by_subject.get(subject, ()))

    def all_relations(self) -> list[Relation]:
        return list(self._relations)

    def drop_entity(self, entity_id: str) -> list[Relation]:
        """
        Remove every relation touching `entity_id` (used when an entity is
        deleted from the world). Returns the removed relations so the
        caller can log/undo them if desired. This is a direct mutation
        (not command-wrapped) intended to be called from within
        `WorldEngine.remove_entity`'s cascade, mirroring how physics
        interactions are cascade-removed.
        """
        removed = [r for r in self._relations
                   if r.subject == entity_id or r.obj == entity_id]
        for r in removed:
            self._remove(r.subject, r.predicate, r.obj)
        return removed

    def __len__(self) -> int:
        return len(self._relations)
