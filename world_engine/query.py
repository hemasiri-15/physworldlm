"""
world_engine.query
──────────────────────
World Query Language (WQL) — "almost SQL for WorldSpec".

Two layers, by design:

1. `WorldQuery` — a typed, chainable, programmatic builder. This is the
   primary, robust interface: every other subsystem (constraints, plugins,
   a future UI) should build queries this way, not by hand-parsing text.

2. `WQLParser` — a small, deliberately non-exhaustive pattern matcher over
   a handful of canonical English phrasings ("find all vehicles", "find
   moving vehicles", "find nearest pedestrian to e_car", "find every
   object touching water", "find <type> near <id> within <r>"). It exists
   for convenience (chat-driven or notebook-driven querying) and compiles
   down to a `WorldQuery` — it is NOT a general natural-language
   understanding layer, and unrecognized phrasings raise `QueryError`
   rather than guessing. New phrasings are added via `register_pattern`,
   keeping the parser open for extension without a rewrite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from world_spec import Entity, Vec3
from .exceptions import QueryError

if TYPE_CHECKING:
    from .engine import WorldEngine


_SPEED_EPS = 1e-6


@dataclass
class WorldQuery:
    """
    Chainable, lazily-executed query over a `WorldEngine`'s entities.

    Every `.where`-style method returns `self` for chaining and appends a
    predicate; `.execute()` (or `.first()` / `.nearest_to()`) is what
    actually runs it. Predicates are ANDed together.
    """
    engine: "WorldEngine"
    _predicates: list[Callable[[Entity], bool]] = field(default_factory=list)
    _limit: Optional[int] = None
    _order_by_distance_to: Optional[Vec3] = None

    # ── filters ──────────────────────────────────────────────────────────

    def of_type(self, *entity_types: str) -> "WorldQuery":
        types = set(entity_types)
        self._predicates.append(lambda e: e.entity_type in types)
        return self

    def static(self, is_static: bool = True) -> "WorldQuery":
        self._predicates.append(lambda e: e.is_static == is_static)
        return self

    def moving(self, is_moving: bool = True) -> "WorldQuery":
        def pred(e: Entity) -> bool:
            speed = e.state.velocity.magnitude()
            return (speed > _SPEED_EPS) == is_moving
        self._predicates.append(pred)
        return self

    def material(self, name: str) -> "WorldQuery":
        self._predicates.append(lambda e: e.material == name)
        return self

    def with_tag(self, tag: str) -> "WorldQuery":
        self._predicates.append(lambda e: tag in (e.tags or []))
        return self

    def near(self, point: Vec3, radius: float) -> "WorldQuery":
        """Restrict to entities within `radius` of `point` (uses the spatial index)."""
        nearby_ids = {eid for eid, _d in self.engine.spatial_index.within_radius(point, radius)}
        self._predicates.append(lambda e: e.id in nearby_ids)
        return self

    def near_entity(self, entity_id: str, radius: float) -> "WorldQuery":
        target = self.engine.get_entity(entity_id)
        return self.near(target.state.position, radius)

    def touching(self, material: str, radius: float = 0.5) -> "WorldQuery":
        """
        Entities within `radius` of any entity made of `material` — a
        proximity-based proxy for "touching" appropriate for the symbolic
        layer (exact contact manifolds are SimulationEngine's job).
        """
        material_entities = [e for e in self.engine.list_entities() if e.material == material]

        def pred(e: Entity) -> bool:
            for other in material_entities:
                if other.id == e.id:
                    continue
                if _distance(e.state.position, other.state.position) <= radius:
                    return True
            return False
        self._predicates.append(pred)
        return self

    def related(self, predicate: str, target_id: Optional[str] = None,
                as_subject: bool = True) -> "WorldQuery":
        """Entities connected via an EntityGraph relation, optionally to a specific target."""
        def pred(e: Entity) -> bool:
            edges = (self.engine.entity_graph.relations_from(e.id, predicate) if as_subject
                     else self.engine.entity_graph.relations_to(e.id, predicate))
            if target_id is None:
                return len(edges) > 0
            key = (lambda r: r.obj) if as_subject else (lambda r: r.subject)
            return any(key(r) == target_id for r in edges)
        self._predicates.append(pred)
        return self

    def matching(self, predicate_name: str) -> "WorldQuery":
        """Apply a named predicate contributed by a plugin (see PluginManager)."""
        predicates = self.engine.plugins.collect_query_predicates()
        fn = predicates.get(predicate_name)
        if fn is None:
            raise QueryError(f"No registered query predicate named '{predicate_name}'")
        self._predicates.append(lambda e: fn(self.engine, e))
        return self

    def custom(self, predicate: Callable[[Entity], bool]) -> "WorldQuery":
        self._predicates.append(predicate)
        return self

    def limit(self, n: int) -> "WorldQuery":
        self._limit = n
        return self

    def order_by_distance_to(self, point: Vec3) -> "WorldQuery":
        self._order_by_distance_to = point
        return self

    # ── execution ────────────────────────────────────────────────────────

    def execute(self) -> list[Entity]:
        results = [e for e in self.engine.list_entities()
                   if all(p(e) for p in self._predicates)]
        if self._order_by_distance_to is not None:
            origin = self._order_by_distance_to
            results.sort(key=lambda e: _distance(e.state.position, origin))
        if self._limit is not None:
            results = results[: self._limit]
        return results

    def first(self) -> Optional[Entity]:
        results = self.execute()
        return results[0] if results else None

    def nearest_to(self, point: Vec3, k: int = 1) -> list[tuple[Entity, float]]:
        """
        Nearest `k` entities satisfying the query's filters to `point`,
        using the spatial index for the distance computation/ordering.
        """
        allowed = {e.id for e in self.execute()}
        hits = self.engine.spatial_index.nearest(point, k=len(allowed) or 1,
                                                  predicate=lambda eid: eid in allowed)
        out = []
        for eid, dist in hits[:k]:
            entity = self.engine.spec.get_entity(eid)
            if entity is not None:
                out.append((entity, dist))
        return out

    def count(self) -> int:
        return len(self.execute())


def _distance(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


# ═════════════════════════════════════════════════════════════════════════
# WQL — a small, pragmatic text DSL compiling to WorldQuery
# ═════════════════════════════════════════════════════════════════════════

_PatternHandler = Callable[["WorldEngine", re.Match], WorldQuery]


class WQLParser:
    """
    Pattern-matching text-DSL front end over `WorldQuery`.

    This is intentionally a small, closed set of regex patterns rather
    than a general parser — it covers the phrasings named in the design
    brief ("find moving vehicles", "find nearest pedestrian", "find every
    object touching water") and is meant to be extended with
    `register_pattern`, not grown into a full grammar. For anything more
    expressive, use `WorldQuery` directly.
    """

    def __init__(self) -> None:
        self._patterns: list[tuple[re.Pattern, _PatternHandler]] = []
        self._install_defaults()

    def register_pattern(self, regex: str, handler: _PatternHandler, flags: int = re.IGNORECASE) -> None:
        self._patterns.append((re.compile(regex, flags), handler))

    def parse(self, engine: "WorldEngine", text: str) -> WorldQuery:
        text = text.strip()
        for pattern, handler in self._patterns:
            m = pattern.match(text)
            if m:
                return handler(engine, m)
        raise QueryError(
            f"Could not parse WQL query: '{text}'. "
            "Use WorldQuery directly for anything not covered by a registered pattern."
        )

    def _install_defaults(self) -> None:
        # "find all <type>s" / "find all <type>"
        self.register_pattern(
            r"^find\s+all\s+(?P<type>\w+?)s?$",
            lambda eng, m: WorldQuery(eng).of_type(_singularize(m.group("type"))),
        )
        # "find moving <type>s"
        self.register_pattern(
            r"^find\s+moving\s+(?P<type>\w+?)s?$",
            lambda eng, m: WorldQuery(eng).of_type(_singularize(m.group("type"))).moving(True),
        )
        # "find static <type>s"
        self.register_pattern(
            r"^find\s+static\s+(?P<type>\w+?)s?$",
            lambda eng, m: WorldQuery(eng).of_type(_singularize(m.group("type"))).static(True),
        )
        # "find nearest <type> to <entity_id>"
        self.register_pattern(
            r"^find\s+nearest\s+(?P<type>\w+?)s?\s+to\s+(?P<id>\S+)$",
            lambda eng, m: WorldQuery(eng).of_type(_singularize(m.group("type")))
                .order_by_distance_to(eng.get_entity(m.group("id")).state.position).limit(1),
        )
        # "find every object touching <material>" / "find all touching <material>"
        self.register_pattern(
            r"^find\s+(?:every|all)\s+(?:object|objects)?\s*touching\s+(?P<material>\w+)$",
            lambda eng, m: WorldQuery(eng).touching(m.group("material")),
        )
        # "find <type> near <entity_id> within <radius>"
        self.register_pattern(
            r"^find\s+(?P<type>\w+?)s?\s+near\s+(?P<id>\S+)\s+within\s+(?P<radius>[\d.]+)$",
            lambda eng, m: WorldQuery(eng).of_type(_singularize(m.group("type")))
                .near_entity(m.group("id"), float(m.group("radius"))),
        )


def _singularize(word: str) -> str:
    """Very small heuristic: strip a trailing 's' unless the word is short/irregular-safe."""
    if word.endswith("s") and not word.endswith(("ss", "us")) and len(word) > 3:
        return word[:-1]
    return word
