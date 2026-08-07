"""
spatial_relationship_parser.py
────────────────────────────────
Extracted from world_parser.py during the architecture-audit cleanup
(file-length reduction — original file mixed the LLM pipeline, spatial
parsing, and WorldParser orchestration in one ~500-line file). This class
is otherwise byte-identical to the original.

Purely deterministic; no LLM calls.
Parses natural-language spatial relationships from a scene description
and updates entity positions in-place to satisfy the stated geometry.

Coordinate system (inherited from WorldSpec):
    x = East / forward
    y = Up
    z = North

Axis assignments per relation:
    left / right / in front of / behind / distance  → x-axis
    above / on / below                              → y-axis
    near                                            → x-axis (close separation)
"""

from __future__ import annotations

import re

from world_spec import Entity, Vec3


class SpatialRelationshipParser:
    """
    Detect and resolve natural-language spatial relationships into SI
    coordinates on a set of :class:`~world_spec.Entity` objects.

    Parameters
    ----------
    near_threshold_m : float
        Maximum separation (metres) that "near" implies.  Default 4.0 m.
    gap_m : float
        Minimum clearance gap between bounding boxes when resolving contact
        relations such as "on" or "next to".  Default 0.05 m.

    Public API
    ----------
    apply(description, entities)
        Parse ``description`` for spatial phrases and mutate the
        ``entities`` list's position fields in-place.

    parse_relationships(description, entities)
        Return a list of :class:`SpatialRelation` dicts without mutating
        anything — useful for inspection and testing.
    """

    # Gap kept between bounding boxes to avoid inter-penetration
    _GAP: float = 0.05    # metres
    # Default separation for "near"
    _NEAR_SEP: float = 3.0  # metres

    _NUM = r"(\d+(?:\.\d+)?)"
    _UNIT = r"(?:\s*(?:meters?|metres?|km|ft|feet|centimeters?|cm|m\b))?"

    _RE_DISTANCE = re.compile(
        rf"(\w[\w\s]*?)\s+(?:is|are)\s+{_NUM}\s*{_UNIT}\s+(?:away\s+)?from\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_ABOVE = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?above\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_BELOW = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?below\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_ON = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?on(?:\s+top\s+of|\s+a|\s+the|\s+an)?\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_LEFT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_RIGHT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_BEHIND = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?behind\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_IN_FRONT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?in\s+front\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )
    _RE_NEAR = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?near\s+(?:a\s+|the\s+|an\s+)?(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    def __init__(self, near_threshold_m: float = 4.0, gap_m: float = 0.05) -> None:
        self._near_threshold = near_threshold_m
        self._GAP = gap_m

    def apply(self, description: str, entities: list[Entity]) -> None:
        """Detect spatial relationships in *description* and mutate entity
        positions in-place. Only position coordinates are modified.
        """
        label_map = self._build_label_map(entities)
        relations = self._detect_relations(description, label_map)
        self._resolve(relations, label_map)

    def parse_relationships(self, description: str, entities: list[Entity]) -> list[dict]:
        """Parse without mutating."""
        label_map = self._build_label_map(entities)
        return self._detect_relations(description, label_map)

    @staticmethod
    def _build_label_map(entities: list[Entity]) -> dict[str, Entity]:
        candidates: list[tuple[str, Entity]] = []
        for entity in entities:
            tokens: list[str] = []
            tokens.append(entity.label.lower())
            tokens.extend(entity.label.lower().split())
            tokens.append(entity.entity_type.lower())
            for tag in entity.tags:
                tokens.append(tag.lower())
            for tok in tokens:
                tok = tok.strip(".,;:()")
                if tok:
                    candidates.append((tok, entity))
        candidates.sort(key=lambda x: len(x[0]))
        return {tok: ent for tok, ent in candidates}

    @staticmethod
    def _resolve_entity(raw_name: str, label_map: dict[str, Entity]) -> Entity | None:
        name = raw_name.strip().lower()
        if name in label_map:
            return label_map[name]
        for word in name.split():
            w = word.strip(".,;:()")
            if w in label_map:
                return label_map[w]
        for key, ent in label_map.items():
            if key in name:
                return ent
        return None

    def _detect_relations(self, description: str, label_map: dict[str, Entity]) -> list[dict]:
        relations: list[dict] = []

        def add(rel: str, raw_a: str, raw_b: str, **kw) -> None:
            ea = self._resolve_entity(raw_a, label_map)
            eb = self._resolve_entity(raw_b, label_map)
            if ea is not None and eb is not None and ea is not eb:
                entry = {"relation": rel, "entity_a": ea, "entity_b": eb}
                entry.update(kw)
                relations.append(entry)

        for m in self._RE_DISTANCE.finditer(description):
            raw_a, dist_str, raw_b = m.group(1), m.group(2), m.group(3)
            try:
                add("distance", raw_a, raw_b, distance_m=float(dist_str))
            except ValueError:
                pass

        for m in self._RE_ABOVE.finditer(description):
            add("above", m.group(1), m.group(2))
        for m in self._RE_BELOW.finditer(description):
            add("below", m.group(1), m.group(2))
        for m in self._RE_ON.finditer(description):
            add("on", m.group(1), m.group(2))
        for m in self._RE_LEFT.finditer(description):
            add("left", m.group(1), m.group(2))
        for m in self._RE_RIGHT.finditer(description):
            add("right", m.group(1), m.group(2))
        for m in self._RE_BEHIND.finditer(description):
            add("behind", m.group(1), m.group(2))
        for m in self._RE_IN_FRONT.finditer(description):
            add("in_front", m.group(1), m.group(2))
        for m in self._RE_NEAR.finditer(description):
            add("near", m.group(1), m.group(2))

        return relations

    def _resolve(self, relations: list[dict], label_map: dict[str, Entity]) -> None:
        for rel in relations:
            kind = rel["relation"]
            ea: Entity = rel["entity_a"]
            eb: Entity = rel["entity_b"]

            pos_a = ea.state.position
            pos_b = eb.state.position
            bb_a = ea.bounding_box
            bb_b = eb.bounding_box

            if kind == "distance":
                dist = rel.get("distance_m", 10.0)
                new_x = pos_b.x + dist
                ea.state.position = Vec3(new_x, pos_a.y, pos_a.z)

            elif kind == "above":
                b_top = pos_b.y + bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(pos_a.x, b_top + a_half_h + self._GAP, pos_a.z)

            elif kind == "below":
                b_base = pos_b.y - bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(pos_a.x, b_base - a_half_h - self._GAP, pos_a.z)

            elif kind == "on":
                b_top = pos_b.y + bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(pos_b.x, b_top + a_half_h, pos_b.z)

            elif kind == "left":
                b_left_edge = pos_b.x - bb_b.width / 2.0
                a_half_w = bb_a.width / 2.0
                ea.state.position = Vec3(b_left_edge - a_half_w - self._GAP, pos_a.y, pos_a.z)

            elif kind == "right":
                b_right_edge = pos_b.x + bb_b.width / 2.0
                a_half_w = bb_a.width / 2.0
                ea.state.position = Vec3(b_right_edge + a_half_w + self._GAP, pos_a.y, pos_a.z)

            elif kind == "behind":
                b_back_edge = pos_b.x - bb_b.depth / 2.0
                a_half_d = bb_a.depth / 2.0
                ea.state.position = Vec3(b_back_edge - a_half_d - self._GAP, pos_a.y, pos_a.z)

            elif kind == "in_front":
                b_front_edge = pos_b.x + bb_b.depth / 2.0
                a_half_d = bb_a.depth / 2.0
                ea.state.position = Vec3(b_front_edge + a_half_d + self._GAP, pos_a.y, pos_a.z)

            elif kind == "near":
                ea.state.position = Vec3(pos_b.x + self._NEAR_SEP, pos_a.y, pos_a.z)
