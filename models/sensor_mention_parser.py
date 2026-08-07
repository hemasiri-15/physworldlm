"""
sensor_mention_parser.py
──────────────────────────
Deterministic, rule-based detector for sensor mentions in natural
language, following the exact same architectural pattern as
SpatialRelationshipParser (spatial_relationship_parser.py): a standalone
class with an `apply(description, entities)` method that mutates already-
extracted Entity objects in place — here, appending SensorSpec objects to
`entity.sensors` rather than repositioning them.

No LLM calls. No modification to entity/physics/interaction extraction —
this runs strictly AFTER PromptParser has already produced entities, the
same ordering SpatialRelationshipParser already uses in world_parser.py.

Scope, by design (per the task's explicit constraints):
  - Detects sensor mentions attached to a subject via a small set of
    trigger verbs ("has", "have", "carries", "carry", "is/are equipped
    with", "contains", "contain").
  - Resolves the subject to an already-extracted Entity using the same
    kind of nearest-token, label/entity_type/tag matching
    SpatialRelationshipParser already uses — NOT a new entity-extraction
    pass. If no matching Entity exists (e.g. the sentence's subject noun
    isn't in PromptParser's existing entity vocabulary — "robot" is a
    known example, see module-level note below), the sensor mention is
    recorded as a warning and skipped, never invented as a phantom entity.
  - Extracts ONLY the numeric parameters this module explicitly knows how
    to extract (field-of-view in degrees, WxH resolution, range in
    metres) from text immediately around the sensor mention. Every other
    case leaves `params={}`, per the explicit "never invent unsupported
    parameters" requirement — this module does not guess resolution,
    noise models, or any other sensor-specific field it didn't literally
    read from the text.
  - A sentence with no sensor-trigger phrase produces zero mutation,
    so `entity.sensors` stays `[]` — output is byte-identical to before
    this module existed for any prompt that doesn't mention sensors.

Known limitation (inherent to leaving entity extraction unmodified, not
a bug in this module): PromptParser's entity vocabulary does not include
the noun "robot" today (ENTITY_ALIASES has no "robot" in any category).
A sentence like "The robot has two stereo cameras." will not have a
matching Entity to attach sensors to, and the mention is skipped with a
warning rather than silently dropped or given a fabricated entity. If
"robot" should resolve to an agent/vehicle entity, that's a one-line
addition to prompt_parser.py's ENTITY_ALIASES — deliberately out of
scope here since modifying entity extraction was explicitly excluded.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from world_spec import Entity, SensorSpec

# ─────────────────────────────────────────────
# Sensor vocabulary
# ─────────────────────────────────────────────
# Ordered longest-phrase-first so "thermal camera" matches before the
# bare "camera" fallback. Values match sensors/sensor_types.py's
# SensorType string values as already used elsewhere in this codebase
# (SensorBuilder, SensorManager) — "camera", "depth_camera",
# "thermal_camera", "lidar", "radar", "gps", "imu".
_SENSOR_PHRASES: list[tuple[str, str]] = [
    ("thermal cameras", "thermal_camera"),
    ("thermal camera", "thermal_camera"),
    ("depth cameras", "depth_camera"),
    ("depth camera", "depth_camera"),
    ("stereo cameras", "camera"),
    ("stereo camera", "camera"),
    ("rgb cameras", "camera"),
    ("rgb camera", "camera"),
    ("cameras", "camera"),
    ("camera", "camera"),
    ("lidars", "lidar"),
    ("lidar", "lidar"),
    ("radars", "radar"),
    ("radar", "radar"),
    ("gnss", "gps"),
    ("gps", "gps"),
    ("imus", "imu"),
    ("imu", "imu"),
]

# Words dropped from the constructed sensor NAME because they're already
# implied by sensor_type (e.g. "rgb camera" -> sensor_type is already
# "camera", so "rgb" would be redundant in the name). Positional/
# descriptive words NOT in this set (front, rear, left, right, stereo,
# wide, ...) are kept in the name.
_REDUNDANT_NAME_WORDS = {
    "rgb", "camera", "cameras", "lidar", "lidars", "radar", "radars",
    "gps", "gnss", "imu", "imus", "thermal", "depth",
}

_QUANTITY_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "pair": 2, "couple": 2, "several": 3,
}

_TRIGGER_RE = re.compile(
    r"\b(?:has|have|carries|carry|is\s+equipped\s+with|are\s+equipped\s+with|"
    r"contains|contain)\b",
    re.IGNORECASE,
)

_NUM = r"(\d+(?:\.\d+)?)"
_RE_FOV = re.compile(rf"{_NUM}\s*(?:degree|deg|°)\s*(?:fov|field\s+of\s+view)|"
                      rf"fov\s*(?:of)?\s*{_NUM}\s*(?:degree|deg|°)?", re.IGNORECASE)
_RE_RESOLUTION = re.compile(rf"{_NUM}\s*[x×]\s*{_NUM}", re.IGNORECASE)
_RE_RANGE = re.compile(rf"{_NUM}\s*(?:m|meter|meters|metre|metres)\s*range|"
                        rf"range\s*(?:of)?\s*{_NUM}\s*(?:m|meter|meters|metre|metres)", re.IGNORECASE)

# Char window scanned after a trigger verb for the sensor list, and
# before it to resolve the subject entity.
_FORWARD_WINDOW = 120
_BACKWARD_WINDOW = 80


class SensorMentionParser:
    """Detects sensor mentions and attaches SensorSpec objects to the
    entities that already exist in `entities`.

    Public API mirrors SpatialRelationshipParser:
        apply(description, entities)          — mutate in place
        parse_mentions(description, entities) — inspect without mutating
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    # ── public API ───────────────────────────────────────────────────

    def apply(self, description: str, entities: list[Entity]) -> None:
        """Detect sensor mentions in `description` and append SensorSpec
        objects to the owning entity's `.sensors` list in place.

        Entities are matched against `entities` exactly as already
        extracted — no new Entity is ever created here.
        """
        self.warnings = []
        label_map = self._build_label_map(entities)
        text = description  # keep original casing for extraction windows;
        # matching itself is done case-insensitively via regex flags.

        for trigger_match in _TRIGGER_RE.finditer(text):
            subject_entity = self._resolve_subject(text, trigger_match.start(), label_map)
            if subject_entity is None:
                self.warnings.append(
                    f"Sensor-bearing phrase near {trigger_match.group(0)!r} "
                    f"has no matching entity in the already-extracted entity "
                    f"list; skipping (subject noun may not be in "
                    f"PromptParser's entity vocabulary)."
                )
                continue

            list_start = trigger_match.end()
            list_segment = text[list_start: list_start + _FORWARD_WINDOW]
            # Stop at sentence end so a later, unrelated sentence's
            # content is never swept into this entity's sensor list.
            sentence_end = re.search(r"[.!?]", list_segment)
            if sentence_end:
                list_segment = list_segment[: sentence_end.start()]

            for sensor_type, name, params in self._extract_sensor_mentions(list_segment):
                existing_names = {s.name for s in subject_entity.sensors}
                unique_name = self._dedupe_name(name, existing_names)
                subject_entity.sensors.append(
                    SensorSpec(sensor_type=sensor_type, name=unique_name, params=params)
                )

    def parse_mentions(self, description: str, entities: list[Entity]) -> list[dict]:
        """Return detected mentions as plain dicts without mutating anything."""
        import copy
        entities_copy = copy.deepcopy(entities)
        self.apply(description, entities_copy)
        results = []
        for e in entities_copy:
            for s in e.sensors:
                results.append({"entity_id": e.id, "sensor_type": s.sensor_type, "name": s.name, "params": s.params})
        return results

    # ── subject resolution (same style as SpatialRelationshipParser) ──

    @staticmethod
    def _build_label_map(entities: list[Entity]) -> dict[str, Entity]:
        candidates: list[tuple[str, Entity]] = []
        for entity in entities:
            tokens = [entity.label.lower(), *entity.label.lower().split(), entity.entity_type.lower()]
            tokens.extend(t.lower() for t in entity.tags)
            for tok in tokens:
                tok = tok.strip(".,;:()")
                if tok:
                    candidates.append((tok, entity))
        candidates.sort(key=lambda x: len(x[0]))
        return {tok: ent for tok, ent in candidates}

    def _resolve_subject(self, text: str, trigger_pos: int, label_map: dict[str, Entity]) -> Optional[Entity]:
        window = text[max(0, trigger_pos - _BACKWARD_WINDOW): trigger_pos].lower()
        best_entity: Optional[Entity] = None
        best_pos = -1
        for token, entity in label_map.items():
            idx = window.rfind(token)
            if idx > best_pos:
                best_pos = idx
                best_entity = entity
        return best_entity

    # ── sensor-list extraction ──────────────────────────────────────

    # Lookahead distance after "and" used to decide whether it introduces
    # a NEW sensor mention (list separator) or continues describing the
    # current one (e.g. "90 degree fov and 1920x1080 resolution" — one
    # camera, two params, not two sensors).
    _AND_LOOKAHEAD = 35
    # Words immediately before the matched sensor phrase considered for
    # the descriptive name prefix — deliberately narrow so trailing
    # parameter clauses ("with 90 degree fov") never leak into the name.
    _NAME_CONTEXT_WORDS = 2

    def _split_segment(self, segment: str) -> list[str]:
        """Split a sensor-list segment into per-sensor-mention chunks.

        Always splits on commas. Splits on standalone "and" ONLY when a
        recognized sensor phrase actually appears shortly after it —
        otherwise "and" is treated as joining two descriptive clauses
        about the SAME sensor (e.g. a fov clause and a resolution
        clause), not as introducing a second sensor.
        """
        chunks: list[str] = []
        for comma_part in segment.split(","):
            last = 0
            for m in re.finditer(r"\band\b", comma_part, re.IGNORECASE):
                lookahead = comma_part[m.end(): m.end() + self._AND_LOOKAHEAD].lower()
                if any(phrase in lookahead for phrase, _ in _SENSOR_PHRASES):
                    chunks.append(comma_part[last:m.start()])
                    last = m.end()
            chunks.append(comma_part[last:])
        return chunks

    def _extract_sensor_mentions(self, segment: str) -> list[tuple[str, str, dict]]:
        """Return a list of (sensor_type, name, params) for one sensor-list
        segment (e.g. "a front RGB camera" or "two stereo cameras and GPS").
        """
        chunks = self._split_segment(segment)
        results: list[tuple[str, str, dict]] = []
        type_counts: dict[str, int] = {}

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_lower = chunk.lower()
            matched_phrase = None
            sensor_type = None
            match_start = -1
            for phrase, stype in _SENSOR_PHRASES:
                idx = chunk_lower.find(phrase)
                if idx != -1:
                    matched_phrase, sensor_type, match_start = phrase, stype, idx
                    break
            if sensor_type is None:
                continue  # not a recognized sensor mention

            quantity = 1
            for w in chunk_lower[:match_start].split():
                w_clean = w.strip(".,;:()")
                if w_clean in _QUANTITY_WORDS:
                    quantity = _QUANTITY_WORDS[w_clean]
                    break

            # Name context: only the ~2 words immediately before the
            # matched phrase, not the whole chunk — keeps trailing
            # parameter clauses out of the name.
            preceding = chunk_lower[:match_start].split()[-self._NAME_CONTEXT_WORDS:]
            descriptive = [
                w.strip(".,;:()") for w in preceding
                if w.strip(".,;:()") not in _QUANTITY_WORDS
                and w.strip(".,;:()") not in _REDUNDANT_NAME_WORDS
                and w.strip(".,;:()") not in {"a", "an", "the"}
                and w.strip(".,;:()")
            ]
            base_name = "_".join([*descriptive, sensor_type]) if descriptive else sensor_type

            params = self._extract_params(chunk, sensor_type)

            type_counts.setdefault(sensor_type, 0)
            for _ in range(quantity):
                type_counts[sensor_type] += 1
                name = base_name if quantity == 1 else f"{base_name}_{type_counts[sensor_type]}"
                results.append((sensor_type, name, dict(params)))

        return results

    @staticmethod
    def _extract_params(chunk: str, sensor_type: str) -> dict:
        """Extract ONLY explicitly-stated numeric parameters from `chunk`.
        Never fills in a default — an unmentioned parameter is simply
        absent from the returned dict, per the "never invent unsupported
        parameters" requirement.
        """
        params: dict = {}

        fov_match = _RE_FOV.search(chunk)
        if fov_match:
            value = fov_match.group(1) or fov_match.group(2)
            if value:
                params["fov_deg"] = float(value)

        res_match = _RE_RESOLUTION.search(chunk)
        if res_match and sensor_type in ("camera", "depth_camera", "thermal_camera"):
            params["resolution"] = [int(float(res_match.group(1))), int(float(res_match.group(2)))]

        range_match = _RE_RANGE.search(chunk)
        if range_match and sensor_type in ("lidar", "radar"):
            value = range_match.group(1) or range_match.group(2)
            if value:
                params["range_m"] = float(value)

        return params

    @staticmethod
    def _dedupe_name(name: str, existing_names: set[str]) -> str:
        """Ensure uniqueness within one entity's sensor list — matches the
        (entity_id, name) uniqueness SceneCompiler.SensorBuilder already
        enforces, so a collision here is caught before compilation rather
        than silently deduplicated later.
        """
        if name not in existing_names:
            return name
        i = 2
        while f"{name}_{i}" in existing_names:
            i += 1
        return f"{name}_{i}"
