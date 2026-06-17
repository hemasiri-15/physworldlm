"""
prompt_parser.py
────────────────
Deterministic, rule-based natural-language → WorldSpec converter for PhysWorldLM.

No external LLM APIs are used.  Everything is done through regex extraction,
keyword matching, and a small set of heuristic rules.  The output is a fully
populated WorldSpec that passes models/validator.py and feeds directly into
models/state_engine.py.

Design decisions
────────────────
1. *Single-pass tokenisation* – the raw prompt is lower-cased once and all
   sub-parsers operate on the same normalised string to avoid redundant work.

2. *Sub-parser isolation* – each ``_parse_*`` method returns a plain Python
   object (list, dict, or dataclass).  ``parse()`` orchestrates them and
   assembles the final WorldSpec.

3. *SI enforcement* – all unit conversions (km/h → m/s, mph → m/s, °→ rad,
   ft → m, etc.) are handled inside helper functions.

4. *Conservative defaults* – when a quantity is not mentioned, a physically
   plausible default is used.

5. *Duration heuristics* – if no explicit duration is stated, the parser
   estimates one from scene type:
     • free-fall:     t = √(2h/g) × 1.5
     • projectile:    t = 2 vy₀/g × 1.5
     • spring:        t = max(5 × 2π/√(k/m) + 0.5, explicit)
     • collision:     t = separation / relative_speed  (or 5 s default)
     • vehicle+dist:  t = distance / speed
     • otherwise:     10 s

6. *Spring interaction* – rest_length and amplitude are extracted from the
   prompt (via dedicated helpers) or fall back to sensible defaults.

7. *Entity-local mass* – mass phrases are matched to the nearest entity noun
   rather than using the first match globally.

8. *Multi-entity motion* – velocity is assigned to every entity that is
   mentioned near a speed phrase, not only the first dynamic entity.

9. *Plural + quantity* – "two cars", "3 balls", "multiple people" are parsed
   into separate _RawEntity instances with unique IDs.

Coordinate conventions (inherited from WorldSpec)
──────────────────────────────────────────────────
  x = East / forward, y = Up (gravity = −y), z = North.
  All SI units: m, m/s, m/s², rad, rad/s, N, J, s.

Usage
─────
    from models.prompt_parser import PromptParser

    parser = PromptParser()
    spec   = parser.parse("A ball falls from 100 m.")
    spec.save("output/scene.json")
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from models.world_spec import (
    BoundingBox,
    Entity,
    Environment,
    Interaction,
    PhysicsState,
    SimulationGraph,
    Vec3,
    Wind,
    WorldSpec,
    MATERIAL_DEFAULTS,
    kmh_to_ms,
    mph_to_ms,
    deg_to_rad,
)


# ─────────────────────────────────────────────
# Internal intermediate representations
# ─────────────────────────────────────────────

@dataclass
class _RawEntity:
    """Intermediate entity before WorldSpec.Entity is constructed."""
    id:           str
    label:        str
    entity_type:  str                    # vehicle|projectile|object|structure|terrain
    is_static:    bool          = False
    mass_kg:      Optional[float] = None
    material:     str           = "generic"
    position:     Vec3          = field(default_factory=Vec3)
    velocity:     Vec3          = field(default_factory=Vec3)
    bounding_box: BoundingBox   = field(default_factory=BoundingBox)
    tags:         list[str]     = field(default_factory=list)
    # character offset in normalised text where this entity was found
    text_offset:  int           = 0


@dataclass
class _RawInteraction:
    """Intermediate interaction before WorldSpec.Interaction is constructed."""
    type:       str
    entity_a:   str
    entity_b:   str
    parameters: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Hierarchical entity alias dictionary
# ─────────────────────────────────────────────

# Maps every recognised noun/alias to a canonical entity_type.
# Extend any sub-dict freely — _classify_word() iterates this structure.
ENTITY_ALIASES: dict[str, set[str]] = {
    "vehicle": {
        # cars
        "car", "cars", "automobile", "automobiles", "vehicle", "vehicles",
        "sedan", "sedans", "suv", "suvs", "coupe", "coupes", "hatchback",
        "hatchbacks", "convertible", "convertibles", "minivan", "minivans",
        "pickup", "pickups", "sportscar", "sportscars",
        # vans / trucks
        "van", "vans", "truck", "trucks", "lorry", "lorries", "semi",
        "semis", "trailer", "trailers", "bus", "buses", "minibus",
        # motorcycles / bikes
        "motorcycle", "motorcycles", "motorbike", "motorbikes",
        "bike", "bikes", "bicycle", "bicycles", "moped", "mopeds",
        "scooter", "scooters", "trike", "trikes",
        # rail
        "train", "trains", "tram", "trams", "subway", "metro",
        "locomotive", "locomotives", "railcar", "railcars",
        # aircraft
        "helicopter", "helicopters", "plane", "planes", "aircraft",
        "aeroplane", "aeroplanes", "airplane", "airplanes",
        "drone", "drones", "glider", "gliders",
        # watercraft
        "boat", "boats", "ship", "ships", "canoe", "canoes",
        "kayak", "kayaks", "yacht", "yachts", "ferry", "ferries",
        "raft", "rafts", "vessel", "vessels",
        # brand names / model names (common)
        "ferrari", "lamborghini", "tesla", "ford", "honda",
        "toyota", "bmw", "audi", "porsche",
    },
    "projectile": {
        "projectile", "projectiles",
        "bullet", "bullets", "shell", "shells",
        "missile", "missiles", "rocket", "rockets",
        "arrow", "arrows", "dart", "darts",
        "cannonball", "cannonballs", "grenade", "grenades",
        "stone", "stones", "pebble", "pebbles",
        "javelin", "javelins", "discus",
    },
    "agent": {
        "person", "persons", "people",
        "man", "men", "woman", "women",
        "human", "humans", "humanoid", "humanoids",
        "athlete", "athletes", "runner", "runners",
        "pedestrian", "pedestrians", "cyclist", "cyclists",
        "player", "players", "worker", "workers",
        "child", "children", "kid", "kids",
        "soldier", "soldiers", "skier", "skiers",
        "jumper", "jumpers", "diver", "divers",
    },
    "fluid": {
        "water", "fluid", "fluids", "liquid", "liquids",
        "oil", "oils", "river", "rivers", "stream", "streams",
        "lake", "lakes", "ocean", "oceans", "sea", "seas",
        "pool", "pools", "puddle", "puddles",
        "gas", "gases", "air", "steam",
    },
    "structure": {
        "wall", "walls", "building", "buildings",
        "bridge", "bridges", "tower", "towers",
        "anchor", "anchors", "post", "posts",
        "pillar", "pillars", "column", "columns",
        "beam", "beams", "barrier", "barriers",
        "fence", "fences", "gate", "gates",
        "dam", "dams", "staircase", "staircases",
        "stair", "stairs", "step", "steps",
        "platform", "platforms", "scaffold", "scaffolding",
        "pole", "poles", "mast", "masts",
        "slab", "slabs", "panel", "panels",
    },
    "terrain": {
        "ground", "road", "roads", "surface", "surfaces",
        "floor", "floors", "terrain", "terrains",
        "hill", "hills", "slope", "slopes",
        "mountain", "mountains", "ramp", "ramps",
        "cliff", "cliffs", "valley", "valleys",
        "field", "fields", "plain", "plains",
        "track", "tracks", "path", "paths",
        "runway", "runways",
    },
    "object": {
        "ball", "balls", "sphere", "spheres",
        "cube", "cubes", "block", "blocks",
        "box", "boxes", "rock", "rocks",
        "mass", "masses", "weight", "weights",
        "object", "objects", "particle", "particles",
        "body", "bodies", "disc", "discs", "disk", "disks",
        "cylinder", "cylinders", "ring", "rings",
        "crate", "crates", "barrel", "barrels",
        "coin", "coins", "marble", "marbles",
        "brick", "bricks", "tile", "tiles",
        "plank", "planks", "rod", "rods",
        "sphere", "spheres", "egg", "eggs",
    },
}

# Build a flat reverse-lookup: token → entity_type (populated at module load)
_TOKEN_TO_TYPE: dict[str, str] = {}
for _etype, _aliases in ENTITY_ALIASES.items():
    for _alias in _aliases:
        _TOKEN_TO_TYPE[_alias] = _etype

# Quantity words that indicate multiple instances of the following entity
_QUANTITY_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "pair": 2, "couple": 2, "multiple": 2, "several": 3,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
}

# ─────────────────────────────────────────────
# Material mappings
# ─────────────────────────────────────────────

_MATERIAL_WORDS = {
    "steel":    {"steel", "metal", "iron"},
    "rubber":   {"rubber", "bouncy"},
    "wood":     {"wood", "wooden"},
    "concrete": {"concrete", "cement"},
    "water":    {"water"},
    "glass":    {"glass"},
    "flesh":    {"flesh", "person", "human", "man", "woman", "body"},
    "plastic":  {"plastic"},
}

# Typical bounding boxes per entity type (width, height, depth in metres)
_DEFAULT_BBOX = {
    "vehicle":    BoundingBox(4.5, 1.5, 1.8),
    "projectile": BoundingBox(0.1, 0.1, 0.1),
    "agent":      BoundingBox(0.5, 1.8, 0.3),
    "fluid":      BoundingBox(5.0, 1.0, 5.0),
    "structure":  BoundingBox(0.5, 2.0, 0.5),
    "terrain":    BoundingBox(100.0, 0.1, 100.0),
    "object":     BoundingBox(0.2, 0.2, 0.2),
}

# Default masses per entity type (kg)
_DEFAULT_MASS = {
    "vehicle":    1200.0,
    "projectile": 0.5,
    "agent":      70.0,
    "fluid":      1000.0,
    "structure":  500.0,
    "terrain":    0.0,
    "object":     1.0,
}

# Suffix heuristics for _classify_unknown_entity
_SUFFIX_TO_TYPE: list[tuple[str, str]] = [
    ("craft",  "vehicle"),
    ("cycle",  "vehicle"),
    ("mobile", "vehicle"),
    ("plane",  "vehicle"),
    ("copter", "vehicle"),
    ("ship",   "vehicle"),
    ("boat",   "vehicle"),
    ("rail",   "vehicle"),
    ("ject",   "projectile"),   # "object" is caught earlier; catches custom types
    ("tile",   "projectile"),   # mis-stem of "missile" variants
    ("ball",   "object"),
    ("block",  "object"),
    ("cube",   "object"),
    ("sphere", "object"),
    ("wall",   "structure"),
    ("tower",  "structure"),
    ("post",   "structure"),
    ("man",    "agent"),
    ("woman",  "agent"),
    ("person", "agent"),
    ("child",  "agent"),
    ("worker", "agent"),
    ("ground", "terrain"),
    ("road",   "terrain"),
    ("floor",  "terrain"),
    ("water",  "fluid"),
    ("lake",   "fluid"),
    ("river",  "fluid"),
]


# ─────────────────────────────────────────────
# Regex patterns (compiled once at module load)
# ─────────────────────────────────────────────

_NUM = r"[-+]?\d+(?:\.\d+)?"

_RE_SPEED = re.compile(
    rf"({_NUM})\s*(m/s|km/h|kmh|kph|mph|ft/s|fps|ms\b)",
    re.IGNORECASE,
)

_RE_HEIGHT = re.compile(
    rf"({_NUM})\s*(m\b|meters?\b|metres?\b|km\b|ft\b|feet\b|foot\b|cm\b)",
    re.IGNORECASE,
)

_RE_DISTANCE = re.compile(
    rf"(?:distance|travels?|moves?|covers?)\s+(?:of\s+)?({_NUM})\s*"
    rf"(m\b|meters?\b|metres?\b|km\b|ft\b|feet\b|cm\b)",
    re.IGNORECASE,
)

_RE_MASS = re.compile(
    rf"({_NUM})\s*(kg\b|g\b|grams?\b|kilograms?\b|pounds?\b|lbs?\b|tonnes?\b|tons?\b)",
    re.IGNORECASE,
)

_RE_ANGLE = re.compile(
    rf"({_NUM})\s*(?:degrees?|°|deg\b)",
    re.IGNORECASE,
)

_RE_DURATION = re.compile(
    rf"(?:for|over|during|in|after)?\s*({_NUM})\s*(seconds?|secs?|s\b|minutes?|mins?)",
    re.IGNORECASE,
)

_RE_SPRING_K = re.compile(
    rf"(?:spring\s+(?:constant|stiffness|coefficient)|k\s*[=:]\s*)({_NUM})\s*(?:N/m|n/m)?",
    re.IGNORECASE,
)

_RE_REST_LENGTH = re.compile(
    rf"rest\s+(?:length|len)\s+(?:of\s+)?({_NUM})\s*(m\b|meters?\b|metres?\b|cm\b|ft\b|feet\b)?",
    re.IGNORECASE,
)

_RE_AMPLITUDE = re.compile(
    rf"amplitude\s+(?:of\s+)?({_NUM})\s*(m\b|meters?\b|metres?\b|cm\b|ft\b|feet\b)?",
    re.IGNORECASE,
)

_RE_WET   = re.compile(r"\bwet\b",   re.IGNORECASE)
_RE_ICY   = re.compile(r"\bicy?\b",  re.IGNORECASE)
_RE_ROUGH = re.compile(r"\brough\b", re.IGNORECASE)

_RE_WIND = re.compile(
    rf"wind(?:\s+(?:at|of|speed))?\s+({_NUM})\s*(m/s|km/h|mph)",
    re.IGNORECASE,
)

# Direction keywords for velocity sign
_DIRECTION_SIGN: dict[str, float] = {
    "east": 1.0, "forward": 1.0, "right": 1.0,
    "west": -1.0, "backward": -1.0, "back": -1.0, "left": -1.0,
}

# Motion verbs used to pair speed with entity
_MOTION_VERBS = re.compile(
    r"\b(?:moves?\s+(?:at|forward|east|west)?|moving\s+(?:at|forward|east|west)?|"
    r"travels?\s+(?:at)?|travelling\s+(?:at)?|traveling\s+(?:at)?|"
    r"drives?\s+(?:at)?|slides?\s+(?:at)?|rolls?\s+(?:at)?|"
    r"launched?\s+(?:at)?|fires?\s+(?:at)?|thrown?\s+(?:at)?|"
    r"going\s+(?:at)?|speed(?:ing)?\s+(?:at|of)?|"
    r"velocity\s+(?:of)?)\s*" + rf"({_NUM})\s*(m/s|km/h|kmh|kph|mph|ft/s|fps)",
    re.IGNORECASE,
)

# Per-entity speed pattern: "<label> moving at <speed>" or "<speed> <label>"
_RE_ENTITY_SPEED = re.compile(
    rf"(\w+)\s+(?:moving|travelling|traveling|going|drives?|slides?)?\s*"
    rf"(?:at|with\s+(?:a\s+)?(?:speed|velocity)\s+of)?\s*({_NUM})\s*(m/s|km/h|kmh|kph|mph|ft/s|fps)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# Unit-conversion helpers
# ─────────────────────────────────────────────

def _to_ms(value: float, unit: str) -> float:
    unit = unit.lower().strip().rstrip(".")
    if unit in ("m/s", "ms", "metres/s", "meters/s", "mps"):
        return value
    if unit in ("km/h", "kmh", "kph", "km/hr"):
        return kmh_to_ms(value)
    if unit in ("mph", "mi/h", "miles/h", "miles/hr"):
        return mph_to_ms(value)
    if unit in ("ft/s", "fps", "feet/s"):
        return value * 0.3048
    return value


def _to_metres(value: float, unit: str) -> float:
    unit = unit.lower().strip().rstrip(".")
    if unit in ("m", "metre", "metres", "meter", "meters"):
        return value
    if unit in ("km", "kilometre", "kilometres", "kilometer", "kilometers"):
        return value * 1000.0
    if unit in ("ft", "foot", "feet"):
        return value * 0.3048
    if unit in ("cm", "centimetre", "centimetres", "centimeter", "centimeters"):
        return value / 100.0
    if unit in ("mm", "millimetre", "millimetres", "millimeter", "millimeters"):
        return value / 1000.0
    return value


def _to_kg(value: float, unit: str) -> float:
    unit = unit.lower().strip().rstrip(".")
    if unit in ("kg", "kilogram", "kilograms"):
        return value
    if unit in ("g", "gram", "grams"):
        return value / 1000.0
    if unit in ("lb", "lbs", "pound", "pounds"):
        return value * 0.453592
    if unit in ("t", "tonne", "tonnes", "ton", "tons"):
        return value * 1000.0
    return value


# ─────────────────────────────────────────────
# PromptParser
# ─────────────────────────────────────────────

class PromptParser:
    """
    Convert a plain-English scene description into a :class:`~models.world_spec.WorldSpec`.

    This is a deterministic, rule-based parser — no LLM calls are made.

    Parameters
    ----------
    default_dt : float
        Integration timestep forwarded to ``SimulationGraph``.  Default 0.01 s.
    export_fps : int
        Frames per second for the exported trajectory.  Default 30.
    verbose : bool
        When ``True``, print a short parse log to stdout.
    """

    G: float = 9.81

    def __init__(
        self,
        default_dt:  float = 0.01,
        export_fps:  int   = 30,
        verbose:     bool  = False,
    ) -> None:
        self.default_dt  = default_dt
        self.export_fps  = export_fps
        self.verbose     = verbose

    # ── public API ────────────────────────────────────────────────────────────

    def parse(self, prompt: str, scene_id: str = None) -> WorldSpec:
        """
        Parse a natural-language scene description into a :class:`WorldSpec`.

        Raises
        ------
        ValueError
            If no physical entity can be detected in the prompt.
        """
        scene_id = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        text = self._normalise(prompt)
        self._warnings = []

        self._log(f"parsing scene_id={scene_id}")
        self._log(f"  normalised: {text[:80]!r}{'…' if len(text)>80 else ''}")

        raw_entities    = self._parse_entities(text, prompt)
        if not raw_entities:
            raw_entities = [
                _RawEntity(
                    id=f"e_object_{uuid.uuid4().hex[:4]}",
                    label="object",
                    entity_type="object",
                    is_static=False,
                    mass_kg=1.0,
                    material="generic",
                    bounding_box=_DEFAULT_BBOX["object"],
                    tags=["object"],
                )
            ]

        self._parse_motion(text, raw_entities)
        environment      = self._parse_environment(text)
        duration         = self._parse_duration(text, raw_entities, environment)
        raw_interactions = self._parse_interactions(text, raw_entities)

        self._log(
            f"  entities={len(raw_entities)}  "
            f"duration={duration:.2f}s  "
            f"interactions={len(raw_interactions)}"
        )

        return self._assemble(
            scene_id, prompt, raw_entities, environment,
            raw_interactions, duration,
        )

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    # ── entity classification ─────────────────────────────────────────────────

    @staticmethod
    def _classify_word(word: str) -> Optional[str]:
        """
        Return entity_type for a noun token.

        Lookup order:
          1. Exact match in flat reverse-lookup table (_TOKEN_TO_TYPE).
          2. Singularisation heuristic (strip trailing 's' / 'es').
          3. Suffix heuristics (_classify_unknown_entity).
        """
        w = word.lower().strip(".,;:!?()")
        if w in _TOKEN_TO_TYPE:
            return _TOKEN_TO_TYPE[w]
        # Try singularisation
        for suffix, replacement in [("ies", "y"), ("ves", "f"), ("ses", "s"),
                                     ("xes", "x"), ("oes", "o"), ("es", ""),
                                     ("s", "")]:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                singular = w[:-len(suffix)] + replacement
                if singular in _TOKEN_TO_TYPE:
                    return _TOKEN_TO_TYPE[singular]
        return PromptParser._classify_unknown_entity(w)

    @staticmethod
    def _classify_unknown_entity(token: str) -> Optional[str]:
        """
        Fallback heuristic classification for unknown entity tokens.

        Strategies (in priority order):
          1. Suffix matching against ``_SUFFIX_TO_TYPE``.
          2. Substring matching for compound nouns
             (e.g. ``"racingcar"`` → vehicle).
          3. Returns ``None`` if none of the above matches.
        """
        t = token.lower()
        for suffix, etype in _SUFFIX_TO_TYPE:
            if t.endswith(suffix) and len(t) > len(suffix):
                return etype
        # Substring match for compound / brand tokens
        for alias, etype in _TOKEN_TO_TYPE.items():
            if len(alias) >= 4 and alias in t:
                return etype
        return None

    # ── entity detection ──────────────────────────────────────────────────────

    def _parse_entities(self, text: str, original: str) -> list[_RawEntity]:
        """
        Detect physical entities by scanning for known noun keywords,
        honouring quantity words and deduplication rules.

        Plural nouns and quantity words ("two cars", "3 balls") produce
        multiple _RawEntity instances with unique IDs.
        """
        entities: list[_RawEntity] = []
        # Track (entity_type, label) pairs that have been seen.
        # Deduplication: same type+label only if no quantity word present.
        seen_labels: dict[str, int] = {}  # label → count so far

        words = text.split()

        # Build character-offset index for mass localisation
        offset = 0
        word_offsets: list[int] = []
        for w in words:
            word_offsets.append(offset)
            offset += len(w) + 1

        i = 0
        while i < len(words):
            word = words[i].strip(".,;:!?()")

            # Check for quantity word before the entity noun
            quantity = 1
            if word in _QUANTITY_WORDS:
                quantity = _QUANTITY_WORDS[word]
                # Peek ahead (skip adjectives) for the entity noun
                j = i + 1
                while j < len(words):
                    candidate = words[j].strip(".,;:!?()")
                    # Skip colour/size adjectives
                    if self._classify_word(candidate) is None and not candidate.isdigit():
                        j += 1
                        continue
                    break
                if j < len(words):
                    entity_word = words[j].strip(".,;:!?()")
                    entity_type = self._classify_word(entity_word)
                    if entity_type is not None:
                        material = self._infer_material(words, j)
                        label    = entity_word   # preserve original noun
                        for _ in range(quantity):
                            eid = f"e_{entity_word}_{uuid.uuid4().hex[:4]}"
                            mass = self._infer_entity_mass(text, label, word_offsets[j])
                            is_static = entity_type in ("terrain", "structure") \
                                        and not self._is_moving(text, entity_word)
                            raw = _RawEntity(
                                id=          eid,
                                label=       label,
                                entity_type= entity_type,
                                is_static=   is_static,
                                mass_kg=     mass,
                                material=    material,
                                bounding_box=_DEFAULT_BBOX.get(entity_type, BoundingBox()),
                                tags=        [entity_word, entity_type],
                                text_offset= word_offsets[j],
                            )
                            entities.append(raw)
                        i = j + 1
                        continue
            # No quantity prefix – check current word as entity noun
            entity_type = self._classify_word(word)
            if entity_type is not None:
                material = self._infer_material(words, i)
                label    = self._build_label(words, i, word)

                # Deduplication: allow repeats only for truly distinct phrases
                seen_key = f"{entity_type}:{label}"
                if seen_key in seen_labels:
                    # Only suppress if there is no quantity word anywhere nearby
                    # that might explain the repeated mention
                    i += 1
                    continue
                seen_labels[seen_key] = 1

                is_static = entity_type in ("terrain", "structure") \
                            and not self._is_moving(text, word)
                mass = self._infer_entity_mass(text, label, word_offsets[i])

                eid = f"e_{word}_{uuid.uuid4().hex[:4]}"
                raw = _RawEntity(
                    id=          eid,
                    label=       label,
                    entity_type= entity_type,
                    is_static=   is_static,
                    mass_kg=     mass,
                    material=    material,
                    bounding_box=_DEFAULT_BBOX.get(entity_type, BoundingBox()),
                    tags=        [word, entity_type],
                    text_offset= word_offsets[i],
                )
                entities.append(raw)

            i += 1

        # ── Spring scene: ensure a static anchor exists ────────────────────
        if "spring" in text and not any(e.entity_type == "structure" for e in entities):
            anchor = _RawEntity(
                id="e_anchor_0000",
                label="spring anchor",
                entity_type="structure",
                is_static=True,
                mass_kg=1.0,
                material="steel",
                position=Vec3(0.0, 0.1, 0.0),
                bounding_box=BoundingBox(0.1, 0.2, 0.1),
                tags=["anchor", "spring"],
                text_offset=0,
            )
            entities.insert(0, anchor)

        # ── Implicit ground ───────────────────────────────────────────────
        has_static = any(e.is_static for e in entities)
        if not has_static:
            ground = _RawEntity(
                id="e_ground_0000",
                label="ground",
                entity_type="terrain",
                is_static=True,
                mass_kg=0.0,
                material="concrete",
                position=Vec3(0.0, 0.0, 0.0),
                bounding_box=BoundingBox(1000.0, 0.01, 1000.0),
                tags=["ground", "terrain"],
                text_offset=len(text),
            )
            entities.append(ground)

        return entities

    @staticmethod
    def _infer_material(words: list[str], idx: int) -> str:
        window = words[max(0, idx-3): idx+3]
        for mat, synonyms in _MATERIAL_WORDS.items():
            if any(w.strip(".,;:") in synonyms for w in window):
                return mat
        return "generic"

    @staticmethod
    def _build_label(words: list[str], idx: int, noun: str) -> str:
        colour_adj = {
            "red", "blue", "green", "black", "white", "yellow", "orange",
            "grey", "gray", "silver", "dark", "light", "small", "large",
            "heavy", "big", "tiny",
        }
        if idx > 0:
            prev = words[idx-1].strip(".,;:()")
            if prev in colour_adj:
                return f"{prev} {noun}"
        return noun

    @staticmethod
    def _is_moving(text: str, noun: str) -> bool:
        motion_verbs = {"moves", "moving", "travels", "travelling", "slides",
                        "rolls", "flies", "launched", "thrown", "falls",
                        "drops", "oscillates", "drives", "runs", "speeds"}
        words = text.split()
        for i, w in enumerate(words):
            if w.strip(".,;:()") == noun:
                window = words[max(0, i-5): i+6]
                if any(v in motion_verbs for v in window):
                    return True
        return False

    # ── Entity-local mass extraction ───────────────────────────────────────────

    def _infer_entity_mass(
        self,
        text: str,
        label: str,
        entity_offset: int,
        entity_type: str = "object",
    ) -> Optional[float]:
        """
        Extract the mass most likely associated with *this* entity.

        Strategy:
          1. Find all mass matches in the text.
          2. For each match, compute distance to ``entity_offset``.
          3. Return the mass of the nearest match within a 60-character window.
          4. Fall back to type-based default.

        Parameters
        ----------
        text : str
            Normalised prompt.
        label : str
            Entity label (used to find entity_type default).
        entity_offset : int
            Character offset in ``text`` where the entity noun appears.
        entity_type : str
            Entity type for default mass lookup.
        """
        best_mass: Optional[float] = None
        best_dist = float("inf")

        for m in _RE_MASS.finditer(text):
            try:
                mass_val = _to_kg(float(m.group(1)), m.group(2))
            except ValueError:
                continue
            dist = abs(m.start() - entity_offset)
            if dist < best_dist and dist < 60:
                best_mass = mass_val
                best_dist = dist

        return best_mass  # None → caller uses _DEFAULT_MASS lookup

    def _infer_mass(self, text: str, entity_type: str) -> float:
        """Legacy single-entity mass extraction; returns default if no match."""
        m = _RE_MASS.search(text)
        if m:
            try:
                return _to_kg(float(m.group(1)), m.group(2))
            except ValueError:
                pass
        return _DEFAULT_MASS.get(entity_type, 1.0)

    # ── Spring helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_spring_rest_length(text: str, default: float = 1.0) -> float:
        """
        Extract spring rest/natural length from the prompt.

        Supports patterns like:
          • "rest length 2 m"
          • "rest length of 0.5 meters"
          • "natural length 3 m"

        Returns ``default`` when no pattern matches.
        """
        m = _RE_REST_LENGTH.search(text)
        if m:
            try:
                val  = float(m.group(1))
                unit = m.group(2) or "m"
                return _to_metres(val, unit)
            except ValueError:
                pass
        # Also try "natural length"
        nl = re.search(
            rf"natural\s+(?:length|len)\s+(?:of\s+)?({_NUM})\s*(m\b|meters?\b|metres?\b|cm\b|ft\b|feet\b)?",
            text, re.IGNORECASE,
        )
        if nl:
            try:
                val  = float(nl.group(1))
                unit = nl.group(2) or "m"
                return _to_metres(val, unit)
            except ValueError:
                pass
        return default

    @staticmethod
    def _extract_spring_amplitude(text: str, default: float = 0.2) -> float:
        """
        Extract spring displacement amplitude from the prompt.

        Supports patterns like:
          • "amplitude 0.5 m"
          • "amplitude of 0.3 meters"
          • "displaced by 0.2 m"

        Returns ``default`` when no pattern matches.
        """
        m = _RE_AMPLITUDE.search(text)
        if m:
            try:
                val  = float(m.group(1))
                unit = m.group(2) or "m"
                return _to_metres(val, unit)
            except ValueError:
                pass
        disp = re.search(
            rf"displaced?\s+(?:by\s+)?({_NUM})\s*(m\b|meters?\b|metres?\b|cm\b|ft\b|feet\b)?",
            text, re.IGNORECASE,
        )
        if disp:
            try:
                val  = float(disp.group(1))
                unit = disp.group(2) or "m"
                return _to_metres(val, unit)
            except ValueError:
                pass
        return default

    # ── motion (position + velocity) ──────────────────────────────────────────

    def _parse_motion(self, text: str, entities: list[_RawEntity]) -> None:
        """
        Attach initial positions and velocities to entities in-place.

        Multi-entity motion:
          • Each dynamic entity is checked for a speed phrase in its
            vicinity (±80 chars).  This handles "A car and a person move at
            10 m/s" as well as "A car at 20 m/s hits a stationary wall".
          • Direction keywords (east/west/forward/backward) flip the sign of
            the x-velocity.
          • For free-fall, projectile, and spring scenes the primary entity
            gets special treatment; secondary entities keep their detected
            velocities.
        """
        # ── global scene flags ────────────────────────────────────────────
        is_fall       = bool(re.search(r"\b(?:falls?|drops?|released?)\b", text))
        is_projectile = bool(re.search(r"\b(?:projectile|launched?|thrown?|fired?)\b", text))
        is_spring     = "spring" in text
        is_forward    = bool(re.search(
            r"\b(?:moves?|moving|travels?|drives?|slides?|rolls?)\b", text
        ))

        # ── Extract global speed, height, angle once ──────────────────────
        height: Optional[float] = None
        fall_patterns = [
            re.compile(
                rf"(?:falls?\s+from|dropped?\s+from|released?\s+from|"
                rf"at\s+(?:a\s+)?height\s+of|from\s+(?:a\s+)?height\s+of|"
                rf"at\s+altitude|from)\s+({_NUM})\s*(m\b|meters?\b|metres?\b|ft\b|feet\b|km\b)",
                re.IGNORECASE,
            ),
        ]
        for pat in fall_patterns:
            hm = pat.search(text)
            if hm:
                try:
                    height = _to_metres(float(hm.group(1)), hm.group(2))
                except ValueError:
                    pass
                break
        # Also try bare height: "100 m" near "fall"
        if height is None and is_fall:
            hm = _RE_HEIGHT.search(text)
            if hm:
                try:
                    height = _to_metres(float(hm.group(1)), hm.group(2))
                except ValueError:
                    pass

        global_speed: Optional[float] = None
        sm = _RE_SPEED.search(text)
        if sm:
            try:
                global_speed = _to_ms(float(sm.group(1)), sm.group(2))
            except ValueError:
                pass

        angle_rad: Optional[float] = None
        am = _RE_ANGLE.search(text)
        if am:
            try:
                angle_rad = deg_to_rad(float(am.group(1)))
            except ValueError:
                pass

        rest_length = self._extract_spring_rest_length(text)
        amplitude   = self._extract_spring_amplitude(text)

        dynamic = [e for e in entities if not e.is_static]
        if not dynamic:
            return

        primary = dynamic[0]

        # ── Free fall ──────────────────────────────────────────────────────
        if is_fall and height is not None:
            half_h = primary.bounding_box.height / 2.0
            primary.position = Vec3(0.0, height + half_h, 0.0)
            primary.velocity = Vec3(0.0, 0.0, 0.0)
            return

        # ── Projectile ────────────────────────────────────────────────────
        if is_projectile:
            if global_speed is None:

                self._warnings.append(
                    "projectile speed not specified; using default"
                )
            spd = global_speed or 20.0

            theta = angle_rad if angle_rad is not None else deg_to_rad(45.0)
            vx    = spd * math.cos(theta)
            vy    = spd * math.sin(theta)
            half_h = primary.bounding_box.height / 2.0
            primary.position = Vec3(0.0, half_h, 0.0)
            primary.velocity = Vec3(vx, vy, 0.0)
            return

        # ── Spring oscillator ─────────────────────────────────────────────
        if is_spring:
            half_h = primary.bounding_box.height / 2.0
            primary.position = Vec3(rest_length + amplitude, half_h, 0.0)
            primary.velocity = Vec3(0.0, 0.0, 0.0)
            return

        # ── Multi-entity velocity assignment ──────────────────────────────
        # For each dynamic entity look for an associated speed near it in the text.
        for entity in dynamic:
            entity_speed  = self._find_entity_speed(text, entity)
            if entity_speed is not None:
                half_h = entity.bounding_box.height / 2.0
                entity.position = Vec3(0.0, half_h, 0.0)
                entity.velocity = Vec3(entity_speed, 0.0, 0.0)
            elif is_forward and global_speed is not None:
                # Assign global speed to all dynamic entities that have no
                # dedicated speed unless marked "stationary"
                if not self._is_stationary_in_context(text, entity.label):
                    half_h = entity.bounding_box.height / 2.0
                    entity.position = Vec3(0.0, half_h, 0.0)
                    entity.velocity = Vec3(global_speed, 0.0, 0.0)
            else:
                half_h = entity.bounding_box.height / 2.0
                entity.position = Vec3(0.0, half_h, 0.0)
                entity.velocity = Vec3(0.0, 0.0, 0.0)

    def _find_entity_speed(self, text: str, entity: _RawEntity) -> Optional[float]:
        """
        Search for a speed phrase within ±100 characters of the entity label
        in the text.  Returns speed in m/s, or None if not found.

        Also applies directional sign:
          "moving west at 10 m/s" → -10 m/s
        """
        label = entity.label.lower()
        # Find occurrence of label in text
        match = re.search(re.escape(label), text)
        if match is None:
            return None
        centre = match.start()
        window = text[max(0, centre - 100): centre + 150]

        # Check for "stationary" / "at rest" near this entity
        #if re.search(r"\b(?:stationary|at\s+rest|standing\s+still|stopped)\b", window):
        #    return 0.0  # explicitly zero

        sm = _RE_SPEED.search(window)
        if sm is None:
            return None
        try:
            speed = _to_ms(float(sm.group(1)), sm.group(2))
        except ValueError:
            return None

        # Apply direction sign
        for direction, sign in _DIRECTION_SIGN.items():
            if re.search(rf"\b{direction}\b", window, re.IGNORECASE):
                speed *= sign
                break

        return speed

    @staticmethod
    def _is_stationary_in_context(text: str, label: str) -> bool:
        """Return True if the entity is described as stationary."""
        label = label.lower()
        m = re.search(re.escape(label), text)
        if m is None:
            return False
        window = text[max(0, m.start() - 60): m.start() + 80]
        return bool(re.search(
            r"\b(?:stationary|at\s+rest|standing\s+still|stopped|parked)\b",
            window,
        ))

    # ── environment ───────────────────────────────────────────────────────────

    def _parse_environment(self, text: str) -> Environment:
        friction = 0.5
        if _RE_ICY.search(text):
            friction = 0.1
        elif _RE_WET.search(text):
            friction = 0.3
        elif _RE_ROUGH.search(text):
            friction = 0.8

        wind_speed = 0.0
        wm = _RE_WIND.search(text)
        if wm:
            try:
                wind_speed = _to_ms(float(wm.group(1)), wm.group(2))
            except ValueError:
                pass

        air_density = 1.225
        if re.search(r"\b(?:vacuum|no\s+air|airless)\b", text):
            air_density = 0.0

        weather = "clear"
        if re.search(r"\brain\b",  text): weather = "rain"
        elif re.search(r"\bsnow\b", text): weather = "snow"
        elif re.search(r"\bfog\b",  text): weather = "fog"
        elif wind_speed > 0:              weather = "wind"

        time_of_day = "day"
        if re.search(r"\bnight\b", text):  time_of_day = "night"
        elif re.search(r"\bdawn\b",  text): time_of_day = "dawn"
        elif re.search(r"\bdusk\b",  text): time_of_day = "dusk"

        return Environment(
            gravity=         Vec3(0.0, -self.G, 0.0),
            temperature_K=   293.15,
            pressure_Pa=     101325.0,
            air_density=     air_density,
            wind=            Wind(speed=wind_speed, direction=0.0),
            terrain_type=    "flat",
            friction_global= friction,
            time_of_day=     time_of_day,
            weather=         weather,
        )

    # ── duration ──────────────────────────────────────────────────────────────

    def _parse_duration(
        self,
        text: str,
        entities: list[_RawEntity],
        env: Environment,
    ) -> float:
        """
        Determine simulation duration with improved heuristics.

        Priority:
          1. Explicit duration phrase.
          2. Scene-specific heuristic:
             • free-fall
             • projectile
             • spring (oscillation-based)
             • collision (separation / relative speed)
             • vehicle + distance (distance / speed)
          3. Default 10 s.
        """
        # 1. Explicit
        dm = _RE_DURATION.search(text)
        if dm:
            try:
                val  = float(dm.group(1))
                unit = dm.group(2).lower()
                if unit.startswith("min"):
                    val *= 60.0
                if val > 0:
                    return val
            except ValueError:
                pass

        g = abs(env.gravity.y) or self.G
        dynamic = [e for e in entities if not e.is_static]

        # 2a. Free fall
        if re.search(r"\b(?:falls?|drops?|released?)\b", text):
            hm = _RE_HEIGHT.search(text)
            if hm:
                try:
                    h = _to_metres(float(hm.group(1)), hm.group(2))
                    return math.sqrt(2 * h / g) * 1.5
                except ValueError:
                    pass
            return 5.0

        # 2b. Projectile
        if re.search(r"\b(?:projectile|launched?|thrown?|fired?)\b", text):
            if dynamic:
                vy = dynamic[0].velocity.y
                if vy > 0:
                    return (2 * vy / g) * 1.5
            sm = _RE_SPEED.search(text)
            if sm:
                try:
                    spd = _to_ms(float(sm.group(1)), sm.group(2))
                    am  = _RE_ANGLE.search(text)
                    theta = deg_to_rad(float(am.group(1))) if am else deg_to_rad(45.0)
                    vy = spd * math.sin(theta)
                    if vy > 0:
                        return (2 * vy / g) * 1.5
                except ValueError:
                    pass
            return 5.0

        # 2c. Spring oscillation
        if "spring" in text:
            km_m = _RE_SPRING_K.search(text)

            if km_m:
                k = float(km_m.group(1))
            else:
                k = 100.0
                self._warnings.append(
                    "spring stiffness not specified; using default"
                )

            mass = dynamic[0].mass_kg if dynamic and dynamic[0].mass_kg else 1.0
            omega  = math.sqrt(k / mass)
            period = 2.0 * math.pi / omega
            return max(5.0 * period + 0.5, period)

        # 2d. Collision scene — estimate from speed and some separation
        if re.search(r"\b(?:collide|collision|crash|impact|hits?)\b", text):
            speeds: list[float] = []
            for m in _RE_SPEED.finditer(text):
                try:
                    speeds.append(abs(_to_ms(float(m.group(1)), m.group(2))))
                except ValueError:
                    pass
            if speeds:
                relative_speed = sum(speeds)
                # Assume entities start ~50 m apart by default
                separation = 50.0
                return min(max(separation / relative_speed, 2.0), 30.0)
            return 5.0

        # 2e. Vehicle + distance
        if re.search(r"\b(?:travels?|drives?|moves?|covers?)\b", text):
            hm = _RE_HEIGHT.search(text)
            sm = _RE_SPEED.search(text)
            if hm and sm:
                try:
                    dist  = _to_metres(float(hm.group(1)), hm.group(2))
                    speed = _to_ms(float(sm.group(1)), sm.group(2))
                    if speed > 0:
                        return dist / speed * 1.1
                except ValueError:
                    pass

        # 3. Default
        return 10.0

    # ── interactions ──────────────────────────────────────────────────────────

    def _parse_interactions(
        self,
        text: str,
        entities: list[_RawEntity],
    ) -> list[_RawInteraction]:
        """
        Detect physical interactions between entities.

        Spring rest_length and k are extracted from the prompt (with defaults).
        """
        interactions: list[_RawInteraction] = []
        dynamic  = [e for e in entities if not e.is_static]
        static_  = [e for e in entities if e.is_static]
        terrain  = next((e for e in static_ if e.entity_type == "terrain"), None)
        terrain_id = terrain.id if terrain else "environment"

        is_projectile = bool(re.search(r"\b(?:projectile|launched?|thrown?)\b", text))
        is_fall       = bool(re.search(r"\b(?:falls?|drops?)\b", text))

        for dyn in dynamic:
            interactions.append(_RawInteraction(
                type="contact",
                entity_a=dyn.id,
                entity_b=terrain_id,
                parameters={"normal": {"x": 0, "y": 1, "z": 0}},
            ))
            if not (is_projectile or is_fall):
                interactions.append(_RawInteraction(
                    type="friction",
                    entity_a=dyn.id,
                    entity_b=terrain_id,
                    parameters={},
                ))

        # ── Spring ────────────────────────────────────────────────────────
        if "spring" in text:
            anchor = next(
                (e for e in entities if e.entity_type == "structure"), None
            )
            mass_ent = next(
                (e for e in dynamic if e.entity_type != "structure"), None
            )
            if anchor and mass_ent:
                km_m = _RE_SPRING_K.search(text)
                k    = float(km_m.group(1)) if km_m else 100.0
                rest_length = self._extract_spring_rest_length(text)
                amplitude   = self._extract_spring_amplitude(text)
                interactions.append(_RawInteraction(
                    type="spring",
                    entity_a=anchor.id,
                    entity_b=mass_ent.id,
                    parameters={
                        "k_Nm":          k,
                        "rest_length_m": rest_length,
                        "amplitude_m":   amplitude,
                        "damping_Nsm":   0.0,
                    },
                ))
 
        # ── Collision ─────────────────────────────────────────────────────

        if len(dynamic) >= 2:

            if re.search(
                r"\b(?:collide|collision|crash|impact|hits?)\b",
                text
            ):

                interactions.append(
                    _RawInteraction(
                        type="collision",
                        entity_a=dynamic[0].id,
                        entity_b=dynamic[1].id,
                        parameters={},
                    )
                )

        return interactions

    # ── assembly ──────────────────────────────────────────────────────────────
    def _infer_scene_type(self, text: str) -> str:

        if "spring" in text:
            return "spring"

        if re.search(
            r"\b(?:falls?|drops?|released?)\b",
            text
        ):
            return "free_fall"

        if re.search(
            r"\b(?:projectile|launched?|thrown?|fired?)\b",
            text
        ):
            return "projectile"

        if re.search(
            r"\b(?:collide|collision|crash|impact|hits?)\b",
            text
        ):
            return "collision"

        if (
            re.search(r"\b(?:car|truck|vehicle|bus|van|lorry|automobile)\b", text)
            and
            re.search(r"\b(?:moves?|travels?|drives?|moving)\b", text)
        ):
            return "vehicle_motion"

        if re.search(
            r"\b(?:moves?|travels?|drives?)\b",
            text
        ):
            return "motion"

        return "static"

    def _estimate_confidence(
        self,
        entities,
        interactions,
        duration,
    ):

        score = 0.4

        if entities:
            score += 0.2

        if interactions:
            score += 0.2

        if any(
            e.velocity.magnitude() > 0
            for e in entities
        ):
            score += 0.1

        if duration != 10.0:
            score += 0.1

        return min(score, 1.0)

    def _assemble(
        self,
        scene_id:     str,
        prompt:       str,
        raw_entities: list[_RawEntity],
        env:          Environment,
        raw_ints:     list[_RawInteraction],
        duration:     float,
    ) -> WorldSpec:
        entities: list[Entity] = []

        for re_ in raw_entities:
            mat     = re_.material
            mat_def = MATERIAL_DEFAULTS.get(mat, MATERIAL_DEFAULTS["generic"])

            mass = re_.mass_kg
            if mass is None:
                vol  = re_.bounding_box.volume()
                mass = mat_def["density"] * vol
                if mass is None or mass <= 0:
                    mass = _DEFAULT_MASS.get(re_.entity_type, 1.0)

            if re_.entity_type == "terrain":
                mass = 0.0
                re_.is_static = True

            half_h = re_.bounding_box.height / 2.0
            if re_.entity_type == "terrain" and re_.position.y == 0.0:
                position = Vec3(0.0, -half_h, 0.0)
            else:
                position = re_.position

            state = PhysicsState(
                position=    position,
                velocity=    re_.velocity,
                orientation= Vec3(0.0, 0.0, 0.0),
            )

            entities.append(Entity(
                id=           re_.id,
                label=        re_.label,
                entity_type=  re_.entity_type,
                is_static=    re_.is_static,
                mass=         float(mass),
                material=     mat,
                restitution=  mat_def["restitution"],
                friction=     mat_def["friction"],
                bounding_box= re_.bounding_box,
                state=        state,
                forces=       [],
                constraints=  [],
                tags=         re_.tags,
            ))

        interactions: list[Interaction] = [
            Interaction(
                type=       ri.type,
                entity_a=   ri.entity_a,
                entity_b=   ri.entity_b,
                parameters= ri.parameters,
            )
            for ri in raw_ints
        ]

        sim_graph = SimulationGraph(
            dt=          self.default_dt,
            duration=    round(duration, 3),
            integrator=  "rk4",
            export_fps=  self.export_fps,
            events=      [],
        )

        return WorldSpec(
            scene_id=         scene_id,
            description=      prompt,
            entities=         entities,
            environment=      env,
            interactions=     interactions,
            simulation_graph= sim_graph,
            metadata={
                "parser":         "PromptParser",
                "parser_version": "2.0",
                "entity_count":   len(entities),
                "dynamic_count":  sum(1 for e in entities if not e.is_static),
                "static_count":   sum(1 for e in entities if e.is_static),
                "scene_type": self._infer_scene_type(prompt),
                "parse_confidence":
                    self._estimate_confidence(
                        raw_entities,
                        raw_ints,
                        duration,
                    ),
                "schema_version": "1.0",
                "warnings": getattr(
                    self,
                    "_warnings",
                    []
                ),
            },
        )

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PromptParser] {msg}")
