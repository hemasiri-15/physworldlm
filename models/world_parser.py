"""
world_parser.py
───────────────
Prompt → WorldSpec using the Anthropic API (claude-sonnet-4-6).

Pipeline:
  1. Entity extraction  (what objects exist?)
  2. Static/dynamic classification
  3. Physics parameter assignment  (mass, friction, restitution …)
  4. Environmental condition inference
  5. Interaction & constraint detection
  6. SimulationGraph specification
  7. WorldSpec assembly & validation
  8. Spatial relationship post-processing  ← NEW (purely deterministic, no LLM)

Each step is a separate LLM call with a tight JSON schema enforced via
a system prompt.  Results are composed into a final WorldSpec object.

Usage:
    parser = WorldParser()
    spec   = parser.parse("A red car moves at 60 km/h along a wet road …")
    spec.save("output/scene_001.json")
"""

from __future__ import annotations
import json
import math
import re
import uuid
import time
from typing import Any

import anthropic
from models.prompt_parser import PromptParser

from models.world_spec import (
    WorldSpec, Entity, PhysicsState, Environment,
    Interaction, SimulationGraph, SimulationGraph,
    BoundingBox, Vec3, Wind,
    MATERIAL_DEFAULTS, kmh_to_ms, mph_to_ms, deg_to_rad,
    celsius_to_kelvin, fahrenheit_to_kelvin,
)


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

_SYSTEM_PHYSICS = """You are a physics simulation expert and scene analyst.
Your job is to convert natural language scene descriptions into structured
JSON that can be fed directly into a physics engine (Bullet / MuJoCo / Gazebo).

Rules:
- All values must be in SI units (kg, m, m/s, rad, Pa, K) unless a key name
  says otherwise.
- Never invent entities that are not implied by the description.
- For every quantity you are uncertain about, use a physically plausible
  default and set "estimated": true in the JSON.
- Return ONLY valid JSON. No markdown fences, no commentary.
"""

_ENTITY_EXTRACTION_PROMPT = """Given this scene description, extract ALL physical entities.

Scene: {description}

Return a JSON array. Each element:
{{
  "id": "e_<short_slug>",
  "label": "human readable name",
  "entity_type": "vehicle|projectile|fluid|agent|structure|terrain|object",
  "is_static": true|false,
  "implied_material": "steel|rubber|wood|concrete|water|glass|flesh|plastic|air|generic",
  "implied_mass_kg": <number or null if truly unknown>,
  "bounding_box_m": {{"width": w, "height": h, "depth": d}},
  "tags": ["list", "of", "semantic", "tags"],
  "estimated": true|false
}}

Examples of static: ground, road, wall, building, tree, lake, mountain.
Examples of dynamic: car, ball, person, drone, projectile.
"""

_STATE_EXTRACTION_PROMPT = """Given this scene description and entity list, assign initial kinematic state to each entity.

Scene: {description}
Entities: {entities_json}

For each entity id, return:
{{
  "entity_id": "e_xxx",
  "position_m": {{"x": 0, "y": 0, "z": 0}},
  "velocity_ms": {{"x": 0, "y": 0, "z": 0}},
  "orientation_rad": {{"x": 0, "y": 0, "z": 0}},
  "forces": [
    {{"label": "gravity", "vector_N": {{"x":0,"y":-9.81,"z":0}}, "per_unit_mass": true}}
  ]
}}

Spatial conventions: x=East, y=Up, z=North.
Convert any km/h or mph to m/s.
Place ground at y=0. Objects resting on ground have their base at y=0.
Return a JSON array.
"""

_ENVIRONMENT_PROMPT = """Extract environmental conditions from this scene description.

Scene: {description}

Return a single JSON object:
{{
  "gravity_ms2": {{"x": 0, "y": -9.81, "z": 0}},
  "temperature_K": 293.15,
  "pressure_Pa": 101325.0,
  "air_density_kgm3": 1.225,
  "wind": {{
    "speed_ms": 0.0,
    "direction_rad": 0.0
  }},
  "terrain_type": "flat|hilly|urban|water|mixed",
  "friction_global": 0.5,
  "time_of_day": "day|night|dawn|dusk",
  "weather": "clear|rain|snow|fog|wind"
}}

Adjust air_density and pressure for altitude if mentioned.
If road is wet set friction_global to 0.3. If icy set to 0.1.
"""

_INTERACTIONS_PROMPT = """Given the scene and entities, identify all physical interactions and constraints.

Scene: {description}
Entities: {entities_json}

Return a JSON array of interactions:
{{
  "type": "collision|joint|contact|fluid_drag|friction|magnetic|gravity",
  "entity_a": "e_xxx",
  "entity_b": "e_yyy_or_environment",
  "parameters": {{}}
}}

Include at minimum:
- Ground contact for every non-flying dynamic entity.
- Friction between every moving object and its surface.
- Collision pairs for objects on a path toward each other.
"""

_SIMGRAPH_PROMPT = """Given this scene description, specify the simulation parameters.

Scene: {description}

Return a single JSON object:
{{
  "dt_s": 0.01,
  "duration_s": 10.0,
  "integrator": "rk4",
  "export_fps": 30,
  "events": [
    {{
      "t_s": 1.5,
      "type": "collision|force_change|entity_spawn|entity_remove",
      "entity_ids": ["e_xxx"],
      "description": "what happens"
    }}
  ]
}}

Choose duration long enough to see the described action complete.
Add events only if the description explicitly implies a discrete change.
"""


# ─────────────────────────────────────────────
# SpatialRelationshipParser
# ─────────────────────────────────────────────
#
# Purely deterministic; no LLM calls.
# Parses natural-language spatial relationships from a scene description
# and updates entity positions in-place to satisfy the stated geometry.
#
# Coordinate system (inherited from WorldSpec):
#   x = East / forward
#   y = Up
#   z = North
#
# Axis assignments per relation:
#   left / right / in front of / behind / distance  → x-axis
#   above / on / below                              → y-axis
#   near                                            → x-axis (close separation)

class SpatialRelationshipParser:
    """
    Detect and resolve natural-language spatial relationships into SI
    coordinates on a set of :class:`~models.world_spec.Entity` objects.

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

    # ── compiled regex patterns ────────────────────────────────────────────
    #
    # Number with optional decimal
    _NUM = r"(\d+(?:\.\d+)?)"
    # Optional unit (m, meters, metres, km, ft, feet, cm)
    _UNIT = r"(?:\s*(?:meters?|metres?|km|ft|feet|centimeters?|cm|m\b))?"

    # Pattern: "<entity A> is/are <N> m from <entity B>"
    _RE_DISTANCE = re.compile(
        rf"(\w[\w\s]*?)\s+(?:is|are)\s+{_NUM}\s*{_UNIT}\s+(?:away\s+)?from\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> above <entity B>"
    _RE_ABOVE = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?above\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> below <entity B>"
    _RE_BELOW = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?below\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> on (top of / a / the) <entity B>"
    _RE_ON = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?on(?:\s+top\s+of|\s+a|\s+the|\s+an)?\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> left of <entity B>" or "to the left of"
    _RE_LEFT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?(?:to\s+the\s+)?left\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> right of <entity B>"
    _RE_RIGHT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?(?:to\s+the\s+)?right\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> behind <entity B>"
    _RE_BEHIND = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?behind\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> in front of <entity B>"
    _RE_IN_FRONT = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?in\s+front\s+of\s+(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    # Pattern: "<entity A> near <entity B>"
    _RE_NEAR = re.compile(
        r"(\w[\w\s]*?)\s+(?:is\s+)?near\s+(?:a\s+|the\s+|an\s+)?(\w[\w\s]*?)(?=[,.]|$)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        near_threshold_m: float = 4.0,
        gap_m: float = 0.05,
    ) -> None:
        self._near_threshold = near_threshold_m
        self._GAP = gap_m

    # ── public API ─────────────────────────────────────────────────────────

    def apply(self, description: str, entities: list[Entity]) -> None:
        """
        Detect spatial relationships in *description* and mutate entity
        positions in-place to satisfy the stated geometry.

        Only position coordinates are modified; velocities, masses,
        orientations, and all other fields are untouched.

        Parameters
        ----------
        description : str
            The original plain-English scene description.
        entities : list[Entity]
            The fully constructed entity list (positions may be arbitrary).
        """
        label_map = self._build_label_map(entities)
        relations  = self._detect_relations(description, label_map)
        self._resolve(relations, label_map)

    def parse_relationships(
        self,
        description: str,
        entities: list[Entity],
    ) -> list[dict]:
        """
        Parse without mutating.

        Returns
        -------
        list[dict]
            Each dict has keys: ``relation``, ``entity_a``, ``entity_b``,
            and optionally ``distance_m``.
        """
        label_map = self._build_label_map(entities)
        return self._detect_relations(description, label_map)

    # ── entity name resolution ─────────────────────────────────────────────

    @staticmethod
    def _build_label_map(entities: list[Entity]) -> dict[str, Entity]:
        """
        Build a case-insensitive token → Entity lookup.

        Each entity contributes:
          • Its full label (e.g. "red car")
          • Each individual word of its label
          • Its entity_type
          • Each tag
        All keys are lower-cased.  Longer / more specific tokens take
        precedence (inserted last, so early broad matches are overwritten
        by more specific ones if processed in length order).
        """
        # Build candidate tokens sorted by specificity (shorter first so
        # longer labels overwrite)
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

        # Sort by token length ascending; longer tokens overwrite shorter
        # ones and thus take precedence in ambiguous cases
        candidates.sort(key=lambda x: len(x[0]))
        return {tok: ent for tok, ent in candidates}

    @staticmethod
    def _resolve_entity(
        raw_name: str,
        label_map: dict[str, Entity],
    ) -> Entity | None:
        """
        Resolve a raw matched string to an Entity.

        Tries (in order):
          1. Exact lower-case match
          2. Each word in the string
          3. Any label_map key that is a substring of the raw name
        Returns None when no match is found.
        """
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

    # ── relation detection ─────────────────────────────────────────────────

    def _detect_relations(
        self,
        description: str,
        label_map: dict[str, Entity],
    ) -> list[dict]:
        """
        Scan *description* for all supported spatial phrases and return a
        structured list of relationship dicts.

        Each dict contains:
          ``relation``   – one of: distance, above, below, on, left, right,
                           behind, in_front, near
          ``entity_a``   – the Entity that is described relative to entity_b
          ``entity_b``   – the reference Entity
          ``distance_m`` – (distance relation only) explicit distance in metres
        """
        relations: list[dict] = []

        # Helper to append only when both entities resolve
        def add(rel: str, raw_a: str, raw_b: str, **kw) -> None:
            ea = self._resolve_entity(raw_a, label_map)
            eb = self._resolve_entity(raw_b, label_map)
            if ea is not None and eb is not None and ea is not eb:
                entry = {"relation": rel, "entity_a": ea, "entity_b": eb}
                entry.update(kw)
                relations.append(entry)

        # ── distance ──────────────────────────────────────────────────────
        for m in self._RE_DISTANCE.finditer(description):
            raw_a, dist_str, raw_b = m.group(1), m.group(2), m.group(3)
            try:
                add("distance", raw_a, raw_b, distance_m=float(dist_str))
            except ValueError:
                pass

        # ── above ─────────────────────────────────────────────────────────
        for m in self._RE_ABOVE.finditer(description):
            add("above", m.group(1), m.group(2))

        # ── below ─────────────────────────────────────────────────────────
        for m in self._RE_BELOW.finditer(description):
            add("below", m.group(1), m.group(2))

        # ── on ────────────────────────────────────────────────────────────
        for m in self._RE_ON.finditer(description):
            add("on", m.group(1), m.group(2))

        # ── left ──────────────────────────────────────────────────────────
        for m in self._RE_LEFT.finditer(description):
            add("left", m.group(1), m.group(2))

        # ── right ─────────────────────────────────────────────────────────
        for m in self._RE_RIGHT.finditer(description):
            add("right", m.group(1), m.group(2))

        # ── behind ────────────────────────────────────────────────────────
        for m in self._RE_BEHIND.finditer(description):
            add("behind", m.group(1), m.group(2))

        # ── in front of ───────────────────────────────────────────────────
        for m in self._RE_IN_FRONT.finditer(description):
            add("in_front", m.group(1), m.group(2))

        # ── near ──────────────────────────────────────────────────────────
        for m in self._RE_NEAR.finditer(description):
            add("near", m.group(1), m.group(2))

        return relations

    # ── coordinate resolution ─────────────────────────────────────────────

    def _resolve(
        self,
        relations: list[dict],
        label_map: dict[str, Entity],
    ) -> None:
        """
        Apply each relation to update entity positions in-place.

        Relations are processed in the order they were detected.
        Each relation adjusts only the relevant axis, leaving the others
        unchanged.  This means a description like "A above B and A left of C"
        can set both y and x independently.
        """
        for rel in relations:
            kind = rel["relation"]
            ea: Entity = rel["entity_a"]
            eb: Entity = rel["entity_b"]

            pos_a = ea.state.position
            pos_b = eb.state.position
            bb_a  = ea.bounding_box
            bb_b  = eb.bounding_box

            if kind == "distance":
                dist = rel.get("distance_m", 10.0)
                # Place A at +dist along x from B; keep y/z the same as B.
                new_x = pos_b.x + dist
                ea.state.position = Vec3(new_x, pos_a.y, pos_a.z)

            elif kind == "above":
                # A's base is above B's top surface; add a small gap.
                b_top = pos_b.y + bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(
                    pos_a.x,
                    b_top + a_half_h + self._GAP,
                    pos_a.z,
                )

            elif kind == "below":
                # A's top is below B's base; add a small gap downward.
                b_base = pos_b.y - bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(
                    pos_a.x,
                    b_base - a_half_h - self._GAP,
                    pos_a.z,
                )

            elif kind == "on":
                # A rests directly on top of B's upper surface.
                b_top = pos_b.y + bb_b.height / 2.0
                a_half_h = bb_a.height / 2.0
                ea.state.position = Vec3(
                    pos_b.x,   # centre A over B horizontally
                    b_top + a_half_h,
                    pos_b.z,
                )

            elif kind == "left":
                # A is to the left (negative x) of B.
                b_left_edge = pos_b.x - bb_b.width / 2.0
                a_half_w    = bb_a.width / 2.0
                ea.state.position = Vec3(
                    b_left_edge - a_half_w - self._GAP,
                    pos_a.y,
                    pos_a.z,
                )

            elif kind == "right":
                # A is to the right (positive x) of B.
                b_right_edge = pos_b.x + bb_b.width / 2.0
                a_half_w     = bb_a.width / 2.0
                ea.state.position = Vec3(
                    b_right_edge + a_half_w + self._GAP,
                    pos_a.y,
                    pos_a.z,
                )

            elif kind == "behind":
                # "Behind" = smaller x value (B is forward / East of A).
                b_back_edge = pos_b.x - bb_b.depth / 2.0
                a_half_d    = bb_a.depth / 2.0
                ea.state.position = Vec3(
                    b_back_edge - a_half_d - self._GAP,
                    pos_a.y,
                    pos_a.z,
                )

            elif kind == "in_front":
                # "In front of" = larger x value.
                b_front_edge = pos_b.x + bb_b.depth / 2.0
                a_half_d     = bb_a.depth / 2.0
                ea.state.position = Vec3(
                    b_front_edge + a_half_d + self._GAP,
                    pos_a.y,
                    pos_a.z,
                )

            elif kind == "near":
                # Place A within _NEAR_SEP metres of B along x.
                ea.state.position = Vec3(
                    pos_b.x + self._NEAR_SEP,
                    pos_a.y,
                    pos_a.z,
                )


# ─────────────────────────────────────────────
# WorldParser
# ─────────────────────────────────────────────

class WorldParser:
    """
    Converts a natural-language scene description into a WorldSpec
    via a multi-step LLM pipeline.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", verbose: bool = True):
        self.client  = anthropic.Anthropic()
        self.model   = model
        self.verbose = verbose
        self._spatial_parser = SpatialRelationshipParser()
        self._prompt_parser = PromptParser()

    # ── private helpers ──────────────────────

    def _llm(self, user_prompt: str, max_tokens: int = 2048) -> str:
        """Single LLM call; returns raw text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=_SYSTEM_PHYSICS,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _parse_json(self, text: str) -> Any:
        """Strip markdown fences if present and parse JSON."""
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```$",       "", text, flags=re.MULTILINE)
        return json.loads(text.strip())

    def _log(self, step: str, msg: str = "") -> None:
        if self.verbose:
            print(f"[WorldParser] {step} {msg}")

    # ── Step 1+2: Entity extraction ──────────

    def _extract_entities_raw(self, description: str) -> list[dict]:
        self._log("Step 1+2", "extracting entities …")
        prompt = _ENTITY_EXTRACTION_PROMPT.format(description=description)
        raw    = self._llm(prompt, max_tokens=2048)
        return self._parse_json(raw)

    # ── Step 4: Physics parameters ───────────

    def _build_entities(self, raw_entities: list[dict],
                        states: list[dict]) -> list[Entity]:
        self._log("Step 4", "assigning physics parameters …")
        state_map = {s["entity_id"]: s for s in states}
        entities  = []

        for re_ in raw_entities:
            eid      = re_["id"]
            material = re_.get("implied_material", "generic")
            mat_def  = MATERIAL_DEFAULTS.get(material, MATERIAL_DEFAULTS["generic"])

            bb_raw   = re_.get("bounding_box_m", {})
            bb       = BoundingBox(
                width=  bb_raw.get("width",  1.0),
                height= bb_raw.get("height", 1.0),
                depth=  bb_raw.get("depth",  1.0),
            )

            # Mass: use LLM estimate or derive from density × volume
            mass = re_.get("implied_mass_kg")
            if mass is None:
                mass = mat_def["density"] * bb.volume()

            # Build PhysicsState from state extraction step
            s_raw = state_map.get(eid, {})
            pos_d = s_raw.get("position_m",    {"x":0,"y":0,"z":0})
            vel_d = s_raw.get("velocity_ms",   {"x":0,"y":0,"z":0})
            ori_d = s_raw.get("orientation_rad",{"x":0,"y":0,"z":0})

            state = PhysicsState(
                position=    Vec3(**pos_d),
                velocity=    Vec3(**vel_d),
                orientation= Vec3(**ori_d),
            )

            entity = Entity(
                id=           eid,
                label=        re_["label"],
                entity_type=  re_["entity_type"],
                is_static=    re_.get("is_static", False),
                mass=         float(mass),
                material=     material,
                restitution=  mat_def["restitution"],
                friction=     mat_def["friction"],
                bounding_box= bb,
                state=        state,
                forces=       s_raw.get("forces", []),
                constraints=  [],
                tags=         re_.get("tags", []),
            )
            entities.append(entity)

        return entities

    # ── Step 5: Environment ──────────────────

    def _extract_environment(self, description: str) -> Environment:
        self._log("Step 5", "inferring environmental conditions …")
        raw  = self._llm(
            _ENVIRONMENT_PROMPT.format(description=description), max_tokens=512
        )
        d    = self._parse_json(raw)
        grav = d.get("gravity_ms2", {"x":0,"y":-9.81,"z":0})
        wind = d.get("wind", {})
        return Environment(
            gravity=         Vec3(**grav),
            temperature_K=   d.get("temperature_K", 293.15),
            pressure_Pa=     d.get("pressure_Pa", 101325.0),
            air_density=     d.get("air_density_kgm3", 1.225),
            wind=            Wind(
                                speed=    wind.get("speed_ms", 0.0),
                                direction=wind.get("direction_rad", 0.0),
                             ),
            terrain_type=    d.get("terrain_type", "flat"),
            friction_global= d.get("friction_global", 0.5),
            time_of_day=     d.get("time_of_day", "day"),
            weather=         d.get("weather", "clear"),
        )

    # ── Step 6: Interactions ─────────────────

    def _extract_interactions(self, description: str,
                               entities: list[Entity]) -> list[Interaction]:
        self._log("Step 6", "detecting interactions & constraints …")
        ents_json = json.dumps([{"id": e.id, "label": e.label,
                                  "is_static": e.is_static} for e in entities])
        raw  = self._llm(
            _INTERACTIONS_PROMPT.format(
                description=description, entities_json=ents_json
            ), max_tokens=1024
        )
        raw_list = self._parse_json(raw)
        return [
            Interaction(
                type=       r["type"],
                entity_a=   r["entity_a"],
                entity_b=   r.get("entity_b", "environment"),
                parameters= r.get("parameters", {}),
            )
            for r in raw_list
        ]

    # ── Step 7: SimulationGraph ──────────────

    def _extract_simgraph(self, description: str) -> SimulationGraph:
        self._log("Step 7", "building simulation graph …")
        raw = self._llm(
            _SIMGRAPH_PROMPT.format(description=description), max_tokens=512
        )
        d   = self._parse_json(raw)
        return SimulationGraph(
            dt=         d.get("dt_s", 0.01),
            duration=   d.get("duration_s", 10.0),
            integrator= d.get("integrator", "rk4"),
            export_fps= d.get("export_fps", 30),
            events=     d.get("events", []),
        )

    # ── Step 3: State extraction (needs entity list) ─

    def _extract_states(self, description: str,
                         raw_entities: list[dict]) -> list[dict]:
        self._log("Step 3", "extracting initial kinematic states …")
        ents_json = json.dumps(
            [{"id": e["id"], "label": e["label"],
              "is_static": e["is_static"]} for e in raw_entities]
        )
        raw = self._llm(
            _STATE_EXTRACTION_PROMPT.format(
                description=description, entities_json=ents_json
            ), max_tokens=2048
        )
        return self._parse_json(raw)

    # ── Step 8: Spatial relationship post-processing ─

    def _parse_spatial_relationships(
        self,
        description: str,
        entities: list[Entity],
    ) -> None:
        """
        Detect natural-language spatial relationships in *description* and
        update entity positions in-place to satisfy the stated geometry.

        This step runs **after** the LLM pipeline has assembled all entities,
        so bounding-box dimensions are already known and can be used to
        compute non-overlapping positions.

        Supported relations
        -------------------
        ``<A> is <N> m from <B>``
            Place A at B.x + N along the x-axis.
        ``<A> above <B>``
            A's base is above B's top surface (y-axis).
        ``<A> below <B>``
            A's top is below B's base (y-axis).
        ``<A> on <B>``
            A rests on top of B (y-axis contact).
        ``<A> left of <B>``
            A is to the left (negative x) of B.
        ``<A> right of <B>``
            A is to the right (positive x) of B.
        ``<A> behind <B>``
            A's x-coordinate is less than B's (B is "forward").
        ``<A> in front of <B>``
            A's x-coordinate is greater than B's.
        ``<A> near <B>``
            A is placed within ``SpatialRelationshipParser._NEAR_SEP`` m
            of B along the x-axis.

        Velocities, masses, orientations, and all other entity fields are
        never modified by this step.

        Parameters
        ----------
        description : str
            The original plain-English scene description (passed through
            unchanged from :meth:`parse`).
        entities : list[Entity]
            Fully constructed entity list; positions are mutated in-place.
        """
        self._log("Step 8", "resolving spatial relationships …")
        self._spatial_parser.apply(description, entities)

    # ── Validation ───────────────────────────

    def _validate(self, spec: WorldSpec) -> list[str]:
        """Return list of warning strings; empty = OK."""
        warnings = []
        ids = {e.id for e in spec.entities}

        for itr in spec.interactions:
            if itr.entity_a not in ids:
                warnings.append(f"Interaction references unknown entity: {itr.entity_a}")
            if itr.entity_b not in ids and itr.entity_b != "environment":
                warnings.append(f"Interaction references unknown entity: {itr.entity_b}")

        for e in spec.entities:
            if not e.is_static and e.mass <= 0:
                warnings.append(f"Entity {e.id} has non-positive mass: {e.mass}")

        if spec.simulation_graph.dt <= 0:
            warnings.append("SimulationGraph dt must be positive")

        return warnings

    # ── Public API ───────────────────────────

    def parse(self, description: str, scene_id: str = None) -> WorldSpec:
        """
        Convert a natural-language scene description to a WorldSpec.

        Args:
            description : the scene in plain English (or any language)
            scene_id    : optional stable id; auto-generated if None

        Returns:
            WorldSpec  (fully populated, SI units throughout)
        """
        """
        t0       = time.time()
        scene_id = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        self._log("START", f"scene_id={scene_id}")

        # ── pipeline ──────────────────────────
        raw_entities = self._extract_entities_raw(description)          # Step 1+2
        states       = self._extract_states(description, raw_entities)  # Step 3
        entities     = self._build_entities(raw_entities, states)       # Step 4
        environment  = self._extract_environment(description)           # Step 5
        interactions = self._extract_interactions(description, entities) # Step 6
        sim_graph    = self._extract_simgraph(description)              # Step 7

        # ── spatial relationship post-processing (deterministic, no LLM) ─
        self._parse_spatial_relationships(description, entities)        # Step 8

        # ── assembly ──────────────────────────
        spec = WorldSpec(
            scene_id=         scene_id,
            description=      description,
            entities=         entities,
            environment=      environment,
            interactions=     interactions,
            simulation_graph= sim_graph,
            metadata={
                "parser_model":    self.model,
                "parse_time_s":    round(time.time() - t0, 2),
                "entity_count":    len(entities),
                "dynamic_count":   len([e for e in entities if not e.is_static]),
                "static_count":    len([e for e in entities if e.is_static]),
                "schema_version":  "1.0",
            },
        )

        warnings = self._validate(spec)
        if warnings:
            spec.metadata["warnings"] = warnings
            for w in warnings:
                self._log("WARNING", w)

        elapsed = time.time() - t0
        self._log("DONE", f"entities={len(entities)} elapsed={elapsed:.1f}s")
        return spec
        """
        t0 = time.time()

        scene_id = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        self._log("START", f"scene_id={scene_id}")

        # ---- deterministic parser ----
        spec = self._prompt_parser.parse(
            description,
            scene_id=scene_id,
        )

        # ---- spatial post-processing ----
        self._spatial_parser.apply(
            description,
            spec.entities,
        )

        # ---- metadata ----
        spec.metadata["parser_model"] = "PromptParser+Spatial"
        spec.metadata["parse_time_s"] = round(
            time.time() - t0,
            2,
        )

        warnings = self._validate(spec)

        if warnings:
            spec.metadata["warnings"] = warnings

        self._log(
            "DONE",
            f"entities={len(spec.entities)}"
        )

        return spec
