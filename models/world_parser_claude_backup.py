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

Each step is a separate LLM call with a tight JSON schema enforced via
a system prompt.  Results are composed into a final WorldSpec object.

Usage:
    parser = WorldParser()
    spec   = parser.parse("A red car moves at 60 km/h along a wet road …")
    spec.save("output/scene_001.json")
"""

from __future__ import annotations
import json
import re
import uuid
import math
import time
from typing import Any

import anthropic

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

