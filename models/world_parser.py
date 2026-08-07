"""
world_parser.py
───────────────
Prompt → WorldSpec, primary front-end pipeline.

    Prompt
        │
        ▼
    PromptParser              (deterministic, regex/keyword-based — see
        │                       models/prompt_parser.py; this is the
        │                       PRIMARY extraction stage, not an LLM)
        ▼
    SpatialRelationshipParser (deterministic; resolves "A left of B" etc.)
        │
        ▼
    OntologyResolver          (OPTIONAL — only runs if an EntityEncoder
        │                       checkpoint is supplied; grounds each entity
        │                       against the trained ontology classifier)
        ▼
    ValidationCompilerPass    (Tier 1/2/3 checks + bounded auto-repair,
        │                       see models/validator.py + models/repair.py)
        ▼
    LLM-assisted repair       (OPTIONAL — only invoked for the SPECIFIC
        │                       fields PromptParser itself flagged as
        │                       unknown in metadata["reasoning"]["unknown"];
        │                       never a full-scene LLM parse)
        ▼
    WorldSpec  (final)

CHANGE LOG (architecture-audit resolution — see chat history for full
rationale; summarized here for anyone reading the source directly):

  * REMOVED the old six-call Gemini/LLM pipeline (_extract_entities_raw,
    _extract_states, _build_entities, _extract_environment,
    _extract_interactions, _extract_simgraph, and their prompt template
    strings). It was DEAD CODE — wrapped in an inert triple-quoted string
    literal in WorldParser.parse() and never executed. It was never
    finished (no Gemini client existed anywhere in this codebase), and its
    docstring is what the paper's Section IV-B / Table II were incorrectly
    written from. Deleted rather than resurrected: finishing six
    independent LLM calls per scene is expensive, hard to test
    deterministically, and duplicates work PromptParser already does more
    reliably.
  * ADDED a bounded, targeted LLM repair step instead: PromptParser already
    records exactly which fields it couldn't confidently determine
    (metadata["reasoning"]["unknown"]). Only those specific fields are
    sent to the LLM for disambiguation — this is a real, working
    implementation of the architecture's `Validator → Repair` arrow using
    the anthropic client already imported here, not a placeholder.
  * ADDED OntologyResolver integration (models/ontology_resolver.py) —
    the trained EntityEncoder is now actually reachable from this pipeline,
    gated behind an explicit checkpoint path so the pipeline still runs
    (deterministic-only) when no checkpoint is available.
  * PhysicsValidator promoted to a first-class CompilerPass
    (models/repair.py: ValidationCompilerPass) with real, bounded repair
    instead of being called ad hoc.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from models.prompt_parser import PromptParser
from models.repair import ValidationCompilerPass
from models.world_compiler import WorldCompiler, OntologyPass
from models.pipeline_interfaces import PassResult

from world_spec import Entity, WorldSpec, Vec3


# ─────────────────────────────────────────────
# SpatialRelationshipParser
# ─────────────────────────────────────────────
# Unchanged from the original file — purely deterministic, no LLM calls.
# (Kept verbatim; omitted here for brevity in this excerpt. Copy the
# original SpatialRelationshipParser class body from the previous version
# of this file unchanged — nothing about it was dead code or incorrect.)

from spatial_relationship_parser import SpatialRelationshipParser  # noqa: E402
# NOTE: extracted to its own module (spatial_relationship_parser.py) as
# part of the file-length cleanup requested in the architecture audit
# ("split files larger than ~800-1000 lines"). See that file — the class
# body is byte-identical to the original world_parser.py's version.


# ─────────────────────────────────────────────
# Targeted LLM repair (bounded — NOT a full-scene parser)
# ─────────────────────────────────────────────

_REPAIR_SYSTEM_PROMPT = """You are assisting a deterministic physics-scene parser.
It has already extracted a scene into structured form and identified SPECIFIC
fields it could not confidently determine. Your only job is to fill in those
exact fields with a physically plausible value, given the original scene text.

Rules:
- Only answer the fields you are asked about. Do not add, remove, or modify
  anything else.
- Return ONLY a JSON object mapping each requested field key to a value.
- All numeric values must be in SI units (kg, m, m/s, rad).
- No markdown fences, no commentary, no explanation.
"""


class LLMFieldRepairer:
    """Fills in specific fields flagged as unknown by PromptParser, using a
    single, bounded LLM call — not a full-scene re-parse.

    This is intentionally the ONLY point in the pipeline where an LLM is
    called, and it only ever sees the original prompt plus a short list of
    named fields to resolve.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        # Lazy import: `anthropic` should not be a hard dependency for the
        # deterministic-only pipeline. Only imported when repair is
        # actually requested.
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def repair_unknowns(self, spec: WorldSpec) -> WorldSpec:
        unknowns = spec.metadata.get("reasoning", {}).get("unknown", [])
        if not unknowns:
            return spec

        fields_desc = "\n".join(
            f"- entity '{u['entity']}': {u['property']} "
            f"(required for {u.get('required_for', 'simulation')})"
            for u in unknowns
        )
        user_prompt = (
            f"Original scene description:\n{spec.description}\n\n"
            f"Fields to resolve:\n{fields_desc}\n\n"
            'Return JSON like: {"car.mass": 1200.0}'
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_REPAIR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
            text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE)
            resolved = json.loads(text)
        except Exception as exc:  # noqa: BLE001 — repair is best-effort
            spec.metadata.setdefault("warnings", []).append(
                f"LLM repair failed, unknowns left unresolved: {exc}"
            )
            return spec

        self._apply_resolved(spec, resolved)
        return spec

    @staticmethod
    def _apply_resolved(spec: WorldSpec, resolved: dict) -> None:
        for key, value in resolved.items():
            if "." not in key:
                continue
            label, prop = key.split(".", 1)
            entity = next((e for e in spec.entities if e.label == label), None)
            if entity is None:
                continue
            if prop == "mass" and isinstance(value, (int, float)):
                entity.mass = float(value)
            # extend here as more unknown-field types are observed in
            # practice — deliberately not guessing at properties beyond
            # mass until real usage shows what else commonly goes unknown.


class LLMRepairPass:
    """Wraps LLMFieldRepairer as a CompilerPass, so it sits in the same
    WorldCompiler pass list as OntologyPass/ValidationCompilerPass instead
    of being a special-cased stage WorldParser manages directly.

    Only does work if the spec actually has unresolved-unknown fields
    (checked before ever constructing the LLM client, so passes that never
    hit an unknown don't pay any LLM-related cost).
    """

    name = "llm_repair"

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model
        self._repairer: Optional[LLMFieldRepairer] = None  # lazy, same reasoning as OntologyResolver's tokenizer

    def run(self, spec: WorldSpec) -> PassResult:
        unknowns = spec.metadata.get("reasoning", {}).get("unknown", [])
        if not unknowns:
            return PassResult(ok=True, payload=None, notes="no unresolved unknowns — skipped")

        if self._repairer is None:
            self._repairer = LLMFieldRepairer(model=self._model)

        before = len(unknowns)
        spec = self._repairer.repair_unknowns(spec)
        after = len(spec.metadata.get("reasoning", {}).get("unknown", []))
        return PassResult(
            ok=True,
            payload={"unknowns_before": before, "unknowns_targeted": before},
            notes=f"targeted LLM repair for {before} unknown field(s)",
        )


# ─────────────────────────────────────────────
# WorldParser
# ─────────────────────────────────────────────

class WorldParser:
    """
    Converts a natural-language scene description into a validated WorldSpec.

    Orchestration only: PromptParser + SpatialRelationshipParser run first
    (they produce the WorldSpec candidate), then a WorldCompiler runs the
    grounding/validation/repair passes. Stage logic itself lives in the
    passes, not here — this class just sequences them and logs.
    """

    def __init__(
        self,
        verbose: bool = True,
        entity_encoder_checkpoint: Optional[str] = None,
        enable_llm_repair: bool = False,
        llm_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.verbose = verbose
        self._prompt_parser = PromptParser()
        self._spatial_parser = SpatialRelationshipParser()

        ontology_resolver = None
        if entity_encoder_checkpoint:
            ontology_resolver = self._build_ontology_resolver(entity_encoder_checkpoint)

        passes = [
            OntologyPass(ontology_resolver),
            ValidationCompilerPass(),
        ]
        if enable_llm_repair:
            passes.append(LLMRepairPass(model=llm_model))
            # re-validate after LLM repair may have touched fields —
            # ValidationCompilerPass is idempotent (no-ops when there are
            # no errors), so running it a second time is cheap and correct.
            passes.append(ValidationCompilerPass())

        self._compiler = WorldCompiler(passes=passes)

    @staticmethod
    def _build_ontology_resolver(checkpoint_path: str):
        import torch
        from models.entity_encoder import EntityEncoder, EntityEncoderConfig
        from models.ontology_resolver import OntologyResolver

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # NOTE: head_dimensions must match what the checkpoint was trained
        # with — wire config loading from checkpoints/model_config.json
        # once its schema is confirmed (see original comment history).
        encoder = EntityEncoder(EntityEncoderConfig())
        encoder.load_state_dict(state_dict)
        return OntologyResolver(encoder)

    def _log(self, step: str, msg: str = "") -> None:
        if self.verbose:
            print(f"[WorldParser] {step} {msg}")

    def parse(self, description: str, scene_id: str = None) -> WorldSpec:
        t0 = time.time()
        scene_id = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        self._log("START", f"scene_id={scene_id}")

        spec = self._prompt_parser.parse(description, scene_id=scene_id)
        self._spatial_parser.apply(description, spec.entities)

        spec, pass_results = self._compiler.compile(spec)
        for name, result in pass_results.items():
            self._log(f"pass:{name}", result.notes)

        # ValidationCompilerPass may run twice (before/after LLM repair,
        # when enabled) — pass_results is a dict keyed by pass name, so the
        # SECOND run's result is what survives under "validation". That's
        # the correct one to read: it reflects post-repair state.
        validation_result = pass_results["validation"]
        spec.metadata["validation"] = validation_result.payload.validation.summary_dict()
        spec.metadata["repair"] = validation_result.payload.repair.summary_dict()

        ontology_result = pass_results.get("ontology")
        if ontology_result is not None and ontology_result.payload:
            spec.metadata["ontology_grounding"] = ontology_result.payload

        llm_repair_result = pass_results.get("llm_repair")
        if llm_repair_result is not None and llm_repair_result.payload:
            spec.metadata["llm_repair"] = llm_repair_result.payload

        spec.metadata["parser_model"] = "PromptParser+Spatial+WorldCompiler"
        spec.metadata["parse_time_s"] = round(time.time() - t0, 2)
        spec.metadata["pipeline_ok"] = validation_result.ok

        self._log("DONE", f"entities={len(spec.entities)} ok={validation_result.ok}")
        return spec
