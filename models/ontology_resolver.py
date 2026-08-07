"""
models/ontology_resolver.py
────────────────────────────
Translation layer: EntityEncoder logits → grounded Entity fields.

    EntityEncoder
        │  EntityEncoderOutput (shared_embedding, 17 head logits)
        ▼
    OntologyResolver.resolve()
        │  decode each head's argmax/threshold → (label, confidence)
        │  ConflictResolver decides parser-value vs encoder-value per field
        ▼
    CandidateEntity (still unvalidated — Validator runs next)

ASSUMPTION FLAGGED FOR VERIFICATION
────────────────────────────────────
This module assumes ``datasets/entity/label_maps/`` contains one JSON file
per head, named ``{head_name}.json``, mapping ``{"0": "label_a", "1":
"label_b", ...}`` (index → label string). This matches the directory name
visible in the repository tree, but the actual file contents were not
available when this was written — confirm the key format and filename
pattern against the real files before relying on this in production.

Direct-mapping heads (subject to ConflictResolver, not a blind overwrite):
    entity_type → Entity.entity_type
    material    → Entity.material

Every other head is stored in Entity.ontology, each as
``{"value": ..., "confidence": ..., "source": "EntityEncoder"}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer

from models.entity_encoder import (
    EntityEncoder,
    SINGLE_LABEL_HEADS,
    MULTI_LABEL_HEADS,
    BACKBONE_NAME,
)
from world_spec import Entity


_DIRECT_MAP = {"entity_type": "entity_type", "material": "material"}
_MULTI_LABEL_THRESHOLD = 0.5

# Confidence for parser-sourced values, tiered by HOW the parser arrived at
# the value — not a flat 1.0. A regex match is a real finding but not a
# certainty; using 1.0 for every parser value was fabricated precision, not
# measured precision. These tiers map onto PromptParser's own distinction
# between metadata["reasoning"]["known"] (explicit in the prompt text,
# e.g. "60 km/h" matched directly) and ["derived"] (computed from an
# explicit value via a formula, e.g. velocity components from a speed+angle)
# — "default" covers a value that came from neither and is just a type
# default. Set to 0.99/0.85/0.50 respectively per the discussion; adjust
# empirically once the benchmark suite's parser-accuracy numbers exist.
_PARSER_CONFIDENCE_EXPLICIT = 0.99
_PARSER_CONFIDENCE_DERIVED  = 0.85
_PARSER_CONFIDENCE_DEFAULT  = 0.50

# Fields the deterministic PromptParser sets to these placeholder values
# when it genuinely didn't determine anything — i.e. "not a real parser
# finding," safe for the encoder to overwrite without it counting as a
# conflict.
_PARSER_PLACEHOLDER_VALUES = {
    "entity_type": {None, "object", "unknown"},
    "material":    {None, "unknown", "generic"},
}


# ─────────────────────────────────────────────
# ConflictResolver
# ─────────────────────────────────────────────

@dataclass
class ResolvedField:
    value: object
    confidence: float
    source: str          # "parser" | "encoder" | "encoder(no-conflict)"
    conflict: bool = False
    parser_value: Optional[object] = None
    encoder_value: Optional[object] = None


class ConflictResolver:
    """Decides, per field, whether the parser's value or the encoder's
    value wins.

    Rule (matches the review's spec):
      - If the parser's value is a real, non-placeholder finding, it wins —
        the encoder never overrides something the prompt explicitly stated.
      - If the parser's value is a placeholder (didn't find anything), the
        encoder's value fills the gap.
      - Either way, both values and whether they disagreed are recorded in
        the returned ResolvedField — nothing is silently dropped.
    """

    @staticmethod
    def resolve(
        field_name: str,
        parser_value: object,
        encoder_value: object,
        encoder_confidence: float,
    ) -> ResolvedField:
        placeholders = _PARSER_PLACEHOLDER_VALUES.get(field_name, {None})
        parser_is_placeholder = parser_value in placeholders

        if not parser_is_placeholder:
            conflict = (
                encoder_value is not None
                and encoder_value != parser_value
            )
            # HONESTY NOTE: the deterministic parser does not currently
            # distinguish "explicit literal match" from "keyword/heuristic
            # match" for entity_type/material specifically (unlike speed
            # or mass, which it does tag as known/derived/unknown). Using
            # the DERIVED tier here — not EXPLICIT — because a keyword
            # classification ("car" -> vehicle) is a real finding but not
            # the same certainty as a directly-quoted number. If
            # PromptParser is later extended to tag entity_type/material
            # provenance the same way it tags speed/mass, wire that
            # distinction through here instead of this fixed tier.
            return ResolvedField(
                value=parser_value,
                confidence=_PARSER_CONFIDENCE_DERIVED,
                source="parser",
                conflict=conflict,
                parser_value=parser_value,
                encoder_value=encoder_value,
            )

        if encoder_value is not None:
            return ResolvedField(
                value=encoder_value,
                confidence=encoder_confidence,
                source="encoder(no-conflict)",
                conflict=False,
                parser_value=parser_value,
                encoder_value=encoder_value,
            )

        # Neither source had anything — keep the placeholder, flag low confidence.
        return ResolvedField(
            value=parser_value,
            confidence=_PARSER_CONFIDENCE_DEFAULT,
            source="parser",
            conflict=False,
            parser_value=parser_value,
            encoder_value=None,
        )


class OntologyResolver:
    """Runs EntityEncoder on an entity's label text and grounds the result
    onto a WorldSpec Entity via ConflictResolver.

    Parameters
    ----------
    encoder : EntityEncoder
        A constructed (and ideally checkpoint-loaded) encoder instance.
    label_maps_dir : str
        Path to the directory containing one JSON file per head.
    device : str
        torch device string.
    cache_size : int
        Number of distinct entity labels to cache decoded results for.
        Set to 0 to disable caching. Default 256 — cheap, and directly
        addresses the "car car car" repeated-label case the review raised.
    """

    def __init__(
        self,
        encoder: EntityEncoder,
        label_maps_dir: str = "datasets/entity/label_maps",
        device: str = "cpu",
        cache_size: int = 256,
    ) -> None:
        self.encoder = encoder.to(device).eval()
        self.device = device
        self._label_maps_dir = label_maps_dir
        self._tokenizer = None  # lazy — see .tokenizer property
        self.label_maps = self._load_label_maps(label_maps_dir)
        self._resolver = ConflictResolver()

        if cache_size > 0:
            self._decode_label = lru_cache(maxsize=cache_size)(self._decode_label_uncached)
        else:
            self._decode_label = self._decode_label_uncached

    # ── lazy tokenizer ────────────────────────────────────────────────

    @property
    def tokenizer(self):
        """Loaded on first access, not at construction time — avoids
        paying HuggingFace tokenizer load cost when OntologyResolver is
        instantiated but grounding is never actually invoked (e.g. a
        WorldParser configured with entity_encoder_checkpoint set but a
        run that hits an early validation failure before Stage 3).
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
        return self._tokenizer

    # ── label map loading ────────────────────────────────────────────

    @staticmethod
    def _load_label_maps(label_maps_dir: str) -> Dict[str, Dict[int, str]]:
        maps: Dict[str, Dict[int, str]] = {}
        base = Path(label_maps_dir)
        for head in (*SINGLE_LABEL_HEADS, *MULTI_LABEL_HEADS):
            fpath = base / f"{head}.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                raw = json.load(f)
            maps[head] = {int(k): v for k, v in raw.items()}
        return maps

    # ── cached per-label decoding ────────────────────────────────────

    def _decode_label_uncached(self, label: str) -> Dict[str, tuple]:
        """Run the encoder on one label string; return
        {head_name: (value, confidence)} for every decodable head.

        Note on caching correctness: this is cached by label text only.
        That's valid because the encoder is deterministic given fixed
        weights (eval mode, no dropout active) — same text always produces
        the same logits. If the encoder is ever put in .train() mode or
        checkpoints are hot-swapped at runtime, the cache must be cleared
        (``self._decode_label.cache_clear()`` when using the lru_cache
        path) or results will go stale.
        """
        if not self.label_maps:
            return {}

        enc = self.tokenizer(
            label, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.encoder(enc["input_ids"], enc["attention_mask"])

        decoded: Dict[str, tuple] = {}

        for head in SINGLE_LABEL_HEADS:
            if head not in output.logits or head not in self.label_maps:
                continue
            probs = torch.softmax(output.logits[head], dim=-1).squeeze(0)
            idx = int(probs.argmax().item())
            confidence = float(probs[idx].item())
            decoded[head] = (self.label_maps[head].get(idx), confidence)

        for head in MULTI_LABEL_HEADS:
            if head not in output.logits or head not in self.label_maps:
                continue
            probs = torch.sigmoid(output.logits[head]).squeeze(0)
            active = (probs > _MULTI_LABEL_THRESHOLD).nonzero(as_tuple=True)[0]
            values = [self.label_maps[head].get(int(i)) for i in active]
            mean_conf = float(probs[active].mean().item()) if len(active) else 0.0
            decoded[head] = (values, mean_conf)

        return decoded

    # ── public API ────────────────────────────────────────────────────

    def resolve(self, entity: Entity) -> Entity:
        """Ground a single Entity using its label text, via ConflictResolver."""
        decoded = self._decode_label(entity.label)
        if not decoded:
            return entity

        for field_name, mapped_attr in _DIRECT_MAP.items():
            if field_name not in decoded:
                continue
            enc_value, enc_conf = decoded[field_name]
            parser_value = getattr(entity, mapped_attr)
            resolved = self._resolver.resolve(field_name, parser_value, enc_value, enc_conf)
            setattr(entity, mapped_attr, resolved.value)
            entity.ontology[f"_resolution.{field_name}"] = {
                "value": resolved.value,
                "confidence": resolved.confidence,
                "source": resolved.source,
                "conflict": resolved.conflict,
                "parser_value": resolved.parser_value,
                "encoder_value": resolved.encoder_value,
            }

        for head, (value, confidence) in decoded.items():
            if head in _DIRECT_MAP:
                continue
            entity.ontology[head] = {
                "value": value,
                "confidence": confidence,
                "source": "EntityEncoder",
            }

        entity.ontology["_grounded"] = True
        return entity

    def resolve_entities(self, entities: list[Entity]) -> list[Entity]:
        """Sequential per-entity resolution.

        NOTE: this is NOT the batched-tensor version the review asked for
        (single forward pass over a padded batch) — that requires handling
        variable-length padding and mapping batch outputs back to entities,
        which is real additional work, not a one-line change. Deferred to
        Priority 2 as originally scoped. What IS implemented here is the
        cheap win: repeated labels within the same call (or across calls)
        hit the LRU cache and skip the forward pass entirely, which covers
        the "car car car" case the review specifically raised without
        needing the batching machinery.
        """
        return [self.resolve(e) for e in entities]
