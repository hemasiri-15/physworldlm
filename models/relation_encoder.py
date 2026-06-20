"""
models/relation_encoder.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — second learned component.

Architecture
────────────
    EntityEncoder.get_embedding()
        e1 : (B, 512)        e2 : (B, 512)
              │                    │
              └────────┬───────────┘
                       ▼
            build_pair_features()
      [e1 | e2 | |e1-e2| | e1*e2 | cos | dot]      (B, ~2050)
                       │
                       ▼
              PairProjector  (MLP, residual)        (B, 512)
                       │
                       ▼
         CrossAttention(e1, e2)  [optional]         (B, 512), attn weights
                       │
                       ▼
        TransformerEncoderLayer × num_transformer_layers
                       │
                       ▼
          relation_embedding  (L2-normalised)        (B, 512)
                       │
        ┌──────────────┼──────────────────────────────┐
        ▼              ▼                               ▼
   spatial_head   contact_head   ...   interaction_head (multi-label)
   (B, n_cls)     (B, n_cls)           (B, n_labels)

The 512-d ``relation_embedding`` is the canonical pairwise relation
representation.  It is designed to be consumed downstream by:

    GraphBuilder        → scene graph edge features
    TemporalWorldModel   → relation-conditioned dynamics
    Trajectory Engine    → interaction priors
    Physics Simulator    → contact / support / collision priors
    Renderer             → relation-aware composition

Design principles (mirrors models/entity_encoder.py)
─────────────────────────────────────────────────────
* RelationEncoderConfig  — zero magic numbers; all dimensions configurable.
* ProjectionBlock        — reused, identical residual projection unit.
* HeadConfig             — typed descriptor for each relation head
                           (name, num_classes, task_type).
* nn.ModuleDict heads    — clean registry, no eight named attributes.
* RelationEncoderOutput  — typed output container.
* Freeze stages          — freeze_backbone / freeze_heads / unfreeze variants
                           ("backbone" here means pair-projector + cross-
                           attention + transformer stack, i.e. everything
                           upstream of the heads).
* parameter_groups()     — differential-LR groups for an external optimizer.
* Xavier init            — all Linear layers.
* Gradient checkpointing — enable_gradient_checkpointing() / disable_*().
* save_pretrained / load_pretrained — torch.save/load-compatible.

Scope discipline
─────────────────
This module implements *only* the pairwise relation encoder described in
PHYSWORLDLM FILE 6.  It deliberately does NOT implement: training loops,
GraphBuilder, TemporalWorldModel, physics simulation, contrastive losses,
uncertainty estimation, a relation memory bank, or N-body / set-level
reasoning.  Those are real, valuable directions (see "FUTURE EXTENSION
SEAMS" below) but adding them here would mean inventing relation taxonomy
and training machinery that isn't part of this file's contract, and would
risk silently diverging from what GraphBuilder and friends will assume.
Every extension point below is structured so it can be added later via
composition, without modifying the public API of this file.

FUTURE EXTENSION SEAMS (documented, intentionally NOT implemented here)
─────────────────────────────────────────────────────────────────────────
* N-body / set reasoning : ``encode_pair()`` is a thin wrapper around
  ``_build_pair_features()``; an ``encode_set()`` for k-way entity tuples
  can be added later by calling the same cross-attention + transformer
  stack on a (B, k, 512) stack instead of a (B, 2, 512) stack.
* Temporal / physics heads : the head registry is a plain ``nn.ModuleDict``
  keyed off ``RelationEncoderConfig.head_configs``; new heads (e.g.
  ``before/after``, ``collision_possible``) are added by appending
  ``HeadConfig`` entries — no change to forward() is required.
* Confidence / uncertainty : ``RelationEncoderOutput`` has room to grow an
  optional ``head_confidence: Dict[str, Tensor]`` field; the natural place
  to populate it is a small auxiliary linear layer per head, added without
  touching the backbone.
* Relation memory bank / retrieval : ``nearest_relations()`` already
  implements the lookup half; a ``RelationMemoryBank`` wrapper can sit
  outside this module and call ``encode_pair()`` + ``nearest_relations()``.
* Contrastive relation space : would be a *training-time* loss (triplet /
  InfoNCE) applied to ``relation_embedding`` — no architectural change
  needed here, since the embedding is already L2-normalised and exposed.
* Multi-scale transformer / hierarchy : ``num_transformer_layers`` is a
  config field; swapping in a deeper or hierarchical stack is a drop-in
  change to ``self.transformer`` only.

No training, no GraphBuilder, no TemporalWorldModel, no Bullet, no
renderer, no WorldSpec generation.  Only models/relation_encoder.py.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_DIM: int = 512

TaskType = Literal["single", "multi"]

#: Spatial relation classes.
SPATIAL_CLASSES: Tuple[str, ...] = (
    "none",
    "above", "below",
    "left_of", "right_of",
    "in_front_of", "behind",
    "inside", "contains",
    "intersecting",
    "near", "far",
    "overlapping", "touching",
    "surrounding",
)

#: Contact relation classes.
CONTACT_CLASSES: Tuple[str, ...] = (
    "none",
    "touching", "resting_on", "attached_to",
    "colliding_with",
    "penetrating",
    "grasping",
    "stacked_on",
    "supported_by",
)

#: Interaction relation classes (multi-label).
INTERACTION_CLASSES: Tuple[str, ...] = (
    "none",
    "holding",
    "sitting_on",
    "standing_on",
    "wearing",
    "using",
    "pulling",
    "pushing",
    "throwing",
    "carrying",
    "driving",
    "riding",
    "opening",
    "closing",
    "cutting",
    "hitting",
)

#: Motion relation classes.
MOTION_CLASSES: Tuple[str, ...] = (
    "none",
    "approaching",
    "moving_away",
    "orbiting",
    "following",
    "leading",
    "rotating_around",
    "crossing",
    "overtaking",
)

#: Support relation classes.
SUPPORT_CLASSES: Tuple[str, ...] = (
    "none",
    "supports",
    "supported_by",
    "hanging_from",
    "attached_to",
    "balanced_on",
)

#: Containment relation classes.
CONTAINMENT_CLASSES: Tuple[str, ...] = (
    "none",
    "inside",
    "contains",
    "enclosed_by",
    "part_of",
    "has_part",
)

#: Visibility relation classes.
VISIBILITY_CLASSES: Tuple[str, ...] = (
    "none",
    "visible_to",
    "occluded_by",
    "blocking",
    "facing",
)

#: Causality relation classes.
CAUSALITY_CLASSES: Tuple[str, ...] = (
    "none",
    "causes",
    "affected_by",
    "triggering",
    "producing",
    "destroying",
)

#: Single-label heads — CrossEntropyLoss targets, mutually-exclusive classes.
SINGLE_LABEL_RELATION_HEADS: Tuple[str, ...] = (
    "spatial",
    "contact",
    "motion",
    "support",
    "containment",
    "visibility",
    "causality",
)

#: Multi-label heads — BCEWithLogitsLoss targets, co-occurring classes.
MULTI_LABEL_RELATION_HEADS: Tuple[str, ...] = (
    "interaction",
)

#: head_name -> ordered class tuple, used to derive default num_classes.
RELATION_CLASS_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "spatial":      SPATIAL_CLASSES,
    "contact":      CONTACT_CLASSES,
    "interaction":  INTERACTION_CLASSES,
    "motion":       MOTION_CLASSES,
    "support":      SUPPORT_CLASSES,
    "containment":  CONTAINMENT_CLASSES,
    "visibility":   VISIBILITY_CLASSES,
    "causality":    CAUSALITY_CLASSES,
}

ALL_RELATION_HEADS: Tuple[str, ...] = (
    *SINGLE_LABEL_RELATION_HEADS,
    *MULTI_LABEL_RELATION_HEADS,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  HeadConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadConfig:
    """Typed descriptor for a single relation output head.

    Attributes:
        name:         Head name (e.g. "spatial", "interaction").
        num_classes:  Output dimension (classes for single-label heads,
                      label-vocabulary size for multi-label heads).
        loss_type:    ``"cross_entropy"`` for single-label heads (mutually
                      exclusive classes); ``"bce"`` for multi-label heads
                      (independent co-occurring labels, e.g. a person can
                      simultaneously hold, carry, and push).
        task_type:    ``"single"`` or ``"multi"`` — mirrors loss_type, kept
                      separate so downstream code can branch on task shape
                      without string-matching the loss name.
    """

    name:        str
    num_classes: int
    loss_type:   Literal["cross_entropy", "bce"] = "cross_entropy"
    task_type:   TaskType = "single"

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError(
                f"HeadConfig({self.name!r}): num_classes must be ≥ 1, "
                f"got {self.num_classes}"
            )
        expected_loss = "bce" if self.task_type == "multi" else "cross_entropy"
        if self.loss_type != expected_loss:
            raise ValueError(
                f"HeadConfig({self.name!r}): task_type={self.task_type!r} "
                f"implies loss_type={expected_loss!r}, got {self.loss_type!r}"
            )


def _default_head_configs() -> Dict[str, HeadConfig]:
    """Build the default head_configs dict from the relation class registry."""
    configs: Dict[str, HeadConfig] = {}
    for name in SINGLE_LABEL_RELATION_HEADS:
        configs[name] = HeadConfig(
            name=name,
            num_classes=len(RELATION_CLASS_REGISTRY[name]),
            loss_type="cross_entropy",
            task_type="single",
        )
    for name in MULTI_LABEL_RELATION_HEADS:
        configs[name] = HeadConfig(
            name=name,
            num_classes=len(RELATION_CLASS_REGISTRY[name]),
            loss_type="bce",
            task_type="multi",
        )
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  RelationEncoderConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelationEncoderConfig:
    """All hyperparameters for RelationEncoder — no magic numbers inside the model.

    Attributes:
        entity_dim:            Dimension of each input entity embedding
                               (output of EntityEncoder.get_embedding()).
        pair_dim:              Target dimension of the concatenated pair
                               feature vector before the final pair-feature
                               projection (kept for spec compatibility; the
                               actual concatenated width is computed from
                               ``entity_dim`` and may exceed this — see
                               :meth:`RelationEncoder.pair_feature_dim`).
        hidden_dim:            Width of intermediate ProjectionBlocks.
        relation_dim:          Dimension of the final relation_embedding.
        num_attention_heads:   Heads for the entity1↔entity2 cross-attention.
        num_transformer_layers: Number of TransformerEncoderLayers stacked
                               after cross-attention for relation reasoning.
        dropout:               Dropout probability throughout the model.
        activation:            ``"gelu"`` or ``"relu"``.
        layer_norm:            Insert LayerNorm inside ProjectionBlocks.
        residual:              Add residual connections where shapes allow.
        normalize_embedding:   L2-normalise the final relation_embedding.
        cross_attention:       Enable the optional entity1↔entity2
                               cross-attention block.  When False, the
                               pair embedding is fed directly to the
                               transformer stack.
        head_configs:          Dict mapping head name → :class:`HeadConfig`.
                               Defaults to the eight relation categories
                               defined in this module (spatial, contact,
                               interaction, motion, support, containment,
                               visibility, causality).
        head_hidden_dim:       Width of the hidden layer inside each head
                               (512 → head_hidden_dim → num_classes).
        head_dropout:          Dropout probability inside each head.
    """

    entity_dim:             int   = ENTITY_DIM
    pair_dim:                int   = 1024
    hidden_dim:               int   = 512
    relation_dim:             int   = 512
    num_attention_heads:      int   = 8
    num_transformer_layers:   int   = 2
    dropout:                  float = 0.1
    activation:               str   = "gelu"
    layer_norm:               bool  = True
    residual:                 bool  = True
    normalize_embedding:      bool  = True
    cross_attention:          bool  = True
    head_configs:             Dict[str, HeadConfig] = field(
        default_factory=_default_head_configs
    )
    head_hidden_dim:          int   = 256
    head_dropout:             float = 0.1

    def __post_init__(self) -> None:
        if self.entity_dim < 1:
            raise ValueError(f"entity_dim must be ≥ 1, got {self.entity_dim}")
        if self.relation_dim < 1:
            raise ValueError(f"relation_dim must be ≥ 1, got {self.relation_dim}")
        if self.num_attention_heads < 1:
            raise ValueError(
                f"num_attention_heads must be ≥ 1, got {self.num_attention_heads}"
            )
        if self.relation_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"relation_dim ({self.relation_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_transformer_layers < 0:
            raise ValueError(
                "num_transformer_layers must be ≥ 0, got "
                f"{self.num_transformer_layers}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError(
                f"head_dropout must be in [0, 1), got {self.head_dropout}"
            )
        if self.activation not in ("gelu", "relu"):
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {self.activation!r}"
            )
        if not self.head_configs:
            raise ValueError("head_configs must not be empty")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  RelationEncoderOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelationEncoderOutput:
    """Typed container for the output of ``RelationEncoder.forward()``.

    Attributes:
        pair_embedding:      Raw concatenated pair feature vector, shape
                             ``(B, pair_feature_dim)`` — see
                             :meth:`RelationEncoder.pair_feature_dim`.
        relation_embedding:  Final shared relation representation, shape
                             ``(B, relation_dim)``.  L2-normalised when
                             ``config.normalize_embedding=True``.  This is
                             the canonical edge feature for GraphBuilder /
                             TemporalWorldModel.
        spatial_logits:      ``(B, n_spatial)`` raw logits.
        contact_logits:      ``(B, n_contact)`` raw logits.
        interaction_logits:  ``(B, n_interaction)`` raw logits (multi-label;
                             apply sigmoid, not softmax).
        motion_logits:       ``(B, n_motion)`` raw logits.
        support_logits:      ``(B, n_support)`` raw logits.
        containment_logits:  ``(B, n_containment)`` raw logits.
        visibility_logits:   ``(B, n_visibility)`` raw logits.
        causality_logits:    ``(B, n_causality)`` raw logits.
        attention_weights:   Cross-attention weights between entity1 and
                             entity2, shape ``(B, num_heads, 1, 1)``-derived
                             tensor — concretely ``(B, num_heads, 2, 2)``
                             since each entity is treated as a single-token
                             sequence; ``None`` when ``cross_attention=False``.

    No softmax / sigmoid is applied to any logits — the training script
    applies the appropriate loss function per head type (CrossEntropyLoss
    for single-label heads, BCEWithLogitsLoss for ``interaction``).
    """

    pair_embedding:     torch.Tensor
    relation_embedding: torch.Tensor
    spatial_logits:     torch.Tensor
    contact_logits:     torch.Tensor
    interaction_logits: torch.Tensor
    motion_logits:      torch.Tensor
    support_logits:     torch.Tensor
    containment_logits: torch.Tensor
    visibility_logits:  torch.Tensor
    causality_logits:   torch.Tensor
    attention_weights:  Optional[torch.Tensor] = None

    _LOGIT_KEYS: ClassVar[Tuple[str, ...]] = (
        "spatial", "contact", "interaction", "motion",
        "support", "containment", "visibility", "causality",
    )

    def logits_dict(self) -> Dict[str, torch.Tensor]:
        """Return all head logits as a plain dict keyed by head name."""
        return {
            "spatial":     self.spatial_logits,
            "contact":     self.contact_logits,
            "interaction": self.interaction_logits,
            "motion":      self.motion_logits,
            "support":     self.support_logits,
            "containment": self.containment_logits,
            "visibility":  self.visibility_logits,
            "causality":   self.causality_logits,
        }

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dict-style access: ``output["spatial"]``."""
        if key in ("pair_embedding", "relation_embedding", "attention_weights"):
            value = getattr(self, key)
            if value is None:
                raise KeyError(f"{key!r} is None for this forward pass")
            return value
        logits = self.logits_dict()
        if key in logits:
            return logits[key]
        raise KeyError(f"Unknown RelationEncoderOutput key: {key!r}")

    def keys(self) -> List[str]:
        """Return all available output keys."""
        base = ["pair_embedding", "relation_embedding"]
        if self.attention_weights is not None:
            base.append("attention_weights")
        return base + list(self._LOGIT_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  ProjectionBlock  (identical to entity_encoder.py)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionBlock(nn.Module):
    """Single residual projection unit used throughout RelationEncoder.

    Structure
    ─────────
        x  ─────────────────────────────────────────────────────┐
        │                                                        │ (residual)
        ▼                                                        │
    Linear(in_dim → out_dim)                                    │
        ▼                                                        │
    LayerNorm(out_dim)  [optional]                              │
        ▼                                                        │
    Activation (GELU / ReLU)                                    │
        ▼                                                        │
    Dropout(p)                                                  │
        ▼                                                        │
    (+ skip connection if in_dim == out_dim and use_residual)   ┘
        ▼
    output  (B, out_dim)

    Args:
        in_dim:        Input feature dimension.
        out_dim:       Output feature dimension.
        dropout:       Dropout probability.
        use_layernorm: Insert LayerNorm after the linear projection.
        use_residual:  Add residual when ``in_dim == out_dim``.
        activation:    ``"gelu"`` or ``"relu"``.
    """

    def __init__(
        self,
        in_dim:        int,
        out_dim:       int,
        dropout:       float = 0.1,
        use_layernorm: bool  = True,
        use_residual:  bool  = True,
        activation:    str   = "gelu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.act    = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop   = nn.Dropout(p=dropout)
        self.use_residual = use_residual and (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) → (B, out_dim)
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  RelationHead
# ─────────────────────────────────────────────────────────────────────────────

class RelationHead(nn.Module):
    """Single output head: relation_dim → head_hidden_dim → num_classes.

    Structure (per spec)::

        512 → 256 → num_classes
        with LayerNorm, GELU, Dropout on the hidden layer.

    Args:
        in_dim:       Input dimension (relation_dim).
        hidden_dim:   Hidden layer width (256 by default).
        num_classes:  Output dimension.
        dropout:      Dropout probability before the hidden layer.
        activation:   ``"gelu"`` or ``"relu"``.
    """

    def __init__(
        self,
        in_dim:      int,
        hidden_dim:  int,
        num_classes: int,
        dropout:     float = 0.1,
        activation:  str   = "gelu",
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) → (B, num_classes)
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  RelationEncoder
# ─────────────────────────────────────────────────────────────────────────────

class RelationEncoder(nn.Module):
    """Pairwise relation encoder for PhysWorldLM.

    Converts a pair of 512-d entity embeddings (produced by
    ``EntityEncoder.get_embedding()``) into a 512-d relation embedding and
    simultaneously predicts eight relation categories via dedicated output
    heads (seven single-label, one multi-label).

    The ``relation_embedding`` is the canonical pairwise relation
    representation; it is L2-normalised (when
    ``config.normalize_embedding=True``) to enable cosine-similarity
    retrieval over relations, mirroring how ``EntityEncoder.shared_embedding``
    is used for entity retrieval.

    Downstream consumers
    ────────────────────
    * ``GraphBuilder``        — scene graph edge initialisation
    * ``TemporalWorldModel``  — relation-conditioned state transitions
    * ``Trajectory Engine``   — motion/support priors per entity pair
    * ``Physics Simulator``   — contact/support/containment priors
    * Retrieval / ANN search  — relation_embedding as query / key

    Usage
    ─────
    ::

        from models.entity_encoder   import EntityEncoder
        from models.relation_encoder import RelationEncoder, RelationEncoderConfig

        enc = EntityEncoder(...)
        rel = RelationEncoder(RelationEncoderConfig())

        e1 = enc.get_embedding(ids_a, mask_a)   # (B, 512)
        e2 = enc.get_embedding(ids_b, mask_b)   # (B, 512)

        output = rel(e1, e2)
        print(output.relation_embedding.shape)   # (B, 512)
        print(output["spatial"].shape)            # (B, 15)
        print(output["interaction"].shape)        # (B, 16)  multi-label

    Args:
        config: :class:`RelationEncoderConfig` instance.  Defaults are used
                when omitted.
    """

    def __init__(self, config: Optional[RelationEncoderConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = RelationEncoderConfig()
        self.config: RelationEncoderConfig = config

        d = config.entity_dim

        # ── pair feature dimension ────────────────────────────────────────────
        # concat(e1, e2, |e1-e2|, e1*e2) = 4d, plus cosine similarity and dot
        # product scalars (2 extra dims) = 4d + 2.
        self._pair_feature_dim: int = 4 * d + 2

        # ── pair projector: pair_feature_dim → hidden_dim → relation_dim ──────
        self.pair_projector: nn.Sequential = nn.Sequential(
            ProjectionBlock(
                in_dim=self._pair_feature_dim,
                out_dim=config.hidden_dim,
                dropout=config.dropout,
                use_layernorm=config.layer_norm,
                use_residual=False,  # dims differ, no residual on first block
                activation=config.activation,
            ),
            ProjectionBlock(
                in_dim=config.hidden_dim,
                out_dim=config.relation_dim,
                dropout=config.dropout,
                use_layernorm=config.layer_norm,
                use_residual=config.residual,
                activation=config.activation,
            ),
        )

        # ── entity1 / entity2 projection to relation_dim (for cross-attn) ─────
        # Cross-attention operates in relation_dim space so it can be fused
        # with the pair projection output regardless of entity_dim.
        self.entity_proj: nn.Linear = nn.Linear(d, config.relation_dim)

        # ── cross attention between entity1 and entity2 ────────────────────────
        self._use_cross_attention: bool = config.cross_attention
        if config.cross_attention:
            self.cross_attention: nn.MultiheadAttention = nn.MultiheadAttention(
                embed_dim=config.relation_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.cross_attn_norm: nn.LayerNorm = nn.LayerNorm(config.relation_dim)
            # Fuse [pair_proj | cross_attn_pooled] -> relation_dim
            self.fusion: nn.Linear = nn.Linear(
                config.relation_dim * 2, config.relation_dim
            )
        else:
            self.cross_attention = None  # type: ignore[assignment]
            self.cross_attn_norm = None  # type: ignore[assignment]
            self.fusion = None  # type: ignore[assignment]

        # ── transformer stack for relation reasoning ───────────────────────────
        self._num_transformer_layers: int = config.num_transformer_layers
        if config.num_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.relation_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.relation_dim * 4,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_transformer_layers
            )
        else:
            self.transformer = None  # type: ignore[assignment]

        # ── final relation embedding norm ──────────────────────────────────────
        self.output_norm: nn.LayerNorm = nn.LayerNorm(config.relation_dim)

        # ── head registry ───────────────────────────────────────────────────────
        self.head_configs: Dict[str, HeadConfig] = dict(config.head_configs)
        self.heads: nn.ModuleDict = nn.ModuleDict()
        for name, hcfg in self.head_configs.items():
            self.heads[name] = RelationHead(
                in_dim=config.relation_dim,
                hidden_dim=config.head_hidden_dim,
                num_classes=hcfg.num_classes,
                dropout=config.head_dropout,
                activation=config.activation,
            )

        # ── weight init ──────────────────────────────────────────────────────────
        self._init_weights()

        # ── gradient checkpointing flag ─────────────────────────────────────────
        self._gradient_checkpointing: bool = False

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform init for all Linear layers in this module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    # ── validation helpers ───────────────────────────────────────────────────

    def _validate_inputs(self, e1: torch.Tensor, e2: torch.Tensor) -> None:
        """Validate batch sizes, embedding dimensions, and NaNs.

        Raises:
            ValueError: On shape mismatch, wrong dimensionality, or empty
                        tensors.
            RuntimeError: On NaN / Inf values in either input.
        """
        if e1.dim() != 2 or e2.dim() != 2:
            raise ValueError(
                "RelationEncoder expects 2-D tensors (B, entity_dim); got "
                f"e1.dim()={e1.dim()}, e2.dim()={e2.dim()}"
            )
        if e1.shape[0] == 0 or e2.shape[0] == 0:
            raise ValueError(
                "RelationEncoder received an empty batch "
                f"(e1.shape={tuple(e1.shape)}, e2.shape={tuple(e2.shape)})"
            )
        if e1.shape[0] != e2.shape[0]:
            raise ValueError(
                f"Batch size mismatch: e1 has {e1.shape[0]} rows, "
                f"e2 has {e2.shape[0]} rows"
            )
        if e1.shape[1] != self.config.entity_dim or e2.shape[1] != self.config.entity_dim:
            raise ValueError(
                f"Expected entity_dim={self.config.entity_dim}, got "
                f"e1.shape[1]={e1.shape[1]}, e2.shape[1]={e2.shape[1]}"
            )
        if torch.isnan(e1).any() or torch.isinf(e1).any():
            raise RuntimeError("RelationEncoder: NaN/Inf detected in e1")
        if torch.isnan(e2).any() or torch.isinf(e2).any():
            raise RuntimeError("RelationEncoder: NaN/Inf detected in e2")

    # ── pair feature construction ────────────────────────────────────────────

    def pair_feature_dim(self) -> int:
        """Return the dimension of the raw concatenated pair feature vector."""
        return self._pair_feature_dim

    def _build_pair_features(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Construct the raw pair feature vector from two entity embeddings.

        Concatenates::

            e1
            e2
            |e1 - e2|
            e1 * e2
            cosine_similarity(e1, e2)   (scalar, broadcast to 1 column)
            dot_product(e1, e2)         (scalar, broadcast to 1 column)

        Args:
            e1: ``(B, entity_dim)``
            e2: ``(B, entity_dim)``

        Returns:
            ``(B, 4*entity_dim + 2)`` raw pair feature vector.
        """
        diff = torch.abs(e1 - e2)
        prod = e1 * e2

        cos_sim = F.cosine_similarity(e1, e2, dim=-1, eps=1e-8).unsqueeze(-1)  # (B,1)
        dot = (e1 * e2).sum(dim=-1, keepdim=True)                              # (B,1)

        return torch.cat([e1, e2, diff, prod, cos_sim, dot], dim=-1)

    # ── cross attention ───────────────────────────────────────────────────────

    def _cross_attend(
        self,
        e1_proj: torch.Tensor,
        e2_proj: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run cross-attention treating entity1/entity2 as a 2-token sequence.

        Args:
            e1_proj: ``(B, relation_dim)``
            e2_proj: ``(B, relation_dim)``

        Returns:
            Tuple of:
                pooled:            ``(B, relation_dim)`` mean-pooled attended
                                   representation.
                attention_weights: ``(B, num_heads, 2, 2)`` or ``None`` if
                                   cross-attention is disabled.
        """
        if not self._use_cross_attention:
            return (e1_proj + e2_proj) * 0.5, None

        # Treat [e1, e2] as a sequence of length 2: (B, 2, relation_dim)
        seq = torch.stack([e1_proj, e2_proj], dim=1)

        attended, attn_weights = self.cross_attention(
            seq, seq, seq,
            need_weights=True,
            average_attn_weights=False,
        )  # attended: (B, 2, relation_dim); attn_weights: (B, num_heads, 2, 2)

        attended = self.cross_attn_norm(attended + seq)  # residual + norm
        pooled = attended.mean(dim=1)                     # (B, relation_dim)

        return pooled, attn_weights

    # ── transformer reasoning ─────────────────────────────────────────────────

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Run the relation-reasoning transformer stack on a single token.

        Args:
            x: ``(B, relation_dim)``

        Returns:
            ``(B, relation_dim)``
        """
        if self.transformer is None or self._num_transformer_layers == 0:
            return x

        if self._gradient_checkpointing and self.training:
            def _ckpt_fn(inp: torch.Tensor) -> torch.Tensor:
                seq_in = inp.unsqueeze(1)  # (B, 1, relation_dim)
                return self.transformer(seq_in)

            seq_out = torch.utils.checkpoint.checkpoint(
                _ckpt_fn, x, use_reentrant=False
            )
        else:
            seq_in = x.unsqueeze(1)             # (B, 1, relation_dim)
            seq_out = self.transformer(seq_in)  # (B, 1, relation_dim)

        return seq_out.squeeze(1)               # (B, relation_dim)

    # ── public API ────────────────────────────────────────────────────────────

    def encode_pair(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Build the raw pair feature vector (no projection, no attention).

        This is the lowest-level public hook — useful for inspecting the
        hand-crafted pair features in isolation.

        Args:
            e1: ``(B, entity_dim)``
            e2: ``(B, entity_dim)``

        Returns:
            ``(B, pair_feature_dim)`` raw concatenated features.
        """
        self._validate_inputs(e1, e2)
        return self._build_pair_features(e1, e2)

    def encode_relation(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Run the full backbone (pair features → relation_embedding), no heads.

        This is the primary interface for GraphBuilder / TemporalWorldModel
        when only the relation embedding is needed (e.g. as an edge feature),
        without paying for the head forward passes.

        Args:
            e1: ``(B, entity_dim)``
            e2: ``(B, entity_dim)``
            return_attention: If True, also return the cross-attention
                              weights (``None`` when cross-attention is
                              disabled).

        Returns:
            ``(B, relation_dim)`` relation embedding, or a tuple of
            ``(relation_embedding, attention_weights)`` when
            ``return_attention=True``.
        """
        self._validate_inputs(e1, e2)
        relation_embedding, attn = self._backbone_forward(e1, e2)
        if return_attention:
            return relation_embedding, attn
        return relation_embedding

    def get_relation_embedding(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode_relation` without attention — matches the
        naming convention used by ``EntityEncoder.get_embedding()``."""
        return self.encode_relation(e1, e2, return_attention=False)  # type: ignore[return-value]

    def _backbone_forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Shared backbone: pair features → projector → cross-attn → transformer.

        Returns:
            Tuple of (relation_embedding ``(B, relation_dim)``,
            attention_weights or None).
        """
        pair_features = self._build_pair_features(e1, e2)        # (B, 4d+2)
        pair_proj = self.pair_projector(pair_features)            # (B, relation_dim)

        if self._use_cross_attention:
            e1_proj = self.entity_proj(e1)                         # (B, relation_dim)
            e2_proj = self.entity_proj(e2)                         # (B, relation_dim)
            attn_pooled, attn_weights = self._cross_attend(e1_proj, e2_proj)
            fused = self.fusion(torch.cat([pair_proj, attn_pooled], dim=-1))
        else:
            fused = pair_proj
            attn_weights = None

        reasoned = self._transform(fused)                         # (B, relation_dim)
        relation_embedding = self.output_norm(reasoned + fused)    # residual + norm

        if self.config.normalize_embedding:
            relation_embedding = F.normalize(relation_embedding, p=2, dim=-1)

        return relation_embedding, attn_weights

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> RelationEncoderOutput:
        """Full forward pass: pair features → backbone → all heads.

        Args:
            e1: ``(B, entity_dim)`` first entity embedding.
            e2: ``(B, entity_dim)`` second entity embedding.

        Returns:
            :class:`RelationEncoderOutput` with ``pair_embedding``,
            ``relation_embedding``, per-head logits, and
            ``attention_weights``.  All logits are raw — no softmax /
            sigmoid applied.
        """
        self._validate_inputs(e1, e2)

        pair_features = self._build_pair_features(e1, e2)  # (B, 4d+2)
        relation_embedding, attn_weights = self._backbone_forward(e1, e2)

        logits: Dict[str, torch.Tensor] = {
            name: head(relation_embedding) for name, head in self.heads.items()
        }

        return RelationEncoderOutput(
            pair_embedding=pair_features,
            relation_embedding=relation_embedding,
            spatial_logits=logits["spatial"],
            contact_logits=logits["contact"],
            interaction_logits=logits["interaction"],
            motion_logits=logits["motion"],
            support_logits=logits["support"],
            containment_logits=logits["containment"],
            visibility_logits=logits["visibility"],
            causality_logits=logits["causality"],
            attention_weights=attn_weights,
        )

    # ── freeze / unfreeze ─────────────────────────────────────────────────────

    def _backbone_modules(self) -> List[nn.Module]:
        """Return all modules upstream of the heads (the shared 'backbone')."""
        modules: List[nn.Module] = [self.pair_projector, self.entity_proj, self.output_norm]
        if self.cross_attention is not None:
            modules.extend([self.cross_attention, self.cross_attn_norm, self.fusion])
        if self.transformer is not None:
            modules.append(self.transformer)
        return modules

    def freeze_backbone(self) -> None:
        """Freeze the pair-projector, cross-attention, and transformer stack.

        Leaves the per-relation heads trainable — useful when fine-tuning
        only the output heads on a new relation taxonomy.
        """
        for module in self._backbone_modules():
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the pair-projector, cross-attention, and transformer stack."""
        for module in self._backbone_modules():
            for param in module.parameters():
                param.requires_grad = True

    def freeze_heads(self) -> None:
        """Freeze all relation head parameters."""
        for param in self.heads.parameters():
            param.requires_grad = False

    def unfreeze_heads(self) -> None:
        """Unfreeze all relation head parameters."""
        for param in self.heads.parameters():
            param.requires_grad = True

    # ── parameter groups ──────────────────────────────────────────────────────

    def parameter_groups(
        self,
        backbone_lr: float = 5e-5,
        head_lr:     float = 1e-3,
    ) -> List[Dict]:
        """Return parameter groups for differential learning rates.

        Intended usage in an external training script::

            optimizer = torch.optim.AdamW(
                model.parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
            )

        Args:
            backbone_lr: Learning rate for pair-projector / cross-attention /
                        transformer parameters.
            head_lr:     Learning rate for the per-relation head parameters.

        Returns:
            List of two dicts consumable by any ``torch.optim.Optimizer``.
        """
        backbone_params: List[torch.nn.Parameter] = []
        for module in self._backbone_modules():
            backbone_params.extend(module.parameters())
        head_params = list(self.heads.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": head_lr,     "name": "heads"},
        ]

    # ── gradient checkpointing ────────────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the transformer reasoning stack.

        Trades compute for memory: activations are recomputed during the
        backward pass rather than stored.  Call before the training loop.
        """
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    # ── similarity / retrieval ────────────────────────────────────────────────

    def cosine_similarity(
        self,
        rel_a: torch.Tensor,
        rel_b: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between two relation-embedding matrices.

        Args:
            rel_a: ``(B, D)`` or ``(D,)`` relation embedding(s).
            rel_b: ``(B, D)`` or ``(D,)`` relation embedding(s).

        Returns:
            Scalar tensor (if 1-D inputs) or ``(B,)`` tensor of similarities
            in ``[-1, 1]``.
        """
        a = F.normalize(rel_a, p=2, dim=-1)
        b = F.normalize(rel_b, p=2, dim=-1)
        return (a * b).sum(dim=-1)

    def nearest_relations(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the ``k`` nearest relation embeddings to ``query`` by cosine
        similarity.

        Useful for relation retrieval / a future ``RelationMemoryBank`` (e.g.
        "find stored relations most similar to 'car approaching truck'").

        Args:
            query:      ``(D,)`` or ``(1, D)`` single query relation
                        embedding.
            candidates: ``(N, D)`` bank of candidate relation embeddings.
            k:          Number of neighbours to return.

        Returns:
            Tuple of:
                values:  ``(k,)`` cosine similarities, descending.
                indices: ``(k,)`` indices into ``candidates``.

        Raises:
            ValueError: If ``candidates`` is empty or ``k`` exceeds the
                        number of candidates.
        """
        if candidates.shape[0] == 0:
            raise ValueError("nearest_relations: candidates is empty")
        if k < 1:
            raise ValueError(f"nearest_relations: k must be ≥ 1, got {k}")
        k = min(k, candidates.shape[0])

        q = query.reshape(1, -1) if query.dim() == 1 else query  # (1, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        c_norm = F.normalize(candidates, p=2, dim=-1)

        sims = (q_norm @ c_norm.t()).squeeze(0)  # (N,)
        values, indices = torch.topk(sims, k=k, largest=True, sorted=True)
        return values, indices

    # ── info / introspection API ──────────────────────────────────────────────

    def get_head_dimensions(self) -> Dict[str, int]:
        """Return head name → output dimension for all registered heads.

        Inferred directly from the head Linear layer weights — guaranteed to
        be consistent with the actual model.
        """
        return {name: head.fc2.out_features for name, head in self.heads.items()}

    def get_head_task_types(self) -> Dict[str, TaskType]:
        """Return head name → task_type ("single" or "multi")."""
        return {name: cfg.task_type for name, cfg in self.head_configs.items()}

    def get_relation_classes(self, head_name: str) -> Tuple[str, ...]:
        """Return the ordered class-name tuple for ``head_name``.

        Args:
            head_name: One of the eight registered relation heads.

        Raises:
            KeyError: If ``head_name`` is not a registered head.
        """
        if head_name not in RELATION_CLASS_REGISTRY:
            raise KeyError(
                f"Unknown relation head {head_name!r}; expected one of "
                f"{list(RELATION_CLASS_REGISTRY.keys())}"
            )
        return RELATION_CLASS_REGISTRY[head_name]

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters.

        Returns:
            Dict with keys ``"total"`` and ``"trainable"``.
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def print_model_summary(self) -> None:
        """Print a human-readable model summary to stdout."""
        cfg   = self.config
        dims  = self.get_head_dimensions()
        param = self.count_parameters()

        print(f"\n{'═'*64}")
        print("  PhysWorldLM — RelationEncoder")
        print(f"{'═'*64}")
        print(f"  Entity dim          : {cfg.entity_dim}")
        print(f"  Pair feature dim    : {self.pair_feature_dim()}")
        print(f"  Hidden dim          : {cfg.hidden_dim}")
        print(f"  Relation dim        : {cfg.relation_dim}")
        print(f"  Attention heads     : {cfg.num_attention_heads}")
        print(f"  Transformer layers  : {cfg.num_transformer_layers}")
        print(f"  Cross-attention     : {cfg.cross_attention}")
        print(f"  Activation          : {cfg.activation}")
        print(f"  Dropout             : {cfg.dropout}")
        print(f"  LayerNorm           : {cfg.layer_norm}")
        print(f"  Residual            : {cfg.residual}")
        print(f"  Normalize embedding : {cfg.normalize_embedding}")
        print(f"  Grad checkpointing  : {self._gradient_checkpointing}")
        print()

        print("  Relation heads")
        print(f"  {'─'*42}")
        for name, hcfg in self.head_configs.items():
            n = dims.get(name, "—")
            print(f"    {name:<14}  →  {n:>3} classes   [{hcfg.task_type}/{hcfg.loss_type}]")

        print()
        print(f"  Parameters")
        print(f"  {'─'*42}")
        print(f"    Total      : {param['total']:>12,}")
        print(f"    Trainable  : {param['trainable']:>12,}")
        print(f"    Frozen     : {param['total'] - param['trainable']:>12,}")
        print(f"{'═'*64}\n")

    # ── save / load ───────────────────────────────────────────────────────────

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model weights and config to ``path`` via ``torch.save``.

        Args:
            path: Destination file path (e.g. "checkpoints/relation_encoder.pt").
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "config": {
                "entity_dim":             self.config.entity_dim,
                "pair_dim":               self.config.pair_dim,
                "hidden_dim":             self.config.hidden_dim,
                "relation_dim":           self.config.relation_dim,
                "num_attention_heads":    self.config.num_attention_heads,
                "num_transformer_layers": self.config.num_transformer_layers,
                "dropout":                self.config.dropout,
                "activation":             self.config.activation,
                "layer_norm":             self.config.layer_norm,
                "residual":               self.config.residual,
                "normalize_embedding":    self.config.normalize_embedding,
                "cross_attention":        self.config.cross_attention,
                "head_hidden_dim":        self.config.head_hidden_dim,
                "head_dropout":           self.config.head_dropout,
                "head_configs": {
                    name: {
                        "num_classes": hcfg.num_classes,
                        "loss_type":   hcfg.loss_type,
                        "task_type":   hcfg.task_type,
                    }
                    for name, hcfg in self.head_configs.items()
                },
            },
        }
        torch.save(state, path)
        print(f"[RelationEncoder] saved → {path}")

    @classmethod
    def load_pretrained(
        cls,
        path: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> "RelationEncoder":
        """Load a :class:`RelationEncoder` previously saved with
        :meth:`save_pretrained`.

        Args:
            path:         Path to the saved checkpoint.
            map_location: Passed through to ``torch.load`` (e.g. "cpu").

        Returns:
            A reconstructed :class:`RelationEncoder` with weights loaded.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"RelationEncoder checkpoint not found: {path}")

        state = torch.load(path, map_location=map_location or "cpu")
        cfg_dict = dict(state["config"])

        head_configs_raw = cfg_dict.pop("head_configs")
        head_configs = {
            name: HeadConfig(
                name=name,
                num_classes=hc["num_classes"],
                loss_type=hc["loss_type"],
                task_type=hc["task_type"],
            )
            for name, hc in head_configs_raw.items()
        }

        config = RelationEncoderConfig(head_configs=head_configs, **cfg_dict)
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        print(f"[RelationEncoder] loaded ← {path}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  debug_forward()
# ─────────────────────────────────────────────────────────────────────────────

def debug_forward(model: RelationEncoder, batch_size: int = 2) -> RelationEncoderOutput:
    """Run one synthetic forward pass and print all output shapes.

    Generates random entity-embedding pairs (no EntityEncoder / dataset
    required) to exercise the full pipeline in isolation.

    Args:
        model:      A :class:`RelationEncoder` instance.
        batch_size: Number of synthetic pairs to generate (default 2).

    Returns:
        The :class:`RelationEncoderOutput` from the synthetic forward pass.
    """
    device = next(model.parameters()).device
    d = model.config.entity_dim

    e1 = F.normalize(torch.randn(batch_size, d, device=device), p=2, dim=-1)
    e2 = F.normalize(torch.randn(batch_size, d, device=device), p=2, dim=-1)

    model.eval()
    with torch.no_grad():
        output = model(e1, e2)

    print(f"\n{'─'*52}")
    print("  debug_forward()  output shapes")
    print(f"{'─'*52}")
    print(f"  {'pair_embedding':<20}  {tuple(output.pair_embedding.shape)}")
    print(f"  {'relation_embedding':<20}  {tuple(output.relation_embedding.shape)}")
    for name in ALL_RELATION_HEADS:
        logit = output[name]
        print(f"  {name:<20}  {tuple(logit.shape)}")
    if output.attention_weights is not None:
        print(f"  {'attention_weights':<20}  {tuple(output.attention_weights.shape)}")
    print(f"{'─'*52}\n")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: build encoder, print summary, run debug forward pass."""

    print("=" * 64)
    print("  PhysWorldLM — RelationEncoder smoke-test")
    print("=" * 64)

    config = RelationEncoderConfig(
        entity_dim=512,
        pair_dim=1024,
        hidden_dim=512,
        relation_dim=512,
        num_attention_heads=8,
        num_transformer_layers=2,
        dropout=0.1,
        activation="gelu",
        layer_norm=True,
        residual=True,
        normalize_embedding=True,
        cross_attention=True,
    )
    model = RelationEncoder(config)

    print(f"\n  Entity dim          : {config.entity_dim}")
    print(f"  Relation dim        : {config.relation_dim}")
    print(f"  Attention heads     : {config.num_attention_heads}")
    print(f"  Transformer layers  : {config.num_transformer_layers}")

    params = model.count_parameters()
    print(f"  Parameter count     : {params['total']:,}")

    print("\n  Head dimensions:")
    for name, dim in model.get_head_dimensions().items():
        print(f"    {name:<14} {dim}")

    model.print_model_summary()
    output = debug_forward(model, batch_size=2)

    # Sanity-check expected shapes per the spec.
    assert output.pair_embedding.shape == (2, model.pair_feature_dim())
    assert output.relation_embedding.shape == (2, config.relation_dim)
    for name in SINGLE_LABEL_RELATION_HEADS:
        expected_n = len(RELATION_CLASS_REGISTRY[name])
        assert output[name].shape == (2, expected_n), name
    for name in MULTI_LABEL_RELATION_HEADS:
        expected_n = len(RELATION_CLASS_REGISTRY[name])
        assert output[name].shape == (2, expected_n), name

    # Similarity / retrieval smoke test.
    bank = F.normalize(torch.randn(16, config.relation_dim), p=2, dim=-1)
    query = output.relation_embedding[0]
    values, indices = model.nearest_relations(query, bank, k=3)
    print(f"  nearest_relations() → values={values.tolist()}  indices={indices.tolist()}")

    sim = model.cosine_similarity(output.relation_embedding[0], output.relation_embedding[1])
    print(f"  cosine_similarity(rel_0, rel_1) = {sim.item():.4f}")

    print("\n[main] all assertions passed. done.")


if __name__ == "__main__":
    main()
