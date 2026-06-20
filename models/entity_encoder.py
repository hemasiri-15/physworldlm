"""
models/entity_encoder.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — first learned component.

Architecture
────────────
    Natural-language entity token  (e.g. "Ferrari", "Mars rover")
        │
        ▼
    sentence-transformers/all-MiniLM-L6-v2     (frozen or fine-tuned)
        │   last_hidden_state  (B, seq, 384)
        ▼
    mean_pool()                                (B, 384)
        │
        ▼
    ProjectionBlock × mlp_depth               (B, 512)
        │   LayerNorm + GELU + Dropout + optional residual
        ▼
    L2-normalised  shared_embedding            (B, 512)
        │
        ├──▶  single-label heads  ×14   logits (B, n_classes)
        │
        └──▶  multi-label  heads  ×3    logits (B, n_labels)

The 512-dimensional shared_embedding is the canonical entity representation.
It is designed to be consumed downstream by:

    RelationEncoder → SceneGraphBuilder → TemporalWorldModel
    Retrieval / nearest-neighbour search
    Open-vocabulary ontology matching
    RAG pipelines

Design principles
─────────────────
* EntityEncoderConfig  — zero magic numbers; all dimensions configurable.
* ProjectionBlock      — reusable residual projection unit.
* HeadConfig           — typed descriptor for each output head.
* nn.ModuleDict heads  — clean registry, no 17 named attributes.
* EntityEncoderOutput  — typed output container.
* Freeze stages        — freeze_backbone / freeze_heads / unfreeze variants.
* Parameter groups     — get_parameter_groups() for differential LR.
* Xavier init          — all Linear layers outside MiniLM.
* Gradient checkpointing placeholder — enable_gradient_checkpointing().
* Attention-weight hook placeholder  — return_attention flag.

No losses, no optimizer, no scheduler, no training loop, no dataloaders,
no inference script.  Only models/entity_encoder.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
BACKBONE_DIM:  int = 384

#: Ordered single-label head names — must match EntityDataset.SINGLE_LABEL_FIELDS.
SINGLE_LABEL_HEADS: Tuple[str, ...] = (
    "entity_type",
    "parent_class",
    "root_class",
    "coarse_class",
    "material",
    "phase",
    "mobility",
    "size_class",
    "shape",
    "mass_class",
    "contact_type",
    "stability",
    "friction_class",
    "restitution_class",
)

#: Ordered multi-label head names — must match EntityDataset.MULTI_LABEL_FIELDS.
MULTI_LABEL_HEADS: Tuple[str, ...] = (
    "capabilities",
    "affordances",
    "scene_roles",
)

TaskType = Literal["single", "multi"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  HeadConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadConfig:
    """Typed descriptor for a single output head.

    Attributes:
        name:         Field name matching EntityDataset's label map key.
        num_classes:  Output dimension (classes for single, vocab size for multi).
        task_type:    ``"single"`` → CrossEntropyLoss target;
                      ``"multi"``  → BCEWithLogitsLoss target.
        class_weights: Optional class-frequency weights for the loss.
                       Populated by the training script, not here.
    """

    name:          str
    num_classes:   int
    task_type:     TaskType = "single"
    class_weights: Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  EntityEncoderConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityEncoderConfig:
    """All hyperparameters for EntityEncoder — no magic numbers inside the model.

    Attributes:
        backbone_name:         HuggingFace model id for the sentence encoder.
        embedding_dim:         Backbone output dimension (384 for MiniLM-L6).
        hidden_dim:            Width of each ProjectionBlock layer.
        projection_dim:        Dimension of the final shared embedding
                               (equals hidden_dim by default).
        dropout:               Dropout probability in ProjectionBlock.
        head_dropout:          Dropout probability before each output head.
        freeze_backbone:       If True, backbone parameters are frozen at init.
        use_layernorm:         Insert LayerNorm inside each ProjectionBlock.
        use_residual:          Add a residual connection in ProjectionBlock
                               when input and output dims match.
        mlp_depth:             Number of stacked ProjectionBlocks.
        normalize_embeddings:  L2-normalise the shared_embedding output.
        activation:            Activation name; only ``"gelu"`` and ``"relu"``
                               are currently supported.
        head_dimensions:       Dict mapping head name → output dimension.
                               Must be provided at construction time, sourced
                               from ``EntityDataset.get_head_dimensions()``.
        return_attention:      Placeholder; when True the forward pass will
                               return backbone attention weights (not yet
                               implemented, reserved for interpretability).
    """

    backbone_name:        str  = BACKBONE_NAME
    embedding_dim:        int  = BACKBONE_DIM
    hidden_dim:           int  = 512
    projection_dim:       int  = 512
    dropout:              float = 0.1
    head_dropout:         float = 0.1
    freeze_backbone:      bool  = False
    use_layernorm:        bool  = True
    use_residual:         bool  = True
    mlp_depth:            int   = 2
    normalize_embeddings: bool  = True
    activation:           str   = "gelu"
    head_dimensions:      Dict[str, int] = field(default_factory=dict)
    return_attention:     bool  = False

    def __post_init__(self) -> None:
        if self.mlp_depth < 1:
            raise ValueError(f"mlp_depth must be ≥ 1, got {self.mlp_depth}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError(f"head_dropout must be in [0, 1), got {self.head_dropout}")
        if self.activation not in ("gelu", "relu"):
            raise ValueError(
                f"activation must be 'gelu' or 'relu', got {self.activation!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  EntityEncoderOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityEncoderOutput:
    """Typed container for the output of ``EntityEncoder.forward()``.

    Attributes:
        shared_embedding: L2-normalised projection, shape ``(B, projection_dim)``.
                          This is the canonical entity representation consumed
                          by RelationEncoder, SceneGraphBuilder, retrieval, etc.
        logits:           Dict of head_name → raw logit tensor.
                          Single-label heads: ``(B, n_classes)``
                          Multi-label heads:  ``(B, n_labels)``
                          No softmax / sigmoid applied — the training script
                          applies the appropriate loss function.
    """

    shared_embedding: torch.Tensor
    logits:           Dict[str, torch.Tensor]

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dict-style access: ``output["entity_type"]``."""
        if key == "shared_embedding":
            return self.shared_embedding
        return self.logits[key]

    def keys(self) -> List[str]:
        """Return all available output keys."""
        return ["shared_embedding"] + list(self.logits.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  ProjectionBlock
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionBlock(nn.Module):
    """Single residual projection unit used inside the shared MLP tower.

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
        in_dim:       Input feature dimension.
        out_dim:      Output feature dimension.
        dropout:      Dropout probability.
        use_layernorm: Insert LayerNorm after the linear projection.
        use_residual: Add residual when ``in_dim == out_dim``.
        activation:   ``"gelu"`` or ``"relu"``.
    """

    def __init__(
        self,
        in_dim:       int,
        out_dim:      int,
        dropout:      float = 0.1,
        use_layernorm: bool = True,
        use_residual: bool  = True,
        activation:   str   = "gelu",
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
# SECTION 6  –  EntityEncoder
# ─────────────────────────────────────────────────────────────────────────────

class EntityEncoder(nn.Module):
    """Multi-task entity encoder for PhysWorldLM.

    Converts entity surface-form tokens into a shared 512-d embedding and
    simultaneously predicts 17 ontology attributes via dedicated output heads.

    The 512-d ``shared_embedding`` is the canonical entity representation; it
    is L2-normalised (when ``config.normalize_embeddings=True``) to enable
    cosine-similarity retrieval and ontology matching without additional
    post-processing.

    Downstream consumers
    ────────────────────
    * ``train_entity_classifier.py`` — multi-task training
    * ``RelationEncoder``            — consumes shared_embedding as node features
    * ``SceneGraphBuilder``          — entity node initialisation
    * ``TemporalWorldModel``         — world state initialisation
    * Retrieval / ANN search         — shared_embedding as query / key
    * Open-vocabulary ontology match — cosine similarity on shared_embedding

    Usage
    ─────
    ::

        from training.entity_dataset import EntityDataset, DatasetConfig
        from models.entity_encoder   import EntityEncoder, EntityEncoderConfig

        ds  = EntityDataset(DatasetConfig(split="train"))
        cfg = EntityEncoderConfig(head_dimensions=ds.get_head_dimensions())
        enc = EntityEncoder(cfg)

        # forward pass (batched)
        output = enc(batch["input_ids"], batch["attention_mask"])
        print(output.shared_embedding.shape)   # (B, 512)
        print(output["entity_type"].shape)     # (B, 19)

        # embedding only (for retrieval)
        emb = enc.encode(batch["input_ids"], batch["attention_mask"])
        # (B, 512)

    Args:
        config: :class:`EntityEncoderConfig` instance.  If omitted, defaults
                are used with an empty ``head_dimensions`` dict — useful only
                for unit-testing the projection tower; you must supply
                ``head_dimensions`` before calling ``forward()``.
    """

    def __init__(self, config: Optional[EntityEncoderConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = EntityEncoderConfig()
        self.config: EntityEncoderConfig = config

        # ── backbone ──────────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        if config.freeze_backbone:
            self._set_backbone_grad(requires_grad=False)

        # ── projection tower (shared MLP) ─────────────────────────────────────
        # First block: embedding_dim (384) → hidden_dim (512); no residual
        # because dims differ.  Subsequent blocks: hidden_dim → hidden_dim
        # with residual when use_residual=True.
        projection_blocks: List[nn.Module] = []
        in_dim = config.embedding_dim
        for i in range(config.mlp_depth):
            out_dim = config.hidden_dim if i < config.mlp_depth - 1 else config.projection_dim
            projection_blocks.append(
                ProjectionBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=config.dropout,
                    use_layernorm=config.use_layernorm,
                    use_residual=config.use_residual,
                    activation=config.activation,
                )
            )
            in_dim = out_dim
        self.projection: nn.Sequential = nn.Sequential(*projection_blocks)

        # ── head dropout (shared pre-head layer) ──────────────────────────────
        self.head_drop: nn.Dropout = nn.Dropout(p=config.head_dropout)

        # ── head registry ─────────────────────────────────────────────────────
        # single_label_heads: CrossEntropyLoss targets
        # multi_label_heads:  BCEWithLogitsLoss targets
        self.single_label_heads: nn.ModuleDict = nn.ModuleDict()
        self.multi_label_heads:  nn.ModuleDict = nn.ModuleDict()

        head_dims = config.head_dimensions
        for name in SINGLE_LABEL_HEADS:
            n = head_dims.get(name, 0)
            if n == 0:
                warnings.warn(
                    f"EntityEncoder: head_dimensions missing key {name!r}; "
                    "skipping head (call forward() only after providing dims).",
                    stacklevel=2,
                )
                continue
            self.single_label_heads[name] = nn.Linear(config.projection_dim, n)

        for name in MULTI_LABEL_HEADS:
            n = head_dims.get(name, 0)
            if n == 0:
                warnings.warn(
                    f"EntityEncoder: head_dimensions missing key {name!r}; "
                    "skipping head.",
                    stacklevel=2,
                )
                continue
            self.multi_label_heads[name] = nn.Linear(config.projection_dim, n)

        # ── weight initialisation (projection + heads only) ───────────────────
        self._init_weights()

        # ── gradient-checkpointing flag (off by default) ──────────────────────
        self._gradient_checkpointing: bool = False

        # ── attention-weight hook placeholder ─────────────────────────────────
        self._return_attention: bool = config.return_attention

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform init for all Linear layers in projection and heads.

        MiniLM pre-trained weights are never touched.
        """
        for module in (*self.projection.modules(),
                       *self.single_label_heads.modules(),
                       *self.multi_label_heads.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ── pooling ───────────────────────────────────────────────────────────────

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask:    torch.Tensor,
    ) -> torch.Tensor:
        """Attention-mask-weighted mean pool over the token dimension.

        Args:
            last_hidden_state: Shape ``(B, seq_len, hidden)``.
            attention_mask:    Shape ``(B, seq_len)``, 1 for real tokens.

        Returns:
            Shape ``(B, hidden)``.  Never uses the [CLS] token.
        """
        mask = attention_mask.unsqueeze(-1).float()          # (B, seq, 1)
        summed  = (last_hidden_state * mask).sum(dim=1)      # (B, hidden)
        counts  = mask.sum(dim=1).clamp(min=1e-9)            # (B, 1)
        return summed / counts                               # (B, hidden)

    # ── forward utilities ─────────────────────────────────────────────────────

    def _backbone_forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run backbone and return mean-pooled embedding ``(B, embedding_dim)``.

        Supports gradient checkpointing when enabled.
        """
        if self._gradient_checkpointing and self.training:
            # torch.utils.checkpoint requires a function with no kwargs.
            def _ckpt_fn(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                out = self.backbone(input_ids=ids, attention_mask=mask)
                return out.last_hidden_state

            hidden = torch.utils.checkpoint.checkpoint(
                _ckpt_fn, input_ids, attention_mask, use_reentrant=False
            )
        else:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = out.last_hidden_state  # (B, seq, 384)

        return self.mean_pool(hidden, attention_mask)       # (B, 384)

    def _project(self, pooled: torch.Tensor) -> torch.Tensor:
        """Run projection tower + optional L2 normalisation.

        Args:
            pooled: ``(B, embedding_dim)`` mean-pooled backbone output.

        Returns:
            ``(B, projection_dim)`` shared embedding.
        """
        emb = self.projection(pooled)                       # (B, 512)
        if self.config.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared embedding without running any output heads.

        This is the primary interface for RelationEncoder, SceneGraphBuilder,
        retrieval, and similarity search.

        Args:
            input_ids:      ``(B, seq_len)`` token IDs, dtype long.
            attention_mask: ``(B, seq_len)`` mask, dtype long.

        Returns:
            ``(B, projection_dim)`` L2-normalised shared embedding.
        """
        pooled = self._backbone_forward(input_ids, attention_mask)
        return self._project(pooled)

    def encode_batch(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`encode` — explicit name for batch-embedding calls.

        Identical behaviour but semantically distinguishes "batch inference
        without heads" from training forward passes.
        """
        return self.encode(input_ids, attention_mask)

    def get_embedding(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`encode` — matches naming expected by the RAG layer."""
        return self.encode(input_ids, attention_mask)

    def cosine_similarity(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """Element-wise cosine similarity between two embedding matrices.

        Args:
            emb_a: ``(B, D)`` or ``(D,)`` embedding(s).
            emb_b: ``(B, D)`` or ``(D,)`` embedding(s).

        Returns:
            Scalar tensor (if 1-D inputs) or ``(B,)`` tensor of similarities
            in ``[-1, 1]``.
        """
        a = F.normalize(emb_a, p=2, dim=-1)
        b = F.normalize(emb_b, p=2, dim=-1)
        return (a * b).sum(dim=-1)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> EntityEncoderOutput:
        """Full forward pass: backbone → projection → all heads.

        Args:
            input_ids:      ``(B, seq_len)`` long tensor.
            attention_mask: ``(B, seq_len)`` long tensor.

        Returns:
            :class:`EntityEncoderOutput` with ``shared_embedding`` and
            ``logits`` dict.  All values are **raw logits** — no softmax
            or sigmoid is applied.  The training script applies the
            appropriate loss function per head type.
        """
        # Step 1: backbone + mean pool
        pooled = self._backbone_forward(input_ids, attention_mask)  # (B, 384)

        # Step 2: shared projection + normalisation
        shared_emb = self._project(pooled)                          # (B, 512)

        # Step 3: head dropout
        head_input = self.head_drop(shared_emb)                     # (B, 512)

        # Step 4: all output heads
        logits: Dict[str, torch.Tensor] = {}
        for name, head in self.single_label_heads.items():
            logits[name] = head(head_input)
        for name, head in self.multi_label_heads.items():
            logits[name] = head(head_input)

        return EntityEncoderOutput(shared_embedding=shared_emb, logits=logits)

    # ── freeze / unfreeze ─────────────────────────────────────────────────────

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad

    def _set_heads_grad(self, requires_grad: bool) -> None:
        for param in self.single_label_heads.parameters():
            param.requires_grad = requires_grad
        for param in self.multi_label_heads.parameters():
            param.requires_grad = requires_grad

    def freeze_backbone(self) -> None:
        """Set all MiniLM parameters to ``requires_grad=False``."""
        self._set_backbone_grad(requires_grad=False)

    def unfreeze_backbone(self) -> None:
        """Restore MiniLM parameters to trainable (``requires_grad=True``)."""
        self._set_backbone_grad(requires_grad=True)

    def freeze_heads(self) -> None:
        """Freeze all output head Linear layers."""
        self._set_heads_grad(requires_grad=False)

    def unfreeze_heads(self) -> None:
        """Unfreeze all output head Linear layers."""
        self._set_heads_grad(requires_grad=True)

    # ── parameter groups ──────────────────────────────────────────────────────

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr:     float = 1e-3,
    ) -> List[Dict]:
        """Return parameter groups for differential learning rates.

        Intended usage in the training script::

            optimizer = torch.optim.AdamW(
                model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
            )

        Args:
            backbone_lr: Learning rate for MiniLM backbone parameters.
            head_lr:     Learning rate for projection + head parameters.

        Returns:
            List of two dicts consumable by any ``torch.optim.Optimizer``.
        """
        backbone_params = list(self.backbone.parameters())
        other_params = [
            p for p in self.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": other_params,    "lr": head_lr,     "name": "heads+projection"},
        ]

    # ── gradient checkpointing ────────────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the backbone.

        Trades compute for memory: activations are recomputed during the
        backward pass rather than stored.  Call before the training loop.
        """
        self._gradient_checkpointing = True
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    # ── head grouping ─────────────────────────────────────────────────────────

    def get_single_label_heads(self) -> Dict[str, nn.Linear]:
        """Return the single-label head registry.

        Training script usage::

            for name, head in model.get_single_label_heads().items():
                loss += ce_loss(logits[name], targets[name])
        """
        return dict(self.single_label_heads)  # type: ignore[return-value]

    def get_multi_label_heads(self) -> Dict[str, nn.Linear]:
        """Return the multi-label head registry.

        Training script usage::

            for name, head in model.get_multi_label_heads().items():
                loss += bce_loss(logits[name], targets[name])
        """
        return dict(self.multi_label_heads)   # type: ignore[return-value]

    # ── info API ──────────────────────────────────────────────────────────────

    def get_embedding_dim(self) -> int:
        """Return the shared projection dimension (e.g. 512)."""
        return self.config.projection_dim

    def get_head_dimensions(self) -> Dict[str, int]:
        """Return head name → output dimension for all registered heads.

        Identical to the ``head_dimensions`` supplied at construction time,
        but inferred directly from the Linear layer weights — guaranteed to
        be consistent with the actual model.
        """
        dims: Dict[str, int] = {}
        for name, head in self.single_label_heads.items():
            dims[name] = head.out_features
        for name, head in self.multi_label_heads.items():
            dims[name] = head.out_features
        return dims

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
        print("  PhysWorldLM — EntityEncoder")
        print(f"{'═'*64}")
        print(f"  Backbone           : {cfg.backbone_name}")
        print(f"  Embedding dim      : {cfg.embedding_dim}  (backbone output)")
        print(f"  Hidden dim         : {cfg.hidden_dim}")
        print(f"  Projection dim     : {cfg.projection_dim}  (shared embedding)")
        print(f"  MLP depth          : {cfg.mlp_depth} × ProjectionBlock")
        print(f"  Activation         : {cfg.activation}")
        print(f"  Dropout            : {cfg.dropout}  (projection)")
        print(f"  Head dropout       : {cfg.head_dropout}")
        print(f"  LayerNorm          : {cfg.use_layernorm}")
        print(f"  Residual           : {cfg.use_residual}")
        print(f"  Normalize emb      : {cfg.normalize_embeddings}")
        print(f"  Backbone frozen    : {cfg.freeze_backbone}")
        print(f"  Grad checkpointing : {self._gradient_checkpointing}")
        print()

        print("  Single-label heads")
        print(f"  {'─'*42}")
        for name in SINGLE_LABEL_HEADS:
            n = dims.get(name, "—")
            print(f"    {name:<22}  →  {n} classes")

        print()
        print("  Multi-label heads")
        print(f"  {'─'*42}")
        for name in MULTI_LABEL_HEADS:
            n = dims.get(name, "—")
            print(f"    {name:<22}  →  {n} labels")

        print()
        print(f"  Parameters")
        print(f"  {'─'*42}")
        print(f"    Total      : {param['total']:>12,}")
        print(f"    Trainable  : {param['trainable']:>12,}")
        print(f"    Frozen     : {param['total'] - param['trainable']:>12,}")
        print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  debug_forward()
# ─────────────────────────────────────────────────────────────────────────────

def debug_forward(model: EntityEncoder) -> None:
    """Run one synthetic forward pass and print all output shapes.

    Uses batch_size=2, seq_len=32 fake tensors to exercise the full pipeline
    without requiring real data or a DataLoader.

    Expected output (with default config and metadata from datasets/entity)::

        shared_embedding   (2, 512)
        entity_type        (2, 19)
        parent_class       (2, 6)
        material           (2, 20)
        shape              (2, 9)
        capabilities       (2, 96)
        affordances        (2, 98)
        scene_roles        (2, 10)
        ...

    Args:
        model: An :class:`EntityEncoder` instance.
    """
    device = next(model.parameters()).device
    B, L  = 2, 32

    input_ids      = torch.ones(B, L, dtype=torch.long,  device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long,  device=device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    print(f"\n{'─'*52}")
    print("  debug_forward()  output shapes")
    print(f"{'─'*52}")
    print(f"  {'shared_embedding':<26}  {tuple(output.shared_embedding.shape)}")
    for name in SINGLE_LABEL_HEADS:
        if name in output.logits:
            print(f"  {name:<26}  {tuple(output.logits[name].shape)}")
    for name in MULTI_LABEL_HEADS:
        if name in output.logits:
            print(f"  {name:<26}  {tuple(output.logits[name].shape)}")
    print(f"{'─'*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_encoder(
    dataset_dir: str = "datasets/entity",
    freeze_backbone: bool = False,
    **config_kwargs,
) -> EntityEncoder:
    """Convenience factory: load head dimensions from EntityDataset and build.

    This is the recommended way to construct ``EntityEncoder`` in training
    scripts — it guarantees that head dimensions match the on-disk label maps.

    Args:
        dataset_dir:     Root of the split dataset (contains label_maps/).
        freeze_backbone: Freeze MiniLM at init.
        **config_kwargs: Additional keyword arguments forwarded to
                         :class:`EntityEncoderConfig`.

    Returns:
        Fully configured :class:`EntityEncoder` instance.

    Example::

        model = build_entity_encoder(dataset_dir="datasets/entity",
                                     hidden_dim=512, mlp_depth=2)
    """
    # Deferred import to avoid circular dependency at module load time.
    # entity_dataset.py lives in training/; entity_encoder.py in models/.
    from training.entity_dataset import EntityDataset, DatasetConfig  # noqa: PLC0415

    ds = EntityDataset(DatasetConfig(split="train", debug=False))
    head_dims = ds.get_head_dimensions()

    cfg = EntityEncoderConfig(
        head_dimensions=head_dims,
        freeze_backbone=freeze_backbone,
        **config_kwargs,
    )
    return EntityEncoder(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: build encoder, print summary, run debug forward pass."""

    print("=" * 64)
    print("  PhysWorldLM — EntityEncoder smoke-test")
    print("=" * 64)

    model = build_entity_encoder(
        dataset_dir="datasets/entity",
        hidden_dim=512,
        projection_dim=512,
        mlp_depth=2,
        dropout=0.1,
        head_dropout=0.1,
        freeze_backbone=False,
        normalize_embeddings=True,
    )

    model.print_model_summary()
    debug_forward(model)

    params = model.count_parameters()
    print(f"  Total parameters     : {params['total']:>12,}")
    print(f"  Trainable parameters : {params['trainable']:>12,}")

    print("\n  Head dimensions:")
    for name, dim in model.get_head_dimensions().items():
        print(f"    {name:<24}  {dim}")

    print(f"\n  get_embedding_dim()  → {model.get_embedding_dim()}")
    print("\n[main] done.")


if __name__ == "__main__":
    main()
