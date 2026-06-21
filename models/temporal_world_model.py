"""
models/temporal_world_model.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — fourth learned component: the core world-evolution model.

Architecture
────────────
    SceneGraph(t-T..t)                     current node embeddings (t)
        │  scene_embedding sequence              │
        ▼  (B, T, D)                              │
    TemporalPositionalEncoding                    │
        ▼                                          │
    Multi-scale TransformerEncoder                 │
      (local + global, residual fusion)            │
        ▼                                          │
    TemporalMemory (GRU)  ──▶ memory_state (B, M)  │
        ▼                                          │
    Variational latent space                       │
      mu, logvar → reparameterize → latent_state   │
        ▼                                          │
    world_token  (CLS-style, like scene_token)      │
        ▼                                          │
    Transformer "decoder" (expand over horizon)     │
        ▼                                          │
    ┌─────────────┬─────────────┬───────────────┬──┴────────────┬───────────┐
    ▼              ▼              ▼                ▼              ▼
 PositionHead  VelocityHead  AccelerationHead  OrientationHead  AngularVelHead
    │              │              │                │              │
    └──────────────┴──────────────┴────────────────┴──────────────┘
                                   ▼
                    UncertaintyHead   PhysicsLatentHead   SceneEmbeddingHead
                                   ▼
                              PredictedState(s)

``TemporalWorldModel`` is the bridge between a static world representation
(``GraphBuilder.SceneGraph`` / ``WorldSpec``) and genuine world *evolution*:
it learns ``Graph(t) → Graph(t+1)`` and is designed to feed directly into:

    StateEngine          → numerical integration / collision resolution
    Trajectory Engine     → multi-step rollouts for animation
    Bullet / MuJoCo / Isaac / Gazebo → physics back-ends (interfaces only)
    Renderer              → visual composition
    Video diffusion       → via the exposed ``world_latent`` / ``world_token``

Design principles (mirrors models/entity_encoder.py, relation_encoder.py,
graph_builder.py)
─────────────────────────────────────────────────────────────────────────
* TemporalWorldModelConfig — zero magic numbers; all dimensions configurable.
* WorldState / PredictedState / TemporalOutput — typed dataclasses; physical
  quantities and embeddings are never discarded in favour of summaries.
* TemporalPositionalEncoding — sinusoidal, covers history + horizon.
* TemporalMemory — GRU-based short-term memory; ``world_memory`` dict is the
  long-term memory bank (hierarchical memory, per the DRDO-grade critique).
* Variational + deterministic hybrid latent space — ``mu``/``logvar`` are
  always computed; sampling only happens in ``self.training`` mode (eval
  uses the mean), so a single model serves both stochastic-future-modelling
  and deterministic-inference use cases without a config switch.
* Delta prediction — heads predict Δposition/Δvelocity/Δacceleration rather
  than absolute values (more stable, matches the spec).
* Physics latent — a dedicated head exposes a coarse physically-meaningful
  17-d vector (momentum, kinetic energy, angular momentum, force, torque,
  constraint state, contact energy — see ``PHYSICS_LATENT_LAYOUT``)
  alongside the raw latent, so later Bullet/MuJoCo integration has a
  natural handle without having to reverse-engineer it from the opaque
  512-d latent.
* Uncertainty — every physical head has a paired variance estimate.
* Rollout — teacher-forcing, free (autoregressive), and beam-search modes.
* World memory bank — ``nearest_memories()`` over stored scene/latent
  embeddings (the long-term half of the hierarchical memory).
* Loss "interfaces" — position/velocity/orientation/KL/trajectory/physics-
  consistency losses are implemented as plain stateless functions (pure
  math, not a training loop), so a future training script can import and
  use them directly without re-deriving the formulas; nothing here builds
  an optimizer, a scheduler, or a training loop.
* Diffusion / flow-matching / rectified-flow futures predictors / Bullet /
  MuJoCo / Isaac / Gazebo — explicit placeholder interfaces that raise
  ``NotImplementedError``, exactly mirroring ``graph_builder.py``'s
  ``to_bullet()`` etc.

Scope discipline
─────────────────
This module implements *only* ``models/temporal_world_model.py``. It
deliberately does NOT implement: training loops, the StateEngine, Bullet /
MuJoCo / Isaac / Gazebo back-ends, a renderer, or video generation. The
following upstream components are COMPLETE and are NOT modified here:

    models/entity_encoder.py
    models/relation_encoder.py
    models/graph_builder.py
    world_spec.py

Per-node multi-step rollout — a documented design decision
─────────────────────────────────────────────────────────────
Physical-quantity heads (position/velocity/acceleration/orientation/
angular-velocity) operate on a *fixed* set of node embeddings. Carrying a
consistent node correspondence across an arbitrarily long horizon (objects
appearing, merging, fragmenting) is GraphBuilder's / StateEngine's job, not
this file's. ``predict_sequence()`` therefore predicts the *scene-level*
latent / scene_embedding trajectory (the world's evolution as a whole,
directly useful for video-diffusion conditioning), while node-level
physical quantities are predicted one step at a time via
``predict_next_state()``. ``rollout()`` composes the two: it advances the
scene-level latent autoregressively and, when node embeddings are supplied,
also Euler-integrates per-node positions/velocities from the predicted
deltas at each step. This keeps the contract simple and avoids silently
inventing a node-tracking algorithm that belongs in a later file.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_DIM:   int = 512
HIDDEN_DIM:  int = 512
STATE_DIM:   int = 512
LATENT_DIM:  int = 512
MEMORY_DIM:  int = 512

RolloutMode = Literal["teacher_force", "autoregressive", "beam"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  TemporalWorldModelConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalWorldModelConfig:
    """All hyperparameters for TemporalWorldModel — no magic numbers inside the model.

    Attributes:
        graph_dim:               Dimension of incoming scene/node embeddings
                                 (output of GraphBuilder).
        hidden_dim:               Width of intermediate projections.
        state_dim:                Dimension of the per-node decode state.
        latent_dim:                Dimension of the variational latent.
        memory_dim:                Dimension of the GRU memory state.
        num_transformer_layers:    Layers in the *global* temporal encoder.
                                   The *local* encoder uses
                                   ``max(1, num_transformer_layers // 2)``.
        num_attention_heads:       Heads for all attention modules.
        dropout:                   Dropout probability throughout.
        activation:                ``"gelu"`` or ``"relu"``.
        layer_norm:                Insert LayerNorm where applicable.
        residual:                  Add residual connections where shapes allow.
        normalize_latents:         L2-normalise ``scene_token`` / latent mean.
        history_length:            Max input sequence length supported by the
                                   positional encoding.
        prediction_horizon:        Max rollout horizon supported by the
                                   positional encoding.
        use_scene_token:           Compute a CLS-style world token.
        use_graph_memory:          Maintain the long-term ``world_memory``
                                   bank and enable ``nearest_memories()``.
        use_cross_attention:       Enable scene↔memory cross-attention before
                                   the GRU.
        use_gru_memory:            Use a GRU for short-term memory.
        use_lstm_memory:           Use an LSTM instead of / alongside GRU.
                                   Mutually exclusive with ``use_gru_memory``
                                   unless both are False (in which case mean
                                   pooling is used).
        use_delta_prediction:      Predict Δposition/Δvelocity/Δacceleration.
        use_uncertainty:           Predict per-quantity variance.
        use_variational_latents:   Sample from ``N(mu, exp(logvar))`` during
                                   training; use ``mu`` directly at eval time
                                   when False.
        gradient_checkpointing:    Enable checkpointing on the transformer
                                   stacks at construction time.
        memory_bank_capacity:      Max entries retained in ``world_memory``.
        dt:                        Default integration timestep (seconds)
                                   used by Euler integration in ``rollout()``.
    """

    graph_dim:               int   = GRAPH_DIM
    hidden_dim:               int   = HIDDEN_DIM
    state_dim:                int   = STATE_DIM
    latent_dim:                int   = LATENT_DIM
    memory_dim:                int   = MEMORY_DIM
    num_transformer_layers:    int   = 4
    num_attention_heads:        int   = 8
    dropout:                    float = 0.1
    activation:                  str   = "gelu"
    layer_norm:                  bool  = True
    residual:                    bool  = True
    normalize_latents:            bool  = True
    history_length:                int   = 32
    prediction_horizon:             int   = 128
    use_scene_token:                 bool  = True
    use_graph_memory:                 bool  = True
    use_cross_attention:               bool  = True
    use_gru_memory:                     bool  = True
    use_lstm_memory:                     bool  = False
    use_delta_prediction:                 bool  = True
    use_uncertainty:                       bool  = True
    use_variational_latents:                 bool  = True
    gradient_checkpointing:                   bool  = False
    memory_bank_capacity:                      int   = 10_000
    dt:                                          float = 0.01

    def __post_init__(self) -> None:
        if self.graph_dim < 1:
            raise ValueError(f"graph_dim must be ≥ 1, got {self.graph_dim}")
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be ≥ 1, got {self.latent_dim}")
        if self.graph_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"graph_dim ({self.graph_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_transformer_layers < 1:
            raise ValueError(
                f"num_transformer_layers must be ≥ 1, got {self.num_transformer_layers}"
            )
        if self.history_length < 1:
            raise ValueError(f"history_length must be ≥ 1, got {self.history_length}")
        if self.prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be ≥ 1, got {self.prediction_horizon}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in ("gelu", "relu"):
            raise ValueError(f"activation must be 'gelu' or 'relu', got {self.activation!r}")
        if self.use_gru_memory and self.use_lstm_memory:
            warnings.warn(
                "TemporalWorldModelConfig: both use_gru_memory and "
                "use_lstm_memory are True; GRU output will be used as the "
                "memory_state and the LSTM will run alongside but its "
                "output is exposed only via metadata.",
                stacklevel=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Dataclasses: WorldState, PredictedState, TemporalOutput
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """Full kinematic snapshot of the world at one timestep.

    Mirrors ``world_spec.PhysicsState`` but batched over nodes and carrying
    learned embeddings, so it can serve as both model input and the
    canonical bridge back to ``WorldSpec``/``GraphBuilder``.

    Attributes:
        time_index:           Integer timestep index (0 = present).
        scene_embedding:       ``(D,)`` or ``(B, D)`` pooled scene embedding.
        node_embeddings:        ``(N, D)`` or ``(B, N, D)`` per-node embeddings,
                                or None.
        edge_embeddings:         ``(E, D)`` or ``(B, E, D)`` per-edge embeddings,
                                or None.
        positions:                ``(N, 3)`` / ``(B, N, 3)`` world positions.
        velocities:                ``(N, 3)`` / ``(B, N, 3)``.
        accelerations:              ``(N, 3)`` / ``(B, N, 3)``.
        orientations:                ``(N, 4)`` / ``(B, N, 4)`` quaternions
                                    (w, x, y, z).
        angular_velocities:           ``(N, 3)`` / ``(B, N, 3)``.
        metadata:                      Free-form bag (timestamps, scene_id, …).
    """

    time_index:           int = 0
    scene_embedding:       Optional[torch.Tensor] = None
    node_embeddings:        Optional[torch.Tensor] = None
    edge_embeddings:         Optional[torch.Tensor] = None
    positions:                 Optional[torch.Tensor] = None
    velocities:                 Optional[torch.Tensor] = None
    accelerations:               Optional[torch.Tensor] = None
    orientations:                 Optional[torch.Tensor] = None
    angular_velocities:             Optional[torch.Tensor] = None
    metadata:                         Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _l(t: Optional[torch.Tensor]) -> Optional[list]:
            return t.tolist() if t is not None else None
        return {
            "time_index":          self.time_index,
            "scene_embedding":      _l(self.scene_embedding),
            "node_embeddings":       _l(self.node_embeddings),
            "edge_embeddings":        _l(self.edge_embeddings),
            "positions":               _l(self.positions),
            "velocities":               _l(self.velocities),
            "accelerations":             _l(self.accelerations),
            "orientations":               _l(self.orientations),
            "angular_velocities":          _l(self.angular_velocities),
            "metadata":                      self.metadata,
        }


@dataclass
class PredictedState:
    """Model output for a single future timestep.

    Attributes:
        next_scene_embedding:    ``(B, D)`` predicted scene embedding.
        next_positions:           ``(B, N, 3)`` or None (node-level only when
                                  node embeddings were supplied).
        next_velocities:           ``(B, N, 3)`` or None.
        next_accelerations:          ``(B, N, 3)`` or None.
        next_orientations:             ``(B, N, 4)`` quaternion, or None.
        next_angular_velocities:         ``(B, N, 3)`` or None.
        uncertainty:                      Dict of quantity name → variance
                                          tensor, or None.
        physics_latent:                    ``(B, physics_latent_dim)`` coarse
                                           physical-quantity vector
                                           (momentum, energy, angular
                                           momentum), or None.
        latent_state:                       ``(B, latent_dim)`` sampled/mean
                                            latent used to produce this state.
    """

    next_scene_embedding:    Optional[torch.Tensor] = None
    next_positions:           Optional[torch.Tensor] = None
    next_velocities:           Optional[torch.Tensor] = None
    next_accelerations:          Optional[torch.Tensor] = None
    next_orientations:             Optional[torch.Tensor] = None
    next_angular_velocities:         Optional[torch.Tensor] = None
    uncertainty:                      Optional[Dict[str, torch.Tensor]] = None
    physics_latent:                    Optional[torch.Tensor] = None
    latent_state:                       Optional[torch.Tensor] = None


@dataclass
class TemporalOutput:
    """Typed container for the output of ``TemporalWorldModel.forward()``.

    Attributes:
        predictions:       List of :class:`PredictedState`, length =
                           the requested horizon (≥ 1).
        memory_state:        ``(B, memory_dim)`` GRU/LSTM short-term memory.
        scene_token:           ``(B, D)`` CLS-style world token, or None.
        attention_weights:        Dict of attention-weight tensors for
                                  explainability (encoder self-attention,
                                  scene↔memory cross-attention).
        latent_mu:                 ``(B, latent_dim)`` variational mean.
        latent_logvar:                ``(B, latent_dim)`` variational
                                      log-variance.
    """

    predictions:          List[PredictedState] = field(default_factory=list)
    memory_state:           Optional[torch.Tensor] = None
    scene_token:              Optional[torch.Tensor] = None
    attention_weights:           Dict[str, torch.Tensor] = field(default_factory=dict)
    latent_mu:                     Optional[torch.Tensor] = None
    latent_logvar:                    Optional[torch.Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  TemporalPositionalEncoding
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding covering history + prediction horizon.

    Args:
        dim:     Embedding dimension.
        max_len: Maximum sequence length to precompute (history_length +
                 prediction_horizon is a safe upper bound).
    """

    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x:      ``(B, T, D)``.
            offset: Starting position index (useful when decoding future
                    steps that continue from the history's positions).

        Returns:
            ``(B, T, D)`` — ``x`` with positional encoding added.

        Raises:
            ValueError: If ``offset + T`` exceeds the precomputed length.
        """
        t = x.shape[1]
        if offset + t > self.pe.shape[1]:
            raise ValueError(
                f"TemporalPositionalEncoding: offset+T={offset + t} exceeds "
                f"max_len={self.pe.shape[1]}"
            )
        return x + self.pe[:, offset: offset + t, :].to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  TemporalMemory  (GRU / optional LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalMemory(nn.Module):
    """Short-term recurrent memory over a scene-embedding sequence.

    Args:
        input_dim:  Dimension of each timestep's input vector.
        memory_dim: Hidden size of the recurrent cell(s).
        use_gru:    Use a GRU.
        use_lstm:   Use an LSTM (can be combined with GRU; see
                    :class:`TemporalWorldModelConfig` for the resulting
                    semantics).
    """

    def __init__(
        self,
        input_dim:  int,
        memory_dim: int,
        use_gru:    bool = True,
        use_lstm:   bool = False,
    ) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.use_lstm = use_lstm
        if use_gru:
            self.gru = nn.GRU(input_dim, memory_dim, batch_first=True)
        else:
            self.gru = None
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, memory_dim, batch_first=True)
        else:
            self.lstm = None
        if not use_gru and not use_lstm:
            self.fallback_proj = nn.Linear(input_dim, memory_dim)
        else:
            self.fallback_proj = None

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Run the recurrent memory over ``x``.

        Args:
            x:      ``(B, T, D)``.
            hidden: Optional GRU hidden state ``(1, B, memory_dim)`` to
                    continue from (used by ``rollout()``).

        Returns:
            Tuple of (``memory_state`` ``(B, memory_dim)``, new GRU hidden
            state or None, ``extras`` dict — contains ``lstm_state`` when
            ``use_lstm=True``).
        """
        extras: Dict[str, torch.Tensor] = {}
        if self.use_gru:
            seq_out, h_n = self.gru(x, hidden)
            memory_state = h_n.squeeze(0)  # (B, memory_dim)
            new_hidden = h_n
        else:
            new_hidden = None
            memory_state = x.mean(dim=1)  # fallback: mean pool
            if self.fallback_proj is not None:
                memory_state = self.fallback_proj(memory_state)

        if self.use_lstm:
            lstm_out, (h_lstm, c_lstm) = self.lstm(x)
            extras["lstm_state"] = h_lstm.squeeze(0)
            if not self.use_gru:
                memory_state = h_lstm.squeeze(0)

        return memory_state, new_hidden, extras


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  Output heads
# ─────────────────────────────────────────────────────────────────────────────

class _MLPHead(nn.Module):
    """Generic ``in_dim → hidden → out_dim`` head with LayerNorm/GELU/Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1, activation: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)


class OrientationHead(nn.Module):
    """Predicts a unit quaternion ``(w, x, y, z)``.

    Structure: ``512 → 256 → 4``, L2-normalised so the output always lies on
    the unit quaternion manifold.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.mlp = _MLPHead(in_dim, hidden_dim, 4, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.mlp(x)
        return F.normalize(q, p=2, dim=-1, eps=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  Diffusion future-predictor interface (placeholder only)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionFuturePredictor(nn.Module):
    """Interface placeholder for a future multimodal-futures predictor
    (DDPM / flow-matching / rectified-flow over latent trajectories).

    Not trained or wired into :class:`TemporalWorldModel` in this file.
    Exists so downstream code can type-check against a stable interface
    ahead of the real implementation, and so the deterministic/variational
    rollout in this file can later be swapped for genuinely multimodal
    futures without changing the public API.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, num_steps: int = 50) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps

    def forward(self, latent: torch.Tensor, horizon: int) -> torch.Tensor:
        raise NotImplementedError(
            "DiffusionFuturePredictor is a placeholder interface; implement "
            "a real diffusion / flow-matching sampler in a future file."
        )


class FlowMatchingPredictor(nn.Module):
    """Interface placeholder for a future flow-matching futures predictor.

    Mirrors :class:`DiffusionFuturePredictor` exactly — same non-wired,
    placeholder-only status. Kept as a distinct class (rather than a flag
    on ``DiffusionFuturePredictor``) because flow-matching and DDPM
    samplers have different training objectives and inference loops; a
    future implementation will not share weights or code between them.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, num_steps: int = 50) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps

    def forward(self, latent: torch.Tensor, horizon: int) -> torch.Tensor:
        raise NotImplementedError(
            "FlowMatchingPredictor is a placeholder interface; implement a "
            "real flow-matching sampler in a future file."
        )


class RectifiedFlowPredictor(nn.Module):
    """Interface placeholder for a future rectified-flow futures predictor.

    Mirrors :class:`DiffusionFuturePredictor` exactly — same non-wired,
    placeholder-only status.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, num_steps: int = 50) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps

    def forward(self, latent: torch.Tensor, horizon: int) -> torch.Tensor:
        raise NotImplementedError(
            "RectifiedFlowPredictor is a placeholder interface; implement a "
            "real rectified-flow sampler in a future file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  TemporalWorldModel
# ─────────────────────────────────────────────────────────────────────────────

class TemporalWorldModel(nn.Module):
    """Learns ``Graph(t) → Graph(t+1)`` — the core world-evolution model.

    Pipeline (see module docstring for the full diagram)::

        scene_embedding sequence (B, T, D)
            → positional encoding
            → multi-scale transformer encoder (local + global)
            → GRU memory  → memory_state (B, memory_dim)
            → variational latent (mu, logvar) → latent_state (B, latent_dim)
            → world_token (CLS-style)
            → transformer "decoder" expanded over the requested horizon
            → per-step output heads (position / velocity / acceleration /
              orientation / angular_velocity / uncertainty / physics_latent /
              scene_embedding)

    Downstream consumers
    ────────────────────
    * ``StateEngine``        — numerical integration, collision resolution
    * ``Trajectory Engine``  — multi-step rollouts for animation
    * Bullet / MuJoCo / Isaac / Gazebo — via the (placeholder) export hooks
    * Renderer / video diffusion — via ``scene_token`` / ``world_latent``

    Args:
        config: :class:`TemporalWorldModelConfig` instance. Defaults are
                used when omitted.
    """

    #: Dimension of the physics-latent head's output:
    #: momentum(3) + kinetic_energy(1) + angular_momentum(3) + force(3) +
    #: torque(3) + constraint_state(3) + contact_energy(1) = 17
    PHYSICS_LATENT_DIM: int = 17

    #: Named slices into the physics-latent vector, in concatenation order.
    #: Mirrors ``_UNCERTAINTY_LAYOUT``'s (name, dim) convention so callers
    #: can address sub-quantities without hard-coding offsets. Kept on the
    #: class (not computed) so physics-latent loss functions and any future
    #: training script can import the layout directly.
    PHYSICS_LATENT_LAYOUT: Tuple[Tuple[str, int], ...] = (
        ("momentum", 3), ("kinetic_energy", 1), ("angular_momentum", 3),
        ("force", 3), ("torque", 3), ("constraint_state", 3), ("contact_energy", 1),
    )

    #: Names + dims of the uncertainty vector, concatenated in this order.
    _UNCERTAINTY_LAYOUT: Tuple[Tuple[str, int], ...] = (
        ("position", 3), ("velocity", 3), ("acceleration", 3),
        ("orientation", 4), ("angular_velocity", 3),
    )

    def __init__(self, config: Optional[TemporalWorldModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = TemporalWorldModelConfig()
        self.config: TemporalWorldModelConfig = config
        d, h = config.graph_dim, config.hidden_dim

        # ── positional encoding ────────────────────────────────────────────
        max_len = config.history_length + config.prediction_horizon + 1
        self.pos_encoding = TemporalPositionalEncoding(d, max_len)

        # ── multi-scale transformer encoder (local + global, fused) ────────
        local_layers = max(1, config.num_transformer_layers // 2)

        def _make_encoder(num_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.num_attention_heads,
                dim_feedforward=d * 4, dropout=config.dropout,
                activation=config.activation, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.local_encoder = _make_encoder(local_layers)
        self.global_encoder = _make_encoder(config.num_transformer_layers)
        self.scale_fusion = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.dropout),
        )

        # ── GRU/LSTM short-term memory ──────────────────────────────────────
        self.memory = TemporalMemory(
            input_dim=d, memory_dim=config.memory_dim,
            use_gru=config.use_gru_memory, use_lstm=config.use_lstm_memory,
        )

        # ── scene ↔ memory cross-attention ──────────────────────────────────
        self._use_cross_attention = config.use_cross_attention
        if config.use_cross_attention:
            self.memory_proj = nn.Linear(config.memory_dim, d)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d, num_heads=config.num_attention_heads,
                dropout=config.dropout, batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(d)
        else:
            self.memory_proj = None
            self.cross_attention = None
            self.cross_attn_norm = None

        # ── variational latent space ────────────────────────────────────────
        fusion_in = config.memory_dim + (d if config.use_cross_attention else 0)
        self.to_latent = nn.Linear(fusion_in, config.hidden_dim)
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

        # ── world / scene token (CLS-style) ─────────────────────────────────
        self._use_scene_token = config.use_scene_token
        if config.use_scene_token:
            self.scene_token_proj = nn.Sequential(
                nn.Linear(config.latent_dim, d), nn.LayerNorm(d),
            )
        else:
            self.scene_token_proj = None

        # ── decoder: latent → per-horizon-step tokens ───────────────────────
        self.latent_to_decode = nn.Linear(config.latent_dim, config.state_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.state_dim, nhead=config.num_attention_heads,
            dim_feedforward=config.state_dim * 4, dropout=config.dropout,
            activation=config.activation, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.num_transformer_layers)
        self.decoder_pos = TemporalPositionalEncoding(config.state_dim, max_len)

        # ── per-node conditioning (broadcast latent onto node embeddings) ──
        self.node_fuse = nn.Sequential(
            nn.Linear(config.state_dim + d, config.state_dim),
            nn.LayerNorm(config.state_dim), nn.GELU(), nn.Dropout(config.dropout),
        )

        # ── output heads ─────────────────────────────────────────────────────
        s = config.state_dim
        self.position_head = _MLPHead(s, 256, 3, config.dropout, config.activation)
        self.velocity_head = _MLPHead(s, 256, 3, config.dropout, config.activation)
        self.acceleration_head = _MLPHead(s, 256, 3, config.dropout, config.activation)
        self.orientation_head = OrientationHead(s, 256, config.dropout, config.activation)
        self.angular_velocity_head = _MLPHead(s, 256, 3, config.dropout, config.activation)
        self.scene_embedding_head = _MLPHead(s, 256, d, config.dropout, config.activation)

        uncertainty_dim = sum(n for _, n in self._UNCERTAINTY_LAYOUT)
        self.uncertainty_head = _MLPHead(s, 256, uncertainty_dim, config.dropout, config.activation) \
            if config.use_uncertainty else None

        self.physics_latent_head = _MLPHead(s, 256, self.PHYSICS_LATENT_DIM, config.dropout, config.activation)

        # ── diffusion / flow interfaces (not trained/used here) ─────────────
        self.diffusion_predictor = DiffusionFuturePredictor(config.latent_dim)
        self.flow_matching_predictor = FlowMatchingPredictor(config.latent_dim)
        self.rectified_flow_predictor = RectifiedFlowPredictor(config.latent_dim)

        # ── world memory bank (long-term hierarchical memory) ──────────────
        self.world_memory: Dict[str, Dict[str, Any]] = {}
        self._memory_counter: int = 0

        # ── weight init ──────────────────────────────────────────────────────
        self._init_weights()

        # ── gradient checkpointing ────────────────────────────────────────────
        self._gradient_checkpointing: bool = config.gradient_checkpointing

        # ── persistent GRU hidden state, used by rollout()/reset_memory() ───
        self._hidden_state: Optional[torch.Tensor] = None

    @classmethod
    def physics_latent_slice(cls, physics_latent: torch.Tensor, name: str) -> torch.Tensor:
        """Slice a named quantity out of a ``(..., PHYSICS_LATENT_DIM)`` tensor.

        Args:
            physics_latent: ``(..., PHYSICS_LATENT_DIM)`` tensor (the
                ``physics_latent_head`` output).
            name:            One of the names in :attr:`PHYSICS_LATENT_LAYOUT`
                             (e.g. ``"momentum"``, ``"force"``).

        Returns:
            ``(..., dim)`` slice, where ``dim`` is that quantity's width.

        Raises:
            ValueError: If ``name`` isn't in :attr:`PHYSICS_LATENT_LAYOUT` or
                the last dim of ``physics_latent`` doesn't match
                :attr:`PHYSICS_LATENT_DIM`.
        """
        if physics_latent.shape[-1] != cls.PHYSICS_LATENT_DIM:
            raise ValueError(
                f"physics_latent_slice: expected last dim "
                f"{cls.PHYSICS_LATENT_DIM}, got {physics_latent.shape[-1]}"
            )
        idx = 0
        for n, dim_ in cls.PHYSICS_LATENT_LAYOUT:
            if n == name:
                return physics_latent[..., idx: idx + dim_]
            idx += dim_
        valid = ", ".join(n for n, _ in cls.PHYSICS_LATENT_LAYOUT)
        raise ValueError(f"physics_latent_slice: unknown name {name!r}; expected one of: {valid}")

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
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

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_sequence(self, scene_embeddings: torch.Tensor) -> None:
        if scene_embeddings.dim() != 3:
            raise ValueError(
                "TemporalWorldModel expects scene_embeddings of shape "
                f"(B, T, D); got dim()={scene_embeddings.dim()}"
            )
        b, t, d = scene_embeddings.shape
        if t == 0:
            raise ValueError("TemporalWorldModel: empty sequence (T=0)")
        if d != self.config.graph_dim:
            raise ValueError(
                f"TemporalWorldModel: expected graph_dim={self.config.graph_dim}, "
                f"got D={d}"
            )
        if t > self.config.history_length:
            raise ValueError(
                f"TemporalWorldModel: sequence length T={t} exceeds "
                f"history_length={self.config.history_length}"
            )
        if torch.isnan(scene_embeddings).any() or torch.isinf(scene_embeddings).any():
            raise RuntimeError("TemporalWorldModel: NaN/Inf in scene_embeddings")

    def _validate_horizon(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be ≥ 1, got {horizon}")
        if horizon > self.config.prediction_horizon:
            raise ValueError(
                f"horizon={horizon} exceeds prediction_horizon="
                f"{self.config.prediction_horizon}"
            )

    # ── encoder / memory / latent ────────────────────────────────────────────

    def _encode_sequence(
        self, scene_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Positional encoding → multi-scale transformer → fused sequence.

        Args:
            scene_embeddings: ``(B, T, D)``.

        Returns:
            Tuple of (fused encoded sequence ``(B, T, D)``, attention dict —
            currently empty; reserved for future per-layer attention export).
        """
        x = self.pos_encoding(scene_embeddings)
        local_out = self._run_transformer(self.local_encoder, x)
        global_out = self._run_transformer(self.global_encoder, x)
        fused = self.scale_fusion(torch.cat([local_out, global_out], dim=-1))
        return fused, {}

    def _run_transformer(self, encoder: nn.TransformerEncoder, x: torch.Tensor) -> torch.Tensor:
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(encoder, x, use_reentrant=False)
        return encoder(x)

    def _memory_and_latent(
        self, encoded: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run GRU memory + (optional) cross-attention + variational latent.

        Returns:
            Tuple of (memory_state ``(B,M)``, new_hidden, latent_state
            ``(B,L)``, mu ``(B,L)``, logvar ``(B,L)``, attention_weights dict).
        """
        memory_state, new_hidden, _extras = self.memory(encoded, hidden)
        attn_weights: Dict[str, torch.Tensor] = {}

        if self._use_cross_attention:
            mem_q = self.memory_proj(memory_state).unsqueeze(1)        # (B,1,D)
            attended, attn = self.cross_attention(
                mem_q, encoded, encoded, need_weights=True,
            )
            attended = self.cross_attn_norm(attended + mem_q).squeeze(1)  # (B,D)
            attn_weights["memory_cross_attention"] = attn
            fused = torch.cat([memory_state, attended], dim=-1)
        else:
            fused = memory_state

        h = F.gelu(self.to_latent(fused))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        latent_state = self._reparameterize(mu, logvar) if (self.training and self.config.use_variational_latents) else mu
        if self.config.normalize_latents:
            latent_state = F.normalize(latent_state, p=2, dim=-1)

        return memory_state, new_hidden, latent_state, mu, logvar, attn_weights

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z ~ N(mu, exp(logvar))`` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Public sampler — draw ``num_samples`` latents per batch element.

        Args:
            mu:           ``(B, L)``.
            logvar:        ``(B, L)``.
            num_samples:    Number of samples per row.

        Returns:
            ``(B, num_samples, L)`` if ``num_samples > 1`` else ``(B, L)``.
        """
        if num_samples == 1:
            return self._reparameterize(mu, logvar)
        std = torch.exp(0.5 * logvar).unsqueeze(1).expand(-1, num_samples, -1)
        mu_exp = mu.unsqueeze(1).expand(-1, num_samples, -1)
        eps = torch.randn_like(std)
        return mu_exp + eps * std

    # ── decode ────────────────────────────────────────────────────────────────

    def _decode_horizon(self, latent_state: torch.Tensor, horizon: int) -> torch.Tensor:
        """Expand a single latent into ``horizon`` decoded step tokens.

        Args:
            latent_state: ``(B, latent_dim)``.
            horizon:       Number of future steps to decode.

        Returns:
            ``(B, horizon, state_dim)``.
        """
        b = latent_state.shape[0]
        step0 = self.latent_to_decode(latent_state).unsqueeze(1)         # (B,1,state_dim)
        tokens = step0.expand(-1, horizon, -1).contiguous()              # (B,H,state_dim)
        tokens = self.decoder_pos(tokens)
        decoded = self._run_transformer(self.decoder, tokens)            # (B,H,state_dim)
        return decoded

    def _apply_heads(
        self, decoded_step: torch.Tensor, node_embeddings: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply all output heads to a single decoded step token.

        Args:
            decoded_step:     ``(B, state_dim)``.
            node_embeddings:    ``(B, N, D)`` or None. When provided, the
                                token is fused per-node so position/velocity/
                                etc. are predicted *per entity*; otherwise
                                a single "virtual node" (the scene itself)
                                is used.

        Returns:
            Dict with keys: position, velocity, acceleration, orientation,
            angular_velocity, scene_embedding, uncertainty (optional dict),
            physics_latent.
        """
        if node_embeddings is not None:
            n = node_embeddings.shape[1]
            tok = decoded_step.unsqueeze(1).expand(-1, n, -1)             # (B,N,state_dim)
            fused = self.node_fuse(torch.cat([tok, node_embeddings], dim=-1))  # (B,N,state_dim)
        else:
            fused = decoded_step.unsqueeze(1)                              # (B,1,state_dim)

        position = self.position_head(fused)
        velocity = self.velocity_head(fused)
        acceleration = self.acceleration_head(fused)
        orientation = self.orientation_head(fused)
        angular_velocity = self.angular_velocity_head(fused)
        scene_embedding = self.scene_embedding_head(decoded_step)          # (B,D), scene-level only
        physics_latent = self.physics_latent_head(decoded_step)            # (B,PHYSICS_LATENT_DIM)

        uncertainty: Optional[Dict[str, torch.Tensor]] = None
        if self.uncertainty_head is not None:
            raw = F.softplus(self.uncertainty_head(fused))                 # (B,N or 1, U) — positivity
            uncertainty = {}
            idx = 0
            for name, dim_ in self._UNCERTAINTY_LAYOUT:
                uncertainty[name] = raw[..., idx: idx + dim_]
                idx += dim_

        if node_embeddings is None:
            position = position.squeeze(1)
            velocity = velocity.squeeze(1)
            acceleration = acceleration.squeeze(1)
            orientation = orientation.squeeze(1)
            angular_velocity = angular_velocity.squeeze(1)
            if uncertainty is not None:
                uncertainty = {k: v.squeeze(1) for k, v in uncertainty.items()}

        return {
            "position": position, "velocity": velocity, "acceleration": acceleration,
            "orientation": orientation, "angular_velocity": angular_velocity,
            "scene_embedding": scene_embedding, "uncertainty": uncertainty,
            "physics_latent": physics_latent,
        }

    # ── public API: forward / predict_next_state / predict_sequence ─────────

    def forward(
        self,
        scene_embeddings: torch.Tensor,
        node_embeddings: Optional[torch.Tensor] = None,
        horizon: int = 1,
        hidden: Optional[torch.Tensor] = None,
    ) -> TemporalOutput:
        """Full forward pass: encode history → memory → latent → decode horizon.

        Args:
            scene_embeddings: ``(B, T, D)`` history of scene embeddings.
            node_embeddings:   ``(B, N, D)`` current-step node embeddings, or
                              None for scene-level-only prediction.
            horizon:            Number of future steps to predict (≥ 1).
            hidden:              Optional GRU hidden state to continue from.

        Returns:
            :class:`TemporalOutput` with ``horizon`` :class:`PredictedState`
            entries plus memory/latent/attention diagnostics.
        """
        self._validate_sequence(scene_embeddings)
        self._validate_horizon(horizon)

        encoded, _enc_attn = self._encode_sequence(scene_embeddings)
        memory_state, new_hidden, latent_state, mu, logvar, attn = \
            self._memory_and_latent(encoded, hidden)

        scene_token: Optional[torch.Tensor] = None
        if self._use_scene_token:
            scene_token = self.scene_token_proj(latent_state)

        decoded = self._decode_horizon(latent_state, horizon)  # (B,H,state_dim)

        predictions: List[PredictedState] = []
        for step in range(horizon):
            heads = self._apply_heads(decoded[:, step, :], node_embeddings)
            predictions.append(PredictedState(
                next_scene_embedding=heads["scene_embedding"],
                next_positions=heads["position"] if node_embeddings is not None else None,
                next_velocities=heads["velocity"] if node_embeddings is not None else None,
                next_accelerations=heads["acceleration"] if node_embeddings is not None else None,
                next_orientations=heads["orientation"] if node_embeddings is not None else None,
                next_angular_velocities=heads["angular_velocity"] if node_embeddings is not None else None,
                uncertainty=heads["uncertainty"],
                physics_latent=heads["physics_latent"],
                latent_state=latent_state,
            ))

        self._hidden_state = new_hidden

        return TemporalOutput(
            predictions=predictions,
            memory_state=memory_state,
            scene_token=scene_token,
            attention_weights=attn,
            latent_mu=mu,
            latent_logvar=logvar,
        )

    def predict_next_state(
        self,
        scene_embeddings: torch.Tensor,
        node_embeddings: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> PredictedState:
        """Convenience wrapper: ``forward(..., horizon=1).predictions[0]``."""
        out = self.forward(scene_embeddings, node_embeddings, horizon=1, hidden=hidden)
        return out.predictions[0]

    def predict_sequence(
        self,
        scene_embeddings: torch.Tensor,
        horizon: int,
        node_embeddings: Optional[torch.Tensor] = None,
    ) -> TemporalOutput:
        """Predict a multi-step scene-level (and optionally node-level)
        trajectory in one shot (non-autoregressive — all ``horizon`` steps
        are decoded from a single latent in parallel).

        See the module docstring's "Per-node multi-step rollout" note for
        why node-level outputs from this method should be treated as a
        single shared per-step decode rather than a physically integrated
        trajectory; use :meth:`rollout` for the latter.
        """
        return self.forward(scene_embeddings, node_embeddings, horizon=horizon)

    def get_world_latent(
        self, output: TemporalOutput, step: int = 0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Bundle the tensors a downstream consumer (renderer, video
        diffusion) would want into a single dict, without recomputing
        anything — pure accessor over an already-computed
        :class:`TemporalOutput`.

        Args:
            output: A :class:`TemporalOutput` returned by :meth:`forward`,
                    :meth:`predict_sequence`, or :meth:`predict_next_state`
                    (wrap the latter in ``forward(..., horizon=1)`` if you
                    need the dict form, since ``predict_next_state`` itself
                    returns a bare :class:`PredictedState`).
            step:    Which horizon step's ``physics_latent`` /
                     ``latent_state`` to read off ``output.predictions``.
                     Ignored for ``scene_token`` / ``memory_state``, which
                     are per-sequence (not per-step).

        Returns:
            Dict with keys ``scene_token``, ``memory_state``,
            ``latent_state``, ``physics_latent`` — any of which may be
            ``None`` if the corresponding feature is disabled in config or
            ``output.predictions`` is empty.

        Raises:
            IndexError: If ``step`` is out of range for
                ``output.predictions``.
        """
        if output.predictions and not (0 <= step < len(output.predictions)):
            raise IndexError(
                f"get_world_latent: step={step} out of range for "
                f"{len(output.predictions)} prediction(s)"
            )
        pred = output.predictions[step] if output.predictions else None
        return {
            "scene_token":    output.scene_token,
            "memory_state":   output.memory_state,
            "latent_state":   pred.latent_state if pred is not None else None,
            "physics_latent": pred.physics_latent if pred is not None else None,
        }

    # ── state encode/decode helpers ──────────────────────────────────────────

    def encode_state(
        self,
        node_embeddings: Optional[torch.Tensor],
        edge_embeddings: Optional[torch.Tensor],
        scene_embedding: torch.Tensor,
        time_index: int = 0,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        accelerations: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
        angular_velocities: Optional[torch.Tensor] = None,
    ) -> WorldState:
        """Package raw tensors into a :class:`WorldState` (no learned transform
        — this is a structural convenience, mirroring GraphBuilder's
        dataclass packaging)."""
        return WorldState(
            time_index=time_index, scene_embedding=scene_embedding,
            node_embeddings=node_embeddings, edge_embeddings=edge_embeddings,
            positions=positions, velocities=velocities, accelerations=accelerations,
            orientations=orientations, angular_velocities=angular_velocities,
        )

    def decode_state(self, latent_state: torch.Tensor, node_embeddings: Optional[torch.Tensor] = None) -> PredictedState:
        """Decode a single latent vector directly into one :class:`PredictedState`
        (bypasses history encoding/memory — useful for sampling from a stored
        ``world_memory`` latent)."""
        decoded = self._decode_horizon(latent_state, horizon=1)[:, 0, :]
        heads = self._apply_heads(decoded, node_embeddings)
        return PredictedState(
            next_scene_embedding=heads["scene_embedding"],
            next_positions=heads["position"] if node_embeddings is not None else None,
            next_velocities=heads["velocity"] if node_embeddings is not None else None,
            next_accelerations=heads["acceleration"] if node_embeddings is not None else None,
            next_orientations=heads["orientation"] if node_embeddings is not None else None,
            next_angular_velocities=heads["angular_velocity"] if node_embeddings is not None else None,
            uncertainty=heads["uncertainty"], physics_latent=heads["physics_latent"],
            latent_state=latent_state,
        )

    # ── rollout ───────────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        """Clear the persistent GRU hidden state used by autoregressive rollout."""
        self._hidden_state = None

    def rollout(
        self,
        scene_embeddings: torch.Tensor,
        horizon: int,
        mode: RolloutMode = "autoregressive",
        node_embeddings: Optional[torch.Tensor] = None,
        ground_truth_future: Optional[torch.Tensor] = None,
        beam_width: int = 4,
        dt: Optional[float] = None,
        physics_weight: float = 0.1,
    ) -> List[PredictedState]:
        """Roll the model forward for ``horizon`` steps.

        Args:
            scene_embeddings:      ``(B, T, D)`` initial history.
            horizon:                 Number of steps to roll out.
            mode:                     ``"teacher_force"`` (requires
                                     ``ground_truth_future``),
                                     ``"autoregressive"`` (feeds back its own
                                     predictions), or ``"beam"`` (keeps the
                                     top-``beam_width`` latent trajectories by
                                     cumulative log-likelihood proxy).
            node_embeddings:          ``(B, N, D)`` current entity embeddings;
                                     when given, positions/velocities are
                                     Euler-integrated forward at each step
                                     using the predicted deltas.
            ground_truth_future:       ``(B, horizon, D)`` required for
                                      ``"teacher_force"``.
            beam_width:                 Beam size for ``"beam"`` mode.
            dt:                          Integration timestep; defaults to
                                        ``config.dt``.
            physics_weight:              Weight on the physics-consistency
                                        penalty in ``"beam"`` mode's scoring
                                        (ignored by other modes); see
                                        :meth:`_beam_rollout`.

        Returns:
            List of ``horizon`` :class:`PredictedState`.

        Raises:
            ValueError: On invalid mode/arguments.
        """
        self._validate_sequence(scene_embeddings)
        self._validate_horizon(horizon)
        dt = self.config.dt if dt is None else dt

        if mode == "teacher_force":
            return self._teacher_force_rollout(scene_embeddings, horizon, ground_truth_future, node_embeddings, dt)
        if mode == "autoregressive":
            return self._autoregressive_rollout(scene_embeddings, horizon, node_embeddings, dt)
        if mode == "beam":
            return self._beam_rollout(scene_embeddings, horizon, beam_width, node_embeddings, dt, physics_weight)
        raise ValueError(f"rollout: unknown mode {mode!r}; expected teacher_force/autoregressive/beam")

    def teacher_force_rollout(
        self, scene_embeddings: torch.Tensor, ground_truth_future: torch.Tensor,
        node_embeddings: Optional[torch.Tensor] = None,
    ) -> List[PredictedState]:
        """Named entry point for teacher-forced rollout (see :meth:`rollout`)."""
        horizon = ground_truth_future.shape[1]
        return self.rollout(scene_embeddings, horizon, mode="teacher_force",
                             node_embeddings=node_embeddings, ground_truth_future=ground_truth_future)

    def autoregressive_rollout(
        self, scene_embeddings: torch.Tensor, horizon: int,
        node_embeddings: Optional[torch.Tensor] = None,
    ) -> List[PredictedState]:
        """Named entry point for free/autoregressive rollout (see :meth:`rollout`)."""
        return self.rollout(scene_embeddings, horizon, mode="autoregressive", node_embeddings=node_embeddings)

    def beam_rollout(
        self, scene_embeddings: torch.Tensor, horizon: int, beam_width: int = 4,
        node_embeddings: Optional[torch.Tensor] = None, physics_weight: float = 0.1,
    ) -> List[PredictedState]:
        """Named entry point for beam-search rollout (see :meth:`rollout`)."""
        return self.rollout(scene_embeddings, horizon, mode="beam", beam_width=beam_width,
                             node_embeddings=node_embeddings, physics_weight=physics_weight)

    def sample_trajectory(
        self,
        scene_embeddings: torch.Tensor,
        horizon: int,
        num_samples: int = 1,
        node_embeddings: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> List[List[PredictedState]]:
        """Draw ``num_samples`` independent stochastic future trajectories
        from the variational latent space.

        Unlike :meth:`rollout` (which always uses the latent's mean at eval
        time — see :meth:`_memory_and_latent`), this method explicitly draws
        from ``N(mu, exp(logvar))`` via the existing :meth:`sample_latent`,
        decodes each sample autoregressively with :meth:`_decode_horizon` /
        :meth:`_apply_heads`, and Euler-integrates node-level positions with
        the same :meth:`_euler_integrate` helper :meth:`rollout` uses — so
        the only thing that differs from autoregressive rollout is *which*
        latent seeds each step (a draw, not the mean), not how decoding or
        integration work.

        Args:
            scene_embeddings: ``(B, T, D)`` initial history.
            horizon:            Number of steps per trajectory.
            num_samples:          Number of independent trajectories to draw.
            node_embeddings:       ``(B, N, D)`` current entity embeddings,
                                  or None for scene-level-only trajectories.
            dt:                     Integration timestep; defaults to
                                   ``config.dt``.

        Returns:
            List of length ``num_samples``, each a list of ``horizon``
            :class:`PredictedState` (same shape as :meth:`rollout`'s return
            value, one per sample).
        """
        self._validate_sequence(scene_embeddings)
        self._validate_horizon(horizon)
        if num_samples < 1:
            raise ValueError(f"sample_trajectory: num_samples must be ≥ 1, got {num_samples}")
        dt = self.config.dt if dt is None else dt

        # One shared history encode → (mu, logvar); sampling diverges per-draw
        # from there, matching how the variational latent space is defined
        # everywhere else in this file (encode once, sample many).
        encoded, _ = self._encode_sequence(scene_embeddings)
        _, _, _latent_state, mu, logvar, _ = self._memory_and_latent(encoded)

        all_trajectories: List[List[PredictedState]] = []
        for _ in range(num_samples):
            latent = self.sample_latent(mu, logvar, num_samples=1)
            if self.config.normalize_latents:
                latent = F.normalize(latent, p=2, dim=-1)

            predictions: List[PredictedState] = []
            prev_pos, prev_vel = None, None
            cur_latent = latent

            for _step in range(horizon):
                decoded = self._decode_horizon(cur_latent, horizon=1)[:, 0, :]
                heads = self._apply_heads(decoded, node_embeddings)
                pred = PredictedState(
                    next_scene_embedding=heads["scene_embedding"],
                    next_positions=heads["position"] if node_embeddings is not None else None,
                    next_velocities=heads["velocity"] if node_embeddings is not None else None,
                    next_accelerations=heads["acceleration"] if node_embeddings is not None else None,
                    next_orientations=heads["orientation"] if node_embeddings is not None else None,
                    next_angular_velocities=heads["angular_velocity"] if node_embeddings is not None else None,
                    uncertainty=heads["uncertainty"],
                    physics_latent=heads["physics_latent"],
                    latent_state=cur_latent,
                )

                if node_embeddings is not None:
                    new_pos, new_vel = self._euler_integrate(
                        node_embeddings, pred, dt, prev_pos, prev_vel
                    )
                    pred.next_positions, pred.next_velocities = new_pos, new_vel
                    prev_pos, prev_vel = new_pos, new_vel

                predictions.append(pred)

                # Re-encode the freshly decoded scene embedding through the
                # same history → memory → latent path, so each step's latent
                # reflects the trajectory so far rather than freezing at the
                # t0 draw. This mirrors _autoregressive_rollout's history
                # feedback, just operating on a single-sample latent instead
                # of recomputing mu/logvar from scratch every step.
                next_token = pred.next_scene_embedding.unsqueeze(1)  # (B,1,D)
                new_history = torch.cat([scene_embeddings[:, 1:, :], next_token], dim=1) \
                    if scene_embeddings.shape[1] >= self.config.history_length \
                    else torch.cat([scene_embeddings, next_token], dim=1)
                scene_embeddings = new_history
                step_encoded, _ = self._encode_sequence(scene_embeddings)
                _, _, cur_latent, _, _, _ = self._memory_and_latent(step_encoded)

            all_trajectories.append(predictions)

        return all_trajectories

    def _euler_integrate(
        self, node_embeddings: torch.Tensor, pred: PredictedState, dt: float,
        prev_positions: Optional[torch.Tensor], prev_velocities: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine delta predictions with the previous physical state via a
        simple Euler step. Falls back to absolute values when no previous
        state is supplied (e.g. the very first rollout step)."""
        if self.config.use_delta_prediction and prev_positions is not None and prev_velocities is not None:
            new_velocity = prev_velocities + pred.next_accelerations * dt
            new_position = prev_positions + prev_velocities * dt + pred.next_positions * dt
        else:
            new_velocity = pred.next_velocities
            new_position = pred.next_positions
        return new_position, new_velocity

    def _autoregressive_rollout(
        self, scene_embeddings: torch.Tensor, horizon: int,
        node_embeddings: Optional[torch.Tensor], dt: float,
    ) -> List[PredictedState]:
        self.reset_memory()
        history = scene_embeddings
        predictions: List[PredictedState] = []
        prev_pos, prev_vel = None, None
        if node_embeddings is not None:
            cur_node_embeddings = node_embeddings
        else:
            cur_node_embeddings = None

        for _ in range(horizon):
            out = self.forward(history, cur_node_embeddings, horizon=1, hidden=self._hidden_state)
            pred = out.predictions[0]

            if cur_node_embeddings is not None:
                new_pos, new_vel = self._euler_integrate(cur_node_embeddings, pred, dt, prev_pos, prev_vel)
                pred.next_positions, pred.next_velocities = new_pos, new_vel
                prev_pos, prev_vel = new_pos, new_vel

            predictions.append(pred)

            next_token = pred.next_scene_embedding.unsqueeze(1)  # (B,1,D)
            history = torch.cat([history[:, 1:, :], next_token], dim=1) \
                if history.shape[1] >= self.config.history_length else torch.cat([history, next_token], dim=1)

        return predictions

    def _teacher_force_rollout(
        self, scene_embeddings: torch.Tensor, horizon: int,
        ground_truth_future: Optional[torch.Tensor],
        node_embeddings: Optional[torch.Tensor], dt: float,
    ) -> List[PredictedState]:
        if ground_truth_future is None:
            raise ValueError("teacher_force rollout requires ground_truth_future")
        if ground_truth_future.shape[1] != horizon:
            raise ValueError(
                f"ground_truth_future horizon ({ground_truth_future.shape[1]}) "
                f"!= requested horizon ({horizon})"
            )
        self.reset_memory()
        history = scene_embeddings
        predictions: List[PredictedState] = []
        prev_pos, prev_vel = None, None
        cur_node_embeddings = node_embeddings

        for step in range(horizon):
            out = self.forward(history, cur_node_embeddings, horizon=1, hidden=self._hidden_state)
            pred = out.predictions[0]

            if cur_node_embeddings is not None:
                new_pos, new_vel = self._euler_integrate(cur_node_embeddings, pred, dt, prev_pos, prev_vel)
                pred.next_positions, pred.next_velocities = new_pos, new_vel
                prev_pos, prev_vel = new_pos, new_vel

            predictions.append(pred)

            gt_token = ground_truth_future[:, step: step + 1, :]
            history = torch.cat([history[:, 1:, :], gt_token], dim=1) \
                if history.shape[1] >= self.config.history_length else torch.cat([history, gt_token], dim=1)

        return predictions

    def _beam_rollout(
        self, scene_embeddings: torch.Tensor, horizon: int, beam_width: int,
        node_embeddings: Optional[torch.Tensor], dt: float,
        physics_weight: float = 0.1,
    ) -> List[PredictedState]:
        """Beam search over latent samples, scored by negative latent-space
        displacement (a cheap likelihood proxy — favours temporally coherent
        trajectories without requiring a trained density model), plus a
        physics-consistency term when node embeddings are supplied.

        The physics term reuses :meth:`physics_consistency_loss` (already
        defined as a stateless loss "interface" elsewhere in this file) to
        penalise beams whose Euler-integrated positions disagree with
        first-order kinematics from the previous step. It only activates
        from the second step onward, since the first step has no previous
        physical state to be consistent with.

        Args:
            physics_weight: Weight on the (negative) physics-consistency
                term in the per-step score. Default is small relative to
                the latent-smoothness term so existing beam-search behaviour
                is only nudged, not dominated, by the new term — there is no
                trained calibration for this weight yet.
        """
        if beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1, got {beam_width}")
        b = scene_embeddings.shape[0]

        # Each beam: (history, predictions_so_far, score, prev_positions, prev_velocities)
        BeamState = Tuple[torch.Tensor, List[PredictedState], float, Optional[torch.Tensor], Optional[torch.Tensor]]
        beams: List[BeamState] = [
            (scene_embeddings, [], 0.0, None, None)
        ]

        for _ in range(horizon):
            candidates: List[BeamState] = []
            for hist, preds_so_far, score, prev_pos, prev_vel in beams:
                out = self.forward(hist, node_embeddings, horizon=1)
                pred = out.predictions[0]
                disp = pred.next_scene_embedding.detach()
                step_score = -float(disp.pow(2).mean().item())  # smoother trajectories score higher

                new_pos, new_vel = prev_pos, prev_vel
                if node_embeddings is not None:
                    new_pos, new_vel = self._euler_integrate(node_embeddings, pred, dt, prev_pos, prev_vel)
                    pred.next_positions, pred.next_velocities = new_pos, new_vel
                    if prev_pos is not None and prev_vel is not None:
                        physics_penalty = float(self.physics_consistency_loss(
                            new_pos.detach(), new_vel.detach(),
                            prev_pos.detach(), prev_vel.detach(), dt,
                        ).item())
                        step_score -= physics_weight * physics_penalty

                next_token = pred.next_scene_embedding.unsqueeze(1)
                new_hist = torch.cat([hist[:, 1:, :], next_token], dim=1) \
                    if hist.shape[1] >= self.config.history_length else torch.cat([hist, next_token], dim=1)
                candidates.append((new_hist, preds_so_far + [pred], score + step_score, new_pos, new_vel))

            candidates.sort(key=lambda c: c[2], reverse=True)
            beams = candidates[:beam_width]

        best_hist, best_preds, best_score, _best_pos, _best_vel = max(beams, key=lambda c: c[2])
        return best_preds

    # ── world memory bank ─────────────────────────────────────────────────────

    def _record_memory(self, scene_embedding: torch.Tensor, latent_state: torch.Tensor, timestamp: float) -> str:
        if not self.config.use_graph_memory:
            return ""
        if len(self.world_memory) >= self.config.memory_bank_capacity:
            oldest = next(iter(self.world_memory))
            del self.world_memory[oldest]
        mem_id = f"mem_{self._memory_counter}"
        self._memory_counter += 1
        self.world_memory[mem_id] = {
            "scene_embedding": scene_embedding.detach().clone(),
            "latent_state":    latent_state.detach().clone(),
            "timestamp":        timestamp,
        }
        return mem_id

    def remember(self, scene_embedding: torch.Tensor, latent_state: torch.Tensor, timestamp: float = 0.0) -> str:
        """Public entry point to store a (scene_embedding, latent_state) pair
        in the long-term ``world_memory`` bank. Returns the memory id."""
        return self._record_memory(scene_embedding, latent_state, timestamp)

    def nearest_memories(self, query: torch.Tensor, k: int = 5, by: str = "scene_embedding") -> List[Tuple[str, float]]:
        """Find the ``k`` most similar stored memories to ``query`` by cosine
        similarity.

        Args:
            query: ``(D,)`` (for ``by="scene_embedding"``) or ``(L,)``
                   (for ``by="latent_state"``) query vector.
            k:      Number of neighbours.
            by:      ``"scene_embedding"`` or ``"latent_state"``.

        Returns:
            List of ``(memory_id, similarity)`` tuples, descending.
        """
        if by not in ("scene_embedding", "latent_state"):
            raise ValueError(f"nearest_memories: by must be scene_embedding/latent_state, got {by!r}")
        if not self.world_memory:
            return []
        q = F.normalize(query.reshape(1, -1), p=2, dim=-1)
        sims: List[Tuple[str, float]] = []
        for mid, entry in self.world_memory.items():
            v = F.normalize(entry[by].reshape(1, -1), p=2, dim=-1)
            sims.append((mid, float((q * v).sum().item())))
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[: max(1, k)]

    # ── freeze / unfreeze ─────────────────────────────────────────────────────

    def _backbone_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [
            self.local_encoder, self.global_encoder, self.scale_fusion,
            self.memory, self.to_latent, self.mu_head, self.logvar_head,
            self.latent_to_decode, self.decoder,
        ]
        if self._use_cross_attention:
            modules.extend([self.memory_proj, self.cross_attention, self.cross_attn_norm])
        if self._use_scene_token:
            modules.append(self.scene_token_proj)
        return modules

    def _head_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [
            self.node_fuse, self.position_head, self.velocity_head, self.acceleration_head,
            self.orientation_head, self.angular_velocity_head, self.scene_embedding_head,
            self.physics_latent_head,
        ]
        if self.uncertainty_head is not None:
            modules.append(self.uncertainty_head)
        return modules

    def freeze_backbone(self) -> None:
        """Freeze the encoder/memory/latent stack; leave output heads trainable."""
        for module in self._backbone_modules():
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for module in self._backbone_modules():
            for p in module.parameters():
                p.requires_grad = True

    def freeze_heads(self) -> None:
        """Freeze all output heads; leave the backbone trainable."""
        for module in self._head_modules():
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_heads(self) -> None:
        for module in self._head_modules():
            for p in module.parameters():
                p.requires_grad = True

    # ── parameter groups ──────────────────────────────────────────────────────

    def parameter_groups(self, backbone_lr: float = 5e-5, head_lr: float = 1e-3) -> List[Dict]:
        """Return parameter groups for differential learning rates.

        Usage::

            optimizer = torch.optim.AdamW(
                model.parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
            )
        """
        backbone_params: List[torch.nn.Parameter] = []
        for module in self._backbone_modules():
            backbone_params.extend(module.parameters())
        head_params: List[torch.nn.Parameter] = []
        for module in self._head_modules():
            head_params.extend(module.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": head_lr,     "name": "heads"},
        ]

    # ── gradient checkpointing ────────────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the transformer stacks.

        Trades compute for memory: activations are recomputed during the
        backward pass rather than stored. Call before the training loop.
        """
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    # ── physics back-end export interfaces (placeholders only) ─────────────

    def to_bullet(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future PyBullet exporter.

        Raises:
            NotImplementedError: Always — implement in a dedicated
                Bullet-export module that owns the pybullet dependency.
        """
        raise NotImplementedError(
            "to_bullet() is an interface placeholder; implement in a "
            "dedicated Bullet-export module."
        )

    def to_mujoco(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future MuJoCo exporter."""
        raise NotImplementedError(
            "to_mujoco() is an interface placeholder; implement in a "
            "dedicated MuJoCo-export module."
        )

    def to_isaac(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future Isaac Sim exporter."""
        raise NotImplementedError(
            "to_isaac() is an interface placeholder; implement in a "
            "dedicated Isaac-export module."
        )

    def to_gazebo(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future Gazebo exporter."""
        raise NotImplementedError(
            "to_gazebo() is an interface placeholder; implement in a "
            "dedicated Gazebo-export module."
        )

    # ── downstream pipeline export interfaces (placeholders only) ──────────
    # Named directly after the consumers in this module's architecture
    # diagram (StateEngine, Trajectory Engine, Renderer, video diffusion).
    # Same status as the physics back-ends above: explicit, non-wired,
    # always raise — implemented in their own dedicated files.

    def to_state_engine(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future StateEngine handoff (numerical
        integration / collision resolution).

        Raises:
            NotImplementedError: Always — implement in the dedicated
                StateEngine module.
        """
        raise NotImplementedError(
            "to_state_engine() is an interface placeholder; implement in a "
            "dedicated StateEngine module."
        )

    def to_trajectory_engine(self, predictions: List[PredictedState]) -> Any:
        """Interface placeholder for a future Trajectory Engine handoff
        (multi-step rollouts for animation).

        Args:
            predictions: A rollout's worth of :class:`PredictedState`
                (e.g. the output of :meth:`rollout`), since the Trajectory
                Engine operates on whole trajectories rather than single
                steps.

        Raises:
            NotImplementedError: Always — implement in a dedicated
                Trajectory Engine module.
        """
        raise NotImplementedError(
            "to_trajectory_engine() is an interface placeholder; implement "
            "in a dedicated Trajectory Engine module."
        )

    def to_renderer(self, predicted_state: PredictedState) -> Any:
        """Interface placeholder for a future Renderer handoff (visual
        composition from ``scene_token`` / predicted physical state).

        Raises:
            NotImplementedError: Always — implement in a dedicated
                Renderer module.
        """
        raise NotImplementedError(
            "to_renderer() is an interface placeholder; implement in a "
            "dedicated Renderer module."
        )

    def to_video_diffusion(self, output: TemporalOutput) -> Any:
        """Interface placeholder for a future video-diffusion handoff,
        conditioning on the exposed ``world_latent`` / ``scene_token``.

        Args:
            output: A :class:`TemporalOutput` (e.g. from :meth:`forward` /
                :meth:`predict_sequence`) — video diffusion conditions on
                the sequence-level ``scene_token`` / latent state, not a
                single :class:`PredictedState`.

        Raises:
            NotImplementedError: Always — implement in a dedicated
                video-diffusion module.
        """
        raise NotImplementedError(
            "to_video_diffusion() is an interface placeholder; implement in "
            "a dedicated video-diffusion module."
        )

    # ── save / load ───────────────────────────────────────────────────────────

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model weights + config to ``path`` via ``torch.save``.

        Note: ``world_memory`` (the long-term memory bank) is intentionally
        NOT persisted here — it is run-time state, not a trained parameter.
        Use a separate mechanism (e.g. pickling ``self.world_memory``) if a
        given application needs memory persistence across processes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(state, path)
        print(f"[TemporalWorldModel] saved → {path}")

    @classmethod
    def load_pretrained(cls, path: Union[str, Path], map_location: Optional[str] = None) -> "TemporalWorldModel":
        """Load a :class:`TemporalWorldModel` previously saved with
        :meth:`save_pretrained`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TemporalWorldModel checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location or "cpu")
        config = TemporalWorldModelConfig(**state["config"])
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        print(f"[TemporalWorldModel] loaded ← {path}")
        return model

    # ── loss "interfaces" (plain functions — no training loop) ─────────────

    @staticmethod
    def position_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean-squared error between predicted and target positions."""
        return F.mse_loss(pred, target)

    @staticmethod
    def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean-squared error between predicted and target velocities."""
        return F.mse_loss(pred, target)

    @staticmethod
    def orientation_loss(pred_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Geodesic-style quaternion loss: ``1 - |<q_pred, q_target>|``.

        Using the absolute dot product makes the loss invariant to the
        double-cover ambiguity of unit quaternions (``q`` and ``-q``
        represent the same rotation).
        """
        pred_n = F.normalize(pred_quat, p=2, dim=-1)
        target_n = F.normalize(target_quat, p=2, dim=-1)
        dot = (pred_n * target_n).sum(dim=-1).abs().clamp(max=1.0)
        return (1.0 - dot).mean()

    @staticmethod
    def latent_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence between ``N(mu, exp(logvar))`` and the standard normal."""
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()

    @classmethod
    def trajectory_loss(
        cls, predictions: List[PredictedState], targets: List[WorldState],
        position_weight: float = 1.0, velocity_weight: float = 1.0, orientation_weight: float = 1.0,
    ) -> torch.Tensor:
        """Aggregate position + velocity + orientation loss over a rollout.

        Args:
            predictions: Model rollout output.
            targets:      Ground-truth :class:`WorldState` per step, same
                          length as ``predictions``.

        Raises:
            ValueError: If lengths mismatch.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"trajectory_loss: predictions ({len(predictions)}) and "
                f"targets ({len(targets)}) length mismatch"
            )
        total = torch.tensor(0.0)
        count = 0
        for pred, tgt in zip(predictions, targets):
            if pred.next_positions is not None and tgt.positions is not None:
                total = total + position_weight * cls.position_loss(pred.next_positions, tgt.positions)
                count += 1
            if pred.next_velocities is not None and tgt.velocities is not None:
                total = total + velocity_weight * cls.velocity_loss(pred.next_velocities, tgt.velocities)
                count += 1
            if pred.next_orientations is not None and tgt.orientations is not None:
                total = total + orientation_weight * cls.orientation_loss(pred.next_orientations, tgt.orientations)
                count += 1
        return total / max(count, 1)

    @staticmethod
    def physics_consistency_loss(
        predicted_positions: torch.Tensor, predicted_velocities: torch.Tensor,
        prev_positions: torch.Tensor, prev_velocities: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """Penalises predictions that are inconsistent with first-order
        kinematics: ``position ≈ prev_position + prev_velocity * dt``.

        This is a cheap, differentiable proxy for "did the model respect
        basic Newtonian motion" — not a substitute for a real constraint
        solver, but useful as an auxiliary training signal.
        """
        expected_position = prev_positions + prev_velocities * dt
        return F.mse_loss(predicted_positions, expected_position)

    @staticmethod
    def energy_loss(
        predicted_velocities: torch.Tensor, target_velocities: torch.Tensor,
        mass: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """MSE between predicted and target kinetic energy, ``½·m·|v|²``.

        Operates on velocities (the quantity the model already predicts via
        ``velocity_head`` / ``next_velocities``) rather than on
        ``physics_latent``'s ``kinetic_energy`` slot directly, so this loss
        is usable without assuming the physics-latent head is supervised at
        all — it only assumes velocity predictions exist, which is always
        true. Pair with :meth:`PHYSICS_LATENT_LAYOUT`'s ``"kinetic_energy"``
        slot if/when that head gets its own supervision target.

        Args:
            predicted_velocities: ``(..., 3)`` predicted velocity.
            target_velocities:     ``(..., 3)`` ground-truth velocity.
            mass:                   Scalar, or per-node mass broadcastable
                                   against ``(...,)`` — i.e. the velocity
                                   shape *without* its trailing 3-component
                                   axis (e.g. ``(B, N)`` or ``(B, N, 1)``
                                   for ``(B, N, 3)`` velocities).
                                   Defaults to 1.0 (i.e. compares specific
                                   kinetic energy) when per-entity mass
                                   isn't available to the caller.

        Returns:
            Scalar MSE loss between predicted and target kinetic energy.
        """
        pred_ke = 0.5 * mass * predicted_velocities.pow(2).sum(dim=-1, keepdim=True)
        target_ke = 0.5 * mass * target_velocities.pow(2).sum(dim=-1, keepdim=True)
        return F.mse_loss(pred_ke, target_ke)

    @staticmethod
    def momentum_loss(
        predicted_velocities: torch.Tensor, target_velocities: torch.Tensor,
        mass: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        """MSE between predicted and target linear momentum, ``m·v``.

        Same design rationale as :meth:`energy_loss`: operates on the
        velocity heads' existing output rather than requiring
        ``physics_latent`` supervision, so it's usable standalone.

        Args:
            predicted_velocities: ``(..., 3)`` predicted velocity.
            target_velocities:     ``(..., 3)`` ground-truth velocity.
            mass:                   Scalar or broadcastable per-node mass.

        Returns:
            Scalar MSE loss between predicted and target momentum vectors.
        """
        pred_p = mass * predicted_velocities
        target_p = mass * target_velocities
        return F.mse_loss(pred_p, target_p)

    # ── debug ─────────────────────────────────────────────────────────────────

    def debug_forward(self, batch_size: int = 2, seq_len: int = 8, num_nodes: int = 3) -> TemporalOutput:
        """Run one synthetic forward pass and print all output shapes.

        Args:
            batch_size: B.
            seq_len:     T (history length).
            num_nodes:    N (entities decoded per step).

        Returns:
            The :class:`TemporalOutput` from the synthetic forward pass.
        """
        device = next(self.parameters()).device
        d = self.config.graph_dim

        scene_embeddings = F.normalize(torch.randn(batch_size, seq_len, d, device=device), p=2, dim=-1)
        node_embeddings = F.normalize(torch.randn(batch_size, num_nodes, d, device=device), p=2, dim=-1)

        self.eval()
        with torch.no_grad():
            output = self.forward(scene_embeddings, node_embeddings, horizon=1)

        pred = output.predictions[0]
        print(f"\n{'─'*52}")
        print("  debug_forward()  output shapes")
        print(f"{'─'*52}")
        print(f"  {'input sequence':<24}  {tuple(scene_embeddings.shape)}")
        print(f"  {'memory_state':<24}  {tuple(output.memory_state.shape)}")
        print(f"  {'latent (mu)':<24}  {tuple(output.latent_mu.shape)}")
        print(f"  {'position output':<24}  {tuple(pred.next_positions.shape)}")
        print(f"  {'velocity output':<24}  {tuple(pred.next_velocities.shape)}")
        print(f"  {'orientation output':<24}  {tuple(pred.next_orientations.shape)}")
        if pred.uncertainty is not None:
            for name, t in pred.uncertainty.items():
                print(f"  {'uncertainty/' + name:<24}  {tuple(t.shape)}")
        print(f"  {'physics_latent':<24}  {tuple(pred.physics_latent.shape)}")
        print(f"{'─'*52}\n")

        return output

    # ── info ──────────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def print_model_summary(self) -> None:
        cfg = self.config
        params = self.count_parameters()
        print(f"\n{'═'*64}")
        print("  PhysWorldLM — TemporalWorldModel")
        print(f"{'═'*64}")
        print(f"  Graph dim           : {cfg.graph_dim}")
        print(f"  Hidden dim          : {cfg.hidden_dim}")
        print(f"  Latent dim          : {cfg.latent_dim}")
        print(f"  Memory dim          : {cfg.memory_dim}")
        print(f"  History length      : {cfg.history_length}")
        print(f"  Prediction horizon  : {cfg.prediction_horizon}")
        print(f"  Attention heads     : {cfg.num_attention_heads}")
        print(f"  Transformer layers  : {cfg.num_transformer_layers}  (global) "
              f"/ {max(1, cfg.num_transformer_layers // 2)} (local)")
        print(f"  GRU / LSTM memory   : {cfg.use_gru_memory} / {cfg.use_lstm_memory}")
        print(f"  Variational latents : {cfg.use_variational_latents}")
        print(f"  Uncertainty heads   : {cfg.use_uncertainty}")
        print(f"  Parameter count     : {params['total']:,}")
        print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: build model, print summary, run debug_forward(), test
    rollout / save / load / nearest_memories, assert shapes."""

    print("=" * 64)
    print("  PhysWorldLM — TemporalWorldModel")
    print("=" * 64)

    config = TemporalWorldModelConfig(
        graph_dim=512, hidden_dim=512, state_dim=512, latent_dim=512,
        memory_dim=512, num_transformer_layers=4, num_attention_heads=8,
        dropout=0.1, history_length=32, prediction_horizon=128,
    )
    model = TemporalWorldModel(config)
    model.print_model_summary()

    output = model.debug_forward(batch_size=2, seq_len=8, num_nodes=3)

    # ── shape assertions ──────────────────────────────────────────────────
    assert output.memory_state.shape == (2, config.memory_dim)
    assert output.latent_mu.shape == (2, config.latent_dim)
    assert output.latent_logvar.shape == (2, config.latent_dim)
    pred = output.predictions[0]
    assert pred.next_positions.shape == (2, 3, 3)
    assert pred.next_velocities.shape == (2, 3, 3)
    assert pred.next_orientations.shape == (2, 3, 4)
    assert pred.physics_latent.shape == (2, TemporalWorldModel.PHYSICS_LATENT_DIM)
    print("  [main] forward() shape assertions             ... OK")

    # ── rollout smoke tests ──────────────────────────────────────────────
    scene_embeddings = F.normalize(torch.randn(2, 8, config.graph_dim), p=2, dim=-1)
    node_embeddings = F.normalize(torch.randn(2, 3, config.graph_dim), p=2, dim=-1)

    model.eval()
    with torch.no_grad():
        ar_preds = model.rollout(scene_embeddings, horizon=5, mode="autoregressive",
                                  node_embeddings=node_embeddings)
        assert len(ar_preds) == 5
        assert ar_preds[-1].next_positions.shape == (2, 3, 3)
        print("  [main] autoregressive_rollout()                ... OK")

        gt_future = F.normalize(torch.randn(2, 5, config.graph_dim), p=2, dim=-1)
        tf_preds = model.rollout(scene_embeddings, horizon=5, mode="teacher_force",
                                  node_embeddings=node_embeddings, ground_truth_future=gt_future)
        assert len(tf_preds) == 5
        print("  [main] teacher_force_rollout()                 ... OK")

        beam_preds = model.rollout(scene_embeddings, horizon=4, mode="beam",
                                    node_embeddings=node_embeddings, beam_width=3)
        assert len(beam_preds) == 4
        print("  [main] beam_rollout()                          ... OK")

    # ── world memory bank ─────────────────────────────────────────────────
    mem_id = model.remember(output.predictions[0].next_scene_embedding[0],
                             output.predictions[0].latent_state[0], timestamp=0.0)
    nearest = model.nearest_memories(output.predictions[0].next_scene_embedding[0], k=1)
    assert nearest and nearest[0][0] == mem_id
    print("  [main] remember() / nearest_memories()         ... OK")

    # ── loss interfaces ──────────────────────────────────────────────────
    pos_a = torch.randn(2, 3, 3)
    pos_b = torch.randn(2, 3, 3)
    _ = TemporalWorldModel.position_loss(pos_a, pos_b)
    quat_a = F.normalize(torch.randn(2, 3, 4), dim=-1)
    quat_b = F.normalize(torch.randn(2, 3, 4), dim=-1)
    _ = TemporalWorldModel.orientation_loss(quat_a, quat_b)
    _ = TemporalWorldModel.latent_kl_loss(output.latent_mu, output.latent_logvar)
    _ = TemporalWorldModel.physics_consistency_loss(pos_a, torch.randn(2, 3, 3), pos_b, torch.randn(2, 3, 3), dt=0.01)
    print("  [main] loss interfaces (position/orientation/KL/physics) ... OK")

    # ── expanded physics latent (item 4) ──────────────────────────────────
    assert TemporalWorldModel.PHYSICS_LATENT_DIM == 17
    assert sum(d for _, d in TemporalWorldModel.PHYSICS_LATENT_LAYOUT) == 17
    force_slice = TemporalWorldModel.physics_latent_slice(pred.physics_latent, "force")
    assert force_slice.shape == (2, 3)
    print("  [main] PHYSICS_LATENT_DIM=17 / physics_latent_slice()    ... OK")

    # ── energy / momentum losses (item 5) ─────────────────────────────────
    vel_a = torch.randn(2, 3, 3)
    vel_b = torch.randn(2, 3, 3)
    _ = TemporalWorldModel.energy_loss(vel_a, vel_b)
    _ = TemporalWorldModel.momentum_loss(vel_a, vel_b, mass=torch.rand(2, 3, 1) + 0.5)
    assert TemporalWorldModel.energy_loss(vel_a, vel_a).item() == 0.0
    print("  [main] energy_loss() / momentum_loss()                  ... OK")

    # ── diffusion-family placeholders (item 9) ────────────────────────────
    assert hasattr(model, "flow_matching_predictor")
    assert hasattr(model, "rectified_flow_predictor")
    try:
        model.flow_matching_predictor(output.latent_mu, horizon=5)
        raise AssertionError("FlowMatchingPredictor should have raised NotImplementedError")
    except NotImplementedError:
        pass
    try:
        model.rectified_flow_predictor(output.latent_mu, horizon=5)
        raise AssertionError("RectifiedFlowPredictor should have raised NotImplementedError")
    except NotImplementedError:
        pass
    print("  [main] FlowMatchingPredictor / RectifiedFlowPredictor   ... OK")

    # ── get_world_latent() (item 10) ──────────────────────────────────────
    world_latent = model.get_world_latent(output, step=0)
    assert set(world_latent.keys()) == {"scene_token", "memory_state", "latent_state", "physics_latent"}
    assert world_latent["physics_latent"].shape == (2, TemporalWorldModel.PHYSICS_LATENT_DIM)
    print("  [main] get_world_latent()                               ... OK")

    # ── sample_trajectory() (item 12) ─────────────────────────────────────
    with torch.no_grad():
        trajectories = model.sample_trajectory(
            scene_embeddings, horizon=3, num_samples=3, node_embeddings=node_embeddings,
        )
    assert len(trajectories) == 3
    assert all(len(t) == 3 for t in trajectories)
    assert trajectories[0][-1].next_positions.shape == (2, 3, 3)
    print("  [main] sample_trajectory()                              ... OK")

    # ── beam rollout with physics-weighted scoring (item 13) ──────────────
    with torch.no_grad():
        beam_preds_weighted = model.beam_rollout(
            scene_embeddings, horizon=4, beam_width=3,
            node_embeddings=node_embeddings, physics_weight=0.5,
        )
    assert len(beam_preds_weighted) == 4
    assert beam_preds_weighted[-1].next_positions.shape == (2, 3, 3)
    print("  [main] beam_rollout() with physics_weight               ... OK")

    # ── downstream pipeline export placeholders (item 15) ──────────────────
    for name, args in (
        ("to_state_engine",     (pred,)),
        ("to_trajectory_engine", (output.predictions,)),
        ("to_renderer",          (pred,)),
        ("to_video_diffusion",   (output,)),
    ):
        try:
            getattr(model, name)(*args)
            raise AssertionError(f"{name} should have raised NotImplementedError")
        except NotImplementedError:
            pass
    print("  [main] to_state_engine/to_trajectory_engine/to_renderer/to_video_diffusion ... OK")

    # ── save / load round-trip ───────────────────────────────────────────
    save_path = Path("/tmp/physworldlm_temporal_world_model_debug.pt")
    model.save_pretrained(save_path)
    loaded = TemporalWorldModel.load_pretrained(save_path)
    assert loaded.count_parameters()["total"] == model.count_parameters()["total"]
    print("  [main] save_pretrained() / load_pretrained()   ... OK")

    print("\n[main] all assertions passed. done.")


if __name__ == "__main__":
    main()
