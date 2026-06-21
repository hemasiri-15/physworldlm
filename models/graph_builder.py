"""
models/graph_builder.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — third learned/structural component.

Architecture
────────────
    EntityEncoder.get_embedding()        RelationEncoder.encode_relation()
            entity embeddings                  relation embeddings
                    │                                   │
                    └─────────────────┬─────────────────┘
                                       ▼
                                 GraphBuilder
                                       │
            ┌──────────────┬──────────┼──────────┬──────────────┐
            ▼              ▼          ▼          ▼              ▼
      SemanticGraph  SpatialGraph PhysicsGraph TemporalGraph EventGraph
            │              │          │          │              │
            └──────────────┴──────────┴──────────┴──────────────┘
                                       ▼
                              GraphAttentionAggregator
                                       │
                                       ▼
                         scene_token / scene_embedding
                                       │
                                       ▼
                                  SceneGraph
                                       │
                                       ▼
                                  WorldSpec
                                       │
                                       ▼
                            TemporalWorldModel (future)
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼             ▼
                       Bullet       MuJoCo       Isaac / Gazebo
                                       │
                                       ▼
                              Trajectory Engine
                                       │
                                       ▼
                                   Renderer
                                       │
                                       ▼
                                    Video

GraphBuilder is the bridge between learned intelligence (EntityEncoder +
RelationEncoder) and physics simulation (WorldSpec → Bullet/MuJoCo/Isaac).
This file is designed to never require redesign downstream: every node and
edge carries the full embedding + physics + temporal payload that later
stages (WorldSpec, TemporalWorldModel, physics engines) will need.

Design principles (mirrors models/entity_encoder.py, models/relation_encoder.py)
─────────────────────────────────────────────────────────────────────────────
* GraphBuilderConfig        — zero magic numbers; all dimensions configurable.
* GraphNode / GraphEdge / HyperEdge / SceneGraph — typed dataclasses, never
  reduce embeddings to labels; embeddings are preserved end-to-end.
* Hierarchical sub-graphs   — Semantic / Spatial / Physics / Temporal /
  Event / Constraint, all views over the same node/edge registries.
* GraphAttentionAggregator  — learned node+edge attention → scene_embedding.
* Node/edge registries      — duplicate prevention, confidence-aware updates.
* Adjacency + edge_index    — PyTorch Geometric compatible.
* Graph statistics          — degree, density, components, isolated nodes.
* Query API                 — neighbors, paths, subgraphs, k-NN over scenes.
* Export/import             — networkx, dict, JSON, YAML, WorldSpec.
* Serialization             — torch.save / torch.load round-trip.
* Graph memory bank         — nearest_graphs() over stored scene embeddings.
* Explainability            — attention weights, edge confidence, stats.
* Validation                — duplicate/orphan/self-loop/NaN/shape checks.
* GNN placeholders          — GraphConv / GAT / MessagePassing interfaces
  only; NOT trained here.
* Physics/Bullet/MuJoCo/Isaac/Gazebo export hooks — interfaces only,
  intentionally raising NotImplementedError; real export lives in later
  files (WorldSpec exporters, TemporalWorldModel).

Scope discipline
─────────────────
This module implements *only* models/graph_builder.py as specified for
PHYSWORLDLM FILE 7. It deliberately does NOT implement: training, the
TemporalWorldModel, Bullet/MuJoCo/Isaac/Gazebo physics back-ends, a
renderer, or trajectory generation. Those are real downstream files; this
module exposes typed interfaces / placeholders for them so the contract is
visible without taking on their implementation.

The following upstream components are COMPLETE and are NOT modified here:

    dataset_gen/generate_entity_dataset.py
    dataset_gen/split_entity_dataset.py
    training/entity_dataset.py
    models/entity_encoder.py
    training/train_entity_encoder.py
    models/relation_encoder.py
    world_spec.py
"""

from __future__ import annotations

import json
import math
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Deferred import to avoid a hard dependency / circular import at module
# load time — world_spec.py lives at the project root.
try:
    from world_spec import (
        WorldSpec, Entity, Environment, Interaction, SimulationGraph,
        PhysicsState, BoundingBox, Vec3,
    )
    _WORLDSPEC_AVAILABLE = True
except ImportError:  # pragma: no cover - degrade gracefully if not on path
    _WORLDSPEC_AVAILABLE = False

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NETWORKX_AVAILABLE = False

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants & vocabularies
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_DIM:   int = 512
RELATION_DIM: int = 512
GRAPH_DIM:    int = 512

GraphLayer = Literal[
    "semantic", "spatial", "physics", "temporal", "event", "constraint",
]

AggregateMethod = Literal["mean", "max", "attention"]

#: Sub-graph membership — which physics/relation categories route to which
#: hierarchical layer. Mirrors RelationEncoder's relation taxonomy.
PHYSICS_RELATION_TYPES: Tuple[str, ...] = (
    "contact", "collision", "support", "constraint",
    "friction", "spring", "joint", "gravity",
)

TEMPORAL_RELATION_TYPES: Tuple[str, ...] = (
    "before", "after", "simultaneous", "causes", "triggered_by",
)

SPATIAL_RELATION_TYPES: Tuple[str, ...] = (
    "above", "below", "left_of", "right_of", "in_front_of", "behind",
    "inside", "contains", "intersecting", "near", "far",
    "overlapping", "touching", "surrounding",
)

EVENT_TYPES: Tuple[str, ...] = (
    "collision_event", "explosion_event", "launch_event", "impact_event",
    "fracture_event", "ignition_event",
)


def _relation_layer(relation_type: str) -> GraphLayer:
    """Route a relation_type string to its hierarchical sub-graph layer.

    Args:
        relation_type: Raw relation/edge type string (e.g. "colliding_with").

    Returns:
        One of the six :data:`GraphLayer` values. Defaults to ``"semantic"``
        for anything not recognised, so no relation is ever dropped.
    """
    rt = relation_type.lower()
    if rt in PHYSICS_RELATION_TYPES or "collid" in rt or "contact" in rt:
        return "physics"
    if rt in TEMPORAL_RELATION_TYPES:
        return "temporal"
    if rt in SPATIAL_RELATION_TYPES:
        return "spatial"
    if rt in ("causes", "triggered_by") or "event" in rt:
        return "event"
    if rt in ("constraint", "joint"):
        return "constraint"
    return "semantic"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  GraphBuilderConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphBuilderConfig:
    """All hyperparameters for GraphBuilder — no magic numbers inside the model.

    Attributes:
        entity_dim:               Dimension of incoming entity embeddings.
        relation_dim:              Dimension of incoming relation embeddings.
        graph_dim:                 Dimension of the final scene_embedding /
                                   scene_token.
        max_nodes:                 Hard cap on nodes per scene graph.
        max_edges:                 Hard cap on edges per scene graph.
        normalize_embeddings:      L2-normalise node/edge/scene embeddings.
        aggregate_method:          ``"mean"``, ``"max"``, or ``"attention"``.
        store_edge_embeddings:     Persist relation embeddings on edges.
        store_node_embeddings:     Persist entity embeddings on nodes.
        build_adjacency_matrix:    Build a dense ``(N, N)`` adjacency matrix.
        build_edge_index:          Build a PyG-style ``(2, E)`` edge_index.
        compute_scene_embedding:   Run the attention aggregator on build.
        compute_graph_statistics:  Compute degree/density/components on build.
        num_attention_heads:       Heads for :class:`GraphAttentionAggregator`.
        duplicate_confidence_eps:  Minimum confidence improvement required to
                                   overwrite an existing duplicate edge.
        memory_bank_capacity:      Max scenes retained in ``graph_memory``.
    """

    entity_dim:               int   = ENTITY_DIM
    relation_dim:              int   = RELATION_DIM
    graph_dim:                 int   = GRAPH_DIM
    max_nodes:                 int   = 256
    max_edges:                 int   = 4096
    normalize_embeddings:      bool  = True
    aggregate_method:          AggregateMethod = "attention"
    store_edge_embeddings:     bool  = True
    store_node_embeddings:     bool  = True
    build_adjacency_matrix:    bool  = True
    build_edge_index:          bool  = True
    compute_scene_embedding:   bool  = True
    compute_graph_statistics:  bool  = True
    num_attention_heads:       int   = 8
    duplicate_confidence_eps:  float = 1e-3
    memory_bank_capacity:      int   = 10_000

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be ≥ 1, got {self.max_nodes}")
        if self.max_edges < 1:
            raise ValueError(f"max_edges must be ≥ 1, got {self.max_edges}")
        if self.graph_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"graph_dim ({self.graph_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.aggregate_method not in ("mean", "max", "attention"):
            raise ValueError(
                f"aggregate_method must be one of mean/max/attention, "
                f"got {self.aggregate_method!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Core dataclasses: GraphNode, GraphEdge, HyperEdge
# ─────────────────────────────────────────────────────────────────────────────

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class GraphNode:
    """One entity (or event) node in the scene graph.

    Embeddings are never discarded in favour of labels — ``embedding`` is
    the canonical 512-d representation produced by ``EntityEncoder``, kept
    alongside the human-readable attribute fields for explainability.

    Attributes:
        node_id:      Unique identifier within the graph.
        name:         Human-readable entity name (e.g. "red car").
        entity_type:  Coarse type (vehicle / structure / event / ...).
        embedding:    ``(entity_dim,)`` tensor from EntityEncoder, or
                      ``None`` if this is a pure structural/event node.
        attributes:   Free-form attribute dict (colour, label confidences…).
        material:     Material string (see world_spec.MATERIAL_DEFAULTS).
        mass_class:   Coarse mass bucket (e.g. "light", "heavy").
        shape:        Coarse shape descriptor.
        mobility:     "static" / "dynamic" / "kinematic".
        size_class:   Coarse size bucket.
        position:     ``(3,)`` world position tensor, or None.
        velocity:     ``(3,)`` world velocity tensor, or None.
        orientation:  ``(3,)`` Euler-angle orientation tensor, or None.
        is_event:     True if this node represents an event rather than a
                      physical object (collision_event, explosion_event…).
        event_type:   Populated when ``is_event=True``; one of
                      :data:`EVENT_TYPES` or a custom event string.
        importance:   Explainability score in [0, 1]; populated by the
                      attention aggregator after a forward pass.
        metadata:     Free-form bag for anything not covered above.
    """

    node_id:      str
    name:         str
    entity_type:  str = "object"
    embedding:    Optional[torch.Tensor] = None
    attributes:   Dict[str, Any] = field(default_factory=dict)
    material:     str = "generic"
    mass_class:   str = "medium"
    shape:        str = "generic"
    mobility:     str = "dynamic"
    size_class:   str = "medium"
    position:     Optional[torch.Tensor] = None
    velocity:     Optional[torch.Tensor] = None
    orientation:  Optional[torch.Tensor] = None
    is_event:     bool = False
    event_type:   Optional[str] = None
    importance:   float = 0.0
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_embeddings: bool = True) -> dict:
        """Serialise to a plain dict. Tensors become nested lists."""
        return {
            "node_id":     self.node_id,
            "name":        self.name,
            "entity_type": self.entity_type,
            "embedding":   (self.embedding.tolist()
                             if include_embeddings and self.embedding is not None
                             else None),
            "attributes":  self.attributes,
            "material":    self.material,
            "mass_class":  self.mass_class,
            "shape":       self.shape,
            "mobility":    self.mobility,
            "size_class":  self.size_class,
            "position":    self.position.tolist() if self.position is not None else None,
            "velocity":    self.velocity.tolist() if self.velocity is not None else None,
            "orientation": self.orientation.tolist() if self.orientation is not None else None,
            "is_event":    self.is_event,
            "event_type":  self.event_type,
            "importance":  self.importance,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphNode":
        def _t(v: Optional[list]) -> Optional[torch.Tensor]:
            return torch.tensor(v, dtype=torch.float32) if v is not None else None

        return cls(
            node_id=d["node_id"],
            name=d["name"],
            entity_type=d.get("entity_type", "object"),
            embedding=_t(d.get("embedding")),
            attributes=d.get("attributes", {}),
            material=d.get("material", "generic"),
            mass_class=d.get("mass_class", "medium"),
            shape=d.get("shape", "generic"),
            mobility=d.get("mobility", "dynamic"),
            size_class=d.get("size_class", "medium"),
            position=_t(d.get("position")),
            velocity=_t(d.get("velocity")),
            orientation=_t(d.get("orientation")),
            is_event=d.get("is_event", False),
            event_type=d.get("event_type"),
            importance=d.get("importance", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class GraphEdge:
    """One pairwise relation (edge) in the scene graph.

    Attributes:
        edge_id:             Unique identifier within the graph.
        source_id:            Source ``GraphNode.node_id``.
        target_id:            Target ``GraphNode.node_id``.
        relation_type:        Fine-grained relation string (e.g. "resting_on").
        relation_category:    Coarse RelationEncoder head name (spatial /
                              contact / interaction / motion / support /
                              containment / visibility / causality).
        relation_embedding:   ``(relation_dim,)`` tensor from RelationEncoder,
                              or None.
        confidence:           Scalar confidence in [0, 1].
        variance:             Optional scalar uncertainty estimate; populated
                              alongside ``confidence`` when available.
        layer:                Hierarchical sub-graph this edge belongs to;
                              auto-derived from ``relation_type`` if not set.
        directed:             Whether this edge is directed (default True).
        physics_properties:   mass/friction/restitution/joint_type/etc.
        temporal_properties:  timestamp/duration/ordering info.
        metadata:             Free-form bag.
    """

    source_id:           str
    target_id:            str
    relation_type:        str
    edge_id:              str = field(default_factory=lambda: _new_id("edge"))
    relation_category:    str = "semantic"
    relation_embedding:   Optional[torch.Tensor] = None
    confidence:           float = 1.0
    variance:             Optional[float] = None
    layer:                GraphLayer = "semantic"
    directed:              bool = True
    physics_properties:    Dict[str, Any] = field(default_factory=dict)
    temporal_properties:   Dict[str, Any] = field(default_factory=dict)
    metadata:              Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"GraphEdge confidence must be in [0, 1], got {self.confidence}"
            )

    def to_dict(self, include_embeddings: bool = True) -> dict:
        return {
            "edge_id":             self.edge_id,
            "source_id":           self.source_id,
            "target_id":           self.target_id,
            "relation_type":       self.relation_type,
            "relation_category":   self.relation_category,
            "relation_embedding":  (self.relation_embedding.tolist()
                                     if include_embeddings and self.relation_embedding is not None
                                     else None),
            "confidence":          self.confidence,
            "variance":            self.variance,
            "layer":               self.layer,
            "directed":            self.directed,
            "physics_properties":  self.physics_properties,
            "temporal_properties": self.temporal_properties,
            "metadata":            self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphEdge":
        emb = d.get("relation_embedding")
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            relation_type=d["relation_type"],
            edge_id=d.get("edge_id", _new_id("edge")),
            relation_category=d.get("relation_category", "semantic"),
            relation_embedding=(torch.tensor(emb, dtype=torch.float32)
                                 if emb is not None else None),
            confidence=d.get("confidence", 1.0),
            variance=d.get("variance"),
            layer=d.get("layer", "semantic"),
            directed=d.get("directed", True),
            physics_properties=d.get("physics_properties", {}),
            temporal_properties=d.get("temporal_properties", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass
class HyperEdge:
    """A relation spanning more than two nodes (e.g. a multi-object collision
    chain, or a support system: ``[car, jack, ground]``).

    Attributes:
        hyperedge_id:  Unique identifier.
        node_ids:      Ordered list of participating node ids (len ≥ 2).
        relation_type: Description of the joint relation (e.g. "pile_up").
        relation_category: Coarse category, mirrors :class:`GraphEdge`.
        embedding:     Optional pooled embedding over participants.
        confidence:    Scalar confidence in [0, 1].
        metadata:      Free-form bag.
    """

    node_ids:           List[str]
    relation_type:       str
    hyperedge_id:        str = field(default_factory=lambda: _new_id("hyper"))
    relation_category:    str = "semantic"
    embedding:            Optional[torch.Tensor] = None
    confidence:           float = 1.0
    metadata:             Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.node_ids) < 2:
            raise ValueError(
                f"HyperEdge requires ≥ 2 participating nodes, got "
                f"{len(self.node_ids)}"
            )

    def to_dict(self, include_embeddings: bool = True) -> dict:
        return {
            "hyperedge_id":      self.hyperedge_id,
            "node_ids":          self.node_ids,
            "relation_type":     self.relation_type,
            "relation_category": self.relation_category,
            "embedding":         (self.embedding.tolist()
                                   if include_embeddings and self.embedding is not None
                                   else None),
            "confidence":        self.confidence,
            "metadata":          self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HyperEdge":
        emb = d.get("embedding")
        return cls(
            node_ids=d["node_ids"],
            relation_type=d["relation_type"],
            hyperedge_id=d.get("hyperedge_id", _new_id("hyper")),
            relation_category=d.get("relation_category", "semantic"),
            embedding=(torch.tensor(emb, dtype=torch.float32)
                       if emb is not None else None),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  SceneGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneGraph:
    """Complete structured world representation for one scene at one moment.

    Attributes:
        scene_id:         Unique scene identifier.
        version:           Monotonically increasing version counter, bumped
                           on every structural mutation (see
                           ``GraphBuilder._bump_version``).
        parent_scene_id:   ``scene_id`` of the previous timestep's graph, or
                           ``None`` for t0. Enables ``t0 → t1 → t2`` chains
                           without rewriting graphs.
        nodes:             ``node_id → GraphNode``.
        edges:             ``edge_id → GraphEdge``.
        hyperedges:        ``hyperedge_id → HyperEdge``.
        adjacency:          Dense ``(N, N)`` adjacency matrix, or None.
        edge_index:         PyG-style ``(2, E)`` long tensor, or None.
        scene_embedding:    ``(graph_dim,)`` pooled representation, or None.
        scene_token:        Alias/CLS-style token identical in shape to
                            ``scene_embedding``; kept distinct so a future
                            TemporalWorldModel can treat it as a dedicated
                            sequence token without ambiguity.
        global_attributes:  Scene-level metadata (weather, lighting, …).
        timestamp:           Simulation time this graph represents (seconds).
        statistics:          Populated by ``compute_statistics()``.
        attention_weights:   Last computed node/edge attention, for
                             explainability.
        metadata:            Free-form bag.
    """

    scene_id:           str = field(default_factory=lambda: _new_id("scene"))
    version:             int = 0
    parent_scene_id:      Optional[str] = None
    nodes:                Dict[str, GraphNode] = field(default_factory=dict)
    edges:                Dict[str, GraphEdge] = field(default_factory=dict)
    hyperedges:           Dict[str, HyperEdge] = field(default_factory=dict)
    adjacency:            Optional[torch.Tensor] = None
    edge_index:            Optional[torch.Tensor] = None
    scene_embedding:        Optional[torch.Tensor] = None
    scene_token:             Optional[torch.Tensor] = None
    global_attributes:        Dict[str, Any] = field(default_factory=dict)
    timestamp:                 float = 0.0
    statistics:                 Dict[str, Any] = field(default_factory=dict)
    attention_weights:           Optional[Dict[str, torch.Tensor]] = None
    metadata:                     Dict[str, Any] = field(default_factory=dict)

    # ── convenience views ────────────────────────────────────────────────

    def node_list(self) -> List[GraphNode]:
        return list(self.nodes.values())

    def edge_list(self) -> List[GraphEdge]:
        return list(self.edges.values())

    def layer_edges(self, layer: GraphLayer) -> List[GraphEdge]:
        """Return all edges belonging to a given hierarchical sub-graph."""
        return [e for e in self.edges.values() if e.layer == layer]

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    # ── serialisation ────────────────────────────────────────────────────

    def to_dict(self, include_embeddings: bool = True) -> dict:
        return {
            "scene_id":          self.scene_id,
            "version":           self.version,
            "parent_scene_id":   self.parent_scene_id,
            "nodes":             {k: v.to_dict(include_embeddings) for k, v in self.nodes.items()},
            "edges":             {k: v.to_dict(include_embeddings) for k, v in self.edges.items()},
            "hyperedges":        {k: v.to_dict(include_embeddings) for k, v in self.hyperedges.items()},
            "adjacency":         self.adjacency.tolist() if self.adjacency is not None else None,
            "edge_index":        self.edge_index.tolist() if self.edge_index is not None else None,
            "scene_embedding":   self.scene_embedding.tolist() if (include_embeddings and self.scene_embedding is not None) else None,
            "scene_token":       self.scene_token.tolist() if (include_embeddings and self.scene_token is not None) else None,
            "global_attributes": self.global_attributes,
            "timestamp":         self.timestamp,
            "statistics":        self.statistics,
            "metadata":          self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for to_yaml(); pip install pyyaml")
        return _yaml.dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneGraph":
        nodes = {k: GraphNode.from_dict(v) for k, v in d.get("nodes", {}).items()}
        edges = {k: GraphEdge.from_dict(v) for k, v in d.get("edges", {}).items()}
        hyperedges = {k: HyperEdge.from_dict(v) for k, v in d.get("hyperedges", {}).items()}
        adj = d.get("adjacency")
        ei  = d.get("edge_index")
        se  = d.get("scene_embedding")
        st  = d.get("scene_token")
        return cls(
            scene_id=d.get("scene_id", _new_id("scene")),
            version=d.get("version", 0),
            parent_scene_id=d.get("parent_scene_id"),
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            adjacency=torch.tensor(adj, dtype=torch.float32) if adj is not None else None,
            edge_index=torch.tensor(ei, dtype=torch.long) if ei is not None else None,
            scene_embedding=torch.tensor(se, dtype=torch.float32) if se is not None else None,
            scene_token=torch.tensor(st, dtype=torch.float32) if st is not None else None,
            global_attributes=d.get("global_attributes", {}),
            timestamp=d.get("timestamp", 0.0),
            statistics=d.get("statistics", {}),
            metadata=d.get("metadata", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  GraphAttentionAggregator
# ─────────────────────────────────────────────────────────────────────────────

class GraphAttentionAggregator(nn.Module):
    """Learned attention pooling over node and edge embeddings → scene_embedding.

    Structure
    ─────────
        node_embeddings  (N, D) ──┐
                                   ├─ NodeAttention  ──┐
        edge_embeddings  (E, D) ──┤                    ├─ CrossAttention ─▶ scene_embedding (D,)
                                   └─ EdgeAttention  ──┘

    Falls back to mean/max pooling when ``method != "attention"`` so the
    rest of the pipeline never has to special-case the aggregation method.

    Args:
        dim:        Embedding dimension (graph_dim).
        num_heads:  Attention heads.
        dropout:    Dropout probability.
    """

    def __init__(self, dim: int = GRAPH_DIM, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.node_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.node_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.edge_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_embeddings: Optional[torch.Tensor],
        edge_embeddings: Optional[torch.Tensor],
        method: AggregateMethod = "attention",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate node/edge embeddings into a single scene embedding.

        Args:
            node_embeddings: ``(N, dim)`` or None if no nodes have embeddings.
            edge_embeddings: ``(E, dim)`` or None if no edges have embeddings.
            method:          ``"mean"``, ``"max"``, or ``"attention"``.

        Returns:
            Tuple of (``scene_embedding`` shape ``(dim,)``, attention-weight
            dict — empty unless ``method == "attention"``).
        """
        device = next(self.parameters()).device
        pieces: List[torch.Tensor] = []
        if node_embeddings is not None and node_embeddings.numel() > 0:
            pieces.append(node_embeddings.to(device))
        if edge_embeddings is not None and edge_embeddings.numel() > 0:
            pieces.append(edge_embeddings.to(device))

        if not pieces:
            return torch.zeros(self.dim, device=device), {}

        if method == "mean":
            return torch.cat(pieces, dim=0).mean(dim=0), {}
        if method == "max":
            return torch.cat(pieces, dim=0).max(dim=0).values, {}

        # attention pooling
        attn_weights: Dict[str, torch.Tensor] = {}
        node_pooled = torch.zeros(1, self.dim, device=device)
        edge_pooled = torch.zeros(1, self.dim, device=device)

        if node_embeddings is not None and node_embeddings.numel() > 0:
            seq = node_embeddings.to(device).unsqueeze(0)        # (1, N, dim)
            q = self.node_query.to(device)
            pooled, w = self.node_attn(q, seq, seq, need_weights=True)
            node_pooled = pooled.squeeze(0)                       # (1, dim)
            attn_weights["node_attention"] = w.squeeze(0)         # (1, N)

        if edge_embeddings is not None and edge_embeddings.numel() > 0:
            seq = edge_embeddings.to(device).unsqueeze(0)         # (1, E, dim)
            q = self.node_query.to(device)
            pooled, w = self.edge_attn(q, seq, seq, need_weights=True)
            edge_pooled = pooled.squeeze(0)
            attn_weights["edge_attention"] = w.squeeze(0)

        fused = self.fusion(torch.cat([node_pooled, edge_pooled], dim=-1))  # (1, dim)
        scene_embedding = self.out_norm(fused).squeeze(0)                    # (dim,)
        return scene_embedding, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  GNN placeholders (interfaces only — NOT trained here)
# ─────────────────────────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """Placeholder interface for a future spectral/spatial graph-conv layer.

    Not trained or wired into :class:`GraphBuilder` in this file. Exists so
    downstream code can type-check against a stable interface ahead of the
    real implementation.
    """

    def __init__(self, in_dim: int = GRAPH_DIM, out_dim: int = GRAPH_DIM) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "GraphConv is a placeholder interface; implement message passing "
            "in a future training-stage file."
        )


class GAT(nn.Module):
    """Placeholder interface for a future Graph Attention Network layer."""

    def __init__(self, in_dim: int = GRAPH_DIM, out_dim: int = GRAPH_DIM, heads: int = 4) -> None:
        super().__init__()
        self.in_dim, self.out_dim, self.heads = in_dim, out_dim, heads

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "GAT is a placeholder interface; not implemented in graph_builder.py."
        )


class MessagePassing(nn.Module):
    """Placeholder interface mirroring PyTorch Geometric's MessagePassing base."""

    def __init__(self, aggr: str = "add") -> None:
        super().__init__()
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "MessagePassing is a placeholder interface; not implemented in "
            "graph_builder.py."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  GraphBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """Converts entities + relations into a fully structured :class:`SceneGraph`.

    This is the bridge between learned intelligence (``EntityEncoder`` /
    ``RelationEncoder``) and physics simulation (``WorldSpec`` →
    Bullet/MuJoCo/Isaac/Gazebo). It is NOT an ``nn.Module`` itself — it is a
    stateful registry/orchestrator that *owns* a :class:`GraphAttentionAggregator`
    for scene-embedding computation, but the graph data structures
    (:class:`GraphNode`, :class:`GraphEdge`) are plain dataclasses so they
    serialise cleanly and stay framework-agnostic everywhere except the
    embedding tensors themselves.

    Usage
    ─────
    ::

        builder = GraphBuilder(GraphBuilderConfig())

        car   = builder.add_node(name="car",   entity_type="vehicle",
                                  embedding=car_emb)
        truck = builder.add_node(name="truck", entity_type="vehicle",
                                  embedding=truck_emb)
        builder.add_edge(car, truck, relation_type="colliding_with",
                          relation_category="contact",
                          relation_embedding=rel_emb, confidence=0.92)

        scene_graph = builder.build()
        print(scene_graph.scene_embedding.shape)   # (512,)

    Args:
        config: :class:`GraphBuilderConfig` instance. Defaults are used when
                omitted.
    """

    def __init__(self, config: Optional[GraphBuilderConfig] = None) -> None:
        self.config: GraphBuilderConfig = config or GraphBuilderConfig()

        self.node_registry: Dict[str, GraphNode] = {}
        self.edge_registry: Dict[str, GraphEdge] = {}
        self.hyperedge_registry: Dict[str, HyperEdge] = {}

        # source_id|target_id|relation_type -> edge_id, for O(1) duplicate checks
        self._edge_key_index: Dict[str, str] = {}

        self.aggregator = GraphAttentionAggregator(
            dim=self.config.graph_dim,
            num_heads=self.config.num_attention_heads,
        )

        self._scene_id: str = _new_id("scene")
        self._version: int = 0
        self._parent_scene_id: Optional[str] = None
        self._timestamp: float = 0.0
        self.global_attributes: Dict[str, Any] = {}

        # graph_memory: scene_id -> {"embedding": Tensor, "timestamp": float, ...}
        self.graph_memory: Dict[str, Dict[str, Any]] = {}

        self._last_scene_graph: Optional[SceneGraph] = None

    # ── id / versioning ───────────────────────────────────────────────────

    def _bump_version(self) -> None:
        self._version += 1

    def new_scene(self, scene_id: Optional[str] = None, parent_scene_id: Optional[str] = None) -> None:
        """Reset the builder for a fresh scene, optionally chained to a parent.

        Args:
            scene_id:        Explicit scene id; auto-generated if omitted.
            parent_scene_id: Previous timestep's scene_id, for ``t0→t1→t2``
                             chains without rewriting graphs.
        """
        self.node_registry.clear()
        self.edge_registry.clear()
        self.hyperedge_registry.clear()
        self._edge_key_index.clear()
        self._scene_id = scene_id or _new_id("scene")
        self._parent_scene_id = parent_scene_id
        self._version = 0
        self._timestamp = 0.0
        self._last_scene_graph = None

    # ── node operations ───────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        entity_type: str = "object",
        embedding: Optional[torch.Tensor] = None,
        node_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Add a node to the graph.

        Args:
            name:        Human-readable entity name.
            entity_type:  Coarse type string.
            embedding:    ``(entity_dim,)`` tensor, or None.
            node_id:      Explicit id; auto-generated if omitted.
            **kwargs:     Forwarded to :class:`GraphNode` (material, shape,
                          position, is_event, event_type, …).

        Returns:
            The node's ``node_id``.

        Raises:
            ValueError: If ``max_nodes`` would be exceeded, or the embedding
                        has the wrong dimensionality.
        """
        if len(self.node_registry) >= self.config.max_nodes:
            raise ValueError(
                f"add_node: max_nodes={self.config.max_nodes} exceeded"
            )
        if embedding is not None:
            self._validate_embedding(embedding, self.config.entity_dim, "node embedding")
            if self.config.normalize_embeddings:
                embedding = F.normalize(embedding.reshape(1, -1), p=2, dim=-1).squeeze(0)
        nid = node_id or _new_id("node")
        if nid in self.node_registry:
            raise ValueError(f"add_node: node_id {nid!r} already exists")

        node = GraphNode(
            node_id=nid,
            name=name,
            entity_type=entity_type,
            embedding=embedding if self.config.store_node_embeddings else None,
            **kwargs,
        )
        self.node_registry[nid] = node
        self._bump_version()
        return nid

    def remove_node(self, node_id: str, cascade: bool = True) -> None:
        """Remove a node, optionally cascading to incident edges.

        Args:
            node_id: Node to remove.
            cascade: If True (default), also remove all edges touching this
                     node. If False and incident edges exist, raises.

        Raises:
            KeyError: If ``node_id`` doesn't exist.
            ValueError: If ``cascade=False`` and incident edges exist.
        """
        if node_id not in self.node_registry:
            raise KeyError(f"remove_node: unknown node_id {node_id!r}")
        incident = [e for e in self.edge_registry.values()
                    if e.source_id == node_id or e.target_id == node_id]
        if incident and not cascade:
            raise ValueError(
                f"remove_node: node {node_id!r} has {len(incident)} incident "
                "edges; pass cascade=True to remove them too"
            )
        for e in incident:
            self.remove_edge(e.edge_id)
        del self.node_registry[node_id]
        self._bump_version()

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        """Update fields on an existing node in-place.

        Args:
            node_id:  Node to update.
            **kwargs: Any :class:`GraphNode` field name → new value.

        Raises:
            KeyError: If ``node_id`` doesn't exist.
            AttributeError: If a kwarg isn't a valid GraphNode field.
        """
        if node_id not in self.node_registry:
            raise KeyError(f"update_node: unknown node_id {node_id!r}")
        node = self.node_registry[node_id]
        for k, v in kwargs.items():
            if not hasattr(node, k):
                raise AttributeError(f"update_node: GraphNode has no field {k!r}")
            if k == "embedding" and v is not None and self.config.normalize_embeddings:
                v = F.normalize(v.reshape(1, -1), p=2, dim=-1).squeeze(0)
            setattr(node, k, v)
        self._bump_version()

    def get_node(self, node_id: str) -> GraphNode:
        """Return the :class:`GraphNode` for ``node_id``.

        Raises:
            KeyError: If not found.
        """
        if node_id not in self.node_registry:
            raise KeyError(f"get_node: unknown node_id {node_id!r}")
        return self.node_registry[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_registry

    # ── edge operations ───────────────────────────────────────────────────

    def _edge_key(self, source_id: str, target_id: str, relation_type: str) -> str:
        return f"{source_id}|{target_id}|{relation_type}"

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        relation_category: str = "semantic",
        relation_embedding: Optional[torch.Tensor] = None,
        confidence: float = 1.0,
        variance: Optional[float] = None,
        layer: Optional[GraphLayer] = None,
        **kwargs: Any,
    ) -> str:
        """Add an edge, with duplicate-prevention by ``(source, target, type)``.

        If an identical edge already exists, it is overwritten only when the
        new ``confidence`` strictly improves on the existing one (by at
        least ``config.duplicate_confidence_eps``); otherwise the call is a
        no-op that returns the existing edge's id. This prevents repeated
        relation extraction (e.g. "car → truck" detected three times) from
        creating duplicate edges.

        Args:
            source_id:           Source node id.
            target_id:            Target node id.
            relation_type:        Fine-grained relation string.
            relation_category:    Coarse category (spatial/contact/...).
            relation_embedding:   ``(relation_dim,)`` tensor, or None.
            confidence:           Scalar confidence in [0, 1].
            variance:             Optional uncertainty estimate.
            layer:                Explicit sub-graph layer; auto-derived
                                  from ``relation_type`` if omitted.
            **kwargs:             Forwarded to :class:`GraphEdge`
                                  (physics_properties, temporal_properties,
                                  directed, metadata).

        Returns:
            The edge's ``edge_id`` (new or pre-existing).

        Raises:
            ValueError: If ``max_edges`` exceeded, embedding shape mismatch,
                        or referenced nodes don't exist.
        """
        if source_id not in self.node_registry:
            raise ValueError(f"add_edge: unknown source_id {source_id!r}")
        if target_id not in self.node_registry:
            raise ValueError(f"add_edge: unknown target_id {target_id!r}")

        key = self._edge_key(source_id, target_id, relation_type)
        if key in self._edge_key_index:
            existing_id = self._edge_key_index[key]
            existing = self.edge_registry[existing_id]
            if confidence > existing.confidence + self.config.duplicate_confidence_eps:
                existing.confidence = confidence
                existing.variance = variance
                if relation_embedding is not None:
                    existing.relation_embedding = self._prep_edge_embedding(relation_embedding)
                self._bump_version()
            return existing_id

        if len(self.edge_registry) >= self.config.max_edges:
            raise ValueError(f"add_edge: max_edges={self.config.max_edges} exceeded")

        if relation_embedding is not None:
            relation_embedding = self._prep_edge_embedding(relation_embedding)

        eid = _new_id("edge")
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            edge_id=eid,
            relation_category=relation_category,
            relation_embedding=relation_embedding if self.config.store_edge_embeddings else None,
            confidence=confidence,
            variance=variance,
            layer=layer or _relation_layer(relation_type),
            **kwargs,
        )
        self.edge_registry[eid] = edge
        self._edge_key_index[key] = eid
        self._bump_version()
        return eid

    def _prep_edge_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        self._validate_embedding(embedding, self.config.relation_dim, "edge embedding")
        if self.config.normalize_embeddings:
            embedding = F.normalize(embedding.reshape(1, -1), p=2, dim=-1).squeeze(0)
        return embedding

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge by id.

        Raises:
            KeyError: If ``edge_id`` doesn't exist.
        """
        if edge_id not in self.edge_registry:
            raise KeyError(f"remove_edge: unknown edge_id {edge_id!r}")
        edge = self.edge_registry.pop(edge_id)
        key = self._edge_key(edge.source_id, edge.target_id, edge.relation_type)
        self._edge_key_index.pop(key, None)
        self._bump_version()

    def update_edge(self, edge_id: str, **kwargs: Any) -> None:
        """Update fields on an existing edge in-place.

        Raises:
            KeyError: If ``edge_id`` doesn't exist.
            AttributeError: If a kwarg isn't a valid GraphEdge field.
        """
        if edge_id not in self.edge_registry:
            raise KeyError(f"update_edge: unknown edge_id {edge_id!r}")
        edge = self.edge_registry[edge_id]
        for k, v in kwargs.items():
            if not hasattr(edge, k):
                raise AttributeError(f"update_edge: GraphEdge has no field {k!r}")
            if k == "relation_embedding" and v is not None:
                v = self._prep_edge_embedding(v)
            setattr(edge, k, v)
        self._bump_version()

    def get_edge(self, edge_id: str) -> GraphEdge:
        if edge_id not in self.edge_registry:
            raise KeyError(f"get_edge: unknown edge_id {edge_id!r}")
        return self.edge_registry[edge_id]

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self.edge_registry

    # ── hyperedge operations ──────────────────────────────────────────────

    def add_hyperedge(
        self,
        node_ids: Sequence[str],
        relation_type: str,
        relation_category: str = "semantic",
        embedding: Optional[torch.Tensor] = None,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Add a hyperedge spanning ≥ 2 nodes (e.g. multi-object collisions).

        Args:
            node_ids:           Participating node ids (length ≥ 2).
            relation_type:       Joint relation description.
            relation_category:   Coarse category.
            embedding:           Optional pooled embedding over participants.
            confidence:           Scalar confidence in [0, 1].
            **kwargs:             Forwarded to :class:`HyperEdge`.

        Returns:
            The hyperedge's id.

        Raises:
            ValueError: If any node_id is unknown.
        """
        for nid in node_ids:
            if nid not in self.node_registry:
                raise ValueError(f"add_hyperedge: unknown node_id {nid!r}")
        if embedding is not None:
            embedding = self._prep_edge_embedding(embedding)
        hid = _new_id("hyper")
        he = HyperEdge(
            node_ids=list(node_ids),
            relation_type=relation_type,
            hyperedge_id=hid,
            relation_category=relation_category,
            embedding=embedding,
            confidence=confidence,
            **kwargs,
        )
        self.hyperedge_registry[hid] = he
        self._bump_version()
        return hid

    # ── query functions ───────────────────────────────────────────────────

    def neighbors(self, node_id: str) -> List[str]:
        """All node ids adjacent to ``node_id`` via any edge direction."""
        out: Set[str] = set()
        for e in self.edge_registry.values():
            if e.source_id == node_id:
                out.add(e.target_id)
            elif e.target_id == node_id and not e.directed:
                out.add(e.source_id)
            elif e.target_id == node_id:
                # directed edge pointing INTO node_id still counts as a neighbor
                out.add(e.source_id)
        out.discard(node_id)
        return sorted(out)

    def incoming_edges(self, node_id: str) -> List[GraphEdge]:
        """All edges with ``target_id == node_id``."""
        return [e for e in self.edge_registry.values() if e.target_id == node_id]

    def outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """All edges with ``source_id == node_id``."""
        return [e for e in self.edge_registry.values() if e.source_id == node_id]

    def connected_nodes(self, node_id: str) -> Set[str]:
        """Full connected component containing ``node_id`` (BFS, undirected)."""
        if node_id not in self.node_registry:
            raise KeyError(f"connected_nodes: unknown node_id {node_id!r}")
        visited: Set[str] = {node_id}
        queue: deque = deque([node_id])
        adj = self._undirected_adjacency()
        while queue:
            cur = queue.popleft()
            for nxt in adj.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _undirected_adjacency(self) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for e in self.edge_registry.values():
            adj[e.source_id].add(e.target_id)
            adj[e.target_id].add(e.source_id)
        return adj

    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """BFS path (any path, undirected) from ``source_id`` to ``target_id``.

        Returns:
            List of node ids from source to target inclusive, or ``None`` if
            unreachable.
        """
        if source_id not in self.node_registry or target_id not in self.node_registry:
            raise KeyError("find_path: source/target node not in graph")
        if source_id == target_id:
            return [source_id]
        adj = self._undirected_adjacency()
        visited = {source_id}
        queue: deque = deque([[source_id]])
        while queue:
            path = queue.popleft()
            cur = path[-1]
            for nxt in adj.get(cur, ()):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if nxt == target_id:
                    return new_path
                visited.add(nxt)
                queue.append(new_path)
        return None

    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Alias for :meth:`find_path` — BFS is already shortest-path for
        unweighted graphs."""
        return self.find_path(source_id, target_id)

    def subgraph(self, node_ids: Sequence[str]) -> "SceneGraph":
        """Return a new :class:`SceneGraph` induced on ``node_ids``.

        Only edges with both endpoints in ``node_ids`` are kept.

        Args:
            node_ids: Node ids to include.

        Raises:
            KeyError: If any node_id is unknown.
        """
        node_set = set(node_ids)
        for nid in node_set:
            if nid not in self.node_registry:
                raise KeyError(f"subgraph: unknown node_id {nid!r}")
        nodes = {nid: self.node_registry[nid] for nid in node_set}
        edges = {
            eid: e for eid, e in self.edge_registry.items()
            if e.source_id in node_set and e.target_id in node_set
        }
        return SceneGraph(
            scene_id=_new_id("subscene"),
            nodes=nodes,
            edges=edges,
            global_attributes=dict(self.global_attributes),
            timestamp=self._timestamp,
        )

    # ── validation ────────────────────────────────────────────────────────

    def _validate_embedding(self, embedding: torch.Tensor, expected_dim: int, what: str) -> None:
        if embedding.dim() != 1 or embedding.shape[0] != expected_dim:
            raise ValueError(
                f"{what} must be a 1-D tensor of dim {expected_dim}, got "
                f"shape {tuple(embedding.shape)}"
            )
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            raise RuntimeError(f"{what} contains NaN/Inf values")

    def validate(self) -> List[str]:
        """Run structural validation checks; never raises, returns issues.

        Checks: duplicate node ids (impossible by registry construction, but
        re-verified), orphan edges (referencing missing nodes), self-loops,
        invalid references, NaN/Inf embeddings, and dimension mismatches.

        Returns:
            List of human-readable issue descriptions; empty if the graph is
            structurally valid.
        """
        issues: List[str] = []
        seen_node_ids: Set[str] = set()
        for nid, node in self.node_registry.items():
            if nid in seen_node_ids:
                issues.append(f"duplicate node id: {nid}")
            seen_node_ids.add(nid)
            if node.embedding is not None:
                try:
                    self._validate_embedding(node.embedding, self.config.entity_dim, f"node {nid} embedding")
                except (ValueError, RuntimeError) as exc:
                    issues.append(str(exc))

        for eid, edge in self.edge_registry.items():
            if edge.source_id not in self.node_registry:
                issues.append(f"edge {eid}: orphan source_id {edge.source_id!r}")
            if edge.target_id not in self.node_registry:
                issues.append(f"edge {eid}: orphan target_id {edge.target_id!r}")
            if edge.source_id == edge.target_id:
                issues.append(f"edge {eid}: self-loop on {edge.source_id!r}")
            if edge.relation_embedding is not None:
                try:
                    self._validate_embedding(edge.relation_embedding, self.config.relation_dim, f"edge {eid} embedding")
                except (ValueError, RuntimeError) as exc:
                    issues.append(str(exc))

        for hid, he in self.hyperedge_registry.items():
            for nid in he.node_ids:
                if nid not in self.node_registry:
                    issues.append(f"hyperedge {hid}: orphan node_id {nid!r}")

        return issues

    def validate_or_raise(self) -> None:
        """Run :meth:`validate` and raise a descriptive ``ValueError`` if any
        issues are found."""
        issues = self.validate()
        if issues:
            raise ValueError(
                "GraphBuilder.validate_or_raise: found "
                f"{len(issues)} issue(s):\n  - " + "\n  - ".join(issues)
            )

    # ── adjacency / edge_index ────────────────────────────────────────────

    def build_adjacency_matrix(self) -> Tuple[torch.Tensor, List[str]]:
        """Build a dense ``(N, N)`` adjacency matrix.

        Returns:
            Tuple of (adjacency matrix, ordered list of node ids matching
            the matrix's row/column order).
        """
        node_ids = sorted(self.node_registry.keys())
        index = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        adj = torch.zeros(n, n, dtype=torch.float32)
        for e in self.edge_registry.values():
            i, j = index[e.source_id], index[e.target_id]
            adj[i, j] = max(adj[i, j].item(), e.confidence)
            if not e.directed:
                adj[j, i] = max(adj[j, i].item(), e.confidence)
        return adj, node_ids

    def build_edge_index(self, node_order: Optional[List[str]] = None) -> torch.Tensor:
        """Build a PyTorch-Geometric-style ``(2, E)`` long edge_index tensor.

        Args:
            node_order: Explicit node ordering; defaults to sorted node ids
                       (must match whatever ordering node features are
                       stacked in downstream).

        Returns:
            ``(2, E)`` long tensor of [source_indices; target_indices].
        """
        node_ids = node_order or sorted(self.node_registry.keys())
        index = {nid: i for i, nid in enumerate(node_ids)}
        sources, targets = [], []
        for e in self.edge_registry.values():
            sources.append(index[e.source_id])
            targets.append(index[e.target_id])
            if not e.directed:
                sources.append(index[e.target_id])
                targets.append(index[e.source_id])
        if not sources:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor([sources, targets], dtype=torch.long)

    # ── statistics ────────────────────────────────────────────────────────

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute graph-level statistics.

        Returns:
            Dict with num_nodes, num_edges, average_degree, density,
            connected_components, isolated_nodes.
        """
        n = len(self.node_registry)
        m = len(self.edge_registry)

        degree: Dict[str, int] = defaultdict(int)
        for e in self.edge_registry.values():
            degree[e.source_id] += 1
            degree[e.target_id] += 1

        avg_degree = (sum(degree.values()) / n) if n > 0 else 0.0
        density = (m / (n * (n - 1))) if n > 1 else 0.0

        visited: Set[str] = set()
        components = 0
        for nid in self.node_registry:
            if nid not in visited:
                comp = self.connected_nodes(nid)
                visited |= comp
                components += 1

        isolated = [nid for nid in self.node_registry if degree.get(nid, 0) == 0]

        return {
            "num_nodes":            n,
            "num_edges":            m,
            "num_hyperedges":       len(self.hyperedge_registry),
            "average_degree":       avg_degree,
            "density":              density,
            "connected_components": components,
            "isolated_nodes":       isolated,
        }

    # ── scene embedding ───────────────────────────────────────────────────

    def compute_scene_embedding(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate all node + edge embeddings into a single scene embedding.

        Returns:
            Tuple of (``scene_embedding`` shape ``(graph_dim,)``,
            attention-weight dict for explainability).
        """
        node_embs = [n.embedding for n in self.node_registry.values() if n.embedding is not None]
        edge_embs = [e.relation_embedding for e in self.edge_registry.values() if e.relation_embedding is not None]

        node_tensor = torch.stack(node_embs, dim=0) if node_embs else None
        edge_tensor = torch.stack(edge_embs, dim=0) if edge_embs else None

        with torch.no_grad():
            scene_embedding, attn = self.aggregator(
                node_tensor, edge_tensor, method=self.config.aggregate_method
            )
        if self.config.normalize_embeddings and scene_embedding.numel() > 0:
            scene_embedding = F.normalize(scene_embedding.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return scene_embedding, attn

    # ── build ─────────────────────────────────────────────────────────────

    def build(self, timestamp: Optional[float] = None) -> SceneGraph:
        """Assemble the current registries into an immutable :class:`SceneGraph`
        snapshot.

        Args:
            timestamp: Simulation time for this scene; defaults to the
                       builder's internal clock.

        Returns:
            A fully populated :class:`SceneGraph`.

        Raises:
            ValueError: If :meth:`validate` finds structural issues.
        """
        self.validate_or_raise()
        if timestamp is not None:
            self._timestamp = timestamp

        adjacency: Optional[torch.Tensor] = None
        node_order: Optional[List[str]] = None
        if self.config.build_adjacency_matrix:
            adjacency, node_order = self.build_adjacency_matrix()

        edge_index: Optional[torch.Tensor] = None
        if self.config.build_edge_index:
            edge_index = self.build_edge_index(node_order)

        scene_embedding: Optional[torch.Tensor] = None
        attn: Dict[str, torch.Tensor] = {}
        if self.config.compute_scene_embedding:
            scene_embedding, attn = self.compute_scene_embedding()

        statistics: Dict[str, Any] = {}
        if self.config.compute_graph_statistics:
            statistics = self.compute_statistics()

        scene = SceneGraph(
            scene_id=self._scene_id,
            version=self._version,
            parent_scene_id=self._parent_scene_id,
            nodes=dict(self.node_registry),
            edges=dict(self.edge_registry),
            hyperedges=dict(self.hyperedge_registry),
            adjacency=adjacency,
            edge_index=edge_index,
            scene_embedding=scene_embedding,
            scene_token=scene_embedding.clone() if scene_embedding is not None else None,
            global_attributes=dict(self.global_attributes),
            timestamp=self._timestamp,
            statistics=statistics,
            attention_weights=attn or None,
        )
        self._last_scene_graph = scene
        self._record_in_memory(scene)
        return scene

    # ── embedding retrieval ───────────────────────────────────────────────

    def get_scene_embedding(self) -> Optional[torch.Tensor]:
        """Return the last built scene's ``scene_embedding``, or None."""
        return self._last_scene_graph.scene_embedding if self._last_scene_graph else None

    def get_node_embedding(self, node_id: str) -> Optional[torch.Tensor]:
        return self.get_node(node_id).embedding

    def get_edge_embedding(self, edge_id: str) -> Optional[torch.Tensor]:
        return self.get_edge(edge_id).relation_embedding

    # ── similarity ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a = F.normalize(a.reshape(1, -1), p=2, dim=-1)
        b = F.normalize(b.reshape(1, -1), p=2, dim=-1)
        return float((a * b).sum().item())

    def graph_similarity(self, other: SceneGraph) -> float:
        """Cosine similarity between this builder's last scene embedding and
        ``other``'s.

        Raises:
            ValueError: If either scene embedding is missing.
        """
        mine = self.get_scene_embedding()
        if mine is None or other.scene_embedding is None:
            raise ValueError("graph_similarity: both graphs need a scene_embedding")
        return self._cosine(mine, other.scene_embedding)

    def node_similarity(self, node_id_a: str, node_id_b: str) -> float:
        a, b = self.get_node(node_id_a).embedding, self.get_node(node_id_b).embedding
        if a is None or b is None:
            raise ValueError("node_similarity: both nodes need embeddings")
        return self._cosine(a, b)

    def edge_similarity(self, edge_id_a: str, edge_id_b: str) -> float:
        a, b = self.get_edge(edge_id_a).relation_embedding, self.get_edge(edge_id_b).relation_embedding
        if a is None or b is None:
            raise ValueError("edge_similarity: both edges need relation_embeddings")
        return self._cosine(a, b)

    # ── memory bank ───────────────────────────────────────────────────────

    def _record_in_memory(self, scene: SceneGraph) -> None:
        if scene.scene_embedding is None:
            return
        if len(self.graph_memory) >= self.config.memory_bank_capacity:
            oldest_id = next(iter(self.graph_memory))
            del self.graph_memory[oldest_id]
        self.graph_memory[scene.scene_id] = {
            "embedding":  scene.scene_embedding.clone(),
            "scene_id":   scene.scene_id,
            "timestamp":  scene.timestamp,
        }

    def nearest_graphs(self, query: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Find the ``k`` most similar stored scenes to ``query`` by cosine
        similarity over ``graph_memory``.

        Args:
            query: ``(graph_dim,)`` query embedding.
            k:     Number of neighbours to return.

        Returns:
            List of ``(scene_id, similarity)`` tuples, descending similarity.
        """
        if not self.graph_memory:
            return []
        sims = [
            (sid, self._cosine(query, entry["embedding"]))
            for sid, entry in self.graph_memory.items()
        ]
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[: max(1, k)]

    # ── exports ───────────────────────────────────────────────────────────

    def to_networkx(self) -> Any:
        """Export the current graph to a ``networkx.MultiDiGraph``.

        Raises:
            ImportError: If networkx is not installed.
        """
        if not _NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for to_networkx(); pip install networkx")
        g = nx.MultiDiGraph()
        for nid, node in self.node_registry.items():
            g.add_node(nid, **node.to_dict(include_embeddings=False))
        for e in self.edge_registry.values():
            g.add_edge(e.source_id, e.target_id, key=e.edge_id,
                       **e.to_dict(include_embeddings=False))
        return g

    def to_dict(self, include_embeddings: bool = True) -> dict:
        """Serialise the current (unbuilt) registries to a plain dict."""
        return {
            "scene_id":          self._scene_id,
            "version":           self._version,
            "parent_scene_id":   self._parent_scene_id,
            "nodes":             {k: v.to_dict(include_embeddings) for k, v in self.node_registry.items()},
            "edges":             {k: v.to_dict(include_embeddings) for k, v in self.edge_registry.items()},
            "hyperedges":        {k: v.to_dict(include_embeddings) for k, v in self.hyperedge_registry.items()},
            "global_attributes": self.global_attributes,
            "timestamp":         self._timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required for to_yaml(); pip install pyyaml")
        return _yaml.dump(self.to_dict(), sort_keys=False)

    def to_worldspec(self, description: str = "") -> "WorldSpec":
        """Convert the current graph into a physics-ready :class:`WorldSpec`.

        Nodes become :class:`Entity` objects; physics-layer edges become
        :class:`Interaction` objects. Spatial/semantic/temporal/event edges
        are not lossy-converted to ``Interaction`` (they don't map cleanly
        onto a single physics interaction) and are instead preserved under
        ``WorldSpec.metadata["scene_graph_edges"]`` for round-tripping.

        Args:
            description: Natural-language description carried into
                        ``WorldSpec.description``.

        Raises:
            ImportError: If ``world_spec.py`` is not importable.
        """
        if not _WORLDSPEC_AVAILABLE:
            raise ImportError(
                "world_spec.py must be importable to use to_worldspec(); "
                "ensure it is on the Python path."
            )

        entities: List[Entity] = []
        for node in self.node_registry.values():
            pos = node.position if node.position is not None else torch.zeros(3)
            vel = node.velocity if node.velocity is not None else torch.zeros(3)
            ori = node.orientation if node.orientation is not None else torch.zeros(3)
            state = PhysicsState(
                position=Vec3(*[float(x) for x in pos.tolist()]),
                velocity=Vec3(*[float(x) for x in vel.tolist()]),
                orientation=Vec3(*[float(x) for x in ori.tolist()]),
            )
            entities.append(Entity(
                id=node.node_id,
                label=node.name,
                entity_type=node.entity_type,
                is_static=(node.mobility == "static"),
                material=node.material,
                state=state,
                tags=[node.shape, node.size_class, node.mass_class],
            ))

        interactions: List[Interaction] = []
        non_physics_edges = []
        for e in self.edge_registry.values():
            if e.layer == "physics":
                interactions.append(Interaction(
                    type=e.relation_type,
                    entity_a=e.source_id,
                    entity_b=e.target_id,
                    parameters={**e.physics_properties, "confidence": e.confidence},
                ))
            else:
                non_physics_edges.append(e.to_dict(include_embeddings=False))

        return WorldSpec(
            scene_id=self._scene_id,
            description=description,
            entities=entities,
            interactions=interactions,
            metadata={
                "scene_graph_edges": non_physics_edges,
                "scene_graph_version": self._version,
                "scene_graph_statistics": self.compute_statistics(),
            },
        )

    # ── exports — physics back-ends (interfaces only) ────────────────────

    def to_bullet(self) -> Any:
        """Interface placeholder for a future PyBullet exporter.

        Raises:
            NotImplementedError: Always — implement in a later file that
                owns the PyBullet dependency.
        """
        raise NotImplementedError(
            "to_bullet() is an interface placeholder; implement in a "
            "dedicated Bullet-export module."
        )

    def to_mujoco(self) -> Any:
        """Interface placeholder for a future MuJoCo MJCF exporter."""
        raise NotImplementedError(
            "to_mujoco() is an interface placeholder; implement in a "
            "dedicated MuJoCo-export module."
        )

    def to_isaac(self) -> Any:
        """Interface placeholder for a future Isaac Sim USD exporter."""
        raise NotImplementedError(
            "to_isaac() is an interface placeholder; implement in a "
            "dedicated Isaac-export module."
        )

    def to_gazebo(self) -> Any:
        """Interface placeholder for a future Gazebo SDF exporter."""
        raise NotImplementedError(
            "to_gazebo() is an interface placeholder; implement in a "
            "dedicated Gazebo-export module."
        )

    # ── imports ───────────────────────────────────────────────────────────

    def from_dict(self, d: dict) -> "GraphBuilder":
        """Replace current registries with the contents of ``d`` (in-place).

        Returns:
            ``self``, for chaining.
        """
        self.node_registry = {k: GraphNode.from_dict(v) for k, v in d.get("nodes", {}).items()}
        self.edge_registry = {k: GraphEdge.from_dict(v) for k, v in d.get("edges", {}).items()}
        self.hyperedge_registry = {k: HyperEdge.from_dict(v) for k, v in d.get("hyperedges", {}).items()}
        self._edge_key_index = {
            self._edge_key(e.source_id, e.target_id, e.relation_type): eid
            for eid, e in self.edge_registry.items()
        }
        self._scene_id = d.get("scene_id", _new_id("scene"))
        self._version = d.get("version", 0)
        self._parent_scene_id = d.get("parent_scene_id")
        self.global_attributes = d.get("global_attributes", {})
        self._timestamp = d.get("timestamp", 0.0)
        return self

    def from_json(self, json_str: str) -> "GraphBuilder":
        return self.from_dict(json.loads(json_str))

    def from_worldspec(self, world_spec: "WorldSpec") -> "GraphBuilder":
        """Populate the builder's registries from a :class:`WorldSpec`.

        Inverse of :meth:`to_worldspec`. Entities become nodes (without
        learned embeddings, since WorldSpec doesn't carry them); Interactions
        become physics-layer edges.

        Returns:
            ``self``, for chaining.
        """
        if not _WORLDSPEC_AVAILABLE:
            raise ImportError(
                "world_spec.py must be importable to use from_worldspec()."
            )
        self.node_registry.clear()
        self.edge_registry.clear()
        self._edge_key_index.clear()

        for entity in world_spec.entities:
            pos = torch.tensor([entity.state.position.x, entity.state.position.y, entity.state.position.z])
            vel = torch.tensor([entity.state.velocity.x, entity.state.velocity.y, entity.state.velocity.z])
            ori = torch.tensor([entity.state.orientation.x, entity.state.orientation.y, entity.state.orientation.z])
            self.node_registry[entity.id] = GraphNode(
                node_id=entity.id,
                name=entity.label,
                entity_type=entity.entity_type,
                material=entity.material,
                mobility="static" if entity.is_static else "dynamic",
                position=pos, velocity=vel, orientation=ori,
            )

        for interaction in world_spec.interactions:
            if interaction.entity_b == "environment":
                continue
            key = self._edge_key(interaction.entity_a, interaction.entity_b, interaction.type)
            eid = _new_id("edge")
            edge = GraphEdge(
                source_id=interaction.entity_a,
                target_id=interaction.entity_b,
                relation_type=interaction.type,
                edge_id=eid,
                relation_category="physics",
                layer="physics",
                physics_properties=dict(interaction.parameters),
                confidence=float(interaction.parameters.get("confidence", 1.0)),
            )
            self.edge_registry[eid] = edge
            self._edge_key_index[key] = eid

        # restore any non-physics edges preserved in metadata
        for ed in world_spec.metadata.get("scene_graph_edges", []):
            edge = GraphEdge.from_dict(ed)
            self.edge_registry[edge.edge_id] = edge
            self._edge_key_index[
                self._edge_key(edge.source_id, edge.target_id, edge.relation_type)
            ] = edge.edge_id

        self._scene_id = world_spec.scene_id
        return self

    # ── serialization (save/load) ────────────────────────────────────────

    def save_graph(self, path: Union[str, Path]) -> None:
        """Persist the current registries to ``path`` via ``torch.save``.

        Args:
            path: Destination ``.pt`` file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config":             asdict(self.config),
            "scene_id":           self._scene_id,
            "version":            self._version,
            "parent_scene_id":    self._parent_scene_id,
            "timestamp":          self._timestamp,
            "global_attributes":  self.global_attributes,
            "nodes":              {k: v.to_dict() for k, v in self.node_registry.items()},
            "edges":              {k: v.to_dict() for k, v in self.edge_registry.items()},
            "hyperedges":         {k: v.to_dict() for k, v in self.hyperedge_registry.items()},
            "aggregator_state":   self.aggregator.state_dict(),
        }
        torch.save(state, path)
        print(f"[GraphBuilder] saved → {path}")

    @classmethod
    def load_graph(cls, path: Union[str, Path], map_location: Optional[str] = None) -> "GraphBuilder":
        """Load a :class:`GraphBuilder` previously saved with :meth:`save_graph`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GraphBuilder checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location or "cpu")

        config = GraphBuilderConfig(**state["config"])
        builder = cls(config)
        builder._scene_id = state["scene_id"]
        builder._version = state["version"]
        builder._parent_scene_id = state["parent_scene_id"]
        builder._timestamp = state["timestamp"]
        builder.global_attributes = state["global_attributes"]
        builder.node_registry = {k: GraphNode.from_dict(v) for k, v in state["nodes"].items()}
        builder.edge_registry = {k: GraphEdge.from_dict(v) for k, v in state["edges"].items()}
        builder.hyperedge_registry = {k: HyperEdge.from_dict(v) for k, v in state.get("hyperedges", {}).items()}
        builder._edge_key_index = {
            builder._edge_key(e.source_id, e.target_id, e.relation_type): eid
            for eid, e in builder.edge_registry.items()
        }
        builder.aggregator.load_state_dict(state["aggregator_state"])
        print(f"[GraphBuilder] loaded ← {path}")
        return builder

    # ── debug ─────────────────────────────────────────────────────────────

    def debug_build(self) -> SceneGraph:
        """Build a small synthetic 3-node / 3-edge graph and print diagnostics.

        Creates::

            car, truck, road
            car → road    (supported_by)
            truck → road  (supported_by)
            car → truck   (colliding_with)

        Returns:
            The built :class:`SceneGraph`.
        """
        self.new_scene()
        d = self.config.entity_dim
        r = self.config.relation_dim

        car   = self.add_node("car",   entity_type="vehicle",
                               embedding=F.normalize(torch.randn(d), p=2, dim=-1),
                               material="steel", mobility="dynamic")
        truck = self.add_node("truck", entity_type="vehicle",
                               embedding=F.normalize(torch.randn(d), p=2, dim=-1),
                               material="steel", mobility="dynamic")
        road  = self.add_node("road",  entity_type="structure",
                               embedding=F.normalize(torch.randn(d), p=2, dim=-1),
                               material="concrete", mobility="static")

        self.add_edge(car, road, relation_type="supported_by",
                      relation_category="support", layer="physics",
                      relation_embedding=F.normalize(torch.randn(r), p=2, dim=-1),
                      confidence=0.95)
        self.add_edge(truck, road, relation_type="supported_by",
                      relation_category="support", layer="physics",
                      relation_embedding=F.normalize(torch.randn(r), p=2, dim=-1),
                      confidence=0.95)
        self.add_edge(car, truck, relation_type="colliding_with",
                      relation_category="contact", layer="physics",
                      relation_embedding=F.normalize(torch.randn(r), p=2, dim=-1),
                      confidence=0.88)

        scene = self.build()

        print(f"\n{'─'*52}")
        print("  debug_build()  diagnostics")
        print(f"{'─'*52}")
        print(f"  Nodes:                 {scene.num_nodes()}")
        print(f"  Edges:                 {scene.num_edges()}")
        adj_shape = tuple(scene.adjacency.shape) if scene.adjacency is not None else None
        print(f"  Adjacency matrix shape: {adj_shape}")
        emb_shape = tuple(scene.scene_embedding.shape) if scene.scene_embedding is not None else None
        print(f"  Scene embedding shape:  {emb_shape}")
        print(f"  Average degree:        {scene.statistics.get('average_degree'):.3f}")
        print(f"  Density:               {scene.statistics.get('density'):.3f}")
        print(f"{'─'*52}\n")

        return scene

    # ── info ──────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        """Count total/trainable parameters of the learned aggregator."""
        total = sum(p.numel() for p in self.aggregator.parameters())
        trainable = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def print_summary(self) -> None:
        cfg = self.config
        params = self.count_parameters()
        print(f"\n{'═'*64}")
        print("  PhysWorldLM — GraphBuilder")
        print(f"{'═'*64}")
        print(f"  Graph dim           : {cfg.graph_dim}")
        print(f"  Aggregation method  : {cfg.aggregate_method}")
        print(f"  Max nodes           : {cfg.max_nodes}")
        print(f"  Max edges           : {cfg.max_edges}")
        print(f"  Attention heads     : {cfg.num_attention_heads}")
        print(f"  Normalize embeddings: {cfg.normalize_embeddings}")
        print(f"  Aggregator params   : {params['total']:,}")
        print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: build GraphBuilder, run debug_build(), verify the full
    export/import/serialization round-trip, assert shapes."""

    print("=" * 64)
    print("  PhysWorldLM — GraphBuilder")
    print("=" * 64)

    config = GraphBuilderConfig(
        entity_dim=512,
        relation_dim=512,
        graph_dim=512,
        max_nodes=256,
        max_edges=4096,
        normalize_embeddings=True,
        aggregate_method="attention",
    )
    builder = GraphBuilder(config)
    builder.print_summary()

    scene = builder.debug_build()

    # ── verify export round-trips ────────────────────────────────────────
    d = builder.to_dict()
    assert set(d["nodes"].keys()) == set(builder.node_registry.keys())

    json_str = builder.to_json()
    assert isinstance(json_str, str) and len(json_str) > 0

    if _WORLDSPEC_AVAILABLE:
        ws = builder.to_worldspec(description="debug_build smoke test")
        assert len(ws.entities) == scene.num_nodes()
        rebuilt = GraphBuilder(config).from_worldspec(ws)
        assert len(rebuilt.node_registry) == scene.num_nodes()
        print("  [main] to_worldspec() / from_worldspec()  ... OK")
    else:
        print("  [main] world_spec.py not importable; skipping WorldSpec round-trip")

    # ── verify save/load round-trip ──────────────────────────────────────
    save_path = Path("/tmp/physworldlm_graph_builder_debug.pt")
    builder.save_graph(save_path)
    loaded = GraphBuilder.load_graph(save_path)
    assert len(loaded.node_registry) == len(builder.node_registry)
    assert len(loaded.edge_registry) == len(builder.edge_registry)
    print("  [main] save_graph() / load_graph()          ... OK")

    # ── query API smoke test ─────────────────────────────────────────────
    node_ids = list(builder.node_registry.keys())
    path = builder.find_path(node_ids[0], node_ids[-1])
    assert path is not None
    print(f"  [main] find_path()                          ... OK  ({len(path)} hops)")

    # ── similarity / memory bank smoke test ──────────────────────────────
    nearest = builder.nearest_graphs(scene.scene_embedding, k=1)
    assert nearest and nearest[0][0] == scene.scene_id
    print("  [main] nearest_graphs()                      ... OK")

    print("\n[main] all assertions passed. done.")


if __name__ == "__main__":
    main()
