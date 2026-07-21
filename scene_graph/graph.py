"""The ``SceneGraph`` IR container.

``SceneGraph`` owns two coupled structures over the same node set:

1. A rooted transform hierarchy (``parent_id`` / ``children_ids`` on each
   :class:`~physworldlm.scene_graph.nodes.SceneNode`), used for world-space
   transform resolution.
2. A flat set of non-hierarchical :class:`~physworldlm.scene_graph.edges.SceneEdge`
   relationships (contacts, joints, spatial relations), used for physical
   interaction resolution.

This is the backend-independent IR that ``WorldSpecBuilder`` lowers a
``WorldSpec`` into, and that the USD/PhysX exporters compile out of --
analogous to an LLVM IR module sitting between a source AST and a target
backend.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from physworldlm.scene_graph.edges import EdgeKind, SceneEdge
from physworldlm.scene_graph.exceptions import (
    CyclicGraphError,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
)
from physworldlm.scene_graph.nodes import NodeKind, SceneNode
from physworldlm.scene_graph.transform import Transform

_ROOT_NODE_NAME = "__scene_root__"


@dataclass(slots=True)
class SceneGraph:
    """A rooted, backend-independent scene graph intermediate representation.

    Parameters
    ----------
    scene_id:
        Identifier of the source scene, propagated from ``WorldSpec.scene_id``.
    schema_version:
        IR schema version, incremented on any breaking change to node/edge
        attribute contracts so downstream compilers can guard against skew.
    """

    scene_id: str
    schema_version: str = "1.0"
    root_id: str = field(default="")
    _nodes: dict[str, SceneNode] = field(default_factory=dict)
    _edges: dict[str, SceneEdge] = field(default_factory=dict)
    _edges_by_node: dict[str, set[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.root_id:
            root = SceneNode.create(NodeKind.ROOT, _ROOT_NODE_NAME)
            self._nodes[root.id] = root
            self.root_id = root.id

    # ------------------------------------------------------------------ #
    # Node management
    # ------------------------------------------------------------------ #

    def add_node(self, node: SceneNode, *, parent_id: str | None = None) -> str:
        """Insert ``node`` into the graph, attached under ``parent_id``.

        Parameters
        ----------
        node:
            A freshly constructed, unattached node (``node.parent_id`` and
            ``node.children_ids`` are ignored and overwritten).
        parent_id:
            The node to attach under. Defaults to the implicit scene root.

        Returns
        -------
        str
            The id of the inserted node (``node.id``).

        Raises
        ------
        DuplicateNodeError
            If ``node.id`` already exists in the graph.
        NodeNotFoundError
            If ``parent_id`` does not exist in the graph.
        """
        if node.id in self._nodes:
            raise DuplicateNodeError(node.id)

        effective_parent_id = parent_id if parent_id is not None else self.root_id
        if effective_parent_id not in self._nodes:
            raise NodeNotFoundError(effective_parent_id)

        node.parent_id = effective_parent_id
        node.children_ids = []
        self._nodes[node.id] = node
        self._nodes[effective_parent_id].children_ids.append(node.id)
        self._edges_by_node.setdefault(node.id, set())
        return node.id

    def remove_node(self, node_id: str, *, recursive: bool = True) -> None:
        """Remove ``node_id`` from the graph.

        Parameters
        ----------
        node_id:
            The node to remove. Must not be the scene root.
        recursive:
            If ``True`` (default), all descendants are removed too. If
            ``False``, descendants are re-parented onto the removed node's
            former parent.

        Raises
        ------
        NodeNotFoundError
            If ``node_id`` does not exist, or is the scene root.
        """
        if node_id == self.root_id:
            raise ValueError("cannot remove the scene root node")
        node = self._require_node(node_id)

        if recursive:
            for child_id in list(node.children_ids):
                self.remove_node(child_id, recursive=True)
        else:
            for child_id in list(node.children_ids):
                self.reparent(child_id, node.parent_id)

        parent = self._nodes.get(node.parent_id) if node.parent_id else None
        if parent is not None and node_id in parent.children_ids:
            parent.children_ids.remove(node_id)

        for edge_id in list(self._edges_by_node.get(node_id, set())):
            self.remove_edge(edge_id)

        del self._nodes[node_id]
        self._edges_by_node.pop(node_id, None)

    def get_node(self, node_id: str) -> SceneNode:
        """Return the node with id ``node_id``.

        Raises
        ------
        NodeNotFoundError
            If no such node exists.
        """
        return self._require_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """Return ``True`` if a node with id ``node_id`` exists in the graph."""
        return node_id in self._nodes

    def reparent(self, node_id: str, new_parent_id: str) -> None:
        """Move ``node_id`` (and its subtree) to become a child of ``new_parent_id``.

        Raises
        ------
        NodeNotFoundError
            If either id does not exist.
        CyclicGraphError
            If ``new_parent_id`` is ``node_id`` itself or a descendant of
            ``node_id`` (which would introduce a cycle).
        """
        node = self._require_node(node_id)
        self._require_node(new_parent_id)

        if new_parent_id == node_id or self._is_descendant(new_parent_id, node_id):
            raise CyclicGraphError(node_id)

        old_parent = self._nodes.get(node.parent_id) if node.parent_id else None
        if old_parent is not None and node_id in old_parent.children_ids:
            old_parent.children_ids.remove(node_id)

        node.parent_id = new_parent_id
        self._nodes[new_parent_id].children_ids.append(node_id)

    def nodes(self) -> Iterator[SceneNode]:
        """Iterate over every node in the graph, including the root."""
        return iter(self._nodes.values())

    def nodes_by_kind(self, kind: NodeKind) -> list[SceneNode]:
        """Return all nodes of the given ``kind``."""
        return [n for n in self._nodes.values() if n.kind is kind]

    def nodes_by_tag(self, tag: str) -> list[SceneNode]:
        """Return all nodes carrying ``tag`` in their ``tags`` list."""
        return [n for n in self._nodes.values() if tag in n.tags]

    def children_of(self, node_id: str) -> list[SceneNode]:
        """Return the direct children of ``node_id``."""
        node = self._require_node(node_id)
        return [self._nodes[cid] for cid in node.children_ids]

    def node_count(self) -> int:
        """Return the total number of nodes, including the root."""
        return len(self._nodes)

    # ------------------------------------------------------------------ #
    # Edge management
    # ------------------------------------------------------------------ #

    def add_edge(self, edge: SceneEdge) -> str:
        """Insert ``edge`` into the graph.

        Raises
        ------
        DuplicateEdgeError
            If ``edge.id`` already exists in the graph.
        NodeNotFoundError
            If either endpoint does not exist in the graph.
        """
        if edge.id in self._edges:
            raise DuplicateEdgeError(edge.id)
        self._require_node(edge.source_id)
        self._require_node(edge.target_id)

        self._edges[edge.id] = edge
        self._edges_by_node.setdefault(edge.source_id, set()).add(edge.id)
        self._edges_by_node.setdefault(edge.target_id, set()).add(edge.id)
        return edge.id

    def remove_edge(self, edge_id: str) -> None:
        """Remove the edge with id ``edge_id``.

        Raises
        ------
        EdgeNotFoundError
            If no such edge exists.
        """
        edge = self._require_edge(edge_id)
        self._edges_by_node.get(edge.source_id, set()).discard(edge_id)
        self._edges_by_node.get(edge.target_id, set()).discard(edge_id)
        del self._edges[edge_id]

    def get_edge(self, edge_id: str) -> SceneEdge:
        """Return the edge with id ``edge_id``.

        Raises
        ------
        EdgeNotFoundError
            If no such edge exists.
        """
        return self._require_edge(edge_id)

    def edges(self) -> Iterator[SceneEdge]:
        """Iterate over every edge in the graph."""
        return iter(self._edges.values())

    def edges_by_kind(self, kind: EdgeKind) -> list[SceneEdge]:
        """Return all edges of the given ``kind``."""
        return [e for e in self._edges.values() if e.kind is kind]

    def edges_of(self, node_id: str) -> list[SceneEdge]:
        """Return every edge incident to ``node_id``, in either direction."""
        self._require_node(node_id)
        return [self._edges[eid] for eid in self._edges_by_node.get(node_id, set())]

    def neighbors(self, node_id: str) -> list[str]:
        """Return the ids of all nodes connected to ``node_id`` by any edge."""
        return [e.other_endpoint(node_id) for e in self.edges_of(node_id)]

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        return len(self._edges)

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def dfs(self, start_id: str | None = None) -> Iterator[SceneNode]:
        """Depth-first traversal of the transform hierarchy from ``start_id``.

        Defaults to starting at the scene root.
        """
        start = start_id if start_id is not None else self.root_id
        self._require_node(start)
        stack = [start]
        visited: set[str] = set()
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            node = self._nodes[current_id]
            yield node
            stack.extend(reversed(node.children_ids))

    def bfs(self, start_id: str | None = None) -> Iterator[SceneNode]:
        """Breadth-first traversal of the transform hierarchy from ``start_id``.

        Defaults to starting at the scene root.
        """
        start = start_id if start_id is not None else self.root_id
        self._require_node(start)
        queue: deque[str] = deque([start])
        visited: set[str] = set()
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            node = self._nodes[current_id]
            yield node
            queue.extend(node.children_ids)

    def topological_order(self) -> list[SceneNode]:
        """Return all nodes ordered so that every parent precedes its children.

        This ordering is a precondition for compilation stages (e.g. USD
        Xform authoring, PhysX body creation) that require a body to exist
        before any of its attachments are authored.
        """
        return list(self.bfs(self.root_id))

    def world_transform(self, node_id: str) -> Transform:
        """Compute the accumulated world-space transform of ``node_id``.

        Walks from the scene root down to ``node_id``, composing local
        transforms along the way. Complexity is O(depth) per call; callers
        performing many lookups should cache results keyed by node id.
        """
        chain: list[SceneNode] = []
        current_id: str | None = node_id
        while current_id is not None:
            node = self._require_node(current_id)
            chain.append(node)
            current_id = node.parent_id
        chain.reverse()

        accumulated = Transform.identity()
        for node in chain:
            accumulated = accumulated.compose(node.local_transform)
        return accumulated

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire graph to a plain JSON-compatible ``dict``."""
        return {
            "scene_id": self.scene_id,
            "schema_version": self.schema_version,
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SceneGraph":
        """Deserialize a graph from the structure produced by :meth:`to_dict`."""
        graph = SceneGraph(
            scene_id=payload["scene_id"],
            schema_version=payload.get("schema_version", "1.0"),
            root_id=payload["root_id"],
        )
        graph._nodes.clear()
        graph._edges_by_node.clear()

        for node_payload in payload["nodes"]:
            node = SceneNode.from_dict(node_payload)
            graph._nodes[node.id] = node
            graph._edges_by_node.setdefault(node.id, set())

        for edge_payload in payload.get("edges", []):
            edge = SceneEdge.from_dict(edge_payload)
            graph._edges[edge.id] = edge
            graph._edges_by_node.setdefault(edge.source_id, set()).add(edge.id)
            graph._edges_by_node.setdefault(edge.target_id, set()).add(edge.id)

        graph.metadata = dict(payload.get("metadata", {}))
        return graph

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _require_node(self, node_id: str) -> SceneNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)
        return node

    def _require_edge(self, edge_id: str) -> SceneEdge:
        edge = self._edges.get(edge_id)
        if edge is None:
            raise EdgeNotFoundError(edge_id)
        return edge

    def _is_descendant(self, candidate_id: str, ancestor_id: str) -> bool:
        """Return ``True`` if ``candidate_id`` lies within ``ancestor_id``'s subtree."""
        for node in self.dfs(ancestor_id):
            if node.id == candidate_id:
                return True
        return False
