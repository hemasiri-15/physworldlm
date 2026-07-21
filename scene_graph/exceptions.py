"""Exception hierarchy for the ``physworldlm.scene_graph`` package.

All exceptions raised anywhere in this package derive from
:class:`SceneGraphError`, allowing callers to catch every scene-graph
related failure with a single ``except`` clause while still being able
to discriminate on the more specific subclasses when needed.
"""

from __future__ import annotations


class SceneGraphError(Exception):
    """Base class for every exception raised by ``physworldlm.scene_graph``."""


class NodeNotFoundError(SceneGraphError):
    """Raised when a referenced node id does not exist in the graph."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"node not found: {node_id!r}")


class EdgeNotFoundError(SceneGraphError):
    """Raised when a referenced edge id does not exist in the graph."""

    def __init__(self, edge_id: str) -> None:
        self.edge_id = edge_id
        super().__init__(f"edge not found: {edge_id!r}")


class DuplicateNodeError(SceneGraphError):
    """Raised when attempting to insert a node id that already exists."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"node already exists: {node_id!r}")


class DuplicateEdgeError(SceneGraphError):
    """Raised when attempting to insert an edge id that already exists."""

    def __init__(self, edge_id: str) -> None:
        self.edge_id = edge_id
        super().__init__(f"edge already exists: {edge_id!r}")


class CyclicGraphError(SceneGraphError):
    """Raised when an operation would introduce a cycle in the transform hierarchy."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"operation would introduce a cycle at node: {node_id!r}")


class SceneGraphValidationError(SceneGraphError):
    """Raised when strict validation fails and errors must abort the caller."""

    def __init__(self, issue_count: int) -> None:
        self.issue_count = issue_count
        super().__init__(f"scene graph validation failed with {issue_count} error(s)")


class SceneGraphSerializationError(SceneGraphError):
    """Raised when serialization or deserialization of a scene graph fails."""
