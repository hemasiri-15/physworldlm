"""Non-hierarchical edge representation for the scene graph IR.

Parent/child transform relationships live directly on
:class:`~physworldlm.scene_graph.nodes.SceneNode`. Every *other* relationship
between two nodes -- physical contact, a joint constraint, friction
coupling, or a purely descriptive spatial relation carried over from the
entity knowledge graph -- is represented as a :class:`SceneEdge` instead,
so the transform hierarchy and the physical-interaction graph can be
queried, validated, and topologically sorted independently of each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from physworldlm.scene_graph.attributes import Attribute, AttributeSet, Provenance
from physworldlm.scene_graph.ids import new_edge_id


class EdgeKind(str, Enum):
    """The taxonomy of edge kinds representable in the scene graph IR."""

    CONTACT = "contact"
    """A resting/collision contact relationship between two bodies."""

    FRICTION = "friction"
    """A friction coupling between a dynamic body and a surface."""

    COLLISION = "collision"
    """An anticipated or scripted collision event between two dynamic bodies."""

    JOINT_SPRING = "joint_spring"
    """A spring constraint between two bodies."""

    JOINT_HINGE = "joint_hinge"
    """A revolute/hinge constraint between two bodies."""

    JOINT_FIXED = "joint_fixed"
    """A rigid, zero-degree-of-freedom weld constraint between two bodies."""

    JOINT_DISTANCE = "joint_distance"
    """A distance/rope constraint between two bodies."""

    SPATIAL_RELATION = "spatial_relation"
    """A non-physical descriptive relation (e.g. ``on_top_of``, ``near``)."""


# Edge kinds that must be realized as an actual PhysX constraint at compile time.
CONSTRAINT_EDGE_KINDS: frozenset[EdgeKind] = frozenset(
    {
        EdgeKind.JOINT_SPRING,
        EdgeKind.JOINT_HINGE,
        EdgeKind.JOINT_FIXED,
        EdgeKind.JOINT_DISTANCE,
    }
)


@dataclass(slots=True)
class SceneEdge:
    """A directed, typed, attributed edge between two scene graph nodes.

    Direction (``source_id`` -> ``target_id``) does not imply physical
    causality for symmetric kinds such as :attr:`EdgeKind.CONTACT`; it is
    preserved purely so serialization is deterministic and diffable.
    """

    id: str
    kind: EdgeKind
    source_id: str
    target_id: str
    attributes: AttributeSet = field(default_factory=AttributeSet)
    bidirectional: bool = True

    @staticmethod
    def create(
        kind: EdgeKind,
        source_id: str,
        target_id: str,
        *,
        bidirectional: bool = True,
    ) -> "SceneEdge":
        """Construct a new ``SceneEdge`` with a freshly generated id."""
        return SceneEdge(
            id=new_edge_id(kind.value, source_id, target_id),
            kind=kind,
            source_id=source_id,
            target_id=target_id,
            attributes=AttributeSet(),
            bidirectional=bidirectional,
        )

    def set_attribute(
        self,
        name: str,
        value: Any,
        *,
        provenance: Provenance = Provenance.ASSUMED,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """Set (or overwrite) a named attribute with full provenance metadata."""
        self.attributes.set(
            Attribute(
                name=name,
                value=value,
                provenance=provenance,
                confidence=confidence,
                source=source,
            )
        )

    def attribute_value(self, name: str, default: Any = None) -> Any:
        """Return the raw value of the named attribute, or ``default`` if absent."""
        return self.attributes.value_of(name, default)

    def is_constraint(self) -> bool:
        """Return ``True`` if this edge must be realized as a physics constraint."""
        return self.kind in CONSTRAINT_EDGE_KINDS

    def endpoints(self) -> tuple[str, str]:
        """Return the ``(source_id, target_id)`` pair."""
        return (self.source_id, self.target_id)

    def other_endpoint(self, node_id: str) -> str:
        """Return the id at the opposite end of this edge from ``node_id``.

        Raises
        ------
        ValueError
            If ``node_id`` is not one of this edge's endpoints.
        """
        if node_id == self.source_id:
            return self.target_id
        if node_id == self.target_id:
            return self.source_id
        raise ValueError(f"node {node_id!r} is not an endpoint of edge {self.id!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this edge to a plain JSON-compatible ``dict``."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "attributes": self.attributes.to_dict(),
            "bidirectional": self.bidirectional,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SceneEdge":
        """Deserialize an edge from the structure produced by :meth:`to_dict`."""
        return SceneEdge(
            id=payload["id"],
            kind=EdgeKind(payload["kind"]),
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            attributes=AttributeSet.from_dict(payload.get("attributes", {})),
            bidirectional=bool(payload.get("bidirectional", True)),
        )
