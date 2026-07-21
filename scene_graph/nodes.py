"""Node representation for the backend-independent ``SceneGraph`` IR.

A ``SceneGraph`` is a rooted forest of :class:`SceneNode` instances linked
by parent/child transform-hierarchy pointers, plus a separate set of
non-hierarchical :class:`~physworldlm.scene_graph.edges.SceneEdge`
relationships (joints, contacts, spatial relations) layered on top. Node
*kind* is intentionally a flat enum rather than a class hierarchy: this
keeps the IR trivially serializable and lets the ``WorldSpecBuilder`` and
downstream compilers (USD exporter, PhysX exporter) pattern-match on
``kind`` without needing ``isinstance`` checks across module boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from physworldlm.scene_graph.attributes import Attribute, AttributeSet, Provenance
from physworldlm.scene_graph.ids import new_node_id
from physworldlm.scene_graph.transform import Transform


class NodeKind(str, Enum):
    """The taxonomy of node kinds representable in the scene graph IR."""

    ROOT = "root"
    """The single implicit root of the transform hierarchy for a scene."""

    ENTITY = "entity"
    """A physical, simulatable thing (vehicle, agent, object, structure, ...)."""

    XFORM = "xform"
    """A pure grouping/transform node with no physical or render payload."""

    PHYSICS_BODY = "physics_body"
    """A rigid or deformable body attached to an :attr:`ENTITY` node."""

    COLLIDER = "collider"
    """A collision geometry attached to a :attr:`PHYSICS_BODY` node."""

    MATERIAL = "material"
    """A physical material definition (friction, restitution, density)."""

    JOINT = "joint"
    """A constraint node (spring, hinge, fixed, distance, ...) between two bodies."""

    ENVIRONMENT = "environment"
    """Global environment state: gravity, wind, atmosphere, ground plane."""

    CAMERA = "camera"
    """An optional viewpoint node for rendering/inspection purposes only."""


# Node kinds that must always resolve to a physically meaningful mass.
PHYSICAL_KINDS: frozenset[NodeKind] = frozenset({NodeKind.PHYSICS_BODY})

# Node kinds that participate in the transform hierarchy (have a Transform).
TRANSFORMABLE_KINDS: frozenset[NodeKind] = frozenset(
    {
        NodeKind.ROOT,
        NodeKind.ENTITY,
        NodeKind.XFORM,
        NodeKind.PHYSICS_BODY,
        NodeKind.COLLIDER,
        NodeKind.CAMERA,
    }
)


@dataclass(slots=True)
class SceneNode:
    """A single node in the scene graph IR.

    ``SceneNode`` is a plain data container; all hierarchy-mutating
    operations (attach, detach, reparent) live on
    :class:`~physworldlm.scene_graph.graph.SceneGraph` so that invariants
    such as acyclicity are enforced in exactly one place.
    """

    id: str
    kind: NodeKind
    name: str
    local_transform: Transform = field(default_factory=Transform.identity)
    attributes: AttributeSet = field(default_factory=AttributeSet)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def create(
        kind: NodeKind,
        name: str,
        *,
        local_transform: Transform | None = None,
        tags: list[str] | None = None,
    ) -> "SceneNode":
        """Construct a new, unattached ``SceneNode`` with a freshly generated id."""
        return SceneNode(
            id=new_node_id(kind.value, name),
            kind=kind,
            name=name,
            local_transform=local_transform if local_transform is not None else Transform.identity(),
            attributes=AttributeSet(),
            parent_id=None,
            children_ids=[],
            tags=list(tags) if tags else [],
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

    def is_transformable(self) -> bool:
        """Return ``True`` if this node kind participates in the transform hierarchy."""
        return self.kind in TRANSFORMABLE_KINDS

    def is_leaf(self) -> bool:
        """Return ``True`` if this node currently has no children."""
        return len(self.children_ids) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to a plain JSON-compatible ``dict``."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "local_transform": {
                "translation": self.local_transform.translation.to_tuple(),
                "rotation_wxyz": self.local_transform.rotation.to_tuple_wxyz(),
                "scale": self.local_transform.scale.to_tuple(),
            },
            "attributes": self.attributes.to_dict(),
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "tags": list(self.tags),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "SceneNode":
        """Deserialize a node from the structure produced by :meth:`to_dict`."""
        from physworldlm.scene_graph.transform import Quaternion, Vec3

        xf = payload["local_transform"]
        tx, ty, tz = xf["translation"]
        rw, rx, ry, rz = xf["rotation_wxyz"]
        sx, sy, sz = xf["scale"]
        local_transform = Transform(
            translation=Vec3(tx, ty, tz),
            rotation=Quaternion(rw, rx, ry, rz),
            scale=Vec3(sx, sy, sz),
        )
        return SceneNode(
            id=payload["id"],
            kind=NodeKind(payload["kind"]),
            name=payload["name"],
            local_transform=local_transform,
            attributes=AttributeSet.from_dict(payload.get("attributes", {})),
            parent_id=payload.get("parent_id"),
            children_ids=list(payload.get("children_ids", [])),
            tags=list(payload.get("tags", [])),
        )
