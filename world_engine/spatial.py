"""
world_engine.spatial
───────────────────────
Spatial indexing over entity positions.

`WorldEngine`'s naive entity lookups are O(n) (fine for tens/hundreds of
entities, the common case for LLM-authored scenes). For large worlds
(procedural cities, crowds, particle-heavy fluids) proximity queries need
sub-linear structures. This module defines a small `SpatialIndex`
interface and ships one concrete implementation, `KDTreeIndex`, with the
interface designed so `OctreeIndex` / `BVHIndex` can be dropped in later
without touching `WorldEngine` call sites — `WorldEngine.spatial_index` is
swappable via the constructor.

The KD-tree here is a plain-Python, dependency-free implementation
(no scipy/numpy requirement) rebuilt lazily on the first query after any
entity insertion/removal/move — appropriate for the "rebuild every few
ticks" access pattern typical of a symbolic edit layer, as opposed to a
per-physics-substep index (that belongs in SimulationEngine/PhysX, which
already has an optimized broadphase).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from world_spec import Entity, Vec3


def _dist2(a: Vec3, b: Vec3) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2


class SpatialIndex(ABC):
    """
    Abstract interface for a 3D spatial index over `(entity_id, position)`
    pairs. Implementations must support incremental rebuild from a full
    entity list (`build`) plus nearest-neighbor and radius queries.

    Future extension notes:
        - `OctreeIndex`: better for very non-uniform density and cheap
          incremental insert/remove without a full rebuild.
        - `BVHIndex`: better when entities have significant bounding-box
          extent (not point approximations) — needed once WorldEngine does
          overlap/containment queries ("find entities inside this
          building") rather than point-proximity queries.
    """

    @abstractmethod
    def build(self, entities: list[Entity]) -> None:
        """(Re)build the index from scratch over the given entities."""

    @abstractmethod
    def nearest(self, point: Vec3, k: int = 1,
                predicate: Optional[callable] = None) -> list[tuple[str, float]]:
        """Return up to `k` (entity_id, distance) pairs nearest to `point`, ascending."""

    @abstractmethod
    def within_radius(self, point: Vec3, radius: float,
                       predicate: Optional[callable] = None) -> list[tuple[str, float]]:
        """Return every (entity_id, distance) pair within `radius` of `point`."""


@dataclass
class _KDNode:
    entity_id: str
    position: Vec3
    axis: int
    left: Optional["_KDNode"] = None
    right: Optional["_KDNode"] = None


class KDTreeIndex(SpatialIndex):
    """
    Balanced 3D k-d tree over entity positions.

    Complexity:
        build:          O(n log n)
        nearest(k):     O(log n) average, O(n) worst case (highly skewed
                         distributions) — acceptable for symbolic-layer
                         query volumes (not a physics broadphase).
        within_radius:  O(sqrt(n) + m) average, m = result count.
    """

    def __init__(self) -> None:
        self._root: Optional[_KDNode] = None
        self._size: int = 0

    def build(self, entities: list[Entity]) -> None:
        points = [(e.id, e.state.position) for e in entities]
        self._root = self._build_recursive(points, depth=0)
        self._size = len(points)

    def _build_recursive(self, points: list[tuple[str, Vec3]], depth: int) -> Optional[_KDNode]:
        if not points:
            return None
        axis = depth % 3
        key = (lambda p: p[1].x) if axis == 0 else (lambda p: p[1].y) if axis == 1 else (lambda p: p[1].z)
        points = sorted(points, key=key)
        mid = len(points) // 2
        eid, pos = points[mid]
        node = _KDNode(eid, pos, axis)
        node.left = self._build_recursive(points[:mid], depth + 1)
        node.right = self._build_recursive(points[mid + 1:], depth + 1)
        return node

    def nearest(self, point: Vec3, k: int = 1,
                predicate: Optional[callable] = None) -> list[tuple[str, float]]:
        if self._root is None or k <= 0:
            return []
        best: list[tuple[float, str]] = []  # max-heap emulated via sorted list (k is small in practice)

        def visit(node: Optional[_KDNode]) -> None:
            if node is None:
                return
            if predicate is None or predicate(node.entity_id):
                d2 = _dist2(point, node.position)
                if len(best) < k:
                    best.append((d2, node.entity_id))
                    best.sort(key=lambda t: t[0])
                elif d2 < best[-1][0]:
                    best[-1] = (d2, node.entity_id)
                    best.sort(key=lambda t: t[0])

            axis_val = (point.x, point.y, point.z)[node.axis]
            node_val = (node.position.x, node.position.y, node.position.z)[node.axis]
            first, second = (node.left, node.right) if axis_val < node_val else (node.right, node.left)
            visit(first)
            # Only descend into the far side if the splitting-plane distance
            # could still beat our current worst kept candidate.
            plane_dist2 = (axis_val - node_val) ** 2
            if len(best) < k or plane_dist2 < best[-1][0]:
                visit(second)

        visit(self._root)
        return [(eid, math.sqrt(d2)) for d2, eid in best]

    def within_radius(self, point: Vec3, radius: float,
                       predicate: Optional[callable] = None) -> list[tuple[str, float]]:
        if self._root is None or radius < 0:
            return []
        r2 = radius * radius
        found: list[tuple[str, float]] = []

        def visit(node: Optional[_KDNode]) -> None:
            if node is None:
                return
            if predicate is None or predicate(node.entity_id):
                d2 = _dist2(point, node.position)
                if d2 <= r2:
                    found.append((node.entity_id, math.sqrt(d2)))

            axis_val = (point.x, point.y, point.z)[node.axis]
            node_val = (node.position.x, node.position.y, node.position.z)[node.axis]
            plane_dist = axis_val - node_val
            near, far = (node.left, node.right) if axis_val < node_val else (node.right, node.left)
            visit(near)
            if plane_dist * plane_dist <= r2:
                visit(far)

        visit(self._root)
        found.sort(key=lambda t: t[1])
        return found

    def __len__(self) -> int:
        return self._size
