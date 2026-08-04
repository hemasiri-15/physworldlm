"""
world_engine.memory
──────────────────────
WorldMemory — the "Past → Current → Future" layer, à la Dreamer-style
world models, implemented as a git-like version DAG over `WorldSpec`
snapshots.

Distinction from `history.HistoryManager`:
    HistoryManager is cheap, linear, command-level undo/redo — appropriate
    for every single edit. WorldMemory is coarser and deep-copy based:
    callers explicitly `checkpoint()` named points worth remembering
    (roughly, "past" states), and `PredictiveEngine` (not implemented in
    this file) is expected to call `add_future()` to attach speculative,
    not-yet-real predicted continuations without disturbing the actual
    timeline. This gives exactly the

        Past World -> Current World -> Future World(s)

    structure requested, where "Future World" is explicitly plural and
    branching: `PredictiveEngine` can hang multiple candidate futures off
    the same current version and let downstream code (a planner, a
    risk-scorer) pick one to `promote()`.

Version kinds:
    "past"    — an actual, committed point the world has been in.
    "current" — exactly one node is marked current at any time (the live
                world tracks this node's spec).
    "future"  — a speculative, not-yet-realized continuation. Multiple
                future nodes may share the same parent (branching). A
                future node becomes "past" (and possibly "current") only
                via `promote()`, modeling a predicted world being accepted
                into the real timeline.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from world_spec import WorldSpec
from .diff import DiffEngine, WorldDiff
from .exceptions import VersionError


@dataclass
class VersionNode:
    """One committed point in the world's history/branch DAG."""
    version_id: str
    parent_id: Optional[str]
    spec: WorldSpec               # deep-copied, owned by this node
    label: str
    kind: str                     # "past" | "current" | "future"
    tick: int
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class WorldMemory:
    """
    Version DAG of `WorldSpec` snapshots supporting linear history,
    branching, and speculative future exploration.

    Every node owns its own deep copy of the `WorldSpec` at that point —
    this trades memory for simplicity and safety (no node can be mutated
    by later edits to the live world). For very long-running sessions with
    many checkpoints, consider periodic pruning via `prune_before(tick)`.
    """

    def __init__(self, initial_spec: WorldSpec, initial_tick: int = 0) -> None:
        root_id = self._new_id()
        root = VersionNode(
            version_id=root_id, parent_id=None,
            spec=copy.deepcopy(initial_spec), label="root",
            kind="current", tick=initial_tick,
        )
        self._nodes: dict[str, VersionNode] = {root_id: root}
        self._children: dict[str, list[str]] = {root_id: []}
        self._current_id: str = root_id
        self._root_id: str = root_id

    @staticmethod
    def _new_id() -> str:
        return f"v_{uuid.uuid4().hex[:10]}"

    # ── committing the timeline ─────────────────────────────────────────

    def checkpoint(self, spec: WorldSpec, tick: int, label: str = "") -> str:
        """
        Purpose:
            Record `spec` as a new "past" node, child of the current node,
            and advance "current" to it — the normal, linear-timeline case
            (no branching).
        Inputs:
            spec: the live WorldSpec to snapshot (deep-copied internally).
            tick: WorldEngine tick at checkpoint time, for ordering/audit.
            label: optional human-readable name (e.g. "before_rain").
        Outputs:
            The new version_id.
        Complexity:
            O(E + I) for the deep copy.
        """
        old_current = self._nodes[self._current_id]
        old_current.kind = "past" if old_current.kind == "current" else old_current.kind
        return self._commit(parent_id=self._current_id, spec=spec, tick=tick,
                             label=label, kind="current", make_current=True)

    def branch(self, spec: WorldSpec, tick: int, from_version: Optional[str] = None,
               label: str = "") -> str:
        """
        Purpose:
            Create a new branch (a node whose parent need not be the
            current node) WITHOUT moving "current" — useful for exploring
            an alternate past without disturbing the live timeline.
        Inputs:
            spec: WorldSpec content for the new branch head.
            tick: tick value to record.
            from_version: version_id to branch from; defaults to current.
            label: human-readable branch name.
        Outputs:
            The new version_id (kind="past", NOT made current).
        Exceptions:
            VersionError if `from_version` does not exist.
        Complexity:
            O(E + I).
        """
        parent_id = from_version or self._current_id
        if parent_id not in self._nodes:
            raise VersionError(f"No such version '{parent_id}'")
        return self._commit(parent_id=parent_id, spec=spec, tick=tick,
                             label=label, kind="past", make_current=False)

    def add_future(self, predicted_spec: WorldSpec, tick: int,
                    from_version: Optional[str] = None, label: str = "",
                    metadata: Optional[dict] = None) -> str:
        """
        Purpose:
            Attach a speculative predicted continuation (from
            PredictiveEngine) to `from_version` (default: current) without
            altering the real timeline. Multiple calls from the same
            parent model multiple candidate futures.
        Inputs:
            predicted_spec: the predicted WorldSpec.
            tick: predicted tick (may be > current tick to represent
                "N steps ahead").
            from_version: version to branch the prediction from.
            label: human-readable label (e.g. "collision_in_2s").
            metadata: arbitrary extra info (e.g. {"confidence": 0.82,
                "risk_score": 0.4}) — the seam PredictiveEngine uses to
                attach its own scoring without WorldMemory needing to know
                about it.
        Outputs:
            The new version_id (kind="future").
        Complexity:
            O(E + I).
        """
        parent_id = from_version or self._current_id
        if parent_id not in self._nodes:
            raise VersionError(f"No such version '{parent_id}'")
        vid = self._commit(parent_id=parent_id, spec=predicted_spec, tick=tick,
                            label=label, kind="future", make_current=False)
        if metadata:
            self._nodes[vid].metadata.update(metadata)
        return vid

    def promote(self, future_version_id: str) -> None:
        """
        Purpose:
            Accept a "future" node into the real timeline: relabel it
            "current" (demoting the previous current node to "past") and
            move the current pointer to it. Models a predicted world being
            realized (e.g. PredictiveEngine's chosen forecast turned out to
            match what actually happened, or a planner committed to it).
        Inputs:
            future_version_id: id of a node with kind == "future".
        Exceptions:
            VersionError if the id doesn't exist or isn't a future node.
        Complexity:
            O(1).
        """
        node = self._nodes.get(future_version_id)
        if node is None:
            raise VersionError(f"No such version '{future_version_id}'")
        if node.kind != "future":
            raise VersionError(f"Version '{future_version_id}' is not a future node (kind={node.kind})")
        old_current = self._nodes[self._current_id]
        old_current.kind = "past"
        node.kind = "current"
        self._current_id = future_version_id

    def _commit(self, parent_id: str, spec: WorldSpec, tick: int, label: str,
                kind: str, make_current: bool) -> str:
        vid = self._new_id()
        node = VersionNode(
            version_id=vid, parent_id=parent_id, spec=copy.deepcopy(spec),
            label=label or vid, kind=kind, tick=tick,
        )
        self._nodes[vid] = node
        self._children.setdefault(parent_id, []).append(vid)
        self._children.setdefault(vid, [])
        if make_current:
            self._current_id = vid
        return vid

    # ── navigation / read access ────────────────────────────────────────

    @property
    def current_id(self) -> str:
        return self._current_id

    @property
    def root_id(self) -> str:
        return self._root_id

    def get(self, version_id: str) -> VersionNode:
        node = self._nodes.get(version_id)
        if node is None:
            raise VersionError(f"No such version '{version_id}'")
        return node

    def checkout(self, version_id: str) -> WorldSpec:
        """
        Purpose:
            Return a deep copy of the WorldSpec at `version_id`, for
            `WorldEngine.checkout()` to load as the live world. Does NOT
            itself change which node is "current" — callers that want the
            timeline pointer moved should treat checkout as loading a
            snapshot (`WorldEngine.checkout` handles both).
        Complexity:
            O(E + I) for the deep copy.
        """
        return copy.deepcopy(self.get(version_id).spec)

    def children(self, version_id: str) -> list[VersionNode]:
        return [self._nodes[cid] for cid in self._children.get(version_id, [])]

    def futures_of(self, version_id: Optional[str] = None) -> list[VersionNode]:
        """All 'future' nodes branching from `version_id` (default: current)."""
        vid = version_id or self._current_id
        return [c for c in self.children(vid) if c.kind == "future"]

    def path_to_root(self, version_id: Optional[str] = None) -> list[VersionNode]:
        """Root-to-`version_id` path, oldest first — the linear "past" chain."""
        vid = version_id or self._current_id
        chain: list[VersionNode] = []
        while vid is not None:
            node = self._nodes[vid]
            chain.append(node)
            vid = node.parent_id
        return list(reversed(chain))

    def timeline(self) -> list[VersionNode]:
        """
        Convenience view matching the requested
        `Past World -> Current World -> Future World` narrative: the
        linear past-to-current chain, followed by every future branching
        directly off the current node.
        """
        return self.path_to_root(self._current_id) + self.futures_of(self._current_id)

    def diff(self, version_a: str, version_b: str) -> WorldDiff:
        return DiffEngine.compute(self.get(version_a).spec, self.get(version_b).spec)

    def prune_before(self, tick: int) -> int:
        """
        Purpose:
            Drop "past" (not current/future) nodes with tick < `tick` and
            no children, to bound memory growth in long sessions. Leaves
            branch points (nodes with children) intact regardless of tick,
            since removing them would disconnect their descendants.
        Outputs:
            Number of nodes actually pruned.
        Complexity:
            O(V) over all versions.
        """
        removed = 0
        for vid in list(self._nodes.keys()):
            node = self._nodes[vid]
            if (node.kind == "past" and node.tick < tick
                    and not self._children.get(vid) and vid != self._root_id):
                parent = node.parent_id
                if parent is not None:
                    self._children[parent] = [c for c in self._children[parent] if c != vid]
                del self._nodes[vid]
                self._children.pop(vid, None)
                removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._nodes)
