"""
indexes.py
──────────
In-memory lookup tables the World Controller maintains alongside the
WorldSpec so that entity/selection/group/lock queries are O(1) instead
of O(n) scans over `WorldSpec.entities` on every access.

The `EntityIndex` is rebuilt from scratch whenever a WorldSpec is
loaded/created/replaced/reset, and incrementally updated by commands as
they execute/undo. It is never treated as a source of truth -- the
WorldSpec always is -- so a full `rebuild()` is always safe and always
correct.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterable, Optional

from world_spec import Entity, WorldSpec

# Reserved tag prefixes the World Controller layers onto Entity.tags to
# author state that WorldSpec's data contract does not model directly
# (grouping, parenting, kinematic/sensor body mode). SceneCompiler and
# any other WorldSpec consumer sees these as ordinary tags; only the
# World Controller interprets them structurally.
GROUP_TAG_PREFIX = "__group__:"
PARENT_TAG_PREFIX = "__parent__:"
BODY_MODE_TAG_PREFIX = "__body_mode__:"


@dataclass
class EntityIndex:
    """Fast lookup tables over the entities of the currently loaded WorldSpec.

    All mutating methods are intended to be called only by the
    `WorldController` (directly or via commands) while holding the
    controller's lock; `EntityIndex` itself adds a lock so it remains
    safe to query concurrently from read-only callers (e.g. a UI thread
    polling selection state).
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _by_id: dict[str, Entity] = field(default_factory=dict)
    _by_type: dict[str, set[str]] = field(default_factory=dict)
    _by_tag: dict[str, set[str]] = field(default_factory=dict)
    _by_material: dict[str, set[str]] = field(default_factory=dict)
    _children_of: dict[str, set[str]] = field(default_factory=dict)  # parent_id -> child ids
    _groups: dict[str, set[str]] = field(default_factory=dict)       # group_name -> member ids
    _selected: set[str] = field(default_factory=set)
    _locked: set[str] = field(default_factory=set)

    # ── (re)building ─────────────────────────────────────────────────

    def rebuild(self, world_spec: WorldSpec) -> None:
        """Recompute every index from `world_spec.entities` from scratch."""
        with self._lock:
            self._by_id.clear()
            self._by_type.clear()
            self._by_tag.clear()
            self._by_material.clear()
            self._children_of.clear()
            self._groups.clear()
            self._selected.clear()
            self._locked.clear()

            for entity in world_spec.entities:
                self._index_entity(entity)

    def _index_entity(self, entity: Entity) -> None:
        self._by_id[entity.id] = entity
        self._by_type.setdefault(entity.entity_type, set()).add(entity.id)
        self._by_material.setdefault(entity.material, set()).add(entity.id)

        for tag in entity.tags:
            if tag.startswith(GROUP_TAG_PREFIX):
                self._groups.setdefault(tag[len(GROUP_TAG_PREFIX):], set()).add(entity.id)
            elif tag.startswith(PARENT_TAG_PREFIX):
                parent_id = tag[len(PARENT_TAG_PREFIX):]
                self._children_of.setdefault(parent_id, set()).add(entity.id)
            else:
                self._by_tag.setdefault(tag, set()).add(entity.id)

    def _unindex_entity(self, entity: Entity) -> None:
        self._by_id.pop(entity.id, None)
        self._by_type.get(entity.entity_type, set()).discard(entity.id)
        self._by_material.get(entity.material, set()).discard(entity.id)
        for tag in entity.tags:
            if tag.startswith(GROUP_TAG_PREFIX):
                self._groups.get(tag[len(GROUP_TAG_PREFIX):], set()).discard(entity.id)
            elif tag.startswith(PARENT_TAG_PREFIX):
                self._children_of.get(tag[len(PARENT_TAG_PREFIX):], set()).discard(entity.id)
            else:
                self._by_tag.get(tag, set()).discard(entity.id)
        self._selected.discard(entity.id)
        self._locked.discard(entity.id)

    # ── incremental maintenance (called by commands) ────────────────

    def on_entity_added(self, entity: Entity) -> None:
        with self._lock:
            self._index_entity(entity)

    def on_entity_removed(self, entity: Entity) -> None:
        with self._lock:
            self._unindex_entity(entity)

    def on_entity_reindexed(self, entity: Entity) -> None:
        """Call after any in-place mutation of `entity` (type/material/tags changed)."""
        with self._lock:
            # Cheap approach: drop and re-add. Correct regardless of which
            # field changed, and index sizes are small relative to the
            # cost of a targeted diff for a scene-authoring workload.
            for id_set in (
                *self._by_type.values(),
                *self._by_tag.values(),
                *self._by_material.values(),
                *self._children_of.values(),
                *self._groups.values(),
            ):
                id_set.discard(entity.id)
            self._index_entity(entity)

    # ── queries ──────────────────────────────────────────────────────

    def get(self, entity_id: str) -> Optional[Entity]:
        with self._lock:
            return self._by_id.get(entity_id)

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._by_id.keys())

    def by_type(self, entity_type: str) -> list[str]:
        with self._lock:
            return sorted(self._by_type.get(entity_type, set()))

    def by_tag(self, tag: str) -> list[str]:
        with self._lock:
            return sorted(self._by_tag.get(tag, set()))

    def by_material(self, material: str) -> list[str]:
        with self._lock:
            return sorted(self._by_material.get(material, set()))

    def children_of(self, parent_id: str) -> list[str]:
        with self._lock:
            return sorted(self._children_of.get(parent_id, set()))

    def group_members(self, group_name: str) -> list[str]:
        with self._lock:
            return sorted(self._groups.get(group_name, set()))

    def group_names(self) -> list[str]:
        with self._lock:
            return sorted(self._groups.keys())

    def search(self, predicate) -> list[str]:
        """Return ids of every entity for which `predicate(entity)` is truthy."""
        with self._lock:
            return [eid for eid, e in self._by_id.items() if predicate(e)]

    def filter_by_label_substring(self, substring: str, case_sensitive: bool = False) -> list[str]:
        needle = substring if case_sensitive else substring.lower()
        with self._lock:
            results = []
            for eid, e in self._by_id.items():
                haystack = e.label if case_sensitive else e.label.lower()
                if needle in haystack:
                    results.append(eid)
            return results

    # ── selection ────────────────────────────────────────────────────

    def selected_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._selected)

    def is_selected(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._selected

    def set_selection(self, entity_ids: Iterable[str]) -> None:
        with self._lock:
            self._selected = {eid for eid in entity_ids if eid in self._by_id}

    def add_to_selection(self, entity_ids: Iterable[str]) -> None:
        with self._lock:
            self._selected.update(eid for eid in entity_ids if eid in self._by_id)

    def remove_from_selection(self, entity_ids: Iterable[str]) -> None:
        with self._lock:
            self._selected.difference_update(entity_ids)

    def toggle_selection(self, entity_ids: Iterable[str]) -> None:
        with self._lock:
            for eid in entity_ids:
                if eid not in self._by_id:
                    continue
                if eid in self._selected:
                    self._selected.discard(eid)
                else:
                    self._selected.add(eid)

    def clear_selection(self) -> None:
        with self._lock:
            self._selected.clear()

    # ── locking ──────────────────────────────────────────────────────

    def is_locked(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._locked

    def lock(self, entity_id: str) -> None:
        with self._lock:
            self._locked.add(entity_id)

    def unlock(self, entity_id: str) -> None:
        with self._lock:
            self._locked.discard(entity_id)

    def locked_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._locked)
