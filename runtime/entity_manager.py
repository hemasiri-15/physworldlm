"""
entity_manager.py
══════════════════════════════════════════════════════════════════════════
Runtime entity bookkeeping for the PhysWorldLM Runtime Engine.

Scope
-----
`EntityManager` owns exactly one responsibility: tracking *which*
entities exist at runtime, their hierarchy/attachment relationships,
ownership, and per-entity component/state bags -- it does not simulate
physics, render anything, or talk to a backend. Instantiation from a
compiled `SceneGraph` is a thin, optional convenience
(`instantiate_from_scene_graph()`); the rest of the API is scene-graph
agnostic so a caller (or a test) can build entities up by hand.

This module has no simulator-specific imports and does not depend on
`Timeline` or `SimulationController` -- `SimulationController` is the
only collaborator that reaches into both.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional, TYPE_CHECKING

from runtime.runtime_context import (
    EntityAttachmentError,
    EntityNotFoundError,
    RuntimeContext,
    RuntimeEventType,
)

if TYPE_CHECKING:
    from scene_compiler import SceneGraph


# ════════════════════════════════════════════════════════════════════════
# Handles & records
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EntityHandle:
    """Opaque, stable reference to a runtime entity.

    `entity_id` is a fresh runtime identity assigned by `EntityManager`
    at instantiation time (distinct from whatever id a `WorldSpec`/
    `SceneGraph` entity carried at compile time, which is preserved as
    `source_id` for provenance/debugging only).
    """

    entity_id: str
    source_id: Optional[str] = None

    def __str__(self) -> str:
        return self.entity_id


@dataclass
class EntityRecord:
    """Full runtime bookkeeping for a single entity.

    Attributes:
        handle: This entity's stable `EntityHandle`.
        name: Human-readable label (from the source scene entity, or
            caller-supplied).
        kind: Free-form category string (e.g. `"rigid_body"`, `"camera"`,
            `"terrain"`), used for lookup/filtering only -- the runtime
            attaches no behavior to it.
        components: Arbitrary named component payloads (transform,
            physics material, sensor config, ...). The runtime treats
            these as opaque data; components are interpreted by backend
            adapters/sensor managers, never by `EntityManager` itself.
        parent: Handle of this entity's parent, or `None` if root.
        children: Handles of this entity's direct children.
        owner: Free-form owner tag (e.g. an AI-agent plugin name or
            `"scene"` for scene-authored entities), for filtering/cleanup.
        active: Whether this entity currently participates in stepping;
            `False` entities are retained (not destroyed) but skipped by
            state sync.
        created_at: UTC creation timestamp.
        destroyed_at: UTC destruction timestamp, set by `destroy()`
            rather than by removing the record outright, so a destroyed
            entity remains inspectable for one more diagnostics pass.
    """

    handle: EntityHandle
    name: str
    kind: str = "generic"
    components: dict[str, Any] = field(default_factory=dict)
    parent: Optional[EntityHandle] = None
    children: list[EntityHandle] = field(default_factory=list)
    owner: str = "scene"
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    destroyed_at: Optional[datetime] = None

    @property
    def is_destroyed(self) -> bool:
        return self.destroyed_at is not None

    def to_dict(self) -> dict:
        return {
            "entity_id": self.handle.entity_id,
            "source_id": self.handle.source_id,
            "name": self.name,
            "kind": self.kind,
            "components": list(self.components.keys()),
            "parent": self.parent.entity_id if self.parent else None,
            "children": [c.entity_id for c in self.children],
            "owner": self.owner,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "destroyed_at": self.destroyed_at.isoformat() if self.destroyed_at else None,
        }


# ════════════════════════════════════════════════════════════════════════
# EntityManager
# ════════════════════════════════════════════════════════════════════════

class EntityManager:
    """Thread-safe registry of runtime entities, their hierarchy, and
    ownership.

    All mutation happens behind a single `RLock`; iteration helpers
    (`all()`, `roots()`, `children_of()`) return snapshots (tuples) taken
    under that lock rather than live views, so callers may safely iterate
    while another thread mutates the manager.

    Example:
        >>> em = EntityManager()
        >>> ball = em.create_entity("ball", kind="rigid_body")
        >>> ground = em.create_entity("ground", kind="static_body")
        >>> em.attach(ball, parent=None)   # already root; no-op example
        >>> em.destroy_entity(ball)
    """

    def __init__(self, context: Optional[RuntimeContext] = None) -> None:
        """Initialize an empty entity manager.

        Args:
            context: Optional `RuntimeContext`. If provided, lifecycle
                operations publish `RuntimeEventType` diagnostics on
                `context.event_bus` and update `context.statistics`.
                Omitting it keeps `EntityManager` usable standalone.
        """
        self._lock = threading.RLock()
        self._context = context
        self._entities: dict[str, EntityRecord] = {}
        self._roots: set[str] = set()

    # ── creation / destruction ─────────────────────────────────────────

    def create_entity(
        self,
        name: str,
        *,
        kind: str = "generic",
        components: Optional[dict[str, Any]] = None,
        parent: Optional[EntityHandle] = None,
        owner: str = "scene",
        source_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> EntityHandle:
        """Instantiate a new runtime entity.

        Args:
            name: Human-readable label.
            kind: Free-form category string.
            components: Initial component payload (copied, not aliased).
            parent: Parent to attach under at creation time, or `None`
                for a root entity.
            owner: Owner tag for filtering/cleanup.
            source_id: Original compile-time entity id, for provenance.
            entity_id: Explicit runtime id override; a fresh UUID4 is
                generated if omitted. Callers doing deterministic replay
                may want to supply this explicitly.

        Returns:
            The new entity's `EntityHandle`.

        Raises:
            EntityAttachmentError: If `parent` does not exist.
        """
        with self._lock:
            resolved_id = entity_id or str(uuid.uuid4())
            handle = EntityHandle(entity_id=resolved_id, source_id=source_id)
            if parent is not None and parent.entity_id not in self._entities:
                raise EntityAttachmentError(f"Cannot create '{name}' under unknown parent '{parent.entity_id}'.")

            record = EntityRecord(handle=handle, name=name, kind=kind, components=dict(components or {}), owner=owner)
            self._entities[resolved_id] = record

            if parent is not None:
                record.parent = parent
                self._entities[parent.entity_id].children.append(handle)
            else:
                self._roots.add(resolved_id)

            self._touch_statistics(created=1)
            self._publish(RuntimeEventType.ENTITY_CREATED, entity_id=resolved_id, name=name, kind=kind)
            return handle

    def destroy_entity(self, handle: EntityHandle, *, cascade: bool = True) -> None:
        """Destroy an entity (and, by default, its subtree).

        Args:
            handle: Entity to destroy.
            cascade: If True (default), all descendants are destroyed
                too. If False, children are re-parented to this entity's
                parent (or promoted to root) before it is removed.

        Raises:
            EntityNotFoundError: If `handle` is unknown.
        """
        with self._lock:
            record = self._require(handle)
            if cascade:
                for child in tuple(record.children):
                    self.destroy_entity(child, cascade=True)
            else:
                for child in tuple(record.children):
                    self._reparent_locked(child, record.parent)

            if record.parent is not None:
                parent_record = self._entities.get(record.parent.entity_id)
                if parent_record is not None:
                    parent_record.children = [c for c in parent_record.children if c.entity_id != handle.entity_id]
            self._roots.discard(handle.entity_id)

            record.active = False
            record.destroyed_at = datetime.now(timezone.utc)
            del self._entities[handle.entity_id]

            self._touch_statistics(destroyed=1)
            self._publish(RuntimeEventType.ENTITY_DESTROYED, entity_id=handle.entity_id)

    def clear(self) -> None:
        """Destroy every entity. Used by `SimulationController.reset()`."""
        with self._lock:
            for handle in tuple(EntityHandle(eid) for eid in self._roots):
                self.destroy_entity(handle, cascade=True)
            # Catch any orphaned records (shouldn't normally happen).
            for eid in tuple(self._entities.keys()):
                self.destroy_entity(EntityHandle(eid), cascade=True)

    # ── hierarchy / attachment ──────────────────────────────────────────

    def attach(self, handle: EntityHandle, parent: Optional[EntityHandle]) -> None:
        """Re-parent `handle` under `parent` (or promote to root if `None`).

        Raises:
            EntityNotFoundError: If `handle` or `parent` is unknown.
            EntityAttachmentError: If the operation would create a cycle
                (attaching an entity under one of its own descendants),
                or if `parent == handle`.
        """
        with self._lock:
            self._require(handle)
            if parent is not None:
                self._require(parent)
                if parent.entity_id == handle.entity_id:
                    raise EntityAttachmentError("An entity cannot be attached to itself.")
                if self._is_descendant(candidate=parent, ancestor=handle):
                    raise EntityAttachmentError(
                        f"Attaching '{handle.entity_id}' under '{parent.entity_id}' would create a cycle."
                    )
            self._reparent_locked(handle, parent)
            self._publish(
                RuntimeEventType.ENTITY_ATTACHED,
                entity_id=handle.entity_id,
                parent_id=parent.entity_id if parent else None,
            )

    def detach(self, handle: EntityHandle) -> None:
        """Promote `handle` to a root entity (equivalent to `attach(handle, None)`)."""
        with self._lock:
            self._require(handle)
            self._reparent_locked(handle, None)
            self._publish(RuntimeEventType.ENTITY_DETACHED, entity_id=handle.entity_id)

    def _reparent_locked(self, handle: EntityHandle, new_parent: Optional[EntityHandle]) -> None:
        record = self._entities[handle.entity_id]
        old_parent = record.parent
        if old_parent is not None:
            old_record = self._entities.get(old_parent.entity_id)
            if old_record is not None:
                old_record.children = [c for c in old_record.children if c.entity_id != handle.entity_id]
        else:
            self._roots.discard(handle.entity_id)

        record.parent = new_parent
        if new_parent is not None:
            self._entities[new_parent.entity_id].children.append(handle)
        else:
            self._roots.add(handle.entity_id)

    def _is_descendant(self, *, candidate: EntityHandle, ancestor: EntityHandle) -> bool:
        """True if `candidate` is `ancestor` or lies within its subtree."""
        stack = [ancestor]
        while stack:
            current = stack.pop()
            if current.entity_id == candidate.entity_id:
                return True
            stack.extend(self._entities[current.entity_id].children)
        return False

    # ── lookup ───────────────────────────────────────────────────────

    def get(self, handle: EntityHandle) -> EntityRecord:
        """Fetch an entity's full record.

        Raises:
            EntityNotFoundError: If `handle` is unknown.
        """
        with self._lock:
            return self._require(handle)

    def find_by_name(self, name: str) -> tuple[EntityHandle, ...]:
        """Return handles of every live entity with the given `name`."""
        with self._lock:
            return tuple(rec.handle for rec in self._entities.values() if rec.name == name)

    def find_by_kind(self, kind: str) -> tuple[EntityHandle, ...]:
        """Return handles of every live entity of the given `kind`."""
        with self._lock:
            return tuple(rec.handle for rec in self._entities.values() if rec.kind == kind)

    def find_by_owner(self, owner: str) -> tuple[EntityHandle, ...]:
        """Return handles of every live entity tagged with the given `owner`."""
        with self._lock:
            return tuple(rec.handle for rec in self._entities.values() if rec.owner == owner)

    def children_of(self, handle: EntityHandle) -> tuple[EntityHandle, ...]:
        """Return direct children of `handle`.

        Raises:
            EntityNotFoundError: If `handle` is unknown.
        """
        with self._lock:
            return tuple(self._require(handle).children)

    def roots(self) -> tuple[EntityHandle, ...]:
        """Return every root-level (parentless) entity."""
        with self._lock:
            return tuple(EntityHandle(eid) for eid in sorted(self._roots))

    def all(self) -> tuple[EntityRecord, ...]:
        """Return a snapshot of every live entity's record."""
        with self._lock:
            return tuple(self._entities.values())

    def count(self) -> int:
        with self._lock:
            return len(self._entities)

    def __len__(self) -> int:
        return self.count()

    def __iter__(self) -> Iterator[EntityRecord]:
        return iter(self.all())

    def __contains__(self, handle: EntityHandle) -> bool:
        with self._lock:
            return handle.entity_id in self._entities

    # ── components ────────────────────────────────────────────────────

    def set_component(self, handle: EntityHandle, key: str, value: Any) -> None:
        """Set (or replace) a named component payload on `handle`.

        Raises:
            EntityNotFoundError: If `handle` is unknown.
        """
        with self._lock:
            self._require(handle).components[key] = value

    def get_component(self, handle: EntityHandle, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._require(handle).components.get(key, default)

    def remove_component(self, handle: EntityHandle, key: str) -> None:
        with self._lock:
            self._require(handle).components.pop(key, None)

    def set_active(self, handle: EntityHandle, active: bool) -> None:
        """Enable/disable an entity's participation in stepping without destroying it."""
        with self._lock:
            self._require(handle).active = active

    # ── scene-graph instantiation (thin, optional convenience) ────────

    def instantiate_from_scene_graph(self, scene_graph: "SceneGraph", *, owner: str = "scene") -> tuple[EntityHandle, ...]:
        """Instantiate one runtime entity per node in a compiled `SceneGraph`.

        This is intentionally a thin adapter: it does not interpret,
        validate, or lower `SceneGraph` semantics -- that is
        `worldspec_builder.WorldSpecBuilder`'s and `scene_compiler.SceneCompiler`'s
        job, already done by the time a `SceneGraph` reaches here. This
        method only walks `scene_graph.nodes` (or `.entities`, whichever
        the compiled graph exposes) and calls `create_entity()` once per
        node, preserving parent/child edges the compiler already resolved.

        Args:
            scene_graph: A compiled scene graph, already produced
                upstream by `WorldSpecBuilder`/`SceneCompiler`.
            owner: Owner tag applied to every instantiated entity.

        Returns:
            Handles for every root entity created (children are reachable
            via `children_of()`), in the scene graph's own node order.
        """
        nodes = getattr(scene_graph, "nodes", None) or getattr(scene_graph, "entities", ())
        id_map: dict[str, EntityHandle] = {}
        roots: list[EntityHandle] = []

        with self._lock:
            for node in nodes:
                source_id = getattr(node, "id", None) or getattr(node, "entity_id", None) or str(uuid.uuid4())
                name = getattr(node, "name", source_id)
                kind = getattr(node, "kind", None) or getattr(node, "type", "generic")
                parent_source_id = getattr(node, "parent_id", None) or getattr(node, "parent", None)
                parent_handle = id_map.get(parent_source_id) if parent_source_id else None
                components = getattr(node, "components", None) or getattr(node, "properties", None) or {}

                handle = self.create_entity(
                    name=name, kind=kind, components=dict(components) if isinstance(components, dict) else {},
                    parent=parent_handle, owner=owner, source_id=source_id,
                )
                id_map[source_id] = handle
                if parent_handle is None:
                    roots.append(handle)

        return tuple(roots)

    # ── internal ────────────────────────────────────────────────────

    def _require(self, handle: EntityHandle) -> EntityRecord:
        record = self._entities.get(handle.entity_id)
        if record is None:
            raise EntityNotFoundError(f"No live entity with id '{handle.entity_id}'.")
        return record

    def _touch_statistics(self, *, created: int = 0, destroyed: int = 0) -> None:
        if self._context is None:
            return
        self._context.statistics.entities_created += created
        self._context.statistics.entities_destroyed += destroyed

    def _publish(self, event_type: RuntimeEventType, **payload: Any) -> None:
        if self._context is not None:
            self._context.publish(event_type, **payload)


__all__ = [
    "EntityHandle",
    "EntityRecord",
    "EntityManager",
]
