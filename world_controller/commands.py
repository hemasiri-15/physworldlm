"""
commands.py
───────────
The Command pattern implementation for the World Controller.

Every mutation applied to a `WorldSpec` — whether triggered by the UI, an
AI agent, a CSP solver, the Physics Validator, an Omniverse callback, or a
future plugin — is expressed as a `Command` object. Commands are the ONLY
thing that mutates a WorldSpec once a `WorldController` owns it: nothing
calls `entity.mass = x` directly, everything goes through
`controller.execute(SomeCommand(...))`.

Design
------
`Command` (Protocol) is the structural contract: `execute`, `undo`, `redo`,
`serialize`, `description`, `timestamp`, `command_id`.

`AbstractCommand` (ABC) implements the boilerplate (id, timestamp,
executed-state tracking, error wrapping) and asks subclasses for two
private hooks: `_do(context)` and `_undo(context)`. `redo()` is defined,
once, on the base class as "call `_do` again" — this is correct as long as
`_do` is idempotent given the command's own stored state, which every
concrete command below is written to satisfy (see each class's docstring
for its redo-safety argument).

Two generic bases, `_EntityFieldCommand` and `_EnvironmentFieldCommand`,
implement "capture old value / apply new value" for simple scalar or
vector field edits via getter/setter closures. Every simple setter-style
command in this module (mass, friction, restitution, gravity, weather,
...) is a thin, explicitly-named subclass of one of these two bases —
this keeps ~30 distinct, individually undoable, individually serializable
command types without duplicating capture/apply/undo/serialize logic
thirty times over. Structural commands (add/delete/duplicate entity,
grouping, relationships, transactions, ...) implement `_do`/`_undo`
directly because their undo semantics are not a simple value swap.
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from world_spec import (
    BoundingBox,
    Entity,
    Environment,
    Interaction,
    PhysicsState,
    Vec3,
    Wind,
    WorldSpec,
)

from world_controller.enums import ChangeEventType, EntityBodyMode
from world_controller.events import ChangeEvent, EventBus
from world_controller.indexes import (
    BODY_MODE_TAG_PREFIX,
    GROUP_TAG_PREFIX,
    PARENT_TAG_PREFIX,
    EntityIndex,
)
from world_controller.exceptions import (
    CommandExecutionError,
    CommandUndoError,
    DuplicateEntityIdError,
    EntityLockedError,
    EntityNotFoundError,
    InvalidParameterError,
    MaterialNotFoundError,
    RelationshipError,
    WorldControllerError,
)

# Reserved boolean-marker tags layered onto Entity.tags by the World
# Controller. WorldSpec has no `enabled` / `visible` fields, so these are
# authored as tags rather than by extending the data contract.
DISABLED_TAG = "__disabled__"
HIDDEN_TAG = "__hidden__"


def _to_serializable(value: Any) -> Any:
    """Best-effort conversion of a value into something `json.dumps`-safe."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


# ════════════════════════════════════════════════════════════════════════
# Command context
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CommandContext:
    """Everything a `Command` needs in order to execute/undo itself.

    A single `CommandContext` is owned by the `WorldController` and handed
    to every command it executes. Commands must not retain a reference to
    it beyond the call in which it was received — the controller may
    replace `world_spec` wholesale (e.g. on `load()`/`replace()`), which
    would silently invalidate a cached reference.
    """

    world_spec: WorldSpec
    index: EntityIndex
    event_bus: EventBus
    source: str = "command"

    def publish(
        self,
        event_type: ChangeEventType,
        entity_ids: tuple[str, ...] = (),
        payload: Optional[dict] = None,
    ) -> None:
        self.event_bus.publish(
            ChangeEvent(
                event_type=event_type,
                source=self.source,
                entity_ids=entity_ids,
                payload=payload or {},
            )
        )

    def resolve_entity(self, entity_id: str, *, require_unlocked: bool = True) -> Entity:
        """Fetch an entity by id, raising if missing or (optionally) locked."""
        entity = self.world_spec.get_entity(entity_id)
        if entity is None:
            raise EntityNotFoundError(entity_id)
        if require_unlocked and self.index.is_locked(entity_id):
            raise EntityLockedError(entity_id)
        return entity


# ════════════════════════════════════════════════════════════════════════
# Command protocol + abstract base
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Command(Protocol):
    """Structural contract every command must satisfy."""

    command_id: str
    timestamp: datetime

    def execute(self, context: CommandContext) -> None: ...

    def undo(self, context: CommandContext) -> None: ...

    def redo(self, context: CommandContext) -> None: ...

    def serialize(self) -> dict: ...

    @property
    def description(self) -> str: ...


class AbstractCommand(ABC):
    """Base implementation shared by every concrete command in this module.

    Attributes:
        command_id: Unique id assigned at construction, stable across the
            command's entire execute/undo/redo lifecycle.
        timestamp: UTC time the command instance was constructed.
    """

    def __init__(self) -> None:
        self.command_id: str = uuid.uuid4().hex
        self.timestamp: datetime = datetime.now(timezone.utc)
        self._executed: bool = False

    # ── public lifecycle ─────────────────────────────────────────────

    def execute(self, context: CommandContext) -> None:
        """Apply this command's effect to `context.world_spec`.

        Wraps `_do()` so every subclass gets consistent error translation
        into `CommandExecutionError` (WorldControllerError subclasses
        raised by `_do` — e.g. `EntityNotFoundError` — propagate as-is,
        since they are already meaningful and typed).
        """
        try:
            self._do(context)
        except WorldControllerError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-wrapped with command context
            raise CommandExecutionError(self.description, cause=exc) from exc
        self._executed = True

    def undo(self, context: CommandContext) -> None:
        """Reverse this command's effect on `context.world_spec`."""
        if not self._executed:
            raise CommandUndoError(self.description, cause=RuntimeError("command was never executed"))
        try:
            self._undo(context)
        except WorldControllerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CommandUndoError(self.description, cause=exc) from exc
        self._executed = False

    def redo(self, context: CommandContext) -> None:
        """Re-apply this command after it has been undone.

        Implemented as a second call to `execute()`. Every concrete `_do`
        in this module is written to be safe to call a second time after
        an intervening `_undo()` restored prior state — see each class's
        docstring.
        """
        self.execute(context)

    # ── subclass contract ────────────────────────────────────────────

    @abstractmethod
    def _do(self, context: CommandContext) -> None:
        """Perform the mutation. Must be safe to call again after `_undo`."""

    @abstractmethod
    def _undo(self, context: CommandContext) -> None:
        """Reverse the mutation performed by the most recent `_do()`."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short, human-readable summary of what this command does."""

    def serialize(self) -> dict:
        """Default serialization; subclasses extend via `_serialize_payload()`."""
        return {
            "command": self.__class__.__name__,
            "command_id": self.command_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            **self._serialize_payload(),
        }

    def _serialize_payload(self) -> dict:
        """Subclasses override to add command-specific serialized fields."""
        return {}


# ════════════════════════════════════════════════════════════════════════
# Generic field-edit bases
# ════════════════════════════════════════════════════════════════════════

class _EntityFieldCommand(AbstractCommand):
    """Generic "set one field on one Entity" command.

    Redo-safety: `_do` always re-reads the *current* value via `getter`
    before overwriting it, so calling `_do` again after `_undo` restored
    the prior value simply re-captures that same prior value and re-applies
    `new_value` — identical to the first execution.
    """

    def __init__(
        self,
        entity_id: str,
        new_value: Any,
        *,
        field_label: str,
        event_type: ChangeEventType,
        getter: Callable[[Entity], Any],
        setter: Callable[[Entity, Any], None],
        validator: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        super().__init__()
        if validator is not None:
            error = validator(new_value)
            if error is not None:
                raise InvalidParameterError(field_label, new_value, error)
        self.entity_id = entity_id
        self.new_value = new_value
        self._old_value: Any = None
        self._field_label = field_label
        self._event_type = event_type
        self._getter = getter
        self._setter = setter

    @property
    def description(self) -> str:
        return f"Set {self._field_label} of entity '{self.entity_id}' to {self.new_value!r}"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_value = self._getter(entity)
        self._setter(entity, self.new_value)
        context.index.on_entity_reindexed(entity)
        context.publish(
            self._event_type,
            entity_ids=(self.entity_id,),
            payload={"field": self._field_label, "old": _to_serializable(self._old_value), "new": _to_serializable(self.new_value)},
        )

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        self._setter(entity, self._old_value)
        context.index.on_entity_reindexed(entity)
        context.publish(
            self._event_type,
            entity_ids=(self.entity_id,),
            payload={"field": self._field_label, "old": _to_serializable(self.new_value), "new": _to_serializable(self._old_value)},
        )

    def _serialize_payload(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "field": self._field_label,
            "new_value": _to_serializable(self.new_value),
        }


class _EnvironmentFieldCommand(AbstractCommand):
    """Generic "set one field on the world Environment" command.

    Redo-safety: identical argument to `_EntityFieldCommand` — `_do`
    re-captures the current value every time it runs.
    """

    def __init__(
        self,
        new_value: Any,
        *,
        field_label: str,
        getter: Callable[[Environment], Any],
        setter: Callable[[Environment, Any], None],
        validator: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        super().__init__()
        if validator is not None:
            error = validator(new_value)
            if error is not None:
                raise InvalidParameterError(field_label, new_value, error)
        self.new_value = new_value
        self._old_value: Any = None
        self._field_label = field_label
        self._getter = getter
        self._setter = setter

    @property
    def description(self) -> str:
        return f"Set environment.{self._field_label} to {self.new_value!r}"

    def _do(self, context: CommandContext) -> None:
        env = context.world_spec.environment
        self._old_value = self._getter(env)
        self._setter(env, self.new_value)
        context.publish(
            ChangeEventType.ENVIRONMENT_CHANGED,
            payload={"field": self._field_label, "old": _to_serializable(self._old_value), "new": _to_serializable(self.new_value)},
        )

    def _undo(self, context: CommandContext) -> None:
        env = context.world_spec.environment
        self._setter(env, self._old_value)
        context.publish(
            ChangeEventType.ENVIRONMENT_CHANGED,
            payload={"field": self._field_label, "old": _to_serializable(self.new_value), "new": _to_serializable(self._old_value)},
        )

    def _serialize_payload(self) -> dict:
        return {"field": self._field_label, "new_value": _to_serializable(self.new_value)}


# ════════════════════════════════════════════════════════════════════════
# Entity lifecycle commands
# ════════════════════════════════════════════════════════════════════════

class AddEntityCommand(AbstractCommand):
    """Adds a fully-constructed `Entity` to the WorldSpec.

    Redo-safety: `_undo` removes the entity by id; `_do` re-appends the
    same `Entity` instance, so a second `_do` after `_undo` reproduces the
    original state exactly.
    """

    def __init__(self, entity: Entity) -> None:
        super().__init__()
        self._entity = entity

    @property
    def description(self) -> str:
        return f"Add entity '{self._entity.id}' ({self._entity.entity_type})"

    def _do(self, context: CommandContext) -> None:
        if context.world_spec.get_entity(self._entity.id) is not None:
            raise DuplicateEntityIdError(self._entity.id)
        context.world_spec.entities.append(self._entity)
        context.index.on_entity_added(self._entity)
        context.publish(ChangeEventType.ENTITY_ADDED, entity_ids=(self._entity.id,), payload={"entity": self._entity.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self._entity.id)
        if entity is None:
            raise EntityNotFoundError(self._entity.id)
        context.world_spec.entities.remove(entity)
        context.index.on_entity_removed(entity)
        context.publish(ChangeEventType.ENTITY_DELETED, entity_ids=(self._entity.id,))

    def _serialize_payload(self) -> dict:
        return {"entity": self._entity.to_dict()}


class DeleteEntityCommand(AbstractCommand):
    """Removes an entity and cascades removal of every Interaction that references it.

    Redo-safety: `_do` looks the entity up by id every time; after
    `_undo` re-inserts the entity (at its original list index) and the
    cascaded interactions (at their original list indices), a second
    `_do` finds the same state to remove again.
    """

    def __init__(self, entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._removed_entity: Optional[Entity] = None
        self._removed_index: int = -1
        self._removed_interactions: list[tuple[int, Interaction]] = []

    @property
    def description(self) -> str:
        return f"Delete entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._removed_index = context.world_spec.entities.index(entity)
        self._removed_entity = entity
        context.world_spec.entities.pop(self._removed_index)
        context.index.on_entity_removed(entity)

        self._removed_interactions = []
        remaining: list[Interaction] = []
        for idx, interaction in enumerate(context.world_spec.interactions):
            if interaction.entity_a == self.entity_id or interaction.entity_b == self.entity_id:
                self._removed_interactions.append((idx, interaction))
            else:
                remaining.append(interaction)
        context.world_spec.interactions = remaining

        context.publish(
            ChangeEventType.ENTITY_DELETED,
            entity_ids=(self.entity_id,),
            payload={"cascaded_interactions": len(self._removed_interactions)},
        )

    def _undo(self, context: CommandContext) -> None:
        if self._removed_entity is None:
            raise CommandUndoError(self.description, cause=RuntimeError("no captured entity to restore"))
        insert_at = min(self._removed_index, len(context.world_spec.entities))
        context.world_spec.entities.insert(insert_at, self._removed_entity)
        context.index.on_entity_added(self._removed_entity)

        for idx, interaction in sorted(self._removed_interactions, key=lambda pair: pair[0]):
            insert_pos = min(idx, len(context.world_spec.interactions))
            context.world_spec.interactions.insert(insert_pos, interaction)

        context.publish(ChangeEventType.ENTITY_ADDED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


class DuplicateEntityCommand(AbstractCommand):
    """Deep-copies an entity under a new id, optionally offsetting its position.

    Redo-safety: the new entity's id is generated once (at construction or
    on first `_do`) and cached, so `_undo` (which removes that id) followed
    by another `_do` (which re-inserts a fresh deep copy under the SAME
    cached id) reproduces identical state.
    """

    def __init__(self, source_entity_id: str, new_entity_id: Optional[str] = None, position_offset: Optional[Vec3] = None) -> None:
        super().__init__()
        self.source_entity_id = source_entity_id
        self._new_entity_id = new_entity_id
        self._position_offset = position_offset or Vec3(0.0, 0.0, 0.0)
        self._created_entity: Optional[Entity] = None

    @property
    def description(self) -> str:
        target = self._new_entity_id or "<auto>"
        return f"Duplicate entity '{self.source_entity_id}' -> '{target}'"

    @property
    def new_entity_id(self) -> Optional[str]:
        """The id assigned to the duplicate, available after `execute()`."""
        return self._created_entity.id if self._created_entity is not None else self._new_entity_id

    def _do(self, context: CommandContext) -> None:
        source = context.resolve_entity(self.source_entity_id, require_unlocked=False)
        if self._new_entity_id is None:
            self._new_entity_id = f"{source.id}_copy_{uuid.uuid4().hex[:8]}"
        if context.world_spec.get_entity(self._new_entity_id) is not None:
            raise DuplicateEntityIdError(self._new_entity_id)

        clone = copy.deepcopy(source)
        clone.id = self._new_entity_id
        clone.state.position = Vec3(
            source.state.position.x + self._position_offset.x,
            source.state.position.y + self._position_offset.y,
            source.state.position.z + self._position_offset.z,
        )
        context.world_spec.entities.append(clone)
        context.index.on_entity_added(clone)
        self._created_entity = clone
        context.publish(ChangeEventType.ENTITY_DUPLICATED, entity_ids=(clone.id,), payload={"source": self.source_entity_id})

    def _undo(self, context: CommandContext) -> None:
        if self._created_entity is None:
            raise CommandUndoError(self.description, cause=RuntimeError("no duplicate to remove"))
        entity = context.world_spec.get_entity(self._created_entity.id)
        if entity is not None:
            context.world_spec.entities.remove(entity)
            context.index.on_entity_removed(entity)
        context.publish(ChangeEventType.ENTITY_DELETED, entity_ids=(self._created_entity.id,))

    def _serialize_payload(self) -> dict:
        return {"source_entity_id": self.source_entity_id, "new_entity_id": self._new_entity_id}


class RenameEntityCommand(_EntityFieldCommand):
    """Changes an entity's human-readable label."""

    def __init__(self, entity_id: str, new_label: str) -> None:
        def _validate(v: str) -> Optional[str]:
            return "label must be a non-empty string" if not v or not v.strip() else None

        super().__init__(
            entity_id,
            new_label,
            field_label="label",
            event_type=ChangeEventType.ENTITY_RENAMED,
            getter=lambda e: e.label,
            setter=lambda e, v: setattr(e, "label", v),
            validator=_validate,
        )


# ════════════════════════════════════════════════════════════════════════
# Entity state commands (enable/disable, show/hide, lock/unlock)
# ════════════════════════════════════════════════════════════════════════

class _TagToggleCommand(AbstractCommand):
    """Generic "add/remove a reserved boolean marker tag" command.

    Redo-safety: `_do` sets tag membership to the target state
    unconditionally (idempotent), so re-running it after `_undo` (which
    restores the opposite state) reproduces the same result.
    """

    def __init__(self, entity_id: str, tag: str, want_present: bool, *, on_event: ChangeEventType, off_event: ChangeEventType) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._tag = tag
        self._want_present = want_present
        self._on_event = on_event
        self._off_event = off_event
        self._was_present: Optional[bool] = None

    @property
    def description(self) -> str:
        return f"Set tag '{self._tag}' presence={self._want_present} on entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._was_present = self._tag in entity.tags
        if self._want_present and not self._was_present:
            entity.tags.append(self._tag)
        elif not self._want_present and self._was_present:
            entity.tags.remove(self._tag)
        context.index.on_entity_reindexed(entity)
        context.publish(
            self._on_event if self._want_present else self._off_event,
            entity_ids=(self.entity_id,),
        )

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        currently_present = self._tag in entity.tags
        if self._was_present and not currently_present:
            entity.tags.append(self._tag)
        elif not self._was_present and currently_present:
            entity.tags.remove(self._tag)
        context.index.on_entity_reindexed(entity)
        context.publish(
            self._off_event if self._want_present else self._on_event,
            entity_ids=(self.entity_id,),
        )

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "tag": self._tag, "want_present": self._want_present}


class SetEntityEnabledCommand(_TagToggleCommand):
    """Enables an entity (removes the `__disabled__` marker tag)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(entity_id, DISABLED_TAG, want_present=False, on_event=ChangeEventType.ENTITY_ENABLED, off_event=ChangeEventType.ENTITY_DISABLED)


class SetEntityDisabledCommand(_TagToggleCommand):
    """Disables an entity (adds the `__disabled__` marker tag)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(entity_id, DISABLED_TAG, want_present=True, on_event=ChangeEventType.ENTITY_DISABLED, off_event=ChangeEventType.ENTITY_ENABLED)


class ShowEntityCommand(_TagToggleCommand):
    """Makes an entity visible (removes the `__hidden__` marker tag)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(entity_id, HIDDEN_TAG, want_present=False, on_event=ChangeEventType.ENTITY_SHOWN, off_event=ChangeEventType.ENTITY_HIDDEN)


class HideEntityCommand(_TagToggleCommand):
    """Hides an entity (adds the `__hidden__` marker tag)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(entity_id, HIDDEN_TAG, want_present=True, on_event=ChangeEventType.ENTITY_HIDDEN, off_event=ChangeEventType.ENTITY_SHOWN)


class LockEntityCommand(AbstractCommand):
    """Locks an entity against further mutation (index-only state; not persisted in WorldSpec).

    Redo-safety: `_do` unconditionally marks the entity locked.
    """

    def __init__(self, entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id

    @property
    def description(self) -> str:
        return f"Lock entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        if context.world_spec.get_entity(self.entity_id) is None:
            raise EntityNotFoundError(self.entity_id)
        context.index.lock(self.entity_id)
        context.publish(ChangeEventType.ENTITY_LOCKED, entity_ids=(self.entity_id,))

    def _undo(self, context: CommandContext) -> None:
        context.index.unlock(self.entity_id)
        context.publish(ChangeEventType.ENTITY_UNLOCKED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


class UnlockEntityCommand(AbstractCommand):
    """Unlocks a previously locked entity."""

    def __init__(self, entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id

    @property
    def description(self) -> str:
        return f"Unlock entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        if context.world_spec.get_entity(self.entity_id) is None:
            raise EntityNotFoundError(self.entity_id)
        context.index.unlock(self.entity_id)
        context.publish(ChangeEventType.ENTITY_UNLOCKED, entity_ids=(self.entity_id,))

    def _undo(self, context: CommandContext) -> None:
        context.index.lock(self.entity_id)
        context.publish(ChangeEventType.ENTITY_LOCKED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


# ════════════════════════════════════════════════════════════════════════
# Grouping / hierarchy commands
# ════════════════════════════════════════════════════════════════════════

class GroupEntitiesCommand(AbstractCommand):
    """Tags a set of entities as members of a named group.

    Redo-safety: `_do` adds the group tag to every entity that does not
    already have it; idempotent by construction.
    """

    def __init__(self, entity_ids: list[str], group_name: str) -> None:
        super().__init__()
        if not group_name or not group_name.strip():
            raise InvalidParameterError("group_name", group_name, "group_name must be non-empty")
        self.entity_ids = list(entity_ids)
        self.group_name = group_name
        self._tag = f"{GROUP_TAG_PREFIX}{group_name}"
        self._added_to: list[str] = []

    @property
    def description(self) -> str:
        return f"Group {len(self.entity_ids)} entit(y/ies) into '{self.group_name}'"

    def _do(self, context: CommandContext) -> None:
        self._added_to = []
        for eid in self.entity_ids:
            entity = context.resolve_entity(eid)
            if self._tag not in entity.tags:
                entity.tags.append(self._tag)
                context.index.on_entity_reindexed(entity)
                self._added_to.append(eid)
        context.publish(ChangeEventType.ENTITY_GROUPED, entity_ids=tuple(self.entity_ids), payload={"group": self.group_name})

    def _undo(self, context: CommandContext) -> None:
        for eid in self._added_to:
            entity = context.world_spec.get_entity(eid)
            if entity is not None and self._tag in entity.tags:
                entity.tags.remove(self._tag)
                context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_UNGROUPED, entity_ids=tuple(self.entity_ids), payload={"group": self.group_name})

    def _serialize_payload(self) -> dict:
        return {"entity_ids": self.entity_ids, "group_name": self.group_name}


class UngroupEntitiesCommand(AbstractCommand):
    """Removes a set of entities from a named group.

    Redo-safety: `_do` removes the group tag from every entity that
    currently has it; idempotent by construction.
    """

    def __init__(self, entity_ids: list[str], group_name: str) -> None:
        super().__init__()
        self.entity_ids = list(entity_ids)
        self.group_name = group_name
        self._tag = f"{GROUP_TAG_PREFIX}{group_name}"
        self._removed_from: list[str] = []

    @property
    def description(self) -> str:
        return f"Ungroup {len(self.entity_ids)} entit(y/ies) from '{self.group_name}'"

    def _do(self, context: CommandContext) -> None:
        self._removed_from = []
        for eid in self.entity_ids:
            entity = context.resolve_entity(eid)
            if self._tag in entity.tags:
                entity.tags.remove(self._tag)
                context.index.on_entity_reindexed(entity)
                self._removed_from.append(eid)
        context.publish(ChangeEventType.ENTITY_UNGROUPED, entity_ids=tuple(self.entity_ids), payload={"group": self.group_name})

    def _undo(self, context: CommandContext) -> None:
        for eid in self._removed_from:
            entity = context.world_spec.get_entity(eid)
            if entity is not None and self._tag not in entity.tags:
                entity.tags.append(self._tag)
                context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_GROUPED, entity_ids=tuple(self.entity_ids), payload={"group": self.group_name})

    def _serialize_payload(self) -> dict:
        return {"entity_ids": self.entity_ids, "group_name": self.group_name}


class ParentEntityCommand(AbstractCommand):
    """Sets an entity's parent, replacing any previous parent tag.

    Redo-safety: `_do` strips any existing `__parent__:*` tag before
    adding the new one, so re-running after `_undo` (which restores the
    original parent tag, if any) reproduces the same end state.
    """

    def __init__(self, entity_id: str, parent_id: str) -> None:
        super().__init__()
        if entity_id == parent_id:
            raise InvalidParameterError("parent_id", parent_id, "an entity cannot be its own parent")
        self.entity_id = entity_id
        self.parent_id = parent_id
        self._previous_parent_tag: Optional[str] = None

    @property
    def description(self) -> str:
        return f"Parent entity '{self.entity_id}' under '{self.parent_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        if context.world_spec.get_entity(self.parent_id) is None:
            raise EntityNotFoundError(self.parent_id)
        self._previous_parent_tag = next((t for t in entity.tags if t.startswith(PARENT_TAG_PREFIX)), None)
        if self._previous_parent_tag is not None:
            entity.tags.remove(self._previous_parent_tag)
        entity.tags.append(f"{PARENT_TAG_PREFIX}{self.parent_id}")
        context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_PARENTED, entity_ids=(self.entity_id,), payload={"parent_id": self.parent_id})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        current_tag = f"{PARENT_TAG_PREFIX}{self.parent_id}"
        if current_tag in entity.tags:
            entity.tags.remove(current_tag)
        if self._previous_parent_tag is not None:
            entity.tags.append(self._previous_parent_tag)
        context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_UNPARENTED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "parent_id": self.parent_id}


class UnparentEntityCommand(AbstractCommand):
    """Removes an entity's parent tag, if any.

    Redo-safety: `_do` removes any `__parent__:*` tag unconditionally
    (idempotent no-op if none present).
    """

    def __init__(self, entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._removed_tag: Optional[str] = None

    @property
    def description(self) -> str:
        return f"Unparent entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._removed_tag = next((t for t in entity.tags if t.startswith(PARENT_TAG_PREFIX)), None)
        if self._removed_tag is not None:
            entity.tags.remove(self._removed_tag)
            context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_UNPARENTED, entity_ids=(self.entity_id,))

    def _undo(self, context: CommandContext) -> None:
        if self._removed_tag is None:
            return
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.tags.append(self._removed_tag)
        context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.ENTITY_PARENTED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


# ════════════════════════════════════════════════════════════════════════
# Transform commands
# ════════════════════════════════════════════════════════════════════════

def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)


class MoveEntityCommand(AbstractCommand):
    """Moves an entity, either to an absolute position or by a relative delta.

    Redo-safety: `_do` recomputes the target from the *current* position
    every time (for relative moves) or uses the fixed absolute target (for
    absolute moves); either way it re-captures `_old_position` from
    whatever is current, so a second `_do` after `_undo` reproduces the
    original transition.
    """

    def __init__(self, entity_id: str, vector: Vec3, relative: bool = True) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.vector = vector
        self.relative = relative
        self._old_position: Optional[Vec3] = None

    @property
    def description(self) -> str:
        mode = "by" if self.relative else "to"
        return f"Move entity '{self.entity_id}' {mode} ({self.vector.x}, {self.vector.y}, {self.vector.z})"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_position = entity.state.position
        entity.state.position = (
            _add_vec3(self._old_position, self.vector) if self.relative else Vec3(self.vector.x, self.vector.y, self.vector.z)
        )
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"position": entity.state.position.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state.position = self._old_position
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"position": entity.state.position.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "vector": self.vector.to_dict(), "relative": self.relative}


class RotateEntityCommand(AbstractCommand):
    """Rotates an entity's Euler orientation, absolute or relative (radians)."""

    def __init__(self, entity_id: str, vector: Vec3, relative: bool = True) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.vector = vector
        self.relative = relative
        self._old_orientation: Optional[Vec3] = None

    @property
    def description(self) -> str:
        mode = "by" if self.relative else "to"
        return f"Rotate entity '{self.entity_id}' {mode} ({self.vector.x}, {self.vector.y}, {self.vector.z}) rad"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_orientation = entity.state.orientation
        entity.state.orientation = (
            _add_vec3(self._old_orientation, self.vector) if self.relative else Vec3(self.vector.x, self.vector.y, self.vector.z)
        )
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"orientation": entity.state.orientation.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state.orientation = self._old_orientation
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"orientation": entity.state.orientation.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "vector": self.vector.to_dict(), "relative": self.relative}


class ScaleEntityCommand(AbstractCommand):
    """Scales an entity's bounding box, either to absolute dimensions or by a relative factor."""

    def __init__(self, entity_id: str, scale: BoundingBox, relative: bool = True) -> None:
        super().__init__()
        if scale.width <= 0 or scale.height <= 0 or scale.depth <= 0:
            reason = "relative scale factors must be > 0" if relative else "absolute dimensions must be > 0"
            raise InvalidParameterError("scale", scale, reason)
        self.entity_id = entity_id
        self.scale = scale
        self.relative = relative
        self._old_bbox: Optional[BoundingBox] = None

    @property
    def description(self) -> str:
        mode = "by factor" if self.relative else "to"
        return f"Scale entity '{self.entity_id}' {mode} ({self.scale.width}, {self.scale.height}, {self.scale.depth})"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_bbox = entity.bounding_box
        if self.relative:
            entity.bounding_box = BoundingBox(
                width=self._old_bbox.width * self.scale.width,
                height=self._old_bbox.height * self.scale.height,
                depth=self._old_bbox.depth * self.scale.depth,
            )
        else:
            entity.bounding_box = BoundingBox(width=self.scale.width, height=self.scale.height, depth=self.scale.depth)
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"bounding_box": entity.bounding_box.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.bounding_box = self._old_bbox
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"bounding_box": entity.bounding_box.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "scale": self.scale.to_dict(), "relative": self.relative}


class MirrorEntityCommand(AbstractCommand):
    """Mirrors an entity's position and orientation across a world axis through the origin."""

    _AXES = ("x", "y", "z")

    def __init__(self, entity_id: str, axis: str) -> None:
        super().__init__()
        if axis not in self._AXES:
            raise InvalidParameterError("axis", axis, f"axis must be one of {self._AXES}")
        self.entity_id = entity_id
        self.axis = axis
        self._old_position: Optional[Vec3] = None
        self._old_orientation: Optional[Vec3] = None

    @property
    def description(self) -> str:
        return f"Mirror entity '{self.entity_id}' across the {self.axis}-axis"

    def _mirror_vec(self, v: Vec3) -> Vec3:
        x, y, z = v.x, v.y, v.z
        if self.axis == "x":
            x = -x
        elif self.axis == "y":
            y = -y
        else:
            z = -z
        return Vec3(x, y, z)

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_position = entity.state.position
        self._old_orientation = entity.state.orientation
        entity.state.position = self._mirror_vec(self._old_position)
        entity.state.orientation = self._mirror_vec(self._old_orientation)
        context.publish(
            ChangeEventType.TRANSFORM_CHANGED,
            entity_ids=(self.entity_id,),
            payload={"position": entity.state.position.to_dict(), "orientation": entity.state.orientation.to_dict()},
        )

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state.position = self._old_position
        entity.state.orientation = self._old_orientation
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "axis": self.axis}


class AlignEntitiesCommand(AbstractCommand):
    """Aligns multiple entities along one axis to the min, max, or centroid of the group.

    Redo-safety: `_do` recomputes the alignment target from the entities'
    *current* positions each time it runs; after `_undo` restores every
    entity's original coordinate, a second `_do` recomputes the identical
    target and reproduces the identical result.
    """

    _AXES = ("x", "y", "z")
    _MODES = ("min", "max", "center")

    def __init__(self, entity_ids: list[str], axis: str, mode: str = "center") -> None:
        super().__init__()
        if axis not in self._AXES:
            raise InvalidParameterError("axis", axis, f"axis must be one of {self._AXES}")
        if mode not in self._MODES:
            raise InvalidParameterError("mode", mode, f"mode must be one of {self._MODES}")
        if len(entity_ids) < 2:
            raise InvalidParameterError("entity_ids", entity_ids, "alignment requires at least 2 entities")
        self.entity_ids = list(entity_ids)
        self.axis = axis
        self.mode = mode
        self._old_values: dict[str, float] = {}

    @property
    def description(self) -> str:
        return f"Align {len(self.entity_ids)} entities on {self.axis}-axis ({self.mode})"

    def _coord(self, pos: Vec3) -> float:
        return getattr(pos, self.axis)

    def _set_coord(self, pos: Vec3, value: float) -> Vec3:
        kwargs = {"x": pos.x, "y": pos.y, "z": pos.z}
        kwargs[self.axis] = value
        return Vec3(**kwargs)

    def _do(self, context: CommandContext) -> None:
        entities = [context.resolve_entity(eid) for eid in self.entity_ids]
        values = [self._coord(e.state.position) for e in entities]
        if self.mode == "min":
            target = min(values)
        elif self.mode == "max":
            target = max(values)
        else:
            target = sum(values) / len(values)

        self._old_values = {}
        for entity in entities:
            self._old_values[entity.id] = self._coord(entity.state.position)
            entity.state.position = self._set_coord(entity.state.position, target)

        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=tuple(self.entity_ids), payload={"axis": self.axis, "target": target})

    def _undo(self, context: CommandContext) -> None:
        for eid, old_value in self._old_values.items():
            entity = context.world_spec.get_entity(eid)
            if entity is None:
                raise EntityNotFoundError(eid)
            entity.state.position = self._set_coord(entity.state.position, old_value)
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=tuple(self.entity_ids))

    def _serialize_payload(self) -> dict:
        return {"entity_ids": self.entity_ids, "axis": self.axis, "mode": self.mode}


class SnapEntityCommand(AbstractCommand):
    """Snaps an entity's position to the nearest point on a uniform grid."""

    def __init__(self, entity_id: str, grid_size: float) -> None:
        super().__init__()
        if grid_size <= 0:
            raise InvalidParameterError("grid_size", grid_size, "grid_size must be > 0")
        self.entity_id = entity_id
        self.grid_size = grid_size
        self._old_position: Optional[Vec3] = None

    @property
    def description(self) -> str:
        return f"Snap entity '{self.entity_id}' to {self.grid_size}m grid"

    @staticmethod
    def _snap(value: float, grid: float) -> float:
        return round(value / grid) * grid

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_position = entity.state.position
        entity.state.position = Vec3(
            self._snap(self._old_position.x, self.grid_size),
            self._snap(self._old_position.y, self.grid_size),
            self._snap(self._old_position.z, self.grid_size),
        )
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,), payload={"position": entity.state.position.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state.position = self._old_position
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "grid_size": self.grid_size}


class ResetTransformCommand(AbstractCommand):
    """Resets an entity's position, orientation, velocity, and angular velocity to zero."""

    def __init__(self, entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._old_state: Optional[PhysicsState] = None

    @property
    def description(self) -> str:
        return f"Reset transform of entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_state = copy.deepcopy(entity.state)
        entity.state.position = Vec3()
        entity.state.orientation = Vec3()
        entity.state.velocity = Vec3()
        entity.state.angular_vel = Vec3()
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state = copy.deepcopy(self._old_state)
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


class PasteTransformCommand(AbstractCommand):
    """Applies a previously copied `PhysicsState` transform onto a target entity.

    Pairs with `WorldController.copy_transform()`, which is a read-only
    clipboard operation and therefore not itself a `Command` (nothing
    about the WorldSpec changes when a transform is copied).
    """

    def __init__(self, entity_id: str, source_state: PhysicsState) -> None:
        super().__init__()
        self.entity_id = entity_id
        self._source_state = copy.deepcopy(source_state)
        self._old_state: Optional[PhysicsState] = None

    @property
    def description(self) -> str:
        return f"Paste transform onto entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        self._old_state = copy.deepcopy(entity.state)
        entity.state = copy.deepcopy(self._source_state)
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.state = copy.deepcopy(self._old_state)
        context.publish(ChangeEventType.TRANSFORM_CHANGED, entity_ids=(self.entity_id,))

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id}


# ════════════════════════════════════════════════════════════════════════
# Physics commands
# ════════════════════════════════════════════════════════════════════════

def _positive(label: str):
    def _v(value: float) -> Optional[str]:
        return f"{label} must be > 0" if value <= 0 else None
    return _v


def _unit_interval(label: str):
    def _v(value: float) -> Optional[str]:
        return f"{label} must be within [0, 1]" if not (0.0 <= value <= 1.0) else None
    return _v


class SetMassCommand(_EntityFieldCommand):
    """Sets an entity's mass (kg). Dynamic entities must have mass > 0."""

    def __init__(self, entity_id: str, mass_kg: float) -> None:
        super().__init__(
            entity_id, mass_kg,
            field_label="mass", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.mass, setter=lambda e, v: setattr(e, "mass", v),
            validator=_positive("mass"),
        )


class SetFrictionCommand(_EntityFieldCommand):
    """Sets an entity's kinetic friction coefficient."""

    def __init__(self, entity_id: str, friction: float) -> None:
        super().__init__(
            entity_id, friction,
            field_label="friction", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.friction, setter=lambda e, v: setattr(e, "friction", v),
            validator=_unit_interval("friction"),
        )


class SetRestitutionCommand(_EntityFieldCommand):
    """Sets an entity's coefficient of restitution."""

    def __init__(self, entity_id: str, restitution: float) -> None:
        super().__init__(
            entity_id, restitution,
            field_label="restitution", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.restitution, setter=lambda e, v: setattr(e, "restitution", v),
            validator=_unit_interval("restitution"),
        )


class SetVelocityCommand(_EntityFieldCommand):
    """Sets an entity's linear velocity vector (m/s)."""

    def __init__(self, entity_id: str, velocity: Vec3) -> None:
        super().__init__(
            entity_id, velocity,
            field_label="velocity", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.state.velocity, setter=lambda e, v: setattr(e.state, "velocity", v),
        )


class SetAccelerationCommand(_EntityFieldCommand):
    """Sets an entity's linear acceleration vector (m/s²)."""

    def __init__(self, entity_id: str, acceleration: Vec3) -> None:
        super().__init__(
            entity_id, acceleration,
            field_label="acceleration", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.state.acceleration, setter=lambda e, v: setattr(e.state, "acceleration", v),
        )


class SetAngularVelocityCommand(_EntityFieldCommand):
    """Sets an entity's angular velocity vector (rad/s)."""

    def __init__(self, entity_id: str, angular_velocity: Vec3) -> None:
        super().__init__(
            entity_id, angular_velocity,
            field_label="angular_vel", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.state.angular_vel, setter=lambda e, v: setattr(e.state, "angular_vel", v),
        )


class SetStaticDynamicCommand(_EntityFieldCommand):
    """Toggles an entity between static (immovable) and dynamic."""

    def __init__(self, entity_id: str, is_static: bool) -> None:
        super().__init__(
            entity_id, is_static,
            field_label="is_static", event_type=ChangeEventType.PHYSICS_CHANGED,
            getter=lambda e: e.is_static, setter=lambda e, v: setattr(e, "is_static", v),
        )


class SetBodyModeCommand(_TagToggleCommand):
    """Marks an entity's simulation body mode as KINEMATIC or SENSOR via a reserved tag.

    STATIC/DYNAMIC are governed by `Entity.is_static` directly (see
    `SetStaticDynamicCommand`); KINEMATIC and SENSOR are additional
    authored states layered on top via a tag, since `WorldSpec.Entity`
    only models a static/dynamic boolean.
    """

    def __init__(self, entity_id: str, mode: EntityBodyMode) -> None:
        if mode not in (EntityBodyMode.KINEMATIC, EntityBodyMode.SENSOR):
            raise InvalidParameterError("mode", mode, "SetBodyModeCommand only supports KINEMATIC or SENSOR")
        self.mode = mode
        super().__init__(
            entity_id, f"{BODY_MODE_TAG_PREFIX}{mode.value}", want_present=True,
            on_event=ChangeEventType.PHYSICS_CHANGED, off_event=ChangeEventType.PHYSICS_CHANGED,
        )


class AddForceCommand(AbstractCommand):
    """Appends an applied-force descriptor to an entity's force list.

    Redo-safety: `_do` appends `self.force` and remembers its resulting
    list index; `_undo` pops that exact index. A second `_do` after
    `_undo` appends the same dict to the end again.
    """

    def __init__(self, entity_id: str, force: dict) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.force = dict(force)
        self._inserted_index: Optional[int] = None

    @property
    def description(self) -> str:
        return f"Add force {self.force.get('label', '<unnamed>')} to entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        entity.forces.append(dict(self.force))
        self._inserted_index = len(entity.forces) - 1
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"force_added": self.force})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        if self._inserted_index is not None and 0 <= self._inserted_index < len(entity.forces):
            entity.forces.pop(self._inserted_index)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"force_removed": self.force})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "force": self.force}


class RemoveForceCommand(AbstractCommand):
    """Removes the force at a given index from an entity's force list.

    Redo-safety: `_do` re-validates the index against the current list
    length and captures the value being removed at that index every time,
    so it is correct whether this is the first execution or a redo.
    """

    def __init__(self, entity_id: str, force_index: int) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.force_index = force_index
        self._removed_force: Optional[dict] = None

    @property
    def description(self) -> str:
        return f"Remove force[{self.force_index}] from entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        if not (0 <= self.force_index < len(entity.forces)):
            raise InvalidParameterError("force_index", self.force_index, "index out of range")
        self._removed_force = entity.forces.pop(self.force_index)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"force_removed": self._removed_force})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        insert_at = min(self.force_index, len(entity.forces))
        entity.forces.insert(insert_at, self._removed_force)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"force_restored": self._removed_force})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "force_index": self.force_index}


class AddConstraintCommand(AbstractCommand):
    """Adds a constraint reference (another entity's id) to an entity's constraint list."""

    def __init__(self, entity_id: str, constraint_entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.constraint_entity_id = constraint_entity_id

    @property
    def description(self) -> str:
        return f"Add constraint '{self.constraint_entity_id}' to entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        if context.world_spec.get_entity(self.constraint_entity_id) is None:
            raise EntityNotFoundError(self.constraint_entity_id)
        if self.constraint_entity_id not in entity.constraints:
            entity.constraints.append(self.constraint_entity_id)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"constraint_added": self.constraint_entity_id})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        if self.constraint_entity_id in entity.constraints:
            entity.constraints.remove(self.constraint_entity_id)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"constraint_removed": self.constraint_entity_id})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "constraint_entity_id": self.constraint_entity_id}


class RemoveConstraintCommand(AbstractCommand):
    """Removes a constraint reference from an entity's constraint list."""

    def __init__(self, entity_id: str, constraint_entity_id: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.constraint_entity_id = constraint_entity_id

    @property
    def description(self) -> str:
        return f"Remove constraint '{self.constraint_entity_id}' from entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        if self.constraint_entity_id in entity.constraints:
            entity.constraints.remove(self.constraint_entity_id)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"constraint_removed": self.constraint_entity_id})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        if self.constraint_entity_id not in entity.constraints:
            entity.constraints.append(self.constraint_entity_id)
        context.publish(ChangeEventType.PHYSICS_CHANGED, entity_ids=(self.entity_id,), payload={"constraint_added": self.constraint_entity_id})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "constraint_entity_id": self.constraint_entity_id}


# ════════════════════════════════════════════════════════════════════════
# Material commands
# ════════════════════════════════════════════════════════════════════════
# Custom materials are persisted under WorldSpec.metadata["custom_materials"]
# (keyed by name -> {density, restitution, friction}) so they survive
# save/load without altering the world_spec.py data contract. Lookups
# consult custom materials first, then fall back to MATERIAL_DEFAULTS.

_CUSTOM_MATERIALS_KEY = "custom_materials"


def _material_table(world_spec: WorldSpec) -> dict:
    return world_spec.metadata.setdefault(_CUSTOM_MATERIALS_KEY, {})


def resolve_material(world_spec: WorldSpec, name: str) -> dict:
    """Look up a material's physical defaults, custom materials taking priority."""
    from world_spec import MATERIAL_DEFAULTS

    custom = _material_table(world_spec)
    if name in custom:
        return custom[name]
    if name in MATERIAL_DEFAULTS:
        return MATERIAL_DEFAULTS[name]
    raise MaterialNotFoundError(name)


class AssignMaterialCommand(AbstractCommand):
    """Assigns a material to an entity, resetting its restitution/friction to that material's defaults.

    Redo-safety: `_do` re-reads the entity's current material/restitution/
    friction each time, so a second `_do` after `_undo` restores the same
    transition.
    """

    def __init__(self, entity_id: str, material_name: str) -> None:
        super().__init__()
        self.entity_id = entity_id
        self.material_name = material_name
        self._old_material: Optional[str] = None
        self._old_restitution: Optional[float] = None
        self._old_friction: Optional[float] = None

    @property
    def description(self) -> str:
        return f"Assign material '{self.material_name}' to entity '{self.entity_id}'"

    def _do(self, context: CommandContext) -> None:
        entity = context.resolve_entity(self.entity_id)
        defaults = resolve_material(context.world_spec, self.material_name)
        self._old_material = entity.material
        self._old_restitution = entity.restitution
        self._old_friction = entity.friction
        entity.material = self.material_name
        entity.restitution = defaults["restitution"]
        entity.friction = defaults["friction"]
        context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.MATERIAL_CHANGED, entity_ids=(self.entity_id,), payload={"material": self.material_name})

    def _undo(self, context: CommandContext) -> None:
        entity = context.world_spec.get_entity(self.entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_id)
        entity.material = self._old_material
        entity.restitution = self._old_restitution
        entity.friction = self._old_friction
        context.index.on_entity_reindexed(entity)
        context.publish(ChangeEventType.MATERIAL_CHANGED, entity_ids=(self.entity_id,), payload={"material": self._old_material})

    def _serialize_payload(self) -> dict:
        return {"entity_id": self.entity_id, "material_name": self.material_name}


class CreateMaterialCommand(AbstractCommand):
    """Registers a new named material (density kg/m³, restitution, friction) in WorldSpec metadata."""

    def __init__(self, name: str, density: float, restitution: float, friction: float) -> None:
        super().__init__()
        if density <= 0:
            raise InvalidParameterError("density", density, "density must be > 0")
        if not (0.0 <= restitution <= 1.0):
            raise InvalidParameterError("restitution", restitution, "must be within [0, 1]")
        if not (0.0 <= friction <= 1.0):
            raise InvalidParameterError("friction", friction, "must be within [0, 1]")
        self.name = name
        self.density = density
        self.restitution = restitution
        self.friction = friction

    @property
    def description(self) -> str:
        return f"Create material '{self.name}'"

    def _do(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        if self.name in table:
            raise InvalidParameterError("name", self.name, "a custom material with this name already exists")
        table[self.name] = {"density": self.density, "restitution": self.restitution, "friction": self.friction}
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_created": self.name})

    def _undo(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        table.pop(self.name, None)
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_deleted": self.name})

    def _serialize_payload(self) -> dict:
        return {"name": self.name, "density": self.density, "restitution": self.restitution, "friction": self.friction}


class DeleteMaterialCommand(AbstractCommand):
    """Removes a custom material definition (built-in `MATERIAL_DEFAULTS` entries cannot be deleted)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._removed_definition: Optional[dict] = None

    @property
    def description(self) -> str:
        return f"Delete material '{self.name}'"

    def _do(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        if self.name not in table:
            raise MaterialNotFoundError(self.name)
        self._removed_definition = table.pop(self.name)
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_deleted": self.name})

    def _undo(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        if self._removed_definition is not None:
            table[self.name] = self._removed_definition
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_created": self.name})

    def _serialize_payload(self) -> dict:
        return {"name": self.name}


class UpdateMaterialCommand(AbstractCommand):
    """Updates one or more fields (density/restitution/friction) of an existing custom material."""

    def __init__(self, name: str, **fields: float) -> None:
        super().__init__()
        allowed = {"density", "restitution", "friction"}
        unknown = set(fields) - allowed
        if unknown:
            raise InvalidParameterError("fields", fields, f"unknown material field(s): {sorted(unknown)}")
        self.name = name
        self.fields = dict(fields)
        self._old_values: dict = {}

    @property
    def description(self) -> str:
        return f"Update material '{self.name}' fields={list(self.fields)}"

    def _do(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        if self.name not in table:
            raise MaterialNotFoundError(self.name)
        definition = table[self.name]
        self._old_values = {k: definition[k] for k in self.fields}
        definition.update(self.fields)
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_updated": self.name, "fields": self.fields})

    def _undo(self, context: CommandContext) -> None:
        table = _material_table(context.world_spec)
        if self.name in table:
            table[self.name].update(self._old_values)
        context.publish(ChangeEventType.MATERIAL_CHANGED, payload={"material_updated": self.name, "fields": self._old_values})

    def _serialize_payload(self) -> dict:
        return {"name": self.name, "fields": self.fields}


# ════════════════════════════════════════════════════════════════════════
# Environment commands
# ════════════════════════════════════════════════════════════════════════

class SetGravityCommand(_EnvironmentFieldCommand):
    """Sets world gravity (m/s²)."""

    def __init__(self, gravity: Vec3) -> None:
        super().__init__(gravity, field_label="gravity", getter=lambda env: env.gravity, setter=lambda env, v: setattr(env, "gravity", v))


class SetWeatherCommand(_EnvironmentFieldCommand):
    """Sets the world's weather state."""

    _VALID = {"clear", "rain", "snow", "fog", "wind"}

    def __init__(self, weather: str) -> None:
        def _v(value: str) -> Optional[str]:
            return f"weather must be one of {sorted(self._VALID)}" if value not in self._VALID else None
        super().__init__(weather, field_label="weather", getter=lambda env: env.weather, setter=lambda env, v: setattr(env, "weather", v), validator=_v)


class SetWindCommand(_EnvironmentFieldCommand):
    """Sets wind speed (m/s) and direction (rad from north)."""

    def __init__(self, speed_ms: float, direction_rad: float) -> None:
        if speed_ms < 0:
            raise InvalidParameterError("speed_ms", speed_ms, "speed_ms must be >= 0")
        super().__init__(
            Wind(speed=speed_ms, direction=direction_rad),
            field_label="wind", getter=lambda env: env.wind, setter=lambda env, v: setattr(env, "wind", v),
        )


class SetTemperatureCommand(_EnvironmentFieldCommand):
    """Sets ambient temperature (Kelvin)."""

    def __init__(self, temperature_k: float) -> None:
        super().__init__(
            temperature_k, field_label="temperature_K",
            getter=lambda env: env.temperature_K, setter=lambda env, v: setattr(env, "temperature_K", v),
            validator=lambda v: "temperature_K must be > 0 (absolute zero bound)" if v <= 0 else None,
        )


class SetPressureCommand(_EnvironmentFieldCommand):
    """Sets ambient pressure (Pa)."""

    def __init__(self, pressure_pa: float) -> None:
        super().__init__(
            pressure_pa, field_label="pressure_Pa",
            getter=lambda env: env.pressure_Pa, setter=lambda env, v: setattr(env, "pressure_Pa", v),
            validator=_positive("pressure_Pa"),
        )


class SetLightingCommand(_EnvironmentFieldCommand):
    """Sets the world's time-of-day (drives Lighting scene-node authoring)."""

    _VALID = {"day", "night", "dawn", "dusk"}

    def __init__(self, time_of_day: str) -> None:
        def _v(value: str) -> Optional[str]:
            return f"time_of_day must be one of {sorted(self._VALID)}" if value not in self._VALID else None
        super().__init__(time_of_day, field_label="time_of_day", getter=lambda env: env.time_of_day, setter=lambda env, v: setattr(env, "time_of_day", v), validator=_v)


class SetTerrainCommand(_EnvironmentFieldCommand):
    """Sets the global terrain type."""

    _VALID = {"flat", "hilly", "urban", "water", "mixed"}

    def __init__(self, terrain_type: str) -> None:
        def _v(value: str) -> Optional[str]:
            return f"terrain_type must be one of {sorted(self._VALID)}" if value not in self._VALID else None
        super().__init__(terrain_type, field_label="terrain_type", getter=lambda env: env.terrain_type, setter=lambda env, v: setattr(env, "terrain_type", v), validator=_v)


class SetGroundMaterialCommand(_EnvironmentFieldCommand):
    """Sets the global ground friction coefficient (WorldSpec's proxy for "ground material" in the absence of a dedicated field)."""

    def __init__(self, friction_global: float) -> None:
        super().__init__(
            friction_global, field_label="friction_global",
            getter=lambda env: env.friction_global, setter=lambda env, v: setattr(env, "friction_global", v),
            validator=_unit_interval("friction_global"),
        )


# ════════════════════════════════════════════════════════════════════════
# Relationship / interaction commands
# ════════════════════════════════════════════════════════════════════════

class AddRelationshipCommand(AbstractCommand):
    """Adds an `Interaction` (collision/joint/contact/fluid_drag/magnetic/...) to the WorldSpec.

    Redo-safety: `_do` appends and records the resulting index; `_undo`
    pops that index; a second `_do` appends again to the end.
    """

    def __init__(self, interaction: Interaction) -> None:
        super().__init__()
        self.interaction = interaction
        self._inserted_index: Optional[int] = None

    @property
    def description(self) -> str:
        return f"Add relationship {self.interaction.type} ({self.interaction.entity_a} -> {self.interaction.entity_b})"

    def _do(self, context: CommandContext) -> None:
        if self.interaction.entity_a not in context.index.all_ids() and context.world_spec.get_entity(self.interaction.entity_a) is None:
            raise EntityNotFoundError(self.interaction.entity_a)
        if self.interaction.entity_b != "environment" and context.world_spec.get_entity(self.interaction.entity_b) is None:
            raise EntityNotFoundError(self.interaction.entity_b)
        context.world_spec.interactions.append(self.interaction)
        self._inserted_index = len(context.world_spec.interactions) - 1
        context.publish(
            ChangeEventType.RELATIONSHIP_CHANGED,
            entity_ids=(self.interaction.entity_a, self.interaction.entity_b),
            payload={"relationship_added": self.interaction.to_dict()},
        )

    def _undo(self, context: CommandContext) -> None:
        if self._inserted_index is not None and 0 <= self._inserted_index < len(context.world_spec.interactions):
            context.world_spec.interactions.pop(self._inserted_index)
        context.publish(ChangeEventType.RELATIONSHIP_CHANGED, payload={"relationship_removed": self.interaction.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"interaction": self.interaction.to_dict()}


class RemoveRelationshipCommand(AbstractCommand):
    """Removes the interaction at a given index from `WorldSpec.interactions`."""

    def __init__(self, interaction_index: int) -> None:
        super().__init__()
        self.interaction_index = interaction_index
        self._removed_interaction: Optional[Interaction] = None

    @property
    def description(self) -> str:
        return f"Remove relationship[{self.interaction_index}]"

    def _do(self, context: CommandContext) -> None:
        if not (0 <= self.interaction_index < len(context.world_spec.interactions)):
            raise RelationshipError(f"interaction index {self.interaction_index} out of range")
        self._removed_interaction = context.world_spec.interactions.pop(self.interaction_index)
        context.publish(ChangeEventType.RELATIONSHIP_CHANGED, payload={"relationship_removed": self._removed_interaction.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        if self._removed_interaction is None:
            raise CommandUndoError(self.description, cause=RuntimeError("no captured interaction to restore"))
        insert_at = min(self.interaction_index, len(context.world_spec.interactions))
        context.world_spec.interactions.insert(insert_at, self._removed_interaction)
        context.publish(ChangeEventType.RELATIONSHIP_CHANGED, payload={"relationship_added": self._removed_interaction.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"interaction_index": self.interaction_index}


class UpdateRelationshipCommand(AbstractCommand):
    """Merges new parameters into the interaction at a given index."""

    def __init__(self, interaction_index: int, parameters: dict) -> None:
        super().__init__()
        self.interaction_index = interaction_index
        self.parameters = dict(parameters)
        self._old_parameters: Optional[dict] = None

    @property
    def description(self) -> str:
        return f"Update relationship[{self.interaction_index}] parameters"

    def _do(self, context: CommandContext) -> None:
        if not (0 <= self.interaction_index < len(context.world_spec.interactions)):
            raise RelationshipError(f"interaction index {self.interaction_index} out of range")
        interaction = context.world_spec.interactions[self.interaction_index]
        self._old_parameters = dict(interaction.parameters)
        interaction.parameters.update(self.parameters)
        context.publish(ChangeEventType.RELATIONSHIP_CHANGED, payload={"relationship_updated": interaction.to_dict()})

    def _undo(self, context: CommandContext) -> None:
        interaction = context.world_spec.interactions[self.interaction_index]
        interaction.parameters = dict(self._old_parameters)
        context.publish(ChangeEventType.RELATIONSHIP_CHANGED, payload={"relationship_updated": interaction.to_dict()})

    def _serialize_payload(self) -> dict:
        return {"interaction_index": self.interaction_index, "parameters": self.parameters}


# ════════════════════════════════════════════════════════════════════════
# Composite / transaction command
# ════════════════════════════════════════════════════════════════════════

class TransactionCommand(AbstractCommand):
    """Groups an ordered sequence of commands so they execute/undo as one unit.

    Used by `history.CommandHistory` to implement transaction grouping:
    `begin_transaction()` / `commit_transaction()` collect individual
    commands and, on commit, push a single `TransactionCommand` onto the
    undo stack instead of N separate entries.

    Redo-safety: executes its children in order on `_do`, and undoes them
    in reverse order on `_undo`; each child is itself required to be
    redo-safe, so the composite is redo-safe by induction.
    """

    def __init__(self, commands: list[Command], label: Optional[str] = None) -> None:
        super().__init__()
        if not commands:
            raise InvalidParameterError("commands", commands, "a transaction must contain at least one command")
        self.commands = list(commands)
        self.label = label or f"Transaction ({len(commands)} commands)"

    @property
    def description(self) -> str:
        return self.label

    def _do(self, context: CommandContext) -> None:
        executed: list[Command] = []
        try:
            for cmd in self.commands:
                cmd.execute(context)
                executed.append(cmd)
        except Exception:
            # Roll back everything that succeeded before the failure, so a
            # partially-applied transaction never leaves the WorldSpec in
            # an inconsistent state.
            for cmd in reversed(executed):
                try:
                    cmd.undo(context)
                except Exception:  # noqa: BLE001 - best-effort rollback
                    pass
            raise
        context.publish(ChangeEventType.TRANSACTION_COMMITTED, payload={"label": self.label, "command_count": len(self.commands)})

    def _undo(self, context: CommandContext) -> None:
        for cmd in reversed(self.commands):
            cmd.undo(context)
        context.publish(ChangeEventType.TRANSACTION_ROLLED_BACK, payload={"label": self.label, "command_count": len(self.commands)})

    def _serialize_payload(self) -> dict:
        return {"label": self.label, "commands": [c.serialize() for c in self.commands]}


__all__ = [
    "CommandContext",
    "Command",
    "AbstractCommand",
    "AddEntityCommand",
    "DeleteEntityCommand",
    "DuplicateEntityCommand",
    "RenameEntityCommand",
    "SetEntityEnabledCommand",
    "SetEntityDisabledCommand",
    "ShowEntityCommand",
    "HideEntityCommand",
    "LockEntityCommand",
    "UnlockEntityCommand",
    "GroupEntitiesCommand",
    "UngroupEntitiesCommand",
    "ParentEntityCommand",
    "UnparentEntityCommand",
    "MoveEntityCommand",
    "RotateEntityCommand",
    "ScaleEntityCommand",
    "MirrorEntityCommand",
    "AlignEntitiesCommand",
    "SnapEntityCommand",
    "ResetTransformCommand",
    "PasteTransformCommand",
    "SetMassCommand",
    "SetFrictionCommand",
    "SetRestitutionCommand",
    "SetVelocityCommand",
    "SetAccelerationCommand",
    "SetAngularVelocityCommand",
    "SetStaticDynamicCommand",
    "SetBodyModeCommand",
    "AddForceCommand",
    "RemoveForceCommand",
    "AddConstraintCommand",
    "RemoveConstraintCommand",
    "AssignMaterialCommand",
    "CreateMaterialCommand",
    "DeleteMaterialCommand",
    "UpdateMaterialCommand",
    "resolve_material",
    "SetGravityCommand",
    "SetWeatherCommand",
    "SetWindCommand",
    "SetTemperatureCommand",
    "SetPressureCommand",
    "SetLightingCommand",
    "SetTerrainCommand",
    "SetGroundMaterialCommand",
    "AddRelationshipCommand",
    "RemoveRelationshipCommand",
    "UpdateRelationshipCommand",
    "TransactionCommand",
    "DISABLED_TAG",
    "HIDDEN_TAG",
]
