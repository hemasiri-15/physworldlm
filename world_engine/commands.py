"""
world_engine.commands
───────────────────────
The Command pattern is the sole mutation path into a live `WorldSpec`.
Every state change is a `Command` object with `execute`/`undo`, giving
`WorldEngine` uniform undo/redo, an auditable history log, and one choke
point (`WorldEngine._execute`) where invariants are enforced before a
mutation lands.

New mutation types (e.g. a future `ReparentEntityCommand` for hierarchical
scene graphs) are added here by subclassing `Command` — no other module
needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from world_spec import WorldSpec, Entity, Interaction, Vec3
from .exceptions import EntityNotFoundError, InteractionNotFoundError, InvalidCommandError


def find_entity(spec: WorldSpec, entity_id: str) -> Entity:
    """Look up an entity by id or raise EntityNotFoundError. O(n)."""
    entity = spec.get_entity(entity_id)
    if entity is None:
        raise EntityNotFoundError(f"No entity with id '{entity_id}'")
    return entity


class Command(ABC):
    """Abstract base for every symbolic world mutation."""

    @abstractmethod
    def execute(self, spec: WorldSpec) -> None:
        """Apply this command's mutation to `spec` in place."""

    @abstractmethod
    def undo(self, spec: WorldSpec) -> None:
        """Reverse this command's mutation on `spec` in place."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description, used in the history log."""


@dataclass
class AddEntityCommand(Command):
    """Adds a new entity to the world. Undo removes it."""
    entity: Entity

    def execute(self, spec: WorldSpec) -> None:
        spec.entities.append(self.entity)

    def undo(self, spec: WorldSpec) -> None:
        spec.entities = [e for e in spec.entities if e.id != self.entity.id]

    @property
    def description(self) -> str:
        return f"add_entity({self.entity.id})"


@dataclass
class RemoveEntityCommand(Command):
    """Removes an entity and (optionally) all interactions referencing it."""
    entity_id: str
    _removed_entity: Optional[Entity] = field(default=None, init=False, repr=False)
    _removed_index: int = field(default=-1, init=False, repr=False)
    _removed_interactions: list[tuple[int, Interaction]] = field(
        default_factory=list, init=False, repr=False
    )
    cascade: bool = True

    def execute(self, spec: WorldSpec) -> None:
        idx = next((i for i, e in enumerate(spec.entities) if e.id == self.entity_id), None)
        if idx is None:
            raise EntityNotFoundError(f"No entity with id '{self.entity_id}'")
        self._removed_entity = spec.entities[idx]
        self._removed_index = idx
        del spec.entities[idx]

        if self.cascade:
            kept: list[Interaction] = []
            for i, itr in enumerate(spec.interactions):
                if itr.entity_a == self.entity_id or itr.entity_b == self.entity_id:
                    self._removed_interactions.append((i, itr))
                else:
                    kept.append(itr)
            spec.interactions = kept

    def undo(self, spec: WorldSpec) -> None:
        if self._removed_entity is None:
            raise InvalidCommandError("undo() called before execute()")
        spec.entities.insert(min(self._removed_index, len(spec.entities)), self._removed_entity)
        for i, itr in sorted(self._removed_interactions, key=lambda p: p[0]):
            spec.interactions.insert(min(i, len(spec.interactions)), itr)
        self._removed_interactions.clear()

    @property
    def description(self) -> str:
        return f"remove_entity({self.entity_id}, cascade={self.cascade})"


@dataclass
class UpdateEntityFieldsCommand(Command):
    """Sets arbitrary scalar fields on an entity (mass, friction, is_static, ...)."""
    entity_id: str
    fields: dict[str, Any]
    _previous: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, spec: WorldSpec) -> None:
        entity = find_entity(spec, self.entity_id)
        for key, value in self.fields.items():
            if not hasattr(entity, key):
                raise InvalidCommandError(f"Entity has no field '{key}'")
            self._previous[key] = getattr(entity, key)
            setattr(entity, key, value)

    def undo(self, spec: WorldSpec) -> None:
        entity = find_entity(spec, self.entity_id)
        for key, value in self._previous.items():
            setattr(entity, key, value)

    @property
    def description(self) -> str:
        return f"update_entity({self.entity_id}, {list(self.fields)})"


@dataclass
class UpdateEntityStateCommand(Command):
    """Updates one or more PhysicsState vector fields (position/velocity/…)."""
    entity_id: str
    fields: dict[str, Vec3]
    _previous: dict[str, Vec3] = field(default_factory=dict, init=False, repr=False)

    def execute(self, spec: WorldSpec) -> None:
        entity = find_entity(spec, self.entity_id)
        for key, value in self.fields.items():
            if not hasattr(entity.state, key):
                raise InvalidCommandError(f"PhysicsState has no field '{key}'")
            self._previous[key] = getattr(entity.state, key)
            setattr(entity.state, key, value)

    def undo(self, spec: WorldSpec) -> None:
        entity = find_entity(spec, self.entity_id)
        for key, value in self._previous.items():
            setattr(entity.state, key, value)

    @property
    def description(self) -> str:
        return f"update_entity_state({self.entity_id}, {list(self.fields)})"


@dataclass
class AddInteractionCommand(Command):
    """Appends a new physics interaction. Undo removes it by identity."""
    interaction: Interaction

    def execute(self, spec: WorldSpec) -> None:
        spec.interactions.append(self.interaction)

    def undo(self, spec: WorldSpec) -> None:
        try:
            spec.interactions.remove(self.interaction)
        except ValueError:
            pass  # best-effort: already removed by a later command

    @property
    def description(self) -> str:
        return f"add_interaction({self.interaction.type}: {self.interaction.entity_a}->{self.interaction.entity_b})"


@dataclass
class RemoveInteractionCommand(Command):
    """Removes the interaction at a given index. Undo re-inserts it there."""
    index: int
    _removed: Optional[Interaction] = field(default=None, init=False, repr=False)

    def execute(self, spec: WorldSpec) -> None:
        if not (0 <= self.index < len(spec.interactions)):
            raise InteractionNotFoundError(f"No interaction at index {self.index}")
        self._removed = spec.interactions.pop(self.index)

    def undo(self, spec: WorldSpec) -> None:
        if self._removed is None:
            raise InvalidCommandError("undo() called before execute()")
        spec.interactions.insert(min(self.index, len(spec.interactions)), self._removed)

    @property
    def description(self) -> str:
        return f"remove_interaction(index={self.index})"


@dataclass
class UpdateEnvironmentCommand(Command):
    """Sets arbitrary scalar/simple fields on the world Environment."""
    fields: dict[str, Any]
    _previous: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, spec: WorldSpec) -> None:
        env = spec.environment
        for key, value in self.fields.items():
            if not hasattr(env, key):
                raise InvalidCommandError(f"Environment has no field '{key}'")
            self._previous[key] = getattr(env, key)
            setattr(env, key, value)

    def undo(self, spec: WorldSpec) -> None:
        env = spec.environment
        for key, value in self._previous.items():
            setattr(env, key, value)

    @property
    def description(self) -> str:
        return f"update_environment({list(self.fields)})"


@dataclass
class UpdateSimulationGraphCommand(Command):
    """Sets arbitrary scalar fields on the SimulationGraph (dt, duration, …)."""
    fields: dict[str, Any]
    _previous: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, spec: WorldSpec) -> None:
        sg = spec.simulation_graph
        for key, value in self.fields.items():
            if not hasattr(sg, key):
                raise InvalidCommandError(f"SimulationGraph has no field '{key}'")
            self._previous[key] = getattr(sg, key)
            setattr(sg, key, value)

    def undo(self, spec: WorldSpec) -> None:
        sg = spec.simulation_graph
        for key, value in self._previous.items():
            setattr(sg, key, value)

    @property
    def description(self) -> str:
        return f"update_simulation_graph({list(self.fields)})"


@dataclass
class GraphRelationCommand(Command):
    """
    Adds or removes a semantic (knowledge-graph) relation edge.

    Relations are symbolic facts ("e_car on_road e_road", "e_pedestrian
    near e_car") kept in `world_engine.graph.EntityGraph`, deliberately
    separate from physics `Interaction`s (which live inside `WorldSpec`
    and are numeric-coupling declarations, not symbolic facts).

    Because `EntityGraph` is not part of `WorldSpec`, this command ignores
    the `spec` argument passed by `HistoryManager`/`WorldEngine._execute`
    and instead mutates the `EntityGraph` instance captured at
    construction time — the uniform `Command.execute(spec)` contract still
    holds (every command is called the same way), only the mutation target
    differs. This keeps `WorldEngine` from needing a second, parallel
    history/undo mechanism for graph edits.
    """
    graph: Any  # world_engine.graph.EntityGraph — Any to avoid a circular import
    subject: str
    predicate: str
    obj: str
    attributes: dict[str, Any] = field(default_factory=dict)
    adding: bool = True  # True = add relation, False = remove relation

    def execute(self, spec: WorldSpec) -> None:
        if self.adding:
            self.graph._add(self.subject, self.predicate, self.obj, self.attributes)
        else:
            self.graph._remove(self.subject, self.predicate, self.obj)

    def undo(self, spec: WorldSpec) -> None:
        if self.adding:
            self.graph._remove(self.subject, self.predicate, self.obj)
        else:
            self.graph._add(self.subject, self.predicate, self.obj, self.attributes)

    @property
    def description(self) -> str:
        verb = "add_relation" if self.adding else "remove_relation"
        return f"{verb}({self.subject} -{self.predicate}-> {self.obj})"
