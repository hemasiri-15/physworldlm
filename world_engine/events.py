"""
world_engine.events
──────────────────────
Translates due `SimulationGraph.events` entries into `WorldEngine` API
calls, so scripted events, LLM edits, and manual API calls are all
indistinguishable to the history/diff machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from world_spec import Entity, BoundingBox, PhysicsState, Vec3, Interaction
from .exceptions import EventProcessingError

if TYPE_CHECKING:
    from .engine import WorldEngine


def entity_from_dict(d: dict[str, Any]) -> Entity:
    """Construct an Entity from a partial dict payload (event-spawn convenience)."""
    bb = BoundingBox(**d.get("bounding_box", {})) if "bounding_box" in d else BoundingBox()
    state_d = d.get("state", {})
    state = PhysicsState(
        position=Vec3(**state_d.get("position", {})) if "position" in state_d else Vec3(),
        velocity=Vec3(**state_d.get("velocity", {})) if "velocity" in state_d else Vec3(),
        acceleration=Vec3(**state_d.get("acceleration", {})) if "acceleration" in state_d else Vec3(),
        orientation=Vec3(**state_d.get("orientation", {})) if "orientation" in state_d else Vec3(),
        angular_vel=Vec3(**state_d.get("angular_vel", {})) if "angular_vel" in state_d else Vec3(),
    )
    return Entity(
        id=d["id"],
        label=d.get("label", d["id"]),
        entity_type=d.get("entity_type", "object"),
        is_static=d.get("is_static", False),
        mass=d.get("mass_kg", d.get("mass", 1.0)),
        material=d.get("material", "generic"),
        restitution=d.get("restitution", 0.5),
        friction=d.get("friction", 0.5),
        bounding_box=bb,
        state=state,
        forces=d.get("forces", []),
        constraints=d.get("constraints", []),
        tags=d.get("tags", []),
    )


class EventProcessor:
    """
    Dispatch table from SimulationGraph event `type` to a handler that
    submits the equivalent `WorldEngine` API calls. Extensible at runtime
    via `register_handler`, and plugins can contribute handlers through
    `PluginManager.collect_event_handlers()` (merged in by `WorldEngine`).
    """

    def __init__(self, engine: "WorldEngine") -> None:
        self._engine = engine
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "entity_spawn": self._handle_entity_spawn,
            "entity_remove": self._handle_entity_remove,
            "force_change": self._handle_force_change,
            "collision": self._handle_collision,
        }

    def register_handler(self, event_type: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handlers[event_type] = handler

    def process(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if not event_type:
            raise EventProcessingError(f"Event missing 'type': {event}")
        handler = self._handlers.get(event_type)
        if handler is None:
            raise EventProcessingError(f"No handler registered for event type '{event_type}'")
        handler(event)

    def _handle_entity_spawn(self, event: dict[str, Any]) -> None:
        payload = event.get("entity")
        if payload is None:
            raise EventProcessingError("entity_spawn event missing 'entity' payload")
        entity = payload if isinstance(payload, Entity) else entity_from_dict(payload)
        self._engine.add_entity(entity)

    def _handle_entity_remove(self, event: dict[str, Any]) -> None:
        entity_ids = event.get("entity_ids")
        if not entity_ids:
            raise EventProcessingError("entity_remove event missing 'entity_ids'")
        for eid in entity_ids:
            self._engine.remove_entity(eid)

    def _handle_force_change(self, event: dict[str, Any]) -> None:
        entity_ids = event.get("entity_ids")
        forces = event.get("forces")
        if not entity_ids or forces is None:
            raise EventProcessingError("force_change event missing 'entity_ids' or 'forces'")
        for eid in entity_ids:
            self._engine.update_entity(eid, forces=forces)

    def _handle_collision(self, event: dict[str, Any]) -> None:
        entity_ids = event.get("entity_ids") or []
        if len(entity_ids) < 2:
            raise EventProcessingError("collision event requires at least 2 entity_ids")
        interaction = Interaction(
            type="collision",
            entity_a=entity_ids[0],
            entity_b=entity_ids[1],
            parameters={"scripted": True, "description": event.get("description", "")},
        )
        self._engine.add_interaction(interaction)
