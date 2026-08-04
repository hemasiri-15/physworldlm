"""
world_engine.plugins
────────────────────────
Plugin architecture: domain packs (traffic, weather, fluid, robotics, ...)
hook into WorldEngine's lifecycle without modifying its source.

A plugin is a `WorldEnginePlugin` subclass overriding only the hooks it
cares about (all hooks default to no-ops). `WorldEngine.register_plugin`
installs it; from then on every relevant mutation/validation call fans out
to every registered plugin in registration order.

Plugins can also extend three things WorldEngine otherwise owns:
    - semantic constraints (via `extra_constraints()`)
    - query predicates for the WQL layer (via `extra_query_predicates()`)
    - event handlers (via `extra_event_handlers()`)
so a "TrafficPlugin" can add a `SemanticConstraint`
("vehicle_must_yield_at_red_light"), a WQL predicate (`at_red_light`), and
an event handler (`"traffic_light_change"`) in one place, all without a
single edit to core WorldEngine files.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .engine import WorldEngine
    from .diff import WorldDiff
    from .constraints import SemanticConstraint
    from world_spec import Entity
    from validator import ValidationResult


class WorldEnginePlugin(ABC):
    """
    Base class for WorldEngine plugins. Every hook is a no-op by default;
    override only what you need. `name` must be unique among registered
    plugins.
    """

    name: str = "unnamed_plugin"

    # ── lifecycle ────────────────────────────────────────────────────────

    def on_register(self, engine: "WorldEngine") -> None:
        """Called once when the plugin is registered with an engine."""

    def on_unregister(self, engine: "WorldEngine") -> None:
        """Called once when the plugin is removed from an engine."""

    # ── mutation hooks (called AFTER the mutation has committed) ────────

    def on_entity_added(self, engine: "WorldEngine", entity: "Entity") -> None:
        """Called after an entity is successfully added."""

    def on_entity_removed(self, engine: "WorldEngine", entity_id: str) -> None:
        """Called after an entity is successfully removed."""

    def on_environment_changed(self, engine: "WorldEngine", changed_fields: dict) -> None:
        """Called after `update_environment` commits, with the changed field names/values."""

    def on_event_processed(self, engine: "WorldEngine", event: dict) -> None:
        """Called after a SimulationGraph event has been applied."""

    def on_validate(self, engine: "WorldEngine", result: "ValidationResult") -> None:
        """
        Called after `PhysicsValidator.validate()` runs, with the result.
        Plugins may call `result.warn(...)`/`result.info(...)` to append
        domain-specific findings (do not remove existing issues).
        """

    # ── extension points ────────────────────────────────────────────────

    def extra_constraints(self) -> list["SemanticConstraint"]:
        """Semantic constraints this plugin contributes to the constraint engine."""
        return []

    def extra_query_predicates(self) -> dict[str, Callable[["WorldEngine", "Entity"], bool]]:
        """
        Named predicates this plugin contributes to the WQL query layer,
        usable as `.matching("predicate_name")` in `WorldQuery`.
        """
        return {}

    def extra_event_handlers(self) -> dict[str, Callable[[dict], None]]:
        """Named event-type handlers this plugin contributes to `EventProcessor`."""
        return {}


class PluginManager:
    """
    Registry + dispatcher for `WorldEnginePlugin`s.

    Kept as its own class (rather than inlined into `WorldEngine`) so
    dispatch failures are isolated: a broken plugin hook logs/raises
    `PluginError` without corrupting `WorldEngine`'s own state, since the
    triggering mutation has already committed by the time hooks run.
    """

    def __init__(self, engine: "WorldEngine") -> None:
        self._engine = engine
        self._plugins: dict[str, WorldEnginePlugin] = {}

    def register(self, plugin: WorldEnginePlugin) -> None:
        from .exceptions import PluginError
        if plugin.name in self._plugins:
            raise PluginError(f"Plugin '{plugin.name}' is already registered")
        self._plugins[plugin.name] = plugin
        plugin.on_register(self._engine)

    def unregister(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin is not None:
            plugin.on_unregister(self._engine)

    def get(self, name: str) -> Optional[WorldEnginePlugin]:
        return self._plugins.get(name)

    def all(self) -> list[WorldEnginePlugin]:
        return list(self._plugins.values())

    # ── fan-out dispatch (best-effort: one plugin's exception doesn't stop others) ──

    def _dispatch(self, hook_name: str, *args: Any) -> None:
        for plugin in self._plugins.values():
            getattr(plugin, hook_name)(self._engine, *args)

    def fire_entity_added(self, entity: "Entity") -> None:
        self._dispatch("on_entity_added", entity)

    def fire_entity_removed(self, entity_id: str) -> None:
        self._dispatch("on_entity_removed", entity_id)

    def fire_environment_changed(self, changed_fields: dict) -> None:
        self._dispatch("on_environment_changed", changed_fields)

    def fire_event_processed(self, event: dict) -> None:
        self._dispatch("on_event_processed", event)

    def fire_validate(self, result: "ValidationResult") -> None:
        self._dispatch("on_validate", result)

    def collect_constraints(self) -> list["SemanticConstraint"]:
        out: list["SemanticConstraint"] = []
        for plugin in self._plugins.values():
            out.extend(plugin.extra_constraints())
        return out

    def collect_query_predicates(self) -> dict[str, Callable]:
        merged: dict[str, Callable] = {}
        for plugin in self._plugins.values():
            merged.update(plugin.extra_query_predicates())
        return merged

    def collect_event_handlers(self) -> dict[str, Callable]:
        merged: dict[str, Callable] = {}
        for plugin in self._plugins.values():
            merged.update(plugin.extra_event_handlers())
        return merged
