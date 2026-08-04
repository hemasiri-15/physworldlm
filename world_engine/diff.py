"""
world_engine.diff
────────────────────
Stateless computation of structured diffs between two `WorldSpec`
snapshots. Equally useful for diffing two points in one engine's history
as for comparing two branches produced by `WorldMemory`/`PredictiveEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world_spec import WorldSpec, Entity, Interaction


@dataclass
class EntityDiff:
    """Describes how a single entity changed between two snapshots."""
    entity_id: str
    change_type: str  # "added" | "removed" | "modified"
    changed_fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class WorldDiff:
    """
    Structured difference between two `WorldSpec` snapshots — the artifact
    downstream consumers (a USD stage updater, a UI, a training-data
    logger) consume instead of re-deriving changes themselves.
    """
    entities_added: list[str] = field(default_factory=list)
    entities_removed: list[str] = field(default_factory=list)
    entities_modified: list[EntityDiff] = field(default_factory=list)
    interactions_added: list[Interaction] = field(default_factory=list)
    interactions_removed: list[Interaction] = field(default_factory=list)
    environment_changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    simulation_graph_changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.entities_added or self.entities_removed or self.entities_modified
            or self.interactions_added or self.interactions_removed
            or self.environment_changed or self.simulation_graph_changed
        )

    def summary(self) -> str:
        if self.is_empty():
            return "WorldDiff(no changes)"
        parts = []
        if self.entities_added:
            parts.append(f"+{len(self.entities_added)} entities")
        if self.entities_removed:
            parts.append(f"-{len(self.entities_removed)} entities")
        if self.entities_modified:
            parts.append(f"~{len(self.entities_modified)} entities")
        if self.interactions_added:
            parts.append(f"+{len(self.interactions_added)} interactions")
        if self.interactions_removed:
            parts.append(f"-{len(self.interactions_removed)} interactions")
        if self.environment_changed:
            parts.append(f"env[{', '.join(self.environment_changed)}]")
        if self.simulation_graph_changed:
            parts.append(f"simgraph[{', '.join(self.simulation_graph_changed)}]")
        return "WorldDiff(" + ", ".join(parts) + ")"


class DiffEngine:
    """Stateless computation of `WorldDiff` between two `WorldSpec` instances."""

    @staticmethod
    def compute(old: WorldSpec, new: WorldSpec) -> WorldDiff:
        """
        Purpose:
            Compute a structured diff between two WorldSpec snapshots.
        Complexity:
            O(E + I + F): E = entity count, I = interaction count, F =
            scalar fields inspected per entity/environment/simgraph.
        """
        diff = WorldDiff()

        old_entities = {e.id: e for e in old.entities}
        new_entities = {e.id: e for e in new.entities}

        diff.entities_added = [eid for eid in new_entities if eid not in old_entities]
        diff.entities_removed = [eid for eid in old_entities if eid not in new_entities]

        for eid in new_entities.keys() & old_entities.keys():
            changed = DiffEngine._entity_field_diff(old_entities[eid], new_entities[eid])
            if changed:
                diff.entities_modified.append(EntityDiff(eid, "modified", changed))

        old_interactions = old.interactions
        new_interactions = new.interactions
        diff.interactions_added = [i for i in new_interactions if i not in old_interactions]
        diff.interactions_removed = [i for i in old_interactions if i not in new_interactions]

        diff.environment_changed = DiffEngine._dict_field_diff(
            old.environment.to_dict(), new.environment.to_dict()
        )
        diff.simulation_graph_changed = DiffEngine._dict_field_diff(
            old.simulation_graph.to_dict(), new.simulation_graph.to_dict()
        )
        return diff

    @staticmethod
    def _entity_field_diff(old: Entity, new: Entity) -> dict[str, tuple[Any, Any]]:
        changed: dict[str, tuple[Any, Any]] = {}
        for field_name in ("label", "entity_type", "is_static", "mass",
                           "material", "restitution", "friction", "tags", "constraints"):
            old_val, new_val = getattr(old, field_name), getattr(new, field_name)
            if old_val != new_val:
                changed[field_name] = (old_val, new_val)

        if old.bounding_box.to_dict() != new.bounding_box.to_dict():
            changed["bounding_box"] = (old.bounding_box.to_dict(), new.bounding_box.to_dict())

        old_state, new_state = old.state.to_dict(), new.state.to_dict()
        if old_state != new_state:
            changed["state"] = (old_state, new_state)

        if old.forces != new.forces:
            changed["forces"] = (old.forces, new.forces)

        return changed

    @staticmethod
    def _dict_field_diff(old: dict, new: dict) -> dict[str, tuple[Any, Any]]:
        keys = old.keys() | new.keys()
        return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}
