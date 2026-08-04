"""
world_engine.engine
──────────────────────
WorldEngine — symbolic reasoning and world-state management core of
PhysWorldLM.

WorldEngine owns exactly one live `WorldSpec` and is the sole authority
for mutating it. It composes (does not inherit) every subsystem in this
package:

    OntologyRegistry           category-level structural expectations
    SemanticConstraintEngine   instance-level domain reasoning
    EntityGraph                symbolic relations (knowledge graph)
    SpatialIndex (KDTreeIndex) proximity queries
    HistoryManager             fine-grained undo/redo
    WorldMemory                past -> current -> future version DAG
    PluginManager               domain-pack extension points
    EventProcessor              scripted event -> API-call translation
    WorldQuery / WQLParser      declarative entity queries

Each is independently testable and swappable via the constructor, keeping
WorldEngine itself a thin, auditable orchestrator rather than a monolith.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Optional

from world_spec import WorldSpec, Entity, Interaction
from validator import PhysicsValidator, ValidationResult

from .exceptions import (
    WorldEngineError, EntityNotFoundError, DuplicateEntityError,
    OntologyViolationError, SemanticConstraintError,
)
from .ontology import OntologyRegistry
from .commands import (
    AddEntityCommand, RemoveEntityCommand, UpdateEntityFieldsCommand,
    UpdateEntityStateCommand, AddInteractionCommand, RemoveInteractionCommand,
    UpdateEnvironmentCommand, UpdateSimulationGraphCommand, find_entity,
)
from .history import HistoryManager
from .diff import DiffEngine, WorldDiff
from .memory import WorldMemory, VersionNode
from .graph import EntityGraph
from .spatial import SpatialIndex, KDTreeIndex
from .constraints import SemanticConstraintEngine, ConstraintViolation
from .plugins import PluginManager, WorldEnginePlugin
from .events import EventProcessor
from .query import WorldQuery, WQLParser
from world_spec import Vec3


class WorldEngine:
    """
    Symbolic reasoning and world-state management core of PhysWorldLM.

    Thread-safety: NOT thread-safe. Concurrent access must be externally
    synchronised (a future `AsyncWorldEngine` wrapper could serialize
    calls onto a single event loop).
    """

    def __init__(
        self,
        spec: Optional[WorldSpec] = None,
        *,
        ontology: Optional[OntologyRegistry] = None,
        validator: Optional[PhysicsValidator] = None,
        constraint_engine: Optional[SemanticConstraintEngine] = None,
        spatial_index: Optional[SpatialIndex] = None,
        max_history: int = 500,
        strict_ontology: bool = False,
        strict_constraints: bool = False,
        auto_reindex: bool = True,
    ) -> None:
        """
        Purpose:
            Construct a WorldEngine around a (possibly empty) WorldSpec,
            wiring up every subsystem.
        Inputs:
            spec: initial WorldSpec; an empty one is created if None.
            ontology / validator / constraint_engine / spatial_index:
                inject custom implementations; each defaults to the
                package's standard implementation.
            max_history: bound on the fine-grained undo stack.
            strict_ontology: raise OntologyViolationError instead of
                silently allowing advisories on add_entity/update_entity.
            strict_constraints: raise SemanticConstraintError on ERROR-
                severity constraint violations during validate_semantics().
            auto_reindex: rebuild the spatial index after every mutation
                that could move/add/remove an entity. Disable for bulk
                loads and call `reindex()` once at the end for O(n log n)
                instead of O(n · n log n).
        Complexity:
            O(1) (plus O(E) initial spatial index build if `spec` has
            entities).
        """
        self._spec: WorldSpec = spec if spec is not None else WorldSpec(
            scene_id=f"scene_{uuid.uuid4().hex[:8]}", description=""
        )
        self.ontology = ontology or OntologyRegistry()
        self.validator = validator or PhysicsValidator()
        self.constraint_engine = constraint_engine or SemanticConstraintEngine()
        self.entity_graph = EntityGraph()
        self.spatial_index: SpatialIndex = spatial_index or KDTreeIndex()
        self.plugins = PluginManager(self)

        self._history = HistoryManager(max_history)
        self._events = EventProcessor(self)
        self._query_parser = WQLParser()
        self.memory = WorldMemory(self._spec, initial_tick=0)

        self._tick: int = 0
        self._processed_event_ids: set[int] = set()
        self.strict_ontology = strict_ontology
        self.strict_constraints = strict_constraints
        self._auto_reindex = auto_reindex

        self.reindex()

    # ── world lifecycle ─────────────────────────────────────────────────

    @property
    def spec(self) -> WorldSpec:
        """The live, mutable WorldSpec. Prefer `snapshot()` for an isolated copy."""
        return self._spec

    @property
    def tick(self) -> int:
        """Monotonically increasing counter, incremented once per executed command."""
        return self._tick

    def load(self, spec: WorldSpec) -> None:
        """Replace the live world with `spec`; resets history, tick, memory, and index."""
        self._spec = spec
        self._history.clear()
        self._tick = 0
        self._processed_event_ids.clear()
        self.entity_graph = EntityGraph()
        self.memory = WorldMemory(self._spec, initial_tick=0)
        self.reindex()

    def snapshot(self) -> WorldSpec:
        """Deep copy of the current world, safe to mutate/store independently. O(E+I)."""
        return copy.deepcopy(self._spec)

    def reset(self, spec: Optional[WorldSpec] = None) -> None:
        self.load(spec if spec is not None else WorldSpec(
            scene_id=f"scene_{uuid.uuid4().hex[:8]}", description=""
        ))

    # ── internal execution ──────────────────────────────────────────────

    def _execute(self, command) -> None:
        """The single choke point every mutating command passes through."""
        command.execute(self._spec)
        self._tick += 1
        self._history.push(command, self._tick)

    # ── entity management ───────────────────────────────────────────────

    def add_entity(self, entity: Entity, *, validate_ontology: bool = True,
                    validate_semantics: bool = False) -> None:
        """
        Purpose:
            Add a new entity to the world.
        Inputs:
            entity: fully constructed Entity (id must be unique).
            validate_ontology: run OntologyRegistry checks; raises if
                `strict_ontology` and advisories are found.
            validate_semantics: additionally run SemanticConstraintEngine
                against this single entity post-insertion; raises if
                `strict_constraints` and an ERROR-severity violation is
                found (the entity remains added — validation is advisory
                unless strict, matching `validate()`'s semantics).
        Exceptions:
            DuplicateEntityError, OntologyViolationError, SemanticConstraintError.
        Complexity:
            O(n) duplicate-id check + O(E log E) spatial reindex (if
            auto_reindex) + O(C) constraint check if requested.
        """
        if self._spec.get_entity(entity.id) is not None:
            raise DuplicateEntityError(f"Entity id '{entity.id}' already exists")

        if validate_ontology:
            advisories = self.ontology.evaluate_entity(entity)
            if advisories and self.strict_ontology:
                raise OntologyViolationError("; ".join(advisories))

        self._execute(AddEntityCommand(entity))
        if self._auto_reindex:
            self.reindex()
        self.plugins.fire_entity_added(entity)

        if validate_semantics:
            violations = [v for v in self.constraint_engine.evaluate(self)
                          if v.entity_id == entity.id]
            fatal = [v for v in violations if v.severity == "ERROR"]
            if fatal and self.strict_constraints:
                raise SemanticConstraintError("; ".join(v.message for v in fatal))

    def remove_entity(self, entity_id: str, *, cascade: bool = True) -> None:
        """
        Purpose:
            Remove an entity; cascades to referencing physics interactions
            AND symbolic relations (EntityGraph) by default.
        Exceptions:
            EntityNotFoundError.
        Complexity:
            O(E + I) for cascade + O(E log E) reindex if auto_reindex.
        """
        self._execute(RemoveEntityCommand(entity_id, cascade=cascade))
        if cascade:
            self.entity_graph.drop_entity(entity_id)
        if self._auto_reindex:
            self.reindex()
        self.plugins.fire_entity_removed(entity_id)

    def update_entity(self, entity_id: str, **fields: Any) -> None:
        """Update scalar Entity fields (mass, friction, material, forces, ...). O(1)+O(k)."""
        if not fields:
            return
        self._execute(UpdateEntityFieldsCommand(entity_id, dict(fields)))
        if self._auto_reindex and "state" not in fields:
            pass  # scalar fields never move an entity; skip needless reindex

    def update_entity_state(self, entity_id: str, **fields: Vec3) -> None:
        """Update PhysicsState vector fields (position, velocity, ...). O(1)+O(k)."""
        if not fields:
            return
        self._execute(UpdateEntityStateCommand(entity_id, dict(fields)))
        if self._auto_reindex and "position" in fields:
            self.reindex()

    def get_entity(self, entity_id: str) -> Entity:
        return find_entity(self._spec, entity_id)

    def list_entities(self, entity_type: Optional[str] = None) -> list[Entity]:
        if entity_type is None:
            return list(self._spec.entities)
        return [e for e in self._spec.entities if e.entity_type == entity_type]

    # ── interaction management (physics) ────────────────────────────────

    def add_interaction(self, interaction: Interaction, *, verify_refs: bool = True) -> None:
        if verify_refs:
            ids = {e.id for e in self._spec.entities}
            if interaction.entity_a not in ids:
                raise EntityNotFoundError(f"entity_a '{interaction.entity_a}' not found")
            if interaction.entity_b not in ids and interaction.entity_b != "environment":
                raise EntityNotFoundError(f"entity_b '{interaction.entity_b}' not found")
        self._execute(AddInteractionCommand(interaction))

    def remove_interaction(self, index: int) -> None:
        self._execute(RemoveInteractionCommand(index))

    def list_interactions(self, entity_id: Optional[str] = None) -> list[Interaction]:
        if entity_id is None:
            return list(self._spec.interactions)
        return [i for i in self._spec.interactions
                if i.entity_a == entity_id or i.entity_b == entity_id]

    # ── relation management (symbolic knowledge graph) ──────────────────

    def add_relation(self, subject: str, predicate: str, obj: str,
                      *, attributes: Optional[dict[str, Any]] = None,
                      verify_refs: bool = True) -> None:
        """
        Purpose:
            Add a symbolic relation edge (e.g. "e_car" "on_road" "e_road")
            to the EntityGraph, participating in the same undo/redo
            history as physics/environment mutations.
        Exceptions:
            EntityNotFoundError if verify_refs and subject/object are not
            live entity ids ("environment" is always allowed as object).
        Complexity:
            O(1).
        """
        if verify_refs:
            ids = {e.id for e in self._spec.entities}
            if subject not in ids:
                raise EntityNotFoundError(f"relation subject '{subject}' not found")
            if obj not in ids and obj != "environment":
                raise EntityNotFoundError(f"relation object '{obj}' not found")
        self._execute(self.entity_graph.make_add_command(subject, predicate, obj, attributes))

    def remove_relation(self, subject: str, predicate: str, obj: str) -> None:
        self._execute(self.entity_graph.make_remove_command(subject, predicate, obj))

    # ── environment / simulation graph ──────────────────────────────────

    def update_environment(self, **fields: Any) -> None:
        if not fields:
            return
        self._execute(UpdateEnvironmentCommand(dict(fields)))
        self.plugins.fire_environment_changed(fields)

    def update_simulation_graph(self, **fields: Any) -> None:
        if not fields:
            return
        self._execute(UpdateSimulationGraphCommand(dict(fields)))

    # ── spatial index ────────────────────────────────────────────────────

    def reindex(self) -> None:
        """Rebuild `spatial_index` from the current entity positions. O(E log E)."""
        self.spatial_index.build(self._spec.entities)

    # ── event processing ────────────────────────────────────────────────

    def process_event(self, event: dict[str, Any]) -> None:
        for name, handler in self.plugins.collect_event_handlers().items():
            self._events.register_handler(name, handler)
        self._events.process(event)
        self.plugins.fire_event_processed(event)

    def advance_events(self, up_to_time: float) -> list[dict[str, Any]]:
        """Apply every due, not-yet-processed event with `t_s <= up_to_time`, in order."""
        events = self._spec.simulation_graph.events
        due = sorted(
            (e for e in events if id(e) not in self._processed_event_ids
             and e.get("t_s", 0.0) <= up_to_time),
            key=lambda e: e.get("t_s", 0.0),
        )
        applied: list[dict[str, Any]] = []
        for event in due:
            self.process_event(event)
            self._processed_event_ids.add(id(event))
            applied.append(event)
        return applied

    # ── history (undo/redo) ─────────────────────────────────────────────

    def undo(self) -> str:
        command = self._history.undo(self._spec)
        if self._auto_reindex:
            self.reindex()
        return command.description

    def redo(self) -> str:
        command = self._history.redo(self._spec)
        if self._auto_reindex:
            self.reindex()
        return command.description

    def can_undo(self) -> bool:
        return self._history.can_undo()

    def can_redo(self) -> bool:
        return self._history.can_redo()

    def history_log(self) -> list[str]:
        return self._history.log()

    # ── world memory (versioning / branching / futures) ─────────────────

    def checkpoint(self, label: str = "") -> str:
        """
        Purpose:
            Record the current live world as a named point in `WorldMemory`
            (the coarse "Past" layer). Cheap edits use `undo`/`redo`;
            checkpoints are for meaningful milestones ("before_rain",
            "scene_v2") worth branching from later.
        Outputs:
            The new version_id.
        Complexity:
            O(E + I) (deep copy).
        """
        return self.memory.checkpoint(self._spec, self._tick, label)

    def branch(self, label: str = "", from_version: Optional[str] = None) -> str:
        """Create a named branch point from `from_version` (default: current) without moving current."""
        return self.memory.branch(self._spec, self._tick, from_version=from_version, label=label)

    def checkout(self, version_id: str) -> WorldDiff:
        """
        Purpose:
            Load a WorldMemory version as the live world (used to switch
            to a branch, or to inspect a past checkpoint). Resets
            fine-grained undo/redo history (it belonged to the world state
            being replaced) but leaves WorldMemory's DAG intact.
        Outputs:
            WorldDiff from the pre-checkout live world to the checked-out
            version, so callers/UIs can show what changed.
        Complexity:
            O(E + I).
        """
        old_spec = self._spec
        new_spec = self.memory.checkout(version_id)
        diff = DiffEngine.compute(old_spec, new_spec)
        self._spec = new_spec
        self._history.clear()
        self.entity_graph = EntityGraph()
        self.reindex()
        return diff

    def predict_future(self, predicted_spec: WorldSpec, tick: int, label: str = "",
                        metadata: Optional[dict[str, Any]] = None,
                        from_version: Optional[str] = None) -> str:
        """
        Purpose:
            Attach a speculative continuation (typically produced by
            `PredictiveEngine`) to the version DAG WITHOUT altering the
            live world. This is the "Future World" slot in the
            Past -> Current -> Future narrative: multiple calls from the
            same parent model multiple candidate futures for a planner or
            risk-scorer to compare.
        Outputs:
            The new (unpromoted) future version_id.
        Complexity:
            O(E + I).
        """
        return self.memory.add_future(predicted_spec, tick, from_version=from_version,
                                       label=label, metadata=metadata)

    def promote_future(self, future_version_id: str) -> WorldDiff:
        """Accept a predicted future into the real timeline and load it as the live world."""
        diff = self.checkout(future_version_id)
        self.memory.promote(future_version_id)
        return diff

    def timeline(self) -> list[VersionNode]:
        """Past-chain-to-current plus every future branching off current — see WorldMemory.timeline."""
        return self.memory.timeline()

    # ── diffing ──────────────────────────────────────────────────────────

    def diff_against(self, other: WorldSpec) -> WorldDiff:
        return DiffEngine.compute(other, self._spec)

    def sync_from_worldspec(self, new_spec: WorldSpec) -> WorldDiff:
        """
        Replace the live world with `new_spec` (e.g. output of a fresh
        `WorldParser.parse()` re-run) while preserving undo continuity:
        returns the diff, and `undo()` immediately after restores the
        previous world.
        """
        old_spec = self._spec
        diff = DiffEngine.compute(old_spec, new_spec)
        self._execute(_ReplaceWorldCommand(old_spec, new_spec))
        if self._auto_reindex:
            self.reindex()
        return diff

    # ── validation ───────────────────────────────────────────────────────

    def validate(self) -> ValidationResult:
        """Run the numeric-physical PhysicsValidator pipeline, then let plugins append findings."""
        result = self.validator.validate(self._spec)
        self.plugins.fire_validate(result)
        return result

    def check_ontology(self) -> dict[str, list[str]]:
        """Category-level structural advisories per entity (see OntologyRegistry)."""
        report: dict[str, list[str]] = {}
        for entity in self._spec.entities:
            advisories = self.ontology.evaluate_entity(entity)
            if self.ontology.requires_ground_contact(entity.entity_type):
                has_ground = any(
                    itr.entity_a == entity.id and itr.type in ("contact", "friction")
                    for itr in self._spec.interactions
                )
                if not has_ground and entity.state.position.y <= 1.0:
                    advisories.append(
                        "expected a ground-contact interaction for this grounded entity_type"
                    )
            if advisories:
                report[entity.id] = advisories
        return report

    def validate_semantics(self) -> list[ConstraintViolation]:
        """
        Instance-level domain reasoning (SemanticConstraintEngine) — e.g.
        "this boat has no water nearby". Includes constraints contributed
        by registered plugins.
        """
        for constraint in self.plugins.collect_constraints():
            self.constraint_engine.registry.register(constraint)
        violations = self.constraint_engine.evaluate(self)
        fatal = [v for v in violations if v.severity == "ERROR"]
        if fatal and self.strict_constraints:
            raise SemanticConstraintError("; ".join(v.message for v in fatal))
        return violations

    def is_simulation_ready(self) -> bool:
        """Minimum bar for SimulationEngine handoff: no numeric-physical ERRORs."""
        return self.validate().is_valid

    # ── scene graph ──────────────────────────────────────────────────────

    def scene_graph(self) -> dict[str, list[str]]:
        """Simple hierarchical grouping of entity ids by entity_type."""
        graph: dict[str, list[str]] = {}
        for entity in self._spec.entities:
            graph.setdefault(entity.entity_type, []).append(entity.id)
        return graph

    # ── query (WQL) ──────────────────────────────────────────────────────

    def query(self, wql: Optional[str] = None) -> "WorldQuery | list[Entity]":
        """
        Purpose:
            Query entities either programmatically (`engine.query()` ->
            `WorldQuery` builder, chain filters, call `.execute()`) or via
            the small WQL text DSL (`engine.query("find moving vehicles")`
            -> executed list[Entity] directly).
        Inputs:
            wql: optional WQL text query. If omitted, returns an empty
                `WorldQuery` builder for programmatic chaining.
        Outputs:
            A `WorldQuery` (no `wql` given) or `list[Entity]` (executed
            result of the parsed `wql` string).
        Exceptions:
            QueryError if `wql` is given but does not match any registered
            pattern (see `WQLParser`).
        Complexity:
            O(E) filter evaluation (+O(log n) if `.near`/`.touching` uses
            the spatial index).
        """
        if wql is None:
            return WorldQuery(self)
        return self._query_parser.parse(self, wql).execute()

    def register_query_pattern(self, regex: str, handler) -> None:
        """Extend the WQL DSL with a new phrasing -> WorldQuery compiler."""
        self._query_parser.register_pattern(regex, handler)

    # ── plugins ──────────────────────────────────────────────────────────

    def register_plugin(self, plugin: WorldEnginePlugin) -> None:
        self.plugins.register(plugin)

    def unregister_plugin(self, name: str) -> None:
        self.plugins.unregister(name)

    # ── simulation handoff ──────────────────────────────────────────────

    def to_worldspec(self) -> WorldSpec:
        return self.snapshot()

    def export_for_simulation(self, *, require_valid: bool = True) -> WorldSpec:
        """
        Purpose:
            Produce the artifact handed to SimulationEngine: a validated,
            deep-copied WorldSpec, decoupled from further WorldEngine
            mutation.
        Exceptions:
            WorldEngineError if `require_valid` and PhysicsValidator
            reports any ERROR-severity issue.
        Complexity:
            O(E + I) validation + O(E + I) deep copy.
        """
        result = self.validate()
        if require_valid and not result.is_valid:
            raise WorldEngineError(f"World is not simulation-ready:\n{result.report()}")
        return self.snapshot()


class _ReplaceWorldCommand:
    """Internal command backing `sync_from_worldspec`: swaps live WorldSpec content in place."""

    def __init__(self, old_spec: WorldSpec, new_spec: WorldSpec) -> None:
        self.old_spec = old_spec
        self.new_spec = new_spec

    def execute(self, spec: WorldSpec) -> None:
        self._copy_into(spec, self.new_spec)

    def undo(self, spec: WorldSpec) -> None:
        self._copy_into(spec, self.old_spec)

    @staticmethod
    def _copy_into(target: WorldSpec, source: WorldSpec) -> None:
        source_copy = copy.deepcopy(source)
        target.scene_id = source_copy.scene_id
        target.description = source_copy.description
        target.entities = source_copy.entities
        target.environment = source_copy.environment
        target.interactions = source_copy.interactions
        target.simulation_graph = source_copy.simulation_graph
        target.metadata = source_copy.metadata

    @property
    def description(self) -> str:
        return f"sync_from_worldspec(scene_id={self.new_spec.scene_id})"
