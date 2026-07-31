"""
enums.py
────────
All enumerations shared across the World Controller subsystem.
"""

from __future__ import annotations

from enum import Enum, auto


class ChangeEventType(Enum):
    """Every discrete event the World Controller's EventBus can publish."""

    # WorldSpec lifecycle
    WORLD_CREATED = auto()
    WORLD_LOADED = auto()
    WORLD_SAVED = auto()
    WORLD_RESET = auto()
    WORLD_REPLACED = auto()

    # Entity lifecycle
    ENTITY_ADDED = auto()
    ENTITY_DELETED = auto()
    ENTITY_UPDATED = auto()
    ENTITY_RENAMED = auto()
    ENTITY_DUPLICATED = auto()
    ENTITY_GROUPED = auto()
    ENTITY_UNGROUPED = auto()
    ENTITY_PARENTED = auto()
    ENTITY_UNPARENTED = auto()

    # Entity state
    ENTITY_ENABLED = auto()
    ENTITY_DISABLED = auto()
    ENTITY_SHOWN = auto()
    ENTITY_HIDDEN = auto()
    ENTITY_LOCKED = auto()
    ENTITY_UNLOCKED = auto()

    # Selection
    SELECTION_CHANGED = auto()

    # Transform / physics / material / environment
    TRANSFORM_CHANGED = auto()
    PHYSICS_CHANGED = auto()
    MATERIAL_CHANGED = auto()
    ENVIRONMENT_CHANGED = auto()
    RELATIONSHIP_CHANGED = auto()

    # Compilation / validation / CSP
    COMPILE_REQUESTED = auto()
    COMPILE_COMPLETED = auto()
    COMPILE_FAILED = auto()
    VALIDATION_REQUESTED = auto()
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    CSP_REQUESTED = auto()
    CSP_COMPLETED = auto()
    CSP_FAILED = auto()

    # History
    COMMAND_EXECUTED = auto()
    UNDO_PERFORMED = auto()
    REDO_PERFORMED = auto()
    HISTORY_CLEARED = auto()
    HISTORY_JUMPED = auto()

    # Transactions
    TRANSACTION_STARTED = auto()
    TRANSACTION_COMMITTED = auto()
    TRANSACTION_ROLLED_BACK = auto()

    # Plugins / sessions
    PLUGIN_REGISTERED = auto()
    PLUGIN_UNREGISTERED = auto()
    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    DIRTY_STATE_CHANGED = auto()


class HookPoint(Enum):
    """Extension points plugins may attach callbacks to."""

    BEFORE_VALIDATION = auto()
    AFTER_VALIDATION = auto()
    BEFORE_COMPILE = auto()
    AFTER_COMPILE = auto()
    BEFORE_CSP = auto()
    AFTER_CSP = auto()
    BEFORE_AI = auto()
    AFTER_AI = auto()
    BEFORE_COMMAND_EXECUTE = auto()
    AFTER_COMMAND_EXECUTE = auto()


class SelectionMode(Enum):
    """How a new selection interacts with the existing selection set."""

    REPLACE = auto()
    ADD = auto()
    TOGGLE = auto()
    SUBTRACT = auto()


class EntityBodyMode(Enum):
    """Simulation body classification, mirrored onto Entity.tags / forces metadata.

    `world_spec.Entity` only models a boolean `is_static` flag. The World
    Controller layers KINEMATIC and SENSOR as additional authored states
    on top of that boolean via a reserved tag (see `indexes.BODY_MODE_TAG_PREFIX`),
    without altering the WorldSpec data contract.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    KINEMATIC = "kinematic"
    SENSOR = "sensor"


class PluginKind(Enum):
    """Category of a registered plugin, for discovery/introspection."""

    VALIDATOR = auto()
    AI_EDITOR = auto()
    EXPORTER = auto()
    PHYSICS_MODULE = auto()
    CONSTRAINT_SOLVER = auto()
    SCENE_OPTIMIZER = auto()
    GENERIC = auto()
