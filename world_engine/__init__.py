"""
world_engine
──────────────
Symbolic reasoning and world-state management core of PhysWorldLM.

    from world_engine import WorldEngine

is the intended import surface for consumers; everything else here is
exported for advanced use (custom plugins, custom spatial indices, direct
WorldQuery construction, etc.).
"""

from .exceptions import (
    WorldEngineError, EntityNotFoundError, DuplicateEntityError,
    InteractionNotFoundError, InvalidCommandError, OntologyViolationError,
    SemanticConstraintError, HistoryError, EventProcessingError,
    VersionError, PluginError, QueryError,
)
from .ontology import OntologyRegistry, OntologyRule, EntityCategory
from .commands import Command
from .history import HistoryManager, HistoryEntry
from .diff import DiffEngine, WorldDiff, EntityDiff
from .graph import EntityGraph, Relation
from .spatial import SpatialIndex, KDTreeIndex
from .constraints import (
    SemanticConstraint, SemanticConstraintEngine, ConstraintRegistry,
    ConstraintViolation, default_constraints,
)
from .memory import WorldMemory, VersionNode
from .plugins import WorldEnginePlugin, PluginManager
from .events import EventProcessor
from .query import WorldQuery, WQLParser
from .engine import WorldEngine

__all__ = [
    "WorldEngine",
    "WorldEngineError", "EntityNotFoundError", "DuplicateEntityError",
    "InteractionNotFoundError", "InvalidCommandError", "OntologyViolationError",
    "SemanticConstraintError", "HistoryError", "EventProcessingError",
    "VersionError", "PluginError", "QueryError",
    "OntologyRegistry", "OntologyRule", "EntityCategory",
    "Command", "HistoryManager", "HistoryEntry",
    "DiffEngine", "WorldDiff", "EntityDiff",
    "EntityGraph", "Relation",
    "SpatialIndex", "KDTreeIndex",
    "SemanticConstraint", "SemanticConstraintEngine", "ConstraintRegistry",
    "ConstraintViolation", "default_constraints",
    "WorldMemory", "VersionNode",
    "WorldEnginePlugin", "PluginManager",
    "EventProcessor",
    "WorldQuery", "WQLParser",
]

__version__ = "0.2.0"
