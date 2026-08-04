"""
world_engine.exceptions
────────────────────────
Central exception hierarchy for the WorldEngine package. Every subsystem
raises from this hierarchy so callers can catch `WorldEngineError` broadly
or a specific subtype narrowly.
"""

from __future__ import annotations


class WorldEngineError(Exception):
    """Base class for all WorldEngine-raised errors."""


class EntityNotFoundError(WorldEngineError):
    """Raised when an operation references an entity id that does not exist."""


class DuplicateEntityError(WorldEngineError):
    """Raised when adding an entity whose id already exists in the world."""


class InteractionNotFoundError(WorldEngineError):
    """Raised when an operation references an interaction index that does not exist."""


class InvalidCommandError(WorldEngineError):
    """Raised when a command cannot be constructed or applied as specified."""


class OntologyViolationError(WorldEngineError):
    """Raised when a mutation would violate a hard (non-warning) ontology rule."""


class SemanticConstraintError(WorldEngineError):
    """Raised when a mutation would violate a hard domain-specific semantic constraint."""


class HistoryError(WorldEngineError):
    """Raised on invalid undo/redo requests (e.g. undo with empty history)."""


class EventProcessingError(WorldEngineError):
    """Raised when a SimulationGraph event cannot be translated into commands."""


class VersionError(WorldEngineError):
    """Raised on invalid version/branch/checkout operations in WorldMemory."""


class PluginError(WorldEngineError):
    """Raised when a plugin fails to register or a hook raises unexpectedly."""


class QueryError(WorldEngineError):
    """Raised when a WQL query string cannot be parsed or executed."""
