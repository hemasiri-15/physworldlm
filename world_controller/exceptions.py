"""
exceptions.py
─────────────
Exception hierarchy for the World Controller subsystem.

Every exception raised anywhere inside `world_controller` derives from
`WorldControllerError`, so callers can catch the whole subsystem with a
single `except WorldControllerError:` when that is appropriate, while
still being able to catch narrower failure modes precisely.
"""

from __future__ import annotations

from typing import Optional


class WorldControllerError(Exception):
    """Base class for every exception raised by the World Controller."""


# ──────────────────────────────────────────────────────────────────────
# WorldSpec lifecycle
# ──────────────────────────────────────────────────────────────────────

class WorldSpecNotLoadedError(WorldControllerError):
    """Raised when an operation requires a loaded WorldSpec but none is present."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"Cannot perform '{operation}': no WorldSpec is currently loaded. "
            f"Call create(), load(), or replace() first."
        )


class WorldSpecLoadError(WorldControllerError):
    """Raised when a WorldSpec fails to load from disk or from a dict."""


class WorldSpecSaveError(WorldControllerError):
    """Raised when a WorldSpec fails to persist to disk."""


class WorldValidationError(WorldControllerError):
    """Raised when a WorldSpec (or a proposed mutation to it) fails validation."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        super().__init__("; ".join(messages) if messages else "WorldSpec validation failed.")


# ──────────────────────────────────────────────────────────────────────
# Entity resolution
# ──────────────────────────────────────────────────────────────────────

class EntityNotFoundError(WorldControllerError):
    """Raised when an operation references an entity id that does not exist."""

    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id
        super().__init__(f"No entity with id '{entity_id}' exists in the current WorldSpec.")


class DuplicateEntityIdError(WorldControllerError):
    """Raised when attempting to create/duplicate an entity under an id already in use."""

    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id
        super().__init__(f"An entity with id '{entity_id}' already exists.")


class EntityLockedError(WorldControllerError):
    """Raised when a mutation targets an entity that is currently locked."""

    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id
        super().__init__(f"Entity '{entity_id}' is locked and cannot be modified while locked.")


class MaterialNotFoundError(WorldControllerError):
    """Raised when a referenced material name is not registered."""

    def __init__(self, material_name: str) -> None:
        self.material_name = material_name
        super().__init__(f"No material named '{material_name}' is registered.")


class RelationshipError(WorldControllerError):
    """Raised for invalid relationship/interaction operations (bad endpoints, duplicates, ...)."""


class InvalidParameterError(WorldControllerError):
    """Raised when a caller supplies a structurally or physically invalid parameter."""

    def __init__(self, parameter: str, value: object, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid value for '{parameter}' ({value!r}): {reason}")


# ──────────────────────────────────────────────────────────────────────
# Command system
# ──────────────────────────────────────────────────────────────────────

class CommandError(WorldControllerError):
    """Base class for command execution/undo/redo failures."""


class CommandExecutionError(CommandError):
    """Raised when `Command.execute()` fails."""

    def __init__(self, command_description: str, cause: Optional[Exception] = None) -> None:
        self.command_description = command_description
        self.cause = cause
        message = f"Command '{command_description}' failed to execute"
        if cause is not None:
            message += f": {cause}"
        super().__init__(message)


class CommandUndoError(CommandError):
    """Raised when `Command.undo()` fails."""

    def __init__(self, command_description: str, cause: Optional[Exception] = None) -> None:
        self.command_description = command_description
        self.cause = cause
        message = f"Command '{command_description}' failed to undo"
        if cause is not None:
            message += f": {cause}"
        super().__init__(message)


# ──────────────────────────────────────────────────────────────────────
# History / transactions
# ──────────────────────────────────────────────────────────────────────

class HistoryError(WorldControllerError):
    """Base class for command-history failures."""


class NothingToUndoError(HistoryError):
    """Raised when `undo()` is called with an empty undo stack."""

    def __init__(self) -> None:
        super().__init__("Undo stack is empty; there is nothing to undo.")


class NothingToRedoError(HistoryError):
    """Raised when `redo()` is called with an empty redo stack."""

    def __init__(self) -> None:
        super().__init__("Redo stack is empty; there is nothing to redo.")


class InvalidHistoryVersionError(HistoryError):
    """Raised when `jump_to_version()` is given an out-of-range version index."""

    def __init__(self, version: int, valid_range: tuple[int, int]) -> None:
        self.version = version
        self.valid_range = valid_range
        super().__init__(
            f"Version {version} is out of range; valid range is "
            f"[{valid_range[0]}, {valid_range[1]}]."
        )


class TransactionError(WorldControllerError):
    """Base class for transaction-management failures."""


class TransactionAlreadyOpenError(TransactionError):
    """Raised when `begin_transaction()` is called while a transaction is already open.

    PhysWorldLM's World Controller supports only one open transaction per
    controller instance; nested transactions must be modeled by grouping
    multiple commands inside a single transaction rather than nesting.
    """

    def __init__(self) -> None:
        super().__init__(
            "A transaction is already open. Commit or rollback the current "
            "transaction before beginning a new one."
        )


class NoOpenTransactionError(TransactionError):
    """Raised when `commit_transaction()` / `rollback_transaction()` is called with none open."""

    def __init__(self) -> None:
        super().__init__("No transaction is currently open.")


# ──────────────────────────────────────────────────────────────────────
# Plugins
# ──────────────────────────────────────────────────────────────────────

class PluginError(WorldControllerError):
    """Base class for plugin registration/execution failures."""


class PluginAlreadyRegisteredError(PluginError):
    """Raised when registering a plugin whose name is already in use."""

    def __init__(self, plugin_name: str) -> None:
        self.plugin_name = plugin_name
        super().__init__(f"A plugin named '{plugin_name}' is already registered.")


class PluginNotFoundError(PluginError):
    """Raised when unregistering or looking up a plugin name that does not exist."""

    def __init__(self, plugin_name: str) -> None:
        self.plugin_name = plugin_name
        super().__init__(f"No plugin named '{plugin_name}' is registered.")


class HookExecutionError(PluginError):
    """Raised when a registered extension hook callback raises an exception."""

    def __init__(self, hook_name: str, plugin_name: str, cause: Exception) -> None:
        self.hook_name = hook_name
        self.plugin_name = plugin_name
        self.cause = cause
        super().__init__(
            f"Hook '{hook_name}' registered by plugin '{plugin_name}' raised "
            f"an exception: {cause}"
        )


# ──────────────────────────────────────────────────────────────────────
# Sessions / concurrency
# ──────────────────────────────────────────────────────────────────────

class SessionError(WorldControllerError):
    """Base class for edit-session failures."""


class NoActiveSessionError(SessionError):
    """Raised when a session-scoped operation is attempted with no active session."""

    def __init__(self) -> None:
        super().__init__("No active edit session. Call begin_session() first.")


class SessionAlreadyActiveError(SessionError):
    """Raised when `begin_session()` is called while a session is already active."""

    def __init__(self) -> None:
        super().__init__("An edit session is already active. End it before beginning a new one.")


# ──────────────────────────────────────────────────────────────────────
# Compilation coordination
# ──────────────────────────────────────────────────────────────────────

class CompilationCoordinationError(WorldControllerError):
    """Raised when the World Controller cannot coordinate a compile request.

    This wraps failures surfaced by the (external, untouched) SceneCompiler
    so callers of the World Controller only need to catch WorldControllerError
    subclasses, without importing scene_compiler's exception types directly.
    """

    def __init__(self, cause: Exception) -> None:
        self.cause = cause
        super().__init__(f"Compilation failed: {cause}")
