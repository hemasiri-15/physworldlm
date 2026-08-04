"""
world_engine.history
──────────────────────
Fine-grained, linear undo/redo over `Command` objects.

This is intentionally the *cheap, in-place* layer: every edit is O(1) to
record and its cost to reverse is whatever the command itself costs. It is
NOT the versioning/branching layer — for git-like checkpoints and branches
(needed by PredictiveEngine's what-if exploration) see
`world_engine.memory.WorldMemory`, which is a coarser, deep-copy-based
layer built on top of explicit `WorldEngine.checkpoint()` calls.

Keeping the two separate is a deliberate cost/granularity trade-off: you
would not want every single field tweak to trigger a deep copy of the
world (that's what HistoryManager avoids), but you also don't want
undo/redo alone to answer "show me every future I explored from version 3"
(that's what WorldMemory is for).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from world_spec import WorldSpec
from .commands import Command
from .exceptions import HistoryError


@dataclass
class HistoryEntry:
    """One executed command plus bookkeeping metadata for the audit log."""
    command: Command
    tick: int
    timestamp: float


class HistoryManager:
    """Linear undo/redo stack over `Command` objects, bounded in size."""

    def __init__(self, max_history: int = 1000) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self._max_history = max_history
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []

    def push(self, command: Command, tick: int) -> None:
        """Record a successfully-executed command; clears the redo stack. O(1) amortized."""
        self._undo_stack.append(HistoryEntry(command, tick, time.time()))
        self._redo_stack.clear()
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def undo(self, spec: WorldSpec) -> Command:
        """Reverse the most recent command against `spec`. Raises HistoryError if empty."""
        if not self._undo_stack:
            raise HistoryError("Nothing to undo")
        entry = self._undo_stack.pop()
        entry.command.undo(spec)
        self._redo_stack.append(entry)
        return entry.command

    def redo(self, spec: WorldSpec) -> Command:
        """Re-apply the most recently undone command against `spec`. Raises HistoryError if empty."""
        if not self._redo_stack:
            raise HistoryError("Nothing to redo")
        entry = self._redo_stack.pop()
        entry.command.execute(spec)
        self._undo_stack.append(entry)
        return entry.command

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def log(self) -> list[str]:
        """Human-readable audit trail, oldest first."""
        return [f"[t={e.tick}] {e.command.description}" for e in self._undo_stack]
