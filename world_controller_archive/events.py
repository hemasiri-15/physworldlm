"""
events.py
─────────
Observer-pattern event system for the World Controller.

Every mutation the World Controller performs is announced on the
`EventBus` as a `ChangeEvent`. Subscribers (UI panels, loggers, AI
agents, plugins, autosave routines, ...) register callbacks and are
notified synchronously, in registration order, on the thread that
performed the mutation.

Thread safety: subscription bookkeeping is guarded by an internal
`threading.RLock`. Notification itself iterates over a snapshot of the
subscriber list taken under the lock, so a subscriber callback that
subscribes/unsubscribes during notification cannot corrupt iteration
and cannot deadlock against the bus.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from world_controller.enums import ChangeEventType

logger = logging.getLogger("physworldlm.world_controller.events")


@dataclass(frozen=True)
class ChangeEvent:
    """An immutable record of a single change published on the EventBus.

    Attributes:
        event_type: What kind of change occurred.
        timestamp: UTC time the event was published.
        source: Free-text identifier of what produced the event
            (e.g. a command class name, "controller", a plugin name).
        entity_ids: Entity ids this event concerns, if any.
        payload: Arbitrary structured data describing the change
            (old/new values, diffs, report objects, etc.). Consumers
            should treat this as read-only.
        event_id: Unique id for this event instance.
    """

    event_type: ChangeEventType
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entity_ids: tuple[str, ...] = field(default_factory=tuple)
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "entity_ids": list(self.entity_ids),
            "payload": self.payload,
        }


@runtime_checkable
class EventSubscriber(Protocol):
    """Structural contract for a callable event subscriber."""

    def __call__(self, event: ChangeEvent) -> None:
        ...


SubscriptionToken = str


@dataclass(frozen=True)
class _Subscription:
    token: SubscriptionToken
    callback: Callable[[ChangeEvent], None]
    event_type: Optional[ChangeEventType]  # None == subscribe to all event types


class EventBus:
    """Thread-safe publish/subscribe hub for `ChangeEvent`s.

    Subscribers may register for a specific `ChangeEventType` or for
    every event type by passing `event_type=None` to `subscribe()`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: dict[SubscriptionToken, _Subscription] = {}
        self._by_type: dict[Optional[ChangeEventType], list[SubscriptionToken]] = {}

    # ── subscription management ─────────────────────────────────────

    def subscribe(
        self,
        callback: Callable[[ChangeEvent], None],
        event_type: Optional[ChangeEventType] = None,
    ) -> SubscriptionToken:
        """Register `callback` to be invoked on publish.

        Args:
            callback: A callable accepting a single `ChangeEvent`.
            event_type: If given, `callback` only fires for that event
                type. If `None`, `callback` fires for every event.

        Returns:
            An opaque token that can later be passed to `unsubscribe()`.
        """
        token = uuid.uuid4().hex
        subscription = _Subscription(token=token, callback=callback, event_type=event_type)
        with self._lock:
            self._subscriptions[token] = subscription
            self._by_type.setdefault(event_type, []).append(token)
        return token

    def unsubscribe(self, token: SubscriptionToken) -> bool:
        """Remove a previously registered subscription.

        Returns:
            True if a subscription with `token` was found and removed.
        """
        with self._lock:
            subscription = self._subscriptions.pop(token, None)
            if subscription is None:
                return False
            bucket = self._by_type.get(subscription.event_type, [])
            if token in bucket:
                bucket.remove(token)
            return True

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            self._by_type.clear()

    def subscriber_count(self, event_type: Optional[ChangeEventType] = None) -> int:
        """Number of subscribers for `event_type` (or overall, if None)."""
        with self._lock:
            if event_type is None:
                return len(self._subscriptions)
            specific = len(self._by_type.get(event_type, []))
            catch_all = len(self._by_type.get(None, []))
            return specific + catch_all

    # ── publication ──────────────────────────────────────────────────

    def publish(self, event: ChangeEvent) -> None:
        """Synchronously notify every matching subscriber of `event`.

        Subscriber exceptions are caught, logged, and do not prevent
        other subscribers from being notified, and do not propagate to
        the caller -- a misbehaving observer must never break a mutation
        that has already been committed to the WorldSpec.
        """
        with self._lock:
            tokens = list(self._by_type.get(event.event_type, ())) + list(self._by_type.get(None, ()))
            callbacks = [self._subscriptions[t].callback for t in tokens if t in self._subscriptions]

        for callback in callbacks:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - observers must never break the publisher
                logger.exception(
                    "Event subscriber raised while handling %s (event_id=%s)",
                    event.event_type.name,
                    event.event_id,
                )
