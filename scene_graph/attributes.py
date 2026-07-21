"""Provenance-tagged attribute values.

Every scalar or structured value attached to a :class:`~physworldlm.scene_graph.nodes.SceneNode`
carries a :class:`Provenance` tag so that downstream consumers (validators,
confidence-calibration studies, explainability tooling) can distinguish a
value the parser read directly off the prompt from one it derived through
physics defaults or unit conversion, mirroring the
``reasoning.known / derived / unknown / assumptions`` buckets already
emitted by ``PromptParser``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    """The evidential source of an attribute value."""

    KNOWN = "known"
    """Read directly and unambiguously from the input prompt."""

    DERIVED = "derived"
    """Computed from one or more ``KNOWN`` values (e.g. unit conversion)."""

    ASSUMED = "assumed"
    """Filled in from an ontology/type default because no evidence existed."""

    UNRESOLVED = "unresolved"
    """Required by the schema but could not be determined at all."""


@dataclass(frozen=True, slots=True)
class Attribute:
    """A single named, typed, provenance-tagged value on a scene node.

    Parameters
    ----------
    name:
        Attribute key, unique within the owning node (e.g. ``"mass_kg"``).
    value:
        The attribute payload. ``None`` is only valid when ``provenance``
        is :attr:`Provenance.UNRESOLVED`.
    provenance:
        Evidential source of ``value``.
    confidence:
        Calibrated confidence in ``[0.0, 1.0]`` that ``value`` is correct.
    source:
        Free-form description of where the value came from (e.g. a regex
        name, an ontology default table, or a upstream node id), used for
        explainability reporting.
    """

    name: str
    value: Any
    provenance: Provenance = Provenance.ASSUMED
    confidence: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Attribute.name must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Attribute.confidence must be within [0, 1], got {self.confidence}"
            )
        if self.value is None and self.provenance is not Provenance.UNRESOLVED:
            raise ValueError(
                f"Attribute {self.name!r} has value=None but provenance="
                f"{self.provenance!r}; only UNRESOLVED attributes may be valueless"
            )


@dataclass(slots=True)
class AttributeSet:
    """An ordered, name-keyed collection of :class:`Attribute` values."""

    _items: dict[str, Attribute] = field(default_factory=dict)

    def set(self, attribute: Attribute) -> None:
        """Insert or overwrite an attribute by name."""
        self._items[attribute.name] = attribute

    def get(self, name: str) -> Attribute | None:
        """Return the attribute named ``name``, or ``None`` if absent."""
        return self._items.get(name)

    def value_of(self, name: str, default: Any = None) -> Any:
        """Return the raw value of attribute ``name``, or ``default`` if absent."""
        attr = self._items.get(name)
        return attr.value if attr is not None else default

    def has(self, name: str) -> bool:
        """Return ``True`` if an attribute named ``name`` is present."""
        return name in self._items

    def remove(self, name: str) -> None:
        """Remove the attribute named ``name`` if present; no-op otherwise."""
        self._items.pop(name, None)

    def names(self) -> list[str]:
        """Return attribute names in insertion order."""
        return list(self._items.keys())

    def by_provenance(self, provenance: Provenance) -> list[Attribute]:
        """Return all attributes tagged with the given ``provenance``."""
        return [a for a in self._items.values() if a.provenance is provenance]

    def unresolved(self) -> list[Attribute]:
        """Return all attributes whose value could not be determined."""
        return self.by_provenance(Provenance.UNRESOLVED)

    def mean_confidence(self) -> float:
        """Return the mean confidence across all attributes, or ``1.0`` if empty."""
        if not self._items:
            return 1.0
        return sum(a.confidence for a in self._items.values()) / len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize to a plain JSON-compatible ``dict``."""
        return {
            attr.name: {
                "value": attr.value,
                "provenance": attr.provenance.value,
                "confidence": attr.confidence,
                "source": attr.source,
            }
            for attr in self._items.values()
        }

    @staticmethod
    def from_dict(payload: dict[str, dict[str, Any]]) -> "AttributeSet":
        """Deserialize from the structure produced by :meth:`to_dict`."""
        attribute_set = AttributeSet()
        for name, fields in payload.items():
            attribute_set.set(
                Attribute(
                    name=name,
                    value=fields.get("value"),
                    provenance=Provenance(fields.get("provenance", Provenance.ASSUMED.value)),
                    confidence=float(fields.get("confidence", 1.0)),
                    source=str(fields.get("source", "")),
                )
            )
        return attribute_set
