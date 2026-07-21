"""Deterministic, human-readable id generation for scene graph elements.

Every node and edge in the :class:`~physworldlm.scene_graph.graph.SceneGraph`
is addressed by a short, prefixed, collision-resistant string id rather than
an opaque UUID, so that dumped IR files remain diffable and readable during
debugging (mirroring the ``e_<label>_<hex4>`` convention already used by the
``PromptParser`` entity ids upstream).
"""

from __future__ import annotations

import re
import uuid

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slugify(text: str) -> str:
    """Normalise arbitrary text into a lowercase, underscore-separated slug."""
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug or "node"


def new_node_id(kind: str, name: str) -> str:
    """Generate a new unique node id of the form ``n_<kind>_<slug>_<hex8>``."""
    return f"n_{_slugify(kind)}_{_slugify(name)}_{uuid.uuid4().hex[:8]}"


def new_edge_id(kind: str, source_id: str, target_id: str) -> str:
    """Generate a new unique edge id of the form ``x_<kind>_<hex8>``."""
    return f"x_{_slugify(kind)}_{uuid.uuid4().hex[:8]}"


def is_valid_id(candidate: str, *, prefix: str) -> bool:
    """Return ``True`` if ``candidate`` looks like a well-formed id with ``prefix``."""
    return bool(candidate) and candidate.startswith(f"{prefix}_")
