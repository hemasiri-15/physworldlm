"""
kit_locator.py
══════════════════════════════════════════════════════════════════════════
Discovers a usable NVIDIA Omniverse Kit executable on the local machine.

This module does exactly one job: given zero or more hints (an explicit
path, an environment variable, or nothing at all), find a Kit executable
on disk and describe it. It has no knowledge of PhysWorldLM, WorldSpec,
USD, or the connector that consumes it -- it is a pure filesystem probe,
intentionally kept separate so it can be tested and reused on its own.

No ``carb``/``omni`` imports and no dependency on a running Kit process:
everything here is plain ``pathlib``/``os`` filesystem inspection.

Search strategy
----------------
1. An explicit path passed by the caller (file or directory).
2. The ``PHYSWORLDLM_KIT_PATH`` environment variable (file or directory).
3. The Packman dependency cache -- where a bare ``kit-kernel`` dependency
   (pulled directly by an app's ``.kit``/``deps.packman.xml``, as on a
   headless build/DGX box with no Omniverse Launcher installed) actually
   lands:

     - ``~/.cache/packman/chk/**/kit-kernel/**/kit``

   This is searched *first* among the automatic strategies, since it's
   the common case for CI/server/DGX deployments that never ran the
   Launcher.
4. A fixed set of well-known Omniverse Launcher / app install locations
   on Linux, used as a fallback:

     - ``~/.local/share/ov/pkg/<app>-<version>/``
     - ``~/.nvidia-omniverse/pkg/<app>-<version>/``
     - ``/opt/nvidia/omniverse/pkg/<app>-<version>/``
     - ``/opt/ov/pkg/<app>-<version>/``

   Every immediate subdirectory of these roots is treated as a candidate
   Kit app install; ``<candidate>/kit`` or ``<candidate>/kit/kit`` is
   probed for an executable named ``kit`` or ``kit.sh``. When several
   installs are found within one strategy, the one with the highest
   parsed version wins.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

logger = logging.getLogger("physworldlm.omniverse.kit_locator")

ENV_VAR = "PHYSWORLDLM_KIT_PATH"

_KIT_EXECUTABLE_NAMES: tuple[str, ...] = ("kit", "kit.sh")

# Packman-cache / Omniverse Launcher install roots. Hardcoded per design
# constraints -- these are the only fixed paths this module knows about;
# everything under them is discovered dynamically.
_DEFAULT_PACKMAN_CACHE_ROOT: Path = Path.home() / ".cache" / "packman" / "chk"

_DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = (
    Path.home() / ".cache" / "packman" / "chk",
    Path.home() / ".local" / "share" / "ov" / "pkg",
    Path.home() / ".nvidia-omniverse" / "pkg",
    Path("/opt/nvidia/omniverse/pkg"),
    Path("/opt/ov/pkg"),
)

# Packman-cache hashes/version dirs nest arbitrarily; bound how deep we
# walk so a large, unrelated cache directory can't make discovery slow.
_PACKMAN_MAX_DEPTH = 6

_VERSION_RE = re.compile(r"(\d+(?:\.\d+)+)")


class KitDiscoveryError(Exception):
    """Raised when no usable Kit executable can be located or validated."""


@dataclass(frozen=True)
class KitInstallation:
    """A located, validated Kit installation.

    Attributes:
        executable: Absolute path to the executable Kit launcher
            (typically ``.../kit/kit.sh`` or ``.../kit/kit``).
        root: Directory the executable was found in.
        version: Best-effort version string parsed from the install
            path (e.g. "105.1.2"), or ``None`` if it couldn't be
            determined.
    """

    executable: Path
    root: Path
    version: Optional[str]


class KitLocator:
    """Locates an Omniverse Kit executable using a fixed, ordered strategy.

    Stateless aside from the configured search roots -- safe to share a
    single instance across multiple :class:`OmniverseConnector` objects.
    """

    def __init__(
        self,
        search_roots: Optional[Sequence[Path]] = None,
        packman_cache_root: Optional[Path] = _DEFAULT_PACKMAN_CACHE_ROOT,
    ) -> None:
        """Initialize the locator.

        Args:
            search_roots: Override the default set of Launcher install-root
                directories to scan. Primarily useful for tests.
            packman_cache_root: Override the Packman dependency-cache root
                (``~/.cache/packman/chk`` by default) searched for a bare
                ``kit-kernel`` deployment. Pass ``None`` to skip this
                strategy entirely.
        """
        self.search_roots: tuple[Path, ...] = (
            tuple(Path(r) for r in search_roots) if search_roots else _DEFAULT_SEARCH_ROOTS
        )
        self.packman_cache_root: Optional[Path] = Path(packman_cache_root) if packman_cache_root else None

    def locate(self, explicit_path: Optional[Path] = None) -> KitInstallation:
        """Find a Kit installation, preferring more specific hints first.

        Args:
            explicit_path: A caller-supplied file or directory to check
                before falling back to the environment variable and the
                default search roots.

        Returns:
            The chosen :class:`KitInstallation`.

        Raises:
            KitDiscoveryError: If ``explicit_path`` was given but does
                not resolve to a valid Kit executable, or if no
                installation can be found anywhere.
        """
        if explicit_path is not None:
            return self._from_path(Path(explicit_path))

        env_value = os.environ.get(ENV_VAR)
        if env_value:
            logger.debug("Using Kit path from %s=%s", ENV_VAR, env_value)
            return self._from_path(Path(env_value))

        candidates = list(self._discover_packman())
        if not candidates:
            candidates = list(self._discover_legacy())
        if not candidates:
            raise KitDiscoveryError(
                "No Omniverse Kit installation found under "
                f"'{self.packman_cache_root}' or {[str(r) for r in self.search_roots]}. "
                f"Set the {ENV_VAR} environment variable, or pass kit_executable "
                "explicitly to OmniverseConnector."
            )
        candidates.sort(key=lambda c: self._version_sort_key(c.version), reverse=True)
        chosen = candidates[0]
        logger.info(
            "Discovered %d Kit installation(s); selected '%s' (version=%s).",
            len(candidates), chosen.executable, chosen.version,
        )
        return chosen

    # ── internals ────────────────────────────────────────────────────

    def _from_path(self, path: Path) -> KitInstallation:
        exe = self._resolve_executable(path)
        if exe is None:
            raise KitDiscoveryError(
                f"'{path}' is not a Kit executable and no '{_KIT_EXECUTABLE_NAMES}' "
                "was found inside it."
            )
        return KitInstallation(executable=exe, root=exe.parent, version=self._extract_version(path))

    def _resolve_executable(self, path: Path) -> Optional[Path]:
        """Return an executable Kit binary at or under `path`, if any."""
        if path.is_file():
            return path if os.access(path, os.X_OK) else None
        if not path.is_dir():
            return None
        # Direct child (e.g. path == .../kit/)
        for name in _KIT_EXECUTABLE_NAMES:
            candidate = path / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        # One level deeper (e.g. path == .../kit-app-1.2.3/, exe in ./kit/)
        nested = path / "kit"
        if nested.is_dir():
            for name in _KIT_EXECUTABLE_NAMES:
                candidate = nested / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate

        # Recursive search for Packman cache layouts
        try:
            for exe in path.rglob("kit"):
                if exe.is_file() and os.access(exe, os.X_OK):
                    return exe
        except PermissionError:
            pass

        return None

    def _discover_legacy(self) -> Iterator[KitInstallation]:
        for root in self.search_roots:
            if not root.is_dir():
                continue
            for entry in root.rglob("*"):
                if not entry.is_dir():
                    continue
                exe = self._resolve_executable(entry)
                if exe is not None:
                    yield KitInstallation(executable=exe, root=exe.parent, version=self._extract_version(entry))

    def _discover_packman(self) -> Iterator[KitInstallation]:
        """Find a bare `kit-kernel` deployment under the Packman dependency cache.

        Packman-managed caches nest an unpredictable, hashed directory
        structure (``chk/<hash>/kit-kernel/<version>/...``), so this walks
        the tree looking for a directory literally named ``kit-kernel``
        and then probes beneath it, rather than assuming a fixed layout.
        Depth is bounded (`_PACKMAN_MAX_DEPTH`) so an unrelated, large
        cache directory can't make discovery slow.
        """
        root = self.packman_cache_root
        if root is None or not root.is_dir():
            return
        for kit_kernel_dir in self._find_dirs_named(root, "kit-kernel", _PACKMAN_MAX_DEPTH):
            for version_entry in sorted(kit_kernel_dir.glob("*")):
                if not version_entry.is_dir():
                    continue
                exe = self._resolve_executable(version_entry)
                if exe is not None:
                    yield KitInstallation(
                        executable=exe, root=exe.parent, version=self._extract_version(version_entry)
                    )

    @classmethod
    def _find_dirs_named(cls, root: Path, name: str, max_depth: int) -> Iterator[Path]:
        """Depth-bounded search for directories called `name` under `root`."""
        if max_depth < 0:
            return
        try:
            entries = list(root.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            if entry.name == name:
                yield entry
                continue  # kit-kernel dirs aren't nested inside each other
            yield from cls._find_dirs_named(entry, name, max_depth - 1)

    @staticmethod
    def _extract_version(path: Path) -> Optional[str]:
        match = _VERSION_RE.search(path.name)
        return match.group(1) if match else None

    @staticmethod
    def _version_sort_key(version: Optional[str]) -> tuple[int, ...]:
        if not version:
            return (0,)
        parts: list[int] = []
        for chunk in version.split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(0)
        return tuple(parts)


def iter_search_roots() -> Iterable[Path]:
    """Expose the default search roots for diagnostics/CLI tooling."""
    return _DEFAULT_SEARCH_ROOTS


__all__ = [
    "KitLocator",
    "KitInstallation",
    "KitDiscoveryError",
    "ENV_VAR",
    "iter_search_roots",
]
