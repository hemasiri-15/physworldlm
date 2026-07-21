"""
asset_server.py
══════════════════════════════════════════════════════════════════════════
Asset registry, cache, and delivery layer for the Omniverse Connector
layer of PhysWorldLM.

Pipeline position
------------------
    Natural Language → Ontology → WorldSpec → Scene Compiler → scene.usda
                                                                      │
              ┌───────────────────────────────────────────────────────┤
              │                                                       ▼
    ┌────────────────────┐                                ┌────────────────────┐
    │  asset_server.py    │◄──── asset lookups / caching ──┤  usd_exporter.py    │
    │  (this module)      │                                └────────────────────┘
    └──────────┬───────────┘
               │ resolved local paths / URIs
               ▼
    ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
    │  app_launcher.py     │   │  stage_manager.py    │   │  USDLoader           │
    └────────────────────┘   └────────────────────┘   └────────────────────┘

Scope
-----
This module owns exactly one concern: everything to do with *assets* --
meshes, materials, textures, skeletons, animations, USD references,
terrain sources, and any other file-backed content PhysWorldLM's scenes
are composed from. It maintains a registry of known assets, indexes and
searches them, resolves them to a local, on-disk path regardless of
where they actually live (local disk, a Nucleus server, or a remote
HTTP(S) endpoint), caches and invalidates that local copy, verifies
integrity via checksums, resolves inter-asset dependencies, and reports
statistics/diagnostics about the whole registry.

This module explicitly does NOT:
    * launch Omniverse Kit / Isaac Sim (``app_launcher.OmniverseLauncher``
      owns process lifecycle)
    * open, create, or otherwise manage a USD *Stage*
      (``stage_manager.StageManager`` owns stage lifecycle)
    * simulate physics (``physics_scene.py``)
    * render frames (``renderer.py``)
    * parse natural language, ontologies, or ``WorldSpec`` objects, or
      compile/export USD content (those are upstream compiler stages)
    * spawn entities, sensors, robots, or terrain content into a stage
      (those components look assets up here, then author them
      themselves via whatever prim/path a live ``StageManager`` gives
      them)

Those concerns belong to ``app_launcher.py``, ``stage_manager.py``,
``physics_scene.py``, ``renderer.py``, and the domain-specific
compiler/exporter stages upstream of Omniverse. This module is imported
by none of them, and imports none of them -- it hands callers a
resolved local path or URI (see :meth:`AssetServer.resolve_for_stage`)
and consumes nothing from them in return.

Design constraints
-------------------
    * No ``omni``/``pxr`` import happens at module load time. The only
      such dependency this module ever needs -- ``omni.client``, for
      Nucleus transfers -- is deferred to the call site that actually
      needs it, behind :func:`_lazy_import`, so this module loads and
      is fully usable (for local-asset workflows) on a machine with no
      Omniverse installation at all.
    * All failure modes raise a documented, specific
      :class:`AssetServerError` subclass. Nothing lets a raw
      ``ImportError``, ``OSError``, or opaque network stack trace
      escape uncaught.
    * No global mutable state. Every piece of runtime state (the asset
      registry, alias/tag indices, cache bookkeeping, download/upload
      history, watch registrations, ...) lives on the
      :class:`AssetServer` instance, guarded by an internal lock, so
      multiple independent servers (e.g. in tests, or one per scene)
      never interfere with one another.
    * Dependency resolution and cycle detection operate purely on
      registered asset IDs; this module never inspects file contents
      to *discover* dependencies (e.g. parsing a USD file's sublayers)
      -- dependencies are declared explicitly at :meth:`register_asset`
      time by the caller (typically the Scene Compiler).
    * Designed to be handed to, and consumed by, both
      ``stage_manager.StageManager`` (which authors references at the
      paths this module resolves) and a ``USDLoader``-style component
      (which reads resolved local paths directly) without either being
      imported here.

Public API
----------
    server = AssetServer(cache_dir="./.cache/assets")
    server.initialize()
    server.register_asset("mesh.crate", kind=AssetKind.MESH,
                           source=AssetSource.LOCAL, uri="assets/crate.usd")
    path = server.load_asset("mesh.crate")
    server.shutdown()

Or, as a context manager::

    with AssetServer() as server:
        server.register_asset(...)
        path = server.load_asset(...)
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.asset_server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class AssetServerError(Exception):
    """Base class for all errors raised by :class:`AssetServer`."""


class NotInitializedError(AssetServerError):
    """Raised when an operation requires :meth:`AssetServer.initialize`
    to have been called successfully first."""


class AlreadyInitializedError(AssetServerError):
    """Raised when :meth:`AssetServer.initialize` is called while
    already initialized."""


class NucleusImportError(AssetServerError):
    """Raised when the ``omni.client`` module can't be imported.

    Distinct from a bare ``ImportError`` so callers can catch exactly
    "Nucleus/Omniverse isn't available here" without accidentally
    swallowing an unrelated import bug in their own code. Only raised
    by operations that actually touch a Nucleus (``omniverse://``)
    asset -- local and plain-HTTP(S) workflows never need it.
    """


class AssetRegistrationError(AssetServerError):
    """Raised when registering an asset fails."""


class AssetAlreadyExistsError(AssetRegistrationError):
    """Raised by :meth:`AssetServer.register_asset` when an asset with
    the same ID is already registered and ``overwrite=False``."""


class AssetNotFoundError(AssetServerError):
    """Raised when a referenced asset ID/alias is not in the registry."""


class AssetLoadError(AssetServerError):
    """Raised when resolving/materializing an asset for use fails."""


class AssetUnloadError(AssetServerError):
    """Raised when releasing a loaded asset fails."""


class AssetCacheError(AssetServerError):
    """Raised when caching (or clearing the cache for) an asset fails."""


class AssetDownloadError(AssetServerError):
    """Raised when downloading an asset from Nucleus or a remote URL fails."""


class AssetUploadError(AssetServerError):
    """Raised when uploading an asset to Nucleus fails, or is requested
    for a source that does not support uploads (e.g. plain HTTP(S))."""


class AssetValidationError(AssetServerError):
    """Raised when validation itself cannot be performed (not when an
    asset is merely found invalid -- that is reported via
    :class:`AssetValidationReport` instead)."""


class ChecksumMismatchError(AssetServerError):
    """Raised when a computed checksum does not match the recorded one."""


class DependencyResolutionError(AssetServerError):
    """Raised when an asset's dependency graph cannot be resolved, e.g.
    because of a missing dependency or a circular reference."""


class AssetVersionError(AssetServerError):
    """Raised when an asset version string is malformed or a requested
    version does not match what is registered."""


class ManifestError(AssetServerError):
    """Base class for registry import/export (manifest) failures."""


class RegistryExportError(ManifestError):
    """Raised when :meth:`AssetServer.export_registry` fails."""


class RegistryImportError(ManifestError):
    """Raised when :meth:`AssetServer.import_registry` fails."""


class AssetWatchError(AssetServerError):
    """Raised when registering, starting, or stopping asset watching fails."""


class AssetPackagingError(AssetServerError):
    """Raised when :meth:`AssetServer.package_assets` fails."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class AssetServerState(str, Enum):
    """Lifecycle state of the :class:`AssetServer` instance itself.

    Transitions (happy path)::

        UNINITIALIZED -> INITIALIZING -> READY -> SHUTDOWN

    ``ERROR`` is reachable from ``INITIALIZING`` and is terminal until
    :meth:`AssetServer.shutdown` resets the server.
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class AssetKind(str, Enum):
    """Coarse content category an asset belongs to.

    Used for indexing, search filters, and the convenience lookup
    methods (:meth:`AssetServer.find_material`,
    :meth:`AssetServer.find_texture`, ...). Purely descriptive -- this
    module never inspects an asset's bytes to infer or verify its kind.
    """

    MESH = "mesh"
    MATERIAL = "material"
    TEXTURE = "texture"
    SKELETON = "skeleton"
    ANIMATION = "animation"
    USD_REFERENCE = "usd_reference"
    SCENE = "scene"
    TERRAIN = "terrain"
    AUDIO = "audio"
    SENSOR_PROFILE = "sensor_profile"
    ROBOT = "robot"
    MANIFEST = "manifest"
    OTHER = "other"


class AssetSource(str, Enum):
    """Where an asset's canonical content lives.

    * ``LOCAL`` -- already on disk; ``uri`` is a filesystem path.
    * ``NUCLEUS`` -- lives on an Omniverse Nucleus server; ``uri`` is an
      ``omniverse://`` URL. Downloads/uploads go through ``omni.client``.
    * ``REMOTE`` -- lives behind a plain HTTP(S) URL. Downloads only;
      uploads are not supported for this source.
    * ``GENERATED`` -- produced at runtime by an upstream component
      (e.g. a procedurally generated texture) rather than fetched from
      anywhere; this module tracks its metadata but has nothing to
      download or cache until the generator writes it and re-registers
      it as ``LOCAL``.
    """

    LOCAL = "local"
    NUCLEUS = "nucleus"
    REMOTE = "remote"
    GENERATED = "generated"


class AssetState(str, Enum):
    """Lifecycle state of a single registered asset.

    Transitions (happy path)::

        REGISTERED -> CACHED -> LOADED
                         ^          │
                         └──────────┘  (unload_asset)

    ``STALE`` is reachable from ``CACHED``/``LOADED`` when the source
    changes underneath a cached copy (see :meth:`AssetServer.check_staleness`
    and asset watching). ``ERROR`` is reachable from any state.
    """

    REGISTERED = "registered"
    CACHED = "cached"
    LOADED = "loaded"
    STALE = "stale"
    ERROR = "error"


class ValidationSeverity(str, Enum):
    """Severity of a single :class:`ValidationIssue`."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════

#: Default subdirectory (relative to the current working directory) used
#: for the on-disk asset cache when no explicit ``cache_dir`` is given.
_DEFAULT_CACHE_DIRNAME = ".cache/physworldlm/assets"

#: Default checksum algorithm, must be a name accepted by ``hashlib.new``.
_DEFAULT_CHECKSUM_ALGORITHM = "sha256"

#: Chunk size, in bytes, used while streaming files for checksumming or
#: downloading, to bound peak memory use on large assets.
_STREAM_CHUNK_BYTES = 1024 * 1024

#: Schema version embedded in exported registry manifests.
_MANIFEST_VERSION = "1.0.0"

#: Default polling interval, in seconds, for the background asset-watch pump.
_DEFAULT_WATCH_INTERVAL_SECONDS = 2.0

#: Best-effort file-extension -> AssetKind hints, used only by
#: :meth:`AssetServer.register_asset` when no explicit ``kind`` is given
#: is not supported (kind is always required) -- retained instead as a
#: lookup for :meth:`AssetServer._infer_kind_hint`, a diagnostic helper
#: used by :meth:`AssetServer.validate_asset` to flag likely mismatches.
_EXTENSION_KIND_HINTS: dict[str, AssetKind] = {
    ".usd": AssetKind.USD_REFERENCE,
    ".usda": AssetKind.USD_REFERENCE,
    ".usdc": AssetKind.USD_REFERENCE,
    ".usdz": AssetKind.USD_REFERENCE,
    ".fbx": AssetKind.MESH,
    ".obj": AssetKind.MESH,
    ".gltf": AssetKind.MESH,
    ".glb": AssetKind.MESH,
    ".png": AssetKind.TEXTURE,
    ".jpg": AssetKind.TEXTURE,
    ".jpeg": AssetKind.TEXTURE,
    ".exr": AssetKind.TEXTURE,
    ".hdr": AssetKind.TEXTURE,
    ".mdl": AssetKind.MATERIAL,
    ".wav": AssetKind.AUDIO,
    ".mp3": AssetKind.AUDIO,
    ".json": AssetKind.MANIFEST,
    ".yaml": AssetKind.MANIFEST,
    ".yml": AssetKind.MANIFEST,
}


# ════════════════════════════════════════════════════════════════════════
# Lazy import helper
# ════════════════════════════════════════════════════════════════════════

def _lazy_import(module_name: str, *, hint: str = "") -> Any:
    """Import ``module_name``, raising :class:`NucleusImportError` on failure.

    Every ``omni.*`` import used by this module goes through this
    function so that (a) importing ``asset_server`` itself never
    requires Omniverse to be installed, and (b) a missing dependency
    surfaces as one clear, catchable exception instead of a raw
    ``ImportError``.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = f"Failed to import '{module_name}'."
        if hint:
            message = f"{message} {hint}"
        raise NucleusImportError(message) from exc


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AssetMetadata:
    """Declarative description of a single registered asset.

    This is the part of an asset's record that is persisted by
    :meth:`AssetServer.export_registry` / restored by
    :meth:`AssetServer.import_registry`. Runtime bookkeeping (cache
    path, load count, last error, ...) lives on the enclosing
    :class:`AssetRecord` instead.

    Attributes:
        asset_id: Stable, unique identifier for this asset within the
            registry (e.g. ``"mesh.crate_01"``).
        kind: Coarse content category.
        source: Where the canonical content lives.
        uri: Location of the canonical content -- a filesystem path for
            ``LOCAL``, an ``omniverse://`` URL for ``NUCLEUS``, an
            ``http(s)://`` URL for ``REMOTE``, or an arbitrary
            caller-defined tag for ``GENERATED``.
        version: Free-form version string (default ``"1.0.0"``). This
            module does not implement semver range resolution -- version
            matching, where used, is exact-string only.
        checksum: Recorded content checksum, or ``None`` if not yet
            computed/known.
        checksum_algorithm: Name of the algorithm ``checksum`` was
            computed with (must be accepted by ``hashlib.new``).
        size_bytes: Recorded content size, or ``None`` if unknown.
        tags: Free-form semantic tags (e.g. ``{"outdoor", "urban"}``)
            used by :meth:`AssetServer.search_assets`.
        aliases: Additional names this asset can be looked up by, in
            addition to ``asset_id``.
        dependencies: IDs of other registered assets this one requires
            (e.g. a mesh's material, a material's textures).
        lod_paths: Mapping of LOD level (``0`` = highest detail) to a
            URI override for that level. Level ``0`` need not be
            present if ``uri`` itself already represents the highest
            detail level.
        extra: Arbitrary caller-defined metadata not modeled explicitly
            above (e.g. author, license, source project).
        created_at: Wall-clock timestamp this asset was first registered.
        updated_at: Wall-clock timestamp this metadata was last modified.
    """

    asset_id: str
    kind: AssetKind
    source: AssetSource
    uri: str
    version: str = "1.0.0"
    checksum: Optional[str] = None
    checksum_algorithm: str = _DEFAULT_CHECKSUM_ALGORITHM
    size_bytes: Optional[int] = None
    tags: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)
    lod_paths: dict[int, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this metadata."""
        return {
            "asset_id": self.asset_id,
            "kind": self.kind.value,
            "source": self.source.value,
            "uri": self.uri,
            "version": self.version,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "size_bytes": self.size_bytes,
            "tags": sorted(self.tags),
            "aliases": sorted(self.aliases),
            "dependencies": sorted(self.dependencies),
            "lod_paths": {str(level): path for level, path in self.lod_paths.items()},
            "extra": dict(self.extra),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssetMetadata":
        """Reconstruct :class:`AssetMetadata` from :meth:`to_dict` output.

        Raises:
            ManifestError: If required fields are missing or malformed.
        """
        try:
            return cls(
                asset_id=str(payload["asset_id"]),
                kind=AssetKind(payload["kind"]),
                source=AssetSource(payload["source"]),
                uri=str(payload["uri"]),
                version=str(payload.get("version", "1.0.0")),
                checksum=payload.get("checksum"),
                checksum_algorithm=str(payload.get("checksum_algorithm", _DEFAULT_CHECKSUM_ALGORITHM)),
                size_bytes=payload.get("size_bytes"),
                tags=set(payload.get("tags", []) or []),
                aliases=set(payload.get("aliases", []) or []),
                dependencies=set(payload.get("dependencies", []) or []),
                lod_paths={int(level): path for level, path in (payload.get("lod_paths") or {}).items()},
                extra=dict(payload.get("extra", {}) or {}),
                created_at=float(payload.get("created_at", time.time())),
                updated_at=float(payload.get("updated_at", time.time())),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ManifestError(f"Malformed asset metadata entry: {exc}") from exc


@dataclass
class AssetRecord:
    """Runtime record for a registered asset: metadata plus live state.

    Attributes:
        metadata: The asset's declarative :class:`AssetMetadata`.
        state: Current lifecycle state.
        local_cache_path: Resolved local filesystem path, once cached
            (``None`` beforehand, or for ``GENERATED`` assets that have
            not yet been produced).
        load_count: Number of times :meth:`AssetServer.load_asset` has
            successfully completed for this asset since it was
            registered (never reset by :meth:`AssetServer.unload_asset`).
        last_loaded_at: Wall-clock timestamp of the most recent
            successful load, or ``None``.
        last_error: Human-readable description of the most recent
            failure for this asset, or ``None``.
        source_mtime: Last-observed modification time of the source
            file, used by the staleness/watch machinery. ``None`` for
            non-``LOCAL`` sources or before first check.
        watch_enabled: Whether hot-reload watching is enabled for this
            asset.
    """

    metadata: AssetMetadata
    state: AssetState = AssetState.REGISTERED
    local_cache_path: Optional[Path] = None
    load_count: int = 0
    last_loaded_at: Optional[float] = None
    last_error: Optional[str] = None
    source_mtime: Optional[float] = None
    watch_enabled: bool = False


@dataclass(frozen=True)
class ValidationIssue:
    """A single finding from :meth:`AssetServer.validate_asset`.

    Attributes:
        severity: How serious the finding is.
        message: Human-readable description.
        asset_id: The asset the issue concerns.
    """

    severity: ValidationSeverity
    message: str
    asset_id: str


@dataclass
class AssetValidationReport:
    """Aggregate result of :meth:`AssetServer.validate_asset`.

    Attributes:
        issues: All findings, in the order they were discovered.
        generated_at: Wall-clock timestamp this report was produced.
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """True if no issue at :attr:`ValidationSeverity.ERROR` was found."""
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this report."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "generated_at": self.generated_at,
            "issues": [
                {"severity": issue.severity.value, "message": issue.message, "asset_id": issue.asset_id}
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class AssetStatistics:
    """Snapshot of aggregate statistics about the registry.

    Attributes:
        total_assets: Total number of registered assets.
        by_kind: Count of assets per :class:`AssetKind` value.
        by_source: Count of assets per :class:`AssetSource` value.
        by_state: Count of assets per :class:`AssetState` value.
        cached_count: Number of assets with a resolved local cache copy.
        loaded_count: Number of assets currently in the ``LOADED`` state.
        total_cache_bytes: Sum of on-disk sizes of all cached files this
            server knows about (best-effort; ``0`` if none could be
            determined).
        total_registered_bytes: Sum of :attr:`AssetMetadata.size_bytes`
            across all assets that have a recorded size.
    """

    total_assets: int
    by_kind: dict[str, int]
    by_source: dict[str, int]
    by_state: dict[str, int]
    cached_count: int
    loaded_count: int
    total_cache_bytes: int
    total_registered_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of these statistics."""
        return asdict(self)


@dataclass(frozen=True)
class DownloadRecord:
    """Record of a single completed or failed download.

    Attributes:
        asset_id: The asset that was downloaded.
        source_uri: The URI it was downloaded from.
        destination: The local path it was written to.
        bytes_downloaded: Number of bytes written, or ``0`` on failure.
        started_at: Wall-clock timestamp the download began.
        completed_at: Wall-clock timestamp the download finished
            (successfully or not).
        success: Whether the download completed successfully.
    """

    asset_id: str
    source_uri: str
    destination: Path
    bytes_downloaded: int
    started_at: float
    completed_at: float
    success: bool


@dataclass(frozen=True)
class UploadRecord:
    """Record of a single completed or failed upload.

    Attributes:
        asset_id: The asset that was uploaded.
        destination_uri: The URI it was uploaded to.
        bytes_uploaded: Number of bytes sent, or ``0`` on failure.
        started_at: Wall-clock timestamp the upload began.
        completed_at: Wall-clock timestamp the upload finished
            (successfully or not).
        success: Whether the upload completed successfully.
    """

    asset_id: str
    destination_uri: str
    bytes_uploaded: int
    started_at: float
    completed_at: float
    success: bool


# ════════════════════════════════════════════════════════════════════════
# Internal: background asset-watch pump
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _WatchPump:
    """Background thread that periodically polls watched assets for
    source changes and invokes registered callbacks.

    Kept private and minimal: it knows nothing about assets itself, it
    only calls ``poll_fn`` on an interval, mirroring the event-loop pump
    pattern used by ``app_launcher.OmniverseLauncher`` so both modules'
    background-thread lifecycle looks and behaves the same way.
    """

    interval_seconds: float
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self, poll_fn: Callable[[], None]) -> None:
        """Start polling ``poll_fn`` on a daemon thread, if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _run() -> None:
            while not self._stop_event.is_set():
                try:
                    poll_fn()
                except Exception:  # noqa: BLE001 - never let the pump die silently
                    logger.exception("Asset watch pump: poll() raised; continuing.")
                self._stop_event.wait(self.interval_seconds)

        self._thread = threading.Thread(
            target=_run, name="physworldlm-asset-watch", daemon=True
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        """Signal the pump to stop and wait briefly for it to exit."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=join_timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ════════════════════════════════════════════════════════════════════════
# AssetServer
# ════════════════════════════════════════════════════════════════════════

class AssetServer:
    """Registers, resolves, caches, and validates every asset PhysWorldLM uses.

    ``AssetServer`` sits alongside (never above or below)
    ``app_launcher.OmniverseLauncher`` and ``stage_manager.StageManager``
    in the Omniverse connector layer: it contains no scene-authoring or
    process-lifecycle logic of its own. Its only job is turning an asset
    ID into a trustworthy local path, and keeping the bookkeeping
    (registry, cache, checksums, dependencies, statistics) around that
    consistent.

    Thread-safety: all state-mutating operations are guarded by an
    internal lock, so this server is safe to call from multiple threads
    (e.g. the watch pump running alongside interactive registrations).

    Example:
        >>> server = AssetServer(cache_dir="./.cache/assets")
        >>> server.initialize()
        >>> server.register_asset(
        ...     "mesh.crate", kind=AssetKind.MESH,
        ...     source=AssetSource.LOCAL, uri="assets/crate.usd",
        ... )
        >>> path = server.load_asset("mesh.crate")
        >>> server.shutdown()
    """

    def __init__(
        self,
        *,
        cache_dir: Optional["str | Path"] = None,
        checksum_algorithm: str = _DEFAULT_CHECKSUM_ALGORITHM,
        nucleus_server: Optional[str] = None,
        asset_search_paths: Optional[list["str | Path"]] = None,
    ) -> None:
        """Create a server. Does not touch disk or Omniverse yet.

        Args:
            cache_dir: Directory downloaded/materialized assets are
                cached under. Defaults to
                ``.cache/physworldlm/assets`` (relative to the current
                working directory). Created on :meth:`initialize`.
            checksum_algorithm: Default algorithm (accepted by
                ``hashlib.new``) used when computing checksums without
                an explicit override.
            nucleus_server: Default Nucleus server URL used to resolve
                relative ``omniverse://`` asset URIs. Purely a
                convenience default -- fully qualified URIs never need
                it.
            asset_search_paths: Ordered list of directories searched,
                in addition to a URI's own path, when resolving a
                ``LOCAL`` asset that is not found at its recorded path
                verbatim (e.g. a relative path recorded against a
                different working directory).
        """
        self._lock = threading.RLock()
        self._state: AssetServerState = AssetServerState.UNINITIALIZED
        self._assets: dict[str, AssetRecord] = {}
        self._alias_index: dict[str, str] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._cache_dir: Path = Path(cache_dir) if cache_dir is not None else Path(_DEFAULT_CACHE_DIRNAME)
        self._checksum_algorithm = checksum_algorithm
        self._nucleus_server = nucleus_server
        self._asset_search_paths: list[Path] = [Path(p) for p in (asset_search_paths or [])]
        self._downloads: list[DownloadRecord] = []
        self._uploads: list[UploadRecord] = []
        self._watch_callbacks: dict[str, list[Callable[[str], None]]] = {}
        self._watch_pump: Optional[_WatchPump] = None
        self._watch_interval_seconds = _DEFAULT_WATCH_INTERVAL_SECONDS
        self._auto_reload_on_watch = False
        self._last_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> AssetServerState:
        """Current lifecycle state of this server."""
        with self._lock:
            return self._state

    def is_initialized(self) -> bool:
        """Whether this server has completed :meth:`initialize`."""
        return self.state is AssetServerState.READY

    @property
    def cache_dir(self) -> Path:
        """The directory this server caches materialized assets under."""
        with self._lock:
            return self._cache_dir

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that most recently drove this server into ``ERROR``, if any."""
        with self._lock:
            return self._last_error

    @property
    def downloads(self) -> list[DownloadRecord]:
        """All downloads performed by this server instance, oldest first."""
        with self._lock:
            return list(self._downloads)

    @property
    def uploads(self) -> list[UploadRecord]:
        """All uploads performed by this server instance, oldest first."""
        with self._lock:
            return list(self._uploads)

    def _require_ready(self) -> None:
        if self._state is not AssetServerState.READY:
            raise NotInitializedError(
                f"AssetServer is not initialized (state='{self._state.value}'). "
                "Call initialize() first."
            )

    # ------------------------------------------------------------------
    # Lifecycle: initialize / shutdown
    # ------------------------------------------------------------------

    def initialize(
        self,
        *,
        create_cache_dir: bool = True,
        registry_path: Optional["str | Path"] = None,
    ) -> None:
        """Bring this server up: create the cache directory and, optionally,
        restore a previously exported registry.

        Args:
            create_cache_dir: If True (default), create :attr:`cache_dir`
                (and any missing parents) if it does not already exist.
            registry_path: If given and the file exists, imported via
                :meth:`import_registry` after the cache directory is
                ready.

        Raises:
            AlreadyInitializedError: If already initialized.
            NotInitializedError: Never raised here; listed for symmetry
                with other lifecycle methods.
        """
        with self._lock:
            if self._state is AssetServerState.READY:
                raise AlreadyInitializedError(
                    "AssetServer is already initialized; call shutdown() first."
                )
            self._state = AssetServerState.INITIALIZING
            self._last_error = None

        logger.info("Initializing AssetServer (cache_dir='%s').", self._cache_dir)
        try:
            if create_cache_dir:
                self._cache_dir.mkdir(parents=True, exist_ok=True)

            with self._lock:
                self._state = AssetServerState.READY

            if registry_path is not None and Path(registry_path).exists():
                count = self.import_registry(registry_path, merge=True)
                logger.info("Restored %d asset(s) from '%s'.", count, registry_path)

        except AssetServerError as exc:
            with self._lock:
                self._state = AssetServerState.ERROR
                self._last_error = exc
            logger.error("AssetServer initialization failed: %s", exc)
            raise
        except Exception as exc:  # noqa: BLE001 - never leak an opaque OS error
            wrapped = AssetServerError(f"Unexpected failure during initialize(): {exc}")
            with self._lock:
                self._state = AssetServerState.ERROR
                self._last_error = wrapped
            logger.error("AssetServer initialization failed: %s", wrapped)
            raise wrapped from exc

        logger.info("AssetServer ready (%d asset(s) registered).", len(self._assets))

    def shutdown(self, *, save_registry_path: Optional["str | Path"] = None) -> None:
        """Stop watching, optionally persist the registry, and reset state.

        Idempotent: calling ``shutdown()`` when not initialized logs and
        returns rather than raising. Never deletes the on-disk cache --
        callers that want that should call :meth:`clear_cache` first.

        Args:
            save_registry_path: If given, :meth:`export_registry` is
                called before the server's in-memory state is cleared.

        Raises:
            RegistryExportError: If ``save_registry_path`` is given and
                the export fails.
        """
        with self._lock:
            if self._state in (AssetServerState.UNINITIALIZED, AssetServerState.SHUTDOWN):
                logger.info("shutdown() called with nothing initialized; nothing to do.")
                return

        self.stop_watching()

        if save_registry_path is not None:
            self.export_registry(save_registry_path)

        with self._lock:
            self._assets.clear()
            self._alias_index.clear()
            self._tag_index.clear()
            self._watch_callbacks.clear()
            self._state = AssetServerState.SHUTDOWN
        logger.info("AssetServer shut down.")

    def __enter__(self) -> "AssetServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal: lookup helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, identifier: str) -> str:
        """Resolve ``identifier`` (an asset ID or alias) to a canonical asset ID."""
        if identifier in self._assets:
            return identifier
        resolved = self._alias_index.get(identifier)
        if resolved is not None:
            return resolved
        raise AssetNotFoundError(f"No asset registered with ID or alias '{identifier}'.")

    def _get_record(self, identifier: str) -> AssetRecord:
        self._require_ready()
        with self._lock:
            asset_id = self._resolve_id(identifier)
            return self._assets[asset_id]

    def _index_asset(self, record: AssetRecord) -> None:
        """Add ``record`` to the alias/tag indices. Caller holds the lock."""
        for alias in record.metadata.aliases:
            self._alias_index[alias] = record.metadata.asset_id
        for tag in record.metadata.tags:
            self._tag_index.setdefault(tag, set()).add(record.metadata.asset_id)

    def _deindex_asset(self, record: AssetRecord) -> None:
        """Remove ``record`` from the alias/tag indices. Caller holds the lock."""
        for alias in record.metadata.aliases:
            if self._alias_index.get(alias) == record.metadata.asset_id:
                del self._alias_index[alias]
        for tag in record.metadata.tags:
            members = self._tag_index.get(tag)
            if members is not None:
                members.discard(record.metadata.asset_id)
                if not members:
                    del self._tag_index[tag]

    def _infer_kind_hint(self, uri: str) -> Optional[AssetKind]:
        """Best-effort extension -> :class:`AssetKind` hint, for diagnostics only."""
        suffix = Path(uri).suffix.lower()
        return _EXTENSION_KIND_HINTS.get(suffix)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_asset(
        self,
        asset_id: str,
        *,
        kind: AssetKind,
        source: AssetSource,
        uri: str,
        version: str = "1.0.0",
        tags: Optional[set[str]] = None,
        aliases: Optional[set[str]] = None,
        dependencies: Optional[set[str]] = None,
        checksum: Optional[str] = None,
        checksum_algorithm: Optional[str] = None,
        lod_paths: Optional[dict[int, str]] = None,
        extra: Optional[dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> AssetMetadata:
        """Register a new asset (or overwrite an existing one) in the registry.

        Args:
            asset_id: Stable, unique identifier for this asset.
            kind: Coarse content category.
            source: Where the canonical content lives.
            uri: Location of the canonical content.
            version: Free-form version string.
            tags: Semantic tags for search.
            aliases: Additional lookup names.
            dependencies: IDs of other registered (or not-yet-registered)
                assets this one requires. Existence is checked lazily,
                at :meth:`resolve_dependencies` / :meth:`load_asset`
                time, not at registration time, so dependency order
                does not matter.
            checksum: Known content checksum, if already computed.
            checksum_algorithm: Algorithm ``checksum`` was computed
                with. Defaults to this server's configured default.
            lod_paths: LOD level -> URI override mapping.
            extra: Arbitrary caller-defined metadata.
            overwrite: If True, silently replace an existing asset with
                the same ID instead of raising.

        Returns:
            The stored :class:`AssetMetadata`.

        Raises:
            NotInitializedError: If not initialized.
            AssetAlreadyExistsError: If ``asset_id`` is already
                registered and ``overwrite`` is False.
            AssetRegistrationError: If any given alias is already used
                by a different asset.
        """
        self._require_ready()
        metadata = AssetMetadata(
            asset_id=asset_id,
            kind=kind,
            source=source,
            uri=uri,
            version=version,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm or self._checksum_algorithm,
            tags=set(tags or ()),
            aliases=set(aliases or ()),
            dependencies=set(dependencies or ()),
            lod_paths=dict(lod_paths or {}),
            extra=dict(extra or {}),
        )

        with self._lock:
            if asset_id in self._assets and not overwrite:
                raise AssetAlreadyExistsError(
                    f"Asset '{asset_id}' is already registered; pass overwrite=True to replace it."
                )
            for alias in metadata.aliases:
                owner = self._alias_index.get(alias)
                if owner is not None and owner != asset_id:
                    raise AssetRegistrationError(
                        f"Alias '{alias}' is already used by asset '{owner}'."
                    )

            existing = self._assets.get(asset_id)
            if existing is not None:
                self._deindex_asset(existing)

            record = AssetRecord(metadata=metadata)
            self._assets[asset_id] = record
            self._index_asset(record)

        logger.info("Registered asset '%s' (kind=%s, source=%s).", asset_id, kind.value, source.value)
        return metadata

    def unregister_asset(self, identifier: str, *, remove_cache: bool = False) -> None:
        """Remove an asset from the registry.

        Args:
            identifier: Asset ID or alias.
            remove_cache: If True, also delete its cached local file (if
                any) from disk.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If no such asset is registered.
            AssetCacheError: If ``remove_cache`` is True and deletion fails.
        """
        self._require_ready()
        with self._lock:
            asset_id = self._resolve_id(identifier)
            record = self._assets[asset_id]
            self._deindex_asset(record)
            del self._assets[asset_id]
            self._watch_callbacks.pop(asset_id, None)

        if remove_cache and record.local_cache_path is not None:
            try:
                if record.local_cache_path.exists():
                    record.local_cache_path.unlink()
            except OSError as exc:
                raise AssetCacheError(
                    f"Failed to remove cached file for '{asset_id}': {exc}"
                ) from exc

        logger.info("Unregistered asset '%s'.", asset_id)

    # ------------------------------------------------------------------
    # Lookup / existence / listing / search
    # ------------------------------------------------------------------

    def find_asset(self, identifier: str) -> AssetRecord:
        """Return the :class:`AssetRecord` for ``identifier``.

        Args:
            identifier: Asset ID or alias.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If no such asset is registered.
        """
        return self._get_record(identifier)

    def asset_exists(self, identifier: str) -> bool:
        """Return whether ``identifier`` (an ID or alias) is registered."""
        self._require_ready()
        with self._lock:
            return identifier in self._assets or identifier in self._alias_index

    def list_assets(
        self,
        *,
        kind: Optional[AssetKind] = None,
        source: Optional[AssetSource] = None,
        state: Optional[AssetState] = None,
        tag: Optional[str] = None,
    ) -> list[AssetRecord]:
        """List registered assets, optionally filtered.

        Args:
            kind: Only include assets of this kind.
            source: Only include assets from this source.
            state: Only include assets in this state.
            tag: Only include assets carrying this tag.

        Raises:
            NotInitializedError: If not initialized.
        """
        self._require_ready()
        with self._lock:
            records = list(self._assets.values())
        if kind is not None:
            records = [r for r in records if r.metadata.kind is kind]
        if source is not None:
            records = [r for r in records if r.metadata.source is source]
        if state is not None:
            records = [r for r in records if r.state is state]
        if tag is not None:
            records = [r for r in records if tag in r.metadata.tags]
        return records

    def search_assets(
        self,
        query: str = "",
        *,
        kind: Optional[AssetKind] = None,
        source: Optional[AssetSource] = None,
        tags: Optional[set[str]] = None,
        limit: int = 50,
    ) -> list[AssetRecord]:
        """Search assets by a free-text substring against ID/aliases/tags.

        Args:
            query: Case-insensitive substring matched against each
                candidate's asset ID, aliases, and tags. Empty string
                matches everything (subject to the other filters).
            kind: Restrict to this kind.
            source: Restrict to this source.
            tags: Restrict to assets carrying all of these tags.
            limit: Maximum number of results returned.

        Raises:
            NotInitializedError: If not initialized.
        """
        candidates = self.list_assets(kind=kind, source=source)
        if tags:
            candidates = [r for r in candidates if tags.issubset(r.metadata.tags)]

        needle = query.strip().lower()
        if needle:
            def _matches(record: AssetRecord) -> bool:
                haystacks = [record.metadata.asset_id.lower(), *[a.lower() for a in record.metadata.aliases]]
                haystacks += [t.lower() for t in record.metadata.tags]
                return any(needle in h for h in haystacks)

            candidates = [r for r in candidates if _matches(r)]

        return candidates[:limit]

    def _find_by_kind(self, kind: AssetKind, name: str) -> AssetRecord:
        """Shared implementation for the per-kind convenience finders below."""
        try:
            record = self._get_record(name)
        except AssetNotFoundError:
            matches = self.search_assets(name, kind=kind, limit=1)
            if not matches:
                raise AssetNotFoundError(
                    f"No {kind.value} asset found matching '{name}'."
                ) from None
            return matches[0]
        if record.metadata.kind is not kind:
            raise AssetNotFoundError(
                f"Asset '{name}' exists but is a '{record.metadata.kind.value}', not a '{kind.value}'."
            )
        return record

    def find_material(self, name: str) -> AssetRecord:
        """Look up a registered material by ID, alias, or fuzzy name match."""
        return self._find_by_kind(AssetKind.MATERIAL, name)

    def find_texture(self, name: str) -> AssetRecord:
        """Look up a registered texture by ID, alias, or fuzzy name match."""
        return self._find_by_kind(AssetKind.TEXTURE, name)

    def find_mesh(self, name: str) -> AssetRecord:
        """Look up a registered mesh by ID, alias, or fuzzy name match."""
        return self._find_by_kind(AssetKind.MESH, name)

    def find_skeleton(self, name: str) -> AssetRecord:
        """Look up a registered skeleton by ID, alias, or fuzzy name match."""
        return self._find_by_kind(AssetKind.SKELETON, name)

    def find_animation(self, name: str) -> AssetRecord:
        """Look up a registered animation by ID, alias, or fuzzy name match."""
        return self._find_by_kind(AssetKind.ANIMATION, name)

    # ------------------------------------------------------------------
    # Aliases / tags / LOD
    # ------------------------------------------------------------------

    def add_alias(self, identifier: str, alias: str) -> None:
        """Add ``alias`` as an additional lookup name for an asset.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetRegistrationError: If ``alias`` is already used by a
                different asset.
        """
        with self._lock:
            asset_id = self._resolve_id(identifier)
            owner = self._alias_index.get(alias)
            if owner is not None and owner != asset_id:
                raise AssetRegistrationError(f"Alias '{alias}' is already used by asset '{owner}'.")
            self._assets[asset_id].metadata.aliases.add(alias)
            self._alias_index[alias] = asset_id
            self._assets[asset_id].metadata.updated_at = time.time()

    def remove_alias(self, alias: str) -> None:
        """Remove a previously registered alias. No-op if it does not exist."""
        with self._lock:
            asset_id = self._alias_index.pop(alias, None)
            if asset_id is not None and asset_id in self._assets:
                self._assets[asset_id].metadata.aliases.discard(alias)

    def add_tag(self, identifier: str, tag: str) -> None:
        """Add a semantic search tag to an asset.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        with self._lock:
            asset_id = self._resolve_id(identifier)
            self._assets[asset_id].metadata.tags.add(tag)
            self._tag_index.setdefault(tag, set()).add(asset_id)
            self._assets[asset_id].metadata.updated_at = time.time()

    def remove_tag(self, identifier: str, tag: str) -> None:
        """Remove a semantic search tag from an asset.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        with self._lock:
            asset_id = self._resolve_id(identifier)
            self._assets[asset_id].metadata.tags.discard(tag)
            members = self._tag_index.get(tag)
            if members is not None:
                members.discard(asset_id)
                if not members:
                    del self._tag_index[tag]

    def set_lod_path(self, identifier: str, level: int, uri: str) -> None:
        """Register a URI override for LOD ``level`` of an asset.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        with self._lock:
            asset_id = self._resolve_id(identifier)
            self._assets[asset_id].metadata.lod_paths[level] = uri
            self._assets[asset_id].metadata.updated_at = time.time()

    def get_lod_path(self, identifier: str, level: int) -> str:
        """Return the URI for a specific LOD level, falling back to the
        asset's base ``uri`` if that level has no override.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        record = self._get_record(identifier)
        return record.metadata.lod_paths.get(level, record.metadata.uri)

    # ------------------------------------------------------------------
    # Checksums / integrity
    # ------------------------------------------------------------------

    def compute_checksum(self, path: "str | Path", *, algorithm: Optional[str] = None) -> str:
        """Compute the hex-digest checksum of a local file.

        Args:
            path: Local file to hash.
            algorithm: Algorithm name (accepted by ``hashlib.new``).
                Defaults to this server's configured default.

        Raises:
            AssetValidationError: If the file cannot be read or the
                algorithm is unknown to ``hashlib``.
        """
        file_path = Path(path)
        algo = algorithm or self._checksum_algorithm
        try:
            hasher = hashlib.new(algo)
        except ValueError as exc:
            raise AssetValidationError(f"Unknown checksum algorithm '{algo}': {exc}") from exc

        try:
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_STREAM_CHUNK_BYTES), b""):
                    hasher.update(chunk)
        except OSError as exc:
            raise AssetValidationError(f"Failed to read '{file_path}' for checksumming: {exc}") from exc

        return hasher.hexdigest()

    def verify_checksum(self, identifier: str) -> bool:
        """Verify a cached asset's local file against its recorded checksum.

        Returns:
            True if the checksums match, or if the asset has no recorded
            checksum (nothing to verify against). False on a mismatch.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetValidationError: If no local cache path is available to
                check, or the file cannot be read.
        """
        record = self._get_record(identifier)
        if record.metadata.checksum is None:
            return True
        if record.local_cache_path is None:
            raise AssetValidationError(
                f"Asset '{record.metadata.asset_id}' has no cached local file to verify."
            )
        actual = self.compute_checksum(
            record.local_cache_path, algorithm=record.metadata.checksum_algorithm
        )
        return actual == record.metadata.checksum

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve_dependencies(self, identifier: str, *, recursive: bool = True) -> list[str]:
        """Return this asset's dependencies in load order (dependencies first).

        Args:
            identifier: Asset ID or alias.
            recursive: If True (default), transitively resolve
                dependencies-of-dependencies. If False, only the
                asset's direct dependencies are returned (unordered
                relative to each other).

        Returns:
            A list of asset IDs, topologically ordered so that every
            dependency appears before anything that depends on it. The
            queried asset itself is not included.

        Raises:
            AssetNotFoundError: If ``identifier``, or any (transitive)
                dependency it declares, is not registered.
            DependencyResolutionError: If a circular dependency is
                detected.
        """
        root_record = self._get_record(identifier)
        root_id = root_record.metadata.asset_id

        if not recursive:
            direct = sorted(root_record.metadata.dependencies)
            missing = [dep for dep in direct if dep not in self._assets]
            if missing:
                raise AssetNotFoundError(
                    f"Asset '{root_id}' declares missing dependencies: {missing}."
                )
            return direct

        order: list[str] = []
        visited: set[str] = set()
        in_progress: set[str] = set()

        def _visit(asset_id: str) -> None:
            if asset_id in visited:
                return
            if asset_id in in_progress:
                raise DependencyResolutionError(
                    f"Circular dependency detected involving asset '{asset_id}'."
                )
            record = self._assets.get(asset_id)
            if record is None:
                raise AssetNotFoundError(f"Dependency '{asset_id}' is not registered.")
            in_progress.add(asset_id)
            for dependency_id in sorted(record.metadata.dependencies):
                _visit(dependency_id)
            in_progress.discard(asset_id)
            visited.add(asset_id)
            if asset_id != root_id:
                order.append(asset_id)

        with self._lock:
            for dependency_id in sorted(root_record.metadata.dependencies):
                _visit(dependency_id)

        return order

    # ------------------------------------------------------------------
    # Caching / staleness
    # ------------------------------------------------------------------

    def _cache_destination(self, record: AssetRecord) -> Path:
        """Compute the cache file path a non-local asset should be materialized to."""
        asset_id = record.metadata.asset_id
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in asset_id)
        suffix = Path(record.metadata.uri).suffix or ""
        return self._cache_dir / f"{safe_name}{suffix}"

    def _resolve_local_path(self, uri: str) -> Path:
        """Resolve a ``LOCAL``-source URI against configured search paths."""
        direct = Path(uri)
        if direct.exists():
            return direct
        for search_path in self._asset_search_paths:
            candidate = search_path / uri
            if candidate.exists():
                return candidate
        return direct  # let the caller raise with the original, most-informative path

    def cache_asset(self, identifier: str, *, force: bool = False) -> Path:
        """Ensure a local, on-disk copy of an asset exists and return its path.

        For ``LOCAL`` assets this simply verifies (and resolves against
        :attr:`_asset_search_paths`, if needed) the recorded path. For
        ``NUCLEUS`` and ``REMOTE`` assets, the content is downloaded
        into :attr:`cache_dir` if not already cached (or if ``force``).
        ``GENERATED`` assets have nothing to cache until produced and
        re-registered as ``LOCAL``.

        Args:
            identifier: Asset ID or alias.
            force: If True, re-fetch/re-verify even if already cached.

        Returns:
            The resolved local path.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetCacheError: If the asset cannot be materialized.
        """
        self._require_ready()
        record = self._get_record(identifier)
        metadata = record.metadata

        if record.local_cache_path is not None and not force and record.local_cache_path.exists():
            return record.local_cache_path

        if metadata.source is AssetSource.LOCAL:
            resolved = self._resolve_local_path(metadata.uri)
            if not resolved.exists():
                raise AssetCacheError(
                    f"Local asset '{metadata.asset_id}' not found at '{metadata.uri}' "
                    f"(also checked {len(self._asset_search_paths)} search path(s))."
                )
            with self._lock:
                record.local_cache_path = resolved
                record.state = AssetState.CACHED
                record.source_mtime = resolved.stat().st_mtime
            return resolved

        if metadata.source is AssetSource.GENERATED:
            raise AssetCacheError(
                f"Asset '{metadata.asset_id}' is GENERATED and has not been produced yet; "
                "nothing to cache."
            )

        destination = self._cache_destination(record)
        try:
            self._transfer_to_local(metadata, destination)
        except AssetServerError:
            with self._lock:
                record.state = AssetState.ERROR
                record.last_error = "cache transfer failed"
            raise

        with self._lock:
            record.local_cache_path = destination
            record.state = AssetState.CACHED
            record.source_mtime = destination.stat().st_mtime if destination.exists() else None

        return destination

    def _transfer_to_local(self, metadata: AssetMetadata, destination: Path) -> None:
        """Materialize a NUCLEUS or REMOTE asset's content at ``destination``."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if metadata.source is AssetSource.NUCLEUS:
            self._download_from_nucleus(metadata.uri, destination)
        elif metadata.source is AssetSource.REMOTE:
            self._download_from_remote(metadata.uri, destination)
        else:
            raise AssetCacheError(
                f"Unsupported source '{metadata.source.value}' for a cache transfer."
            )

    def _download_from_nucleus(self, uri: str, destination: Path) -> None:
        omni_client = _lazy_import(
            "omni.client", hint="Nucleus transfers require the 'omni.client' module."
        )
        try:
            result, _version, _content = omni_client.read_file(uri)
            if result != omni_client.Result.OK:
                raise AssetDownloadError(f"Nucleus read of '{uri}' failed with result '{result}'.")
        except AssetDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssetDownloadError(f"Failed to read '{uri}' from Nucleus: {exc}") from exc

        try:
            destination.write_bytes(bytes(_content))
        except OSError as exc:
            raise AssetDownloadError(f"Failed to write Nucleus content to '{destination}': {exc}") from exc

    def _download_from_remote(self, uri: str, destination: Path) -> None:
        try:
            with urllib.request.urlopen(uri) as response, destination.open("wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle)
        except Exception as exc:  # noqa: BLE001
            raise AssetDownloadError(f"Failed to download '{uri}' to '{destination}': {exc}") from exc

    def clear_cache(self, identifier: Optional[str] = None, *, delete_files: bool = True) -> int:
        """Clear cached local copies for one asset or the whole registry.

        Args:
            identifier: If given, clear only this asset's cache entry.
                If ``None``, clear every asset's cache entry.
            delete_files: If True (default), also delete the cached
                file(s) from disk. ``LOCAL``-source assets' canonical
                files are never deleted, even if they happen to be the
                "cached" path.

        Returns:
            Number of assets whose cache entry was cleared.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is given but not registered.
            AssetCacheError: If a file deletion fails.
        """
        self._require_ready()
        if identifier is not None:
            targets = [self._get_record(identifier)]
        else:
            with self._lock:
                targets = list(self._assets.values())

        cleared = 0
        for record in targets:
            if record.local_cache_path is None:
                continue
            is_local_canonical = record.metadata.source is AssetSource.LOCAL
            if delete_files and not is_local_canonical:
                try:
                    if record.local_cache_path.exists():
                        record.local_cache_path.unlink()
                except OSError as exc:
                    raise AssetCacheError(
                        f"Failed to delete cache file for '{record.metadata.asset_id}': {exc}"
                    ) from exc
            with self._lock:
                record.local_cache_path = None
                record.source_mtime = None
                if record.state in (AssetState.CACHED, AssetState.LOADED, AssetState.STALE):
                    record.state = AssetState.REGISTERED
            cleared += 1

        logger.info("Cleared cache for %d asset(s).", cleared)
        return cleared

    def check_staleness(self, identifier: str) -> bool:
        """Return whether a cached ``LOCAL`` asset's source has changed
        since it was last cached.

        Non-``LOCAL`` sources and assets with no cached copy are never
        considered stale by this check (their staleness is instead a
        question of "has the remote content changed", which this
        module does not poll for).

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        record = self._get_record(identifier)
        if record.metadata.source is not AssetSource.LOCAL or record.local_cache_path is None:
            return False
        try:
            current_mtime = record.local_cache_path.stat().st_mtime
        except OSError:
            return True  # file disappeared out from under us -- treat as stale
        is_stale = record.source_mtime is not None and current_mtime > record.source_mtime
        if is_stale:
            with self._lock:
                record.state = AssetState.STALE
        return is_stale

    # ------------------------------------------------------------------
    # Load / unload / reload
    # ------------------------------------------------------------------

    def load_asset(self, identifier: str, *, force_reload: bool = False) -> Path:
        """Resolve an asset (and, recursively, its dependencies) to a
        usable local path.

        This is the primary entry point downstream components use to go
        from an asset ID to something they can actually open. It never
        opens the file itself (that remains the caller's / USDLoader's
        job) -- it only guarantees the returned path exists on disk.

        Args:
            identifier: Asset ID or alias.
            force_reload: If True, re-cache even if a valid local copy
                already exists.

        Returns:
            The resolved local path.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If the asset, or any dependency it
                declares, is not registered.
            DependencyResolutionError: If a circular dependency is
                detected among its dependencies.
            AssetLoadError: If caching the asset (or any dependency)
                fails.
        """
        self._require_ready()
        record = self._get_record(identifier)
        asset_id = record.metadata.asset_id

        try:
            for dependency_id in self.resolve_dependencies(asset_id, recursive=True):
                self.load_asset(dependency_id, force_reload=force_reload)

            path = self.cache_asset(asset_id, force=force_reload or self.check_staleness(asset_id))
        except AssetServerError as exc:
            with self._lock:
                record.state = AssetState.ERROR
                record.last_error = str(exc)
            raise AssetLoadError(f"Failed to load asset '{asset_id}': {exc}") from exc

        with self._lock:
            record.state = AssetState.LOADED
            record.load_count += 1
            record.last_loaded_at = time.time()
            record.last_error = None

        logger.info("Loaded asset '%s' -> '%s'.", asset_id, path)
        return path

    def unload_asset(self, identifier: str) -> None:
        """Release a loaded asset back to the ``CACHED`` state.

        This never deletes the cached file (use :meth:`clear_cache` for
        that) -- it only demotes the asset's lifecycle state, freeing
        any caller that tracked "in use" state against it.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetUnloadError: If the asset is not currently loaded.
        """
        record = self._get_record(identifier)
        if record.state is not AssetState.LOADED:
            raise AssetUnloadError(
                f"Asset '{record.metadata.asset_id}' is not loaded (state='{record.state.value}')."
            )
        with self._lock:
            record.state = AssetState.CACHED if record.local_cache_path is not None else AssetState.REGISTERED
        logger.info("Unloaded asset '%s'.", record.metadata.asset_id)

    def reload_asset(self, identifier: str) -> Path:
        """Force-refresh an asset's cache and reload it.

        Equivalent to :meth:`load_asset` with ``force_reload=True``,
        provided as a distinct, explicit entry point for hot-reload
        call sites (including the watch-pump callback) where "reload"
        better conveys intent than "load".

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetLoadError: If the underlying reload fails.
        """
        return self.load_asset(identifier, force_reload=True)

    # ------------------------------------------------------------------
    # Download / upload
    # ------------------------------------------------------------------

    def download_asset(
        self, identifier: str, destination: Optional["str | Path"] = None, *, overwrite: bool = False
    ) -> Path:
        """Explicitly download a ``NUCLEUS``/``REMOTE`` asset, recording a
        :class:`DownloadRecord`.

        Args:
            identifier: Asset ID or alias.
            destination: Explicit destination path. Defaults to this
                asset's standard cache location.
            overwrite: If True, re-download even if ``destination``
                already exists.

        Returns:
            The path the asset was downloaded to.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetDownloadError: If the asset's source is ``LOCAL``/
                ``GENERATED`` (nothing to download), or the transfer fails.
        """
        self._require_ready()
        record = self._get_record(identifier)
        metadata = record.metadata
        if metadata.source not in (AssetSource.NUCLEUS, AssetSource.REMOTE):
            raise AssetDownloadError(
                f"Asset '{metadata.asset_id}' has source '{metadata.source.value}'; nothing to download."
            )

        target = Path(destination) if destination is not None else self._cache_destination(record)
        started_at = time.time()
        success = False
        bytes_downloaded = 0
        try:
            if target.exists() and not overwrite:
                logger.info("Download destination '%s' already exists; skipping (overwrite=False).", target)
            else:
                self._transfer_to_local(metadata, target)
            bytes_downloaded = target.stat().st_size if target.exists() else 0
            success = True
            with self._lock:
                record.local_cache_path = target
                record.state = AssetState.CACHED
        finally:
            record_entry = DownloadRecord(
                asset_id=metadata.asset_id,
                source_uri=metadata.uri,
                destination=target,
                bytes_downloaded=bytes_downloaded,
                started_at=started_at,
                completed_at=time.time(),
                success=success,
            )
            with self._lock:
                self._downloads.append(record_entry)

        return target

    def upload_asset(self, identifier: str, destination_uri: str, *, overwrite: bool = False) -> str:
        """Upload a cached asset's local file to a Nucleus destination.

        Args:
            identifier: Asset ID or alias.
            destination_uri: Target ``omniverse://`` URL.
            overwrite: If True, overwrite an existing object at
                ``destination_uri``. If False and something already
                exists there, the underlying Nucleus write may fail
                (behavior is server-dependent); this module does not
                itself pre-check remote existence.

        Returns:
            ``destination_uri``, for chaining.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetUploadError: If the asset has no local cached copy yet,
                the destination is not an ``omniverse://`` URL, or the
                transfer fails.
        """
        self._require_ready()
        record = self._get_record(identifier)
        if record.local_cache_path is None or not record.local_cache_path.exists():
            raise AssetUploadError(
                f"Asset '{record.metadata.asset_id}' has no local cached copy to upload; "
                "call cache_asset()/load_asset() first."
            )
        if not destination_uri.startswith("omniverse://"):
            raise AssetUploadError(
                f"upload_asset() only supports Nucleus destinations (got '{destination_uri}'); "
                "plain HTTP(S) uploads are not supported."
            )

        omni_client = _lazy_import(
            "omni.client", hint="Nucleus uploads require the 'omni.client' module."
        )
        started_at = time.time()
        success = False
        bytes_uploaded = 0
        try:
            content = record.local_cache_path.read_bytes()
            result = omni_client.write_file(destination_uri, content)
            if result != omni_client.Result.OK:
                raise AssetUploadError(
                    f"Nucleus write to '{destination_uri}' failed with result '{result}'."
                )
            bytes_uploaded = len(content)
            success = True
        except AssetUploadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssetUploadError(f"Failed to upload '{identifier}' to '{destination_uri}': {exc}") from exc
        finally:
            entry = UploadRecord(
                asset_id=record.metadata.asset_id,
                destination_uri=destination_uri,
                bytes_uploaded=bytes_uploaded,
                started_at=started_at,
                completed_at=time.time(),
                success=success,
            )
            with self._lock:
                self._uploads.append(entry)

        return destination_uri

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_asset(self, identifier: str) -> AssetValidationReport:
        """Run consistency checks against a single registered asset.

        Checks performed:
            * The source can be reached (local file exists / Nucleus
              stat succeeds / remote URL responds), when possible to
              check cheaply.
            * If a checksum is recorded and the asset is cached, the
              checksum matches (see :meth:`verify_checksum`).
            * Every declared dependency is itself registered, and the
              dependency graph has no cycles (see
              :meth:`resolve_dependencies`).
            * The asset's file extension (if any) matches its declared
              ``kind``, as a soft warning only.

        Returns:
            A :class:`AssetValidationReport` describing every finding.
            This method itself only raises if validation *cannot be
            attempted at all* -- an invalid asset is a normal,
            non-raising result reported via the returned report.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        record = self._get_record(identifier)
        metadata = record.metadata
        issues: list[ValidationIssue] = []

        if metadata.source is AssetSource.LOCAL:
            resolved = self._resolve_local_path(metadata.uri)
            if not resolved.exists():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Local asset not found at '{metadata.uri}'.",
                        asset_id=metadata.asset_id,
                    )
                )
        elif metadata.source is AssetSource.GENERATED and record.local_cache_path is None:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Generated asset has not been produced yet.",
                    asset_id=metadata.asset_id,
                )
            )

        if metadata.checksum is not None and record.local_cache_path is not None:
            try:
                if not self.verify_checksum(metadata.asset_id):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message="Cached file checksum does not match the recorded checksum.",
                            asset_id=metadata.asset_id,
                        )
                    )
            except AssetValidationError as exc:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Could not verify checksum: {exc}",
                        asset_id=metadata.asset_id,
                    )
                )

        try:
            self.resolve_dependencies(metadata.asset_id, recursive=True)
        except AssetNotFoundError as exc:
            issues.append(
                ValidationIssue(severity=ValidationSeverity.ERROR, message=str(exc), asset_id=metadata.asset_id)
            )
        except DependencyResolutionError as exc:
            issues.append(
                ValidationIssue(severity=ValidationSeverity.ERROR, message=str(exc), asset_id=metadata.asset_id)
            )

        hint = self._infer_kind_hint(metadata.uri)
        if hint is not None and hint is not metadata.kind:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"File extension of '{metadata.uri}' suggests kind '{hint.value}', "
                        f"but asset is registered as '{metadata.kind.value}'."
                    ),
                    asset_id=metadata.asset_id,
                )
            )

        report = AssetValidationReport(issues=issues)
        logger.info(
            "Validated asset '%s' (errors=%d, warnings=%d).",
            metadata.asset_id, report.error_count, report.warning_count,
        )
        return report

    # ------------------------------------------------------------------
    # Watching / hot reload
    # ------------------------------------------------------------------

    def register_watch(self, identifier: str, callback: Callable[[str], None]) -> None:
        """Register ``callback`` to be invoked (with the asset ID) whenever
        this asset's source changes while watching is active.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        record = self._get_record(identifier)
        asset_id = record.metadata.asset_id
        with self._lock:
            record.watch_enabled = True
            self._watch_callbacks.setdefault(asset_id, []).append(callback)

    def unregister_watch(self, identifier: str, callback: Optional[Callable[[str], None]] = None) -> None:
        """Remove a watch callback (or all callbacks) for an asset.

        Raises:
            AssetNotFoundError: If ``identifier`` is not registered.
        """
        record = self._get_record(identifier)
        asset_id = record.metadata.asset_id
        with self._lock:
            if callback is None:
                self._watch_callbacks.pop(asset_id, None)
            else:
                callbacks = self._watch_callbacks.get(asset_id, [])
                if callback in callbacks:
                    callbacks.remove(callback)
                if not callbacks:
                    self._watch_callbacks.pop(asset_id, None)
            if asset_id not in self._watch_callbacks:
                record.watch_enabled = False

    def start_watching(
        self, *, interval_seconds: Optional[float] = None, auto_reload: bool = False
    ) -> None:
        """Start a background thread polling watched assets for source changes.

        Args:
            interval_seconds: Poll interval. Defaults to
                :data:`_DEFAULT_WATCH_INTERVAL_SECONDS`.
            auto_reload: If True, automatically call :meth:`reload_asset`
                for any asset found stale, before notifying callbacks.
                If False, callers are expected to call
                :meth:`reload_asset` themselves from within a callback.

        Raises:
            NotInitializedError: If not initialized.
            AssetWatchError: If watching is already running.
        """
        self._require_ready()
        with self._lock:
            if self._watch_pump is not None and self._watch_pump.is_running:
                raise AssetWatchError("Asset watching is already running.")
            self._watch_interval_seconds = interval_seconds or self._DEFAULT_WATCH_INTERVAL_SECONDS \
                if hasattr(self, "_DEFAULT_WATCH_INTERVAL_SECONDS") else (interval_seconds or _DEFAULT_WATCH_INTERVAL_SECONDS)
            self._auto_reload_on_watch = auto_reload
            self._watch_pump = _WatchPump(interval_seconds=self._watch_interval_seconds)

        self._watch_pump.start(self._poll_watched_assets)
        logger.info("Asset watching started (interval=%.2fs, auto_reload=%s).", self._watch_interval_seconds, auto_reload)

    def stop_watching(self) -> None:
        """Stop the background watch thread, if running. Idempotent."""
        with self._lock:
            pump = self._watch_pump
        if pump is not None:
            pump.stop()
        with self._lock:
            self._watch_pump = None
        logger.info("Asset watching stopped.")

    def _poll_watched_assets(self) -> None:
        with self._lock:
            watched_ids = [asset_id for asset_id, callbacks in self._watch_callbacks.items() if callbacks]

        for asset_id in watched_ids:
            try:
                if not self.check_staleness(asset_id):
                    continue
            except AssetNotFoundError:
                continue

            if self._auto_reload_on_watch:
                try:
                    self.reload_asset(asset_id)
                except AssetServerError:
                    logger.exception("Auto-reload failed for watched asset '%s'.", asset_id)

            with self._lock:
                callbacks = list(self._watch_callbacks.get(asset_id, []))
            for callback in callbacks:
                try:
                    callback(asset_id)
                except Exception:  # noqa: BLE001 - a callback's failure is not the server's
                    logger.exception("Watch callback for '%s' raised an exception.", asset_id)

    # ------------------------------------------------------------------
    # Preloading / packaging
    # ------------------------------------------------------------------

    def preload_assets(self, identifiers: list[str]) -> dict[str, "Path | AssetServerError"]:
        """Eagerly load a batch of assets, continuing past individual failures.

        Args:
            identifiers: Asset IDs or aliases to load.

        Returns:
            A mapping from each requested identifier to either the
            resolved local :class:`Path` (on success) or the
            :class:`AssetServerError` raised for it (on failure).

        Raises:
            NotInitializedError: If not initialized.
        """
        self._require_ready()
        results: dict[str, "Path | AssetServerError"] = {}
        for identifier in identifiers:
            try:
                results[identifier] = self.load_asset(identifier)
            except AssetServerError as exc:
                results[identifier] = exc
        succeeded = sum(1 for v in results.values() if isinstance(v, Path))
        logger.info("Preloaded %d/%d asset(s).", succeeded, len(identifiers))
        return results

    def package_assets(
        self, identifiers: list[str], destination_dir: "str | Path", *, include_dependencies: bool = True
    ) -> Path:
        """Copy a set of assets' cached files (and their manifest) into a
        self-contained directory for distribution.

        Args:
            identifiers: Asset IDs or aliases to package.
            destination_dir: Directory to copy files into (created if
                missing).
            include_dependencies: If True (default), transitively
                include every dependency of each requested asset.

        Returns:
            ``destination_dir``, containing the copied asset files plus
            an ``manifest.json`` describing them (see
            :meth:`export_registry`).

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If any identifier or dependency is not registered.
            AssetPackagingError: If caching or copying any asset fails.
        """
        self._require_ready()
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)

        asset_ids: set[str] = set()
        for identifier in identifiers:
            record = self._get_record(identifier)
            asset_ids.add(record.metadata.asset_id)
            if include_dependencies:
                asset_ids.update(self.resolve_dependencies(record.metadata.asset_id, recursive=True))

        for asset_id in sorted(asset_ids):
            try:
                source_path = self.cache_asset(asset_id)
                shutil.copy2(source_path, destination / source_path.name)
            except (AssetServerError, OSError) as exc:
                raise AssetPackagingError(f"Failed to package asset '{asset_id}': {exc}") from exc

        manifest_path = self.export_registry(destination / "manifest.json", asset_ids=asset_ids)
        logger.info("Packaged %d asset(s) into '%s'.", len(asset_ids), destination)
        return destination

    # ------------------------------------------------------------------
    # Statistics / reporting
    # ------------------------------------------------------------------

    def asset_statistics(self) -> AssetStatistics:
        """Compute and return aggregate statistics about the registry.

        Raises:
            NotInitializedError: If not initialized.
        """
        self._require_ready()
        with self._lock:
            records = list(self._assets.values())

        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_state: dict[str, int] = {}
        cached_count = 0
        loaded_count = 0
        total_cache_bytes = 0
        total_registered_bytes = 0

        for record in records:
            by_kind[record.metadata.kind.value] = by_kind.get(record.metadata.kind.value, 0) + 1
            by_source[record.metadata.source.value] = by_source.get(record.metadata.source.value, 0) + 1
            by_state[record.state.value] = by_state.get(record.state.value, 0) + 1
            if record.local_cache_path is not None:
                cached_count += 1
                try:
                    total_cache_bytes += record.local_cache_path.stat().st_size
                except OSError:
                    pass
            if record.state is AssetState.LOADED:
                loaded_count += 1
            if record.metadata.size_bytes:
                total_registered_bytes += record.metadata.size_bytes

        return AssetStatistics(
            total_assets=len(records),
            by_kind=by_kind,
            by_source=by_source,
            by_state=by_state,
            cached_count=cached_count,
            loaded_count=loaded_count,
            total_cache_bytes=total_cache_bytes,
            total_registered_bytes=total_registered_bytes,
        )

    def export_asset_report(self, path: "str | Path") -> Path:
        """Write a combined statistics + per-asset validation report to disk.

        Args:
            path: Destination JSON file path.

        Returns:
            The destination path.

        Raises:
            NotInitializedError: If not initialized.
            ManifestError: If the report cannot be written.
        """
        statistics = self.asset_statistics()
        with self._lock:
            asset_ids = sorted(self._assets.keys())

        validations = {asset_id: self.validate_asset(asset_id).to_dict() for asset_id in asset_ids}

        report = {
            "generated_at": time.time(),
            "statistics": statistics.to_dict(),
            "validations": validations,
        }

        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"Failed to write asset report to '{destination}': {exc}") from exc

        logger.info("Asset report written to '%s'.", destination)
        return destination

    # ------------------------------------------------------------------
    # Registry import / export (manifests)
    # ------------------------------------------------------------------

    def export_registry(
        self, path: "str | Path", *, asset_ids: Optional[set[str]] = None
    ) -> Path:
        """Serialize registered assets' metadata to a JSON manifest file.

        Only :class:`AssetMetadata` is persisted -- runtime state
        (cache paths, load counts, watch registrations, ...) is not,
        since it is only meaningful within a single server's lifetime
        on a single machine.

        Args:
            path: Destination JSON file path.
            asset_ids: If given, export only these assets. Defaults to
                the entire registry.

        Returns:
            The destination path.

        Raises:
            NotInitializedError: If not initialized.
            RegistryExportError: If the file cannot be written.
        """
        self._require_ready()
        with self._lock:
            if asset_ids is None:
                records = list(self._assets.values())
            else:
                records = [self._assets[aid] for aid in asset_ids if aid in self._assets]

        payload = {
            "manifest_version": _MANIFEST_VERSION,
            "generated_at": time.time(),
            "assets": [record.metadata.to_dict() for record in records],
        }

        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise RegistryExportError(f"Failed to write registry manifest to '{destination}': {exc}") from exc

        logger.info("Exported %d asset(s) to '%s'.", len(records), destination)
        return destination

    def import_registry(self, path: "str | Path", *, merge: bool = True) -> int:
        """Load a JSON manifest previously written by :meth:`export_registry`.

        Args:
            path: Source manifest file path.
            merge: If True (default), assets already registered under
                the same ID are overwritten with the manifest's version.
                If False, an asset already registered under the same ID
                as one in the manifest is skipped (existing registration wins).

        Returns:
            Number of assets registered (or overwritten) as a result.

        Raises:
            NotInitializedError: If not initialized.
            RegistryImportError: If the file cannot be read or parsed.
        """
        self._require_ready()
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RegistryImportError(f"Failed to read registry manifest '{source}': {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RegistryImportError(f"Failed to parse registry manifest '{source}': {exc}") from exc

        entries = raw.get("assets") if isinstance(raw, dict) else None
        if entries is None:
            raise RegistryImportError(f"Manifest '{source}' has no 'assets' array.")

        imported = 0
        for entry in entries:
            try:
                metadata = AssetMetadata.from_dict(entry)
            except ManifestError:
                logger.warning("Skipping malformed manifest entry in '%s'.", source)
                continue

            with self._lock:
                already_present = metadata.asset_id in self._assets
            if already_present and not merge:
                continue

            self.register_asset(
                metadata.asset_id,
                kind=metadata.kind,
                source=metadata.source,
                uri=metadata.uri,
                version=metadata.version,
                tags=set(metadata.tags),
                aliases=set(metadata.aliases),
                dependencies=set(metadata.dependencies),
                checksum=metadata.checksum,
                checksum_algorithm=metadata.checksum_algorithm,
                lod_paths=dict(metadata.lod_paths),
                extra=dict(metadata.extra),
                overwrite=True,
            )
            imported += 1

        logger.info("Imported %d asset(s) from '%s'.", imported, source)
        return imported

    # ------------------------------------------------------------------
    # Compatibility helpers (StageManager / USDLoader)
    # ------------------------------------------------------------------

    def resolve_for_stage(self, identifier: str, *, lod_level: Optional[int] = None) -> str:
        """Resolve an asset to a string suitable for authoring a USD
        reference/payload via ``stage_manager.StageManager``, or for a
        ``USDLoader``-style component to open directly.

        This is the sanctioned hand-off point between this module and
        both of those components: neither is imported here, and this
        module never authors anything into a stage itself.

        Args:
            identifier: Asset ID or alias.
            lod_level: If given, resolve this specific LOD level instead
                of the asset's base representation.

        Returns:
            An absolute local filesystem path string. The asset is
            cached (downloading it first, if necessary) as a side
            effect of this call.

        Raises:
            NotInitializedError: If not initialized.
            AssetNotFoundError: If ``identifier`` is not registered.
            AssetLoadError: If the asset cannot be materialized locally.
        """
        record = self._get_record(identifier)
        asset_id = record.metadata.asset_id

        if lod_level is not None:
            override_uri = record.metadata.lod_paths.get(lod_level)
            if override_uri is not None and override_uri != record.metadata.uri:
                temp_id = f"{asset_id}::lod{lod_level}"
                with self._lock:
                    existing = self._assets.get(temp_id)
                if existing is None:
                    self.register_asset(
                        temp_id,
                        kind=record.metadata.kind,
                        source=record.metadata.source,
                        uri=override_uri,
                    )
                path = self.load_asset(temp_id)
                return str(path.resolve())

        path = self.load_asset(asset_id)
        return str(path.resolve())


__all__ = [
    "AssetServer",
    "AssetMetadata",
    "AssetRecord",
    "ValidationIssue",
    "AssetValidationReport",
    "AssetStatistics",
    "DownloadRecord",
    "UploadRecord",
    "AssetServerState",
    "AssetKind",
    "AssetSource",
    "AssetState",
    "ValidationSeverity",
    "AssetServerError",
    "NotInitializedError",
    "AlreadyInitializedError",
    "NucleusImportError",
    "AssetRegistrationError",
    "AssetAlreadyExistsError",
    "AssetNotFoundError",
    "AssetLoadError",
    "AssetUnloadError",
    "AssetCacheError",
    "AssetDownloadError",
    "AssetUploadError",
    "AssetValidationError",
    "ChecksumMismatchError",
    "DependencyResolutionError",
    "AssetVersionError",
    "ManifestError",
    "RegistryExportError",
    "RegistryImportError",
    "AssetWatchError",
    "AssetPackagingError",
]
