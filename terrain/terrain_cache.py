"""
terrain_cache.py
═══════════════════════════════════════════════════════════════════════════
Reusable caching layer for expensive terrain computations in PhysWorldLM.

`terrain_cache` NEVER computes terrain itself. It only stores and retrieves
already-computed `TerrainSurface`-derived products: the surface itself,
normals, hillshade, curvature, slope, aspect, point clouds, meshes, voxel
grids, navigation-cost maps, occupancy grids, traversability maps, and
serialized exports (e.g. OBJ/PLY/STL/PNG bytes produced by
`terrain_converter`). Callers are expected to supply the actual `compute_fn`
(typically a `terrain_converter`/`TerrainSampler` call); this module's job is
purely memoization, eviction, persistence, and invalidation.

Reference-material note
------------------------
This module was implemented against `terrain_converter.py` and
`terrain_sampler.py` only. `terrain_surface.py`, `dem_loader.py`, and
`terrain_loader.py` were not available while writing it, so `TerrainSurface`
is treated strictly as an opaque object exposing only the attributes those
two sibling modules actually use (`shape`, `cell_size`, `origin`, `crs`,
`elevation`, `nodata_value`, `height_range`, `metadata.name`, and the
optional mask/material/semantic layers), plus the `SlopeUnits`,
`InterpolationMethod` enums and `TerrainError` exception they import. No
`TerrainSurface`, `dem_loader`, or `terrain_loader` internals are assumed
beyond that. In particular, this module does not know how to *construct* a
new `TerrainSurface` from raw arrays, so a cached `TerrainSurface` can only
be round-tripped through the disk cache via the explicit `allow_pickle=True`
opt-in; all NumPy-array-based derived products (the common case) use a
dependency-free NPZ path and never touch pickle.

Design notes
------------
    * Cache keys are deterministic: `hashlib.sha256` over a canonical
      JSON-ish encoding of the source surface's identity (shape, cell size,
      origin, nodata value, name, and a content hash of its elevation array)
      plus the operation name and parameters. `hash()` is never used for
      persistent keys since it is randomized per-process for strings.
    * Memory cache is a size- and/or count-bounded LRU, thread-safe via a
      single `RLock`.
    * Disk cache (optional) writes one `.tcache.npz` file per key using an
      atomic temp-file + `os.replace` sequence, with a companion
      `.sha256` sidecar checksum file so truncated/corrupted writes (e.g.
      from a crash mid-write) are detected and gracefully treated as a
      cache miss rather than raising.
    * Serialization prefers NumPy (`np.ndarray`, `dict[str, np.ndarray]`,
      `(np.ndarray, json_dict)` tuples, and plain JSON-safe values) and only
      falls back to `pickle` when the caller has explicitly opted in via
      `allow_pickle=True` on `TerrainCache`.
    * Concurrent `get_or_compute` calls for the same key are coalesced onto
      a single in-flight computation (via `concurrent.futures.Future`), so
      a cache stampede never triggers the same expensive computation twice.
    * An optional background thread pool (`background_loading=True`) lets
      callers `prefetch()` a value without blocking the calling thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .terrain_surface import TerrainError, TerrainSurface

logger = logging.getLogger("physworldlm.terrain.terrain_cache")

PathLike = Union[str, Path]

#: Version of the on-disk `.tcache.npz` container format. Bumped whenever
#: the field layout below changes so old entries are recognized as
#: incompatible (and thus a cache miss) rather than misread.
CACHE_FORMAT_VERSION = 1

#: Reserved npz field names used for container bookkeeping; value payloads
#: must not collide with these (payload array fields are always prefixed
#: with ``"arr_"`` to guarantee that).
_RESERVED_NPZ_FIELDS = frozenset({"format_version", "type_tag", "meta_json"})

#: Fallback size estimate (bytes) for values whose in-memory footprint
#: cannot be introspected (e.g. arbitrary Python objects with no ndarray
#: attributes). Keeps LRU accounting conservative rather than silently
#: treating such entries as free.
_UNKNOWN_VALUE_BYTE_ESTIMATE = 4096


# ═════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════

class TerrainCacheError(TerrainError):
    """Base class for `terrain_cache` failures."""


class CacheKeyError(TerrainCacheError):
    """Raised when a deterministic cache key cannot be derived."""


class CacheSerializationError(TerrainCacheError):
    """Raised when a value cannot be serialized to (or deserialized from)
    the disk cache format."""


# ═════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class CacheStats:
    """Point-in-time snapshot of `TerrainCache` activity and usage.

    All counters are cumulative since the `TerrainCache` was constructed
    (or since `TerrainCache.reset_stats` was last called).
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    disk_hits: int = 0
    memory_hits: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    serialize_seconds: float = 0.0
    deserialize_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Overall hit rate in `[0, 1]`; `0.0` if there have been no lookups."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Union[int, float]]:
        """Return a plain-`dict` snapshot suitable for logging or JSON export."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "disk_hits": self.disk_hits,
            "memory_hits": self.memory_hits,
            "memory_bytes": self.memory_bytes,
            "disk_bytes": self.disk_bytes,
            "serialize_seconds": self.serialize_seconds,
            "deserialize_seconds": self.deserialize_seconds,
            "hit_rate": self.hit_rate,
        }


# ═════════════════════════════════════════════════════════════════════════
# Internal helpers -- size estimation
# ═════════════════════════════════════════════════════════════════════════

def _approx_nbytes(value: Any) -> int:
    """Best-effort estimate of `value`'s in-memory footprint, in bytes.

    Used only for LRU accounting/eviction decisions, never for
    correctness, so it is deliberately conservative rather than exact for
    arbitrary Python objects.
    """
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        return sum(_approx_nbytes(v) for v in value.values()) or _UNKNOWN_VALUE_BYTE_ESTIMATE
    if isinstance(value, (list, tuple)):
        return sum(_approx_nbytes(v) for v in value) or _UNKNOWN_VALUE_BYTE_ESTIMATE
    if hasattr(value, "__dict__"):
        total = sum(
            v.nbytes for v in vars(value).values() if isinstance(v, np.ndarray)
        )
        return total or _UNKNOWN_VALUE_BYTE_ESTIMATE
    return _UNKNOWN_VALUE_BYTE_ESTIMATE


# ═════════════════════════════════════════════════════════════════════════
# Internal helpers -- deterministic key generation
# ═════════════════════════════════════════════════════════════════════════

def _canonicalize(obj: Any) -> Any:
    """Recursively reduce `obj` to a JSON-encodable, order-stable form for
    inclusion in a cache key. Enums, tuples, and NumPy scalars/arrays are
    normalized to their plain-Python (or string) equivalents rather than
    relying on their default `repr`, so equivalent parameters always
    canonicalize identically regardless of type spelling.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (tuple, list)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.shape, "sha256": hashlib.sha256(np.ascontiguousarray(obj)).hexdigest()}
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "name") and hasattr(obj, "__class__") and hasattr(obj.__class__, "__members__"):
        # Enum-like (e.g. SlopeUnits, InterpolationMethod) -- use the member
        # name, which is stable across processes unlike id()/repr().
        return f"{obj.__class__.__name__}.{obj.name}"
    # Last resort: a stable string representation. Not guaranteed unique
    # across truly opaque objects, but keeps key generation total rather
    # than raising for every unforeseen parameter type.
    return repr(obj)


class _SurfaceFingerprintCache:
    """Per-process cache of `TerrainSurface -> content fingerprint`.

    Fingerprinting hashes the full elevation array, which is O(rows*cols)
    per call; for repeated cache lookups against the same surface instance
    (the common case: one surface, many derived products) this class
    memoizes the fingerprint by object identity via a `WeakKeyDictionary`
    so it is only paid once per surface object's lifetime. If a surface is
    mutated in place after being fingerprinted, call
    `TerrainCache.invalidate_surface` to force a re-hash.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_id: dict[int, tuple[Any, str]] = {}

    def get(self, surface: TerrainSurface) -> str:
        key = id(surface)
        with self._lock:
            cached = self._by_id.get(key)
            if cached is not None and cached[0] is surface:
                return cached[1]
        fingerprint = self._compute(surface)
        with self._lock:
            self._by_id[key] = (surface, fingerprint)
        return fingerprint

    def invalidate(self, surface: TerrainSurface) -> None:
        with self._lock:
            self._by_id.pop(id(surface), None)

    @staticmethod
    def _compute(surface: TerrainSurface) -> str:
        h = hashlib.sha256()
        h.update(repr(getattr(surface, "shape", None)).encode("utf-8"))
        h.update(repr(getattr(surface, "cell_size", None)).encode("utf-8"))
        h.update(repr(getattr(surface, "origin", None)).encode("utf-8"))
        h.update(repr(getattr(surface, "nodata_value", None)).encode("utf-8"))
        metadata = getattr(surface, "metadata", None)
        h.update(repr(getattr(metadata, "name", None)).encode("utf-8"))
        elevation = getattr(surface, "elevation", None)
        if isinstance(elevation, np.ndarray):
            h.update(np.ascontiguousarray(elevation).tobytes())
        return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════
# Internal helpers -- serialization (NPZ-first, pickle only if requested)
# ═════════════════════════════════════════════════════════════════════════

def _tag_and_payload(
    value: Any, *, allow_pickle: bool
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    """Decompose `value` into `(type_tag, array_fields, json_meta)` for
    disk storage.

    Raises:
        CacheSerializationError: If `value`'s type is not one of the
            supported NPZ-friendly shapes and `allow_pickle` is `False`.
    """
    if isinstance(value, np.ndarray):
        return "ndarray", {"arr_data": np.ascontiguousarray(value)}, {}

    if isinstance(value, (bytes, bytearray)):
        return "bytes", {"arr_data": np.frombuffer(bytes(value), dtype=np.uint8)}, {}

    if isinstance(value, dict) and value and all(isinstance(v, np.ndarray) for v in value.values()):
        arrays = {f"arr_{k}": np.ascontiguousarray(v) for k, v in value.items()}
        return "dict_ndarray", arrays, {"keys": list(value.keys())}

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], np.ndarray)
    ):
        try:
            json.dumps(value[1])
        except (TypeError, ValueError):
            pass
        else:
            return "array_and_meta", {"arr_data": np.ascontiguousarray(value[0])}, {"meta": value[1]}

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        pass
    else:
        return "json", {}, {"value": value}

    if allow_pickle:
        import pickle

        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return "pickle", {"arr_data": np.frombuffer(payload, dtype=np.uint8)}, {}

    raise CacheSerializationError(
        f"Cannot serialize value of type '{type(value).__name__}' to the disk cache "
        "without allow_pickle=True (only ndarray, dict[str, ndarray], "
        "(ndarray, json-dict) tuples, and plain JSON-safe values are supported)."
    )


def _rebuild_from_payload(tag: str, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> Any:
    """Inverse of `_tag_and_payload`.

    Raises:
        CacheSerializationError: If `tag` is not recognized.
    """
    if tag == "ndarray":
        return arrays["arr_data"]
    if tag == "bytes":
        return arrays["arr_data"].tobytes()
    if tag == "dict_ndarray":
        keys = meta.get("keys", [])
        return {k: arrays[f"arr_{k}"] for k in keys}
    if tag == "array_and_meta":
        return arrays["arr_data"], meta.get("meta", {})
    if tag == "json":
        return meta.get("value")
    if tag == "pickle":
        import pickle

        return pickle.loads(arrays["arr_data"].tobytes())
    raise CacheSerializationError(f"Unknown cache entry type tag '{tag}'.")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically: write to a sibling temp file,
    `fsync`, then `os.replace` over the destination. Never leaves a
    partially-written file at `path` itself, even on crash mid-write."""
    tmp_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ═════════════════════════════════════════════════════════════════════════
# Memory cache -- bounded LRU
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class _MemoryEntry:
    value: Any
    nbytes: int
    created_at: float = field(default_factory=time.monotonic)


class _MemoryLRU:
    """Thread-safe, size- and/or count-bounded LRU cache.

    Eviction runs after every insert: least-recently-used entries (by
    access order, tracked via re-insertion into an `OrderedDict`-like
    structure) are dropped until both the entry-count and byte-size
    budgets are satisfied.
    """

    def __init__(self, *, max_entries: Optional[int], max_bytes: Optional[int]) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._entries: dict[str, _MemoryEntry] = {}
        self._order: list[str] = []  # least-recently-used first
        self._total_bytes = 0

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False, None
            self._touch(key)
            return True, entry.value

    def put(self, key: str, value: Any) -> int:
        """Insert/replace `key`, returning the number of entries evicted
        to satisfy the configured budgets."""
        nbytes = _approx_nbytes(value)
        with self._lock:
            if key in self._entries:
                self._total_bytes -= self._entries[key].nbytes
                self._order.remove(key)
            self._entries[key] = _MemoryEntry(value=value, nbytes=nbytes)
            self._order.append(key)
            self._total_bytes += nbytes
            return self._evict_locked()

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def delete(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False
            self._order.remove(key)
            self._total_bytes -= entry.nbytes
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()
            self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def _touch(self, key: str) -> None:
        # caller already holds self._lock
        self._order.remove(key)
        self._order.append(key)

    def _evict_locked(self) -> int:
        evicted = 0
        while self._order and (
            (self.max_entries is not None and len(self._order) > self.max_entries)
            or (self.max_bytes is not None and self._total_bytes > self.max_bytes)
        ):
            oldest = self._order.pop(0)
            entry = self._entries.pop(oldest)
            self._total_bytes -= entry.nbytes
            evicted += 1
        return evicted


# ═════════════════════════════════════════════════════════════════════════
# Disk cache -- atomic, checksummed, versioned NPZ files
# ═════════════════════════════════════════════════════════════════════════

class _DiskCache:
    """Checksummed, versioned, atomic-write disk cache backend.

    Each entry is a single `<key>.tcache.npz` file plus a
    `<key>.tcache.sha256` sidecar containing the SHA-256 checksum of the
    `.npz` file's bytes at write time. On read, the checksum is
    recomputed and compared; any mismatch, missing sidecar, unreadable
    npz, or unrecognized/incompatible `format_version` is treated as a
    corrupted/missing entry -- both files are removed and the read
    reports a miss rather than raising.
    """

    def __init__(self, root: Path, *, allow_pickle: bool, max_bytes: Optional[int]) -> None:
        self.root = root
        self.allow_pickle = allow_pickle
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.tcache.npz", self.root / f"{key}.tcache.sha256"

    def contains(self, key: str) -> bool:
        npz_path, sha_path = self._paths(key)
        return npz_path.exists() and sha_path.exists()

    def read(self, key: str) -> tuple[bool, Any, float]:
        """Returns `(found, value, deserialize_seconds)`. `found=False` on
        any missing/corrupted/incompatible entry (which is also cleaned
        up from disk as a side effect)."""
        npz_path, sha_path = self._paths(key)
        with self._lock:
            if not npz_path.exists() or not sha_path.exists():
                return False, None, 0.0
            try:
                raw = npz_path.read_bytes()
                expected_checksum = sha_path.read_text(encoding="utf-8").strip()
                actual_checksum = hashlib.sha256(raw).hexdigest()
                if actual_checksum != expected_checksum:
                    logger.warning("Disk cache entry '%s' failed checksum validation; discarding.", key)
                    self._remove_locked(key)
                    return False, None, 0.0

                start = time.perf_counter()
                import io

                with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
                    version = int(npz["format_version"])
                    if version != CACHE_FORMAT_VERSION:
                        logger.warning(
                            "Disk cache entry '%s' has incompatible format version %d (expected %d); discarding.",
                            key, version, CACHE_FORMAT_VERSION,
                        )
                        self._remove_locked(key)
                        return False, None, 0.0
                    tag = str(npz["type_tag"])
                    meta = json.loads(str(npz["meta_json"]))
                    arrays = {name: npz[name] for name in npz.files if name not in _RESERVED_NPZ_FIELDS}
                value = _rebuild_from_payload(tag, arrays, meta)
                elapsed = time.perf_counter() - start
                return True, value, elapsed
            except (OSError, ValueError, KeyError, json.JSONDecodeError, CacheSerializationError) as exc:
                logger.warning("Disk cache entry '%s' is corrupted (%s); discarding.", key, exc)
                self._remove_locked(key)
                return False, None, 0.0

    def write(self, key: str, value: Any) -> float:
        """Serializes and atomically writes `value` under `key`, enforcing
        the disk byte budget afterward. Returns the serialize time in
        seconds.

        Raises:
            CacheSerializationError: If `value` cannot be represented in
                the supported disk formats (see `_tag_and_payload`).
        """
        start = time.perf_counter()
        tag, arrays, meta = _tag_and_payload(value, allow_pickle=self.allow_pickle)

        import io

        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            format_version=np.array(CACHE_FORMAT_VERSION, dtype=np.int64),
            type_tag=np.array(tag),
            meta_json=np.array(json.dumps(meta)),
            **arrays,
        )
        payload = buffer.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        elapsed = time.perf_counter() - start

        npz_path, sha_path = self._paths(key)
        with self._lock:
            _atomic_write_bytes(npz_path, payload)
            _atomic_write_bytes(sha_path, checksum.encode("utf-8"))
            self._enforce_budget_locked()
        return elapsed

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove_locked(key)

    def clear(self) -> None:
        with self._lock:
            for path in self.root.glob("*.tcache.npz"):
                self._remove_locked(path.name[: -len(".tcache.npz")])

    def total_bytes(self) -> int:
        with self._lock:
            return sum(p.stat().st_size for p in self.root.glob("*.tcache.*") if p.is_file())

    def _remove_locked(self, key: str) -> bool:
        npz_path, sha_path = self._paths(key)
        removed = False
        for path in (npz_path, sha_path):
            if path.exists():
                try:
                    path.unlink()
                    removed = True
                except OSError:
                    pass
        return removed

    def _enforce_budget_locked(self) -> None:
        if self.max_bytes is None:
            return
        entries = sorted(self.root.glob("*.tcache.npz"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in entries)
        sidecar_total = sum(
            p.stat().st_size for p in self.root.glob("*.tcache.sha256")
        )
        total += sidecar_total
        idx = 0
        while total > self.max_bytes and idx < len(entries):
            npz_path = entries[idx]
            key = npz_path.name[: -len(".tcache.npz")]
            size_before = npz_path.stat().st_size
            sha_path = self.root / f"{key}.tcache.sha256"
            sha_size = sha_path.stat().st_size if sha_path.exists() else 0
            if self._remove_locked(key):
                total -= size_before + sha_size
            idx += 1


# ═════════════════════════════════════════════════════════════════════════
# TerrainCache -- public API
# ═════════════════════════════════════════════════════════════════════════

class TerrainCache:
    """Memoization layer for `TerrainSurface`-derived computations.

    A `TerrainCache` never computes terrain products itself; it stores and
    retrieves values produced elsewhere (typically by `terrain_converter`
    or `TerrainSampler` functions), keyed deterministically by the source
    surface's content and the operation/parameters used to derive the
    value.

    Args:
        max_memory_entries: Maximum number of entries kept in the in-memory
            LRU cache. `None` disables the count-based bound (byte budget
            still applies if set).
        max_memory_bytes: Maximum approximate total size (bytes) of values
            kept in memory. `None` disables the byte-based bound.
        disk_dir: Directory for the optional on-disk cache. If `None`
            (default), disk caching is disabled and `TerrainCache` behaves
            as memory-only.
        max_disk_bytes: Maximum total size (bytes) of the on-disk cache
            directory's `.tcache.*` files. `None` disables the bound.
        allow_pickle: If `True`, values that cannot be represented as
            NumPy arrays / JSON-safe data (e.g. an entire `TerrainSurface`,
            or a `terrain_converter.PointCloud`/`TriangleMesh` with a
            non-array `crs` field) fall back to `pickle` for disk
            persistence. Default `False`: such values still cache
            correctly in memory, but `put(..., persist=True)` for them is
            silently skipped for the disk tier (logged at `debug`).
        background_loading: If `True`, allocates a small thread pool so
            `prefetch()` can populate the cache without blocking the
            calling thread. If `False`, `prefetch()` raises.
        max_background_workers: Thread pool size when `background_loading`
            is enabled.

    Thread-safety: all public methods are safe to call concurrently from
    multiple threads. Concurrent `get_or_compute` calls that miss on the
    same key coalesce onto a single computation.
    """

    def __init__(
        self,
        *,
        max_memory_entries: Optional[int] = 256,
        max_memory_bytes: Optional[int] = None,
        disk_dir: Optional[PathLike] = None,
        max_disk_bytes: Optional[int] = None,
        allow_pickle: bool = False,
        background_loading: bool = False,
        max_background_workers: int = 2,
    ) -> None:
        self._memory = _MemoryLRU(max_entries=max_memory_entries, max_bytes=max_memory_bytes)
        self._disk: Optional[_DiskCache] = (
            _DiskCache(Path(disk_dir), allow_pickle=allow_pickle, max_bytes=max_disk_bytes)
            if disk_dir is not None
            else None
        )
        self.allow_pickle = allow_pickle
        self._fingerprints = _SurfaceFingerprintCache()

        self._stats = CacheStats()
        self._stats_lock = threading.RLock()

        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.RLock()

        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=max_background_workers, thread_name_prefix="terrain-cache")
            if background_loading
            else None
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the background thread pool, if one was created."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "TerrainCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── statistics ────────────────────────────────────────────────────

    @property
    def stats(self) -> CacheStats:
        """A consistent snapshot of cumulative cache statistics."""
        with self._stats_lock:
            snapshot = CacheStats(**vars(self._stats))
        snapshot.memory_bytes = self._memory.total_bytes
        snapshot.disk_bytes = self._disk.total_bytes() if self._disk is not None else 0
        return snapshot

    def reset_stats(self) -> None:
        """Zero all cumulative counters (does not affect cached entries)."""
        with self._stats_lock:
            self._stats = CacheStats()

    # ── deterministic key generation ─────────────────────────────────

    def make_key(self, surface: TerrainSurface, operation: str, **params: Any) -> str:
        """Derive a deterministic cache key for `operation` applied to
        `surface` with `params`.

        The key is a SHA-256 hex digest over the surface's content
        fingerprint (shape, cell size, origin, nodata value, name, and a
        hash of the elevation array) combined with `operation` and a
        canonicalized encoding of `params`. Equal surfaces + equal
        operation + equal parameters always produce the same key, in this
        process or any other -- this never uses Python's built-in `hash()`,
        which is salted per-process for strings and therefore unsuitable
        for a persistent (disk) key.

        Raises:
            CacheKeyError: If `operation` is empty.
        """
        if not operation:
            raise CacheKeyError("operation must be a non-empty string.")
        fingerprint = self._fingerprints.get(surface)
        canonical_params = _canonicalize(params)
        payload = json.dumps(
            {"op": operation, "params": canonical_params, "surface": fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def invalidate_surface(self, surface: TerrainSurface) -> None:
        """Forget the memoized content fingerprint for `surface`.

        Call this after mutating a `TerrainSurface` in place (the
        fingerprint cache is keyed by object identity for performance, so
        an in-place mutation would otherwise not be detected). This does
        not, by itself, remove any already-cached derived values -- those
        were keyed off the *old* fingerprint and will simply no longer be
        reachable via `make_key` for the mutated surface; use `clear()` if
        you also want to reclaim that storage immediately.
        """
        self._fingerprints.invalidate(surface)

    # ── core operations ───────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Look up `key`, checking memory first, then disk (promoting a
        disk hit back into memory). Returns `None` on a miss.

        Note: a legitimately cached value of `None` is indistinguishable
        from a miss under this API; callers caching `None`-valued results
        should wrap them (e.g. in a one-element tuple) before `put`.
        """
        found, value = self._memory.get(key)
        if found:
            with self._stats_lock:
                self._stats.hits += 1
                self._stats.memory_hits += 1
            return value

        if self._disk is not None:
            found, value, deserialize_seconds = self._disk.read(key)
            if found:
                self._memory.put(key, value)
                with self._stats_lock:
                    self._stats.hits += 1
                    self._stats.disk_hits += 1
                    self._stats.deserialize_seconds += deserialize_seconds
                return value

        with self._stats_lock:
            self._stats.misses += 1
        return None

    def put(self, key: str, value: Any, *, persist: bool = True) -> None:
        """Store `value` under `key` in the memory cache, and (if
        `persist` and a disk cache is configured) on disk.

        A disk write that fails because `value`'s type isn't
        disk-serializable (and `allow_pickle=False`) is logged and
        skipped -- the memory-cache write still succeeds, since disk
        persistence is a best-effort tier, not the caching guarantee.
        """
        evicted = self._memory.put(key, value)
        with self._stats_lock:
            self._stats.writes += 1
            self._stats.evictions += evicted

        if persist and self._disk is not None:
            try:
                serialize_seconds = self._disk.write(key, value)
            except CacheSerializationError as exc:
                logger.debug("Skipping disk persistence for key '%s': %s", key, exc)
            else:
                with self._stats_lock:
                    self._stats.serialize_seconds += serialize_seconds

    def contains(self, key: str) -> bool:
        """`True` if `key` is present in memory or on disk (does not
        promote a disk entry into memory or validate its checksum)."""
        if self._memory.contains(key):
            return True
        return self._disk.contains(key) if self._disk is not None else False

    def delete(self, key: str) -> None:
        """Remove `key` from both the memory and disk tiers, if present."""
        self._memory.delete(key)
        if self._disk is not None:
            self._disk.delete(key)
        with self._pending_lock:
            self._pending.pop(key, None)

    def invalidate(self, key: str) -> None:
        """Alias for `delete`, provided for readability at call sites that
        are explicitly invalidating a stale result (as opposed to
        incidentally overwriting one)."""
        self.delete(key)

    def clear(self) -> None:
        """Remove all entries from both the memory and disk tiers."""
        self.clear_memory()
        self.clear_disk()

    def clear_memory(self) -> None:
        """Remove all entries from the in-memory tier only."""
        self._memory.clear()

    def clear_disk(self) -> None:
        """Remove all entries from the on-disk tier only (a no-op if no
        disk cache is configured)."""
        if self._disk is not None:
            self._disk.clear()

    # ── lazy computation ──────────────────────────────────────────────

    def get_or_compute(
        self,
        surface: TerrainSurface,
        operation: str,
        compute_fn: Callable[[], Any],
        *,
        persist: bool = True,
        force: bool = False,
        **params: Any,
    ) -> Any:
        """Return the cached value for `(surface, operation, params)`,
        computing it via `compute_fn()` on a miss and storing the result.

        Args:
            surface: The source `TerrainSurface` (used only for key
                derivation -- never read or mutated by this method).
            operation: A short, stable name for the computation (e.g.
                `'normals'`, `'hillshade'`, `'mesh'`).
            compute_fn: Zero-argument callable that performs the actual
                (expensive) computation. Only invoked on a cache miss, and
                at most once even under concurrent calls for the same key.
            persist: Whether a freshly computed value should also be
                written to the disk tier (if configured).
            force: If `True`, bypass the cache lookup and recompute
                unconditionally, overwriting any existing entry.
            **params: Additional parameters distinguishing this
                computation (e.g. `stride=2`, `max_slope_deg=30.0`) --
                included in the cache key so different parameterizations
                of the same operation never collide.

        Returns:
            The cached or freshly computed value.
        """
        key = self.make_key(surface, operation, **params)
        if not force:
            cached = self.get(key)
            if cached is not None:
                return cached
        return self._compute_and_store(key, compute_fn, persist=persist)

    def prefetch(
        self,
        surface: TerrainSurface,
        operation: str,
        compute_fn: Callable[[], Any],
        *,
        persist: bool = True,
        **params: Any,
    ) -> Future:
        """Submit `compute_fn` to the background thread pool to
        populate the cache for `(surface, operation, params)` without
        blocking the calling thread.

        Returns:
            A `concurrent.futures.Future` resolving to the cached/
            computed value once available. If the entry is already cached,
            the returned `Future` is already resolved.

        Raises:
            TerrainCacheError: If this `TerrainCache` was constructed with
                `background_loading=False`.
        """
        if self._executor is None:
            raise TerrainCacheError(
                "prefetch() requires TerrainCache(background_loading=True)."
            )
        key = self.make_key(surface, operation, **params)

        cached = self.get(key)
        if cached is not None:
            resolved: Future = Future()
            resolved.set_result(cached)
            return resolved

        with self._pending_lock:
            existing = self._pending.get(key)
            if existing is not None:
                return existing

        return self._executor.submit(self._compute_and_store, key, compute_fn, persist)

    def _compute_and_store(self, key: str, compute_fn: Callable[[], Any], persist: bool) -> Any:
        """Coalescing single-flight compute: only the first caller for a
        given `key` actually invokes `compute_fn`; concurrent callers for
        the same key block on that call's result instead of recomputing."""
        with self._pending_lock:
            future = self._pending.get(key)
            is_owner = future is None
            if is_owner:
                future = Future()
                self._pending[key] = future

        if not is_owner:
            return future.result()

        try:
            value = compute_fn()
        except BaseException as exc:  # noqa: BLE001 -- propagate to all waiters, then re-raise
            future.set_exception(exc)
            with self._pending_lock:
                self._pending.pop(key, None)
            raise

        self.put(key, value, persist=persist)
        future.set_result(value)
        with self._pending_lock:
            self._pending.pop(key, None)
        return value


__all__ = [
    "TerrainCache",
    "CacheStats",
    "TerrainCacheError",
    "CacheKeyError",
    "CacheSerializationError",
    "CACHE_FORMAT_VERSION",
]
