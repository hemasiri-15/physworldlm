"""
terrain_surface.py
═══════════════════════════════════════════════════════════════════════════
Canonical terrain representation for PhysWorldLM.

`TerrainSurface` is the single source of truth for "what is the ground."
Every other module in this subsystem either *produces* a `TerrainSurface`
(`terrain_loader`, `dem_loader`), *queries* one (`terrain_sampler`),
*transforms* one (`terrain_converter`), or *caches* one (`terrain_cache`).

Design intent
-------------
This module is intentionally the most restricted file in the subsystem:

    * No simulator code (no Omniverse/Gazebo/MuJoCo/Unity/Unreal imports).
    * No renderer code.
    * No file-format I/O (that belongs to `terrain_loader` / `dem_loader` /
      `terrain_converter`).
    * Pure data + the analytical derivations (normals/slope/aspect/
      curvature) that are properties of the elevation grid itself, not of
      any particular source format or backend.

Everything downstream of this module -- `WorldSpec`, `SceneCompiler`,
`RuntimeContext`, `EnvironmentBuilder`, `SensorManager`, the physics
engine, UAV planners -- speaks in terms of `TerrainSurface`. Backend
adapters translate *from* `TerrainSurface`; they never define their own
parallel terrain representation.

Coordinate conventions
-----------------------
    * The elevation grid is stored row-major as `elevation[row, col]`,
      with `row` increasing along -Y (north-to-south, image convention)
      and `col` increasing along +X (west-to-east), matching the GDAL /
      GeoTIFF raster convention so no y-flip surprises loaders.
    * `origin` is the world-space (x, y) of the *top-left* grid cell
      center (row=0, col=0), consistent with GDAL's GeoTransform origin.
    * `cell_size` is (dx, dy) in world units (typically metres); dy is
      stored as a positive magnitude here (row *increases* southward),
      unlike a raw GDAL GeoTransform where pixel height is negative.
    * Z (elevation) is always "up" in world units.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger("physworldlm.terrain.surface")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

CURRENT_SCHEMA_VERSION = "1.0.0"


# ═════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═════════════════════════════════════════════════════════════════════════

class TerrainError(Exception):
    """Root of every exception raised anywhere in the terrain subsystem."""


class TerrainSurfaceError(TerrainError):
    """Base class for `TerrainSurface`-level failures."""


class InvalidGridError(TerrainSurfaceError):
    """Raised when an elevation grid is malformed (wrong rank, non-finite
    values where finite is required, degenerate shape, etc.)."""


class DimensionMismatchError(TerrainSurfaceError):
    """Raised when an auxiliary layer (mask/material/label grid) does not
    match the elevation grid's (rows, cols) shape."""


class SerializationError(TerrainSurfaceError):
    """Raised when `TerrainSurface` (de)serialization fails."""


class ChecksumMismatchError(SerializationError):
    """Raised when a loaded surface's checksum does not match its stored
    checksum, indicating corruption or tampering."""


class SpatialIndexError(TerrainSurfaceError):
    """Raised when spatial-index construction or querying fails."""


# ═════════════════════════════════════════════════════════════════════════
# Enums
# ═════════════════════════════════════════════════════════════════════════

class InterpolationMethod(Enum):
    """Interpolation strategies shared by `terrain_sampler` and any
    resampling performed by `terrain_converter`."""

    NEAREST = auto()
    BILINEAR = auto()
    BICUBIC = auto()


class VerticalDatum(Enum):
    """Vertical reference used by the elevation values themselves."""

    WGS84_ELLIPSOID = auto()
    EGM96_GEOID = auto()
    EGM2008_GEOID = auto()
    LOCAL = auto()          # arbitrary local/relative datum (e.g. simulation origin)
    UNKNOWN = auto()


class SlopeUnits(Enum):
    DEGREES = auto()
    RADIANS = auto()
    PERCENT = auto()


# ═════════════════════════════════════════════════════════════════════════
# Coordinate reference system
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CoordinateReferenceSystem:
    """A minimal, dependency-free CRS descriptor.

    Deliberately does not depend on `pyproj`/`GDAL` so this module has no
    hard geospatial-library dependency; `terrain_loader`/`dem_loader`
    populate this from whatever library they use internally, and
    `terrain_converter` reads it back out without ever needing to
    reconstruct a live transform object here.

    Attributes:
        epsg: EPSG code if known (e.g. 4326 for WGS84, 32633 for a UTM zone).
        wkt: Full WKT string, if available (authoritative if present).
        proj4: PROJ4 string, if available.
        name: Human-readable name, e.g. "WGS 84" or "Local Cartesian".
        is_geographic: True if coordinates are lon/lat degrees, False if
            projected (metres) or purely local/simulation cartesian.
        vertical_datum: Reference for the elevation (Z) values.
    """

    epsg: Optional[int] = None
    wkt: Optional[str] = None
    proj4: Optional[str] = None
    name: str = "unknown"
    is_geographic: bool = True
    vertical_datum: VerticalDatum = VerticalDatum.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "epsg": self.epsg,
            "wkt": self.wkt,
            "proj4": self.proj4,
            "name": self.name,
            "is_geographic": self.is_geographic,
            "vertical_datum": self.vertical_datum.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoordinateReferenceSystem":
        return cls(
            epsg=d.get("epsg"),
            wkt=d.get("wkt"),
            proj4=d.get("proj4"),
            name=d.get("name", "unknown"),
            is_geographic=d.get("is_geographic", True),
            vertical_datum=VerticalDatum[d.get("vertical_datum", "UNKNOWN")],
        )

    @classmethod
    def wgs84(cls) -> "CoordinateReferenceSystem":
        return cls(epsg=4326, name="WGS 84", is_geographic=True,
                    vertical_datum=VerticalDatum.WGS84_ELLIPSOID)

    @classmethod
    def local_cartesian(cls) -> "CoordinateReferenceSystem":
        """A backend-independent local ENU/simulation frame, metres, no
        geographic meaning. This is the default for procedurally
        generated or unit-test terrain."""
        return cls(epsg=None, name="Local Cartesian", is_geographic=False,
                    vertical_datum=VerticalDatum.LOCAL)


# ═════════════════════════════════════════════════════════════════════════
# Bounding volume
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundingBox3D:
    """Axis-aligned bounding box, world units."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )

    def contains_point_xy(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def to_dict(self) -> dict:
        return {
            "min": [self.min_x, self.min_y, self.min_z],
            "max": [self.max_x, self.max_y, self.max_z],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox3D":
        mn, mx = d["min"], d["max"]
        return cls(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])


# ═════════════════════════════════════════════════════════════════════════
# Metadata
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class TerrainMetadata:
    """Provenance and descriptive metadata carried alongside every
    `TerrainSurface`, independent of the numeric grid content."""

    name: str = "unnamed_terrain"
    source_format: str = "unknown"
    source_path: Optional[str] = None
    provider: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = CURRENT_SCHEMA_VERSION
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_format": self.source_format,
            "source_path": self.source_path,
            "provider": self.provider,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TerrainMetadata":
        return cls(
            name=d.get("name", "unnamed_terrain"),
            source_format=d.get("source_format", "unknown"),
            source_path=d.get("source_path"),
            provider=d.get("provider"),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            schema_version=d.get("schema_version", CURRENT_SCHEMA_VERSION),
            tags=list(d.get("tags", [])),
            extra=dict(d.get("extra", {})),
        )


# ═════════════════════════════════════════════════════════════════════════
# Spatial index
# ═════════════════════════════════════════════════════════════════════════

class SpatialIndex:
    """Nearest-neighbor spatial index over an (N, 2) or (N, 3) point set.

    Uses `scipy.spatial.cKDTree` when SciPy is importable (the common
    case, and the fast path); otherwise transparently falls back to a
    uniform spatial hash grid so the terrain subsystem never hard-fails
    for lack of an optional dependency. Callers never need to know which
    backend is active -- both expose the same `query`/`query_radius` API.

    This class is intentionally decoupled from `TerrainSurface`'s grid
    layout: it operates on plain point arrays, so it is equally usable
    for masked/irregular point sets (e.g. LiDAR point clouds) as for a
    regular elevation grid flattened to (row*col, 2) sample coordinates.
    """

    def __init__(self, points: np.ndarray) -> None:
        """Args:
            points: (N, 2) or (N, 3) array of point coordinates.

        Raises:
            SpatialIndexError: If `points` is empty or malformed.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (2, 3):
            raise SpatialIndexError(
                f"SpatialIndex requires an (N, 2) or (N, 3) array; got shape {points.shape}."
            )
        self._points = points
        self._backend_name: str
        self._tree = None
        self._grid_index: Optional[dict[tuple[int, int], list[int]]] = None
        self._cell_size: float = 1.0
        self._build()

    def _build(self) -> None:
        try:
            from scipy.spatial import cKDTree  # noqa: WPS433 -- optional dependency, lazy by design
            self._tree = cKDTree(self._points)
            self._backend_name = "scipy.cKDTree"
        except ImportError:
            logger.info("scipy not available; SpatialIndex falling back to uniform grid hash.")
            self._build_grid_fallback()
            self._backend_name = "uniform_grid_fallback"

    def _build_grid_fallback(self) -> None:
        xy = self._points[:, :2]
        span = np.ptp(xy, axis=0)
        span = np.where(span <= 0, 1.0, span)
        n = max(1, int(np.sqrt(len(xy))))
        self._cell_size = float(max(span) / n) or 1.0
        grid: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(xy):
            key = (int(x // self._cell_size), int(y // self._cell_size))
            grid.setdefault(key, []).append(i)
        self._grid_index = grid

    @property
    def backend(self) -> str:
        """Name of the active backend, e.g. `'scipy.cKDTree'`."""
        return self._backend_name

    def query(self, point: Sequence[float], k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Find the `k` nearest points to `point`.

        Returns:
            (distances, indices), each shape (k,) (or scalar-wrapped if k=1
            for the scipy backend, normalized here to always be arrays).
        """
        point = np.asarray(point, dtype=np.float64)
        if self._tree is not None:
            dist, idx = self._tree.query(point, k=k)
            dist = np.atleast_1d(dist)
            idx = np.atleast_1d(idx)
            return dist, idx
        # Fallback: brute-force over an expanding ring of grid cells.
        cx, cy = int(point[0] // self._cell_size), int(point[1] // self._cell_size)
        candidates: list[int] = []
        ring = 0
        while len(candidates) < k and ring < 1000:
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    candidates.extend(self._grid_index.get((cx + dx, cy + dy), []))  # type: ignore[union-attr]
            ring += 1
        if not candidates:
            candidates = list(range(len(self._points)))
        candidates = list(dict.fromkeys(candidates))  # dedupe, preserve order
        pts = self._points[candidates]
        d = np.linalg.norm(pts[:, :2] - point[:2], axis=1)
        order = np.argsort(d)[:k]
        return d[order], np.asarray(candidates)[order]

    def query_radius(self, point: Sequence[float], radius: float) -> np.ndarray:
        """Return indices of all points within `radius` of `point`."""
        point = np.asarray(point, dtype=np.float64)
        if self._tree is not None:
            return np.asarray(self._tree.query_ball_point(point, r=radius))
        d = np.linalg.norm(self._points[:, :2] - point[:2], axis=1)
        return np.nonzero(d <= radius)[0]


# ═════════════════════════════════════════════════════════════════════════
# TerrainSurface
# ═════════════════════════════════════════════════════════════════════════

class TerrainSurface:
    """The canonical, backend-independent terrain representation.

    A `TerrainSurface` wraps a regular elevation grid plus every derived
    or auxiliary layer needed by the rest of PhysWorldLM: normals, slope,
    aspect, curvature, material/semantic/mask layers, and enough
    provenance metadata to round-trip through serialization without
    losing CRS information.

    Thread-safety: read access (sampling, property access) is safe from
    multiple threads. Lazily-computed derived layers (normals, slope,
    aspect, curvature, spatial index) are computed under an internal
    lock the first time they are requested, so concurrent first-access
    from multiple threads cannot race and compute the same layer twice.

    Attributes:
        elevation: (rows, cols) float64 grid of elevation values, world
            units (typically metres). Row 0 = north/top edge (image
            convention); see module docstring for the full coordinate
            convention.
        cell_size: (dx, dy) world-unit size of one grid cell.
        origin: (x, y) world coordinate of cell (row=0, col=0)'s center.
        crs: Coordinate reference system of the (x, y) plane.
        metadata: Provenance/descriptive metadata.
        nodata_value: Sentinel value in `elevation` denoting missing data,
            or `None` if the grid is fully populated.
    """

    def __init__(
        self,
        elevation: np.ndarray,
        cell_size: tuple[float, float] = (1.0, 1.0),
        origin: tuple[float, float] = (0.0, 0.0),
        crs: Optional[CoordinateReferenceSystem] = None,
        metadata: Optional[TerrainMetadata] = None,
        nodata_value: Optional[float] = None,
        validate: bool = True,
    ) -> None:
        elevation = np.asarray(elevation, dtype=np.float64)
        if validate:
            _validate_elevation_grid(elevation)
            if cell_size[0] <= 0 or cell_size[1] <= 0:
                raise InvalidGridError(f"cell_size must be positive; got {cell_size}.")

        self.elevation = elevation
        self.cell_size = (float(cell_size[0]), float(cell_size[1]))
        self.origin = (float(origin[0]), float(origin[1]))
        self.crs = crs or CoordinateReferenceSystem.local_cartesian()
        self.metadata = metadata or TerrainMetadata()
        self.nodata_value = nodata_value

        # Optional auxiliary layers -- all validated to match grid shape
        # on assignment via the property setters below.
        self._material_map: Optional[np.ndarray] = None
        self._semantic_labels: Optional[np.ndarray] = None
        self._vegetation_mask: Optional[np.ndarray] = None
        self._road_mask: Optional[np.ndarray] = None
        self._water_mask: Optional[np.ndarray] = None
        self._obstacle_mask: Optional[np.ndarray] = None

        # Lazily-computed derived layers.
        self._normals: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._aspect: Optional[np.ndarray] = None
        self._curvature: Optional[np.ndarray] = None
        self._spatial_index: Optional[SpatialIndex] = None

        self._lock = threading.RLock()

    # ── shape / geometry properties ────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) of the elevation grid."""
        return self.elevation.shape

    @property
    def resolution(self) -> tuple[int, int]:
        """Alias for `shape`, in (width_cols, height_rows) order -- the
        convention most image/graphics callers expect."""
        rows, cols = self.elevation.shape
        return (cols, rows)

    @property
    def height_range(self) -> tuple[float, float]:
        """(min_elevation, max_elevation) ignoring NODATA cells."""
        valid = self._valid_mask()
        if not np.any(valid):
            return (0.0, 0.0)
        vals = self.elevation[valid]
        return (float(np.min(vals)), float(np.max(vals)))

    @property
    def bounding_box(self) -> BoundingBox3D:
        rows, cols = self.shape
        dx, dy = self.cell_size
        ox, oy = self.origin
        min_z, max_z = self.height_range
        return BoundingBox3D(
            min_x=ox, min_y=oy - (rows - 1) * dy,
            min_z=min_z,
            max_x=ox + (cols - 1) * dx, max_y=oy,
            max_z=max_z,
        )

    def _valid_mask(self) -> np.ndarray:
        """Boolean mask of cells that are *not* NODATA / NaN."""
        mask = np.isfinite(self.elevation)
        if self.nodata_value is not None:
            mask &= self.elevation != self.nodata_value
        return mask

    # ── coordinate transforms ────────────────────────────────────────

    def world_to_grid(self, x: float, y: float) -> tuple[float, float]:
        """Convert a world (x, y) coordinate to fractional (row, col).

        Fractional so callers doing bilinear/bicubic sampling do not need
        a second transform; truncate/round for nearest-cell lookups.
        """
        dx, dy = self.cell_size
        ox, oy = self.origin
        col = (x - ox) / dx
        row = (oy - y) / dy
        return (row, col)

    def grid_to_world(self, row: float, col: float) -> tuple[float, float]:
        """Convert a fractional (row, col) grid coordinate to world (x, y)."""
        dx, dy = self.cell_size
        ox, oy = self.origin
        x = ox + col * dx
        y = oy - row * dy
        return (x, y)

    def in_bounds(self, x: float, y: float) -> bool:
        row, col = self.world_to_grid(x, y)
        rows, cols = self.shape
        return 0 <= row <= rows - 1 and 0 <= col <= cols - 1

    # ── auxiliary layer accessors (validated on write) ───────────────

    def _set_layer(self, attr: str, value: Optional[np.ndarray], dtype) -> None:
        if value is not None:
            value = np.asarray(value, dtype=dtype)
            if value.shape != self.shape:
                raise DimensionMismatchError(
                    f"{attr} shape {value.shape} does not match elevation shape {self.shape}."
                )
        setattr(self, f"_{attr}", value)

    @property
    def material_map(self) -> Optional[np.ndarray]:
        """(rows, cols) integer array of material ids, or `None`."""
        return self._material_map

    @material_map.setter
    def material_map(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("material_map", value, np.int32)

    @property
    def semantic_labels(self) -> Optional[np.ndarray]:
        """(rows, cols) integer array of semantic class ids, or `None`."""
        return self._semantic_labels

    @semantic_labels.setter
    def semantic_labels(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("semantic_labels", value, np.int32)

    @property
    def vegetation_mask(self) -> Optional[np.ndarray]:
        return self._vegetation_mask

    @vegetation_mask.setter
    def vegetation_mask(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("vegetation_mask", value, bool)

    @property
    def road_mask(self) -> Optional[np.ndarray]:
        return self._road_mask

    @road_mask.setter
    def road_mask(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("road_mask", value, bool)

    @property
    def water_mask(self) -> Optional[np.ndarray]:
        return self._water_mask

    @water_mask.setter
    def water_mask(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("water_mask", value, bool)

    @property
    def obstacle_mask(self) -> Optional[np.ndarray]:
        return self._obstacle_mask

    @obstacle_mask.setter
    def obstacle_mask(self, value: Optional[np.ndarray]) -> None:
        self._set_layer("obstacle_mask", value, bool)

    # ── derived layers (lazy, thread-safe, cached) ────────────────────

    def invalidate_derived(self) -> None:
        """Drop all cached derived layers (call after mutating `elevation`
        in place)."""
        with self._lock:
            self._normals = None
            self._slope = None
            self._aspect = None
            self._curvature = None
            self._spatial_index = None

    def compute_normals(self, force: bool = False) -> np.ndarray:
        """Per-cell surface normals via central differences.

        Returns:
            (rows, cols, 3) unit-normal-vector array, world-frame
            (x=east, y=north, z=up).
        """
        with self._lock:
            if self._normals is not None and not force:
                return self._normals
            dx, dy = self.cell_size
            gy, gx = np.gradient(self.elevation, dy, dx)
            # Surface normal of z = f(x, y) is proportional to (-df/dx, -df/dy, 1).
            # gy here is d(elevation)/d(row) which runs along -y, so flip sign
            # to express the gradient in the +y (north) direction.
            nx = -gx
            ny = gy
            nz = np.ones_like(self.elevation)
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            norm[norm == 0] = 1.0
            normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
            self._normals = normals
            return normals

    def compute_slope(self, units: SlopeUnits = SlopeUnits.DEGREES, force: bool = False) -> np.ndarray:
        """Per-cell terrain slope (angle from horizontal)."""
        with self._lock:
            if self._slope is not None and not force:
                slope_rad = self._slope
            else:
                dx, dy = self.cell_size
                gy, gx = np.gradient(self.elevation, dy, dx)
                slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
                self._slope = slope_rad
            if units is SlopeUnits.RADIANS:
                return slope_rad
            if units is SlopeUnits.DEGREES:
                return np.degrees(slope_rad)
            return np.tan(slope_rad) * 100.0  # percent grade

    def compute_aspect(self, force: bool = False) -> np.ndarray:
        """Per-cell aspect (downslope compass direction, degrees clockwise
        from north; flat cells are assigned -1)."""
        with self._lock:
            if self._aspect is not None and not force:
                return self._aspect
            dx, dy = self.cell_size
            gy, gx = np.gradient(self.elevation, dy, dx)
            aspect = np.degrees(np.arctan2(gx, gy))  # 0 = north, clockwise
            aspect = np.mod(aspect + 180.0, 360.0)   # downslope direction
            flat = (gx == 0) & (gy == 0)
            aspect = np.where(flat, -1.0, aspect)
            self._aspect = aspect
            return aspect

    def compute_curvature(self, force: bool = False) -> np.ndarray:
        """Per-cell profile curvature (second derivative of elevation);
        positive = convex, negative = concave."""
        with self._lock:
            if self._curvature is not None and not force:
                return self._curvature
            dx, dy = self.cell_size
            gy, gx = np.gradient(self.elevation, dy, dx)
            gyy, gyx = np.gradient(gy, dy, dx)
            gxy, gxx = np.gradient(gx, dy, dx)
            curvature = gxx + gyy  # discrete Laplacian
            self._curvature = curvature
            return curvature

    def compute_hillshade(self, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
        """Grayscale hillshade in [0, 255], standard cartographic
        illumination model (matches GDAL's `gdaldem hillshade` convention)."""
        normals = self.compute_normals()
        az = np.radians(azimuth_deg)
        alt = np.radians(altitude_deg)
        light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
        shade = np.clip(normals @ light, 0.0, 1.0)
        return (shade * 255).astype(np.uint8)

    # ── spatial index ─────────────────────────────────────────────────

    @property
    def spatial_index(self) -> SpatialIndex:
        """Lazily-built `SpatialIndex` over all valid (x, y, z) grid
        points, for nearest-point / KNN queries from `terrain_sampler`."""
        with self._lock:
            if self._spatial_index is None:
                rows, cols = self.shape
                row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
                x, y = self.grid_to_world(row_idx.astype(np.float64), col_idx.astype(np.float64))
                valid = self._valid_mask()
                points = np.stack([x[valid], y[valid], self.elevation[valid]], axis=-1)
                self._spatial_index = SpatialIndex(points)
            return self._spatial_index

    # ── integrity ─────────────────────────────────────────────────────

    def compute_checksum(self) -> str:
        """SHA-256 checksum over the elevation grid + geometric metadata.

        Used by `terrain_cache` for cache-key validation and by
        serialization round-trips to detect corruption. Auxiliary
        layers (materials/masks) are intentionally excluded so cosmetic
        re-tagging does not appear as a different terrain to the cache.
        """
        hasher = hashlib.sha256()
        hasher.update(self.elevation.tobytes())
        hasher.update(json.dumps({
            "cell_size": self.cell_size,
            "origin": self.origin,
            "shape": self.shape,
        }, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    # ── serialization ─────────────────────────────────────────────────

    def to_dict(self, include_arrays: bool = False) -> dict:
        """Structured metadata (and optionally raw arrays as nested
        lists -- expensive; prefer `to_npz`/`from_npz` for real data)."""
        d = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "shape": list(self.shape),
            "cell_size": list(self.cell_size),
            "origin": list(self.origin),
            "crs": self.crs.to_dict(),
            "metadata": self.metadata.to_dict(),
            "nodata_value": self.nodata_value,
            "height_range": list(self.height_range),
            "checksum": self.compute_checksum(),
        }
        if include_arrays:
            d["elevation"] = self.elevation.tolist()
        return d

    def to_npz(self, path: Union[str, Path]) -> None:
        """Serialize the full surface (grid + all layers + metadata) to a
        compressed `.npz` archive. This is the canonical persistence
        format for `TerrainSurface`."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {"elevation": self.elevation}
        for name in ("material_map", "semantic_labels", "vegetation_mask",
                     "road_mask", "water_mask", "obstacle_mask"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        header = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "cell_size": self.cell_size,
            "origin": self.origin,
            "crs": self.crs.to_dict(),
            "metadata": self.metadata.to_dict(),
            "nodata_value": self.nodata_value,
            "checksum": self.compute_checksum(),
            "layers_present": [k for k in arrays if k != "elevation"],
        }
        arrays["__header__"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        try:
            np.savez_compressed(path, **arrays)
        except Exception as exc:  # noqa: BLE001
            raise SerializationError(f"Failed writing TerrainSurface to '{path}': {exc}") from exc
        logger.info("TerrainSurface serialized -> %s (%d layers).", path, len(arrays) - 1)

    @classmethod
    def from_npz(cls, path: Union[str, Path], verify_checksum: bool = True) -> "TerrainSurface":
        """Deserialize a `TerrainSurface` previously written by `to_npz`.

        Raises:
            SerializationError: If the archive is malformed.
            ChecksumMismatchError: If `verify_checksum` is True and the
                stored checksum does not match the loaded elevation grid.
        """
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as npz:
                if "__header__" not in npz:
                    raise SerializationError(f"'{path}' is not a valid TerrainSurface archive (no header).")
                header = json.loads(bytes(npz["__header__"]).decode("utf-8"))
                elevation = npz["elevation"]
                surface = cls(
                    elevation=elevation,
                    cell_size=tuple(header["cell_size"]),
                    origin=tuple(header["origin"]),
                    crs=CoordinateReferenceSystem.from_dict(header["crs"]),
                    metadata=TerrainMetadata.from_dict(header["metadata"]),
                    nodata_value=header.get("nodata_value"),
                )
                for layer_name in header.get("layers_present", []):
                    if layer_name in npz:
                        setattr(surface, layer_name, npz[layer_name])
        except SerializationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SerializationError(f"Failed reading TerrainSurface from '{path}': {exc}") from exc

        if verify_checksum:
            actual = surface.compute_checksum()
            expected = header.get("checksum")
            if expected is not None and actual != expected:
                raise ChecksumMismatchError(
                    f"Checksum mismatch loading '{path}': expected {expected}, got {actual}."
                )
        logger.info("TerrainSurface deserialized <- %s (shape=%s).", path, surface.shape)
        return surface

    # ── misc ──────────────────────────────────────────────────────────

    def clone(self) -> "TerrainSurface":
        """Deep copy (arrays copied, not views)."""
        clone = TerrainSurface(
            elevation=self.elevation.copy(),
            cell_size=self.cell_size,
            origin=self.origin,
            crs=self.crs,
            metadata=TerrainMetadata.from_dict(self.metadata.to_dict()),
            nodata_value=self.nodata_value,
            validate=False,
        )
        for name in ("material_map", "semantic_labels", "vegetation_mask",
                     "road_mask", "water_mask", "obstacle_mask"):
            value = getattr(self, name)
            if value is not None:
                setattr(clone, name, value.copy())
        return clone

    def __repr__(self) -> str:
        rows, cols = self.shape
        zmin, zmax = self.height_range
        return (
            f"TerrainSurface(name={self.metadata.name!r}, shape=({rows}x{cols}), "
            f"cell_size={self.cell_size}, height_range=({zmin:.2f}, {zmax:.2f}), "
            f"crs={self.crs.name!r})"
        )


# ═════════════════════════════════════════════════════════════════════════
# Module-level validation helpers
# ═════════════════════════════════════════════════════════════════════════

def _validate_elevation_grid(elevation: np.ndarray) -> None:
    if elevation.ndim != 2:
        raise InvalidGridError(f"elevation grid must be 2D (rows, cols); got ndim={elevation.ndim}.")
    if elevation.shape[0] < 2 or elevation.shape[1] < 2:
        raise InvalidGridError(f"elevation grid too small for derivative ops: shape={elevation.shape}.")


__all__ = [
    "TerrainSurface",
    "CoordinateReferenceSystem",
    "BoundingBox3D",
    "TerrainMetadata",
    "SpatialIndex",
    "InterpolationMethod",
    "VerticalDatum",
    "SlopeUnits",
    "TerrainError",
    "TerrainSurfaceError",
    "InvalidGridError",
    "DimensionMismatchError",
    "SerializationError",
    "ChecksumMismatchError",
    "SpatialIndexError",
    "CURRENT_SCHEMA_VERSION",
]
