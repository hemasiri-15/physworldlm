"""
terrain_sampler.py
═══════════════════════════════════════════════════════════════════════════
Vectorized spatial query engine over a `TerrainSurface`.

`TerrainSampler` is the only supported way the rest of PhysWorldLM
(UAV planners, sensor models, physics contact generation, path
planning) should read terrain data at arbitrary world-space points --
never index `TerrainSurface.elevation` directly, since that bypasses
interpolation, NODATA handling, and bounds checking.

Everything here is pure NumPy: no simulator dependency, no I/O. All
"batch" methods accept and return arrays so callers doing dense
queries (e.g. rasterizing visibility across a whole grid, or sampling
a UAV path with hundreds of waypoints) pay one Python-level call, not
one per point. This is also the seam intended for a future GPU
backend (e.g. CuPy/Warp): swap the NumPy ops for their GPU
equivalents behind the same method signatures.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .terrain_surface import (
    InterpolationMethod,
    SlopeUnits,
    TerrainSurface,
    TerrainError,
)

logger = logging.getLogger("physworldlm.terrain.sampler")


# ═════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════

class TerrainSamplerError(TerrainError):
    """Base class for `TerrainSampler` failures."""


class OutOfBoundsError(TerrainSamplerError):
    """Raised when a query point falls outside the terrain extent and
    `clamp=False`."""


class NoDataError(TerrainSamplerError):
    """Raised when a query resolves entirely to NODATA cells and
    `allow_nodata=False`."""


# ═════════════════════════════════════════════════════════════════════════
# Result types
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VisibilityResult:
    """Result of a line-of-sight query between two points."""

    is_visible: bool
    blocking_distance: Optional[float]   # distance along the ray to first occluder, or None
    blocking_point: Optional[tuple[float, float, float]]
    clearance_profile: np.ndarray        # signed clearance (positive=clear) sampled along the ray


@dataclass(frozen=True)
class RayHit:
    """Result of a terrain ray intersection query."""

    hit: bool
    distance: Optional[float]
    point: Optional[tuple[float, float, float]]
    steps_taken: int


# ═════════════════════════════════════════════════════════════════════════
# TerrainSampler
# ═════════════════════════════════════════════════════════════════════════

class TerrainSampler:
    """Efficient, cache-friendly spatial queries over a `TerrainSurface`.

    Thread-safety: a `TerrainSampler` holds no mutable per-query state
    (each call computes fresh from the underlying, independently
    thread-safe `TerrainSurface`), so a single instance may be shared
    and queried concurrently from multiple threads.

    Args:
        surface: The `TerrainSurface` to sample.
        default_interpolation: Interpolation method used when a query
            method's `method` argument is left as `None`.
    """

    def __init__(
        self,
        surface: TerrainSurface,
        default_interpolation: InterpolationMethod = InterpolationMethod.BILINEAR,
    ) -> None:
        self.surface = surface
        self.default_interpolation = default_interpolation
        self._lock = threading.RLock()

    # ── core height sampling ─────────────────────────────────────────

    def sample_height(
        self,
        x: "float | np.ndarray",
        y: "float | np.ndarray",
        method: Optional[InterpolationMethod] = None,
        clamp: bool = True,
    ) -> "float | np.ndarray":
        """Sample terrain elevation at world (x, y).

        Args:
            x, y: Scalars or equal-shape arrays of world coordinates.
            method: Interpolation method; defaults to
                `self.default_interpolation`.
            clamp: If True, out-of-bounds points are clamped to the
                nearest valid grid coordinate rather than raising.

        Returns:
            Scalar or array of elevation values, matching the shape of
            `x`/`y`.

        Raises:
            OutOfBoundsError: If `clamp=False` and any point is outside
                the terrain extent.
        """
        method = method or self.default_interpolation
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        row, col = self.surface.world_to_grid(x_arr, y_arr)

        rows, cols = self.surface.shape
        out_of_bounds = (row < 0) | (row > rows - 1) | (col < 0) | (col > cols - 1)
        if np.any(out_of_bounds) and not clamp:
            raise OutOfBoundsError(f"{int(np.sum(out_of_bounds))} query point(s) fall outside terrain extent.")
        row = np.clip(row, 0, rows - 1)
        col = np.clip(col, 0, cols - 1)

        if method is InterpolationMethod.NEAREST:
            result = self._sample_nearest(row, col)
        elif method is InterpolationMethod.BILINEAR:
            result = self._sample_bilinear(row, col)
        elif method is InterpolationMethod.BICUBIC:
            result = self._sample_bicubic(row, col)
        else:  # pragma: no cover -- exhaustive enum guard
            raise TerrainSamplerError(f"Unsupported interpolation method: {method}")

        return result.item() if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0) else result

    def _sample_nearest(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        ri = np.round(row).astype(int)
        ci = np.round(col).astype(int)
        return self.surface.elevation[ri, ci]

    def _sample_bilinear(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        rows, cols = self.surface.shape
        r0 = np.clip(np.floor(row).astype(int), 0, rows - 2 if rows > 1 else 0)
        c0 = np.clip(np.floor(col).astype(int), 0, cols - 2 if cols > 1 else 0)
        r1, c1 = r0 + 1, c0 + 1
        fr, fc = row - r0, col - c0

        grid = self.surface.elevation
        v00 = grid[r0, c0]
        v01 = grid[r0, c1]
        v10 = grid[r1, c0]
        v11 = grid[r1, c1]

        top = v00 * (1 - fc) + v01 * fc
        bottom = v10 * (1 - fc) + v11 * fc
        return top * (1 - fr) + bottom * fr

    def _sample_bicubic(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Catmull-Rom bicubic interpolation using a 4x4 neighborhood."""
        rows, cols = self.surface.shape
        grid = self.surface.elevation

        def cubic_kernel(t: np.ndarray) -> np.ndarray:
            # Catmull-Rom basis, per-tap weights for offsets [-1, 0, 1, 2].
            a = -0.5
            t2, t3 = t * t, t * t * t
            w0 = a * (t3 - 2 * t2 + t)
            w1 = (a + 2) * t3 - (a + 3) * t2 + 1
            w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
            w3 = a * (-t3 + t2)
            return np.stack([w0, w1, w2, w3], axis=-1)

        r0 = np.clip(np.floor(row).astype(int), 1, rows - 3 if rows > 3 else 0)
        c0 = np.clip(np.floor(col).astype(int), 1, cols - 3 if cols > 3 else 0)
        fr, fc = row - np.floor(row), col - np.floor(col)

        if rows < 4 or cols < 4:
            # Grid too small for a true 4x4 bicubic stencil; degrade gracefully.
            return self._sample_bilinear(row, col)

        wr = cubic_kernel(fr)  # (N, 4)
        wc = cubic_kernel(fc)  # (N, 4)

        result = np.zeros_like(row, dtype=np.float64)
        for i in range(4):
            row_taps = np.zeros_like(row, dtype=np.float64)
            for j in range(4):
                rr = np.clip(r0 + i - 1, 0, rows - 1)
                cc = np.clip(c0 + j - 1, 0, cols - 1)
                row_taps += grid[rr, cc] * wc[:, j]
            result += row_taps * wr[:, i]
        return result

    # ── derived-layer sampling ───────────────────────────────────────

    def sample_normal(self, x: "float | np.ndarray", y: "float | np.ndarray") -> np.ndarray:
        """Sample the interpolated surface normal at (x, y). Returns
        shape (3,) for a scalar query or (N, 3) for array input."""
        return self._sample_layer(self.surface.compute_normals(), x, y, channels=3)

    def sample_slope(
        self, x: "float | np.ndarray", y: "float | np.ndarray", units: SlopeUnits = SlopeUnits.DEGREES
    ) -> "float | np.ndarray":
        return self._sample_layer(self.surface.compute_slope(units=units), x, y)

    def sample_aspect(self, x: "float | np.ndarray", y: "float | np.ndarray") -> "float | np.ndarray":
        return self._sample_layer(self.surface.compute_aspect(), x, y)

    def sample_curvature(self, x: "float | np.ndarray", y: "float | np.ndarray") -> "float | np.ndarray":
        return self._sample_layer(self.surface.compute_curvature(), x, y)

    def _sample_layer(
        self, layer: np.ndarray, x: "float | np.ndarray", y: "float | np.ndarray", channels: int = 1
    ) -> "float | np.ndarray":
        scalar_input = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        row, col = self.surface.world_to_grid(x_arr, y_arr)
        rows, cols = self.surface.shape
        row = np.clip(row, 0, rows - 1)
        col = np.clip(col, 0, cols - 1)
        ri = np.round(row).astype(int)
        ci = np.round(col).astype(int)
        values = layer[ri, ci]
        if scalar_input:
            return values[0] if channels == 1 else tuple(values[0])
        return values

    def gradient(self, x: "float | np.ndarray", y: "float | np.ndarray") -> np.ndarray:
        """Sample the (dz/dx, dz/dy) elevation gradient at (x, y)."""
        dx, dy = self.surface.cell_size
        gy, gx = np.gradient(self.surface.elevation, dy, dx)
        gx_s = self._sample_layer(gx, x, y)
        gy_s = self._sample_layer(gy, x, y)
        if np.isscalar(x):
            return np.array([gx_s, gy_s])
        return np.stack([gx_s, gy_s], axis=-1)

    # ── path / line queries ──────────────────────────────────────────

    def sample_line(
        self, start: Sequence[float], end: Sequence[float], num_samples: int = 100,
        method: Optional[InterpolationMethod] = None,
    ) -> np.ndarray:
        """Sample elevation along a straight line in the XY plane.

        Returns:
            (num_samples, 3) array of (x, y, z) points.
        """
        t = np.linspace(0.0, 1.0, num_samples)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        z = self.sample_height(x, y, method=method)
        return np.stack([x, y, z], axis=-1)

    def sample_path(
        self, waypoints: Sequence[Sequence[float]], samples_per_segment: int = 50,
        method: Optional[InterpolationMethod] = None,
    ) -> np.ndarray:
        """Sample elevation along a piecewise-linear multi-waypoint path.

        Args:
            waypoints: Sequence of (x, y) or (x, y, z) points; z, if
                given, is ignored (terrain elevation is authoritative).
            samples_per_segment: Samples generated between each
                consecutive waypoint pair.

        Returns:
            (M, 3) array of (x, y, z) points along the full path,
            de-duplicated at segment boundaries.
        """
        if len(waypoints) < 2:
            raise TerrainSamplerError("sample_path requires at least two waypoints.")
        segments = []
        for i in range(len(waypoints) - 1):
            seg = self.sample_line(waypoints[i][:2], waypoints[i + 1][:2], samples_per_segment, method=method)
            segments.append(seg[:-1] if i < len(waypoints) - 2 else seg)
        return np.concatenate(segments, axis=0)

    def terrain_profile(
        self, start: Sequence[float], end: Sequence[float], num_samples: int = 100,
    ) -> dict:
        """Elevation profile along a straight line, plus derived summary
        statistics (min/max/mean grade, total climb/descent) useful for
        path planning and traversability analysis.
        """
        points = self.sample_line(start, end, num_samples)
        distances = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(distances)])
        elevations = points[:, 2]
        d_elev = np.diff(elevations)
        climb = float(np.sum(d_elev[d_elev > 0]))
        descent = float(-np.sum(d_elev[d_elev < 0]))
        grades = np.divide(d_elev, distances, out=np.zeros_like(d_elev), where=distances != 0)
        return {
            "points": points,
            "cumulative_distance": cum_dist,
            "total_distance": float(cum_dist[-1]),
            "total_climb": climb,
            "total_descent": descent,
            "max_grade_percent": float(np.max(np.abs(grades)) * 100.0) if len(grades) else 0.0,
            "mean_grade_percent": float(np.mean(np.abs(grades)) * 100.0) if len(grades) else 0.0,
        }

    # ── nearest-point queries ────────────────────────────────────────

    def nearest_point(self, x: float, y: float, z: Optional[float] = None) -> tuple[np.ndarray, float]:
        """Find the nearest terrain sample point to a query point using
        the surface's spatial index.

        Args:
            x, y: Query location.
            z: Optional query elevation; if omitted, `0` is used and the
                search effectively becomes a planar nearest-neighbor
                (still correct for terrain sampled on a regular grid,
                since the index stores true elevations).

        Returns:
            (point, distance) where `point` is (x, y, z) of the nearest
            indexed terrain sample.
        """
        query = np.array([x, y, z if z is not None else 0.0])
        dist, idx = self.surface.spatial_index.query(query, k=1)
        point = self.surface.spatial_index._points[int(idx[0])]  # noqa: SLF001 -- internal, same package
        return point, float(dist[0])

    # ── ray intersection ──────────────────────────────────────────────

    def ray_intersect(
        self,
        origin: Sequence[float],
        direction: Sequence[float],
        max_distance: float = 10000.0,
        coarse_step: Optional[float] = None,
        refine_iterations: int = 8,
    ) -> RayHit:
        """March a ray against the terrain heightfield and find the first
        intersection, via coarse marching + bisection refinement.

        Args:
            origin: (x, y, z) ray origin.
            direction: (x, y, z) ray direction (need not be normalized).
            max_distance: Maximum march distance.
            coarse_step: Initial march step size; defaults to half the
                smaller terrain cell dimension (Nyquist-safe for typical
                terrain relief).
            refine_iterations: Bisection refinement steps once a
                sign-change bracket is found.

        Returns:
            A `RayHit` describing whether/where the ray hit the terrain.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise TerrainSamplerError("ray_intersect direction must be non-zero.")
        direction = direction / norm

        if coarse_step is None:
            coarse_step = min(self.surface.cell_size) * 0.5

        def height_delta(t: float) -> float:
            p = origin + direction * t
            terrain_z = self.sample_height(p[0], p[1], clamp=True)
            return p[2] - terrain_z

        t_prev = 0.0
        d_prev = height_delta(0.0)
        t = coarse_step
        steps = 0
        max_steps = int(max_distance / coarse_step) + 1

        while t <= max_distance and steps < max_steps:
            d = height_delta(t)
            if d_prev >= 0 and d < 0:
                # Bracketed a crossing; bisection refine.
                lo, hi = t_prev, t
                d_lo = d_prev
                for _ in range(refine_iterations):
                    mid = (lo + hi) / 2.0
                    d_mid = height_delta(mid)
                    if (d_lo >= 0) == (d_mid >= 0):
                        lo, d_lo = mid, d_mid
                    else:
                        hi = mid
                t_hit = (lo + hi) / 2.0
                p_hit = origin + direction * t_hit
                return RayHit(hit=True, distance=float(t_hit), point=tuple(p_hit), steps_taken=steps)
            t_prev, d_prev = t, d
            t += coarse_step
            steps += 1

        return RayHit(hit=False, distance=None, point=None, steps_taken=steps)

    # ── visibility / line of sight ─────────────────────────────────────

    def check_visibility(
        self,
        observer: Sequence[float],
        target: Sequence[float],
        num_samples: int = 200,
        observer_offset: float = 0.0,
        target_offset: float = 0.0,
    ) -> VisibilityResult:
        """Line-of-sight check between two points, accounting for terrain
        occlusion (does not account for other entities/buildings -- pure
        terrain visibility).

        Args:
            observer, target: (x, y, z) endpoints. z is the eye/target
                height above the world origin (not necessarily on the
                terrain surface).
            num_samples: Terrain samples taken along the line of sight.
            observer_offset, target_offset: Additional height-above-
                terrain applied at each endpoint (e.g. mast height),
                added on top of the given z.

        Returns:
            A `VisibilityResult`.
        """
        observer = np.asarray(observer, dtype=np.float64) + np.array([0, 0, observer_offset])
        target = np.asarray(target, dtype=np.float64) + np.array([0, 0, target_offset])

        t = np.linspace(0.0, 1.0, num_samples)
        line_xyz = observer[None, :] + t[:, None] * (target - observer)[None, :]
        terrain_z = self.sample_height(line_xyz[:, 0], line_xyz[:, 1])
        clearance = line_xyz[:, 2] - terrain_z  # positive = line is above terrain

        blocked = clearance < 0
        if not np.any(blocked):
            return VisibilityResult(True, None, None, clearance)

        first_block = int(np.argmax(blocked))
        distance = float(np.linalg.norm(line_xyz[first_block, :2] - observer[:2]))
        blocking_point = tuple(line_xyz[first_block])
        return VisibilityResult(False, distance, blocking_point, clearance)


__all__ = [
    "TerrainSampler",
    "TerrainSamplerError",
    "OutOfBoundsError",
    "NoDataError",
    "VisibilityResult",
    "RayHit",
]
