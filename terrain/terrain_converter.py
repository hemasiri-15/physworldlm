"""
terrain_converter.py
═══════════════════════════════════════════════════════════════════════════
Backend-independent conversion layer for PhysWorldLM.

`terrain_converter` turns a `TerrainSurface` -- the single canonical
terrain representation used across the terrain subsystem -- into other
backend-neutral representations: NumPy arrays, point clouds, triangle
meshes, heightmaps, normal maps, hillshade, occupancy/traversability/
navigation-cost rasters, curvature/aspect/slope rasters, binary masks,
material/semantic rasters, voxel grids, and signed distance fields. It
also provides optional, lazily-imported exporters for OBJ, PLY, STL,
GeoTIFF, PNG, USD, and `trimesh`.

Design notes
------------
    * This module never re-implements interpolation, differential
      geometry (slope/aspect/curvature/normals), or spatial indexing --
      those live on `TerrainSurface` and `TerrainSampler` and are always
      delegated to. `terrain_converter` only *reshapes* their outputs
      into other representations.
    * Every function is a pure, stateless transformation: it reads from
      the given `TerrainSurface` (and, for arbitrary-point queries, a
      fresh `TerrainSampler` built on top of it) and returns new data. No
      shared mutable module state is written on the hot path, so these
      functions are safe to call concurrently from multiple threads, the
      same as `TerrainSampler` itself.
    * Optional third-party writers (Pillow, `trimesh`, OpenUSD/`pxr`) are
      imported lazily, inside the function that needs them, so importing
      `terrain_converter` never requires any of them to be installed.
      OBJ/PLY/STL writers are implemented natively (no dependency).
    * `origin`, `cell_size` (including anisotropic dx != dy), and `crs`
      are always respected when mapping grid indices to world
      coordinates, and NODATA / non-finite cells are consistently
      excluded (or explicitly flagged) rather than silently corrupting
      downstream geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .terrain_sampler import TerrainSampler
from .terrain_surface import (
    CoordinateReferenceSystem,
    SlopeUnits,
    TerrainError,
    TerrainSurface,
)
from . import dem_loader

logger = logging.getLogger("physworldlm.terrain.terrain_converter")

PathLike = Union[str, Path]


# ═════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════

class TerrainConverterError(TerrainError):
    """Base class for `terrain_converter` failures."""


class UnsupportedConversionError(TerrainConverterError):
    """Raised when a requested layer/format/mode is not recognized."""


class MeshGenerationError(TerrainConverterError):
    """Raised when a triangle mesh cannot be generated from the surface
    (e.g. grid too small after striding, or entirely NODATA)."""


class VoxelizationError(TerrainConverterError):
    """Raised when a voxel grid / signed distance field cannot be built
    (e.g. non-positive voxel size, degenerate z-range)."""


class ExportError(TerrainConverterError):
    """Raised when writing a converted representation to disk fails,
    including missing optional third-party dependencies."""


# ═════════════════════════════════════════════════════════════════════════
# Lightweight backend-neutral result types
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PointCloud:
    """A backend-neutral scattered point set.

    Attributes:
        points: `(N, 3)` float64 array of `(x, y, z)` world coordinates.
        colors: Optional `(N, 3)` uint8 RGB array.
        normals: Optional `(N, 3)` float64 unit normal vectors.
        attributes: Optional extra per-point scalar/vector layers (e.g.
            `{'slope': (N,) array}`), keyed by layer name.
        crs: The `CoordinateReferenceSystem` the points are expressed in.
        source_name: Name of the `TerrainSurface` this was derived from.
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    crs: Optional[CoordinateReferenceSystem] = None
    source_name: Optional[str] = None

    def __len__(self) -> int:
        return int(self.points.shape[0])


@dataclass
class TriangleMesh:
    """A backend-neutral indexed triangle mesh.

    Attributes:
        vertices: `(V, 3)` float64 array of `(x, y, z)` world coordinates.
        faces: `(F, 3)` int64 array of vertex indices (CCW winding).
        normals: Optional `(V, 3)` float64 per-vertex unit normals.
        uvs: Optional `(V, 2)` float64 per-vertex texture coordinates.
        colors: Optional `(V, 3)` uint8 per-vertex RGB.
        crs: The `CoordinateReferenceSystem` the vertices are expressed in.
        source_name: Name of the `TerrainSurface` this was derived from.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    crs: Optional[CoordinateReferenceSystem] = None
    source_name: Optional[str] = None

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass
class VoxelGrid:
    """A backend-neutral dense boolean occupancy voxel grid.

    Attributes:
        occupancy: `(ny, nx, nz)` bool array; `occupancy[j, i, k]` is the
            voxel centered at `(origin[0] + i*vx, origin[1] - j*vy,
            origin[2] + k*vz)`, i.e. row `j` runs south from `origin`,
            column `i` runs east, layer `k` runs up -- the same row/col
            convention as `TerrainSurface.elevation`.
        origin: World `(x, y, z)` of the *center* of voxel `(0, 0, 0)`.
        voxel_size: `(vx, vy, vz)` voxel edge lengths.
        crs: The `CoordinateReferenceSystem` the grid is expressed in.
    """

    occupancy: np.ndarray
    origin: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    crs: Optional[CoordinateReferenceSystem] = None


@dataclass
class SignedDistanceField:
    """A backend-neutral dense signed-distance voxel grid.

    Values follow a heightfield-SDF convention: `values[j, i, k] =
    voxel_z - terrain_height_at(voxel_x, voxel_y)`, so positive values
    are above the terrain surface (outside) and negative values are
    below it (inside). Indexing/`origin`/`voxel_size` conventions match
    `VoxelGrid`.
    """

    values: np.ndarray
    origin: tuple[float, float, float]
    voxel_size: tuple[float, float, float]
    crs: Optional[CoordinateReferenceSystem] = None


# ═════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════

def _get_layer(surface: TerrainSurface, name: str) -> np.ndarray:
    """Resolve a named layer -- computed on demand or a stored auxiliary
    array -- from `surface`. Never mutates `surface`."""
    computed = {
        "elevation": lambda: surface.elevation,
        "slope": lambda: surface.compute_slope(units=SlopeUnits.DEGREES),
        "slope_degrees": lambda: surface.compute_slope(units=SlopeUnits.DEGREES),
        "aspect": lambda: surface.compute_aspect(),
        "curvature": lambda: surface.compute_curvature(),
        "normal": lambda: surface.compute_normals(),
        "normals": lambda: surface.compute_normals(),
    }
    if name in computed:
        return computed[name]()

    stored = {
        "material_map": surface.material_map,
        "semantic_labels": surface.semantic_labels,
        "vegetation_mask": surface.vegetation_mask,
        "road_mask": surface.road_mask,
        "water_mask": surface.water_mask,
        "obstacle_mask": surface.obstacle_mask,
    }
    if name in stored:
        layer = stored[name]
        if layer is None:
            raise UnsupportedConversionError(f"TerrainSurface has no '{name}' layer populated.")
        return layer

    raise UnsupportedConversionError(f"Unknown terrain layer '{name}'.")


def _grid_world_coords(
    surface: TerrainSurface, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build strided world-space `(x, y)` coordinate grids aligned with
    `surface.elevation[::stride, ::stride]`, honoring `origin` and
    (possibly anisotropic) `cell_size`.

    Returns:
        `(grid_x, grid_y, elevation, valid)`, all shape
        `(ceil(rows/stride), ceil(cols/stride))`. `valid` is `True` where
        elevation is finite and not equal to `nodata_value`.
    """
    rows, cols = surface.shape
    dx, dy = surface.cell_size
    ox, oy = surface.origin

    row_idx = np.arange(0, rows, stride)
    col_idx = np.arange(0, cols, stride)
    xs = ox + col_idx * dx
    ys = oy - row_idx * dy
    grid_x, grid_y = np.meshgrid(xs, ys)

    elevation = surface.elevation[np.ix_(row_idx, col_idx)]
    valid = np.isfinite(elevation)
    if surface.nodata_value is not None:
        valid &= elevation != surface.nodata_value

    return grid_x, grid_y, elevation, valid


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vectorized area-weighted vertex normals via face-normal scatter-add."""
    normals = np.zeros_like(vertices)
    tri = vertices[faces]  # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    face_normals = np.cross(e1, e2)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths[:, None]


def _voxel_grid_setup(
    surface: TerrainSurface,
    voxel_size: Optional[float],
    z_min: Optional[float],
    z_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Shared grid construction + terrain-height sampling for
    `to_voxel_grid` and `to_sdf`.

    Returns:
        `(terrain_z, zs, voxel_size, min_x, max_y, z_min)` where
        `terrain_z` has shape `(ny, nx)` and `zs` has shape `(nz,)`.
    """
    dx, dy = surface.cell_size
    if voxel_size is None:
        voxel_size = min(dx, dy)
    if voxel_size <= 0:
        raise VoxelizationError(f"voxel_size must be positive; got {voxel_size}.")

    rows, cols = surface.shape
    ox, oy = surface.origin
    world_width = (cols - 1) * dx
    world_height = (rows - 1) * dy
    min_x, max_x = ox - dx / 2.0, ox + world_width + dx / 2.0
    max_y, min_y = oy + dy / 2.0, oy - world_height - dy / 2.0

    zlo, zhi = surface.height_range
    z_min = zlo if z_min is None else z_min
    z_max = zhi if z_max is None else z_max
    if z_max <= z_min:
        raise VoxelizationError(f"Degenerate z-range for voxelization: [{z_min}, {z_max}].")

    nx = max(1, int(np.ceil((max_x - min_x) / voxel_size)))
    ny = max(1, int(np.ceil((max_y - min_y) / voxel_size)))
    nz = max(1, int(np.ceil((z_max - z_min) / voxel_size)))

    xs = min_x + (np.arange(nx) + 0.5) * voxel_size
    ys = max_y - (np.arange(ny) + 0.5) * voxel_size
    zs = z_min + (np.arange(nz) + 0.5) * voxel_size

    grid_x, grid_y = np.meshgrid(xs, ys)  # shape (ny, nx)
    sampler = TerrainSampler(surface)
    terrain_z = sampler.sample_height(grid_x.ravel(), grid_y.ravel(), clamp=True).reshape(grid_x.shape)

    return terrain_z, zs, voxel_size, min_x, max_y, z_min


# ═════════════════════════════════════════════════════════════════════════
# NumPy array export
# ═════════════════════════════════════════════════════════════════════════

def to_numpy(surface: TerrainSurface, *, layer: str = "elevation", copy: bool = True) -> np.ndarray:
    """Extract a named layer (`'elevation'`, `'slope'`, `'aspect'`,
    `'curvature'`, `'normal'`, or any populated auxiliary layer such as
    `'material_map'`) as a plain NumPy array.

    Args:
        copy: If `True` (default), return a copy so callers cannot
            mutate `surface`'s internal state through the result.

    Raises:
        UnsupportedConversionError: If `layer` is unknown or not
            populated on `surface`.
    """
    data = _get_layer(surface, layer)
    return data.copy() if copy else data


# ═════════════════════════════════════════════════════════════════════════
# Point cloud
# ═════════════════════════════════════════════════════════════════════════

def to_point_cloud(
    surface: TerrainSurface,
    *,
    stride: int = 1,
    include_nodata: bool = False,
    include_normals: bool = False,
    attribute_layers: Optional[Sequence[str]] = None,
) -> PointCloud:
    """Convert `surface` to a scattered `PointCloud` in world coordinates.

    Args:
        stride: Sub-sampling stride over rows/cols (1 = every cell).
        include_nodata: If `False` (default), NODATA/non-finite cells
            are dropped entirely rather than emitted as degenerate points.
        include_normals: If `True`, attach per-point normals from
            `surface.compute_normals()`.
        attribute_layers: Optional extra layer names (see `_get_layer`)
            to attach as per-point attributes.

    Raises:
        TerrainConverterError: If `stride < 1`.
    """
    if stride < 1:
        raise TerrainConverterError(f"stride must be >= 1; got {stride}.")

    grid_x, grid_y, elevation, valid = _grid_world_coords(surface, stride)
    mask = np.ones_like(valid) if include_nodata else valid

    points = np.stack([grid_x[mask], grid_y[mask], elevation[mask]], axis=-1).astype(np.float64)

    normals_out = None
    if include_normals:
        normals_full = surface.compute_normals()[::stride, ::stride]
        normals_out = normals_full[mask]

    attributes: dict[str, np.ndarray] = {}
    for layer_name in attribute_layers or ():
        layer = _get_layer(surface, layer_name)[::stride, ::stride]
        attributes[layer_name] = layer[mask]

    logger.debug(
        "to_point_cloud: %d points (stride=%d, include_nodata=%s)", points.shape[0], stride, include_nodata
    )
    return PointCloud(
        points=points, normals=normals_out, attributes=attributes,
        crs=surface.crs, source_name=surface.metadata.name,
    )


# ═════════════════════════════════════════════════════════════════════════
# Triangle mesh
# ═════════════════════════════════════════════════════════════════════════

def to_mesh(
    surface: TerrainSurface,
    *,
    stride: int = 1,
    include_nodata: bool = False,
    compute_normals: bool = True,
) -> TriangleMesh:
    """Convert `surface` to a regular-grid `TriangleMesh` (two triangles
    per grid cell) in world coordinates.

    Args:
        stride: Sub-sampling stride over rows/cols (1 = every cell).
        include_nodata: If `False` (default), any quad touching a
            NODATA/non-finite corner is skipped entirely.
        compute_normals: If `True`, attach area-weighted vertex normals.

    Raises:
        TerrainConverterError: If `stride < 1`.
        MeshGenerationError: If the (strided) grid is smaller than 2x2,
            or every quad is excluded as NODATA.
    """
    if stride < 1:
        raise TerrainConverterError(f"stride must be >= 1; got {stride}.")

    grid_x, grid_y, elevation, valid = _grid_world_coords(surface, stride)
    rows, cols = elevation.shape
    if rows < 2 or cols < 2:
        raise MeshGenerationError(f"Grid too small to mesh after striding ({rows}x{cols}); reduce stride.")

    vertices = np.stack([grid_x, grid_y, elevation], axis=-1).reshape(-1, 3).astype(np.float64)
    vert_idx = np.arange(rows * cols).reshape(rows, cols)

    v00 = vert_idx[:-1, :-1].ravel()
    v10 = vert_idx[1:, :-1].ravel()
    v01 = vert_idx[:-1, 1:].ravel()
    v11 = vert_idx[1:, 1:].ravel()

    tri_a = np.stack([v00, v10, v01], axis=-1)
    tri_b = np.stack([v01, v10, v11], axis=-1)

    if not include_nodata:
        valid_flat = valid.ravel()
        quad_ok = valid_flat[v00] & valid_flat[v10] & valid_flat[v01] & valid_flat[v11]
        tri_a, tri_b = tri_a[quad_ok], tri_b[quad_ok]

    faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int64)
    if faces.shape[0] == 0:
        raise MeshGenerationError("No valid triangles could be generated (all quads touch NODATA?).")

    normals = _compute_vertex_normals(vertices, faces) if compute_normals else None
    logger.debug("to_mesh: %d vertices, %d faces (stride=%d)", vertices.shape[0], faces.shape[0], stride)
    return TriangleMesh(
        vertices=vertices, faces=faces, normals=normals,
        crs=surface.crs, source_name=surface.metadata.name,
    )


# ═════════════════════════════════════════════════════════════════════════
# Heightmap / normal map / hillshade
# ═════════════════════════════════════════════════════════════════════════

def to_heightmap(
    surface: TerrainSurface,
    *,
    bit_depth: int = 16,
    min_elevation: Optional[float] = None,
    max_elevation: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rescale `surface.elevation` into a `bit_depth`-bit grayscale
    heightmap image array.

    Args:
        bit_depth: `8` -> `uint8`, anything else -> `uint16`.
        min_elevation, max_elevation: Elevation range mapped to
            `[0, 2**bit_depth - 1]`; defaults to `surface.height_range`.

    Returns:
        `(image, params)` where `image` is the quantized array and
        `params` records `{'min_elevation', 'max_elevation',
        'bit_depth'}` for exact round-tripping back to elevation.

    Raises:
        TerrainConverterError: If the resolved elevation range is
            degenerate (`max_elevation <= min_elevation`).
    """
    lo = surface.height_range[0] if min_elevation is None else min_elevation
    hi = surface.height_range[1] if max_elevation is None else max_elevation
    if hi <= lo:
        raise TerrainConverterError(f"Degenerate elevation range for heightmap: [{lo}, {hi}].")

    max_val = (2 ** bit_depth) - 1
    normalized = np.clip((surface.elevation - lo) / (hi - lo), 0.0, 1.0)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    image = (normalized * max_val).astype(dtype)

    return image, {"min_elevation": float(lo), "max_elevation": float(hi), "bit_depth": bit_depth}


def to_normal_map(surface: TerrainSurface) -> np.ndarray:
    """Encode `surface.compute_normals()` (unit vectors in `[-1, 1]`) as
    a `uint8` RGB tangent-space-style normal map image via the standard
    `(n * 0.5 + 0.5) * 255` mapping."""
    normals = surface.compute_normals()
    return np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def to_hillshade(surface: TerrainSurface, *, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Compute a `uint8` grayscale hillshade image using the standard
    ESRI illumination model, built from `surface.compute_slope()` and
    `surface.compute_aspect()` (never re-deriving slope/aspect itself).

    Args:
        azimuth: Sun azimuth in degrees, clockwise from north.
        altitude: Sun altitude above the horizon in degrees.
    """
    slope_rad = np.radians(surface.compute_slope(units=SlopeUnits.DEGREES))
    aspect_rad = np.radians(surface.compute_aspect())

    zenith_rad = np.radians(90.0 - altitude)
    azimuth_math = 360.0 - azimuth + 90.0
    if azimuth_math >= 360.0:
        azimuth_math -= 360.0
    azimuth_rad = np.radians(azimuth_math)

    shade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    return np.clip(shade, 0.0, 1.0).astype(np.float64) * 255.0 if False else (
        np.clip(shade * 255.0, 0, 255).astype(np.uint8)
    )


# ═════════════════════════════════════════════════════════════════════════
# Occupancy / traversability / navigation cost
# ═════════════════════════════════════════════════════════════════════════

def to_occupancy_grid(
    surface: TerrainSurface,
    *,
    max_traversable_slope_deg: float = 30.0,
    use_obstacle_mask: bool = True,
) -> np.ndarray:
    """Compute a boolean occupancy grid: `True` where a cell is blocked
    (slope exceeds `max_traversable_slope_deg`, is flagged in
    `surface.obstacle_mask`, or is NODATA/non-finite)."""
    occupancy = surface.compute_slope(units=SlopeUnits.DEGREES) > max_traversable_slope_deg
    if use_obstacle_mask and surface.obstacle_mask is not None:
        occupancy = occupancy | surface.obstacle_mask.astype(bool)
    if surface.nodata_value is not None:
        occupancy = occupancy | (surface.elevation == surface.nodata_value)
    occupancy = occupancy | ~np.isfinite(surface.elevation)
    return occupancy


def to_traversability_map(
    surface: TerrainSurface,
    *,
    max_slope_deg: float = 30.0,
    slope_weight: float = 0.6,
    roughness_weight: float = 0.4,
    impassable_value: float = 0.0,
) -> np.ndarray:
    """Compute a continuous `[0, 1]` traversability score (1 = fully
    traversable, `impassable_value` = blocked), combining a slope-based
    score and a curvature-based roughness score, then zeroing out any
    cell flagged in `surface.water_mask` / `surface.obstacle_mask` or
    that is NODATA/non-finite.
    """
    slope = surface.compute_slope(units=SlopeUnits.DEGREES)
    slope_score = np.clip(1.0 - slope / max_slope_deg, 0.0, 1.0)

    curvature = surface.compute_curvature()
    roughness = np.abs(curvature)
    finite_roughness = roughness[np.isfinite(roughness)]
    r_max = float(np.percentile(finite_roughness, 95)) if finite_roughness.size else 1.0
    r_max = r_max if r_max > 0 else 1.0
    roughness_score = np.clip(1.0 - roughness / r_max, 0.0, 1.0)

    score = slope_weight * slope_score + roughness_weight * roughness_score

    if surface.water_mask is not None:
        score = np.where(surface.water_mask.astype(bool), impassable_value, score)
    if surface.obstacle_mask is not None:
        score = np.where(surface.obstacle_mask.astype(bool), impassable_value, score)
    if surface.nodata_value is not None:
        score = np.where(surface.elevation == surface.nodata_value, impassable_value, score)
    score = np.where(np.isfinite(surface.elevation), score, impassable_value)

    return np.clip(score, 0.0, 1.0)


def to_navigation_cost_map(
    surface: TerrainSurface,
    *,
    base_cost: float = 1.0,
    max_slope_deg: float = 30.0,
    slope_weight: float = 0.6,
    roughness_weight: float = 0.4,
    impassable_cost: float = np.inf,
) -> np.ndarray:
    """Compute a per-cell path-planning cost grid: `base_cost /
    traversability` where traversable, `impassable_cost` elsewhere.
    Built on top of `to_traversability_map`, not a separate model."""
    traversability = to_traversability_map(
        surface, max_slope_deg=max_slope_deg, slope_weight=slope_weight,
        roughness_weight=roughness_weight, impassable_value=0.0,
    )
    cost = np.full(surface.shape, impassable_cost, dtype=np.float64)
    passable = traversability > 0.0
    cost[passable] = base_cost / traversability[passable]
    return cost


# ═════════════════════════════════════════════════════════════════════════
# Curvature / aspect / slope rasters
# ═════════════════════════════════════════════════════════════════════════

def to_slope_map(surface: TerrainSurface, *, units: SlopeUnits = SlopeUnits.DEGREES) -> np.ndarray:
    """Return `surface.compute_slope(units=units)`."""
    return surface.compute_slope(units=units)


def to_aspect_map(surface: TerrainSurface) -> np.ndarray:
    """Return `surface.compute_aspect()`."""
    return surface.compute_aspect()


def to_curvature_map(surface: TerrainSurface) -> np.ndarray:
    """Return `surface.compute_curvature()`."""
    return surface.compute_curvature()


# ═════════════════════════════════════════════════════════════════════════
# Binary masks / material / semantic rasters
# ═════════════════════════════════════════════════════════════════════════

def to_binary_mask(
    surface: TerrainSurface,
    *,
    layer: str = "elevation",
    mode: str = "nodata",
    threshold: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> np.ndarray:
    """Threshold any named layer (see `_get_layer`) into a boolean mask.

    Args:
        mode: One of `'nodata'` (default; NODATA/non-finite cells),
            `'above'`/`'below'`/`'equal'` (vs. `threshold`), or
            `'between'` (inclusive `[low, high]`).

    Raises:
        TerrainConverterError: If a mode-required parameter is missing.
        UnsupportedConversionError: If `mode` is unrecognized.
    """
    data = _get_layer(surface, layer)

    if mode == "nodata":
        mask = ~np.isfinite(data)
        if layer == "elevation" and surface.nodata_value is not None:
            mask = mask | (data == surface.nodata_value)
    elif mode == "above":
        if threshold is None:
            raise TerrainConverterError("mode='above' requires 'threshold'.")
        mask = data > threshold
    elif mode == "below":
        if threshold is None:
            raise TerrainConverterError("mode='below' requires 'threshold'.")
        mask = data < threshold
    elif mode == "equal":
        if threshold is None:
            raise TerrainConverterError("mode='equal' requires 'threshold'.")
        mask = data == threshold
    elif mode == "between":
        if low is None or high is None:
            raise TerrainConverterError("mode='between' requires 'low' and 'high'.")
        mask = (data >= low) & (data <= high)
    else:
        raise UnsupportedConversionError(f"Unknown binary mask mode '{mode}'.")

    return mask


def to_material_raster(surface: TerrainSurface) -> np.ndarray:
    """Return a copy of `surface.material_map`.

    Raises:
        UnsupportedConversionError: If no material map is populated.
    """
    return _get_layer(surface, "material_map").copy()


def to_semantic_raster(surface: TerrainSurface) -> np.ndarray:
    """Return a copy of `surface.semantic_labels`.

    Raises:
        UnsupportedConversionError: If no semantic label layer is populated.
    """
    return _get_layer(surface, "semantic_labels").copy()


# ═════════════════════════════════════════════════════════════════════════
# Voxel grid / signed distance field
# ═════════════════════════════════════════════════════════════════════════

def to_voxel_grid(
    surface: TerrainSurface,
    *,
    voxel_size: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    mode: str = "solid_below",
) -> VoxelGrid:
    """Voxelize `surface` into a dense boolean `VoxelGrid`, sampling
    terrain height at each voxel column's center via `TerrainSampler`
    (never re-implementing interpolation here).

    Args:
        voxel_size: Cubic voxel edge length; defaults to
            `min(surface.cell_size)`.
        z_min, z_max: Vertical extent to voxelize; defaults to
            `surface.height_range`.
        mode: `'solid_below'` (default; voxel is occupied if its center
            is at or below the terrain, i.e. a solid half-space under
            the heightfield) or `'shell'` (only voxels within half a
            voxel of the terrain surface are occupied).

    Raises:
        VoxelizationError: If `voxel_size <= 0` or the z-range is
            degenerate, or `mode` is unrecognized.
    """
    terrain_z, zs, voxel_size, min_x, max_y, z_min = _voxel_grid_setup(surface, voxel_size, z_min, z_max)
    ny, nx = terrain_z.shape
    nz = zs.shape[0]

    occupancy = np.zeros((ny, nx, nz), dtype=bool)
    if mode == "solid_below":
        for k, z in enumerate(zs):
            occupancy[:, :, k] = z <= terrain_z
    elif mode == "shell":
        half = voxel_size / 2.0
        for k, z in enumerate(zs):
            occupancy[:, :, k] = np.abs(z - terrain_z) <= half
    else:
        raise VoxelizationError(f"Unknown voxelization mode '{mode}'.")

    logger.debug("to_voxel_grid: %dx%dx%d voxels (mode=%s, voxel_size=%.4f)", ny, nx, nz, mode, voxel_size)
    return VoxelGrid(
        occupancy=occupancy, origin=(min_x, max_y, z_min),
        voxel_size=(voxel_size, voxel_size, voxel_size), crs=surface.crs,
    )


def to_sdf(
    surface: TerrainSurface,
    *,
    voxel_size: Optional[float] = None,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> SignedDistanceField:
    """Build a heightfield-based `SignedDistanceField`: `values[j, i, k]
    = voxel_z - terrain_height_at(voxel_x, voxel_y)` (positive above the
    surface, negative below), sampling terrain height via
    `TerrainSampler`.

    Args:
        voxel_size: Cubic voxel edge length; defaults to
            `min(surface.cell_size)`.
        z_min, z_max: Vertical extent to evaluate; defaults to
            `surface.height_range`.

    Raises:
        VoxelizationError: If `voxel_size <= 0` or the z-range is
            degenerate.
    """
    terrain_z, zs, voxel_size, min_x, max_y, z_min = _voxel_grid_setup(surface, voxel_size, z_min, z_max)
    ny, nx = terrain_z.shape
    nz = zs.shape[0]

    values = np.empty((ny, nx, nz), dtype=np.float64)
    for k, z in enumerate(zs):
        values[:, :, k] = z - terrain_z

    logger.debug("to_sdf: %dx%dx%d voxels (voxel_size=%.4f)", ny, nx, nz, voxel_size)
    return SignedDistanceField(
        values=values, origin=(min_x, max_y, z_min),
        voxel_size=(voxel_size, voxel_size, voxel_size), crs=surface.crs,
    )


# ═════════════════════════════════════════════════════════════════════════
# Optional exporters -- native (no dependency)
# ═════════════════════════════════════════════════════════════════════════

def export_obj(mesh: TriangleMesh, path: PathLike) -> None:
    """Write `mesh` as a Wavefront OBJ file. Pure-Python; no dependency."""
    path = Path(path)
    has_vn = mesh.normals is not None
    has_vt = mesh.uvs is not None
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# exported by terrain_converter.export_obj\n")
            for v in mesh.vertices:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            if has_vn:
                for n in mesh.normals:
                    fh.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            if has_vt:
                for uv in mesh.uvs:
                    fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for tri in mesh.faces:
                tokens = []
                for vi in tri:
                    vi1 = int(vi) + 1
                    if has_vt and has_vn:
                        tokens.append(f"{vi1}/{vi1}/{vi1}")
                    elif has_vn:
                        tokens.append(f"{vi1}//{vi1}")
                    elif has_vt:
                        tokens.append(f"{vi1}/{vi1}")
                    else:
                        tokens.append(f"{vi1}")
                fh.write("f " + " ".join(tokens) + "\n")
    except OSError as exc:
        raise ExportError(f"Failed writing OBJ to '{path}': {exc}") from exc
    logger.info("OBJ mesh written -> %s (%d verts, %d faces)", path, mesh.vertex_count, mesh.face_count)


def export_ply(obj: Union[PointCloud, TriangleMesh], path: PathLike, *, binary: bool = True) -> None:
    """Write a `PointCloud` or `TriangleMesh` as a Stanford PLY file.
    Pure-Python; no `plyfile`/`trimesh` dependency required.

    Raises:
        UnsupportedConversionError: If `obj` is neither a `PointCloud`
            nor a `TriangleMesh`.
    """
    path = Path(path)
    if isinstance(obj, TriangleMesh):
        vertices, faces, colors = obj.vertices, obj.faces, obj.colors
    elif isinstance(obj, PointCloud):
        vertices, faces, colors = obj.points, None, obj.colors
    else:
        raise UnsupportedConversionError(f"export_ply does not support type {type(obj).__name__}.")

    n_verts = int(vertices.shape[0])
    n_faces = 0 if faces is None else int(faces.shape[0])
    has_color = colors is not None

    header_lines = [
        "ply",
        f"format {'binary_little_endian 1.0' if binary else 'ascii 1.0'}",
        f"element vertex {n_verts}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        header_lines += ["property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        header_lines += [f"element face {n_faces}", "property list uchar int vertex_indices"]
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    try:
        with open(path, "wb") as fh:
            fh.write(header)
            if binary:
                vdtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
                if has_color:
                    vdtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
                record = np.zeros(n_verts, dtype=vdtype)
                record["x"], record["y"], record["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                if has_color:
                    record["red"], record["green"], record["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
                fh.write(record.tobytes())
                if faces is not None:
                    for tri in faces:
                        fh.write(np.uint8(3).tobytes())
                        fh.write(np.asarray(tri, dtype="<i4").tobytes())
            else:
                lines: list[str] = []
                for i in range(n_verts):
                    row = f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}"
                    if has_color:
                        row += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
                    lines.append(row)
                if faces is not None:
                    for tri in faces:
                        lines.append(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}")
                fh.write(("\n".join(lines) + "\n").encode("ascii"))
    except OSError as exc:
        raise ExportError(f"Failed writing PLY to '{path}': {exc}") from exc
    logger.info("PLY written -> %s (%d vertices, %d faces, binary=%s)", path, n_verts, n_faces, binary)


def export_stl(mesh: TriangleMesh, path: PathLike, *, solid_name: str = "terrain") -> None:
    """Write `mesh` as a binary STL file. Pure-Python; no dependency.
    Face normals are (re-)computed from triangle geometry rather than
    reusing per-vertex normals, per the STL spec's per-face convention.
    """
    path = Path(path)
    tri = mesh.vertices[mesh.faces]  # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    face_normals = np.cross(e1, e2)
    lengths = np.linalg.norm(face_normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    face_normals = face_normals / lengths[:, None]
    n_faces = tri.shape[0]

    try:
        with open(path, "wb") as fh:
            header = solid_name.encode("ascii", errors="replace")[:80].ljust(80, b"\0")
            fh.write(header)
            fh.write(np.uint32(n_faces).tobytes())
            dtype = [("normal", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3), ("attr", "<u2")]
            records = np.zeros(n_faces, dtype=dtype)
            records["normal"] = face_normals.astype("<f4")
            records["v0"] = tri[:, 0].astype("<f4")
            records["v1"] = tri[:, 1].astype("<f4")
            records["v2"] = tri[:, 2].astype("<f4")
            fh.write(records.tobytes())
    except OSError as exc:
        raise ExportError(f"Failed writing STL to '{path}': {exc}") from exc
    logger.info("Binary STL written -> %s (%d triangles)", path, n_faces)


# ═════════════════════════════════════════════════════════════════════════
# Optional exporters -- lazy third-party dependencies
# ═════════════════════════════════════════════════════════════════════════

def export_geotiff(surface: TerrainSurface, path: PathLike, **kwargs: Any) -> None:
    """Write `surface.elevation` as a georeferenced GeoTIFF, delegating
    entirely to `dem_loader.write_dem` (never reimplementing raster I/O).
    Accepts the same keyword arguments as `dem_loader.write_dem`.

    Raises:
        ExportError: If the DEM backend (`rasterio`/GDAL) is unavailable
            or the write fails.
    """
    try:
        dem_loader.write_dem(surface, path, **kwargs)
    except dem_loader.DemLoaderError as exc:
        raise ExportError(f"Failed writing GeoTIFF to '{path}': {exc}") from exc


def export_png(image: np.ndarray, path: PathLike) -> None:
    """Write a `uint8` (grayscale `(H, W)` or RGB `(H, W, 3)`) or
    `uint16` (grayscale `(H, W)`) image array as a PNG, via a lazily
    imported Pillow.

    Raises:
        ExportError: If Pillow is not installed, or the write fails.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ExportError("Exporting PNG requires Pillow ('pip install Pillow').") from exc

    path = Path(path)
    if image.dtype == np.uint16 and image.ndim == 2:
        mode = "I;16"
    elif image.dtype == np.uint8 and image.ndim == 2:
        mode = "L"
    elif image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        mode = "RGB"
    else:
        raise ExportError(f"Unsupported image dtype/shape for PNG export: {image.dtype}, {image.shape}.")

    try:
        Image.fromarray(image, mode=mode).save(path)
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"Failed writing PNG to '{path}': {exc}") from exc
    logger.info("PNG written -> %s (mode=%s, shape=%s)", path, mode, image.shape)


def to_trimesh(mesh: TriangleMesh) -> Any:
    """Convert `mesh` to a `trimesh.Trimesh`, via a lazily imported
    `trimesh`. Useful as a bridge to `trimesh`'s own broader export
    ecosystem (glTF, collision hulls, etc.) without this module
    depending on it directly.

    Raises:
        ExportError: If `trimesh` is not installed.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ExportError("Converting to trimesh requires the 'trimesh' package.") from exc
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals, process=False)


def export_usd(mesh: TriangleMesh, path: PathLike, *, mesh_path: str = "/Terrain") -> None:
    """Write `mesh` as a USD geometry stage, via a lazily imported
    `pxr` (OpenUSD). Only core `UsdGeom.Mesh` attributes are set
    (points, face-vertex counts/indices, normals) -- no materials,
    physics schemas, or scenegraph beyond a single mesh prim, keeping
    this a pure geometry export with no simulator coupling.

    Raises:
        ExportError: If `pxr` is not installed, or the write fails.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise ExportError("Exporting USD requires the 'pxr' (OpenUSD) package.") from exc

    path = Path(path)
    try:
        stage = Usd.Stage.CreateNew(str(path))
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        usd_mesh.CreatePointsAttr(mesh.vertices.astype(np.float32).tolist())
        usd_mesh.CreateFaceVertexCountsAttr([3] * mesh.face_count)
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.astype(np.int32).ravel().tolist())
        if mesh.normals is not None:
            usd_mesh.CreateNormalsAttr(mesh.normals.astype(np.float32).tolist())
        stage.GetRootLayer().Save()
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"Failed writing USD to '{path}': {exc}") from exc
    logger.info("USD mesh written -> %s (prim=%s)", path, mesh_path)


__all__ = [
    # result types
    "PointCloud",
    "TriangleMesh",
    "VoxelGrid",
    "SignedDistanceField",
    # array / point / mesh conversions
    "to_numpy",
    "to_point_cloud",
    "to_mesh",
    # image-space conversions
    "to_heightmap",
    "to_normal_map",
    "to_hillshade",
    # planning / analysis rasters
    "to_occupancy_grid",
    "to_traversability_map",
    "to_navigation_cost_map",
    "to_slope_map",
    "to_aspect_map",
    "to_curvature_map",
    "to_binary_mask",
    "to_material_raster",
    "to_semantic_raster",
    # volumetric conversions
    "to_voxel_grid",
    "to_sdf",
    # exporters
    "export_obj",
    "export_ply",
    "export_stl",
    "export_geotiff",
    "export_png",
    "export_usd",
    "to_trimesh",
    # exceptions
    "TerrainConverterError",
    "UnsupportedConversionError",
    "MeshGenerationError",
    "VoxelizationError",
    "ExportError",
]
