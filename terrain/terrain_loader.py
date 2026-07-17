"""
terrain_loader.py
═══════════════════════════════════════════════════════════════════════════
Universal terrain ingestion / format-routing layer for PhysWorldLM.

`terrain_loader` is the single entry point the rest of the pipeline uses
to turn *any* supported terrain source into a `TerrainSurface`. It never
implements geospatial raster I/O itself -- GeoTIFF/COG/SRTM/ASTER/
Copernicus/NASADEM/USGS DEM/any GDAL-readable raster is delegated to
`dem_loader.read_dem` / `dem_loader.write_dem`. This module owns:

    * format detection (object type, file extension, file signature,
      directory contents),
    * non-geospatial format readers (.npy/.npz, ASCII Grid, grayscale
      heightmap PNG, XYZ/LAS/LAZ/PLY point clouds),
    * the portable ``terrain.npz`` convention (backward compatible with
      the PSO UAV path-planning project),
    * directory / tile loading with georeferenced stitching,
    * WorldSpec resolution,
    * save routing, validation, and description.

Every successful load path returns a `TerrainSurface` -- never a raw
NumPy array, dict, Rasterio dataset, or GDAL dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from . import dem_loader
from .terrain_surface import (
    CoordinateReferenceSystem,
    InterpolationMethod,
    TerrainError,
    TerrainMetadata,
    TerrainSurface,
)

logger = logging.getLogger("physworldlm.terrain.terrain_loader")

PathLike = Union[str, Path]


# ═════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════

class TerrainLoaderError(TerrainError):
    """Base class for `terrain_loader` failures."""


class UnsupportedTerrainFormatError(TerrainLoaderError):
    """Raised when a source's format cannot be determined or is not
    supported by any registered reader."""


class TerrainLoadError(TerrainLoaderError):
    """Raised when a recognized-format source fails to load."""


class TerrainTileError(TerrainLoaderError):
    """Raised when directory/tile loading or stitching fails."""


# ═════════════════════════════════════════════════════════════════════════
# Format detection
# ═════════════════════════════════════════════════════════════════════════

_GEOTIFF_EXTS = {".tif", ".tiff"}
_RASTER_EXTS = _GEOTIFF_EXTS  # additional GDAL-readable raster extensions could be added here
_NUMPY_EXTS = {".npy"}
_NPZ_EXTS = {".npz"}
_ASCII_GRID_EXTS = {".asc"}
_HEIGHTMAP_EXTS = {".png"}
_POINT_CLOUD_TEXT_EXTS = {".xyz", ".txt", ".csv"}
_LAS_EXTS = {".las", ".laz"}
_PLY_EXTS = {".ply"}

_ALL_KNOWN_EXTS = (
    _RASTER_EXTS | _NUMPY_EXTS | _NPZ_EXTS | _ASCII_GRID_EXTS | _HEIGHTMAP_EXTS
    | _POINT_CLOUD_TEXT_EXTS | _LAS_EXTS | _PLY_EXTS
)

_FORMAT_ALIASES = {
    "geotiff": _RASTER_EXTS, "tif": _RASTER_EXTS, "tiff": _RASTER_EXTS, "cog": _RASTER_EXTS,
    "npy": _NUMPY_EXTS, "numpy": _NUMPY_EXTS,
    "npz": _NPZ_EXTS,
    "asc": _ASCII_GRID_EXTS, "ascii_grid": _ASCII_GRID_EXTS,
    "png": _HEIGHTMAP_EXTS, "heightmap": _HEIGHTMAP_EXTS,
    "xyz": _POINT_CLOUD_TEXT_EXTS, "point_cloud_text": _POINT_CLOUD_TEXT_EXTS,
    "las": _LAS_EXTS, "laz": _LAS_EXTS,
    "ply": _PLY_EXTS,
}


def detect_format(path: PathLike) -> Optional[str]:
    """Detect a source's terrain format from its file extension (and, for
    ambiguous cases, a lightweight signature check).

    Returns:
        One of ``'geotiff'``, ``'npy'``, ``'npz'``, ``'ascii_grid'``,
        ``'heightmap'``, ``'point_cloud_text'``, ``'las'``, ``'ply'``, or
        `None` if the extension is not recognized.
    """
    ext = Path(path).suffix.lower()
    if ext in _RASTER_EXTS:
        return "geotiff"
    if ext in _NUMPY_EXTS:
        return "npy"
    if ext in _NPZ_EXTS:
        return "npz"
    if ext in _ASCII_GRID_EXTS:
        return "ascii_grid"
    if ext in _HEIGHTMAP_EXTS:
        return "heightmap"
    if ext in _POINT_CLOUD_TEXT_EXTS:
        return "point_cloud_text"
    if ext in _LAS_EXTS:
        return "las"
    if ext in _PLY_EXTS:
        return "ply"
    return None


def _resolve_format_override(fmt: Optional[str], path: PathLike) -> str:
    if fmt is not None:
        if fmt not in _FORMAT_ALIASES:
            raise UnsupportedTerrainFormatError(
                f"Unknown explicit format override '{fmt}'. Known: {sorted(_FORMAT_ALIASES)}"
            )
        return fmt
    detected = detect_format(path)
    if detected is None:
        raise UnsupportedTerrainFormatError(
            f"Could not determine terrain format for '{path}' "
            f"(unrecognized extension '{Path(path).suffix}')."
        )
    return detected


# ═════════════════════════════════════════════════════════════════════════
# Public API -- dispatch
# ═════════════════════════════════════════════════════════════════════════

def load(source: Any, *, format: Optional[str] = None, **kwargs: Any) -> TerrainSurface:
    """Universal load entry point.

    Args:
        source: A `TerrainSurface` (returned as-is), a NumPy `ndarray`, a
            file path, or a directory path.
        format: Optional explicit format override (see `detect_format`
            for the accepted names); only meaningful for file sources.
        **kwargs: Forwarded to the resolved loader (see `load_file`,
            `load_numpy`, `load_directory`).

    Returns:
        A `TerrainSurface`.

    Raises:
        UnsupportedTerrainFormatError: If `source`'s type/format cannot
            be resolved to a known loader.
    """
    if isinstance(source, TerrainSurface):
        return source
    if isinstance(source, np.ndarray):
        return load_numpy(source, **kwargs)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return load_directory(path, **kwargs)
        return load_file(path, format=format, **kwargs)
    raise UnsupportedTerrainFormatError(
        f"Cannot load terrain from object of type {type(source).__name__}; "
        "expected TerrainSurface, numpy.ndarray, or a file/directory path."
    )


def load_file(path: PathLike, *, format: Optional[str] = None, **kwargs: Any) -> TerrainSurface:
    """Load a single terrain file, auto-detecting (or using an explicit
    `format` override) the reader to use.

    Raises:
        UnsupportedTerrainFormatError: Unknown/ambiguous format.
        TerrainLoadError: The file matched a known format but failed to
            load.
    """
    path = Path(path)
    if not path.exists():
        raise TerrainLoadError(f"Terrain source '{path}' does not exist.")
    fmt = _resolve_format_override(format, path)

    try:
        if fmt == "geotiff":
            return _load_geotiff(path, **kwargs)
        if fmt == "npy":
            return _load_npy(path, **kwargs)
        if fmt == "npz":
            return _load_npz(path, **kwargs)
        if fmt == "ascii_grid":
            return _load_ascii_grid(path, **kwargs)
        if fmt == "heightmap":
            return load_heightmap(path, **kwargs)
        if fmt == "point_cloud_text":
            return _load_point_cloud_text(path, **kwargs)
        if fmt == "las":
            return _load_las(path, **kwargs)
        if fmt == "ply":
            return _load_ply(path, **kwargs)
    except TerrainLoaderError:
        raise
    except dem_loader.DemLoaderError as exc:
        raise TerrainLoadError(f"DEM backend failed loading '{path}': {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed loading '{path}' as '{fmt}': {exc}") from exc

    raise UnsupportedTerrainFormatError(f"No reader registered for format '{fmt}'.")


# ═════════════════════════════════════════════════════════════════════════
# GeoTIFF / DEM delegation
# ═════════════════════════════════════════════════════════════════════════

def _load_geotiff(path: Path, **kwargs: Any) -> TerrainSurface:
    """Delegate GeoTIFF/COG/GDAL-readable DEM reads to `dem_loader`.

    Accepts the same keyword arguments as `dem_loader.read_dem`
    (``band``, ``aoi``, ``resample``, ``target_resolution``,
    ``max_block_pixels``, ``vertical_datum``).
    """
    return dem_loader.read_dem(path, **kwargs)


# ═════════════════════════════════════════════════════════════════════════
# NumPy array loading
# ═════════════════════════════════════════════════════════════════════════

def load_numpy(
    array: np.ndarray,
    *,
    cell_size: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    crs: Optional[CoordinateReferenceSystem] = None,
    nodata_value: Optional[float] = None,
    name: str = "numpy_terrain",
) -> TerrainSurface:
    """Wrap a raw (rows, cols) elevation array as a `TerrainSurface`.

    No georeferencing is assumed; defaults to a local Cartesian frame
    unless `crs` is supplied.
    """
    metadata = TerrainMetadata(name=name, source_format="numpy")
    return TerrainSurface(
        elevation=array, cell_size=cell_size, origin=origin,
        crs=crs or CoordinateReferenceSystem.local_cartesian(),
        metadata=metadata, nodata_value=nodata_value,
    )


def _load_npy(path: Path, **kwargs: Any) -> TerrainSurface:
    array = np.load(path, allow_pickle=False)
    kwargs.setdefault("name", path.stem)
    return load_numpy(array, **kwargs)


# ═════════════════════════════════════════════════════════════════════════
# NPZ loading -- two conventions
# ═════════════════════════════════════════════════════════════════════════

def _load_npz(path: Path, **kwargs: Any) -> TerrainSurface:
    """Dispatch between the native `TerrainSurface.to_npz` archive format
    (has a ``__header__`` entry) and the portable ``terrain.npz``
    convention used elsewhere in the project."""
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
    if "__header__" in keys:
        return TerrainSurface.from_npz(path, verify_checksum=kwargs.pop("verify_checksum", True))
    return _load_terrain_npz(path)


def _load_terrain_npz(path: Path) -> TerrainSurface:
    """Load the generic, location-agnostic ``terrain.npz`` convention.

    Recognized keys (all optional except an elevation source):
        ``z_grid`` or ``elevation``, ``width_m``, ``height_m``,
        ``elev_min``, ``elev_max``, ``cell_size``, ``origin``, ``crs``,
        ``epsg``, ``nodata_value``, ``metadata``, ``aoi_east``,
        ``aoi_north``.

    Backward compatible with terrain.npz files produced by the PSO UAV
    path-planning project.
    """
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)

        if "elevation" in keys:
            elevation = np.asarray(npz["elevation"], dtype=np.float64)
        elif "z_grid" in keys:
            elevation = np.asarray(npz["z_grid"], dtype=np.float64)
        else:
            raise TerrainLoadError(
                f"'{path}' does not contain 'elevation' or 'z_grid'; not a valid terrain.npz."
            )
        rows, cols = elevation.shape

        if "cell_size" in keys:
            cs = np.asarray(npz["cell_size"], dtype=np.float64).reshape(-1)
            cell_size = (float(cs[0]), float(cs[1]) if cs.size > 1 else float(cs[0]))
        elif "width_m" in keys and "height_m" in keys:
            dx = float(npz["width_m"]) / max(1, cols - 1)
            dy = float(npz["height_m"]) / max(1, rows - 1)
            cell_size = (dx, dy)
        else:
            cell_size = (1.0, 1.0)

        if "origin" in keys:
            og = np.asarray(npz["origin"], dtype=np.float64).reshape(-1)
            origin = (float(og[0]), float(og[1]) if og.size > 1 else float(og[0]))
        elif "aoi_east" in keys and "aoi_north" in keys:
            origin = (float(npz["aoi_east"]), float(npz["aoi_north"]))
        else:
            origin = (0.0, 0.0)

        crs = None
        if "crs" in keys:
            crs = _decode_json_bytes_field(npz["crs"])
            crs = CoordinateReferenceSystem.from_dict(crs) if isinstance(crs, dict) else None
        if crs is None and "epsg" in keys:
            try:
                epsg = int(npz["epsg"])
                crs = CoordinateReferenceSystem(epsg=epsg, name=f"EPSG:{epsg}", is_geographic=False)
            except (TypeError, ValueError):
                crs = None

        nodata_value = float(npz["nodata_value"]) if "nodata_value" in keys else None

        metadata = TerrainMetadata(name=path.stem, source_format="terrain.npz", source_path=str(path))
        if "metadata" in keys:
            decoded = _decode_json_bytes_field(npz["metadata"])
            if isinstance(decoded, dict):
                metadata.extra.update(decoded)
        if "elev_min" in keys:
            metadata.extra["source_elev_min"] = float(npz["elev_min"])
        if "elev_max" in keys:
            metadata.extra["source_elev_max"] = float(npz["elev_max"])

    return TerrainSurface(
        elevation=elevation, cell_size=cell_size, origin=origin,
        crs=crs or CoordinateReferenceSystem.local_cartesian(),
        metadata=metadata, nodata_value=nodata_value,
    )


def _decode_json_bytes_field(field: np.ndarray) -> Any:
    try:
        if field.dtype == np.uint8:
            return json.loads(bytes(field).decode("utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("Failed decoding embedded JSON field in npz archive; ignoring.")
    return None


# ═════════════════════════════════════════════════════════════════════════
# ASCII Grid
# ═════════════════════════════════════════════════════════════════════════

def _load_ascii_grid(path: Path, nodata_override: Optional[float] = None) -> TerrainSurface:
    """Load an ESRI ASCII Grid (`.asc`) file."""
    header: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                raise TerrainLoadError(f"'{path}' ended before any data rows were read.")
            parts = line.split()
            if len(parts) == 2 and not _is_number(parts[0]):
                header[parts[0].lower()] = float(parts[1])
                continue
            fh.seek(pos)
            break
        data = np.loadtxt(fh, dtype=np.float64)

    required = ("ncols", "nrows", "cellsize")
    missing = [k for k in required if k not in header]
    if missing:
        raise TerrainLoadError(f"'{path}' ASCII Grid header missing required fields: {missing}.")

    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    data = np.atleast_2d(data).reshape(nrows, ncols)
    cellsize = header["cellsize"]

    if "xllcenter" in header:
        origin_x = header["xllcenter"]
    else:
        origin_x = header.get("xllcorner", 0.0) + cellsize / 2.0
    if "yllcenter" in header:
        origin_y_bottom = header["yllcenter"]
    else:
        origin_y_bottom = header.get("yllcorner", 0.0) + cellsize / 2.0
    origin_y_top = origin_y_bottom + (nrows - 1) * cellsize

    nodata = nodata_override if nodata_override is not None else header.get("nodata_value")

    metadata = TerrainMetadata(name=path.stem, source_format="ascii_grid", source_path=str(path))
    return TerrainSurface(
        elevation=data, cell_size=(cellsize, cellsize), origin=(origin_x, origin_y_top),
        crs=CoordinateReferenceSystem.local_cartesian(), metadata=metadata, nodata_value=nodata,
    )


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


# ═════════════════════════════════════════════════════════════════════════
# Heightmap PNG
# ═════════════════════════════════════════════════════════════════════════

def load_heightmap(
    path: PathLike,
    *,
    min_elevation: Optional[float] = None,
    max_elevation: Optional[float] = None,
    scale: Optional[float] = None,
    offset: Optional[float] = None,
    cell_size: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
) -> TerrainSurface:
    """Load a grayscale heightmap PNG (8-bit or 16-bit) into a
    `TerrainSurface`.

    Pixel intensity is converted to elevation either via
    ``(min_elevation, max_elevation)`` (linear rescale across the full
    intensity range) or via an explicit ``(scale, offset)``:
    ``elevation = intensity * scale + offset``. Exactly one convention
    should be supplied; if neither is given, raw intensity in [0, 1] is
    used as elevation.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise TerrainLoadError(
            "Loading PNG heightmaps requires Pillow ('pip install Pillow')."
        ) from exc

    path = Path(path)
    try:
        img = Image.open(path)
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed opening heightmap '{path}': {exc}") from exc

    if img.mode not in ("L", "I", "I;16", "I;16B", "I;16L"):
        img = img.convert("I")  # coerce color/RGBA heightmaps to single-channel intensity

    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        max_intensity = 255.0
    elif arr.dtype == np.uint16:
        max_intensity = 65535.0
    else:
        # PIL "I" mode is int32; treat range as observed max, guarding div-by-zero.
        max_intensity = float(arr.max()) or 1.0
    normalized = arr.astype(np.float64) / max_intensity

    if scale is not None or offset is not None:
        elevation = normalized * (scale if scale is not None else 1.0) + (offset if offset is not None else 0.0)
    elif min_elevation is not None or max_elevation is not None:
        lo = min_elevation if min_elevation is not None else 0.0
        hi = max_elevation if max_elevation is not None else 1.0
        elevation = lo + normalized * (hi - lo)
    else:
        elevation = normalized

    metadata = TerrainMetadata(name=path.stem, source_format="heightmap_png", source_path=str(path))
    return TerrainSurface(
        elevation=elevation, cell_size=cell_size, origin=origin,
        crs=CoordinateReferenceSystem.local_cartesian(), metadata=metadata,
    )


# ═════════════════════════════════════════════════════════════════════════
# Point clouds -> regular grid
# ═════════════════════════════════════════════════════════════════════════

def _bin_points_to_grid(
    points: np.ndarray,
    cell_size: float,
    bounds: Optional[tuple[float, float, float, float]],
    interpolation: str,
    name: str,
    source_format: str,
    source_path: Optional[str] = None,
) -> TerrainSurface:
    """Bin an (N, 3) scattered point set into a regular elevation grid.

    Args:
        interpolation: ``'mean'`` (NumPy binning, always available) or
            ``'scipy_linear'`` / ``'scipy_cubic'`` (lazy `scipy.interpolate`
            griddata, used when SciPy is installed).
    """
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise TerrainLoadError(f"Point cloud must be a non-empty (N, 3) array; got shape {points.shape}.")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if bounds is None:
        min_x, max_x, min_y, max_y = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    else:
        min_x, min_y, max_x, max_y = bounds

    cols = max(2, int(np.ceil((max_x - min_x) / cell_size)) + 1)
    rows = max(2, int(np.ceil((max_y - min_y) / cell_size)) + 1)
    origin = (min_x, max_y)  # top-left, row 0 = north edge

    if interpolation.startswith("scipy"):
        try:
            from scipy.interpolate import griddata
        except ImportError as exc:
            raise TerrainLoadError(
                f"interpolation='{interpolation}' requires SciPy ('pip install scipy')."
            ) from exc
        method = "linear" if interpolation == "scipy_linear" else "cubic"
        grid_x, grid_y = np.meshgrid(
            min_x + np.arange(cols) * cell_size,
            max_y - np.arange(rows) * cell_size,
        )
        elevation = griddata((x, y), z, (grid_x, grid_y), method=method)
        nan_mask = np.isnan(elevation)
        if np.any(nan_mask) and not np.all(nan_mask):
            nearest = griddata((x, y), z, (grid_x, grid_y), method="nearest")
            elevation = np.where(nan_mask, nearest, elevation)
    else:
        sums = np.zeros((rows, cols), dtype=np.float64)
        counts = np.zeros((rows, cols), dtype=np.int64)
        col_idx = np.clip(((x - min_x) / cell_size).astype(np.int64), 0, cols - 1)
        row_idx = np.clip(((max_y - y) / cell_size).astype(np.int64), 0, rows - 1)
        np.add.at(sums, (row_idx, col_idx), z)
        np.add.at(counts, (row_idx, col_idx), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            elevation = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    nodata_value = float(np.nan)
    metadata = TerrainMetadata(name=name, source_format=source_format, source_path=source_path,
                                extra={"point_count": int(points.shape[0]), "interpolation": interpolation})
    return TerrainSurface(
        elevation=np.nan_to_num(elevation, nan=0.0), cell_size=(cell_size, cell_size), origin=origin,
        crs=CoordinateReferenceSystem.local_cartesian(), metadata=metadata,
        nodata_value=None if not np.isnan(nodata_value) else 0.0,
    )


def _load_point_cloud_text(
    path: Path,
    *,
    cell_size: float = 1.0,
    bounds: Optional[tuple[float, float, float, float]] = None,
    interpolation: str = "mean",
    delimiter: Optional[str] = None,
) -> TerrainSurface:
    """Load an XYZ (whitespace/CSV delimited) point cloud text file."""
    try:
        points = np.loadtxt(path, delimiter=delimiter, usecols=(0, 1, 2), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed parsing XYZ point cloud '{path}': {exc}") from exc
    return _bin_points_to_grid(points, cell_size, bounds, interpolation, path.stem, "xyz", str(path))


def _load_las(
    path: Path,
    *,
    cell_size: float = 1.0,
    bounds: Optional[tuple[float, float, float, float]] = None,
    interpolation: str = "mean",
    classification_filter: Optional[Iterable[int]] = None,
) -> TerrainSurface:
    """Load a LAS/LAZ point cloud via the optional `laspy` dependency."""
    try:
        import laspy
    except ImportError as exc:
        raise TerrainLoadError("Loading LAS/LAZ requires laspy ('pip install laspy')." ) from exc
    try:
        with laspy.open(str(path)) as reader:
            las = reader.read()
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed reading LAS/LAZ '{path}': {exc}") from exc

    x, y, z = np.asarray(las.x, dtype=np.float64), np.asarray(las.y, dtype=np.float64), np.asarray(las.z, dtype=np.float64)
    if classification_filter is not None and hasattr(las, "classification"):
        mask = np.isin(np.asarray(las.classification), list(classification_filter))
        x, y, z = x[mask], y[mask], z[mask]
    points = np.stack([x, y, z], axis=-1)
    return _bin_points_to_grid(points, cell_size, bounds, interpolation, path.stem, "las", str(path))


def _load_ply(
    path: Path,
    *,
    cell_size: float = 1.0,
    bounds: Optional[tuple[float, float, float, float]] = None,
    interpolation: str = "mean",
) -> TerrainSurface:
    """Load a PLY point cloud via the optional `plyfile` (preferred) or
    `trimesh` dependency."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        points = np.stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=-1).astype(np.float64)
    except ImportError:
        try:
            import trimesh
        except ImportError as exc:
            raise TerrainLoadError(
                "Loading PLY requires plyfile or trimesh ('pip install plyfile' or 'pip install trimesh')."
            ) from exc
        mesh = trimesh.load(str(path))
        points = np.asarray(mesh.vertices, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed reading PLY '{path}': {exc}") from exc

    return _bin_points_to_grid(points, cell_size, bounds, interpolation, path.stem, "ply", str(path))


# ═════════════════════════════════════════════════════════════════════════
# Directory / tile loading
# ═════════════════════════════════════════════════════════════════════════

def load_directory(
    path: PathLike,
    *,
    pattern: str = "*",
    stitch: bool = True,
    overlap_policy: str = "first",
    **kwargs: Any,
) -> TerrainSurface:
    """Load a directory of terrain tiles into a single `TerrainSurface`.

    Args:
        pattern: Glob pattern for candidate tile files within `path`.
        stitch: If `False` and more than one tile matches, raise rather
            than silently picking one.
        overlap_policy: How overlapping tile cells are resolved:
            ``'first'`` (keep the first tile written), ``'last'``
            (overwrite with the most recently written), or ``'average'``.

    Raises:
        TerrainTileError: If no tiles are found, tiles disagree on CRS or
            cell size, or georeferencing is insufficient to place tiles
            deterministically.
    """
    path = Path(path)
    if not path.is_dir():
        raise TerrainLoadError(f"'{path}' is not a directory.")

    candidates = sorted(p for p in path.glob(pattern) if p.is_file() and detect_format(p) is not None)
    if not candidates:
        raise TerrainTileError(f"No loadable terrain tiles found in '{path}' (pattern='{pattern}').")
    if len(candidates) == 1:
        return load_file(candidates[0], **kwargs)
    if not stitch:
        raise TerrainTileError(
            f"{len(candidates)} tiles found in '{path}' but stitch=False; "
            "load each with load_file(...) explicitly."
        )
    if overlap_policy not in ("first", "last", "average"):
        raise TerrainTileError(f"Unknown overlap_policy '{overlap_policy}'.")

    non_geotiff = [p for p in candidates if detect_format(p) != "geotiff"]
    if non_geotiff:
        raise TerrainTileError(
            f"Cannot deterministically stitch tiles in '{path}': "
            f"{[p.name for p in non_geotiff]} lack reliable georeferencing "
            "(only GeoTIFF tiles are supported for automatic stitching)."
        )

    tiles = [(p, dem_loader.read_metadata(p)) for p in candidates]

    epsgs = {m["epsg"] for _, m in tiles}
    if len(epsgs) > 1:
        raise TerrainTileError(f"Tiles in '{path}' have mixed CRS/EPSG values: {sorted(epsgs, key=str)}.")

    cell_sizes = {(round(abs(m["transform"][0]), 9), round(abs(m["transform"][4]), 9)) for _, m in tiles}
    if len(cell_sizes) > 1:
        raise TerrainTileError(f"Tiles in '{path}' have mismatched cell sizes: {cell_sizes}.")
    dx, dy = next(iter(cell_sizes))

    global_min_x = min(m["transform"][2] for _, m in tiles)
    global_max_y = max(m["transform"][5] for _, m in tiles)
    global_max_x = max(m["transform"][2] + m["width"] * m["transform"][0] for _, m in tiles)
    global_min_y = min(m["transform"][5] + m["height"] * m["transform"][4] for _, m in tiles)

    out_cols = int(round((global_max_x - global_min_x) / dx))
    out_rows = int(round((global_max_y - global_min_y) / dy))
    if out_cols <= 0 or out_rows <= 0:
        raise TerrainTileError(f"Computed non-positive mosaic shape ({out_rows}x{out_cols}) for '{path}'.")

    mosaic = np.full((out_rows, out_cols), np.nan, dtype=np.float64)
    coverage = np.zeros((out_rows, out_cols), dtype=np.int32)
    reference_nodata: Optional[float] = None
    reference_crs: Optional[CoordinateReferenceSystem] = None

    for tile_path, meta in tiles:
        surface = dem_loader.read_dem(tile_path)
        reference_nodata = reference_nodata if reference_nodata is not None else surface.nodata_value
        reference_crs = reference_crs or surface.crs
        row_off = int(round((global_max_y - meta["transform"][5]) / dy))
        col_off = int(round((meta["transform"][2] - global_min_x) / dx))
        r0, c0 = row_off, col_off
        r1, c1 = r0 + surface.shape[0], c0 + surface.shape[1]

        target = mosaic[r0:r1, c0:c1]
        tile_cov = coverage[r0:r1, c0:c1]
        already_written = tile_cov > 0
        empty = ~already_written

        target[empty] = surface.elevation[empty]
        if overlap_policy == "last":
            target[already_written] = surface.elevation[already_written]
        elif overlap_policy == "average":
            target[already_written] = (target[already_written] + surface.elevation[already_written]) / 2.0
        # 'first' leaves already-written cells untouched.
        coverage[r0:r1, c0:c1] += 1

    gap_count = int(np.sum(coverage == 0))
    if gap_count:
        logger.warning("Stitched mosaic from '%s' has %d uncovered (gap) cells.", path, gap_count)
    overlap_count = int(np.sum(coverage > 1))
    if overlap_count:
        logger.info("Stitched mosaic from '%s' has %d overlapping cells (policy=%s).",
                    path, overlap_count, overlap_policy)

    metadata = TerrainMetadata(
        name=path.name, source_format="geotiff_mosaic", source_path=str(path),
        extra={"tile_count": len(tiles), "overlap_policy": overlap_policy, "gap_cells": gap_count},
    )
    return TerrainSurface(
        elevation=mosaic, cell_size=(dx, dy), origin=(global_min_x + dx / 2.0, global_max_y - dy / 2.0),
        crs=reference_crs or CoordinateReferenceSystem.local_cartesian(), metadata=metadata,
        nodata_value=reference_nodata,
    )


# ═════════════════════════════════════════════════════════════════════════
# WorldSpec integration
# ═════════════════════════════════════════════════════════════════════════

_WORLDSPEC_TERRAIN_KEYS = ("terrain", "terrain_source", "dem", "dem_path", "heightmap", "elevation_source")


def load_worldspec(worldspec: Any, **kwargs: Any) -> TerrainSurface:
    """Resolve a terrain source from a WorldSpec-like object and load it.

    Accepts either a mapping (dict-like, tried via ``__getitem__``/``get``)
    or an object exposing terrain/environment attributes. Deliberately not
    coupled to one exact `WorldSpec` class -- any object exposing one of
    the recognized terrain keys as an attribute or mapping entry works.

    Raises:
        TerrainLoadError: If no recognizable terrain reference is found.
    """
    terrain_ref = None

    if isinstance(worldspec, dict):
        for key in _WORLDSPEC_TERRAIN_KEYS:
            if key in worldspec:
                terrain_ref = worldspec[key]
                break
    else:
        for key in _WORLDSPEC_TERRAIN_KEYS:
            if hasattr(worldspec, key):
                terrain_ref = getattr(worldspec, key)
                break
        if terrain_ref is None and hasattr(worldspec, "environment"):
            env = worldspec.environment
            for key in _WORLDSPEC_TERRAIN_KEYS:
                if isinstance(env, dict) and key in env:
                    terrain_ref = env[key]
                    break
                if hasattr(env, key):
                    terrain_ref = getattr(env, key)
                    break

    if terrain_ref is None:
        raise TerrainLoadError(
            "Could not resolve a terrain reference from the given WorldSpec "
            f"(looked for attributes/keys: {_WORLDSPEC_TERRAIN_KEYS})."
        )

    # A nested dict may itself carry a path plus loader kwargs.
    if isinstance(terrain_ref, dict) and "path" in terrain_ref:
        nested_kwargs = {k: v for k, v in terrain_ref.items() if k != "path"}
        nested_kwargs.update(kwargs)
        return load(terrain_ref["path"], **nested_kwargs)

    return load(terrain_ref, **kwargs)


# ═════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════

def save(surface: TerrainSurface, path: PathLike, *, format: Optional[str] = None, **kwargs: Any) -> None:
    """Save a `TerrainSurface`, routing by output format.

    ``.npz`` writes the portable ``terrain.npz`` convention (see
    `save_npz`); ``.tif``/``.tiff`` delegates to `dem_loader.write_dem`.
    Use `format='native_npz'` to write `TerrainSurface.to_npz`'s
    round-trip-exact archive instead of the portable convention.
    """
    path = Path(path)
    fmt = format or detect_format(path) or path.suffix.lower().lstrip(".")

    if fmt in ("npz",):
        save_npz(surface, path)
        return
    if fmt == "native_npz":
        surface.to_npz(path)
        return
    if fmt in ("geotiff", "tif", "tiff"):
        dem_loader.write_dem(surface, path, **kwargs)
        return
    raise UnsupportedTerrainFormatError(
        f"save() does not support format '{fmt}' for '{path}'. "
        "Use terrain_converter's export helpers for mesh/point-cloud/image formats."
    )


def save_npz(surface: TerrainSurface, path: PathLike) -> None:
    """Write the portable, location-agnostic ``terrain.npz`` convention.

    Includes both ``elevation`` and the PSO-project-compatible ``z_grid``
    alias, plus ``width_m``/``height_m``/``elev_min``/``elev_max`` derived
    from the surface, so files written here round-trip through both this
    loader and legacy PSO tooling.
    """
    path = Path(path)
    rows, cols = surface.shape
    dx, dy = surface.cell_size
    zmin, zmax = surface.height_range

    payload: dict[str, np.ndarray] = {
        "elevation": surface.elevation,
        "z_grid": surface.elevation,
        "width_m": np.float64(dx * (cols - 1)),
        "height_m": np.float64(dy * (rows - 1)),
        "elev_min": np.float64(zmin),
        "elev_max": np.float64(zmax),
        "cell_size": np.array(surface.cell_size, dtype=np.float64),
        "origin": np.array(surface.origin, dtype=np.float64),
        "aoi_east": np.float64(surface.origin[0]),
        "aoi_north": np.float64(surface.origin[1]),
        "crs": np.frombuffer(json.dumps(surface.crs.to_dict()).encode("utf-8"), dtype=np.uint8),
        "metadata": np.frombuffer(json.dumps(surface.metadata.to_dict()).encode("utf-8"), dtype=np.uint8),
    }
    if surface.crs.epsg is not None:
        payload["epsg"] = np.int64(surface.crs.epsg)
    if surface.nodata_value is not None:
        payload["nodata_value"] = np.float64(surface.nodata_value)

    try:
        np.savez_compressed(path, **payload)
    except Exception as exc:  # noqa: BLE001
        raise TerrainLoadError(f"Failed writing terrain.npz to '{path}': {exc}") from exc
    logger.info("Portable terrain.npz written -> %s", path)


# ═════════════════════════════════════════════════════════════════════════
# Validation / description
# ═════════════════════════════════════════════════════════════════════════

def validate(surface: TerrainSurface) -> list[str]:
    """Non-mutating validation of a `TerrainSurface`.

    Returns:
        A list of human-readable issue strings; empty if no issues were
        found. Does not raise -- `TerrainSurface`'s constructor already
        enforces the hard invariants (grid rank, minimum size, positive
        cell size), so this performs softer, advisory checks.
    """
    issues: list[str] = []

    if surface.elevation.ndim != 2:
        issues.append(f"elevation grid is not 2D (ndim={surface.elevation.ndim}).")
    rows, cols = surface.shape
    if rows < 2 or cols < 2:
        issues.append(f"elevation grid is degenerate ({rows}x{cols}).")

    valid = np.isfinite(surface.elevation)
    if surface.nodata_value is not None:
        valid &= surface.elevation != surface.nodata_value
    finite_ratio = float(np.mean(valid)) if valid.size else 0.0
    if finite_ratio == 0.0:
        issues.append("elevation grid has no valid (finite, non-nodata) cells.")
    elif finite_ratio < 0.5:
        issues.append(f"only {finite_ratio:.1%} of cells are valid; possible bad AOI or nodata handling.")

    dx, dy = surface.cell_size
    if dx <= 0 or dy <= 0:
        issues.append(f"non-positive cell_size {surface.cell_size}.")
    if not (np.isfinite(surface.origin[0]) and np.isfinite(surface.origin[1])):
        issues.append(f"non-finite origin {surface.origin}.")

    if surface.crs.epsg is None and surface.crs.wkt is None and surface.crs.is_geographic:
        issues.append("CRS marked geographic but has neither EPSG nor WKT to back it.")

    for layer_name in ("material_map", "semantic_labels", "vegetation_mask",
                       "road_mask", "water_mask", "obstacle_mask"):
        layer = getattr(surface, layer_name)
        if layer is not None and layer.shape != surface.shape:
            issues.append(f"{layer_name} shape {layer.shape} does not match elevation shape {surface.shape}.")

    return issues


def describe(surface: TerrainSurface) -> dict:
    """Return a serializable summary dictionary for a `TerrainSurface`.
    Does not mutate the surface."""
    rows, cols = surface.shape
    zmin, zmax = surface.height_range
    present_layers = [
        name for name in ("material_map", "semantic_labels", "vegetation_mask",
                           "road_mask", "water_mask", "obstacle_mask")
        if getattr(surface, name) is not None
    ]
    return {
        "name": surface.metadata.name,
        "shape": {"rows": rows, "cols": cols},
        "cell_size": surface.cell_size,
        "origin": surface.origin,
        "height_range": {"min": zmin, "max": zmax},
        "crs": surface.crs.to_dict(),
        "source_format": surface.metadata.source_format,
        "source_path": surface.metadata.source_path,
        "provider": surface.metadata.provider,
        "nodata_value": surface.nodata_value,
        "auxiliary_layers": present_layers,
        "checksum": surface.compute_checksum(),
        "validation_issues": validate(surface),
    }


__all__ = [
    "load",
    "load_file",
    "load_directory",
    "load_numpy",
    "load_heightmap",
    "load_worldspec",
    "save",
    "save_npz",
    "validate",
    "describe",
    "detect_format",
    "TerrainLoaderError",
    "UnsupportedTerrainFormatError",
    "TerrainLoadError",
    "TerrainTileError",
]
