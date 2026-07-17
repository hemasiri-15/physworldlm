"""
dem_loader.py
═══════════════════════════════════════════════════════════════════════════
Geospatial DEM ingestion for PhysWorldLM.

This module owns every dependency on a geospatial raster I/O library
(`rasterio` if available, else raw GDAL bindings). It is the *only* place
in the terrain subsystem that imports either -- `terrain_loader` delegates
here for GeoTIFF/COG/SRTM/ASTER/Copernicus/NASADEM/USGS DEM/OpenTopography
inputs, and everything downstream still only ever sees a `TerrainSurface`.

Design notes
------------
    * `rasterio` is tried first (simpler API, wraps GDAL); if unavailable,
      the raw `osgeo.gdal` bindings are used as a fallback. If neither is
      importable, every public function raises a clear
      `GeoBackendUnavailableError` rather than failing with an obscure
      `ImportError` deep in a call stack.
    * Reads are windowed by default for large DEMs (tiled/blocked reads
      via the backend's native block structure) so a multi-GB DEM does
      not require multi-GB of RAM just to inspect its metadata.
    * AOI clipping and resampling are applied at read time (not as a
      post-hoc crop/resize of a fully materialized array), so a small AOI
      out of a huge DEM only ever touches the relevant blocks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .terrain_surface import (
    CoordinateReferenceSystem,
    InterpolationMethod,
    TerrainError,
    TerrainMetadata,
    TerrainSurface,
    VerticalDatum,
)

logger = logging.getLogger("physworldlm.terrain.dem_loader")

PathLike = Union[str, Path]


# ═════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════

class DemLoaderError(TerrainError):
    """Base class for `dem_loader` failures."""


class GeoBackendUnavailableError(DemLoaderError):
    """Raised when neither `rasterio` nor GDAL is importable."""


class DemReadError(DemLoaderError):
    """Raised when a DEM file cannot be opened or read."""


class AoiError(DemLoaderError):
    """Raised when an AOI clip is invalid (outside raster extent, zero
    area, malformed bounds)."""


# ═════════════════════════════════════════════════════════════════════════
# Backend selection
# ═════════════════════════════════════════════════════════════════════════

class _Backend:
    RASTERIO = "rasterio"
    GDAL = "gdal"
    NONE = "none"


def _detect_backend() -> str:
    try:
        import rasterio  # noqa: F401
        return _Backend.RASTERIO
    except ImportError:
        pass
    try:
        from osgeo import gdal  # noqa: F401
        return _Backend.GDAL
    except ImportError:
        pass
    return _Backend.NONE


_BACKEND = None  # lazily resolved, cached at module scope


def active_backend() -> str:
    """Return the geospatial backend in use: `'rasterio'`, `'gdal'`, or
    `'none'` if neither is installed."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _detect_backend()
        logger.info("dem_loader geospatial backend: %s", _BACKEND)
    return _BACKEND


def _require_backend() -> str:
    backend = active_backend()
    if backend == _Backend.NONE:
        raise GeoBackendUnavailableError(
            "dem_loader requires either 'rasterio' (recommended: pip install rasterio) "
            "or GDAL Python bindings (pip install GDAL). Neither is importable."
        )
    return backend


# ═════════════════════════════════════════════════════════════════════════
# AOI / windows
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AOI:
    """Area of interest for clipped reads, in the raster's native CRS
    units (world coordinates, not pixel coordinates)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise AoiError(f"Degenerate AOI: {self}")


# ═════════════════════════════════════════════════════════════════════════
# Metadata inspection (cheap, no pixel data read)
# ═════════════════════════════════════════════════════════════════════════

def read_metadata(path: PathLike) -> dict:
    """Read only the header/metadata of a DEM file -- CRS, affine
    transform, shape, dtype, band count, nodata -- without materializing
    any pixel data. Cheap even for very large rasters.

    Returns:
        dict with keys: `driver`, `crs_wkt`, `epsg`, `transform`
        (`(a, b, c, d, e, f)` GDAL-style affine 6-tuple), `width`,
        `height`, `count` (band count), `dtypes`, `nodata`.
    """
    backend = _require_backend()
    path = str(path)

    if backend == _Backend.RASTERIO:
        import rasterio
        try:
            with rasterio.open(path) as ds:
                return {
                    "driver": ds.driver,
                    "crs_wkt": ds.crs.to_wkt() if ds.crs else None,
                    "epsg": ds.crs.to_epsg() if ds.crs else None,
                    "transform": tuple(ds.transform)[:6],
                    "width": ds.width,
                    "height": ds.height,
                    "count": ds.count,
                    "dtypes": list(ds.dtypes),
                    "nodata": ds.nodata,
                }
        except Exception as exc:  # noqa: BLE001
            raise DemReadError(f"Failed reading metadata from '{path}': {exc}") from exc

    # GDAL fallback
    from osgeo import gdal, osr
    ds = gdal.Open(path)
    if ds is None:
        raise DemReadError(f"GDAL could not open '{path}'.")
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(wkt=ds.GetProjection()) if ds.GetProjection() else None
    epsg = None
    if srs is not None and srs.IsProjected() or (srs is not None and srs.IsGeographic()):
        try:
            epsg = int(srs.GetAuthorityCode(None))
        except (TypeError, ValueError):
            epsg = None
    band = ds.GetRasterBand(1)
    return {
        "driver": ds.GetDriver().ShortName,
        "crs_wkt": ds.GetProjection() or None,
        "epsg": epsg,
        "transform": gt,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "count": ds.RasterCount,
        "dtypes": [gdal.GetDataTypeName(ds.GetRasterBand(i + 1).DataType) for i in range(ds.RasterCount)],
        "nodata": band.GetNoDataValue(),
    }


# ═════════════════════════════════════════════════════════════════════════
# Core read path
# ═════════════════════════════════════════════════════════════════════════

def read_dem(
    path: PathLike,
    band: int = 1,
    aoi: Optional[Union[AOI, tuple[float, float, float, float]]] = None,
    resample: Optional[InterpolationMethod] = None,
    target_resolution: Optional[tuple[float, float]] = None,
    max_block_pixels: int = 64_000_000,
    vertical_datum: Optional[VerticalDatum] = None,
) -> TerrainSurface:
    """Read a geospatial DEM (GeoTIFF/COG/SRTM/ASTER/Copernicus/NASADEM/
    USGS DEM/any GDAL-readable raster) into a `TerrainSurface`.

    Args:
        path: Path to the raster (local file; GDAL virtual filesystem
            paths like `/vsicurl/...` also work if GDAL is built with
            that support).
        band: 1-indexed band to read as elevation.
        aoi: Optional area-of-interest clip, either an `AOI` or a
            `(min_x, min_y, max_x, max_y)` tuple in the raster's native
            CRS units. When given, only the intersecting window is read
            from disk.
        resample: If set together with `target_resolution`, resamples
            during the read (GDAL-side, not a separate NumPy pass).
        target_resolution: Desired (dx, dy) output cell size; triggers
            resampling if it differs from the source resolution.
        max_block_pixels: Safety cap; if the requested read (after AOI
            clip) would exceed this many pixels, the read is
            automatically tiled internally (see `_read_tiled`) to bound
            peak memory instead of loading it all at once.
        vertical_datum: Override the vertical datum recorded in the
            resulting surface's CRS (source rasters rarely encode this
            reliably); defaults to `VerticalDatum.UNKNOWN`.

    Returns:
        A `TerrainSurface`.

    Raises:
        GeoBackendUnavailableError: If neither rasterio nor GDAL is
            installed.
        DemReadError: If the file cannot be opened/read.
        AoiError: If `aoi` does not intersect the raster.
    """
    backend = _require_backend()
    path = str(path)
    aoi_obj = AOI(*aoi) if isinstance(aoi, tuple) else aoi

    if backend == _Backend.RASTERIO:
        return _read_dem_rasterio(
            path, band, aoi_obj, resample, target_resolution, max_block_pixels, vertical_datum
        )
    return _read_dem_gdal(
        path, band, aoi_obj, resample, target_resolution, max_block_pixels, vertical_datum
    )


def _resample_enum_to_rasterio(method: Optional[InterpolationMethod]):
    from rasterio.enums import Resampling
    mapping = {
        InterpolationMethod.NEAREST: Resampling.nearest,
        InterpolationMethod.BILINEAR: Resampling.bilinear,
        InterpolationMethod.BICUBIC: Resampling.cubic,
    }
    return mapping.get(method, Resampling.bilinear)


def _read_dem_rasterio(
    path: str,
    band: int,
    aoi: Optional[AOI],
    resample: Optional[InterpolationMethod],
    target_resolution: Optional[tuple[float, float]],
    max_block_pixels: int,
    vertical_datum: Optional[VerticalDatum],
) -> TerrainSurface:
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import calculate_default_transform, reproject

    try:
        ds = rasterio.open(path)
    except Exception as exc:  # noqa: BLE001
        raise DemReadError(f"rasterio failed to open '{path}': {exc}") from exc

    with ds:
        window: Optional[Window] = None
        if aoi is not None:
            full_bounds = ds.bounds
            if (aoi.max_x < full_bounds.left or aoi.min_x > full_bounds.right or
                    aoi.max_y < full_bounds.bottom or aoi.min_y > full_bounds.top):
                raise AoiError(f"AOI {aoi} does not intersect raster bounds {tuple(full_bounds)}.")
            window = from_bounds(aoi.min_x, aoi.min_y, aoi.max_x, aoi.max_y, transform=ds.transform)
            window = window.round_offsets().round_lengths()

        win_transform = ds.window_transform(window) if window is not None else ds.transform
        win_height = int(window.height) if window is not None else ds.height
        win_width = int(window.width) if window is not None else ds.width

        src_dx, src_dy = abs(win_transform.a), abs(win_transform.e)
        out_shape = None
        out_transform = win_transform
        if target_resolution is not None:
            tdx, tdy = target_resolution
            out_width = max(1, int(round(win_width * src_dx / tdx)))
            out_height = max(1, int(round(win_height * src_dy / tdy)))
            out_shape = (out_height, out_width)
            out_transform = rasterio.Affine(tdx, win_transform.b, win_transform.c,
                                             win_transform.d, -tdy, win_transform.f)

        n_pixels = (out_shape[0] * out_shape[1]) if out_shape else (win_height * win_width)

        if n_pixels > max_block_pixels and out_shape is None:
            elevation = _read_tiled_rasterio(ds, band, window, win_height, win_width, max_block_pixels)
        else:
            resampling = _resample_enum_to_rasterio(resample) if out_shape else None
            read_kwargs: dict[str, Any] = {"window": window}
            if out_shape:
                read_kwargs["out_shape"] = out_shape
                read_kwargs["resampling"] = resampling
            elevation = ds.read(band, **read_kwargs).astype(np.float64)

        crs_epsg = ds.crs.to_epsg() if ds.crs else None
        crs = CoordinateReferenceSystem(
            epsg=crs_epsg,
            wkt=ds.crs.to_wkt() if ds.crs else None,
            name=ds.crs.to_string() if ds.crs else "unknown",
            is_geographic=bool(ds.crs and ds.crs.is_geographic),
            vertical_datum=vertical_datum or VerticalDatum.UNKNOWN,
        )
        nodata = ds.nodata
        dx, dy = abs(out_transform.a), abs(out_transform.e)
        origin = (out_transform.c + dx / 2.0, out_transform.f - dy / 2.0)

        metadata = TerrainMetadata(
            name=Path(path).stem,
            source_format=ds.driver,
            source_path=path,
            provider=_guess_provider(path),
            extra={"band": band, "backend": "rasterio"},
        )

    return TerrainSurface(
        elevation=elevation, cell_size=(dx, dy), origin=origin,
        crs=crs, metadata=metadata, nodata_value=nodata,
    )


def _read_tiled_rasterio(ds, band: int, window, win_height: int, win_width: int, max_block_pixels: int) -> np.ndarray:
    """Read a large window in row-block chunks bounded by
    `max_block_pixels`, so peak memory stays proportional to one chunk
    rather than the whole (potentially huge) requested extent."""
    from rasterio.windows import Window

    rows_per_chunk = max(1, max_block_pixels // max(1, win_width))
    out = np.empty((win_height, win_width), dtype=np.float64)
    base_col_off = window.col_off if window is not None else 0
    base_row_off = window.row_off if window is not None else 0

    logger.info(
        "DEM read exceeds max_block_pixels (%d px); tiling in %d-row chunks.",
        win_height * win_width, rows_per_chunk,
    )
    for row_start in range(0, win_height, rows_per_chunk):
        rows_here = min(rows_per_chunk, win_height - row_start)
        chunk_window = Window(base_col_off, base_row_off + row_start, win_width, rows_here)
        out[row_start:row_start + rows_here, :] = ds.read(band, window=chunk_window).astype(np.float64)
    return out


def _read_dem_gdal(
    path: str,
    band: int,
    aoi: Optional[AOI],
    resample: Optional[InterpolationMethod],
    target_resolution: Optional[tuple[float, float]],
    max_block_pixels: int,
    vertical_datum: Optional[VerticalDatum],
) -> TerrainSurface:
    from osgeo import gdal, osr
    gdal.UseExceptions()

    ds = gdal.Open(path)
    if ds is None:
        raise DemReadError(f"GDAL failed to open '{path}'.")

    gt = ds.GetGeoTransform()
    full_width, full_height = ds.RasterXSize, ds.RasterYSize

    x_off, y_off, width, height = 0, 0, full_width, full_height
    if aoi is not None:
        inv_col = lambda x: (x - gt[0]) / gt[1]
        inv_row = lambda y: (y - gt[3]) / gt[5]
        c0, c1 = sorted([inv_col(aoi.min_x), inv_col(aoi.max_x)])
        r0, r1 = sorted([inv_row(aoi.min_y), inv_row(aoi.max_y)])
        x_off, y_off = max(0, int(math.floor(c0))), max(0, int(math.floor(r0)))
        x_end, y_end = min(full_width, int(math.ceil(c1))), min(full_height, int(math.ceil(r1)))
        width, height = x_end - x_off, y_end - y_off
        if width <= 0 or height <= 0:
            raise AoiError(f"AOI {aoi} does not intersect raster bounds.")

    band_ds = ds.GetRasterBand(band)
    nodata = band_ds.GetNoDataValue()

    out_width, out_height = width, height
    if target_resolution is not None:
        tdx, tdy = target_resolution
        src_dx, src_dy = abs(gt[1]), abs(gt[5])
        out_width = max(1, int(round(width * src_dx / tdx)))
        out_height = max(1, int(round(height * src_dy / tdy)))

    n_pixels = out_width * out_height
    resample_alg = {
        InterpolationMethod.NEAREST: gdal.GRA_NearestNeighbour,
        InterpolationMethod.BILINEAR: gdal.GRA_Bilinear,
        InterpolationMethod.BICUBIC: gdal.GRA_Cubic,
    }.get(resample, gdal.GRA_Bilinear)

    if n_pixels > max_block_pixels:
        elevation = _read_tiled_gdal(band_ds, x_off, y_off, width, height, max_block_pixels)
        out_width, out_height = width, height
    else:
        elevation = band_ds.ReadAsArray(
            xoff=x_off, yoff=y_off, win_xsize=width, win_ysize=height,
            buf_xsize=out_width, buf_ysize=out_height, resample_alg=resample_alg,
        ).astype(np.float64)

    srs = osr.SpatialReference(wkt=ds.GetProjection()) if ds.GetProjection() else None
    epsg = None
    if srs is not None:
        try:
            epsg = int(srs.GetAuthorityCode(None))
        except (TypeError, ValueError):
            epsg = None
    crs = CoordinateReferenceSystem(
        epsg=epsg, wkt=ds.GetProjection() or None,
        name=srs.GetName() if srs else "unknown",
        is_geographic=bool(srs and srs.IsGeographic()),
        vertical_datum=vertical_datum or VerticalDatum.UNKNOWN,
    )

    dx = abs(gt[1]) * (width / out_width)
    dy = abs(gt[5]) * (height / out_height)
    origin_x = gt[0] + x_off * gt[1] + dx / 2.0
    origin_y = gt[3] + y_off * gt[5] - dy / 2.0 if gt[5] < 0 else gt[3] + y_off * gt[5] + dy / 2.0
    # GDAL geotransform e (gt[5]) is conventionally negative (north-up).
    origin_y = gt[3] + y_off * gt[5] + dy / 2.0

    metadata = TerrainMetadata(
        name=Path(path).stem, source_format=ds.GetDriver().ShortName, source_path=path,
        provider=_guess_provider(path), extra={"band": band, "backend": "gdal"},
    )
    return TerrainSurface(
        elevation=elevation, cell_size=(dx, dy), origin=(origin_x, origin_y),
        crs=crs, metadata=metadata, nodata_value=nodata,
    )


def _read_tiled_gdal(band_ds, x_off: int, y_off: int, width: int, height: int, max_block_pixels: int) -> np.ndarray:
    rows_per_chunk = max(1, max_block_pixels // max(1, width))
    out = np.empty((height, width), dtype=np.float64)
    logger.info(
        "DEM read exceeds max_block_pixels (%d px); tiling in %d-row chunks (GDAL backend).",
        height * width, rows_per_chunk,
    )
    for row_start in range(0, height, rows_per_chunk):
        rows_here = min(rows_per_chunk, height - row_start)
        out[row_start:row_start + rows_here, :] = band_ds.ReadAsArray(
            xoff=x_off, yoff=y_off + row_start, win_xsize=width, win_ysize=rows_here,
        ).astype(np.float64)
    return out


def _guess_provider(path: str) -> Optional[str]:
    lowered = path.lower()
    hints = {
        "srtm": "SRTM", "aster": "ASTER GDEM", "cop30": "Copernicus DEM",
        "cop90": "Copernicus DEM", "nasadem": "NASADEM", "usgs": "USGS 3DEP",
    }
    for key, name in hints.items():
        if key in lowered:
            return name
    return None


# ═════════════════════════════════════════════════════════════════════════
# Writing
# ═════════════════════════════════════════════════════════════════════════

def write_dem(surface: TerrainSurface, path: PathLike, driver: str = "GTiff", compress: str = "DEFLATE") -> None:
    """Write a `TerrainSurface` out as a georeferenced raster (GeoTIFF by
    default).

    Args:
        surface: Surface to export.
        path: Output path.
        driver: GDAL/rasterio short driver name (`'GTiff'`, `'COG'`, ...).
        compress: Compression method passed to the writer, if supported
            by `driver`.
    """
    backend = _require_backend()
    path = str(path)
    dx, dy = surface.cell_size
    ox, oy = surface.origin

    if backend == _Backend.RASTERIO:
        import rasterio
        transform = rasterio.Affine(dx, 0.0, ox - dx / 2.0, 0.0, -dy, oy + dy / 2.0)
        crs = None
        if surface.crs.wkt:
            crs = rasterio.crs.CRS.from_wkt(surface.crs.wkt)
        elif surface.crs.epsg:
            crs = rasterio.crs.CRS.from_epsg(surface.crs.epsg)
        profile = {
            "driver": driver, "height": surface.shape[0], "width": surface.shape[1],
            "count": 1, "dtype": "float64", "crs": crs, "transform": transform,
            "nodata": surface.nodata_value, "compress": compress,
        }
        try:
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(surface.elevation, 1)
        except Exception as exc:  # noqa: BLE001
            raise DemLoaderError(f"Failed writing DEM to '{path}': {exc}") from exc
        return

    from osgeo import gdal, osr
    gdal.UseExceptions()
    rows, cols = surface.shape
    driver_obj = gdal.GetDriverByName(driver)
    if driver_obj is None:
        raise DemLoaderError(f"GDAL driver '{driver}' not available.")
    creation_opts = [f"COMPRESS={compress}"] if compress else []
    ds = driver_obj.Create(path, cols, rows, 1, gdal.GDT_Float64, options=creation_opts)
    ds.SetGeoTransform((ox - dx / 2.0, dx, 0.0, oy + dy / 2.0, 0.0, -dy))
    if surface.crs.wkt or surface.crs.epsg:
        srs = osr.SpatialReference()
        if surface.crs.wkt:
            srs.ImportFromWkt(surface.crs.wkt)
        elif surface.crs.epsg:
            srs.ImportFromEPSG(surface.crs.epsg)
        ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if surface.nodata_value is not None:
        band.SetNoDataValue(surface.nodata_value)
    band.WriteArray(surface.elevation)
    band.FlushCache()
    ds = None


__all__ = [
    "read_dem",
    "read_metadata",
    "write_dem",
    "active_backend",
    "AOI",
    "DemLoaderError",
    "GeoBackendUnavailableError",
    "DemReadError",
    "AoiError",
]
