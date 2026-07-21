"""
gps.py
══════════════════════════════════════════════════════════════════════════
GPS/GNSS sensor: emits geodetic position (lat/lon/alt), horizontal
accuracy, ground velocity, and heading. Coordinate conversion to
ENU/NED/ECEF is delegated to `sensor_types.CoordinateTransformer` given
a configured local-tangent-plane origin.

Extended with:
  - Multi-constellation selection (GPS, GLONASS, Galileo, BeiDou, NavIC)
  - Satellite count (visible/used) and DOP (HDOP/VDOP/PDOP)
  - Fix-quality tiers (Standard, DGPS, SBAS, RTK Float, RTK Fixed)
  - Multi-frequency tracking (L1/L2/L5) as an accuracy modifier
  - GPS time / UTC time / leap seconds

All new supporting types (GNSSConstellation, GNSSFrequencyBand,
FixQuality, GPSReading) are defined in this file rather than in
sensor_types.py, so the whole GPS feature set lives in one place. If you
later want GPSReading alongside your other *_types in sensor_types.py,
it can be moved verbatim -- nothing here depends on it staying local.

DOP and satellite-count modeling here is a deliberately simple,
tunable heuristic (count- and dropout-driven), not a geometrically
accurate constellation simulation -- see `_estimate_dop()` for the
reasoning. If you later want real satellite geometry (elevation/azimuth
per SV, actual DOP from the geometry matrix), that's a bigger, separate
piece of work and shouldn't block using these fields as degradation
knobs today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from base_sensor import Sensor
from sensor_types import CoordinateFrame, CoordinateTransformer, SensorType, Vec3

__all__ = ["Gps", "GNSSConstellation", "GNSSFrequencyBand", "FixQuality", "GPSReading"]

# GPS epoch: 1980-01-06T00:00:00 UTC, as a Unix timestamp.
_GPS_EPOCH_UNIX_S = 315964800.0
# GPS-UTC leap second offset as of the last announced leap second
# (2017-01-01). This is an IERS announcement, not something computable
# from a formula -- update this constant (or pass leap_seconds=) if a
# new leap second is inserted.
_DEFAULT_LEAP_SECONDS = 18.0

# Nominal visible-satellite contribution per constellation, used as a
# deterministic baseline in the absence of real orbit propagation.
# NavIC is a regional (India-centric) constellation, hence the low but
# nonzero nominal count -- it contributes meaningfully over India and
# little/nothing elsewhere, which a real implementation would gate on
# receiver position; kept as a flat nominal count here for simplicity.
_NOMINAL_VISIBLE_SATELLITES: dict["GNSSConstellation", int] = {}


class GNSSConstellation(Enum):
    GPS = "gps"
    GLONASS = "glonass"
    GALILEO = "galileo"
    BEIDOU = "beidou"
    NAVIC = "navic"


_NOMINAL_VISIBLE_SATELLITES.update(
    {
        GNSSConstellation.GPS: 8,
        GNSSConstellation.GLONASS: 6,
        GNSSConstellation.GALILEO: 6,
        GNSSConstellation.BEIDOU: 7,
        GNSSConstellation.NAVIC: 4,
    }
)


class GNSSFrequencyBand(Enum):
    L1 = "L1"  # ~1575.42 MHz -- all constellations broadcast an L1-class signal
    L2 = "L2"  # ~1227.60 MHz
    L5 = "L5"  # ~1176.45 MHz -- newer, higher-precision civilian signal


class FixQuality(Enum):
    NO_FIX = "no_fix"
    STANDARD = "standard"    # autonomous single-frequency fix
    DGPS = "dgps"             # differential correction
    SBAS = "sbas"              # satellite-based augmentation (WAAS/EGNOS/GAGAN)
    RTK_FLOAT = "rtk_float"
    RTK_FIXED = "rtk_fixed"

    @property
    def accuracy_floor_m(self) -> float:
        """Best-case horizontal accuracy this fix quality can realistically
        deliver. Used to clamp the reported `accuracy_m` so a config can't
        claim RTK-Fixed-grade precision while sitting at standard-GPS
        noise levels -- the fix quality should drive the achievable
        accuracy, not sit beside it as an independent, possibly
        contradictory field."""
        return {
            FixQuality.NO_FIX: float("inf"),
            FixQuality.STANDARD: 2.5,
            FixQuality.DGPS: 1.0,
            FixQuality.SBAS: 0.5,
            FixQuality.RTK_FLOAT: 0.1,
            FixQuality.RTK_FIXED: 0.02,
        }[self]

    @property
    def min_satellites_required(self) -> int:
        """Minimum used-satellite count below which this fix quality
        cannot be sustained (RTK needs more satellites/geometry than a
        standard fix to resolve carrier-phase ambiguities)."""
        return {
            FixQuality.NO_FIX: 0,
            FixQuality.STANDARD: 4,
            FixQuality.DGPS: 4,
            FixQuality.SBAS: 5,
            FixQuality.RTK_FLOAT: 5,
            FixQuality.RTK_FIXED: 6,
        }[self]


@dataclass
class GPSReading:
    """GNSS reading. Defined locally in gps.py (see module docstring) --
    move into sensor_types.py alongside your other *Reading types if you
    want a single shared location later."""

    sensor_id: str
    timestamp: float

    latitude_deg: float
    longitude_deg: float
    altitude_m: float

    velocity_ms: Vec3
    heading_rad: float

    # --- fix quality & accuracy ---
    fix_quality: FixQuality
    has_fix: bool
    accuracy_m: float
    velocity_accuracy_ms: float
    heading_accuracy_deg: float

    # --- satellites & DOP ---
    constellations: tuple[GNSSConstellation, ...]
    visible_satellites: int
    used_satellites: int
    hdop: float
    vdop: float
    pdop: float

    # --- signal ---
    tracked_frequencies: tuple[GNSSFrequencyBand, ...]

    # --- time ---
    gps_time_s: float
    utc_time_s: float
    leap_seconds: float

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "velocity_ms": self.velocity_ms.to_dict(),
            "heading_rad": self.heading_rad,
            "fix_quality": self.fix_quality.value,
            "has_fix": self.has_fix,
            "accuracy_m": self.accuracy_m,
            "velocity_accuracy_ms": self.velocity_accuracy_ms,
            "heading_accuracy_deg": self.heading_accuracy_deg,
            "constellations": [c.value for c in self.constellations],
            "visible_satellites": self.visible_satellites,
            "used_satellites": self.used_satellites,
            "hdop": self.hdop,
            "vdop": self.vdop,
            "pdop": self.pdop,
            "tracked_frequencies": [b.value for b in self.tracked_frequencies],
            "gps_time_s": self.gps_time_s,
            "utc_time_s": self.utc_time_s,
            "leap_seconds": self.leap_seconds,
        }


class Gps(Sensor):
    """GPS/GNSS receiver."""

    sensor_type = SensorType.GPS

    def __init__(
        self,
        name: str,
        *,
        accuracy_m: float = 2.5,
        velocity_accuracy_ms: float = 0.1,
        heading_accuracy_deg: float = 1.0,
        coordinate_frame: CoordinateFrame = CoordinateFrame.ENU,
        datum: str = "WGS84",
        origin_lat_deg: float = 0.0,
        origin_lon_deg: float = 0.0,
        origin_alt_m: float = 0.0,
        # --- GNSS constellation / signal / fix config ---------------------
        constellations: Optional[set[GNSSConstellation]] = None,
        tracked_frequencies: Optional[set[GNSSFrequencyBand]] = None,
        fix_quality: FixQuality = FixQuality.STANDARD,
        satellite_dropout: int = 0,
        leap_seconds: float = _DEFAULT_LEAP_SECONDS,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("update_rate_hz", 10.0)
        super().__init__(name, **kwargs)
        self.accuracy_m = accuracy_m
        self.velocity_accuracy_ms = velocity_accuracy_ms
        self.heading_accuracy_deg = heading_accuracy_deg
        self.coordinate_frame = coordinate_frame
        self.datum = datum
        self.origin_lat_deg = origin_lat_deg
        self.origin_lon_deg = origin_lon_deg
        self.origin_alt_m = origin_alt_m

        self.constellations = constellations or {GNSSConstellation.GPS}
        self.tracked_frequencies = tracked_frequencies or {GNSSFrequencyBand.L1}
        self.fix_quality = fix_quality
        # Config knob for simulating partial sky visibility (urban canyon,
        # canopy, jamming) without modeling real satellite geometry.
        self.satellite_dropout = satellite_dropout
        self.leap_seconds = leap_seconds

        # State expected to be supplied by the caller/backend via
        # `set_true_position()`, so `capture()` has something physically
        # meaningful to perturb -- the framework does not simulate GNSS
        # constellations or propagation itself.
        self._true_lat_deg: float = origin_lat_deg
        self._true_lon_deg: float = origin_lon_deg
        self._true_alt_m: float = origin_alt_m
        self._true_velocity: Vec3 = Vec3()
        self._true_heading_rad: float = 0.0

    def set_true_position(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        velocity: Optional[Vec3] = None,
        heading_rad: Optional[float] = None,
    ) -> None:
        """Feed ground-truth position/velocity/heading in (typically from
        the owning entity's physics state each tick); `capture()` then
        applies this sensor's noise/accuracy model on top."""
        with self._lock:
            self._true_lat_deg = lat_deg
            self._true_lon_deg = lon_deg
            self._true_alt_m = alt_m
            if velocity is not None:
                self._true_velocity = velocity
            if heading_rad is not None:
                self._true_heading_rad = heading_rad

    def enu_position(self) -> Vec3:
        ecef = CoordinateTransformer.geodetic_to_ecef(self._true_lat_deg, self._true_lon_deg, self._true_alt_m)
        return CoordinateTransformer.ecef_to_enu(ecef, self.origin_lat_deg, self.origin_lon_deg, self.origin_alt_m)

    # ------------------------------------------------------------------ #
    # Satellites / DOP
    # ------------------------------------------------------------------ #
    def _visible_satellite_count(self) -> int:
        return sum(_NOMINAL_VISIBLE_SATELLITES[c] for c in self.constellations)

    def _used_satellite_count(self, visible: int) -> int:
        return max(0, visible - self.satellite_dropout)

    @staticmethod
    def _estimate_dop(used_satellites: int) -> tuple[float, float, float]:
        """Rough DOP estimate as a function of used-satellite count.

        Real DOP depends on satellite geometry (elevation/azimuth spread),
        not just count -- this is a simplified count-based heuristic meant
        as a tunable degradation knob (e.g. urban-canyon scenarios via
        `satellite_dropout`), not a geometrically accurate GNSS
        simulation. PDOP asymptotes toward ~1.0 with many satellites and
        grows sharply below 4, the minimum for a 3D fix.
        """
        if used_satellites < 4:
            return float("inf"), float("inf"), float("inf")
        pdop = 1.0 + 6.0 / used_satellites
        hdop = pdop * 0.6
        vdop = pdop * 0.8
        return hdop, vdop, pdop

    def _frequency_accuracy_multiplier(self) -> float:
        """More tracked frequency bands allow direct ionospheric-delay
        cancellation (dual/triple-frequency receivers estimate and remove
        the ionospheric term from the signals themselves instead of
        relying on a broadcast correction model), improving accuracy.
        Modeled as a simple multiplier rather than simulating per-band
        delay, consistent with the DOP heuristic above."""
        band_count = len(self.tracked_frequencies)
        if band_count >= 3:
            return 0.5
        if band_count == 2:
            return 0.7
        return 1.0

    def _resolve_fix(self) -> tuple[FixQuality, bool, int, int, float, float, float, float]:
        """Combine constellation/dropout/fix-quality config into the set
        of values that actually go on the wire: effective fix quality
        (demoted if satellite count can't support it), has_fix,
        satellite counts, DOP, and effective accuracy_m.
        """
        visible = self._visible_satellite_count()
        used = self._used_satellite_count(visible)
        hdop, vdop, pdop = self._estimate_dop(used)

        effective_fix = self.fix_quality
        if used < effective_fix.min_satellites_required:
            # Demote step-by-step rather than jumping straight to NO_FIX,
            # so a marginal satellite count degrades gracefully (e.g. RTK
            # Fixed -> RTK Float) instead of dropping out entirely.
            demotion_order = [
                FixQuality.RTK_FIXED,
                FixQuality.RTK_FLOAT,
                FixQuality.SBAS,
                FixQuality.DGPS,
                FixQuality.STANDARD,
                FixQuality.NO_FIX,
            ]
            start = demotion_order.index(effective_fix)
            for candidate in demotion_order[start:]:
                if used >= candidate.min_satellites_required:
                    effective_fix = candidate
                    break
            else:
                effective_fix = FixQuality.NO_FIX

        has_fix = effective_fix != FixQuality.NO_FIX and used >= 4

        if not has_fix:
            accuracy_m = float("inf")
        else:
            freq_multiplier = self._frequency_accuracy_multiplier()
            # hdop of 1.0 is the nominal baseline the configured
            # accuracy_m is assumed to represent; scale up/down from there.
            dop_scale = hdop if hdop != float("inf") else 1.0
            accuracy_m = max(self.accuracy_m * freq_multiplier * dop_scale, effective_fix.accuracy_floor_m)

        return effective_fix, has_fix, visible, used, hdop, vdop, pdop, accuracy_m

    # ------------------------------------------------------------------ #
    # Time
    # ------------------------------------------------------------------ #
    def _gnss_time(self, sim_timestamp: float) -> tuple[float, float]:
        """Derive GPS time (seconds since GPS epoch, leap-second-free) and
        UTC time from the simulation timestamp, which is treated as UTC
        Unix time -- the same assumption used for `Imu`'s software
        timestamp, so the two sensors stay time-consistent."""
        utc_unix_s = sim_timestamp
        gps_time_s = (utc_unix_s - _GPS_EPOCH_UNIX_S) + self.leap_seconds
        return gps_time_s, utc_unix_s

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    def _build_sample(self, raw_data: Any) -> GPSReading:
        lat, lon, alt = self._true_lat_deg, self._true_lon_deg, self._true_alt_m
        if self.noise_model is not None:
            # Perturb the ECEF representation isotropically, then convert
            # back, so noise doesn't need separately-scaled lat/lon/alt
            # handling.
            ecef = CoordinateTransformer.geodetic_to_ecef(lat, lon, alt)
            noisy_ecef = ecef.__class__(
                self.noise_model.apply(ecef.x),
                self.noise_model.apply(ecef.y),
                self.noise_model.apply(ecef.z),
            )
            lat, lon, alt = CoordinateTransformer.ecef_to_geodetic(noisy_ecef)

        effective_fix, has_fix, visible, used, hdop, vdop, pdop, accuracy_m = self._resolve_fix()

        data_kwargs = self._next_data_kwargs()
        sim_timestamp = data_kwargs.get("timestamp", 0.0)
        gps_time_s, utc_time_s = self._gnss_time(sim_timestamp)

        return GPSReading(
            **data_kwargs,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt,
            velocity_ms=self._true_velocity,
            heading_rad=self._true_heading_rad,
            fix_quality=effective_fix,
            has_fix=has_fix,
            accuracy_m=accuracy_m,
            velocity_accuracy_ms=self.velocity_accuracy_ms,
            heading_accuracy_deg=self.heading_accuracy_deg,
            constellations=tuple(sorted(self.constellations, key=lambda c: c.value)),
            visible_satellites=visible,
            used_satellites=used,
            hdop=hdop,
            vdop=vdop,
            pdop=pdop,
            tracked_frequencies=tuple(sorted(self.tracked_frequencies, key=lambda b: b.value)),
            gps_time_s=gps_time_s,
            utc_time_s=utc_time_s,
            leap_seconds=self.leap_seconds,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.accuracy_m < 0:
            problems.append(f"accuracy_m must be >= 0 (got {self.accuracy_m}).")
        if not (-90.0 <= self.origin_lat_deg <= 90.0):
            problems.append(f"origin_lat_deg must be in [-90, 90] (got {self.origin_lat_deg}).")
        if not (-180.0 <= self.origin_lon_deg <= 180.0):
            problems.append(f"origin_lon_deg must be in [-180, 180] (got {self.origin_lon_deg}).")
        if not self.constellations:
            problems.append("constellations must contain at least one GNSSConstellation.")
        if not self.tracked_frequencies:
            problems.append("tracked_frequencies must contain at least one GNSSFrequencyBand.")
        if self.satellite_dropout < 0:
            problems.append(f"satellite_dropout must be >= 0 (got {self.satellite_dropout}).")
        min_required = self.fix_quality.min_satellites_required
        if self._visible_satellite_count() < min_required:
            problems.append(
                f"configured constellations provide only {self._visible_satellite_count()} nominal "
                f"visible satellites, below the {min_required} required for fix_quality="
                f"{self.fix_quality.value} even with zero dropout."
            )
        return problems

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def _serialize_specific(self) -> dict:
        return {
            "accuracy_m": self.accuracy_m,
            "velocity_accuracy_ms": self.velocity_accuracy_ms,
            "heading_accuracy_deg": self.heading_accuracy_deg,
            "coordinate_frame": self.coordinate_frame.value,
            "datum": self.datum,
            "origin_lat_deg": self.origin_lat_deg,
            "origin_lon_deg": self.origin_lon_deg,
            "origin_alt_m": self.origin_alt_m,
            "constellations": [c.value for c in self.constellations],
            "tracked_frequencies": [b.value for b in self.tracked_frequencies],
            "fix_quality": self.fix_quality.value,
            "satellite_dropout": self.satellite_dropout,
            "leap_seconds": self.leap_seconds,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Gps":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.accuracy_m = data.get("accuracy_m", 2.5)
        instance.velocity_accuracy_ms = data.get("velocity_accuracy_ms", 0.1)
        instance.heading_accuracy_deg = data.get("heading_accuracy_deg", 1.0)
        instance.coordinate_frame = CoordinateFrame(data.get("coordinate_frame", CoordinateFrame.ENU.value))
        instance.datum = data.get("datum", "WGS84")
        instance.origin_lat_deg = data.get("origin_lat_deg", 0.0)
        instance.origin_lon_deg = data.get("origin_lon_deg", 0.0)
        instance.origin_alt_m = data.get("origin_alt_m", 0.0)
        instance.constellations = {
            GNSSConstellation(v) for v in data.get("constellations", [GNSSConstellation.GPS.value])
        }
        instance.tracked_frequencies = {
            GNSSFrequencyBand(v) for v in data.get("tracked_frequencies", [GNSSFrequencyBand.L1.value])
        }
        instance.fix_quality = FixQuality(data.get("fix_quality", FixQuality.STANDARD.value))
        instance.satellite_dropout = data.get("satellite_dropout", 0)
        instance.leap_seconds = data.get("leap_seconds", _DEFAULT_LEAP_SECONDS)
        return instance
