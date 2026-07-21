"""
runtime/environment_builder.py
══════════════════════════════════════════════════════════════════════════
One-shot environment construction for PhysWorldLM's Omniverse Runtime.

Pipeline position
------------------
    scene.usda (already exported by Scene Compiler)
            │
            ▼
      OmniverseRuntime.load_stage()
            │
            ▼
    ┌────────────────────┐
    │ ENVIRONMENT BUILDER │   <-- this module
    └────────────────────┘
            │
            ▼
      OmniverseRuntime.initialize_physics() / discover_entities()

Scope
-----
This module is a *one-shot builder*, not a `RuntimeSubsystem`. It runs
once, during `OmniverseRuntime.initialize()` (after `load_stage()` and
before `discover_entities()`), and procedurally authors the immersive,
non-entity backdrop of the world onto the already-loaded stage: sky,
sun, terrain, water, roads, forests, fog, clouds, and a military
airbase/runway. It does not touch per-entity prims (aircraft, vehicles,
sensors, ...) -- that is `entity_spawner.EntitySpawner`'s job -- and it
does not run per-frame -- weather *animation* belongs to
`animation_system.AnimationSystem`.

This builder is USD/Omniverse-API-based where `pxr` is available, and
falls back to structured no-op logging (recording intent without
touching the stage) when `pxr` is not installed, mirroring the fallback
pattern already used in `scene_compiler.py` and `omniverse_runtime.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from world_spec import Environment

logger = logging.getLogger("physworldlm.environment_builder")
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

class EnvironmentBuildError(Exception):
    """Raised when environment construction fails."""


# ════════════════════════════════════════════════════════════════════════
# Enums / config
# ════════════════════════════════════════════════════════════════════════

class TerrainType(Enum):
    DESERT = "desert"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    COASTAL = "coastal"
    GRASSLAND = "grassland"
    SNOW = "snow"
    FLAT = "flat"

    @classmethod
    def from_string(cls, value: str) -> "TerrainType":
        try:
            return cls(value.strip().lower())
        except ValueError:
            logger.warning("Unrecognized terrain_type '%s'; defaulting to FLAT.", value)
            return cls.FLAT


_TERRAIN_GROUND_MATERIAL = {
    TerrainType.DESERT: "sand",
    TerrainType.FOREST: "forest_floor",
    TerrainType.MOUNTAIN: "rock",
    TerrainType.URBAN: "asphalt",
    TerrainType.COASTAL: "wet_sand",
    TerrainType.GRASSLAND: "grass",
    TerrainType.SNOW: "snow",
    TerrainType.FLAT: "generic_ground",
}

# Terrain types that, by default, include mountains/hills.
_TERRAINS_WITH_RELIEF = frozenset({TerrainType.MOUNTAIN, TerrainType.COASTAL, TerrainType.FOREST})

# Terrain types that include a body of water by default.
_TERRAINS_WITH_WATER = frozenset({TerrainType.COASTAL})


@dataclass
class EnvironmentBuildConfig:
    """User-configurable settings controlling environment construction.

    Attributes:
        terrain_size_m: Side length of the procedurally generated ground
            plane / terrain mesh, in meters.
        build_airbase: Whether to author a military airbase + runway.
        build_roads: Whether to author a procedural road network.
        build_forest: Whether to author forest/tree instances.
        max_trees: Upper bound on the number of tree instances spawned
            when `build_forest` is True.
        runway_length_m: Length of the authored runway, in meters.
        runway_width_m: Width of the authored runway, in meters.
    """

    terrain_size_m: float = 20000.0
    build_airbase: bool = True
    build_roads: bool = True
    build_forest: bool = True
    max_trees: int = 500
    runway_length_m: float = 3000.0
    runway_width_m: float = 45.0


@dataclass
class EnvironmentBuildReport:
    """Summary of what was authored onto the stage."""

    terrain_type: TerrainType = TerrainType.FLAT
    prims_created: list[str] = field(default_factory=list)
    tree_count: int = 0
    road_segment_count: int = 0
    has_water: bool = False
    has_airbase: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "terrain_type": self.terrain_type.value,
            "prims_created": list(self.prims_created),
            "tree_count": self.tree_count,
            "road_segment_count": self.road_segment_count,
            "has_water": self.has_water,
            "has_airbase": self.has_airbase,
            "warnings": list(self.warnings),
        }


# ════════════════════════════════════════════════════════════════════════
# EnvironmentBuilder
# ════════════════════════════════════════════════════════════════════════

class EnvironmentBuilder:
    """Procedurally authors the environment backdrop of a PhysWorldLM scene.

    Example:
        >>> builder = EnvironmentBuilder(stage, world_spec.environment)
        >>> report = builder.build()
        >>> report.has_water
        True
    """

    def __init__(
        self,
        stage: Any,
        environment: Environment,
        config: Optional[EnvironmentBuildConfig] = None,
    ) -> None:
        """Initialize the builder.

        Args:
            stage: An open USD stage handle, as produced by
                `OmniverseRuntime._open_stage()` (either a real
                `pxr.Usd.Stage` or the `_FallbackStage` stand-in).
            environment: The `Environment` block of the compiled
                `WorldSpec`, carrying terrain type, weather, and wind.
            config: Build-time tuning parameters. Defaults to
                `EnvironmentBuildConfig()`.
        """
        self._stage = stage
        self._env = environment
        self._config = config or EnvironmentBuildConfig()
        self._terrain_type = TerrainType.from_string(environment.terrain_type)
        self._report = EnvironmentBuildReport(terrain_type=self._terrain_type)
        self._pxr_available = self._detect_pxr()

    @staticmethod
    def _detect_pxr() -> bool:
        try:
            import pxr  # noqa: F401

            return True
        except ImportError:
            return False

    # ── orchestration ────────────────────────────────────────────────

    def build(self) -> EnvironmentBuildReport:
        """Run the full environment construction sequence.

        Order: sky → sun → terrain → mountains → water → roads → forest
        → airbase → weather. Each step is independent and failures in
        one step are recorded as warnings rather than aborting the
        remaining steps, since a partially-built environment is still
        useful for visualization/debugging.

        Returns:
            An `EnvironmentBuildReport` summarizing what was authored.
        """
        logger.info("Building environment (terrain_type=%s)", self._terrain_type.value)

        for step in (
            self.create_sky,
            self.create_sun,
            self.create_terrain,
            self.create_mountains,
            self.create_water,
            self.create_roads,
            self.create_forest,
            self.create_airbase,
        ):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - best-effort per-step
                msg = f"{step.__name__} failed: {exc}"
                logger.warning(msg)
                self._report.warnings.append(msg)

        self.apply_weather()

        logger.info(
            "Environment build complete: %d prim(s), %d tree(s), water=%s, airbase=%s",
            len(self._report.prims_created),
            self._report.tree_count,
            self._report.has_water,
            self._report.has_airbase,
        )
        return self._report

    # ── individual steps ─────────────────────────────────────────────

    def create_sky(self) -> None:
        """Author a sky dome (HDRI dome light) and register it on the stage."""
        path = "/World/Environment/Sky/DomeLight"
        self._define_prim(path, "DomeLight", attrs={"intensity": 1000.0, "exposure": 0.0})
        self._report.prims_created.append(path)
        logger.info("Sky dome authored at %s", path)

    def create_sun(self) -> None:
        """Author the primary directional light (sun) consistent with `time_of_day`."""
        path = "/World/Environment/Sky/Sun"
        elevation_deg = {"day": 55.0, "dawn": 8.0, "dusk": 5.0, "night": -20.0}.get(
            self._env.time_of_day, 45.0
        )
        self._define_prim(
            path,
            "DistantLight",
            attrs={"intensity": 3000.0, "angle": 0.53, "elevation_deg": elevation_deg},
        )
        self._report.prims_created.append(path)
        logger.info("Sun authored at %s (elevation=%.1f deg)", path, elevation_deg)

    def create_terrain(self) -> None:
        """Author the ground/terrain mesh and assign its base material."""
        path = "/World/Environment/Terrain/Ground"
        material = _TERRAIN_GROUND_MATERIAL.get(self._terrain_type, "generic_ground")
        self._define_prim(
            path,
            "Mesh",
            attrs={
                "size_m": self._config.terrain_size_m,
                "terrain_type": self._terrain_type.value,
                "material": material,
                "friction": self._env.friction_global,
            },
        )
        self._report.prims_created.append(path)
        logger.info(
            "Terrain authored at %s (type=%s, material=%s, size=%.0fm)",
            path,
            self._terrain_type.value,
            material,
            self._config.terrain_size_m,
        )

    def create_mountains(self) -> None:
        """Author relief geometry (mountains/hills) for terrain types with relief."""
        if self._terrain_type not in _TERRAINS_WITH_RELIEF:
            logger.info("Terrain type '%s' has no default relief; skipping.", self._terrain_type.value)
            return
        path = "/World/Environment/Terrain/Relief"
        self._define_prim(path, "Mesh", attrs={"relief_style": self._terrain_type.value})
        self._report.prims_created.append(path)
        logger.info("Relief geometry authored at %s", path)

    def create_water(self) -> None:
        """Author a water body (lake/coastline) where the terrain type calls for one."""
        if self._terrain_type not in _TERRAINS_WITH_WATER:
            logger.info("Terrain type '%s' has no default water body; skipping.", self._terrain_type.value)
            return
        path = "/World/Environment/Terrain/Water"
        self._define_prim(path, "Mesh", attrs={"material": "water", "wave_amplitude_m": 0.3})
        self._report.prims_created.append(path)
        self._report.has_water = True
        logger.info("Water body authored at %s", path)

    def create_roads(self) -> None:
        """Author a simple procedural road network connecting key points."""
        if not self._config.build_roads:
            logger.info("Road generation disabled by configuration.")
            return
        if self._terrain_type is TerrainType.URBAN:
            segment_count = 12
        elif self._terrain_type in (TerrainType.GRASSLAND, TerrainType.COASTAL):
            segment_count = 4
        else:
            segment_count = 1  # service road to the airbase, at minimum

        group_path = "/World/Environment/Roads"
        self._define_prim(group_path, "Scope")
        for i in range(segment_count):
            seg_path = f"{group_path}/Segment_{i:03d}"
            self._define_prim(seg_path, "Mesh", attrs={"material": "asphalt"})
            self._report.prims_created.append(seg_path)
        self._report.road_segment_count = segment_count
        logger.info("Authored %d road segment(s) under %s", segment_count, group_path)

    def create_forest(self) -> None:
        """Author forest/tree instances, density-scaled by terrain type."""
        if not self._config.build_forest:
            logger.info("Forest generation disabled by configuration.")
            return
        density_fraction = {
            TerrainType.FOREST: 1.0,
            TerrainType.MOUNTAIN: 0.4,
            TerrainType.GRASSLAND: 0.15,
            TerrainType.COASTAL: 0.2,
        }.get(self._terrain_type, 0.0)

        tree_count = int(self._config.max_trees * density_fraction)
        if tree_count <= 0:
            logger.info("Terrain type '%s' yields zero trees; skipping forest authoring.", self._terrain_type.value)
            return

        group_path = "/World/Environment/Forest"
        self._define_prim(group_path, "PointInstancer", attrs={"instance_count": tree_count})
        self._report.prims_created.append(group_path)
        self._report.tree_count = tree_count
        logger.info("Authored forest instancer at %s (%d trees)", group_path, tree_count)

    def create_airbase(self) -> None:
        """Author a military airbase: runway, taxiways, and landing zone."""
        if not self._config.build_airbase:
            logger.info("Airbase generation disabled by configuration.")
            return

        base_path = "/World/Environment/Airbase"
        self._define_prim(base_path, "Scope")

        runway_path = f"{base_path}/Runway"
        self._define_prim(
            runway_path,
            "Mesh",
            attrs={
                "length_m": self._config.runway_length_m,
                "width_m": self._config.runway_width_m,
                "material": "runway_asphalt",
            },
        )

        lz_path = f"{base_path}/LandingZone"
        self._define_prim(lz_path, "Mesh", attrs={"material": "concrete"})

        self._report.prims_created.extend([base_path, runway_path, lz_path])
        self._report.has_airbase = True
        logger.info(
            "Airbase authored at %s (runway=%.0fm x %.0fm)",
            base_path,
            self._config.runway_length_m,
            self._config.runway_width_m,
        )

    def apply_weather(self) -> None:
        """Apply fog/cloud/wind authoring consistent with `Environment.weather`."""
        weather = self._env.weather
        atmosphere_path = "/World/Environment/Atmosphere"
        attrs: dict[str, Any] = {
            "weather": weather,
            "wind_speed_ms": self._env.wind.speed,
            "wind_direction_rad": self._env.wind.direction,
        }
        if weather == "fog":
            attrs["fog_density"] = 0.6
        elif weather == "rain":
            attrs["precipitation"] = "rain"
        elif weather == "snow":
            attrs["precipitation"] = "snow"

        attrs["cloud_coverage"] = 0.3 if weather == "clear" else 0.8
        self._define_prim(atmosphere_path, "Scope", attrs=attrs)
        self._report.prims_created.append(atmosphere_path)
        logger.info("Weather/atmosphere applied at %s (weather=%s)", atmosphere_path, weather)

    # ── internal helpers ─────────────────────────────────────────────

    def _define_prim(self, path: str, prim_type: str, attrs: Optional[dict[str, Any]] = None) -> None:
        """Define (or update) a prim at `path` on the stage.

        Uses `pxr.Usd.Stage.DefinePrim` + `CreateAttribute` when `pxr` is
        available. Otherwise records the intended authoring as a
        structured log entry, so this builder remains exercisable in
        environments without the OpenUSD bindings installed (consistent
        with the fallback pattern in `scene_compiler.USDAsciiExporter`
        and `omniverse_runtime._FallbackStage`).
        """
        attrs = attrs or {}
        if self._pxr_available and hasattr(self._stage, "DefinePrim"):
            try:
                from pxr import Sdf  # type: ignore

                prim = self._stage.DefinePrim(path, prim_type)
                for key, value in attrs.items():
                    try:
                        prim.CreateAttribute(f"physworldlm:{key}", Sdf.ValueTypeNames.String).Set(str(value))
                    except Exception:  # noqa: BLE001 - attribute authoring is best-effort
                        pass
                return
            except Exception as exc:  # noqa: BLE001
                raise EnvironmentBuildError(f"Failed to define prim '{path}': {exc}") from exc

        logger.debug("[fallback] would define %s prim at %s with attrs=%s", prim_type, path, attrs)
