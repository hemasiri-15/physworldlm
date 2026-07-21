"""
config.py
══════════════════════════════════════════════════════════════════════════
Central configuration module for the Omniverse connector layer of
PhysWorldLM.

Pipeline position
------------------
    Natural Language → Ontology → WorldSpec → Scene Compiler → scene.usda
                                                                      │
                                                                      ▼
                                                        ┌───────────────────┐
                                                        │  omniverse/config │  <-- this module
                                                        └───────────────────┘
                                                                      │
                                                                      ▼
                                                   app_launcher.py / stage_manager.py / ...

Scope
-----
This module owns the single source of truth for how every other
``omniverse/*`` component locates and talks to an NVIDIA Omniverse Kit /
Isaac Sim installation: paths, rendering flags, physics timestep, and
feature toggles. It performs *detection*, not *invocation* -- it never
launches Kit, opens a stage, or imports ``omni``/``pxr``. Those concerns
belong to ``app_launcher.py`` and ``stage_manager.py``.

Design constraints
-------------------
    * Only the Python standard library may be imported at module load
      time. Omniverse/Isaac Sim packages are never imported here.
    * No installation path is ever hardcoded. Detection is environment-
      and filesystem-driven, with explicit override via env vars or a
      YAML config file.
    * ``OmniverseConfig`` is a plain, picklable dataclass so it can be
      constructed in a detection-free unit test or passed across a
      future process/cloud boundary without touching disk.

Public API
----------
    config = OmniverseConfig.default()          # auto-detected
    config = OmniverseConfig.load_from_env()     # env-var driven
    config = OmniverseConfig.load_from_yaml(p)   # file driven
    config.validate()
    config.save(path)

Changelog
---------
    * Fixed: ``to_dict()`` did not stringify ``Path`` entries inside the
      ``asset_search_paths`` list, so ``save()`` raised ``TypeError`` on
      any config with a non-empty search path list.
    * Fixed: ``detect_kit_root()`` previously only confirmed the parent
      ``.../ov/pkg`` package directory existed, never descended into the
      versioned app subdirectory (e.g. ``create-2023.2.0``), so
      ``validate(require_kit=True)`` could never actually fail a
      below-minimum Kit version check -- it was silently a no-op.
    * Fixed: ``%USERNAME%`` / ``%LOCALAPPDATA%``-style Windows env-var
      syntax was left untouched by ``os.path.expandvars`` on Linux/WSL
      (which only understands ``$VAR``), so the WSL candidate path
      that reaches into the Windows side of the filesystem never
      resolved. ``_expand()`` now substitutes ``%VAR%`` tokens manually
      before falling back to POSIX-style expansion.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.config")
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

class OmniverseConfigError(Exception):
    """Base class for all configuration errors."""


class ConfigValidationError(OmniverseConfigError):
    """Raised when ``OmniverseConfig.validate()`` finds an invalid state."""


class ConfigLoadError(OmniverseConfigError):
    """Raised when a config cannot be loaded from env vars or a file."""


class ConfigSaveError(OmniverseConfigError):
    """Raised when a config cannot be persisted to disk."""


# ════════════════════════════════════════════════════════════════════════
# Environment variable names
# ════════════════════════════════════════════════════════════════════════
#
# Centralized here so app_launcher.py / stage_manager.py never need to
# know raw env var names -- they only ever see the resolved dataclass.

class EnvVar:
    KIT_ROOT = "PWLM_OMNIVERSE_KIT_ROOT"
    ISAAC_ROOT = "PWLM_ISAAC_ROOT"
    ASSET_ROOT = "PWLM_ASSET_ROOT"
    USD_OUTPUT_DIR = "PWLM_USD_OUTPUT_DIR"
    CACHE_DIR = "PWLM_CACHE_DIR"
    EXTENSION_DIR = "PWLM_EXTENSION_DIR"
    NUCLEUS_SERVER = "PWLM_NUCLEUS_SERVER"
    PHYSICS_DT = "PWLM_PHYSICS_DT"
    RENDER_DT = "PWLM_RENDER_DT"
    HEADLESS = "PWLM_HEADLESS"
    RENDERER = "PWLM_RENDERER"
    GPU_DEVICE = "PWLM_GPU_DEVICE"
    ENABLE_RTX = "PWLM_ENABLE_RTX"
    ENABLE_DLSS = "PWLM_ENABLE_DLSS"
    ENABLE_LIVESTREAM = "PWLM_ENABLE_LIVESTREAM"
    ENABLE_PHYSICS = "PWLM_ENABLE_PHYSICS"
    ENABLE_REPLICATOR = "PWLM_ENABLE_REPLICATOR"
    ENABLE_ROS2 = "PWLM_ENABLE_ROS2"
    CONFIG_FILE = "PWLM_OMNIVERSE_CONFIG"

    SIMULATION_BACKEND = "PWLM_SIMULATION_BACKEND"
    USD_FORMAT = "PWLM_USD_FORMAT"
    PHYSICS_ENGINE = "PWLM_PHYSICS_ENGINE"
    PLANNER = "PWLM_PLANNER"
    TERRAIN_FILE = "PWLM_TERRAIN_FILE"
    TERRAIN_CACHE = "PWLM_TERRAIN_CACHE"
    TERRAIN_RESOLUTION = "PWLM_TERRAIN_RESOLUTION"
    TERRAIN_SCALE = "PWLM_TERRAIN_SCALE"


# Known install locations to probe, by platform, in priority order.
# These are *candidates* to check for existence -- never assumed present.
# Note: these point at the *package root* (e.g. ``.../ov/pkg``), which
# typically contains one versioned subdirectory per installed app
# (``create-2023.2.0``, ``kit-106.0.0``, ...). Version-aware resolution
# of the actual app directory happens in ``_select_versioned_subdir()``.
_KIT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Windows": (
        "%LOCALAPPDATA%/ov/pkg",
        "C:/Users/%USERNAME%/AppData/Local/ov/pkg",
        "C:/NVIDIA/Omniverse",
    ),
    "Linux": (
        "~/.local/share/ov/pkg",
        "/opt/nvidia/omniverse",
        "/opt/ov",
    ),
    "WSL": (
        "~/.local/share/ov/pkg",
        "/opt/nvidia/omniverse",
        "/mnt/c/Users/%USERNAME%/AppData/Local/ov/pkg",
    ),
}

_ISAAC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Windows": (
        "%LOCALAPPDATA%/ov/pkg/isaac-sim-*",
        "C:/isaac-sim",
    ),
    "Linux": (
        "~/.local/share/ov/pkg/isaac-sim-*",
        "/opt/isaac-sim",
    ),
    "WSL": (
        "~/.local/share/ov/pkg/isaac-sim-*",
        "/opt/isaac-sim",
    ),
}

_KIT_MIN_VERSION = (106, 0)
_ISAAC_MIN_VERSION = (4, 0)


# ════════════════════════════════════════════════════════════════════════
# Platform detection
# ════════════════════════════════════════════════════════════════════════

def detect_platform() -> str:
    """Return one of ``"Windows"``, ``"Linux"``, or ``"WSL"``.

    WSL is distinguished from plain Linux by inspecting
    ``/proc/version`` for the ``microsoft`` marker, which is present in
    both WSL1 and WSL2 kernels.
    """
    system = platform.system()
    if system == "Windows":
        return "Windows"
    if system == "Linux":
        proc_version = Path("/proc/version")
        try:
            if proc_version.exists() and "microsoft" in proc_version.read_text().lower():
                return "WSL"
        except OSError:
            pass
        return "Linux"
    # macOS and anything else: Omniverse Kit does not officially support
    # it, but we do not want detection itself to raise -- callers decide
    # whether an unsupported platform is fatal via validate().
    return system


# Matches Windows-style ``%VAR%`` tokens so they can be substituted on
# platforms (Linux/WSL) where ``os.path.expandvars`` only understands
# POSIX ``$VAR`` / ``${VAR}`` syntax and would otherwise leave them
# untouched (silently breaking any WSL candidate that reaches into the
# Windows side of the filesystem, e.g. ``/mnt/c/Users/%USERNAME%/...``).
_WIN_VAR_PATTERN = re.compile(r"%([^%/\\]+)%")


def _expand_windows_style_vars(path_str: str) -> str:
    """Substitute ``%VAR%`` tokens using the current environment.

    ``%USERNAME%`` falls back to the POSIX ``$USER`` variable when
    ``USERNAME`` itself isn't set (the common case on Linux/WSL, where
    only ``$USER`` exists natively). Unresolvable tokens are left as-is
    so a subsequent ``.exists()`` check simply (and correctly) fails
    rather than raising.
    """

    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name.upper() == "USERNAME":
            return os.environ.get("USERNAME") or os.environ.get("USER") or match.group(0)
        return os.environ.get(name, match.group(0))

    return _WIN_VAR_PATTERN.sub(_sub, path_str)


def _expand(path_str: str) -> Path:
    """Expand ``%ENV%``, ``~``, and ``$ENV``-style tokens in a path string."""
    expanded = _expand_windows_style_vars(path_str)
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def _first_existing(candidates: tuple[str, ...]) -> Optional[Path]:
    """Return the first candidate path that exists on disk, expanding globs."""
    for candidate in candidates:
        expanded = _expand(candidate)
        if "*" in expanded.name:
            parent = expanded.parent
            if parent.exists():
                matches = sorted(parent.glob(expanded.name), reverse=True)
                if matches:
                    return matches[0]
            continue
        if expanded.exists():
            return expanded
    return None


def _parse_version_tuple(raw: str) -> Optional[tuple[int, ...]]:
    """Parse a leading dotted-integer version prefix out of a directory name.

    E.g. ``"isaac-sim-4.2.0"`` -> ``(4, 2, 0)``. Returns ``None`` if no
    parsable version prefix is found.
    """
    digits = ""
    parts: list[int] = []
    for ch in raw:
        if ch.isdigit():
            digits += ch
        elif ch == "." and digits:
            parts.append(int(digits))
            digits = ""
        elif parts or digits:
            break
    if digits:
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _meets_minimum(version: Optional[tuple[int, ...]], minimum: tuple[int, int]) -> bool:
    """Return True if ``version >= minimum``. Unknown versions are treated as unmet."""
    if not version:
        return False
    padded = version + (0,) * max(0, len(minimum) - len(version))
    return padded[: len(minimum)] >= minimum


def _select_versioned_subdir(root: Path, minimum: tuple[int, int]) -> Optional[Path]:
    """Among ``root``'s immediate subdirectories, pick the best versioned one.

    ``_KIT_CANDIDATES`` point at package *roots* (e.g. ``.../ov/pkg``)
    that typically contain one subdirectory per installed app version
    (``create-2023.2.0``, ``kit-106.4.0``, ...). Without descending into
    these, ``kit_root.name`` is something generic like ``"pkg"`` with no
    parsable version, which meant version validation silently never ran.

    Returns the highest-versioned subdirectory that meets ``minimum`` if
    one exists; otherwise the highest-versioned subdirectory found at
    all (so validation can still correctly report it as too old);
    otherwise ``None`` if no versioned subdirectory exists.
    """
    if not root.is_dir():
        return None

    candidates: list[tuple[tuple[int, ...], Path]] = []
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            version = _parse_version_tuple(child.name)
            if version:
                candidates.append((version, child))
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    for version, child in candidates:
        if _meets_minimum(version, minimum):
            return child
    return candidates[0][1]


def detect_kit_root(host_platform: Optional[str] = None) -> Optional[Path]:
    """Best-effort detection of an installed Omniverse Kit root.

    Returns ``None`` if no known install location exists -- this is not
    an error at detection time, only at validation time (if a component
    that requires Kit is asked to run).

    When a package root (rather than a directly versioned app
    directory) is found, this descends one level to locate the actual
    versioned app directory so downstream version validation has a
    meaningful ``kit_root.name`` to parse.
    """
    host_platform = host_platform or detect_platform()
    candidates = _KIT_CANDIDATES.get(host_platform, _KIT_CANDIDATES["Linux"])
    found = _first_existing(candidates)
    if found is None:
        return None

    # If `found` itself already has a parsable version (e.g. a glob
    # candidate resolved directly to a versioned dir), use it as-is.
    if _parse_version_tuple(found.name) is not None:
        logger.debug("Detected Kit root at '%s' (platform=%s)", found, host_platform)
        return found

    versioned = _select_versioned_subdir(found, _KIT_MIN_VERSION)
    result = versioned or found
    logger.debug("Detected Kit root at '%s' (platform=%s)", result, host_platform)
    return result


def detect_isaac_root(host_platform: Optional[str] = None) -> Optional[Path]:
    """Best-effort detection of an installed Isaac Sim root."""
    host_platform = host_platform or detect_platform()
    candidates = _ISAAC_CANDIDATES.get(host_platform, _ISAAC_CANDIDATES["Linux"])
    found = _first_existing(candidates)
    if found:
        logger.debug("Detected Isaac Sim root at '%s' (platform=%s)", found, host_platform)
    return found


# ════════════════════════════════════════════════════════════════════════
# Renderer options
# ════════════════════════════════════════════════════════════════════════

_VALID_RENDERERS = ("rtx_realtime", "rtx_pathtracing", "hydra_storm")

# Known-today values. Deliberately not an Enum -- new backends/engines/
# planners are expected to land as plain strings without a code change
# here; validate() only warns-via-exception on genuinely unknown values
# so typos are still caught.
_VALID_SIMULATION_BACKENDS = ("omniverse", "bullet", "mujoco", "gazebo", "unity", "unreal")
_VALID_USD_FORMATS = ("usda", "usdc", "usdz")
_VALID_PHYSICS_ENGINES = ("physx", "bullet", "mujoco")
_VALID_PLANNERS = ("pso", "astar", "rrt", "rrt_star", "ppo", "none")


# ════════════════════════════════════════════════════════════════════════
# OmniverseConfig
# ════════════════════════════════════════════════════════════════════════

@dataclass
class OmniverseConfig:
    """Central configuration for every Omniverse/Isaac Sim component.

    Every field has a safe, environment-detected or conservative
    default so ``OmniverseConfig()`` never raises -- call ``validate()``
    explicitly once a component actually needs a guaranteed-usable
    configuration (e.g. right before ``app_launcher.launch()``).

    Attributes:
        kit_root: Root install directory of Omniverse Kit, or ``None``
            if not detected/configured.
        isaac_root: Root install directory of Isaac Sim, or ``None``.
        asset_root: Root directory PhysWorldLM reads/writes simulation
            assets from (models, textures, USD references).
        usd_output_dir: Directory the Scene Compiler / usd_exporter
            writes generated ``.usda``/``.usdc`` stages to.
        cache_dir: Scratch/cache directory for downloaded or converted
            assets.
        extension_dir: Additional Kit extension search path, if any.
        nucleus_server: Nucleus server URL for collaborative/cloud
            asset storage. ``None`` for local-only operation.
        physics_dt: PhysX simulation timestep, in seconds.
        render_dt: Render loop timestep, in seconds.
        headless: Whether Kit should launch without a viewport window.
        renderer: One of ``"rtx_realtime"``, ``"rtx_pathtracing"``,
            ``"hydra_storm"``.
        gpu_device: CUDA device index to target, or ``None`` for
            Kit's default selection.
        enable_rtx: Enable the RTX renderer extension set.
        enable_dlss: Enable DLSS upscaling (requires ``enable_rtx``).
        enable_livestream: Enable Kit's livestreaming extension.
        enable_physics: Enable the PhysX extension set.
        enable_replicator: Enable Omniverse Replicator (synthetic data
            generation).
        enable_ros2: Enable the ROS2 bridge extension.
        platform: Detected/overridden host platform
            (``"Windows"`` / ``"Linux"`` / ``"WSL"``). Populated
            automatically in ``__post_init__`` if left as ``""``.
        simulation_backend: Target simulator. ``"omniverse"`` today;
            reserved values (``"bullet"``, ``"mujoco"``, ``"gazebo"``,
            ``"unity"``, ``"unreal"``) let the rest of PhysWorldLM stay
            simulator-agnostic as new backends are added. This module
            itself only ever drives the Omniverse/Isaac Sim path --
            other backends are configuration placeholders until their
            own connector packages exist.
        usd_format: Default export format for generated stages
            (``"usda"`` / ``"usdc"`` / ``"usdz"``). Consumed by
            ``compiler/usd_exporter.py``; this field only records the
            preference.
        asset_search_paths: Ordered list of roots searched for assets
            (local disk, Nucleus mounts, Isaac's bundled asset pack,
            downloaded/cached assets, ...). Preferred over the single
            ``asset_root`` for anything beyond a trivial local setup;
            ``asset_root``, if set, is treated as the first search path.
        planner: Motion/mission planner selection (``"pso"`` today;
            ``"astar"``, ``"rrt"``, ``"rrt_star"``, ``"ppo"`` reserved,
            ``"none"`` for stages with no planner attached). This
            module only records the selection -- planner implementations
            live outside the Omniverse connector.
        terrain_file: Path to the active terrain source (heightmap,
            GeoTIFF, USD terrain reference, ...). ``None`` if no terrain
            is configured yet.
        terrain_cache: Directory for terrain tiles/derived data that
            are expensive to regenerate (e.g. resampled heightmaps).
        terrain_resolution: Terrain sampling resolution, in meters per
            sample/pixel.
        terrain_scale: Uniform scale multiplier applied to terrain
            vertical/horizontal units, for switching between planetary
            bodies (e.g. Earth vs. Moon vs. Mars datasets) without
            re-authoring the terrain pipeline.
        physics_engine: Physics backend (``"physx"`` / ``"bullet"`` /
            ``"mujoco"``). Distinct from ``enable_physics``, which is
            just an on/off switch for whichever engine is selected here.
        config_version: Schema version of this config object, for
            forward/backward compatibility as fields are added.
        enabled_plugins: Names of optional PhysWorldLM plugins to load
            (e.g. ``"replicator"``, ``"ros2"``, ``"navigation"``,
            ``"planner"``, ``"sensor"``). Purely declarative here --
            actual plugin loading is owned by whichever component
            bootstraps the runtime.
    """

    kit_root: Optional[Path] = None
    isaac_root: Optional[Path] = None
    asset_root: Optional[Path] = None
    usd_output_dir: Path = field(default_factory=lambda: Path("output/usd"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache/physworldlm"))
    extension_dir: Optional[Path] = None
    nucleus_server: Optional[str] = None

    physics_dt: float = 1.0 / 60.0
    render_dt: float = 1.0 / 60.0

    headless: bool = True
    renderer: str = "rtx_realtime"
    gpu_device: Optional[int] = None

    enable_rtx: bool = True
    enable_dlss: bool = False
    enable_livestream: bool = False
    enable_physics: bool = True
    enable_replicator: bool = False
    enable_ros2: bool = False

    platform: str = ""

    simulation_backend: str = "omniverse"
    usd_format: str = "usda"
    asset_search_paths: list[Path] = field(default_factory=list)

    planner: str = "pso"

    terrain_file: Optional[Path] = None
    terrain_cache: Path = field(default_factory=lambda: Path(".cache/physworldlm/terrain"))
    terrain_resolution: float = 30.0
    terrain_scale: float = 1.0

    physics_engine: str = "physx"

    config_version: str = "1.0.0"
    enabled_plugins: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Post-init normalization
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.platform:
            self.platform = detect_platform()

        path_attrs = (
            "kit_root", "isaac_root", "asset_root", "usd_output_dir",
            "cache_dir", "extension_dir", "terrain_file", "terrain_cache",
        )
        for attr in path_attrs:
            value = getattr(self, attr)
            if value is not None and not isinstance(value, Path):
                setattr(self, attr, Path(value))

        self.asset_search_paths = [
            p if isinstance(p, Path) else Path(p) for p in self.asset_search_paths
        ]
        # asset_root, if set, is treated as the first (highest-priority)
        # search path unless it's already present -- keeps the common
        # single-root case working without callers needing to know
        # asset_search_paths exists.
        if self.asset_root is not None and self.asset_root not in self.asset_search_paths:
            self.asset_search_paths.insert(0, self.asset_root)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "OmniverseConfig":
        """Build a config using automatic environment detection only.

        Detects platform, Kit root, and Isaac root from known install
        locations. Does not consult environment variables or files --
        use ``load_from_env()`` / ``load_from_yaml()`` when explicit
        overrides are required.
        """
        host_platform = detect_platform()
        config = cls(
            kit_root=detect_kit_root(host_platform),
            isaac_root=detect_isaac_root(host_platform),
            platform=host_platform,
        )
        logger.info(
            "Built default OmniverseConfig (platform=%s, kit_root=%s, isaac_root=%s)",
            config.platform, config.kit_root, config.isaac_root,
        )
        return config

    @classmethod
    def load_from_env(cls, base: Optional["OmniverseConfig"] = None) -> "OmniverseConfig":
        """Build a config from environment variables, layered over ``base``.

        Args:
            base: Config to start from (defaults to ``OmniverseConfig.default()``).
                Any ``PWLM_*`` environment variable present overrides the
                corresponding field. A deep copy of ``base`` is taken
                before mutation so callers that hold their own reference
                to ``base`` are never surprised by in-place changes.

        Raises:
            ConfigLoadError: If a numeric/boolean env var cannot be parsed.
        """
        import copy

        config = copy.deepcopy(base) if base is not None else cls.default()
        try:
            config = cls._apply_env_overrides(config)
        except ValueError as exc:
            raise ConfigLoadError(f"Failed to parse environment configuration: {exc}") from exc
        logger.info("Applied environment overrides to OmniverseConfig.")
        return config

    @staticmethod
    def _apply_env_overrides(config: "OmniverseConfig") -> "OmniverseConfig":
        env = os.environ

        def _path_or(current: Optional[Path], key: str) -> Optional[Path]:
            raw = env.get(key)
            return _expand(raw) if raw else current

        def _bool_or(current: bool, key: str) -> bool:
            raw = env.get(key)
            if raw is None:
                return current
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _float_or(current: float, key: str) -> float:
            raw = env.get(key)
            return float(raw) if raw is not None else current

        def _int_or(current: Optional[int], key: str) -> Optional[int]:
            raw = env.get(key)
            return int(raw) if raw is not None else current

        config.kit_root = _path_or(config.kit_root, EnvVar.KIT_ROOT)
        config.isaac_root = _path_or(config.isaac_root, EnvVar.ISAAC_ROOT)
        config.asset_root = _path_or(config.asset_root, EnvVar.ASSET_ROOT)
        usd_out = _path_or(config.usd_output_dir, EnvVar.USD_OUTPUT_DIR)
        config.usd_output_dir = usd_out if usd_out is not None else config.usd_output_dir
        cache = _path_or(config.cache_dir, EnvVar.CACHE_DIR)
        config.cache_dir = cache if cache is not None else config.cache_dir
        config.extension_dir = _path_or(config.extension_dir, EnvVar.EXTENSION_DIR)
        config.nucleus_server = env.get(EnvVar.NUCLEUS_SERVER, config.nucleus_server)

        config.physics_dt = _float_or(config.physics_dt, EnvVar.PHYSICS_DT)
        config.render_dt = _float_or(config.render_dt, EnvVar.RENDER_DT)

        config.headless = _bool_or(config.headless, EnvVar.HEADLESS)
        config.renderer = env.get(EnvVar.RENDERER, config.renderer)
        config.gpu_device = _int_or(config.gpu_device, EnvVar.GPU_DEVICE)

        config.enable_rtx = _bool_or(config.enable_rtx, EnvVar.ENABLE_RTX)
        config.enable_dlss = _bool_or(config.enable_dlss, EnvVar.ENABLE_DLSS)
        config.enable_livestream = _bool_or(config.enable_livestream, EnvVar.ENABLE_LIVESTREAM)
        config.enable_physics = _bool_or(config.enable_physics, EnvVar.ENABLE_PHYSICS)
        config.enable_replicator = _bool_or(config.enable_replicator, EnvVar.ENABLE_REPLICATOR)
        config.enable_ros2 = _bool_or(config.enable_ros2, EnvVar.ENABLE_ROS2)

        config.simulation_backend = env.get(EnvVar.SIMULATION_BACKEND, config.simulation_backend)
        config.usd_format = env.get(EnvVar.USD_FORMAT, config.usd_format)
        config.physics_engine = env.get(EnvVar.PHYSICS_ENGINE, config.physics_engine)
        config.planner = env.get(EnvVar.PLANNER, config.planner)

        config.terrain_file = _path_or(config.terrain_file, EnvVar.TERRAIN_FILE)
        terrain_cache = _path_or(config.terrain_cache, EnvVar.TERRAIN_CACHE)
        config.terrain_cache = terrain_cache if terrain_cache is not None else config.terrain_cache
        config.terrain_resolution = _float_or(config.terrain_resolution, EnvVar.TERRAIN_RESOLUTION)
        config.terrain_scale = _float_or(config.terrain_scale, EnvVar.TERRAIN_SCALE)

        # asset_root may have just changed via env override above;
        # re-run the search-path fold so PWLM_ASSET_ROOT still takes
        # priority even when set only via environment variable.
        if config.asset_root is not None and config.asset_root not in config.asset_search_paths:
            config.asset_search_paths.insert(0, config.asset_root)

        return config

    @classmethod
    def load_from_yaml(cls, path: "str | Path") -> "OmniverseConfig":
        """Load a config from a YAML file.

        Requires ``PyYAML`` to be installed. The import is deferred to
        this call site so importing ``config.py`` itself never requires
        a third-party dependency.

        Raises:
            ConfigLoadError: If PyYAML is missing, the file does not
                exist, or its contents cannot be parsed into the known
                ``OmniverseConfig`` fields.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigLoadError(
                "load_from_yaml() requires PyYAML ('pip install pyyaml'), "
                "which is not installed in this environment."
            ) from exc

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise ConfigLoadError(f"Config file not found: '{yaml_path}'")

        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise ConfigLoadError(f"Failed to parse YAML config '{yaml_path}': {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigLoadError(f"Config file '{yaml_path}' must contain a mapping at the top level.")

        known_fields = {f.name for f in fields(cls)}
        unknown = set(raw) - known_fields
        if unknown:
            logger.warning("Ignoring unknown key(s) in '%s': %s", yaml_path, sorted(unknown))

        filtered = {k: v for k, v in raw.items() if k in known_fields}
        try:
            config = cls(**filtered)
        except TypeError as exc:
            raise ConfigLoadError(f"Invalid config values in '{yaml_path}': {exc}") from exc

        logger.info("Loaded OmniverseConfig from '%s'.", yaml_path)
        return config

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: "str | Path", *, as_yaml: bool = True) -> Path:
        """Persist this config to disk.

        Args:
            path: Destination file path.
            as_yaml: Write YAML if ``True`` (requires PyYAML) and the
                suffix isn't already ``.json``; otherwise writes JSON.
                Paths and other non-JSON-native types are serialized as
                strings.

        Raises:
            ConfigSaveError: If the file cannot be written, or YAML was
                requested but PyYAML is unavailable.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()

        write_yaml = as_yaml and out_path.suffix.lower() not in (".json",)
        try:
            if write_yaml:
                try:
                    import yaml  # type: ignore
                except ImportError as exc:
                    raise ConfigSaveError(
                        "save(as_yaml=True) requires PyYAML ('pip install pyyaml')."
                    ) from exc
                out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            else:
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ConfigSaveError(f"Failed to write config to '{out_path}': {exc}") from exc

        logger.info("Saved OmniverseConfig to '%s'.", out_path)
        return out_path

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a JSON/YAML-friendly plain dict.

        ``dataclasses.asdict()`` recurses into nested dataclasses and
        mappings, but it does not know how to stringify arbitrary
        objects (like ``Path``) sitting inside a plain ``list`` -- so
        ``asset_search_paths`` and ``enabled_plugins`` need an explicit
        pass to make the payload actually JSON/YAML-safe.
        """
        raw = asdict(self)
        for key, value in raw.items():
            if isinstance(value, Path):
                raw[key] = str(value)
            elif isinstance(value, list):
                raw[key] = [str(item) if isinstance(item, Path) else item for item in value]
        return raw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, *, require_kit: bool = True, require_isaac: bool = False) -> None:
        """Validate that this config describes a usable installation.

        Args:
            require_kit: If True, ``kit_root`` must be set, exist, and
                (when a version can be determined) meet the minimum
                supported Kit version (106+).
            require_isaac: If True, applies the equivalent checks to
                ``isaac_root`` (Isaac Sim 4.x+).

        Raises:
            ConfigValidationError: On the first validation failure
                encountered. Callers that want a full report should
                catch this and inspect ``str(exc)``; all issues found
                up to that point are included in the message.
        """
        issues: list[str] = []

        if self.platform not in ("Windows", "Linux", "WSL"):
            issues.append(
                f"Unsupported platform '{self.platform}'; expected Windows, Linux, or WSL."
            )

        if require_kit:
            if self.kit_root is None:
                issues.append("kit_root is not set and no Omniverse Kit installation was detected.")
            elif not self.kit_root.exists():
                issues.append(f"kit_root '{self.kit_root}' does not exist.")
            else:
                version = _parse_version_tuple(self.kit_root.name)
                if version is not None and not _meets_minimum(version, _KIT_MIN_VERSION):
                    issues.append(
                        f"Kit version {version} at '{self.kit_root}' is below the minimum "
                        f"supported version {_KIT_MIN_VERSION}."
                    )

        if require_isaac:
            if self.isaac_root is None:
                issues.append("isaac_root is not set and no Isaac Sim installation was detected.")
            elif not self.isaac_root.exists():
                issues.append(f"isaac_root '{self.isaac_root}' does not exist.")
            else:
                version = _parse_version_tuple(self.isaac_root.name)
                if version is not None and not _meets_minimum(version, _ISAAC_MIN_VERSION):
                    issues.append(
                        f"Isaac Sim version {version} at '{self.isaac_root}' is below the "
                        f"minimum supported version {_ISAAC_MIN_VERSION}."
                    )

        if self.renderer not in _VALID_RENDERERS:
            issues.append(f"renderer '{self.renderer}' must be one of {_VALID_RENDERERS}.")

        if self.enable_dlss and not self.enable_rtx:
            issues.append("enable_dlss=True requires enable_rtx=True.")

        if self.physics_dt <= 0:
            issues.append(f"physics_dt must be > 0 (got {self.physics_dt}).")
        if self.render_dt <= 0:
            issues.append(f"render_dt must be > 0 (got {self.render_dt}).")

        if self.gpu_device is not None and self.gpu_device < 0:
            issues.append(f"gpu_device must be >= 0 if set (got {self.gpu_device}).")

        if self.simulation_backend not in _VALID_SIMULATION_BACKENDS:
            issues.append(
                f"simulation_backend '{self.simulation_backend}' must be one of "
                f"{_VALID_SIMULATION_BACKENDS}."
            )
        elif self.simulation_backend != "omniverse":
            logger.warning(
                "simulation_backend='%s' is configured but this connector package "
                "only drives Omniverse/Isaac Sim; other backends are placeholders "
                "until their own connector exists.", self.simulation_backend,
            )

        if self.usd_format not in _VALID_USD_FORMATS:
            issues.append(f"usd_format '{self.usd_format}' must be one of {_VALID_USD_FORMATS}.")

        if self.physics_engine not in _VALID_PHYSICS_ENGINES:
            issues.append(
                f"physics_engine '{self.physics_engine}' must be one of {_VALID_PHYSICS_ENGINES}."
            )
        elif self.physics_engine != "physx" and self.enable_physics and self.simulation_backend == "omniverse":
            issues.append(
                f"physics_engine='{self.physics_engine}' is not available under "
                f"simulation_backend='omniverse' (only 'physx' is)."
            )

        if self.planner not in _VALID_PLANNERS:
            issues.append(f"planner '{self.planner}' must be one of {_VALID_PLANNERS}.")

        if self.terrain_resolution <= 0:
            issues.append(f"terrain_resolution must be > 0 (got {self.terrain_resolution}).")
        if self.terrain_scale <= 0:
            issues.append(f"terrain_scale must be > 0 (got {self.terrain_scale}).")
        if self.terrain_file is not None and not self.terrain_file.exists():
            issues.append(f"terrain_file '{self.terrain_file}' does not exist.")

        if issues:
            raise ConfigValidationError("; ".join(issues))

        logger.info("OmniverseConfig validated successfully (kit_root=%s).", self.kit_root)

    @property
    def is_kit_available(self) -> bool:
        """Cheap, non-raising check for whether a Kit install was found."""
        return self.kit_root is not None and self.kit_root.exists()

    @property
    def is_isaac_available(self) -> bool:
        """Cheap, non-raising check for whether an Isaac Sim install was found."""
        return self.isaac_root is not None and self.isaac_root.exists()


__all__ = [
    "OmniverseConfig",
    "EnvVar",
    "OmniverseConfigError",
    "ConfigValidationError",
    "ConfigLoadError",
    "ConfigSaveError",
    "detect_platform",
    "detect_kit_root",
    "detect_isaac_root",
]
