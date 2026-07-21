"""
runtime/camera_manager.py
══════════════════════════════════════════════════════════════════════════
Cinematic camera subsystem for PhysWorldLM's Omniverse Runtime.

Pipeline position
------------------
    OmniverseRuntime.initialize_camera()  (creates the default placeholder)
            │
            ▼
    ┌────────────────┐
    │ CAMERA MANAGER  │   <-- this module (registered subsystem)
    └────────────────┘
            │
            ▼
      OmniverseRuntime._step_frame()  (every frame, via SubsystemRegistry)

Scope
-----
This module implements the `omniverse_runtime.RuntimeSubsystem` protocol
and is registered with `OmniverseRuntime.subsystems`. It owns USD Camera
prims and camera-rig behavior (orbit, follow, top-down, cockpit, free)
on top of the placeholder camera handle `OmniverseRuntime.initialize_camera()`
already creates at `/World/Cameras/MainCamera` -- this subsystem is what
gives that placeholder real behavior, rather than replacing it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("physworldlm.camera_manager")
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

CAMERAS_ROOT = "/World/Cameras"
DEFAULT_CAMERA_PATH = f"{CAMERAS_ROOT}/MainCamera"


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class CameraError(Exception):
    """Raised when a camera cannot be created, switched, or animated."""


# ════════════════════════════════════════════════════════════════════════
# Enums / data
# ════════════════════════════════════════════════════════════════════════

class CameraMode(Enum):
    ORBIT = "orbit"
    FOLLOW = "follow"
    TOP_DOWN = "top_down"
    COCKPIT = "cockpit"
    FREE = "free"


@dataclass
class CameraRig:
    """State for a single managed camera."""

    prim_path: str
    mode: CameraMode
    target_prim_path: Optional[str] = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    focal_length_mm: float = 35.0
    orbit_radius_m: float = 50.0
    orbit_angle_deg: float = 0.0
    orbit_speed_deg_s: float = 10.0
    follow_offset_m: tuple[float, float, float] = (-20.0, 8.0, 0.0)


@dataclass
class CameraManagerConfig:
    """Tuning for the camera manager.

    Attributes:
        default_mode: Mode assigned to the default camera at startup.
        top_down_height_m: Altitude of the top-down camera above origin.
    """

    default_mode: CameraMode = CameraMode.ORBIT
    top_down_height_m: float = 2000.0


# ════════════════════════════════════════════════════════════════════════
# CameraManager
# ════════════════════════════════════════════════════════════════════════

class CameraManager:
    """`RuntimeSubsystem` implementation owning USD Camera prims and rigs.

    Example:
        >>> cameras = CameraManager(stage)
        >>> runtime.subsystems.register(cameras)
        >>> runtime.initialize()
        >>> cameras.follow_entity("/World/Entities/F16_01")
    """

    name = "camera_manager"

    def __init__(self, stage: Any, config: CameraManagerConfig | None = None) -> None:
        """Initialize the manager.

        Args:
            stage: An open USD stage handle.
            config: Camera tuning. Defaults to `CameraManagerConfig()`.
        """
        self._stage = stage
        self._config = config or CameraManagerConfig()
        self._pxr_available = self._detect_pxr()
        self._rigs: dict[str, CameraRig] = {}
        self._active_path: Optional[str] = None

    @staticmethod
    def _detect_pxr() -> bool:
        try:
            import pxr  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def active_camera(self) -> Optional[CameraRig]:
        return self._rigs.get(self._active_path) if self._active_path else None

    # ── RuntimeSubsystem protocol ───────────────────────────────────

    def initialize(self, registry: Any, config: Any) -> None:
        """One-time setup: author the default camera and rig."""
        self._define_prim(CAMERAS_ROOT, "Scope")
        self.create_default_camera()
        logger.info("CameraManager initialized with default camera at %s", DEFAULT_CAMERA_PATH)

    def update(self, dt: float, registry: Any) -> None:
        """Advance the active camera rig by `dt` seconds."""
        rig = self.active_camera
        if rig is None:
            return
        self.animate_camera(rig.prim_path, dt, registry)

    def shutdown(self) -> None:
        """Release manager state (no persistent external resources held)."""
        logger.info("CameraManager shutdown (%d rig(s) managed).", len(self._rigs))
        self._rigs.clear()
        self._active_path = None

    # ── camera creation / switching ─────────────────────────────────

    def create_default_camera(self) -> str:
        """Create the default camera using `CameraManagerConfig.default_mode`."""
        rig = CameraRig(prim_path=DEFAULT_CAMERA_PATH, mode=self._config.default_mode)
        self._define_prim(DEFAULT_CAMERA_PATH, "Camera", attrs={"focal_length_mm": rig.focal_length_mm})
        self._rigs[DEFAULT_CAMERA_PATH] = rig
        self._active_path = DEFAULT_CAMERA_PATH
        logger.info("Default camera created at %s (mode=%s)", DEFAULT_CAMERA_PATH, rig.mode.value)
        return DEFAULT_CAMERA_PATH

    def follow_entity(self, target_prim_path: str, *, camera_path: Optional[str] = None) -> str:
        """Create or reconfigure a camera in FOLLOW mode tracking `target_prim_path`."""
        path = camera_path or f"{CAMERAS_ROOT}/FollowCam"
        rig = self._rigs.get(path) or CameraRig(prim_path=path, mode=CameraMode.FOLLOW)
        rig.mode = CameraMode.FOLLOW
        rig.target_prim_path = target_prim_path
        self._define_prim(path, "Camera", attrs={"mode": rig.mode.value, "target": target_prim_path})
        self._rigs[path] = rig
        logger.info("Follow camera '%s' now tracking '%s'", path, target_prim_path)
        return path

    def switch_camera(self, camera_path: str) -> None:
        """Make `camera_path` the active camera.

        Raises:
            CameraError: If `camera_path` has not been created via one
                of `create_default_camera()` / `follow_entity()` / a
                direct rig registration.
        """
        if camera_path not in self._rigs:
            raise CameraError(f"Cannot switch to unknown camera '{camera_path}'.")
        self._active_path = camera_path
        logger.info("Active camera switched to %s", camera_path)

    def create_orbit_camera(self, *, radius_m: float = 50.0, speed_deg_s: float = 10.0) -> str:
        """Create an orbit camera circling the world origin."""
        path = f"{CAMERAS_ROOT}/OrbitCam"
        rig = CameraRig(prim_path=path, mode=CameraMode.ORBIT, orbit_radius_m=radius_m, orbit_speed_deg_s=speed_deg_s)
        self._define_prim(path, "Camera", attrs={"mode": rig.mode.value})
        self._rigs[path] = rig
        logger.info("Orbit camera created at %s (radius=%.1fm)", path, radius_m)
        return path

    def create_top_down_camera(self) -> str:
        """Create a top-down camera fixed above the world origin."""
        path = f"{CAMERAS_ROOT}/TopDownCam"
        rig = CameraRig(
            prim_path=path,
            mode=CameraMode.TOP_DOWN,
            position=(0.0, self._config.top_down_height_m, 0.0),
        )
        self._define_prim(path, "Camera", attrs={"mode": rig.mode.value})
        self._rigs[path] = rig
        logger.info("Top-down camera created at %s (height=%.0fm)", path, self._config.top_down_height_m)
        return path

    def create_cockpit_camera(self, target_prim_path: str) -> str:
        """Create a cockpit (first-person, entity-attached) camera."""
        path = f"{CAMERAS_ROOT}/CockpitCam"
        rig = CameraRig(prim_path=path, mode=CameraMode.COCKPIT, target_prim_path=target_prim_path)
        self._define_prim(path, "Camera", attrs={"mode": rig.mode.value, "target": target_prim_path})
        self._rigs[path] = rig
        logger.info("Cockpit camera created at %s (target=%s)", path, target_prim_path)
        return path

    def create_free_camera(self, position: tuple[float, float, float] = (0.0, 10.0, 0.0)) -> str:
        """Create a free (manually-positioned) camera."""
        path = f"{CAMERAS_ROOT}/FreeCam"
        rig = CameraRig(prim_path=path, mode=CameraMode.FREE, position=position)
        self._define_prim(path, "Camera", attrs={"mode": rig.mode.value})
        self._rigs[path] = rig
        logger.info("Free camera created at %s (position=%s)", path, position)
        return path

    # ── per-frame animation ─────────────────────────────────────────

    def animate_camera(self, camera_path: str, dt: float, registry: Any) -> None:
        """Advance the rig at `camera_path` by `dt` seconds according to its mode."""
        rig = self._rigs.get(camera_path)
        if rig is None:
            return

        if rig.mode is CameraMode.ORBIT:
            rig.orbit_angle_deg = (rig.orbit_angle_deg + rig.orbit_speed_deg_s * dt) % 360.0
            angle_rad = math.radians(rig.orbit_angle_deg)
            rig.position = (
                rig.orbit_radius_m * math.cos(angle_rad),
                rig.position[1] or 20.0,
                rig.orbit_radius_m * math.sin(angle_rad),
            )
            self._set_attrs(camera_path, {"position": rig.position})

        elif rig.mode in (CameraMode.FOLLOW, CameraMode.COCKPIT):
            target_pos = self._target_position(rig.target_prim_path, registry)
            if target_pos is None:
                return
            if rig.mode is CameraMode.FOLLOW:
                ox, oy, oz = rig.follow_offset_m
                rig.position = (target_pos[0] + ox, target_pos[1] + oy, target_pos[2] + oz)
            else:  # COCKPIT rides exactly at the target
                rig.position = target_pos
            self._set_attrs(camera_path, {"position": rig.position})

        # TOP_DOWN and FREE cameras are static unless explicitly repositioned.

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _target_position(target_prim_path: Optional[str], registry: Any) -> Optional[tuple[float, float, float]]:
        if target_prim_path is None:
            return None
        entities = getattr(registry, "entities", {})
        entity = entities.get(target_prim_path)
        if entity is None:
            return None
        metadata = getattr(entity, "metadata", {}) or {}
        try:
            return (
                float(metadata.get("position_x", 0.0)),
                float(metadata.get("position_y", 0.0)),
                float(metadata.get("position_z", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def _define_prim(self, path: str, prim_type: str, attrs: Optional[dict[str, Any]] = None) -> None:
        if self._pxr_available and hasattr(self._stage, "DefinePrim"):
            try:
                self._stage.DefinePrim(path, prim_type)
                if attrs:
                    self._set_attrs(path, attrs)
                return
            except Exception as exc:  # noqa: BLE001
                raise CameraError(f"Failed to define camera prim '{path}': {exc}") from exc
        logger.debug("[fallback] would define %s prim at %s with attrs=%s", prim_type, path, attrs or {})

    def _set_attrs(self, prim_path: str, attrs: dict[str, Any]) -> None:
        if self._pxr_available and hasattr(self._stage, "GetPrimAtPath"):
            try:
                from pxr import Sdf  # type: ignore

                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    return
                for key, value in attrs.items():
                    try:
                        prim.CreateAttribute(f"physworldlm:{key}", Sdf.ValueTypeNames.String).Set(str(value))
                    except Exception:  # noqa: BLE001
                        pass
                return
            except Exception as exc:  # noqa: BLE001
                raise CameraError(f"Failed to set attributes on '{prim_path}': {exc}") from exc
        logger.debug("[fallback] would set attrs on %s: %s", prim_path, attrs)
