"""
camera_controller.py
══════════════════════════════════════════════════════════════════════════
Camera subsystem of the PhysWorldLM Execution Layer.

Pipeline position
------------------
    OmniverseRuntime.initialize_runtime_systems()
            │
            ▼
    ┌──────────────────┐
    │CAMERA CONTROLLER │   <-- this module
    └──────────────────┘
            │
            ▼
    Per-frame camera transform updates

Scope
-----
`CameraController` is a `RuntimeSubsystem` (structurally compatible with
`runtime/omniverse_runtime.py`'s `RuntimeSubsystem` contract) responsible
for managing one or more `CameraRig` definitions and recomputing their
world-space transform every frame according to a simple, generic camera
mode (fixed, follow, orbit, free).

It owns:
    * A registry of `CameraRig` instances (prim path, mode, target
      entity, offset, field of view).
    * The per-frame transform recomputation for FOLLOW and ORBIT rigs,
      reading target-entity kinematic state from
      `entity.metadata["kinematics"]` (the same convention used and
      produced by `simulation_controller.SimulationController`).

It deliberately does NOT:
    * Render anything, or talk to the actual Hydra/RTX renderer --
      `OmniverseRuntime.render_frame()` owns the real render call. This
      module only computes *where the camera should be*.
    * Implement cinematic logic, shot composition, or AI-driven framing.

Integration with `runtime/omniverse_runtime.py`
------------------------------------------------
Like the other subsystems in this package, `CameraController` depends
only on the structural shape of `RuntimeEntity` / `EntityRegistry`
(`EntityLike` / `RegistryLike` protocols below) -- no import of
`omniverse_runtime` is required. Register an instance with the runtime
before `initialize()`::

    >>> runtime.subsystems.register(CameraController())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.camera_controller")
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

class CameraControllerError(Exception):
    """Base exception for all camera-subsystem failures."""


class CameraNotFoundError(CameraControllerError):
    """Raised when an operation references a camera rig that is not registered."""


class CameraTargetError(CameraControllerError):
    """Raised when a FOLLOW/ORBIT rig's target entity cannot be resolved or read."""


# ════════════════════════════════════════════════════════════════════════
# Structural protocols (decoupled from omniverse_runtime)
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class EntityLike(Protocol):
    """Structural contract for entities this subsystem can operate on."""

    prim_path: str
    name: str
    metadata: dict[str, Any]


@runtime_checkable
class RegistryLike(Protocol):
    """Structural contract for an entity registry, matching
    `omniverse_runtime.EntityRegistry`."""

    entities: dict[str, EntityLike]

    def get(self, prim_path: str) -> Optional[EntityLike]:
        ...

    def __len__(self) -> int:
        ...


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class CameraMode(Enum):
    """Behavior mode of a `CameraRig`."""

    FIXED = auto()    # Static world-space transform; never recomputed.
    FOLLOW = auto()    # Tracks a target entity's position at a fixed offset.
    ORBIT = auto()      # Orbits a target entity at a fixed radius/height.
    FREE = auto()        # Externally driven (e.g. by an operator/UI); not auto-updated.


Vector3 = tuple[float, float, float]
_ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CameraTransform:
    """World-space camera transform.

    Attributes:
        position_m: World-space position (x, y, z), in meters.
        look_at_m: World-space point the camera is oriented toward.
        up_vector: World-space up vector used to resolve roll.
        fov_deg: Vertical field of view, in degrees.
    """

    position_m: Vector3 = _ZERO_VECTOR
    look_at_m: Vector3 = _ZERO_VECTOR
    up_vector: Vector3 = (0.0, 1.0, 0.0)
    fov_deg: float = 50.0

    def to_dict(self) -> dict:
        return {
            "position_m": self.position_m,
            "look_at_m": self.look_at_m,
            "up_vector": self.up_vector,
            "fov_deg": self.fov_deg,
        }


@dataclass
class CameraRig:
    """A single managed camera definition.

    Attributes:
        prim_path: USD prim path of the camera (e.g. "/World/Cameras/ChaseCam").
        name: Human-readable display name.
        mode: Behavior mode controlling how the transform is recomputed
            each frame.
        target_prim_path: For FOLLOW / ORBIT modes, the prim path of the
            entity to track. Ignored for FIXED / FREE.
        offset_m: For FOLLOW mode, a fixed world-space offset added to
            the target entity's position.
        orbit_radius_m: For ORBIT mode, the horizontal distance from the
            target entity.
        orbit_height_m: For ORBIT mode, the vertical offset above the
            target entity.
        orbit_angular_speed_rads: For ORBIT mode, the angular speed of
            the orbit, in radians per second.
        fov_deg: Vertical field of view, in degrees.
        transform: The rig's current computed `CameraTransform`.
    """

    prim_path: str
    name: str
    mode: CameraMode = CameraMode.FIXED
    target_prim_path: Optional[str] = None
    offset_m: Vector3 = (0.0, 5.0, -10.0)
    orbit_radius_m: float = 15.0
    orbit_height_m: float = 5.0
    orbit_angular_speed_rads: float = 0.3
    fov_deg: float = 50.0
    transform: CameraTransform = field(default_factory=CameraTransform)
    _orbit_angle_rad: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.mode in (CameraMode.FOLLOW, CameraMode.ORBIT) and not self.target_prim_path:
            raise CameraControllerError(
                f"CameraRig '{self.prim_path}' has mode {self.mode.name} but no target_prim_path."
            )
        self.transform.fov_deg = self.fov_deg


# ════════════════════════════════════════════════════════════════════════
# Camera registry
# ════════════════════════════════════════════════════════════════════════

class CameraRegistry:
    """In-memory index of all managed `CameraRig` instances."""

    def __init__(self) -> None:
        self._rigs: dict[str, CameraRig] = {}
        self._active_prim_path: Optional[str] = None

    def register(self, rig: CameraRig, *, make_active: bool = False) -> None:
        self._rigs[rig.prim_path] = rig
        if make_active or self._active_prim_path is None:
            self._active_prim_path = rig.prim_path
        logger.debug("Registered camera rig '%s' (mode=%s).", rig.prim_path, rig.mode.name)

    def get(self, prim_path: str) -> Optional[CameraRig]:
        return self._rigs.get(prim_path)

    def all_rigs(self) -> list[CameraRig]:
        return list(self._rigs.values())

    @property
    def active(self) -> Optional[CameraRig]:
        if self._active_prim_path is None:
            return None
        return self._rigs.get(self._active_prim_path)

    def set_active(self, prim_path: str) -> None:
        if prim_path not in self._rigs:
            raise CameraNotFoundError(f"Cannot activate unknown camera rig '{prim_path}'.")
        self._active_prim_path = prim_path

    def __len__(self) -> int:
        return len(self._rigs)


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CameraControllerConfig:
    """User-configurable settings controlling `CameraController` behavior.

    Attributes:
        default_camera_prim_path: Prim path used for the runtime's
            default camera, created automatically during `initialize()`
            if no rigs have been registered yet.
        default_fov_deg: Field of view applied to the auto-created
            default camera.
    """

    default_camera_prim_path: str = "/World/Cameras/MainCamera"
    default_fov_deg: float = 50.0


# ════════════════════════════════════════════════════════════════════════
# CameraController
# ════════════════════════════════════════════════════════════════════════

class CameraController:
    """Manages camera rigs and recomputes their transforms every frame.

    Implements the `RuntimeSubsystem` contract structurally (`name`,
    `initialize(registry, config)`, `update(dt, registry)`,
    `shutdown()`), so it can be registered directly with
    `omniverse_runtime.OmniverseRuntime.subsystems`.

    Example:
        >>> cameras = CameraController()
        >>> cameras.cameras.register(
        ...     CameraRig(
        ...         prim_path="/World/Cameras/ChaseCam",
        ...         name="Chase Cam",
        ...         mode=CameraMode.FOLLOW,
        ...         target_prim_path="/World/Entities/F16_01",
        ...     ),
        ...     make_active=True,
        ... )
        >>> runtime.subsystems.register(cameras)
    """

    name = "camera_controller"

    def __init__(self, config: Optional[CameraControllerConfig] = None) -> None:
        """Initialize the camera controller (no rigs registered yet).

        Args:
            config: Controller-wide settings. Defaults to
                `CameraControllerConfig()`.
        """
        self.config = config or CameraControllerConfig()
        self.cameras = CameraRegistry()

    # ── RuntimeSubsystem contract ────────────────────────────────────

    def initialize(self, registry: RegistryLike, config: Any) -> None:
        """Ensure at least one camera rig exists.

        If no rigs have been registered by the time `initialize()` runs,
        a single FIXED default camera is created at
        `CameraControllerConfig.default_camera_prim_path`, matching the
        placeholder camera created by
        `omniverse_runtime.OmniverseRuntime.initialize_camera()`.

        Args:
            registry: The entity registry populated by
                `OmniverseRuntime.discover_entities()` /
                `classify_entities()`.
            config: The owning runtime's `RuntimeConfig` (unused
                directly; accepted for interface compatibility).
        """
        logger.info("Initializing camera controller (%d rig(s) pre-registered)", len(self.cameras))

        if len(self.cameras) == 0:
            default_rig = CameraRig(
                prim_path=self.config.default_camera_prim_path,
                name="Main Camera",
                mode=CameraMode.FIXED,
                fov_deg=self.config.default_fov_deg,
            )
            self.cameras.register(default_rig, make_active=True)
            logger.info("No camera rigs registered; created default FIXED camera '%s'.", default_rig.prim_path)

        for rig in self.cameras.all_rigs():
            if rig.mode in (CameraMode.FOLLOW, CameraMode.ORBIT):
                self._validate_target(rig, registry)

        logger.info("Camera controller initialized (%d rig(s)).", len(self.cameras))

    def update(self, dt: float, registry: RegistryLike) -> None:
        """Recompute the transform of every FOLLOW / ORBIT camera rig.

        FIXED rigs are left untouched. FREE rigs are assumed to be
        driven externally (e.g. by an operator UI) and are also left
        untouched here.

        Args:
            dt: Simulation time delta for this frame, in seconds.
            registry: The current entity registry, used to resolve each
                rig's target entity.

        Raises:
            CameraTargetError: If a FOLLOW/ORBIT rig's target entity
                cannot be resolved or its kinematic state cannot be read.
        """
        for rig in self.cameras.all_rigs():
            if rig.mode is CameraMode.FOLLOW:
                self._update_follow(rig, dt, registry)
            elif rig.mode is CameraMode.ORBIT:
                self._update_orbit(rig, dt, registry)
            # FIXED and FREE rigs are intentionally left untouched.

    def shutdown(self) -> None:
        """Release all managed camera rigs."""
        logger.info("Shutting down camera controller (%d rig(s) released).", len(self.cameras))
        self.cameras = CameraRegistry()

    # ── public accessors ─────────────────────────────────────────────

    def get_transform(self, prim_path: str) -> CameraTransform:
        """Return the current `CameraTransform` for the rig at `prim_path`.

        Raises:
            CameraNotFoundError: If no rig is registered at `prim_path`.
        """
        rig = self.cameras.get(prim_path)
        if rig is None:
            raise CameraNotFoundError(f"No camera rig registered at '{prim_path}'.")
        return rig.transform

    # ── internal helpers ─────────────────────────────────────────────

    def _validate_target(self, rig: CameraRig, registry: RegistryLike) -> None:
        if rig.target_prim_path is None or registry.get(rig.target_prim_path) is None:
            raise CameraTargetError(
                f"Camera rig '{rig.prim_path}' (mode={rig.mode.name}) targets unknown "
                f"entity '{rig.target_prim_path}'."
            )

    def _resolve_target_position(self, rig: CameraRig, registry: RegistryLike) -> Vector3:
        target = registry.get(rig.target_prim_path) if rig.target_prim_path else None
        if target is None:
            raise CameraTargetError(
                f"Camera rig '{rig.prim_path}' target entity '{rig.target_prim_path}' no longer exists."
            )

        kinematics = target.metadata.get("kinematics")
        if isinstance(kinematics, dict) and "position_m" in kinematics:
            pos = kinematics["position_m"]
            return (float(pos[0]), float(pos[1]), float(pos[2]))

        logger.warning(
            "Target entity '%s' has no kinematic state; camera rig '%s' tracking world origin.",
            target.name,
            rig.prim_path,
        )
        return _ZERO_VECTOR

    def _update_follow(self, rig: CameraRig, dt: float, registry: RegistryLike) -> None:
        target_position = self._resolve_target_position(rig, registry)
        rig.transform.position_m = _add(target_position, rig.offset_m)
        rig.transform.look_at_m = target_position
        rig.transform.fov_deg = rig.fov_deg

    def _update_orbit(self, rig: CameraRig, dt: float, registry: RegistryLike) -> None:
        target_position = self._resolve_target_position(rig, registry)
        rig._orbit_angle_rad = (rig._orbit_angle_rad + rig.orbit_angular_speed_rads * dt) % (2 * math.pi)

        offset = (
            rig.orbit_radius_m * math.cos(rig._orbit_angle_rad),
            rig.orbit_height_m,
            rig.orbit_radius_m * math.sin(rig._orbit_angle_rad),
        )
        rig.transform.position_m = _add(target_position, offset)
        rig.transform.look_at_m = target_position
        rig.transform.fov_deg = rig.fov_deg


__all__ = [
    "CameraController",
    "CameraControllerConfig",
    "CameraRig",
    "CameraRegistry",
    "CameraTransform",
    "CameraMode",
    "EntityLike",
    "RegistryLike",
    "CameraControllerError",
    "CameraNotFoundError",
    "CameraTargetError",
]
