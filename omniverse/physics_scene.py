"""
physics_scene.py
══════════════════════════════════════════════════════════════════════════
PhysX scene ownership module for the Omniverse connector layer of
PhysWorldLM.

Pipeline position
------------------
    Natural Language → Ontology → WorldSpec → Scene Compiler → scene.usda
                                                                      │
                                                                      ▼
                                                        omniverse/config.py
                                                                      │
                                                                      ▼
                                                     app_launcher.py opens
                                                     the Kit process and a
                                                     Usd.Stage, then hands
                                                     that stage to:
                                                        ┌────────────────────┐
                                                        │ omniverse/          │  <-- this module
                                                        │   physics_scene.py  │
                                                        └────────────────────┘
                                                                      │
                                                                      ▼
                                                        stage_manager.py / the
                                                        simulation step loop

Scope
-----
This module owns the complete PhysX simulation surface once a stage
already exists: the physics scene prim, gravity, materials, rigid
bodies, articulations, joints/constraints, collision filtering,
broadphase/narrowphase selection, CPU/GPU dynamics, sleeping, solver
iteration counts, CCD, contact reports, triggers, stepping, and
snapshot/serialization/replay of scene state.

Design constraints
-------------------
    * ``omni``, ``pxr``, and ``omni.physx`` are *never* imported at
      module load time. Every reference to them is resolved lazily,
      inside function bodies, via :func:`_require_omni` /
      :func:`_require_pxr` / :func:`_require_physx`, so this module can
      be imported (and its dataclasses/validation logic unit tested)
      in a plain Python environment with no Omniverse Kit installed.
    * This module never launches Omniverse Kit (``app_launcher.py``'s
      job), never opens/loads a ``Usd.Stage`` from disk
      (``stage_manager.py``'s job -- a stage is always handed in), and
      never renders a frame. It also never parses natural-language
      prompts, ontologies, or ``WorldSpec`` documents -- by the time a
      ``PhysicsScene`` is constructed, that has already happened
      upstream and produced concrete rigid body / joint / material
      specs.
    * All public mutating methods are guarded by a single
      :class:`threading.RLock` so a ``PhysicsScene`` instance may be
      driven from a background simulation thread while, e.g., a UI or
      RPC thread concurrently queries :meth:`PhysicsScene.statistics`.
    * Every dataclass here is plain and picklable (no live ``omni``/
      ``pxr`` handles inside them) so specs can be constructed,
      validated, serialized, and diffed without touching Omniverse at
      all -- only :meth:`PhysicsScene.create_scene` and onward actually
      talk to PhysX.

Public API
----------
    scene = PhysicsScene(PhysicsSceneConfig(gravity=(0.0, 0.0, -9.81)))
    scene.initialize()
    scene.create_scene(stage)
    scene.add_rigid_body(RigidBodySpec(...))
    scene.add_joint(JointSpec(...))
    scene.step()
    scene.statistics()
    scene.serialize()
    scene.shutdown()

    # or, as a context manager:
    with PhysicsScene(config) as scene:
        scene.create_scene(stage)
        ...
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for static type checkers / IDEs -- never imported at runtime.
    from .config import OmniverseConfig

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.physics_scene")
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

class PhysicsSceneError(Exception):
    """Base class for all physics-scene errors."""


class PhysicsBackendUnavailableError(PhysicsSceneError):
    """Raised when ``omni``/``pxr``/``omni.physx`` cannot be imported.

    This is only ever raised at the point an operation actually needs
    the live backend (e.g. :meth:`PhysicsScene.initialize`); importing
    this module never raises it.
    """


class SceneNotInitializedError(PhysicsSceneError):
    """Raised when an operation requires :meth:`PhysicsScene.initialize` first."""


class SceneAlreadyExistsError(PhysicsSceneError):
    """Raised by :meth:`PhysicsScene.create_scene` if a scene is already active."""


class SceneNotCreatedError(PhysicsSceneError):
    """Raised when an operation requires :meth:`PhysicsScene.create_scene` first."""


class RigidBodyError(PhysicsSceneError):
    """Raised for invalid rigid-body add/remove/lookup operations."""


class JointError(PhysicsSceneError):
    """Raised for invalid joint add/remove/lookup operations."""


class MaterialError(PhysicsSceneError):
    """Raised for invalid physics-material operations."""


class PhysicsSceneValidationError(PhysicsSceneError):
    """Raised by :meth:`PhysicsScene.validate` when the scene state is invalid.

    The message aggregates every issue found in a single pass, joined
    with ``"; "``, mirroring ``OmniverseConfig.validate()`` in
    ``config.py`` so callers can rely on the same "catch once, read the
    full report" pattern across the connector layer.
    """


class SnapshotError(PhysicsSceneError):
    """Raised when a scene snapshot cannot be captured, restored, or replayed."""


# ════════════════════════════════════════════════════════════════════════
# Lazy backend imports
# ════════════════════════════════════════════════════════════════════════
#
# omni / pxr / omni.physx are only ever available inside a running
# Omniverse Kit process. Every access to them goes through these three
# functions so that (a) importing physics_scene.py never requires Kit,
# and (b) a missing/incompatible backend surfaces as one clear
# PhysicsBackendUnavailableError instead of a raw ImportError deep in
# some method body.

_omni_module: Any = None
_pxr_module: Any = None
_physx_module: Any = None


def _require_omni() -> Any:
    """Lazily import and cache the ``omni`` package.

    Returns:
        The imported ``omni`` module.

    Raises:
        PhysicsBackendUnavailableError: If ``omni`` cannot be imported
            (i.e. this process is not running inside Omniverse Kit).
    """
    global _omni_module
    if _omni_module is None:
        try:
            import omni  # type: ignore
        except ImportError as exc:
            raise PhysicsBackendUnavailableError(
                "The 'omni' package is not importable -- physics_scene.py "
                "must run inside an Omniverse Kit process."
            ) from exc
        _omni_module = omni
    return _omni_module


def _require_pxr() -> Any:
    """Lazily import and cache the ``pxr`` (USD) package.

    Returns:
        The imported ``pxr`` module.

    Raises:
        PhysicsBackendUnavailableError: If ``pxr`` cannot be imported.
    """
    global _pxr_module
    if _pxr_module is None:
        try:
            import pxr  # type: ignore
        except ImportError as exc:
            raise PhysicsBackendUnavailableError(
                "The 'pxr' (USD) package is not importable -- physics_scene.py "
                "must run inside an environment with USD/Omniverse Kit available."
            ) from exc
        _pxr_module = pxr
    return _pxr_module


def _require_physx() -> Any:
    """Lazily import and cache the ``omni.physx`` extension module.

    Returns:
        The imported ``omni.physx`` module.

    Raises:
        PhysicsBackendUnavailableError: If ``omni.physx`` cannot be
            imported (e.g. the extension is not enabled).
    """
    global _physx_module
    if _physx_module is None:
        _require_omni()
        try:
            import omni.physx  # type: ignore
        except ImportError as exc:
            raise PhysicsBackendUnavailableError(
                "The 'omni.physx' extension is not importable -- ensure the "
                "'omni.physx' extension is enabled in this Kit process."
            ) from exc
        _physx_module = omni.physx
    return _physx_module


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class SolverType(str, Enum):
    """PhysX solver algorithm."""

    PGS = "pgs"
    """Projected Gauss-Seidel -- PhysX's legacy, faster, less accurate solver."""

    TGS = "tgs"
    """Temporal Gauss-Seidel -- PhysX's default modern solver, more stable
    for stacking, joints, and high mass ratios."""


class BroadphaseType(str, Enum):
    """Broadphase collision-detection algorithm."""

    SAP = "sap"
    """Sweep-And-Prune -- good general-purpose default."""

    MBP = "mbp"
    """Multi-Box-Pruning -- better for large, sparse, non-uniform scenes."""

    GPU = "gpu"
    """GPU broadphase -- requires ``use_gpu_dynamics=True``."""


class CombineMode(str, Enum):
    """How two materials' friction/restitution values are combined on contact."""

    AVERAGE = "average"
    MIN = "min"
    MULTIPLY = "multiply"
    MAX = "max"


class JointType(str, Enum):
    """Supported PhysX joint / constraint types."""

    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    SPHERICAL = "spherical"
    DISTANCE = "distance"
    D6 = "d6"


class ActorType(str, Enum):
    """Kinematic classification of a rigid body."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    KINEMATIC = "kinematic"
    ARTICULATION_LINK = "articulation_link"


class PhysicsBackendMode(str, Enum):
    """Whether PhysX dynamics run on the CPU or GPU."""

    CPU = "cpu"
    GPU = "gpu"


# ════════════════════════════════════════════════════════════════════════
# Constants / defaults
# ════════════════════════════════════════════════════════════════════════

_MIN_PHYSICS_DT = 1.0 / 1000.0
_MAX_PHYSICS_DT = 1.0 / 10.0
_DEFAULT_SOLVER_POSITION_ITERATIONS = 4
_DEFAULT_SOLVER_VELOCITY_ITERATIONS = 1
_DEFAULT_SLEEP_THRESHOLD = 0.005
_DEFAULT_STABILIZATION_THRESHOLD = 0.0025
_DEFAULT_MAX_REPLAY_SNAPSHOTS = 256

_VALID_COMBINE_MODES = tuple(m.value for m in CombineMode)
_VALID_JOINT_TYPES = tuple(t.value for t in JointType)
_VALID_ACTOR_TYPES = tuple(t.value for t in ActorType)


# ════════════════════════════════════════════════════════════════════════
# Dataclasses: materials, filtering, bodies, joints
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsMaterialSpec:
    """A named PhysX physical material.

    Attributes:
        name: Unique material identifier, referenced by
            :attr:`RigidBodySpec.material_name`.
        static_friction: Coulomb static friction coefficient (>= 0).
        dynamic_friction: Coulomb dynamic friction coefficient (>= 0,
            conventionally <= ``static_friction``).
        restitution: Bounciness, in ``[0, 1]`` (0 = fully inelastic,
            1 = fully elastic).
        friction_combine: How this material's friction combines with
            the friction of the other material in a contact.
        restitution_combine: How this material's restitution combines
            with the other material's in a contact.
    """

    name: str
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    friction_combine: CombineMode = CombineMode.AVERAGE
    restitution_combine: CombineMode = CombineMode.AVERAGE

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if not self.name:
            issues.append("Material 'name' must be non-empty.")
        if self.static_friction < 0:
            issues.append(f"Material '{self.name}': static_friction must be >= 0.")
        if self.dynamic_friction < 0:
            issues.append(f"Material '{self.name}': dynamic_friction must be >= 0.")
        if not 0.0 <= self.restitution <= 1.0:
            issues.append(f"Material '{self.name}': restitution must be in [0, 1].")
        if self.friction_combine.value not in _VALID_COMBINE_MODES:
            issues.append(f"Material '{self.name}': invalid friction_combine.")
        if self.restitution_combine.value not in _VALID_COMBINE_MODES:
            issues.append(f"Material '{self.name}': invalid restitution_combine.")
        return issues


@dataclass
class CollisionFilterSpec:
    """Collision-filtering group membership for a rigid body.

    Mirrors PhysX's group/mask filtering model: a body belongs to
    ``group`` and collides with any other body whose group appears in
    its ``collides_with`` set (symmetric application is the caller's
    responsibility -- this spec just records one side's intent).

    Attributes:
        group: Name of the collision group this body belongs to.
        collides_with: Names of groups this body should collide with.
            An empty set (the default) means "collide with everything
            not explicitly excluded" -- i.e. no filtering is applied.
        excludes: Names of groups this body should *never* collide
            with, regardless of ``collides_with``.
    """

    group: str = "default"
    collides_with: frozenset[str] = field(default_factory=frozenset)
    excludes: frozenset[str] = field(default_factory=frozenset)


@dataclass
class RigidBodySpec:
    """A single rigid body (or articulation link) to be added to the scene.

    Attributes:
        body_id: Unique identifier within the owning :class:`PhysicsScene`.
        prim_path: USD prim path this body corresponds to on the stage
            handed to :meth:`PhysicsScene.create_scene`.
        actor_type: Kinematic classification (static / dynamic /
            kinematic / articulation link).
        mass: Mass in kilograms. Ignored for ``STATIC`` bodies.
        material_name: Name of a :class:`PhysicsMaterialSpec` previously
            registered via :meth:`PhysicsScene.set_material`, or
            ``None`` to use PhysX's built-in default material.
        collision_filter: Collision group/mask membership.
        enable_ccd: Enable continuous collision detection for this
            body (recommended for small/fast-moving bodies prone to
            tunneling).
        sleep_threshold: Linear+angular kinetic-energy threshold below
            which this body is allowed to fall asleep.
        stabilization_threshold: Additional stabilization energy
            threshold used by TGS to reduce jitter in resting contacts.
        solver_position_iterations: Per-body override of solver
            position-iteration count, or ``None`` to inherit the scene
            default.
        solver_velocity_iterations: Per-body override of solver
            velocity-iteration count, or ``None`` to inherit the scene
            default.
        is_trigger: If ``True``, this body's shape generates trigger
            (enter/leave) events instead of physical collision response.
        contact_report_enabled: If ``True``, contact events involving
            this body are forwarded to registered contact-report
            callbacks.
        articulation_id: If this body is an ``ARTICULATION_LINK``, the
            identifier of the owning articulation; otherwise ``None``.
        parent_body_id: For an articulation link, the ``body_id`` of
            its parent link (``None`` for the articulation root).
    """

    body_id: str
    prim_path: str
    actor_type: ActorType = ActorType.DYNAMIC
    mass: float = 1.0
    material_name: Optional[str] = None
    collision_filter: CollisionFilterSpec = field(default_factory=CollisionFilterSpec)
    enable_ccd: bool = False
    sleep_threshold: float = _DEFAULT_SLEEP_THRESHOLD
    stabilization_threshold: float = _DEFAULT_STABILIZATION_THRESHOLD
    solver_position_iterations: Optional[int] = None
    solver_velocity_iterations: Optional[int] = None
    is_trigger: bool = False
    contact_report_enabled: bool = False
    articulation_id: Optional[str] = None
    parent_body_id: Optional[str] = None

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if not self.body_id:
            issues.append("RigidBodySpec.body_id must be non-empty.")
        if not self.prim_path:
            issues.append(f"Body '{self.body_id}': prim_path must be non-empty.")
        if self.actor_type.value not in _VALID_ACTOR_TYPES:
            issues.append(f"Body '{self.body_id}': invalid actor_type.")
        if self.actor_type != ActorType.STATIC and self.mass <= 0:
            issues.append(f"Body '{self.body_id}': mass must be > 0 for non-static bodies.")
        if self.sleep_threshold < 0:
            issues.append(f"Body '{self.body_id}': sleep_threshold must be >= 0.")
        if self.stabilization_threshold < 0:
            issues.append(f"Body '{self.body_id}': stabilization_threshold must be >= 0.")
        if self.solver_position_iterations is not None and self.solver_position_iterations < 1:
            issues.append(f"Body '{self.body_id}': solver_position_iterations must be >= 1.")
        if self.solver_velocity_iterations is not None and self.solver_velocity_iterations < 1:
            issues.append(f"Body '{self.body_id}': solver_velocity_iterations must be >= 1.")
        if self.actor_type == ActorType.ARTICULATION_LINK and not self.articulation_id:
            issues.append(f"Body '{self.body_id}': articulation_id required for articulation links.")
        return issues


@dataclass
class JointLimitSpec:
    """Angular/linear limit for a single degree of freedom on a joint.

    Attributes:
        lower: Lower bound (radians for angular DOFs, meters for linear).
        upper: Upper bound. Must be ``>= lower``.
        stiffness: Soft-limit spring stiffness (0 = hard limit).
        damping: Soft-limit spring damping.
    """

    lower: float
    upper: float
    stiffness: float = 0.0
    damping: float = 0.0

    def validate(self, dof_name: str) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if self.upper < self.lower:
            issues.append(f"Joint limit '{dof_name}': upper ({self.upper}) < lower ({self.lower}).")
        if self.stiffness < 0:
            issues.append(f"Joint limit '{dof_name}': stiffness must be >= 0.")
        if self.damping < 0:
            issues.append(f"Joint limit '{dof_name}': damping must be >= 0.")
        return issues


@dataclass
class JointDriveSpec:
    """A position/velocity drive applied to a single joint degree of freedom.

    Attributes:
        target_position: Desired position (radians or meters).
        target_velocity: Desired velocity.
        stiffness: Proportional gain.
        damping: Derivative gain.
        max_force: Force/torque clamp, or ``None`` for unclamped.
    """

    target_position: float = 0.0
    target_velocity: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0
    max_force: Optional[float] = None

    def validate(self, dof_name: str) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if self.stiffness < 0:
            issues.append(f"Joint drive '{dof_name}': stiffness must be >= 0.")
        if self.damping < 0:
            issues.append(f"Joint drive '{dof_name}': damping must be >= 0.")
        if self.max_force is not None and self.max_force < 0:
            issues.append(f"Joint drive '{dof_name}': max_force must be >= 0 if set.")
        return issues


@dataclass
class JointSpec:
    """A joint/constraint connecting two rigid bodies (or one body and the world).

    Attributes:
        joint_id: Unique identifier within the owning :class:`PhysicsScene`.
        joint_type: The kind of constraint this joint represents.
        body0_id: ``body_id`` of the first connected body.
        body1_id: ``body_id`` of the second connected body, or ``None``
            to connect ``body0_id`` to the static world frame.
        local_pos0: Joint frame origin in ``body0``'s local space
            (x, y, z, in meters).
        local_pos1: Joint frame origin in ``body1``'s local space (or
            world space if ``body1_id`` is ``None``).
        limit: Optional single-DOF limit (for REVOLUTE/PRISMATIC/DISTANCE).
        drive: Optional single-DOF drive (for REVOLUTE/PRISMATIC).
        break_force: Linear force above which the joint breaks, or
            ``None`` for unbreakable.
        break_torque: Torque above which the joint breaks, or ``None``
            for unbreakable.
        enable_collision: If ``True``, the two connected bodies still
            collide with each other despite being jointed.
    """

    joint_id: str
    joint_type: JointType
    body0_id: str
    body1_id: Optional[str] = None
    local_pos0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    limit: Optional[JointLimitSpec] = None
    drive: Optional[JointDriveSpec] = None
    break_force: Optional[float] = None
    break_torque: Optional[float] = None
    enable_collision: bool = False

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if not self.joint_id:
            issues.append("JointSpec.joint_id must be non-empty.")
        if not self.body0_id:
            issues.append(f"Joint '{self.joint_id}': body0_id must be non-empty.")
        if self.body1_id is not None and self.body1_id == self.body0_id:
            issues.append(f"Joint '{self.joint_id}': body0_id and body1_id must differ.")
        if self.joint_type.value not in _VALID_JOINT_TYPES:
            issues.append(f"Joint '{self.joint_id}': invalid joint_type.")
        if self.joint_type in (JointType.FIXED, JointType.SPHERICAL) and self.drive is not None:
            issues.append(f"Joint '{self.joint_id}': {self.joint_type.value} joints cannot have a drive.")
        if self.limit is not None:
            issues.extend(self.limit.validate(f"{self.joint_id}.limit"))
        if self.drive is not None:
            issues.extend(self.drive.validate(f"{self.joint_id}.drive"))
        if self.break_force is not None and self.break_force <= 0:
            issues.append(f"Joint '{self.joint_id}': break_force must be > 0 if set.")
        if self.break_torque is not None and self.break_torque <= 0:
            issues.append(f"Joint '{self.joint_id}': break_torque must be > 0 if set.")
        return issues


@dataclass
class ArticulationSpec:
    """Metadata for an articulation root (a kinematic tree of rigid links).

    Individual links are still added via :meth:`PhysicsScene.add_rigid_body`
    with ``actor_type=ActorType.ARTICULATION_LINK`` and
    ``articulation_id`` set to this spec's ``articulation_id`` -- this
    dataclass only records root-level articulation settings.

    Attributes:
        articulation_id: Unique identifier for this articulation.
        root_prim_path: USD prim path of the articulation root.
        fix_base: If ``True``, the root link is welded to the world
            (e.g. a stationary robot arm base).
        self_collision: If ``True``, links within this articulation can
            collide with each other.
        solver_position_iterations: Override of solver position
            iterations for this articulation, or ``None`` to inherit
            the scene default.
        solver_velocity_iterations: Override of solver velocity
            iterations for this articulation, or ``None`` to inherit
            the scene default.
    """

    articulation_id: str
    root_prim_path: str
    fix_base: bool = False
    self_collision: bool = False
    solver_position_iterations: Optional[int] = None
    solver_velocity_iterations: Optional[int] = None

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if not self.articulation_id:
            issues.append("ArticulationSpec.articulation_id must be non-empty.")
        if not self.root_prim_path:
            issues.append(f"Articulation '{self.articulation_id}': root_prim_path must be non-empty.")
        return issues


# ════════════════════════════════════════════════════════════════════════
# Scene-level configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsSceneConfig:
    """Scene-wide PhysX settings, independent of any single rigid body.

    This is intentionally a separate, smaller dataclass from
    ``omniverse.config.OmniverseConfig`` -- ``OmniverseConfig`` owns
    install-detection and process-launch concerns for the whole
    connector layer, while ``PhysicsSceneConfig`` owns only what a
    ``PhysicsScene`` needs to drive PhysX. Use
    :meth:`PhysicsSceneConfig.from_omniverse_config` to derive one from
    the other without this module importing ``config.py`` at runtime.

    Attributes:
        gravity: World-space gravity vector, in m/s^2 (default: Earth
            gravity, -Z up... callers on a Y-up stage should pass
            ``(0, -9.81, 0)``).
        physics_dt: Fixed simulation timestep, in seconds.
        solver_type: PGS or TGS.
        solver_position_iterations: Scene-default solver position
            iteration count (per-body/-articulation overrides win).
        solver_velocity_iterations: Scene-default solver velocity
            iteration count.
        broadphase_type: Broadphase algorithm.
        use_gpu_dynamics: Run rigid-body dynamics on the GPU.
        gpu_device: CUDA device index for GPU dynamics, or ``None`` to
            let PhysX pick.
        enable_ccd: Scene-wide default for continuous collision
            detection (individual bodies may still opt in/out).
        enable_stabilization: Enable TGS stabilization pass.
        enable_sleeping: Allow bodies to go to sleep when at rest.
        bounce_threshold_velocity: Minimum relative contact velocity
            for restitution to be applied (below this, contacts are
            treated as inelastic to avoid jitter).
        friction_offset_threshold: Distance at which friction
            anchors are established ahead of contact.
        default_material: Fallback material used by bodies with no
            ``material_name`` set.
        max_replay_snapshots: Maximum number of snapshots retained by
            the in-memory replay ring buffer.
    """

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    physics_dt: float = 1.0 / 60.0
    solver_type: SolverType = SolverType.TGS
    solver_position_iterations: int = _DEFAULT_SOLVER_POSITION_ITERATIONS
    solver_velocity_iterations: int = _DEFAULT_SOLVER_VELOCITY_ITERATIONS
    broadphase_type: BroadphaseType = BroadphaseType.SAP
    use_gpu_dynamics: bool = False
    gpu_device: Optional[int] = None
    enable_ccd: bool = False
    enable_stabilization: bool = True
    enable_sleeping: bool = True
    bounce_threshold_velocity: float = 0.2
    friction_offset_threshold: float = 0.04
    default_material: PhysicsMaterialSpec = field(
        default_factory=lambda: PhysicsMaterialSpec(name="__default__")
    )
    max_replay_snapshots: int = _DEFAULT_MAX_REPLAY_SNAPSHOTS

    def validate(self) -> list[str]:
        """Return a list of human-readable validation issues (empty if valid)."""
        issues: list[str] = []
        if len(self.gravity) != 3:
            issues.append("gravity must be a 3-tuple (x, y, z).")
        if not _MIN_PHYSICS_DT <= self.physics_dt <= _MAX_PHYSICS_DT:
            issues.append(
                f"physics_dt must be within [{_MIN_PHYSICS_DT}, {_MAX_PHYSICS_DT}] "
                f"(got {self.physics_dt})."
            )
        if self.solver_position_iterations < 1:
            issues.append("solver_position_iterations must be >= 1.")
        if self.solver_velocity_iterations < 1:
            issues.append("solver_velocity_iterations must be >= 1.")
        if self.broadphase_type == BroadphaseType.GPU and not self.use_gpu_dynamics:
            issues.append("broadphase_type='gpu' requires use_gpu_dynamics=True.")
        if self.use_gpu_dynamics and self.gpu_device is not None and self.gpu_device < 0:
            issues.append("gpu_device must be >= 0 if set.")
        if self.bounce_threshold_velocity < 0:
            issues.append("bounce_threshold_velocity must be >= 0.")
        if self.friction_offset_threshold < 0:
            issues.append("friction_offset_threshold must be >= 0.")
        if self.max_replay_snapshots < 1:
            issues.append("max_replay_snapshots must be >= 1.")
        issues.extend(self.default_material.validate())
        return issues

    @classmethod
    def from_omniverse_config(cls, config: "OmniverseConfig") -> "PhysicsSceneConfig":
        """Derive a ``PhysicsSceneConfig`` from a central ``OmniverseConfig``.

        Uses duck typing (``getattr`` with fallbacks) rather than an
        import of ``omniverse.config`` so this module has no hard
        dependency on it and remains importable/testable standalone.

        Args:
            config: A populated ``OmniverseConfig``-like object (must
                expose at minimum ``physics_dt`` and ``gpu_device``).

        Returns:
            A new :class:`PhysicsSceneConfig` seeded from ``config``.
        """
        physics_dt = getattr(config, "physics_dt", 1.0 / 60.0)
        gpu_device = getattr(config, "gpu_device", None)
        use_gpu = gpu_device is not None
        instance = cls(
            physics_dt=physics_dt,
            gpu_device=gpu_device,
            use_gpu_dynamics=use_gpu,
            broadphase_type=BroadphaseType.GPU if use_gpu else BroadphaseType.SAP,
        )
        logger.debug("Derived PhysicsSceneConfig from OmniverseConfig: %s", instance)
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (enums become their values)."""
        raw = asdict(self)
        return _stringify_enums(raw)


def _stringify_enums(value: Any) -> Any:
    """Recursively convert ``Enum`` members inside nested dict/list/tuple structures to their ``.value``."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _stringify_enums(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_enums(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(_stringify_enums(v) for v in value)
    return value


# ════════════════════════════════════════════════════════════════════════
# Statistics / snapshots
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SceneStatistics:
    """A point-in-time snapshot of scene diagnostics.

    Attributes:
        step_count: Number of :meth:`PhysicsScene.step` calls since the
            scene was created (or last :meth:`PhysicsScene.reset`).
        simulated_time: Total simulated time, in seconds
            (``step_count * physics_dt``).
        wall_time_in_step: Cumulative wall-clock seconds spent inside
            :meth:`PhysicsScene.step`, for perf monitoring.
        rigid_body_count: Number of rigid bodies currently tracked.
        articulation_count: Number of articulations currently tracked.
        joint_count: Number of joints currently tracked.
        material_count: Number of registered materials.
        awake_body_count: Bodies not currently asleep (best-effort;
            ``None`` if the backend can't report it, e.g. before a
            live scene exists).
        backend_mode: Whether dynamics are running on CPU or GPU.
        broadphase_type: Active broadphase algorithm.
        is_paused: Whether the scene is currently paused.
    """

    step_count: int = 0
    simulated_time: float = 0.0
    wall_time_in_step: float = 0.0
    rigid_body_count: int = 0
    articulation_count: int = 0
    joint_count: int = 0
    material_count: int = 0
    awake_body_count: Optional[int] = None
    backend_mode: PhysicsBackendMode = PhysicsBackendMode.CPU
    broadphase_type: BroadphaseType = BroadphaseType.SAP
    is_paused: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return _stringify_enums(asdict(self))


@dataclass
class SceneSnapshot:
    """A fully self-contained, picklable capture of scene state.

    Captures configuration plus every registered material, rigid body,
    articulation, and joint spec -- everything needed to reconstruct
    an equivalent (though not bit-for-bit, since live PhysX state such
    as velocities is not captured) scene via
    :meth:`PhysicsScene.deserialize`.

    Attributes:
        version: Schema version, for forward/backward compatibility.
        timestamp: Unix timestamp (seconds) the snapshot was taken.
        step_count: :attr:`SceneStatistics.step_count` at capture time.
        config: The scene configuration at capture time.
        materials: All registered materials, keyed by name.
        rigid_bodies: All rigid bodies, keyed by ``body_id``.
        articulations: All articulations, keyed by ``articulation_id``.
        joints: All joints, keyed by ``joint_id``.
    """

    version: str
    timestamp: float
    step_count: int
    config: PhysicsSceneConfig
    materials: dict[str, PhysicsMaterialSpec]
    rigid_bodies: dict[str, RigidBodySpec]
    articulations: dict[str, ArticulationSpec]
    joints: dict[str, JointSpec]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "step_count": self.step_count,
            "config": self.config.to_dict(),
            "materials": {k: _stringify_enums(asdict(v)) for k, v in self.materials.items()},
            "rigid_bodies": {k: _stringify_enums(asdict(v)) for k, v in self.rigid_bodies.items()},
            "articulations": {k: _stringify_enums(asdict(v)) for k, v in self.articulations.items()},
            "joints": {k: _stringify_enums(asdict(v)) for k, v in self.joints.items()},
        }


_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


# ════════════════════════════════════════════════════════════════════════
# PhysicsScene
# ════════════════════════════════════════════════════════════════════════

class PhysicsScene:
    """Owns and drives a single PhysX physics scene.

    Lifecycle::

        scene = PhysicsScene(config)
        scene.initialize()          # resolve the PhysX backend
        scene.create_scene(stage)   # create the UsdPhysics scene prim
        scene.add_rigid_body(...)
        scene.add_joint(...)
        scene.step()                 # advance simulation by physics_dt
        ...
        scene.destroy_scene()
        scene.shutdown()

    Or, equivalently, as a context manager (which calls
    :meth:`initialize` on entry and :meth:`shutdown` on exit)::

        with PhysicsScene(config) as scene:
            scene.create_scene(stage)
            ...

    Thread safety:
        Every public method that reads or mutates internal state
        acquires ``self._lock`` (a re-entrant lock), so a single
        ``PhysicsScene`` instance may safely be driven by a dedicated
        simulation thread while other threads call read-only methods
        such as :meth:`statistics` or :meth:`validate`.

    Attributes:
        config: The immutable-in-practice scene configuration this
            instance was constructed with (mutate via
            :meth:`set_gravity` / :meth:`set_material` rather than
            editing ``config`` directly once :meth:`create_scene` has
            been called).
    """

    def __init__(self, config: Optional[PhysicsSceneConfig] = None) -> None:
        """Construct a ``PhysicsScene``. Never touches Omniverse/PhysX.

        Args:
            config: Scene-wide settings. Defaults to
                ``PhysicsSceneConfig()`` (Earth gravity, CPU dynamics,
                60 Hz) if omitted.
        """
        self.config = config or PhysicsSceneConfig()

        self._lock = threading.RLock()
        self._initialized = False
        self._scene_created = False
        self._paused = False

        self._stage: Any = None
        self._scene_path: Optional[str] = None
        self._physx_sim_interface: Any = None
        self._physx_interface: Any = None

        self._materials: dict[str, PhysicsMaterialSpec] = {}
        self._rigid_bodies: dict[str, RigidBodySpec] = {}
        self._articulations: dict[str, ArticulationSpec] = {}
        self._joints: dict[str, JointSpec] = {}

        self._contact_report_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._trigger_callbacks: list[Callable[[dict[str, Any]], None]] = []

        self._step_count = 0
        self._simulated_time = 0.0
        self._wall_time_in_step = 0.0

        self._snapshot_recording_enabled = False
        self._replay_buffer: Deque[SceneSnapshot] = deque(maxlen=self.config.max_replay_snapshots)

        self._materials[self.config.default_material.name] = self.config.default_material

        logger.debug("Constructed PhysicsScene with config=%s", self.config)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PhysicsScene":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PhysicsScene(initialized={self._initialized}, "
            f"scene_created={self._scene_created}, "
            f"bodies={len(self._rigid_bodies)}, joints={len(self._joints)})"
        )

    # ------------------------------------------------------------------
    # Lifecycle: initialize / shutdown
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Resolve and cache the ``omni.physx`` backend interfaces.

        Idempotent -- calling this more than once is a no-op after the
        first successful call. Does not create a scene or touch a
        stage; that happens in :meth:`create_scene`.

        Raises:
            PhysicsBackendUnavailableError: If ``omni``, ``pxr``, or
                ``omni.physx`` cannot be imported.
        """
        with self._lock:
            if self._initialized:
                logger.debug("PhysicsScene.initialize() called on an already-initialized scene; no-op.")
                return

            physx = _require_physx()
            _require_pxr()

            try:
                self._physx_interface = physx.get_physx_interface()
                self._physx_sim_interface = physx.get_physx_simulation_interface()
            except AttributeError as exc:
                raise PhysicsBackendUnavailableError(
                    "omni.physx was imported but does not expose the expected "
                    "get_physx_interface()/get_physx_simulation_interface() API."
                ) from exc

            self._initialized = True
            logger.info("PhysicsScene initialized (backend=omni.physx).")

    def shutdown(self) -> None:
        """Tear down the scene (if any) and release backend interfaces.

        Idempotent -- safe to call on an already-shut-down or
        never-initialized scene.
        """
        with self._lock:
            if self._scene_created:
                try:
                    self.destroy_scene()
                except PhysicsSceneError:
                    logger.warning("destroy_scene() failed during shutdown(); continuing.", exc_info=True)

            self._physx_interface = None
            self._physx_sim_interface = None
            self._initialized = False
            logger.info("PhysicsScene shut down.")

    @property
    def is_initialized(self) -> bool:
        """Whether :meth:`initialize` has been successfully called."""
        return self._initialized

    # ------------------------------------------------------------------
    # Lifecycle: create_scene / destroy_scene
    # ------------------------------------------------------------------

    def create_scene(self, stage: Any, scene_path: str = "/World/PhysicsScene") -> None:
        """Create the UsdPhysics scene prim on an already-open stage.

        This method never opens, loads, or saves a ``Usd.Stage`` --
        ``stage`` must already be a live ``pxr.Usd.Stage`` handed in by
        the caller (typically ``stage_manager.py``).

        Args:
            stage: The live ``pxr.Usd.Stage`` to author the physics
                scene prim onto.
            scene_path: USD prim path for the new PhysicsScene prim.

        Raises:
            SceneNotInitializedError: If :meth:`initialize` has not
                been called yet.
            SceneAlreadyExistsError: If a scene has already been
                created on this instance.
            PhysicsBackendUnavailableError: If ``pxr`` is unavailable.
        """
        with self._lock:
            if not self._initialized:
                raise SceneNotInitializedError("Call initialize() before create_scene().")
            if self._scene_created:
                raise SceneAlreadyExistsError(
                    f"A physics scene already exists at '{self._scene_path}'; "
                    "call destroy_scene() first."
                )

            pxr = _require_pxr()
            UsdPhysics = pxr.UsdPhysics
            PhysxSchema = getattr(pxr, "PhysxSchema", None)

            usd_scene = UsdPhysics.Scene.Define(stage, scene_path)
            gx, gy, gz = self.config.gravity
            magnitude = (gx ** 2 + gy ** 2 + gz ** 2) ** 0.5
            direction = (gx / magnitude, gy / magnitude, gz / magnitude) if magnitude > 0 else (0.0, 0.0, -1.0)
            usd_scene.CreateGravityDirectionAttr().Set(direction)
            usd_scene.CreateGravityMagnitudeAttr().Set(magnitude)

            if PhysxSchema is not None:
                physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(usd_scene.GetPrim())
                physx_scene_api.CreateTimeStepsPerSecondAttr().Set(round(1.0 / self.config.physics_dt))
                physx_scene_api.CreateSolverTypeAttr().Set(self.config.solver_type.value.upper())
                physx_scene_api.CreateEnableCCDAttr().Set(self.config.enable_ccd)
                physx_scene_api.CreateEnableStabilizationAttr().Set(self.config.enable_stabilization)
                physx_scene_api.CreateEnableGPUDynamicsAttr().Set(self.config.use_gpu_dynamics)
                physx_scene_api.CreateBroadphaseTypeAttr().Set(self.config.broadphase_type.value.upper())
                physx_scene_api.CreateBounceThresholdAttr().Set(self.config.bounce_threshold_velocity)
                physx_scene_api.CreateFrictionOffsetThresholdAttr().Set(self.config.friction_offset_threshold)
            else:
                logger.warning(
                    "PhysxSchema not available on this 'pxr' build; scene created with "
                    "core UsdPhysics gravity only -- solver/CCD/GPU settings were not applied."
                )

            self._stage = stage
            self._scene_path = scene_path
            self._scene_created = True
            self._step_count = 0
            self._simulated_time = 0.0
            self._wall_time_in_step = 0.0

            logger.info(
                "Created PhysicsScene prim at '%s' (gravity=%s, dt=%s, solver=%s, gpu=%s).",
                scene_path, self.config.gravity, self.config.physics_dt,
                self.config.solver_type.value, self.config.use_gpu_dynamics,
            )

    def destroy_scene(self) -> None:
        """Remove the physics scene prim and clear all body/joint/material state.

        Does not close or unload the stage itself -- only the physics
        scene prim and this instance's bookkeeping.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
        """
        with self._lock:
            if not self._scene_created:
                raise SceneNotCreatedError("No active scene to destroy.")

            if self._stage is not None and self._scene_path is not None:
                try:
                    pxr = _require_pxr()
                    self._stage.RemovePrim(pxr.Sdf.Path(self._scene_path))
                except PhysicsBackendUnavailableError:
                    logger.warning("pxr unavailable while destroying scene; skipping prim removal.")
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to remove physics scene prim '%s'.", self._scene_path, exc_info=True)

            self._rigid_bodies.clear()
            self._articulations.clear()
            self._joints.clear()
            self._materials = {self.config.default_material.name: self.config.default_material}
            self._stage = None
            path = self._scene_path
            self._scene_path = None
            self._scene_created = False
            self._paused = False

            logger.info("Destroyed PhysicsScene prim at '%s'.", path)

    @property
    def is_scene_created(self) -> bool:
        """Whether :meth:`create_scene` has succeeded and :meth:`destroy_scene` has not since been called."""
        return self._scene_created

    def _require_scene(self) -> None:
        """Raise if no scene is currently active. Internal helper."""
        if not self._scene_created:
            raise SceneNotCreatedError("No active physics scene; call create_scene() first.")

    # ------------------------------------------------------------------
    # Stepping / reset / pause / resume
    # ------------------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> None:
        """Advance the simulation by one fixed timestep.

        Args:
            dt: Override for this single step's duration, in seconds.
                Defaults to ``self.config.physics_dt``. Providing a
                per-step override does not change
                ``self.config.physics_dt`` for subsequent steps.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
        """
        with self._lock:
            self._require_scene()
            if self._paused:
                logger.debug("step() called while paused; ignoring.")
                return

            step_dt = dt if dt is not None else self.config.physics_dt
            physx = _require_physx()

            start = time.perf_counter()
            try:
                if self._physx_sim_interface is not None:
                    self._physx_sim_interface.simulate(step_dt, self._simulated_time)
                    self._physx_sim_interface.fetch_results()
                else:  # pragma: no cover - defensive; initialize() always sets this
                    physx.get_physx_simulation_interface().simulate(step_dt, self._simulated_time)
            except Exception as exc:  # noqa: BLE001
                raise PhysicsSceneError(f"PhysX step failed: {exc}") from exc
            elapsed = time.perf_counter() - start

            self._step_count += 1
            self._simulated_time += step_dt
            self._wall_time_in_step += elapsed

            if self._snapshot_recording_enabled:
                self._record_snapshot_locked()

            logger.debug(
                "Stepped physics scene: step=%d dt=%.6f sim_time=%.4f wall_ms=%.3f",
                self._step_count, step_dt, self._simulated_time, elapsed * 1000.0,
            )

    def reset(self) -> None:
        """Reset simulation time/step counters and clear the replay buffer.

        Does not remove rigid bodies, joints, or materials -- only
        resets the stepping clock and (if the backend is live) asks
        PhysX to reset transforms/velocities to their authored USD
        state. Use :meth:`destroy_scene` followed by
        :meth:`create_scene` for a full teardown.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
        """
        with self._lock:
            self._require_scene()

            physx = _require_physx()
            reset_fn = getattr(self._physx_interface, "reset_simulation", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:  # noqa: BLE001
                    logger.warning("Backend reset_simulation() failed; counters were still reset.", exc_info=True)
            else:
                logger.debug("Backend has no reset_simulation(); only local counters were reset.")

            self._step_count = 0
            self._simulated_time = 0.0
            self._wall_time_in_step = 0.0
            self._replay_buffer.clear()
            self._paused = False

            logger.info("PhysicsScene reset.")

    def pause(self) -> None:
        """Pause stepping. Subsequent :meth:`step` calls become no-ops until :meth:`resume`."""
        with self._lock:
            self._require_scene()
            self._paused = True
            logger.info("PhysicsScene paused at step=%d.", self._step_count)

    def resume(self) -> None:
        """Resume stepping after :meth:`pause`."""
        with self._lock:
            self._require_scene()
            self._paused = False
            logger.info("PhysicsScene resumed at step=%d.", self._step_count)

    @property
    def is_paused(self) -> bool:
        """Whether the scene is currently paused."""
        return self._paused

    # ------------------------------------------------------------------
    # Gravity
    # ------------------------------------------------------------------

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Update world gravity, both locally and (if live) on the USD scene prim.

        Args:
            gravity: New gravity vector, in m/s^2.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
        """
        with self._lock:
            self._require_scene()
            self.config.gravity = gravity

            if self._stage is not None and self._scene_path is not None:
                pxr = _require_pxr()
                UsdPhysics = pxr.UsdPhysics
                usd_scene = UsdPhysics.Scene.Get(self._stage, self._scene_path)
                gx, gy, gz = gravity
                magnitude = (gx ** 2 + gy ** 2 + gz ** 2) ** 0.5
                direction = (gx / magnitude, gy / magnitude, gz / magnitude) if magnitude > 0 else (0.0, 0.0, -1.0)
                usd_scene.CreateGravityDirectionAttr().Set(direction)
                usd_scene.CreateGravityMagnitudeAttr().Set(magnitude)

            logger.info("Gravity set to %s.", gravity)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def set_material(self, name: str, material: Optional[PhysicsMaterialSpec] = None) -> PhysicsMaterialSpec:
        """Register or update a named physics material.

        Args:
            name: Material identifier.
            material: The material to store. If ``None``, a
                default-valued ``PhysicsMaterialSpec(name=name)`` is
                created. If provided, ``material.name`` is overwritten
                to match ``name`` so the registry key and the spec
                never disagree.

        Returns:
            The stored :class:`PhysicsMaterialSpec` (post-normalization).

        Raises:
            MaterialError: If the material fails validation.
        """
        with self._lock:
            spec = material or PhysicsMaterialSpec(name=name)
            spec.name = name
            issues = spec.validate()
            if issues:
                raise MaterialError("; ".join(issues))

            self._materials[name] = spec
            logger.info("Registered material '%s': %s", name, spec)
            return spec

    def get_material(self, name: str) -> PhysicsMaterialSpec:
        """Look up a previously registered material by name.

        Raises:
            MaterialError: If no material with that name is registered.
        """
        with self._lock:
            try:
                return self._materials[name]
            except KeyError as exc:
                raise MaterialError(f"No material registered under name '{name}'.") from exc

    # ------------------------------------------------------------------
    # Rigid bodies
    # ------------------------------------------------------------------

    def add_rigid_body(self, spec: RigidBodySpec) -> None:
        """Add a rigid body (or articulation link) to the scene.

        Args:
            spec: The rigid body to add.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
            RigidBodyError: If ``spec`` fails validation, a body with
                the same ``body_id`` already exists, ``spec`` references
                an unregistered ``material_name``, or (for articulation
                links) an unregistered ``articulation_id``.
        """
        with self._lock:
            self._require_scene()

            issues = spec.validate()
            if spec.body_id in self._rigid_bodies:
                issues.append(f"Body '{spec.body_id}' already exists; remove it first or choose a new id.")
            if spec.material_name is not None and spec.material_name not in self._materials:
                issues.append(f"Body '{spec.body_id}' references unknown material '{spec.material_name}'.")
            if spec.actor_type == ActorType.ARTICULATION_LINK and spec.articulation_id not in self._articulations:
                issues.append(
                    f"Body '{spec.body_id}' references unknown articulation "
                    f"'{spec.articulation_id}'; call add_articulation() first."
                )
            if issues:
                raise RigidBodyError("; ".join(issues))

            self._apply_rigid_body_to_stage(spec)
            self._rigid_bodies[spec.body_id] = spec
            logger.info("Added rigid body '%s' (%s) at '%s'.", spec.body_id, spec.actor_type.value, spec.prim_path)

    def _apply_rigid_body_to_stage(self, spec: RigidBodySpec) -> None:
        """Author the USD/PhysX schema APIs for one rigid body. Internal helper."""
        if self._stage is None:
            logger.debug("No live stage attached; recorded rigid body '%s' without authoring USD.", spec.body_id)
            return

        pxr = _require_pxr()
        UsdPhysics = pxr.UsdPhysics
        PhysxSchema = getattr(pxr, "PhysxSchema", None)

        prim = self._stage.GetPrimAtPath(spec.prim_path)
        if not prim or not prim.IsValid():
            raise RigidBodyError(f"Prim path '{spec.prim_path}' does not exist on the current stage.")

        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr().Set(spec.actor_type != ActorType.STATIC)
        rigid_body_api.CreateKinematicEnabledAttr().Set(spec.actor_type == ActorType.KINEMATIC)

        if spec.actor_type != ActorType.STATIC:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(spec.mass)

        UsdPhysics.CollisionAPI.Apply(prim)

        if PhysxSchema is not None:
            rigid_body_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_body_physx_api.CreateEnableCCDAttr().Set(spec.enable_ccd or self.config.enable_ccd)
            rigid_body_physx_api.CreateSleepThresholdAttr().Set(spec.sleep_threshold)
            rigid_body_physx_api.CreateStabilizationThresholdAttr().Set(spec.stabilization_threshold)
            rigid_body_physx_api.CreateEnableGyroscopicForcesAttr().Set(True)
            rigid_body_physx_api.CreateSolverPositionIterationCountAttr().Set(
                spec.solver_position_iterations or self.config.solver_position_iterations
            )
            rigid_body_physx_api.CreateSolverVelocityIterationCountAttr().Set(
                spec.solver_velocity_iterations or self.config.solver_velocity_iterations
            )

            if spec.contact_report_enabled:
                contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                contact_report_api.CreateThresholdAttr().Set(0.0)

            if spec.is_trigger:
                PhysxSchema.PhysxTriggerAPI.Apply(prim)

        if spec.material_name is not None:
            material = self._materials[spec.material_name]
            material_api = UsdPhysics.MaterialAPI.Apply(prim)
            material_api.CreateStaticFrictionAttr().Set(material.static_friction)
            material_api.CreateDynamicFrictionAttr().Set(material.dynamic_friction)
            material_api.CreateRestitutionAttr().Set(material.restitution)

    def remove_rigid_body(self, body_id: str) -> None:
        """Remove a previously added rigid body from the scene.

        Any joints still referencing ``body_id`` are left in place but
        will fail :meth:`validate`; remove dependent joints first for a
        clean scene graph.

        Raises:
            RigidBodyError: If no body with that ``body_id`` exists.
        """
        with self._lock:
            if body_id not in self._rigid_bodies:
                raise RigidBodyError(f"No rigid body registered under id '{body_id}'.")

            spec = self._rigid_bodies.pop(body_id)
            if self._stage is not None:
                try:
                    pxr = _require_pxr()
                    UsdPhysics = pxr.UsdPhysics
                    prim = self._stage.GetPrimAtPath(spec.prim_path)
                    if prim and prim.IsValid():
                        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(False)
                except PhysicsBackendUnavailableError:
                    logger.warning("pxr unavailable while removing rigid body '%s'.", body_id)

            logger.info("Removed rigid body '%s'.", body_id)

    def get_rigid_body(self, body_id: str) -> RigidBodySpec:
        """Look up a previously added rigid body by id.

        Raises:
            RigidBodyError: If no body with that id exists.
        """
        with self._lock:
            try:
                return self._rigid_bodies[body_id]
            except KeyError as exc:
                raise RigidBodyError(f"No rigid body registered under id '{body_id}'.") from exc

    @property
    def rigid_body_count(self) -> int:
        """Number of rigid bodies currently tracked."""
        return len(self._rigid_bodies)

    # ------------------------------------------------------------------
    # Articulations
    # ------------------------------------------------------------------

    def add_articulation(self, spec: ArticulationSpec) -> None:
        """Register an articulation root.

        Must be called before adding any ``ARTICULATION_LINK`` rigid
        bodies that reference ``spec.articulation_id``.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
            RigidBodyError: If ``spec`` fails validation or its id is
                already registered.
        """
        with self._lock:
            self._require_scene()
            issues = spec.validate()
            if spec.articulation_id in self._articulations:
                issues.append(f"Articulation '{spec.articulation_id}' already exists.")
            if issues:
                raise RigidBodyError("; ".join(issues))

            if self._stage is not None:
                pxr = _require_pxr()
                UsdPhysics = pxr.UsdPhysics
                PhysxSchema = getattr(pxr, "PhysxSchema", None)
                prim = self._stage.GetPrimAtPath(spec.root_prim_path)
                if not prim or not prim.IsValid():
                    raise RigidBodyError(f"Articulation root prim '{spec.root_prim_path}' does not exist.")
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                if PhysxSchema is not None:
                    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                    articulation_api.CreateEnabledSelfCollisionsAttr().Set(spec.self_collision)
                    articulation_api.CreateFixBaseAttr().Set(spec.fix_base)
                    articulation_api.CreateSolverPositionIterationCountAttr().Set(
                        spec.solver_position_iterations or self.config.solver_position_iterations
                    )
                    articulation_api.CreateSolverVelocityIterationCountAttr().Set(
                        spec.solver_velocity_iterations or self.config.solver_velocity_iterations
                    )

            self._articulations[spec.articulation_id] = spec
            logger.info("Added articulation '%s' rooted at '%s'.", spec.articulation_id, spec.root_prim_path)

    def remove_articulation(self, articulation_id: str) -> None:
        """Remove an articulation root registration.

        Does not remove the link bodies themselves -- remove those
        individually via :meth:`remove_rigid_body` first.

        Raises:
            RigidBodyError: If no articulation with that id exists, or
                if link bodies still reference it.
        """
        with self._lock:
            if articulation_id not in self._articulations:
                raise RigidBodyError(f"No articulation registered under id '{articulation_id}'.")

            dependents = [
                b.body_id for b in self._rigid_bodies.values() if b.articulation_id == articulation_id
            ]
            if dependents:
                raise RigidBodyError(
                    f"Cannot remove articulation '{articulation_id}': still referenced by "
                    f"link bodies {dependents}. Remove those first."
                )

            self._articulations.pop(articulation_id)
            logger.info("Removed articulation '%s'.", articulation_id)

    @property
    def articulation_count(self) -> int:
        """Number of articulations currently tracked."""
        return len(self._articulations)

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    def add_joint(self, spec: JointSpec) -> None:
        """Add a joint/constraint connecting one or two rigid bodies.

        Args:
            spec: The joint to add.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
            JointError: If ``spec`` fails validation, a joint with the
                same id already exists, or it references unregistered
                body ids.
        """
        with self._lock:
            self._require_scene()

            issues = spec.validate()
            if spec.joint_id in self._joints:
                issues.append(f"Joint '{spec.joint_id}' already exists.")
            if spec.body0_id not in self._rigid_bodies:
                issues.append(f"Joint '{spec.joint_id}' references unknown body0_id '{spec.body0_id}'.")
            if spec.body1_id is not None and spec.body1_id not in self._rigid_bodies:
                issues.append(f"Joint '{spec.joint_id}' references unknown body1_id '{spec.body1_id}'.")
            if issues:
                raise JointError("; ".join(issues))

            self._apply_joint_to_stage(spec)
            self._joints[spec.joint_id] = spec
            logger.info(
                "Added %s joint '%s' between '%s' and '%s'.",
                spec.joint_type.value, spec.joint_id, spec.body0_id, spec.body1_id or "<world>",
            )

    def _apply_joint_to_stage(self, spec: JointSpec) -> None:
        """Author the USD/PhysX joint schema for one joint. Internal helper."""
        if self._stage is None:
            logger.debug("No live stage attached; recorded joint '%s' without authoring USD.", spec.joint_id)
            return

        pxr = _require_pxr()
        UsdPhysics = pxr.UsdPhysics

        joint_path = f"{self._scene_path}/joints/{spec.joint_id}"
        joint_type_map = {
            JointType.FIXED: UsdPhysics.FixedJoint,
            JointType.REVOLUTE: UsdPhysics.RevoluteJoint,
            JointType.PRISMATIC: UsdPhysics.PrismaticJoint,
            JointType.SPHERICAL: UsdPhysics.SphericalJoint,
            JointType.DISTANCE: UsdPhysics.DistanceJoint,
            JointType.D6: UsdPhysics.Joint,
        }
        joint_class = joint_type_map[spec.joint_type]
        usd_joint = joint_class.Define(self._stage, joint_path)

        body0 = self._rigid_bodies[spec.body0_id]
        usd_joint.CreateBody0Rel().SetTargets([pxr.Sdf.Path(body0.prim_path)])
        if spec.body1_id is not None:
            body1 = self._rigid_bodies[spec.body1_id]
            usd_joint.CreateBody1Rel().SetTargets([pxr.Sdf.Path(body1.prim_path)])

        usd_joint.CreateLocalPos0Attr().Set(spec.local_pos0)
        usd_joint.CreateLocalPos1Attr().Set(spec.local_pos1)
        usd_joint.CreateCollisionEnabledAttr().Set(spec.enable_collision)

        if spec.break_force is not None:
            usd_joint.CreateBreakForceAttr().Set(spec.break_force)
        if spec.break_torque is not None:
            usd_joint.CreateBreakTorqueAttr().Set(spec.break_torque)

        if spec.limit is not None and hasattr(usd_joint, "CreateLowerLimitAttr"):
            usd_joint.CreateLowerLimitAttr().Set(spec.limit.lower)
            usd_joint.CreateUpperLimitAttr().Set(spec.limit.upper)

        if spec.drive is not None:
            drive_api = UsdPhysics.DriveAPI.Apply(
                usd_joint.GetPrim(), "angular" if spec.joint_type == JointType.REVOLUTE else "linear"
            )
            drive_api.CreateTargetPositionAttr().Set(spec.drive.target_position)
            drive_api.CreateTargetVelocityAttr().Set(spec.drive.target_velocity)
            drive_api.CreateStiffnessAttr().Set(spec.drive.stiffness)
            drive_api.CreateDampingAttr().Set(spec.drive.damping)
            if spec.drive.max_force is not None:
                drive_api.CreateMaxForceAttr().Set(spec.drive.max_force)

    def remove_joint(self, joint_id: str) -> None:
        """Remove a previously added joint.

        Raises:
            JointError: If no joint with that id exists.
        """
        with self._lock:
            if joint_id not in self._joints:
                raise JointError(f"No joint registered under id '{joint_id}'.")

            self._joints.pop(joint_id)
            if self._stage is not None:
                try:
                    pxr = _require_pxr()
                    self._stage.RemovePrim(pxr.Sdf.Path(f"{self._scene_path}/joints/{joint_id}"))
                except PhysicsBackendUnavailableError:
                    logger.warning("pxr unavailable while removing joint '%s'.", joint_id)

            logger.info("Removed joint '%s'.", joint_id)

    def get_joint(self, joint_id: str) -> JointSpec:
        """Look up a previously added joint by id.

        Raises:
            JointError: If no joint with that id exists.
        """
        with self._lock:
            try:
                return self._joints[joint_id]
            except KeyError as exc:
                raise JointError(f"No joint registered under id '{joint_id}'.") from exc

    @property
    def joint_count(self) -> int:
        """Number of joints currently tracked."""
        return len(self._joints)

    # ------------------------------------------------------------------
    # CCD
    # ------------------------------------------------------------------

    def enable_ccd(self, body_id: Optional[str] = None) -> None:
        """Enable continuous collision detection scene-wide or for one body.

        Args:
            body_id: If given, enables CCD only for that body. If
                ``None``, enables CCD as the scene-wide default
                (``PhysicsSceneConfig.enable_ccd``); already-added
                bodies are not retroactively changed unless they also
                had ``enable_ccd=False`` purely by inheriting the old
                scene default -- for guaranteed effect on existing
                bodies, pass explicit ``body_id`` values.

        Raises:
            RigidBodyError: If ``body_id`` is given but not found.
        """
        with self._lock:
            if body_id is None:
                self.config.enable_ccd = True
                logger.info("Enabled scene-wide default CCD.")
                return

            spec = self.get_rigid_body(body_id)
            spec.enable_ccd = True
            self._apply_rigid_body_to_stage(spec)
            logger.info("Enabled CCD for body '%s'.", body_id)

    def disable_ccd(self, body_id: Optional[str] = None) -> None:
        """Disable continuous collision detection scene-wide or for one body.

        See :meth:`enable_ccd` for argument semantics (mirrored).

        Raises:
            RigidBodyError: If ``body_id`` is given but not found.
        """
        with self._lock:
            if body_id is None:
                self.config.enable_ccd = False
                logger.info("Disabled scene-wide default CCD.")
                return

            spec = self.get_rigid_body(body_id)
            spec.enable_ccd = False
            self._apply_rigid_body_to_stage(spec)
            logger.info("Disabled CCD for body '%s'.", body_id)

    # ------------------------------------------------------------------
    # Contact reports / triggers
    # ------------------------------------------------------------------

    def register_contact_report_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked with contact-event payloads.

        Args:
            callback: Called with a plain ``dict`` describing a contact
                event (actor ids, contact point, impulse, etc.) whenever
                the backend reports one. The exact payload shape is
                backend-dependent; callers should treat unknown keys as
                forward-compatible extras.
        """
        with self._lock:
            self._contact_report_callbacks.append(callback)
            logger.debug("Registered contact-report callback (total=%d).", len(self._contact_report_callbacks))

    def unregister_contact_report_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered contact-report callback.

        Raises:
            PhysicsSceneError: If ``callback`` was never registered.
        """
        with self._lock:
            try:
                self._contact_report_callbacks.remove(callback)
            except ValueError as exc:
                raise PhysicsSceneError("Callback was not registered.") from exc

    def register_trigger_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked with trigger enter/leave event payloads.

        Args:
            callback: Called with a plain ``dict`` describing a trigger
                event whenever a body flagged ``is_trigger=True`` is
                entered or exited by another collider.
        """
        with self._lock:
            self._trigger_callbacks.append(callback)
            logger.debug("Registered trigger callback (total=%d).", len(self._trigger_callbacks))

    def unregister_trigger_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered trigger callback.

        Raises:
            PhysicsSceneError: If ``callback`` was never registered.
        """
        with self._lock:
            try:
                self._trigger_callbacks.remove(callback)
            except ValueError as exc:
                raise PhysicsSceneError("Callback was not registered.") from exc

    def _dispatch_contact_report(self, payload: dict[str, Any]) -> None:
        """Fan a contact event out to all registered callbacks. Internal helper."""
        for callback in list(self._contact_report_callbacks):
            try:
                callback(payload)
            except Exception:  # noqa: BLE001
                logger.warning("Contact-report callback raised.", exc_info=True)

    def _dispatch_trigger_event(self, payload: dict[str, Any]) -> None:
        """Fan a trigger event out to all registered callbacks. Internal helper."""
        for callback in list(self._trigger_callbacks):
            try:
                callback(payload)
            except Exception:  # noqa: BLE001
                logger.warning("Trigger callback raised.", exc_info=True)

    # ------------------------------------------------------------------
    # Statistics / diagnostics / validation
    # ------------------------------------------------------------------

    def statistics(self) -> SceneStatistics:
        """Return a point-in-time snapshot of scene diagnostics.

        Never raises for an uncreated scene -- returns zeroed-out
        counters instead, so monitoring code can poll unconditionally.
        """
        with self._lock:
            awake_count: Optional[int] = None
            if self._physx_sim_interface is not None:
                getter = getattr(self._physx_sim_interface, "get_awake_actor_count", None)
                if callable(getter):
                    try:
                        awake_count = int(getter())
                    except Exception:  # noqa: BLE001
                        logger.debug("Backend get_awake_actor_count() failed; omitting from statistics.")

            return SceneStatistics(
                step_count=self._step_count,
                simulated_time=self._simulated_time,
                wall_time_in_step=self._wall_time_in_step,
                rigid_body_count=len(self._rigid_bodies),
                articulation_count=len(self._articulations),
                joint_count=len(self._joints),
                material_count=len(self._materials),
                awake_body_count=awake_count,
                backend_mode=PhysicsBackendMode.GPU if self.config.use_gpu_dynamics else PhysicsBackendMode.CPU,
                broadphase_type=self.config.broadphase_type,
                is_paused=self._paused,
            )

    def validate(self) -> None:
        """Validate the full in-memory scene graph in a single pass.

        Checks configuration, every registered material/body/
        articulation/joint, and cross-references between them (e.g.
        joints pointing at bodies that still exist, bodies pointing at
        materials/articulations that still exist).

        Raises:
            PhysicsSceneValidationError: Aggregating every issue found,
                joined with ``"; "``.
        """
        with self._lock:
            issues: list[str] = list(self.config.validate())

            for material in self._materials.values():
                issues.extend(material.validate())

            for body in self._rigid_bodies.values():
                issues.extend(body.validate())
                if body.material_name is not None and body.material_name not in self._materials:
                    issues.append(f"Body '{body.body_id}' references missing material '{body.material_name}'.")
                if body.actor_type == ActorType.ARTICULATION_LINK and body.articulation_id not in self._articulations:
                    issues.append(
                        f"Body '{body.body_id}' references missing articulation '{body.articulation_id}'."
                    )

            for articulation in self._articulations.values():
                issues.extend(articulation.validate())

            for joint in self._joints.values():
                issues.extend(joint.validate())
                if joint.body0_id not in self._rigid_bodies:
                    issues.append(f"Joint '{joint.joint_id}' references missing body0 '{joint.body0_id}'.")
                if joint.body1_id is not None and joint.body1_id not in self._rigid_bodies:
                    issues.append(f"Joint '{joint.joint_id}' references missing body1 '{joint.body1_id}'.")

            if issues:
                raise PhysicsSceneValidationError("; ".join(issues))

            logger.info(
                "PhysicsScene validated successfully (bodies=%d, joints=%d, articulations=%d, materials=%d).",
                len(self._rigid_bodies), len(self._joints), len(self._articulations), len(self._materials),
            )

    def diagnose(self) -> list[str]:
        """Non-raising variant of :meth:`validate`.

        Returns:
            A list of human-readable issues (empty if the scene is
            valid). Useful for UI/logging code that wants a full report
            without exception-handling boilerplate.
        """
        with self._lock:
            try:
                self.validate()
                return []
            except PhysicsSceneValidationError as exc:
                return str(exc).split("; ")

    # ------------------------------------------------------------------
    # Serialization / snapshots / replay
    # ------------------------------------------------------------------

    def serialize(self) -> SceneSnapshot:
        """Capture a picklable, JSON-serializable snapshot of scene state.

        Captures configuration and every registered material/body/
        articulation/joint spec. Does not capture live PhysX state
        such as current velocities or contact points.

        Returns:
            A new :class:`SceneSnapshot`.
        """
        with self._lock:
            snapshot = SceneSnapshot(
                version=_SNAPSHOT_SCHEMA_VERSION,
                timestamp=time.time(),
                step_count=self._step_count,
                config=copy.deepcopy(self.config),
                materials=copy.deepcopy(self._materials),
                rigid_bodies=copy.deepcopy(self._rigid_bodies),
                articulations=copy.deepcopy(self._articulations),
                joints=copy.deepcopy(self._joints),
            )
            logger.info("Serialized PhysicsScene snapshot at step=%d.", self._step_count)
            return snapshot

    def deserialize(self, snapshot: SceneSnapshot) -> None:
        """Restore scene state (config + all specs) from a snapshot.

        If a live stage/scene is already active, existing bodies/
        joints/materials are replaced by the snapshot's contents and
        re-authored onto the current stage. Requires
        :meth:`create_scene` to already have been called.

        Args:
            snapshot: A snapshot previously produced by :meth:`serialize`.

        Raises:
            SceneNotCreatedError: If no scene is currently active.
            SnapshotError: If the snapshot's schema version is
                incompatible.
        """
        with self._lock:
            self._require_scene()

            if snapshot.version.split(".")[0] != _SNAPSHOT_SCHEMA_VERSION.split(".")[0]:
                raise SnapshotError(
                    f"Snapshot schema version '{snapshot.version}' is incompatible with "
                    f"this module's version '{_SNAPSHOT_SCHEMA_VERSION}' (major version mismatch)."
                )

            self.config = copy.deepcopy(snapshot.config)
            self._materials = copy.deepcopy(snapshot.materials)
            self._articulations = copy.deepcopy(snapshot.articulations)
            self._rigid_bodies = {}
            self._joints = {}

            for articulation in snapshot.articulations.values():
                if self._stage is not None:
                    try:
                        self.add_articulation(copy.deepcopy(articulation))
                        self._articulations.pop(articulation.articulation_id, None)
                    except RigidBodyError:
                        logger.warning("Failed to re-author articulation '%s' on restore.", articulation.articulation_id)
                self._articulations[articulation.articulation_id] = articulation

            for body in snapshot.rigid_bodies.values():
                self._rigid_bodies[body.body_id] = copy.deepcopy(body)
                if self._stage is not None:
                    try:
                        self._apply_rigid_body_to_stage(self._rigid_bodies[body.body_id])
                    except RigidBodyError:
                        logger.warning("Failed to re-author rigid body '%s' on restore.", body.body_id)

            for joint in snapshot.joints.values():
                self._joints[joint.joint_id] = copy.deepcopy(joint)
                if self._stage is not None:
                    try:
                        self._apply_joint_to_stage(self._joints[joint.joint_id])
                    except (JointError, KeyError):
                        logger.warning("Failed to re-author joint '%s' on restore.", joint.joint_id)

            self._step_count = snapshot.step_count
            logger.info("Deserialized PhysicsScene snapshot from step=%d.", snapshot.step_count)

    def save_snapshot(self, path: "str | Path") -> Path:
        """Serialize the current scene state and write it to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The resolved output path.

        Raises:
            SnapshotError: If the file cannot be written.
        """
        out_path = Path(path)
        snapshot = self.serialize()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        except OSError as exc:
            raise SnapshotError(f"Failed to write snapshot to '{out_path}': {exc}") from exc
        logger.info("Saved scene snapshot to '%s'.", out_path)
        return out_path

    def enable_snapshot_recording(self) -> None:
        """Begin recording a snapshot into the replay ring buffer after every :meth:`step`."""
        with self._lock:
            self._snapshot_recording_enabled = True
            logger.info("Enabled per-step snapshot recording (max=%d).", self.config.max_replay_snapshots)

    def disable_snapshot_recording(self) -> None:
        """Stop recording snapshots after every :meth:`step` (existing buffer is retained)."""
        with self._lock:
            self._snapshot_recording_enabled = False
            logger.info("Disabled per-step snapshot recording.")

    def _record_snapshot_locked(self) -> None:
        """Append a snapshot to the replay buffer. Caller must already hold ``self._lock``."""
        self._replay_buffer.append(self.serialize())

    @property
    def replay_length(self) -> int:
        """Number of snapshots currently held in the replay buffer."""
        return len(self._replay_buffer)

    def replay_at(self, index: int) -> SceneSnapshot:
        """Return the snapshot at ``index`` in the replay buffer without restoring it.

        Args:
            index: Position in the buffer (``0`` is the oldest retained
                snapshot, ``-1`` the most recent).

        Raises:
            SnapshotError: If the replay buffer is empty or ``index``
                is out of range.
        """
        with self._lock:
            if not self._replay_buffer:
                raise SnapshotError("Replay buffer is empty; call enable_snapshot_recording() and step() first.")
            try:
                return self._replay_buffer[index]
            except IndexError as exc:
                raise SnapshotError(
                    f"Replay index {index} out of range for buffer of length {len(self._replay_buffer)}."
                ) from exc

    def replay_restore(self, index: int) -> None:
        """Restore the scene to the state captured at ``index`` in the replay buffer.

        Args:
            index: Position in the buffer, as in :meth:`replay_at`.

        Raises:
            SnapshotError: If the replay buffer is empty or ``index``
                is out of range.
            SceneNotCreatedError: If no scene is currently active.
        """
        snapshot = self.replay_at(index)
        self.deserialize(snapshot)
        logger.info("Restored scene from replay index %d.", index)


__all__ = [
    "PhysicsScene",
    "PhysicsSceneConfig",
    "PhysicsMaterialSpec",
    "CollisionFilterSpec",
    "RigidBodySpec",
    "JointLimitSpec",
    "JointDriveSpec",
    "JointSpec",
    "ArticulationSpec",
    "SceneStatistics",
    "SceneSnapshot",
    "SolverType",
    "BroadphaseType",
    "CombineMode",
    "JointType",
    "ActorType",
    "PhysicsBackendMode",
    "PhysicsSceneError",
    "PhysicsBackendUnavailableError",
    "SceneNotInitializedError",
    "SceneAlreadyExistsError",
    "SceneNotCreatedError",
    "RigidBodyError",
    "JointError",
    "MaterialError",
    "PhysicsSceneValidationError",
    "SnapshotError",
]
