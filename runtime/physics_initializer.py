"""
runtime/physics_initializer.py
══════════════════════════════════════════════════════════════════════════
PhysX bring-up for PhysWorldLM's Omniverse Runtime.

Pipeline position
------------------
    EntitySpawner.spawn_entities()
            │
            ▼
    ┌──────────────────────┐
    │ PHYSICS INITIALIZER   │   <-- this module
    └──────────────────────┘
            │
            ▼
      OmniverseRuntime.discover_entities()

Scope
-----
This module is a *one-shot builder* invoked once per run, after entities
have been spawned and before they are discovered/classified by
`OmniverseRuntime`. It does not run per-frame and it deliberately does
not duplicate `OmniverseRuntime.initialize_physics()` / `RuntimeConfig`
(gravity, substeps) -- it *consumes* that existing configuration and
performs the more detailed scene-level PhysX setup (rigid bodies,
collision, vehicle physics, projectile CCD, constraints, ground plane)
that `OmniverseRuntime` intentionally leaves out of scope.

No guidance, targeting, or lethality modeling is implemented or implied
here -- "projectile" support means giving fast-moving rigid bodies
continuous collision detection (CCD) so they don't tunnel through thin
geometry, nothing more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("physworldlm.physics_initializer")
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
    """Raised when the PhysX scene cannot be configured."""


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsInitializerConfig:
    """Scene-level PhysX tuning, layered on top of `RuntimeConfig`.

    Attributes:
        gravity_ms2: Gravitational acceleration magnitude, m/s^2. Should
            normally be sourced from `RuntimeConfig.gravity_ms2` so the
            two stay consistent.
        solver_position_iterations: PhysX solver position iteration count.
        solver_velocity_iterations: PhysX solver velocity iteration count.
        enable_ccd: Whether continuous collision detection is enabled
            scene-wide (recommended when fast-moving / "projectile"
            dynamic bodies are present).
        enable_gpu_dynamics: Whether GPU-accelerated rigid body dynamics
            are requested (falls back to CPU silently if unavailable).
        ground_plane_enabled: Whether a PhysX ground-plane collider is
            added beneath the authored terrain.
        ground_plane_height_m: Height (along the up-axis) of the ground
            plane collider.
        vehicle_physics_enabled: Whether the PhysX vehicle extension is
            initialized for wheeled/tracked ground vehicles.
        projectile_speed_threshold_ms: Dynamic bodies with a velocity
            magnitude at/above this threshold are tagged for CCD/per-body
            sub-stepping in `enable_projectiles()`.
    """

    gravity_ms2: float = 9.81
    solver_position_iterations: int = 4
    solver_velocity_iterations: int = 1
    enable_ccd: bool = True
    enable_gpu_dynamics: bool = False
    ground_plane_enabled: bool = True
    ground_plane_height_m: float = 0.0
    vehicle_physics_enabled: bool = True
    projectile_speed_threshold_ms: float = 50.0


@dataclass
class PhysicsSceneReport:
    """Summary of what was configured on the PhysX scene."""

    rigid_body_count: int = 0
    static_collider_count: int = 0
    vehicle_count: int = 0
    projectile_candidate_count: int = 0
    constraint_count: int = 0
    ground_plane_added: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rigid_body_count": self.rigid_body_count,
            "static_collider_count": self.static_collider_count,
            "vehicle_count": self.vehicle_count,
            "projectile_candidate_count": self.projectile_candidate_count,
            "constraint_count": self.constraint_count,
            "ground_plane_added": self.ground_plane_added,
            "warnings": list(self.warnings),
        }


# ════════════════════════════════════════════════════════════════════════
# PhysicsInitializer
# ════════════════════════════════════════════════════════════════════════

class PhysicsInitializer:
    """Configures the PhysX scene for a loaded, entity-populated USD stage.

    Example:
        >>> initializer = PhysicsInitializer(stage, registry, config)
        >>> report = initializer.initialize()
        >>> report.rigid_body_count
        12
    """

    def __init__(
        self,
        stage: Any,
        registry: Any,
        config: Optional[PhysicsInitializerConfig] = None,
    ) -> None:
        """Initialize the physics initializer.

        Args:
            stage: An open USD stage handle.
            registry: An `omniverse_runtime.EntityRegistry`-compatible
                object exposing `.entities` (mapping of prim_path ->
                object with `.is_static` / `.metadata`). Passing an
                already-populated registry lets this module reuse
                `OmniverseRuntime`'s discovery results instead of
                re-walking the stage.
            config: Scene-level PhysX tuning. Defaults to
                `PhysicsInitializerConfig()`.
        """
        self._stage = stage
        self._registry = registry
        self._config = config or PhysicsInitializerConfig()
        self._pxr_available = self._detect_pxr()
        self._report = PhysicsSceneReport()

    @staticmethod
    def _detect_pxr() -> bool:
        try:
            import pxr  # noqa: F401

            return True
        except ImportError:
            return False

    # ── orchestration ────────────────────────────────────────────────

    def initialize(self) -> PhysicsSceneReport:
        """Run the full PhysX scene bring-up sequence.

        Order: configure_scene → add_rigid_bodies → enable_collision →
        enable_vehicle_physics → enable_projectiles.

        Returns:
            A `PhysicsSceneReport` summarizing the configured scene.
        """
        logger.info("Initializing PhysX scene")
        self.configure_scene()
        self.add_rigid_bodies()
        self.enable_collision()
        if self._config.vehicle_physics_enabled:
            self.enable_vehicle_physics()
        self.enable_projectiles()
        logger.info(
            "PhysX scene ready: %d rigid bodies, %d static colliders, %d vehicles, %d projectile candidate(s)",
            self._report.rigid_body_count,
            self._report.static_collider_count,
            self._report.vehicle_count,
            self._report.projectile_candidate_count,
        )
        return self._report

    # ── individual steps ─────────────────────────────────────────────

    def configure_scene(self) -> None:
        """Create the PhysX scene prim and apply solver/gravity/CCD settings."""
        path = "/World/PhysicsScene"
        attrs = {
            "gravity_ms2": self._config.gravity_ms2,
            "solver_position_iterations": self._config.solver_position_iterations,
            "solver_velocity_iterations": self._config.solver_velocity_iterations,
            "enable_ccd": self._config.enable_ccd,
            "enable_gpu_dynamics": self._config.enable_gpu_dynamics,
        }
        self._define_prim(path, "PhysicsScene", attrs)
        if self._config.ground_plane_enabled:
            self._add_ground_plane()
        logger.info("PhysX scene configured at %s (gravity=%.3f m/s^2)", path, self._config.gravity_ms2)

    def add_rigid_bodies(self) -> None:
        """Apply rigid-body APIs to every dynamic entity in the registry."""
        for prim_path, entity in self._iter_entities():
            if self._is_static(entity):
                continue
            self._apply_rigid_body(prim_path, dynamic=True)
            self._report.rigid_body_count += 1
        logger.info("Applied rigid-body physics to %d dynamic entit(y/ies).", self._report.rigid_body_count)

    def enable_collision(self) -> None:
        """Apply collision APIs to every entity (static and dynamic) and the terrain."""
        for prim_path, entity in self._iter_entities():
            self._apply_collider(prim_path)
            if self._is_static(entity):
                self._report.static_collider_count += 1

        terrain_path = "/World/Environment/Terrain/Ground"
        self._apply_collider(terrain_path, kind="triangle_mesh")
        logger.info(
            "Collision enabled for %d entit(y/ies) plus terrain.",
            self._report.rigid_body_count + self._report.static_collider_count,
        )

    def enable_vehicle_physics(self) -> None:
        """Initialize the PhysX vehicle extension for wheeled/tracked ground vehicles."""
        vehicle_count = 0
        for prim_path, entity in self._iter_entities():
            if self._category_of(entity) not in ("vehicle", "vehicle_ground", "tank", "truck"):
                continue
            self._apply_vehicle_api(prim_path)
            vehicle_count += 1
        self._report.vehicle_count = vehicle_count
        logger.info("Vehicle physics enabled for %d ground vehicle(s).", vehicle_count)

    def enable_projectiles(self) -> None:
        """Mark fast-moving dynamic bodies as projectile candidates for CCD.

        This does not implement flight/guidance dynamics -- it only
        ensures continuous collision detection (and, where available,
        finer per-body solver sub-stepping) is applied to bodies whose
        declared/expected speed exceeds `projectile_speed_threshold_ms`,
        so they don't tunnel through thin geometry like a runway or a
        building wall during a single physics step.
        """
        candidates = 0
        threshold = self._config.projectile_speed_threshold_ms
        for prim_path, entity in self._iter_entities():
            category = self._category_of(entity)
            speed = self._estimated_speed(entity)
            if category in ("missile", "weapon") or speed >= threshold:
                self._apply_ccd(prim_path)
                candidates += 1
        self._report.projectile_candidate_count = candidates
        logger.info("Flagged %d projectile candidate(s) for CCD.", candidates)

    # ── entity introspection helpers ────────────────────────────────

    def _iter_entities(self):
        entities = getattr(self._registry, "entities", {})
        for prim_path, entity in entities.items():
            yield prim_path, entity

    @staticmethod
    def _is_static(entity: Any) -> bool:
        is_static = getattr(entity, "is_static", None)
        if is_static is not None:
            return bool(is_static)
        motion_class = getattr(entity, "motion_class", None)
        if motion_class is not None:
            return getattr(motion_class, "value", str(motion_class)) == "static"
        return True

    @staticmethod
    def _category_of(entity: Any) -> str:
        category = getattr(entity, "category", None)
        if category is not None:
            return str(getattr(category, "value", category)).lower()
        metadata = getattr(entity, "metadata", {}) or {}
        return str(metadata.get("entity_type", "")).lower()

    @staticmethod
    def _estimated_speed(entity: Any) -> float:
        metadata = getattr(entity, "metadata", {}) or {}
        try:
            return float(metadata.get("estimated_speed_ms", 0.0))
        except (TypeError, ValueError):
            return 0.0

    # ── internal stage helpers ──────────────────────────────────────

    def _add_ground_plane(self) -> None:
        path = "/World/PhysicsScene/GroundPlane"
        self._define_prim(
            path,
            "PhysicsCollisionGroup",
            {"height_m": self._config.ground_plane_height_m, "type": "ground_plane"},
        )
        self._report.ground_plane_added = True
        logger.info("Ground-plane collider added at %s", path)

    def _apply_rigid_body(self, prim_path: str, *, dynamic: bool) -> None:
        self._set_attrs(prim_path, {"physx:rigidBodyEnabled": True, "physx:kinematic": not dynamic})

    def _apply_collider(self, prim_path: str, *, kind: str = "convex_hull") -> None:
        self._set_attrs(prim_path, {"physx:collisionEnabled": True, "physx:approximation": kind})

    def _apply_vehicle_api(self, prim_path: str) -> None:
        self._set_attrs(prim_path, {"physx:vehicleEnabled": True})

    def _apply_ccd(self, prim_path: str) -> None:
        self._set_attrs(prim_path, {"physx:ccdEnabled": True})

    def _define_prim(self, path: str, prim_type: str, attrs: dict[str, Any]) -> None:
        if self._pxr_available and hasattr(self._stage, "DefinePrim"):
            try:
                self._stage.DefinePrim(path, prim_type)
                self._set_attrs(path, attrs)
                return
            except Exception as exc:  # noqa: BLE001
                raise PhysicsSceneError(f"Failed to define prim '{path}': {exc}") from exc
        logger.debug("[fallback] would define %s prim at %s with attrs=%s", prim_type, path, attrs)

    def _set_attrs(self, prim_path: str, attrs: dict[str, Any]) -> None:
        if self._pxr_available and hasattr(self._stage, "GetPrimAtPath"):
            try:
                from pxr import Sdf  # type: ignore

                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    return
                for key, value in attrs.items():
                    try:
                        prim.CreateAttribute(key, Sdf.ValueTypeNames.String).Set(str(value))
                    except Exception:  # noqa: BLE001
                        pass
                return
            except Exception as exc:  # noqa: BLE001
                raise PhysicsSceneError(f"Failed to set attributes on '{prim_path}': {exc}") from exc
        logger.debug("[fallback] would set attrs on %s: %s", prim_path, attrs)
