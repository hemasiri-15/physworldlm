"""stage_builder.py

OpenUSD Stage Builder for the PhysWorldLM framework.

This module constructs the **canonical scene skeleton** that every
PhysWorldLM-generated OpenUSD stage shares, immediately after the Scene
Compiler initializes a new stage and before any other builder module
(Entity Builder, Transform Builder, Material Builder, Physics Builder,
Sensor Builder, Relationship Builder, ...) runs.

This module is intentionally narrow in scope. It does NOT:
    * create entities, aircraft, vehicles, or sensors
    * assign materials
    * configure physics
    * apply transforms
    * export metadata or run the final USD export

Its only job is to guarantee that every scene begins with the same
predictable, stable namespace so downstream builders always know where
to attach their prims, and so every exported stage is structurally
interoperable with Omniverse, Isaac Sim, PhysX, Cesium, and future
backends.

Canonical hierarchy produced by ``StageBuilder.build()``::

    /World
    /World/Environment
    /World/Environment/Terrain
    /World/Environment/Atmosphere
    /World/Environment/Weather
    /World/Environment/Lighting
    /World/Entities
    /World/Sensors
    /World/Physics
    /World/Materials
    /World/Relationships
    /World/Metadata

This hierarchy matches the structure assumed by ``usd_exporter.py`` and
the Scene Graph produced by ``scene_compiler.py``; this module does not
redefine either of those.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger("physworldlm.stage_builder")


# =============================================================================
# Exceptions
# =============================================================================


class StageBuilderError(Exception):
    """Base class for all errors raised by the Stage Builder."""


class StageCreationError(StageBuilderError):
    """Raised when a prim required by the canonical hierarchy cannot be created."""


class InvalidStageError(StageBuilderError):
    """Raised when ``build()`` is called with a missing or invalid ``Usd.Stage``."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class StageBuilderConfig:
    """Configuration for the canonical hierarchy's root naming.

    Attributes:
        world_prim_name: Name of the root prim (default ``"World"``).
        environment_groups: Sub-groups created under ``Environment``.
        top_level_groups: Sibling groups created directly under
            ``World`` (excluding ``Environment``, which is handled
            separately since it has its own sub-groups).
    """

    world_prim_name: str = "World"
    environment_groups: tuple = ("Terrain", "Atmosphere", "Weather", "Lighting")
    top_level_groups: tuple = (
        "Entities",
        "Sensors",
        "Physics",
        "Materials",
        "Relationships",
        "Metadata",
    )


_DEFAULT_CONFIG: Final[StageBuilderConfig] = StageBuilderConfig()


# =============================================================================
# Stage Builder
# =============================================================================


class StageBuilder:
    """Builds the canonical OpenUSD scene skeleton used by PhysWorldLM.

    The Stage Builder accepts an already-created ``Usd.Stage`` (the
    Scene Compiler is responsible for creating it) and populates it
    in place with the organizational ``Xform`` hierarchy that every
    other builder module attaches prims to.

    Example:
        >>> from pxr import Usd
        >>> stage = Usd.Stage.CreateInMemory()
        >>> builder = StageBuilder()
        >>> world_prim = builder.build(stage)
        >>> stage.GetPrimAtPath("/World/Environment/Terrain").IsValid()
        True
    """

    def __init__(self, config: StageBuilderConfig = _DEFAULT_CONFIG) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, stage: Usd.Stage) -> Usd.Prim:
        """Populate ``stage`` with the canonical PhysWorldLM hierarchy.

        Args:
            stage: An existing, already-created ``Usd.Stage``. This
                method modifies it in place; it does not create or
                save the stage.

        Returns:
            The ``Usd.Prim`` for ``/World``, set as the stage's
            default prim.

        Raises:
            InvalidStageError: If ``stage`` is ``None`` or not a valid
                ``Usd.Stage``.
            StageCreationError: If any required prim cannot be created.
        """
        self._validate_stage(stage)

        world_prim = self._create_world_root(stage)
        self._create_environment(stage, world_prim)
        self._create_entities_group(stage, world_prim)
        self._create_sensor_group(stage, world_prim)
        self._create_physics_group(stage, world_prim)
        self._create_material_group(stage, world_prim)
        self._create_relationship_group(stage, world_prim)
        self._create_metadata_group(stage, world_prim)

        logger.info("Stage hierarchy completed")
        return world_prim

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_stage(self, stage: Usd.Stage) -> None:
        if stage is None:
            raise InvalidStageError("stage_builder requires an existing Usd.Stage, got None.")
        if not isinstance(stage, Usd.Stage):
            raise InvalidStageError(
                f"stage_builder requires a Usd.Stage instance, got {type(stage).__name__}."
            )

    # ------------------------------------------------------------------
    # Hierarchy construction
    # ------------------------------------------------------------------

    def _define_group(self, stage: Usd.Stage, parent_path: Sdf.Path, name: str) -> Usd.Prim:
        """Define a single organizational ``Xform`` prim and return it.

        Args:
            stage: The stage to create the prim on.
            parent_path: Path of the parent prim.
            name: Name of the group to create.

        Returns:
            The newly created (or pre-existing) ``Usd.Prim``.

        Raises:
            StageCreationError: If the prim could not be defined.
        """
        path = parent_path.AppendChild(name)
        logger.info("Creating %s", name)
        xform = UsdGeom.Xform.Define(stage, path)
        if not xform:
            raise StageCreationError(f"Failed to create prim at '{path}'.")
        return xform.GetPrim()

    def _create_world_root(self, stage: Usd.Stage) -> Usd.Prim:
        """Create ``/World`` and set it as the stage's default prim."""
        logger.info("Creating World root")
        world_path = Sdf.Path(f"/{self._config.world_prim_name}")
        world_xform = UsdGeom.Xform.Define(stage, world_path)
        if not world_xform:
            raise StageCreationError(f"Failed to create world root prim at '{world_path}'.")

        world_prim = world_xform.GetPrim()
        stage.SetDefaultPrim(world_prim)
        return world_prim

    def _create_environment(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Environment`` and its fixed sub-groups."""
        logger.info("Creating Environment")
        env_prim = self._define_group(stage, world_prim.GetPath(), "Environment")
        for sub_group in self._config.environment_groups:
            self._define_group(stage, env_prim.GetPath(), sub_group)
        return env_prim

    def _create_entities_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Entities``, the attachment point for Entity Builder."""
        return self._define_group(stage, world_prim.GetPath(), "Entities")

    def _create_sensor_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Sensors``, the attachment point for Sensor Builder."""
        return self._define_group(stage, world_prim.GetPath(), "Sensors")

    def _create_physics_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Physics``, the attachment point for Physics Builder."""
        return self._define_group(stage, world_prim.GetPath(), "Physics")

    def _create_material_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Materials``, the attachment point for Material Builder."""
        return self._define_group(stage, world_prim.GetPath(), "Materials")

    def _create_relationship_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Relationships``, the attachment point for Relationship Builder."""
        return self._define_group(stage, world_prim.GetPath(), "Relationships")

    def _create_metadata_group(self, stage: Usd.Stage, world_prim: Usd.Prim) -> Usd.Prim:
        """Create ``/World/Metadata``, the attachment point for stage-level metadata."""
        return self._define_group(stage, world_prim.GetPath(), "Metadata")


# =============================================================================
# Self-contained smoke test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    demo_stage = Usd.Stage.CreateInMemory()
    StageBuilder().build(demo_stage)

    expected_paths = [
        "/World",
        "/World/Environment",
        "/World/Environment/Terrain",
        "/World/Environment/Atmosphere",
        "/World/Environment/Weather",
        "/World/Environment/Lighting",
        "/World/Entities",
        "/World/Sensors",
        "/World/Physics",
        "/World/Materials",
        "/World/Relationships",
        "/World/Metadata",
    ]
    for expected_path in expected_paths:
        prim_exists = demo_stage.GetPrimAtPath(expected_path).IsValid()
        print(f"{expected_path}: {'OK' if prim_exists else 'MISSING'}")
