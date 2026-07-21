"""
runtime/entity_spawner.py
══════════════════════════════════════════════════════════════════════════
One-shot entity instantiation for PhysWorldLM's Omniverse Runtime.

Pipeline position
------------------
    EnvironmentBuilder.build()
            │
            ▼
    ┌────────────────┐
    │ ENTITY SPAWNER  │   <-- this module
    └────────────────┘
            │
            ▼
      OmniverseRuntime.initialize_physics() / discover_entities()

Scope
-----
This module is a *one-shot builder*. It reads `Entity` records straight
from a `WorldSpec` (the same canonical contract used by `world_spec.py`
and `scene_compiler.py`) and instantiates one prim per entity under
`/World/Entities`, applying transform, material, and metadata. It does
NOT attach physics bodies (that is `physics_initializer.PhysicsInitializer`),
does NOT animate anything per-frame (that is `animation_system`), and
attaches no behavior, guidance, or targeting logic of any kind -- it is
purely "place a labeled prim with the right transform/material at the
right path."

Spawning here is independent of (and a superset of) `OmniverseRuntime`'s
own `discover_entities()`/`classify_entities()`: those methods *read
back* whatever prims already exist on the stage (e.g. ones authored by
this spawner, or hand-authored in the source `.usda`). This module is
what makes sure those prims exist in the first place when entities come
from a `WorldSpec` rather than being hand-authored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from world_spec import Entity, WorldSpec

logger = logging.getLogger("physworldlm.entity_spawner")
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

ENTITIES_ROOT = "/World/Entities"


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class EntitySpawnError(Exception):
    """Raised when an entity cannot be spawned onto the stage."""


# ════════════════════════════════════════════════════════════════════════
# Spawn classification
# ════════════════════════════════════════════════════════════════════════

# Maps WorldSpec.Entity.entity_type / tags to the spawn method that
# should handle it. Order matters: first matching predicate wins.
_TYPE_DISPATCH: tuple[tuple[str, str], ...] = (
    ("aircraft", "spawn_aircraft"),
    ("missile", "spawn_weapon"),
    ("weapon", "spawn_weapon"),
    ("ship", "spawn_ship"),
    ("vessel", "spawn_ship"),
    ("vehicle", "spawn_vehicle"),
    ("tank", "spawn_vehicle"),
    ("truck", "spawn_vehicle"),
    ("building", "spawn_building"),
    ("structure", "spawn_building"),
    ("human", "spawn_human"),
    ("agent", "spawn_human"),
    ("tree", "spawn_tree"),
    ("vegetation", "spawn_tree"),
    ("sensor", "spawn_sensor"),
    ("radar", "spawn_sensor"),
)


@dataclass
class SpawnReport:
    """Summary of a spawn run."""

    spawned_paths: dict[str, str] = field(default_factory=dict)  # entity.id -> prim path
    skipped_entity_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spawned_count": len(self.spawned_paths),
            "spawned_paths": dict(self.spawned_paths),
            "skipped_entity_ids": list(self.skipped_entity_ids),
            "warnings": list(self.warnings),
        }


# ════════════════════════════════════════════════════════════════════════
# EntitySpawner
# ════════════════════════════════════════════════════════════════════════

class EntitySpawner:
    """Spawns `WorldSpec` entities onto a USD stage under `/World/Entities`.

    Example:
        >>> spawner = EntitySpawner(stage)
        >>> report = spawner.spawn_entities(world_spec)
        >>> report.spawned_paths["aircraft_01"]
        '/World/Entities/aircraft_01'
    """

    def __init__(self, stage: Any) -> None:
        """Initialize the spawner.

        Args:
            stage: An open USD stage handle, as produced by
                `OmniverseRuntime._open_stage()`.
        """
        self._stage = stage
        self._pxr_available = self._detect_pxr()
        self._report = SpawnReport()

    @staticmethod
    def _detect_pxr() -> bool:
        try:
            import pxr  # noqa: F401

            return True
        except ImportError:
            return False

    # ── orchestration ────────────────────────────────────────────────

    def spawn_entities(self, world_spec: WorldSpec) -> SpawnReport:
        """Spawn every entity declared in `world_spec` onto the stage.

        Args:
            world_spec: The compiled, validated `WorldSpec`.

        Returns:
            A `SpawnReport` mapping entity ids to authored prim paths.
        """
        self._define_prim(ENTITIES_ROOT, "Scope")
        logger.info("Spawning %d entit%s", len(world_spec.entities), "y" if len(world_spec.entities) == 1 else "ies")

        for entity in world_spec.entities:
            try:
                handler = self._resolve_handler(entity)
                prim_path = handler(entity)
                self.assign_transform(prim_path, entity)
                self.assign_material(prim_path, entity)
                self.assign_metadata(prim_path, entity)
                self._report.spawned_paths[entity.id] = prim_path
            except Exception as exc:  # noqa: BLE001 - one bad entity should not abort the batch
                msg = f"Failed to spawn entity '{entity.id}': {exc}"
                logger.warning(msg)
                self._report.warnings.append(msg)
                self._report.skipped_entity_ids.append(entity.id)

        logger.info(
            "Spawn complete: %d spawned, %d skipped",
            len(self._report.spawned_paths),
            len(self._report.skipped_entity_ids),
        )
        return self._report

    def _resolve_handler(self, entity: Entity) -> Callable[[Entity], str]:
        haystack = " ".join([entity.entity_type.lower(), *[t.lower() for t in entity.tags]])
        for keyword, method_name in _TYPE_DISPATCH:
            if keyword in haystack:
                return getattr(self, method_name)
        logger.info("No specific spawn handler for entity '%s' (type=%s); using generic spawn.", entity.id, entity.entity_type)
        return self._spawn_generic

    # ── per-category spawn methods ──────────────────────────────────

    def spawn_aircraft(self, entity: Entity) -> str:
        """Spawn an aircraft entity prim."""
        return self._spawn(entity, kind="aircraft")

    def spawn_vehicle(self, entity: Entity) -> str:
        """Spawn a ground vehicle entity prim (tank, truck, car, ...)."""
        return self._spawn(entity, kind="vehicle")

    def spawn_ship(self, entity: Entity) -> str:
        """Spawn a ship/vessel entity prim."""
        return self._spawn(entity, kind="ship")

    def spawn_building(self, entity: Entity) -> str:
        """Spawn a static building/structure entity prim."""
        return self._spawn(entity, kind="building")

    def spawn_human(self, entity: Entity) -> str:
        """Spawn a human/agent entity prim."""
        return self._spawn(entity, kind="human")

    def spawn_tree(self, entity: Entity) -> str:
        """Spawn a single tree/vegetation entity prim (for trees authored as
        discrete WorldSpec entities, as opposed to the bulk forest
        instancer authored by `EnvironmentBuilder.create_forest`)."""
        return self._spawn(entity, kind="tree")

    def spawn_sensor(self, entity: Entity) -> str:
        """Spawn a sensor/radar entity prim. Carries no signal-processing
        or detection logic -- placement and metadata only."""
        return self._spawn(entity, kind="sensor")

    def spawn_weapon(self, entity: Entity) -> str:
        """Spawn a weapon-system or munition entity prim.

        Note:
            This authors a labeled, positioned prim only -- geometry and
            scene-graph bookkeeping. It carries no guidance, targeting,
            fuzing, or lethality logic of any kind; that is explicitly
            out of scope for this runtime layer.
        """
        return self._spawn(entity, kind="weapon")

    def _spawn_generic(self, entity: Entity) -> str:
        """Fallback spawn path for entity types with no specific handler."""
        return self._spawn(entity, kind="generic")

    # ── shared spawn / assignment logic ─────────────────────────────

    def _spawn(self, entity: Entity, *, kind: str) -> str:
        prim_path = f"{ENTITIES_ROOT}/{self._safe_name(entity.label or entity.id)}"
        self._define_prim(prim_path, "Xform", attrs={"spawn_kind": kind})
        logger.info("Spawned %s '%s' -> %s", kind, entity.id, prim_path)
        return prim_path

    def assign_transform(self, prim_path: str, entity: Entity) -> None:
        """Author position/orientation/scale onto the prim at `prim_path`."""
        pos = entity.state.position
        rot = entity.state.orientation
        bbox = entity.bounding_box
        self._set_attrs(
            prim_path,
            {
                "translate": (pos.x, pos.y, pos.z),
                "rotateXYZ": (rot.x, rot.y, rot.z),
                "scale": (bbox.width, bbox.height, bbox.depth),
            },
        )

    def assign_material(self, prim_path: str, entity: Entity) -> None:
        """Bind the entity's material reference onto the prim at `prim_path`."""
        self._set_attrs(prim_path, {"material": entity.material})

    def assign_metadata(self, prim_path: str, entity: Entity) -> None:
        """Author identifying / semantic metadata (id, type, tags) onto the prim.

        This metadata is what `OmniverseRuntime._infer_category()` reads
        back via `entity_type` custom data during entity discovery, so
        the key name must stay in sync with that consumer.
        """
        self._set_attrs(
            prim_path,
            {
                "world_spec_id": entity.id,
                "entity_type": entity.entity_type,
                "is_static": entity.is_static,
                "tags": ",".join(entity.tags),
            },
        )

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
        if not safe or safe[0].isdigit():
            safe = f"_{safe}"
        return safe

    def _define_prim(self, path: str, prim_type: str, attrs: Optional[dict[str, Any]] = None) -> None:
        if self._pxr_available and hasattr(self._stage, "DefinePrim"):
            try:
                self._stage.DefinePrim(path, prim_type)
                if attrs:
                    self._set_attrs(path, attrs)
                return
            except Exception as exc:  # noqa: BLE001
                raise EntitySpawnError(f"Failed to define prim '{path}': {exc}") from exc
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
                    except Exception:  # noqa: BLE001 - attribute authoring is best-effort
                        pass
                return
            except Exception as exc:  # noqa: BLE001
                raise EntitySpawnError(f"Failed to set attributes on '{prim_path}': {exc}") from exc
        logger.debug("[fallback] would set attrs on %s: %s", prim_path, attrs)
