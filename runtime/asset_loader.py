"""
asset_loader.py
══════════════════════════════════════════════════════════════════════════
Asset Management subsystem of the PhysWorldLM Execution Layer.

Pipeline position
------------------
    scene.usda  (Scene Compiler output)
            │
            ▼
    ┌──────────────────┐
    │ OMNIVERSE RUNTIME │  (runtime/omniverse_runtime.py)
    └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │   ASSET LOADER     │   <-- this module
    └──────────────────┘
            │
            ▼
    Instantiated USD asset references on the live stage

Scope
-----
This module owns the translation of an *ontology entity* (an aircraft, a
missile, a building, ...) discovered on the stage by `OmniverseRuntime`
into a concrete, on-disk USD asset, and the mechanics of bringing that
asset onto the live stage as a referenced prim.

It is a pure asset-management layer:
    * It resolves WHICH asset file represents a given entity.
    * It loads and validates that asset.
    * It instantiates the asset as a USD reference at a target prim path.
    * It caches and unloads assets to avoid redundant I/O.

It deliberately does NOT:
    * Render anything.
    * Configure or attach PhysX physics to instantiated assets.
    * Implement AI, targeting, or behavior of any kind.

Integration with `runtime/omniverse_runtime.py`
------------------------------------------------
`AssetLoader.resolve_asset()` accepts any object exposing `.category`
(or `.metadata["entity_type"]`) and `.name` attributes -- which is
exactly the shape of `omniverse_runtime.RuntimeEntity`. The two modules
are intentionally decoupled (no import of `omniverse_runtime` here) so
that `asset_loader.py` can be used standalone, tested independently, and
registered as a `RuntimeSubsystem`-compatible collaborator by whichever
subsystem (e.g. a future `SceneAssemblySystem`) owns asset placement.

Public API
----------
    loader = AssetLoader(AssetLoaderConfig(asset_root=Path("assets/")))
    loader.load_registry()
    asset = loader.resolve_asset(entity)
    handle = loader.load_asset(asset.file_path)
    loader.validate_asset(handle)
    loader.instantiate_asset(stage, "/World/Entities/F16_01", asset)
    loader.unload_asset(asset.asset_id)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.asset_loader")
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

class AssetManagementError(Exception):
    """Base exception for all asset-management failures."""


class AssetNotFoundError(AssetManagementError):
    """Raised when no registry entry or on-disk file can be resolved for a request."""


class AssetLoadError(AssetManagementError):
    """Raised when a located asset file fails to load."""


class AssetValidationError(AssetManagementError):
    """Raised when a loaded asset fails structural validation."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class AssetCategory(Enum):
    """Ontology-level asset categories recognized by the loader."""

    AIRCRAFT = "aircraft"
    MISSILES = "missiles"
    VEHICLES = "vehicles"
    SHIPS = "ships"
    BUILDINGS = "buildings"
    HUMANS = "humans"
    RADAR = "radar"
    SENSORS = "sensors"
    TERRAIN = "terrain"
    VEGETATION = "vegetation"
    ROADS = "roads"
    RUNWAYS = "runways"
    WEATHER = "weather"
    UNKNOWN = "unknown"


class AssetFormat(Enum):
    """Supported on-disk USD asset formats."""

    USD = ".usd"
    USDA = ".usda"
    USDC = ".usdc"

    @classmethod
    def from_path(cls, path: Path) -> "AssetFormat":
        suffix = path.suffix.lower()
        for fmt in cls:
            if fmt.value == suffix:
                return fmt
        raise AssetValidationError(
            f"Unsupported asset file extension '{suffix}' for path '{path}'. "
            f"Supported formats: {[f.value for f in cls]}"
        )


class AssetLoadState(Enum):
    """Lifecycle state of an asset handle managed by `AssetLoader`."""

    UNLOADED = auto()
    LOADED = auto()
    VALIDATED = auto()
    INVALID = auto()


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AssetMetadata:
    """Registry-level description of a single on-disk USD asset.

    Attributes:
        asset_id: Stable unique identifier (e.g. "aircraft.f16.block50").
        name: Human-readable display name (e.g. "F-16 Block 50").
        category: Ontology-level asset category.
        file_path: Absolute path to the asset's root USD layer.
        format: USD file format (.usd / .usda / .usdc), derived from
            `file_path` if not explicitly supplied.
        entity_types: Ontology entity-type strings this asset may satisfy
            (e.g. {"aircraft", "fighter_jet"}). Used by `resolve_asset()`
            for entity-type-based lookup.
        tags: Free-form descriptive tags (e.g. {"fixed_wing", "military"}).
        default_scale: Uniform scale applied at instantiation time unless
            overridden by the requesting entity.
        bounding_box_m: Optional (width, height, depth) bounding box, in
            meters, used for sanity-checking against entity metadata.
        provenance: Free-form notes on asset origin/licensing.
    """

    asset_id: str
    name: str
    category: AssetCategory
    file_path: Path
    format: AssetFormat = field(init=False)
    entity_types: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)
    default_scale: float = 1.0
    bounding_box_m: Optional[tuple[float, float, float]] = None
    provenance: str = ""

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self.format = AssetFormat.from_path(self.file_path)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "category": self.category.value,
            "file_path": str(self.file_path),
            "format": self.format.value,
            "entity_types": sorted(self.entity_types),
            "tags": sorted(self.tags),
            "default_scale": self.default_scale,
            "bounding_box_m": self.bounding_box_m,
            "provenance": self.provenance,
        }


@dataclass
class AssetHandle:
    """A loaded (and optionally validated) in-memory representation of an asset.

    Attributes:
        metadata: The `AssetMetadata` this handle was loaded from.
        state: Current lifecycle state.
        stage_or_layer: Opaque handle to the loaded USD layer/stage. Uses
            `pxr.Sdf.Layer` / `pxr.Usd.Stage` when `pxr` is available, or
            a lightweight fallback handle otherwise.
        prim_count: Number of prims discovered in the loaded layer.
        load_warnings: Non-fatal issues recorded while loading/validating.
    """

    metadata: AssetMetadata
    state: AssetLoadState = AssetLoadState.UNLOADED
    stage_or_layer: Any = None
    prim_count: int = 0
    load_warnings: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# Entity protocol (structural — avoids importing omniverse_runtime)
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ResolvableEntity(Protocol):
    """Structural contract for anything `resolve_asset()` can accept.

    `omniverse_runtime.RuntimeEntity` satisfies this protocol without any
    import dependency between the two modules.
    """

    name: str
    metadata: dict[str, Any]

    @property
    def category(self) -> Any:
        ...


# ════════════════════════════════════════════════════════════════════════
# Asset registry
# ════════════════════════════════════════════════════════════════════════

class AssetRegistry:
    """In-memory index of all known `AssetMetadata`, with multi-key lookup.

    The registry supports lookup by asset id, asset name, ontology
    category, and ontology entity type, since different callers know
    different things about the asset they want (the runtime knows an
    entity's category and type; a debugging tool may know the asset id
    or name directly).
    """

    def __init__(self) -> None:
        self._by_id: dict[str, AssetMetadata] = {}
        self._by_name: dict[str, list[AssetMetadata]] = {}
        self._by_category: dict[AssetCategory, list[AssetMetadata]] = {}
        self._by_entity_type: dict[str, list[AssetMetadata]] = {}

    def register(self, asset: AssetMetadata) -> None:
        """Add `asset` to the registry, indexing it under all lookup keys."""
        if asset.asset_id in self._by_id:
            logger.warning("Overwriting existing asset registration for id '%s'.", asset.asset_id)

        self._by_id[asset.asset_id] = asset
        self._by_name.setdefault(asset.name.lower(), []).append(asset)
        self._by_category.setdefault(asset.category, []).append(asset)
        for entity_type in asset.entity_types:
            self._by_entity_type.setdefault(entity_type.lower(), []).append(asset)

        logger.debug("Registered asset '%s' (category=%s).", asset.asset_id, asset.category.value)

    def by_id(self, asset_id: str) -> Optional[AssetMetadata]:
        return self._by_id.get(asset_id)

    def by_name(self, name: str) -> list[AssetMetadata]:
        return list(self._by_name.get(name.lower(), []))

    def by_category(self, category: AssetCategory) -> list[AssetMetadata]:
        return list(self._by_category.get(category, []))

    def by_entity_type(self, entity_type: str) -> list[AssetMetadata]:
        return list(self._by_entity_type.get(entity_type.lower(), []))

    def all_assets(self) -> list[AssetMetadata]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._by_id


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AssetLoaderConfig:
    """User-configurable settings controlling `AssetLoader` behavior.

    Attributes:
        asset_root: Root directory under which asset files are resolved
            when `AssetMetadata.file_path` is relative.
        manifest_path: Optional path to a JSON manifest file describing
            the asset registry (see `AssetLoader.load_registry`). When
            None, `load_registry()` falls back to scanning `asset_root`.
        manifest_glob: Glob pattern used when scanning `asset_root` in
            the absence of a manifest.
        strict_validation: If True, `validate_asset()` raises on any
            issue. If False, issues are recorded as warnings on the
            `AssetHandle` and validation proceeds best-effort.
        enable_cache: If True, `load_asset()` reuses previously loaded
            handles instead of re-reading from disk.
    """

    asset_root: Path
    manifest_path: Optional[Path] = None
    manifest_glob: str = "**/*.usd*"
    strict_validation: bool = True
    enable_cache: bool = True

    def __post_init__(self) -> None:
        self.asset_root = Path(self.asset_root)
        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path)


# ════════════════════════════════════════════════════════════════════════
# AssetLoader
# ════════════════════════════════════════════════════════════════════════

class AssetLoader:
    """Resolves ontology entities into USD assets and instantiates them on a stage.

    `AssetLoader` is the bridge between the abstract entity taxonomy
    produced upstream (Ontology -> WorldSpec -> Scene Compiler) and
    concrete, on-disk USD asset files. It is used by `OmniverseRuntime`
    (or a subsystem registered with it) after entity discovery to give
    each discovered entity a physical USD representation on the stage.

    Example:
        >>> config = AssetLoaderConfig(asset_root=Path("assets/"))
        >>> loader = AssetLoader(config)
        >>> loader.load_registry()
        >>> asset = loader.resolve_asset(entity)
        >>> handle = loader.load_asset(asset.file_path)
        >>> loader.validate_asset(handle)
        >>> loader.instantiate_asset(stage, "/World/Entities/F16_01", asset)
    """

    def __init__(self, config: AssetLoaderConfig) -> None:
        """Initialize the loader (does not touch disk until `load_registry()`).

        Args:
            config: Asset loader configuration, including the asset root
                directory and optional manifest path.
        """
        self.config = config
        self.registry = AssetRegistry()
        self._cache: dict[str, AssetHandle] = {}

    # ── registry loading ─────────────────────────────────────────────

    def load_registry(self) -> None:
        """Populate `self.registry` from a manifest file or directory scan.

        If `AssetLoaderConfig.manifest_path` is set, the registry is
        built from that JSON manifest (a list of objects matching
        `AssetMetadata` fields). Otherwise, `asset_root` is scanned using
        `manifest_glob` and each discovered file is registered under a
        best-effort `AssetCategory` inferred from its parent directory
        name, with `asset_id` derived from its relative path.

        Raises:
            AssetNotFoundError: If neither a valid manifest nor any
                matching files can be found.
        """
        logger.info("Loading asset registry")

        if self.config.manifest_path is not None:
            self._load_registry_from_manifest(self.config.manifest_path)
        else:
            self._load_registry_from_directory_scan(self.config.asset_root)

        if len(self.registry) == 0:
            raise AssetNotFoundError(
                f"Asset registry is empty after loading "
                f"(manifest='{self.config.manifest_path}', asset_root='{self.config.asset_root}')."
            )

        logger.info("Asset registry loaded: %d asset(s).", len(self.registry))

    def _load_registry_from_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise AssetNotFoundError(f"Asset manifest not found: {manifest_path}")

        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetLoadError(f"Failed to read/parse asset manifest '{manifest_path}': {exc}") from exc

        if not isinstance(entries, list):
            raise AssetLoadError(f"Asset manifest '{manifest_path}' must contain a JSON array of asset entries.")

        for entry in entries:
            try:
                category = AssetCategory(str(entry["category"]).lower())
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed manifest entry %s: %s", entry, exc)
                continue

            raw_path = Path(entry["file_path"])
            file_path = raw_path if raw_path.is_absolute() else self.config.asset_root / raw_path

            try:
                asset = AssetMetadata(
                    asset_id=entry["asset_id"],
                    name=entry.get("name", entry["asset_id"]),
                    category=category,
                    file_path=file_path,
                    entity_types=frozenset(entry.get("entity_types", [])),
                    tags=frozenset(entry.get("tags", [])),
                    default_scale=float(entry.get("default_scale", 1.0)),
                    bounding_box_m=tuple(entry["bounding_box_m"]) if entry.get("bounding_box_m") else None,
                    provenance=entry.get("provenance", ""),
                )
            except (KeyError, AssetValidationError) as exc:
                logger.warning("Skipping malformed manifest entry %s: %s", entry, exc)
                continue

            self.registry.register(asset)

    def _load_registry_from_directory_scan(self, asset_root: Path) -> None:
        if not asset_root.exists():
            raise AssetNotFoundError(f"Asset root directory not found: {asset_root}")

        for file_path in sorted(asset_root.glob(self.config.manifest_glob)):
            if not file_path.is_file():
                continue
            try:
                fmt = AssetFormat.from_path(file_path)
            except AssetValidationError:
                continue  # not a recognized USD file extension; skip silently

            del fmt  # validated, format itself is re-derived inside AssetMetadata
            category = self._infer_category_from_path(file_path, asset_root)
            relative = file_path.relative_to(asset_root)
            asset_id = relative.with_suffix("").as_posix().replace("/", ".")

            asset = AssetMetadata(
                asset_id=asset_id,
                name=file_path.stem,
                category=category,
                file_path=file_path,
                entity_types=frozenset({category.value}),
            )
            self.registry.register(asset)

    @staticmethod
    def _infer_category_from_path(file_path: Path, asset_root: Path) -> AssetCategory:
        try:
            relative_parts = file_path.relative_to(asset_root).parts
        except ValueError:
            relative_parts = file_path.parts

        for part in relative_parts:
            lowered = part.lower()
            for category in AssetCategory:
                if category.value == lowered:
                    return category
        return AssetCategory.UNKNOWN

    # ── resolution ────────────────────────────────────────────────────

    def resolve_asset(self, entity: ResolvableEntity) -> AssetMetadata:
        """Resolve the best-matching `AssetMetadata` for a discovered entity.

        Resolution order:
            1. Exact `entity_types` match against `entity.metadata["entity_type"]`.
            2. Exact `entity_types` match against `entity.name` (lowercased).
            3. Category-level match against `entity.category`, picking the
               first registered asset in that category.

        Args:
            entity: Any object satisfying `ResolvableEntity` (e.g. an
                `omniverse_runtime.RuntimeEntity`).

        Returns:
            The resolved `AssetMetadata`.

        Raises:
            AssetNotFoundError: If no asset can be resolved for the
                entity by type or category.
        """
        entity_type = str(entity.metadata.get("entity_type", "")).strip().lower()
        if entity_type:
            candidates = self.registry.by_entity_type(entity_type)
            if candidates:
                logger.debug("Resolved entity '%s' via entity_type '%s'.", entity.name, entity_type)
                return candidates[0]

        name_candidates = self.registry.by_entity_type(entity.name.strip().lower())
        if name_candidates:
            logger.debug("Resolved entity '%s' via name match.", entity.name)
            return name_candidates[0]

        category = self._coerce_category(entity)
        category_candidates = self.registry.by_category(category)
        if category_candidates:
            logger.debug("Resolved entity '%s' via category fallback '%s'.", entity.name, category.value)
            return category_candidates[0]

        raise AssetNotFoundError(
            f"No asset could be resolved for entity '{entity.name}' "
            f"(entity_type='{entity_type}', category='{category.value}')."
        )

    @staticmethod
    def _coerce_category(entity: ResolvableEntity) -> AssetCategory:
        """Map an entity's `.category` (any enum/str) onto `AssetCategory`."""
        raw = entity.category
        value = getattr(raw, "value", raw)
        value = str(value).strip().lower()

        # omniverse_runtime.EntityCategory uses singular names (e.g. "missile");
        # AssetCategory uses plural names (e.g. "missiles"). Normalize both ways.
        for category in AssetCategory:
            if category.value == value or category.value == f"{value}s" or category.value.rstrip("s") == value:
                return category
        return AssetCategory.UNKNOWN

    # ── loading ───────────────────────────────────────────────────────

    def load_asset(self, asset_path: Path) -> AssetHandle:
        """Load the USD layer at `asset_path` into an `AssetHandle`.

        Uses the `pxr` (OpenUSD) bindings when available; falls back to a
        lightweight existence/format check when `pxr` is not installed,
        so the loader remains usable outside a full Omniverse Kit
        environment for testing and pipeline development.

        Args:
            asset_path: Absolute or registry-relative path to the asset's
                root USD layer.

        Returns:
            A populated `AssetHandle` in state `LOADED`.

        Raises:
            AssetNotFoundError: If `asset_path` does not exist.
            AssetLoadError: If the file exists but fails to load.
        """
        asset_path = Path(asset_path)
        if not asset_path.is_absolute():
            asset_path = self.config.asset_root / asset_path

        if not asset_path.exists():
            raise AssetNotFoundError(f"Asset file not found: {asset_path}")

        metadata = self._find_metadata_by_path(asset_path) or AssetMetadata(
            asset_id=asset_path.stem,
            name=asset_path.stem,
            category=AssetCategory.UNKNOWN,
            file_path=asset_path,
        )

        if self.config.enable_cache and metadata.asset_id in self._cache:
            logger.debug("Cache hit for asset '%s'.", metadata.asset_id)
            return self._cache[metadata.asset_id]

        logger.info("Loading asset '%s' from '%s'.", metadata.asset_id, asset_path)

        try:
            stage_or_layer, prim_count = self._open_layer(asset_path)
        except AssetManagementError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssetLoadError(f"Failed to load asset '{asset_path}': {exc}") from exc

        handle = AssetHandle(
            metadata=metadata,
            state=AssetLoadState.LOADED,
            stage_or_layer=stage_or_layer,
            prim_count=prim_count,
        )

        if self.config.enable_cache:
            self.cache_asset(handle)

        return handle

    def _open_layer(self, asset_path: Path) -> tuple[Any, int]:
        """Open `asset_path` and return `(handle, prim_count)`."""
        try:
            from pxr import Sdf  # type: ignore

            layer = Sdf.Layer.FindOrOpen(str(asset_path))
            if layer is None:
                raise AssetLoadError(f"`pxr.Sdf.Layer.FindOrOpen` returned None for '{asset_path}'.")
            prim_count = sum(1 for _ in layer.rootPrims)
            return layer, prim_count
        except ImportError:
            logger.warning(
                "`pxr` (OpenUSD) bindings not found; using fallback asset handle for '%s'. "
                "Install `usd-core` / run inside Omniverse Kit for full fidelity.",
                asset_path,
            )
            return _FallbackAssetLayer(asset_path), 1

    def _find_metadata_by_path(self, asset_path: Path) -> Optional[AssetMetadata]:
        for asset in self.registry.all_assets():
            if asset.file_path == asset_path:
                return asset
        return None

    # ── validation ────────────────────────────────────────────────────

    def validate_asset(self, handle: AssetHandle) -> None:
        """Validate a loaded asset handle.

        Checks performed:
            * The handle is in a loadable state (not UNLOADED).
            * The underlying layer/stage reports at least one root prim.
            * The file extension matches a supported `AssetFormat`.

        Args:
            handle: The `AssetHandle` to validate, as returned by
                `load_asset()`.

        Raises:
            AssetValidationError: If validation fails and
                `AssetLoaderConfig.strict_validation` is True.
        """
        issues: list[str] = []

        if handle.state is AssetLoadState.UNLOADED:
            issues.append("Asset handle has not been loaded.")
        if handle.prim_count <= 0:
            issues.append("Loaded asset contains zero root prims.")
        try:
            AssetFormat.from_path(handle.metadata.file_path)
        except AssetValidationError as exc:
            issues.append(str(exc))

        if issues:
            handle.load_warnings.extend(issues)
            message = "; ".join(issues)
            if self.config.strict_validation:
                handle.state = AssetLoadState.INVALID
                raise AssetValidationError(
                    f"Asset '{handle.metadata.asset_id}' failed validation: {message}"
                )
            logger.warning("Asset '%s' validation issue(s) (non-strict mode): %s", handle.metadata.asset_id, message)

        handle.state = AssetLoadState.VALIDATED
        logger.info("Asset '%s' validated (%d root prim(s)).", handle.metadata.asset_id, handle.prim_count)

    # ── instantiation ────────────────────────────────────────────────

    def instantiate_asset(self, stage: Any, prim_path: str, asset: AssetMetadata) -> str:
        """Instantiate `asset` as a USD reference at `prim_path` on `stage`.

        This creates (or reuses) an `Xform`/`def` prim at `prim_path` and
        adds a USD reference to `asset.file_path`. No physics or
        rendering setup is performed here -- this method's only
        responsibility is placing the asset's geometry reference on the
        stage.

        Args:
            stage: An open USD stage handle, as produced by
                `omniverse_runtime.OmniverseRuntime.load_stage()` (a
                `pxr.Usd.Stage` or compatible fallback handle).
            prim_path: Target USD prim path for the instantiated
                reference (e.g. "/World/Entities/F16_01").
            asset: The resolved `AssetMetadata` to instantiate.

        Returns:
            The prim path the asset was instantiated at (equal to
            `prim_path`).

        Raises:
            AssetLoadError: If instantiation onto the stage fails.
        """
        logger.info("Instantiating asset '%s' at prim path '%s'.", asset.asset_id, prim_path)

        try:
            from pxr import Sdf, Usd, UsdGeom  # type: ignore

            if not isinstance(stage, Usd.Stage):
                raise AssetLoadError(
                    f"instantiate_asset() requires a pxr.Usd.Stage when `pxr` is available "
                    f"(got {type(stage).__name__})."
                )
            xform = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))
            xform.AddScaleOp().Set((asset.default_scale,) * 3)
            xform.GetPrim().GetReferences().AddReference(str(asset.file_path))
        except ImportError:
            logger.warning(
                "`pxr` (OpenUSD) bindings not found; recording instantiation on fallback "
                "stage handle for '%s' without writing real USD references.",
                prim_path,
            )
            if isinstance(stage, dict):
                stage.setdefault("instantiated_assets", {})[prim_path] = asset.to_dict()
            else:
                raise AssetLoadError(
                    "Cannot instantiate asset: no `pxr` bindings and `stage` is not a "
                    "recognized fallback handle."
                )
        except AssetLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AssetLoadError(f"Failed to instantiate asset '{asset.asset_id}' at '{prim_path}': {exc}") from exc

        return prim_path

    # ── caching / unloading ──────────────────────────────────────────

    def cache_asset(self, handle: AssetHandle) -> None:
        """Insert `handle` into the in-memory asset cache, keyed by asset id.

        Args:
            handle: The asset handle to cache.
        """
        self._cache[handle.metadata.asset_id] = handle
        logger.debug("Cached asset '%s' (cache size=%d).", handle.metadata.asset_id, len(self._cache))

    def get_cached(self, asset_id: str) -> Optional[AssetHandle]:
        """Return the cached `AssetHandle` for `asset_id`, if present."""
        return self._cache.get(asset_id)

    def unload_asset(self, asset_id: str) -> None:
        """Remove `asset_id` from the cache and release its handle.

        Args:
            asset_id: The asset id to unload.

        Raises:
            AssetNotFoundError: If `asset_id` is not currently cached.
        """
        if asset_id not in self._cache:
            raise AssetNotFoundError(f"Cannot unload asset '{asset_id}': not currently cached.")

        handle = self._cache.pop(asset_id)
        handle.stage_or_layer = None
        handle.state = AssetLoadState.UNLOADED
        logger.info("Unloaded asset '%s' (cache size=%d).", asset_id, len(self._cache))

    def unload_all(self) -> None:
        """Unload every currently cached asset."""
        for asset_id in list(self._cache.keys()):
            self.unload_asset(asset_id)

    # ── introspection ────────────────────────────────────────────────

    def cached_assets(self) -> Iterable[AssetHandle]:
        """Return all currently cached asset handles."""
        return list(self._cache.values())


# ════════════════════════════════════════════════════════════════════════
# Fallback asset layer (used only when `pxr` is unavailable)
# ════════════════════════════════════════════════════════════════════════

class _FallbackAssetLayer:
    """Minimal stand-in for a `pxr.Sdf.Layer`, used when the `pxr`
    (OpenUSD) bindings are not installed in the current environment.

    Confirms the asset file exists and is readable, without attempting
    real USD parsing. Sufficient for exercising `AssetLoader`'s caching,
    validation, and lifecycle logic outside a full Omniverse Kit
    installation.
    """

    def __init__(self, asset_path: Path) -> None:
        self.asset_path = asset_path
        if not asset_path.exists():
            raise AssetLoadError(f"Fallback asset layer: file not found at '{asset_path}'.")
        self.size_bytes = asset_path.stat().st_size


__all__ = [
    "AssetLoader",
    "AssetLoaderConfig",
    "AssetRegistry",
    "AssetMetadata",
    "AssetHandle",
    "AssetCategory",
    "AssetFormat",
    "AssetLoadState",
    "ResolvableEntity",
    "AssetManagementError",
    "AssetNotFoundError",
    "AssetLoadError",
    "AssetValidationError",
]
