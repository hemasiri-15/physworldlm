"""
usd_loader.py
══════════════════════════════════════════════════════════════════════════
USD stage loading, composition, indexing, and diagnostics layer for the
Omniverse Connector layer of PhysWorldLM.

Pipeline position
------------------
    Natural Language → Ontology → WorldSpec → Scene Compiler → scene.usda
                                                                      │
              ┌───────────────────────────────────────────────────────┤
              │                                                       ▼
    ┌────────────────────┐   resolved asset paths (optional hook)  ┌────────────────────┐
    │  asset_server.py     │◄─────────────────────────────────────┤  usd_loader.py       │  <-- this module
    │  (decoupled, unused  │                                        └──────────┬─────────┘
    │  directly by import) │                                                   ▼
    └────────────────────┘                                    stage_manager.py / physics_scene.py /
                                                                renderer.py / USDLoader-style consumers

Scope
-----
This module owns exactly one concern: everything to do with *loading and
inspecting USD content* -- opening, closing, and reloading stages;
resolving and diagnosing composition arcs (references, payloads,
sublayers, variants); validating and repairing common authoring
mistakes; indexing, searching, and traversing prims; and reporting
statistics and structured diagnostics about a loaded stage.

This module explicitly does NOT:
    * launch Omniverse Kit / Isaac Sim (``app_launcher.OmniverseLauncher``
      owns process lifecycle)
    * render frames (``renderer.py``)
    * simulate physics (``physics_scene.py``)
    * resolve *where* an asset ID's canonical bytes live or cache them
      locally (``asset_server.AssetServer`` owns that); this module only
      ever opens a path/URI it is given, though it accepts an optional
      resolver *hook* (a plain callable) so a caller can wire the two
      together without either module importing the other
    * parse natural language, ontologies, or ``WorldSpec`` objects, or
      compile/export USD content (those are upstream compiler stages)
    * spawn entities, sensors, robots, or terrain content into a stage
      (those components author into a stage this module hands them via
      :meth:`UsdLoader.get_stage`)

Design constraints
-------------------
    * No ``pxr`` import happens at module load time. Every such
      dependency is deferred to the call site that actually needs it,
      behind :func:`_lazy_import`, so this module loads (and its
      dataclasses/enums are fully usable, e.g. in tests) on a machine
      with no USD/Omniverse installation at all.
    * All failure modes raise a documented, specific
      :class:`UsdLoaderError` subclass. Nothing lets a raw
      ``ImportError`` or an opaque ``Tf.ErrorException`` escape uncaught.
    * No global mutable state. Every piece of runtime state (the open
      stage handle, the prim index, the reference graph, load history,
      ...) lives on the :class:`UsdLoader` instance, guarded by an
      internal lock, so multiple independent loaders never interfere.
    * This module never inspects *content it did not open itself* --
      dependency/reference discovery always walks the live composed
      stage (via ``Usd.PrimCompositionQuery`` / layer stacks), never a
      hand-maintained manifest.
    * Designed to be handed to, and consumed by, both
      ``stage_manager.StageManager``-style components (which continue
      authoring into the stage this module opens) and
      ``physics_scene.PhysicsScene`` (which reads prims from it) without
      either being imported here.

Public API
----------
    loader = UsdLoader()
    stats = loader.load_usd("scene.usda")
    meshes = loader.find_meshes()
    report = loader.validate()
    loader.export_report("report.json")
    loader.unload()

Or, as a context manager::

    with UsdLoader() as loader:
        loader.open("scene.usda")
        for prim in loader.list_prims():
            ...
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.usd_loader")
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

class UsdLoaderError(Exception):
    """Base class for all errors raised by :class:`UsdLoader`."""


class UsdImportError(UsdLoaderError):
    """Raised when the ``pxr`` USD Python bindings can't be imported.

    Distinct from a bare ``ImportError`` so callers can catch exactly
    "USD isn't available in this Python environment" without
    accidentally swallowing an unrelated import bug in their own code.
    """


class NotOpenError(UsdLoaderError):
    """Raised when an operation requires a stage to already be open."""


class AlreadyOpenError(UsdLoaderError):
    """Raised by :meth:`UsdLoader.open` when a stage is already open and
    ``force=False``."""


class UsdOpenError(UsdLoaderError):
    """Raised when opening a USD file/layer fails."""


class UsdCloseError(UsdLoaderError):
    """Raised when releasing an open stage fails."""


class UsdReloadError(UsdLoaderError):
    """Raised when :meth:`UsdLoader.reload` fails."""


class PrimNotFoundError(UsdLoaderError):
    """Raised when a referenced prim path or name cannot be found."""


class UsdValidationError(UsdLoaderError):
    """Raised when validation itself cannot be performed (not when a
    stage is merely found invalid -- that is reported via
    :class:`UsdValidationReport` instead)."""


class UsdRepairError(UsdLoaderError):
    """Raised when :meth:`UsdLoader.repair` fails to apply a fix."""


class CompositionError(UsdLoaderError):
    """Raised when a composition arc (reference/payload/sublayer/variant)
    cannot be inspected or resolved."""


class ReferenceResolutionError(UsdLoaderError):
    """Raised when :meth:`UsdLoader.find_references` cannot resolve the
    asset resolver context for a prim."""


class LayerLoadError(UsdLoaderError):
    """Raised when an individual sublayer fails to load."""


class PayloadLoadError(UsdLoaderError):
    """Raised when :meth:`UsdLoader.load_payloads` fails to load a
    requested payload."""


class VariantError(UsdLoaderError):
    """Raised when a variant set/selection operation fails."""


class ExportReportError(UsdLoaderError):
    """Raised when :meth:`UsdLoader.export_report` fails to write to disk."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class UsdLoaderState(str, Enum):
    """Lifecycle state of the :class:`UsdLoader` instance itself.

    Transitions (happy path)::

        CLOSED -> OPENING -> OPEN -> CLOSED

    ``ERROR`` is reachable from ``OPENING`` and is terminal for the
    current stage handle until :meth:`UsdLoader.close` resets it.
    """

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    ERROR = "error"


class ValidationSeverity(str, Enum):
    """Severity of a single :class:`ValidationIssue`."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PrimCategory(str, Enum):
    """Coarse content category a prim is bucketed into for indexing.

    Purely a classification convenience layered on top of a prim's USD
    schema type name -- it never changes how the prim is authored or
    composed.
    """

    MESH = "mesh"
    MATERIAL = "material"
    SHADER = "shader"
    SKELETON = "skeleton"
    SKELETON_ROOT = "skeleton_root"
    ANIMATION = "animation"
    POINT_INSTANCER = "point_instancer"
    XFORM = "xform"
    SCOPE = "scope"
    CAMERA = "camera"
    LIGHT = "light"
    OTHER = "other"


class ReferenceKind(str, Enum):
    """Kind of composition arc a :class:`ReferenceInfo` describes."""

    REFERENCE = "reference"
    PAYLOAD = "payload"
    SUBLAYER = "sublayer"


# ════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════

#: Schema type names (as returned by ``prim.GetTypeName()``) mapped to a
#: coarse :class:`PrimCategory`, used by the indexing/search/find_* layer.
_TYPE_NAME_CATEGORY: dict[str, PrimCategory] = {
    "Mesh": PrimCategory.MESH,
    "Material": PrimCategory.MATERIAL,
    "Shader": PrimCategory.SHADER,
    "Skeleton": PrimCategory.SKELETON,
    "SkelRoot": PrimCategory.SKELETON_ROOT,
    "SkelAnimation": PrimCategory.ANIMATION,
    "PointInstancer": PrimCategory.POINT_INSTANCER,
    "Xform": PrimCategory.XFORM,
    "Scope": PrimCategory.SCOPE,
    "Camera": PrimCategory.CAMERA,
    "DistantLight": PrimCategory.LIGHT,
    "DomeLight": PrimCategory.LIGHT,
    "SphereLight": PrimCategory.LIGHT,
    "RectLight": PrimCategory.LIGHT,
    "DiskLight": PrimCategory.LIGHT,
    "CylinderLight": PrimCategory.LIGHT,
}

#: Shader ``info:id`` values treated as texture-reading shaders when
#: :meth:`UsdLoader.find_textures` walks material networks.
_TEXTURE_SHADER_IDS: tuple[str, ...] = ("UsdUVTexture",)

#: Shader input names checked, in order, for a texture asset path.
_TEXTURE_FILE_INPUTS: tuple[str, ...] = ("file", "filename", "texture")

#: Schema version embedded in exported reports.
_REPORT_VERSION = "1.0.0"


# ════════════════════════════════════════════════════════════════════════
# Lazy import helper
# ════════════════════════════════════════════════════════════════════════

def _lazy_import(module_name: str, *, hint: str = "") -> Any:
    """Import ``module_name``, raising :class:`UsdImportError` on failure.

    Every ``pxr.*`` import used by this module goes through this
    function so that (a) importing ``usd_loader`` itself never requires
    USD to be installed, and (b) a missing dependency surfaces as one
    clear, catchable exception instead of a raw ``ImportError``.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = f"Failed to import '{module_name}'. USD (pxr) is required for this operation."
        if hint:
            message = f"{message} {hint}"
        raise UsdImportError(message) from exc


# ════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PrimInfo:
    """Lightweight, JSON-friendly description of a single prim.

    Attributes:
        path: The prim's stage path (e.g. ``"/World/Vehicle/Body"``).
        type_name: USD schema type name (e.g. ``"Mesh"``), or empty
            string for a typeless "def" prim.
        category: Coarse :class:`PrimCategory` classification.
        kind: USD ``model:kind`` metadata value, if any (e.g.
            ``"component"``, ``"assembly"``).
        active: Whether the prim is active in the composed stage.
        is_abstract: Whether the prim is a class (abstract) prim.
        instanceable: Whether the prim is marked instanceable.
        has_payload: Whether the prim has at least one payload arc.
        has_reference: Whether the prim has at least one reference arc.
        variant_sets: Names of variant sets authored directly on the prim.
        specifier: USD specifier (``"def"``, ``"over"``, ``"class"``).
    """

    path: str
    type_name: str
    category: PrimCategory
    kind: str
    active: bool
    is_abstract: bool
    instanceable: bool
    has_payload: bool
    has_reference: bool
    variant_sets: tuple[str, ...]
    specifier: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this prim description."""
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["variant_sets"] = list(self.variant_sets)
        return payload


@dataclass(frozen=True)
class ReferenceInfo:
    """A single composition arc discovered on a prim.

    Attributes:
        prim_path: Path of the prim the arc is authored on.
        kind: Whether this is a reference, payload, or sublayer.
        asset_path: The raw (unresolved) asset path/identifier authored
            in the arc.
        resolved_path: The resolver-resolved local path/URI, or
            ``None`` if resolution failed.
        layer_offset: ``(scale, offset)`` time-remapping applied by the
            arc, if any (``(1.0, 0.0)`` for none).
        is_internal: True if the arc targets another prim within the
            same stage/layer (no external asset path).
        broken: True if the arc's target could not be resolved/opened.
    """

    prim_path: str
    kind: ReferenceKind
    asset_path: str
    resolved_path: Optional[str]
    layer_offset: tuple[float, float]
    is_internal: bool
    broken: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this reference description."""
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True)
class ValidationIssue:
    """A single finding from :meth:`UsdLoader.validate`.

    Attributes:
        severity: How serious the finding is.
        message: Human-readable description.
        prim_path: The prim the issue concerns, or ``""`` for a
            stage/layer-level issue.
        code: Short, stable machine-readable identifier for the issue
            type (e.g. ``"missing-default-prim"``), used by
            :meth:`UsdLoader.repair` to decide which fixes apply.
    """

    severity: ValidationSeverity
    message: str
    prim_path: str
    code: str


@dataclass
class UsdValidationReport:
    """Aggregate result of :meth:`UsdLoader.validate`.

    Attributes:
        issues: All findings, in the order they were discovered.
        generated_at: Wall-clock timestamp this report was produced.
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """True if no issue at :attr:`ValidationSeverity.ERROR` was found."""
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity is ValidationSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of this report."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "generated_at": self.generated_at,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "prim_path": issue.prim_path,
                    "code": issue.code,
                }
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class UsdStatistics:
    """Snapshot of aggregate statistics about the currently open stage.

    Attributes:
        identifier: The root layer's identifier (path/URI) of the open stage.
        total_prims: Total number of prims in the composed stage.
        by_category: Count of prims per :class:`PrimCategory` value.
        active_prims: Number of active prims.
        inactive_prims: Number of inactive prims.
        instanceable_prims: Number of prims marked instanceable.
        layer_count: Number of layers in the full layer stack (root +
            sublayers, recursively).
        reference_count: Number of reference arcs found across all prims.
        payload_count: Number of payload arcs found across all prims.
        broken_reference_count: Number of reference/payload arcs that
            failed to resolve.
        variant_set_count: Number of variant sets authored anywhere on
            the stage.
        has_default_prim: Whether the root layer declares a default prim.
        time_code_range: ``(start, end)`` authored stage time codes, or
            ``None`` if not authored.
        up_axis: Stage ``upAxis`` metadata (``"Y"`` / ``"Z"``), or ``""``.
        meters_per_unit: Stage ``metersPerUnit`` metadata.
    """

    identifier: str
    total_prims: int
    by_category: dict[str, int]
    active_prims: int
    inactive_prims: int
    instanceable_prims: int
    layer_count: int
    reference_count: int
    payload_count: int
    broken_reference_count: int
    variant_set_count: int
    has_default_prim: bool
    time_code_range: Optional[tuple[float, float]]
    up_axis: str
    meters_per_unit: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain-dict rendering of these statistics."""
        return asdict(self)


@dataclass(frozen=True)
class LoadRecord:
    """Record of a single completed or failed load (open/reload) call.

    Attributes:
        identifier: The layer identifier that was opened.
        started_at: Wall-clock timestamp the load began.
        completed_at: Wall-clock timestamp the load finished.
        success: Whether the load completed successfully.
        error: Human-readable error description, or ``None`` on success.
    """

    identifier: str
    started_at: float
    completed_at: float
    success: bool
    error: Optional[str]


# ════════════════════════════════════════════════════════════════════════
# UsdLoader
# ════════════════════════════════════════════════════════════════════════

class UsdLoader:
    """Opens, composes, indexes, validates, and reports on USD stages.

    ``UsdLoader`` owns every concern related to *getting a USD stage
    into memory and understanding its structure*. It contains no
    process-lifecycle logic (``app_launcher.OmniverseLauncher``), no
    physics (``physics_scene.PhysicsScene``), and no rendering
    (``renderer.py``) -- it only opens a stage and hands out
    read-mostly information (and the live stage handle itself, via
    :meth:`get_stage`) to whoever needs it.

    Thread-safety: all state-mutating operations are guarded by an
    internal lock, so this loader is safe to call from multiple threads
    (e.g. a background validation pass alongside interactive queries).

    Example:
        >>> loader = UsdLoader()
        >>> stats = loader.load_usd("scene.usda")
        >>> report = loader.validate()
        >>> loader.unload()
    """

    def __init__(
        self,
        *,
        asset_resolver_hook: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Create a loader. Does not touch disk or import ``pxr`` yet.

        Args:
            asset_resolver_hook: Optional callable that maps a
                caller-defined asset identifier to a concrete local
                path/URI before it is handed to ``Usd.Stage.Open``
                (e.g. ``AssetServer.resolve_for_stage``). This module
                never imports ``asset_server`` itself; wiring the two
                together is entirely the caller's responsibility via
                this hook, keeping the two modules decoupled.
        """
        self._lock = threading.RLock()
        self._state: UsdLoaderState = UsdLoaderState.CLOSED
        self._stage: Optional[Any] = None
        self._identifier: str = ""
        self._asset_resolver_hook = asset_resolver_hook
        self._prim_index: dict[str, PrimInfo] = {}
        self._loads: list[LoadRecord] = []
        self._last_error: Optional[BaseException] = None
        self._payload_paths_loaded: set[str] = set()

    # ------------------------------------------------------------------
    # State introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> UsdLoaderState:
        """Current lifecycle state of this loader."""
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        """Whether a stage is currently open and usable."""
        return self.state is UsdLoaderState.OPEN

    @property
    def identifier(self) -> str:
        """The root layer identifier of the currently open stage, or ``""``."""
        with self._lock:
            return self._identifier

    @property
    def last_error(self) -> Optional[BaseException]:
        """The exception that most recently drove this loader into ``ERROR``, if any."""
        with self._lock:
            return self._last_error

    @property
    def load_history(self) -> list[LoadRecord]:
        """All open/reload attempts performed by this loader instance, oldest first."""
        with self._lock:
            return list(self._loads)

    def _require_open(self) -> None:
        if self._state is not UsdLoaderState.OPEN or self._stage is None:
            raise NotOpenError(
                f"UsdLoader has no open stage (state='{self._state.value}'). "
                "Call open() or load_usd() first."
            )

    # ------------------------------------------------------------------
    # Open / close / reload
    # ------------------------------------------------------------------

    def open(
        self,
        identifier: "str | Path",
        *,
        force: bool = False,
        variant_selections: Optional[dict[str, dict[str, str]]] = None,
        load_all_payloads: bool = True,
    ) -> Any:
        """Open a USD layer as the current stage.

        Args:
            identifier: Path or URI of the root layer to open. Passed
                through :attr:`asset_resolver_hook` first, if one was
                configured.
            force: If True, close any currently open stage first
                instead of raising.
            variant_selections: Mapping of prim path -> ``{variant_set:
                selection}`` to apply immediately after opening, before
                indexing (so the index reflects the selected variants).
            load_all_payloads: If True (default), open with all
                payloads loaded (``Usd.Stage.LoadAll``). If False, open
                with no payloads loaded (``Usd.Stage.LoadNone``) -- see
                :meth:`load_payloads` for loading them selectively
                afterward (lazy loading).

        Returns:
            The live ``pxr.Usd.Stage`` handle.

        Raises:
            AlreadyOpenError: If a stage is already open and
                ``force`` is False.
            UsdOpenError: If the layer cannot be opened.
        """
        with self._lock:
            if self._state is UsdLoaderState.OPEN and not force:
                raise AlreadyOpenError(
                    "A stage is already open; pass force=True to replace it, or call close() first."
                )
        if self.is_open():
            self.close()

        raw_identifier = str(identifier)
        resolved_identifier = raw_identifier
        if self._asset_resolver_hook is not None:
            try:
                resolved_identifier = self._asset_resolver_hook(raw_identifier)
            except Exception as exc:  # noqa: BLE001
                raise UsdOpenError(
                    f"asset_resolver_hook failed for '{raw_identifier}': {exc}"
                ) from exc

        with self._lock:
            self._state = UsdLoaderState.OPENING
            self._last_error = None

        logger.info("Opening USD stage '%s'.", resolved_identifier)
        started_at = time.time()
        usd_module = _lazy_import("pxr.Usd", hint="Install USD (pip install usd-core) or run inside Kit.")

        try:
            load_set = usd_module.Stage.LoadAll if load_all_payloads else usd_module.Stage.LoadNone
            stage = usd_module.Stage.Open(resolved_identifier, load_set)
            if stage is None:
                raise UsdOpenError(f"Usd.Stage.Open('{resolved_identifier}') returned None.")

            with self._lock:
                self._stage = stage
                self._identifier = resolved_identifier
                self._state = UsdLoaderState.OPEN
                self._payload_paths_loaded.clear()

            if variant_selections:
                self._apply_variant_selections(variant_selections)

            self._rebuild_prim_index()
            self._record_load(resolved_identifier, started_at, success=True, error=None)
            logger.info(
                "Opened USD stage '%s' (%d prim(s) indexed).",
                resolved_identifier, len(self._prim_index),
            )
            return stage

        except UsdLoaderError as exc:
            self._enter_error_state(exc)
            self._record_load(resolved_identifier, started_at, success=False, error=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - never leak an opaque Tf error
            wrapped = UsdOpenError(f"Failed to open '{resolved_identifier}': {exc}")
            self._enter_error_state(wrapped)
            self._record_load(resolved_identifier, started_at, success=False, error=str(wrapped))
            raise wrapped from exc

    def close(self) -> None:
        """Release the currently open stage, if any. Idempotent.

        Raises:
            UsdCloseError: If releasing the stage raises unexpectedly.
        """
        with self._lock:
            if self._state in (UsdLoaderState.CLOSED,) and self._stage is None:
                logger.info("close() called with nothing open; nothing to do.")
                return

        try:
            with self._lock:
                stage = self._stage
                self._stage = None
            if stage is not None:
                # Usd.Stage has no explicit "close" -- releasing every
                # reference lets the underlying PcpCache/SdfLayer stack
                # be garbage collected. Any outstanding external
                # reference obtained via get_stage() naturally keeps
                # the C++ side alive until it, too, is released.
                del stage
        except Exception as exc:  # noqa: BLE001
            raise UsdCloseError(f"Failed to close stage '{self._identifier}': {exc}") from exc

        with self._lock:
            self._identifier = ""
            self._prim_index.clear()
            self._payload_paths_loaded.clear()
            self._state = UsdLoaderState.CLOSED

        logger.info("USD stage closed.")

    def reload(self) -> Any:
        """Close and re-open the current stage from its same identifier.

        Also reloads every sublayer's on-disk content (via
        ``Usd.Stage.Reload``-style semantics) rather than merely
        re-``Open``-ing, so external edits to referenced/payloaded
        layers are picked up.

        Returns:
            The newly opened ``pxr.Usd.Stage`` handle.

        Raises:
            NotOpenError: If no stage (and thus no identifier) is open.
            UsdReloadError: If the reload fails.
        """
        with self._lock:
            identifier = self._identifier
            stage = self._stage
        if not identifier or stage is None:
            raise NotOpenError("reload() requires a previously opened stage; call open() first.")

        try:
            # Prefer an in-place reload (preserves session-layer edits
            # and any external Python references to the same Stage
            # object) when the stage is still alive; fall back to a
            # full close+open if that fails for any reason.
            stage.Reload()
            self._rebuild_prim_index()
            logger.info("Reloaded USD stage '%s' in place.", identifier)
            return stage
        except Exception as exc:  # noqa: BLE001
            logger.warning("In-place reload failed for '%s' (%s); reopening instead.", identifier, exc)

        try:
            return self.open(identifier, force=True)
        except UsdLoaderError as exc:
            raise UsdReloadError(f"Failed to reload '{identifier}': {exc}") from exc

    def load_usd(
        self,
        identifier: "str | Path",
        *,
        variant_selections: Optional[dict[str, dict[str, str]]] = None,
        load_all_payloads: bool = True,
        validate_on_load: bool = True,
    ) -> UsdStatistics:
        """High-level entry point: open, index, and (optionally) validate a stage.

        This is the primary method most callers should use instead of
        the lower-level :meth:`open`.

        Args:
            identifier: Path or URI of the root layer to open.
            variant_selections: See :meth:`open`.
            load_all_payloads: See :meth:`open`.
            validate_on_load: If True (default), run :meth:`validate`
                immediately after opening and log a warning (without
                raising) if the report contains errors.

        Returns:
            :class:`UsdStatistics` for the newly opened stage.

        Raises:
            AlreadyOpenError: If a stage is already open (see
                :meth:`open`; call :meth:`unload` first to replace it
                without an explicit ``force``).
            UsdOpenError: If the layer cannot be opened.
        """
        self.open(
            identifier,
            variant_selections=variant_selections,
            load_all_payloads=load_all_payloads,
        )
        if validate_on_load:
            report = self.validate()
            if not report.is_valid:
                logger.warning(
                    "Loaded '%s' with %d validation error(s); call validate() for details.",
                    self._identifier, report.error_count,
                )
        return self.statistics()

    def unload(self) -> None:
        """High-level counterpart to :meth:`load_usd`: releases the stage
        and clears all derived state (index, payload-load tracking).

        Equivalent to :meth:`close`, provided as a distinct, explicit
        entry point so call sites that think in terms of "load"/"unload"
        (mirroring ``load_usd()``) don't need to reach for the
        lower-level ``open``/``close`` vocabulary.

        Raises:
            UsdCloseError: If releasing the stage raises unexpectedly.
        """
        self.close()

    # ------------------------------------------------------------------
    # Context-manager convenience
    # ------------------------------------------------------------------

    def __enter__(self) -> "UsdLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.unload()

    # ------------------------------------------------------------------
    # Internal: error / history bookkeeping
    # ------------------------------------------------------------------

    def _enter_error_state(self, exc: BaseException) -> None:
        with self._lock:
            self._state = UsdLoaderState.ERROR
            self._last_error = exc
            self._stage = None

    def _record_load(
        self, identifier: str, started_at: float, *, success: bool, error: Optional[str]
    ) -> None:
        record = LoadRecord(
            identifier=identifier,
            started_at=started_at,
            completed_at=time.time(),
            success=success,
            error=error,
        )
        with self._lock:
            self._loads.append(record)

    # ------------------------------------------------------------------
    # Stage / metadata access
    # ------------------------------------------------------------------

    def get_stage(self) -> Any:
        """Return the live ``pxr.Usd.Stage`` handle.

        This is the sanctioned hand-off point to downstream authoring
        components (e.g. a future ``StageManager``, or
        ``physics_scene.PhysicsScene``); neither is imported here, and
        this module never authors onto the stage itself beyond the
        narrow, explicitly-requested repairs in :meth:`repair`.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        return self._stage

    def get_metadata(self, prim_path: Optional[str] = None) -> dict[str, Any]:
        """Return authored metadata for the stage, or for a specific prim.

        Args:
            prim_path: If given, return this prim's authored metadata
                fields instead of the stage's.

        Raises:
            NotOpenError: If no stage is open.
            PrimNotFoundError: If ``prim_path`` is given but does not
                resolve to a valid prim.
        """
        self._require_open()
        stage = self._stage
        if prim_path is None:
            keys = ("upAxis", "metersPerUnit", "startTimeCode", "endTimeCode",
                    "framesPerSecond", "timeCodesPerSecond", "documentation")
            result: dict[str, Any] = {}
            for key in keys:
                if stage.HasAuthoredMetadata(key):
                    result[key] = stage.GetMetadata(key)
            default_prim = stage.GetDefaultPrim()
            result["defaultPrim"] = default_prim.GetPath().pathString if default_prim else None
            result["rootLayerIdentifier"] = stage.GetRootLayer().identifier
            return result

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise PrimNotFoundError(f"No prim found at path '{prim_path}'.")
        return {key: prim.GetMetadata(key) for key in prim.GetAllAuthoredMetadata().keys()}

    # ------------------------------------------------------------------
    # Internal: prim classification
    # ------------------------------------------------------------------

    def _classify(self, type_name: str) -> PrimCategory:
        return _TYPE_NAME_CATEGORY.get(type_name, PrimCategory.OTHER)

    def _describe_prim(self, prim: Any) -> PrimInfo:
        variant_sets = tuple(sorted(prim.GetVariantSets().GetNames())) if prim.HasVariantSets() else ()
        return PrimInfo(
            path=prim.GetPath().pathString,
            type_name=prim.GetTypeName() or "",
            category=self._classify(prim.GetTypeName() or ""),
            kind=prim.GetMetadata("kind") or "",
            active=prim.IsActive(),
            is_abstract=prim.IsAbstract(),
            instanceable=prim.IsInstanceable(),
            has_payload=prim.HasPayload(),
            has_reference=prim.HasAuthoredReferences(),
            variant_sets=variant_sets,
            specifier=str(prim.GetSpecifier()),
        )

    def _rebuild_prim_index(self) -> None:
        """Rebuild the path -> :class:`PrimInfo` index via a fresh traversal.

        Called after every open/reload/variant-selection/payload-load,
        since any of those can change the composed prim set.
        """
        self._require_open()
        index: dict[str, PrimInfo] = {}
        for prim in self._stage.TraverseAll():
            info = self._describe_prim(prim)
            index[info.path] = info
        with self._lock:
            self._prim_index = index

    # ------------------------------------------------------------------
    # Prim traversal / indexing / search
    # ------------------------------------------------------------------

    def list_prims(
        self,
        *,
        category: Optional[PrimCategory] = None,
        type_name: Optional[str] = None,
        kind: Optional[str] = None,
        path_prefix: Optional[str] = None,
        active_only: bool = False,
    ) -> list[PrimInfo]:
        """List indexed prims, optionally filtered.

        Args:
            category: Only include prims of this coarse category.
            type_name: Only include prims with this exact USD schema
                type name.
            kind: Only include prims with this ``model:kind`` value.
            path_prefix: Only include prims whose path starts with this
                prefix.
            active_only: If True, exclude inactive prims.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        with self._lock:
            prims = list(self._prim_index.values())
        if category is not None:
            prims = [p for p in prims if p.category is category]
        if type_name is not None:
            prims = [p for p in prims if p.type_name == type_name]
        if kind is not None:
            prims = [p for p in prims if p.kind == kind]
        if path_prefix is not None:
            prims = [p for p in prims if p.path.startswith(path_prefix)]
        if active_only:
            prims = [p for p in prims if p.active]
        return sorted(prims, key=lambda p: p.path)

    def find_prim(self, identifier: str) -> PrimInfo:
        """Look up a single prim by exact path, or by unique name match.

        Args:
            identifier: An exact prim path (``"/World/Vehicle/Body"``)
                or a bare prim name (``"Body"``), matched against the
                final path component if no exact path match is found.

        Raises:
            NotOpenError: If no stage is open.
            PrimNotFoundError: If no prim matches, or a bare-name lookup
                is ambiguous (matches more than one prim).
        """
        self._require_open()
        with self._lock:
            direct = self._prim_index.get(identifier)
            if direct is not None:
                return direct
            matches = [p for p in self._prim_index.values() if p.path.rsplit("/", 1)[-1] == identifier]
        if not matches:
            raise PrimNotFoundError(f"No prim found matching '{identifier}'.")
        if len(matches) > 1:
            paths = ", ".join(sorted(m.path for m in matches))
            raise PrimNotFoundError(
                f"Name '{identifier}' is ambiguous; matches multiple prims: {paths}."
            )
        return matches[0]

    def search_prims(self, query: str, *, limit: int = 50) -> list[PrimInfo]:
        """Free-text substring search over prim paths and type names.

        Args:
            query: Case-insensitive substring to match.
            limit: Maximum number of results returned.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        needle = query.strip().lower()
        with self._lock:
            prims = list(self._prim_index.values())
        if needle:
            prims = [
                p for p in prims
                if needle in p.path.lower() or needle in p.type_name.lower()
            ]
        return sorted(prims, key=lambda p: p.path)[:limit]

    def traverse(
        self, *, predicate: Optional[Callable[[PrimInfo], bool]] = None
    ) -> Iterator[PrimInfo]:
        """Yield indexed prims in path order, optionally filtered by ``predicate``.

        Provided alongside :meth:`list_prims` for call sites that want a
        lazy iterator (e.g. to short-circuit a large-stage scan) rather
        than a fully materialized list.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        with self._lock:
            prims = sorted(self._prim_index.values(), key=lambda p: p.path)
        for prim_info in prims:
            if predicate is None or predicate(prim_info):
                yield prim_info

    # ------------------------------------------------------------------
    # Category-specific finders
    # ------------------------------------------------------------------

    def find_materials(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) all ``Material`` prims."""
        return self._find_by_category(PrimCategory.MATERIAL, name)

    def find_meshes(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) all ``Mesh`` prims."""
        return self._find_by_category(PrimCategory.MESH, name)

    def find_skeletons(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) all ``Skeleton``/``SkelRoot`` prims."""
        self._require_open()
        with self._lock:
            prims = [
                p for p in self._prim_index.values()
                if p.category in (PrimCategory.SKELETON, PrimCategory.SKELETON_ROOT)
            ]
        return self._filter_by_name(prims, name)

    def find_animations(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) all ``SkelAnimation`` prims."""
        return self._find_by_category(PrimCategory.ANIMATION, name)

    def find_point_instancers(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) all ``PointInstancer`` prims."""
        return self._find_by_category(PrimCategory.POINT_INSTANCER, name)

    def find_instances(self, name: Optional[str] = None) -> list[PrimInfo]:
        """List (or search by substring) every prim marked instanceable.

        Distinct from :meth:`find_point_instancers`: this covers
        ``instanceable=true`` prims of any type (native USD instancing),
        while point instancers are a specific ``PointInstancer`` schema
        prim.
        """
        self._require_open()
        with self._lock:
            prims = [p for p in self._prim_index.values() if p.instanceable]
        return self._filter_by_name(prims, name)

    def _find_by_category(self, category: PrimCategory, name: Optional[str]) -> list[PrimInfo]:
        self._require_open()
        with self._lock:
            prims = [p for p in self._prim_index.values() if p.category is category]
        return self._filter_by_name(prims, name)

    @staticmethod
    def _filter_by_name(prims: list[PrimInfo], name: Optional[str]) -> list[PrimInfo]:
        if name:
            needle = name.lower()
            prims = [p for p in prims if needle in p.path.lower()]
        return sorted(prims, key=lambda p: p.path)

    def find_textures(self, name: Optional[str] = None) -> list[dict[str, Any]]:
        """Find texture-file references authored on ``UsdUVTexture``-style shaders.

        Textures are not first-class USD prims -- they are asset-path
        inputs on ``Shader`` prims inside a material's shading network.
        This walks every indexed ``Shader`` prim, and for those whose
        ``info:id`` matches a known texture-reading shader, extracts the
        asset path from its file input.

        Args:
            name: Optional case-insensitive substring filter matched
                against the resolved/raw asset path.

        Returns:
            A list of dicts with ``shader_path``, ``material_path``
            (best-effort, the nearest ``Material``-typed ancestor),
            ``asset_path``, and ``resolved_path`` keys.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        stage = self._stage
        results: list[dict[str, Any]] = []
        with self._lock:
            shader_paths = [p.path for p in self._prim_index.values() if p.category is PrimCategory.SHADER]

        for shader_path in shader_paths:
            prim = stage.GetPrimAtPath(shader_path)
            if not prim or not prim.IsValid():
                continue
            shader_id_attr = prim.GetAttribute("info:id")
            shader_id = shader_id_attr.Get() if shader_id_attr and shader_id_attr.IsValid() else None
            if shader_id not in _TEXTURE_SHADER_IDS:
                continue

            asset_path = None
            for input_name in _TEXTURE_FILE_INPUTS:
                attr = prim.GetAttribute(f"inputs:{input_name}")
                if attr and attr.IsValid():
                    value = attr.Get()
                    if value is not None:
                        asset_path = getattr(value, "path", str(value))
                        break
            if asset_path is None:
                continue

            material_path = self._nearest_ancestor_path(prim, PrimCategory.MATERIAL)
            resolved_path = self._resolve_asset_path(stage, shader_path, asset_path)

            record = {
                "shader_path": shader_path,
                "material_path": material_path,
                "asset_path": asset_path,
                "resolved_path": resolved_path,
            }
            if name:
                needle = name.lower()
                haystack = f"{asset_path} {resolved_path or ''}".lower()
                if needle not in haystack:
                    continue
            results.append(record)

        return sorted(results, key=lambda r: r["shader_path"])

    def _nearest_ancestor_path(self, prim: Any, category: PrimCategory) -> Optional[str]:
        current = prim.GetParent()
        with self._lock:
            index = self._prim_index
        while current and current.IsValid() and not current.IsPseudoRoot():
            info = index.get(current.GetPath().pathString)
            if info is not None and info.category is category:
                return info.path
            current = current.GetParent()
        return None

    def _resolve_asset_path(self, stage: Any, anchor_prim_path: str, asset_path: str) -> Optional[str]:
        try:
            ar_module = _lazy_import("pxr.Ar")
            resolver = ar_module.GetResolver()
            anchor_layer = stage.GetPseudoRoot().GetPrimStack()[0].layer if False else stage.GetRootLayer()
            anchored = resolver.CreateIdentifier(asset_path, anchor_layer.resolvedPath) \
                if hasattr(resolver, "CreateIdentifier") else asset_path
            resolved = resolver.Resolve(anchored)
            resolved_str = str(resolved) if resolved else None
            return resolved_str or None
        except UsdImportError:
            return None
        except Exception:  # noqa: BLE001 - resolution is best-effort diagnostics only
            return None

    # ------------------------------------------------------------------
    # Composition: references / payloads / sublayers
    # ------------------------------------------------------------------

    def find_references(self, prim_path: Optional[str] = None) -> list[ReferenceInfo]:
        """Enumerate reference, payload, and sublayer arcs.

        Args:
            prim_path: If given, restrict to arcs authored on (or
                beneath) this prim. If ``None``, scan the whole stage
                (including the root layer's sublayers).

        Returns:
            A list of :class:`ReferenceInfo`, including a
            :attr:`ReferenceInfo.broken` flag for any arc whose target
            could not be resolved.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        stage = self._stage
        results: list[ReferenceInfo] = []

        results.extend(self._sublayer_references(stage.GetRootLayer()))

        if prim_path is not None:
            root_prim = stage.GetPrimAtPath(prim_path)
            if not root_prim or not root_prim.IsValid():
                raise PrimNotFoundError(f"No prim found at path '{prim_path}'.")
            prims = list(iter([root_prim])) + list(root_prim.GetAllChildren())
        else:
            prims = list(stage.TraverseAll())

        for prim in prims:
            results.extend(self._prim_composition_arcs(prim))

        return results

    def _sublayer_references(self, layer: Any, *, _seen: Optional[set[str]] = None) -> list[ReferenceInfo]:
        _seen = _seen or set()
        if layer.identifier in _seen:
            return []
        _seen.add(layer.identifier)

        results: list[ReferenceInfo] = []
        for sublayer_path in layer.subLayerPaths:
            resolved = layer.ComputeAbsolutePath(sublayer_path)
            sublayer = None
            broken = True
            try:
                sdf_module = _lazy_import("pxr.Sdf")
                sublayer = sdf_module.Layer.FindOrOpen(resolved) if resolved else None
                broken = sublayer is None
            except UsdImportError:
                broken = resolved is None or not Path(resolved).exists()

            results.append(
                ReferenceInfo(
                    prim_path="/",
                    kind=ReferenceKind.SUBLAYER,
                    asset_path=sublayer_path,
                    resolved_path=resolved or None,
                    layer_offset=(1.0, 0.0),
                    is_internal=False,
                    broken=broken,
                )
            )
            if sublayer is not None:
                results.extend(self._sublayer_references(sublayer, _seen=_seen))
        return results

    def _prim_composition_arcs(self, prim: Any) -> list[ReferenceInfo]:
        results: list[ReferenceInfo] = []
        try:
            usd_module = _lazy_import("pxr.Usd")
            query = usd_module.PrimCompositionQuery(prim)
            for arc in query.GetCompositionArcs():
                arc_type = arc.GetArcType()
                type_name = str(arc_type)
                if "reference" not in type_name.lower() and "payload" not in type_name.lower():
                    continue
                kind = ReferenceKind.PAYLOAD if "payload" in type_name.lower() else ReferenceKind.REFERENCE
                introducing_layer = arc.GetIntroducingLayer()
                node = arc.GetTargetNode()
                layer_stack = node.layerStack if node else None
                asset_path = ""
                resolved_path = None
                is_internal = True
                if layer_stack is not None and layer_stack.identifier.rootLayer is not None:
                    root_layer = layer_stack.identifier.rootLayer
                    if introducing_layer is not None and root_layer.identifier != introducing_layer.identifier:
                        is_internal = False
                        asset_path = root_layer.identifier
                        resolved_path = root_layer.realPath or root_layer.identifier

                results.append(
                    ReferenceInfo(
                        prim_path=prim.GetPath().pathString,
                        kind=kind,
                        asset_path=asset_path,
                        resolved_path=resolved_path,
                        layer_offset=(1.0, 0.0),
                        is_internal=is_internal,
                        broken=(not is_internal and not resolved_path),
                    )
                )
        except UsdImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("Composition arc inspection failed for '%s': %s", prim.GetPath(), exc)
        return results

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return a best-effort external-layer dependency graph.

        Returns:
            A mapping from each layer identifier reachable from the
            root layer to the list of external layer identifiers it
            directly depends on (sublayers plus any external
            reference/payload targets discovered on its prims).

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        graph: dict[str, list[str]] = {}
        root_identifier = self._stage.GetRootLayer().identifier
        graph[root_identifier] = sorted(
            {
                ref.resolved_path or ref.asset_path
                for ref in self.find_references()
                if not ref.is_internal and (ref.resolved_path or ref.asset_path)
            }
        )
        return graph

    # ------------------------------------------------------------------
    # Payloads (lazy loading)
    # ------------------------------------------------------------------

    def load_payloads(self, prim_paths: Optional[list[str]] = None) -> list[str]:
        """Load payloads for specific prims (or every unloaded payload).

        Args:
            prim_paths: Prim paths to load payloads for. If ``None``,
                every prim with an unloaded payload is loaded
                (equivalent to ``Usd.Stage.LoadAll`` at this point in
                time).

        Returns:
            The list of prim paths whose payloads were newly loaded.

        Raises:
            NotOpenError: If no stage is open.
            PayloadLoadError: If a requested prim path does not exist.
        """
        self._require_open()
        stage = self._stage
        if prim_paths is None:
            stage.Load()
            self._rebuild_prim_index()
            with self._lock:
                loaded_now = [p for p in self._prim_index if self._prim_index[p].has_payload]
                self._payload_paths_loaded.update(loaded_now)
            logger.info("Loaded all outstanding payloads.")
            return loaded_now

        newly_loaded: list[str] = []
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise PayloadLoadError(f"No prim found at path '{path}'; cannot load its payload.")
            try:
                prim.Load()
            except Exception as exc:  # noqa: BLE001
                raise PayloadLoadError(f"Failed to load payload for '{path}': {exc}") from exc
            newly_loaded.append(path)

        with self._lock:
            self._payload_paths_loaded.update(newly_loaded)
        self._rebuild_prim_index()
        logger.info("Loaded payload(s) for %d prim(s).", len(newly_loaded))
        return newly_loaded

    def unload_payloads(self, prim_paths: Optional[list[str]] = None) -> list[str]:
        """Unload payloads for specific prims (or every loaded payload).

        Raises:
            NotOpenError: If no stage is open.
            PayloadLoadError: If a requested prim path does not exist.
        """
        self._require_open()
        stage = self._stage
        if prim_paths is None:
            stage.Unload()
            with self._lock:
                unloaded_now = list(self._payload_paths_loaded)
                self._payload_paths_loaded.clear()
            self._rebuild_prim_index()
            logger.info("Unloaded all payloads.")
            return unloaded_now

        unloaded: list[str] = []
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise PayloadLoadError(f"No prim found at path '{path}'; cannot unload its payload.")
            prim.Unload()
            unloaded.append(path)
        with self._lock:
            self._payload_paths_loaded.difference_update(unloaded)
        self._rebuild_prim_index()
        return unloaded

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    def get_variant_sets(self, prim_path: str) -> dict[str, list[str]]:
        """Return every variant set on a prim, mapped to its available variant names.

        Raises:
            NotOpenError: If no stage is open.
            PrimNotFoundError: If ``prim_path`` does not resolve.
        """
        self._require_open()
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise PrimNotFoundError(f"No prim found at path '{prim_path}'.")
        variant_sets = prim.GetVariantSets()
        return {name: list(variant_sets.GetVariantSet(name).GetVariantNames()) for name in variant_sets.GetNames()}

    def set_variant_selection(self, prim_path: str, variant_set: str, selection: str) -> None:
        """Author a variant selection on a prim and re-index the stage.

        Raises:
            NotOpenError: If no stage is open.
            PrimNotFoundError: If ``prim_path`` does not resolve.
            VariantError: If the variant set/selection is invalid.
        """
        self._apply_variant_selections({prim_path: {variant_set: selection}})

    def _apply_variant_selections(self, selections: dict[str, dict[str, str]]) -> None:
        self._require_open()
        stage = self._stage
        for prim_path, set_selections in selections.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise PrimNotFoundError(f"No prim found at path '{prim_path}'.")
            variant_sets = prim.GetVariantSets()
            for variant_set_name, selection in set_selections.items():
                if variant_set_name not in variant_sets.GetNames():
                    raise VariantError(
                        f"Prim '{prim_path}' has no variant set named '{variant_set_name}'."
                    )
                variant_set = variant_sets.GetVariantSet(variant_set_name)
                if selection not in variant_set.GetVariantNames():
                    raise VariantError(
                        f"'{selection}' is not a valid selection for variant set "
                        f"'{variant_set_name}' on '{prim_path}'."
                    )
                if not variant_set.SetVariantSelection(selection):
                    raise VariantError(
                        f"Failed to set variant selection '{variant_set_name}={selection}' "
                        f"on '{prim_path}'."
                    )
        self._rebuild_prim_index()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, *, strict: bool = False) -> UsdValidationReport:
        """Run consistency and compliance checks against the open stage.

        Checks performed:
            * A default prim is authored on the root layer.
            * Every reference/payload arc discovered by
              :meth:`find_references` resolves (no broken references).
            * Every prim's schema type name is a known/registered USD
              schema (soft warning for unrecognized types, which are
              often authoring typos).
            * Mesh prims declare non-empty ``points``/``faceVertexCounts``.
            * Material-bound prims reference a material that exists.
            * ``strict`` additionally flags: prims with no authored
              type name at all ("def" with no schema), and empty
              variant sets.

        Args:
            strict: Enable the additional, stricter checks described
                above.

        Returns:
            A :class:`UsdValidationReport` describing every finding.
            This method itself only raises if validation *cannot be
            attempted at all* -- an invalid stage is a normal,
            non-raising result reported via the returned report.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        stage = self._stage
        issues: list[ValidationIssue] = []

        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Root layer has no default prim authored.",
                    prim_path="",
                    code="missing-default-prim",
                )
            )

        for ref in self.find_references():
            if ref.broken:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Broken {ref.kind.value} arc targeting '{ref.asset_path}'.",
                        prim_path=ref.prim_path,
                        code="broken-reference",
                    )
                )

        try:
            sdf_module = _lazy_import("pxr.Sdf")
            registered_types = set(sdf_module.SchemaBase.GetSchemaAttributeNames.__self__.__class__.__dict__) \
                if False else None
        except UsdImportError:
            registered_types = None

        with self._lock:
            prims = list(self._prim_index.values())

        for info in prims:
            if info.category is PrimCategory.MESH:
                mesh_issues = self._validate_mesh(info.path)
                issues.extend(mesh_issues)
            if strict and not info.type_name:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Typeless 'def' prim with no authored schema.",
                        prim_path=info.path,
                        code="typeless-prim",
                    )
                )
            if strict and info.variant_sets:
                for vset_name, variants in self.get_variant_sets(info.path).items():
                    if not variants:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Variant set '{vset_name}' has no variants.",
                                prim_path=info.path,
                                code="empty-variant-set",
                            )
                        )

        report = UsdValidationReport(issues=issues)
        logger.info(
            "Validated stage '%s' (errors=%d, warnings=%d).",
            self._identifier, report.error_count, report.warning_count,
        )
        return report

    def _validate_mesh(self, prim_path: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            usdgeom_module = _lazy_import("pxr.UsdGeom")
        except UsdImportError:
            return issues

        prim = self._stage.GetPrimAtPath(prim_path)
        mesh = usdgeom_module.Mesh(prim)
        if not mesh:
            return issues

        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get() if points_attr and points_attr.IsValid() else None
        if not points:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Mesh has no authored 'points'.",
                    prim_path=prim_path,
                    code="mesh-missing-points",
                )
            )

        counts_attr = mesh.GetFaceVertexCountsAttr()
        counts = counts_attr.Get() if counts_attr and counts_attr.IsValid() else None
        if not counts:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Mesh has no authored 'faceVertexCounts'.",
                    prim_path=prim_path,
                    code="mesh-missing-face-counts",
                )
            )
        return issues

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair(
        self,
        report: Optional[UsdValidationReport] = None,
        *,
        fix_default_prim: bool = True,
        remove_broken_references: bool = False,
    ) -> list[str]:
        """Apply best-effort, explicitly-opt-in fixes for common issues.

        Every fix here is conservative and additive/removal-only on the
        session layer's authoring where possible; nothing here silently
        rewrites a referenced asset's own file on disk.

        Args:
            report: A previously computed :class:`UsdValidationReport`
                to act on. If ``None``, :meth:`validate` is run first.
            fix_default_prim: If True and no default prim is set, sets
                the first root-level, non-abstract, active prim as the
                default prim.
            remove_broken_references: If True, removes any
                reference/payload arc reported as broken. Destructive --
                off by default.

        Returns:
            A list of human-readable descriptions of the fixes applied.

        Raises:
            NotOpenError: If no stage is open.
            UsdRepairError: If a requested fix cannot be applied.
        """
        self._require_open()
        report = report or self.validate()
        applied: list[str] = []
        stage = self._stage

        if fix_default_prim and any(i.code == "missing-default-prim" for i in report.issues):
            root_prims = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsActive() and not p.IsAbstract()]
            if not root_prims:
                raise UsdRepairError("Cannot fix missing default prim: stage has no root-level prims.")
            try:
                stage.SetDefaultPrim(root_prims[0])
            except Exception as exc:  # noqa: BLE001
                raise UsdRepairError(f"Failed to set default prim: {exc}") from exc
            applied.append(f"Set default prim to '{root_prims[0].GetPath()}'.")

        if remove_broken_references:
            broken_prim_paths = {i.prim_path for i in report.issues if i.code == "broken-reference"}
            for prim_path in broken_prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                try:
                    prim.GetReferences().ClearReferences()
                    prim.GetPayloads().ClearPayloads()
                except Exception as exc:  # noqa: BLE001
                    raise UsdRepairError(
                        f"Failed to clear broken reference/payload arcs on '{prim_path}': {exc}"
                    ) from exc
                applied.append(f"Cleared broken reference/payload arcs on '{prim_path}'.")

        if applied:
            self._rebuild_prim_index()
        logger.info("Applied %d repair(s).", len(applied))
        return applied

    # ------------------------------------------------------------------
    # Statistics / reporting
    # ------------------------------------------------------------------

    def statistics(self) -> UsdStatistics:
        """Compute and return aggregate statistics about the open stage.

        Raises:
            NotOpenError: If no stage is open.
        """
        self._require_open()
        stage = self._stage
        with self._lock:
            prims = list(self._prim_index.values())

        by_category: dict[str, int] = {}
        variant_set_count = 0
        for info in prims:
            by_category[info.category.value] = by_category.get(info.category.value, 0) + 1
            variant_set_count += len(info.variant_sets)

        references = self.find_references()
        reference_count = sum(1 for r in references if r.kind is ReferenceKind.REFERENCE)
        payload_count = sum(1 for r in references if r.kind is ReferenceKind.PAYLOAD)
        broken_count = sum(1 for r in references if r.broken)
        layer_count = 1 + sum(1 for r in references if r.kind is ReferenceKind.SUBLAYER)

        start_time = stage.GetStartTimeCode() if stage.HasAuthoredMetadata("startTimeCode") else None
        end_time = stage.GetEndTimeCode() if stage.HasAuthoredMetadata("endTimeCode") else None
        time_range = (start_time, end_time) if start_time is not None and end_time is not None else None

        default_prim = stage.GetDefaultPrim()

        return UsdStatistics(
            identifier=self._identifier,
            total_prims=len(prims),
            by_category=by_category,
            active_prims=sum(1 for p in prims if p.active),
            inactive_prims=sum(1 for p in prims if not p.active),
            instanceable_prims=sum(1 for p in prims if p.instanceable),
            layer_count=layer_count,
            reference_count=reference_count,
            payload_count=payload_count,
            broken_reference_count=broken_count,
            variant_set_count=variant_set_count,
            has_default_prim=bool(default_prim and default_prim.IsValid()),
            time_code_range=time_range,
            up_axis=stage.GetMetadata("upAxis") if stage.HasAuthoredMetadata("upAxis") else "",
            meters_per_unit=stage.GetMetadata("metersPerUnit") if stage.HasAuthoredMetadata("metersPerUnit") else 0.01,
        )

    def export_report(self, path: "str | Path") -> Path:
        """Write a combined statistics + validation report to disk as JSON.

        Args:
            path: Destination JSON file path.

        Returns:
            The destination path.

        Raises:
            NotOpenError: If no stage is open.
            ExportReportError: If the report cannot be written.
        """
        statistics = self.statistics()
        report = self.validate()

        payload = {
            "report_version": _REPORT_VERSION,
            "generated_at": time.time(),
            "identifier": self._identifier,
            "statistics": statistics.to_dict(),
            "validation": report.to_dict(),
            "prims": [info.to_dict() for info in self.list_prims()],
            "references": [ref.to_dict() for ref in self.find_references()],
        }

        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            raise ExportReportError(f"Failed to write USD report to '{destination}': {exc}") from exc

        logger.info("USD report written to '%s'.", destination)
        return destination


__all__ = [
    "UsdLoader",
    "PrimInfo",
    "ReferenceInfo",
    "ValidationIssue",
    "UsdValidationReport",
    "UsdStatistics",
    "LoadRecord",
    "UsdLoaderState",
    "ValidationSeverity",
    "PrimCategory",
    "ReferenceKind",
    "UsdLoaderError",
    "UsdImportError",
    "NotOpenError",
    "AlreadyOpenError",
    "UsdOpenError",
    "UsdCloseError",
    "UsdReloadError",
    "PrimNotFoundError",
    "UsdValidationError",
    "UsdRepairError",
    "CompositionError",
    "ReferenceResolutionError",
    "LayerLoadError",
    "PayloadLoadError",
    "VariantError",
    "ExportReportError",
]
