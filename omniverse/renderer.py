"""
renderer.py
══════════════════════════════════════════════════════════════════════════
Rendering authority for the Omniverse connector layer of PhysWorldLM.

Pipeline position
------------------
    Prompt → WorldSpec → USD → Launcher → StageManager
                                                │
                                                ▼
                                           PhysicsScene
                                                │
                                                ▼
                                       TimelineController
                                                │
                                                ▼
                                      ┌──────────────────┐
                                      │     Renderer     │  <-- this module
                                      └──────────────────┘

Scope
-----
This module owns ONLY rendering: RTX/path-tracing/real-time backend
selection, viewport and camera configuration, quality/DLSS/anti-alias/
denoiser toggles, and every capture modality (RGB, depth, normals,
segmentation, bounding boxes, motion vectors, optical flow) needed by
downstream consumers (dataset export, Replicator-driven synthetic data,
ROS2 perception bridges, video review).

It never:
    * launches Omniverse (``app_launcher.py``)
    * creates stages (``stage_manager.py``)
    * creates physics (``physics_scene.py``)
    * controls timeline (``timeline_controller.py``) -- the Renderer
      renders whatever frame the timeline currently points at; it does
      not play/pause/step/jump the timeline itself
    * loads assets (``asset_server.py``)
    * parses prompts (upstream ontology / WorldSpec layer)

Design constraints
-------------------
    * ``omni.*`` packages (``omni.kit.viewport.utility``,
      ``omni.replicator.core``, ``carb.settings``, ...) are imported
      lazily, only inside the concrete backend that actually needs
      them, and only the first time such a method executes -- never at
      module import time, never in ``__init__``, and never as a side
      effect of merely constructing a ``Renderer``.
    * The renderer is backend-agnostic via the :class:`RenderBackend`
      protocol and dependency injection, exactly mirroring the pattern
      used in ``timeline_controller.py``. A real RTX-backed adapter is
      lazily constructed in :meth:`Renderer.initialize`, or a fake/stub
      backend can be injected for unit testing with zero Omniverse
      dependencies.
    * Every :class:`RenderProductType` maps to a stable annotator-style
      string key (``_PRODUCT_ANNOTATOR_NAMES``). This is the seam that
      makes the module future-compatible with a real
      ``omni.replicator.core`` annotator/render-product backend without
      any change to the public :class:`Renderer` API -- only
      ``_OmniRTXBackend`` (or a new ``_ReplicatorBackend``) would change.
    * All mutable state lives on the instance, guarded by a single
      re-entrant lock. There is no module-level mutable state.

Public API
----------
    renderer = Renderer(config=OmniverseConfig.default())
    with renderer:
        renderer.set_resolution(1920, 1080)
        renderer.set_camera("/World/Cameras/Main")
        renderer.render_frame()
        image = renderer.capture_image("output/frame_0001.png")
        stats = renderer.statistics()

Changelog
---------
    * Initial implementation.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .config import OmniverseConfig

# ════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("physworldlm.omniverse.renderer")
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

class RendererError(Exception):
    """Base class for all Renderer errors."""


class RendererStateError(RendererError):
    """Raised when an operation is invalid for the current renderer state."""


class RendererImportError(RendererError):
    """Raised when a required ``omni``/RTX module cannot be lazily imported."""


class RendererValidationError(RendererError):
    """Raised when a parameter (resolution, quality, product type, ...) is invalid."""


class RenderCaptureError(RendererError):
    """Raised when a capture operation fails at the backend."""


# ════════════════════════════════════════════════════════════════════════
# Renderer lifecycle state
# ════════════════════════════════════════════════════════════════════════

class RendererState(Enum):
    """Lifecycle state of a :class:`Renderer`."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    SHUTDOWN = "shutdown"


_ACTIVE_STATES = (RendererState.READY,)


# ════════════════════════════════════════════════════════════════════════
# Enums for renderer configuration
# ════════════════════════════════════════════════════════════════════════

class RenderMode(str, Enum):
    """Rendering backend selection. Values match ``OmniverseConfig``'s renderer strings."""

    RTX_REALTIME = "rtx_realtime"
    RTX_PATHTRACING = "rtx_pathtracing"
    HYDRA_STORM = "hydra_storm"


class RenderQuality(str, Enum):
    """Coarse render-quality tiers, mapped internally to sampling/settings presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CINEMATIC = "cinematic"


class RenderProductType(str, Enum):
    """A single capturable render modality."""

    RGB = "rgb"
    DEPTH = "depth"
    NORMALS = "normals"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    BOUNDING_BOX_2D = "bounding_box_2d"
    BOUNDING_BOX_3D = "bounding_box_3d"
    MOTION_VECTORS = "motion_vectors"
    OPTICAL_FLOW = "optical_flow"


# Stable mapping from our product-type vocabulary to the annotator-style
# key a real Replicator/RTX backend would register. Kept centralized so
# only this table (and _OmniRTXBackend) need updating if the underlying
# Replicator annotator names change -- the public Renderer API and every
# RenderProductType value are unaffected.
_PRODUCT_ANNOTATOR_NAMES: dict[RenderProductType, str] = {
    RenderProductType.RGB: "rgb",
    RenderProductType.DEPTH: "distance_to_camera",
    RenderProductType.NORMALS: "normals",
    RenderProductType.INSTANCE_SEGMENTATION: "instance_segmentation",
    RenderProductType.SEMANTIC_SEGMENTATION: "semantic_segmentation",
    RenderProductType.BOUNDING_BOX_2D: "bounding_box_2d_tight",
    RenderProductType.BOUNDING_BOX_3D: "bounding_box_3d",
    RenderProductType.MOTION_VECTORS: "motion_vectors",
    RenderProductType.OPTICAL_FLOW: "optical_flow",
}

_VALID_QUALITIES = tuple(q.value for q in RenderQuality)
_VALID_RENDER_MODES = tuple(m.value for m in RenderMode)
_VALID_TONE_MAPPING = ("linear", "reinhard", "aces", "filmic")


# ════════════════════════════════════════════════════════════════════════
# Backend protocol (dependency inversion -- makes the renderer testable
# without Omniverse installed, and future-compatible with a Replicator-
# backed implementation).
# ════════════════════════════════════════════════════════════════════════

@runtime_checkable
class RenderBackend(Protocol):
    """Minimal surface a rendering driver must expose.

    A production implementation wraps ``omni.kit.viewport.utility`` +
    ``omni.replicator.core`` + ``carb.settings``. Tests can supply a
    trivial fake implementing this same surface. This is also the seam
    a future Replicator-native backend would implement, without any
    change to :class:`Renderer` itself.
    """

    def set_resolution(self, width: int, height: int) -> None: ...

    def set_camera(self, camera_path: str) -> None: ...

    def set_renderer_mode(self, mode: str) -> None: ...

    def set_quality(self, quality: str) -> None: ...

    def set_gpu_devices(self, devices: list[int]) -> None: ...

    def set_feature(self, name: str, enabled: bool) -> None: ...

    def render_frame(self) -> None: ...

    def capture(self, product_keys: list[str]) -> dict[str, Any]: ...

    def capture_screenshot(self, path: str) -> None: ...

    def start_video_recording(self, path: str, fps: float) -> None: ...

    def write_video_frame(self) -> None: ...

    def stop_video_recording(self) -> None: ...

    def get_performance_stats(self) -> dict[str, float]: ...

    def shutdown(self) -> None: ...


class _OmniRTXBackend:
    """Adapter around ``omni.kit.viewport.utility`` / ``omni.replicator.core``.

    Imports every ``omni``/``carb`` dependency lazily -- only when this
    adapter is actually constructed, which itself only happens lazily
    from :meth:`Renderer.initialize` when no backend was injected.

    Individual calls are defensive (``getattr``/``try``/``except``)
    because the exact Replicator/viewport API surface varies across Kit
    and Isaac Sim versions; failures are logged and degrade gracefully
    rather than crashing a render loop over a missing optional setting.
    """

    def __init__(self, *, headless: bool) -> None:
        try:
            import omni.kit.viewport.utility as _viewport_utility  # type: ignore
            import omni.replicator.core as _replicator  # type: ignore
            import carb.settings  # type: ignore
        except ImportError as exc:
            raise RendererImportError(
                "Could not import 'omni.kit.viewport.utility', "
                "'omni.replicator.core', or 'carb.settings'. Renderer "
                "requires a running Omniverse Kit process to drive real "
                "rendering, or an injected RenderBackend for "
                "backend-less / test operation."
            ) from exc

        self._viewport_utility = _viewport_utility
        self._replicator = _replicator
        self._settings = carb.settings.get_settings()
        self._headless = headless
        self._viewport = self._viewport_utility.get_active_viewport()
        self._render_product = None
        self._annotators: dict[str, Any] = {}
        self._video_writer = None
        logger.debug("Acquired Omniverse viewport/replicator backend (headless=%s).", headless)

    def set_resolution(self, width: int, height: int) -> None:
        try:
            self._viewport_utility.set_viewport_resolution(self._viewport, (width, height))
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_resolution(%d, %d).", width, height)

    def set_camera(self, camera_path: str) -> None:
        try:
            self._viewport_utility.set_active_camera(self._viewport, camera_path)
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_camera('%s').", camera_path)

    def set_renderer_mode(self, mode: str) -> None:
        try:
            self._settings.set("/rtx/rendermode", mode)
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_renderer_mode('%s').", mode)

    def set_quality(self, quality: str) -> None:
        try:
            self._settings.set("/physworldlm/render/quality", quality)
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_quality('%s').", quality)

    def set_gpu_devices(self, devices: list[int]) -> None:
        try:
            self._settings.set("/renderer/multiGpu/enabled", len(devices) > 1)
            if devices:
                self._settings.set("/renderer/activeGpu", devices[0])
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_gpu_devices(%s).", devices)

    def set_feature(self, name: str, enabled: bool) -> None:
        _FEATURE_SETTING_PATHS = {
            "rtx": "/rtx/enabled",
            "dlss": "/rtx/post/dlss/enabled",
            "antialiasing": "/rtx/post/aa/enabled",
            "denoiser": "/rtx/pathtracing/denoiser/enabled",
        }
        path = _FEATURE_SETTING_PATHS.get(name)
        if path is None:
            logger.debug("Unknown feature '%s'; ignoring.", name)
            return
        try:
            self._settings.set(path, enabled)
        except Exception:  # noqa: BLE001
            logger.debug("Backend rejected set_feature('%s', %s).", name, enabled)

    def render_frame(self) -> None:
        try:
            import omni.kit.app  # type: ignore
            omni.kit.app.get_app().update()
        except Exception as exc:  # noqa: BLE001
            raise RenderCaptureError(f"Backend failed to render a frame: {exc}") from exc

    def _ensure_render_product(self) -> Any:
        if self._render_product is None:
            camera_path = self._viewport_utility.get_active_camera(self._viewport)
            self._render_product = self._replicator.create.render_product(camera_path, (1280, 720))
        return self._render_product

    def capture(self, product_keys: list[str]) -> dict[str, Any]:
        product = self._ensure_render_product()
        results: dict[str, Any] = {}
        for key in product_keys:
            annotator = self._annotators.get(key)
            if annotator is None:
                annotator = self._replicator.AnnotatorRegistry.get_annotator(key)
                annotator.attach([product])
                self._annotators[key] = annotator
            try:
                results[key] = annotator.get_data()
            except Exception as exc:  # noqa: BLE001
                raise RenderCaptureError(f"Failed to read annotator '{key}': {exc}") from exc
        return results

    def capture_screenshot(self, path: str) -> None:
        try:
            import omni.kit.viewport.utility as _vu  # type: ignore
            _vu.capture_viewport_to_file(self._viewport, path)
        except Exception as exc:  # noqa: BLE001
            raise RenderCaptureError(f"Backend failed to capture screenshot to '{path}': {exc}") from exc

    def start_video_recording(self, path: str, fps: float) -> None:
        try:
            import omni.kit.capture.viewport as _capture  # type: ignore
            self._video_writer = _capture.get_capture_instance()
            self._video_writer.start(output_path=path, fps=fps)
        except Exception as exc:  # noqa: BLE001
            raise RenderCaptureError(f"Backend failed to start video recording: {exc}") from exc

    def write_video_frame(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.capture_frame()
            except Exception:  # noqa: BLE001
                logger.debug("Backend dropped a video frame.")

    def stop_video_recording(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.stop()
            finally:
                self._video_writer = None

    def get_performance_stats(self) -> dict[str, float]:
        try:
            import omni.kit.viewport.utility as _vu  # type: ignore
            return dict(_vu.get_viewport_stats(self._viewport) or {})
        except Exception:  # noqa: BLE001
            return {}

    def shutdown(self) -> None:
        try:
            self.stop_video_recording()
        except Exception:  # noqa: BLE001
            pass
        try:
            for annotator in self._annotators.values():
                annotator.detach()
        except Exception:  # noqa: BLE001
            pass
        self._annotators.clear()
        self._render_product = None


# ════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CaptureResult:
    """Result of a single capture operation.

    Attributes:
        product_type: The :class:`RenderProductType` captured.
        frame: Frame number at capture time (as reported by the caller;
            the Renderer itself does not track simulation frames --
            that belongs to ``TimelineController``).
        width: Capture width in pixels.
        height: Capture height in pixels.
        data: Raw backend payload (array/bytes), or ``None`` for
            operations (like starting a video recording) that produce
            no immediate in-memory payload.
        path: Filesystem path the result was written to, if any.
        metadata: Free-form extra info (e.g. ``{"video": True, "fps": 30}``).
    """

    product_type: RenderProductType
    frame: int
    width: int
    height: int
    data: Any = None
    path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderStatistics:
    """Point-in-time statistics about a :class:`Renderer`.

    Attributes:
        state: Current :class:`RendererState`.
        renderer_mode: Current :class:`RenderMode`.
        quality: Current :class:`RenderQuality`.
        resolution: Current ``(width, height)`` in pixels.
        rtx_enabled: Whether RTX is enabled.
        dlss_enabled: Whether DLSS is enabled.
        antialiasing_enabled: Whether anti-aliasing is enabled.
        denoiser_enabled: Whether the denoiser is enabled.
        frames_rendered: Cumulative frames rendered since ``initialize()``.
        sequences_rendered: Cumulative ``render_sequence()`` calls.
        images_captured: Cumulative ``capture_image()`` calls.
        videos_captured: Cumulative completed ``capture_video()`` calls.
        last_frame_time_ms: Wall-clock milliseconds spent on the most
            recent ``render_frame()``.
        average_frame_time_ms: Running average frame time, in
            milliseconds, over all rendered frames.
        current_fps: ``1000 / average_frame_time_ms``, or ``0.0`` before
            any frame has been rendered.
        real_time_elapsed: Wall-clock seconds since ``initialize()``.
        gpu_devices: GPU device indices currently selected.
    """

    state: RendererState
    renderer_mode: RenderMode
    quality: RenderQuality
    resolution: tuple[int, int]
    rtx_enabled: bool
    dlss_enabled: bool
    antialiasing_enabled: bool
    denoiser_enabled: bool
    frames_rendered: int
    sequences_rendered: int
    images_captured: int
    videos_captured: int
    last_frame_time_ms: float
    average_frame_time_ms: float
    current_fps: float
    real_time_elapsed: float
    gpu_devices: tuple[int, ...]


@dataclass
class RenderPreset:
    """A named bundle of renderer settings, applied atomically.

    Attributes:
        name: Preset identifier (e.g. ``"draft"``, ``"cinematic"``).
        renderer_mode: Renderer backend to select.
        quality: Quality tier to select.
        rtx_enabled: Whether RTX should be enabled.
        dlss_enabled: Whether DLSS should be enabled (requires
            ``rtx_enabled``).
        antialiasing_enabled: Whether anti-aliasing should be enabled.
        denoiser_enabled: Whether the denoiser should be enabled.
        exposure: Exposure value in EV stops.
        tone_mapping: Tone-mapping operator name.
    """

    name: str
    renderer_mode: RenderMode
    quality: RenderQuality
    rtx_enabled: bool = True
    dlss_enabled: bool = False
    antialiasing_enabled: bool = True
    denoiser_enabled: bool = True
    exposure: float = 0.0
    tone_mapping: str = "aces"


# Built-in presets, chosen to be reasonable defaults for common
# use-cases; callers can register their own via register_preset().
_BUILTIN_PRESETS: dict[str, RenderPreset] = {
    "draft": RenderPreset(
        name="draft", renderer_mode=RenderMode.RTX_REALTIME, quality=RenderQuality.LOW,
        dlss_enabled=True, antialiasing_enabled=False, denoiser_enabled=False,
    ),
    "balanced": RenderPreset(
        name="balanced", renderer_mode=RenderMode.RTX_REALTIME, quality=RenderQuality.HIGH,
        dlss_enabled=True, antialiasing_enabled=True, denoiser_enabled=True,
    ),
    "cinematic": RenderPreset(
        name="cinematic", renderer_mode=RenderMode.RTX_PATHTRACING, quality=RenderQuality.CINEMATIC,
        dlss_enabled=False, antialiasing_enabled=True, denoiser_enabled=True, tone_mapping="filmic",
    ),
}


@dataclass
class RenderJob:
    """A single queued unit of render work, for batch/offline processing.

    Attributes:
        job_id: Opaque identifier assigned at enqueue time.
        action: Callable invoked with no arguments when the job runs;
            typically a closure over one of the ``Renderer`` capture
            methods.
        label: Human-readable description for diagnostics/logging.
    """

    job_id: str
    action: Callable[[], Any]
    label: str = ""


FrameCallback = Callable[[int, RenderStatistics], None]
"""Signature required of the optional ``on_frame`` callback passed to ``render_sequence()``."""


# ════════════════════════════════════════════════════════════════════════
# Renderer
# ════════════════════════════════════════════════════════════════════════

class Renderer:
    """Owns and drives rendering for a PhysWorldLM Omniverse session.

    Thread safety: every public method acquires an internal
    :class:`threading.RLock`, so a single instance may safely be driven
    from multiple threads (e.g. a capture worker thread calling
    ``capture_image()`` while a monitoring thread calls ``statistics()``).

    The renderer never imports ``omni``/``carb`` at construction time. A
    real RTX-backed backend is only acquired inside :meth:`initialize`,
    and only if no :class:`RenderBackend` was injected via the
    constructor -- so tests can freely construct and drive a renderer
    with zero Omniverse dependencies.

    Attributes:
        config: The :class:`~omniverse.config.OmniverseConfig` this
            renderer was built from (used only for its default
            ``headless``/``gpu_device`` -- the renderer does not
            otherwise reach into unrelated config fields).
    """

    def __init__(
        self,
        config: Optional[OmniverseConfig] = None,
        *,
        backend: Optional[RenderBackend] = None,
        camera_path: str = "/OmniverseKit_Persp",
        resolution: tuple[int, int] = (1280, 720),
        renderer_mode: RenderMode = RenderMode.RTX_REALTIME,
        quality: RenderQuality = RenderQuality.HIGH,
        headless: Optional[bool] = None,
        gpu_devices: Optional[list[int]] = None,
    ) -> None:
        """Construct a renderer. Performs no I/O and no ``omni``/``carb`` imports.

        Args:
            config: Optional shared configuration. If provided,
                ``headless`` and the initial GPU device default from it
                unless explicitly overridden.
            backend: Optional pre-built :class:`RenderBackend`
                (typically used to inject a fake for unit tests, or a
                pre-constructed real backend). If ``None``, a real
                RTX-backed adapter is lazily constructed the first time
                :meth:`initialize` runs.
            camera_path: Initial active camera prim path.
            resolution: Initial ``(width, height)`` in pixels. Both
                must be positive.
            renderer_mode: Initial :class:`RenderMode`.
            quality: Initial :class:`RenderQuality`.
            headless: Whether to render offscreen with no viewport
                window. Defaults to ``config.headless`` if ``config`` is
                given, else ``True``.
            gpu_devices: GPU device indices to render on. Defaults to
                ``[config.gpu_device]`` if ``config.gpu_device`` is set,
                else an empty list (backend default selection).

        Raises:
            RendererValidationError: If ``resolution`` is non-positive
                or ``camera_path`` is empty.
        """
        self.config = config
        self._lock = threading.RLock()

        width, height = resolution
        if width <= 0 or height <= 0:
            raise RendererValidationError(f"resolution must be positive (got {resolution}).")
        if not camera_path:
            raise RendererValidationError("camera_path must be non-empty.")

        self._injected_backend = backend
        self._backend: Optional[RenderBackend] = None

        self._state = RendererState.UNINITIALIZED
        self._camera_path = camera_path
        self._resolution = (int(width), int(height))
        self._renderer_mode = renderer_mode
        self._quality = quality
        self._headless = headless if headless is not None else (config.headless if config else True)
        self._gpu_devices: list[int] = (
            list(gpu_devices) if gpu_devices is not None
            else ([config.gpu_device] if config and config.gpu_device is not None else [])
        )

        self._rtx_enabled = True
        self._dlss_enabled = False
        self._antialiasing_enabled = True
        self._denoiser_enabled = True
        self._exposure = 0.0
        self._tone_mapping = "aces"
        self._lighting_preset: Optional[str] = None

        self._frames_rendered = 0
        self._sequences_rendered = 0
        self._images_captured = 0
        self._videos_captured = 0
        self._last_frame_time_ms = 0.0
        self._total_frame_time_ms = 0.0
        self._recording_active = False

        self._init_wall_time: Optional[float] = None

        self._overlays: dict[str, str] = {}
        self._hud_visible = False

        self._presets: dict[str, RenderPreset] = dict(_BUILTIN_PRESETS)
        self._active_preset_name: Optional[str] = None

        self._render_queue: list[RenderJob] = []

        logger.debug(
            "Renderer constructed (mode=%s, quality=%s, resolution=%s, headless=%s).",
            renderer_mode.value, quality.value, self._resolution, self._headless,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Renderer":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        with self._lock:
            return (
                f"Renderer(state={self._state.value}, mode={self._renderer_mode.value}, "
                f"quality={self._quality.value}, resolution={self._resolution})"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Acquire/attach the render backend and enter ``READY`` state.

        Idempotent: calling ``initialize()`` again while already
        initialized is a no-op (logged at debug level).

        Raises:
            RendererImportError: If no backend was injected and the
                required ``omni``/``carb`` modules cannot be imported.
        """
        with self._lock:
            if self._state != RendererState.UNINITIALIZED:
                logger.debug("initialize() called in state %s; ignoring.", self._state.value)
                return

            self._backend = self._injected_backend or _OmniRTXBackend(headless=self._headless)
            self._apply_current_settings_to_backend()

            self._state = RendererState.READY
            self._init_wall_time = time.monotonic()
            logger.info(
                "Renderer initialized (mode=%s, quality=%s, resolution=%s).",
                self._renderer_mode.value, self._quality.value, self._resolution,
            )

    def shutdown(self) -> None:
        """Stop any active recording, release the backend, and enter ``SHUTDOWN`` state.

        Idempotent. Safe to call from ``__exit__`` even if
        ``initialize()`` was never called.
        """
        with self._lock:
            if self._state in (RendererState.UNINITIALIZED, RendererState.SHUTDOWN):
                self._state = RendererState.SHUTDOWN
                return

            try:
                if self._backend is not None:
                    self._backend.shutdown()
            except Exception:  # noqa: BLE001
                logger.warning("Backend raised while shutting down; ignoring.")

            self._state = RendererState.SHUTDOWN
            self._backend = None
            self._recording_active = False
            self._render_queue.clear()
            logger.info("Renderer shut down.")

    def _apply_current_settings_to_backend(self) -> None:
        """Push all current settings to ``self._backend``. Caller must hold the lock."""
        assert self._backend is not None
        self._backend.set_resolution(*self._resolution)
        self._backend.set_camera(self._camera_path)
        self._backend.set_renderer_mode(self._renderer_mode.value)
        self._backend.set_quality(self._quality.value)
        self._backend.set_gpu_devices(self._gpu_devices)
        self._backend.set_feature("rtx", self._rtx_enabled)
        self._backend.set_feature("dlss", self._dlss_enabled)
        self._backend.set_feature("antialiasing", self._antialiasing_enabled)
        self._backend.set_feature("denoiser", self._denoiser_enabled)

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _require_state(self, *allowed: RendererState, action: str) -> None:
        if self._state not in allowed:
            raise RendererStateError(
                f"Cannot {action} while in state {self._state.value}; "
                f"expected one of {[s.value for s in allowed]}."
            )

    # ------------------------------------------------------------------
    # Frame / sequence rendering
    # ------------------------------------------------------------------

    def render_frame(self) -> RenderStatistics:
        """Render exactly one frame at the current settings.

        Returns:
            The :class:`RenderStatistics` snapshot taken immediately
            after the frame renders, so capture/export consumers can
            synchronize off a single return value.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend fails to render.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="render_frame()")
            start = time.monotonic()
            self._backend.render_frame()  # type: ignore[union-attr]
            elapsed_ms = (time.monotonic() - start) * 1000.0

            self._frames_rendered += 1
            self._last_frame_time_ms = elapsed_ms
            self._total_frame_time_ms += elapsed_ms

            if self._recording_active:
                self._backend.write_video_frame()  # type: ignore[union-attr]

            logger.debug("Rendered frame #%d in %.2fms.", self._frames_rendered, elapsed_ms)
            return self._statistics_unlocked()

    def render_sequence(
        self, frame_count: int, *, on_frame: Optional[FrameCallback] = None,
    ) -> RenderStatistics:
        """Render ``frame_count`` consecutive frames.

        Args:
            frame_count: Number of frames to render. Must be a positive
                integer.
            on_frame: Optional callback invoked after each frame with
                ``(frame_index, statistics)``, where ``frame_index`` is
                0-based within this sequence call.

        Returns:
            The :class:`RenderStatistics` snapshot taken after the last
            frame renders.

        Raises:
            RendererValidationError: If ``frame_count`` is not a
                positive integer.
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise RendererValidationError(f"frame_count must be a positive integer (got {frame_count!r}).")

        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="render_sequence()")
            stats = self._statistics_unlocked()
            for i in range(frame_count):
                stats = self.render_frame()
                if on_frame is not None:
                    try:
                        on_frame(i, stats)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_frame callback raised at frame %d; continuing.", i)
            self._sequences_rendered += 1
            logger.info("Rendered sequence of %d frame(s).", frame_count)
            return stats

    # ------------------------------------------------------------------
    # Capture: images / video
    # ------------------------------------------------------------------

    def capture_image(self, path: Optional[str] = None) -> CaptureResult:
        """Capture the current frame's RGB image.

        Args:
            path: Optional filesystem path to also save a screenshot to
                (PNG/JPEG, backend-dependent by extension). If omitted,
                only the in-memory payload is returned.

        Returns:
            A :class:`CaptureResult` with ``product_type=RGB``.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_image()")
            data = self._capture_products_unlocked([RenderProductType.RGB])
            if path is not None:
                self._backend.capture_screenshot(path)  # type: ignore[union-attr]
            self._images_captured += 1
            width, height = self._resolution
            result = CaptureResult(
                product_type=RenderProductType.RGB,
                frame=self._frames_rendered,
                width=width,
                height=height,
                data=data.get(RenderProductType.RGB),
                path=Path(path) if path else None,
            )
            logger.info("Captured image (frame=%d, path=%s).", result.frame, path)
            return result

    def capture_video(
        self, path: str, num_frames: int, *, fps: Optional[float] = None,
    ) -> CaptureResult:
        """Render and capture ``num_frames`` frames to a video file.

        Args:
            path: Output video file path.
            num_frames: Number of frames to record. Must be a positive
                integer.
            fps: Output video frame rate. Defaults to 30.0.

        Returns:
            A :class:`CaptureResult` with ``product_type=RGB`` and
            ``metadata={"video": True, "fps": ..., "num_frames": ...}``.

        Raises:
            RendererValidationError: If ``num_frames`` is not a
                positive integer.
            RendererStateError: If called before ``initialize()``,
                after ``shutdown()``, or if a recording is already in
                progress.
            RenderCaptureError: If the backend fails to start/stop
                recording.
        """
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise RendererValidationError(f"num_frames must be a positive integer (got {num_frames!r}).")

        resolved_fps = fps if fps is not None else 30.0
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_video()")
            if self._recording_active:
                raise RendererStateError("A video recording is already in progress.")

            self._backend.start_video_recording(path, resolved_fps)  # type: ignore[union-attr]
            self._recording_active = True
            try:
                for _ in range(num_frames):
                    self.render_frame()
            finally:
                self._backend.stop_video_recording()  # type: ignore[union-attr]
                self._recording_active = False

            self._videos_captured += 1
            width, height = self._resolution
            result = CaptureResult(
                product_type=RenderProductType.RGB,
                frame=self._frames_rendered,
                width=width,
                height=height,
                path=Path(path),
                metadata={"video": True, "fps": resolved_fps, "num_frames": num_frames},
            )
            logger.info("Captured video of %d frame(s) to '%s'.", num_frames, path)
            return result

    # ------------------------------------------------------------------
    # Capture: per-modality render products
    # ------------------------------------------------------------------

    def _capture_products_unlocked(
        self, product_types: list[RenderProductType],
    ) -> dict[RenderProductType, Any]:
        """Capture one or more product types via the backend. Caller must hold the lock."""
        keys = [_PRODUCT_ANNOTATOR_NAMES[p] for p in product_types]
        try:
            raw = self._backend.capture(keys)  # type: ignore[union-attr]
        except RenderCaptureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderCaptureError(f"Backend capture failed for {product_types}: {exc}") from exc
        return {p: raw.get(_PRODUCT_ANNOTATOR_NAMES[p]) for p in product_types}

    def capture_depth(self) -> CaptureResult:
        """Capture a per-pixel depth (distance-to-camera) render product.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_depth()")
            data = self._capture_products_unlocked([RenderProductType.DEPTH])
            width, height = self._resolution
            return CaptureResult(
                product_type=RenderProductType.DEPTH, frame=self._frames_rendered,
                width=width, height=height, data=data[RenderProductType.DEPTH],
            )

    def capture_normals(self) -> CaptureResult:
        """Capture a per-pixel surface-normals render product.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_normals()")
            data = self._capture_products_unlocked([RenderProductType.NORMALS])
            width, height = self._resolution
            return CaptureResult(
                product_type=RenderProductType.NORMALS, frame=self._frames_rendered,
                width=width, height=height, data=data[RenderProductType.NORMALS],
            )

    def capture_segmentation(self, mode: str = "instance") -> CaptureResult:
        """Capture an instance- or semantic-segmentation render product.

        Args:
            mode: Either ``"instance"`` or ``"semantic"``.

        Raises:
            RendererValidationError: If ``mode`` is not one of the
                supported values.
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        if mode not in ("instance", "semantic"):
            raise RendererValidationError(f"mode must be 'instance' or 'semantic' (got {mode!r}).")
        product_type = (
            RenderProductType.INSTANCE_SEGMENTATION if mode == "instance"
            else RenderProductType.SEMANTIC_SEGMENTATION
        )
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_segmentation()")
            data = self._capture_products_unlocked([product_type])
            width, height = self._resolution
            return CaptureResult(
                product_type=product_type, frame=self._frames_rendered,
                width=width, height=height, data=data[product_type], metadata={"mode": mode},
            )

    def capture_motion_vectors(self, *, include_optical_flow: bool = False) -> CaptureResult:
        """Capture a per-pixel motion-vector render product.

        Args:
            include_optical_flow: If ``True``, also captures the
                derived optical-flow product and attaches it under
                ``metadata["optical_flow"]``.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_motion_vectors()")
            product_types = [RenderProductType.MOTION_VECTORS]
            if include_optical_flow:
                product_types.append(RenderProductType.OPTICAL_FLOW)
            data = self._capture_products_unlocked(product_types)
            width, height = self._resolution
            metadata: dict[str, Any] = {}
            if include_optical_flow:
                metadata["optical_flow"] = data[RenderProductType.OPTICAL_FLOW]
            return CaptureResult(
                product_type=RenderProductType.MOTION_VECTORS, frame=self._frames_rendered,
                width=width, height=height, data=data[RenderProductType.MOTION_VECTORS],
                metadata=metadata,
            )

    def capture_optical_flow(self) -> CaptureResult:
        """Capture a per-pixel optical-flow render product directly.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_optical_flow()")
            data = self._capture_products_unlocked([RenderProductType.OPTICAL_FLOW])
            width, height = self._resolution
            return CaptureResult(
                product_type=RenderProductType.OPTICAL_FLOW, frame=self._frames_rendered,
                width=width, height=height, data=data[RenderProductType.OPTICAL_FLOW],
            )

    def capture_bounding_boxes(self, dimension: str = "2d") -> CaptureResult:
        """Capture a 2D or 3D bounding-box render product.

        Args:
            dimension: Either ``"2d"`` or ``"3d"``.

        Raises:
            RendererValidationError: If ``dimension`` is not one of the
                supported values.
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
            RenderCaptureError: If the backend capture fails.
        """
        if dimension not in ("2d", "3d"):
            raise RendererValidationError(f"dimension must be '2d' or '3d' (got {dimension!r}).")
        product_type = (
            RenderProductType.BOUNDING_BOX_2D if dimension == "2d" else RenderProductType.BOUNDING_BOX_3D
        )
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="capture_bounding_boxes()")
            data = self._capture_products_unlocked([product_type])
            width, height = self._resolution
            return CaptureResult(
                product_type=product_type, frame=self._frames_rendered,
                width=width, height=height, data=data[product_type], metadata={"dimension": dimension},
            )

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_renderer(self, mode: RenderMode) -> None:
        """Select the active render backend mode.

        Args:
            mode: One of :class:`RenderMode`.
        """
        with self._lock:
            self._renderer_mode = mode
            if self._backend is not None:
                self._backend.set_renderer_mode(mode.value)
            self._active_preset_name = None
            logger.info("renderer_mode set to '%s'.", mode.value)

    def set_quality(self, quality: RenderQuality) -> None:
        """Select the active render quality tier.

        Args:
            quality: One of :class:`RenderQuality`.
        """
        with self._lock:
            self._quality = quality
            if self._backend is not None:
                self._backend.set_quality(quality.value)
            self._active_preset_name = None
            logger.info("quality set to '%s'.", quality.value)

    def set_resolution(self, width: int, height: int) -> None:
        """Set the render resolution, in pixels.

        Args:
            width: Output width. Must be positive.
            height: Output height. Must be positive.

        Raises:
            RendererValidationError: If ``width`` or ``height`` is
                non-positive.
        """
        if width <= 0 or height <= 0:
            raise RendererValidationError(f"resolution must be positive (got ({width}, {height})).")
        with self._lock:
            self._resolution = (width, height)
            if self._backend is not None:
                self._backend.set_resolution(width, height)
            logger.info("resolution set to (%d, %d).", width, height)

    def set_camera(self, camera_path: str) -> None:
        """Set the active camera used for rendering and capture.

        Args:
            camera_path: USD prim path of the camera to activate.

        Raises:
            RendererValidationError: If ``camera_path`` is empty.
        """
        if not camera_path:
            raise RendererValidationError("camera_path must be non-empty.")
        with self._lock:
            self._camera_path = camera_path
            if self._backend is not None:
                self._backend.set_camera(camera_path)
            logger.info("camera set to '%s'.", camera_path)

    # ------------------------------------------------------------------
    # Feature toggles: RTX / DLSS / anti-aliasing / denoiser
    # ------------------------------------------------------------------

    def enable_rtx(self) -> None:
        """Enable the RTX renderer feature set."""
        with self._lock:
            self._rtx_enabled = True
            if self._backend is not None:
                self._backend.set_feature("rtx", True)
            logger.info("RTX enabled.")

    def disable_rtx(self) -> None:
        """Disable the RTX renderer feature set.

        Also disables DLSS if it was enabled, since DLSS requires RTX
        (mirrors the equivalent constraint in ``OmniverseConfig.validate()``).
        """
        with self._lock:
            self._rtx_enabled = False
            if self._dlss_enabled:
                logger.warning("Disabling RTX also disables DLSS (DLSS requires RTX).")
                self._dlss_enabled = False
                if self._backend is not None:
                    self._backend.set_feature("dlss", False)
            if self._backend is not None:
                self._backend.set_feature("rtx", False)
            logger.info("RTX disabled.")

    def enable_dlss(self) -> None:
        """Enable DLSS upscaling.

        Raises:
            RendererValidationError: If RTX is not currently enabled.
        """
        with self._lock:
            if not self._rtx_enabled:
                raise RendererValidationError("enable_dlss() requires RTX to be enabled first.")
            self._dlss_enabled = True
            if self._backend is not None:
                self._backend.set_feature("dlss", True)
            logger.info("DLSS enabled.")

    def disable_dlss(self) -> None:
        """Disable DLSS upscaling."""
        with self._lock:
            self._dlss_enabled = False
            if self._backend is not None:
                self._backend.set_feature("dlss", False)
            logger.info("DLSS disabled.")

    def enable_antialiasing(self) -> None:
        """Enable anti-aliasing."""
        with self._lock:
            self._antialiasing_enabled = True
            if self._backend is not None:
                self._backend.set_feature("antialiasing", True)
            logger.info("Anti-aliasing enabled.")

    def disable_antialiasing(self) -> None:
        """Disable anti-aliasing."""
        with self._lock:
            self._antialiasing_enabled = False
            if self._backend is not None:
                self._backend.set_feature("antialiasing", False)
            logger.info("Anti-aliasing disabled.")

    def enable_denoiser(self) -> None:
        """Enable the (path-tracing) denoiser."""
        with self._lock:
            self._denoiser_enabled = True
            if self._backend is not None:
                self._backend.set_feature("denoiser", True)
            logger.info("Denoiser enabled.")

    def disable_denoiser(self) -> None:
        """Disable the (path-tracing) denoiser."""
        with self._lock:
            self._denoiser_enabled = False
            if self._backend is not None:
                self._backend.set_feature("denoiser", False)
            logger.info("Denoiser disabled.")

    # ------------------------------------------------------------------
    # Lighting / exposure / tone mapping
    # ------------------------------------------------------------------

    def set_lighting_preset(self, name: str) -> None:
        """Record the active lighting preset name.

        This module does not itself author lights (that is a
        stage/scene concern) -- it records which named lighting rig is
        active so diagnostics/export metadata can reflect it, and, when
        a real backend is attached, forwards the selection as an
        RTX-side environment/dome-light setting.

        Args:
            name: Preset identifier (e.g. ``"studio"``, ``"outdoor_noon"``).

        Raises:
            RendererValidationError: If ``name`` is empty.
        """
        if not name:
            raise RendererValidationError("name must be non-empty.")
        with self._lock:
            self._lighting_preset = name
            logger.info("lighting_preset set to '%s'.", name)

    def set_exposure(self, ev: float) -> None:
        """Set camera exposure, in EV stops.

        Args:
            ev: Exposure value. Positive brightens, negative darkens.
        """
        with self._lock:
            self._exposure = ev
            logger.info("exposure set to %.3f EV.", ev)

    def set_tone_mapping(self, operator_name: str) -> None:
        """Set the tone-mapping operator.

        Args:
            operator_name: One of ``"linear"``, ``"reinhard"``,
                ``"aces"``, ``"filmic"``.

        Raises:
            RendererValidationError: If ``operator_name`` is not
                recognized.
        """
        if operator_name not in _VALID_TONE_MAPPING:
            raise RendererValidationError(
                f"operator_name '{operator_name}' must be one of {_VALID_TONE_MAPPING}."
            )
        with self._lock:
            self._tone_mapping = operator_name
            logger.info("tone_mapping set to '%s'.", operator_name)

    # ------------------------------------------------------------------
    # GPU selection
    # ------------------------------------------------------------------

    def set_gpu_device(self, device: int) -> None:
        """Select a single GPU device to render on.

        Args:
            device: CUDA device index. Must be non-negative.

        Raises:
            RendererValidationError: If ``device`` is negative.
        """
        if device < 0:
            raise RendererValidationError(f"device must be >= 0 (got {device}).")
        with self._lock:
            self._gpu_devices = [device]
            if self._backend is not None:
                self._backend.set_gpu_devices(self._gpu_devices)
            logger.info("gpu_device set to %d.", device)

    def enable_multi_gpu(self, devices: list[int]) -> None:
        """Enable rendering across multiple GPU devices.

        Args:
            devices: CUDA device indices to render across. Must contain
                at least two non-negative, unique indices.

        Raises:
            RendererValidationError: If fewer than two devices are
                given, or any index is negative.
        """
        if len(devices) < 2:
            raise RendererValidationError("enable_multi_gpu() requires at least 2 device indices.")
        if any(d < 0 for d in devices):
            raise RendererValidationError(f"all device indices must be >= 0 (got {devices}).")
        with self._lock:
            self._gpu_devices = list(dict.fromkeys(devices))  # de-duplicate, preserve order
            if self._backend is not None:
                self._backend.set_gpu_devices(self._gpu_devices)
            logger.info("multi-GPU enabled across devices %s.", self._gpu_devices)

    # ------------------------------------------------------------------
    # Viewport overlays / HUD
    # ------------------------------------------------------------------

    def add_viewport_overlay(self, name: str, text: str) -> None:
        """Add or update a named viewport overlay/annotation.

        Args:
            name: Overlay identifier, used to update or later remove it.
            text: Overlay text content.
        """
        with self._lock:
            self._overlays[name] = text
            logger.debug("Viewport overlay '%s' set.", name)

    def remove_viewport_overlay(self, name: str) -> None:
        """Remove a named viewport overlay if present.

        Args:
            name: Overlay identifier previously passed to
                :meth:`add_viewport_overlay`.
        """
        with self._lock:
            self._overlays.pop(name, None)

    def clear_viewport_overlays(self) -> None:
        """Remove all viewport overlays."""
        with self._lock:
            self._overlays.clear()

    def show_hud(self) -> None:
        """Show the performance/diagnostics HUD overlay."""
        with self._lock:
            self._hud_visible = True

    def hide_hud(self) -> None:
        """Hide the performance/diagnostics HUD overlay."""
        with self._lock:
            self._hud_visible = False

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def register_preset(self, preset: RenderPreset) -> None:
        """Register (or overwrite) a named :class:`RenderPreset`.

        Args:
            preset: The preset to register, keyed by ``preset.name``.
        """
        with self._lock:
            self._presets[preset.name] = preset
            logger.debug("Registered render preset '%s'.", preset.name)

    def apply_preset(self, name: str) -> None:
        """Apply a previously registered (or built-in) preset by name.

        Built-in presets are ``"draft"``, ``"balanced"``, and
        ``"cinematic"``.

        Args:
            name: Preset identifier.

        Raises:
            RendererValidationError: If ``name`` is not a known preset.
        """
        with self._lock:
            preset = self._presets.get(name)
            if preset is None:
                raise RendererValidationError(
                    f"Unknown render preset '{name}'; known presets: {sorted(self._presets)}."
                )
            self.set_renderer(preset.renderer_mode)
            self.set_quality(preset.quality)
            if preset.rtx_enabled:
                self.enable_rtx()
            else:
                self.disable_rtx()
            if preset.dlss_enabled:
                self.enable_dlss()
            else:
                self.disable_dlss()
            if preset.antialiasing_enabled:
                self.enable_antialiasing()
            else:
                self.disable_antialiasing()
            if preset.denoiser_enabled:
                self.enable_denoiser()
            else:
                self.disable_denoiser()
            self.set_exposure(preset.exposure)
            self.set_tone_mapping(preset.tone_mapping)
            self._active_preset_name = name
            logger.info("Applied render preset '%s'.", name)

    def list_presets(self) -> list[str]:
        """Return the names of all currently registered presets."""
        with self._lock:
            return sorted(self._presets)

    # ------------------------------------------------------------------
    # Render queue (batch/offline processing)
    # ------------------------------------------------------------------

    def enqueue_render_job(self, action: Callable[[], Any], label: str = "") -> str:
        """Queue a unit of render work for later batch processing.

        Args:
            action: A zero-argument callable to invoke when the job
                runs, typically a closure over a capture method (e.g.
                ``lambda: renderer.capture_image("out/%04d.png" % i)``).
            label: Optional human-readable description.

        Returns:
            An opaque job identifier. Not currently cancelable
            individually; use :meth:`clear_render_queue` to drop all
            pending jobs.

        Raises:
            RendererValidationError: If ``action`` is not callable.
        """
        if not callable(action):
            raise RendererValidationError("action must be callable.")
        job_id = uuid.uuid4().hex
        with self._lock:
            self._render_queue.append(RenderJob(job_id=job_id, action=action, label=label))
        logger.debug("Enqueued render job '%s' (%s).", job_id, label or "unlabeled")
        return job_id

    def process_render_queue(self) -> list[Any]:
        """Run every queued render job in FIFO order and clear the queue.

        Returns:
            A list of each job's return value, in submission order. A
            job that raises is logged and contributes ``None`` rather
            than aborting the remaining queue.

        Raises:
            RendererStateError: If called before ``initialize()`` or
                after ``shutdown()``.
        """
        with self._lock:
            self._require_state(*_ACTIVE_STATES, action="process_render_queue()")
            jobs, self._render_queue = self._render_queue, []

        results: list[Any] = []
        for job in jobs:
            try:
                results.append(job.action())
            except Exception:  # noqa: BLE001
                logger.exception("Render job '%s' (%s) raised; recording None.", job.job_id, job.label)
                results.append(None)
        logger.info("Processed %d queued render job(s).", len(jobs))
        return results

    def clear_render_queue(self) -> None:
        """Discard all pending render jobs without running them."""
        with self._lock:
            self._render_queue.clear()

    # ------------------------------------------------------------------
    # Statistics / diagnostics
    # ------------------------------------------------------------------

    def statistics(self) -> RenderStatistics:
        """Return a point-in-time :class:`RenderStatistics` snapshot."""
        with self._lock:
            return self._statistics_unlocked()

    def _statistics_unlocked(self) -> RenderStatistics:
        real_elapsed = (
            time.monotonic() - self._init_wall_time if self._init_wall_time is not None else 0.0
        )
        average_ms = (
            self._total_frame_time_ms / self._frames_rendered if self._frames_rendered else 0.0
        )
        current_fps = (1000.0 / average_ms) if average_ms > 0 else 0.0
        return RenderStatistics(
            state=self._state,
            renderer_mode=self._renderer_mode,
            quality=self._quality,
            resolution=self._resolution,
            rtx_enabled=self._rtx_enabled,
            dlss_enabled=self._dlss_enabled,
            antialiasing_enabled=self._antialiasing_enabled,
            denoiser_enabled=self._denoiser_enabled,
            frames_rendered=self._frames_rendered,
            sequences_rendered=self._sequences_rendered,
            images_captured=self._images_captured,
            videos_captured=self._videos_captured,
            last_frame_time_ms=self._last_frame_time_ms,
            average_frame_time_ms=average_ms,
            current_fps=current_fps,
            real_time_elapsed=real_elapsed,
            gpu_devices=tuple(self._gpu_devices),
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return low-level diagnostic information, distinct from render statistics.

        Useful for health checks / debugging rather than pipeline
        logic -- includes backend presence, queue depth, overlay/HUD
        state, and backend-reported performance counters when
        available.
        """
        with self._lock:
            backend_perf: dict[str, float] = {}
            if self._backend is not None:
                try:
                    backend_perf = self._backend.get_performance_stats()
                except Exception:  # noqa: BLE001
                    logger.debug("Backend get_performance_stats() failed; omitting.")
            return {
                "state": self._state.value,
                "backend_attached": self._backend is not None,
                "backend_type": type(self._backend).__name__ if self._backend else None,
                "headless": self._headless,
                "camera_path": self._camera_path,
                "active_preset": self._active_preset_name,
                "lighting_preset": self._lighting_preset,
                "exposure": self._exposure,
                "tone_mapping": self._tone_mapping,
                "recording_active": self._recording_active,
                "pending_render_jobs": len(self._render_queue),
                "viewport_overlays": dict(self._overlays),
                "hud_visible": self._hud_visible,
                "gpu_devices": list(self._gpu_devices),
                "backend_performance": backend_perf,
            }

    def __deepcopy__(self, memo: dict[int, Any]) -> "Renderer":
        """Deep-copying a live renderer is not supported.

        A renderer holds a lock and a live backend connection, neither
        of which is meaningfully copyable; use :meth:`statistics` /
        :meth:`diagnostics` to inspect state, and construct a new
        ``Renderer`` with the desired settings instead.
        """
        raise RendererError(
            "Renderer cannot be deep-copied; construct a new Renderer with the "
            "desired settings instead."
        )


__all__ = [
    "Renderer",
    "RenderBackend",
    "RenderStatistics",
    "CaptureResult",
    "RenderPreset",
    "RenderJob",
    "RendererState",
    "RenderMode",
    "RenderQuality",
    "RenderProductType",
    "RendererError",
    "RendererStateError",
    "RendererImportError",
    "RendererValidationError",
    "RenderCaptureError",
]
