"""
utilities/performance.py

Reusable, repository-agnostic performance engineering toolkit for PyTorch
research code (profiling, timing, throughput, GPU memory, benchmarking,
torch.compile, and SDPA backend introspection).

Design goals
------------
* Zero repository-specific assumptions: only depends on PyTorch (+ the
  Python standard library).
* Nothing executes automatically on import. Every capability is opt-in and
  must be explicitly invoked by calling code.
* Safe to use around models wrapped in DDP, AMP autocast, and/or
  torch.compile -- this module never mutates model internals except when a
  helper is explicitly asked to (e.g. `maybe_compile`).

Typical usage
-------------
    from utilities.performance import Profiler, Timer, benchmark_forward

    with Timer("forward") as t:
        out = model(x)
    print(t.elapsed_ms)

    with Profiler(enabled=True, trace_path="trace.json") as prof:
        for _ in range(10):
            out = model(x)

    stats = benchmark_forward(model, (x, timesteps), warmup=5, iters=20)

    # Environment + aggregated performance reporting
    from utilities.performance import EnvironmentReport, PerformanceLogger

    env = EnvironmentReport.capture()
    print(env.to_markdown())

    plog = PerformanceLogger(out_dir="perf_logs")
    plog.capture_environment()
    plog.log_benchmark("baseline", benchmark_forward(model, (x, timesteps)))
    plog.save_markdown()
    plog.save_csv()
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import os
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Timer",
    "Profiler",
    "ThroughputMeter",
    "GPUMonitor",
    "BenchmarkResult",
    "benchmark_callable",
    "maybe_compile",
    "sdpa_backend_report",
    "profile_model",
    "benchmark_forward",
    "benchmark_training_step",
    "PerformanceLogger",
    "EnvironmentReport",
]


# ============================================================================
# Timer
# ============================================================================


class Timer:
    """
    A lightweight, nestable context-manager timer.

    Supports CPU wall-clock timing and, when CUDA is available and
    requested, CUDA-synchronized timing so that measurements reflect actual
    device completion time rather than kernel-launch latency.

    Nesting is supported: child timers report their own elapsed time
    independently of any parent timer that also wraps them.

    Parameters
    ----------
    name:
        Human readable label for this timing region (used only for
        ``__repr__`` / logging convenience; not required).
    device:
        Optional device for CUDA synchronization. If ``None`` and CUDA is
        available, the current default CUDA device is used. Pass
        ``torch.device("cpu")`` to force pure CPU timing even when CUDA is
        available.
    sync_cuda:
        If True (default) and CUDA is available on the resolved device,
        ``torch.cuda.synchronize()`` is called on both enter and exit so
        that ``elapsed_ms`` reflects true device-side duration.

    Examples
    --------
        with Timer("attention_block") as t:
            y = block(x)
        print(f"{t.elapsed_ms:.3f} ms")

        # Nested usage
        with Timer("outer") as outer:
            with Timer("inner") as inner:
                do_work()
            print(inner.elapsed_ms)
        print(outer.elapsed_ms)
    """

    def __init__(
        self,
        name: str = "timer",
        device: Optional[torch.device] = None,
        sync_cuda: bool = True,
    ) -> None:
        self.name = name
        self._requested_device = device
        self._sync_cuda = sync_cuda
        self._use_cuda_sync = False
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed_ms: float = 0.0

    def _resolve_use_cuda(self) -> bool:
        if not self._sync_cuda or not torch.cuda.is_available():
            return False
        if self._requested_device is not None and self._requested_device.type != "cuda":
            return False
        return True

    def __enter__(self) -> "Timer":
        self._use_cuda_sync = self._resolve_use_cuda()
        if self._use_cuda_sync:
            torch.cuda.synchronize(self._requested_device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._use_cuda_sync:
            torch.cuda.synchronize(self._requested_device)
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1000.0
        return None

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed_ms={self.elapsed_ms:.3f})"


# ============================================================================
# Profiler
# ============================================================================


class Profiler:
    """
    Thin, optional wrapper around ``torch.profiler.profile``.

    The profiler does nothing unless ``enabled=True`` is passed, so it can
    be left in place in training scripts as a no-op context manager during
    normal runs.

    Parameters
    ----------
    enabled:
        Master on/off switch. When False, entering/exiting this context
        manager is a pure no-op (no CPU/CUDA activity is recorded).
    use_cuda:
        Whether to additionally record CUDA activity. Ignored if CUDA is
        unavailable.
    warmup:
        Number of warmup steps passed to ``torch.profiler.schedule``. These
        steps are executed but not recorded.
    active:
        Number of active (recorded) steps passed to the schedule.
    wait:
        Number of initial steps to skip before warmup begins.
    repeat:
        Number of times to repeat the wait/warmup/active cycle. 0 means
        repeat indefinitely.
    record_shapes, profile_memory, with_stack:
        Forwarded directly to ``torch.profiler.profile``.
    trace_path:
        If provided, a Chrome trace JSON is exported to this path on
        context exit via ``export_chrome_trace``.

    Notes
    -----
    Call ``prof.step()`` once per iteration inside the ``with`` block if you
    are using the schedule-based wait/warmup/active cycle (matches standard
    ``torch.profiler`` usage). If you only need to profile a single
    contiguous region, you can omit ``step()`` calls entirely.
    """

    def __init__(
        self,
        enabled: bool = False,
        use_cuda: Optional[bool] = None,
        warmup: int = 1,
        active: int = 3,
        wait: int = 0,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        trace_path: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.warmup = warmup
        self.active = active
        self.wait = wait
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.trace_path = trace_path

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self._prof: Optional["torch.profiler.profile"] = None

    def _build_activities(self) -> List["torch.profiler.ProfilerActivity"]:
        from torch.profiler import ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(ProfilerActivity.CUDA)
        return activities

    def __enter__(self) -> "Profiler":
        if not self.enabled:
            return self

        from torch.profiler import schedule as torch_schedule

        self._prof = torch.profiler.profile(
            activities=self._build_activities(),
            schedule=torch_schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._prof.__enter__()
        return self

    def step(self) -> None:
        """Advance the profiler schedule by one iteration (no-op if disabled)."""
        if self.enabled and self._prof is not None:
            self._prof.step()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.enabled or self._prof is None:
            return None

        self._prof.__exit__(exc_type, exc_val, exc_tb)

        if self.trace_path:
            self.export_chrome_trace(self.trace_path)

        return None

    def export_chrome_trace(self, path: str) -> None:
        """Export a Chrome trace JSON file. No-op if profiling was disabled."""
        if self._prof is not None:
            self._prof.export_chrome_trace(path)

    def key_averages_table(
        self,
        sort_by: str = "self_cuda_time_total",
        row_limit: int = 20,
    ) -> str:
        """
        Return a formatted table string of averaged profiler events.

        Falls back to sorting by ``self_cpu_time_total`` if CUDA events are
        unavailable (e.g. CPU-only run).
        """
        if self._prof is None:
            return ""
        try:
            return self._prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        except (KeyError, AssertionError):
            return self._prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=row_limit
            )


# ============================================================================
# ThroughputMeter
# ============================================================================


class ThroughputMeter:
    """
    Accumulates timing/sample-count observations and reports throughput
    statistics (images/sec, iterations/sec, average batch time).

    Usage
    -----
        meter = ThroughputMeter()
        for batch in loader:
            with Timer(sync_cuda=True) as t:
                model(batch)
            meter.update(batch_size=batch.shape[0], elapsed_ms=t.elapsed_ms)
        print(meter.images_per_sec, meter.iters_per_sec, meter.avg_batch_time_ms)
    """

    def __init__(self) -> None:
        self._n_iters: int = 0
        self._n_samples: int = 0
        self._total_ms: float = 0.0

    def reset(self) -> None:
        self._n_iters = 0
        self._n_samples = 0
        self._total_ms = 0.0

    def update(self, batch_size: int, elapsed_ms: float) -> None:
        """Register one measured iteration."""
        self._n_iters += 1
        self._n_samples += batch_size
        self._total_ms += elapsed_ms

    @property
    def avg_batch_time_ms(self) -> float:
        if self._n_iters == 0:
            return 0.0
        return self._total_ms / self._n_iters

    @property
    def images_per_sec(self) -> float:
        if self._total_ms <= 0.0:
            return 0.0
        return self._n_samples / (self._total_ms / 1000.0)

    @property
    def iters_per_sec(self) -> float:
        if self._total_ms <= 0.0:
            return 0.0
        return self._n_iters / (self._total_ms / 1000.0)

    def summary(self) -> Dict[str, float]:
        return {
            "images_per_sec": self.images_per_sec,
            "iters_per_sec": self.iters_per_sec,
            "avg_batch_time_ms": self.avg_batch_time_ms,
            "n_iters": float(self._n_iters),
            "n_samples": float(self._n_samples),
        }


# ============================================================================
# GPUMonitor
# ============================================================================


class GPUMonitor:
    """
    Convenience wrapper around ``torch.cuda`` memory/utilization queries.

    All methods return ``None`` (or an empty dict) gracefully when CUDA is
    unavailable, so this class is safe to instantiate and use unconditionally
    on CPU-only machines.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device
        self.available = torch.cuda.is_available()

    def allocated_mb(self) -> Optional[float]:
        if not self.available:
            return None
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def reserved_mb(self) -> Optional[float]:
        if not self.available:
            return None
        return torch.cuda.memory_reserved(self.device) / (1024 ** 2)

    def max_allocated_mb(self) -> Optional[float]:
        if not self.available:
            return None
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def max_reserved_mb(self) -> Optional[float]:
        if not self.available:
            return None
        return torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

    def reset_peak_stats(self) -> None:
        if self.available:
            torch.cuda.reset_peak_memory_stats(self.device)

    def utilization_pct(self) -> Optional[float]:
        """
        Best-effort GPU utilization percentage via ``torch.cuda.utilization``.
        Returns None if unsupported by the current PyTorch/driver combo.
        """
        if not self.available:
            return None
        try:
            return float(torch.cuda.utilization(self.device))
        except Exception:
            return None

    def snapshot(self) -> Dict[str, Optional[float]]:
        """Return all available memory/utilization stats as a single dict."""
        return {
            "allocated_mb": self.allocated_mb(),
            "reserved_mb": self.reserved_mb(),
            "max_allocated_mb": self.max_allocated_mb(),
            "max_reserved_mb": self.max_reserved_mb(),
            "utilization_pct": self.utilization_pct(),
        }


# ============================================================================
# Benchmark helper
# ============================================================================


@dataclasses.dataclass
class BenchmarkResult:
    """Summary statistics from repeated timed invocations of a callable."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_iters: int
    raw_ms: List[float] = dataclasses.field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(mean_ms={self.mean_ms:.3f}, std_ms={self.std_ms:.3f}, "
            f"min_ms={self.min_ms:.3f}, max_ms={self.max_ms:.3f}, n_iters={self.n_iters})"
        )


def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 20,
    sync_cuda: Optional[bool] = None,
) -> BenchmarkResult:
    """
    Benchmark an arbitrary zero-argument callable.

    Parameters
    ----------
    fn:
        Callable taking no arguments (use ``functools.partial`` / a closure
        to bind arguments beforehand).
    warmup:
        Number of untimed warmup calls (lets cuDNN autotuning, lazy
        compilation, and cache warmup settle before measurement).
    iters:
        Number of timed repetitions.
    sync_cuda:
        Whether to synchronize CUDA before/after each timed call. Defaults
        to True automatically when CUDA is available.

    Returns
    -------
    BenchmarkResult
        Mean/std/min/max timings in milliseconds across ``iters`` runs.
    """
    if sync_cuda is None:
        sync_cuda = torch.cuda.is_available()

    for _ in range(max(0, warmup)):
        fn()

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    samples_ms: List[float] = []
    for _ in range(max(1, iters)):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    std_ms = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0

    return BenchmarkResult(
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        n_iters=len(samples_ms),
        raw_ms=samples_ms,
    )


# ============================================================================
# torch.compile helper
# ============================================================================


def compile_available() -> bool:
    """Return True if ``torch.compile`` exists in this PyTorch build."""
    return hasattr(torch, "compile")


def maybe_compile(
    model: torch.nn.Module,
    enabled: bool = False,
    **compile_kwargs: Any,
) -> torch.nn.Module:
    """
    Compile ``model`` with ``torch.compile`` only when explicitly requested.

    Parameters
    ----------
    model:
        The module to (optionally) compile.
    enabled:
        If False, ``model`` is returned unchanged -- this function never
        compiles implicitly.
    **compile_kwargs:
        Forwarded to ``torch.compile`` (e.g. ``mode="reduce-overhead"``).

    Returns
    -------
    torch.nn.Module
        Either the original module (compile disabled / unavailable) or the
        ``torch.compile``-wrapped module.
    """
    if not enabled:
        return model
    if not compile_available():
        return model
    return torch.compile(model, **compile_kwargs)


# ============================================================================
# SDPA backend report
# ============================================================================


def sdpa_backend_report() -> Dict[str, Optional[bool]]:
    """
    Report which ``torch.nn.functional.scaled_dot_product_attention``
    backends are available/enabled in the current PyTorch build.

    Returns
    -------
    dict
        Keys: ``flash_available``, ``flash_enabled``,
        ``mem_efficient_enabled``, ``math_enabled``. Values are ``None``
        when the corresponding query API is unavailable in this PyTorch
        version.
    """
    report: Dict[str, Optional[bool]] = {
        "flash_available": None,
        "flash_enabled": None,
        "mem_efficient_enabled": None,
        "math_enabled": None,
    }

    backends = getattr(torch.backends, "cuda", None)
    if backends is None:
        return report

    try:
        report["flash_available"] = bool(torch.backends.cuda.flash_sdp_enabled.__self__)
    except Exception:
        pass

    # Preferred modern API (PyTorch >= 2.1): can_use_flash_attention etc. are
    # context-dependent, so we report the *enabled* toggles instead, which
    # are stable across versions.
    for key, getter_name in (
        ("flash_enabled", "flash_sdp_enabled"),
        ("mem_efficient_enabled", "mem_efficient_sdp_enabled"),
        ("math_enabled", "math_sdp_enabled"),
    ):
        getter = getattr(backends, getter_name, None)
        if getter is not None:
            try:
                report[key] = bool(getter())
            except Exception:
                report[key] = None

    # flash_available: whether flash attention *can* run on this hardware.
    try:
        report["flash_available"] = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        report["flash_available"] = None

    return report


# ============================================================================
# High-level utility functions
# ============================================================================


def profile_model(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    warmup: int = 2,
    active: int = 3,
    trace_path: Optional[str] = None,
    use_cuda: Optional[bool] = None,
) -> str:
    """
    Profile a single model's forward pass and return a human-readable table
    of top operators by cost.

    Parameters
    ----------
    model:
        Any ``torch.nn.Module`` (or compiled/DDP-wrapped module).
    inputs:
        Positional arguments passed to ``model(*inputs, **kwargs)``.
    kwargs:
        Optional keyword arguments passed to the model call.
    warmup, active:
        Forwarded to :class:`Profiler`.
    trace_path:
        Optional Chrome trace export path.
    use_cuda:
        Forwarded to :class:`Profiler`; defaults to CUDA availability.

    Returns
    -------
    str
        A formatted key-averages table (sorted by self CUDA/CPU time).
    """
    kwargs = kwargs or {}
    total_steps = warmup + active

    with Profiler(
        enabled=True,
        use_cuda=use_cuda,
        warmup=warmup,
        active=active,
        wait=0,
        repeat=1,
        trace_path=trace_path,
    ) as prof:
        with torch.no_grad():
            for _ in range(total_steps):
                model(*inputs, **kwargs)
                prof.step()

    return prof.key_averages_table()


def benchmark_forward(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    """
    Benchmark an inference-mode forward pass of ``model``.

    Wraps the call in ``torch.no_grad()`` and delegates timing to
    :func:`benchmark_callable`.
    """
    kwargs = kwargs or {}

    def _call() -> None:
        with torch.no_grad():
            model(*inputs, **kwargs)

    return benchmark_callable(_call, warmup=warmup, iters=iters)


def benchmark_training_step(
    step_fn: Callable[[], Any],
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    """
    Benchmark a full training step (forward + backward + optimizer step).

    Parameters
    ----------
    step_fn:
        A zero-argument callable that performs one complete training step
        (e.g. a closure wrapping ``TrainLoop.run_step``). Gradients are
        expected to be managed by the callable itself.
    warmup, iters:
        Forwarded to :func:`benchmark_callable`.
    """
    return benchmark_callable(step_fn, warmup=warmup, iters=iters)


# ============================================================================
# EnvironmentReport
# ============================================================================


@dataclasses.dataclass
class EnvironmentReport:
    """
    Snapshot of the PyTorch/CUDA/hardware environment, for reproducibility
    and for explaining benchmark deltas across machines.

    Construct via :meth:`capture` rather than the constructor directly, so
    that field derivation stays in one place.
    """

    torch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    gpu_name: Optional[str]
    compute_capability: Optional[str]
    tf32_matmul_allowed: Optional[bool]
    tf32_cudnn_allowed: Optional[bool]
    amp_available: bool
    compile_available: bool
    sdpa_backends: Dict[str, Optional[bool]]

    @classmethod
    def capture(cls) -> "EnvironmentReport":
        """Build a report by introspecting the current process's environment."""
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None

        cudnn_version: Optional[str] = None
        try:
            if torch.backends.cudnn.is_available():
                cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            cudnn_version = None

        gpu_name: Optional[str] = None
        compute_capability: Optional[str] = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                compute_capability = f"{major}.{minor}"
            except Exception:
                pass

        tf32_matmul: Optional[bool] = None
        tf32_cudnn: Optional[bool] = None
        try:
            tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
            tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)
        except Exception:
            pass

        amp_available = hasattr(torch.cuda, "amp") or hasattr(torch, "amp")

        return cls(
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            gpu_name=gpu_name,
            compute_capability=compute_capability,
            tf32_matmul_allowed=tf32_matmul,
            tf32_cudnn_allowed=tf32_cudnn,
            amp_available=amp_available,
            compile_available=compile_available(),
            sdpa_backends=sdpa_backend_report(),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # flatten the nested sdpa_backends dict for table-friendly output
        sdpa = d.pop("sdpa_backends")
        for k, v in sdpa.items():
            d[f"sdpa_{k}"] = v
        return d

    def to_markdown(self) -> str:
        lines = ["| Field | Value |", "|---|---|"]
        for k, v in self.to_dict().items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    def save(self, path: str) -> str:
        """Write this report as a standalone Markdown file at ``path``."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_markdown())
        return path


# ============================================================================
# PerformanceLogger
# ============================================================================


class PerformanceLogger:
    """
    Aggregates results from :class:`GPUMonitor`, :class:`ThroughputMeter`,
    :class:`BenchmarkResult`, and :class:`EnvironmentReport` into a single
    research report (Markdown + CSV). This class does no measurement of its
    own -- it only records and formats results produced by the other
    utilities in this module.

    Usage
    -----
        plog = PerformanceLogger(out_dir="perf_logs")
        plog.capture_environment()
        plog.log_benchmark("baseline", benchmark_forward(model, (x, t)))
        plog.log_benchmark("compiled", benchmark_forward(compiled_model, (x, t)))
        plog.log_memory("baseline", gpu_monitor)
        plog.save_markdown()
        plog.save_csv()
    """

    def __init__(self, out_dir: str = "perf_logs") -> None:
        self.out_dir = out_dir
        self.environment: Optional[EnvironmentReport] = None
        self._benchmarks: Dict[str, BenchmarkResult] = {}
        self._memory_snapshots: Dict[str, Dict[str, Optional[float]]] = {}
        self._throughput_summaries: Dict[str, Dict[str, float]] = {}

    def _ensure_out_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    # -- recording -----------------------------------------------------

    def capture_environment(self) -> EnvironmentReport:
        self.environment = EnvironmentReport.capture()
        return self.environment

    def log_benchmark(self, name: str, result: BenchmarkResult) -> None:
        self._benchmarks[name] = result

    def log_memory(self, name: str, monitor: GPUMonitor) -> None:
        self._memory_snapshots[name] = monitor.snapshot()

    def log_throughput(self, name: str, meter: ThroughputMeter) -> None:
        self._throughput_summaries[name] = meter.summary()

    # -- comparison ------------------------------------------------------

    @staticmethod
    def compare_benchmarks(results: Dict[str, BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """
        Compare several named :class:`BenchmarkResult` objects against the
        fastest one. Returns, per name, its mean latency and speedup factor
        relative to the fastest entry (1.0 == fastest).
        """
        if not results:
            return {}
        fastest = min(results.values(), key=lambda r: r.mean_ms)
        comparison: Dict[str, Dict[str, float]] = {}
        for name, r in results.items():
            comparison[name] = {
                "mean_ms": r.mean_ms,
                "speedup_vs_fastest": (fastest.mean_ms / r.mean_ms) if r.mean_ms > 0 else float("nan"),
            }
        return comparison

    def compare_logged_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Convenience: :meth:`compare_benchmarks` over everything logged so far."""
        return self.compare_benchmarks(self._benchmarks)

    # -- export ------------------------------------------------------------

    def to_markdown(self) -> str:
        lines = ["# Performance Report", ""]

        if self.environment is not None:
            lines += ["## Environment", self.environment.to_markdown(), ""]

        if self._benchmarks:
            lines += [
                "## Benchmarks",
                "| Name | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Iters | Speedup vs fastest |",
                "|---|---|---|---|---|---|---|",
            ]
            comparison = self.compare_logged_benchmarks()
            for name, r in self._benchmarks.items():
                speedup = comparison[name]["speedup_vs_fastest"]
                lines.append(
                    f"| {name} | {r.mean_ms:.3f} | {r.std_ms:.3f} | {r.min_ms:.3f} | "
                    f"{r.max_ms:.3f} | {r.n_iters} | {speedup:.2f}x |"
                )
            lines.append("")

        if self._memory_snapshots:
            lines += [
                "## GPU Memory",
                "| Name | Allocated MB | Reserved MB | Max Allocated MB | Max Reserved MB | Util % |",
                "|---|---|---|---|---|---|",
            ]
            for name, s in self._memory_snapshots.items():
                lines.append(
                    f"| {name} | {s.get('allocated_mb')} | {s.get('reserved_mb')} | "
                    f"{s.get('max_allocated_mb')} | {s.get('max_reserved_mb')} | {s.get('utilization_pct')} |"
                )
            lines.append("")

        if self._throughput_summaries:
            lines += [
                "## Throughput",
                "| Name | Images/sec | Iters/sec | Avg batch (ms) |",
                "|---|---|---|---|",
            ]
            for name, s in self._throughput_summaries.items():
                lines.append(
                    f"| {name} | {s.get('images_per_sec', 0.0):.3f} | "
                    f"{s.get('iters_per_sec', 0.0):.3f} | {s.get('avg_batch_time_ms', 0.0):.3f} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save_markdown(self, name: str = "performance_report") -> str:
        self._ensure_out_dir()
        path = os.path.join(self.out_dir, f"{name}.md")
        with open(path, "w") as f:
            f.write(self.to_markdown())
        return path

    def save_csv(self, name: str = "performance_report") -> Dict[str, str]:
        """Export benchmark/memory/throughput tables as separate CSV files."""
        self._ensure_out_dir()
        paths: Dict[str, str] = {}

        if self._benchmarks:
            p = os.path.join(self.out_dir, f"{name}_benchmarks.csv")
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["name", "mean_ms", "std_ms", "min_ms", "max_ms", "n_iters"]
                )
                writer.writeheader()
                for k, r in self._benchmarks.items():
                    writer.writerow(
                        {
                            "name": k,
                            "mean_ms": r.mean_ms,
                            "std_ms": r.std_ms,
                            "min_ms": r.min_ms,
                            "max_ms": r.max_ms,
                            "n_iters": r.n_iters,
                        }
                    )
            paths["benchmarks"] = p

        if self._memory_snapshots:
            p = os.path.join(self.out_dir, f"{name}_memory.csv")
            with open(p, "w", newline="") as f:
                fieldnames = [
                    "name",
                    "allocated_mb",
                    "reserved_mb",
                    "max_allocated_mb",
                    "max_reserved_mb",
                    "utilization_pct",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for k, s in self._memory_snapshots.items():
                    row = {"name": k}
                    row.update(s)
                    writer.writerow(row)
            paths["memory"] = p

        if self._throughput_summaries:
            p = os.path.join(self.out_dir, f"{name}_throughput.csv")
            with open(p, "w", newline="") as f:
                fieldnames = ["name", "images_per_sec", "iters_per_sec", "avg_batch_time_ms", "n_iters", "n_samples"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for k, s in self._throughput_summaries.items():
                    row = {"name": k}
                    row.update(s)
                    writer.writerow(row)
            paths["throughput"] = p

        return paths
