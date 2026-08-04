"""
utilities/visualization.py

Reusable, publication-quality visualization helpers for the physics-aware
conditional diffusion SAR despeckling research pipeline.

Design goals
------------
* No repository-specific imports: this module consumes plain
  numpy/torch arrays and Python dicts/lists -- it does not import from
  ``guided_diffusion`` or ``structdiff``. Callers pull data out of the
  model (e.g. ``model.get_physics_metrics()``) and pass it in.
* Nothing executes on import: all figure generation happens inside
  explicit method calls.
* IEEE-paper-friendly defaults (serif fonts, tight layout, high DPI,
  vector PDF + raster PNG export) applied via a scoped matplotlib
  ``rc_context`` so global pyplot state elsewhere is not disturbed.

Typical usage
-------------
    from utilities.visualization import ResearchVisualizer

    viz = ResearchVisualizer(out_dir="figures")
    viz.plot_condition_weights(history=condition_weight_history)
    viz.plot_attention(attention_map=attn, title="Physics Attention")
    viz.plot_metrics(loss=loss_hist, psnr=psnr_hist, ssim=ssim_hist)
    viz.save_heatmap(bias_map, name="physics_bias")

    # Dashboard / table / comparison helpers
    viz.plot_progress_csv("logs/progress.csv")
    viz.plot_experiment_dashboard(loss=loss_hist, psnr=psnr_hist, ssim=ssim_hist)
    viz.plot_ablation_table(rows=[{"variant": "A2", "psnr": 28.1}, {"variant": "A26e", "psnr": 29.4}])
    viz.plot_multihead_attention(attention_maps=attn_heads)
    viz.compare_experiments({"run_a": loss_a, "run_b": loss_b}, metric_name="Loss")
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard dependency in practice
    _HAS_TORCH = False

import matplotlib

matplotlib.use("Agg")  # safe headless default; caller can switch backend beforehand if needed
import matplotlib.pyplot as plt

__all__ = ["ResearchVisualizer"]


# ============================================================================
# IEEE-style publication defaults
# ============================================================================

_IEEE_RC: Dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _to_numpy(x: Union["np.ndarray", "torch.Tensor", Sequence[float]]) -> np.ndarray:
    """Convert torch tensors / sequences to a detached CPU numpy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


class ResearchVisualizer:
    """
    Central entry point for all research plotting/logging helpers.

    Parameters
    ----------
    out_dir:
        Directory where figures/CSVs are written. Created lazily on first
        save (not on construction), so instantiating this class has no
        filesystem side effects.
    formats:
        File extensions to export for every figure. Defaults to both a
        vector PDF (for camera-ready IEEE submission) and a raster PNG
        (for quick viewing / slides).
    tensorboard_writer:
        Optional pre-constructed ``torch.utils.tensorboard.SummaryWriter``.
        If omitted, TensorBoard helper methods are no-ops unless a writer
        is supplied per-call.
    """

    def __init__(
        self,
        out_dir: str = "figures",
        formats: Sequence[str] = ("pdf", "png"),
        tensorboard_writer: Optional[Any] = None,
    ) -> None:
        self.out_dir = out_dir
        self.formats = tuple(formats)
        self.tensorboard_writer = tensorboard_writer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_out_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def _save_figure(self, fig: "plt.Figure", name: str) -> List[str]:
        """Save ``fig`` under ``self.out_dir`` in every configured format."""
        self._ensure_out_dir()
        paths = []
        for ext in self.formats:
            path = os.path.join(self.out_dir, f"{name}.{ext}")
            fig.savefig(path)
            paths.append(path)
        return paths

    # ------------------------------------------------------------------
    # Physics Attention visualization
    # ------------------------------------------------------------------

    def plot_attention(
        self,
        attention_map: Union[np.ndarray, "torch.Tensor"],
        title: str = "Physics Attention",
        name: str = "physics_attention",
        cmap: str = "viridis",
        xlabel: str = "Key position",
        ylabel: str = "Query position",
    ) -> List[str]:
        """
        Plot a 2D physics attention map (e.g. averaged attention weights for
        one head/timestep) as a heatmap.

        Parameters
        ----------
        attention_map:
            2D array of shape ``(query_len, key_len)``. If a higher-rank
            tensor is passed (e.g. including a head or batch dimension), it
            is averaged over all leading dimensions.
        """
        arr = _to_numpy(attention_map)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1]).mean(axis=0)

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.0, 3.4))
            im = ax.imshow(arr, cmap=cmap, aspect="auto")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    def plot_attention_shift(
        self,
        shift_history: Sequence[float],
        name: str = "attention_shift",
        xlabel: str = "Training step",
        ylabel: str = "Attention shift",
    ) -> List[str]:
        """
        Plot the evolution of the scalar "attention shift" research metric
        exposed by ``UNetModel.get_attention_shift`` / physics transformer
        blocks over training.
        """
        y = _to_numpy(shift_history)
        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.0, 2.6))
            ax.plot(np.arange(len(y)), y, color="#1f77b4")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("Physics Attention Shift Over Training")
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Orientation relation visualization
    # ------------------------------------------------------------------

    def plot_orientation_relation(
        self,
        orientations: Union[np.ndarray, "torch.Tensor"],
        magnitudes: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        name: str = "orientation_relation",
        title: str = "Local Orientation Field",
        subsample: int = 8,
    ) -> List[str]:
        """
        Visualize a dense orientation field (e.g. derived from the structure
        tensor) as a quiver plot, optionally scaled by a magnitude/coherence
        map.

        Parameters
        ----------
        orientations:
            2D array of angles in radians, shape ``(H, W)``.
        magnitudes:
            Optional 2D array of the same shape used to scale arrow length
            (e.g. structure-tensor coherence). Defaults to unit length.
        subsample:
            Stride used to subsample the grid for a legible quiver plot.
        """
        theta = _to_numpy(orientations)
        h, w = theta.shape
        mag = np.ones_like(theta) if magnitudes is None else _to_numpy(magnitudes)

        ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
        u = np.cos(theta[::subsample, ::subsample]) * mag[::subsample, ::subsample]
        v = np.sin(theta[::subsample, ::subsample]) * mag[::subsample, ::subsample]

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.0, 4.0))
            ax.quiver(xs, ys, u, v, angles="xy", scale_units="xy", scale=1.0, width=0.003)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.set_aspect("equal")
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Physics Bias heatmap
    # ------------------------------------------------------------------

    def save_heatmap(
        self,
        data: Union[np.ndarray, "torch.Tensor"],
        name: str,
        title: str = "",
        cmap: str = "magma",
        symmetric: bool = False,
    ) -> List[str]:
        """
        Generic heatmap saver, used for e.g. physics attention bias maps.

        Parameters
        ----------
        data:
            2D array to visualize.
        symmetric:
            If True, colormap limits are set symmetrically around zero
            (useful for signed bias values).
        """
        arr = _to_numpy(data)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1]).mean(axis=0)

        vmin, vmax = None, None
        if symmetric:
            vmax = float(np.abs(arr).max())
            vmin = -vmax

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.0, 3.4))
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            if title:
                ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    def plot_bias_norm_evolution(
        self,
        norm_history: Sequence[float],
        name: str = "bias_norm_evolution",
    ) -> List[str]:
        """Plot the L2 norm of the physics attention bias over training."""
        y = _to_numpy(norm_history)
        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.0, 2.6))
            ax.plot(np.arange(len(y)), y, color="#d62728")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Bias norm")
            ax.set_title("Physics Attention Bias Norm")
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Condition weight / gate value evolution
    # ------------------------------------------------------------------

    def plot_condition_weights(
        self,
        history: Union[Dict[str, Sequence[float]], Sequence[Dict[str, float]]],
        name: str = "condition_weights",
        title: str = "Adaptive Condition Fusion Weights",
    ) -> List[str]:
        """
        Plot the evolution of adaptive condition-fusion softmax weights
        (e.g. ``UNetModel.last_condition_weights``) over training.

        Parameters
        ----------
        history:
            Either a dict mapping condition name -> sequence of weights over
            time, or a list of per-step dicts (as directly logged by
            ``TrainLoop.log_step``), which is transposed automatically.
        """
        series = self._normalize_named_series(history)

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.4, 2.8))
            for cond_name, values in series.items():
                ax.plot(np.arange(len(values)), values, label=cond_name)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Fusion weight")
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.legend(loc="best", ncol=2)
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    def plot_gate_values(
        self,
        history: Union[Dict[str, Sequence[float]], Sequence[Dict[str, float]]],
        name: str = "gate_values",
        title: str = "Condition Gate Values",
    ) -> List[str]:
        """Plot gate-value evolution (pre-softmax condition gate scores)."""
        series = self._normalize_named_series(history)

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.4, 2.8))
            for cond_name, values in series.items():
                ax.plot(np.arange(len(values)), values, label=cond_name)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Gate value")
            ax.set_title(title)
            ax.legend(loc="best", ncol=2)
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    @staticmethod
    def _normalize_named_series(
        history: Union[Dict[str, Sequence[float]], Sequence[Dict[str, float]]]
    ) -> Dict[str, np.ndarray]:
        """Normalize either a dict-of-series or a list-of-step-dicts into
        a dict-of-series representation."""
        if isinstance(history, dict):
            return {k: _to_numpy(v) for k, v in history.items()}

        # list-of-dicts: transpose to name -> array
        keys: List[str] = []
        for step in history:
            for k in step.keys():
                if k not in keys:
                    keys.append(k)
        series: Dict[str, List[float]] = {k: [] for k in keys}
        for step in history:
            for k in keys:
                series[k].append(step.get(k, np.nan))
        return {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}

    # ------------------------------------------------------------------
    # Loss / PSNR / SSIM / training history plotting
    # ------------------------------------------------------------------

    def plot_metrics(
        self,
        loss: Optional[Sequence[float]] = None,
        psnr: Optional[Sequence[float]] = None,
        ssim: Optional[Sequence[float]] = None,
        steps: Optional[Sequence[float]] = None,
        name: str = "training_metrics",
    ) -> List[str]:
        """
        Plot loss, PSNR, and SSIM curves side by side (any subset may be
        omitted).

        Parameters
        ----------
        loss, psnr, ssim:
            Optional sequences of per-logged-step values. At least one must
            be provided.
        steps:
            Optional shared x-axis values (defaults to ``range(len(series))``
            per curve).
        """
        panels = [(k, v) for k, v in (("Loss", loss), ("PSNR (dB)", psnr), ("SSIM", ssim)) if v is not None]
        if not panels:
            raise ValueError("At least one of loss, psnr, or ssim must be provided.")

        with plt.rc_context(_IEEE_RC):
            fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8))
            if len(panels) == 1:
                axes = [axes]
            for ax, (label, values) in zip(axes, panels):
                y = _to_numpy(values)
                x = _to_numpy(steps) if steps is not None else np.arange(len(y))
                ax.plot(x, y, color="#2ca02c" if label == "PSNR (dB)" else "#1f77b4")
                ax.set_xlabel("Step")
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.grid(True)
            fig.tight_layout()
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    def plot_training_history(
        self,
        history: Dict[str, Sequence[float]],
        name: str = "training_history",
        title: str = "Training History",
    ) -> List[str]:
        """
        Plot an arbitrary set of named scalar curves (e.g. every key logged
        via ``logger.logkv_mean``) on a single shared-x-axis figure.
        """
        series = {k: _to_numpy(v) for k, v in history.items()}

        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            for label, values in series.items():
                ax.plot(np.arange(len(values)), values, label=label)
            ax.set_xlabel("Logged step")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.legend(loc="best", ncol=2, fontsize=7)
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Progress CSV plotting
    # ------------------------------------------------------------------

    def plot_progress_csv(
        self,
        csv_path: str,
        columns: Optional[Sequence[str]] = None,
        name: str = "progress",
        title: str = "Training Progress",
    ) -> List[str]:
        """
        Load a logger-style progress CSV (as produced by ``write_csv`` /
        ``append_csv_row``, or by ``TrainLoop``'s own progress logging) and
        plot the requested numeric columns via :meth:`plot_training_history`.

        Parameters
        ----------
        columns:
            Column names to plot. Defaults to every column except common
            step/epoch index columns.
        """
        rows = self.read_csv(csv_path)
        if not rows:
            raise ValueError(f"No rows found in {csv_path}")

        all_columns = list(rows[0].keys())
        selected = (
            list(columns)
            if columns is not None
            else [c for c in all_columns if c.lower() not in ("step", "epoch")]
        )

        history: Dict[str, List[float]] = {col: [] for col in selected}
        for row in rows:
            for col in selected:
                raw = row.get(col, "")
                try:
                    history[col].append(float(raw))
                except (TypeError, ValueError):
                    history[col].append(np.nan)

        return self.plot_training_history(history, name=name, title=title)

    # ------------------------------------------------------------------
    # Ablation table
    # ------------------------------------------------------------------

    def plot_ablation_table(
        self,
        rows: Sequence[Dict[str, Any]],
        name: str = "ablation_table",
        title: str = "",
        float_fmt: str = "{:.3f}",
    ) -> List[str]:
        """
        Render a list of row-dicts (e.g. ablation study results) as a
        publication-ready table figure. Column order follows first-seen
        key order across rows.
        """
        if not rows:
            raise ValueError("rows must be non-empty.")

        columns: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in columns:
                    columns.append(k)

        cell_text: List[List[str]] = []
        for row in rows:
            cell_row = []
            for c in columns:
                v = row.get(c, "")
                if isinstance(v, float):
                    v = float_fmt.format(v)
                cell_row.append(str(v))
            cell_text.append(cell_row)

        with plt.rc_context(_IEEE_RC):
            fig_height = 0.35 * (len(rows) + 1) + 0.4
            fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(columns)), fig_height))
            ax.axis("off")
            if title:
                ax.set_title(title, pad=10)
            table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.3)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Multi-panel experiment dashboard
    # ------------------------------------------------------------------

    def plot_experiment_dashboard(
        self,
        loss: Optional[Sequence[float]] = None,
        psnr: Optional[Sequence[float]] = None,
        ssim: Optional[Sequence[float]] = None,
        bias_norm: Optional[Sequence[float]] = None,
        attention_shift: Optional[Sequence[float]] = None,
        condition_weights: Optional[Union[Dict[str, Sequence[float]], Sequence[Dict[str, float]]]] = None,
        name: str = "experiment_dashboard",
        title: str = "Experiment Dashboard",
    ) -> List[str]:
        """
        Single-figure overview combining the most commonly reported curves
        (loss, PSNR, SSIM, physics-attention bias norm, attention shift,
        condition-fusion weights). Any subset of arguments may be omitted;
        a panel is only created for series that are provided.
        """
        panels: List[Tuple[str, str, Any, Optional[str]]] = []
        if loss is not None:
            panels.append(("Loss", "line", _to_numpy(loss), "#1f77b4"))
        if psnr is not None:
            panels.append(("PSNR (dB)", "line", _to_numpy(psnr), "#2ca02c"))
        if ssim is not None:
            panels.append(("SSIM", "line", _to_numpy(ssim), "#9467bd"))
        if bias_norm is not None:
            panels.append(("Bias Norm", "line", _to_numpy(bias_norm), "#d62728"))
        if attention_shift is not None:
            panels.append(("Attention Shift", "line", _to_numpy(attention_shift), "#ff7f0e"))
        if condition_weights is not None:
            panels.append(("Condition Weights", "multi", self._normalize_named_series(condition_weights), None))

        if not panels:
            raise ValueError("At least one series must be provided.")

        ncols = 3
        nrows = int(np.ceil(len(panels) / ncols))

        with plt.rc_context(_IEEE_RC):
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for ax, (label, kind, data, color) in zip(axes, panels):
                if kind == "line":
                    ax.plot(np.arange(len(data)), data, color=color)
                else:
                    for cond_name, values in data.items():
                        ax.plot(np.arange(len(values)), values, label=cond_name)
                    ax.legend(loc="best", fontsize=6, ncol=2)
                ax.set_title(label)
                ax.set_xlabel("Step")
                ax.grid(True)
            for ax in axes[len(panels):]:
                ax.axis("off")
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Multi-head attention visualization
    # ------------------------------------------------------------------

    def plot_multihead_attention(
        self,
        attention_maps: Union[np.ndarray, "torch.Tensor"],
        name: str = "multihead_attention",
        title: str = "Multi-Head Physics Attention",
        cmap: str = "viridis",
        max_heads: int = 8,
    ) -> List[str]:
        """
        Plot each attention head separately in a grid, given a tensor/array
        of shape ``(num_heads, query_len, key_len)``. Complements
        :meth:`plot_attention`, which averages over leading dimensions
        instead of showing heads individually.
        """
        arr = _to_numpy(attention_maps)
        if arr.ndim != 3:
            raise ValueError("attention_maps must have shape (num_heads, query_len, key_len).")

        n_heads = min(arr.shape[0], max_heads)
        ncols = min(4, n_heads)
        nrows = int(np.ceil(n_heads / ncols))

        with plt.rc_context(_IEEE_RC):
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.2 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for i in range(n_heads):
                ax = axes[i]
                ax.imshow(arr[i], cmap=cmap, aspect="auto")
                ax.set_title(f"Head {i}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes[n_heads:]:
                ax.axis("off")
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Cross-experiment comparison
    # ------------------------------------------------------------------

    def compare_experiments(
        self,
        experiments: Dict[str, Sequence[float]],
        metric_name: str = "Value",
        name: str = "experiment_comparison",
        title: Optional[str] = None,
    ) -> List[str]:
        """
        Overlay a single named metric (e.g. loss, PSNR) across several
        experiments/runs for direct comparison.

        Parameters
        ----------
        experiments:
            Mapping of experiment/run name -> sequence of metric values.
        """
        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(4.6, 3.0))
            for exp_name, values in experiments.items():
                y = _to_numpy(values)
                ax.plot(np.arange(len(y)), y, label=exp_name)
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_name)
            ax.set_title(title or f"{metric_name} Comparison")
            ax.legend(loc="best", fontsize=7, ncol=2)
            ax.grid(True)
            paths = self._save_figure(fig, name)
            plt.close(fig)
        return paths

    # ------------------------------------------------------------------
    # Batch export of externally-built figures
    # ------------------------------------------------------------------

    def export_ieee_figures(
        self,
        figures: Dict[str, "plt.Figure"],
    ) -> Dict[str, List[str]]:
        """
        Batch-export externally constructed matplotlib figures using the
        same save conventions (formats, DPI, tight bbox) as every other
        method in this class, without re-styling their contents.
        """
        self._ensure_out_dir()
        all_paths: Dict[str, List[str]] = {}
        for fig_name, fig in figures.items():
            paths = []
            for ext in self.formats:
                path = os.path.join(self.out_dir, f"{fig_name}.{ext}")
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
                paths.append(path)
            all_paths[fig_name] = paths
        return all_paths

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------

    def log_scalars_to_tensorboard(
        self,
        scalars: Dict[str, float],
        step: int,
        writer: Optional[Any] = None,
    ) -> None:
        """
        Log a dict of scalars to TensorBoard via
        ``torch.utils.tensorboard.SummaryWriter.add_scalar``.

        A writer must be available either via the constructor
        (``tensorboard_writer=``) or passed explicitly per-call; if neither
        is available this is a no-op (TensorBoard support is optional).
        """
        w = writer or self.tensorboard_writer
        if w is None:
            return
        for key, value in scalars.items():
            if value is None:
                continue
            w.add_scalar(key, value, global_step=step)

    def log_figure_to_tensorboard(
        self,
        tag: str,
        fig: "plt.Figure",
        step: int,
        writer: Optional[Any] = None,
    ) -> None:
        """Log a matplotlib figure to TensorBoard via ``add_figure``."""
        w = writer or self.tensorboard_writer
        if w is None:
            return
        w.add_figure(tag, fig, global_step=step)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def write_csv(
        self,
        rows: Sequence[Dict[str, Any]],
        name: str,
    ) -> str:
        """
        Write a list of row-dicts to ``{out_dir}/{name}.csv``. The union of
        all keys across rows is used as the column header, in first-seen
        order.
        """
        self._ensure_out_dir()
        path = os.path.join(self.out_dir, f"{name}.csv")

        fieldnames: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def read_csv(self, path: str) -> List[Dict[str, str]]:
        """Read a CSV file back into a list of row-dicts (string values)."""
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def append_csv_row(self, row: Dict[str, Any], name: str) -> str:
        """
        Append a single row to ``{out_dir}/{name}.csv``, writing a header
        first if the file does not yet exist.
        """
        self._ensure_out_dir()
        path = os.path.join(self.out_dir, f"{name}.csv")
        file_exists = os.path.exists(path)

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return path
