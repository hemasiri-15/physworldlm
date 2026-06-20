"""
training/train_entity_encoder.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — first learning stage.

Trains EntityEncoder on the entity classification dataset and produces:

    checkpoints/
    ├── best_loss/
    │   └── entity_encoder_best_loss.pt
    ├── best_entity/
    │   └── entity_encoder_best_entity.pt
    ├── best_capability/
    │   └── entity_encoder_best_capability.pt
    ├── ema/
    │   └── entity_encoder_ema.pt
    ├── history/
    │   └── training_history.json
    ├── plots/
    │   ├── loss_curve.png
    │   ├── accuracy_curve.png
    │   └── f1_curve.png
    ├── train_config.json
    ├── model_config.json
    ├── metrics.json
    └── entity_encoder_last.pt

Design features
───────────────
* Per-head loss objects (single_head_losses / multi_head_losses dicts) —
  enables class weighting and focal loss without redesign.
* Dynamic loss weights dict — critical for curriculum / head balancing.
* Exponential Moving Average (EMA, decay=0.999) — generalises better.
* Multiple best checkpoints — best_loss, best_entity, best_capability.
* Gradient accumulation — supports effective large-batch training.
* Linear warmup + cosine decay scheduler.
* Rich metrics: accuracy, top-3 accuracy, precision, recall, F1, mAP.
* Gradient norm tracking — detects exploding gradients early.
* NaN-safe batch skipping — never crashes on corrupt records.
* Layerwise LR decay — backbone < projection < heads.
* TensorBoard logging.
* Optional WandB logging.
* Full resume: model + optimizer + scheduler + scaler + EMA + history.
* Confusion-matrix PNGs saved per epoch (entity_type, material, shape).
* Config persistence (train_config.json, model_config.json).

No relation encoder, no graph builder, no temporal model, no inference.
"""

from __future__ import annotations

import copy
import json
import random
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.entity_encoder import EntityEncoder, EntityEncoderConfig, build_entity_encoder
from training.entity_dataset import DatasetConfig, EntityDataset

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix as _sklearn_cm
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from transformers import get_cosine_schedule_with_warmup as _hf_warmup
    _HF_SCHEDULER_AVAILABLE = True
except ImportError:
    _HF_SCHEDULER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all global RNG seeds for deterministic training.

    Args:
        seed: Integer seed (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  TrainConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All training hyperparameters — no magic numbers inside train().

    Attributes:
        batch_size:             Samples per forward pass (before accumulation).
        num_workers:            DataLoader worker count.
        epochs:                 Maximum training epochs.
        lr_backbone:            Learning rate for MiniLM backbone.
        lr_projection:          Learning rate for projection tower.
        lr_heads:               Learning rate for output heads.
        weight_decay:           AdamW weight decay.
        gradient_clip:          Max gradient norm for clipping.
        accumulation_steps:     Gradient accumulation steps (effective batch =
                                batch_size × accumulation_steps).
        warmup_ratio:           Fraction of total steps used for linear warmup.
        patience:               Early stopping patience (epochs).
        checkpoint_dir:         Root directory for all saved artifacts.
        device:                 ``"cuda"``, ``"mps"``, or ``"cpu"``.  ``"auto"``
                                selects automatically.
        mixed_precision:        Enable AMP (CUDA only).
        scheduler_tmax:         T_max for fallback CosineAnnealingLR.
        save_every_epoch:       Save ``entity_encoder_last.pt`` every epoch.
        freeze_backbone_epochs: Freeze MiniLM for this many initial epochs.
        resume_checkpoint:      Path to checkpoint dir to resume from.
        ema_decay:              EMA model decay coefficient.
        use_ema:                Maintain and save an EMA model.
        loss_weights:           Per-head loss multipliers.  Missing keys
                                default to 1.0.
        use_tensorboard:        Write TensorBoard logs.
        use_wandb:              Log to Weights & Biases.
        wandb_project:          W&B project name.
        dataset_dir:            Root of the entity split dataset.
        seed:                   RNG seed.
        debug:                  If True, run 1 epoch on 100 samples.
        save_confusion_matrices: Save confusion-matrix PNGs for key heads.
        save_plots:             Save loss / accuracy / F1 curves.
    """

    batch_size:              int   = 64
    num_workers:             int   = 4
    epochs:                  int   = 50
    lr_backbone:             float = 2e-5
    lr_projection:           float = 5e-5
    lr_heads:                float = 1e-4
    weight_decay:            float = 1e-2
    gradient_clip:           float = 1.0
    accumulation_steps:      int   = 1
    warmup_ratio:            float = 0.06
    patience:                int   = 8
    checkpoint_dir:          str   = "checkpoints"
    device:                  str   = "auto"
    mixed_precision:         bool  = True
    scheduler_tmax:          int   = 50
    save_every_epoch:        bool  = True
    freeze_backbone_epochs:  int   = 0
    resume_checkpoint:       Optional[str] = None
    ema_decay:               float = 0.999
    use_ema:                 bool  = True
    loss_weights:            Dict[str, float] = field(default_factory=dict)
    use_tensorboard:         bool  = True
    use_wandb:               bool  = False
    wandb_project:           str   = "physworldlm-entity"
    dataset_dir:             str   = "datasets/entity"
    seed:                    int   = 42
    debug:                   bool  = False
    save_confusion_matrices: bool  = True
    save_plots:              bool  = True

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be ≥ 1")
        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1)")

    def effective_batch_size(self) -> int:
        """Return batch_size × accumulation_steps."""
        return self.batch_size * self.accumulation_steps


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Default loss weights
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "entity_type":      1.0,
    "parent_class":     1.0,
    "root_class":       1.0,
    "coarse_class":     1.0,
    "material":         1.0,
    "phase":            1.0,
    "mobility":         1.0,
    "size_class":       1.0,
    "shape":            1.0,
    "mass_class":       1.0,
    "contact_type":     1.0,
    "stability":        1.0,
    "friction_class":   1.0,
    "restitution_class":1.0,
    "capabilities":     0.5,
    "affordances":      0.5,
    "scene_roles":      0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  Device selection
# ─────────────────────────────────────────────────────────────────────────────

def select_device(requested: str) -> torch.device:
    """Select compute device, printing the result.

    Args:
        requested: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns:
        :class:`torch.device`
    """
    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(requested)

    print(f"[device] selected: {dev}")
    if dev.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(dev)}")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMAModel:
    """Exponential Moving Average over model parameters.

    Maintains a shadow copy of the model weights updated as::

        ema_param = decay * ema_param + (1 - decay) * param

    Usage::

        ema = EMAModel(model, decay=0.999)
        for batch in loader:
            loss.backward()
            optimizer.step()
            ema.update(model)        # call after every optimizer step
        ema.apply(model)             # swap EMA weights in for evaluation
        ...
        ema.restore(model)           # swap original weights back

    Args:
        model: The live model whose weights are tracked.
        decay: EMA decay coefficient (0.999 is a good default).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    def update(self, model: nn.Module) -> None:
        """Update EMA shadow weights from the current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.float()
                )

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into the model (save originals in _backup)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.shadow = state["shadow"]
        self.decay  = state["decay"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  Loss construction
# ─────────────────────────────────────────────────────────────────────────────

def build_loss_dicts(
    head_names: List[str],
) -> Tuple[Dict[str, nn.CrossEntropyLoss], Dict[str, nn.BCEWithLogitsLoss]]:
    """Create per-head loss objects.

    Having one loss object per head enables future class-weighting and
    focal-loss without changing the training loop.

    Args:
        head_names: All head names as returned by model.get_head_dimensions().

    Returns:
        Tuple of (single_head_losses, multi_head_losses).
    """
    multi_names = {"capabilities", "affordances", "scene_roles"}

    single_losses: Dict[str, nn.CrossEntropyLoss]    = {}
    multi_losses:  Dict[str, nn.BCEWithLogitsLoss]   = {}

    for name in head_names:
        if name in multi_names:
            multi_losses[name]  = nn.BCEWithLogitsLoss()
        else:
            single_losses[name] = nn.CrossEntropyLoss()

    return single_losses, multi_losses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a single-label head."""
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def _top3_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-3 accuracy for a single-label head."""
    k = min(3, logits.size(-1))
    top_k = logits.topk(k, dim=-1).indices          # (B, k)
    correct = top_k.eq(targets.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def _binary_metrics(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Precision, recall, F1 (micro) and per-sample mAP for a multi-label head."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # Average Precision (per sample, then averaged)
    ap_list: List[float] = []
    for i in range(probs.size(0)):
        p  = probs[i].cpu().numpy()
        gt = targets[i].cpu().numpy()
        if gt.sum() == 0:
            continue
        order = np.argsort(-p)
        gt_sorted = gt[order]
        precision_at_k = np.cumsum(gt_sorted) / (np.arange(len(gt_sorted)) + 1)
        ap_list.append(float((precision_at_k * gt_sorted).sum() / (gt.sum() + 1e-8)))

    map_score = float(np.mean(ap_list)) if ap_list else 0.0

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "mAP":       map_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def _save_confusion_matrix(
    all_preds:  List[int],
    all_targets: List[int],
    head_name:  str,
    out_path:   Path,
    max_classes: int = 20,
) -> None:
    """Save a confusion-matrix PNG for ``head_name``."""
    if not _MPL_AVAILABLE or not _SKLEARN_AVAILABLE:
        return
    try:
        cm = _sklearn_cm(all_targets, all_preds)
        n  = cm.shape[0]
        if n > max_classes:
            return  # too many classes to display readably
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{head_name} — confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(f"[confusion_matrix] failed for {head_name}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  Training curves
# ─────────────────────────────────────────────────────────────────────────────

def _save_curves(history: List[Dict[str, Any]], plots_dir: Path) -> None:
    """Save loss / accuracy / F1 curve PNGs from training history."""
    if not _MPL_AVAILABLE or len(history) < 2:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    # Loss curve
    try:
        fig, ax = plt.subplots()
        ax.plot(epochs, [h["train_loss"] for h in history], label="train")
        ax.plot(epochs, [h["val_loss"]   for h in history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()
        fig.savefig(plots_dir / "loss_curve.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"[plots] loss_curve failed: {exc}")

    # Accuracy curve (entity_type)
    try:
        fig, ax = plt.subplots()
        key = "val_metrics/entity_type/accuracy"
        if history[0].get(key) is not None:
            ax.plot(epochs, [h.get(key, 0.0) for h in history], label="entity_type acc")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy curves")
            ax.legend()
            fig.savefig(plots_dir / "accuracy_curve.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"[plots] accuracy_curve failed: {exc}")

    # F1 curve
    try:
        fig, ax = plt.subplots()
        for head in ("capabilities", "affordances", "scene_roles"):
            key = f"val_metrics/{head}/f1"
            if history[0].get(key) is not None:
                ax.plot(epochs, [h.get(key, 0.0) for h in history], label=head)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.set_title("Multi-label F1 curves")
        ax.legend()
        fig.savefig(plots_dir / "f1_curve.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"[plots] f1_curve failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  –  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    path: Path,
    model: EntityEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[EMAModel],
    epoch: int,
    best_metrics: Dict[str, float],
    history: List[Dict[str, Any]],
) -> None:
    """Persist all training state to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "best_metrics":         best_metrics,
        "history":              history,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    torch.save(state, path)


def _load_checkpoint(
    path: Path,
    model: EntityEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[EMAModel],
) -> Tuple[int, Dict[str, float], List[Dict[str, Any]]]:
    """Load training state from ``path`` and return (start_epoch, best_metrics, history)."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    if ema is not None and "ema_state_dict" in state:
        ema.load_state_dict(state["ema_state_dict"])
    epoch        = state.get("epoch", 0)
    best_metrics = state.get("best_metrics", {})
    history      = state.get("history", [])
    print(f"[resume] loaded checkpoint from epoch {epoch}: {path}")
    return epoch, best_metrics, history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  –  Scheduler builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
) -> Any:
    """Build a linear-warmup + cosine-decay scheduler when transformers is available,
    falling back to CosineAnnealingLR otherwise."""
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    if _HF_SCHEDULER_AVAILABLE:
        return _hf_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: PLC0415
    return CosineAnnealingLR(optimizer, T_max=cfg.scheduler_tmax)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  –  Single epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model:         EntityEncoder,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    scheduler:     Any,
    scaler:        Optional[torch.cuda.amp.GradScaler],
    ema:           Optional[EMAModel],
    single_losses: Dict[str, nn.CrossEntropyLoss],
    multi_losses:  Dict[str, nn.BCEWithLogitsLoss],
    loss_weights:  Dict[str, float],
    cfg:           TrainConfig,
    device:        torch.device,
    epoch:         int,
    tb_writer:     Any,
    global_step:   int,
) -> Tuple[float, float, Dict[str, float], int]:
    """Run one training epoch.

    Returns:
        (train_loss, grad_norm_mean, head_losses_dict, updated_global_step)
    """
    model.train()
    use_amp = cfg.mixed_precision and device.type == "cuda"

    running_loss    = 0.0
    running_head    : Dict[str, float] = defaultdict(float)
    grad_norms      : List[float] = []
    n_batches       = 0
    n_nan_skipped   = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(input_ids, attention_mask)
            logits = output.logits

            total_loss  = torch.tensor(0.0, device=device)
            single_sum  = torch.tensor(0.0, device=device)
            multi_sum   = torch.tensor(0.0, device=device)

            for name, loss_fn in single_losses.items():
                if name not in logits:
                    continue
                targets = batch[name].to(device, non_blocking=True)
                w       = loss_weights.get(name, 1.0)
                h_loss  = loss_fn(logits[name], targets)
                total_loss = total_loss + w * h_loss
                single_sum = single_sum + w * h_loss
                running_head[name] += h_loss.item()

            for name, loss_fn in multi_losses.items():
                if name not in logits:
                    continue
                targets = batch[name].to(device, non_blocking=True)
                w       = loss_weights.get(name, 1.0)
                h_loss  = loss_fn(logits[name], targets)
                total_loss = total_loss + w * h_loss
                multi_sum  = multi_sum  + w * h_loss
                running_head[name] += h_loss.item()

        # NaN safety guard
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            n_nan_skipped += 1
            warnings.warn(
                f"[train] NaN/Inf loss at step {step} (epoch {epoch}); skipping batch."
            )
            optimizer.zero_grad()
            continue

        # Scale loss for gradient accumulation
        scaled = total_loss / cfg.accumulation_steps
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        # Optimizer step every accumulation_steps batches
        if (step + 1) % cfg.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.gradient_clip
            ).item()
            grad_norms.append(grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)

            global_step += 1

        running_loss += total_loss.item()
        n_batches    += 1

        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/step_loss", total_loss.item(), global_step)

    if n_nan_skipped > 0:
        print(f"  [warn] {n_nan_skipped} NaN batches skipped this epoch.")

    avg_loss     = running_loss / max(n_batches, 1)
    avg_grad     = float(np.mean(grad_norms)) if grad_norms else 0.0
    avg_head     = {k: v / max(n_batches, 1) for k, v in running_head.items()}

    return avg_loss, avg_grad, avg_head, global_step


@torch.no_grad()
def _val_epoch(
    model:         EntityEncoder,
    loader:        DataLoader,
    single_losses: Dict[str, nn.CrossEntropyLoss],
    multi_losses:  Dict[str, nn.BCEWithLogitsLoss],
    loss_weights:  Dict[str, float],
    cfg:           TrainConfig,
    device:        torch.device,
    epoch:         int,
    plots_dir:     Path,
) -> Tuple[float, Dict[str, Any]]:
    """Run one validation pass.

    Returns:
        (val_loss, metrics_dict)
    """
    model.eval()
    use_amp = cfg.mixed_precision and device.type == "cuda"

    running_loss = 0.0
    n_batches    = 0

    # Accumulators
    all_single_preds:   Dict[str, List[int]]            = defaultdict(list)
    all_single_targets: Dict[str, List[int]]            = defaultdict(list)
    all_multi_logits:   Dict[str, List[torch.Tensor]]   = defaultdict(list)
    all_multi_targets:  Dict[str, List[torch.Tensor]]   = defaultdict(list)

    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(input_ids, attention_mask)
            logits = output.logits
            total  = torch.tensor(0.0, device=device)

            for name, loss_fn in single_losses.items():
                if name not in logits:
                    continue
                targets = batch[name].to(device, non_blocking=True)
                w = loss_weights.get(name, 1.0)
                total = total + w * loss_fn(logits[name], targets)
                all_single_preds[name].extend(
                    logits[name].argmax(-1).cpu().tolist()
                )
                all_single_targets[name].extend(targets.cpu().tolist())

            for name, loss_fn in multi_losses.items():
                if name not in logits:
                    continue
                targets = batch[name].to(device, non_blocking=True)
                w = loss_weights.get(name, 1.0)
                total = total + w * loss_fn(logits[name], targets)
                all_multi_logits[name].append(logits[name].cpu())
                all_multi_targets[name].append(targets.cpu())

        if not (torch.isnan(total) or torch.isinf(total)):
            running_loss += total.item()
            n_batches    += 1

    avg_loss = running_loss / max(n_batches, 1)
    metrics: Dict[str, Any] = {}

    # Single-label metrics
    for name in all_single_preds:
        preds_t   = torch.tensor(all_single_preds[name])
        targets_t = torch.tensor(all_single_targets[name])
        acc  = _accuracy(
            torch.zeros(len(preds_t), preds_t.max().item() + 1).scatter_(
                1, preds_t.unsqueeze(1), 1.0
            ),
            targets_t,
        )
        top3 = _top3_accuracy(
            torch.zeros(len(preds_t), preds_t.max().item() + 1).scatter_(
                1, preds_t.unsqueeze(1), 1.0
            ),
            targets_t,
        )
        metrics[f"{name}/accuracy"] = acc
        metrics[f"{name}/top3_accuracy"] = top3

        # Confusion matrix for key heads
        if cfg.save_confusion_matrices and name in ("entity_type", "material", "shape"):
            _save_confusion_matrix(
                all_single_preds[name],
                all_single_targets[name],
                name,
                plots_dir / f"{name}_confusion_e{epoch:03d}.png",
            )

    # Multi-label metrics
    for name in all_multi_logits:
        cat_logits  = torch.cat(all_multi_logits[name],  dim=0)
        cat_targets = torch.cat(all_multi_targets[name], dim=0)
        bm = _binary_metrics(cat_logits, cat_targets)
        for k, v in bm.items():
            metrics[f"{name}/{k}"] = v

    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13  –  Main train() function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Optional[TrainConfig] = None) -> EntityEncoder:
    """Full training pipeline for EntityEncoder.

    Args:
        cfg: :class:`TrainConfig` instance.  Defaults are used when None.

    Returns:
        The trained :class:`EntityEncoder` (with best-loss weights loaded).
    """
    if cfg is None:
        cfg = TrainConfig()

    set_seed(cfg.seed)

    # ── checkpoint directories ────────────────────────────────────────────────
    ckpt_root   = Path(cfg.checkpoint_dir)
    dirs = {
        "best_loss":    ckpt_root / "best_loss",
        "best_entity":  ckpt_root / "best_entity",
        "best_cap":     ckpt_root / "best_capability",
        "ema":          ckpt_root / "ema",
        "history":      ckpt_root / "history",
        "plots":        ckpt_root / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = select_device(cfg.device)

    # ── datasets ──────────────────────────────────────────────────────────────
    print("[data] loading datasets …")
    train_ds: EntityDataset = EntityDataset(
        DatasetConfig(split="train", dataset_dir=cfg.dataset_dir, verify_labels=False)
    )
    val_ds:   EntityDataset = EntityDataset(
        DatasetConfig(split="val",   dataset_dir=cfg.dataset_dir, verify_labels=False)
    )

    if cfg.debug:
        print("[debug] truncating to 100 samples, 1 epoch")
        train_ds = Subset(train_ds, range(min(100, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(100, len(val_ds))))
        cfg.epochs = 1

    # ── data loaders ──────────────────────────────────────────────────────────
    num_workers = 0 if cfg.debug else cfg.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print("[model] building EntityEncoder …")
    model = build_entity_encoder(dataset_dir=cfg.dataset_dir)
    model.to(device)

    if cfg.freeze_backbone_epochs > 0:
        print(f"[model] backbone frozen for first {cfg.freeze_backbone_epochs} epochs")
        model.freeze_backbone()

    # ── loss objects ──────────────────────────────────────────────────────────
    head_names   = list(model.get_head_dimensions().keys())
    single_losses, multi_losses = build_loss_dicts(head_names)

    # Resolve loss weights (user overrides merged with defaults)
    loss_weights: Dict[str, float] = {**_DEFAULT_LOSS_WEIGHTS, **cfg.loss_weights}

    # ── optimizer — layerwise LR ──────────────────────────────────────────────
    backbone_params   = list(model.backbone.parameters())
    projection_params = list(model.projection.parameters())
    head_params       = [
        p for p in model.parameters()
        if not any(p is b for b in backbone_params)
        and not any(p is proj for proj in projection_params)
    ]
    optimizer = AdamW(
        [
            {"params": backbone_params,   "lr": cfg.lr_backbone,   "name": "backbone"},
            {"params": projection_params, "lr": cfg.lr_projection, "name": "projection"},
            {"params": head_params,       "lr": cfg.lr_heads,      "name": "heads"},
        ],
        weight_decay=cfg.weight_decay,
    )

    # ── scheduler ─────────────────────────────────────────────────────────────
    total_steps = (len(train_loader) // cfg.accumulation_steps) * cfg.epochs
    scheduler   = _build_scheduler(optimizer, cfg, total_steps)

    # ── AMP ───────────────────────────────────────────────────────────────────
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if cfg.mixed_precision and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema: Optional[EMAModel] = None
    if cfg.use_ema:
        ema = EMAModel(model, decay=cfg.ema_decay)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_writer = None
    if cfg.use_tensorboard and _TENSORBOARD_AVAILABLE:
        tb_writer = _TBWriter(log_dir=str(ckpt_root / "tensorboard"))
        print("[tensorboard] logging enabled")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if cfg.use_wandb and _WANDB_AVAILABLE:
        _wandb.init(project=cfg.wandb_project, config=asdict(cfg))
        print("[wandb] logging enabled")

    # ── persist configs ───────────────────────────────────────────────────────
    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    (ckpt_root / "train_config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )
    model_cfg = asdict(model.config)
    model_cfg["head_dimensions"] = model.get_head_dimensions()
    (ckpt_root / "model_config.json").write_text(
        json.dumps(model_cfg, indent=2), encoding="utf-8"
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_metrics  : Dict[str, float] = {
        "best_val_loss":       float("inf"),
        "best_entity_acc":     0.0,
        "best_capability_f1":  0.0,
    }
    history: List[Dict[str, Any]] = []

    if cfg.resume_checkpoint is not None:
        resume_path = Path(cfg.resume_checkpoint)
        start_epoch, best_metrics, history = _load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, ema
        )
        start_epoch += 1   # continue from next epoch

    # ── early stopping state ──────────────────────────────────────────────────
    patience_counter = 0
    global_step      = start_epoch * (len(train_loader) // cfg.accumulation_steps)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n[train] starting — {cfg.epochs} epochs | "
          f"effective batch = {cfg.effective_batch_size()} | "
          f"device = {device}")

    for epoch in range(start_epoch, cfg.epochs):

        # Backbone freeze / unfreeze schedule
        if epoch == cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
            print(f"[epoch {epoch}] backbone unfrozen")

        # ── train ─────────────────────────────────────────────────────────────
        train_loss, grad_norm, head_train_losses, global_step = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema,
            single_losses, multi_losses, loss_weights,
            cfg, device, epoch, tb_writer, global_step,
        )

        # ── validation (raw model) ─────────────────────────────────────────────
        val_loss, val_metrics = _val_epoch(
            model, val_loader, single_losses, multi_losses, loss_weights,
            cfg, device, epoch, dirs["plots"],
        )

        # ── EMA validation ────────────────────────────────────────────────────
        ema_val_loss: Optional[float] = None
        if ema is not None:
            ema.apply(model)
            ema_val_loss, ema_metrics = _val_epoch(
                model, val_loader, single_losses, multi_losses, loss_weights,
                cfg, device, epoch, dirs["plots"],
            )
            ema.restore(model)

        # ── logging ───────────────────────────────────────────────────────────
        entity_acc   = val_metrics.get("entity_type/accuracy", 0.0)
        material_acc = val_metrics.get("material/accuracy", 0.0)
        shape_acc    = val_metrics.get("shape/accuracy", 0.0)
        cap_f1       = val_metrics.get("capabilities/f1", 0.0)
        aff_f1       = val_metrics.get("affordances/f1", 0.0)
        role_f1      = val_metrics.get("scene_roles/f1", 0.0)
        lr_now       = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d}/{cfg.epochs - 1}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
            + (f"  ema_val={ema_val_loss:.4f}" if ema_val_loss is not None else "")
            + f"\n"
            f"  entity_acc={entity_acc:.3f}  material_acc={material_acc:.3f}  "
            f"shape_acc={shape_acc:.3f}\n"
            f"  cap_F1={cap_f1:.3f}  aff_F1={aff_f1:.3f}  "
            f"role_F1={role_f1:.3f}\n"
            f"  grad_norm={grad_norm:.3f}  lr={lr_now:.2e}"
        )

        # ── history entry ─────────────────────────────────────────────────────
        history_entry: Dict[str, Any] = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "grad_norm":  grad_norm,
            "lr":         lr_now,
        }
        for k, v in val_metrics.items():
            history_entry[f"val_metrics/{k}"] = v
        if ema_val_loss is not None:
            history_entry["ema_val_loss"] = ema_val_loss
        history.append(history_entry)

        # ── TensorBoard ───────────────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
            tb_writer.add_scalar("epoch/val_loss",   val_loss,   epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"epoch/val/{k}", v, epoch)

        # ── WandB ─────────────────────────────────────────────────────────────
        if cfg.use_wandb and _WANDB_AVAILABLE:
            log_dict = {"train_loss": train_loss, "val_loss": val_loss, **val_metrics}
            _wandb.log(log_dict, step=epoch)

        # ── best checkpoints ──────────────────────────────────────────────────
        def _ckpt_state() -> Dict[str, Any]:
            """Shared state for all checkpoint saves."""
            return dict(
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, ema=ema, epoch=epoch,
                best_metrics=best_metrics, history=history,
            )

        if val_loss < best_metrics["best_val_loss"]:
            best_metrics["best_val_loss"] = val_loss
            _save_checkpoint(
                dirs["best_loss"] / "entity_encoder_best_loss.pt",
                **_ckpt_state(),
            )
            print(f"  ✓ best_loss checkpoint saved  (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if entity_acc > best_metrics["best_entity_acc"]:
            best_metrics["best_entity_acc"] = entity_acc
            _save_checkpoint(
                dirs["best_entity"] / "entity_encoder_best_entity.pt",
                **_ckpt_state(),
            )
            print(f"  ✓ best_entity checkpoint saved  (entity_acc={entity_acc:.3f})")

        if cap_f1 > best_metrics["best_capability_f1"]:
            best_metrics["best_capability_f1"] = cap_f1
            _save_checkpoint(
                dirs["best_cap"] / "entity_encoder_best_capability.pt",
                **_ckpt_state(),
            )
            print(f"  ✓ best_capability checkpoint saved  (cap_F1={cap_f1:.3f})")

        # EMA checkpoint
        if ema is not None:
            ema.apply(model)
            _save_checkpoint(
                dirs["ema"] / "entity_encoder_ema.pt",
                **_ckpt_state(),
            )
            ema.restore(model)

        # Last checkpoint
        if cfg.save_every_epoch:
            _save_checkpoint(
                ckpt_root / "entity_encoder_last.pt",
                **_ckpt_state(),
            )

        # History JSON
        (dirs["history"] / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        # Plots
        if cfg.save_plots:
            _save_curves(history, dirs["plots"])

        # ── early stopping ────────────────────────────────────────────────────
        if patience_counter >= cfg.patience:
            print(f"\n[early stopping] no improvement for {cfg.patience} epochs. stopping.")
            break

    # ── final metrics ─────────────────────────────────────────────────────────
    final_metrics: Dict[str, Any] = {
        "best_val_loss":      best_metrics["best_val_loss"],
        "best_entity_acc":    best_metrics["best_entity_acc"],
        "best_capability_f1": best_metrics["best_capability_f1"],
        "total_epochs":       epoch + 1,
    }
    if history:
        final_metrics["last_val_metrics"] = {
            k: v for k, v in history[-1].items()
            if k.startswith("val_metrics/")
        }

    (ckpt_root / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    if tb_writer is not None:
        tb_writer.close()
    if cfg.use_wandb and _WANDB_AVAILABLE:
        _wandb.finish()

    # ── final report ──────────────────────────────────────────────────────────
    best_epoch = max(
        (h["epoch"] for h in history if h["val_loss"] == best_metrics["best_val_loss"]),
        default=0,
    )
    print(f"\n{'═'*60}")
    print("  Training complete")
    print(f"{'═'*60}")
    print(f"  Best epoch         : {best_epoch}")
    print(f"  Best val loss      : {best_metrics['best_val_loss']:.4f}")
    print(f"  Best entity acc    : {best_metrics['best_entity_acc']:.3f}")
    print(f"  Best capability F1 : {best_metrics['best_capability_f1']:.3f}")
    print(f"  Checkpoints        : {ckpt_root.resolve()}")
    print(f"{'═'*60}\n")

    # Load best-loss weights before returning
    best_ckpt = dirs["best_loss"] / "entity_encoder_best_loss.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: build a TrainConfig and launch training."""
    cfg = TrainConfig(
        batch_size=64,
        num_workers=4,
        epochs=50,
        lr_backbone=2e-5,
        lr_projection=5e-5,
        lr_heads=1e-4,
        weight_decay=1e-2,
        gradient_clip=1.0,
        accumulation_steps=1,
        warmup_ratio=0.06,
        patience=8,
        checkpoint_dir="checkpoints",
        device="auto",
        mixed_precision=True,
        freeze_backbone_epochs=0,
        use_ema=True,
        ema_decay=0.999,
        use_tensorboard=True,
        use_wandb=False,
        save_confusion_matrices=True,
        save_plots=True,
        seed=42,
        debug=False,
    )
    train(cfg)


if __name__ == "__main__":
    main()
