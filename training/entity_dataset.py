"""
training/entity_dataset.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — first PyTorch component.

Bridges the dataset preparation pipeline (split_entity_dataset.py) to the
neural entity encoder (models/entity_encoder.py).

Reads:
    datasets/entity/{train,val,test}.jsonl
    datasets/entity/label_maps/<field>_to_id.json
    datasets/entity/metadata.json
    datasets/entity/dataset_registry.json

Produces per-sample dicts of:
    input_ids, attention_mask          (torch.long, shape [max_length])
    entity_type ... restitution_class  (torch.long, scalar)
    capabilities, affordances,
    scene_roles                        (torch.float32, multi-hot vectors)

Design principles
─────────────────
* DatasetConfig dataclass — no hardcoded arguments inside EntityDataset.
* DatasetMetadata dataclass — structured view of metadata.json; EntityEncoder
  reads metadata.num_capabilities etc. directly.
* LabelMapManager — centralises encode / decode / num_classes; prevents dict
  access scattered across the codebase.
* Tokenization cache (self.token_cache) — each unique token string is
  tokenised exactly once; critical when the same entity surface form appears
  across many records.
* Dataset registry support — resolves filenames from dataset_registry.json
  rather than hardcoding them; supports future entity_dataset_v2/v3.
* Shared multi-hot encoder — encode_multihot() used for all three multi-label
  heads.
* Tensor validator — validates shapes against config at __getitem__ time when
  verify_labels is True.
* Optional raw-record return — return_raw_record=True appends "raw_record" to
  each sample for debugging.
* Placeholder relation/graph targets — self.relation_targets,
  self.graph_targets reserved for future RelationEncoder / GraphBuilder.
* API functions — get_num_classes(), get_head_dimensions(),
  get_vocab_sizes(), decode_sample(), print_sample(), validate_sample().

No DataLoader, no batching, no collate_fn, no neural networks, no losses,
no optimiser, no scheduler, no training loop, no inference code.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH: int = 32

#: Every single-label head that maps to one integer class ID.
SINGLE_LABEL_FIELDS: Tuple[str, ...] = (
    "entity_type",
    "parent_class",
    "root_class",
    "coarse_class",
    "material",
    "phase",
    "mobility",
    "size_class",
    "shape",
    "mass_class",
    "contact_type",
    "stability",
    "friction_class",
    "restitution_class",
)

#: Every multi-label head that maps to a float multi-hot vector.
MULTI_LABEL_FIELDS: Tuple[str, ...] = (
    "capabilities",
    "affordances",
    "scene_roles",
)

#: The registry key for the current dataset version.
REGISTRY_KEY: str = "entity_dataset_v1"

#: Valid split names.
VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  DatasetConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """All configuration for EntityDataset — no magic numbers inside the class.

    Attributes:
        split:               Which split to load ("train", "val", "test").
        dataset_dir:         Root directory produced by split_entity_dataset.py.
        tokenizer_name:      HuggingFace model name / local path for the
                             tokenizer used to encode entity tokens.
        max_length:          Tokenizer padding / truncation length.
        cache_tokenization:  When True, each unique entity-token string is
                             tokenised only once and cached in self.token_cache.
        return_raw_record:   When True, __getitem__ appends the original dict
                             as "raw_record" — useful during debugging.
        use_metadata:        Load and expose DatasetMetadata from metadata.json.
        verify_labels:       Warn (never crash) when a label is absent from its
                             map.  Also validates tensor shapes at __getitem__.
        debug:               Print a sample and head dimensions on init.
        registry_key:        Which key to look up inside dataset_registry.json.
    """

    split: str = "train"
    dataset_dir: str = "datasets/entity"
    tokenizer_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    cache_tokenization: bool = True
    return_raw_record: bool = False
    use_metadata: bool = True
    verify_labels: bool = True
    debug: bool = False
    registry_key: str = REGISTRY_KEY

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {VALID_SPLITS}, got {self.split!r}"
            )
        if self.max_length < 1:
            raise ValueError(f"max_length must be ≥ 1, got {self.max_length}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  DatasetMetadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetMetadata:
    """Structured representation of metadata.json.

    EntityEncoder (and downstream consumers) should read class counts from
    this object — e.g. ``metadata.num_capabilities`` — rather than calling
    ``len(label_map)`` at every site.

    Attributes are populated from the JSON file; any key absent in the file
    is silently left at 0 so that newer metadata formats do not break older
    consumers.
    """

    version: str = "unknown"
    created_at: str = ""
    seed: int = 42
    num_samples: int = 0
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0

    num_entity_types: int = 0
    num_parent_classes: int = 0
    num_root_classes: int = 0
    num_coarse_classes: int = 0
    num_materials: int = 0
    num_phases: int = 0
    num_mobility_classes: int = 0
    num_size_classes: int = 0
    num_shapes: int = 0
    num_mass_classes: int = 0
    num_contact_types: int = 0
    num_stability_classes: int = 0
    num_friction_classes: int = 0
    num_restitution_classes: int = 0

    num_capabilities: int = 0
    num_affordances: int = 0
    num_scene_roles: int = 0

    negative_examples_count: int = 0
    hard_examples_count: int = 0
    ambiguous_examples_count: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "DatasetMetadata":
        """Load from metadata.json, tolerating missing keys."""
        with path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
        obj = cls()
        for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if f in raw:
                setattr(obj, f, raw[f])
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  LabelMapManager
# ─────────────────────────────────────────────────────────────────────────────

class LabelMapManager:
    """Centralised label-encoding / decoding hub.

    Instead of ``self.label_maps["entity_type"][value]`` scattered everywhere,
    callers use::

        mgr.encode("entity_type", value)   -> int
        mgr.decode("entity_type", 3)       -> str
        mgr.num_classes("entity_type")     -> int

    For multi-hot fields (capabilities, affordances, scene_roles) the forward
    map is also available via encode(), which returns an int index into the
    multi-hot vector.

    Missing labels are handled by returning the ``__unknown__`` index when
    present in the map, or emitting a warning and returning -1 otherwise.
    Decoding an out-of-range index emits a warning and returns "__unknown__".
    """

    def __init__(self, label_maps: Dict[str, Dict[str, int]]) -> None:
        """
        Args:
            label_maps: dict of {map_name -> {label_str -> int_id}}.
                        map_name is the field name for single-label heads
                        ("entity_type", …) and the base vocabulary name for
                        multi-label heads ("capability", "affordance",
                        "scene_role" — note: singular, matching the JSON
                        filename convention from split_entity_dataset.py).
        """
        self._forward: Dict[str, Dict[str, int]] = label_maps
        # Build reverse maps lazily to avoid wasted work if decode is unused.
        self._reverse: Dict[str, Dict[int, str]] = {}

    # ── forward ──────────────────────────────────────────────────────────────

    def encode(self, map_name: str, value: str) -> int:
        """Return the integer ID for ``value`` in ``map_name``.

        Falls back to the ``__unknown__`` entry if present; otherwise emits a
        warning and returns -1 (never raises, to keep DataLoaders alive).
        """
        forward = self._forward.get(map_name)
        if forward is None:
            warnings.warn(
                f"LabelMapManager: unknown map {map_name!r}; returning -1",
                stacklevel=2,
            )
            return -1
        if value in forward:
            return forward[value]
        if "__unknown__" in forward:
            warnings.warn(
                f"LabelMapManager: label {value!r} not in map {map_name!r}; "
                "mapping to __unknown__",
                stacklevel=2,
            )
            return forward["__unknown__"]
        warnings.warn(
            f"LabelMapManager: label {value!r} not in map {map_name!r} and "
            "no __unknown__ entry; returning -1",
            stacklevel=2,
        )
        return -1

    # ── reverse ───────────────────────────────────────────────────────────────

    def decode(self, map_name: str, idx: int) -> str:
        """Return the label string for integer ``idx`` in ``map_name``."""
        if map_name not in self._reverse:
            self._build_reverse(map_name)
        reverse = self._reverse.get(map_name, {})
        if idx in reverse:
            return reverse[idx]
        warnings.warn(
            f"LabelMapManager: index {idx} out of range for map {map_name!r}; "
            "returning '__unknown__'",
            stacklevel=2,
        )
        return "__unknown__"

    def _build_reverse(self, map_name: str) -> None:
        forward = self._forward.get(map_name, {})
        self._reverse[map_name] = {v: k for k, v in forward.items()}

    # ── sizes ─────────────────────────────────────────────────────────────────

    def num_classes(self, map_name: str) -> int:
        """Return the vocabulary size for ``map_name``."""
        forward = self._forward.get(map_name)
        if forward is None:
            warnings.warn(
                f"LabelMapManager: unknown map {map_name!r}; returning 0",
                stacklevel=2,
            )
            return 0
        return len(forward)

    def map_names(self) -> List[str]:
        """Return all loaded map names."""
        return list(self._forward.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  Dataset registry resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_split_path(
    dataset_dir: Path,
    split: str,
    registry_key: str,
) -> Path:
    """Resolve the JSONL filename for ``split`` via dataset_registry.json.

    Falls back to ``<split>.jsonl`` if the registry is absent or the key is
    not found — this keeps the dataset usable in environments where the
    registry was not written.
    """
    registry_path = dataset_dir / "dataset_registry.json"
    fallback = dataset_dir / f"{split}.jsonl"

    if not registry_path.exists():
        return fallback

    with registry_path.open("r", encoding="utf-8") as fh:
        registry: Dict[str, Any] = json.load(fh)

    entry = registry.get(registry_key)
    if entry is None:
        return fallback

    filename = entry.get(split, f"{split}.jsonl")
    return dataset_dir / filename


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  Label-map loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_label_maps(label_maps_dir: Path) -> Dict[str, Dict[str, int]]:
    """Load every ``*_to_id.json`` file from ``label_maps_dir``.

    Map names are derived from filenames: ``entity_type_to_id.json`` → key
    ``"entity_type"``, ``capability_to_id.json`` → key ``"capability"``.
    """
    maps: Dict[str, Dict[str, int]] = {}
    if not label_maps_dir.exists():
        raise FileNotFoundError(
            f"label_maps directory not found: {label_maps_dir}\n"
            "Run dataset_gen/split_entity_dataset.py first."
        )
    for json_file in sorted(label_maps_dir.glob("*_to_id.json")):
        # "entity_type_to_id.json" -> "entity_type"
        map_name = json_file.stem.replace("_to_id", "")
        with json_file.open("r", encoding="utf-8") as fh:
            maps[map_name] = json.load(fh)
    if not maps:
        raise RuntimeError(
            f"No *_to_id.json files found in {label_maps_dir}. "
            "The split pipeline may not have run successfully."
        )
    return maps


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  JSONL loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, skipping blank lines.  Raises on missing file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run dataset_gen/split_entity_dataset.py first."
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{lineno}: {exc}") from exc
            records.append(obj)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  EntityDataset
# ─────────────────────────────────────────────────────────────────────────────

class EntityDataset(Dataset):
    """PyTorch Dataset bridging split JSONL files to the entity encoder.

    Each ``__getitem__`` call returns a dict of tensors keyed by field name.
    All label-to-ID conversions, tokenisation caching, multi-hot encoding,
    and tensor shape validation are handled transparently.

    Usage::

        cfg = DatasetConfig(split="train")
        ds  = EntityDataset(cfg)
        sample = ds[0]
        print(sample["input_ids"].shape)   # torch.Size([32])
        print(ds.metadata.num_capabilities)

    Backward-compatible constructor::

        ds = EntityDataset("train")           # classic positional usage
        ds = EntityDataset("train", debug=True)
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        split_or_config: "str | DatasetConfig" = "train",
        dataset_dir: str = "datasets/entity",
        tokenizer_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        *,
        cache_tokenization: bool = True,
        return_raw_record: bool = False,
        use_metadata: bool = True,
        verify_labels: bool = True,
        debug: bool = False,
        registry_key: str = REGISTRY_KEY,
    ) -> None:
        """Initialise the dataset.

        Accepts either a :class:`DatasetConfig` as the first argument (new
        API) or the legacy positional ``split`` string (old API), so that
        existing code does not break.

        Args:
            split_or_config:    ``"train"``, ``"val"``, ``"test"`` *or* a
                                :class:`DatasetConfig` instance.
            dataset_dir:        Root of the entity dataset tree (ignored when
                                ``split_or_config`` is a DatasetConfig).
            tokenizer_name:     HuggingFace identifier (ignored when config).
            max_length:         Token sequence length (ignored when config).
            cache_tokenization: Cache tokenised token strings (default True).
            return_raw_record:  Include "raw_record" in each sample dict.
            use_metadata:       Load and expose DatasetMetadata.
            verify_labels:      Warn on missing labels / wrong shapes.
            debug:              Print shapes and a sample after init.
            registry_key:       Entry key in dataset_registry.json.
        """
        if isinstance(split_or_config, DatasetConfig):
            cfg = split_or_config
        else:
            cfg = DatasetConfig(
                split=split_or_config,
                dataset_dir=dataset_dir,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                cache_tokenization=cache_tokenization,
                return_raw_record=return_raw_record,
                use_metadata=use_metadata,
                verify_labels=verify_labels,
                debug=debug,
                registry_key=registry_key,
            )

        self.config: DatasetConfig = cfg
        self._dataset_dir: Path = Path(cfg.dataset_dir)

        # ── records ───────────────────────────────────────────────────────────
        split_path = _resolve_split_path(
            self._dataset_dir, cfg.split, cfg.registry_key
        )
        self.records: List[Dict[str, Any]] = _load_jsonl(split_path)

        # ── label maps ────────────────────────────────────────────────────────
        label_maps_dir = self._dataset_dir / "label_maps"
        raw_maps = _load_label_maps(label_maps_dir)
        self.label_maps: Dict[str, Dict[str, int]] = raw_maps
        self.label_map_manager: LabelMapManager = LabelMapManager(raw_maps)

        # ── metadata ──────────────────────────────────────────────────────────
        metadata_path = self._dataset_dir / "metadata.json"
        if cfg.use_metadata and metadata_path.exists():
            self.metadata: DatasetMetadata = DatasetMetadata.from_json(metadata_path)
        else:
            self.metadata = DatasetMetadata()

        # ── tokenizer (loaded once; shared across all workers via fork) ───────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.max_length: int = cfg.max_length

        # ── tokenization cache ────────────────────────────────────────────────
        # Maps raw token string -> {"input_ids": Tensor, "attention_mask": Tensor}
        self.token_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # ── future relation / graph placeholders ──────────────────────────────
        # Reserved for RelationEncoder and GraphBuilder integration.
        self.relation_targets: Optional[Any] = None
        self.graph_targets: Optional[Any] = None

        if cfg.debug:
            print(f"\n[EntityDataset] split={cfg.split!r} "
                  f"| {len(self.records):,} records "
                  f"| max_length={cfg.max_length}")
            print("[EntityDataset] head dimensions:")
            for k, v in self.get_head_dimensions().items():
                print(f"  {k:<22} {v}")
            if len(self.records) > 0:
                print("[EntityDataset] sample[0]:")
                self.print_sample(0)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of records in this split."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the tensor dict for record at position ``idx``.

        Returns:
            Dict with keys:
                input_ids, attention_mask — shape (max_length,), torch.long
                entity_type … restitution_class — scalar torch.long
                capabilities, affordances, scene_roles — multi-hot torch.float32
            Optionally:
                raw_record — the original dict (when config.return_raw_record)
        """
        record = self.records[idx]

        # ── tokenise entity token ─────────────────────────────────────────────
        token_str: str = str(record.get("token", ""))
        enc = self._tokenize(token_str)

        # ── single-label heads ────────────────────────────────────────────────
        single: Dict[str, torch.Tensor] = {}
        for fname in SINGLE_LABEL_FIELDS:
            single[fname] = self.encode_single_label(
                record.get(fname, "__unknown__"), fname
            )

        # ── multi-label heads ─────────────────────────────────────────────────
        capabilities = self.encode_multihot(
            record.get("capabilities") or [], "capability"
        )
        affordances = self.encode_multihot(
            record.get("affordances") or [], "affordance"
        )
        scene_roles = self.encode_multihot(
            record.get("scene_roles") or [], "scene_role"
        )

        sample: Dict[str, Any] = {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            **single,
            "capabilities": capabilities,
            "affordances":  affordances,
            "scene_roles":  scene_roles,
        }

        if self.config.verify_labels:
            self.validate_sample(sample)

        if self.config.return_raw_record:
            sample["raw_record"] = record  # type: ignore[assignment]

        return sample

    # ── tokenisation ─────────────────────────────────────────────────────────

    def _tokenize(self, token: str) -> Dict[str, torch.Tensor]:
        """Tokenise ``token``, using the cache when enabled.

        Returns:
            dict with "input_ids" and "attention_mask", each shape (max_length,)
            and dtype torch.long.
        """
        if self.config.cache_tokenization and token in self.token_cache:
            return self.token_cache[token]

        encoded = self.tokenizer(
            token,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        result: Dict[str, torch.Tensor] = {
            "input_ids":      encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        }

        if self.config.cache_tokenization:
            self.token_cache[token] = result

        return result

    # ── label encoding helpers ────────────────────────────────────────────────

    def encode_single_label(self, value: Any, map_name: str) -> torch.Tensor:
        """Encode a single categorical label to a scalar ``torch.long`` tensor.

        Args:
            value:    The label string (e.g. "particle").
            map_name: The label map key (e.g. "entity_type").

        Returns:
            Scalar torch.long tensor containing the integer class ID.

        Raises:
            ValueError: Only when ``__unknown__`` is absent and
                        ``verify_labels`` is False (strict mode).  In all
                        other cases a warning is emitted and -1 is returned so
                        that DataLoader workers do not die.
        """
        str_value = str(value) if value is not None else "__unknown__"
        forward = self.label_maps.get(map_name, {})

        if str_value in forward:
            return torch.tensor(forward[str_value], dtype=torch.long)

        if "__unknown__" in forward:
            if self.config.verify_labels:
                warnings.warn(
                    f"encode_single_label: '{str_value}' not in map "
                    f"'{map_name}'; using __unknown__",
                    stacklevel=2,
                )
            return torch.tensor(forward["__unknown__"], dtype=torch.long)

        if not self.config.verify_labels:
            raise ValueError(
                f"encode_single_label: label {str_value!r} absent from map "
                f"{map_name!r} and no __unknown__ entry exists."
            )

        warnings.warn(
            f"encode_single_label: '{str_value}' not in map '{map_name}' "
            "and no __unknown__ entry; returning -1",
            stacklevel=2,
        )
        return torch.tensor(-1, dtype=torch.long)

    def encode_multihot(self, values: List[str], vocab_name: str) -> torch.Tensor:
        """Encode a list of labels into a multi-hot ``torch.float32`` vector.

        The vector length equals the vocabulary size for ``vocab_name``.
        Each position is set to 1.0 if the corresponding label is present in
        ``values``.  Unknown labels are silently skipped (with a warning when
        verify_labels is True).

        Args:
            values:     List of active label strings.
            vocab_name: Vocabulary key, e.g. "capability", "affordance",
                        "scene_role" (singular, matching *_to_id.json names).

        Returns:
            1-D torch.float32 tensor of length ``num_classes(vocab_name)``.
        """
        forward = self.label_maps.get(vocab_name, {})
        n = len(forward)
        vec = torch.zeros(n, dtype=torch.float32)

        for v in values:
            key = str(v)
            if key in forward:
                vec[forward[key]] = 1.0
            elif self.config.verify_labels:
                warnings.warn(
                    f"encode_multihot: '{key}' not in vocab '{vocab_name}'; skipped",
                    stacklevel=2,
                )

        return vec

    # ── encode_multi_label alias (spec compliance) ────────────────────────────

    def encode_multi_label(self, values: List[str], vocab_name: str) -> torch.Tensor:
        """Alias for :meth:`encode_multihot` (satisfies original spec name)."""
        return self.encode_multihot(values, vocab_name)

    # ── API functions ─────────────────────────────────────────────────────────

    def get_num_classes(self) -> Dict[str, int]:
        """Return per-head vocabulary sizes.

        For single-label heads this is the number of discrete classes.
        For multi-label heads this is the multi-hot vector length.

        These values are consumed by EntityEncoder to set output projection
        sizes.

        Returns:
            Ordered dict, one entry per output head.
        """
        mgr = self.label_map_manager
        return {
            "entity_type":     mgr.num_classes("entity_type"),
            "parent_class":    mgr.num_classes("parent_class"),
            "root_class":      mgr.num_classes("root_class"),
            "coarse_class":    mgr.num_classes("coarse_class"),
            "material":        mgr.num_classes("material"),
            "phase":           mgr.num_classes("phase"),
            "mobility":        mgr.num_classes("mobility"),
            "size_class":      mgr.num_classes("size_class"),
            "shape":           mgr.num_classes("shape"),
            "mass_class":      mgr.num_classes("mass_class"),
            "contact_type":    mgr.num_classes("contact_type"),
            "stability":       mgr.num_classes("stability"),
            "friction_class":  mgr.num_classes("friction_class"),
            "restitution_class": mgr.num_classes("restitution_class"),
            "capabilities":    mgr.num_classes("capability"),
            "affordances":     mgr.num_classes("affordance"),
            "scene_roles":     mgr.num_classes("scene_role"),
        }

    def get_head_dimensions(self) -> Dict[str, int]:
        """Alias for :meth:`get_num_classes` — explicit naming for encoder config.

        EntityEncoder reads::

            dims = dataset.get_head_dimensions()
            self.entity_type_head = nn.Linear(hidden, dims["entity_type"])
        """
        return self.get_num_classes()

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Return vocabulary sizes for every loaded label map, keyed by map name.

        Includes both singular multi-label keys (``"capability"``,
        ``"affordance"``, ``"scene_role"``) and single-label field keys
        (``"entity_type"``, …) — i.e. all keys as they appear in
        ``label_maps/``.

        Useful for direct low-level access when :meth:`get_num_classes` is too
        coarse.
        """
        return {
            name: len(mapping)
            for name, mapping in self.label_maps.items()
        }

    def decode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Decode integer / multi-hot tensors in ``sample`` back to label strings.

        Provides a human-readable view of any sample returned by
        ``__getitem__``.  The returned dict uses the same keys as the input but
        replaces tensors with decoded strings / lists.

        Args:
            sample: Dict as returned by ``__getitem__``.

        Returns:
            Dict with string / list values (no tensors).
        """
        mgr = self.label_map_manager
        decoded: Dict[str, Any] = {}

        for fname in SINGLE_LABEL_FIELDS:
            if fname in sample:
                idx: int = int(sample[fname].item())
                decoded[fname] = mgr.decode(fname, idx)

        # Multi-label: invert multi-hot vectors.
        for field_name, vocab_name in (
            ("capabilities", "capability"),
            ("affordances",  "affordance"),
            ("scene_roles",  "scene_role"),
        ):
            if field_name in sample:
                vec: torch.Tensor = sample[field_name]
                forward = self.label_maps.get(vocab_name, {})
                reverse = {v: k for k, v in forward.items()}
                active = [
                    reverse[i] for i in range(len(vec)) if vec[i].item() == 1.0
                ]
                decoded[field_name] = active

        if "raw_record" in sample:
            decoded["raw_record"] = sample["raw_record"]

        return decoded

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Check tensor shapes and dtypes for a sample from ``__getitem__``.

        Emits ``warnings.warn`` for each violation — never raises — so that
        DataLoader workers survive bad data.

        Args:
            sample: Dict returned by the encoding pipeline inside __getitem__.

        Returns:
            True if no issues were found.
        """
        ok = True
        expected_len = self.max_length

        for key in ("input_ids", "attention_mask"):
            t = sample.get(key)
            if t is None:
                warnings.warn(f"validate_sample: missing key {key!r}", stacklevel=2)
                ok = False
                continue
            if t.shape != torch.Size([expected_len]):
                warnings.warn(
                    f"validate_sample: {key} shape {tuple(t.shape)} != "
                    f"({expected_len},)",
                    stacklevel=2,
                )
                ok = False
            if t.dtype != torch.long:
                warnings.warn(
                    f"validate_sample: {key} dtype {t.dtype} != torch.long",
                    stacklevel=2,
                )
                ok = False

        for fname in SINGLE_LABEL_FIELDS:
            t = sample.get(fname)
            if t is None:
                warnings.warn(
                    f"validate_sample: missing single-label key {fname!r}",
                    stacklevel=2,
                )
                ok = False
                continue
            if t.shape != torch.Size([]):
                warnings.warn(
                    f"validate_sample: {fname} shape {tuple(t.shape)} != ()",
                    stacklevel=2,
                )
                ok = False
            if t.dtype != torch.long:
                warnings.warn(
                    f"validate_sample: {fname} dtype {t.dtype} != torch.long",
                    stacklevel=2,
                )
                ok = False

        expected_sizes = {
            "capabilities": self.label_map_manager.num_classes("capability"),
            "affordances":  self.label_map_manager.num_classes("affordance"),
            "scene_roles":  self.label_map_manager.num_classes("scene_role"),
        }
        for fname, expected in expected_sizes.items():
            t = sample.get(fname)
            if t is None:
                warnings.warn(
                    f"validate_sample: missing multi-label key {fname!r}",
                    stacklevel=2,
                )
                ok = False
                continue
            if t.shape != torch.Size([expected]):
                warnings.warn(
                    f"validate_sample: {fname} shape {tuple(t.shape)} != "
                    f"({expected},)",
                    stacklevel=2,
                )
                ok = False
            if t.dtype != torch.float32:
                warnings.warn(
                    f"validate_sample: {fname} dtype {t.dtype} != torch.float32",
                    stacklevel=2,
                )
                ok = False

        return ok

    def print_sample(self, idx: int = 0) -> None:
        """Pretty-print the tensors for record at ``idx``.

        Shows:
          * Each key with its shape and dtype.
          * Decoded label names for single-label heads.
          * Active label list for multi-hot heads.
          * Raw record dict if ``return_raw_record=True``.
        """
        sample = self[idx]
        decoded = self.decode_sample(sample)
        dims = self.get_num_classes()

        print(f"\n{'─'*60}")
        print(f"  EntityDataset.print_sample(idx={idx})")
        print(f"{'─'*60}")

        for key in ("input_ids", "attention_mask"):
            t = sample[key]
            print(f"  {key:<22} shape={tuple(t.shape)}  dtype={t.dtype}")

        print()
        for fname in SINGLE_LABEL_FIELDS:
            t = sample[fname]
            label = decoded.get(fname, "?")
            n_cls = dims.get(fname, "?")
            print(
                f"  {fname:<22} id={int(t.item()):>4}  "
                f"label={label!r:<30}  n_classes={n_cls}"
            )

        print()
        for field_name, vocab_name in (
            ("capabilities", "capability"),
            ("affordances",  "affordance"),
            ("scene_roles",  "scene_role"),
        ):
            t = sample[field_name]
            active = decoded.get(field_name, [])
            n_active = int(t.sum().item())
            n_total = len(t)
            print(
                f"  {field_name:<22} shape=({n_total},)  "
                f"active={n_active}  labels={active}"
            )

        if "raw_record" in sample:
            print(f"\n  raw_record:")
            print(
                "  " + json.dumps(sample["raw_record"], indent=4,
                                  ensure_ascii=False)
                      .replace("\n", "\n  ")
            )

        print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  main()
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Smoke-test: instantiate all three splits, print sizes, dims, and a sample."""

    print("=" * 60)
    print("  PhysWorldLM — EntityDataset smoke-test")
    print("=" * 60)

    train_dataset = EntityDataset(
        DatasetConfig(split="train", debug=False, verify_labels=True)
    )
    val_dataset   = EntityDataset(
        DatasetConfig(split="val",   debug=False, verify_labels=True)
    )
    test_dataset  = EntityDataset(
        DatasetConfig(split="test",  debug=False, verify_labels=True)
    )

    print(f"\n  Train size : {len(train_dataset):>8,}")
    print(f"  Val   size : {len(val_dataset):>8,}")
    print(f"  Test  size : {len(test_dataset):>8,}")

    print("\n  Head dimensions (from train split):")
    dims = train_dataset.get_num_classes()
    for head, n in dims.items():
        print(f"    {head:<24} {n}")

    print("\n  Metadata:")
    md = train_dataset.metadata
    print(f"    num_entity_types   : {md.num_entity_types}")
    print(f"    num_capabilities   : {md.num_capabilities}")
    print(f"    num_affordances    : {md.num_affordances}")
    print(f"    num_scene_roles    : {md.num_scene_roles}")
    print(f"    num_samples        : {md.num_samples}")

    print("\n  Sample[0] from train:")
    train_dataset.print_sample(0)


if __name__ == "__main__":
    main()
