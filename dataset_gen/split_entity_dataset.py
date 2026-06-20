"""
dataset_gen/split_entity_dataset.py
───────────────────────────────────────────────────────────────────────────────
PhysWorldLM — research-grade dataset preparation layer.

Transforms a flat entity-classification JSONL file into a fully-prepared,
versioned, reproducible training corpus:

    datasets/entity_classification.jsonl
        │
        ▼
    datasets/entity/
        ├── train.jsonl / train.jsonl.gz
        ├── val.jsonl   / val.jsonl.gz
        ├── test.jsonl  / test.jsonl.gz
        ├── metadata.json
        ├── statistics.txt
        ├── capability_counts.json
        ├── affordance_counts.json
        ├── scene_role_counts.json
        ├── dataset_registry.json
        └── label_maps/
                <field>_to_id.json
                <field>_from_id.json   (for every categorical field)

This module is intentionally dependency-free (stdlib only) so that it can sit
permanently between ontology generation (`generate_entity_dataset.py`) and
every downstream consumer (`entity_dataset.py`, `entity_encoder.py`,
`train_entity_classifier.py`, ...) without ever needing to be touched again.

No PyTorch. No NumPy. No pandas. No sklearn. No tokenizers. No neural nets.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Schema constants
# ─────────────────────────────────────────────────────────────────────────────

#: Fields that MUST be present on every record (validated by validate_records).
REQUIRED_FIELDS: tuple[str, ...] = (
    "token",
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
    "affected_by_gravity",
    "floats",
    "friction_class",
    "restitution_class",
    "capabilities",
    "affordances",
    "scene_roles",
)

#: Single-label categorical fields that get a forward/reverse ID map each.
SINGLE_LABEL_FIELDS: tuple[str, ...] = (
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

#: Multi-label categorical fields, handled by dedicated vocab builders.
MULTI_LABEL_FIELDS: tuple[str, ...] = (
    "capabilities",
    "affordances",
    "scene_roles",
)

#: Fields expected to be lists (used by validate_records type checking).
LIST_FIELDS: frozenset[str] = frozenset(
    {"capabilities", "affordances", "scene_roles", "properties",
     "interaction_properties", "aliases", "possible_classes"}
)

#: Fields expected to be bool.
BOOL_FIELDS: frozenset[str] = frozenset({"affected_by_gravity", "floats"})

#: Fields expected to be str.
STR_FIELDS: frozenset[str] = frozenset(
    {"token", "entity_type", "parent_class", "root_class", "coarse_class",
     "material", "phase", "mobility", "size_class", "shape", "mass_class",
     "contact_type", "stability", "friction_class", "restitution_class"}
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    """Configuration for the dataset split pipeline.

    All defaults are chosen to be reproducible across machines and across
    time: a fixed seed, fixed ratios, and deterministic, alphabetically
    sorted label maps whose IDs never change across runs (given the same
    input vocabulary).
    """

    input_path:  str
    output_dir:  str

    train_ratio: float = 0.80
    val_ratio:   float = 0.10
    test_ratio:  float = 0.10

    seed: int = 42

    stratify_by_entity_type: bool = True

    save_reverse_maps: bool = True
    save_metadata:     bool = True
    save_statistics:   bool = True
    compress_files:    bool = True

    version: str = "1.0"

    def __post_init__(self) -> None:
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0 "
                f"(got {ratio_sum})"
            )
        for name, val in (
            ("train_ratio", self.train_ratio),
            ("val_ratio",   self.val_ratio),
            ("test_ratio",  self.test_ratio),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1] (got {val})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  I/O primitives
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    - Blank / whitespace-only lines are skipped.
    - Record order is preserved (this matters for reproducibility: sorting,
      if any, happens explicitly downstream, never implicitly here).

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if any non-blank line is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {p}")

    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {p}:{line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected a JSON object at {p}:{line_no}, "
                    f"got {type(obj).__name__}"
                )
            records.append(obj)
    return records


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write records to `path`, one JSON object per line, preserving order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Summary of issues found by `validate_records`."""

    total_records:      int = 0
    missing_fields:     dict[str, int] = field(default_factory=dict)
    wrong_types:        dict[str, int] = field(default_factory=dict)
    empty_capabilities: int = 0
    duplicate_tokens:   list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(
            self.missing_fields
            or self.wrong_types
            or self.empty_capabilities
            or self.duplicate_tokens
        )

    def print_warnings(self) -> None:
        if not self.has_issues():
            print("[validate_records] OK — no issues found "
                  f"({self.total_records:,} records).")
            return

        print(f"[validate_records] WARNINGS for {self.total_records:,} records:")
        for fname, count in sorted(self.missing_fields.items()):
            print(f"  - missing field {fname!r}: {count} record(s)")
        for fname, count in sorted(self.wrong_types.items()):
            print(f"  - wrong type for {fname!r}: {count} record(s)")
        if self.empty_capabilities:
            print(f"  - empty 'capabilities' list: "
                  f"{self.empty_capabilities} record(s)")
        if self.duplicate_tokens:
            n = len(self.duplicate_tokens)
            preview = ", ".join(self.duplicate_tokens[:10])
            more = f" (+{n - 10} more)" if n > 10 else ""
            print(f"  - duplicate (token, entity_type) pairs: {n} "
                  f"-> {preview}{more}")


def validate_records(records: list[dict[str, Any]]) -> ValidationReport:
    """Validate records against REQUIRED_FIELDS and basic type expectations.

    Detects (but does not raise on, since the dataset may legitimately
    contain a handful of edge cases):
      - missing required fields
      - wrong types (lists where a string is expected and vice versa,
        bool fields, etc.)
      - empty `capabilities` lists
      - duplicate (token, entity_type) pairs

    Returns a ValidationReport; the caller decides whether to abort.
    """
    report = ValidationReport(total_records=len(records))
    seen_tokens: dict[tuple[str, str], int] = {}
    dup_set: set[str] = set()

    for rec in records:
        for fname in REQUIRED_FIELDS:
            if fname not in rec:
                report.missing_fields[fname] = (
                    report.missing_fields.get(fname, 0) + 1
                )
                continue

            value = rec[fname]
            if fname in LIST_FIELDS and not isinstance(value, list):
                report.wrong_types[fname] = report.wrong_types.get(fname, 0) + 1
            elif fname in BOOL_FIELDS and not isinstance(value, bool):
                report.wrong_types[fname] = report.wrong_types.get(fname, 0) + 1
            elif fname in STR_FIELDS and not isinstance(value, str):
                report.wrong_types[fname] = report.wrong_types.get(fname, 0) + 1

        capabilities = rec.get("capabilities")
        if isinstance(capabilities, list) and len(capabilities) == 0:
            report.empty_capabilities += 1

        token = rec.get("token")
        entity_type = rec.get("entity_type")
        if isinstance(token, str) and isinstance(entity_type, str):
            key = (token.lower(), entity_type)
            if key in seen_tokens:
                dup_set.add(token)
            else:
                seen_tokens[key] = 1

    report.duplicate_tokens = sorted(dup_set)
    report.print_warnings()
    return report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  –  Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    records: list[dict[str, Any]],
    config: SplitConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split `records` into (train, val, test).

    Stratifies by `entity_type` when `config.stratify_by_entity_type` is
    True (the default and recommended setting), guaranteeing every class
    is represented proportionally in every split. Falls back to a single
    global shuffle-and-slice if stratification is disabled.

    Guarantees:
      - deterministic given the same `config.seed` and input order
      - train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅
        (enforced by construction: each record is assigned to exactly one
        split and never duplicated)
      - class distribution in each split mirrors the source distribution
        (when stratified)
    """
    rng = random.Random(config.seed)

    train: list[dict[str, Any]] = []
    val:   list[dict[str, Any]] = []
    test:  list[dict[str, Any]] = []

    if config.stratify_by_entity_type:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            buckets[rec.get("entity_type", "__unknown__")].append(rec)

        for etype in sorted(buckets.keys()):
            bucket = list(buckets[etype])
            rng.shuffle(bucket)
            n = len(bucket)

            n_train = int(round(n * config.train_ratio))
            n_val = int(round(n * config.val_ratio))
            # test gets the remainder, guaranteeing all n records are used
            # exactly once even with rounding.
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            n_test = n - n_train - n_val

            train.extend(bucket[:n_train])
            val.extend(bucket[n_train:n_train + n_val])
            test.extend(bucket[n_train + n_val:n_train + n_val + n_test])
    else:
        pool = list(records)
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(round(n * config.train_ratio))
        n_val = int(round(n * config.val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train = pool[:n_train]
        val = pool[n_train:n_train + n_val]
        test = pool[n_train + n_val:]

    # Final shuffle within each split so that the stratification bucket
    # order doesn't leak into the on-disk record order.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    _assert_no_leakage(train, val, test)
    return train, val, test


def _record_identity(rec: dict[str, Any]) -> tuple[str, str]:
    """Identity key used for leakage checks: (token, entity_type)."""
    return (str(rec.get("token", "")).lower(), str(rec.get("entity_type", "")))


def _assert_no_leakage(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    """Raise AssertionError if any record identity appears in 2+ splits."""
    train_ids = {_record_identity(r) for r in train}
    val_ids = {_record_identity(r) for r in val}
    test_ids = {_record_identity(r) for r in test}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    if tv or tt or vt:
        raise AssertionError(
            "Token leakage detected across splits! "
            f"train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  –  Label maps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelMaps:
    """Container for every forward/reverse label map produced by this module."""

    single_label: dict[str, dict[str, int]] = field(default_factory=dict)
    #: entity_type -> {label: id}, etc. Reverse maps are derived on save.

    capability_to_id: dict[str, int] = field(default_factory=dict)
    affordance_to_id: dict[str, int] = field(default_factory=dict)
    scene_role_to_id: dict[str, int] = field(default_factory=dict)

    capability_counts: dict[str, int] = field(default_factory=dict)
    affordance_counts: dict[str, int] = field(default_factory=dict)
    scene_role_counts: dict[str, int] = field(default_factory=dict)


def build_label_maps(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Build forward `{label: id}` maps for every single-label field.

    Labels are sorted alphabetically before ID assignment, so that IDs are
    stable and reproducible across runs given the same vocabulary — this is
    what makes the maps safe to bake into a deployed model's preprocessing
    layer.

    Unknown/missing values are recorded under the literal string
    `"__unknown__"` if encountered, sorted along with everything else.
    """
    maps: dict[str, dict[str, int]] = {}
    for fname in SINGLE_LABEL_FIELDS:
        values: set[str] = set()
        for rec in records:
            v = rec.get(fname)
            if v is None:
                v = "__unknown__"
            values.add(str(v))
        sorted_values = sorted(values)
        maps[fname] = {label: idx for idx, label in enumerate(sorted_values)}
    return maps


def _build_multi_label_vocab(
    records: list[dict[str, Any]], field_name: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Shared implementation for capability/affordance/scene_role vocabs.

    Returns (label_to_id, frequency_counts), both alphabetically ordered.
    """
    counter: Counter[str] = Counter()
    for rec in records:
        values = rec.get(field_name) or []
        if not isinstance(values, list):
            continue
        for v in values:
            counter[str(v)] += 1

    sorted_labels = sorted(counter.keys())
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    counts = {label: counter[label] for label in sorted_labels}
    return label_to_id, counts


def build_capability_vocab(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Flatten all `capabilities` lists into a sorted vocabulary + counts."""
    return _build_multi_label_vocab(records, "capabilities")


def build_affordance_vocab(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Flatten all `affordances` lists into a sorted vocabulary + counts."""
    return _build_multi_label_vocab(records, "affordances")


def build_scene_role_vocab(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Flatten all `scene_roles` lists into a sorted vocabulary + counts."""
    return _build_multi_label_vocab(records, "scene_roles")


def save_label_maps(
    output_dir: Path,
    single_label_maps: dict[str, dict[str, int]],
    capability_to_id: dict[str, int],
    affordance_to_id: dict[str, int],
    scene_role_to_id: dict[str, int],
    save_reverse_maps: bool = True,
) -> None:
    """Persist every forward map (and, optionally, its reverse) under
    `output_dir / label_maps /`.

    File naming follows `<field>_to_id.json` / `<field>_from_id.json`.
    """
    label_maps_dir = output_dir / "label_maps"
    label_maps_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, mapping: dict[str, int]) -> None:
        _save_json(label_maps_dir / f"{name}_to_id.json", mapping)
        if save_reverse_maps:
            reverse = {str(idx): label for label, idx in mapping.items()}
            _save_json(label_maps_dir / f"{name}_from_id.json", reverse)

    for fname, mapping in single_label_maps.items():
        _write(fname, mapping)

    _write("capability", capability_to_id)
    _write("affordance", affordance_to_id)
    _write("scene_role", scene_role_to_id)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  –  Compression
# ─────────────────────────────────────────────────────────────────────────────

def compress_dataset(output_dir: Path, split_names: Iterable[str] = ("train", "val", "test")) -> None:
    """Gzip each `<split>.jsonl` file in `output_dir` to `<split>.jsonl.gz`.

    The uncompressed `.jsonl` files are left in place; `.gz` variants are
    provided alongside for bandwidth-constrained transfer/storage.
    """
    for split in split_names:
        src = output_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        dst = output_dir / f"{split}.jsonl.gz"
        with src.open("rb") as fin, gzip.open(dst, "wb") as fout:
            fout.writelines(fin)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  –  Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStatistics:
    total_samples: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)

    entity_type_distribution:   dict[str, int] = field(default_factory=dict)
    parent_class_distribution:  dict[str, int] = field(default_factory=dict)
    root_class_distribution:    dict[str, int] = field(default_factory=dict)
    coarse_class_distribution:  dict[str, int] = field(default_factory=dict)
    material_distribution:      dict[str, int] = field(default_factory=dict)
    phase_distribution:         dict[str, int] = field(default_factory=dict)
    mobility_distribution:      dict[str, int] = field(default_factory=dict)
    shape_distribution:         dict[str, int] = field(default_factory=dict)
    mass_distribution:          dict[str, int] = field(default_factory=dict)
    contact_type_distribution:  dict[str, int] = field(default_factory=dict)
    stability_distribution:     dict[str, int] = field(default_factory=dict)
    friction_distribution:      dict[str, int] = field(default_factory=dict)
    restitution_distribution:   dict[str, int] = field(default_factory=dict)

    capability_frequency: dict[str, int] = field(default_factory=dict)
    affordance_frequency: dict[str, int] = field(default_factory=dict)
    scene_role_frequency: dict[str, int] = field(default_factory=dict)

    negative_examples_count:  int = 0
    hard_examples_count:      int = 0
    ambiguous_examples_count: int = 0

    variant_type_distribution: dict[str, int] = field(default_factory=dict)

    duplicate_token_count: int = 0
    rare_class_thresholds: dict[str, int] = field(default_factory=dict)


def _distribution(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rec in records:
        v = rec.get(field_name)
        counter[str(v) if v is not None else "__unknown__"] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _multi_label_frequency(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rec in records:
        values = rec.get(field_name) or []
        if isinstance(values, list):
            for v in values:
                counter[str(v)] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def compute_statistics(
    all_records: list[dict[str, Any]],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    validation_report: Optional[ValidationReport] = None,
) -> DatasetStatistics:
    """Compute the full statistics bundle described in the spec."""
    stats = DatasetStatistics()

    stats.total_samples = len(all_records)
    stats.split_sizes = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    stats.entity_type_distribution  = _distribution(all_records, "entity_type")
    stats.parent_class_distribution = _distribution(all_records, "parent_class")
    stats.root_class_distribution   = _distribution(all_records, "root_class")
    stats.coarse_class_distribution = _distribution(all_records, "coarse_class")
    stats.material_distribution     = _distribution(all_records, "material")
    stats.phase_distribution        = _distribution(all_records, "phase")
    stats.mobility_distribution     = _distribution(all_records, "mobility")
    stats.shape_distribution        = _distribution(all_records, "shape")
    stats.mass_distribution         = _distribution(all_records, "mass_class")
    stats.contact_type_distribution = _distribution(all_records, "contact_type")
    stats.stability_distribution    = _distribution(all_records, "stability")
    stats.friction_distribution     = _distribution(all_records, "friction_class")
    stats.restitution_distribution  = _distribution(all_records, "restitution_class")

    stats.capability_frequency = _multi_label_frequency(all_records, "capabilities")
    stats.affordance_frequency = _multi_label_frequency(all_records, "affordances")
    stats.scene_role_frequency = _multi_label_frequency(all_records, "scene_roles")

    stats.negative_examples_count = sum(
        1 for r in all_records if r.get("negative") is True
    )
    stats.hard_examples_count = sum(
        1 for r in all_records if r.get("variant_type") == "hard_example"
    )
    stats.ambiguous_examples_count = sum(
        1 for r in all_records if r.get("variant_type") == "ambiguous"
    )

    stats.variant_type_distribution = _distribution(all_records, "variant_type")

    stats.duplicate_token_count = (
        len(validation_report.duplicate_tokens) if validation_report else 0
    )

    stats.rare_class_thresholds = {
        "count<10": sum(1 for v in stats.entity_type_distribution.values() if v < 10),
        "count<20": sum(1 for v in stats.entity_type_distribution.values() if v < 20),
        "count<50": sum(1 for v in stats.entity_type_distribution.values() if v < 50),
    }

    return stats


def save_statistics(
    stats: DatasetStatistics,
    validation_report: ValidationReport,
    output_path: Path,
) -> None:
    """Write a human-readable statistics.txt file."""

    def _section(title: str, dist: dict[str, int], limit: int = 25) -> str:
        lines = [title, "-" * len(title)]
        items = list(dist.items())[:limit]
        for label, count in items:
            lines.append(f"  {label:<28} {count:>8,}")
        if len(dist) > limit:
            lines.append(f"  ... ({len(dist) - limit} more)")
        lines.append("")
        return "\n".join(lines)

    out_lines: list[str] = []
    out_lines.append("=" * 60)
    out_lines.append("DATASET STATISTICS")
    out_lines.append("=" * 60)
    out_lines.append(f"Generated:    {datetime.now(timezone.utc).isoformat()}")
    out_lines.append("")
    out_lines.append(f"Total Samples: {stats.total_samples:,}")
    out_lines.append("")
    out_lines.append(f"Train: {stats.split_sizes.get('train', 0):,}")
    out_lines.append(f"Val:   {stats.split_sizes.get('val', 0):,}")
    out_lines.append(f"Test:  {stats.split_sizes.get('test', 0):,}")
    out_lines.append("")

    out_lines.append(_section("Entity Types", stats.entity_type_distribution))
    out_lines.append(_section("Parent Classes", stats.parent_class_distribution))
    out_lines.append(_section("Root Classes", stats.root_class_distribution))
    out_lines.append(_section("Coarse Classes", stats.coarse_class_distribution))
    out_lines.append(_section("Materials", stats.material_distribution))
    out_lines.append(_section("Phases", stats.phase_distribution))
    out_lines.append(_section("Mobility", stats.mobility_distribution))
    out_lines.append(_section("Shapes", stats.shape_distribution))
    out_lines.append(_section("Mass Classes", stats.mass_distribution))
    out_lines.append(_section("Contact Types", stats.contact_type_distribution))
    out_lines.append(_section("Stability", stats.stability_distribution))
    out_lines.append(_section("Friction", stats.friction_distribution))
    out_lines.append(_section("Restitution", stats.restitution_distribution))
    out_lines.append(_section("Capabilities", stats.capability_frequency, limit=50))
    out_lines.append(_section("Affordances", stats.affordance_frequency, limit=50))
    out_lines.append(_section("Scene Roles", stats.scene_role_frequency, limit=50))
    out_lines.append(_section("Variant Types", stats.variant_type_distribution))

    out_lines.append("Rare Classes (by entity_type)")
    out_lines.append("-" * 30)
    out_lines.append(f"  count < 10: {stats.rare_class_thresholds.get('count<10', 0)}")
    out_lines.append(f"  count < 20: {stats.rare_class_thresholds.get('count<20', 0)}")
    out_lines.append(f"  count < 50: {stats.rare_class_thresholds.get('count<50', 0)}")
    out_lines.append("")

    out_lines.append(f"Duplicate Tokens: {stats.duplicate_token_count}")
    out_lines.append(f"Negative Examples: {stats.negative_examples_count:,}")
    out_lines.append(f"Hard Examples: {stats.hard_examples_count:,}")
    out_lines.append(f"Ambiguous Examples: {stats.ambiguous_examples_count:,}")
    out_lines.append("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  –  Metadata + registry
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(
    config: SplitConfig,
    stats: DatasetStatistics,
    single_label_maps: dict[str, dict[str, int]],
    capability_to_id: dict[str, int],
    affordance_to_id: dict[str, int],
    scene_role_to_id: dict[str, int],
    output_path: Path,
) -> None:
    """Write `metadata.json` summarizing the dataset for downstream tooling."""
    metadata = {
        "version": config.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "stratify_by_entity_type": config.stratify_by_entity_type,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "num_samples": stats.total_samples,
        "train_size": stats.split_sizes.get("train", 0),
        "val_size": stats.split_sizes.get("val", 0),
        "test_size": stats.split_sizes.get("test", 0),
        "num_entity_types": len(single_label_maps.get("entity_type", {})),
        "num_parent_classes": len(single_label_maps.get("parent_class", {})),
        "num_root_classes": len(single_label_maps.get("root_class", {})),
        "num_coarse_classes": len(single_label_maps.get("coarse_class", {})),
        "num_materials": len(single_label_maps.get("material", {})),
        "num_phases": len(single_label_maps.get("phase", {})),
        "num_mobility_classes": len(single_label_maps.get("mobility", {})),
        "num_size_classes": len(single_label_maps.get("size_class", {})),
        "num_shapes": len(single_label_maps.get("shape", {})),
        "num_mass_classes": len(single_label_maps.get("mass_class", {})),
        "num_contact_types": len(single_label_maps.get("contact_type", {})),
        "num_stability_classes": len(single_label_maps.get("stability", {})),
        "num_friction_classes": len(single_label_maps.get("friction_class", {})),
        "num_restitution_classes": len(single_label_maps.get("restitution_class", {})),
        "num_capabilities": len(capability_to_id),
        "num_affordances": len(affordance_to_id),
        "num_scene_roles": len(scene_role_to_id),
        "negative_examples_count": stats.negative_examples_count,
        "hard_examples_count": stats.hard_examples_count,
        "ambiguous_examples_count": stats.ambiguous_examples_count,
    }
    _save_json(output_path, metadata)


def save_registry(config: SplitConfig, output_path: Path) -> None:
    """Write `dataset_registry.json`, the single source of truth pointing
    downstream training code at the right files for this dataset version.
    """
    registry = {
        "entity_dataset_v1": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
            "train_compressed": "train.jsonl.gz",
            "val_compressed": "val.jsonl.gz",
            "test_compressed": "test.jsonl.gz",
            "metadata": "metadata.json",
            "statistics": "statistics.txt",
            "label_maps_dir": "label_maps",
            "capability_counts": "capability_counts.json",
            "affordance_counts": "affordance_counts.json",
            "scene_role_counts": "scene_role_counts.json",
            "version": config.version,
            "seed": config.seed,
        }
    }
    _save_json(output_path, registry)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10  –  Debug helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_examples(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    """Print one sample record from each split, for sanity-checking."""
    for name, split in (("train", train), ("val", val), ("test", test)):
        print(f"\n[print_examples] sample from {name}:")
        if split:
            print(json.dumps(split[0], indent=2, ensure_ascii=False))
        else:
            print("  (empty split)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11  –  Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: SplitConfig) -> DatasetStatistics:
    """Execute the full split pipeline end to end, per `config`.

    Returns the computed DatasetStatistics for programmatic use (e.g. in
    tests), in addition to writing every artifact described in the module
    docstring to `config.output_dir`.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split_entity_dataset] loading records from {config.input_path}")
    records = load_jsonl(config.input_path)
    print(f"[split_entity_dataset] loaded {len(records):,} records")

    validation_report = validate_records(records)

    print("[split_entity_dataset] performing stratified split "
          f"(seed={config.seed}, ratios="
          f"{config.train_ratio}/{config.val_ratio}/{config.test_ratio})")
    train, val, test = stratified_split(records, config)
    print(f"[split_entity_dataset] split sizes -> "
          f"train={len(train):,} val={len(val):,} test={len(test):,}")

    print("[split_entity_dataset] building label maps")
    single_label_maps = build_label_maps(records)
    capability_to_id, capability_counts = build_capability_vocab(records)
    affordance_to_id, affordance_counts = build_affordance_vocab(records)
    scene_role_to_id, scene_role_counts = build_scene_role_vocab(records)

    print(f"[split_entity_dataset] writing JSONL splits to {output_dir}")
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    print("[split_entity_dataset] writing label maps")
    save_label_maps(
        output_dir,
        single_label_maps,
        capability_to_id,
        affordance_to_id,
        scene_role_to_id,
        save_reverse_maps=config.save_reverse_maps,
    )

    print("[split_entity_dataset] writing vocab frequency files")
    _save_json(output_dir / "capability_counts.json", capability_counts)
    _save_json(output_dir / "affordance_counts.json", affordance_counts)
    _save_json(output_dir / "scene_role_counts.json", scene_role_counts)

    print("[split_entity_dataset] computing statistics")
    stats = compute_statistics(records, train, val, test, validation_report)

    if config.save_metadata:
        print("[split_entity_dataset] writing metadata.json")
        save_metadata(
            config, stats, single_label_maps,
            capability_to_id, affordance_to_id, scene_role_to_id,
            output_dir / "metadata.json",
        )

    if config.save_statistics:
        print("[split_entity_dataset] writing statistics.txt")
        save_statistics(stats, validation_report, output_dir / "statistics.txt")

    print("[split_entity_dataset] writing dataset_registry.json")
    save_registry(config, output_dir / "dataset_registry.json")

    if config.compress_files:
        print("[split_entity_dataset] compressing splits (.jsonl.gz)")
        compress_dataset(output_dir)

    print_examples(train, val, test)

    print(f"\n[split_entity_dataset] done. Output written to: {output_dir}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12  –  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> "argparse.Namespace":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split datasets/entity_classification.jsonl into "
                    "stratified train/val/test sets with full label maps, "
                    "statistics, and a dataset registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="datasets/entity_classification.jsonl",
        help="Path to the source entity_classification.jsonl file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="datasets/entity",
        help="Directory to write the split dataset and artifacts into.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratification by entity_type (plain random split).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Skip writing .jsonl.gz compressed copies of the splits.",
    )
    parser.add_argument(
        "--no-reverse-maps", action="store_true",
        help="Skip writing <field>_from_id.json reverse label maps.",
    )
    parser.add_argument("--version", default="1.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = SplitConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_entity_type=not args.no_stratify,
        save_reverse_maps=not args.no_reverse_maps,
        compress_files=not args.no_compress,
        version=args.version,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
