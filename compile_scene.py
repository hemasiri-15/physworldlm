"""compile_scene.py

Entry point for the PhysWorldLM execution pipeline.

This script is **not** part of the compiler itself; it is the
executable program that loads a WorldSpec JSON file and runs it
through ``SceneCompiler`` to produce an OpenUSD scene:

    WorldSpec JSON -> WorldSpec.from_json() -> SceneCompiler.compile()
        -> outputs/scene.usda

Usage:
    python compile_scene.py
    python compile_scene.py --input examples/air_intercept.json --output outputs/scene.usda
    python compile_scene.py --debug

Integration notes (read this if you're touching this file)
------------------------------------------------------------
``scene_compiler.py`` is a self-contained orchestrator. Its
``SceneCompiler.compile(world_spec, output_path)`` already runs the
full pipeline internally -- world-root, environment, entities,
transforms, assets, materials, physics, sensors, relationships,
metadata, and export -- via its own *internal* ``StageBuilder`` /
``EntityBuilder`` / ``TransformBuilder`` classes and its
``USDAsciiExporter``.

Those internal classes are separate from, and now supersede, the
earlier standalone ``compiler/stage_builder.py`` /
``compiler/entity_builder.py`` / ``compiler/transform_builder.py``
modules generated earlier in this project: this script does **not**
import or call those modules. Reconciling or retiring them is a
follow-up task, not something this script papers over.

Because ``SceneCompiler.compile()`` is a single call that returns only
once the whole pipeline has finished, this script cannot print
*live* per-stage checkmarks as each stage starts. Instead, after
``compile()`` returns, it reports each stage's real, measured duration
from ``CompilationReport.statistics.stage_durations_s`` -- so the
per-stage breakdown shown to the user reflects what actually ran, not
a hand-typed approximation of it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from compiler.scene_compiler import (
    CompilationError,
    CompilationReport,
    CompilationStatus,
    CompilerConfig,
    SceneCompiler,
)
from world_spec import WorldSpec

logger = logging.getLogger("physworldlm.compile_scene")

DEFAULT_INPUT = Path("examples/air_intercept.json")
DEFAULT_OUTPUT = Path("outputs/scene.usda")


# =============================================================================
# Pipeline steps
# =============================================================================


def _load_world_spec(input_path: Path) -> WorldSpec:
    """Load a WorldSpec from disk via ``WorldSpec.from_json``.

    Raises:
        CompilationError: If the file is missing, not valid JSON, or
            the JSON does not match the WorldSpec shape (e.g. an
            entity missing a required field).
    """
    if not input_path.exists():
        raise CompilationError(f"Input WorldSpec not found: '{input_path}'")

    try:
        return WorldSpec.from_json(str(input_path))
    except Exception as exc:
        # WorldSpec.from_dict() does direct dict indexing (e.g. ed["id"])
        # for required Entity fields, so malformed input can surface as
        # KeyError / TypeError / json.JSONDecodeError -- all are reported
        # uniformly here rather than crashing the script.
        raise CompilationError(f"Could not load WorldSpec from '{input_path}': {exc}") from exc


def _run_compiler(world_spec: WorldSpec, output_path: Path, config: CompilerConfig) -> CompilationReport:
    """Run the WorldSpec through SceneCompiler and return its report.

    Raises:
        CompilationError: Propagated from SceneCompiler for unrecoverable
            failures. Ordinary validation/build failures are instead
            captured in a FAILED CompilationReport and returned, not
            raised -- callers should check ``report.success``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = SceneCompiler(config=config)
    return compiler.compile(world_spec, output_path)


# =============================================================================
# Console reporting
# =============================================================================


def _print_banner() -> None:
    print("=" * 40)
    print("PhysWorldLM Scene Compiler")
    print("=" * 40 + "\n")


def _print_report(report: CompilationReport) -> None:
    print()
    if report.success:
        print("Compilation Successful\n")
        print("Scene saved to:\n")
        print(f"  {report.output_path}\n")
    else:
        print("Compilation Failed\n")
        for diag in report.errors():
            print(f"  ✗ {diag}")
        print()

    if report.statistics.stage_durations_s:
        print("Stage breakdown:")
        for stage_label, duration in report.statistics.stage_durations_s.items():
            print(f"  ✓ {stage_label} ({duration:.4f}s)")
        print()

    print(f"Entities compiled: {report.statistics.entity_count}")
    print(f"Materials: {report.statistics.material_count}")
    print(f"Relationships: {report.statistics.relationship_count}")
    if report.success:
        print(f"Exported file size: {report.statistics.exported_file_size_bytes} bytes")
    print(f"\nCompilation time:\n  {report.statistics.compilation_time_s:.2f} seconds")

    warnings = report.warnings()
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    print("=" * 40)


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysWorldLM Scene Compiler: WorldSpec -> OpenUSD scene."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the WorldSpec JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write scene.usda to.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _print_banner()

    try:
        print("Loading WorldSpec...")
        world_spec = _load_world_spec(args.input)
        print(f"✓ WorldSpec loaded ({len(world_spec.entities)} entities)\n")

        print("Compiling scene...")
        config = CompilerConfig(log_level="DEBUG" if args.debug else "INFO")
        report = _run_compiler(world_spec, args.output, config)

    except CompilationError as exc:
        print(f"\n✗ {exc}\n")
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive catch-all
        print(f"\n✗ Unexpected error: {exc}\n")
        logger.exception("Unexpected error")
        return 1

    _print_report(report)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
