"""
parse_scene.py
──────────────
CLI: convert one natural-language scene description to a WorldSpec JSON.

Usage:
    # Interactive (prompts you to type a description)
    python parse_scene.py

    # Pipe in a description
    echo "A 1200 kg car moves at 60 km/h on a wet road" | python parse_scene.py

    # Pass as argument
    python parse_scene.py --scene "A ball is thrown horizontally at 15 m/s from 10 m height"

    # Save to file
    python parse_scene.py --scene "..." --output output/scene.json

    # Quiet mode (only JSON, no progress logs)
    python parse_scene.py --scene "..." --quiet
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.world_parser import WorldParser


def main():
    ap = argparse.ArgumentParser(
        description="Prompt → WorldSpec JSON (PhysWorldLM)"
    )
    ap.add_argument("--scene",  "-s", type=str, default=None,
                    help="Scene description (reads stdin if omitted)")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="Path to save JSON output (prints to stdout if omitted)")
    ap.add_argument("--quiet",  "-q", action="store_true",
                    help="Suppress progress logs; output pure JSON only")
    ap.add_argument("--scene-id", type=str, default=None,
                    help="Override auto-generated scene_id")
    args = ap.parse_args()

    # ── get description ──────────────────────
    if args.scene:
        description = args.scene.strip()
    elif not sys.stdin.isatty():
        description = sys.stdin.read().strip()
    else:
        print("Enter scene description (press Enter twice when done):")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        description = "\n".join(lines).strip()

    if not description:
        print("Error: no scene description provided.", file=sys.stderr)
        sys.exit(1)

    # ── parse ────────────────────────────────
    parser = WorldParser(verbose=not args.quiet)
    spec   = parser.parse(description, scene_id=args.scene_id)

    # ── output ───────────────────────────────
    json_str = spec.to_json(indent=2)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str)
        if not args.quiet:
            print(f"\n[parse_scene] saved → {out}")
    else:
        if not args.quiet:
            print("\n" + "─" * 60)
        print(json_str)

    # ── quick summary ────────────────────────
    if not args.quiet:
        m = spec.metadata
        print(f"\n{'─'*60}")
        print(f"  scene_id      : {spec.scene_id}")
        print(f"  entities      : {m.get('entity_count', '?')} "
              f"({m.get('dynamic_count','?')} dynamic, "
              f"{m.get('static_count','?')} static)")
        print(f"  interactions  : {len(spec.interactions)}")
        print(f"  sim duration  : {spec.simulation_graph.duration} s")
        print(f"  integrator    : {spec.simulation_graph.integrator}")
        print(f"  parse time    : {m.get('parse_time_s','?')} s")
        if "warnings" in m:
            for w in m["warnings"]:
                print(f"  ⚠  {w}")
        print("─" * 60)


if __name__ == "__main__":
    main()
