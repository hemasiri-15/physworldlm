"""
generate_dataset.py
───────────────────
Bootstrap the training dataset: generate N (Prompt → WorldSpec) pairs
using the WorldParser and a bank of diverse scene templates.

Usage:
    python -m dataset_gen.generate_dataset \
        --output data/train.jsonl \
        --count 100 \
        --workers 4

Each output line is newline-delimited JSON:
    {"prompt": "...", "world_spec": {...}}

For 10 000 pairs run with --count 10000 --workers 8.
The generator uses template variation + LLM augmentation so every
scene is unique even within the same template family.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# Add repo root to path so relative imports work when called as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.world_parser import WorldParser


# ─────────────────────────────────────────────
# Scene template bank
# ─────────────────────────────────────────────

# Each template is a (category, template_string) pair.
# {placeholders} are filled with random values below.

TEMPLATES = [
    # ── Ground vehicles ───────────────────────
    ("vehicle", "A {color} {car} is travelling at {speed_kmh} km/h along a {road_condition} road. "
                "The road has a gentle {slope}° downward slope."),

    ("vehicle", "Two cars approach an intersection. "
                "Car A moves east at {speed_a} km/h, Car B moves north at {speed_b} km/h. "
                "The road surface is {road_condition}."),

    ("vehicle", "A {mass_kg} kg truck decelerates from {speed_kmh} km/h to a stop over {distance} metres "
                "on a flat concrete road. The driver applies maximum braking at t=0."),

    ("vehicle", "A motorcycle at {speed_kmh} km/h takes a curve of radius {radius} m. "
                "The road is banked at {bank}° and is {road_condition}."),

    # ── Projectiles ───────────────────────────
    ("projectile", "A ball of mass {mass} kg is thrown horizontally at {speed} m/s "
                   "from a height of {height} m above flat ground. Wind blows at {wind} m/s eastward."),

    ("projectile", "A {mass} kg cannonball is fired at {angle}° elevation with initial speed {speed} m/s. "
                   "Temperature is {temp}°C. There is no wind."),

    ("projectile", "A rubber ball bounces on a concrete floor. "
                   "Initial drop height: {height} m. Ball mass: {mass} kg. "
                   "Coefficient of restitution is 0.8."),

    # ── Fluid / weather ───────────────────────
    ("fluid", "Heavy rain ({rain_rate} mm/h) falls on a {slope}° inclined asphalt road. "
              "Wind blows at {wind} m/s from the west. Temperature is {temp}°C."),

    ("fluid", "A {volume} m³ tank of water at {temp}°C sits on a flat surface. "
              "At t=0 a 0.05 m diameter hole opens at the base."),

    # ── Aerial / drone ────────────────────────
    ("aerial", "A {mass} kg drone hovers at {height} m altitude in {wind} m/s crosswind. "
               "At t=2s the drone accelerates north at {accel} m/s²."),

    ("aerial", "A {mass} kg payload is dropped from an aircraft flying at {altitude} m altitude "
               "and {speed} m/s horizontal speed. Air density is {air_density} kg/m³."),

    # ── Human / agent ─────────────────────────
    ("agent", "A {mass} kg person runs at {speed} m/s on {surface} surface and jumps from "
              "the edge of a {height} m platform."),

    ("agent", "Two football players collide. Player A: {mass_a} kg at {speed_a} m/s heading east. "
              "Player B: {mass_b} kg at {speed_b} m/s heading west. Grass surface."),

    # ── Structural / collapse ─────────────────
    ("structural", "A {mass} kg steel beam of length {length} m falls from {height} m "
                   "and impacts a concrete slab at {angle}°."),

    ("structural", "A pendulum with {mass} kg bob and {length} m arm is released from {angle}° "
                   "in a room with air at {temp}°C."),

    # ── Space / low gravity ───────────────────
    ("space", "On the Moon (g=1.62 m/s²) a {mass} kg rover moves at {speed} m/s over {terrain} terrain. "
              "There is no atmosphere."),

    ("space", "A {mass} kg satellite in low Earth orbit decelerates at {decel} m/s² due to drag "
              "at altitude {altitude} km. Air density at this altitude is {air_density} kg/m³."),
]

# ─────────────────────────────────────────────
# Value banks for placeholders
# ─────────────────────────────────────────────

VALUES = {
    "color":          ["red", "blue", "white", "black", "silver", "green", "yellow"],
    "car":            ["sedan", "SUV", "sports car", "compact car", "pickup truck"],
    "speed_kmh":      [30, 50, 60, 80, 100, 120],
    "speed_a":        [40, 60, 80],
    "speed_b":        [30, 50, 70],
    "speed":          [5, 10, 15, 20, 30, 50],
    "speed_ms":       [5, 10, 15, 20],
    "road_condition": ["dry", "wet", "icy", "gravel", "dry asphalt"],
    "surface":        ["grass", "rubber track", "concrete", "sand"],
    "terrain":        ["flat", "rocky", "sandy"],
    "slope":          [0, 2, 5, 8, 10, 15],
    "bank":           [5, 10, 15, 20],
    "angle":          [15, 30, 45, 60, 75],
    "radius":         [20, 50, 100, 200],
    "height":         [1, 2, 5, 10, 20, 50, 100],
    "altitude":       [100, 200, 500, 1000],
    "mass":           [0.1, 0.5, 1, 2, 5, 10, 50, 100],
    "mass_kg":        [500, 1000, 2000, 5000, 10000],
    "mass_a":         [60, 70, 80, 90],
    "mass_b":         [60, 70, 80, 90],
    "volume":         [1, 5, 10, 50],
    "length":         [2, 5, 10, 20],
    "temp":           [-20, -5, 0, 10, 20, 30, 40],
    "wind":           [0, 2, 5, 10, 15],
    "rain_rate":      [5, 10, 25, 50],
    "accel":          [1, 2, 3, 5],
    "decel":          [0.1, 0.5, 1],
    "distance":       [20, 50, 100, 200],
    "air_density":    [0.001, 0.01, 0.1, 1.0, 1.225],
}

def fill_template(template: str) -> str:
    """Replace {placeholder} with a random value from VALUES."""
    import re
    placeholders = re.findall(r"\{(\w+)\}", template)
    for ph in placeholders:
        if ph in VALUES:
            template = template.replace("{" + ph + "}", str(random.choice(VALUES[ph])), 1)
    return template


# ─────────────────────────────────────────────
# LLM-based scene augmentation
# ─────────────────────────────────────────────

_AUG_SYSTEM = (
    "You are a physics scenario writer for a simulation dataset. "
    "Return ONLY the new scene description — no explanation, no JSON."
)

_AUG_PROMPT = """Take this base scene and add one or two physically meaningful details 
(e.g. exact dimensions, material properties, environmental conditions, or a secondary object).
Keep the total description under 60 words.

Base scene: {base}

Augmented scene:"""


def augment_scene(client: anthropic.Anthropic, base: str) -> str:
    """Ask the LLM to add one realistic detail to the base scene."""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            system=_AUG_SYSTEM,
            messages=[{"role": "user",
                       "content": _AUG_PROMPT.format(base=base)}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return base   # fall back to un-augmented


# ─────────────────────────────────────────────
# Single example generation
# ─────────────────────────────────────────────

def generate_one(parser: WorldParser,
                 client: anthropic.Anthropic,
                 augment: bool = True) -> dict | None:
    """Generate one (prompt, world_spec) pair. Returns None on failure."""
    cat, tpl = random.choice(TEMPLATES)
    base     = fill_template(tpl)
    prompt   = augment_scene(client, base) if augment else base

    try:
        spec = parser.parse(prompt, scene_id=f"ds_{uuid.uuid4().hex[:8]}")
        return {
            "prompt":     prompt,
            "category":   cat,
            "world_spec": spec.to_dict(),
        }
    except Exception as exc:
        print(f"[generate_dataset] SKIP  parse error: {exc}")
        return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate PhysWorldLM training dataset")
    ap.add_argument("--output",  default="data/train.jsonl",
                    help="Output file (newline-delimited JSON)")
    ap.add_argument("--count",   type=int, default=100,
                    help="Number of examples to generate")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel threads (each uses one API connection)")
    ap.add_argument("--no-augment", action="store_true",
                    help="Skip LLM scene augmentation (faster, less diverse)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client  = anthropic.Anthropic()
    parser  = WorldParser(verbose=False)
    augment = not args.no_augment

    print(f"[generate_dataset] Generating {args.count} examples → {out_path}")
    print(f"[generate_dataset] Workers={args.workers}  augment={augment}")

    done = 0
    skip = 0
    t0   = time.time()

    with open(out_path, "w") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(generate_one, parser, client, augment)
                for _ in range(args.count)
            ]
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    skip += 1
                else:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    done += 1

                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (args.count - done - skip) / rate if rate > 0 else 0
                print(f"\r  done={done:>5}  skip={skip:>3}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s   ", end="", flush=True)

    print(f"\n[generate_dataset] Complete: {done} written, {skip} skipped")
    print(f"[generate_dataset] Output: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
