"""
models/pipeline_interfaces.py
──────────────────────────────
Protocol definitions for the PhysWorldLM pipeline.

Only the interfaces we have a real, working implementation for are fully
specified (Parser, CompilerPass, RepairStrategy, ValidationRule). The
remaining interfaces requested in the architecture audit (SimulationBackend,
Planner, Sensor, Exporter) are declared as signature-only stubs — DO NOT
treat these as implemented. They exist so that future modules can be typed
against a stable contract, but no behavior is defined here. Filling them in
requires the actual simulation_engine/, omniverse/, and sensors/ code, which
was not available when this file was written.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from world_spec import WorldSpec


# ─────────────────────────────────────────────
# Implemented-and-conformed-to interfaces
# ─────────────────────────────────────────────

@runtime_checkable
class Parser(Protocol):
    """Anything that turns raw text into a WorldSpec.

    Conformed to today by: models.prompt_parser.PromptParser
    """

    def parse(self, prompt: str, scene_id: str | None = None) -> WorldSpec:
        ...


@runtime_checkable
class CompilerPass(Protocol):
    """A single, named stage in the WorldSpec compilation pipeline.

    Each pass takes a WorldSpec and returns a (possibly identical) WorldSpec
    plus a structured result describing what it found. Passes must not have
    side effects on modules other than the WorldSpec they're given.

    Conformed to today by: models.validation_pass.ValidationCompilerPass
    """

    name: str

    def run(self, spec: WorldSpec) -> "PassResult":
        ...


class PassResult:
    """Minimal structured result every CompilerPass returns.

    ``ok`` is the pass/fail gate the pipeline checks before continuing.
    ``payload`` carries pass-specific data (e.g. a ValidationResult).
    """

    def __init__(self, ok: bool, payload: object, notes: str = "") -> None:
        self.ok = ok
        self.payload = payload
        self.notes = notes


@runtime_checkable
class ValidationRule(Protocol):
    """A single checkable condition over a WorldSpec.

    This is intentionally narrower than the existing PhysicsValidator's
    internal `_t1_*`/`_t2_*`/`_t3_*` methods — those remain the source of
    truth for now. This protocol exists so *new* rules (e.g. future
    domain-specific checks) can be added without subclassing the validator,
    by registering objects that conform to this shape.
    """

    code: str

    def check(self, spec: WorldSpec) -> list[str]:
        """Return a list of human-readable violation messages (empty = pass)."""
        ...


@runtime_checkable
class RepairStrategy(Protocol):
    """Something that can fix ONE specific validation error code.

    Conformed to today by: models.repair.{DefaultMassRepair, ZeroStaticVelocityRepair,
    AssignMissingIdRepair}
    """

    #: The Issue.code (from models.validator.Issue) this strategy can fix.
    handles_code: str

    def repair(self, spec: WorldSpec, entity_id: str) -> bool:
        """Mutate spec in-place to fix the issue on the given entity.

        Returns True if a fix was applied, False if this instance could not
        find anything to fix (e.g. entity_id not found).
        """
        ...


# ─────────────────────────────────────────────
# NOT YET IMPLEMENTED — signature-only stubs
# ─────────────────────────────────────────────
# These are declared so future code can type-check against a stable shape.
# No behavior lives here. Implementing these requires the actual
# simulation_engine/, omniverse/, sensors/, and planner modules, which were
# not provided in this session — do not assume any of this is wired up.

@runtime_checkable
class SimulationBackend(Protocol):
    """TODO(future work): unify PhysX/MuJoCo/Bullet/Genesis behind this."""

    def create_rigid_body(self, entity_id: str, spec: WorldSpec) -> object: ...
    def apply_constraint(self, constraint: object) -> None: ...
    def step_simulation(self, dt: float) -> None: ...
    def query_contact(self, entity_id: str) -> list[object]: ...


@runtime_checkable
class Planner(Protocol):
    """TODO(future work): unify PSO/A*/RRT*/MPC behind this."""

    def plan(self, spec: WorldSpec, agent_id: str) -> list[object]: ...


@runtime_checkable
class Sensor(Protocol):
    """TODO(future work): camera/depth/LiDAR/radar/IMU/GPS behind this."""

    def sample(self, spec: WorldSpec, entity_id: str) -> object: ...


@runtime_checkable
class Exporter(Protocol):
    """TODO(future work): OpenUSD/other backend serialization behind this."""

    def export(self, spec: WorldSpec, path: str) -> None: ...
