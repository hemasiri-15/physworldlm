"""
models/world_compiler.py
──────────────────────────
Thin, generic runner for a sequence of CompilerPass instances.

Deliberately the ONLY orchestration abstraction in this codebase — the
architecture review that prompted this file also proposed a separate
`Pipeline` class wrapping PromptParser/SpatialRelationshipParser/Compiler.
That's the same idea (run ordered stages) at a different name. Rather than
maintain two classes doing the same job, WorldParser uses ONE WorldCompiler
instance whose pass list includes everything: ontology grounding,
validation, and repair. See world_parser.py.

Usage
─────
    compiler = WorldCompiler(passes=[
        OntologyPass(resolver),      # optional, only if grounding enabled
        ValidationCompilerPass(),    # validate -> repair -> re-validate
    ])
    spec, results = compiler.compile(spec)
"""

from __future__ import annotations

from typing import List, Tuple

from models.pipeline_interfaces import CompilerPass, PassResult
from world_spec import WorldSpec


class OntologyPass:
    """Wraps OntologyResolver as a CompilerPass so it can sit in the same
    pass list as validation/repair, rather than being a special-cased call
    WorldParser makes directly.
    """

    name = "ontology"

    def __init__(self, resolver) -> None:
        self._resolver = resolver

    def run(self, spec: WorldSpec) -> PassResult:
        if self._resolver is None:
            return PassResult(ok=True, payload=None, notes="skipped — no resolver configured")
        spec.entities = self._resolver.resolve_entities(spec.entities)
        grounded = sum(1 for e in spec.entities if e.ontology.get("_grounded"))
        return PassResult(
            ok=True,
            payload={"grounded_entities": grounded, "total_entities": len(spec.entities)},
            notes=f"grounded {grounded}/{len(spec.entities)} entities",
        )


class WorldCompiler:
    """Runs each CompilerPass in order, collecting PassResults.

    Does NOT stop on the first failing pass by default (``ok=False`` from
    e.g. ValidationCompilerPass doesn't halt the sequence) — the caller
    (WorldParser) decides what to do with an unresolved-errors result,
    since "stop" vs. "continue and surface the failure" is a policy
    decision, not something the compiler itself should hardcode.
    """

    def __init__(self, passes: List[CompilerPass]) -> None:
        self.passes = passes

    def compile(self, spec: WorldSpec) -> Tuple[WorldSpec, dict]:
        results: dict = {}
        for p in self.passes:
            result = p.run(spec)
            results[p.name] = result
        return spec, results
