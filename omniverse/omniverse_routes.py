"""
omniverse_routes.py
══════════════════════════════════════════════════════════════════════════
FastAPI integration for the new OmniverseConnector.

This is illustrative glue meant to be merged into PhysWorldLM's existing
backend (the router prefix, response models, and error mapping should be
adapted to match whatever conventions the rest of the API already uses).
It shows the two things that actually change with the new connector:

  1. Connector lifecycle: one process-wide OmniverseConnector, created on
     app startup and shut down on app shutdown -- NOT one per request.
  2. The compile → show_stage handoff: SceneCompiler is completely
     unchanged; only what happens after `compiler.compile(...)` differs.

Merge notes
-----------
- Replace `from physworldlm.omniverse import ...` with your project's
  actual import path if it differs.
- If the existing app already has a startup/shutdown lifespan handler,
  add the two `connector.initialize()` / `connector.shutdown()` calls to
  it rather than registering a second one.
- `OMNIVERSE_ENABLED` lets `/scenes/compile` keep working (compile-only,
  no Kit) in environments without a Kit install, e.g. CI or a laptop --
  this mirrors how the old code treated Omniverse as optional.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel

from omniverse import (
    ConnectorStatistics,
    KitAlreadyRunningError,
    KitLaunchError,
    KitNotFoundError,
    OmniverseConnector,
    OmniverseConnectorError,
    StageLoadError,
)
from scene_compiler import SceneCompiler, CompilationReport
from world_spec import WorldSpec

logger = logging.getLogger("physworldlm.api.omniverse_routes")

OMNIVERSE_ENABLED = os.environ.get("PHYSWORLDLM_OMNIVERSE_ENABLED", "1") != "0"

router = APIRouter(prefix="/omniverse", tags=["omniverse"])

# One connector for the whole process. Built lazily so importing this
# module (e.g. for tests) never touches the filesystem or spawns anything.
_connector: Optional[OmniverseConnector] = None


def get_connector() -> OmniverseConnector:
    """Return the process-wide connector, constructing it on first use."""
    global _connector
    if _connector is None:
        _connector = OmniverseConnector()
    return _connector


def register_lifecycle(app: FastAPI) -> None:
    """Wire connector startup/shutdown into the FastAPI app.

    Call this once from the app factory, e.g.:

        app = FastAPI()
        register_lifecycle(app)
        app.include_router(router)
    """

    @app.on_event("startup")
    def _startup() -> None:  # noqa: ANN202
        if not OMNIVERSE_ENABLED:
            logger.info("PHYSWORLDLM_OMNIVERSE_ENABLED=0; skipping Kit discovery at startup.")
            return
        try:
            get_connector().initialize()
        except KitNotFoundError as exc:
            # Non-fatal: scene compilation still works without Kit; only
            # the /omniverse/show endpoint will fail until this is fixed.
            logger.warning("Omniverse Kit not found at startup: %s", exc)

    @app.on_event("shutdown")
    def _shutdown() -> None:  # noqa: ANN202
        if _connector is not None:
            _connector.shutdown()


# ════════════════════════════════════════════════════════════════════════
# Request/response models
# ════════════════════════════════════════════════════════════════════════

class ShowStageRequest(BaseModel):
    usd_path: str


class CompileAndShowRequest(BaseModel):
    world_spec: dict
    show: bool = True


class CompileAndShowResponse(BaseModel):
    scene_id: str
    status: str
    output_path: Optional[str]
    shown: bool
    connector: Optional["ConnectorStatusResponse"] = None


class ConnectorStatusResponse(BaseModel):
    state: str
    pid: Optional[int]
    current_stage: Optional[str]
    kit_executable: Optional[str]
    kit_version: Optional[str]
    launch_count: int
    uptime_seconds: float
    log_file: Optional[str]

    @classmethod
    def from_stats(cls, stats: ConnectorStatistics) -> "ConnectorStatusResponse":
        return cls(
            state=stats.state.value,
            pid=stats.pid,
            current_stage=stats.current_stage,
            kit_executable=stats.kit_executable,
            kit_version=stats.kit_version,
            launch_count=stats.launch_count,
            uptime_seconds=stats.uptime_seconds,
            log_file=stats.log_file,
        )


# ════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════

@router.post("/show", response_model=ConnectorStatusResponse)
def show_stage(body: ShowStageRequest) -> ConnectorStatusResponse:
    """Show an already-compiled ``.usd*`` file in Omniverse Kit.

    Launches Kit if it isn't running yet, or reloads it (clean restart)
    if it already is -- see ``OmniverseConnector.show_stage`` docstring
    for why a restart is used instead of an in-process reload.
    """
    if not OMNIVERSE_ENABLED:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Omniverse integration is disabled on this server.")

    connector = get_connector()
    try:
        connector.show_stage(body.usd_path)
    except StageLoadError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except KitNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except (KitLaunchError, KitAlreadyRunningError, OmniverseConnectorError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return ConnectorStatusResponse.from_stats(connector.statistics())


@router.post("/compile_and_show", response_model=CompileAndShowResponse)
def compile_and_show(body: CompileAndShowRequest) -> CompileAndShowResponse:
    """Compile a WorldSpec and (optionally) hand the result straight to Kit.

    This is the single call the frontend needs for the common path: it
    never has to know a ``.usda`` file exists on disk. Compilation
    (`scene_compiler.py`) is completely unchanged by any of this --
    only what happens to the output path is new.
    """
    world_spec = WorldSpec.from_dict(body.world_spec)
    compiler = SceneCompiler()
    report: CompilationReport = compiler.compile(world_spec, output_path=f"{world_spec.scene_id}.usda")

    if not report.success:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[str(d) for d in report.errors()],
        )

    connector_response: Optional[ConnectorStatusResponse] = None
    shown = False
    if body.show and OMNIVERSE_ENABLED:
        connector = get_connector()
        try:
            connector.show_stage(report.output_path)
            shown = True
        except OmniverseConnectorError as exc:
            # Compilation succeeded either way -- don't fail the request
            # over Kit; the frontend can retry /omniverse/show once fixed.
            logger.warning("Compiled '%s' but could not show it in Kit: %s", report.output_path, exc)
        connector_response = ConnectorStatusResponse.from_stats(connector.statistics())

    return CompileAndShowResponse(
        scene_id=report.scene_id,
        status=report.status.name,
        output_path=str(report.output_path) if report.output_path else None,
        shown=shown,
        connector=connector_response,
    )


@router.get("/status", response_model=ConnectorStatusResponse)
def get_status() -> ConnectorStatusResponse:
    """Report current connector/Kit-process state, for the frontend's status indicator."""
    connector = get_connector()
    connector.is_running()  # reconciles state if Kit exited on its own
    return ConnectorStatusResponse.from_stats(connector.statistics())


@router.post("/shutdown", status_code=status.HTTP_204_NO_CONTENT)
def shutdown_kit() -> None:
    """Terminate the running Kit process, if any."""
    get_connector().shutdown()


__all__ = ["router", "register_lifecycle", "get_connector", "OMNIVERSE_ENABLED"]
