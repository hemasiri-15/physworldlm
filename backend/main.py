from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from compiler.scene_compiler import SceneCompiler
from models.world_parser import WorldParser


app = FastAPI(
    title="PhysWorldLM API",
    description="Physics-aware natural-language world generation pipeline",
)


if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


parser = WorldParser()
compiler = SceneCompiler()

connector: Optional[object] = None


class Prompt(BaseModel):
    prompt: str


def _create_connector():
    if not settings.omniverse_enabled:
        return None

    from omniverse.omniverse_connector import OmniverseConnector

    kwargs = {}

    if settings.kit_path is not None:
        kwargs["kit_executable"] = settings.kit_path

    if settings.kit_ext_folders:
        kwargs["ext_folders"] = settings.kit_ext_folders

    if settings.kit_extra_args:
        kwargs["extra_kit_args"] = settings.kit_extra_args

    return OmniverseConnector(**kwargs)


@app.on_event("startup")
def startup() -> None:
    global connector

    connector = None

    if not settings.omniverse_enabled:
        return

    connector = _create_connector()

    try:
        connector.initialize()
    except Exception:
        # Kit is an optional runtime. The core API must remain available
        # even when the configured Kit installation is unavailable.
        connector = None


@app.on_event("shutdown")
def shutdown() -> None:
    global connector

    if connector is not None:
        connector.shutdown()
        connector = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "omniverse_enabled": settings.omniverse_enabled,
        "omniverse_available": connector is not None,
    }


@app.post("/generate")
def generate(data: Prompt):
    prompt = data.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt must not be empty.",
        )

    spec = parser.parse(prompt)

    settings.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    usd_file = settings.output_dir / f"{spec.scene_id}.usda"

    report = compiler.compile(
        world_spec=spec,
        output_path=usd_file,
    )

    if not report.success:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Scene compilation failed.",
                "scene_id": spec.scene_id,
                "diagnostics": [
                    diagnostic.to_dict()
                    for diagnostic in report.diagnostics
                ],
            },
        )

    omniverse = None

    if connector is not None:
        try:
            connector.show_stage(usd_file)
            omniverse = connector.statistics()
        except Exception as exc:
            omniverse = {
                "state": "error",
                "message": str(exc),
            }

    return {
        "status": report.status.name,
        "scene_id": spec.scene_id,
        "worldspec": spec.to_dict(),
        "usd_path": str(usd_file),
        "omniverse": omniverse,
        "diagnostics": [
            diagnostic.to_dict()
            for diagnostic in report.diagnostics
        ],
    }

@app.post("/omniverse/connect")
def omniverse_connect():
    if connector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "state": "unavailable",
                "message": "Omniverse Kit is not available in the current runtime.",
            },
        )

    if not connector.is_running():
        connector.launch()

    return connector.statistics()

@app.get("/omniverse/status")
def omniverse_status():
    if connector is None:
        return {
            "state": "unavailable",
            "available": False,
        }

    return connector.statistics()


@app.post("/omniverse/show")
def omniverse_show(body: dict):
    if connector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "state": "unavailable",
                "message": "Omniverse Kit is not available in the current runtime.",
            },
        )

    usd_path = body.get("usd_path")

    if not usd_path:
        raise HTTPException(
            status_code=400,
            detail="usd_path is required.",
        )

    connector.show_stage(usd_path)

    return connector.statistics()
