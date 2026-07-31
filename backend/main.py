import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from compiler.scene_compiler import SceneCompiler
from pathlib import Path

from omniverse.omniverse_connector import OmniverseConnector

from models.world_parser import WorldParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = WorldParser()

compiler = SceneCompiler()

connector = OmniverseConnector()

class Prompt(BaseModel):
    prompt: str

@app.on_event("startup")
def startup():
    connector.initialize()

@app.on_event("shutdown")
def shutdown():
    connector.shutdown()

@app.post("/generate")
def generate(data: Prompt):

    # Parse prompt
    spec = parser.parse(data.prompt)

    # Compile to USD
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    usd_file = output_dir / f"{spec.scene_id}.usda"

    report = compiler.compile(
        world_spec=spec,
        output_path=usd_file,
    )
    connector.load_stage(usd_file)

    return {
        "status": report.status.name,
        "scene_id": spec.scene_id,
        "worldspec": spec.to_dict(),
        "usd_path": str(usd_file),
        "diagnostics": [d.to_dict() for d in report.diagnostics],
    }
