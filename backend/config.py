from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    value = os.getenv(name, "")

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


class Settings:
    """Runtime configuration supplied by the environment."""

    def __init__(self) -> None:
        self.omniverse_enabled = _env_bool(
            "PHYSWORLDLM_OMNIVERSE_ENABLED"
        )

        kit_path = os.getenv("PHYSWORLDLM_KIT_PATH", "").strip()
        self.kit_path = Path(kit_path) if kit_path else None

        output_dir = os.getenv("PHYSWORLDLM_OUTPUT_DIR", "").strip()
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else ROOT / "outputs"
        )

        self.cors_origins = _env_list(
            "PHYSWORLDLM_CORS_ORIGINS"
        )

        self.host = os.getenv("PHYSWORLDLM_HOST", "127.0.0.1")
        self.port = int(os.getenv("PHYSWORLDLM_PORT", "8000"))

        self.kit_ext_folders = [
            Path(path)
            for path in _env_list("PHYSWORLDLM_KIT_EXT_FOLDERS")
        ]

        self.kit_extra_args = _env_list(
            "PHYSWORLDLM_KIT_EXTRA_ARGS"
        )


settings = Settings()
