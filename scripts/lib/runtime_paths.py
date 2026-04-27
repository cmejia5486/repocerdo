from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    raw = (
        os.getenv("VISION360_DATA_DIR", "")
        or os.getenv("AUDIT_DATA_DIR", "")
        or os.getenv("RUNNER_TEMP", "")
    ).strip()

    if raw:
        base = Path(raw)
        if base.name.lower() != "vision360-data":
            base = base / "vision360-data"
    else:
        base = repo_root() / ".vision360-data"

    base.mkdir(parents=True, exist_ok=True)
    return base


def data_path(filename: str) -> Path:
    return data_dir() / filename


def env_path(name: str, default_filename: str) -> Path:
    raw = os.getenv(name, "").strip()
    if raw:
        return Path(raw)
    return data_path(default_filename)