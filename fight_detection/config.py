from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "msar_local.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["person_dim"] = int(cfg["n_landmarks"]) * int(cfg["features_per_landmark"])
    cfg["frame_dim"] = int(cfg["max_people"]) * int(cfg["person_dim"])
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def dataset_repo_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["dataset_repo_dir"])


def dataset_root(cfg: dict[str, Any]) -> Path:
    return dataset_repo_dir(cfg) / "violence-detection-dataset"


def cache_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["cache_root"])


def runs_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["runs_root"])


def weights_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["weights_root"])
