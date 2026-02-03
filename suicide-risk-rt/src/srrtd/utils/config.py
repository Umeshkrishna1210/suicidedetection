from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _set_by_dotted_key(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = target
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_scalar(s: str) -> Any:
    lowered = s.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw = item.split("=", 1)
        _set_by_dotted_key(out, key.strip(), _parse_scalar(raw.strip()))
    return out


def ensure_dirs(cfg: dict[str, Any], project_root: str | os.PathLike) -> dict[str, Any]:
    root = Path(project_root)
    model_dir = root / str(cfg.get("project", {}).get("model_dir", "models"))
    output_dir = root / str(cfg.get("project", {}).get("output_dir", "outputs"))
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@dataclass(frozen=True)
class ResolvedPaths:
    root: Path
    models: Path
    outputs: Path
    data_raw: Path
    data_processed: Path


def resolve_paths(cfg: dict[str, Any], project_root: str | os.PathLike) -> ResolvedPaths:
    root = Path(project_root)
    models = root / str(cfg.get("project", {}).get("model_dir", "models"))
    outputs = root / str(cfg.get("project", {}).get("output_dir", "outputs"))
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    for p in (models, outputs, data_raw, data_processed):
        p.mkdir(parents=True, exist_ok=True)
    return ResolvedPaths(root=root, models=models, outputs=outputs, data_raw=data_raw, data_processed=data_processed)
