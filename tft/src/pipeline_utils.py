from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_config = config.get("base_config")
    if base_config:
        base_path = (config_path.parent / base_config).resolve()
        base = load_yaml_config(base_path)
        return deep_merge(base, config)
    return config


def write_yaml_config(config: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_columns(df: pd.DataFrame, columns: list[str], df_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")

