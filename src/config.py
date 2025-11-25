#!/usr/bin/env python

"""
src/config.py

Helpers for loading main config and montage definitions from YAML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass
class Montage:
    name: str
    description: str
    sampling_rate_default: float
    sample_counter_col: int
    timestamp_col: int
    marker_col: int
    adc_sentinel: float
    channel_map: Dict[int, str]


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_main_config(path: str | Path = "config/config.yaml") -> dict:
    return load_yaml(path)


def load_montages(path: str | Path = "config/montages.yaml") -> Dict[str, Montage]:
    raw = load_yaml(path)
    montages = {}
    for name, m in raw.get("montages", {}).items():
        columns = m.get("columns", {})
        montage = Montage(
            name=name,
            description=m.get("description", ""),
            sampling_rate_default=float(m.get("sampling_rate_default", 0.0)),
            sample_counter_col=int(columns.get("sample_counter", 0)),
            timestamp_col=int(columns.get("timestamp", 33)),
            marker_col=int(columns.get("marker", 34)),
            adc_sentinel=float(m.get("adc_sentinel", -312500.03125)),
            channel_map={int(k): str(v) for k, v in m.get("channels", {}).items()},
        )
        montages[name] = montage
    return montages


def get_montage(main_config: dict, montages: Dict[str, Montage]) -> Montage:
    montage_name = main_config["eeg"]["montage"]
    if montage_name not in montages:
        raise ValueError(f"Montage '{montage_name}' not found in montages.yaml")
    return montages[montage_name]
