#!/usr/bin/env python

"""src/config.py

Helpers for loading main config and montage definitions from YAML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union

import yaml


@dataclass
class Montage:
    """Definition of how to interpret a raw EEG CSV."""
    name: str
    description: str
    sampling_rate_default: float
    sample_counter_col: int
    timestamp_col: int
    marker_col: int
    adc_sentinel: float
    channel_map: Dict[int, str]


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return it as a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {p} did not parse to a mapping (dict).")

    return data


def load_main_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load the main pipeline configuration (config.yaml or config_streaming.yaml)."""
    return _load_yaml(path)


def load_montages(path: Union[str, Path]) -> Dict[str, Montage]:
    """Load all montage definitions from montages.yaml."""
    raw = _load_yaml(path)

    montages_raw = raw.get("montages", None)
    if not isinstance(montages_raw, dict):
        raise ValueError(f"File {path} does not contain a top-level 'montages' mapping.")

    montages: Dict[str, Montage] = {}

    for name, m in montages_raw.items():
        if not isinstance(m, dict):
            raise ValueError(f"Montage '{name}' in {path} is not a mapping.")

        columns = m.get("columns", {})
        if not isinstance(columns, dict):
            raise ValueError(f"Montage '{name}' has invalid 'columns' section.")

        montage = Montage(
            name=name,
            description=str(m.get("description", "")),
            sampling_rate_default=float(m.get("sampling_rate_default", 0.0)),
            sample_counter_col=int(columns.get("sample_counter", 0)),
            timestamp_col=int(columns.get("timestamp", 0)),
            marker_col=int(columns.get("marker", 0)),
            adc_sentinel=float(m.get("adc_sentinel", -312500.03125)),
            channel_map={int(k): str(v) for k, v in m.get("channels", {}).items()},
        )
        montages[name] = montage

    return montages


def get_montage(main_config: Dict[str, Any], montages: Dict[str, Montage]) -> Montage:
    """Read 'eeg.montage' from the main config and return the corresponding Montage."""
    eeg_cfg = main_config.get("eeg", {})
    montage_name = eeg_cfg.get("montage")

    if not montage_name:
        raise ValueError("Main config does not define 'eeg.montage'.")

    if montage_name not in montages:
        available = ", ".join(sorted(montages.keys()))
        raise ValueError(
            f"Montage '{montage_name}' not found in montages.yaml (available: {available})"
        )

    return montages[montage_name]
