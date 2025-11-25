#!/usr/bin/env python

"""
src/io.py

I/O helpers for EEG CSV files from FreeEEG32.

- Uses YAML config + montages to stay parameter-agnostic.
- Returns a structured pandas.DataFrame plus metadata (e.g. fs, good/dead channels).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Montage


@dataclass
class EEGRecording:
    df: pd.DataFrame
    fs: float
    good_channels: List[str]
    dead_channels: List[str]
    montage: Montage
    path: Path


def load_eeg_csv(path: str | Path, montage: Montage) -> EEGRecording:
    """
    Load a FreeEEG32 CSV according to the montage definition.

    - Reads as tab-delimited with no header.
    - Renames channel columns to electrode names.
    - Adds t_rel (seconds since start of file).
    - Detects good vs dead channels based on sentinel and std.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Read full table, infer number of columns
    df_raw = pd.read_csv(path, sep="\t", header=None)

    # Extract core columns
    sc_col = montage.sample_counter_col
    ts_col = montage.timestamp_col
    mk_col = montage.marker_col

    df = pd.DataFrame()
    df["sample_counter"] = df_raw.iloc[:, sc_col]
    df["timestamp"] = df_raw.iloc[:, ts_col]
    df["marker"] = df_raw.iloc[:, mk_col] if mk_col < df_raw.shape[1] else 0.0

    # Compute relative time (seconds since start)
    t0 = df["timestamp"].iloc[0]
    df["t_rel"] = df["timestamp"] - t0

    # Estimate sampling frequency from timestamps (median dt)
    dt = df["timestamp"].diff().dropna()
    fs = float(1.0 / dt.median()) if len(dt) > 0 else montage.sampling_rate_default

    # Add EEG channels with proper names
    for col_idx, ch_name in montage.channel_map.items():
        if col_idx >= df_raw.shape[1]:
            # Column not present
            continue
        df[ch_name] = df_raw.iloc[:, col_idx].astype(float)

    # Detect good vs dead channels
    good_channels: List[str] = []
    dead_channels: List[str] = []
    sentinel = montage.adc_sentinel

    for ch_name in montage.channel_map.values():
        if ch_name not in df.columns:
            continue
        s = df[ch_name]
        if np.allclose(s, sentinel) or s.std() < 1e-6:
            dead_channels.append(ch_name)
        else:
            good_channels.append(ch_name)

    rec = EEGRecording(
        df=df,
        fs=fs,
        good_channels=good_channels,
        dead_channels=dead_channels,
        montage=montage,
        path=path,
    )
    return rec
