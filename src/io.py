#!/usr/bin/env python

"""src/io.py

Helpers for reading raw EEG CSV files and returning a structured EEGRecording
object compatible with the rest of the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from .config import Montage


@dataclass
class EEGRecording:
    """Container for a single EEG recording."""
    df: pd.DataFrame
    fs: float
    good_channels: List[str]
    dead_channels: List[str]


def load_eeg_csv(path: Union[str, Path], montage: Montage) -> EEGRecording:
    """Load a raw EEG CSV and structure it according to a Montage."""
    path = Path(path)

    # 1) Read raw CSV with unknown delimiter; let pandas detect it.
    df_raw = pd.read_csv(path, header=None, sep=None, engine="python")

    # 2) Timestamps & markers
    ts_col = montage.timestamp_col
    marker_col = montage.marker_col

    if ts_col >= df_raw.shape[1]:
        raise IndexError(
            f"Timestamp column index {ts_col} is out of bounds for file {path} (n_cols={df_raw.shape[1]})."
        )
    if marker_col >= df_raw.shape[1]:
        raise IndexError(
            f"Marker column index {marker_col} is out of bounds for file {path} (n_cols={df_raw.shape[1]})."
        )

    t = df_raw.iloc[:, ts_col].to_numpy(dtype=float)

    # 3) Sampling rate estimate (fallback to montage default if timestamps are weird)
    if len(t) > 1:
        total_T = float(t[-1] - t[0])
        if total_T > 0:
            fs_est = (len(t) - 1) / total_T
        else:
            fs_est = montage.sampling_rate_default
    else:
        fs_est = montage.sampling_rate_default

    fs = float(fs_est)

    # 4) Relative time
    if len(t) > 0:
        t_rel = t - t[0]
    else:
        t_rel = t

    # 5) EEG channels by mapping
    ch_map = montage.channel_map
    adc_sentinel = montage.adc_sentinel

    data = {}
    good_channels: List[str] = []
    dead_channels: List[str] = []

    for col_idx, ch_name in ch_map.items():
        if col_idx >= df_raw.shape[1]:
            raise IndexError(
                f"Channel column index {col_idx} for '{ch_name}' is out of bounds "
                f"for file {path} (n_cols={df_raw.shape[1]})."
            )

        # Use float dtype to allow NaNs
        series = df_raw.iloc[:, col_idx].astype(float)

        # Detect "dead" channels: all samples equal to sentinel
        if np.all(series == adc_sentinel):
            dead_channels.append(ch_name)
            # Represent dead channels as NaN everywhere
            series = pd.Series(np.nan, index=series.index)
        else:
            good_channels.append(ch_name)
            # Replace occasional sentinel values with NaN
            series = series.replace(adc_sentinel, np.nan)

        data[ch_name] = series

    # 6) Build structured DataFrame
    df = pd.DataFrame(data)
    df.insert(0, "t_rel", t_rel)
    df.insert(0, "timestamp", t)
    df["marker"] = df_raw.iloc[:, marker_col]

    return EEGRecording(
        df=df,
        fs=fs,
        good_channels=good_channels,
        dead_channels=dead_channels,
    )
