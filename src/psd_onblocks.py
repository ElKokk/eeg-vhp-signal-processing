#!/usr/bin/env python

"""
src/psd_onblocks.py

Compute PSD averaged ONLY over ON-blocks (stim ON → stim OFF).
Uses CAR/Bipolar data if provided.

Features:
- Uses fs from config with same sanity checks as psd.py.
- ON/OFF marker codes from config.eeg.events.{stim_on, stim_off}.
- Frequency range automatically extended to include stim frequency.
- Vertical red line at stim frequency.
- Multi-column legend at bottom.
- Optional --channels to limit which electrodes are analysed.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from .config import load_main_config, load_montages, get_montage


def _estimate_fs(df: pd.DataFrame, cfg: dict) -> float:
    eeg_cfg = cfg.get("eeg", {})
    target_fs = float(eeg_cfg.get("sampling_rate", 512.0))

    if "timestamp" in df.columns:
        dt = df["timestamp"].diff().dropna()
        if len(dt) > 0:
            dt_med = float(dt.median())
            fs_est = 1.0 / dt_med if dt_med > 0 else target_fs
        else:
            fs_est = target_fs
    else:
        fs_est = target_fs

    if fs_est < 0.5 * target_fs or fs_est > 2.0 * target_fs:
        print(
            f"[PSD_ON] estimated fs from timestamps ≈ {fs_est:.2f} Hz, "
            f"but config says {target_fs:.2f} Hz → using config fs"
        )
        return target_fs

    print(f"[PSD_ON] fs ≈ {fs_est:.2f} Hz")
    return fs_est


def _parse_stim_freq_from_name(stem: str) -> Optional[float]:
    m = re.search(r"_f(\d+)(?:_v|_|$)", stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _get_freq_range(fs: float, psd_cfg: dict, stim_freq: Optional[float]) -> tuple[float, float]:
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax_base = float(psd_cfg.get("fmax", 60.0))
    nyq = fs / 2.0

    if stim_freq is not None:
        desired_max = stim_freq * 1.5
        fmax = min(nyq, max(fmax_base, desired_max))
    else:
        fmax = min(fmax_base, nyq)

    return fmin, fmax


def _get_on_off_codes(cfg: dict) -> Tuple[int, int]:
    eeg_cfg = cfg.get("eeg", {})
    ev_cfg = eeg_cfg.get("events", {})
    stim_on = int(ev_cfg.get("stim_on", 1))
    stim_off = int(ev_cfg.get("stim_off", 11))
    return stim_on, stim_off


def _get_channels_with_bipolar(df: pd.DataFrame, cfg: dict, montage) -> List[str]:
    base_eeg = [ch for ch in montage.channel_map.values() if ch in df.columns]
    reref_cfg = cfg.get("analysis", {}).get("reref", {})
    bipolar_pairs = reref_cfg.get("bipolar_pairs", []) or []

    def parse_pair(pair):
        if isinstance(pair, str):
            if "-" not in pair:
                return None
            a, b = pair.split("-", 1)
            return a.strip(), b.strip()
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return str(pair[0]).strip(), str(pair[1]).strip()
        return None

    extra = []
    for pair in bipolar_pairs:
        parsed = parse_pair(pair)
        if parsed:
            col = f"{parsed[0]}-{parsed[1]}"
            if col in df.columns:
                extra.append(col)

    return list(dict.fromkeys(base_eeg + extra))


def find_on_intervals(markers: np.ndarray, on_code: int, off_code: int) -> List[Tuple[int, int]]:
    """
    Find [start, end) indices where markers == on_code until markers == off_code.
    If no ON markers are found, return the whole file as a single interval.
    """
    on_starts = np.where(markers == on_code)[0]
    off_starts = np.where(markers == off_code)[0]

    intervals: List[Tuple[int, int]] = []
    if len(on_starts) == 0:
        return [(0, len(markers))]

    for start in on_starts:
        future_offs = off_starts[off_starts > start]
        if len(future_offs) > 0:
            intervals.append((start, future_offs[0]))
        else:
            intervals.append((start, len(markers)))
    return intervals


def main():
    parser = argparse.ArgumentParser(description="Compute ON-block PSD for EEG (+ bipolar) channels.")
    parser.add_argument("--input", required=True, help="Rereferenced CSV (CAR + bipolar)")
    parser.add_argument("--out_csv", required=True, help="Output ON-block PSD CSV")
    parser.add_argument("--out_fig", required=True, help="Output ON-block PSD figure PNG")
    parser.add_argument("--config", default="config/config.yaml", help="Main config YAML")
    parser.add_argument("--montages", default="config/montages.yaml", help="Montages YAML")
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Optional list of channel names to include (e.g. --channels C3 Cz C4). "
             "If omitted, all montage EEG + bipolar channels are used.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    cfg = load_main_config(args.config)
    montage = get_montage(cfg, load_montages(args.montages))

    df = pd.read_csv(in_path)

    fs = _estimate_fs(df, cfg)
    stim_freq = _parse_stim_freq_from_name(in_path.stem)
    psd_cfg = cfg.get("analysis", {}).get("psd", {})
    fmin, fmax = _get_freq_range(fs, psd_cfg, stim_freq)
    print(f"[PSD_ON] Frequency range: {fmin} → {fmax} Hz (stim={stim_freq})")

    stim_on_code, stim_off_code = _get_on_off_codes(cfg)

    channels = _get_channels_with_bipolar(df, cfg, montage)
    if args.channels:
        requested = {ch.strip() for ch in args.channels}
        channels = [ch for ch in channels if ch in requested]
    if not channels:
        raise ValueError("[PSD_ON] No channels selected/found. Check montage / --channels.")
    print(f"[PSD_ON] Channels: {channels}")

    markers = df["marker"].to_numpy().astype(int)
    intervals = find_on_intervals(markers, stim_on_code, stim_off_code)
    print(f"[PSD_ON] Using marker codes ON={stim_on_code}, OFF={stim_off_code}")
    print(f"[PSD_ON] Found {len(intervals)} ON intervals")

    nperseg = int(fs)  # ~1 second windows

    accumulated_psd = {ch: [] for ch in channels}
    freqs_ref = None

    for (start, end) in intervals:
        dur = end - start
        if dur < nperseg:
            continue

        for ch in channels:
            sig = df[ch].iloc[start:end].to_numpy().astype(float)

            if not np.any(np.isfinite(sig)):
                continue
            mean_val = np.nanmean(sig)
            if not np.isfinite(mean_val):
                mean_val = 0.0
            sig = np.where(np.isfinite(sig), sig, mean_val)

            freqs, Pxx = welch(sig, fs=fs, nperseg=nperseg, scaling="density")
            if freqs_ref is None:
                freqs_ref = freqs
            accumulated_psd[ch].append(Pxx)

    rows = []
    fig, ax = plt.subplots(figsize=(10, 6))

    if freqs_ref is not None:
        mask = (freqs_ref >= fmin) & (freqs_ref <= fmax)
        f_sel = freqs_ref[mask]

        for ch in channels:
            if not accumulated_psd[ch]:
                continue

            P_mean = np.mean(np.stack(accumulated_psd[ch]), axis=0)
            P_db = 10 * np.log10(P_mean + 1e-20)

            p_sel = P_mean[mask]
            db_sel = P_db[mask]

            ax.plot(f_sel, db_sel, label=ch)
            for f, p, db in zip(f_sel, p_sel, db_sel):
                rows.append({"freq": f, "channel": ch, "psd_on": p, "psd_on_db": db})

    # Vertical stim freq
    if stim_freq is not None and fmin <= stim_freq <= fmax:
        ax.axvline(
            stim_freq,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Stim {stim_freq:g} Hz",
        )

    out_fig = Path(args.out_fig)
    out_csv = Path(args.out_csv)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("[PSD_ON] No ON-block PSD records computed; writing empty CSV and blank figure.")
        plt.savefig(out_fig, dpi=150)
        plt.close()
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return

    ax.set_title(f"ON-Block PSD: {in_path.stem}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True, alpha=0.3)

    # Multi-column legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        n_labels = len(labels)
        ncol = max(1, min(n_labels, n_labels // 4 or 1))
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.15, 1, 1])
    else:
        plt.tight_layout()

    plt.savefig(out_fig, dpi=150)
    plt.close()

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[PSD_ON] Saved {out_csv} and {out_fig}")


if __name__ == "__main__":
    main()
