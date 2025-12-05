#!/usr/bin/env python

"""
src/psd.py

Compute Welch PSDs for EEG (+ bipolar) channels.

Features:
- Uses fs from config (eeg.sampling_rate) with sanity check vs timestamps.
- Frequency range automatically extended to include stimulation frequency.
- Vertical red line at stim frequency.
- Multi-column legend at the bottom so electrode labels stay readable.
- Optional --channels to restrict which electrodes are plotted.
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

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
            f"[PSD] estimated fs from timestamps ≈ {fs_est:.2f} Hz, "
            f"but config says {target_fs:.2f} Hz → using config fs"
        )
        return target_fs

    print(f"[PSD] fs ≈ {fs_est:.2f} Hz")
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

    extra_channels = []
    for pair in bipolar_pairs:
        parsed = parse_pair(pair)
        if parsed:
            a, b = parsed
            col_name = f"{a}-{b}"
            if col_name in df.columns:
                extra_channels.append(col_name)

    all_channels = base_eeg + extra_channels
    return list(dict.fromkeys(all_channels))


def main():
    parser = argparse.ArgumentParser(description="Compute Welch PSD for EEG (+ bipolar) channels.")
    parser.add_argument("--input", required=True, help="Preprocessed CSV")
    parser.add_argument("--out_csv", required=True, help="Output PSD CSV")
    parser.add_argument("--out_fig", required=True, help="Output PSD figure (PNG)")
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
        raise FileNotFoundError(f"Input file not found: {in_path}")

    cfg = load_main_config(args.config)
    montages = load_montages(args.montages)
    montage = get_montage(cfg, montages)

    df = pd.read_csv(in_path)

    fs = _estimate_fs(df, cfg)
    stim_freq = _parse_stim_freq_from_name(in_path.stem)

    psd_cfg = cfg.get("analysis", {}).get("psd", {})
    fmin, fmax = _get_freq_range(fs, psd_cfg, stim_freq)
    print(f"[PSD] Frequency range: {fmin} → {fmax} Hz (stim={stim_freq})")

    channels = _get_channels_with_bipolar(df, cfg, montage)
    if args.channels:
        requested = {ch.strip() for ch in args.channels}
        channels = [ch for ch in channels if ch in requested]
    if not channels:
        raise ValueError("[PSD] No channels selected/found to analyse. Check montage / --channels.")
    print(f"[PSD] Analyzing channels: {channels}")

    out_csv = Path(args.out_csv)
    out_fig = Path(args.out_fig)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    records = []
    fig, ax = plt.subplots(figsize=(10, 6))

    nperseg_cfg = int(psd_cfg.get("nperseg", 2048))

    for ch in channels:
        sig = df[ch].to_numpy().astype(float)

        if not np.any(np.isfinite(sig)):
            print(f"[PSD] Channel {ch}: all NaNs, skipping.")
            continue
        mean_val = np.nanmean(sig)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        sig = np.where(np.isfinite(sig), sig, mean_val)

        eff_nperseg = min(nperseg_cfg, len(sig))
        if eff_nperseg < 2:
            print(f"[PSD] Channel {ch}: signal too short (len={len(sig)}), skipping.")
            continue

        freqs, psd_vals = welch(sig, fs=fs, nperseg=eff_nperseg, scaling="density")

        mask = (freqs >= fmin) & (freqs <= fmax)
        f_sel = freqs[mask]
        p_sel = psd_vals[mask]
        if f_sel.size == 0:
            print(f"[PSD] Channel {ch}: no freqs in [{fmin},{fmax}] Hz, skipping.")
            continue

        p_db = 10 * np.log10(p_sel + 1e-20)

        ax.plot(f_sel, p_db, label=ch, linewidth=1.0, alpha=0.9)

        for f, p, db in zip(f_sel, p_sel, p_db):
            records.append({"freq": f, "channel": ch, "psd": p, "psd_db": db})

    # Vertical stim frequency line
    if stim_freq is not None and fmin <= stim_freq <= fmax:
        ax.axvline(
            stim_freq,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Stim {stim_freq:g} Hz",
        )

    if not records:
        print("[PSD] No PSD records computed; writing empty CSV and blank figure.")
        plt.savefig(out_fig, dpi=150)
        plt.close()
        pd.DataFrame(records).to_csv(out_csv, index=False)
        return

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    title_extra = f" (f0={stim_freq:.0f} Hz)" if stim_freq is not None else ""
    ax.set_title(f"PSD (Welch) — {in_path.stem}{title_extra}")
    ax.grid(True, linestyle="--", alpha=0.5)

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

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"[PSD] Done. Saved {out_csv} and {out_fig}")


if __name__ == "__main__":
    main()
