###############################################
# Snakefile – EEG + VHP pipeline (STREAMING BOARD)
###############################################

import os

# Use the STREAMING config
configfile: "config/config_streaming.yaml"

# Paths & recordings defined in config_streaming.yaml
RAW_DIR      = config["paths"]["raw_dir"]
PROC_DIR     = config["paths"]["processed_dir"]
RESULTS_DIR  = config["paths"]["results_dir"]
RECORDINGS   = config["recordings"]

# Result subdirs
PSD_RAW_DIR           = os.path.join(RESULTS_DIR, "psd_raw")
PSD_REREF_DIR         = os.path.join(RESULTS_DIR, "psd_reref")
PSD_ON_DIR            = os.path.join(RESULTS_DIR, "psd_onblocks")
SPECTROGRAM_RAW_DIR   = os.path.join(RESULTS_DIR, "spectrograms_raw")
SPECTROGRAM_REREF_DIR = os.path.join(RESULTS_DIR, "spectrograms_reref")
SNR_DIR               = os.path.join(RESULTS_DIR, "snr")

# Analysis / preprocessing options from config (with sensible fallbacks)
pre_cfg   = config.get("preprocessing", {})
spec_cfg  = config.get("analysis", {}).get("spectrogram", {})
psd_cfg   = config.get("analysis", {}).get("psd", {})
reref_cfg = config.get("analysis", {}).get("reref", {})
snr_cfg   = config.get("analysis", {}).get("snr", {})

# Channels to plot (optional)
SPECTROGRAM_CHANNELS = spec_cfg.get("channels", [])
PSD_CHANNELS         = psd_cfg.get("channels", [])

# SNR parameters
SNR_BW      = float(snr_cfg.get("bandwidth", 1.0))
SNR_TARGETS = snr_cfg.get("targets", [])
SNR_EXCLUDE = snr_cfg.get("exclude", [])

###############################################
# Helper functions
###############################################

def _channels_arg(ch_list):
    """Return a CLI fragment like '--channels C3 Cz' or '' if empty."""
    if not ch_list:
        return ""
    return "--channels " + " ".join(str(ch) for ch in ch_list)

###############################################
# Rule: all
###############################################

rule all:
    input:
        # Structured & preprocessed
        expand(os.path.join(PROC_DIR, "{rec}.structured.csv"), rec=RECORDINGS),
        expand(os.path.join(PROC_DIR, "{rec}.preproc.csv"),   rec=RECORDINGS),
        expand(os.path.join(PROC_DIR, "{rec}.reref.csv"),     rec=RECORDINGS),

        # PSDs
        expand(os.path.join(PSD_RAW_DIR,   "{rec}.psd.csv"),   rec=RECORDINGS),
        expand(os.path.join(PSD_RAW_DIR,   "{rec}.psd.png"),   rec=RECORDINGS),
        expand(os.path.join(PSD_REREF_DIR, "{rec}.psd.csv"),   rec=RECORDINGS),
        expand(os.path.join(PSD_REREF_DIR, "{rec}.psd.png"),   rec=RECORDINGS),
        expand(os.path.join(PSD_ON_DIR,    "{rec}.psd_on.csv"), rec=RECORDINGS),
        expand(os.path.join(PSD_ON_DIR,    "{rec}.psd_on.png"), rec=RECORDINGS),

        # Spectrograms
        expand(os.path.join(SPECTROGRAM_RAW_DIR,   "{rec}.spec.done"), rec=RECORDINGS),
        expand(os.path.join(SPECTROGRAM_REREF_DIR, "{rec}.spec.done"), rec=RECORDINGS),

        # SNR summary
        expand(os.path.join(SNR_DIR, "{rec}.snr.csv"), rec=RECORDINGS),
        expand(os.path.join(SNR_DIR, "{rec}.snr.png"), rec=RECORDINGS)

###############################################
# Rule: structure_raw
###############################################

rule structure_raw:
    input:
        csv=os.path.join(RAW_DIR, "{rec}.csv")
    output:
        structured=os.path.join(PROC_DIR, "{rec}.structured.csv")
    shell:
        """
        python -m src.structure_raw \
            --input {input.csv} \
            --output {output.structured} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml
        """

###############################################
# Rule: preprocess
###############################################

rule preprocess:
    input:
        structured=os.path.join(PROC_DIR, "{rec}.structured.csv")
    output:
        preproc=os.path.join(PROC_DIR, "{rec}.preproc.csv")
    shell:
        """
        python -m src.preprocess \
            --input {input.structured} \
            --output {output.preproc} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml
        """

###############################################
# Rule: rereference (CAR + bipolar)
###############################################

rule reref:
    input:
        preproc=os.path.join(PROC_DIR, "{rec}.preproc.csv")
    output:
        reref=os.path.join(PROC_DIR, "{rec}.reref.csv")
    shell:
        """
        python -m src.reference \
            --input {input.preproc} \
            --output {output.reref} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml
        """

###############################################
# Rule: PSD on raw (preprocessed) channels
###############################################

rule psd_raw:
    input:
        preproc=os.path.join(PROC_DIR, "{rec}.preproc.csv")
    output:
        csv=os.path.join(PSD_RAW_DIR, "{rec}.psd.csv"),
        fig=os.path.join(PSD_RAW_DIR, "{rec}.psd.png")
    params:
        ch_arg=lambda wildcards: _channels_arg(PSD_CHANNELS)
    shell:
        """
        python -m src.psd \
            --input {input.preproc} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml \
            {params.ch_arg}
        """

###############################################
# Rule: PSD on rereferenced channels
###############################################

rule psd_reref:
    input:
        reref=os.path.join(PROC_DIR, "{rec}.reref.csv")
    output:
        csv=os.path.join(PSD_REREF_DIR, "{rec}.psd.csv"),
        fig=os.path.join(PSD_REREF_DIR, "{rec}.psd.png")
    params:
        ch_arg=lambda wildcards: _channels_arg(PSD_CHANNELS)
    shell:
        """
        python -m src.psd \
            --input {input.reref} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml \
            {params.ch_arg}
        """

###############################################
# Rule: PSD on ON-blocks only (from rereferenced data)
###############################################

rule psd_onblocks:
    input:
        reref=os.path.join(PROC_DIR, "{rec}.reref.csv")
    output:
        csv=os.path.join(PSD_ON_DIR, "{rec}.psd_on.csv"),
        fig=os.path.join(PSD_ON_DIR, "{rec}.psd_on.png")
    params:
        ch_arg=lambda wildcards: _channels_arg(PSD_CHANNELS)
    shell:
        """
        python -m src.psd_onblocks \
            --input {input.reref} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml \
            {params.ch_arg}
        """

###############################################
# Rule: Spectrograms on raw (preprocessed) channels
###############################################

rule spectrogram_raw:
    input:
        preproc=os.path.join(PROC_DIR, "{rec}.preproc.csv")
    output:
        flag=os.path.join(SPECTROGRAM_RAW_DIR, "{rec}.spec.done")
    params:
        ch_arg=lambda wildcards: _channels_arg(SPECTROGRAM_CHANNELS)
    shell:
        """
        python -m src.spectrogram \
            --input {input.preproc} \
            --out_flag {output.flag} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml \
            {params.ch_arg}
        """

###############################################
# Rule: Spectrograms on rereferenced channels
###############################################

rule spectrogram_reref:
    input:
        reref=os.path.join(PROC_DIR, "{rec}.reref.csv")
    output:
        flag=os.path.join(SPECTROGRAM_REREF_DIR, "{rec}.spec.done")
    params:
        ch_arg=lambda wildcards: _channels_arg(SPECTROGRAM_CHANNELS)
    shell:
        """
        python -m src.spectrogram \
            --input {input.reref} \
            --out_flag {output.flag} \
            --config config/config_streaming.yaml \
            --montages config/montages.yaml \
            {params.ch_arg}
        """

###############################################
# Rule: SNR from ON-block PSD
###############################################

rule snr:
    input:
        psd_on=os.path.join(PSD_ON_DIR, "{rec}.psd_on.csv")
    output:
        csv=os.path.join(SNR_DIR, "{rec}.snr.csv"),
        fig=os.path.join(SNR_DIR, "{rec}.snr.png")
    params:
        bw=SNR_BW,
        targets=lambda wildcards: " ".join(map(str, SNR_TARGETS)),
        exclude=lambda wildcards: " ".join(map(str, SNR_EXCLUDE)),
    shell:
        """
        python -m src.snr_from_psd \
            --psd_csv {input.psd_on} \
            --targets {params.targets} \
            --bandwidth {params.bw} \
            --exclude {params.exclude} \
            --col psd_on \
            --out_csv {output.csv} \
            --out_fig {output.fig}
        """
