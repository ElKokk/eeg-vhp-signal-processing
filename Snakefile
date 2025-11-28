###############################################
# Snakefile – EEG + VHP pipeline
###############################################

import os

configfile: "config/config.yaml"

RAW_DIR      = config["paths"]["raw_dir"]
PROC_DIR     = config["paths"]["processed_dir"]
RESULTS_DIR  = config["paths"]["results_dir"]
RECORDINGS   = config["recordings"]

# Result subdirs
PSD_RAW_DIR       = os.path.join(RESULTS_DIR, "psd_raw")
PSD_REREF_DIR     = os.path.join(RESULTS_DIR, "psd_reref")
SPEC_RAW_DIR      = os.path.join(RESULTS_DIR, "spectrograms_raw")
SPEC_REREF_DIR    = os.path.join(RESULTS_DIR, "spectrograms_reref")
PSD_ON_DIR        = os.path.join(RESULTS_DIR, "psd_on")
SNR_DIR           = os.path.join(RESULTS_DIR, "snr")

# SNR config
_snr_cfg    = config.get("analysis", {}).get("snr", {})
SNR_TARGETS = _snr_cfg.get("targets", [22, 24, 26, 28, 30, 32, 34, 36])
SNR_BW      = _snr_cfg.get("bw", 2.0)
SNR_EXCLUDE = _snr_cfg.get("exclude", 0.5)


rule all:
    input:
        expand(f"{PSD_RAW_DIR}/{{rec}}.psd.csv", rec=RECORDINGS),
        expand(f"{PSD_REREF_DIR}/{{rec}}.psd.csv", rec=RECORDINGS),
        expand(f"{SPEC_REREF_DIR}/{{rec}}.spec.done", rec=RECORDINGS),
        expand(f"{PSD_ON_DIR}/{{rec}}.psd_on.csv", rec=RECORDINGS),
        expand(f"{SNR_DIR}/{{rec}}.snr.csv", rec=RECORDINGS),

###############################################
# 1) PREPROCESS
###############################################
rule preprocess:
    input:
        f"{RAW_DIR}/{{rec}}.txt"
    output:
        f"{PROC_DIR}/{{rec}}.preproc.csv"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.preprocess \
            --input {input} \
            --output {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 2) PSD (RAW)
###############################################
rule psd_raw:
    input:
        f"{PROC_DIR}/{{rec}}.preproc.csv"
    output:
        csv=f"{PSD_RAW_DIR}/{{rec}}.psd.csv",
        fig=f"{PSD_RAW_DIR}/{{rec}}.psd.png"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.psd \
            --input {input} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 3) SPECTROGRAM (RAW)
###############################################
rule spectrogram_raw:
    input:
        f"{PROC_DIR}/{{rec}}.preproc.csv"
    output:
        f"{SPEC_RAW_DIR}/{{rec}}.spec.done"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.spectrogram \
            --input {input} \
            --out_flag {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 4) REREFERENCE
###############################################
rule reref:
    input:
        f"{PROC_DIR}/{{rec}}.preproc.csv"
    output:
        f"{PROC_DIR}/{{rec}}.reref.csv"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.reference \
            --input {input} \
            --output {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 5) PSD (REREF)
###############################################
rule psd_reref:
    input:
        f"{PROC_DIR}/{{rec}}.reref.csv"
    output:
        csv=f"{PSD_REREF_DIR}/{{rec}}.psd.csv",
        fig=f"{PSD_REREF_DIR}/{{rec}}.psd.png"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.psd \
            --input {input} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 6) SPECTROGRAM (REREF)
###############################################
rule spectrogram_reref:
    input:
        f"{PROC_DIR}/{{rec}}.reref.csv"
    output:
        f"{SPEC_REREF_DIR}/{{rec}}.spec.done"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.spectrogram \
            --input {input} \
            --out_flag {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 7) ON-BLOCK AVERAGED PSD
###############################################
rule psd_onblocks:
    input:
        f"{PROC_DIR}/{{rec}}.reref.csv"
    output:
        csv=f"{PSD_ON_DIR}/{{rec}}.psd_on.csv",
        fig=f"{PSD_ON_DIR}/{{rec}}.psd_on.png"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.psd_onblocks \
            --input {input} \
            --out_csv {output.csv} \
            --out_fig {output.fig} \
            --config {params.cfg} \
            --montages {params.mont}
        """

###############################################
# 8) SNR
###############################################
rule snr:
    input:
        psd_on=f"{PSD_ON_DIR}/{{rec}}.psd_on.csv"
    output:
        csv=f"{SNR_DIR}/{{rec}}.snr.csv",
        fig=f"{SNR_DIR}/{{rec}}.snr.png"
    params:
        bw=SNR_BW,
        targets=" ".join(map(str, SNR_TARGETS)),
        exclude=SNR_EXCLUDE
    shell:
        """
        python -m src.snr_from_psd \
            --psd_csv {input.psd_on} \
            --targets {params.targets} \
            --bw {params.bw} \
            --exclude {params.exclude} \
            --out_csv {output.csv} \
            --out_fig {output.fig}
        """