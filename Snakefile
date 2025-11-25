import os

# Tell Snakemake which YAML config to use
configfile: "config/config.yaml"

RAW_DIR = config["paths"]["raw_dir"]
PROC_DIR = config["paths"]["processed_dir"]
RECORDINGS = config["recordings"]

rule all:
    input:
        # One structured file per recording
        expand(
            f"{PROC_DIR}" + "/{rec}.structured.csv",
            rec=RECORDINGS
        )

rule structure_raw:
    """
    Convert raw FreeEEG32 CSV -> structured CSV with t_rel and channel names.
    """
    input:
        # pattern with wildcard {rec}
        f"{RAW_DIR}" + "/{rec}.csv"
    output:
        f"{PROC_DIR}" + "/{rec}.structured.csv"
    params:
        cfg="config/config.yaml",
        mont="config/montages.yaml"
    shell:
        """
        python -m src.structure_raw \
            --input {input} \
            --output {output} \
            --config {params.cfg} \
            --montages {params.mont}
        """
