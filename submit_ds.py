#!/usr/bin/env python
import yaml
import subprocess
import os

# Load config
with open("submit_ds_config.yaml") as f:
    config = yaml.safe_load(f)

# Chromosome definitions (replaces chroms.sh)
CHROMS = {
    "hg19": [f"chr{i}" for i in range(1, 23)] + ["chrX"],
    "hg38": [f"chr{i}" for i in range(1, 23)] + ["chrX"],
    "mm10": [f"chr{i}" for i in range(1, 20)] + ["chrX"],
    "mm9": [f"chr{i}" for i in range(1, 20)] + ["chrX"],
    "danRer11": [f"chr{i}" for i in range(1, 25)],
    "ce10": ["chrI", "chrII", "chrIII", "chrIV", "chrV", "chrX"],
}

FIXED_ARGS = [
    "--save_dir_root", "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v2.0.2",
    "--num_cores", "4",
    "--verbose",
    "--diagnostic_plots",
]

# Keys that aren't CLI flags
SKIP_KEYS = {"file", "genome", "chroms", "mcool"}

# Only needed if the CLI flag differs from the YAML key
FLAG_RENAMES = {
    "exp": "exp_type",
    "norm": "normalization",
    "res": "resolution",
    "win": "window_size",
    "compartment": "compartment",
    "root_within": "root_within",
}

for sample, params in config["samples"].items():
    genome = params["genome"]
    chroms_val = params.get("chroms", "all")
    chroms = CHROMS[genome] if chroms_val == "all" else chroms_val.split(",")

    for chrom in chroms:
        cmd = ["python", "-m", "miajet", params["file"], "--chrom", chrom]

        # Dynamically add every other key as a CLI flag
        for key, val in params.items():
            if key in SKIP_KEYS:
                continue
            flag = FLAG_RENAMES.get(key, key)
            cmd += [f"--{flag}", str(val)]

        cmd += FIXED_ARGS

        job_name = f"{sample}_{chrom}"
        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            "--account=minjilab99",
            "--partition=standard",
            "--cpus-per-task=4",
            "--mem=60g",
            "--time=4:00:00",
            "--mail-type=FAIL",
            "--output=slurm_out/slurm-%j.out",
            "--wrap", " ".join(cmd),
        ]

        print(f"Submitting {job_name}")
        subprocess.run(sbatch_cmd)