#!/usr/bin/env python3
import os
import shlex
import subprocess
import yaml

# Load config
with open("submit_all_config.yaml") as f:
    config = yaml.safe_load(f)

SAVE_DIR_ROOT = "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v2.0.2"
COMBINE_SCRIPT = "combine_chrom_results.py"

SBATCH_ARGS = [
    "--account=minjilab99",
    "--partition=standard",
    "--cpus-per-task=2",
    "--mem=16g",
    "--gpus=0",
    "--time=1:00:00",
    "--mail-type=FAIL",
    "--output=slurm_out/slurm-%j.out",
]

if not os.path.isfile(COMBINE_SCRIPT):
    raise FileNotFoundError(f"Missing combine script: {COMBINE_SCRIPT}")

for sample, params in config.get("samples", {}).items():
    missing = [k for k in ("file", "res") if k not in params]
    if missing:
        print(f"Skipping {sample}: missing required key(s): {', '.join(missing)}")
        continue

    cmd = [
        "python",
        COMBINE_SCRIPT,
        "--hic_file",
        str(params["file"]),
        "--resolution",
        str(params["res"]),
        "--save_dir_root",
        SAVE_DIR_ROOT,
    ]

    # Optional: if present in YAML, pass through to combine script.
    if "folder_name" in params:
        cmd += ["--folder_name", str(params["folder_name"])]

    result_types = params.get("combine_result_types")
    if result_types:
        cmd += ["--result_types"]
        if isinstance(result_types, list):
            cmd += [str(x) for x in result_types]
        else:
            cmd += [str(result_types)]

    out_dir = params.get("combine_out_dir")
    if out_dir:
        cmd += ["--out_dir", str(out_dir)]

    strict_val = params.get("combine_strict", False)
    if isinstance(strict_val, str):
        strict_val = strict_val.lower() in {"1", "true", "yes", "y"}
    if strict_val:
        cmd += ["--strict"]

    job_name = f"{sample}_combine".replace("/", "_").replace(" ", "_")
    # sbatch_cmd = [
    #     "sbatch",
    #     f"--job-name={job_name}",
    #     *SBATCH_ARGS,
    #     "--wrap",
    #     shlex.join(cmd),
    # ]
    # print(f"Submitting {job_name}")
    # subprocess.run(sbatch_cmd)

    print(f"Running in current session: {job_name}")
    subprocess.run(cmd, check=True)