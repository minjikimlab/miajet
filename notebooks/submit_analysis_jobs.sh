#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

my_job_header

conda activate jet-env


CONFIG_PATH="../submit_all_config.yaml"
readarray -t SAMPLE_KEYS < <(
    python - <<'PY'
import yaml, sys
with open("../submit_all_config.yaml") as cf:
    print("\n".join(yaml.safe_load(cf)["samples"].keys()))
PY
)

# ── 2.  Fire off one job per sample -------------------------------------------
for SAMPLE_KEY in "${SAMPLE_KEYS[@]}"; do
    sbatch --export=ALL,SAMPLE_KEY="${SAMPLE_KEY}",CONFIG_PATH="${CONFIG_PATH}" run_sample.sh
done