#!/bin/bash

#SBATCH --account=minjilab99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE,FAIL

CONFIG=../submit_all_config.yaml

# one unique log per invocation (sortable stamp; $$ guards same-second collisions)
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
JOBS_TSV="submitted_fun_jobs_${RUN_STAMP}_$$.tsv"
printf 'jobid\tsample\tjob_name\n' > "$JOBS_TSV"

yq -r '.samples | keys | .[]' "$CONFIG" | while IFS= read -r sample; do
  hic=$(yq -r ".samples[\"$sample\"].mcool"    "$CONFIG")
  genome=$(yq -r ".samples[\"$sample\"].genome" "$CONFIG")
  res=$(yq -r ".samples[\"$sample\"].res"      "$CONFIG")
  win=$(yq -r ".samples[\"$sample\"].win"      "$CONFIG")

  job_name="${sample}_FUN_${res}"

  jobid=$(sbatch --parsable \
    --job-name="$job_name" \
    --export=HIC_FILE="$hic",DATA_NAME="$sample",GENOME="$genome",RES="$res",WIN="$win" \
    individual_fun.sh)
  jobid=${jobid%%;*}        # strip ';cluster' suffix if the cluster is federated

  if [[ -n "$jobid" ]]; then
    printf '%s\t%s\t%s\n' "$jobid" "$sample" "$job_name" >> "$JOBS_TSV"
    echo "Submitted $job_name as job $jobid"
  else
    echo "WARNING: submission failed for $sample" >&2
  fi
done

echo "Wrote job records to $JOBS_TSV"