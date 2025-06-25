#!/bin/bash

#SBATCH --account=minjilab0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=6
#SBATCH --mem=80g
#SBATCH --gpus=0
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL

conda activate jet-env

# fixed parameters
BIN_SIZE=5000
NPROC=6
tmp_dir=/scratch/minjilab_root/minjilab0/sionkim/temp

# from sbatch --export
pairs="${PAIRS}"
assembly="${ASSEMBLY}"
sizes="${SIZES}"

# Derive sample name (strip .pairs.gz)
sample_with_path="${pairs%.pairs.gz}"
sample="$(basename "${pairs%.pairs.gz}")"

# Define intermediate .cool and final .mcool names
cool_file="${tmp_dir}/${sample}.${BIN_SIZE}.cool"
mcool_file="${sample_with_path}.ice.mcool"

echo "Processing ${pairs} -> ${mcool_file}"

# 1) Load pairs into single-resolution .cool
cooler cload pairs \
    -c1 2 -p1 3 \
    -c2 4 -p2 5 \
    --assembly $assembly \
    "${sizes}:${BIN_SIZE}" \
    "${pairs}" \
    "${cool_file}"

# ── zoomify + ICE balance ──
cooler zoomify \
    --nproc ${NPROC} \
    --resolutions ${BIN_SIZE}N \
    --balance \
    -o "${mcool_file}" \
    "${cool_file}"

# Optional: remove intermediate .cool to save space
rm -f "${cool_file}"

echo "Finished: ${mcool_file}"
