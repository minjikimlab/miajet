#!/bin/bash

#SBATCH --job-name=miajet_demo_mouse
#SBATCH --account=minjilab99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gpus=0
#SBATCH --profile=all
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_out/slurm-%j.out

my_job_header

conda activate jet-env-rev

REPO="/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet"
DEMO="$REPO/notebooks-release/figure_input_files"

cd "$REPO"

python -m miajet "$DEMO/GSE199059_CD69negDPWTR1R2R3R4_merged.ice_small_chr3_.hic" \
 --chrom "chr3" \
 --exp_type "hic" \
 --resolution 25000 \
 --window_size 6000000 \
 --save_dir_root "$REPO/output_demo" \
 --num_cores 4 \
 --verbose \
 --diagnostic_plots
