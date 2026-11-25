#!/bin/bash

#SBATCH --job-name=miajet_demo_zebrafish
#SBATCH --account=minjilab99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --gpus=0
#SBATCH --profile=all
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_out/slurm-%j.out

my_job_header

conda activate jet-env-rev

REPO="/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet"
DEMO="$REPO/notebooks-release/figure_input_files"

cd "$REPO"

python -m miajet "$DEMO/zebrafish-embryo_sperm_hic_Wike-2021_4DNFI4P145EM_z11.ice_small_chr14_.hic" \
 --chrom "chr14" \
 --exp_type "hic" \
 --resolution 50000 \
 --window_size 6000000 \
 --compartment "False" \
 --angle_turbulence 0.75 \
 --consistency 0.4 \
 --q_val 0.5 \
 --save_dir_root "$REPO/output_demo" \
 --num_cores 4 \
 --verbose \
 --diagnostic_plots
