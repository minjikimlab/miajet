#!/bin/bash

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

# Use the CHROM environment variable passed by the wrapper
python -m miajet ./demo_data/zebrafish-embryo_sperm_hic_Wike-2021_4DNFI4P145EM_z11.ice_small_chr14_.hic \
 --chrom "chr14" \
 --exp_type "hic" \
 --resolution 50000 \
 --jet_widths 5 45 \
 --window_size 3000000 \
 --compartment "False" \
 --root_within 10 \
 --angle_turbulence 0.4 \
 --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_demo" \
 --num_cores 4 \
 --verbose \
 --diagnostic_plots