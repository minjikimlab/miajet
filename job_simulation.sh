#!/bin/bash

#SBATCH --account=minjilab99 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g 
#SBATCH --gpus=0 
#SBATCH --profile=all
#SBATCH --time=2:00:00  
#SBATCH --mail-type=FAIL 
#SBATCH --output=slurm_out/slurm-%j.out

my_job_header

conda activate jet-env-rev

python -m miajet "${HIC_FILE}" \
  --chrom "chrS" \
  --exp_type "hic" \
  --resolution "25000" \
  --compartment "False" \
  --normalization "NONE" \
  --window_size "5000000" \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v2.0.3_test" \
  --root_within "5" \
  --q_val "0.25" \
  --num_cores 4 \
  --verbose \
  --diagnostic_plots
