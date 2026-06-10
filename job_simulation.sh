#!/bin/bash

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

python -m miajet "${HIC_FILE}" \
  --chrom "chrS" \
  --exp_type "hic" \
  --resolution "25000" \
  --compartment "False" \
  --normalization "NONE" \
  --window_size "5000000" \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v2.0.3_runtime" \
  --root_within "5" \
  --q_val "0.25" \
  --angle_range 45 135 \
  --num_cores 4 \
  --verbose \
# --angle_range 60 120 \

# Typically, angle range should be 60, 120 for jets and 45, 135 if we want to include stripes as well