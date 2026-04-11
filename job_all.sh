#!/bin/bash

#SBATCH --account=minjilab99 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --profile=all
#SBATCH --time=4:00:00  
#SBATCH --mail-type=FAIL 

my_job_header

conda activate jet-env-rev

python -m miajet "${HIC_FILE}" \
  --chrom "${CHROM}" \
  --exp_type "${EXP}" \
  --resolution "${RES}" \
  --normalization "${NORM}" \
  --window_size "${WIN}" \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v1.1.0" \
  --num_cores 4 \
  --verbose \

