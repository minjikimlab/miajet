#!/bin/bash


#SBATCH --account=minjilab99
#SBATCH --partition=large-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=200g
#SBATCH --gpus=0 
#SBATCH --time=24:00:00  
#SBATCH --mail-type=FAIL 
#SBATCH --output=slurm_out/slurm-%j.out

conda activate fontanka

python run_fontanka_pipeline.py \
  --hic_file "${HIC_FILE}" \
  --data_name "${DATA_NAME}" \
  --genome "${GENOME}" \
  --res "${RES}" \
  --win "${WIN}"

