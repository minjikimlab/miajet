#!/bin/bash


#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=2
#SBATCH --mem=30g
#SBATCH --gpus=0 
#SBATCH --time=2:00:00  
#SBATCH --mail-type=FAIL 

conda activate fontanka

python run_fontanka_pipeline.py \
  --hic_file "${HIC_FILE}" \
  --data_name "${DATA_NAME}" \
  --genome "${GENOME}" \
  --res "${RES}" \
  --win "${WIN}"

