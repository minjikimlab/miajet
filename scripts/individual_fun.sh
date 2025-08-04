#!/bin/bash


#SBATCH --account=rjhryan0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --time=2:00:00  
#SBATCH --mail-type=FAIL 

echo "Using rjhyrna0 account"

conda activate fun-env

python run_fun_pipeline.py \
  --hic_file "${HIC_FILE}" \
  --data_name "${DATA_NAME}" \
  --genome "${GENOME}" \
  --res "${RES}" \
  --win "${WIN}"