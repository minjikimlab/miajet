#!/bin/bash


#SBATCH --account=minjilab99
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --time=2:00:00  
#SBATCH --mail-type=FAIL 
#SBATCH --output=slurm_out/slurm-%j.out

conda activate fun-env

python run_fun_pipeline.py \
  --hic_file "${HIC_FILE}" \
  --data_name "${DATA_NAME}" \
  --genome "${GENOME}" \
  --res "${RES}" \
  --win "${WIN}"