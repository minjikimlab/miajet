#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

conda activate jet-env

# The worker processes exactly one sample, indicated by $SAMPLE_KEY
python process_samples.py --config "${CONFIG_PATH}" --sample "${SAMPLE_KEY}"

