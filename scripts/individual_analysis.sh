#!/bin/bash


#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g 
#SBATCH --gpus=0 
#SBATCH --time=4:00:00  
#SBATCH --mail-type=FAIL 

conda activate jet-env

python run_analysis_pipeline.py --sample "${SAMPLE}"
