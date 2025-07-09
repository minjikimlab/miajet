#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 


conda activate jet-env


yq -r '.samples | keys | .[]' ../submit_all_config.yaml | while IFS= read -r sample; do
  sbatch \
    --job-name="${sample}_MIAJET_ANALYSIS" \
    --export=SAMPLE="$sample" \
    individual_analysis.sh
done