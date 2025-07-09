#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

conda activate fun-env


yq -r '.samples | keys | .[]' ../submit_all_config.yaml | while IFS= read -r sample; do
  hic=$(yq -r ".samples[\"$sample\"].mcool"    ../submit_all_config.yaml)
  genome=$(yq -r ".samples[\"$sample\"].genome" ../submit_all_config.yaml)
  res=$(yq -r ".samples[\"$sample\"].res"      ../submit_all_config.yaml)
  win=$(yq -r ".samples[\"$sample\"].win"      ../submit_all_config.yaml)

  sbatch \
    --job-name="${sample}_FUN_${res}" \
    --export=HIC_FILE="$hic",DATA_NAME="$sample",GENOME="$genome",RES="$res",WIN="$win" \
    individual_fun.sh
done