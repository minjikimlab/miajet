#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

my_job_header

conda activate jet-env

source chroms.sh

yq -r '.samples | keys | .[]' submit_all_config.yaml | while IFS= read -r sample; do

  hic=$(yq -r ".samples[\"$sample\"].file"    submit_all_config.yaml)
  genome=$(yq -r ".samples[\"$sample\"].genome" submit_all_config.yaml)
  norm=$(yq -r ".samples[\"$sample\"].norm"    submit_all_config.yaml)
  res=$(yq -r ".samples[\"$sample\"].res"      submit_all_config.yaml)
  exp=$(yq -r ".samples[\"$sample\"].exp"      submit_all_config.yaml)
  win=$(yq -r ".samples[\"$sample\"].win"      submit_all_config.yaml)
  
  for chrom in ${CHROMS[$genome]}; do
    sbatch \
      --job-name="${sample}_${chrom}_${norm}_${res}" \
      --export=HIC_FILE="$hic",CHROM="$chrom",NORM="$norm",RES="$res",EXP="$exp",WIN="$win" \
      job_all.sh
  done

done






