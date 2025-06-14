#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

chromosomes=(
  "chr1"
  "chr2"
  "chr3"
  "chr4"
  "chr5"
  "chr6"
  "chr7"
  "chr8"
  "chr9"
  "chr10"
  "chr11"
  "chr12"
  "chr13"
  "chr14"
  "chr15"
  "chr16"
  "chr17"
  "chr18"
  "chr19"
  "chrX"
)

for chrom in "${chromosomes[@]}"; do
  # sbatch --job-name="Guo et al. mm10 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte_50Kb.sbat
  # sleep 1
  sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9_50Kb.sbat
  sleep 1
done
