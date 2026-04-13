#!/bin/bash

chromosomes=(
  # "chr1"
  # "chr2"
  # "chr3"
  "chr4"
  # "chr5"
  # "chr6"
  # "chr7"
  # "chr8"
  # "chr9"
  # "chr10"
  # "chr11"
  # "chr12"
  # "chr13"
  # "chr14"
  # "chr15"
  # "chr16"
  # "chr17"
  # "chr18"
  # "chr19"
  # "chrX"
)

# chromosomes=(
#   "chrX"
# )


for chrom in "${chromosomes[@]}"; do
  # sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9.sbat
  sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9_gs1.sbat
  sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9_gs2.sbat
  sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9_gs3.sbat
  sbatch --job-name="Guo et al. mm9 ${chrom}" --export=CHROM=${chrom} job_DP_thymocyte-mm9_gs4.sbat
done
