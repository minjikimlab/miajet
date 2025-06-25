#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE

pairs_list=(
    /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/K562_intacthic_ENCODE-2023_ENCFF808MAG_hg38.pairs.gz
    /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/K562_hic_Rao-2014_4DNFI2R1W3YW_hg38.pairs.gz
    /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/HCT116_RAD21-auxin-6hr_intacthic_Guckelberger-2024_ENCFF461RFV_hg38.pairs.gz
    /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/HCT116_RAD21-auxin-0hr_intacthic_Guckelberger-2024_ENCFF109GNA_hg38.pairs.gz
    /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GM12878_intacthic_ENCFF785BPC.pairs.gz
)
assemblies=(
  hg38
  hg38
  hg38
  hg38
  hg38
)
genome_sizes=(
  /nfs/turbo/umms-minjilab/processing/genomes/hg38/hg38.chrom.sizes
  /nfs/turbo/umms-minjilab/processing/genomes/hg38/hg38.chrom.sizes
  /nfs/turbo/umms-minjilab/processing/genomes/hg38/hg38.chrom.sizes
  /nfs/turbo/umms-minjilab/processing/genomes/hg38/hg38.chrom.sizes
  /nfs/turbo/umms-minjilab/processing/genomes/hg38/hg38.chrom.sizes
)

for i in "${!pairs_list[@]}"; do
    pairs="${pairs_list[i]}"
    assembly="${assemblies[i]}"
    sizes="${genome_sizes[i]}"
    name="$(basename "${pairs%.pairs.gz}")"
    sbatch \
      --job-name="pairs_to_mcool_${name}" \
      --export=PAIRS="${pairs}",ASSEMBLY="${assembly}",SIZES="${sizes}" \
      job_pairs_to_mcool.sh
done
