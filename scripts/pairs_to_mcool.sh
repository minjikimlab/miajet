#!/bin/bash

#SBATCH --job-name=pairs_to_mcool
#SBATCH --account=minjilab0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=6
#SBATCH --mem=80g
#SBATCH --gpus=0
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL

# Convert .pairs.gz files to ICE balanced .mcool
# cooler, genome sizes file 

# Parameters
BIN_SIZE=5000                             
NPROC=6                 
tmp_dir=/scratch/minjilab_root/minjilab0/sionkim/temp                    

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

# For latest version of cooler
conda activate jet-env

for i in "${!pairs_list[@]}"; do
    assembly=${assemblies[i]}
    sizes=${genome_sizes[i]}
    pairs=${pairs_list[i]}

    # Derive sample name (strip .pairs.gz)
    sample_with_path="${pairs%.pairs.gz}"
    sample="$(basename "${pairs%.pairs.gz}")"
    
    # Define intermediate .cool and final .mcool names
    cool_file="${tmp_dir}/${sample}.${BIN_SIZE}.cool"
    mcool_file="${sample_with_path}.ice.mcool"
    
    echo "Processing ${pairs} -> ${mcool_file}"
    
    # 1) Load pairs into single-resolution .cool
    cooler cload pairs \
        -c1 2 -p1 3 \
        -c2 4 -p2 5 \
        --assembly $assembly \
        "${sizes}:${BIN_SIZE}" \
        "${pairs}" \
        "${cool_file}"
    
    # ── zoomify + ICE balance ──
    cooler zoomify \
        --nproc ${NPROC} \
        --resolutions ${BIN_SIZE}N \
        --balance \
        -o "${mcool_file}" \
        "${cool_file}"
    
    # Optional: remove intermediate .cool to save space
    rm -f "${cool_file}"
    
    echo "Finished: ${mcool_file}"
done
