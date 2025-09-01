#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=6
#SBATCH --mem=120g 
#SBATCH --gpus=0 
#SBATCH --time=10:00:00  
#SBATCH --mail-type=FAIL 

my_job_header

conda activate jet-env

# AUTHOR: Ben B from https://www.biostars.org/p/360254/#9516679

# MCOOL="/nfs/turbo/umms-minjilab/downloaded_data/zebrafish-embryo_sperm_hic_Wike-2021_4DNFI4P145EM_z11.ice.mcool"
# GENOME="danRer11"
# RESOLUTION=50000
# CHROM="chr14"

MCOOL="/nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.ice.mcool"
GENOME="mm9"
RESOLUTION=25000
CHROM="chr3"

input_mcool="${MCOOL}"

output_hic=${input_mcool%.*}_small_${CHROM}_${selected_res}.hic

chrom_sizes=/nfs/turbo/umms-minjilab/processing/genomes/${GENOME}/${GENOME}.chrom.sizes

juicer_tools_jar=/nfs/turbo/umms-minjilab/packages/juicer/CPU/juicer_tools_1.22.01.jar

# Get the resolutions stored in the .mcool file
resolutions=$(h5ls -r $input_mcool | grep -Eo 'resolutions/[0-9]+' | cut -d '/' -f 2 | sort -n | uniq)
echo $resolutions
selected_res=$RESOLUTION
echo "Selected resolution: $selected_res"

# Use Cooler to write the .mcool matrix as interactions in bedpe format
output_bedpe="${input_mcool%.mcool}.${selected_res}.bedpe"

# Dump to make a fake pairs file
echo "cooler dump --join $input_mcool::/resolutions/$selected_res > $output_bedpe"
cooler dump -r "$CHROM" --join "$input_mcool::/resolutions/$selected_res" > "$output_bedpe"

# Convert the ginteractions file to short format with score using awk
awk -F "\t" '{print 0, $1, $2, 0, 0, $4, $5, 1, $7}' ${output_bedpe} > ${output_bedpe}.short

# Sort the short format with score file
sort --parallel=6 -k2,2d -k6,6d ${output_bedpe}.short > ${output_bedpe}.short.sorted

# Convert the short format with score file to .hic using juicer pre
java -Xms100g -Xmx100g \
    -jar $juicer_tools_jar pre \
    --threads 6 \
    -r $selected_res \
    -c $CHROM \
    ${output_bedpe}.short.sorted $output_hic $chrom_sizes