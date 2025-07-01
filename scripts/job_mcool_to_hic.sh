#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=6
#SBATCH --mem=120g 
#SBATCH --gpus=0 
#SBATCH --time=16:00:00  
#SBATCH --mail-type=FAIL 

my_job_header

conda activate jet-env

# AUTHOR: Ben B from https://www.biostars.org/p/360254/#9516679

input_mcool="${MCOOL}"

output_hic=${input_mcool%.*}.hic

chrom_sizes=/nfs/turbo/umms-minjilab/processing/genomes/${GENOME}/${GENOME}.chrom.sizes

juicer_tools_jar=/nfs/turbo/umms-minjilab/packages/juicer/CPU/juicer_tools_1.22.01.jar


# Get the resolutions stored in the .mcool file
resolutions=$(h5ls -r $input_mcool | grep -Eo 'resolutions/[0-9]+' | cut -d '/' -f 2 | sort -n | uniq)
echo $resolutions
highest_res=$(echo $resolutions | tr ' ' '\n' | head -n 1)
echo "highest resolution: $highest_res"

# # Use Cooler to write the .mcool matrix as interactions in bedpe format
output_bedpe="${input_mcool%.mcool}.${highest_res}.bedpe"
# echo "cooler dump --join $input_mcool::/resolutions/$highest_res > $output_bedpe"
# cooler dump --join "$input_mcool::/resolutions/$highest_res" > "$output_bedpe"

# # Convert the ginteractions file to short format with score using awk
# awk -F "\t" '{print 0, $1, $2, 0, 0, $4, $5, 1, $7}' ${output_bedpe} > ${output_bedpe}.short

# # Sort the short format with score file
# sort --parallel=6 -k2,2d -k6,6d ${output_bedpe}.short > ${output_bedpe}.short.sorted

# Convert the short format with score file to .hic using juicer pre
java -Xms100g -Xmx120g \
    -jar $juicer_tools_jar pre \
    --threads 6 \
    -r 1000,2000,5000,10000,20000,50000,100000,250000,500000,1000000 \
    ${output_bedpe}.short.sorted $output_hic $chrom_sizes