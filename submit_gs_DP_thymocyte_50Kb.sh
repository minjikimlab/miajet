#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

# Define tuples for (eps_c1, eps_c2)
tuples=("0.1:0.0001" "0.1:0" "0.1:1e-5" "0.2:1e-5" "0.3:1e-5" "0.5:1e-5")

for tup in "${tuples[@]}"; do
    # Split the tuple by ':'
    IFS=":" read -r eps_c1 eps_c2 <<< "$tup"
    jobname="Guo et al. DEBUG eps_c1=${eps_c1}_eps_c2=${eps_c2}"
    sbatch --job-name="$jobname" --export=EPS_C1=${eps_c1},EPS_C2=${eps_c2},CHROM=chr14 gs_DP_thymocyte_50Kb.sbat
done