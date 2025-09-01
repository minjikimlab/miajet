#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=6
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --time=4:00:00  
#SBATCH --mail-type=FAIL 

my_job_header

conda activate jet-env

MCOOL="/nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.ice.mcool"
cooler coarsen -k 5 $MCOOL::/resolutions/5000 \
  -o $MCOOL::/resolutions/25000 --append
