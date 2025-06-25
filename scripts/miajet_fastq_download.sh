#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --time=6:00:00  
#SBATCH --mail-type=FAIL 


SAVE_DIR="/nfs/turbo/umms-minjilab/processing/results/miajet_temp"
mkdir -p "${SAVE_DIR}"

# list of runs (add more SRRs here if needed)
runs=(
  SRR931713 # Seitan et al. 2013, DP Thymocytes NIPBL ChIP-seq
  SRR931714 # Seitan et al. 2013, DP Thymocytes NIPBL ChIP-seq
)

# load sratoolkit module
module load Bioinformatics
module load sratoolkit/3.1.1

# download each run as gzipped FASTQ
for run in "${runs[@]}"; do
  fasterq-dump "${run}" \
    --threads 4 \
    --progress \
    --outdir "${SAVE_DIR}"

  # compress and remove the unzipped
  gzip "${SAVE_DIR}/${run}.fastq"
done

echo "FASTQ saved in ${SAVE_DIR}"
