#!/usr/bin/env bash
#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=30g
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --mail-type=NONE

# list of files to gzip
chip_files=(
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_input-R1_chipseq_Guo-2022_GSM5963432_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_input-R2_chipseq_Guo-2022_GSM5963433_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_H3K27ac-R1_chipseq_Guo-2022_GSM5963434_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_H3K27ac-R2_chipseq_Guo-2022_GSM5963435_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_RAD21-R1_chipseq_Guo-2022_GSM5963436_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_RAD21-R2_chipseq_Guo-2022_GSM5963437_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_CTCF_chipseq_Guo-2022_GSM5963438_mm9.bedGraph.gz"
  "/nfs/turbo/umms-minjilab/processing/results/SRR18396978/dp_thymocytes_NIPBL_SRR18396978_hiseq.q30.nr.sorted.bedgraph"
)

for f in "${chip_files[@]}"; do
  echo "Unzipping: $f"
  gunzip "$f"
done
