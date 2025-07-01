#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

# list all your .mcool files
mcool_list=(
  /nfs/turbo/umms-minjilab/downloaded_data/mESC_CTCF-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  /nfs/turbo/umms-minjilab/downloaded_data/mESC_RAD21-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_WAPL-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_YY1-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
)

for mcool in "${mcool_list[@]}"; do
  name="$(basename "${mcool%.mcool}")"
  sbatch --job-name="mcool2hic_${name}" --export=MCOOL="${mcool}",GENOME="mm10" job_mcool_to_hic.sh
done
