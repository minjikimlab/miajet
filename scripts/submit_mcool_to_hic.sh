#!/bin/bash

#SBATCH --account=minjilab99
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g 
#SBATCH --gpus=0 
#SBATCH --time=1:00:00  
#SBATCH --mail-type=NONE 

# list all your .mcool files
mcool_list=(
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_CTCF-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_RAD21-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_WAPL-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  # /nfs/turbo/umms-minjilab/downloaded_data/mESC_YY1-auxin-3hr_microc_Hsieh-2022_GSE178982_mm10.mcool
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512841_HiC.05_CTCFWAPL_0h.mcool # mESC DKO Liu et al. 
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512842_HiC.06_CTCFWAPL_6h.mcool # mESC DKO Liu et al.
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512843_HiC.07_CTCFWAPL_24h.mcool # mESC DKO Liu et al.
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512844_HiC.08_CTCFWAPL_96h.mcool # mESC DKO Liu et al.
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512837_HiC.01_WAPL_0h.mcool # mESC WAPL KO Liu et al.
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512838_HiC.02_WAPL_6h.mcool # mESC WAPL KO Liu et al.
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512839_HiC.03_WAPL_24h.mcool # mESC WAPL KO Liu et al
  /nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp/GSM5512840_HiC.04_WAPL_96h.mcool # mESC WAPL KO Liu et al.
)

for mcool in "${mcool_list[@]}"; do
  name="$(basename "${mcool%.mcool}")"
  sbatch --job-name="mcool2hic_${name}" --export=MCOOL="${mcool}",GENOME="mm10" job_mcool_to_hic.sh
done
