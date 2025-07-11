#!/bin/bash

#SBATCH --job-name=hic_to_mcool
#SBATCH --account=minjilab0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=6
#SBATCH --mem=80g
#SBATCH --gpus=0
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL

my_job_header

#### LOAD MODULES / SOURCE ENVIRONMENT HERE
#### /sw/pkgs/arc/python3.9-anaconda/2021.11
#### python3.9-anaconda/2021.11

# module load python3.9-anaconda/2021.11
conda activate jet-env

nproc=6

# JUST the c-elegans
hic_list=(
  /nfs/turbo/umms-minjilab/downloaded_data/c-elegans-CA1200-L2-L3-JK07-JK08_control-auxin-1hr_hic_Kim-2023_GSE188849_ce10.hic
  /nfs/turbo/umms-minjilab/downloaded_data/c-elegans-JK05-L3_SMC3-auxin-1hr_hic_Kim-2023_GSE237663_ce10.hic
  /nfs/turbo/umms-minjilab/downloaded_data/c-elegans-JK06-L3_WAPL-auxin-1hr_hic_Kim-2023_GSE237663_ce10.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_insitu-hic_4DNFI1UEG1HD.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_intacthic_ENCFF318GOM.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_cohesin-SMC1-RAD21-pooled_chiadrop_Kim-2024_4DNFI9JN3S8M_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_CTCF_chiadrop_Kim-2024_4DNFIERR7BI3_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_RNAPII_chiadrop_Kim-2024_4DNFI3ZH8UYR_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_control_chiapet_Kim-2024_GSE158897-GM19239_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_CTCF_chiapet_Kim-2024_4DNFIR5BPZ5L_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_RAD21_chiapet_Kim-2024_4DNFIV9RG6YP_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/GM12878_RNAPII_chiapet_Kim-2024_4DNFICWBQKM9_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/HCT116_RAD21-auxin-0hr_hic_Rao-2017_4DNFIP71EWXC_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/HCT116_RAD21-auxin-6hr_hic_Rao-2017_4DNFILIM6FDL_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/HCT116_RAD21-auxin-0hr_intacthic_Guckelberger-2024_ENCFF528XGK_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/HCT116_RAD21-auxin-6hr_intacthic_Guckelberger-2024_4DNF---OIA_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/K562_hic_Rao-2014_ENCFF616PUW_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/K562_intacthic_ENCODE-2023_ENCFF621AIY_hg38.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/zebrafish-embryo_sperm_hic_Wike-2021_4DNFI4P145EM_z11.hic
#   /nfs/turbo/umms-minjilab/downloaded_data/splenic-B-cell_WT_insitu-hic_Kieffer-Kwon-2018_GSE82144_mm9.hic
)

for hic in "${hic_list[@]}"; do
    base="${hic%.hic}"           
    mcool="${base}.ice.mcool"
    out="${mcool}" 
    
    echo "Appending 1 kb to ${mcool} and re-computing weights"
    cooler coarsen \
      --nproc $nproc \
      --factor 2 \
      --append \
      --out "${mcool}::/resolutions/1000" \
      "${mcool}::/resolutions/500"

    cooler balance \
      --force \
      --nproc $nproc \
      "${mcool}::/resolutions/1000"
done
