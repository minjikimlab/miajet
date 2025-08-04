#!/bin/bash

#SBATCH --job-name=download_data
#SBATCH --account=rjhryan0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=30g
#SBATCH --gpus=0
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL


set -euo pipefail

SAVE_DIR="/nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp"
mkdir -p "${SAVE_DIR}"

# Dependencies: wget, curl, jq

# files_gsm=(
#     # Guo et al. 2022 (Input controls)
#     "GSM5963432_InputCD69negDPWTR1.bedGraph.gz"  # Input R1
#     "GSM5963433_InputCD69negDPWTR2.bedGraph.gz"  # Input R2

#     # Guo et al. 2022 (H3K27ac)
#     "GSM5963434_H3K27acCD69negDPWTR1.bedGraph.gz" # H3K27ac R1
#     "GSM5963435_H3K27acCD69negDPWTR2.bedGraph.gz" # H3K27ac R2

#     # Guo et al. 2022 (RAD21)
#     "GSM5963436_Rad21CD69negDPWTR1.bedGraph.gz"   # RAD21 R1
#     "GSM5963437_Rad21CD69negDPWTR2.bedGraph.gz"   # RAD21 R2

#     # Guo et al. 2022 (CTCF)
#     "GSM5963438_CTCFCD69negDPWT.bedGraph.gz"      # CTCF

#     # Seitan et al. 2013 (NIPBL)
#     "GSM1184315_Nipbl_Tcell_filter_rmdup.SWEMBL.3.3.txt.bed.gz" # NIPBL
# )

# # Download GSM
# echo "Downloading GSM files"
# for f in "${files_gsm[@]}"; do
#     gsm="${f%%_*}" # Extract GSM ID from filename
#     # Construct the URL by replacing the last 3 digits with 'nnn' 
#     prefix="${gsm:0:${#gsm}-3}nnn"
#     url="ftp://ftp.ncbi.nlm.nih.gov/geo/samples/${prefix}/${gsm}/suppl/${f}" # the only difference to GSE is the 'samples' directory
#     echo "* ${f}"
#     wget -c -P "${SAVE_DIR}" "${url}"
# done




# Define data 
# files_4dn=(
#     # "4DNFIYO3H24N.txt" # HCT116_0h_repliseq_IZ_4DNFIGOMS9G7 peak
#     # "4DNFIGOMS9G7.txt" # HCT116_6h_repliseq_IZ_4DNFIGOMS9G7 peak
#     # "4DNFI84R4CIL.bw" # 4DNFI84R4CIL_HCT116_auxin6hr_ChIP_NIPBL
#     # "4DNFIS7J9B9X.bedGraph.gz" # GM12878 2-stage Late Repli-seq R1 `GM12878_Late_repliseq_R1_4DNFIS7J9B9X_hg38.bedGraph`
#     # "4DNFIT26294Y.bedGraph.gz" # GM12878 2-stage Late Repli-seq R2 `GM12878_Late_repliseq_R2_4DNFIT26294Y_hg38.bedGraph`
#     # "4DNFI6TILWWX.bedGraph.gz" # GM12878 2-stage Early Repli-seq R1 `GM12878_Early_repliseq_R1_4DNFI6TILWWX_hg38.bedGraph`
#     # "4DNFIIMJQ8NT.bedGraph.gz" # GM12878 2-stage Early Repli-seq R2 `GM12878_Early_repliseq_R2_4DNFIIMJQ8NT_hg38.bedGraph`
#     # "4DNFIVK5JOFU.bed.gz" # in-situ GM12878 (Rao 2014 4DNFI1UEG1HD): boundaries bed file `GM12878_hic_boundaries_Rao-2014_4DNFIVK5JOFU_hg38.bed`
#     # "4DNFIBMOGOZC.bw" # in-situ GM12878 (Rao 2014 4DNFI1UEG1HD): insulation score bigwig `GM12878_hic_insulation_score_Rao-2014_4DNFIBMOGOZC_hg38.bigWig`
#     # "4DNFILYQ1PAY.bw" # in-situ GM12878 (Rao 2014 4DNFI1UEG1HD): compartment score bigwig `GM12878_hic_compartment_score_Rao-2014_4DNFILYQ1PAY_hg38.bigWig`
# )

# # Credentials (4DN)
# KEYFILE="${KEYFILE:-keypairs.json}"
# if [[ -f "${KEYFILE}" ]]; then
#   echo "Loading credentials from ${KEYFILE}..."
#   F4DN_KEY=$(jq -r '.default.key' "${KEYFILE}")
#   F4DN_SECRET=$(jq -r '.default.secret' "${KEYFILE}")
# else
#     echo "Keyfile ${KEYFILE} not found. Please make a keypairs.json file with your 4DN credentials."
#     echo "Instructions are on 4DN documentation: https://data.4dnucleome.org/help/user-guide/downloading-files"
#     exit 1
#     fi


# # Download 4DN
# echo "Downloading 4DN files"
# for f in "${files_4dn[@]}"; do
#     # id="${f%.hic}"

#     # Accomodate for .mcool / .pairs.gz too
#     id="${f%%.*}"         

#     url="https://data.4dnucleome.org/files/${id}/@@download/${f}"
#     echo "* ${f}"
#     cd "${SAVE_DIR}"
#     curl -O -L --user "${F4DN_KEY}:${F4DN_SECRET}" "${url}" 
#     cd - # return to original directory
# done




files_encode=(
    "ENCFF964GSR.bed.gz" # `GM12878_hic_compartments_Rao-2014_ENCFF964GSR_hg38.bed`
    "ENCFF022ZAI.bed.gz" # `GM12878_hic_compartments_Rao-2014_ENCFF022ZAI_hg38.bed`
    "ENCFF265HXK.bed.gz" # `GM12878_hic_compartments_Rao-2014_ENCFF265HXK_hg38.bed`
)


# Download ENCODE files
echo "Downloading direct ENCODE files"
for f in "${files_encode[@]}"; do
    # acc="${f%.hic}"
    # Accomodate for pairs.gz files
    acc="${f%%.*}"
    echo "* ${f}"
    href=$(curl -s -H "Accept: application/json" \
             "https://www.encodeproject.org/files/${acc}/?format=json" \
           | jq -r '.href')
    wget -c -P "${SAVE_DIR}" "https://www.encodeproject.org${href}"
done
