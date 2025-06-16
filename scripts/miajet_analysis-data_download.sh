#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="/nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp"
mkdir -p "${SAVE_DIR}"

# Dependencies: wget, curl, jq

files_gsm=(
    # Guo et al. 2022 (Input controls)
    "GSM5963432_InputCD69negDPWTR1.bedGraph.gz"  # Input R1
    "GSM5963433_InputCD69negDPWTR2.bedGraph.gz"  # Input R2

    # Guo et al. 2022 (H3K27ac)
    "GSM5963434_H3K27acCD69negDPWTR1.bedGraph.gz" # H3K27ac R1
    "GSM5963435_H3K27acCD69negDPWTR2.bedGraph.gz" # H3K27ac R2

    # Guo et al. 2022 (RAD21)
    "GSM5963436_Rad21CD69negDPWTR1.bedGraph.gz"   # RAD21 R1
    "GSM5963437_Rad21CD69negDPWTR2.bedGraph.gz"   # RAD21 R2

    # Guo et al. 2022 (CTCF)
    "GSM5963438_CTCFCD69negDPWT.bedGraph.gz"      # CTCF

    # Seitan et al. 2013 (NIPBL)
    "GSM1184315_Nipbl_Tcell_filter_rmdup.SWEMBL.3.3.txt.bed.gz" # NIPBL
)

# Download GSM
echo "Downloading GSM files"
for f in "${files_gsm[@]}"; do
    gsm="${f%%_*}" # Extract GSM ID from filename
    # Construct the URL by replacing the last 3 digits with 'nnn' 
    prefix="${gsm:0:${#gsm}-3}nnn"
    url="ftp://ftp.ncbi.nlm.nih.gov/geo/samples/${prefix}/${gsm}/suppl/${f}" # the only difference to GSE is the 'samples' directory
    echo "* ${f}"
    wget -c -P "${SAVE_DIR}" "${url}"
done

