#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=2
#SBATCH --mem=30g 
#SBATCH --gpus=0 
#SBATCH --time=6:00:00  
#SBATCH --mail-type=FAIL 

SAVE_DIR="/nfs/turbo/umms-minjilab/downloaded_data/miajet_data_temp"
mkdir -p "${SAVE_DIR}"

# Dependencies: wget, curl, jq

# Define data 
files_4dn=(
    # "4DNFIP71EWXC.hic" # Rao et al. 2017, HCT116 RAD21 0hr
    # "4DNFILIM6FDL.hic" # Rao et al. 2017, HCT116 RAD21 6hr
    # "4DNFI1UEG1HD.hic" # Rao et al. 2014, GM12878
    # "4DNFIR5BPZ5L.hic" # Kim et al. 2024, GM12878, ChIA-PET CTCF
    # "4DNFIV9RG6YP.hic" # Kim et al. 2024, GM12878, ChIA-PET RAD21
    # "4DNFICWBQKM9.hic" # Kim et al. 2024, GM12878, ChIA-PET RNAPII
    # "4DNFI3ZH8UYR.hic" # Kim et al. 2024, GM12878, ChIA-DROP RNAPII
    # "4DNFI9JN3S8M.hic" # Kim et al. 2024, GM12878, ChIA-DROP cohesin (RAD21 + SMC1)
    # "4DNFIERR7BI3.hic" # Kim et al. 2024, GM12878, ChIA-DROP CTCF
    # "4DNFI4P145EM.hic"  # Wike et al. 2021, Zebrafish embryo sperm, In-situ Hi-C (low cell counts), GRCz11
    # "4DNFI2R1W3YW.pairs.gz" # K562_hic_Rao-2014_ENCFF616PUW_hg38.hic Downloading due to failed .hic to .mcool conversion
)

files_encode=(
    "ENCFF528XGK.hic" # Guckelberger et al. 2024, HCT116, RAD21 0hr
    "ENCFF317OIA.hic" # Guckelberger et al. 2024, HCT116, RAD21 6hr
    # "ENCFF318GOM.hic" # ENCODE, GM12878 intact
    "ENCFF616PUW.hic" # Rao et al. 2014, K562
    "ENCFF621AIY.hic" # ENCODE, K562, intact Hi-C
    # "ENCFF808MAG.pairs.gz" # K562_intacthic_ENCODE-2023_ENCFF621AIY_hg38.hic Downloading due to failed .hic to .mcool conversion
    # "ENCFF785BPC.pairs.gz" # GM12878_intacthic_ENCFF318GOM.hic Downloading due to failed .hic to .mcool conversion
    # "ENCFF109GNA.pairs.gz" # HCT116_RAD21-auxin-0hr_intacthic_Guckelberger-2024_ENCFF528XGK_hg38.hic Downloading due to failed .hic to .mcool conversion
    # "ENCFF461RFV.pairs.gz" # HCT116_RAD21-auxin-6hr_intacthic_Guckelberger-2024_ENCFF317OIA_hg38.hic Downloading due to failed .hic to .mcool conversion
)

files_gse=(
    # "GSE158897_GM19239-NE-pooled.hic" # Kim et al. 2024, GM12878, ChIA-DROP control
    # "GSE178982_CTCF-AID_pool.mcool" # Hsieh et al. 2022, mESC, Micro-C, mm10, CTCF depletion
    # "GSE178982_RAD21-AID_pool.mcool" # Hsieh et al. 2022, mESC, Micro-C, mm10, RAD21 depletion
    # "GSE178982_WAPL-AID_pool.mcool" # Hsieh et al. 2022, mESC, Micro-C, mm10, WAPL depletion
    # "GSE178982_YY1-AID_pool.mcool"  # Hsieh et al. 2022, mESC, Micro-C, mm10, YY1 depletion
    # "GSE130275_mESC_WT_combined_2.6B.hic" # Hsieh et al. 2020, mESC, Micro-C, mm10, WT combined 2.6B
    # "GSE199059_CD69negDPWTR1R2R3R4_merged.hic" # Guo et al. 2022, DP thymocytes, Hi-C, mm9 (mm10 remapped), WT
    # "GSE199059_CD69negDPCTCFKOR1R2_merged.hic" # Guo et al. 2022, DP thymocytes, Hi-C, mm9 (mm10 remapped), CTCF KO
    # "GSE199059_CD69negDPDKOR1R2_merged.hic" # Guo et al. 2022, DP thymocytes, Hi-C, mm9 (mm10 remapped), CTCF/RAD21 DKO
    # "GSE82144_Kieffer-Kwon-2017-resting_B_cells_WT_30.hic" # Kieffer-Kwon et al. 2017, Splenic B cells, In-situ Hi-C, mm9, WT
    # "GSE188849_CA1200_auxin1hr_L2-L3_JK07_JK08_30.hic" # Morao et al. 2022 (TIR1 ctrl in Isiaka et al. 2023), Hi-C, ce10, Control
    # "GSE237663_hic_processed.tar.gz" # Kim et al. 2023, C. elegans, Hi-C, ce11, WAPL-1 depletion AND SMC-3 depletion
)

files_gsm=(
    #  "GSM3992961_65_WaplC6-0h_HiC_10000_iced.matrix.txt.gz" # Liu et al. 2021 (nat. gen.), mESC, WAPL 0hr
    #  "GSM3992962_66_WaplC6-24h_HiC_10000_iced.matrix.txt.gz" # Liu et al. 2021 (nat. gen.), mESC, WAPL 24hr
)

# Credentials (4DN)
KEYFILE="${KEYFILE:-keypairs.json}"
if [[ -f "${KEYFILE}" ]]; then
  echo "Loading credentials from ${KEYFILE}..."
  F4DN_KEY=$(jq -r '.default.key' "${KEYFILE}")
  F4DN_SECRET=$(jq -r '.default.secret' "${KEYFILE}")
else
    echo "Keyfile ${KEYFILE} not found. Please make a keypairs.json file with your 4DN credentials."
    echo "Instructions are on 4DN documentation: https://data.4dnucleome.org/help/user-guide/downloading-files"
    exit 1
    fi



# Download 4DN
echo "Downloading 4DN files"
for f in "${files_4dn[@]}"; do
    # id="${f%.hic}"

    # Accomodate for .mcool / .pairs.gz too
    id="${f%%.*}"         

    url="https://data.4dnucleome.org/files/${id}/@@download/${f}"
    echo "* ${f}"
    cd "${SAVE_DIR}"
    curl -O -L --user "${F4DN_KEY}:${F4DN_SECRET}" "${url}" 
    cd - # return to original directory
done

echo

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


echo

# Download GSE
echo "Downloading GSE files"
for f in "${files_gse[@]}"; do
    gse="${f%%_*}" # Extract GSE ID from filename
    # Construct the URL by replacing the last 3 digits with 'nnn' 
    prefix="${gse:0:${#gse}-3}nnn"
    url="ftp://ftp.ncbi.nlm.nih.gov/geo/series/${prefix}/${gse}/suppl/${f}"
    echo "* ${f}"
    wget -c -P "${SAVE_DIR}" "${url}"
done

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

