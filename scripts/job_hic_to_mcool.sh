#!/bin/bash

#SBATCH --account=minjilab0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=6
#SBATCH --mem=80g 
#SBATCH --gpus=0 
#SBATCH --time=72:00:00  
#SBATCH --mail-type=FAIL 

my_job_header

conda activate jet-env

# get the .hic path from the environment
hic="${HIC}"
if [[ -z "$hic" ]]; then
  echo "ERROR: no .hic file passed in!"
  exit 1
fi

base="${hic%.hic}"
sample="$(basename "$base")"       # e.g. GM12878_insitu-hic_4DNFI1UEG1HD
out="${base}.ice.mcool"
tmp_dir=/scratch/minjilab_root/minjilab0/sionkim/temp

echo
echo "[$(date)] Processing: $hic -> $out"

# pick highest resolution
res=$(python3 - <<PYCODE
from hicstraw import HiCFile
h = HiCFile("$hic")
print(min(h.getResolutions()))
PYCODE
)
echo "  Selected resolution: ${res} bp"

# convert .hic -> .cool
prefix="${tmp_dir}/${sample}.cool"
tmp="${tmp_dir}/${sample}_${res}.cool"
hicConvertFormat \
  -m "$hic" \
  --inputFormat hic \
  --outputFormat cool \
  --resolutions "$res" \
  --outFileName "$prefix"
echo "  Generated intermediate: $tmp"

# zoomify + ICE
cooler zoomify \
  --nproc 6 \
  --resolutions "${res}N" \
  --balance \
  -o "$out" \
  "$tmp"
echo "  Written: $out"

# cleanup
rm -f "$tmp"
echo "[$(date)] DONE: $sample"
