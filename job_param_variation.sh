#!/bin/bash

#SBATCH --account=minjilab99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --gpus=0
#SBATCH --profile=all
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_out/slurm-%j.out

my_job_header

conda activate jet-env-rev

if [[ -z "${HIC_FILE}" ]]; then
    echo "ERROR: HIC_FILE is required"
    exit 1
fi

Q_VAL="${Q_VAL:-0.25}"
RESOLUTION="${RESOLUTION:-25000}"
WINDOW_SIZE="${WINDOW_SIZE:-5000000}"

to_tag() {
    local value="$1"
    echo "${value//./p}"
}

hic_base=$(basename "${HIC_FILE}")
hic_base="${hic_base%.hic}"

q_tag=$(to_tag "${Q_VAL}")

if [[ -n "${JET_WIDTH_LOW}" && -n "${JET_WIDTH_HIGH}" ]]; then
    jet_width_low_tag=$(to_tag "${JET_WIDTH_LOW}")
    jet_width_high_tag=$(to_tag "${JET_WIDTH_HIGH}")
    jet_width_tag="jw${jet_width_low_tag}-${jet_width_high_tag}"
else
    jet_width_tag="jwdefault"
fi

FOLDER_NAME="${hic_base}_res${RESOLUTION}_win${WINDOW_SIZE}_q${q_tag}_${jet_width_tag}"

cmd=(
    python -m miajet "${HIC_FILE}"
    --chrom "chrS"
    --exp_type "hic"
    --resolution "${RESOLUTION}"
    --compartment "False"
    --normalization "NONE"
    --window_size "${WINDOW_SIZE}"
    --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output_v2.0.3_param_variation"
    --folder_name "${FOLDER_NAME}"
    --root_within "5"
    --q_val "${Q_VAL}"
    --num_cores 4
    --verbose
    --diagnostic_plots
)

if [[ -n "${JET_WIDTH_LOW}" && -n "${JET_WIDTH_HIGH}" ]]; then
    cmd+=(--jet_widths "${JET_WIDTH_LOW}" "${JET_WIDTH_HIGH}")
fi

"${cmd[@]}"