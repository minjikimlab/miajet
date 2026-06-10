#!/bin/bash

FILES=(
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1b_s-23/out_fast_bounded_jets/r2.14/test1b_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1a_s-23/out_fast_unbounded_jets/r2.14/test1a_s-23-fast_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim/5e+05_0.8_0.6_0.2_0.75_0.33_0.33_s-23/out_fast_unbounded_jets/r2.14/chromsim1-fast-unbounded_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim/5e+05_0.8_0.6_0.2_0.75_0.33_0.33_s-23/out_fast_bounded_jets/r2.14/chromsim1-fast-bounded_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1a_s-23/out_slow_unbounded_jets/r2.14/test1a_s-23-slow_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1c_s-23/out_fast_bounded_jets/r2.14/test1c_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1cii_s-23/out_fast_bounded_jets/r2.14/test1cii_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1e_s-23/out_tads/r2.14/test1e_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1eii_s-23/out_tads/r2.14/test1eii_s-23_hic_003.hic"
)

DEFAULT_Q_VAL="0.25"
DEFAULT_RESOLUTION="25000"
DEFAULT_WINDOW_SIZE="5000000"

Q_VALS=(
    1.0
    0.5
    0.25
    0.1
    0.05
    0.01
)

RESOLUTIONS=(
    100000
    50000
    25000
    10000
)
WINDOW_SIZES=(
    10000000
    7500000
    5000000
    2500000
    1000000
)

JET_WIDTHS=(
    "1 2"
    "2 3"
    "3 5"
    "5 7.5"
    "7.5 10"
    "10 15"
    "1 3"
    "2 5"
    "3 7.5"
    "5 10"
    "7.5 15"
    "1 5"
    "5 15"
    "1 15"
)

for hic in "${FILES[@]}"; do
    # Sweep q_val only.
    for q in "${Q_VALS[@]}"; do
        sbatch --export=HIC_FILE="$hic",Q_VAL="$q",RESOLUTION="$DEFAULT_RESOLUTION",WINDOW_SIZE="$DEFAULT_WINDOW_SIZE" \
          job_param_variation.sh
    done

    # Sweep resolution only.
    for res in "${RESOLUTIONS[@]}"; do
        sbatch --export=HIC_FILE="$hic",Q_VAL="$DEFAULT_Q_VAL",RESOLUTION="$res",WINDOW_SIZE="$DEFAULT_WINDOW_SIZE" \
          job_param_variation.sh
    done

    # Sweep window_size only.
    for window_size in "${WINDOW_SIZES[@]}"; do
        sbatch --export=HIC_FILE="$hic",Q_VAL="$DEFAULT_Q_VAL",RESOLUTION="$DEFAULT_RESOLUTION",WINDOW_SIZE="$window_size" \
          job_param_variation.sh
    done

    # Sweep jet_widths only (do not pass scale_range).
    for pair in "${JET_WIDTHS[@]}"; do
        read -r jet_width_low jet_width_high <<< "$pair"

        sbatch --export=HIC_FILE="$hic",Q_VAL="$DEFAULT_Q_VAL",RESOLUTION="$DEFAULT_RESOLUTION",WINDOW_SIZE="$DEFAULT_WINDOW_SIZE",JET_WIDTH_LOW="$jet_width_low",JET_WIDTH_HIGH="$jet_width_high" \
          job_param_variation.sh
    done
done