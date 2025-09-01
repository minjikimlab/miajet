#!/bin/bash

#SBATCH --account=rjhryan0 
#SBATCH --partition=standard 
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g 
#SBATCH --gpus=0 
#SBATCH --profile=all
#SBATCH --time=1:00:00  
#SBATCH --mail-type=FAIL 
#SBATCH --output=slurm_out/slurm-%j.out

my_job_header

conda activate jet-env-test

python -m miajet ./demo_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic \
 --chrom "chr3" \
 --exp_type "hic" \
 --resolution 25000 \
 --jet_widths 3 60 \
 --alpha 0.1 0.05 0.01 \
 --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_demo" \
 --num_cores 4 \
 --verbose \
 --root_within 10 \
  # --folder_name "v1.0.24_final" \
  # --folder_name "v1.0.25_pp-fixed-reorient-eig0.5" \
  # --eig2_trim 0.5 \
  # --angle_trim "None" \
  # --entropy_thresh "None" \
  # --rmse "None" \
  # --corner_trim "None" \
  # --sum_cond "r" \
  # --angle_trim 0.5 \
  # --corner_trim "None" \
  # --entropy_thresh 0.75 \
  # --im_corner_vmax 100 \
  # --normalization "KR" \
  # --window_size 6000000 \
  # --data_type "oe" \
  # --rem_k_strata 1 \
  # --thresholds 0.01 0.05 \
  # --gamma 0.75 \
  # --ridge_method 1 \
  # --rotation_padding "nearest" \
  # --convolution_padding "nearest" \
  # --angle_range 80 100 \
  # --noise_consec "" \
  # --noise_alt "" \
  # --sum_cond "a-r" \
  # --agg "sum" \
  # --num_bins 10 \
  # --points_min 0 \
  # --points_max 0.04 \
  # --eps_r 0.0005 \
  # --eps_c1 0.1 \
  # --eps_c2 1e-5 \
  # --root_within 10 \
  # --im_vmax 99 \
  # --im_vmin 0 \
  # --im_corner_vmax 98 \
  # --im_corner_vmin 0 \
  # --corner_trim 0 \
  # --entropy_thresh 0.5 \
  # --angle_trim 0.5 \
  # --saliency_thresh 90 \
  # --ang_frac \
  # --rmse 0.01 \
  # --whiten 0.01 \