#!/usr/bin/env python
import bioframe as bf
import subprocess
import cooler
import os
import yaml
import argparse

def main(hic_file, data_name, genome, resolution, window_size):
    # Shared parameters
    # resolution = 50000      # 50 kb 
    # ext_length = 2500000  # 2.5 Mb # half of window size
    coverage_ratio = 0 # sparse region threshold (default option)
    padding_width = 2 # if 2, then 2 bin on left and 2 bin on right (total 5 bin width including center)
    # offset = 5 * resolution
    # interval_length = 5 * resolution # ensure it's ~5-fold of resolution as in example
    # max_merge_distance = 5 * resolution
    p_value = 0.05
    # HARD-CODED PARAMETER: "--extension_pixels", "10", "100", "5",
    # HARD-CODED PARAMETER: signal_noise_background = 1.3
    base_save_dir = "/nfs/turbo/umms-minjilab/sionkim/jet_pred"

    ext_length = int(window_size * 0.75) # STANDARDIZE to 0.75 of window size
    offset = 5 * resolution
    interval_length = 5 * resolution
    max_merge_distance = 5 * resolution

    print(ext_length)

    save_dir = os.path.join(base_save_dir, f"FUN_{data_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Save the chromsizes to a file to give to FUN program
    chromsizes = bf.fetch_chromsizes(genome)
    f_chromsizes = f"{save_dir}/{genome}.chrom.sizes"
    chromsizes.to_csv(f_chromsizes, sep="\t", header=False)

    # Rename chromosomes because FUN requires "1" "2" "3" not "chr1" "chr2" "chr3"
    clr = cooler.Cooler(f"{hic_file}::resolutions/{resolution}", mode="r")
    rename_dict = {c: c.lstrip("chr") for c in clr.chromnames}
    cooler.rename_chroms(clr, rename_dict)

    cmd = [
        "conda", "run", "-n", "fun-env",
        "python", "/nfs/turbo/umms-minjilab/sionkim/Fun/Fun",
        "calculate-son-score",
        f"{hic_file}::resolutions/{resolution}",
        "--out_dir", save_dir,
        "--norm", "weight",
        "--use_mean", "True",
        "--coverage_ratio", f"{coverage_ratio}",
        "--chromsize_path", f_chromsizes,
        "--ext_length", f"{ext_length}",
        "--padding_width", f"{padding_width}",
        "--offset", f"{offset}",
    ]
    subprocess.run(cmd, check=True)

    f_merged_bedgraph = f"{save_dir}/SoN_track_{resolution}/SoN_{resolution}_merged.bedgraph"

    cmd = [
        "conda", "run", "-n", "fun-env",
        "python", "/nfs/turbo/umms-minjilab/sionkim/Fun/Fun",
        "generate-summits",
        f"{hic_file}::resolutions/{resolution}",
        "--track", f_merged_bedgraph,
        "--out_dir", save_dir,
    ]
    subprocess.run(cmd, check=True)

    f_merged_summit = f"{save_dir}/SoN_summits/Summits_{resolution}_merged.bed"
    f_output = f"{save_dir}/FUN-pred_{resolution}"

    cmd = [
        "conda", "run", "-n", "fun-env",
        "python", "/nfs/turbo/umms-minjilab/sionkim/Fun/Fun",
        "find-fountains",
        f"{hic_file}::resolutions/{resolution}",
        "--ext_length", f"{ext_length}",
        "--norm", "weight",
        "--offset", f"{offset}",
        "--coverage_ratio", f"{coverage_ratio}",
        "--half_width", f"{padding_width}",
        "--region_path", f_merged_summit,
        "--extension_pixels", "10", "100", "5",
        "--interval_length", f"{interval_length}",
        "--max_merge_distance", f"{max_merge_distance}",
        "--output", f_output,
        "--p_value", f"{p_value}",
        "--signal_noise_background", "1.1", "1.2", "1.3", "1.4", "1.5",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hic_file", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--res",     type=int, required=True)
    parser.add_argument("--win",     type=int, required=True)
    args = parser.parse_args()
    main(args.hic_file, args.data_name, args.genome, args.res, args.win)
