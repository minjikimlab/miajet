# FILE: run_fontanka_pipeline.py
#!/usr/bin/env python
import cooler
import cooltools
import bioframe as bf
import os
import subprocess
import numpy as np
import pandas as pd
from skimage.filters import threshold_li
import yaml
import argparse

def main(hic_file, data_name, genome, resolution, window_size):
    # Shared parameters
    # resolution = 50000      # 50 kb
    # window_size = int(6e6)  # 6 Mb
    base_save_dir = "/nfs/turbo/umms-minjilab/sionkim/jet_pred"
    angle_leniency_deg = 20
    num_cores = 4
    angle_leniency_rad = np.radians(angle_leniency_deg)
    fountain_threshold = 0 # Require positive correlation

    # save_dir = os.path.join(base_save_dir, data_name)
    # os.makedirs(save_dir, exist_ok=True)
    save_dir = base_save_dir

    # Load cooler at desired resolution
    clr = cooler.Cooler(f"{hic_file}::resolutions/{resolution}", mode="r")

    # Rename chromosomes to ensure they start with "chr"
    rename_dict = {
        name: name if name.startswith("chr") else f"chr{name}"
        for name in clr.chromnames
    }

    # apply the renaming in-place
    cooler.rename_chroms(clr, rename_dict)

    chromsizes = bf.fetch_chromsizes(genome)

    try:
        # Not all genomes have centromeres (e.g. ce10)
        cens = bf.fetch_centromeres(genome)

        if cens is None or cens.empty:
            raise ValueError(f"No centromeres found for genome {genome}.")
        
        # Otherwise, use the centromeres to build arms
        arms = bf.make_chromarms(chromsizes, cens)
    except ValueError:
        # Just use the chromsizes if no centromeres are available
        arms = pd.DataFrame({
            "chrom": chromsizes.index,
            "start": 0,
            "end": chromsizes.values
        })

        # Sort the dataframe to exactly match the cooler's chromnames order
        arms["chrom"] = pd.Categorical(arms["chrom"], categories=clr.chromnames, ordered=True)

        arms = arms.sort_values("chrom").reset_index(drop=True)


    # Select only chromosomes present in the cooler
    arms = arms[arms.chrom.isin(clr.chromnames)].reset_index(drop=True)

    # Trim based on cooler chrom sizes (cannot exceed)
    # This is a rare case but can happen
    chromsizes_cooler = pd.Series(dict(clr.chromsizes), name="length")
    arms["end"] = arms.apply(lambda r: min(r.end, chromsizes_cooler[r.chrom]), axis=1)

    # Overwrite the defult assignment of the "name" column
    # with genomic string coordinate
    arms["name"] = arms.apply(lambda x: f"{x.chrom}:{x.start}-{x.end}", axis=1)

    # Compute expected cis contact vector
    cvd = cooltools.expected_cis(clr=clr,
                                 view_df=arms,
                                 nproc=num_cores)

    # Save arms and expected vector for fontanka
    arms_save_path = os.path.join(save_dir, f"FONTANKA_{data_name}.arms.tsv")
    arms.to_csv(arms_save_path, sep="\t", index=False, header=False)

    cvd_save_path = os.path.join(save_dir, f"FONTANKA_{data_name}.expected.tsv")
    cvd.to_csv(cvd_save_path, sep="\t", index=False)

    # Extract snips
    snips_path = os.path.join(save_dir, f"FONTANKA_{data_name}.{resolution}.snips.npy")

    # cmd = [
    #     "conda", "run", "-n", "fontanka", # this is needed to run the command in the fontanka conda env
    #     "fontanka", "slice-windows",
    #     f"{hic_file}::resolutions/{resolution}",
    #     snips_path, # this is the output file (i.e. snips)
    #     "-W", str(window_size),
    #     "-p", f"{num_cores}", # number of cores
    #     "--view", arms_save_path,
    #     "--expected", cvd_save_path,
    # ]
    # subprocess.run(cmd, check=True)

    # Apply binary fountain mask
    out_path = os.path.join(save_dir, f"FONTANKA_{data_name}.{resolution}.predicted.fountains.tsv")
    # mask_cmd = [
    #     "conda", "run", "-n", "fontanka",
    #     "fontanka", "apply-binary-fountain-mask",
    #     f"{hic_file}::resolutions/{resolution}",
    #     out_path,
    #     "-A", str(angle_leniency_rad),
    #     "-W", str(window_size),
    #     "-p", str(num_cores),
    #     "--snips", snips_path,
    #     "--view", arms_save_path,
    # ]
    # subprocess.run(mask_cmd, check=True)

    # New: thresholding and dropNA
    results = pd.read_csv(out_path, sep="\t", index_col=0)

    results = results.dropna()

    # We apply the same thresholding scheme as in Fontanka example notebook
    li_threshold = threshold_li(results['FS_peaks'].dropna().values)
    peak_threshold = max(fountain_threshold, li_threshold)
    print(f" Using threshold: {peak_threshold}")
    results_thresholded = results.loc[results["FS_peaks"] > peak_threshold].reset_index(drop=True)    

    # Save the thresholding scheme of Zebrafish as "weak"
    results_thresholded.to_csv(out_path.replace(".tsv", ".thresholded-weak.tsv"), sep="\t")

    # Finally, we apply the two thresholding schemes as suggested for Hi-C-like data
    # where there are more structures
    min_FS_ref = np.nanpercentile(results_thresholded["FS_peaks"], 90)
    th_noise = np.nanpercentile(results_thresholded["Scharr_box"], 75)

    results_thresholded = results_thresholded.loc[results_thresholded["FS_peaks"] > min_FS_ref].reset_index(drop=True)  
    results_thresholded = results_thresholded.loc[results_thresholded["Scharr_box"] < th_noise].reset_index(drop=True)  

    results_thresholded.to_csv(out_path.replace(".tsv", ".thresholded.tsv"), sep="\t")

    print(f"Finished processing {data_name} (genome: {genome})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hic_file", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--res", type=int, required=True)
    parser.add_argument("--win", type=int, required=True)
    args = parser.parse_args()
    main(args.hic_file, args.data_name, args.genome, args.res, args.win)
