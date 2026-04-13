import numpy as np
from utils.processing import group_adjacent_numbers
from utils.plotting import convert_imagej_coord_to_numpy
from utils.scale_space import scale_to_width
from miajet.expanded_table import rect_to_square
from miajet.process_imagej import enforce_root_position
from miajet.compute_p_value import compute_test_statistic_quantities

from skimage.filters import threshold_yen, threshold_triangle

import pandas as pd
import os
import sys
from scipy.stats import entropy, ks_2samp
# from utils.file_io import save_csv
# import json
# from scipy.optimize import curve_fit
from .analyze_ridges import plot_top_k
from scipy.signal import find_peaks
from tqdm import tqdm

from multiprocessing import Pool, shared_memory
from functools import partial


def count_alternating_01(boolean_array):
    """
    Count the number of alternating 0/1 transitions in a boolean array

    Parameters
    ----------
    boolean_array : np.ndarray
        A 1D boolean array (True/False or 1/0)
    Returns
    -------
    int
        The number of transitions between True and False in the array
    """
    count = 0
    init = boolean_array[0]
    for x in boolean_array[1:]:
        if x != init:
            count += 1
        init = x

    return count if count != 0 else 1

def consecutive_true(boolean_array, min_consecutive=1):
    """
    Process a boolean array to be True only for indices where there is consecutive True for 
    at least `min_consecutive` times

    Parameters
    ----------
    boolean_array : np.ndarray
        A 1D boolean array (True/False or 1/0)
    min_consecutive : int, optional
        Minimum number of consecutive True values to consider as True
    Returns
    -------
    np.ndarray
        Updated boolean array
    """
    out_vec = np.full_like(boolean_array, False)

    consecutive_true_indices = group_adjacent_numbers(np.where(boolean_array)[0])

    for idx in consecutive_true_indices:
        if len(idx) >= min_consecutive:
            out_vec[idx] = True

    return out_vec


def simulate_filter_ridges(df_agg, df_features, rmse, entropy_thresh, c0_filter, exp_scale_deriv, exp_scale_deriv2, ridge_cond_type, ridge_cond_val,
                            angle_mean_type, angle_range, angle_deriv_thresh, col_mean_diff_std, verbose, save_path, config):
    """
    Simulates the effect of each filter. Plots the output jets (i.e. effect of each filter)
    in save_path directory
    """
    # Plot top K
    save_name = os.path.join(save_path, f"{config.root}_plot_nofilter.png")
    plot_top_k(df_agg, df_features, "all", 
                config.ranking, config.hic_file, config.chrom, config.resolution,
                config.window_size, config.normalization, config.rotation_padding,
                save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)

    sum_rem = 0

    if rmse is not None:
        # filter based on the rmse
        rmse_satisfies = df_agg["rmse"] <= rmse
        df_agg["rmse_bool"] = rmse_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["rmse_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on rmse <= {rmse}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_rmse.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)

    if c0_filter is not None:
        # filter based on the rmse
        c0_satisfies = df_agg["coeffs"].apply(lambda x : x[0] > c0_filter)
        df_agg["coeffs_bool"] = c0_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["coeffs_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on c0 filter > {c0_filter}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_c0-filter.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)

    if exp_scale_deriv is not None:
        # filter based on the exp_scale_deriv
        exp_scale_deriv_satisfies = df_agg["exp_scale_deriv"] <= exp_scale_deriv
        df_agg["exp_scale_deriv_bool"] = exp_scale_deriv_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["exp_scale_deriv_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on exp_scale_deriv <= {exp_scale_deriv}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_exp-scale-deriv.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)
        
    if exp_scale_deriv2 is not None:
        # filter based on the exp_scale_deriv2
        exp_scale_deriv2_satisfies = df_agg["exp_scale_deriv2"] <= exp_scale_deriv2
        df_agg["exp_scale_deriv2_bool"] = exp_scale_deriv2_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["exp_scale_deriv2_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on exp_scale_deriv2 <= {exp_scale_deriv2}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_exp-scale-deriv2.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)
        
    if exp_scale_deriv is not None and exp_scale_deriv2 is not None:
        # filter based on both exp_scale_deriv and exp_scale_deriv2
        exp_scale_deriv_satisfies = df_agg["exp_scale_deriv"] <= exp_scale_deriv
        exp_scale_deriv2_satisfies = df_agg["exp_scale_deriv2"] <= exp_scale_deriv2
        df_agg["exp_scale_both_bool"] = exp_scale_deriv_satisfies & exp_scale_deriv2_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["exp_scale_both_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on both exp_scale_deriv <= {exp_scale_deriv} and exp_scale_deriv2 <= {exp_scale_deriv2}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_exp-scale-both.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)
    

    if col_mean_diff_std is not None:
        # filter based on the std of the col mean diff
        col_mean_diff_satisfies = df_agg["col_mean_diff_std"] <= col_mean_diff_std
        df_agg["col_mean_diff_bool"] = col_mean_diff_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["col_mean_diff_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on col_mean_diff <= {col_mean_diff_std}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_col-mean-diff-std.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)


    if entropy_thresh is not None:
        entropy_satisfies = df_agg["entropy"] <= entropy_thresh
        df_agg["entropy_bool"] = entropy_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["entropy_bool"]].reset_index(drop=True)
        if verbose: print(f"\tEntropy threshold keeping 'normalized entropy' <= {entropy_thresh}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_entropy.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)


    if ridge_cond_type is not None:
        if ridge_cond_type == "frac_zeros":
            ridge_cond_satisfies = df_agg["ridge_cond_fraction"] >= ridge_cond_val
        elif ridge_cond_type == "num_zeros":
            ridge_cond_satisfies = df_agg["ridge_cond_num"] >= ridge_cond_val
        else:
            print("`ridge_cond_type` must be either 'num_zeros' or 'frac_zeros'")
            raise ValueError
        
        df_agg["ridge_cond_bool"] = ridge_cond_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["ridge_cond_bool"]].reset_index(drop=True)
        if verbose: print(f"\tRidge condition filtering '{ridge_cond_type}' >= {ridge_cond_val}: {len_df_agg} -> {len(df_agg_sim)} (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_ridge-cond-bool.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)


    if angle_mean_type is not None:
        if angle_mean_type in df_agg.columns:
            angle_mean_satisfies = (angle_range[0] <= df_agg[angle_mean_type]) &  (df_agg[angle_mean_type] <= angle_range[1])
            df_agg["angle_mean_bool"] = angle_mean_satisfies
            len_df_agg = len(df_agg)
            df_agg_sim = df_agg.loc[df_agg["angle_mean_bool"]].reset_index(drop=True)
            if verbose: print(f"\tAngle condition filtering {angle_range[0]} <= '{angle_mean_type}' <= {angle_range[1]}: {len_df_agg} -> {len(df_agg_sim)}  (removed {len_df_agg - len(df_agg_sim)})")
            sum_rem += len_df_agg - len(df_agg_sim)

            df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
            save_name = os.path.join(save_path, f"{config.root}_plot_angle-mean-type.png")
            plot_top_k(df_agg_sim, df_features_sim, "all", 
                    config.ranking, config.hic_file, config.chrom, config.resolution,
                    config.window_size, config.normalization, config.rotation_padding,
                    save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)
        else:
            print("`angle_mean_type` must be either 'angle_mean' or None")
            raise ValueError

    if angle_deriv_thresh is not None:
        angle_deriv_satisfies = df_agg["angle_deriv_max"] <= angle_deriv_thresh
        df_agg["angle_deriv_bool"] = angle_deriv_satisfies
        len_df_agg = len(df_agg)
        df_agg_sim = df_agg.loc[df_agg["angle_deriv_bool"]].reset_index(drop=True)
        if verbose: print(f"\tAngle derivative filtering 'angle_deriv_max' <= {angle_deriv_thresh}: {len_df_agg} -> {len(df_agg_sim)}  (removed {len_df_agg - len(df_agg_sim)})")
        sum_rem += len_df_agg - len(df_agg_sim)

        df_features_sim = df_features.merge(df_agg_sim[["Contour Number", "s_imagej"]].drop_duplicates(), on=["Contour Number", "s_imagej"], how="inner").reset_index(drop=True)
        save_name = os.path.join(save_path, f"{config.root}_plot_angle-deriv-max.png")
        plot_top_k(df_agg_sim, df_features_sim, "all", 
                   config.ranking, config.hic_file, config.chrom, config.resolution,
                   config.window_size, config.normalization, config.rotation_padding,
                   save_name, config.root, config.parameter_str, config.im_vmin, config.im_vmax)      
    
    if verbose: 
        print(f"\tTotal ridges removed: {sum_rem}")
        print(f"\tTotal ridges remaining: {len(df_agg)}")


    return df_agg



def filter_ridges(df_agg, df_features, resolution,
                  root_within=None, length=None, angle_turbulence=None, blobness=None, consistency=None, 
                  sum_consistency=None, sum_consistency_im=None,
                  ridge_strength_turbulence=None, angle_satisfied=None,
                  verbose=False):
    """
    Filter ridges based on various conditions (if None then no filtering is applied)

    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe containing aggregated ridge features

    Returns
    -------
    df_agg : pd.DataFrame
        Filtered summary dataframe with ridges that satisfy the specified conditions
    """
    mask = pd.Series(True, index=df_agg.index)
    filtered_counts = {}
    individual_masks = {}

    if root_within is not None:
        _before = int(mask.sum())
        m = df_agg["dist_diag"] < root_within * resolution
        individual_masks[f"root_within < {root_within * resolution:.3g}"] = m
        mask &= m
        filtered_counts["root_within"] = _before - int(mask.sum())
        if verbose: print(f"\troot_within filtered < {root_within * resolution}: {filtered_counts['root_within']}")

    if angle_turbulence is not None:
        _before = int(mask.sum())
        m = df_agg["angle_turbulence"] < angle_turbulence
        individual_masks[f"angle_turbulence < {angle_turbulence}"] = m
        mask &= m
        filtered_counts["angle_turbulence"] = _before - int(mask.sum())
        if verbose: print(f"\tangle_turbulence filtered < {angle_turbulence}: {filtered_counts['angle_turbulence']}")

    if blobness is not None:
        _before = int(mask.sum())
        m = df_agg["blobness"] < blobness
        individual_masks[f"blobness < {blobness}"] = m
        mask &= m
        filtered_counts["blobness"] = _before - int(mask.sum())
        if verbose: print(f"\tblobness filtered < {blobness}: {filtered_counts['blobness']}")

    if consistency is not None:
        _before = int(mask.sum())
        m = df_agg["consistency"] >= consistency
        individual_masks[f"consistency >= {consistency}"] = m
        mask &= m
        filtered_counts["consistency"] = _before - int(mask.sum())
        if verbose: print(f"\tconsistency filtered >= {consistency}: {filtered_counts['consistency']}")

    if length is not None:
        _before = int(mask.sum())
        m = df_agg["length"] >= length
        individual_masks[f"length >= {length}"] = m
        mask &= m
        filtered_counts["length"] = _before - int(mask.sum())
        if verbose: print(f"\tlength filtered >= {length}: {filtered_counts['length']}")

    if ridge_strength_turbulence is not None:
        _before = int(mask.sum())
        m = df_agg["ridge_strength_turbulence"] < ridge_strength_turbulence
        individual_masks[f"ridge_strength_turbulence < {ridge_strength_turbulence}"] = m
        mask &= m
        filtered_counts["ridge_strength_turbulence"] = _before - int(mask.sum())
        if verbose: print(f"\tridge_strength_turbulence filtered < {ridge_strength_turbulence}: {filtered_counts['ridge_strength_turbulence']}")

    if angle_satisfied is not None:
        _before = int(mask.sum())
        m = df_agg["angle_satisfied"] > angle_satisfied
        individual_masks[f"angle_satisfied > {angle_satisfied}"] = m
        mask &= m
        filtered_counts["angle_satisfied"] = _before - int(mask.sum())
        if verbose: print(f"\tangle_satisfied filtered > {angle_satisfied}: {filtered_counts['angle_satisfied']}")

    if sum_consistency is not None:
        _before = int(mask.sum())
        m = df_agg["sum_consistency"] > sum_consistency
        individual_masks[f"sum_consistency > {sum_consistency}"] = m
        mask &= m
        filtered_counts["sum_consistency"] = _before - int(mask.sum())
        if verbose: print(f"\tsum_consistency filtered > {sum_consistency}: {filtered_counts['sum_consistency']}")

    if sum_consistency_im:
        _before = int(mask.sum())
        thresh = threshold_yen(df_agg["sum_consistency_im"].values)
        m = df_agg["sum_consistency_im"] > thresh
        individual_masks[f"sum_consistency_im > {thresh:.3g} (yen)"] = m
        mask &= m
        filtered_counts["sum_consistency_im"] = _before - int(mask.sum())
        if verbose: print(f"\tsum_consistency_im filtered (yen threshold {thresh:.3g}): {filtered_counts['sum_consistency_im']}")

    _before = int(mask.sum())
    thresh = threshold_triangle(df_agg["saliency"].values)
    m = df_agg["saliency"] > thresh
    individual_masks[f"saliency > {thresh:.3g} (triangle)"] = m
    mask &= m
    filtered_counts["saliency"] = _before - int(mask.sum())
    if verbose: print(f"\tsaliency filtered (triangle threshold {thresh:.3g}): {filtered_counts['saliency']}")

    df_agg_thresholded = df_agg.loc[mask].reset_index(drop=True)

    if len(df_agg_thresholded) == 0:
        if verbose: print("All ridges filtered. Consider changing parameters.")
        sys.exit(0)

    if verbose:
        print(f"\tTotal ridges removed: {len(df_agg) - len(df_agg_thresholded)}")
        print(f"\tTotal ridges remaining: {len(df_agg_thresholded)}")

    df_features_thresholded = df_features.loc[df_features["unique_id"].isin(df_agg_thresholded["unique_id"])].reset_index(drop=True)

    return df_agg_thresholded, df_features_thresholded, individual_masks 


def parse_noise(noise_str):
    """
    Parse the noise parameter string

    Expected format:
        <prefix>-<mask>
    where <prefix> must include:
        - "alt" if alternating normalization is desired,
        - "consec" if consecutive filtering is desired,
    and <mask> is one of:
        - "a"   (angle-only),
        - "r"   (ridge-only), or
        - "a-r" (combined angle & ridge).
    
    Examples:
        "alt-a"         -> use_alt=True, use_consec=False, mask_type="a"
        "consec-a"      -> use_alt=False, use_consec=True, mask_type="a"
        "alt-consec-a"  -> use_alt=True, use_consec=True, mask_type="a"
        "alt-r"         -> use_alt=True, use_consec=False, mask_type="r"
        "consec-a-r"    -> use_alt=False, use_consec=True, mask_type="a-r"
        "alt-consec-a-r"-> use_alt=True, use_consec=True, mask_type="a-r"
    
    Returns:
        tuple: (use_alt, use_consec, mask_type)
    """
    tokens = noise_str.split("-")
    # Determine mask type:
    if len(tokens) >= 2 and tokens[-2] == "a" and tokens[-1] == "r":
        mask_type = "a-r"
        prefix_tokens = tokens[:-2]
    else:
        mask_type = tokens[-1]
        prefix_tokens = tokens[:-1]
    
    use_alt = "alt" in prefix_tokens
    use_consec = "consec" in prefix_tokens
    return use_alt, use_consec, mask_type

def parse_noise_consec(noise_consec_str):
    """
    Parse a string in the format "INTEGER-a", "INTEGER-r", or "INTEGER-a-r".
    
    Returns:
        tuple: (consec_true, mask_type)
            consec_true (int): The minimum number of consecutive True values.
            mask_type (str): The mask type ("a", "r", or "a-r").
    
    Examples:
        "3-a"   -> (3, "a")
        "3-r"   -> (3, "r")
        "3-a-r" -> (3, "a-r")
    """
    tokens = noise_consec_str.split("-")
    try:
        consec_true = int(tokens[0])
    except Exception as e:
        raise ValueError("The noise_consec string must start with an integer representing the minimum consecutive True values.") from e

    if len(tokens) == 2:
        mask_type = tokens[1]
    elif len(tokens) == 3:
        mask_type = tokens[1] + "-" + tokens[2]
    else:
        raise ValueError("Invalid noise_consec format. Expected 'INTEGER-a', 'INTEGER-r', or 'INTEGER-a-r'.")
    return consec_true, mask_type


def compute_histogram_data(points, points_min, points_max, num_bins=None, bin_size=None):
    """
    Computes histogram data (PMF and bin edges) from a numpy array, using only the data within a specified range
    
    Parameters:
    - points: NumPy array of values
    - num_bins: (Optional) Number of bins to use
                If provided (and bin_size is None), exactly num_bins equal-width bins are generated
                over the [points_min, points_max] interval
    - bin_size: (Optional) Fixed bin size. If provided, it takes precedence over num_bins.
                Bins are created from points_min to points_max with this fixed width
    - points_min: (Optional) Lower bound for the histogram range
                  If None, defaults to np.min(points)
    - points_max: (Optional) Upper bound for the histogram range
                  If None, defaults to np.max(points)
    
    Returns:
    - pmf: Array of probabilities for each bin (counts normalized to sum to 1)
    - bin_edges: Array of bin edges
    """
    if points_min is None:
        points_min = np.min(points)
    if points_max is None:
        points_max = np.max(points)
    
    if bin_size is not None:
        # Fixed bin size approach:
        bins = np.arange(points_min, points_max + bin_size, bin_size)
    elif num_bins is not None:
        # Fixed number of bins approach:
        bins = np.linspace(points_min, points_max, num_bins + 1)
    else:
        raise ValueError("Specify either `num_bins` or `bin_size`")
    
    # Counts Array: Each element corresponds to the frequency (or count) of data points falling within a specific bin
    # Bin Edges Array: This array defines the boundaries of each bin, where each bin is typically represented by an interval 
    counts, bin_edges = np.histogram(points, bins=bins)

    if np.sum(counts) > 0:

        pmf = counts / np.sum(counts)
        # NOTE: this is different from using the np.histogram density=True parameter
        # while that may give us a slightly more accurate PMF to the true
        # it won't sum to 1, which is a crucial invariant of PMF for downstream calculations like entropy

        assert np.isclose(np.sum(pmf), 1)

    else:
        # if you see an all zero PMF, then you know the range needs to be increased
        pmf = counts

    return pmf, bin_edges

def masked_abs_diff(x, N):
    # compute the per‑element diff with a zero at 0
    d = np.abs(np.diff(x, prepend=x[0]))
    # now mask out the first N values
    d[:N] = 0
    return d


def masked_abs_second_diff(x, N, edge_order=1):
    # first derivative
    d1 = np.gradient(x, 1.0, edge_order=edge_order)
    # second derivative
    d2 = np.gradient(d1, 1.0, edge_order=edge_order)
    d2 = np.abs(d2)
    d2[:N] = 0
    return d2


def peaks_with_borders(mat):
    """
    Adds relative maxima for end-points too 
    """
    n_rows, n_cols = mat.shape
    out = []
    for j in range(n_cols):
        col = mat[:, j]
        peaks = list(find_peaks(col)[0])
        # manual border checks:
        if n_rows > 1:
            if col[0] > col[1]:
                peaks.insert(0, 0)
            if col[-1] > col[-2]:
                peaks.append(n_rows - 1)
        out.append(np.array(peaks, dtype=int))
    return out


def detect_conflicting_structures(local_maxima, D_curves, s_idx, conflicting_pmf, conflicting_assignment):
    mu = []
    for i in range(len(local_maxima)):
        l_maxima = local_maxima[i]

        # Subset local maxima to those before the ImageJ scale
        closest_idx = np.argmin(np.abs(l_maxima - s_idx))
        closest_s_idx = l_maxima[closest_idx]
        l_maxima = np.array([m for m in l_maxima if m <= closest_s_idx])

        if len(l_maxima) == 0:
            mu.append(0)
        elif len(l_maxima) == 1:
            mu.append(1)
        else:

            # there is conflicting structures
            Q = np.zeros(len(D_curves[:, i]))
            if conflicting_pmf == "uniform":
                for j in range(len(l_maxima)):
                    Q[l_maxima[j]] = 1 / len(l_maxima)
            elif conflicting_pmf == "ridge_strength":
                for j in range(len(l_maxima)):
                    Q[l_maxima[j]] = D_curves[:, i][l_maxima[j]]
                Q /= np.sum(Q)
            elif conflicting_pmf == "stringent":
                # Assign 0 to any position with > 1 local maxima 
                pass

            if conflicting_assignment == "entropy":
                if conflicting_pmf == "stringent":
                    raise ValueError("conflicting_pmf='stringent' is not compatible with conflicting_assignment='entropy'.")
                # Compute normalized entropy as probability of conflicting structure
                mu.append(-np.sum(Q * np.log(Q + 1e-10)) / np.log(len(l_maxima)))
            
            elif conflicting_assignment == "closest":
                # The probability of the local maxima closest to the imageJ scale
                closest_idx = np.argmin(np.abs(l_maxima - s_idx))
                mu.append(Q[l_maxima[closest_idx]])

    return np.array(mu)

# def compute_consistency(group, scale_range, conflicting_pmf, conflicting_assignment, col_sim, col_agg, adj_nondec):
#     s = group.iloc[0]["s"]
#     s_idx = np.argmin(np.abs(scale_range - s))

#     # Ridge strength at this scale
#     ridge_strength = group["ridge_strength"].values[s_idx, :] 

#     # 1. Conflicting structures
#     local_maxima = peaks_with_borders(group["ridge_strength"]) # absolute value (?)
#     mu = detect_conflicting_structures(local_maxima=local_maxima, D_curves=group["ridge_strength"].values, s_idx=s_idx,
#                                        conflicting_pmf=conflicting_pmf, conflicting_assignment=conflicting_assignment)
    
#     # 2. Fast moving scale
#     if col_sim == "derivative":
#         nu, f, f_1, f_2 = detect_rapid_scale_change(rec=rec, scale_range=scale_range, col_sim=col_sim, N=2)
#     else:
#         nu = detect_rapid_scale_change(rec=rec, scale_range=scale_range, col_sim=col_sim, N=2)
#         f, f_1, f_2 = np.zeros_like(ridge_strength), np.zeros_like(ridge_strength), np.zeros_like(ridge_strength)  
#         # no derivatives computed in this case

#     # 3. Fast changing scale intensity
#     g, g_1 = detect_rapid_intensity_change(rec, intensity_method=intensity_method, N=2)

#     # 4. Adjacent non-decreasing scale selected
#     if adj_nondec:
#         adj = detect_adjacent_nondecreasing_local_maxima(rec)
#     else:
#         adj = np.ones_like(ridge_strength, dtype=float)

#     perc_satisfied = np.mean(mu * nu * adj)

#     return perc_satisfied

def aggregate_ridge_features(group, ranking, angle_label, angle_range,
                             noise_consec_in, noise_alt_in, sum_cond, agg, 
                             num_bins, bin_size, points_min, points_max, ang_frac, 
                             scale_range):
    """
    Aggregate Ridge Features
    
    Computes statistical summaries for a ridge while applying:
      1) sum_cond as the baseline mask
        The basic conditions to sum the ridge strength
      2) optional consecutive filtering (noise_consec_in)
        Mask is true only when consecutively True a certain number of times
      3) optional alternating normalization (noise_alt_in) 
        Divides by the number of alternating True/False values in the mask

    Additionally fits a 3rd order polynomial to the expected scale values 
    and computes the RMSE of the fit

    Returns:
        pd.Series: A Series containing aggregated statistics plus one ranking column (named by 'ranking') 
                   with the final ridge strength value
    """
    # Basic Aggregations
    input_mean = group["input"].mean()
    overall_mean = group["ridge_strength"].mean()
    overall_sum = group["ridge_strength"].sum()
    angle_mean = group["angle"].mean() # all 3 angles
    angle_unwrapped_mean = group["angle_unwrapped"].mean() # all 3 angles
    angle_imagej_mean = group["angle_imagej"].mean() # all 3 angles
    # angle_deriv_max = group["angle_deriv"].max()s
    eig1_mean = group["eig1"].mean()
    eig2_mean = group["eig2"].mean()
    width_mean = group["width"].mean()
    # col_mean_diff_std = group["col_mean_diff"].std()

    
    # Define Base Masks
    if angle_range[0] > angle_range[1]:
        # case where lower bound is greater than upper bound
        # example: when you want to specify [0-10] and [170, 180], you give it lb=170 and ub=10
        # then we should OR them i.e. 
        angle_mask = (group[angle_label].values >= angle_range[0]) | (group[angle_label].values <= angle_range[1])
    else:
        angle_mask = (group[angle_label].values >= angle_range[0]) & (group[angle_label].values <= angle_range[1])

    # Not a corner!
    corner_mask = ~group["corner_condition"].values.astype(bool)

    ridge_mask = (group["ridge_condition"].values > 0)

    
    # Construct mask combinations
    ar_mask = angle_mask & ridge_mask
    ac_mask = angle_mask & corner_mask
    rc_mask = ridge_mask & corner_mask
    arc_mask = angle_mask & ridge_mask & corner_mask

    
    # 1) sum_cond Baseline Mask
    # sum_cond must be one of "a", "r", "c", "ar", "ac", "rc", or "arc".
    if sum_cond == "a":
        sum_mask = angle_mask
    elif sum_cond == "r":
        sum_mask = ridge_mask
    elif sum_cond == "c":
        sum_mask = corner_mask
    elif sum_cond == "a-r":
        sum_mask = ar_mask
    elif sum_cond == "a-c":
        sum_mask = ac_mask
    elif sum_cond == "r-c":
        sum_mask = rc_mask
    elif sum_cond == "a-r-c":
        sum_mask = arc_mask
    else:
        raise ValueError("sum_cond must be one of 'a', 'r', 'c', 'a-r', 'a-c', 'r-c', or 'a-r-c'.")

    
    # 2) Consecutive Filtering
    # If noise_consec_in is empty => no consecutive filtering
    if noise_consec_in == "":
        # The consecutive filtering mask is effectively empty or no-op.
        # So we take only sum_mask as the final mask for now.
        final_mask = sum_mask
    else:
        # Parse "INTEGER-mask_type" => e.g. "3-a"
        num_consec_true, mask_type_consec = parse_noise_consec(noise_consec_in)
        if mask_type_consec == "a":
            base_mask_consec = angle_mask
        elif mask_type_consec == "r":
            base_mask_consec = ridge_mask
        elif mask_type_consec == "c":
            base_mask_consec = corner_mask
        elif mask_type_consec == "a-r":
            base_mask_consec = ar_mask
        elif mask_type_consec == "a-c":
            base_mask_consec = ac_mask
        elif mask_type_consec == "r-c":
            base_mask_consec = rc_mask
        elif mask_type_consec == "a-r-c":
            base_mask_consec = arc_mask
        else:
            raise ValueError("Invalid noise_consec value. Must be one of 'a', 'r', 'c', 'a-r', 'a-c', 'r-c', 'a-r-c'.")


        consec_mask = consecutive_true(base_mask_consec, num_consec_true)
        # The final mask is the union of sum_mask and consec_mask
        final_mask = np.logical_or(sum_mask, consec_mask)

    
    # 3) Alternating Normalization
    # If noise_alt_in is empty => alt_count = 1 => no normalization
    if noise_alt_in == "":
        alt_count = 1
    else:
        # Choose the base mask for counting alternating transitions
        if noise_alt_in == "a":
            base_mask_alt = angle_mask
        elif noise_alt_in == "r":
            base_mask_alt = ridge_mask
        elif noise_alt_in == "c":
            base_mask_alt = corner_mask
        elif noise_alt_in == "a-r":
            base_mask_alt = ar_mask
        elif noise_alt_in == "a-c":
            base_mask_alt = ac_mask
        elif noise_alt_in == "r-c":
            base_mask_alt = rc_mask
        elif noise_alt_in == "a-r-c":
            base_mask_alt = arc_mask
        else:
            raise ValueError("Invalid noise_alt value. Must be '', 'a', 'r', 'c', 'a-r', 'a-c', 'r-c', or 'a-r-c'.")

    # Debug
    # if group.name[0] == 18 and np.round(group.name[1], 3) == 17.086:
    #     pass

    # Aggregation Helper
    def compute_agg(mask, ang_frac, normalize=False, alt_factor=1):
        idx = np.where(mask)[0]
        if idx.size > 0:

            saliency_values = group["ridge_strength"].values
            if ang_frac:
                # Compute the fraction of the angle condition satisfied across scale space
                saliency_values *= group["angle_fraction"].values 

            saliency_values = saliency_values[mask]
            if agg == "mean":
                val = np.mean(saliency_values)
            else:
                val = np.sum(saliency_values)
            return val / alt_factor if normalize else val
        return 0

    
    # Final Aggregation
    # We always apply normalizing by alt_count, but alt_count=1 if not using alt.
    final_value = compute_agg(final_mask, ang_frac=ang_frac, normalize=True, alt_factor=alt_count)

    
    # Ridge Condition Stats
    ridge_cond_values = group["ridge_condition"].values[ridge_mask]
    ridge_cond_mean = ridge_cond_values.mean() if ridge_cond_values.size > 0 else 0
    ridge_cond_num = np.sum(ridge_mask)
    ridge_cond_fraction = ridge_cond_num / len(group)

    
    # Corner Condition Stats
    corner_cond_num = np.sum(corner_mask)
    corner_cond_fraction = corner_cond_num / len(group)
    
    # Compute normalized histogram (i.e. PMF) from "col_ridge_mean"
    if num_bins is not None and bin_size is not None:
        print("\t`num_bins` and `bin_size` cannot both be specified. Only specify one (for ranking)")

    if num_bins is None and bin_size is None:
        # No entropy business
        pmf, edges, H = np.nan, np.nan, np.nan

    else:
        # Assume that we want to do some kind of entropy plotting
        if num_bins is not None:
            pmf, edges = compute_histogram_data(group["col_ridge_mean"].values, num_bins=num_bins, points_min=points_min, points_max=points_max)
            # pmf, edges = compute_histogram_data(group["col_mean_diff"].values, num_bins=num_bins, points_min=points_min, points_max=points_max)
            # pmf, edges = compute_histogram_data(group["col_scale_diff"].values[:-1], num_bins=num_bins, points_min=points_min, points_max=points_max)
        else:
            pmf, edges = compute_histogram_data(group["col_ridge_mean"].values, bin_size=bin_size, points_min=points_min, points_max=points_max)
            # pmf, edges = compute_histogram_data(group["col_mean_diff"].values, num_bins=num_bins, points_min=points_min, points_max=points_max)
            # pmf, edges = compute_histogram_data(group["col_scale_diff"].values[:-1], num_bins=num_bins, points_min=points_min, points_max=points_max) 

        # Compute normalized entropy of the PMF in [0, 1] range
        if np.any(pmf > 0):
            H = entropy(pmf, base=2) / np.log2(pmf.shape[0])

            if pmf.shape[0] == 1:
                # if PMF is delta function with only one state, then set entropy to 0 manually
                H = 0

            assert (H >= 0) & (H <= 1) | np.isclose(H, 1) | np.isclose(H, 0)
        else:
            H = np.nan

    # Fit 3rd order polynomial and save parameters and RMSE only for now
    if len(group) <= 4:
        # then too small
        # simply assign any polynomial
        coeffs = [0, 0, 0, 0]
        rmse = 0
    else:
        expected_values = group["expected_scale"].values
        x_ev = np.arange(len(expected_values))
        coeffs = np.polyfit(x_ev, expected_values, deg=3)
        y_fit_ev = np.polyval(coeffs, x_ev)
        residuals = expected_values - y_fit_ev
        rmse = np.sqrt(np.mean(residuals ** 2)) / len(expected_values) # normalized RMSE

    # exp_scale_deriv = masked_abs_diff(group["expected_scale"].values, N=2) # ignore the first two positions
    # exp_scale_deriv = np.max(exp_scale_deriv) # take the maximum derivative value

    # exp_scale_deriv2 = masked_abs_second_diff(group["expected_scale"].values, N=2) # ignore the first two positions
    # exp_scale_deriv2 = np.max(exp_scale_deriv2) # take the maximum

    # consistency = compute_consistency(group, scale_range, conflicting_pmf, conflicting_assignment, col_sim, col_agg, adj_nondec)
    
    # Build Result
    result = {
        "s": group.iloc[0]["s"],
        "input_mean": input_mean,
        "ridge_strength_mean": overall_mean,
        "ridge_strength_sum": overall_sum,
        "eig1_mean": eig1_mean,
        "eig2_mean": eig2_mean,
        "angle_mean": angle_mean,
        "angle_unwrapped_mean": angle_unwrapped_mean,
        "angle_imagej_mean": angle_imagej_mean,
        # "angle_deriv_max": angle_deriv_max,
        "ridge_cond_mean": ridge_cond_mean,
        "ridge_cond_fraction": ridge_cond_fraction,
        "ridge_cond_num": ridge_cond_num,
        "corner_cond_fraction": corner_cond_fraction,
        "corner_cond_num": corner_cond_num,
        "entropy": H, 
        "pmf": pmf,
        "edges": edges,
        "width_mean": width_mean,
        # "col_mean_diff_std" : col_mean_diff_std, # deprecated
        "direction": 0,
        "rmse": rmse,
        "coeffs": coeffs, # parameters of RMSE
        # "exp_scale_deriv": exp_scale_deriv,
        # "exp_scale_deriv2": exp_scale_deriv2,
        ranking: final_value
    }
    return pd.Series(result)


def generate_summary_table(df_features, ranking, angle_label, angle_range, noise_consec, noise_alt, sum_cond, 
                           agg, save_path, root, parameter_str, num_bins, bin_size, points_min, points_max, ang_frac, scale_range,
                           verbose, contour_label="Contour Number"):
    
    """
    Generate a summary table of aggregated ridge features

    This function groups the input feature DataFrame by the unique identifier:
        ['Contour Number', 's_imagej']
    and aggregates the features. 

    Parameters
    ----------
    df_features : pd.DataFrame
        Expanded table containing ridge features
    ranking : str
        Column name for the final jet strength value (should be 'jet_saliency')
    angle_label : str
        Column name for the angle values to be used (e.g. "angle", "angle_imagej", "angle_unwrapped")
        in the aggregation function for the jet saliency score
    angle_range : list
        Upper and lower bound of the valid angle range
    noise_consec : str
        A string in the format "INTEGER-CONDITION"
        where "INTEGER" is the number of consecutive True values required to be considered True
        and "CONDITION" is 
        * "a" for angle mask
        * "r" for ridge mask
        * "c" for corner mask
        * "a-r" for both angle and ridge masks
        * "a-c" for both angle and corner masks
        * "r-c" for both ridge and corner masks
        * "a-r-c" for all three masks combined

        Example: noise_consec="2-a-r" 
    noise_alt : str
        Specifies which boolean mask to use for alternating normalization (dividing by the number of alternating True/False)
        Can be one of:
        * "a" for angle mask
        * "r" for ridge mask
        * "c" for corner mask
        * "a-r" for both angle and ridge masks
        * "a-c" for both angle and corner masks
        * "r-c" for both ridge and corner masks
        * "a-r-c" for all three masks combined
    sum_cond : str
        Specifies the which conditions need to be true to sum or average the ridge strength
        Can be one of:
        * "a" for angle mask
        * "r" for ridge mask
        * "c" for corner mask
        * "a-r" for both angle and ridge masks
        * "a-c" for both angle and corner masks
        * "r-c" for both ridge and corner masks
        * "a-r-c" for all three masks combined
    agg : str
        Aggregation function to use for the final ridge strength value
        Can be either "sum" or "mean".
    save_path : str (deprecated)
    root : str (deprecated)
    ang_frac : bool
        If True, multiply the corresponding ridge strength by the fraction of the angle condition
        satisfied across scale space
    
    Returns
    -------
    pd.DataFrame
        A summary dataframe of aggregated features with a ranking column
    """
    save_name = os.path.join(save_path, f"{root}_summary_table.csv")
    # if os.path.exists(save_name):
    #     if verbose: print("\tSummary table already exists. Skipping...")

    #     # need to convert the json to a numpy array
    #     df_read = pd.read_csv(save_name, index_col=False, comment="#",
    #                           converters={"pmf" : json.loads, "edges" : json.loads, "coeffs" : json.loads})
    #     for col in ["pmf", "edges", "coeffs"]:
    #         df_read[col] = df_read[col].apply(np.array)

    #     return df_read
    
    if verbose:
        if angle_range[0] > angle_range[1]:
            print(f"\tNOTE: angle lb {angle_range[0]} > ub {angle_range[1]}")
            print(f"\tWill be interpreted as the following range: [{angle_range[0]}, 180] U [0, {angle_range[1]}]")
    
    df_agg = df_features.groupby([contour_label, 's_imagej'], sort=False).apply(lambda x : aggregate_ridge_features(x, 
                                                                                                        ranking=ranking,
                                                                                                        angle_label=angle_label,
                                                                                                        angle_range=angle_range, 
                                                                                                        noise_consec_in=noise_consec,
                                                                                                        noise_alt_in=noise_alt, 
                                                                                                        sum_cond=sum_cond,
                                                                                                        agg=agg, 
                                                                                                        num_bins=num_bins, 
                                                                                                        bin_size=bin_size,
                                                                                                        points_min=points_min,
                                                                                                        points_max=points_max,
                                                                                                        ang_frac=ang_frac,
                                                                                                        scale_range=scale_range,
                                                                                                        ), 
                                                                                                        include_groups=False).reset_index()
    
    
    if verbose:
        len_before_drop = len(df_agg)
        df_agg.drop_duplicates([contour_label, "s_imagej"], inplace=True)
        print(f"\tNumber of rows in expanded table: {len(df_features)}")
        print(f"\tNumber of rows in summary table (possibly non-unique): {len_before_drop}")
        print(f"\tNumber of rows in summary table (unique) i.e. number of ridges : {len(df_agg)}")
    else:
        df_agg.drop_duplicates([contour_label, "s_imagej"], inplace=True)

    df_agg.reset_index(drop=True, inplace=True)

    # Save entropy data as a .csv file (deprecated)
    # if verbose:
    #     print(f"\tGenerating histogram bins from range: [{points_min}, {points_max}]")
    #     verb_max = df_features["col_ridge_mean"].max()
    #     verb_min = df_features["col_ridge_mean"].min()
    #     verb_mean = df_features["col_ridge_mean"].mean()
    #     verb_med = df_features["col_ridge_mean"].median()
    #     verb_75 = np.percentile(df_features["col_ridge_mean"], q=75)

    #     print(f"\tTrue range data statistics: [{verb_min:3f}, {verb_max:.3f}] mean: {verb_mean:.3f} median: {verb_med:.3f} 75th: {verb_75:.3f}")

    # if num_bins is not None:
    #     save_name_entropy = os.path.join(save_path, f"{root}_entropy_{points_min}-{points_max}_nbins-{num_bins}.csv")
    # else:
    #     save_name_entropy = os.path.join(save_path, f"{root}_entropy_{points_min}-{points_max}_bin_size-{bin_size}.csv")

    # save_csv(df_agg[[contour_label, "s_imagej", "entropy", "pmf"]],
    #          save_name=save_name_entropy, root=root, parameter_str=parameter_str,
    #          convert_json=["pmf"])

    # save
    # COMMENTED OUT: save the summary table after filtering
    # save_csv(df_agg, save_name, root, parameter_str, convert_json=["pmf", "edges", "coeffs"])

    return df_agg



def detect_adjacent_nondecreasing_local_maxima_memory(dbscan_s_idx, protection=False, N=None, use_memory=True):
    # Find adjacent non-decreasing local maxima with memory
    adjacent_nondecreasing = []
    memory = 1  # Initialize memory to 1 (optimistic)
    curr_max = -np.inf
    # 15% of ridge length is the max memory span before we simply append 1
    # The reasoning behind this is that because this is a discrete signal
    # A true "flat line" should be considered as optimistic non-decreasing
    # whereas a gradual decrease with some fake "flat lines" should be considered as decreasing
    mem_limit = np.ceil(len(dbscan_s_idx) * 0.1) # Changed from 0.15 to 0.1
    mem_counter = 0

    for i in range(1, len(dbscan_s_idx)):
        if dbscan_s_idx[i] > dbscan_s_idx[i - 1]:
            # Strict increase -> output 1, set memory to 1
            adjacent_nondecreasing.append(1)
            memory = 1
            mem_counter = 0 # Reset memory counter

        elif curr_max != -np.inf and np.abs(dbscan_s_idx[i] - curr_max) <= 3:
            # If its in the protected zone, treat it as non-decreasing
            adjacent_nondecreasing.append(1)
            memory = 1
            mem_counter = 0 # Reset memory counter

        elif dbscan_s_idx[i] < dbscan_s_idx[i - 1]:
            # Strict decrease -> output 0, set memory to 0
            adjacent_nondecreasing.append(0)
            memory = 0
            mem_counter = 0 # Reset memory counter

        else:
            if use_memory and mem_counter <= mem_limit:
                # Equal -> use memory
                adjacent_nondecreasing.append(memory)
                mem_counter += 1
            else:
                adjacent_nondecreasing.append(1)  # treat as increasing

    # Ensure same size as closest_s by prepending a 0
    adjacent_nondecreasing = [1] + adjacent_nondecreasing

    return np.array(adjacent_nondecreasing)



def compute_saliency(df_pos, ridge_datum, im_p_val=None, im_p_val2=None,
                     agg_pval="mean", scale_range=None, adj_nondec=True, ang_frac=True,
                     x_label="X_(px)_unmap", y_label="Y_(px)_unmap"):

    gb = df_pos.groupby('unique_id', sort=False)

    frames = []

    for uid, df_ridge in tqdm(gb, desc="Computing saliency", total=gb.ngroups):
        if uid not in ridge_datum:
            continue

        rec = ridge_datum[uid]
        s_idx = int(rec["s_idx"])
        D_curves = rec["D_curves"]
        A_bool = rec["A_bool_curves"]
        dbscan_s_idx = rec["dbscan_s_idx"]
        n_pos = D_curves.shape[1]

        # --- ridge_strength at selected scale ---
        ridge_strength = D_curves[s_idx, np.arange(n_pos)]

        # --- angle mask (sum_cond="a" only) ---
        angle_mask = A_bool[s_idx, np.arange(n_pos)]

        # --- angle_fraction (ang_frac) ---
        angle_fraction = A_bool.mean(axis=0).astype(float) if ang_frac else 1.0

        # --- adj (adjacent non-decreasing) ---
        if adj_nondec:
            adj = detect_adjacent_nondecreasing_local_maxima_memory(dbscan_s_idx, protection=False, N=3, use_memory=True)
            # adj = detect_adjacent_nondecreasing_local_maxima_memory(dbscan_s_idx, protection=False, N=3, use_memory=False)
        else:
            adj = np.ones(n_pos, dtype=float)

        # --- saliency ---
        saliency_values = ridge_strength * adj * angle_fraction
        jet_saliency = float(np.sum(saliency_values[angle_mask]))

        # --- consistency ---
        perc_satisfied = np.mean(adj)
        sum_consistency = np.sum(adj)
        sum_consistency_im = np.sum(adj * df_ridge["input"].values)

        # --- angle_satisfied (jetness) ---
        jetness = np.mean(angle_mask)

        # --- length ---
        length = len(df_ridge)

        # --- blobness ---
        dbscan_widths = scale_to_width(scale_range[dbscan_s_idx])
        blobness = np.max(dbscan_widths) / length

        # --- angle turbulence (A_curves_cv) ---
        A_curves = rec["A_curves"]
        A_curves_cv = np.std(A_curves) / (np.mean(A_curves) + 1e-6)

        # --- cv_ddbscan ---
        dbscan_D_values = np.array([D_curves[dbscan_s_idx[i], i] for i in range(len(dbscan_s_idx))])
        cv_ddbscan = np.std(dbscan_D_values) / (np.mean(dbscan_D_values) + 1e-8)

        # --- enrichment p-values ---
        enrich_p_val = np.nan
        enrich_p_val2 = np.nan

        if im_p_val is not None:
            ridge_pts = convert_imagej_coord_to_numpy(
                df_ridge[[x_label, y_label]].values,
                im_p_val.shape[0], flip_y=False, start_bin=0)
            A_curves = rec["A_curves"]
            dbscan_angle_values = np.array([A_curves[dbscan_s_idx[i], i] for i in range(len(dbscan_s_idx))])
            ridge_angles = -dbscan_angle_values + 90
            ridge_widths = scale_to_width(scale_range[dbscan_s_idx])

            l_mean, r_mean, c_mean, \
                _, _, _, _, _, _, _, _, _, \
                l_med, r_med, c_med = compute_test_statistic_quantities(
                    im=im_p_val, ridge_points=ridge_pts, ridge_angles=ridge_angles,
                    width_in=ridge_widths, height=1, im_shape=im_p_val.shape, factor_lr=1)

            if agg_pval == "mean":
                c_vals, b_vals = c_mean, np.sqrt(l_mean * r_mean)
            else:
                c_vals, b_vals = c_med, np.sqrt(l_med * r_med)

            if len(c_vals) > 0 and len(b_vals) > 0:
                _, enrich_p_val = ks_2samp(c_vals, b_vals, nan_policy="omit", alternative="less")

            if im_p_val2 is not None:
                l_mean2, r_mean2, c_mean2, \
                    _, _, _, _, _, _, _, _, _, \
                    l_med2, r_med2, c_med2 = compute_test_statistic_quantities(
                        im=im_p_val2, ridge_points=ridge_pts, ridge_angles=ridge_angles,
                        width_in=ridge_widths, height=1, im_shape=im_p_val2.shape, factor_lr=1)

                if agg_pval == "mean":
                    c_vals2, b_vals2 = c_mean2, np.sqrt(l_mean2 * r_mean2)
                else:
                    c_vals2, b_vals2 = c_med2, np.sqrt(l_med2 * r_med2)

                if len(c_vals2) > 0 and len(b_vals2) > 0:
                    _, enrich_p_val2 = ks_2samp(c_vals2, b_vals2, nan_policy="omit", alternative="less")

        
        df_ridge = df_ridge.copy()
        df_ridge["length"] = length
        df_ridge["saliency"] = jet_saliency
        df_ridge["angle_turbulence"] = A_curves_cv
        df_ridge["consistency"] = perc_satisfied
        df_ridge["sum_consistency"] = sum_consistency
        df_ridge["sum_consistency_im"] = sum_consistency_im
        df_ridge["blobness"] = blobness
        df_ridge["p-val"] = enrich_p_val
        df_ridge["p-val_white"] = enrich_p_val2
        df_ridge["ridge_strength_turbulence"] = cv_ddbscan
        df_ridge["angle_satisfied"] = jetness

        frames.append(df_ridge)

    df_features = pd.concat(frames).reset_index(drop=True)

    # Build df_agg as one row per ridge
    # df_agg = df_features.groupby("unique_id", sort=False).first().reset_index()
    # Instead of selecting the first row of each ridge as df_agg
    # Use the row with minimum distance to the diagonal such that it can be used to filter later on
    idx = df_features.groupby("unique_id", sort=False)["dist_diag"].idxmin()
    df_agg = df_features.loc[idx].reset_index(drop=True)

    return df_agg, df_features




def filter_dist_diag(df, df_pos, root_within, window_size, resolution,
                      square_size_original, verbose=False):

    if len(df_pos) == 0:
        if verbose: 
            print("All ridges filtered. Consider changing parameters.")
        sys.exit(0)

    # Need to recompute dist_diag since ridges are split not trimmed
    # Compute the distance diag vectorized
    window_size_bin = np.ceil(window_size / resolution).astype(int)
    # Make sure the coordinates are w.r.t the correct, original coordinates
    coords = rect_to_square(square_size_original, window_size_bin, df_pos[["Y_(px)_orig", "X_(px)_orig"]].values) 
    rows, cols = coords[:, 0], coords[:, 1]
    df_pos["x (bp)"] = cols * resolution
    df_pos["y (bp)"] = rows * resolution 
    df_pos['dist_diag'] = np.abs(df_pos['x (bp)'] - df_pos['y (bp)'])

    # df_dd = df_pos.loc[df_pos.groupby('unique_id')['dist_diag'].idxmin(), ["unique_id", "dist_diag"]].reset_index(drop=True)
    # dist_by_id = df_dd.set_index("unique_id")["dist_diag"]
    # df["dist_diag"] = df["unique_id"].map(dist_by_id)
    # df = df.reset_index(drop=True)

    # if root_within is None:
    #     return df_pos

    # n_before = len(df)
    # df = df.loc[df["dist_diag"] < root_within * resolution * np.sqrt(2)].reset_index(drop=True)
    # if verbose:
    #     print(f"\tRe-filtering by {root_within} bins to diagonal after splitting: {n_before} -> {len(df)}...")

    # df_pos = df_pos.loc[df_pos["unique_id"].isin(df["unique_id"])].reset_index(drop=True)


    window_size_bin = np.ceil(window_size / resolution).astype(int)

    df_pos, df = enforce_root_position(df_pos=df_pos, df=df, root_within=root_within, window_size_bin=window_size_bin, 
                                       verbose=verbose, keys="unique_id", y_label="Y_(px)_orig")

    if len(df_pos) == 0:
        if verbose: 
            print("All ridges filtered. Consider changing parameters.")
        sys.exit(0)


    return df_pos



def _init_worker(shm_name1, shm_shape1, shm_dtype1,
                 shm_name2, shm_shape2, shm_dtype2):
    """Attach shared memory in each worker."""
    global _im_p_val, _im_p_val2, _shm1, _shm2
    _im_p_val = _im_p_val2 = None
    _shm1 = _shm2 = None
    if shm_name1 is not None:
        _shm1 = shared_memory.SharedMemory(name=shm_name1)
        _im_p_val = np.ndarray(shm_shape1, dtype=shm_dtype1, buffer=_shm1.buf)
    if shm_name2 is not None:
        _shm2 = shared_memory.SharedMemory(name=shm_name2)
        _im_p_val2 = np.ndarray(shm_shape2, dtype=shm_dtype2, buffer=_shm2.buf)


def _process_ridge(args, agg_pval, scale_range, adj_nondec, ang_frac,
                   x_label, y_label):
    """Process a single ridge — runs in worker."""
    uid, df_ridge, rec = args
    im_p_val = globals().get('_im_p_val')
    im_p_val2 = globals().get('_im_p_val2')

    s_idx = int(rec["s_idx"])
    D_curves = rec["D_curves"]
    A_bool = rec["A_bool_curves"]
    dbscan_s_idx = rec["dbscan_s_idx"]
    n_pos = D_curves.shape[1]

    ridge_strength = D_curves[s_idx, np.arange(n_pos)]
    angle_mask = A_bool[s_idx, np.arange(n_pos)]
    angle_fraction = A_bool.mean(axis=0).astype(float) if ang_frac else 1.0

    if adj_nondec:
        adj = detect_adjacent_nondecreasing_local_maxima_memory(
            dbscan_s_idx, protection=False, N=3, use_memory=True) # CHANGED TO FALSE
    else:
        adj = np.ones(n_pos, dtype=float)

    saliency_values = ridge_strength * adj * angle_fraction
    jet_saliency = float(np.sum(saliency_values[angle_mask]))

    perc_satisfied = np.mean(adj)
    sum_consistency = np.sum(adj)
    sum_consistency_im = np.sum(adj * df_ridge["input"].values)
    jetness = np.mean(angle_mask)
    length = len(df_ridge)

    dbscan_widths = scale_to_width(scale_range[dbscan_s_idx])
    blobness = np.max(dbscan_widths) / length

    A_curves = rec["A_curves"]
    A_curves_cv = np.std(A_curves) / (np.mean(A_curves) + 1e-6)

    # dbscan_D_values = np.array([D_curves[dbscan_s_idx[i], i] for i in range(len(dbscan_s_idx))])
    dbscan_D_values = D_curves[dbscan_s_idx, np.arange(n_pos)]
    cv_ddbscan = np.std(dbscan_D_values) / (np.mean(dbscan_D_values) + 1e-8)

    enrich_p_val = np.nan
    enrich_p_val2 = np.nan

    if im_p_val is not None:
        ridge_pts = convert_imagej_coord_to_numpy(
            df_ridge[[x_label, y_label]].values,
            im_p_val.shape[0], flip_y=False, start_bin=0)
        # dbscan_angle_values = np.array([A_curves[dbscan_s_idx[i], i] for i in range(len(dbscan_s_idx))])
        dbscan_angle_values = A_curves[dbscan_s_idx, np.arange(n_pos)]
        ridge_angles = -dbscan_angle_values + 90
        # ridge_widths = scale_to_width(scale_range[dbscan_s_idx])

        l_mean, r_mean, c_mean, \
            _, _, _, _, _, _, _, _, _, \
            l_med, r_med, c_med = compute_test_statistic_quantities(
                im=im_p_val, ridge_points=ridge_pts, ridge_angles=ridge_angles,
                width_in=dbscan_widths, height=1, im_shape=im_p_val.shape, factor_lr=1)

        if agg_pval == "mean":
            c_vals, b_vals = c_mean, np.sqrt(l_mean * r_mean)
        else:
            c_vals, b_vals = c_med, np.sqrt(l_med * r_med)

        if len(c_vals) > 0 and len(b_vals) > 0:
            _, enrich_p_val = ks_2samp(c_vals, b_vals, nan_policy="omit", alternative="less")

        if im_p_val2 is not None:
            l_mean2, r_mean2, c_mean2, \
                _, _, _, _, _, _, _, _, _, \
                l_med2, r_med2, c_med2 = compute_test_statistic_quantities(
                    im=im_p_val2, ridge_points=ridge_pts, ridge_angles=ridge_angles,
                    width_in=dbscan_widths, height=1, im_shape=im_p_val2.shape, factor_lr=1)

            if agg_pval == "mean":
                c_vals2, b_vals2 = c_mean2, np.sqrt(l_mean2 * r_mean2)
            else:
                c_vals2, b_vals2 = c_med2, np.sqrt(l_med2 * r_med2)

            if len(c_vals2) > 0 and len(b_vals2) > 0:
                _, enrich_p_val2 = ks_2samp(c_vals2, b_vals2, nan_policy="omit", alternative="less")

    df_ridge = df_ridge.copy()
    df_ridge["length"] = length
    df_ridge["width"] = dbscan_widths
    df_ridge["avg_width"] = np.mean(dbscan_widths)
    df_ridge["saliency"] = jet_saliency
    df_ridge["angle_turbulence"] = A_curves_cv
    df_ridge["consistency"] = perc_satisfied
    df_ridge["sum_consistency"] = sum_consistency
    df_ridge["sum_consistency_im"] = sum_consistency_im
    df_ridge["blobness"] = blobness
    df_ridge["p-val"] = enrich_p_val
    df_ridge["p-val_white"] = enrich_p_val2
    df_ridge["ridge_strength_turbulence"] = cv_ddbscan
    df_ridge["angle_satisfied"] = jetness

    return df_ridge


def compute_saliency_parallel(df_pos, ridge_datum, im_p_val=None, im_p_val2=None,
                     agg_pval="mean", scale_range=None, adj_nondec=True, ang_frac=True,
                     x_label="X_(px)_unmap", y_label="Y_(px)_unmap", num_cores=4):

    # --- create shared memory for images ---
    shm1 = shm2 = None
    shm_name1 = shm_shape1 = shm_dtype1 = None
    shm_name2 = shm_shape2 = shm_dtype2 = None

    try:
        if im_p_val is not None:
            shm1 = shared_memory.SharedMemory(create=True, size=im_p_val.nbytes)
            buf1 = np.ndarray(im_p_val.shape, dtype=im_p_val.dtype, buffer=shm1.buf)
            buf1[:] = im_p_val
            shm_name1, shm_shape1, shm_dtype1 = shm1.name, im_p_val.shape, im_p_val.dtype

        if im_p_val2 is not None:
            shm2 = shared_memory.SharedMemory(create=True, size=im_p_val2.nbytes)
            buf2 = np.ndarray(im_p_val2.shape, dtype=im_p_val2.dtype, buffer=shm2.buf)
            buf2[:] = im_p_val2
            shm_name2, shm_shape2, shm_dtype2 = shm2.name, im_p_val2.shape, im_p_val2.dtype

        # --- build task list ---
        gb = df_pos.groupby('unique_id', sort=False)
        tasks = [(uid, df_ridge, ridge_datum[uid])
                 for uid, df_ridge in gb if uid in ridge_datum]

        worker_fn = partial(_process_ridge,
                            agg_pval=agg_pval, scale_range=scale_range,
                            adj_nondec=adj_nondec, ang_frac=ang_frac,
                            x_label=x_label, y_label=y_label)

        with Pool(num_cores,
                  initializer=_init_worker,
                  initargs=(shm_name1, shm_shape1, shm_dtype1,
                            shm_name2, shm_shape2, shm_dtype2)) as pool:
            frames = list(tqdm(pool.imap(worker_fn, tasks),
                               desc="Computing saliency", total=len(tasks)))

    finally:
        for shm in (shm1, shm2):
            if shm is not None:
                shm.close()
                shm.unlink()

    df_features = pd.concat(frames).reset_index(drop=True)
    idx = df_features.groupby("unique_id", sort=False)["dist_diag"].idxmin()
    df_agg = df_features.loc[idx].reset_index(drop=True)

    return df_agg, df_features
