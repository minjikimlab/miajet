import numpy as np
from utils.processing import group_adjacent_numbers
import pandas as pd
import os
from scipy.stats import entropy
from utils.file_io import save_csv
import json
from scipy.optimize import curve_fit


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

 

def filter_ridges(df_agg, rmse, entropy_thresh, ridge_cond_type, ridge_cond_val,
                  angle_mean_type, angle_range, angle_deriv_thresh, col_mean_diff_std, verbose):
    """
    Filter ridges based on various conditions (if None then no filtering is applied)

    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe containing aggregated ridge features
    rmse : float, optional
        Root Mean Square Error of the 3rd order polynomial fit to expected scale values
    entropy_thresh : float, optional
        Normalized entropy threshold of the PMF of ridge strength values
    ridge_cond_type : str, optional
        Method of ridge condition filtering. Can be either "num_zeros" or "frac_zeros"
    ridge_cond_val : float, optional
        Value for the ridge condition filtering. 
        If `ridge_cond_type` is "num_zeros", this is the minimum number of zeros
        If `ridge_cond_type` is "frac_zeros", this is the minimum fraction of zeros
    angle_mean_type : str, optional
        Column name for the angle mean to be used in filtering (e.g. "angle_mean", "angle_imagej_mean")
    angle_range : tuple, optional
        A tuple of (lower_bound, upper_bound) for the angle mean filtering
    angle_deriv_thresh : float, optional
        Threshold for the maximum angle derivative value
    col_mean_diff_std : float, optional (deprecated)

    Returns
    -------
    df_agg : pd.DataFrame
        Filtered summary dataframe with ridges that satisfy the specified conditions
    """
    sum_rem = 0

    if rmse is not None:
        # filter based on the rmse
        rmse_satisfies = df_agg["rmse"] <= rmse
        df_agg["rmse_bool"] = rmse_satisfies
        len_df_agg = len(df_agg)
        df_agg = df_agg.loc[df_agg["rmse_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on rmse <= {rmse}: {len_df_agg} -> {len(df_agg)} (removed {len_df_agg - len(df_agg)})")
        sum_rem += len_df_agg - len(df_agg)

    if col_mean_diff_std is not None:
        # filter based on the std of the col mean diff
        col_mean_diff_satisfies = df_agg["col_mean_diff_std"] <= col_mean_diff_std
        df_agg["col_mean_diff_bool"] = col_mean_diff_satisfies
        len_df_agg = len(df_agg)
        df_agg = df_agg.loc[df_agg["col_mean_diff_bool"]].reset_index(drop=True)
        if verbose: print(f"\tFiltering based on col_mean_diff <= {col_mean_diff_std}: {len_df_agg} -> {len(df_agg)} (removed {len_df_agg - len(df_agg)})")
        sum_rem += len_df_agg - len(df_agg)

    if entropy_thresh is not None:
        entropy_satisfies = df_agg["entropy"] <= entropy_thresh
        df_agg["entropy_bool"] = entropy_satisfies
        len_df_agg = len(df_agg)
        df_agg = df_agg.loc[df_agg["entropy_bool"]].reset_index(drop=True)
        if verbose: print(f"\tEntropy threshold keeping 'normalized entropy' <= {entropy_thresh}: {len_df_agg} -> {len(df_agg)} (removed {len_df_agg - len(df_agg)})")
        sum_rem += len_df_agg - len(df_agg)

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
        df_agg = df_agg.loc[df_agg["ridge_cond_bool"]].reset_index(drop=True)
        if verbose: print(f"\tRidge condition filtering '{ridge_cond_type}' >= {ridge_cond_val}: {len_df_agg} -> {len(df_agg)} (removed {len_df_agg - len(df_agg)})")
        sum_rem += len_df_agg - len(df_agg)

    if angle_mean_type is not None:
        if angle_mean_type in df_agg.columns:
            angle_mean_satisfies = (angle_range[0] <= df_agg[angle_mean_type]) &  (df_agg[angle_mean_type] <= angle_range[1])
            df_agg["angle_mean_bool"] = angle_mean_satisfies
            len_df_agg = len(df_agg)
            df_agg = df_agg.loc[df_agg["angle_mean_bool"]].reset_index(drop=True)
            if verbose: print(f"\tAngle condition filtering {angle_range[0]} <= '{angle_mean_type}' <= {angle_range[1]}: {len_df_agg} -> {len(df_agg)}  (removed {len_df_agg - len(df_agg)})")
            sum_rem += len_df_agg - len(df_agg)
        else:
            print("`angle_mean_type` must be either 'angle_mean' or None")
            raise ValueError

    if angle_deriv_thresh is not None:
        angle_deriv_satisfies = df_agg["angle_deriv_max"] <= angle_deriv_thresh
        df_agg["angle_deriv_bool"] = angle_deriv_satisfies
        len_df_agg = len(df_agg)
        df_agg = df_agg.loc[df_agg["angle_deriv_bool"]].reset_index(drop=True)
        if verbose: print(f"\tAngle derivative filtering 'angle_deriv_max' <= {angle_deriv_thresh}: {len_df_agg} -> {len(df_agg)}  (removed {len_df_agg - len(df_agg)})")
        sum_rem += len_df_agg - len(df_agg)
    
    if verbose: 
        print(f"\tTotal ridges removed: {sum_rem}")
        print(f"\tTotal ridges remaining: {len(df_agg)}")


    return df_agg


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


def aggregate_ridge_features(group, ranking, angle_label, angle_range,
                             noise_consec_in, noise_alt_in, sum_cond, agg, 
                             num_bins, bin_size, points_min, points_max, ang_frac):
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
    angle_deriv_max = group["angle_deriv"].max()
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
        coeffs = (0, 0, 0, 0)
        rmse = 0
    else:
        expected_values = group["expected_scale"].values
        x_ev = np.arange(len(expected_values))
        coeffs = np.polyfit(x_ev, expected_values, deg=3)
        y_fit_ev = np.polyval(coeffs, x_ev)
        residuals = expected_values - y_fit_ev
        rmse = np.sqrt(np.mean(residuals**2)) / len(expected_values)
    
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
        "angle_deriv_max": angle_deriv_max,
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
        ranking: final_value
    }
    return pd.Series(result)


def generate_summary_table(df_features, ranking, angle_label, angle_range, noise_consec, noise_alt, sum_cond, 
                           agg, save_path, root, parameter_str, num_bins, bin_size, points_min, points_max, ang_frac,
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
    
    df_agg = df_features.groupby([contour_label, 's_imagej']).apply(lambda x : aggregate_ridge_features(x, 
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


