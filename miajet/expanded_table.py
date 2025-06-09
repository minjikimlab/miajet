from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random

from utils.scale_space import extract_line_scale_space, create_maxima_set, round_line_scale_space, extract_angle_scale_space
from utils.plotting import convert_imagej_coord_to_numpy



# function to process each ridge
def process_ridge(args):
    """
    Helper function for `generate_expanded_table` to process each ridge in parallel
    """
    indexer, df_ridge = args
    # Use global variables initialized in each process
    global im, D, A, W1, W2, R, C, scale_range, scale_selection, x_label, y_label, contour_label, angle_label

    # for curve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL (i.e. start_bin=0)
    ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0)


    im_curves, D_curves, W1_curves, W2_curves, R_curves, C_curves = extract_line_scale_space(ridge_coords,
                                                                                             scale_space_container=[
                                                                                                np.expand_dims(im, 0),
                                                                                                D,
                                                                                                W1,
                                                                                                W2, 
                                                                                                R,
                                                                                                C
                                                                                                ])
    
    A_curves = round_line_scale_space(ridge_coords, scale_space_container=[A]) # round
    # A_curves = extract_angle_scale_space(ridge_coords, A) # angle interpolate

    # IMPORTANT: scale selection method
    # assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond = create_maxima_set(scale_range, D_curves, R_curves,
    #                                                                                                                            method=scale_selection)

    # if len(assigned_s) > 0 and not np.isnan(assigned_s).all():
    #     # Save the median scale for this ridge
    #     scale_value = np.nanpercentile(assigned_s, 50, method='lower')
    # else:
    #     scale_value = np.nan

    # 03/19/25: using ImageJ scales as scale assigned
    scale_value = indexer[1]

    # Save the data assigned for this ridge
    # Note: new naming standard
    features_data = pd.DataFrame({
        contour_label : indexer[0],
        x_label: np.tile(df_ridge[x_label], len(scale_range)),  
        y_label: np.tile(df_ridge[y_label], len(scale_range)),  
        "pos" : np.tile(np.arange(len(df_ridge)), len(scale_range)), # added to easily compute angle fraction
        "s_imagej" : indexer[1],
        "s" : np.repeat(scale_range, len(df_ridge)),
        "input" : np.tile(im_curves[0], len(scale_range)), 
        "ridge_strength" : D_curves.reshape(-1), 
        "angle_imagej" : np.tile(df_ridge[angle_label], len(scale_range)), 
        "angle" : A_curves.reshape(-1), 
        "angle_unwrapped" : angle_unwrap(A_curves).reshape(-1),
        "angle_deriv" : np.abs(angle_first_derivative_vectorized(A_curves)).reshape(-1),
        "eig1" : W1_curves.reshape(-1),
        "eig2" : W2_curves.reshape(-1), 
        "ridge_condition" : R_curves.reshape(-1),
        "col_ridge_mean" : np.tile(np.mean(D_curves, axis=0), len(scale_range)), # FOR HISTOGRAM
        # "col_mean_diff" : np.tile(adjacent_abs_mean_diffs(D_curves), len(scale_range)), 
        # "col_scale_diff" : np.tile(adj_expected_value(D_curves, scale_range), len(scale_range)), 
        "corner_condition" : C_curves.reshape(-1), 
        "expected_scale" : np.tile(expected_value(D_curves, scale_range), len(scale_range)), # FOR SIGMOID FITTING
        "width" : np.tile(df_ridge["width"], len(scale_range)),
        })

    return (indexer, scale_value, features_data)

def init_globals(im_, D_, A_, W1_, W2_, R_, C_, scale_range_, x_label_, y_label_, contour_label_, angle_label_):
    """
    Helper function for `generate_expanded_table` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im, D, A, W1, W2, R, C, scale_range, x_label, y_label, contour_label, angle_label
    im = im_
    D = D_
    A = A_
    W1 = W1_
    W2 = W2_
    R = R_
    C = C_
    scale_range = scale_range_
    x_label = x_label_
    y_label = y_label_
    contour_label = contour_label_
    angle_label = angle_label_

def angle_unwrap(angles_deg):
    """
    Unwrap angles in degrees with period of pi
    """
    angles_rad = np.radians(angles_deg)

    angles_unwrapped = np.unwrap(angles_rad, axis=1, period=np.pi)

    return np.degrees(angles_unwrapped)


def angle_first_derivative_vectorized(angles_deg):
    """
    Compute the first derivative of angles in degrees
    This is a vectorized version that works on a matrix of angles where each row is an independent angle list.
    It ensures that the unwrapping and gradient computation is done with respect to each row independently

    Parameters
    ----------
    angles_deg : np.ndarray
        A 2D array of angles in degrees, where each row represents independent set of angles
    Returns
    -------
    np.ndarray
        A 2D array of the same shape as `angles_deg`, containing the first derivative of angles in degrees
    """
    angles_rad = np.radians(angles_deg)
    angles_unwrapped = np.unwrap(angles_rad, axis=1, period=np.pi)
    return np.gradient(np.degrees(angles_unwrapped), axis=1)


def compute_angle_fraction(df_features, angle_range, expectation, contour_label="Contour Number"):
    """
    For each position along the ridge, compute the fraction of angles that satisfy the angle range
    across all scales 

    This corresponds to grouping by contour_label, s_imagej, and pos
    
    This feature was designed to filter out ridges that appeared to "ignore" 
    underlying contact map patterns. These ridges are often found in high scales and corresponds to 
    cases when at lower scales there is clearly another ridge (underlying contact map patterns)  
    that doesn't conform to these ridges. 

    Parameters
    --------
    df_features : pd.DataFrame
        Expanded table with features
    angle_range : list
        A list of two elements [lb, ub] representing the lower and upper bounds of the angle range
    expectation : bool
        If True, compute the expectation of the angles that satisfy the angle range 
        across all scales under a PMF defined by normalized ridge strength
            i.e. 
            p = strengths / strengths.sum() # normalize PMF
            frac = (p * mask).sum() # compute expectation
        If False, compute the expectation under a PMF defined by uniform distribution
    
    Returns
    -------
    np.ndarray
        An array of the same length as df_features, containing the fraction of angles that satisfy the angle range
    """
    lb, ub = angle_range

    # DEBUG MODE
    # print("\tWARNING: DEBUG MODE REMOVE THIS LINE AFTER DEBUGGING!")
    if expectation:

        out = np.zeros(len(df_features), float)

        for indexer, group in df_features.groupby([contour_label, "s_imagej", "pos"]):

            # if indexer[0] == 18 and np.round(indexer[1], 3) == 17.086:
            #     pass

            angles = group["angle"].values
            strengths = group["ridge_strength"].values

            # ensure no negative values
            min_val = np.min(strengths)
            if min_val < 0:
                strengths += np.abs(min_val) 

            # boolean mask of angles in the range
            if lb > ub:
                mask = (angles >= lb) | (angles <= ub)
            else:
                mask = (angles >= lb) & (angles <= ub)

            # take the expectation 
            total = np.sum(strengths)
            if total > 0:
                p = strengths / total
                frac = (p * mask).sum()
            else:
                frac = mask.sum() / len(mask)

            if frac > 1:
                raise ValueError(f"Fraction > 1 for group {indexer}: {frac}")

            out[group.index] = frac

        return out
    
    # otherwise, just compute the average angle fraction
    def frac(x):

        if lb > ub:
            mask = (x >= lb) | (x <= ub)
        else:
            mask = (x >= lb) & (x <= ub)

        frac = mask.sum() / len(x)

        if frac > 1:
            raise ValueError(f"Fraction > 1 for group {indexer}: {frac}")
        
        return frac

    return df_features.groupby([contour_label, "s_imagej", "pos"])["angle"].transform(frac)


def adjacent_abs_mean_diffs(D_curves_in):
    """
    Given ridge strength curves `D_curves_in` of shape (num_scales, n_cols)
    Return the absolute differences between consecutive column means

    Parameters
    ----------
    D_curves_in : np.ndarray
        Input matrix of shape (num_scales, n_cols) representing ridge strength curve
    Returns
    -------
    np.ndarray
        An array of shape (n_cols,) containing the absolute differences between consecutive column means
        The last value is appended to maintain the same length as D_curves_in
    """
    D_curves = D_curves_in.copy()

    # Ensure non-negative
    D_curves[D_curves < 0] = 0

    # Rescale
    D_curves /= np.max(D_curves)

    # Compute column means: shape (n_cols,)
    col_means = D_curves.mean(axis=0)

    # Compute adjacent differences and take absolute value: shape (n_cols-1,)
    diff = np.abs(np.diff(col_means))

    return np.append(diff, diff[-1])  # Append the last value to maintain the same length as D_curves


def adj_expected_value(pmf_matrix_in, value):
    """
    Compute the expected value of a discrete random variable
    given its probability mass function (pmf) and corresponding values.
    """
    expected_values = []

    pmf_matrix = pmf_matrix_in.copy()

    pmf_matrix[pmf_matrix < 0] = 0  # Ensure non-negative

    pmf_matrix /= np.max(pmf_matrix)  # Normalize pmf_matrix to sum to 1

    for i in range(pmf_matrix.shape[1]):
        pmf = pmf_matrix[:, i]

        pmf = pmf / np.sum(pmf)  # Normalize the pmf to sum to 1

        expected_value = np.sum(pmf * value)
        expected_values.append(expected_value)    

    expected_values = np.array(expected_values)    

    diff = np.abs(np.diff(expected_values))

    return np.append(diff, diff[-1])  # Append the last value to maintain the same length as D_curves


def expected_value(pmf_matrix_in, value):
    """
    Compute the expected value of a discrete random variable
    given its probability mass function (pmf) and corresponding values

    Parameters
    ----------
    pmf_matrix_in : np.ndarray
        Input matrix of shape (num_scales, n_cols) where 
        each column is interpreted as a pmf 
    value : np.ndarray
        An array of values corresponding to the pmf
    
    Returns
    -------
    np.ndarray
        An array of expected values for each column in pmf_matrix_in
    """
    expected_values = []

    pmf_matrix = pmf_matrix_in.copy()

    pmf_matrix[pmf_matrix < 0] = 0  # Ensure non-negative

    pmf_matrix /= np.max(pmf_matrix)  # Normalize pmf_matrix to sum to 1

    for i in range(pmf_matrix.shape[1]):
        pmf = pmf_matrix[:, i]

        pmf = pmf / np.sum(pmf)  # Normalize the pmf to sum to 1

        expected_value = np.sum(pmf * value)
        expected_values.append(expected_value) 

    return np.array(expected_values)   
    


def generate_expanded_table(im, df, df_pos, D, A, W1, W2, R, C, scale_range, angle_range, num_cores, verbose, save_path, root,
                            contour_label="Contour Number",                         
                            x_label="X_(px)", 
                            y_label="Y_(px)", 
                            angle_label="Angle_of_normal_(radians)"):
    """
    Using the ridge coordinates from imageJ expanded table and the scale space features generated 
    from the contact map (D, A, W1, W2, R, C), adds new columns (features) to the expanded table 

    These features include:
    - `input`: the image intensity value 
    - `ridge_strength`: the ridge strength
    - `angle`: the angle (should be similar to `angle_imagej`, which is the angle given by ImageJ)
    - `angle_unwrapped`: the unwrapped angle
    - `angle_deriv`: the first derivative of the angle
    - `eig1`: the first eigenvalue
    - `eig2`: the second eigenvalue
    - `ridge_condition`: the ridge condition
    - `col_ridge_mean`: the mean ridge strength across all scales
    - `col_mean_diff`: the absolute differences between consecutive ridge position means
    - `col_scale_diff`: the absolute differences between consecutive expected scales
    - `expected_scale`: the expected scale for each ridge position
    - `corner_condition`: the corner condition

    Parameters
    ----------
    im : np.ndarray
        The rotated contact map
    df : pd.DataFrame
        Summary table
    df_pos : pd.DataFrame
        Expanded table with ridge coordinates
    D, A, W1, W2, R, C : np.ndarray
        Scale space features of dimension (num_scales, height, width)
    scale_range : list
        List of scales generated
    angle_range : list
        Upper and lower bound of the valid angle range
    num_cores : int
        Number of cores to use for parallel processing
    save_path : str (deprecated)
    root : str (deprecated)

    Returns
    -------
    pd.DataFrame
        Expanded table with new features added
    """
    # save_name = os.path.join(save_path, f"{root}_expanded_table.csv")
    # if os.path.exists(save_name):
    #     if verbose: print("\tExpanded table already exists. Skipping...")
    #     return pd.read_csv(save_name, index_col=False, comment="#")

    gb = df_pos.groupby([contour_label, "s_imagej"]) # no need for chromosome selection 

    scale_assignment = dict()
    df_features = []

    # print("\tWARNING DEBUG MODE: UNCOMMENT THE FOLLOWING LINES IN `generate_expanded_table` AFTER DEBUGGING!")
    if num_cores == 1:
        # print("WARNING: NON-PARALLEL")
        for indexer, df_ridge in tqdm(gb):
            # Each ridge
            
            # for curve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL (i.e. start_bin=0)
            ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0)
        
            im_curves, D_curves, W1_curves, W2_curves, R_curves, C_curves = extract_line_scale_space(ridge_coords, 
                                                                                                     scale_space_container=[np.expand_dims(im, 0), D, W1, W2, R, C])
            
            A_curves = round_line_scale_space(ridge_coords, scale_space_container=[A]) # round
            # A_curves = extract_angle_scale_space(ridge_coords, A) # angle interpolate (takes much longer)

            # assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond = create_maxima_set(scale_range, D_curves, R_curves, 
            #                                                                                                                         method=scale_selection) 
        
            # if len(assigned_s) > 0 and not np.isnan(assigned_s).all():
            #     # Save the scale assigned for this ridge
            #     scale_assignment[indexer[0], indexer[1]] = np.nanpercentile(assigned_s, 50, method='lower')
            # else:
            #     scale_assignment[indexer[0], indexer[1]] = np.nan

            # 03/19/25: do not use median but the imageJ scale for the scale assignment
            scale_assignment[indexer[0], indexer[1]] = indexer[1]
            
            # Save the data assigned for this ridge
            # The features dataframe will be an EVEN bigger dataframe that the position dataframe
            # It will contain the postiion dataframe multiplied for each scale of scale space
            features_data = pd.DataFrame({
                contour_label : indexer[0],
                x_label: np.tile(df_ridge[x_label], len(scale_range)),  
                y_label: np.tile(df_ridge[y_label], len(scale_range)),  
                "pos" : np.tile(np.arange(len(df_ridge)), len(scale_range)), # added to easily compute angle fraction
                "s_imagej" : indexer[1],
                "s" : np.repeat(scale_range, len(df_ridge)),
                "input" : np.tile(im_curves[0], len(scale_range)), 
                "ridge_strength" : D_curves.reshape(-1), 
                "angle_imagej" : np.tile(df_ridge[angle_label], len(scale_range)), 
                "angle" : A_curves.reshape(-1), 
                "angle_unwrapped" : angle_unwrap(A_curves).reshape(-1),
                "angle_deriv" : np.abs(angle_first_derivative_vectorized(A_curves)).reshape(-1),
                "eig1" : W1_curves.reshape(-1),
                "eig2" : W2_curves.reshape(-1), 
                "ridge_condition" : R_curves.reshape(-1), 
                "col_ridge_mean" : np.tile(np.mean(D_curves, axis=0), len(scale_range)), # FOR HISTOGRAM
                # "col_mean_diff" : np.tile(adjacent_abs_mean_diffs(D_curves), len(scale_range)), # FOR HISTOGRAM
                # "col_scale_diff" : np.tile(adj_expected_value(D_curves, scale_range), len(scale_range)), # FOR HISTOGRAM
                "expected_scale" : np.tile(expected_value(D_curves, scale_range), len(scale_range)), # FOR SIGMOID FITTING
                "corner_condition" : C_curves.reshape(-1), 
                "width" : np.tile(df_ridge["width"], len(scale_range)),
                })
            
            df_features.append(features_data)

    else:
        # parallel
        initializer_args = (im, D, A, W1, W2, R, C, scale_range, x_label, y_label, contour_label, angle_label)
        # Collect the total number of groups for progress tracking
        total_groups = sum(1 for _ in gb)
        # Create a pool of worker processes
        with Pool(num_cores, initializer=init_globals, initargs=initializer_args) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_ridge, gb), total=total_groups):
                results.append(result)
        # Process the results
        for indexer, scale_value, features_data in results:
            scale_assignment[indexer[0], indexer[1]] = scale_value
            df_features.append(features_data)

    df_features = pd.concat(df_features, ignore_index=True)

    # Convert keys into a DataFrame for easy merging
    keys_df = pd.DataFrame(scale_assignment.keys(), columns=[contour_label, 's_imagej'])
    keys_df['scale_assigned'] = scale_assignment.values()

    # Merge on the 3 columns to add the new column where tuples match
    df = df.merge(keys_df, on=[contour_label, 's_imagej'], how='left')

    # We can technically skip this because the scale assigned is simply the s_imagej
    df_features = df_features.merge(df[[contour_label, "s_imagej", "scale_assigned"]], on=[contour_label, "s_imagej"], how="left")

    # Before we assign the scales, compute the angle fraction satisfying scores for each row of expanded table
    df_features["angle_fraction"] = compute_angle_fraction(df_features, expectation=False, angle_range=angle_range)

    # Now we can drop "pos" since it was used only for computing the angle fraction
    df_features.drop(["pos"], axis=1, inplace=True)

    # Now assign scales
    # df_features actually contains not only the assigned scale but features for any scale.. so we its quite useful (we can consider what if we selected other scales!)
    # So we should select the scale `s` that corresponds to the ridge, uniquely identified by both `contour_lable` and `s_imagej`
    df_features_assigned = df_features.loc[np.isclose(df_features["scale_assigned"].values, df_features["s"].values)].reset_index(drop=True)

    # The line above basically only keeps the ridges that have been assigned scales
    # There are ridges who has not been assigned scales (due to not meeting ridge condition criteria etc.) – They have np.nan in them
    # As a result, since we dropped these ridges, we must also drop them from the other objects too, df, df_pos
    # Ensure that in the future, when you merge with df_pos, keep only the rows that have a match in df_agg

    if verbose:
        print("\tNote: If you use `df` or `df_pos` in the future, ensure that you merge appropriately with")
        print("\t`df_agg` (that is `df_features` grouped by the appropriate columns) to ensure consistency with scales assigned.")


    gb = df_features_assigned.groupby([contour_label, "s_imagej"])

    sampled_ridges = random.sample(list(gb), k=min(50, len(gb))) # check just 50 ridges

    assert np.allclose(df_features["scale_assigned"].values, df_features["s_imagej"].values)

    # final check
    for key, group_features in sampled_ridges:
        # Iterate through every ridge identified and check whether the X, Y position values are consistent
        # i.e. did the merging operation collect the right data?
        # key is a tuple: (contour number, s_imagej)
        # Select all rows in df_pos corresponding to the same ridge (with the same s_imagej)
        group_pos = df_pos[(df_pos[contour_label] == key[0]) & (df_pos["s_imagej"] == key[1])]
        
        # Check if the number of rows in both groups match. 
        # They should be the same if df_features_assigned is supposed to contain all coordinates of that ridge.
        if len(group_features) != len(group_pos):
            raise ValueError(f"Group size mismatch for ridge {key}: df_features_assigned has {len(group_features)} rows, but df_pos has {len(group_pos)} rows.")

        # Compare the entire arrays of X coordinates
        if not np.allclose(group_features["X_(px)"].values, group_pos["X_(px)"].values):
            raise ValueError(f"Mismatch in X coordinates for ridge {key}!")
        
        # Compare the entire arrays of Y coordinates
        if not np.allclose(group_features["Y_(px)"].values, group_pos["Y_(px)"].values):
            raise ValueError(f"Mismatch in Y coordinates for ridge {key}!")

    return df_features_assigned


def intersect_with_true(df_features, f_true_bed, chromosome, tolerance, resolution, num_cores, verbose,
                        contour_label="Contour Number", 
                        x_label="X_(px)_orig"):

    # f_true_bed must be a bed file
    tp = pd.read_csv(f_true_bed, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
    tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
    tp["start"] -= tolerance # don't care about out of chromosome issues
    tp["end"] += tolerance # don't care about out of chromosome issues

    gb = df_features.groupby([contour_label, "s_imagej"])

    frames = []

    for indexer, df_ridge in tqdm(gb):

        genomic_pos_max = df_ridge[x_label].max() * resolution / np.sqrt(2)
        genomic_pos_min = df_ridge[x_label].min() * resolution / np.sqrt(2)

        condition_bool = (genomic_pos_max >= tp["start"].values) & (genomic_pos_min <= tp["end"].values)

        if np.any(condition_bool):
            # keep 
            frames.append(df_ridge)

    if len(frames) == 0:
        if verbose: print(f"\tNo ridges remain after merging with true (tol: {tolerance})")
        return None

    df_features_out = pd.concat(frames).reset_index(drop=True)

    return df_features_out


    

def extract_coords(coords):
    """
    Helper function for `square_to_rect` and `rect_to_square` to extract coordinates
    from a 2D array or return them as is if they are already in (i, j) format
    """

    # If coords is an Nx2 array, split it into two 1D arrays.
    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 2:
        return coords[:, 0], coords[:, 1]
    else:
        # Otherwise, assume coords is already in (i, j) format.
        return coords

def square_to_rect(N, window_size_bin, coords):
    """
    Convert indices (i, j) in the original NxN square to coordinates 
    (r_rect, c_rect) or (y positions, x positions) in the rotated rectangle (i.e. rotated contact map)
    
    Parameters
    ----------
    N : int
        The original size of the square matrix (before removing zero sum rows/columns)
    window_size_bin : int
        Binned window size
    coords : array-like, shape (n, 2)
        The coordinates in the original square matrix (i, j) or (row, column) format.
    
    Returns
    -------
    np.ndarray, shape (n, 2)
        The coordinates in the rotated rectangle (r_rect, c_rect) or (y positions, x positions)
    """
    i, j = extract_coords(coords)

    i = np.minimum(i, j)
    j = np.maximum(i, j)

    # Compute the offset (using ceil in rect_to_square, but here we use exact arithmetic for invertibility)
    offset = (N * np.sqrt(2) / 2) - (window_size_bin / np.sqrt(2))
    
    # Compute raw rotated row coordinate.
    raw_r = (j + (N - 1 - i)) / np.sqrt(2) - offset
    
    # Determine the raw row range for UT coordinates:
    raw_r_min = ((N - 1) / np.sqrt(2)) - offset      # when i == j
    raw_r_max = ((window_size_bin + (N - 1)) / np.sqrt(2)) - offset  # when i = 0, j = window_size_bin
    
    # The extracted rotated rectangle is expected to have height:
    window_size_bin_rect = np.ceil(window_size_bin / np.sqrt(2))
    
    # Scale raw_r to fit into [0, window_size_bin_rect - 1]
    scale = (window_size_bin_rect - 1) / (raw_r_max - raw_r_min)
    r_rect = (raw_r - raw_r_min) * scale
    
    # Compute rotated column coordinate (horizontal flip is built in so that the main diagonal remains invariant)
    c_rect = N * np.sqrt(2) - 1 - ((2 * N - 2 - i - j) / np.sqrt(2))

    r_rect = window_size_bin / np.sqrt(2) - r_rect - 1
    
    return np.array((r_rect, c_rect)).T


def rect_to_square(N, window_size_bin, coords):
    """
    Convert (r_rect, c_rect) or (y positions, x positions) in the rotated rectangle back to indices (i, j) in the original NxN square

    Parameters
    ----------
    N : int
        The original size of the square matrix (before removing zero sum rows/columns)
    window_size_bin : int
        Binned window size
    coords : array-like, shape (n, 2)
        The coordinates in the rotated rectangle (r_rect, c_rect) or (y positions, x positions).

    Returns
    -------
    np.ndarray, shape (n, 2)
        The reconstructed indices (i, j) in the original square matrix
    """
    r_rect, c_rect = extract_coords(coords)

    c_rect = N * np.sqrt(2) - c_rect - 1
    
    center = N * np.sqrt(2) / 2
    window_size_bin_rect = window_size_bin / np.sqrt(2)
    offset = center - window_size_bin_rect
    
    full_r = r_rect + offset
    full_c = c_rect
    
    # standard inverse rotation
    A = full_r * np.sqrt(2)
    B = full_c * np.sqrt(2) - (N - 1)
    i_ = (A - B) / 2
    j_ = (A + B) / 2

    # final flip
    j_ = N - j_ - 1

    # ensure the final (i2, j2) is upper-triangular, i.e. i2 <= j2
    i_new = np.minimum(i_, j_)
    j_new = np.maximum(i_, j_)
    return np.array((i_new, j_new)).T



def reconstruct_indices_float(compressed_indices, rm_idx, N):
    """
    Recover the original indices with indices `rm_idx` removed 
    
    Parameters:
       compressed_indices : array-like, shape (n, 2)
           The compressed indices with indices `rm_idx` removed
           Can be floating point values
       rm_idx : array-like
           The indices that were removed
       N : int
           The original NxN square size (before removals)
           
    Returns:
       original_indices : np.ndarray, shape (n, 2)
           The reconstructed original indices (as floats)
    """
    # Create the array of indices that remain after removal
    remaining = np.delete(np.arange(N), rm_idx).astype(float)
    # xp are the "compressed" positions corresponding to each remaining index
    xp = np.arange(len(remaining))
    
    # Ensure compressed_indices is a numpy array
    compressed_indices = np.asarray(compressed_indices)
    
    # Interpolate along each column
    original_r = np.interp(compressed_indices[:, 0], xp, remaining)
    original_c = np.interp(compressed_indices[:, 1], xp, remaining)
    
    return np.column_stack((original_r, original_c))


def reconstruct_indices(compressed_indices, rm_idx, N):
    """
    Similar function to `reconstruct_indices_float`, but for integer indices only

    Parameters:
         compressed_indices : array-like, shape (n, 2)
              The compressed indices with indices `rm_idx` removed
              Must be integers
         rm_idx : array-like
              The indices that were removed
         N : int
              The original NxN square size (before removals)
    
    Returns:
        original_indices : np.ndarray, shape (n, 2)
            The reconstructed original indices (as integers)
    """
    # Create an array of remaining indices (i.e. those not removed)
    remaining = np.delete(np.arange(N), rm_idx)

    max_idx = len(remaining) - 1
    # Ensure all indices in compressed_indices are within 0 and max_idx.
    compressed_indices = np.clip(compressed_indices, 0, max_idx)
    
    # Map each (r, c) pair in the compressed matrix to the original indices
    original_indices = [(remaining[r], remaining[c]) for r, c in compressed_indices]
    return np.array(original_indices)

def insert_unmapped_regions(df_features, im_orig, rm_idx, N_removed, window_size, resolution, verbose, num_cores, 
                            contour_label="Contour Number", x_label="X_(px)", y_label="Y_(px)"):
    """
    Insert unmapped regions into the x and y coordinates of the expanded table (df_features)

    Recall that ImageJ was given a rotated contact map with the zero sum rows/columns removed

    As a result, the genomic coordinates in the expanded table need to be converted back to the original coordinates

    Strategy: 
        1. Convert into square coordinates
        2. Add offsets
        3. Convert into rectangle coordinates

    Parameters
    ----------
    df_features : pd.DataFrame
        The expanded table with features
    im_orig : np.ndarray
        The contact map without the zero sum columns and rows removed
    rm_idx : list
        The indices of the removed rows/columns
    N_removed : int
        The number of bins in the Hi-C data after removing zero sum rows/columns
    window_size : int
        Size of the window in base pairs to extract from the Hi-C data
    resolution : float
        Resolution of the Hi-C data in base pairs
    num_cores : int
        Number of cores to use for parallel processing

    Returns
    -------
    pd.DataFrame
        The expanded table with:
        * Removed column `x_label` and `y_label`
        * Added columns
            * `x_label+"_orig"` and `y_label+"_orig"` representing the correct genomic coordinates
            * `x_label+"_unmap"` and `y_label+"_unmap"` representing the shifted genomic coordinates
    """
    # print("\tWARNING DEBUG MODE: UNCOMMENT THE FOLLOWING LINES IN `insert_unmapped_regions` AFTER DEBUGGING!!!")
    # if y_label+"_orig" in df_features.columns:
    #     if verbose: print("\tUnmapped coordinates already present. Skipping...")
    #     return df_features

    N = N_removed + len(rm_idx)

    row_col = df_features[[y_label, x_label]].values
    
    # first, convert from rect to square coordinates
    window_size_bin = np.ceil(window_size / resolution).astype(int)
    row_col_square = rect_to_square(N_removed, window_size_bin, row_col)

    # reconstruct original indices by adding appropriate offsets
    new_coords = reconstruct_indices_float(row_col_square, rm_idx, N)

    # finally, convert back from square to rect but now w.r.t larger, original square size
    row_col_new = square_to_rect(N, window_size_bin, new_coords)

    df_features[y_label+"_orig"] = np.clip(row_col_new[:, 0], 0, window_size_bin - 1)
    df_features[x_label+"_orig"] = np.clip(row_col_new[:, 1], 0, N * np.sqrt(2) - 1)

    df_features[y_label+"_unmap"] = df_features[y_label]
    df_features[x_label+"_unmap"] = df_features[x_label]

    df_features.drop([x_label, y_label], axis=1, inplace=True)

    df_features = trim_whitespace_ridges(df_features, im_orig, verbose=verbose, num_cores=num_cores, 
                                         contour_label=contour_label, x_label=x_label, y_label=y_label)

    return df_features


def init_globals_trim_whitespace(im_orig_, x_label_, y_label_, discrepency_threshold_, back_pixel_):
    """
    Helper function for `trim_whitespace_ridges` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im_orig, x_label, y_label, discrepency_threshold, back_pixel
    im_orig = im_orig_
    x_label = x_label_
    y_label = y_label_
    discrepency_threshold = discrepency_threshold_
    back_pixel = back_pixel_

def process_trim_whitespace(df_ridge):
    """
    Helper function for `trim_whitespace_ridges` to process each ridge in parallel
    Returns a tuple: (trimmed_df or None, flag indicating if trimming occurred).
    """
    # Get the global shift by looking at the first element
    global_shift = df_ridge.iloc[0][x_label+"_orig"] - df_ridge.iloc[0][x_label+"_unmap"]
    diff = df_ridge[x_label+"_orig"] - df_ridge[x_label+"_unmap"]

    if np.any(np.abs(global_shift - diff) >= discrepency_threshold):  # at least one pixel inserted
        first_discrepency = np.where(np.abs(global_shift - diff) >= discrepency_threshold)[0][0]

        # For curve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL (i.e. start_bin=0)
        ridge_coords = convert_imagej_coord_to_numpy(
            df_ridge[[x_label+"_orig", y_label+"_orig"]].values,
            im_orig.shape[0], flip_y=False, start_bin=0
        )

        im_curves = extract_line_scale_space(
            ridge_coords, scale_space_container=[np.expand_dims(im_orig, 0)]
        ).squeeze()

        trimmed_flag = 0
        if np.all(im_curves[:first_discrepency] <= back_pixel) or len(im_curves[:first_discrepency]) <= 1:
            # Check if ALL original Hi-C image pixels beyond first discrepancy are all 0, trim
            df_ridge = df_ridge.iloc[:first_discrepency]
            trimmed_flag = 1
        elif np.all(im_curves[first_discrepency:] <= back_pixel) or len(im_curves[first_discrepency:]) <= 1:
            # Otherwise, if ALL original Hi-C image pixels before the first discrepancy are all 0, trim
            df_ridge = df_ridge.iloc[first_discrepency:]
            trimmed_flag = 1

        if len(df_ridge) > 1:
            return df_ridge, trimmed_flag
        else:
            return None, trimmed_flag
    else:
        # No trimming required; keep the ridge as is
        return df_ridge, 0

def trim_whitespace_ridges(df_features, im_orig, verbose, num_cores, 
                           contour_label="Contour Number", x_label="X_(px)", y_label="Y_(px)"):
    """
    This function trims ridges which unnaturally extend between unmapped regions in the Hi-C contact map

    If the ridge turns out to be mostly in the unmapped region, it is trimmed

    Strategy:
    Iterates through each ridge (uniquely identified by `contour_label` and "s_imagej") 
    and identifies ridges with coordinates changed i.e. 
        x_label+"_orig" != x_label+"_unmap"
        OR 
        y_label+"_orig" != y_label+"_unmap"
    At the indices that are changed, indexes into the original Hi-C matrix
    and determines if the values are 0 or not:
        * If the values are all approximately 0, then the portion of the ridge with the 
          first index change onwards is removed entirely (assume ridges are ordered)
        * Otherwise, the ridge is kept as is

    Parameters
    ----------
    df_features : pd.DataFrame
        The expanded table with features
    im_orig : np.ndarray
        The original contact map with zero sum rows/columns removed
    num_cores : int
        Number of cores to use for parallel processing

    Returns
    -------
    pd.DataFrame
        The expanded table with ridges removed based on whitespace criteria
        i.e. ridges that are mostly in unmapped regions are removed
    """
    # Consider trimming if the difference between unmapped and mapped position is greater than 5 
    # This is more for computational efficiency, as we do not want to process ridges that are not affected by unmapped regions
    discrepency_threshold = 5

    gb = df_features.groupby([contour_label, "s_imagej"])
    frames = []
    count_trimmed = 0

    # Get estimate of the background pixel strength
    back_pixel_local = np.percentile(im_orig[im_orig > 0], 25)

    if num_cores == 1:
        for indexer, df_ridge in tqdm(gb):
            global_shift = df_ridge.iloc[0][x_label+"_orig"] - df_ridge.iloc[0][x_label+"_unmap"]
            diff = df_ridge[x_label+"_orig"] - df_ridge[x_label+"_unmap"]

            if np.any(np.abs(global_shift - diff) >= discrepency_threshold):
                # Find the first index where unmapped and mapped coordinates differ
                first_discrepency = np.where(np.abs(global_shift - diff) >= discrepency_threshold)[0][0]

                # Convert coordinates from ImageJ to numpy format
                ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label+"_orig", y_label+"_orig"]].values,
                                                             im_orig.shape[0], flip_y=False, start_bin=0)

                # Extract the image intensity values along the ridge
                im_curves = extract_line_scale_space(ridge_coords, 
                                                     scale_space_container=[np.expand_dims(im_orig, 0)]).squeeze()

                # If all pixel values before the discrepancy are below or equal to background OR
                # that segment is too short (length ≤ 1), remove everything from the start up to the discrepancy                
                if np.all(im_curves[:first_discrepency] <= back_pixel_local) or len(im_curves[:first_discrepency]) <= 1:
                    df_ridge = df_ridge.iloc[:first_discrepency]
                    count_trimmed += 1
                # Else if all pixel values after the discrepancy are below or equal to background OR
                # that segment is too short, remove everything from the discrepancy to the end
                elif np.all(im_curves[first_discrepency:] <= back_pixel_local) or len(im_curves[first_discrepency:]) <= 1:
                    df_ridge = df_ridge.iloc[first_discrepency:]
                    count_trimmed += 1

                # Only keep the ridge if it has more than one point after trimming
                if len(df_ridge) > 1:
                    frames.append(df_ridge)
            else:
                # If no discrepancy exceeds the threshold, keep the ridge intact
                frames.append(df_ridge)
    else:
        groups = list(gb)
        total_groups = len(groups)
        with Pool(num_cores, initializer=init_globals_trim_whitespace,
                  initargs=(im_orig, x_label, y_label, discrepency_threshold, back_pixel_local)) as pool:
            results = []
            for res in tqdm(pool.imap_unordered(process_trim_whitespace,
                                                (group[1] for group in groups)),
                            total=total_groups):
                results.append(res)
        for df_ridge, flag in results:
            if df_ridge is not None:
                frames.append(df_ridge)
            count_trimmed += flag

    df_features_out = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\tNumber of trimmed ridges (unmapped): {count_trimmed} out of {len(gb)}...")

    return df_features_out


from utils.file_io import save_csv


def save_expanded_table(df_features, save_path, root, parameter_str):
    save_name = os.path.join(save_path, f"{root}_expanded_table.csv")
    # if os.path.exists(save_name): 
    #     return 
    
    save_csv(df_features, save_name, root, parameter_str, exclude_rounding=["s_imagej", "ridge_strength", "ridge_condition", "angle_imagej", "angle_fraction"])

    # save a pickle just in case
    # save_name = os.path.join(save_path, f"{root}_expanded_table.pkl")
    # df_features.to_pickle(save_name)

    