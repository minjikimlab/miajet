import pandas as pd
import os
import numpy as np


def read_curve_tracing_results(f, verbose=True):
    """
    Returns dataframes for imageJ curve tracing output 

    Parameters
    ----------
    f : str
        Path to the file containing the curve tracing results
        
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame containing the curve tracing results
    """
    
    try:
        df_results = pd.read_csv(f, index_col=0)
    except pd.errors.EmptyDataError:
        if verbose:
            print(f"\tFile {f} empty. Skipping...")
        df_results = None

    return df_results



def load_imagej_results(save_path, scale_range, verbose, root,
                        contour_label="Contour Number", 
                        pos_label="Point Number", 
                        frame_label="Frame Number"):
    """
    Loads the tables from ImageJ curve tracing and combines them
    * Recall that each table corresponds to the imageJ results of the image blurred at a single scale

    Two tables are returned:
    1. A summary dataframe is generated that collapses each ridge to a single row 
    2. An expanded table is generated that contains all points in each ridge
    Each ridge is uniquely identified by the columns [contour_label, "s_imagej"] 
    present in both tables

    Parameters
    ----------
    save_path : str
        Path to the directory where the ImageJ results are saved
    scale_range : list
        List of scales used in the ImageJ curve tracing
    verbose : bool
    root : str
        Root name for the files to be loaded
    contour_label : str
        Column name in Curve Tracing tables that is a unique identifier for each table
    pos_label : str
        Column name in Curve Tracing tables that numbers the points in each ridge
    frame_label : str
        Column name in Curve Tracing table

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the summary of the ImageJ results
    df_results : pd.DataFrame
        DataFrame containing the expanded table of the ImageJ curve tracing
        i.e. each row corresponds to a point in the ridge
    """
    # f_expanded_table = os.path.join(save_path, f"{root}_expanded_table.csv")
    # if os.path.exists(f_expanded_table):
    #     if verbose: print("\tExpanded table already exists. Skipping load imagej...")
    #     return None, None

    # collect results files
    results_s_imagej = []

    for s in scale_range:
        s_str = f"{s:.3f}"

        # equivalent to the "expected_csv" variable in `call_imagej.process_sigma`
        f = os.path.join(save_path, f"{root}_imagej_results_s-{s_str}_table.csv")

        # read curve tracing results
        df_results = read_curve_tracing_results(f, verbose)

        if df_results is None:
            continue

        # remove duplicate rows in summary and results (most likely due to junctions, but the data is exactly the same)
        df_results.drop_duplicates([contour_label, pos_label, frame_label], inplace=True)

        # assign the scale as "s_imagej"
        df_results["s_imagej"] = s 

        results_s_imagej.append(df_results)

    # combine
    df_results = pd.concat(results_s_imagej).reset_index(drop=True)

    # take average of left and right widths to form the "width" column
    df_results["width"] = (df_results["Width_left_(px)"] + df_results["Width_right_(px)"]) / 2

    # drop duplicates according to the unique identifier
    df = df_results.drop_duplicates([contour_label, "s_imagej"], ignore_index=True) 

    df = df[["Label", frame_label, contour_label, "s_imagej", "Response"]]
        
    return df, df_results

from .expanded_table import rect_to_square

def remove_padding_positions(df_pos, df, N, window_size_bin, verbose, x_label="X_(px)", y_label="Y_(px)", contour_label="Contour Number"):
    """
    Removes ridges with positions in the rotation padding areas (simply clips them but does not discard the ridge)
    Note that coordinates are w.r.t the image with unmappable regions removed. 
    Because of this:
     * 'X_(px)' and 'Y_(px)' labels which come directly from ImageJ 
     * N is the number of bins in the Hi-C data after removing zero sum rows and columns

    Updates both the expanded and summary tables

    Approach: 
    1. Maps indices back from rectangle to square using .expanded_table.rect_to_square function
    2. Determines whether these indices are in valid positions i.e. [0, N - 1]

    Parameters
    ----------
    df_pos : pd.DataFrame
        DataFrame containing the expanded table of ImageJ curve tracing results
    df : pd.DataFrame
        DataFrame containing the summary table of ImageJ curve tracing results
    N : int
        The number of bins in the Hi-C data after removing zero sum rows and columns
    window_size_bin : int
        Binned window size

    Returns
    -------
    df_pos : pd.DataFrame
        Updated expanded table 
    df : pd.DataFrame
        Updated summary table
    """
    # convert 
    coords = rect_to_square(N, window_size_bin, df_pos[[y_label, x_label]].values)

    valid_coords = np.all((coords >= 0) & (coords <= N - 1), axis=1)
    new_df_pos = df_pos.loc[valid_coords].reset_index(drop=True)
    keep_rows = new_df_pos[[contour_label, "s_imagej"]]

    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)
    
    if verbose:
        print(f"\tRemoving positions in rotation padding region {len(df)} -> {len(df_new)}...")
    
    return new_df_pos, df_new



def remove_kth_off_diagonal(df_pos, df, k, window_size_bin, contour_label, y_label, verbose=False):
    """
    Clips ridge positions within k-th off diagonal of the contact map.
    `window_size_bin` is required because the y positions are flipped (largest is closest to diagonal)

    Updates both the expanded and summary tables
    
    Parameters
    ----------
    df_pos : pd.DataFrame
        DataFrame containing the expanded table of ImageJ curve tracing results
    df : pd.DataFrame
        DataFrame containing the summary table of ImageJ curve tracing results
    k : int
        The k-th off diagonal within which ridge positions are clipped
    window_size_bin : int
        Binned window size
    contour_label : str
        Column name in expanded table that is a unique identifier for a given scale (typically just 'Contour Number')
    y_label : str
        Column name in expanded table that contains the y positions of the points in the ridge
    
    Returns
    -------
    df_pos : pd.DataFrame
        Updated expanded table 
    df : pd.DataFrame
        Updated summary table
    """
    num_rows_image = np.ceil(window_size_bin / np.sqrt(2)).astype(int)
    thresh = num_rows_image - 1 - k # convert from size to index
    
    new_df_pos = df_pos.loc[df_pos[y_label] <= thresh].reset_index(drop=True)
    keep_rows = new_df_pos[[contour_label, "s_imagej"]]
    
    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)
    
    if verbose:
        print(f"\tRemoving positions < {k} off diagonal {len(df)} -> {len(df_new)}...")
    
    return new_df_pos, df_new

def remove_small_ridges(df_pos, df, min_points, contour_label, verbose=False):
    """
    Removes small ridges whose length is less than or equal to `min_points`
    Updates both the expanded and summary tables

    Parameters
    ----------
    df_pos : pd.DataFrame
        DataFrame containing the expanded table of ImageJ curve tracing results
    df : pd.DataFrame
        DataFrame containing the summary table of ImageJ curve tracing results
    min_points : int
        The minimum number of points in a ridge > we keep it
    contour_label : str
        Column name in expanded table that is a unique identifier for a given scale (typically just 'Contour Number')
    
    Returns
    -------
    df_pos : pd.DataFrame
        Updated expanded table
    df : pd.DataFrame
        Updated summary table
    """
    new_df_pos = df_pos.loc[df_pos.groupby([contour_label, "s_imagej"], sort=False)[contour_label].transform('size') > min_points]
    keep_rows = new_df_pos[[contour_label, "s_imagej"]]

    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)
    
    if verbose:
        print(f"\tRemoving small ridges <= {min_points} length {len(df)} -> {len(df_new)}...")
        
    return new_df_pos, df_new


# def process_enforce_root(df_ridge, thresh, y_label):
#     """
#     Helper function for enforce_root_position that processes each ridge
#     """
#     # Discard ridge if its maximum y coordinate is less than the threshold
#     if df_ridge[y_label].max() < thresh:
#         return None, 1  # None indicates discard; count_discard=1
#     else:
#         return df_ridge, 0

def enforce_root_position(df_pos, df, root_within, window_size_bin, verbose,
                          contour_label="Contour Number", 
                          y_label="Y_(px)"):
    """
    Enforces that the root (maximum y value or the closest point to the main diagonal) of each ridge
    is within the specifie root_within-th off-diagonal. Ridges failing this criterion are discarded
    
    Updates both the expanded and summary tables

    Parameters
    ----------
    df_pos : pd.DataFrame
        DataFrame containing the expanded table of ImageJ curve tracing results
    df : pd.DataFrame
        DataFrame containing the summary table of ImageJ curve tracing results
    root_within : int
        The k-th off diagonal within which the root position of the ridge must be located
    window_size_bin : int
        Binned window size
    num_cores : int
        Number of cores to use for parallel processing
    
    Returns
    -------
    df_pos : pd.DataFrame
        Updated expanded table
    df : pd.DataFrame
        Updated summary table
    """
    if df is None and df_pos is None:
        if verbose:
            print("\tSkipping process imagej...")
        return None, None

    # Determine threshold based on window size and resolution
    num_rows_image = np.ceil(window_size_bin / np.sqrt(2)).astype(int)
    thresh_val = (num_rows_image - 1 - root_within) if root_within is not None else 0

    keys = [contour_label, "s_imagej"]

    # per-row max of y within each ridge
    max_per_row = df_pos.groupby(keys, sort=False)[y_label].transform('max')
    keep_mask = max_per_row >= thresh_val

    # filtered expanded table
    df_pos_out = df_pos.loc[keep_mask].reset_index(drop=True)

    # filtered summary table (using unique kept keys)
    keep_rows = df_pos_out[keys].drop_duplicates()
    df_new = df.merge(keep_rows, on=keys, how="inner")
    df_new.drop_duplicates(keys, inplace=True, ignore_index=True)

    if verbose:
        total_groups = df_pos[keys].drop_duplicates().shape[0]
        kept_groups = keep_rows.shape[0]
        discarded = total_groups - kept_groups
        print(f"\tEnforcing root position to be within <= {root_within}: discarded {discarded} of {total_groups} ridges.")

    return df_pos_out, df_new


def enforce_y_displacement(df_pos, df, verbose, 
                           contour_label="Contour Number", 
                           y_label="Y_(px)",
                           displacement_thresh=2):
    """
    Enforces that each ridge has a vertical displacement greater than the given threshold.
    Ridges failing this criterion (i.e. displacement <= threshold) are discarded.
    
    Updates both the expanded and summary tables.

    Parameters
    ----------
    df_pos : pd.DataFrame
        DataFrame containing the expanded table of ImageJ curve tracing results
    df : pd.DataFrame
        DataFrame containing the summary table of ImageJ curve tracing results
    verbose : bool
        Whether to print progress and discarded counts
    contour_label : str, default="Contour Number"
        Column name identifying ridges in the expanded table
    y_label : str, default="Y_(px)"
        Column containing y-coordinates
    displacement_thresh : int or float, default=2
        Minimum required y displacement (max - min). Ridges with <= this value are discarded.
    
    Returns
    -------
    df_pos_out : pd.DataFrame
        Updated expanded table with ridges filtered
    df_new : pd.DataFrame
        Updated summary table with ridges filtered
    """
    if df is None and df_pos is None:
        if verbose:
            print("\tSkipping enforce_y_displacement...")
        return None, None

    keys = [contour_label, "s_imagej"]

    # Compute displacement per ridge
    displacement = (
        df_pos.groupby(keys, sort=False)[y_label]
        .transform(lambda x: x.max() - x.min())
    )
    keep_mask = displacement > displacement_thresh

    # Filter expanded table
    df_pos_out = df_pos.loc[keep_mask].reset_index(drop=True)

    # Filter summary table (using unique kept keys)
    keep_rows = df_pos_out[keys].drop_duplicates()
    df_new = df.merge(keep_rows, on=keys, how="inner")
    df_new.drop_duplicates(keys, inplace=True, ignore_index=True)

    if verbose:
        total_groups = df_pos[keys].drop_duplicates().shape[0]
        kept_groups = keep_rows.shape[0]
        discarded = total_groups - kept_groups
        print(f"\tEnforcing y displacement > {displacement_thresh}: discarded {discarded} of {total_groups} ridges.")

    return df_pos_out, df_new



def reorient_ridges(df_results, gb_results, y_label, verbose=False, window_size_bin=None):
    '''
    Reorients the ridges based on the y positions of the first and last points in each ridge
    The invariant is that the y position (vertical axis of the rectangle) should be decreasing along the ridge

    NEW: Now sorts the entire ridge by y_label in decreasing order

    Originally copied from filtering.py

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing the expanded table of the ImageJ curve tracing
    gb_results : pd.core.groupby.generic.DataFrameGroupBy
        The expanded table grouped by each ridge (i.e. grouped by contour_label and s_imagej)
    y_label : str
        The column name in df_results that contains the y positions of the points in the ridge
    
    Returns
    -------
    df_results : pd.DataFrame
        Updated expanded table
    '''
    # Determine 3 points to main diagonal
    root_within_fixed = 3
    num_rows_image = np.ceil(window_size_bin / np.sqrt(2)).astype(int)
    thresh_val = (num_rows_image - 1 - root_within_fixed) if root_within_fixed is not None else 0

    count = 0  # count how many ridges 

    frames = []
    for cid in gb_results.indices:

        # select ridge
        df_ridge = df_results.iloc[gb_results.indices.get(cid)].copy()

        # OLD SORT
        # # sort rows so y_label is in decreasing order
        # if not df_ridge[y_label].is_monotonic_decreasing:
        #     count += 1
        #     df_ridge = df_ridge.sort_values(by=y_label, ascending=False)

        # NEW SORT
        point_a = df_ridge[y_label].values[0]
        point_b = df_ridge[y_label].values[-1]
        point_c = df_ridge[y_label].min()

        if point_c < point_a and point_c < point_b:
            # C-shaped ridges 
            # Its ambiguous how to sort.. so do not do an actual sort of values for these
            # ridges
            if point_a == point_b or (point_a > thresh_val and point_b > thresh_val and point_c < thresh_val):
                # Set the root as the larger value
                if df_ridge["Response"].values[0] < df_ridge["Response"].values[-1]:
                    # Flip order
                    df_ridge = df_ridge[::-1].reset_index(drop=True)
                    count += 1
            elif df_ridge[y_label].values[0] < df_ridge[y_label].values[-1]:
                # Flip order
                df_ridge = df_ridge[::-1].reset_index(drop=True)
                count += 1
        else:
            # But if its not C-shaped, we should sort rigorously i.e. monotonic decreasing
            if not df_ridge[y_label].is_monotonic_decreasing:
                df_ridge = df_ridge.sort_values(by=y_label, ascending=False)
                count += 1

        frames.append(df_ridge)

    if verbose:
        print(f"\tReoriented {count} out of {len(gb_results)} ridges")

    if count == 0:
        return df_results

    return pd.concat(frames).reset_index(drop=True)


def check_ridge_order(df_pos,
                      contour_label="Contour Number",
                      s_label="s_imagej",
                      y_label="Y_(px)"):
    """
    Verify that, for each (contour_label, s_imagej) ridge,
    the y_label series is non-increasing
    """
    for (cid, sid), grp in df_pos.groupby([contour_label, s_label], sort=False):
        if not grp[y_label].is_monotonic_decreasing:
            print(f"Order violation in ridge: contour={cid}, scale={sid}")
            return False
    return True

def process_imagej_results(df, df_pos, window_size, N, resolution, remove_kth_strata, remove_min_size, root_within, 
                           verbose, num_cores,
                           contour_label="Contour Number", 
                           y_label="Y_(px)", 
                           angle_label="Angle_of_normal_(radians)"
                           ):
    """
    Processes the ImageJ results (i.e. summary and expanded tables) by:
    1. Clips ridge positions that are in the padding region 
    2. Clips ridge positions that are within the k-th off diagonal of the contact map
    3. Discards any ridges whose closest position of the ridge to the main diagonal
        is not within the `root_within`-th off diagonal
    4. Removes ridges that are smaller than `remove_min_size` (typically 1)

    Updates both summary and expanded tables and returns them

    Parameters
    ----------
    df : pd.DataFrame
        Summary table of ImageJ results
    df_pos : pd.DataFrame
        Expanded table of ImageJ results
    window_size : int
        Size of the window in base pairs to extract from the Hi-C data
    N : int
        The number of bins in the Hi-C data after removing zero sum rows and columns
    resolution : int
        Resolution of the Hi-C data in base pairs
    remove_kth_strata : int
        The k-th off diagonal within which ridge positions are clipped
    remove_min_size : int
        The minimum number of points in a ridge. Otherwise, it is discarded
    root_within : int
        The k-th off diagonal within which the root position (minimum position of ridge to the main diagonal) 
        of the ridge must be located
    
    Returns
    -------
    df : pd.DataFrame
        Processed summary table of ImageJ results
    df_pos : pd.DataFrame
        Processed expanded table of ImageJ results
    """

    # if df is None and df_pos is None:
    #     # Skip processing if both df and df_pos are None 
    #     # This corresponds to the case when results are already generated (e.g. expanded table, summary table)
    #     # This function is now deprecated
    #     if verbose: print("\tSkipping process imagej...")
    #     return None, None
    
    window_size_bin = np.ceil(window_size / resolution).astype(int)

    # convert ImageJ angles from [0 to 2pi] to [0, 180]
    df_pos[angle_label] = np.degrees(df_pos[angle_label]) % 180

    # Reorder ridges 
    # df_pos = reorient_ridges(df_pos, df_pos.groupby([contour_label, "s_imagej"], sort=False), y_label, True)

    if verbose: print(f"\tNum ridges before processing: {len(df)}")
    df_pos, df = enforce_root_position(df_pos=df_pos, df=df, root_within=root_within, window_size_bin=window_size_bin, 
                                       verbose=verbose, contour_label=contour_label, y_label=y_label)
    # df_pos, df = enforce_y_displacement(df_pos=df_pos, df=df, verbose=verbose, contour_label=contour_label, y_label=y_label, displacement_thresh=2)
    df_pos, df = remove_padding_positions(df_pos=df_pos, df=df, N=N, window_size_bin=window_size_bin, verbose=verbose)
    df_pos, df = remove_kth_off_diagonal(df_pos=df_pos, df=df, k=remove_kth_strata, window_size_bin=window_size_bin, 
                                         contour_label=contour_label, y_label=y_label, verbose=verbose)
    df_pos, df = remove_small_ridges(df_pos=df_pos, df=df, min_points=remove_min_size, contour_label=contour_label, verbose=verbose)

    assert len(df_pos.drop_duplicates([contour_label, "s_imagej"])) == len(df)

    if verbose: print(f"\tNum ridges after processing: {len(df)}")

    df_pos = reorient_ridges(df_pos, df_pos.groupby([contour_label, "s_imagej"], sort=False), y_label, True, window_size_bin)

    # assert check_ridge_order(df_pos)

    return df, df_pos




from utils.plotting import convert_imagej_coord_to_numpy
from utils.scale_space import round_line_scale_space, extract_line_scale_space
from tqdm import tqdm
from multiprocessing import Pool

def init_globals_trim_corner(im_shape_0_, min_trim_size_in_, C_, x_label_, y_label_):
    """
    Helper function for `trim_imagej_results_corner` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im_shape_0, min_trim_size_in, C, x_label, y_label
    im_shape_0 = im_shape_0_
    min_trim_size_in = min_trim_size_in_
    C = C_
    x_label = x_label_
    y_label = y_label_

def process_trim_corner(df_ridge):
    """
    Helper function for `trim_imagej_results_corner` that processes each ridge
    """
    df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

    if min_trim_size_in < 1:
        # interpret as percentile threshold
        min_trim_size = int(len(df_ridge) * min_trim_size_in)
    else:
        # interpret as number of bins
        min_trim_size = min_trim_size_in      

    # Extract coordinates and interpolate using the C tensor
    ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                  im_shape_0, flip_y=False, start_bin=0)
    C_curves = extract_line_scale_space(ridge_coords, [C])
    # C_curves = (C_curves > 0.25).astype(bool)
    C_curves = np.sum(C_curves > 0.25, axis=0) >= 2 # New
    cols_true = np.where(np.all(C_curves, axis=0))[0]    

    # `first_col_idx` is the index of the first column that is all True
    # we must ensure that the ridge is long enough before trimming
    cols_true = cols_true[cols_true >= min_trim_size]

    if len(cols_true):
        first_col_idx = cols_true[0]
        trimmed = df_ridge.iloc[:first_col_idx]
        return trimmed, 1  # trimmed, count_trimmed
    else:
        return df_ridge, 0

def trim_imagej_results_corner(df, df_pos, C, im_shape_0, min_trim_size_in, remove_min_size, num_cores, verbose,
                               contour_label="Contour Number", 
                               x_label="X_(px)", 
                               y_label="Y_(px)"):
    """
    Trims ridges based on the C tensor (corner condition).
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature 
    of the corner condition is True (i.e. at all scales there is a corner), then the ridge is trimmed
    at that point 

    Parameters
    ----------
    df : pd.DataFrame
        Summary table of ImageJ results
    df_pos : pd.DataFrame
        Expanded table of ImageJ results
    C : np.ndarray
        The corner tensor of (num_scales, n, m) where image is of shape (n, m)
    im_shape_0 : int
        Binned window size (i.e. im.shape[0])
    min_trim_size_in : float or int
        Specifies the minimum size of the ridge where no trimming can occur 
        If `min_trim_size_in` is None, then no trimming occurs
        If `min_trim_size_in` = 0, then trimming can occur at any point in the ridge
        If `min_trim_size_in` < 1, then interpret as a fraction of the ridge length where no trimming can occur 
            Example: 0.1 means that 10% trimming cannot leave a ridge smaller than 10% of its original length
        If `min_trim_size_in` >= 1, then interpret as number of bins (i.e. minimum number of points in the ridge)
        where no trimming can occur
    remove_min_size : int
        The minimum number of points in a ridge. Otherwise, it is discarded
    num_cores : int
        Number of cores to use for parallel processing
    
    Returns
    -------
    df : pd.DataFrame
        Updated summary table
    df_pos : pd.DataFrame
        Updated expanded table
    """
    if df is None and df_pos is None:
        if verbose: print("\tSkipping process imagej...")
        return None, None
    
    if min_trim_size_in is None:
        # interpret as no trimming
        if verbose: print("\tNo corner trimming")
        return df, df_pos

    # Group by contour and s_imagej
    gb = df_pos.groupby([contour_label, "s_imagej"], sort=False)
    frames = []
    count_trimmed = 0


    # print("\tWARNING: DEBUG MODE NO PARALLEILIZATION")
    if num_cores == 1:
        for indexer, df_ridge in tqdm(gb):

            df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

            # if indexer[0] == 49 and np.round(indexer[1], 2) == 6.59:
            #     pass

            if min_trim_size_in < 1:
                # interpret as percentile threshold
                min_trim_size = int(len(df_ridge) * min_trim_size_in)
            else:
                # interpret as number of bins
                min_trim_size = min_trim_size_in   

            # for cuve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL 
            ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                          im_shape_0, flip_y=False, start_bin=0)
            
            # interpolate the corner values 
            C_curves = extract_line_scale_space(ridge_coords, [C])

            # hard-code to 0.25 for corner condition 
            # floating point is due to interpolation [0, 1]
            # C_curves = (C_curves > 0.25).astype(bool) 
            # New corner condition: True for at least 2 scales
            C_curves = np.sum(C_curves > 0.25, axis=0) >= 2
            
            # extract the columns that is all true
            cols_true = np.where(np.all(C_curves, axis=0))[0]
            
            # `first_col_idx` is the index of the first column that is all True
            # we must ensure that the ridge is long enough before trimming
            cols_true = cols_true[cols_true >= min_trim_size]

            if len(cols_true):
                first_col_idx = cols_true[0]

                frames.append(df_ridge.iloc[:first_col_idx])
                count_trimmed += 1
            else:
                frames.append(df_ridge)
    else:
        groups = list(gb)
        total_groups = len(groups)
        with Pool(num_cores, initializer=init_globals_trim_corner, initargs=(im_shape_0, min_trim_size_in, C, x_label, y_label)) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_trim_corner,
                                                   (group[1] for group in groups)),
                               total=total_groups):
                results.append(result)
        for trimmed_df, flag_trim in results:
            if trimmed_df is not None:
                frames.append(trimmed_df)
            count_trimmed += flag_trim

    if verbose: 
        print(f"\tNumber of trimmed ridges (corner): {count_trimmed} out of {len(df)}...")
    df_pos_out = pd.concat(frames).reset_index(drop=True)

    # Update `df` in case the trimming inadvertantly filtered out ridges (will happen if first_col_idx = 0)
    keep_rows = df_pos_out[[contour_label, "s_imagej"]]
    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)

    if verbose: print("\tNow removing small ridges (post corner trimming)...")
    df_pos_out, df_new = remove_small_ridges(df_pos_out, df_new, remove_min_size, contour_label, verbose)

    # df_pos_out = reorient_ridges(df_pos_out, df_pos_out.groupby([contour_label, "s_imagej"], sort=False), y_label, True)

    return df_new, df_pos_out

def init_globals_trim_angle(im_shape_0_, A_, x_label_, y_label_, scale_range_, min_trim_size_in_):
    """
    Helper function for `trim_imagej_results_angle` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im_shape_0, A, x_label, y_label, scale_range, min_trim_size_in
    im_shape_0 = im_shape_0_
    A = A_
    x_label = x_label_
    y_label = y_label_
    scale_range = scale_range_
    min_trim_size_in = min_trim_size_in_


def process_trim_angle(df_ridge):
    """
    Helper function for `trim_imagej_results_angle` that processes each ridge
    """
    df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

    # Extract coordinates and interpolate using the A tensor
    ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                 im_shape_0, flip_y=False, start_bin=0)
    
    # A_curves = extract_line_scale_space(ridge_coords, [A]).astype(bool)
    A_curves = round_line_scale_space(ridge_coords, [A]).astype(bool)

    scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["s_imagej"]))[0]
    if len(scale_assigned_set) == 0:
        selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["s_imagej"]))[0][0]
    else:
        selected_scale_idx = scale_assigned_set[0]
    cols_true = np.where(A_curves[selected_scale_idx])[0]
    # --- New min_trim_size_in logic ---
    if min_trim_size_in < 1:
        min_trim_size = int(len(df_ridge) * min_trim_size_in)
    else:
        min_trim_size = min_trim_size_in
    cols_true = cols_true[cols_true >= min_trim_size]
    # ------------------------------------
    if len(cols_true):
        first_col_idx = cols_true[0]
        return df_ridge.iloc[:first_col_idx], 1
    else:
        return df_ridge, 0


def trim_imagej_results_angle(df, df_pos, A, im_shape_0, min_trim_size_in, remove_min_size, num_cores, verbose,
                              scale_range, angle_range,
                              contour_label="Contour Number", 
                              x_label="X_(px)", 
                              y_label="Y_(px)"):
    """
    Trims ridges based on the A tensor (angle condition).
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature
    of the angle condition is False (i.e. at the selected scale the angle of the ridge is not in `angle_range`),
    then the ridge is trimmed at that point

    Parameters
    ----------
    df : pd.DataFrame
        Summary table of ImageJ results
    df_pos : pd.DataFrame
        Expanded table of ImageJ results
    A : np.ndarray
        The angle tensor of (num_scales, n, m) where image is of shape (n, m)
    im_shape_0 : int
        Binned window size (i.e. im.shape[0])
    min_trim_size_in : float or int
        Specifies the minimum size of the ridge where no trimming can occur 
        If `min_trim_size_in` is None, then no trimming occurs
        If `min_trim_size_in` = 0, then trimming can occur at any point in the ridge
        If `min_trim_size_in` < 1, then interpret as a fraction of the ridge length where no trimming can occur 
            Example: 0.1 means that 10% trimming cannot leave a ridge smaller than 10% of its original length
        If `min_trim_size_in` >= 1, then interpret as number of bins (i.e. minimum number of points in the ridge)
        where no trimming can occur
    remove_min_size : int
        The minimum number of points in a ridge. Otherwise, it is discarded
    num_cores : int
        Number of cores to use for parallel processing
    scale_range : list
        List of scales generated
    angle_range : list
        Upper and lower bound of the valid angle range
    
    Returns
    -------
    df : pd.DataFrame
        Updated summary table
    df_pos : pd.DataFrame
        Updated expanded table
    """
    if df is None and df_pos is None:
        if verbose: print("\tSkipping process imagej...")
        return None, None
    
    if min_trim_size_in is None:
        # interpret as no trimming
        if verbose: print("\tNo angle trimming")
        return df, df_pos

    # Make a copy of A and update based on angle_range without modifying the original A
    A = A.copy()
    A = ~np.logical_and(A > angle_range[0], A < angle_range[1]) # not within range (its flipped!)
    
    # Group by contour and s_imagej
    gb = df_pos.groupby([contour_label, "s_imagej"], sort=False)
    frames = []
    count_trimmed = 0

    # print("\tWARNING: DEBUG MODE")
    if num_cores == 1:
        for indexer, df_ridge in tqdm(gb):

            df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

            # if indexer[0] == 2838 and np.round(indexer[1], 2) == 1.67:
            #     pass

            ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                          im_shape_0, flip_y=False, start_bin=0)
            
            # A_curves = extract_line_scale_space(ridge_coords, [A]).astype(bool)
            A_curves = round_line_scale_space(ridge_coords, scale_space_container=[A]).astype(bool) # round
            
            # Determine the selected scale index based on scale_range and df_ridge["s_imagej"]
            scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["s_imagej"]))[0]
            if len(scale_assigned_set) == 0: 
                selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["s_imagej"]))[0][0]
            else:
                selected_scale_idx = scale_assigned_set[0]
                
            cols_true = np.where(A_curves[selected_scale_idx])[0]
            # --- New min_trim_size_in logic ---
            if min_trim_size_in < 1:
                # interpret as percentile threshold
                min_trim_size = int(len(df_ridge) * min_trim_size_in)
            else:
                # interpret as number of bins
                min_trim_size = min_trim_size_in
            # `first_col_idx` is the index of the first column that is all True
            # we must ensure that the ridge is long enough before trimming
            cols_true = cols_true[cols_true >= min_trim_size]
            # ------------------------------------
            if len(cols_true):
                first_col_idx = cols_true[0]
                frames.append(df_ridge.iloc[:first_col_idx])
                count_trimmed += 1
            else:
                frames.append(df_ridge)
    else:
        # Parallel processing block for trim_imagej_results_angle:
        groups = list(gb)
        total_groups = len(groups)
        with Pool(num_cores, initializer=init_globals_trim_angle, initargs=(im_shape_0, A, x_label, y_label, scale_range, min_trim_size_in)) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_trim_angle,
                                                (group[1] for group in groups)),
                            total=total_groups):
                results.append(result)
        for trimmed_df, flag_trim in results:
            if trimmed_df is not None:
                frames.append(trimmed_df)
            count_trimmed += flag_trim

    if verbose: 
        print(f"\tNumber of trimmed ridges (angle): {count_trimmed} out of {len(df)}...")
    df_pos_out = pd.concat(frames).reset_index(drop=True)
    
    # Update `df` in case the trimming inadvertently filtered out ridges (will happen if first_col_idx = 0)
    keep_rows = df_pos_out[[contour_label, "s_imagej"]]
    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)

    if verbose: print("\tNow removing small ridges (post angle trimming)...")
    df_pos_out, df_new = remove_small_ridges(df_pos_out, df_new, remove_min_size, contour_label, verbose)

    # df_pos_out = reorient_ridges(df_pos_out, df_pos_out.groupby([contour_label, "s_imagej"], sort=False), y_label, True)

    return df_new, df_pos_out


from scipy.signal import argrelmax

def init_globals_trim_eig2(im_shape_0_, W2_, x_label_, y_label_, scale_range_, min_trim_size_in_):
    """
    Helper function for `trim_imagej_results_eig2` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im_shape_0, W2, x_label, y_label, scale_range, min_trim_size_in
    im_shape_0 = im_shape_0_
    W2 = W2_
    x_label = x_label_
    y_label = y_label_
    scale_range = scale_range_
    min_trim_size_in = min_trim_size_in_

def process_trim_eig2(df_ridge):
    """
    Helper function for `trim_imagej_results_eig2` that processes each ridge
    """
    df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)


    # Extract coordinates using the provided helper function.
    ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                  im_shape_0, flip_y=False, start_bin=0)
    # Compute the eig2 (W2) scale-space representation.
    W2_curves = extract_line_scale_space(ridge_coords, [W2])
    
    # Determine the selected scale index based on scale_range and df_ridge["s_imagej"]
    scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["s_imagej"]))[0]
    if len(scale_assigned_set) == 0:
        selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["s_imagej"]))[0][0]
    else:
        selected_scale_idx = scale_assigned_set[0]
    

    W2_curves_line = W2_curves[selected_scale_idx]

    local_peaks = argrelmax(W2_curves_line)[0]

    if len(local_peaks) == 0:
        # If no local maxima, find the global max index
        peak_idx = np.argmax(W2_curves_line)

    else:
        # Find the first local maxima that is non-negative
        peak_idx = 0
        for j in local_peaks:
            if W2_curves_line[j] >= 0:
                peak_idx = j
                break

    if W2_curves_line[peak_idx] < 0:
        # If the peak is negative, set peak_idx to 0
        # We should begin trimming from this point onwards
        peak_idx = 0

    # Find all indices where the value is negative in the full array
    neg_indices = np.where(W2_curves_line < 0)[0]

    # Select only those indices that occur after the peak index
    cols_true = neg_indices[neg_indices >= peak_idx]
                            
    # --- min_trim_size_in logic ---
    if min_trim_size_in < 1:
        # Interpret as percentile threshold
        min_trim_size = int(len(df_ridge) * min_trim_size_in)
    else:
        # Interpret as number of bins
        min_trim_size = min_trim_size_in
    # Ensure that the ridge is long enough before trimming
    cols_true = cols_true[cols_true >= min_trim_size]
    # ------------------------------------
    
    if len(cols_true):
        first_col_idx = cols_true[0]
        return df_ridge.iloc[:first_col_idx], 1
    else:
        return df_ridge, 0

def trim_imagej_results_eig2(df, df_pos, W2, im_shape_0, min_trim_size_in, remove_min_size, num_cores, verbose, scale_range,
                              contour_label="Contour Number", 
                              x_label="X_(px)", 
                              y_label="Y_(px)"):
    """
    Trims ridges based on the W2 tensor (eig2 condition).
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature
    of the eig2 condition is negative (i.e. at the selected scale the eig2 is negative),
    then the ridge is trimmed at that point

    Note that this invariant was mainly designed for datatype='oe' not 'observed'

    Parameters
    ----------
    df : pd.DataFrame
        Summary table of ImageJ results
    df_pos : pd.DataFrame
        Expanded table of ImageJ results
    W2 : np.ndarray
        The eig2 tensor of (num_scales, n, m) where image is of shape (n, m)
    im_shape_0 : int
        Binned window size (i.e. im.shape[0])
    min_trim_size_in : float or int
        Specifies the minimum size of the ridge where no trimming can occur
        If `min_trim_size_in` is None, then no trimming occurs
        If `min_trim_size_in` = 0, then trimming can occur at any point in the ridge
        If `min_trim_size_in` < 1, then interpret as a fraction of the ridge length where no trimming can occur
        Example: 0.1 means that 10% trimming cannot leave a ridge smaller than 10% of its original length
        If `min_trim_size_in` >= 1, then interpret as number of bins (i.e. minimum number of points in the ridge)
        where no trimming can occur
    remove_min_size : int
        The minimum number of points in a ridge. Otherwise, it is discarded
    num_cores : int
        Number of cores to use for parallel processing
    scale_range : list
        List of scales generated

    Returns
    -------
    df : pd.DataFrame
        Updated summary table
    df_pos : pd.DataFrame
        Updated expanded table
    """
    if df is None and df_pos is None:
        if verbose: print("\tSkipping process imagej...")
        return None, None
    
    if min_trim_size_in is None:
        # interpret as no trimming
        if verbose: print("\tNo eig2 trimming")
        return df, df_pos

    # Group by contour and s_imagej
    gb = df_pos.groupby([contour_label, "s_imagej"], sort=False)
    frames = []
    count_trimmed = 0

    # print("\tWARNING: DEBUG MODE")
    if num_cores == 1:
        for indexer, df_ridge in tqdm(gb):

            df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

            ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values,
                                                          im_shape_0, flip_y=False, start_bin=0)
            
            W2_curves = extract_line_scale_space(ridge_coords, [W2])
            # W2_curves = round_line_scale_space(ridge_coords, scale_space_container=[W2]) # round
            
            # Determine the selected scale index based on scale_range and df_ridge["s_imagej"]
            scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["s_imagej"]))[0]
            if len(scale_assigned_set) == 0: 
                selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["s_imagej"]))[0][0]
            else:
                selected_scale_idx = scale_assigned_set[0]

            W2_curves_line = W2_curves[selected_scale_idx]

            local_peaks = argrelmax(W2_curves_line)[0]

            if len(local_peaks) == 0:
                # If no local maxima, find the global max index
                peak_idx = np.argmax(W2_curves_line)

            else:
                # Find the first local maxima that is non-negative
                peak_idx = 0
                for j in local_peaks:
                    if W2_curves_line[j] >= 0:
                        peak_idx = j
                        break

            if W2_curves_line[peak_idx] < 0:
                # If the peak is negative, set peak_idx to 0
                # We should begin trimming from this point onwards
                peak_idx = 0

            # Find all indices where the value is negative in the full array
            neg_indices = np.where(W2_curves_line < 0)[0]

            # Select only those indices that occur after the peak index
            cols_true = neg_indices[neg_indices >= peak_idx]
                            
            # --- min_trim_size_in logic ---
            if min_trim_size_in < 1:
                # Interpret as percentile threshold
                min_trim_size = int(len(df_ridge) * min_trim_size_in)
            else:
                # Interpret as number of bins
                min_trim_size = min_trim_size_in
            # `first_col_idx` is the index of the first column that is all True
            # we must ensure that the ridge is long enough before trimming
            cols_true = cols_true[cols_true >= min_trim_size]
            # ------------------------------------

            if len(cols_true):
                first_col_idx = cols_true[0]
                frames.append(df_ridge.iloc[:first_col_idx])
                count_trimmed += 1
            else:
                frames.append(df_ridge)
    else:
        # Parallel processing block for trim_imagej_results_eig2:
        groups = list(gb)
        total_groups = len(groups)
        with Pool(num_cores, initializer=init_globals_trim_eig2, initargs=(im_shape_0, W2, x_label, y_label, scale_range, min_trim_size_in)) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_trim_eig2,
                                                (group[1] for group in groups)),
                            total=total_groups):
                results.append(result)
        for trimmed_df, flag_trim in results:
            if trimmed_df is not None:
                frames.append(trimmed_df)
            count_trimmed += flag_trim

    if verbose: 
        print(f"\tNumber of trimmed ridges (eig2): {count_trimmed} out of {len(df)}...")
    df_pos_out = pd.concat(frames).reset_index(drop=True)
    
    # Update `df` in case the trimming inadvertently filtered out ridges (will happen if first_col_idx = 0)
    keep_rows = df_pos_out[[contour_label, "s_imagej"]]
    df_new = df.merge(keep_rows[[contour_label, "s_imagej"]], on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)

    if verbose: print("\tNow removing small ridges (post eig2 trimming)...")
    df_pos_out, df_new = remove_small_ridges(df_pos_out, df_new, remove_min_size, contour_label, verbose)

    # df_pos_out = reorient_ridges(df_pos_out, df_pos_out.groupby([contour_label, "s_imagej"], sort=False), y_label, True)

    return df_new, df_pos_out




def find_scale_idx(scale_range, s_val):
    idxs = np.where(np.isclose(np.round(scale_range, 3), s_val))[0]
    if len(idxs) == 0:
        return np.where(np.isclose(scale_range, s_val))[0][0]
    return idxs[0]

def init_globals_trim_all(im_shape_0_, x_label_, y_label_,
                          C_, A_mask_, W2_, scale_range_,
                          corner_trim_, angle_trim_, eig2_trim_, remove_min_size_):
    # Globals used by worker
    global im_shape_0, x_label, y_label, C, A_mask, W2, scale_range
    global corner_trim, angle_trim, eig2_trim, remove_min_size
    im_shape_0 = im_shape_0_
    x_label = x_label_
    y_label = y_label_
    C = C_
    A_mask = A_mask_
    W2 = W2_
    scale_range = scale_range_
    corner_trim = corner_trim_
    angle_trim = angle_trim_
    eig2_trim = eig2_trim_
    remove_min_size = remove_min_size_


def process_trim_all(df_ridge):
    """
    Single pass per ridge, but applies the three rules SEQUENTIALLY:
      corner -> (maybe drop if too short) -> angle -> (maybe drop) -> eig2 -> (maybe drop)
    Precomputes curves once; later stages operate on the current prefix [ : end_idx ].
    Returns (trimmed_df_or_None, trimmed_corner_flag, trimmed_angle_flag, trimmed_eig2_flag).
    """
    # df_ridge.sort_values(y_label, inplace=True, ignore_index=True, ascending=False)

    L0 = len(df_ridge)
    if L0 <= remove_min_size:
        return None, 0, 0, 0  # drop now

    coords_px = df_ridge[[x_label, y_label]].to_numpy()
    ridge_coords = convert_imagej_coord_to_numpy(coords_px, im_shape_0, flip_y=False, start_bin=0)
    sel_idx = find_scale_idx(scale_range, df_ridge.iloc[0]["s_imagej"])

    # curves once
    corner_line = None
    if corner_trim is not None:
        C_curves = extract_line_scale_space(ridge_coords, [C]) 
        corner_line = (C_curves > 0.25).all(axis=0) # Enforce all

    angle_line = None
    if angle_trim is not None:
        A_curves = round_line_scale_space(ridge_coords, [A_mask]).astype(bool) # Round 
        angle_line = A_curves[sel_idx]

    w2_line = None
    if eig2_trim is not None:
        W2_curves = extract_line_scale_space(ridge_coords, [W2])
        w2_line = W2_curves[sel_idx]

    end_idx = L0
    t_corner = t_angle = t_eig2 = 0  # per-stage trimmed flags

    # CORNER
    if corner_line is not None:
        m = int(L0 * corner_trim) if (isinstance(corner_trim, float) and corner_trim < 1) else int(corner_trim)
        cand = corner_line.copy()
        if m > 0: 
            cand[:m] = False
        if cand[:end_idx].any():
            cut = int(np.flatnonzero(cand[:end_idx])[0])
            if cut < end_idx:
                end_idx = cut
                t_corner = 1
        # stagewise remove_small
        if end_idx <= remove_min_size:
            return None, t_corner, 0, 0

    # ANGLE
    if angle_line is not None and end_idx > 0:
        L1 = end_idx  # current ridge length at this stage
        m = int(L1 * angle_trim) if (isinstance(angle_trim, float) and angle_trim < 1) else int(angle_trim)
        cand = angle_line[:L1].copy()
        if m > 0: 
            cand[:m] = False
        if cand.any():
            cut = int(np.flatnonzero(cand)[0])
            if cut < end_idx:
                end_idx = cut
                t_angle = 1
        if end_idx <= remove_min_size:
            return None, t_corner, t_angle, 0

    # EIG2
    if w2_line is not None and end_idx > 0:
        w = w2_line[:end_idx]

        peaks = argrelmax(w)[0]
        if len(peaks) == 0:
            peak_idx = int(np.argmax(w))
        else:
            peak_idx = 0
            for j in peaks:
                if w[j] >= 0:
                    peak_idx = int(j)
                    break
        if w[peak_idx] < 0:
            peak_idx = 0

        neg_after_peak = np.flatnonzero(w < 0)
        neg_after_peak = neg_after_peak[neg_after_peak >= peak_idx]
        if len(neg_after_peak):
            L2 = end_idx
            m = int(L2 * eig2_trim) if (isinstance(eig2_trim, float) and eig2_trim < 1) else int(eig2_trim)
            neg_after_peak = neg_after_peak[neg_after_peak >= m]
            if len(neg_after_peak):
                cut = int(neg_after_peak[0])
                if cut < end_idx:
                    end_idx = cut
                    t_eig2 = 1
        if end_idx <= remove_min_size:
            return None, t_corner, t_angle, t_eig2

    # keep current prefix, or drop if too small
    if end_idx <= remove_min_size:
        return None, t_corner, t_angle, t_eig2
    return df_ridge.iloc[:end_idx], t_corner, t_angle, t_eig2


def trim_imagej_results_all(df, df_pos, im_shape_0,
    C=None, corner_trim=None,
    A=None, angle_trim=None, angle_range=None, scale_range=None,
    W2=None, eig2_trim=None,
    remove_min_size=1, num_cores=1, verbose=False,
    contour_label="Contour Number", x_label="X_(px)", y_label="Y_(px)"):
    """
    Trims ridges based on the corner, angle, or eig2 (toggled independently by their respective parameters)

    CORNER:
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature 
    of the corner condition is True (i.e. at all scales there is a corner), then the ridge is trimmed
    at that point 

    ANGLE:
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature
    of the angle condition is False (i.e. at the selected scale the angle of the ridge is not in `angle_range`),
    then the ridge is trimmed at that point

    EIG2:
    As the ridge portrudes from the main diagonal (assuming ordered), if scale space signature
    of the eig2 condition is negative (i.e. at the selected scale the eig2 is negative),
    then the ridge is trimmed at that point
    Note that this invariant was mainly designed for datatype='oe' not 'observed'

    Parameters
    ----------
    df : pd.DataFrame
        Summary table of ImageJ results
    df_pos : pd.DataFrame
        Expanded table of ImageJ results
    C : np.ndarray
        The corner tensor of (num_scales, n, m) where image is of shape (n, m)
    A : np.ndarray
        The angle tensor of (num_scales, n, m) where image is of shape (n, m)
    W2 : np.ndarray
        The eig2 tensor of (num_scales, n, m) where image is of shape (n, m)
    im_shape_0 : int
        Binned window size (i.e. im.shape[0])
    corner_trim or eig2_trim or angle_trim : float or int
        Specifies the minimum size of the ridge where no trimming can occur
        If None, then no trimming occurs
        If 0, then trimming can occur at any point in the ridge
        If < 1, then interpret as a fraction of the ridge length where no trimming can occur
        Example: 0.1 means that 10% trimming cannot leave a ridge smaller than 10% of its original length
        If >= 1, then interpret as number of bins (i.e. minimum number of points in the ridge)
        where no trimming can occur
    remove_min_size : int
        The minimum number of points in a ridge. Otherwise, it is discarded
    num_cores : int
        Number of cores to use for parallel processing
    scale_range : list
        List of scales generated

    Returns
    -------
    df : pd.DataFrame
        Updated summary table
    df_pos : pd.DataFrame
        Updated expanded table
    """
    if df is None and df_pos is None:
        if verbose: print("\tSkipping process imagej...")
        return None, None

    # Quick exits if everything disabled
    if corner_trim is None and angle_trim is None and eig2_trim is None:
        if verbose: print("\tNo trimming (all trimming disabled)")
        return df, df_pos

    # A mask once (angle: 'bad' when NOT within range)
    A_mask = None
    if angle_trim is not None:
        if angle_range is None:
            raise ValueError("angle_range must be provided when angle trimming is enabled.")
        A_mask = ~np.logical_and(A > angle_range[0], A < angle_range[1])

    init_globals_trim_all(
        im_shape_0, x_label, y_label,
        C, A_mask, W2, np.array(scale_range),
        corner_trim, angle_trim, eig2_trim,
        remove_min_size)

    # Group once
    gb = df_pos.groupby([contour_label, "s_imagej"], sort=False)

    frames = []
    n_corner = 0
    n_angle = 0
    n_eig2 = 0
    n_none = 0
    total_groups = len(df)

    if num_cores == 1:
        for _, df_ridge in tqdm(gb, disable=not verbose):
            out, c, a, e2 = process_trim_all(df_ridge.copy(deep=False))
            n_corner += int(bool(c))
            n_angle += int(bool(a))
            n_eig2 += int(bool(e2))
            if not (c or a or e2): 
                n_none += 1
            if out is not None:
                frames.append(out)
    else:
        groups = list(gb)
        total = len(groups)
        results = []
        with Pool(
            num_cores,
            initializer=init_globals_trim_all,
            initargs=(im_shape_0, x_label, y_label, C, A_mask, W2, np.array(scale_range),
                      corner_trim, angle_trim, eig2_trim, remove_min_size)
        ) as pool:
            for out, c, a, e2 in tqdm(pool.imap_unordered(process_trim_all, (g[1] for g in groups), chunksize=8), 
                                      total=total, disable=not verbose):
                results.append((out, c, a, e2))
        for out, c, a, e2 in results:
            n_corner += int(bool(c))
            n_angle += int(bool(a))
            n_eig2 += int(bool(e2))
            if not (c or a or e2): 
                n_none += 1
            if out is not None:
                frames.append(out)

    # Assemble once
    df_pos_out = pd.concat(frames, axis=0, ignore_index=True)

    # Update summary table
    keep_keys = df_pos_out[[contour_label, "s_imagej"]].drop_duplicates()
    df_new = df.merge(keep_keys, on=[contour_label, "s_imagej"], how="inner")
    df_new.drop_duplicates([contour_label, "s_imagej"], inplace=True, ignore_index=True)

    if verbose:
        print(f"\tTrimmed corner: {n_corner} angle: {n_angle}  "
              f"eig2: {n_eig2} none: {n_none} (total: {total_groups})")
        print("\tNow removing small ridges (post trimming)...")

    # Remove small ridges after all trimming
    df_pos_out, df_new = remove_small_ridges(df_pos_out, df_new, remove_min_size, contour_label, verbose)

    return df_new, df_pos_out
