from shapely.geometry import box, Point
from shapely.affinity import rotate, translate
from scipy.stats import ks_2samp, kstest, wilcoxon, ttest_rel
import numpy as np
from utils.plotting import convert_imagej_coord_to_numpy
from tqdm import tqdm
from multiprocessing import Pool

def create_rotated_box(center, width, height, angle, use_radians=False):
    """
    Creates a rectangle (as a Shapely polygon) of size (width x height) 
    centered at 'center' and rotated by 'angle'.
    
    Parameters:
    - center: Tuple (x, y) for the center of the box.
    - width: The width of the box (long dimension).
    - height: The height of the box (short dimension).
    - angle: The rotation angle (in degrees by default, unless use_radians=True).
    - use_radians: Set to True if the angle is provided in radians.
    
    Returns:
    - A Shapely polygon representing the rotated rectangle.
    """
    # Create an axis-aligned rectangle centered at the origin
    rect = box(-width/2, -height/2, width/2, height/2)
    
    # Rotate the rectangle around the origin
    rect_rotated = rotate(rect, angle, origin=(0, 0), use_radians=use_radians)
    
    # Translate the rotated rectangle to the desired center
    rect_translated = translate(rect_rotated, xoff=center[0], yoff=center[1])
    return rect_translated



def compute_boxes(point, angle, w, h, factor_lr):
    """
    Given a point, an angle (in degrees), a fixed box width 'w' (the long dimension), and
    an offset distance, this function returns three rotated boxes:
    
    1. Center box: 1 x w box centered at the point.
    2. Right box: 1 x w box centered at (x + offset, y + offset) computed along the given angle.
    3. Left box: 1 x w box centered at (x - offset, y - offset).
    
    Parameters:
    - point: Tuple (xi, yi) for the center point.
    - angle: The rotation angle in degrees.
    - w: The long dimension of the box (box size is defined as 1 x w).
    
    Returns:
    - A tuple with (center_box, right_box, left_box) as Shapely polygons.
    
    Note:
    The function uses numpy (np) for trigonometric calculations.
    It assumes that a function 'create_rotated_box' is defined elsewhere.
    """
    # Create the center box
    center_box = create_rotated_box(point, w, h, angle, use_radians=False)
    
    # Compute the offset vector using numpy
    offset_angle_rad = np.radians(angle)
    dx = w * np.cos(offset_angle_rad)
    dy = w * np.sin(offset_angle_rad)
    
    # Define centers for right and left boxes
    right_center = (point[0] + dx, point[1] + dy)
    left_center  = (point[0] - dx, point[1] - dy)
    
    # Create the right and left boxes using the same dimensions and rotation angle
    right_box = create_rotated_box(right_center, w * factor_lr, h, angle, use_radians=False)
    left_box  = create_rotated_box(left_center, w * factor_lr, h, angle, use_radians=False)
    
    return center_box, right_box, left_box



def extract_box_indices(polygon):
    """
    Given a Shapely polygon (representing a box), returns:
    
    - interior_indices: a list of (x, y) integer pixel coordinates lying inside the polygon.
    - exterior_indices: a list of (x, y) coordinates representing the polygon's vertices.
    
    This is useful to extract the proper image indices corresponding to the entire box.
    """
    minx, miny, maxx, maxy = polygon.bounds
    interior_indices = []
    # Loop over integer coordinates in the bounding box
    for x in range(int(np.floor(minx)), int(np.ceil(maxx)) + 1):
        for y in range(int(np.floor(miny)), int(np.ceil(maxy)) + 1):
            if polygon.contains(Point(x, y)):
                interior_indices.append((x, y))
    # exterior_indices = list(polygon.exterior.coords)
    # return np.array(interior_indices + exterior_indices)
    return np.array(interior_indices)


def clip_indices(indices, im_shape):
    """
    Clips a numpy array of indices so that all (x, y) pairs fall within the bounds of an image
    """
    clipped = indices.copy()
    # x corresponds to columns: valid range is 0 to image_shape[1] - 1
    clipped[:, 0] = np.clip(clipped[:, 0], 0, im_shape[1] - 1)
    # y corresponds to rows: valid range is 0 to image_shape[0] - 1
    clipped[:, 1] = np.clip(clipped[:, 1], 0, im_shape[0] - 1)
    return clipped


def compute_test_statistic_quantities(im, ridge_points, ridge_angles, width_in, height, im_shape, factor_lr=1):
    """
    Computes test statistic quantities for the ridge according to the image intensity values in `im`

    Parameters
    ----------
    im : np.ndarray
        Contact map
    ridge_points : np.ndarray
        Array of shape (n, 2) with ridge points in the numpy format (i.e. output of `convert_imagej_coord_to_numpy`)
    ridge_angles : np.ndarray
        Array of shape (n,) with angles in degrees corresponding to each ridge point
    width_in : float or list of floats
        Width of the center box. If a list, each ridge point can have a different width.
    height : float
        Height of the boxes
    im_shape : tuple
        Shape of the image (height, width)
    factor_lr : float, optional
        Factor to scale the width of the left and right boxes relative to the center box
        Default is 1, means that center width is the same as left and right widths
    
    Returns
    -------
    left_means, right_means, center_means : np.ndarray
        Mean values of the left, right, and center boxes
    left_box_coords, right_box_coords, center_box_coords : list of lists
        Coordinates of the left, right, and center boxes as lists of (x, y) tuples
    left_num_points, right_num_points, center_num_points : list of int
        Number of points in the left, right, and center boxes
    left_vals, right_vals, center_vals : list of np.ndarray
        Intensity values from the left, right, and center boxes
    left_medians, right_medians, center_medians : np.ndarray
        Median values of the left, right, and center boxes
    """
    center_box_coords = []
    right_box_coords = []
    left_box_coords = []
    
    center_num_points = []
    right_num_points = []
    left_num_points = []

    center_means = []
    right_means = []
    left_means = []

    center_medians = []
    right_medians = []
    left_medians = []

    center_vals = []
    right_vals = []
    left_vals = []

    if isinstance(width_in, list) or isinstance(width_in, np.ndarray):
        width = np.clip(width_in, a_min=1.5, a_max=None) # 04/03/25 minimum width is 1.5
    else:
        width = width_in

    for i, (pt, ang) in enumerate(zip(ridge_points, ridge_angles)):

        if isinstance(width, list) or isinstance(width, np.ndarray):
            C, R, L = compute_boxes(point=pt, angle=ang, w=width[i], h=height, factor_lr=factor_lr)
        else:
            C, R, L = compute_boxes(point=pt, angle=ang, w=width, h=height, factor_lr=factor_lr)
        
        center_box_coords.append(list(C.exterior.coords))
        right_box_coords.append(list(R.exterior.coords))
        left_box_coords.append(list(L.exterior.coords))
        
        C_idx = clip_indices(extract_box_indices(C), im_shape)
        R_idx = clip_indices(extract_box_indices(R), im_shape)

        if len(extract_box_indices(L)) == 0:
            pass
        L_idx = clip_indices(extract_box_indices(L), im_shape)

        center_num_points.append(len(C_idx))
        right_num_points.append(len(R_idx))
        left_num_points.append(len(L_idx))
        
        C_vals = im[C_idx[:, 1], C_idx[:, 0]]
        R_vals = im[R_idx[:, 1], R_idx[:, 0]]
        L_vals = im[L_idx[:, 1], L_idx[:, 0]]
        
        center_means.append(np.mean(C_vals))
        right_means.append(np.mean(R_vals))
        left_means.append(np.mean(L_vals))

        center_medians.append(np.median(C_vals))
        right_medians.append(np.median(R_vals))
        left_medians.append(np.median(L_vals))

        center_vals.append(C_vals.flatten())
        right_vals.append(R_vals.flatten())
        left_vals.append(L_vals.flatten())


    right_means = np.array(right_means)
    left_means = np.array(left_means)
    center_means = np.array(center_means)

    center_medians = np.array(center_medians)
    right_medians = np.array(right_medians)
    left_medians = np.array(left_medians)

    return left_means, right_means, center_means, \
           left_box_coords, right_box_coords, center_box_coords, \
           left_num_points, right_num_points, center_num_points, \
            left_vals, right_vals, center_vals, \
           left_medians, right_medians, center_medians


def compute_test_statistic(left_means, right_means, center_means, pseudo_count=0.1):
    """
    Computes the test statistic based on the means (or medians) of the left, right, and center boxes

    Returns all 3 statistics:
    - CR_subtract: Center box - Average of Left and Right boxes
    - CR_ratio: Center box / Average of Left and Right boxes
    - C2R_ratio: Center box^2 / Average of Left and Right boxes
    """
    # combined = np.concatenate((left_means, right_means))
    LR_average = np.mean(np.vstack((left_means, right_means)), axis=0)

    CR_subtract = center_means - LR_average
    CR_ratio = (center_means + pseudo_count) / (LR_average + pseudo_count)
    C2R_ratio = (center_means + pseudo_count) ** 2 / (LR_average + pseudo_count)

    return CR_subtract, CR_ratio, C2R_ratio


def init_globals_significance(im_p_value_, corr_im_p_value_, factor_lr_, agg_, statistic_,
                              contour_label_, x_label_, y_label_):
    """
    Helper function for `compute_significance` to initialize global variables
    This is used for parallel processing to avoid passing large objects back and forth
    """
    global im_p_value, corr_im_p_value, factor_lr, agg, statistic
    global contour_label, x_label, y_label
    im_p_value = im_p_value_
    corr_im_p_value = corr_im_p_value_
    factor_lr = factor_lr_
    agg = agg_
    statistic = statistic_
    contour_label = contour_label_
    x_label = x_label_
    y_label = y_label_

def process_significance(df_ridge):
    """
    Helper function for `compute_significance` to process each ridge in parallel

    Expects df_ridge to have a column '_orig_idx' indicating its position in df_agg
    Returns tuple: (orig_idx, ks_stat, p_val)
    """
    orig_idx = int(df_ridge["_orig_idx"].iloc[0])
    coords = df_ridge[[x_label, y_label]].values
    ridge_pts = convert_imagej_coord_to_numpy(coords,
                                              im_p_value.shape[0],
                                              flip_y=False,
                                              start_bin=0)
    ridge_angles = -df_ridge["angle_imagej"].values - 90
    ridge_widths = df_ridge["width"].values

    # observed vs. null
    l_mean_obs, r_mean_obs, c_mean_obs, *_, l_med_obs, r_med_obs, c_med_obs = \
        compute_test_statistic_quantities(im=im_p_value,
                                          ridge_points=ridge_pts,
                                          ridge_angles=ridge_angles,
                                          width_in=ridge_widths,
                                          height=1,
                                          im_shape=im_p_value.shape,
                                          factor_lr=factor_lr)
    l_mean_null, r_mean_null, c_mean_null, *_, l_med_null, r_med_null, c_med_null = \
        compute_test_statistic_quantities(im=corr_im_p_value,
                                          ridge_points=ridge_pts,
                                          ridge_angles=ridge_angles,
                                          width_in=ridge_widths,
                                          height=1,
                                          im_shape=im_p_value.shape,
                                          factor_lr=factor_lr)

    # build mean-based stats
    mean_CR_sub_obs, mean_CR_ratio_obs, mean_C2R_ratio_obs = compute_test_statistic(
        l_mean_obs, r_mean_obs, c_mean_obs)
    mean_CR_sub_null, mean_CR_ratio_null, mean_C2R_ratio_null = compute_test_statistic(
        l_mean_null, r_mean_null, c_mean_null)

    # build median-based stats
    med_CR_sub_obs, med_CR_ratio_obs, med_C2R_ratio_obs = compute_test_statistic(
        l_med_obs, r_med_obs, c_med_obs)
    med_CR_sub_null, med_CR_ratio_null, med_C2R_ratio_null = compute_test_statistic(
        l_med_null, r_med_null, c_med_null)

    # select arrays
    if agg == "mean":
        if statistic == 1:
            obs_arr, null_arr = mean_CR_ratio_obs,   mean_CR_ratio_null
        elif statistic == 2:
            obs_arr, null_arr = mean_CR_sub_obs,     mean_CR_sub_null
        elif statistic == 3:
            obs_arr, null_arr = mean_C2R_ratio_obs,  mean_C2R_ratio_null
        else:
            raise ValueError(f"Invalid statistic: {statistic}")
    else:  # median
        if statistic == 1:
            obs_arr, null_arr = med_CR_ratio_obs,    med_CR_ratio_null
        elif statistic == 2:
            obs_arr, null_arr = med_CR_sub_obs,      med_CR_sub_null
        elif statistic == 3:
            obs_arr, null_arr = med_C2R_ratio_obs,   med_C2R_ratio_null
        else:
            raise ValueError(f"Invalid statistic: {statistic}")

    test_statistic, p_val = ks_2samp(obs_arr, null_arr, nan_policy="omit", alternative="less")
    # test_statistic, p_val = wilcoxon(obs_arr - null_arr, alternative='greater', nan_policy='omit')
    # test_statistic, p_val = kstest(obs_arr - null_arr, lambda x : np.where(x < 0, 0.0, 1.0), alternative='less', nan_policy='omit')
    # test_statistic, p_val = ttest_rel(obs_arr, null_arr, alternative='greater', nan_policy='omit') # best for NE

    return orig_idx, test_statistic, p_val

def compute_significance(df_agg, df_features, im_p_value, corr_im_p_value, agg, statistic, factor_lr=1,
                         contour_label="Contour Number",
                         x_label="X_(px)_unmap",
                         y_label="Y_(px)_unmap",
                         num_cores=1,
                         verbose=False):
    """
    Computes significance of ridge features with 
    * observed values as `im_p_value`
    * null values as `corr_im_p_value`
    using the KS test.

    The statistic can differ and can be one of:
    - 1: Center box / Avg(Left box, Right box) 
    - 2: Center box - Avg(Left box, Right box)
    - 3: Center box^2 / Avg(Left box, Right box)

    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe containing aggregated features
    df_features : pd.DataFrame
        Expanded dataframe with features
    im_p_value : np.ndarray
        Observed image values for significance testing
        The datatype is always log of "observed" regardless of the user data_type parameter specification
    corr_im_p_value : np.ndarray
        Null image values for significance testing
        The datatype is always correlation of "observed" regardless of the user data_type parameter specification
    agg : str
        Aggregation method for the boxes, either "mean" or "median"
    statistic : int
        Statistic to compute:
        - 1: Center box / Avg(Left box, Right box)
        - 2: Center box - Avg(Left box, Right box)
        - 3: Center box^2 / Avg(Left box, Right box)
    factor_lr : float, optional
        Factor to scale the width of the left and right boxes relative to the center box.
        Default is 1, meaning no scaling
    num_cores : int, optional
        Number of CPU cores to use for parallel processing

    Returns
    -------
    pd.DataFrame
        Summary dataframe with the additional columns:
        - "ks": KS statistic for each ridge
        - "p-val": p-value for each ridge    
    """
    # 1) tag each ridge with its original index
    df_keys = df_agg[[contour_label, "s_imagej"]].copy()
    df_keys["_orig_idx"] = np.arange(len(df_keys))

    # 2) merge in positional features
    df_merge = df_keys.merge(df_features[[contour_label, "s_imagej", x_label, y_label, "angle_imagej", "width"]],
                             on=[contour_label, "s_imagej"],
                             how="inner")

    n = len(df_agg)
    ks_stat_arr = np.zeros(n)
    p_val_arr   = np.zeros(n)

    # group by ridge
    gb = df_merge.groupby("_orig_idx", sort=False)

    if num_cores == 1:
        # single-core processing
        for orig_idx, df_ridge in tqdm(gb, disable=not verbose):
            idx, ks_stat, p_val = process_significance(df_ridge)
            ks_stat_arr[idx] = ks_stat
            p_val_arr[idx] = p_val
    else:
        # prepare for multi-core
        groups = [df for _, df in gb]
        total = len(groups)

        with Pool(processes=num_cores,
                  initializer=init_globals_significance,
                  initargs=(im_p_value,
                            corr_im_p_value,
                            factor_lr,
                            agg,
                            statistic,
                            contour_label,
                            x_label,
                            y_label)) as pool:
            for idx, ks_stat, p_val in tqdm(
                pool.imap_unordered(process_significance, groups),
                total=total,
                disable=not verbose
            ):
                ks_stat_arr[idx] = ks_stat
                p_val_arr[idx]  = p_val

    # assemble output
    df_out = df_agg.copy()
    df_out["ks"] = ks_stat_arr
    df_out["p-val"] = p_val_arr

    return df_out


from statsmodels.stats.multitest import multipletests


def correct_significance(df_agg, method="fdr_bh"):
    """
    Corrects p-values in column "p-val" using the specified method from statsmodels
    * Note that the input dataframe is a summary table which is important 
    in that we have one row per ridge 

    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe containing a "p-val" column 
    method : str, optional
        Method for p-value correction from statsmodels.stats.multitest.multipletests
        Default is "fdr_bh"

    Returns
    -------
    pd.DataFrame
        Summary dataframe an additional "p-val_corr" column containing corrected p-values
    """
    
    _, p_values_corrected, _, _ = multipletests(df_agg["p-val"].values, method=method)
    df_agg["p-val_corr"] = p_values_corrected

    return df_agg



def threshold_significance(df_agg_in, alpha_range, verbose=False):
    """
    Filter rows of a DataFrame by a p-value cutoff provided in `alpha_range`

    If `alpha_range` is a single float, 
        Returns a list containing one dataframe
    with all rows where "p-val_corr" <= alpha_range
    If `alpha_range` is a list of floats, 
    Returns a list of dataframe, where each is a thresholded dataframe 

    Parameters
    ----------
    df_agg_in : pd.DataFrame
        Summary dataframe. Must contain "p-val_corr" column 
    alpha_range : float or list of float
        Significance threshold(s). Rows with p-val_corr ≤ threshold are kept

    Returns
    -------
    List[pd.DataFrame]
        A list of filtered DataFrames, one per `alpha` in `alpha_range`
    """
    n = len(df_agg_in)

    if not isinstance(alpha_range, list):
        # make a copy
        df_agg = df_agg_in.loc[df_agg_in["p-val_corr"] <= alpha_range].reset_index(drop=True)  

        if verbose:
            print(f"\t{len(df_agg)} / {n} ridges remaining after thresholding at alpha = {alpha_range}...")

        return [df_agg]

    df_agg_alpha = []
    for alpha in alpha_range:
        df_agg = df_agg_in.loc[df_agg_in["p-val_corr"] <= alpha].reset_index(drop=True)  
        df_agg_alpha.append(df_agg)

        if verbose:
            print(f"\t * alpha = {alpha} : {len(df_agg)} / {n} ridges remaining...")

    return df_agg_alpha


                



    



    