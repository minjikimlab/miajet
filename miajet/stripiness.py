import numpy as np
from tqdm import tqdm
import hicstraw
from utils.plotting import convert_imagej_coord_to_numpy
from .compute_p_value import compute_test_statistic_quantities

K_XL = np.array([[-1,  1],
                 [-2,  2],
                 [-1,  1]], dtype=float)

K_XR = np.array([[ 1, -1],
                 [ 2, -2],
                 [ 1, -1]], dtype=float)

K_Y  = np.array([[ 1,  2,  1],
                 [ 0,  0,  0],
                 [-1, -2, -1]], dtype=float)


def generalized_stripiness(L, C, R, p):
    n = len(C)
    if n < 3:
        print("\tWarning: stripiness cannot be computed for less than 3 points. Returning 0")
        return 0
    
    delta_G = np.full(n, np.nan)

    for i in range(1, n - 1):
        patch_xL = np.array([[L[i-1], C[i-1]],
                             [L[i],   C[i]],
                             [L[i+1], C[i+1]]])

        patch_xR = np.array([[C[i-1], R[i-1]],
                             [C[i],   R[i]],
                             [C[i+1], R[i+1]]])

        patch_y  = np.array([[L[i-1], C[i-1], R[i-1]],
                             [L[i],   C[i],   R[i]],
                             [L[i+1], C[i+1], R[i+1]]])

        G_xL = np.sum(K_XL * patch_xL)
        G_xR = np.sum(K_XR * patch_xR)
        G_y = np.sum(K_Y  * patch_y)

        delta_G[i] = min(G_xL, G_xR) - abs(G_y)

    valid = ~np.isnan(delta_G)
    if np.sum(valid) == 0:
        print("\tWarning: stripiness cannot be computed for all points being NaN. Returning 0")
        return 0
    
    # Effective length i.e. after removing nans
    n_eff = np.sum(valid) 
    # G_p = (np.sum(delta_G[valid] ** p)  / n_eff) ** (1 / p)
    G_p = (np.sum(delta_G[valid] ** p)  / n) ** (1 / p) # paper divided by n not n_eff
    M = np.median(C)

    S = M * G_p

    return S


from scipy.ndimage import sobel

def steer_filter_1(angle, dx, dy):
    """
    Angle must be in degrees
    """
    theta = np.radians(angle)
    return np.cos(theta) * dx + np.sin(theta) * dy

def process_stripiness_single(df_ridge, im_oe,
                              factor_lr, p_norm, max_width_bin=None,
                              x_label="X_(px)_unmap",
                              y_label="Y_(px)_unmap"):
    
    # if df_ridge["Contour Number"].values[0] == 39 and df_ridge["s_imagej"].values[0].round(3) == 3.886:
    #     pass
    
    orig_idx = int(df_ridge["_orig_idx"].iloc[0])
    coords = df_ridge[[x_label, y_label]].values
    ridge_pts = convert_imagej_coord_to_numpy(coords,
                                              im_oe.shape[0],
                                              flip_y=False, # Indexing mode
                                              start_bin=0)
    ridge_angles = -df_ridge["angle_imagej"].values - 90
    ridge_widths = df_ridge["width"].values

    # Clip if specified
    if max_width_bin is not None:
        ridge_widths = np.clip(ridge_widths, 0, max_width_bin)

    # height_ratio = 1
    # ridge_heights = np.clip(ridge_widths * height_ratio, 1, None)
    ridge_heights = 1

    l_mean_oe, r_mean_oe, c_mean_oe, *_, l_med_oe, r_med_oe, c_med_oe = \
        compute_test_statistic_quantities(im=im_oe,
                                          ridge_points=ridge_pts,
                                          ridge_angles=ridge_angles,
                                          width_in=ridge_widths,
                                          height=ridge_heights, # same as width!
                                          im_shape=im_oe.shape,
                                          factor_lr=factor_lr)
    
    # return orig_idx, generalized_stripiness(L=l_mean_oe, C=c_mean_oe, R=r_mean_oe, p=p_norm)

    Ix = sobel(im_oe, axis=1)
    Iy = sobel(im_oe, axis=0)
    avg_normal = np.mean(ridge_angles) + 90 
    I_rot = steer_filter_1(avg_normal, Ix, Iy)

    l_mean_noise, r_mean_noise, c_mean_noise, *_ = \
    compute_test_statistic_quantities(im=I_rot,
                                        ridge_points=ridge_pts,
                                        ridge_angles=ridge_angles,
                                        width_in=ridge_widths,
                                        height=ridge_heights, # same as width!
                                        im_shape=im_oe.shape,
                                        factor_lr=factor_lr)

    # #### NORMALIZED ####
    # l_mean_ratio = l_mean_oe - l_mean_noise
    # r_mean_ratio = r_mean_oe - r_mean_noise
    # c_mean_ratio = c_mean_oe - c_mean_noise  

    # S = generalized_stripiness(L=l_mean_ratio, C=c_mean_ratio, R=r_mean_ratio, p=p_norm) 

    # return orig_idx, S

    # #### NORMALIZED 2 ####
    total_energy = np.sqrt(np.sum(l_mean_noise ** 2) + np.sum(r_mean_noise ** 2) + np.sum(c_mean_noise ** 2))

    S = generalized_stripiness(L=l_mean_oe, C=c_mean_oe, R=r_mean_oe, p=p_norm) / total_energy
    
    return orig_idx, S


def compute_stripiness(df_agg, df_features, im_oe,
                       p_norm,
                       stripiness_factor_lr,
                       max_width_bin,
                       contour_label="Contour Number",
                       x_label="X_(px)_unmap",
                       y_label="Y_(px)_unmap",
                       num_cores=1,
                       verbose=False):
    
    # tag each ridge with its original index
    df_keys = df_agg[[contour_label, "s_imagej"]].copy()
    df_keys["_orig_idx"] = np.arange(len(df_keys))

    # merge in positional features
    df_merge = df_keys.merge(df_features[[contour_label, "s_imagej", x_label, y_label, "angle_imagej", "width"]],
                             on=[contour_label, "s_imagej"],
                             how="inner")
    
    # add checks
    if df_merge["_orig_idx"].nunique() < len(df_keys):
        missing = df_keys.loc[~df_keys["_orig_idx"].isin(df_merge["_orig_idx"])]
        msg = (
            f"compute_stripiness: df_features lacks rows for "
            f"{len(missing)} ridge(s). Missing keys:\n"
            + missing[[contour_label, "s_imagej"]].to_string(index=False)
        )
        raise ValueError(msg)


    n = len(df_agg)
    stripiness_arr = np.zeros(n)

    # group by ridge
    gb = df_merge.groupby("_orig_idx", sort=False)


    if num_cores == 1:
        # single-core processing
        for orig_idx, df_ridge in tqdm(gb, disable=not verbose):

            idx, s_val = process_stripiness_single(df_ridge, 
                                                   im_oe=im_oe, 
                                                   factor_lr=stripiness_factor_lr,
                                                   p_norm=p_norm,
                                                   max_width_bin=max_width_bin,
                                                   x_label=x_label,
                                                   y_label=y_label)
            stripiness_arr[idx] = s_val


    # assemble output
    df_out = df_agg.copy()
    df_out["stripiness"] = stripiness_arr

    return df_out


# def extract_expected_vector(hic_file, chrom, data_type, resolution):
#     hic = hicstraw.HiCFile(hic_file)

#     # Same method as `read_hic_file`
#     for chromosome in hic.getChromosomes():
#         if str(chrom) == chromosome.name or ("chr" not in chromosome.name and chromosome.name == str(chrom)[3:]):
#             key = chromosome.name
#             found = True
#             end_pos = chromosome.length

#     mzd = hic.getMatrixZoomData(key, key, data_type, "oe", "BP", int(resolution))

#     return mzd.getExpectedValues()