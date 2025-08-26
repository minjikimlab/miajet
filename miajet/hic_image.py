import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import scipy
from scipy.ndimage import rotate

from utils.processing import read_hic_rectangle, read_hic_corr_rectangle, read_hic_network_enhancement, \
    read_hic_file, remove_zero_sum, whiten_matrix, scalar_products
from utils.plotting import save_histogram
import copy


def zero_outside_window(mat_in, window_size_bin):
    mat = np.copy(mat_in)
    window_size_buffer = min(window_size_bin + 5, mat.shape[0])
    mat[np.triu_indices_from(mat, k=window_size_buffer)] = 0
    mat[np.tril_indices_from(mat, k=-1)] = 0
    return mat


def compute_rm_idx(mat, window_size_bin, verbose=False):
    mat_rm = np.copy(mat)
    mat_rm = zero_outside_window(mat_rm, window_size_bin)
    _, rm_idx = remove_zero_sum(mat_rm, verbose=verbose)
    return rm_idx

def clip_and_normalize(image, vmin_perc, vmax_perc):
    vmin_perc_eff = vmin_perc
    vmax_perc_eff = vmax_perc
    if np.percentile(image, vmin_perc) == np.percentile(image, vmax_perc):
        vmin_perc_eff = 0
        vmax_perc_eff = 100
    image = np.clip(image, np.percentile(image, vmin_perc_eff), np.percentile(image, vmax_perc_eff))
    image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return image


def rotate_stack_extract_rectangle(mats, rotate_mode, cval, center, window_size_bin_rect):
    """
    Rotate a stack of 2D arrays (C, H, W) once using scipy.ndimage.rotate across (H, W).
    Returns a list of rotated 2D arrays preserving order
    """
    stack = np.stack(mats, axis=0) # (C, H, W)
    stack_rot = rotate(stack, 45, axes=(1, 2), reshape=True, order=1, mode=rotate_mode, cval=cval)
    stack_rot = stack_rot[:, center-window_size_bin_rect:center, :]
    return [stack_rot[i] for i in range(stack_rot.shape[0])]

def generate_hic_bundle(hic_file, chromosome, resolution, window_size, data_type, normalization, whiten,
                        rotation_padding, save_path, verbose, root,
                        im_vmax_perc, im_vmin_perc, corner_vmax_perc, corner_vmin_perc):
    """
    Generate all downstream images with a single rotation pass for shared shapes:
    - im: primary contact map (oe or observed)
    - im_oe: always OE (used for stripiness and plotting, currently deprecated)
    - im_p_value: log10(observed+1)
    - im_corner: correlation of OE (zero_before_corr=True)
    - corr_im_p_value: correlation of observed (zero_before_corr=False)
    - im_orig: same as `im` data_type but without zero-sum removal (rotated separately)
    Returns also rm_idx, save_name, and N (post-removal).
    """
    save_name = os.path.join(save_path, f"{root}_contact_map.jpg")
    window_size_bin = np.ceil(window_size / resolution).astype(int)

    # Single access via hic-straw
    mat_obs = read_hic_file(filename=hic_file, chrom=chromosome, resolution=resolution,
                            positions="all", data_type="observed", normalization=normalization, verbose=verbose)
    mat_oe = read_hic_file(filename=hic_file, chrom=chromosome, resolution=resolution,
                           positions="all", data_type="oe", normalization=normalization, verbose=verbose)

    if mat_obs.shape == (1, 1) or mat_oe.shape == (1, 1):
        raise ValueError(f".hic file read in error. This is most likely due to the normalization vectors not being present in the .hic file. "
                         f"Suggestion: retry with a different normalization method than '{normalization}'")

    mat_obs = mat_obs.astype(np.float32)
    mat_oe = mat_oe.astype(np.float32)

    # Fill nans with 0
    mat_obs[np.isnan(mat_obs)] = 0
    mat_oe[np.isnan(mat_oe)] = 0

    # Determine the common removed indices 
    rm_idx = compute_rm_idx(mat_obs, window_size_bin, verbose=verbose)
    mat_obs_sub = np.delete(np.delete(mat_obs, rm_idx, axis=0), rm_idx, axis=1)
    mat_oe_sub = np.delete(np.delete(mat_oe, rm_idx, axis=0), rm_idx, axis=1)

    if mat_obs_sub.shape[0] == 0 or mat_obs_sub.shape[1] == 0:
        raise ValueError(f"Empty contact map generated for chromosome {chromosome} with resolution {resolution} and window size {window_size}. "
                         f"Please check the Hi-C file {hic_file} for coverage and/or parameters")

    if data_type == "observed":
        im_sq = np.log10(mat_obs_sub + 1)
        if whiten is not None:
            # If not None, then interpret the `whiten` parameter as the epsilon parameter
            im_sq = cv.normalize(im_sq, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            im_sq, _ = whiten_matrix(A_for_corr=im_sq, A_for_whiten=im_sq, epsilon=whiten)
    elif data_type == "oe":
        im_sq = mat_oe_sub.copy()
    else:
        raise ValueError(f"data_type {data_type} not supported. Use 'oe' or 'observed'.")
    

    # IMAGE
    im_sq = clip_and_normalize(im_sq, im_vmin_perc, im_vmax_perc)

    # P-VALUE OBSERVED
    im_pval_sq = np.log10(mat_obs_sub + 1)
    im_pval_sq = clip_and_normalize(im_pval_sq, im_vmin_perc, im_vmax_perc)

    # IMAGE CORNER (correlation of oe)
    coe_sq = mat_oe_sub.copy()
    coe_sq = zero_outside_window(coe_sq, window_size_bin) # TEST
    coe_sq = np.log10(coe_sq + 1)
    coe_sq = clip_and_normalize(coe_sq, corner_vmin_perc, corner_vmax_perc)
    coe_sq = scalar_products(coe_sq, out="correlation")
    coe_sq = cv.normalize(coe_sq, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # P-VALUE NULL
    cob_sq = np.log10(mat_obs_sub + 1)
    cob_sq = clip_and_normalize(cob_sq, im_vmin_perc, im_vmax_perc)
    # cob_sq = zero_outside_window(cob_sq, window_size_bin) # new addition
    cob_sq = scalar_products(cob_sq, out="correlation")
    cob_sq = cv.normalize(cob_sq, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # ROTATION CODE
    # Some basic statistics for the rotation code
    N = im_sq.shape[0]
    window_size_bin_rect = np.ceil(window_size_bin / np.sqrt(2)).astype(int)
    center = np.ceil(N * np.sqrt(2) / 2).astype(int)

    # Single rotation with all images stacked as channel dim
    mats_to_rotate = [im_sq, im_pval_sq, coe_sq, cob_sq]
    rot_list = rotate_stack_extract_rectangle(mats_to_rotate, rotate_mode=rotation_padding, cval=0, 
                                              center=center, window_size_bin_rect=window_size_bin_rect)
    im, im_p_value, im_corner, corr_im_p_value = rot_list

    # Save statistics histogram for the IMAGE only
    plt.imsave(save_name, im, cmap="gray", vmax=np.percentile(im, im_vmax_perc), vmin=np.percentile(im, im_vmin_perc))
    save_histogram(im, save_path, file_name=f"{root}_contact_map_intensity_value_histogram.jpg", vmin_perc=im_vmin_perc, vmax_perc=im_vmax_perc)

    # Finally, return the image that has no 0 sum rows/columns removed (i.e. fully intact matrix)
    if data_type == "oe":
        mat_orig = mat_oe 
    else:
        mat_orig = np.log10(mat_obs + 1)

    # Rotation statistics 
    N_orig = mat_orig.shape[0]
    center_orig = np.ceil(N_orig * np.sqrt(2) / 2).astype(int)
    window_size_bin_rect_orig = np.ceil(window_size_bin / np.sqrt(2)).astype(int)
    # Rotation and extract
    mat_orig_rot = rotate(mat_orig, 45, reshape=True, order=1, mode=rotation_padding, cval=0)
    im_orig = mat_orig_rot[center_orig-window_size_bin_rect_orig:center_orig, :]
    im_orig = clip_and_normalize(im_orig, im_vmin_perc, im_vmax_perc)

    return im, im_orig, im_p_value, im_corner, corr_im_p_value, rm_idx, save_name, N


def compute_edge_strength(im):
    """
    Computes the edge strength of the image using Sobel filters
    
    Parameters
    ----------
    im : np.ndarray
        The contact map image as a 2D numpy array
    
    Returns
    -------
    edge_strength : np.ndarray
        The edge strength of the image as a 2D numpy array
    """
    Ix = scipy.ndimage.sobel(im, axis=1, mode='reflect') # dx
    Iy = scipy.ndimage.sobel(im, axis=0, mode='reflect') # dy    
    return Ix**2 + Iy**2


def check_im_corner_vmin_vmax(im, config_in):
    '''
    Checks if the image intensity values are all the same with the current im_corner_vmin and im_corner_vmax percentiles
    If they are, sets im_corner_vmin and im_corner_vmax to 0 and 100, respectively

    Parameters
    ----------
    im : np.ndarray
        The contact map image
    config_in : dict
        Configuration dictionary containing the vmin and vmax percentiles
    Returns
    -------
    config : dict
        Updated configuration dictionary with vmin and vmax set to 0 and 100, respectively, if the image intensity values are all the same
    '''
    config = copy.deepcopy(config_in)

    if np.percentile(im, config.im_corner_vmin) == np.percentile(im, config.im_corner_vmax):
        
        print(f"\tWarning: Image corner intensity values are all the same...")
        print(f"\tSetting corner_vmin and corner_vmax to 0 and 100, respectively")

        config.im_corner_vmin = 0
        config.im_corner_vmax = 100

    return config



def check_im_vmin_vmax(im, config_in):
    '''
    Checks if the image intensity values are all the same with the current vmin and vmax percentiles
    If they are, sets vmin and vmax to 0 and 100, respectively

    Parameters
    ----------
    im : np.ndarray
        The contact map image
    config_in : dict
        Configuration dictionary containing the vmin and vmax percentiles
    Returns
    -------
    config : dict
        Updated configuration dictionary with vmin and vmax set to 0 and 100, respectively, if the image intensity values are all the same
    '''
    config = copy.deepcopy(config_in)

    if np.percentile(im, config.im_vmin) == np.percentile(im, config.im_vmax):
        
        print(f"\tWarning: Image intensity values are all the same...")
        print(f"\tSetting im_vmin and im_vmax to 0 and 100, respectively")

        config.im_vmin = 0
        config.im_vmax = 100

    return config
    

def generate_hic_image(hic_file, chromosome, resolution, window_size, data_type, normalization, whiten,
                       rotation_padding, save_path, verbose, root, vmax_perc=99, vmin_perc=0):
    """
    Generate contact map image from Hi-C (or Repli Hi-C) data
    The contact map is 
    1. zero sum columns (and rows) removed
    2. rotated and so requires a padding method to fill in the corners

    Parameters
    ----------
    hic_file : str
        Path to the Hi-C file
    chromosome : str
        Chromosome name according to the formatting in the .hic file (e.g. "chr1" or "1" depending on the file)
    resolution : int
        Resolution of the Hi-C data in base pairs
    window_size : int
        Size of the window in base pairs to extract from the Hi-C data
    data_type : str
        Type of data to extract from the Hi-C file. Can be "oe" or "observed"
    normalization : str
        Normalization method to apply to the Hi-C data. Can be "KR", "VC", "VC_SQRT" or "NONE" according to the .hic file
    whiten : float or None
        If float, applies a whitening transformation to the Hi-C data with the specified epsilon value
        If None, no whitening is applied
        The whitening process utilizes PCA whitening: http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    rotation_padding : str
        Padding method to use when rotating the Hi-C data
        Same parameter as `rotate_mode` in `scipy.ndimage.rotate`:
            rotate_mode : {‘constant’, ‘reflect’, ‘grid-mirror’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}
    save_path : str
        Path to the directory where the generated image will be saved
    verbose : bool
        Whether to print verbose output during the image generation process
    root : str
        Root name for the saved image file
    vmax_perc : int, optional
        Maximum percentile for clipping the image intensity values. Default is 99
    vmin_perc : int, optional
        Minimum percentile for clipping the image intensity values. Default is 0
    Returns
    -------
    im : np.ndarray
        The contact map image used as input to the program as a numpy array
    im_orig : np.ndarray
        The contact map without the zero sum columns and rows removed
    im_p_value : np.ndarray
        The p-value image of the contact map, which necessarily is data_type="observed", used for statistical significance
    rm_idx : list
        List of indices of the rows and columns that were removed due to zero sum
    save_name : str
        The path to the saved image file
    N : int
        The number of bins in the Hi-C data after removing zero sum rows and columns
    """

    save_name = os.path.join(save_path, f"{root}_contact_map.jpg")

    window_size_bin = np.ceil(window_size / resolution).astype(int)

    im, rm_idx, N = read_hic_rectangle(filename=hic_file, 
                                       chrom=chromosome, 
                                       resolution=resolution, 
                                       window_size_bin=window_size_bin, 
                                       data_type=data_type, 
                                       normalization=normalization,
                                       rotate_mode=rotation_padding, 
                                       cval=0, 
                                       handle_zero_sum="remove", 
                                       whiten=whiten,
                                       verbose=verbose) 
    
    if im.shape[0] == 0 or im.shape[1] == 0:
        raise ValueError(f"Empty contact map generated for chromosome {chromosome} with resolution {resolution} and window size {window_size}. "
                         f"Please check the Hi-C file {hic_file} for coverage and/or parameters")
    
    if data_type != "oe":
         # used to compute stripiness
        im_oe, _, _ = read_hic_rectangle(filename=hic_file, 
                                        chrom=chromosome, 
                                        resolution=resolution, 
                                        window_size_bin=window_size_bin, 
                                        data_type="oe", 
                                        normalization=normalization,
                                        rotate_mode=rotation_padding, 
                                        cval=0, 
                                        handle_zero_sum="remove", 
                                        whiten=None, # used to compute stripiness
                                        verbose=verbose)     
    else:
        im_oe = im.copy()
    
    im_p_value, _, _ = read_hic_rectangle(filename=hic_file, 
                                       chrom=chromosome, 
                                       resolution=resolution, 
                                       window_size_bin=window_size_bin, 
                                       data_type="observed", 
                                       normalization=normalization,
                                       rotate_mode=rotation_padding, 
                                       cval=0, 
                                       handle_zero_sum="remove", 
                                       whiten=None,
                                       verbose=False) 
    
    im_orig = read_hic_rectangle(filename=hic_file, 
                                 chrom=chromosome, 
                                 resolution=resolution, 
                                 window_size_bin=window_size_bin, 
                                 data_type=data_type, 
                                 normalization=normalization,
                                 rotate_mode=rotation_padding, 
                                 cval=0, 
                                 handle_zero_sum=None, 
                                 whiten=None,
                                 verbose=False)    
    

    if verbose: print(f"\tImage dimensions: {im.shape}")

    im_p_value = np.log10(im_p_value + 1)

    if data_type == "observed":
        if whiten is None:
            # only do log transformation if not whitened
            # because whitened already did a log transformation
            im = np.log10(im + 1)
            
        im_orig = np.log10(im_orig + 1)

    # before clipping, let's save an image of histogram of intensity values of the image
    save_histogram(im, save_path, file_name=f"{root}_contact_map_intensity_value_histogram.jpg", vmin_perc=vmin_perc, vmax_perc=vmax_perc) 

    if np.percentile(im, vmin_perc) == np.percentile(im, vmax_perc):
        # if the image intensity values are all the same, set vmin and vmax to 0 and 100, respectively
        # the config variables (config.im_vmin, config.im_vmax) are updated later in the main function
        # by calling `check_im_vmin_vmax`
        vmin_perc = 0
        vmax_perc = 100

    im = cv.normalize(im, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    im_orig = cv.normalize(im_orig, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    im_oe = cv.normalize(im_oe, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    plt.imsave(save_name, im, cmap="gray", vmax=np.percentile(im, vmax_perc), vmin=np.percentile(im, vmin_perc))

    # what imageJ looks at i.e. [0, 1] normalization -> percentile thresholding
    im = np.clip(im, np.percentile(im, vmin_perc), np.percentile(im, vmax_perc))   
    im_orig = np.clip(im_orig, np.percentile(im, vmin_perc), np.percentile(im, vmax_perc))
    im_oe = np.clip(im_oe, np.percentile(im, vmin_perc), np.percentile(im, vmax_perc))

    return im, im_orig, im_p_value, im_oe, rm_idx, save_name, N



def generate_hic_corr_image(hic_file, chromosome, resolution, window_size, data_type, normalization, vmax_perc, vmin_perc, save_path, 
                            zero_before_corr, rotation_padding, root, verbose):
    """
    Similar functionality to `generate_hic_image` except that as opposed to `read_hic_rectangle` we call `read_hic_corr_rectangle`
    
    This modified function generates a contact map after computing the correlation matrix from the Hi-C data

    If data_type is any of ["coe", "cobserved"] then we indeed compute the correlation matrix. The order of events is as follows:
    1. Log
    2. Normalize 0-1
    3. Clip percentile
    4. Compute scalar product (correlation)
    5. Normalize 0-1

    Parameters
    ----------
    hic_file : str
        Path to the Hi-C file
    chromosome : str
        Chromosome name according to the formatting in the .hic file (e.g. "chr1" or "1" depending on the file)
    resolution : int
        Resolution of the Hi-C data in base pairs
    window_size : int
        Size of the window in base pairs to extract from the Hi-C data
    data_type : str
        Type of data to extract from the Hi-C file. Can be 
        * "coe": correlation of "oe"
        * "cobserved": correlation of "observed"
    normalization : str
        Normalization method to apply to the Hi-C data. Can be "KR", "VC", "VC_SQRT" or "NONE" according to the .hic file
    vmax_perc : int, optional
        Maximum percentile for clipping the image intensity values. Default is 99
    vmin_perc : int, optional
        Minimum percentile for clipping the image intensity values. Default is 0
    save_path : str
        Path to the directory where the generated image will be saved
    zero_before_corr : bool
        Whether to zero out the off-diagonal elements beyond the window size before computing the correlation matrix
        If True, the off-diagonal elements beyond the window size will be set to zero before computing the correlation matrix
        It turns out that this has a significant effect on the correlation matrix, so we keep it as an option
        Notably, 
        * To generate the image for corner detection we zero out the off-diagonal elements beyond the window size
        * To generate the image for p-value we do not zero out the off-diagonal elements beyond the window size
    rotation_padding : str
        Padding method to use when rotating the Hi-C data
        Same parameter as `rotate_mode` in `scipy.ndimage.rotate`:
            rotate_mode : {‘constant’, ‘reflect’, ‘grid-mirror’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’}
    root : str
        Root name for the saved image file
    verbose : bool
        Whether to print verbose output during the image generation process

    Returns
    -------
    im : np.ndarray
        The contact map 

    Additionally, saves a histogram of the contact map intensity values
    """
    window_size_bin = np.ceil(window_size / resolution).astype(int)

    if data_type in ["coe", "cobserved"]:

        if data_type == "coe":
            data_type_internal = "oe"
        else:
            data_type_internal = "observed"

        # do correlation 
        im, _, _ = read_hic_corr_rectangle(filename=hic_file, 
                                                chrom=chromosome, 
                                                resolution=resolution, 
                                                window_size_bin=window_size_bin, 
                                                vmin_q=vmin_perc, 
                                                vmax_q=vmax_perc,
                                                data_type=data_type_internal, 
                                                zero_before_corr=zero_before_corr,
                                                normalization=normalization, 
                                                save_path=save_path,
                                                rotate_mode=rotation_padding, 
                                                cval=0, 
                                                handle_zero_sum="remove", 
                                                root=root,
                                                verbose=verbose)
    elif data_type == "ne":
        # New: network enhancement
        current_loc = os.path.dirname(__file__) 
        # In parent directory of current file
        ne_path = os.path.abspath(os.path.join(current_loc, "..", "Network_Enhancement"))
        im, _, _ = read_hic_network_enhancement(filename=hic_file,
                                                chrom=chromosome, 
                                                resolution=resolution, 
                                                window_size_bin=window_size_bin, 
                                                vmin_q=vmin_perc, 
                                                vmax_q=vmax_perc,
                                                normalization=normalization, 
                                                save_path=save_path,
                                                ne_path=ne_path, 
                                                rotate_mode=rotation_padding, 
                                                cval=0, 
                                                handle_zero_sum="remove", 
                                                root=root,
                                                verbose=verbose)

    else:
        raise ValueError(f"data_type {data_type} not supported for correlation image generation"
                         "Use 'coe' or 'cobserved' or call `generate_hic_image` if data_type is 'oe' or 'observed'.") 
    
    # value range should be from [-1, 1]
    # we map this to float in [0, 1] for image processing conventions
    im = cv.normalize(im, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    return im
    

