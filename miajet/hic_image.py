import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

from utils.processing import read_hic_rectangle, read_hic_corr_rectangle, read_hic_network_enhancement
from utils.plotting import save_histogram
import copy

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
                                       verbose=False) 
    
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

    plt.imsave(save_name, im, cmap="gray", vmax=np.percentile(im, vmax_perc), vmin=np.percentile(im, vmin_perc))

    # what imageJ looks at i.e. [0, 1] normalization -> percentile thresholding
    im = np.clip(im, np.percentile(im, vmin_perc), np.percentile(im, vmax_perc))   
    im_orig = np.clip(im_orig, np.percentile(im, vmin_perc), np.percentile(im, vmax_perc))

    return im, im_orig, im_p_value, rm_idx, save_name, N



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
    

