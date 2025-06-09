import numpy as np
import pandas as pd
import hicstraw
import pyBigWig 
import bioframe as bf
from scipy.ndimage import rotate 

def z_standardize(A):
    """
    Helper for `scalar_products`
    """
    return (A - np.mean(A, axis=0)) / np.std(A, axis=0)


def center(A):
    """
    Helper for `scalar_products`
    """
    # print("\tWARNING: changed function to double centering (04/09/25)")
    # return A - np.mean(A, axis=0)

    A_centered = A - np.mean(A, axis=1, keepdims=True) \
             - np.mean(A, axis=0, keepdims=True) \
             + np.mean(A)
    return A_centered

def scalar_products(A, out):
    '''
    Computes A^T @ A (i.e. covariance or correlation matrix)

    Parameters
    ----------
    * A : m by n array or matrix, 
        where m is # of obs and n is # of var
    * out : {"covariance", "correlation", "sample correlation"}
        Note: "correlation" is equivalent to numpy corrcoef()
        It is not the sample covariance matrix (which divides by n - 1)
        
        Note: "covariance" is equivalent to numpy cov
    
    Returns
    -------
    * numpy.ndarray : the covariance or correlation matrix of A
    '''
    if out == "correlation":
        A = z_standardize(A)
        n = A.shape[0]
        return (A.T @ A) / n
    elif out == "sample correlation":
        A = z_standardize(A)
        n = A.shape[0]
        return (A.T @ A) / (n - 1)
    elif out == "covariance":
        A = center(A)
        n = A.shape[0]
        return (A.T @ A) / (n - 1)
    else:
        print("out : {'covariance', 'correlation', 'sample correlation'}")


def whiten_matrix(A_for_corr, A_for_whiten, epsilon):
    """
    Whitening scheme 

    Parameters
    ----------
    A_for_corr : numpy.ndarray
        The matrix for which to compute the covariance or correlation matrix
        This is used to compute the covariance matrix C
    A_for_whiten : numpy.ndarray
        The matrix to be whitened
        This is used to apply the whitening transformation
    epsilon : float
        A small value to prevent division by zero during whitening
    
    Returns
    -------
    numpy.ndarray
        The whitened matrix
    """
    C = scalar_products(A_for_corr, out="covariance")

    d, V = np.linalg.eigh(C)

    d_reg = d + epsilon  # prevent divide by 0 issues
    d_inv_sqrt = 1.0 / np.sqrt(d_reg)
    W = V @ np.diag(d_inv_sqrt) @ V.T

    A_centered = center(A_for_whiten) # apply double centering
    
    return W @ A_centered, C



def read_hic_rectangle(filename, chrom, resolution, window_size_bin, data_type, whiten, normalization='NONE', positions="all", 
                       handle_zero_sum=None, rotate_mode="nearest", cval=0, 
                       verbose=False):
    """
    Reads a Hi-C file as a numpy matrix (square) 
    1. Extracts a window of size `window_size_bin` from the main diagonal 
    2. Rotates the matrix by 45 degrees to obtain a rectangle

    Parameters
    ----------
    filename : str
        Path to the Hi-C file
    chrom : str
        Chromosome name according to the formatting in the .hic file (e.g. "chr1" or "1" depending on the file)
    resolution : int
        Resolution of the Hi-C data in base pairs
    window_size : int
        Binned size of the window 
    data_type : str
        Type of data to extract from the Hi-C file. Can be "oe" or "observed"
    normalization : str
        Normalization method to apply to the Hi-C data. Can be "KR", "VC", "VC_SQRT" or "NONE" according to the .hic file
    positions : tuple or str
        If tuple, specifies the start and end positions (1-indexed) to extract from the Hi-C file
        If "all", extracts the entire chromosome
    whiten : float or None
        If float, applies a whitening transformation to the Hi-C data with the specified epsilon value
        If None, no whitening is applied
        The whitening process utilizes PCA whitening: http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    handle_zero_sum : str or None
        If "remove", removes rows and columns with zero sum from the Hi-C matrix
        If float or int, replaces zero sum rows and columns with the specified value
        If "mean", replaces zero sum rows and columns with the mean of the non-zero sum rows and columns
        If None, no handling of zero sum regions is performed
    rotate_mode : str
        Mode for rotating the matrix. Can be "constant", "reflect", "grid-mirror", "grid-constant", "nearest", "mirror", "grid-wrap", or "wrap"
        This parameter is passed to `scipy.ndimage.rotate`
    cval : float
        Value to use for points outside the boundaries of the input array during rotation
        This parameter is passed to `scipy.ndimage.rotate`

    Returns
    -------
    im : np.ndarray
        The contact map image
    rm_idx : np.ndarray, if handle_zero_sum == "remove"
        A numpy array of the indices of the rows and columns that were removed due to zero sum
    N : int, if handle_zero_sum == "remove"
        The number of rows (and columns) in the Hi-C matrix after removing zero sum regions
    """
    # read Hi-C file as a numpy array
    mat = read_hic_file(filename, chrom, resolution, positions=positions, data_type=data_type, normalization=normalization, verbose=False)

    mat = mat.astype(np.float64)        

    # add 5 bins to the window size to avoid aliasing artefacts
    window_size_buffer = min(window_size_bin + 5, mat.shape[0]) 
    
    if whiten is not None:
        # Zeroing off-diagonal does have an effect on the zero sum removed rows and columns
        # However, for correlation this has the additional effect of causing issues with the dot products
        # So we should just use this mat for computing the remove indices
        mat_rm_idx = np.copy(mat)

        mat_rm_idx[np.triu_indices_from(mat_rm_idx, k=window_size_buffer)] = 0
        mat_rm_idx[np.tril_indices_from(mat_rm_idx, k=-1)] = 0

        # For correlation, we need to do zero sum computation BEFORE computing the correlations
        if handle_zero_sum is not None:
            # padding effect goes here before we rotate
            if handle_zero_sum == "remove":
                _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
            elif isinstance(handle_zero_sum, (int, float)):
                _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
                mat[rm_idx, :] = handle_zero_sum
                mat[:, rm_idx] = handle_zero_sum
                
            elif handle_zero_sum == "mean":
                mean_values, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
                mat[rm_idx, :] = np.mean(mean_values)
                mat[:, rm_idx] = np.mean(mean_values)  

        # New addition to correlation version: 
        # Manually remove the rows and columsn in mat according to rm_idx
        mat = np.delete(mat, rm_idx, axis=0)
        mat = np.delete(mat, rm_idx, axis=1)

        # Before we whiten, lets first apply a log transformation and 0-1 normalization 
        # to replicate notebook
        mat = np.log10(mat + 1)

        mat = cv.normalize(mat, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

        # Whiten matrix
        mat, _ = whiten_matrix(A_for_corr=mat, A_for_whiten=mat, epsilon=whiten)

        # The rest of the code should be identical (rotation code)

    else:
        # Probably has no effect on performance, but we'll do it anyway 
        mat[np.triu_indices_from(mat, k=window_size_buffer)] = 0
        mat[np.tril_indices_from(mat, k=-1)] = 0
            
        # Moved handling zero sum regions AFTER value normalization
        # This way, we can make the border cval and the rotation cval the SAME
        if handle_zero_sum is not None:
            # padding effect goes here before we rotate
            if handle_zero_sum == "remove":
                mat, rm_idx = remove_zero_sum(mat, verbose=verbose)
            elif isinstance(handle_zero_sum, (int, float)):
                _, rm_idx = remove_zero_sum(mat, verbose=verbose)
                mat[rm_idx, :] = handle_zero_sum
                mat[:, rm_idx] = handle_zero_sum
                
            elif handle_zero_sum == "mean":
                mean_values, rm_idx = remove_zero_sum(mat, verbose=verbose)
                mat[rm_idx, :] = np.mean(mean_values)
                mat[:, rm_idx] = np.mean(mean_values)  

    N = mat.shape[0]
    
    # moved center computation after removing zero indices
    center = np.ceil(mat.shape[0] * np.sqrt(2) / 2).astype(int) # the new center after rotation
    
    mat = rotate(mat, 45, reshape=True, order=1, mode=rotate_mode, cval=cval) # scipy.ndimage
    
    # convert the window size to rectangle height
    window_size_bin_rect = np.ceil(window_size_bin / np.sqrt(2)).astype(int) 
        
    if handle_zero_sum == "remove":
        return mat[center-window_size_bin_rect:center, :], rm_idx, N
    return mat[center-window_size_bin_rect:center, :]

from utils.plotting import save_histogram
import cv2 as cv
from Network_Enhancement.runner import enhance_network

def read_hic_network_enhancement(filename, chrom, resolution, window_size_bin, vmin_q, vmax_q, save_path,
                                 root, ne_path, normalization='NONE', positions="all", handle_zero_sum=None, 
                                 rotate_mode="nearest", cval=0, verbose=False):
    """
    In development
    [1] Wang, B., Pourshafeie, A., Zitnik, M. et al. 
    Network enhancement as a general method to denoise weighted biological networks. 
    Nat Commun 9, 3108 (2018). https://doi-org.proxy.lib.umich.edu/10.1038/s41467-018-05469-x
    """

    mat = read_hic_file(filename=filename, chrom=chrom, resolution=resolution, positions=positions, 
                        data_type="oe", # hard-code 
                        normalization=normalization, verbose=False)

    mat = mat.astype(np.float64)

    window_size_buffer = min(window_size_bin + 5, mat.shape[0]) 
    # the 5 bins is for aliasing artefacts

    # Zeroing off-diagonal does have an effect on the zero sum removed rows and columns
    # However, for correlation this has the additional effect of causing issues with the dot products
    # So we should just use this mat for computing the remove indices
    mat_rm_idx = np.copy(mat)

    mat_rm_idx[np.triu_indices_from(mat_rm_idx, k=window_size_buffer)] = 0
    mat_rm_idx[np.tril_indices_from(mat_rm_idx, k=-1)] = 0

    # For correlation, we need to do zero sum computation BEFORE computing the correlations
    if handle_zero_sum is not None:
        # padding effect goes here before we rotate
        if handle_zero_sum == "remove":
            _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
        elif isinstance(handle_zero_sum, (int, float)):
            _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
            mat[rm_idx, :] = handle_zero_sum
            mat[:, rm_idx] = handle_zero_sum
            
        elif handle_zero_sum == "mean":
            mean_values, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
            mat[rm_idx, :] = np.mean(mean_values)
            mat[:, rm_idx] = np.mean(mean_values)  

    # New addition to correlation version: 
    # Manually remove the rows and columsn in mat according to rm_idx
    mat = np.delete(mat, rm_idx, axis=0)
    mat = np.delete(mat, rm_idx, axis=1)

    # Save histogram of the contact map intensity values
    save_histogram(mat, save_path, vmax_perc=vmax_q, vmin_perc=vmin_q, file_name=f"{root}_network_enhancement_intensity_value_histogram.png")

    if np.percentile(mat, vmin_q) == np.percentile(mat, vmax_q):
        # This should not occur as the vmin and vmax is already updated in the main function
        # With the call to 
        #   config = check_im_vmin_vmax(im, config) 
        raise ValueError()

    # mat = cv.normalize(mat, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    # before we rotate, we compute network enhancement
    print("\tWarning: MATLAB module is required for network enhancement. May not work outside GL environment")
    mat = enhance_network(A=mat, order=15, num_neighbors=50, alpha=0.99, workdir=save_path, ne_path=ne_path, matlab_module="matlab/R2024b")

    # below is equivalent to `read_hic_rectangle`
    window_size_buffer = min(window_size_bin + 5, mat.shape[0]) 
    # the 5 bins is for aliasing artefacts
    
    # Probably has no effect on performance, but we'll do it anyway 
    mat[np.triu_indices_from(mat, k=window_size_buffer)] = 0
    mat[np.tril_indices_from(mat, k=-1)] = 0
        
    N = mat.shape[0]
    
    # moved center computation after removing zero indices
    center = np.ceil(mat.shape[0] * np.sqrt(2) / 2).astype(int) # the new center after rotation
    
    mat = rotate(mat, 45, reshape=True, order=1, mode=rotate_mode, cval=cval) # scipy.ndimage

    # new is that we do a log transformation, normalize again, and clip AFTER the rotation
    window_size_bin_rect = np.ceil(window_size_bin / np.sqrt(2)).astype(int) 
    mat = mat[center-window_size_bin_rect:center, :]
    
    mat = np.log10(mat + 1)
    mat = cv.normalize(mat, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    mat = np.clip(mat, np.percentile(mat, vmin_q), np.percentile(mat, vmax_q))
    
    if handle_zero_sum == "remove":
        return mat, rm_idx, N
    return mat


def read_hic_corr_rectangle(filename, chrom, resolution, window_size_bin, data_type, vmin_q, vmax_q, save_path, zero_before_corr, 
                            root, normalization='NONE', positions="all", handle_zero_sum=None, rotate_mode="nearest", cval=0, verbose=False):
    """
    Equivalent to `read_hic_rectangle` except that we read in the Hi-C **correlation** matrix 
    1. Log
    2. Normalize 0-1
    3. Clip percentile
    4. Compute scalar product A^T @ A (i.e. correlation)

    Parameters
    ----------
    filename : str
        Path to the Hi-C file
    chrom : str
        Chromosome name according to the formatting in the .hic file (e.g. "chr1" or "1" depending on the file)
    resolution : int
        Resolution of the Hi-C data in base pairs
    window_size : int
        Binned size of the window 
    data_type : str
        Type of data to extract from the Hi-C file. Can be "oe" or "observed"
    vmin_q : int
        Minimum percentile to clip the Hi-C data
    vmax_q : int
        Maximum percentile to clip the Hi-C data
    zero_before_corr : bool
        Whether to zero out the off-diagonal elements beyond the window size before computing the correlation matrix
        If True, the off-diagonal elements beyond the window size will be set to zero before computing the correlation matrix
        It turns out that this has a significant effect on the correlation matrix, so we keep it as an option
        Notably, 
        * To generate the image for corner detection we zero out the off-diagonal elements beyond the window size
        * To generate the image for p-value we do not zero out the off-diagonal elements beyond the window size
    root : str
        Root name for the saved image file        
    normalization : str
        Normalization method to apply to the Hi-C data. Can be "KR", "VC", "VC_SQRT" or "NONE" according to the .hic file
    positions : tuple or str
        If tuple, specifies the start and end positions (1-indexed) to extract from the Hi-C file
        If "all", extracts the entire chromosome
    handle_zero_sum : str or None
        If "remove", removes rows and columns with zero sum from the Hi-C matrix
        If float or int, replaces zero sum rows and columns with the specified value
        If "mean", replaces zero sum rows and columns with the mean of the non-zero sum rows and columns
        If None, no handling of zero sum regions is performed
    rotate_mode : str
        Mode for rotating the matrix. Can be "constant", "reflect", "grid-mirror", "grid-constant", "nearest", "mirror", "grid-wrap", or "wrap"
        This parameter is passed to `scipy.ndimage.rotate`
    cval : float
        Value to use for points outside the boundaries of the input array during rotation
        This parameter is passed to `scipy.ndimage.rotate`

    Returns
    -------
    im : np.ndarray
        The contact map image
    rm_idx : np.ndarray, if handle_zero_sum == "remove"
        A numpy array of the indices of the rows and columns that were removed due to zero sum
    N : int, if handle_zero_sum == "remove"
        The number of rows (and columns) in the Hi-C matrix after removing zero sum regions

    Additionally, saves a histogram of the contact map intensity values
    """
    
    mat = read_hic_file(filename=filename, chrom=chrom, resolution=resolution, positions=positions, data_type=data_type, normalization=normalization, verbose=False)

    mat = mat.astype(np.float64)

    window_size_buffer = min(window_size_bin + 5, mat.shape[0]) 
    # the 5 bins is for aliasing artefacts
    
    if not zero_before_corr:

        # Zeroing off-diagonal does have an effect on the zero sum removed rows and columns
        # However, for correlation this has the additional effect of causing issues with the dot products
        # So we should just use this mat for computing the remove indices
        mat_rm_idx = np.copy(mat)

        mat_rm_idx[np.triu_indices_from(mat_rm_idx, k=window_size_buffer)] = 0
        mat_rm_idx[np.tril_indices_from(mat_rm_idx, k=-1)] = 0

        # For correlation, we need to do zero sum computation BEFORE computing the correlations
        if handle_zero_sum is not None:
            # padding effect goes here before we rotate
            if handle_zero_sum == "remove":
                _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
            elif isinstance(handle_zero_sum, (int, float)):
                _, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
                mat[rm_idx, :] = handle_zero_sum
                mat[:, rm_idx] = handle_zero_sum
                
            elif handle_zero_sum == "mean":
                mean_values, rm_idx = remove_zero_sum(mat_rm_idx, verbose=verbose)
                mat[rm_idx, :] = np.mean(mean_values)
                mat[:, rm_idx] = np.mean(mean_values)  

        # New addition to correlation version: 
        # Manually remove the rows and columsn in mat according to rm_idx
        mat = np.delete(mat, rm_idx, axis=0)
        mat = np.delete(mat, rm_idx, axis=1)
    else:
        # Identical to `read_hic_rectangle`
        # We ensure that off-diagonal regions are zeroed out on `mat` itself
        mat[np.triu_indices_from(mat, k=window_size_buffer)] = 0
        mat[np.tril_indices_from(mat, k=-1)] = 0   

        if handle_zero_sum is not None:
            # padding effect goes here before we rotate
            if handle_zero_sum == "remove":
                mat, rm_idx = remove_zero_sum(mat, verbose=verbose)
            elif isinstance(handle_zero_sum, (int, float)):
                _, rm_idx = remove_zero_sum(mat, verbose=verbose)
                mat[rm_idx, :] = handle_zero_sum
                mat[:, rm_idx] = handle_zero_sum
                
            elif handle_zero_sum == "mean":
                mean_values, rm_idx = remove_zero_sum(mat, verbose=verbose)
                mat[rm_idx, :] = np.mean(mean_values)
                mat[:, rm_idx] = np.mean(mean_values)          


    if data_type == "observed":
        mat = np.log10(mat + 1)

    # Save histogram of the contact map intensity values
    save_histogram(mat, save_path, vmax_perc=vmax_q, vmin_perc=vmin_q, file_name=f"{root}_correlation_{data_type}_intensity_value_histogram.png")

    if np.percentile(mat, vmin_q) == np.percentile(mat, vmax_q):
        # This should not occur as the vmin and vmax is already updated in the main function
        # With the call to 
        #   config = check_im_vmin_vmax(im, config) 
        raise ValueError()

    mat = cv.normalize(mat, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    # clip values in mat
    mat = np.clip(mat, np.percentile(mat, vmin_q), np.percentile(mat, vmax_q))

    # before we extract rectangle, we first compute the correlation matrix 
    mat = scalar_products(mat, out="correlation")

    # below is equivalent to `read_hic_rectangle`
    window_size_buffer = min(window_size_bin + 5, mat.shape[0]) 
    # the 5 bins is for aliasing artefacts
    
    # Probably has no effect on performance, but we'll do it anyway 
    mat[np.triu_indices_from(mat, k=window_size_buffer)] = 0
    mat[np.tril_indices_from(mat, k=-1)] = 0
        
    N = mat.shape[0]
    
    # moved center computation after removing zero indices
    center = np.ceil(mat.shape[0] * np.sqrt(2) / 2).astype(int) # the new center after rotation
    
    mat = rotate(mat, 45, reshape=True, order=1, mode=rotate_mode, cval=cval) # scipy.ndimage
    
    window_size_bin_rect = np.ceil(window_size_bin / np.sqrt(2)).astype(int) 
        
    if handle_zero_sum == "remove":
        return mat[center-window_size_bin_rect:center, :], rm_idx, N
    return mat[center-window_size_bin_rect:center, :]

    
def read_hic_efficiently(mzd, binned_size, resolution, max_matrix_size=2000, verbose=True):
    '''
    Reads a Hi-C with even size blocks of size (`max_matrix_size` - 1) to avoid
    crashes due to limited RAM; if too small then performance takes a hit
    * max_matrix_size = 2000 is roughly appropriate for a system with 32GB RAM 

    Parameters
    ----------
    mzd : hicstraw.MatrixZoomData
        The MatrixZoomData object from hicstraw
    binned_size : int
        The binned size of Hi-C matrix
    resolution : int
        Resolution of Hi-C data in base pairs
    max_matrix_size : int
        The maximum size of the matrix to read in at once (default is 2000)
    
    Returns
    -------
    H : numpy.ndarray
        A numpy array of the Hi-C matrix, with shape (binned_size, binned_size)
    '''
    H = np.zeros((binned_size, binned_size))
    
    m = np.ceil(binned_size / max_matrix_size).astype(int) # length of final matrix

    for i in range(m):
        # upper triangular elements only
        for j in range(i, m):

            # read in the data
            i0 = max_matrix_size * i 
            i1 = max_matrix_size * (i + 1) - 1

            i2 = max_matrix_size * j
            i3 = max_matrix_size * (j + 1) - 1

            if i1 >= binned_size:
                i1 = binned_size - 1

            if i3 >= binned_size:
                i3 = binned_size - 1

            p0 = i0 * resolution
            p1 = i1 * resolution 
            p2 = i2 * resolution
            p3 = i3 * resolution

            if verbose:
                print("Entry: ({}, {}) of ({}, {})".format(i, j, m - 1, m - 1))

            H[i0:i1+1, i2:i3+1] = mzd.getRecordsAsMatrix(p0, p1, p2, p3)
            H[i2:i3+1, i0:i1+1] = mzd.getRecordsAsMatrix(p0, p1, p2, p3).T
            
    return H


def read_hic_file(filename, chrom, resolution, positions, data_type, normalization="NONE", max_matrix_size=False, verbose=True):
    '''
    Reads a Hi-C file and returns a numpy array. Utilizes the hicstraw library

    Parameters
    ----------
    filename (str) : path to .hic file
    chrom (str) : specify chromosome (e.g., "chr3") – also printed if verbose=True
    resolution (int) : resolution of the Hi-C data in base pairs
    positions (int, int) : tuple of start and end 1-indexed basepair resolution, or "all"
    data_type (str) = ["observed", "oe"] : according to hicstraw documentation 
    normalization (str) = ["NONE", "VC", "VC_SQRT", "KR", "SCALE"]: according to hicstraw documentation 
    max_matrix_size (int or False)
        If False, then reads the entire Hi-C matrix at once, which may cause memory issues
        If int, this reads a Hi-C in chunks of size (`max_matrix_size` - 1) to avoid crashes due to limited RAM 
        (note: if max_matrix_size is too small then performance takes a hit)
        
        max_matrix_size = 2000 is roughly appropriate for a system with 32GB RAM 

    Returns
    -------
    numpy.ndarray : a numpy array of the Hi-C matrix 
    '''
    hic = hicstraw.HiCFile(filename)
    if verbose:
        print("Identified")
        print("Possible resolutions")
        for possible_res in hic.getResolutions():
            print(" * {}".format(possible_res))
        print("Chrom name : Chrom size:")
    found = False
    start_pos = 0
    end_pos = -1
    for chromosome in hic.getChromosomes():
        if verbose:
            print("  -", chromosome.name, chromosome.length, end="")
        
        if str(chrom) == chromosome.name or ("chr" not in chromosome.name and chromosome.name == str(chrom)[3:]):
            key = chromosome.name
            found = True
            end_pos = chromosome.length
            if verbose:
                print(" <- Selected", end="")
                
        if verbose:
            print()
        
    if not found:
        print("Chromosome '{}' could not be identified. Check again.".format(chrom))
        return
    
    mzd = hic.getMatrixZoomData(key, key, data_type, normalization, "BP", int(resolution))
    
    if positions != "all":
        start_pos = positions[0]
        end_pos = positions[1]
        
    if max_matrix_size:
        binned_size = np.ceil((end_pos - start_pos) / resolution).astype(int)
        return read_hic_efficiently(mzd, binned_size, resolution, max_matrix_size, verbose)
        
    return mzd.getRecordsAsMatrix(start_pos, end_pos, start_pos, end_pos)
    


def remove_zero_sum(A_in, verbose=False):
    '''
    Removes zero sum columns (and rows) from a numpy array

    Parameters
    ----------
    A_in : numpy.ndarray
        The input array from which to remove zero sum rows and columns
    verbose : bool
        If True, prints the shape of the input array before and after removing zero sum rows and columns,
        as well as the indices of the removed rows and columns
    
    Returns
    -------
    A_in : numpy.ndarray
        The input array with zero sum rows and columns removed
    remove_idx : numpy.ndarray
        A numpy array of the indices of the rows and columns that were removed due to zero sum
    '''
    if verbose:
        print("{}".format(A_in.shape), end=" ")
    remove_idx = np.where(np.isclose(np.sum(A_in, axis=0), 0))[0]
    A_in = np.delete(A_in, remove_idx, axis=0)
    A_in = np.delete(A_in, remove_idx, axis=1)
    if verbose:
        print("-> {}".format(A_in.shape))
        print("    Removed indices: {}".format(remove_idx))
    return A_in, remove_idx



def kth_diag_indices(a, k):
    """
    Returns the indices of the k-th diagonal of a matrix
    Parameters
    ----------
    a : numpy.ndarray
        The input matrix from which to extract the k-th diagonal indices
    k : int
        The off-diagonal index to extract
    
    Returns
    -------
    rows : numpy.ndarray
        The row indices of the k-th diagonal
    cols : numpy.ndarray
        The column indices of the k-th diagonal
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
    


def group_adjacent_numbers(numbers):
    '''
    Groups adjacent numbers in a list into sublists

    Parameters
    ----------
    numbers : list of int
        The list of numbers to group
    
    Returns
    -------
    groups : list of list of int
        A list of lists, where each sublist contains adjacent numbers from the input list
        For example, [1, 2, 3, 5, 6] would return [[1, 2, 3], [5, 6]]
    '''
    if not list(numbers):
        return []
    
    # Sort the numbers
    numbers.sort()
    
    # Initialize groups
    groups = [[numbers[0]]]
    
    # Iterate over the sorted list and group adjacent numbers
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            # If current number is adjacent to the previous, add to the current group
            groups[-1].append(numbers[i])
        else:
            # Otherwise, start a new group
            groups.append([numbers[i]])
    
    return groups
    
    
    
