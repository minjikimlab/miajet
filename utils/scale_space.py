import numpy as np
import cv2 as cv
from functools import partial
import multiprocessing
import pyscsp

import numpy as np

def clip_scale_range(scale_range, im_shape, verbose, k=2):
    """
    Computes a maximum sigma (scale) for Gaussian convolution
    based on the image of dimensions (im_shape)
    
    The strict no-padding limit (for a kernel size of 2*ceil(k*sigma)+1) is:
        sigma_max_strict = (min_dim - 1) / (2 * k)
    
    Parameters:
      im_shape (tuple): Image dimensions (height, width) 
      k (float): Kernel multiplier; many implementations use k ~2.
    
    Returns:
    scale_range : np.ndarray
        Clipped scale range where scales larger than sigma_max are removed
    """
    if len(im_shape) < 2:
        raise ValueError("im_shape must have at least two dimensions (height, width)")
    
    # Use the smallest dimension (width or height)
    min_dim = min(im_shape)
    
    # Compute maximum sigma
    sigma_max = (min_dim) / (2 * k)

    if verbose:
        print(f"\tRemoving scales {scale_range[scale_range >= sigma_max]} due to image dimension {min_dim} being too small for convolution operation")
        print(f"\tIf you want to include this scale, you can increase the size of the image by some of the following options:")
        print("\t(1) increasing resolution (2) increasing window size (3) choosing a larger chromosome")

    return scale_range[scale_range < sigma_max]


def vector_to_angle(x0, y0, find_normal=False):
    """
    Helper function for `eigenvector_space_to_angle`
    """
    angle = np.arctan2(y0, x0)
    angle = np.degrees(angle) 
    if find_normal:
        return ((angle + 90) % 360) % 180
        
    return (angle % 360) % 180


def eigenvector_space_to_angle(eigvecspace):
    """
    Taking a scale space tensor of eigenvectors, computes angles of the largest (assuming the [:, :, 1, :, :] is the largest)  
    """
    anglespace = []
    for i in range(eigvecspace.shape[0]):

        x0 = eigvecspace[i, 0, 1, :, :]
        y0 = eigvecspace[i, 1, 1, :, :]
                
        anglespace.append(vector_to_angle(x0, y0, find_normal=True))

    return np.array(anglespace)

def zerocross(image, method=1, size_kernel=3):
    """
    There are 2 methods: one from source [1] and the other from source [2]
    
    There is minimal difference betwene the methods in the ideal case (Gaussian noise, clear ridge), except that
        method 1 yields thinner ridges and method 2 yields thicker ridges, noticeably due to the 
        size of the kernel in cv.morphologyEx: np.ones(3, 3)
        In both cases an 8 pixel neighborhood is considered, resulting in my preference for method 1
        It may be more noise robust to consider a 5x5 pixel neighborhood, although the implications
        of doing so are not fully understood
        
        For flexibility, size_kernel is introduced (applicable only for method 2)
        
    
    [1] https://notebook.community/darshanbagul/ComputerVision/EdgeDetection-ZeroCrossings/EdgeDetectionByZeroCrossings
    [2] https://stackoverflow.com/questions/25105916/laplacian-of-gaussian-in-opencv-how-to-find-zero-crossings 
    """
    if method == 1:
        z_c_image = np.zeros(image.shape)

        for i in range(0, image.shape[0]-1):
            for j in range(0, image.shape[1]-1):
                if image[i][j] > 0:
                    if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                        z_c_image[i,j] = 1
                elif image[i][j] < 0:
                    if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                        z_c_image[i,j] = 1
        return z_c_image
    
    elif method == 2:
        # Find the minimum and maximum in the neighborhood of each pixel
        min_image = cv.morphologyEx(image, cv.MORPH_ERODE, np.ones((size_kernel, size_kernel)))
        max_image = cv.morphologyEx(image, cv.MORPH_DILATE, np.ones((size_kernel, size_kernel)))

        # Detect zero-crossings by checking sign changes in the neighborhood
        zero_cross = np.logical_or(
            np.logical_and(min_image < 0, image > 0),
            np.logical_and(max_image > 0, image < 0)
        )
        return zero_cross 
    
    else:
        print("Method must be either 1 or 2")
        raise ValueError
    

def image_eigh(im_matrix):
    """
    im_matrix is size (k, k, n, m)
    where k is the dimension of the square matrix that we eigendecompose
    for each pixel in the image of size n, m
    """
    k = im_matrix.shape[0]
    im_eigvals = np.zeros((k, im_matrix.shape[2], im_matrix.shape[3]))
    im_eigvecs = np.zeros((k, k, im_matrix.shape[2], im_matrix.shape[3]))

    for i in range(im_matrix.shape[2]):
        for j in range(im_matrix.shape[3]):
            # i, j is each pixel
            w, v = np.linalg.eigh(im_matrix[:, :, i, j])

            im_eigvals[:, i, j] = w
            im_eigvecs[:, :, i, j] = v
    return im_eigvals, im_eigvecs

def local_contrast_enhancement(im_eigvals, method):
    """
    Computes the local contrast enhancement for ridges using the eigenvalues of the Hessian matrix

    The methods are outlined in [1]. 
    Methods 5-7 are new and developed to reduce the effect of corners on the ridge strength.

    Parameters:
    ------------
    im_eigvals : np.ndarray
        Eigenvalues of the Hessian matrix of shape (2, n, m) where 
        * 2 is the number of eigenvalues (lambda_1, lambda_2)
        * n and m are the image dimensions 
    method : int or str
        * 1: $\lambda_1$ (M)
        * 2: $(\lambda_1^2 - \lambda_2^2)^2$ (N)
        * 3: $(\lambda_1 - \lambda_2)^2$ (A)
        * 4: $|\lambda_1 - \lambda_2| \cdot |\lambda_1 + \lambda_2|$
        * 5: $\lambda_1 + sign(det(M)) * \sqrt(|det(M)|)$
        * 6: $\lambda_1 - \sqrt(|det(M)|)$
        * 7: $(\lambda_1^2 - \lambda_2^2)^{1/2} - \sqrt(|det(M)|)$
        * "all": returns all methods as a tuple (D1, D2, D3, D4)
    
    Returns:
    ------------
    np.ndarray
        A tensor the same dimension as im_eigvals containing the 
        local enhancement values for the specified method

        If the method is 'all', returns a tuple of tensors for each method

    References:
    [1] Shokouh, Ghulam-Sakhi, et al. "Ridge detection by image filtering techniques: A review and an objective analysis." 
    Pattern Recognition and Image Analysis 31 (2021): 551-570.
    """
    if method == 1:
        return im_eigvals[1, :, :]
    elif method == 2:
        return (im_eigvals[1] ** 2 - im_eigvals[0] ** 2) ** 2
    elif method == 3:
        return (im_eigvals[1] - im_eigvals[0]) ** 2
    elif method == 4:
        return np.abs(im_eigvals[1, :, :] - im_eigvals[0, :, :]) * np.abs(im_eigvals[1, :, :] + im_eigvals[0, :, :])
    elif method == 5:
        lambda_1 = im_eigvals[0]
        det_H = im_eigvals[0] * im_eigvals[1]  # or however you're computing det(H)

        # Compute the correction only if det_H is negative
        # correction = np.where(det_H < 0, np.sqrt(np.abs(det_H)), 0)
        correction = np.where(det_H < 0, np.abs(det_H), 0)

        return lambda_1 - correction
    
    elif method == 6:
        determinant = im_eigvals[0] * im_eigvals[1]
        return im_eigvals[1] - np.sqrt(np.abs(determinant))
    elif method == 7:
        determinant = im_eigvals[0] * im_eigvals[1]
        term = np.sqrt(np.abs(im_eigvals[1] ** 2 - im_eigvals[0] ** 2))
        return term - np.sqrt(np.abs(determinant))
    
    elif method == "all":
        D1 = im_eigvals[1, :, :]
        D2 = (im_eigvals[1] ** 2 - im_eigvals[0] ** 2) ** 2
        D3 =  (im_eigvals[1] - im_eigvals[0]) ** 2
        D4 = np.abs(im_eigvals[1, :, :] - im_eigvals[0, :, :]) * np.abs(im_eigvals[1, :, :] + im_eigvals[0, :, :])
        return D1, D2, D3, D4  
    else:
        print(f"Method {method} unrecognized")
        raise ValueError 



def gamma_to_lp_norm(gamma, m, D=2):
    '''
    Returns the p of the Lp norm to normalize the Gaussian derivative kernel over scales
    m : the order of the derivative of the Gaussian kernel (corresponds to m in equation)
    D : the dimensionality of the signal (default 2 for images) (corresponds to D in equation)
    '''
    return 1 / (1 + (1 - gamma) * (m / D))


def construct_scale_space_helper(s, im, gamma, ridge_strength_method, scale_space_filter,
              filter_mode, zc_method, zc_ks, eps_r, eps_c1, eps_c2):
    """
    Helper function for construct_scale_space that generates a set of scale space features
    of the input image `im` at a specific scale `s`

    Utilizes the pyscsp library to compute the scale space features
    * pyscsp.discscsp.computeNjetfcn for Gaussian derivatives

    The ridge conditions are defined in Haralick 1983 [1] (see also Lindeberg 1996 [2])

    Note that there are some deprecated functionality, such as 
    * scale_space_filter = "2d-mean" which is not recommended for use

    See `construct_scale_space` for the description of the parameters

    References:
    [1] Haralick, Robert M., et al. “The Topographic Primal Sketch.” 
        The International Journal of Robotics Research, vol. 2, no. 1, Mar. 1983, 
        pp. 50-72. DOI.org (Crossref), https://doi.org/10.1177/027836498300200105.
    [2] Lindeberg, T. “Edge Detection and Ridge Detection with Automatic Scale Selection.” 
        Proceedings CVPR IEEE Computer Society Conference on Computer Vision and Pattern Recognition, IEEE, 1996, 
        pp. 465-70. DOI.org (Crossref), https://doi.org/10.1109/CVPR.1996.517113.
    """

    # Which kernel to use? Gaussian or 2D Mean
    if scale_space_filter == "gaussian":

        # Generate image blurred
        im_blur = pyscsp.discscsp.computeNjetfcn(
            im, 'L', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)
        I_s = im_blur

        # Gradient magnitude
        im_Lv = pyscsp.discscsp.computeNjetfcn(im, 'Lv', s, gamma=gamma,
                                               normdermethod="discgaussvar", filter_mode=filter_mode)
        
        # Second derivatives
        im_xx = -pyscsp.discscsp.computeNjetfcn(
            im, 'Lxx', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_yy = -pyscsp.discscsp.computeNjetfcn(
            im, 'Lyy', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_xy = -pyscsp.discscsp.computeNjetfcn(
            im, 'Lxy', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)

        # Must compute the spatial ridgeness too
        # Directional derivatives (first order)
        im_p = pyscsp.discscsp.computeNjetfcn(
            im, 'Lp', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_q = pyscsp.discscsp.computeNjetfcn(
            im, 'Lq', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)

        # Directional derivatives (second order)
        im_pp = pyscsp.discscsp.computeNjetfcn(
            im, 'Lpp', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_qq = pyscsp.discscsp.computeNjetfcn(
            im, 'Lqq', s, gamma=gamma, normdermethod="discgaussvar", filter_mode=filter_mode)

    elif scale_space_filter == "2d-mean":
        # This is not recommended for use but kept just in case
        p0 = gamma_to_lp_norm(gamma, 0, 2)
        p1 = gamma_to_lp_norm(gamma, 1, 2)
        p2 = gamma_to_lp_norm(gamma, 2, 2)

        kx = np.ones(s)
        kx /= np.linalg.norm(kx, ord=p0)
        mean_filter_order0 = np.outer(kx, kx)

        kx = np.ones(s)
        kx /= np.linalg.norm(kx, ord=p1)
        mean_filter_order1 = np.outer(kx, kx)

        kx = np.ones(s)
        kx /= np.linalg.norm(kx, ord=p2)
        mean_filter_order2 = np.outer(kx, kx)

        im_blur = cv.filter2D(im, ddepth=cv.CV_64F, kernel=mean_filter_order0)
        I_s = im_blur

        im_blur_order1 = cv.filter2D(im, ddepth=cv.CV_64F, kernel=mean_filter_order1)
        im_blur_order2 = cv.filter2D(im, ddepth=cv.CV_64F, kernel=mean_filter_order2)

        im_xx = -pyscsp.discscsp.computeNjetfcn(
            im_blur_order2, 'Lxx', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_yy = -pyscsp.discscsp.computeNjetfcn(
            im_blur_order2, 'Lyy', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_xy = -pyscsp.discscsp.computeNjetfcn(
            im_blur_order2, 'Lxy', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)

        # Must compute the spatial ridgeness too
        im_p = pyscsp.discscsp.computeNjetfcn(
            im_blur_order1, 'Lp', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_q = pyscsp.discscsp.computeNjetfcn(
            im_blur_order1, 'Lq', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)

        im_pp = pyscsp.discscsp.computeNjetfcn(
            im_blur_order2, 'Lpp', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)
        im_qq = pyscsp.discscsp.computeNjetfcn(
            im_blur_order2, 'Lqq', 3, gamma=1, normdermethod="discgaussvar", filter_mode=filter_mode)
    else:
        print("`scale_space_filter` must be either 'gaussian' or '2d-mean'")
        raise ValueError

    # Enforce gradient magnitude conditions (Haralick)
    grad_nonzero = (im_Lv >= eps_r)   # Nonflat ridges
    grad_zero    = ~grad_nonzero         # Flat ridges

    # First derivative conditions 
    im_p_nearzero = zerocross(im_p, zc_method, zc_ks)
    im_q_nearzero = zerocross(im_q, zc_method, zc_ks)

    # Second derivative conditions using epsilon thresholds
    im_pp_neg      = (im_pp < -eps_r)                     # Significantly negative
    im_pp_nearzero = (~im_pp_neg) & (np.abs(im_pp) <= eps_r) # Effectively zero, only if not negative

    # For the q-direction:
    im_qq_neg      = (im_qq < -eps_r)
    im_qq_nearzero = (~im_qq_neg) & (np.abs(im_qq) <= eps_r)

    # Nonflat ridges: Case 1 and Case 2
    ridges1 = np.logical_and.reduce([
        grad_nonzero,   # Gradient is non-zero 
        im_p_nearzero,  # Zero crossing in p-direction  Lp
        im_pp_neg       # Negative curvature in p-direction Lpp
    ])
    ridges2 = np.logical_and.reduce([
        grad_nonzero,   # Gradient is non-zero
        im_q_nearzero,  # Zero crossing in q-direction
        im_qq_neg       # Negative curvature in q-direction
    ])

    # Flat ridges (Case 3)
    ridges3a = np.logical_and.reduce([
        grad_zero,      # Gradient is zero
        im_pp_neg,      # Negative curvature in p-direction
        im_qq_nearzero  # Near-zero curvature in q-direction
    ])
    ridges3b = np.logical_and.reduce([
        grad_zero,
        im_qq_neg,
        im_pp_nearzero
    ])

    # We populate the tensor R with the actual value where each is true
    # in order of priority
    R_s = np.zeros_like(im_Lv)
    R_s[ridges3b]  = True
    R_s[ridges3a]  = True
    R_s[ridges2]   = True
    R_s[ridges1]   = True

    # Construct CORNER boolean condition
    # Have a separate epsilon to control 
    # gradient floating `eps_c1` and the determinant condition `eps_c2`
    grad_nonzero = (im_Lv >= eps_c1)  
    grad_zero    = ~grad_nonzero         
    det_neg = im_pp * im_qq < -eps_c2
    C_s = np.logical_and.reduce([grad_zero, det_neg])


    im_eigvals, im_eigvecs = image_eigh(
        np.array([[im_xx, im_xy], [im_xy, im_yy]]))

    V_s = im_eigvecs
    W1_s = im_eigvals[1]  # Larger eigenvalue
    W2_s = im_eigvals[0]  # Smaller eigenvalue

    # Compute ridge strength
    d = local_contrast_enhancement(im_eigvals, ridge_strength_method)

    if ridge_strength_method in [1, 5, 6, 7]:
        D_s = d
    elif ridge_strength_method == 2:
        D_s = d ** 0.25
    elif ridge_strength_method in [3, 4]:
        D_s = d ** 0.5

    return I_s, D_s, W1_s, W2_s, V_s, R_s, C_s




def construct_scale_space(im, s_range, gamma, ridge_strength_method,
                          scale_space_filter, filter_mode, eps_r, eps_c1, eps_c2,
                          zc_method=1, zc_ks=3, num_pools=None):
    """
    Constructs scale space tensors for the input image `im` over the specified scale range `s_range`

    Each scale space feature is a tensor of dimension (len(s_range), im.shape)
    
    The following scale space features are generated:
    1. I: Image where I[s] is the image `im` convolved with a Gaussian kernel of scale `s`
    2. D: Ridge strength tensor where D[s] is the ridge strength at scale `s`
        * The ridge strength method is specified by `ridge_strength_method`
    3. W1: Largest eigenvalue tensor where W1[s] is the first eigenvalue of the 2x2 Hessian matrix at scale `s`
    4. W2: Smallest eigenvalue tensor where W2[s] is the second eigenvalue of the 2x2 Hessian matrix at scale `s`
    5. A: Angle tensor where A[s] is the angle (in degrees) of the eigenvector corresponding to W1 at scale `s`
    6. R: Ridge condition tensor where R[s] is a boolean tensor indicating whether the ridge condition is satisfied at scale `s`
    7. C: Corner condition tensor where C[s] is a boolean tensor indicating whether the corner condition is satisfied at scale `s`

    Parameters:
    im : np.ndarray
        Input image of shape (height, width) 
    s_range : np.ndarray
        Array of scales at which to compute the scale space features
    gamma : float
        The gamma parameter is used to normalize the Gaussian derivative kernel over scales
        A value of 0.75 is suggested to detect ridges [1]
    ridge_strength_method : int
        Method to compute the ridge strength (1-7) as defined in `local_contrast_enhancement`
    scale_space_filter : str
        Type of filter to use for scale space construction, either 'gaussian' or '2d-mean'
    filter_mode : str
        Convolution padding mode for scipy.ndimage.correlate1d
        {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
    eps_r : float
        Epsilon tolerances for the corner condition
    eps_c1 : float
        Epsilon tolerance for the corner condition
    eps_c2 : float
        Epsilon tolerance for the corner condition
    zc_method : int, optional
        Method to use for zero-crossing detection (1 or 2)
        Default is 1. See description in `zerocross` function
    zc_ks : int, optional
        Kernel size for zero-crossing detection
        Default is 3. See description in `zerocross` function
    num_pools : int, optional
        Number of parallel processes to use for scale space construction

    Returns:
    tuple
        Tuple containing the scale space tensors:
        (I, D, W1, W2, A, R, C) where:
        - I: Image tensor
        - D: Ridge strength tensor
        - W1: Largest eigenvalue tensor
        - W2: Smallest eigenvalue tensor
        - A: Angle tensor
        - R: Ridge condition tensor
        - C: Corner condition tensor

    References:
    [1] Lindeberg, T. Edge Detection and Ridge Detection with Automatic Scale Selection. 
    International Journal of Computer Vision 30, 117–156 (1998). https://doi.org/10.1023/A:1008097225773
    """
    # Prepare the partial function with fixed arguments
    process_s_partial = partial(
        construct_scale_space_helper, im=im, gamma=gamma, ridge_strength_method=ridge_strength_method,
        scale_space_filter=scale_space_filter, 
        eps_r=eps_r, eps_c1=eps_c1, eps_c2=eps_c2,
        filter_mode=filter_mode, zc_method=zc_method, zc_ks=zc_ks)

    if num_pools is not None and num_pools > 1:
        with multiprocessing.Pool(processes=num_pools) as pool:
            results = pool.map(process_s_partial, s_range)
    else:
        results = [process_s_partial(s) for s in s_range]

    # Unpack the results
    I_list, D_list, W1_list, W2_list, V_list, R_list, C_list = zip(*results)

    # Convert lists to numpy arrays
    I = np.array(I_list)
    D = np.array(D_list)
    W1 = np.array(W1_list)
    W2 = np.array(W2_list)
    V = np.array(V_list)

    A = eigenvector_space_to_angle(V)

    R = np.array(R_list)
    C = np.array(C_list)

    return I, D, W1, W2, A, R, C



from scipy.interpolate import RegularGridInterpolator
from scipy.signal import argrelmax


def extract_line_scale_space_neighborhood(ridge_coords, scale_space_container, s_range):
    all_curves = []
    
    for tensor in scale_space_container:
        
        curves = []
        for i in range(len(ridge_coords)):

            # one position
            curve = []
            for j in range(len(s_range)):

                r = np.round(ridge_coords[:, 1][i], 0).astype(int) # y is the rows
                c = np.round(ridge_coords[:, 0][i], 0).astype(int) # x is the columns

                voxel = get_neighborhood(tensor, j, r, c)

                val = np.sum(voxel) / np.size(voxel)

                # if R_chunk[j, rows[i], cols[i]]:
                #     # overwrite to 1 if that position is True
                #     val = 1

                curve.append(val)

            curves.append(curve)
        all_curves.append(np.array(curves).T)
        
    return tuple(all_curves)

def get_neighborhood(tensor, x, y, z):
    # Get the dimensions of the tensor
    dim_x, dim_y, dim_z = tensor.shape
    
    # Calculate the safe bounds for extracting the neighborhood
    x_lb = max(0, x - 1)
    x_ub = min(dim_x, x + 2)
    
    y_lb = max(0, y - 1)
    y_ub = min(dim_y, y + 2)
    
    z_lb = max(0, z - 1)
    z_ub = min(dim_z, z + 2)
    
    # Extract the neighborhood within the safe bounds
    neighborhood = tensor[x_lb:x_ub, y_lb:y_ub, z_lb:z_ub]
    
    return neighborhood

def extract_line_scale_space(ridge_coords, scale_space_container):
    """
    Given the `ridge_coords`, extracts values from each tensor in `scale_space_container` at the specified `ridge_coords`
    The extraction is done using linear interpolation to preserve sub-pixel accuracy
    The ridge coordinates should be the output of
        ridge_coords_curve = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0)
    
    Parameters
    ----------
    ridge_coords : np.ndarray
        Array of shape (n, 2) with [x, y] coordinates where the first column is x (column) and the second column is y (row)
    scale_space_container : list or iterable
        Iterable of tensors, each with shape (num_scales, height, width) from which to extract the curves
    
    Returns
    -------
    np.ndarray or tuple
        If `scale_space_container` contains multiple tensors, 
            returns a tuple of arrays, each corresponding to the curves extracted from each tensor.
        If `scale_space_container` is a single nd.array tensor, 
            returns a nd.array of shape (num_scales, l) where l is the number of ridge coordinates
    """
    all_curves = []
    
    for tensor in scale_space_container:
        x = np.arange(tensor.shape[1])  
        y = np.arange(tensor.shape[2])  

        # Clamp ridge_coords to stay within bounds
        clamped_coords = np.clip(ridge_coords, [0, 0], [tensor.shape[2] - 1, tensor.shape[1] - 1])
        
        curves = []
        
        for z in range(len(tensor)):
            interpolator = RegularGridInterpolator((x, y), tensor[z], method='linear')
            curves.append(interpolator(clamped_coords[:, [1, 0]])) # swap 
            # curves.append(interpolator(ridge_coords)) # swap 

        all_curves.append(np.array(curves))
        
    if len(scale_space_container) > 1:
        return tuple(all_curves)
    return np.array(all_curves).squeeze(axis=0)

def round_line_scale_space(ridge_coords, scale_space_container):
    """
    Given the `ridge_coords`, extracts values from each tensor in `scale_space_container` at the specified `ridge_coords`
    The extraction is done by rounding the coordinates to the nearest integer
    The ridge coordinates should be the output of
        ridge_coords_curve = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0)

    Parameters
    ----------
    ridge_coords : np.ndarray
        Array of shape (n, 2) with [x, y] coordinates where the first column is x (column) and the second column is y (row)
    scale_space_container : list or iterable
        Iterable of tensors, each with shape (num_scales, height, width) from which to extract the curves
    Returns
    -------
    np.ndarray or tuple
        If `scale_space_container` contains multiple tensors,
            returns a tuple of arrays, each corresponding to the curves extracted from each tensor.
        If `scale_space_container` is a single nd.array tensor,
            returns a nd.array of shape (num_scales, l) where l is the number of ridge coordinates
    """
    all_curves = []

    for tensor in scale_space_container:
        # Clamp ridge_coords to ensure they are within valid bounds.
        # x (first column) is clamped to [0, tensor.shape[2]-1]
        # y (second column) is clamped to [0, tensor.shape[1]-1]
        clamped_coords = np.clip(ridge_coords, [0, 0], [tensor.shape[2] - 1, tensor.shape[1] - 1])
        
        # Round the coordinates and convert to integers.
        rounded_coords = np.rint(clamped_coords).astype(int)
        
        # For direct extraction from the tensor, use:
        #   tensor[channel, row, col]
        # Here, rows come from the second column (y) and columns from the first column (x).
        rows = rounded_coords[:, 1]  # y-coordinates
        cols = rounded_coords[:, 0]  # x-coordinates
        
        curves = tensor[:, rows, cols]
        all_curves.append(curves)

    if len(scale_space_container) > 1:
        return tuple(all_curves)
    return np.array(all_curves).squeeze(axis=0)


def extract_angle_scale_space(ridge_coords, tensor):
    """
    Given ridge_coords, extracts angular values from the provided tensor by performing
    angular interpolation using a sine-cosine method.
    
    This function assumes:
      - ridge_coords is of shape (n, 2) with [x, y] coordinates (first column is x, second is y).
      - tensor is a 3D numpy array with shape (channels, height, width) containing angles in degrees [0, 180).
      
    For integer coordinates, the output will be identical to direct extraction.
    
    Parameters:
        ridge_coords (np.ndarray): Array of shape (n, 2) with [x, y] coordinates.
        tensor (np.ndarray): A tensor with shape (channels, height, width) containing angular values.
    
    Returns:
        np.ndarray: An array of shape (channels, n) where each row corresponds to the interpolated curve
                    for the respective channel.
    """
    channels, height, width = tensor.shape

    # Clamp ridge_coords to valid bounds:
    # x (first column) is clamped to [0, width-1] and y (second column) to [0, height-1].
    clamped_coords = np.clip(ridge_coords, [0, 0], [width - 1, height - 1])
    
    # Create query coordinates in (row, col) order (i.e., [y, x])
    query_coords = clamped_coords[:, [1, 0]]
    
    # Prepare an array to hold the interpolated values for each channel.
    interpolated_values = np.zeros((channels, query_coords.shape[0]))
    
    # Create grid arrays corresponding to row and column indices.
    grid_y = np.arange(height)
    grid_x = np.arange(width)
    
    for c in range(channels):
        # Extract the 2D angle data for this channel.
        angle_data = tensor[c, :, :]
        
        # Convert angles (in degrees) to doubled angles (in radians) to map [0, 180) to [0, 360).
        angle_data_rad = np.deg2rad(angle_data * 2)
        
        # Compute sine and cosine components.
        sin_data = np.sin(angle_data_rad)
        cos_data = np.cos(angle_data_rad)
        
        # Build interpolators for sine and cosine.
        interp_sin = RegularGridInterpolator((grid_y, grid_x), sin_data, method='linear')
        interp_cos = RegularGridInterpolator((grid_y, grid_x), cos_data, method='linear')
        
        # Interpolate sine and cosine at the sub-pixel coordinates.
        sin_interp = interp_sin(query_coords)
        cos_interp = interp_cos(query_coords)
        
        # Reconstruct the interpolated doubled angle, then divide by 2.
        interp_angle_rad = np.arctan2(sin_interp, cos_interp) / 2.0
        interp_angle_deg = np.rad2deg(interp_angle_rad)
        
        # Ensure the angle is in the range [0, 180)
        interp_angle_deg = np.mod(interp_angle_deg, 180)
        
        # Store the interpolated curve for this channel.
        interpolated_values[c, :] = interp_angle_deg
    
    return interpolated_values




def create_maxima_set(s_range, D_curves, R_curves, method="", verbose=False):
    """
    Method: 
    * "local" :
        Function returns a subset of s_range where 
            1. D_curves is global maximum AND R_curves is true
            2. If (1) is not satisfied for a given position (column),  
                Then D_curves is local maximum AND R_curves is true
                If there are multiple local maximum that has R_curves true, then
                select the one where D_curves value is maximum
            3. If (3) is not satisfied, then does not return an s_range for that position (column)
        Returns assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond
        where
        * assigned_s : final list including NaNs if not assigned properly
        * assigned_s_ridge_cond : final list including NaNs if not assigned properly
        * global_max : scale at global max
        * global_max_ridge_cond : boolean array of whether global max is True/False
        * local max: scale at local max (not global max) 
            Note this is a tuple because there can be multiple local max
        * local_max_ridge_cond: boolean array of whether local max is True/False
            Note this is a tuple because there can be multiple local max
        FROM ip_combine_scale_median.ipynb
    * "min_scale" :
        Function returns a subset of s_range where out of all local and global maxima, 
        the minimum scale is returned
        Same return variables as "local" 

    FROM scale_space.py
    """
    if method == "local":
        return process_local_method(s_range, D_curves, R_curves, verbose)
    elif method == "min_scale":
        return process_min_scale_method(s_range, D_curves, R_curves, verbose)
    elif method == "global":
        assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond = process_local_method(s_range, D_curves, R_curves, verbose)

        return global_max, global_max_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond

    else:
        raise ValueError("Unknown method: {}".format(method))
    



def process_local_method(s_range, D_curves, R_curves, verbose=False):
    assigned_s = []
    assigned_s_ridge_cond = []
    extrema_r, extrema_c = argrelmax(D_curves, order=1, axis=0)

    global_max_r = np.argmax(D_curves, axis=0)
    global_max_c = np.arange(D_curves.shape[1])
    global_max_ridge_cond = R_curves[global_max_r, global_max_c] > 0
    global_max = s_range[global_max_r]

    local_max = []
    local_max_ridge_cond = []

    for c in range(D_curves.shape[1]):
        # Indices of all local maxima at position c
        local_extrema_per_pos = extrema_r[extrema_c == c]
        # Ridge condition for each local maxima
        local_extrema_cond_per_pos = R_curves[local_extrema_per_pos, c] > 0

        if verbose:
            print(f"pos={c} Global max: {global_max_r[c]} ({global_max_ridge_cond[c]}) "
                  f"Local maxima: {local_extrema_per_pos} ({local_extrema_cond_per_pos})")

        if global_max_ridge_cond[c]:
            # Global maximum satisfies ridge condition
            assigned_s.append(s_range[global_max_r[c]])
            assigned_s_ridge_cond.append(True)
            if verbose:
                print(f"\tGlobal max TRUE, assigned scale at {global_max_r[c]}: {s_range[global_max_r[c]]:.2f}")
        else:
            # Check other local maxima
            if len(local_extrema_per_pos) == 0 or not np.any(local_extrema_cond_per_pos):
                assigned_s.append(np.nan)
                assigned_s_ridge_cond.append(False)
                if verbose:
                    print("\tNo suitable local maxima with ridge condition TRUE. Assigned NaN.")
            else:
                # Select local maxima with maximum D_curves value
                local_max_ridge_true = local_extrema_per_pos[local_extrema_cond_per_pos]
                local_max_ridge_true_max = local_max_ridge_true[np.argmax(D_curves[local_max_ridge_true, c])]
                assigned_s.append(s_range[local_max_ridge_true_max])
                assigned_s_ridge_cond.append(True)
                if verbose:
                    print(f"\tAssigned scale at local max {local_max_ridge_true_max}: {s_range[local_max_ridge_true_max]:.2f}")

        # Collect local maxima excluding the global maximum
        local_max_per_pos_r = np.setdiff1d(local_extrema_per_pos, global_max_r[c])
        local_max_per_pos_ridge_cond = R_curves[local_max_per_pos_r, c] > 0

        local_max.append(s_range[local_max_per_pos_r])
        local_max_ridge_cond.append(local_max_per_pos_ridge_cond)

        if verbose:
            print()

    # Pad arrays for consistent output
    local_max = pad_arrays(local_max)
    local_max_ridge_cond = pad_arrays(local_max_ridge_cond, False)

    return assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond

def process_min_scale_method(s_range, D_curves, R_curves, verbose=False):
    assigned_s = []
    assigned_s_ridge_cond = []
    extrema_r, extrema_c = argrelmax(D_curves, order=1, axis=0)

    global_max_r = np.argmax(D_curves, axis=0)
    global_max_c = np.arange(D_curves.shape[1])
    global_max_ridge_cond = R_curves[global_max_r, global_max_c] > 0
    global_max = s_range[global_max_r]

    local_max = []
    local_max_ridge_cond = []

    for c in range(D_curves.shape[1]):
        # Indices of all local maxima at position c
        local_extrema_per_pos = extrema_r[extrema_c == c]
        # Include the global maximum index
        maxima_indices = np.append(local_extrema_per_pos, global_max_r[c])
        maxima_indices = np.unique(maxima_indices)

        # Ridge condition for these maxima
        maxima_ridge_cond = R_curves[maxima_indices, c] > 0

        if verbose:
            print(f"pos={c} Maxima indices: {maxima_indices} ({maxima_ridge_cond})")

        # Maxima indices where ridge condition is True
        maxima_indices_ridge_true = maxima_indices[maxima_ridge_cond]

        if len(maxima_indices_ridge_true) == 0:
            assigned_s.append(np.nan)
            assigned_s_ridge_cond.append(False)
            if verbose:
                print("\tNo maxima with ridge condition TRUE. Assigned NaN.")
        else:
            # Select the one with minimum scale value
            s_values = s_range[maxima_indices_ridge_true]
            min_s_index = maxima_indices_ridge_true[np.argmin(s_values)]
            assigned_s.append(s_range[min_s_index])
            assigned_s_ridge_cond.append(True)
            if verbose:
                print(f"\tAssigned minimum scale at index {min_s_index}: {s_range[min_s_index]:.2f}")

        # Collect local maxima excluding the global maximum
        local_max_per_pos_r = np.setdiff1d(maxima_indices, global_max_r[c])
        local_max_per_pos_ridge_cond = R_curves[local_max_per_pos_r, c] > 0

        local_max.append(s_range[local_max_per_pos_r])
        local_max_ridge_cond.append(local_max_per_pos_ridge_cond)

        if verbose:
            print()

    # Pad arrays for consistent output
    local_max = pad_arrays(local_max)
    local_max_ridge_cond = pad_arrays(local_max_ridge_cond, False)

    return assigned_s, assigned_s_ridge_cond, global_max, global_max_ridge_cond, local_max, local_max_ridge_cond

def pad_arrays(arrays, pad_val=np.nan):
    """
    Helper function for `create_maxima_set`
    FROM ip_combine_scale_median.ipynb
    """
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad each array to the max length with NaNs
    padded_arrays = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=pad_val) for arr in arrays])
    return padded_arrays

def scale_to_width(scales):
    """
    Convert scale(s) to width(s) via
        w = (s - 0.5) * (2 * sqrt(3))
    where s is the scale and w is the width.

    Parameters
    ----------
    scales : float or np.ndarray
        Scale value(s).

    Returns
    -------
    float or np.ndarray
        Width value(s), same shape as input.
    """
    arr = np.asarray(scales)
    widths = (arr - 0.5) * (2 * np.sqrt(3))
    # if input was a scalar, return a Python float
    if arr.ndim == 0:
        return widths.item()
    return widths

def width_to_scale(widths):
    """
    Convert width(s) to scale(s) via
        s = w / (2 * sqrt(3)) + 0.5
    where w is the width and s is the scale.

    Parameters
    ----------
    widths : float or np.ndarray
        Width value(s).

    Returns
    -------
    float or np.ndarray
        Scale value(s), same shape as input.
    """
    arr = np.asarray(widths)
    scales = arr / (2 * np.sqrt(3)) + 0.5
    # if input was a scalar, return a Python float
    if arr.ndim == 0:
        return scales.item()
    return scales


def dec_to_log(val, base):
    return np.log(val) / np.log(base)


def compute_num_scales(s0_log, s1_log, scale_resolution):
    num_scales = np.round((s1_log - s0_log) / scale_resolution).astype(int)

    # min max
    num_scales = np.clip(num_scales, a_min=3, a_max=30)


    return num_scales


def generate_scales_from_widths(w0, w1, base, scale_resolution):
    """
    Generates scales from widths 
    """
    if w0 > w1:
        print("\tLower width larger than upper width. Ensure that w0 <= w1.")
        return ValueError

    s0 = width_to_scale(w0)
    s1 = width_to_scale(w1)

    s0_log = dec_to_log(s0, base)
    s1_log = dec_to_log(s1, base)

    num_scales = compute_num_scales(s0_log, s1_log, scale_resolution)

    scale_range = np.logspace(s0_log, s1_log, num=num_scales, base=base)

    scale_range = np.clip(scale_range, a_min=1, a_max=None)

    scale_range = np.unique(scale_range)

    return scale_range