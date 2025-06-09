import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def extract_line_scale_space(ridge_coords, scale_space_container):
    """
    Given the ridge_coords, extracts values from scale_space_container
        Ridge coords should be output of 
            ridge_coords_curve = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0) 
    FROM scale_space.py
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

        all_curves.append(np.array(curves))
        
    return tuple(all_curves)



def round_line_scale_space(ridge_coords, scale_space_container):
    """
    Given the ridge_coords, extracts values by simply rounding from scale_space_container.
    This version assumes ridge_coords is of shape (n, 2) where the first column is x (column) 
    and the second column is y (row), matching the coordinate convention used in extract_line_scale_space.
    
    For integer coordinates, the output will be identical to that of extract_line_scale_space.
    
    Parameters:
        ridge_coords (np.ndarray): Array of shape (n, 2) with [x, y] coordinates.
        scale_space_container (iterable): Iterable of tensors with shape (channels, height, width).
    
    Returns:
        tuple: A tuple of arrays, each corresponding to the curves extracted from each tensor.
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


from scipy.interpolate import RegularGridInterpolator


def interpolate_line_scale_space(ridge_coords, tensor):
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

class TestExtractLineScaleSpace(unittest.TestCase):


    def test_angle_extract_line_scale_space(self):
        # Create dummy tensor A 
        shape = (3, 4, 4)  # 1 layers of a 4x4 grid
        A = np.zeros(shape)

        # Angle assignments
        A[2, 0:2, 0:2] = 10 
        A[2, 0:2, 2:4] = 170 
        A[2, 2:4, 0:2] = 30 
        A[2, 2:4, 2:4] = 85

        A = A.astype(np.float64)
        
        ridge_coords = np.array([
            [0.5, 0.5], [0.5, 1.5], [0.5, 2.5], 
            [1.5, 2.5], [2.5, 2.5], [2.5, 1.5], 
            [2.5, 0.5], [1.5, 0.5], [1.5, 1.5],
        ])
                
        # test simple linear interpolation
        A_curves_0 = extract_line_scale_space(ridge_coords[:, [1, 0]], scale_space_container=[A])[0]

        # test simple rounding
        A_curves_1 = round_line_scale_space(ridge_coords[:, [1, 0]], scale_space_container=[A])

        A_curves_2 = interpolate_line_scale_space(ridge_coords[:, [1, 0]], A)


        # Second plot: A_curves.
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")

        ax = ax.flatten()


        im0 = ax[0].imshow(A_curves_0, interpolation='none', aspect='equal', cmap="Reds")
        ax[0].set_title('Linear interpolation')
        fig.colorbar(im0, ax=ax[0])
        # Add text annotation for each cell.
        data = A_curves_0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax[0].text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="gray")

        im1 = ax[1].imshow(A_curves_1, interpolation='none', aspect='equal', cmap="Reds")
        ax[1].set_title('Rounded')
        fig.colorbar(im1, ax=ax[1])
        # Add text annotation for each cell.
        data = A_curves_1
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax[1].text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="gray")


        im2 = ax[2].imshow(A_curves_2, interpolation='none', aspect='equal', cmap="Reds")
        ax[2].set_title('Sin Cosine interpolation')
        fig.colorbar(im2, ax=ax[2])
        # Add text annotation for each cell.
        data = A_curves_2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax[2].text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="gray")
        

        plt.savefig("tests/test_angle_interp_scale_space_curves.png", dpi=300)
        plt.close(fig)



    def test_integer_extract_line_scale_space(self):
        # Create dummy tensors A and B of shape [z, x, y]
        shape = (3, 5, 5)  # 3 layers of a 5x5 grid.
        A = np.zeros(shape)
        B = np.zeros(shape)

        # bottom layer modifications for A[0]
        A[0, 0, 4] = 3
        A[0, 2, 2:5] = 2
        A[0, 4, 0:3] = 0.5

        # middle layer modifications for A[1]
        A[1, 1:4, 1:4] = 1
        A[1, 2, 2] = 2

        # top layer modifications for A[2]
        A[2, 1:4, 3] = 1
        A[2, 1:4, 4] = 1

        # B is a copy of A.
        B = np.copy(A)
        
        ridge_coords = np.array([
            [0, 1], [1, 1], [2, 1], [3, 1], [3.5, 1], [3.5, 2], [4, 2], [4, 3], [4, 4], 
            [3, 4], [2, 4], [2, 3], [2, 2.5], [2, 2], [1.5, 2.5], [0.5, 2.5], [0.5, 3.5], [0, 4], [0, 3]
        ])
                
        # Extract curves from both tensors.
        A_curves_1, B_curves_1 = extract_line_scale_space(np.round(ridge_coords[:, [1, 0]], 0), scale_space_container=[A, B])

        A_curves_2 = round_line_scale_space(ridge_coords[:, [1, 0]], scale_space_container=[A])

        assert np.allclose(A_curves_1, A_curves_2)

        # Second plot: A_curves.
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 10))
        im3 = ax2.imshow(A_curves_1.T, interpolation='none', aspect='equal', cmap="Reds")
        ax2.set_title('A_curves rounded')
        fig2.colorbar(im3, ax=ax2)

        # Add text annotation for each cell.
        data = A_curves_1.T
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax2.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="gray")
        
        plt.tight_layout()
        plt.savefig("tests/test_round_line_scale_space_curves.png", dpi=300)
        plt.close(fig2)

        


    def test_extract_line_scale_space(self):
        # Create dummy tensors A and B of shape [z, x, y]
        shape = (3, 5, 5)  # 3 layers of a 5x5 grid.
        A = np.zeros(shape)
        B = np.zeros(shape)

        # bottom layer modifications for A[0]
        A[0, 0, 4] = 3
        A[0, 2, 2:5] = 2
        A[0, 4, 0:3] = 0.5

        # middle layer modifications for A[1]
        A[1, 1:4, 1:4] = 1
        A[1, 2, 2] = 2

        # top layer modifications for A[2]
        A[2, 1:4, 3] = 1
        A[2, 1:4, 4] = 1

        # B is a copy of A.
        B = np.copy(A)
        
        ridge_coords = np.array([
            [0, 1], [1, 1], [2, 1], [3, 1], [3.5, 1], [3.5, 2], [4, 2], [4, 3], [4, 4], 
            [3, 4], [2, 4], [2, 3], [2, 2.5], [2, 2], [1.5, 2.5], [0.5, 2.5], [0.5, 3.5], [0, 4], [0, 3]
        ])
                
        # Extract curves from both tensors.
        A_curves, B_curves = extract_line_scale_space(ridge_coords[:, [1, 0]], scale_space_container=[A, B])
        
        # Plotting:
        # Create a figure with 2 rows and 3 columns.
        # Top row: display the bottom, middle, and top layers of A.
        # Bottom row: display the outputs A_curves and B_curves.
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # First plot: 3 layers of A.
        fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axs1[0].imshow(A[0], interpolation='none', aspect='equal', cmap="Reds")
        axs1[0].set_title('A Bottom Layer (A[0])')
        fig1.colorbar(im0, ax=axs1[0])
        
        im1 = axs1[1].imshow(A[1], interpolation='none', aspect='equal', cmap="Reds")
        axs1[1].set_title('A Middle Layer (A[1])')
        fig1.colorbar(im1, ax=axs1[1])
        
        im2 = axs1[2].imshow(A[2], interpolation='none', aspect='equal', cmap="Reds")
        axs1[2].set_title('A Top Layer (A[2])')
        fig1.colorbar(im2, ax=axs1[2])
        
        plt.tight_layout()
        plt.savefig("tests/test_extract_line_scale_space_layers.png", dpi=300)
        plt.close(fig1)
        
        # Second plot: A_curves.
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 10))
        im3 = ax2.imshow(A_curves.T, interpolation='none', aspect='equal', cmap="Reds")
        ax2.set_title('A_curves')
        fig2.colorbar(im3, ax=ax2)

        # Add text annotation for each cell.
        data = A_curves.T
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax2.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="gray")
        
        plt.tight_layout()
        plt.savefig("tests/test_extract_line_scale_space_curves.png", dpi=300)
        plt.close(fig2)

if __name__ == '__main__':
    unittest.main()
