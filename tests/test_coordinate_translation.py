import unittest
import numpy as np
import logging
import sys
from utils.processing import read_hic_rectangle, read_hic_file, kth_diag_indices
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import stats
import random


logging.basicConfig(
    level=logging.INFO,              # Capture DEBUG and higher messages
    stream=sys.stdout,                # Output to standard output
    format='%(levelname)s: %(message)s'
)

def extract_coords(coords):
    # If coords is an Nx2 array, split it into two 1D arrays.
    if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 2:
        return coords[:, 0], coords[:, 1]
    else:
        # Otherwise, assume coords is already in (i, j) format.
        return coords

def square_to_rect(N, window_size_bin, coords):
    """
    Convert coordinates (i, j) in the original NxN square (in bin units) to coordinates 
    (r_rect, c_rect) in the rotated rectangle (i.e. the submatrix extracted along the main diagonal),
    such that rect_to_square (with its final horizontal flip and swap) recovers the original UT coordinate exactly.
    
    This function enforces the UT region (if i > j, swap them) and then computes:
       offset = (N*sqrt(2)/2) - (window_size_bin/sqrt(2))
       raw_r = ((j + (N - 1 - i)) / sqrt(2)) - offset
       raw_r_min = ((N - 1) / sqrt(2)) - offset     (for i=j)
       raw_r_max = ((window_size_bin + (N - 1)) / sqrt(2)) - offset  (for i=0, j=window_size_bin)
       scale = (window_size_bin_rect - 1) / (raw_r_max - raw_r_min), 
            where window_size_bin_rect = ceil(window_size_bin/sqrt(2))
       r_rect = (raw_r - raw_r_min) * scale
       c_rect = N*sqrt(2) - 1 - ((2*N - 2 - i - j)/sqrt(2))
       
    These formulas were derived by requiring that:
         rect_to_square(N, window_size_bin, square_to_rect(N, window_size_bin, (i,j))) == (i, j)
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
    Convert (r_rect, c_rect) in the rotated rectangle back to (i, j) in the original NxN square.
    
    This version has your final line:
        j = N - j - 1
        return (j, i)
    to do a horizontal flip + swap, but we also ensure the final (i2, j2) has i2<= j2 
    by swapping if needed.
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



def generate_ut_coords(N, window_size_bin):
    """
    Collect all UT indices from an N x N square that satisfy j - i <= window_size_bin,
    and randomly return one coordinate (i, j).
    """
    row_arr = []
    col_arr = []
    for i in range(window_size_bin):
        row, col = kth_diag_indices(np.zeros((N, N)), i)
        row_arr += list(row)
        col_arr += list(col)
    return row_arr, col_arr



class TestCoordTranslationWithOffset(unittest.TestCase):

    
    # def test_round_trip_square_synthetic(self):
    #     """
    #     Test that converting a UT coordinate from the original square to the rotated rectangle
    #     and back recovers the original UT coordinate.
    #     """
    #     N = 100  # synthetic side-length
    #     window_size_bin = 20  # synthetic extraction parameter
        
    #     for _ in range(50):
    #         ut_coords = generate_ut_coords(N, window_size_bin)

    #         ut_coords = np.array(ut_coords)

    #         i, j = random.choice(list(ut_coords.T))

    #         rect_coords = square_to_rect(N, window_size_bin, (i, j))
    #         # logging.info(f"{rect_coords}")
    #         i2, j2 = rect_to_square(N, window_size_bin, rect_coords)
    #         # logging.info(f"{(i2, j2)}")

    #         self.assertAlmostEqual(i, i2, places=0,
    #                                msg=f"Square round-trip failed for i: {i} vs {i2}")
    #         self.assertAlmostEqual(j, j2, places=0,
    #                                msg=f"Square round-trip failed for j: {j} vs {j2}")


    def test_round_trip_square_synthetic_vectorized(self):
        """
        Test that converting a UT coordinate from the original square to the rotated rectangle
        and back recovers the original UT coordinate.
        """
        N = 100  # synthetic side-length
        window_size_bin = 20  # synthetic extraction parameter
        indices = []
        

        for _ in range(50):
            ut_coords = generate_ut_coords(N, window_size_bin)

            ut_coords = np.array(ut_coords)

            i, j = random.choice(list(ut_coords.T))

            indices.append([i, j])

        indices = np.array(indices)



        rect_coords = square_to_rect(N, window_size_bin, indices)
        # logging.info(f"{rect_coords}")
        recon_indices = rect_to_square(N, window_size_bin, rect_coords)
        # logging.info(f"{(i2, j2)}")

        self.assertTrue(np.allclose(recon_indices[:, 0], indices[:, 0], atol=0.5, rtol=0),
                        msg=f"Square round-trip failed")
        self.assertTrue(np.allclose(recon_indices[:, 0], indices[:, 0], atol=0.5, rtol=0),
                         msg=f"Square round-trip failed")



    # def test_round_trip_rectangle_synthetic(self):
    #     """
    #     Instead of generating random rectangle coordinates directly, we generate random UT coordinates,
    #     convert them to rectangle coordinates, and then back. This guarantees that the original square
    #     coordinates are from the UT region.
    #     """
    #     N = 100
    #     window_size_bin = 20
        
    #     for _ in range(50):
    #         ut_coords = generate_ut_coords(N, window_size_bin)

    #         ut_coords = np.array(ut_coords)

    #         i, j = random.choice(list(ut_coords.T))
    #         # Get rectangle coordinates from UT indices.
    #         rect_coords = square_to_rect(N, window_size_bin, (i, j))
    #         # Inverse: from rectangle back to square.
    #         i2, j2 = rect_to_square(N, window_size_bin, rect_coords)
    #         # Then re-compute rectangle coordinates from these recovered UT indices.
    #         rect_coords2 = square_to_rect(N, window_size_bin, (i2, j2))
    #         # They should be identical.

    #         if np.round(rect_coords[0]) < 0 or np.round(rect_coords[1]) < 0:
    #             logging.info(f"")
    #             logging.info(f"{(i, j)}")
    #             logging.info(f"{rect_coords}")
            
    #         if np.round(i2) < 0 or np.round(j2) < 0:
    #             logging.info(f"")
    #             logging.info(f"{(i, j)}")
    #             logging.info(f"{rect_coords}")
    #             logging.info(f"{(i2, j2)}")
            
    #         if np.round(rect_coords2[0]) < 0 or np.round(rect_coords2[1]) < 0:
    #             logging.info(f"")
    #             logging.info(f"{(i, j)}")
    #             logging.info(f"{rect_coords}")
    #             logging.info(f"{(i2, j2)}")
    #             logging.info(f"{rect_coords2}")


    #         self.assertAlmostEqual(rect_coords[0], rect_coords2[0], places=0,
    #                                msg=f"Rectangle round-trip failed for row: {rect_coords[0]} vs {rect_coords2[0]}")
    #         self.assertAlmostEqual(rect_coords[1], rect_coords2[1], places=0,
    #                                msg=f"Rectangle round-trip failed for col: {rect_coords[1]} vs {rect_coords2[1]}")

    def test_round_trip_rectangle_synthetic_vectorized(self):
        """
        Instead of generating random rectangle coordinates directly, we generate random UT coordinates,
        convert them to rectangle coordinates, and then back. This guarantees that the original square
        coordinates are from the UT region.
        """
        N = 100
        window_size_bin = 20

        indices = []
        
        for _ in range(50):
            ut_coords = generate_ut_coords(N, window_size_bin)

            ut_coords = np.array(ut_coords)

            i, j = random.choice(list(ut_coords.T))

            indices.append([i, j])

        indices = np.array(indices)


        # Get rectangle coordinates from UT indices.
        rect_coords = square_to_rect(N, window_size_bin, indices)
        # Inverse: from rectangle back to square.
        indices2 = rect_to_square(N, window_size_bin, rect_coords)
        # Then re-compute rectangle coordinates from these recovered UT indices.
        rect_coords2 = square_to_rect(N, window_size_bin, indices2)
        # They should be identical.


        self.assertTrue(np.allclose(rect_coords[:, 0], rect_coords2[:, 0], atol=0.5, rtol=0),
                        msg=f"Rectangle round-trip failed")
        self.assertTrue(np.allclose(rect_coords[:, 1], rect_coords2[:, 1], atol=0.5, rtol=0),
                         msg=f"Rectangle round-trip failed")
        


            
    # def test_value_positions_rect_to_square(self):
    #     """
    #     For 10 random indices in the Hi-C rectangle, convert the rectangle coordinates
    #     into square coordinates, log the conversion details, and overlay the points with
    #     labels on interactive plots for visual confirmation.
    #     """

    #     hic_file = "/nfs/turbo/umms-minjilab/mingjiay/GSE199059_wt_selected_30_new.hic"
    #     resolution = int(50e3)
    #     window_size_bin = int(np.ceil(6e6 / resolution))
        
    #     # Read in the full square matrix and the extracted rectangle.
    #     square = read_hic_file(hic_file, "chr1", resolution, "all", "oe", "KR", verbose=False)
    #     rect = read_hic_rectangle(hic_file, "chr1", resolution, window_size_bin,
    #                             "oe", "KR", rotate_mode="mirror", cval=0,
    #                             handle_zero_sum=None, verbose=False)
        
    #     logging.info("Square shape: %s", square.shape)
    #     logging.info("Rectangle shape: %s", rect.shape)
        
    #     # Apply Gaussian blurring to both matrices.
    #     square_blurred = gaussian_filter(square, sigma=4)
    #     rect_blurred = gaussian_filter(rect, sigma=4)
        
    #     M, N_rect = rect_blurred.shape  # dimensions of the rectangle
    #     N = square_blurred.shape[0]     # dimensions of the square (N x N)

    #     # Lists to collect points and labels.
    #     rect_points = []   # (col, row) positions in the rectangle
    #     square_points = [] # (col, row) positions in the square
    #     rect_labels = []        # label for each sampled point
    #     square_labels = []
    #     rect_values = []
    #     square_values = []

    #     # Perform 10 random comparisons.
    #     for epoch in range(50):
    #         # Pick a random index in the rectangle.
    #         r_rect = np.random.randint(0, M)
    #         c_rect = np.random.randint(0, N_rect)
    #         # Use the flipped version to match the plotting orientation.
    #         value_rect = rect_blurred[r_rect, c_rect]
            
    #         # Convert rectangle index to the corresponding square coordinates.
    #         i, j = rect_to_square(N, window_size_bin, (r_rect, c_rect))
    #         i_idx = int(round(i))
    #         j_idx = int(round(j))
            
    #         # Skip if out-of-bounds.
    #         if not (0 <= i_idx < N and 0 <= j_idx < N):
    #             continue
            
    #         value_square = square_blurred[i_idx, j_idx]
    #         logging.info("Epoch %d: rect[%d, %d] (value %.3f) -> square[%d, %d] (value %.3f)",
    #                     epoch, r_rect, c_rect, value_rect, i_idx, j_idx, value_square)
            
    #         # Save the points.
    #         rect_points.append((c_rect, r_rect))
    #         square_points.append((j_idx, i_idx))
    #         rect_labels.append(f"p{epoch}:({c_rect},{r_rect})")
    #         square_labels.append(f"p{epoch}:({j_idx},{i_idx})")
    #         rect_values.append(value_rect)
    #         square_values.append(value_square)


    #     cc = stats.spearmanr(rect_values, square_values)
    #     logging.info(f"Correlation (higher the better): {cc.statistic:.2f}")


    #     # Convert lists to numpy arrays.
    #     rect_points = np.array(rect_points)
    #     square_points = np.array(square_points)
        
    #     # --- Plot for the rectangle ---
    #     plt.figure(figsize=(40, 5), layout="constrained")
    #     # Flip the image to match the coordinate conversion.
    #     plt.imshow(np.flipud(rect_blurred), cmap='viridis', origin='lower') # for plotting only
    #     if rect_points.size > 0:
    #         adjusted_rect_points = rect_points.copy()
    #         # Adjust y-coordinates to account for the vertical flip.
    #         adjusted_rect_points[:, 1] = rect_blurred.shape[0] - rect_points[:, 1] - 1 # for plotting only
    #         plt.scatter(adjusted_rect_points[:, 0], adjusted_rect_points[:, 1],
    #                     c='red', marker='x', s=10, label='Random Points')
    #         # Annotate each point with its label.
    #         for idx, (x, y) in enumerate(adjusted_rect_points):
    #             plt.annotate(rect_labels[idx], xy=(x, y), color='white', fontsize=5,
    #                         xytext=(x+5, y+5), textcoords='data')
    #     plt.title(f'Blurred Hi-C Rectangle with Random Points {cc.statistic:.2f}')
    #     plt.savefig("output/test_value_positions_rect.png", dpi=300)
    #     plt.close()  
        
    #     # --- Plot for the square ---
    #     plt.figure(figsize=(40, 40), layout="constrained")
    #     plt.imshow(square_blurred, cmap='viridis')
    #     if square_points.size > 0:
    #         adjusted_square_points = square_points.copy()
    #         # Adjust y-coordinates using the square image height.
    #         # adjusted_square_points[:, 1] = square_blurred.shape[0] - square_points[:, 1] - 1 
    #         plt.scatter(adjusted_square_points[:, 0], adjusted_square_points[:, 1], 
    #                     c='red', marker='x', s=10, label='Converted Points')
    #         # Annotate each point with its label.
    #         for idx, (x, y) in enumerate(adjusted_square_points):
    #             plt.annotate(square_labels[idx], xy=(x, y), color='white', fontsize=5,
    #                         xytext=(x+5, y+5), textcoords='data')
    #     plt.title(f'Blurred Hi-C Square with Converted Points {cc.statistic:.2f}')
    #     plt.savefig("output/test_value_positions_square.png", dpi=300)
    #     plt.close() 



    def test_value_positions_rect_to_square_vectorized(self):
        """
        For random indices in the Hi-C rectangle, convert the rectangle coordinates
        into square coordinates, log the conversion details, and overlay the points with
        labels on interactive plots for visual confirmation.
        """

        hic_file = "/nfs/turbo/umms-minjilab/mingjiay/GSE199059_wt_selected_30_new.hic"
        resolution = int(50e3)
        window_size_bin = int(np.ceil(6e6 / resolution))
        
        # Read in the full square matrix and the extracted rectangle.
        square = read_hic_file(hic_file, "chr1", resolution, "all", "oe", "KR", verbose=False)
        rect = read_hic_rectangle(hic_file, "chr1", resolution, window_size_bin,
                                "oe", "KR", rotate_mode="mirror", cval=0,
                                handle_zero_sum=None, verbose=False)
        
        logging.info("Square shape: %s", square.shape)
        logging.info("Rectangle shape: %s", rect.shape)
        
        # Apply Gaussian blurring to both matrices.
        square_blurred = gaussian_filter(square, sigma=4)
        rect_blurred = gaussian_filter(rect, sigma=4)
        
        M, N_rect = rect_blurred.shape  # dimensions of the rectangle
        N = square_blurred.shape[0]     # dimensions of the square (N x N)

        # Lists to collect points and labels.
        rect_points = []   # (col, row) positions in the rectangle
        square_points = [] # (col, row) positions in the square
        rect_values = []

        # Perform 10 random comparisons.
        for epoch in range(100):
            # Pick a random index in the rectangle.
            r_rect = np.random.randint(0, M)
            c_rect = np.random.randint(0, N_rect)

            # Use the flipped version to match the plotting orientation.
            value_rect = rect_blurred[r_rect, c_rect]            

            # Save the points
            rect_points.append((r_rect, c_rect))
            rect_values.append(value_rect)

        rect_points = np.array(rect_points)
                    
        # Convert rectangle index to the corresponding square coordinates.
        square_points = rect_to_square(N, window_size_bin, rect_points)

        square_values = []
        for epoch, (i, j) in enumerate(square_points):
            i_idx = int(round(i))
            j_idx = int(round(j))

            value_square = square_blurred[i_idx, j_idx]

            r_rect, c_rect = rect_points[epoch, :]
            value_rect = rect_values[epoch]

            # logging.info(f"Epoch {epoch}: rect[{r_rect}, {c_rect}] (value {value_rect}) -> square[{i_idx}, {j_idx}] (value {value_square})")

            square_values.append(value_square)
    
        cc = stats.spearmanr(rect_values, square_values)
        logging.info(f"Correlation (higher the better): {cc.statistic:.2f}")

        


    # def test_value_positions_square_to_rect(self):
    #     """
    #     For 50 iterations, randomly pick one UT coordinate from the Hi-C square such that 
    #     j - i <= window_size_bin. For each such coordinate, convert the UT square coordinate 
    #     to rotated rectangle coordinates using square_to_rect, ensuring the resulting indices 
    #     fall within the bounds of the rotated rectangle image. Then compare the blurred pixel 
    #     values at the corresponding positions. Log the conversion details, compute Spearman 
    #     correlation, and produce plots (with annotations) for both the square and the rectangle.
    #     """
    #     hic_file = "/nfs/turbo/umms-minjilab/mingjiay/GSE199059_wt_selected_30_new.hic"
    #     resolution = int(50e3)
    #     window_size_bin = int(np.ceil(6e6 / resolution))
        
    #     # Read the Hi-C square and rectangle.
    #     square = read_hic_file(hic_file, "chr1", resolution, "all", "oe", "KR", verbose=False)
    #     rect = read_hic_rectangle(hic_file, "chr1", resolution, window_size_bin,
    #                                 "oe", "KR", rotate_mode="mirror", cval=0,
    #                                 handle_zero_sum=None, verbose=False)
        
    #     logging.info("Square shape: %s", square.shape)
    #     logging.info("Rectangle shape: %s", rect.shape)
    #     logging.info(f"Window size bin: {window_size_bin}")
        
    #     # Apply Gaussian blurring.
    #     square_blurred = gaussian_filter(square, sigma=4)
    #     rect_blurred = gaussian_filter(rect, sigma=4)
        
    #     N = square_blurred.shape[0]          # Dimension of the square
    #     M, N_rect = rect_blurred.shape       # Dimensions of the rectangle

    #     # Lists for storing points, labels, and pixel values.
    #     square_points = []  # (col, row) in the square (for plotting)
    #     rect_points = []    # (col, row) in the rectangle (for plotting)
    #     ut_labels = []
    #     square_values = []
    #     rect_values = []

    #     for epoch in range(50):
    #         # Sample a random UT coordinate from the square.
    #         ut_coords = generate_ut_coords(N, window_size_bin)

    #         ut_coords = np.array(ut_coords)

    #         i, j = random.choice(list(ut_coords.T))

    #         # Convert to rectangle coordinates.
    #         rect_coord = square_to_rect(N, window_size_bin, (i, j))
    #         # Convert to integer indices.
    #         r_rect_idx = int(round(rect_coord[0]))
    #         c_rect_idx = int(round(rect_coord[1]))
            
    #         # Check bounds for the rectangle.
    #         if not (0 <= r_rect_idx < M):
    #             # If out-of-bounds, skip this sample.
    #             logging.info(f"Square: {i}, {j}")
    #             logging.info(f"Rectangle: {r_rect_idx} exceeded {M}")
    #             logging.info("")
    #             continue
    #         if not (0 <= c_rect_idx < N_rect):
    #             # If out-of-bounds, skip this sample.
    #             logging.info(f"Square: {i}, {j}")
    #             logging.info(f"Rectangle: {c_rect_idx} exceeded {N_rect}")
    #             logging.info("")
    #             continue
            

    #         # Retrieve the blurred pixel values.
    #         val_square = square_blurred[i, j]
    #         val_rect = rect_blurred[r_rect_idx, c_rect_idx]
            
    #         logging.info(f"Epoch {epoch}: square {(i, j)} (value {val_square:.3f}) -> "
    #                     f"rect [{r_rect_idx}, {c_rect_idx}] (value {val_rect:.3f})")
            
    #         # Save the points.
    #         # For the square, use (col, row) = (j, i)
    #         square_points.append((j, i))
    #         # For the rectangle, use (col, row) = (c_rect_idx, r_rect_idx)
    #         rect_points.append((c_rect_idx, r_rect_idx))
    #         ut_labels.append(f"p{epoch}:({i},{j})")
            
    #         square_values.append(val_square)
    #         rect_values.append(val_rect)
        
    #     # Compute Spearman correlation.
    #     cc = stats.spearmanr(square_values, rect_values)
    #     logging.info(f"Correlation (higher the better): {cc.statistic:.2f}")

    #     square_points = np.array(square_points)
    #     rect_points = np.array(rect_points)

    #     # --- Plot for the square ---
    #     plt.figure(figsize=(40, 40), layout="constrained")
    #     plt.imshow(square_blurred, cmap='viridis', origin='upper')
    #     if square_points.size > 0:
    #         plt.scatter(square_points[:, 0], square_points[:, 1],
    #                     c='red', marker='x', s=10, label='UT Points')
    #         for idx, (x, y) in enumerate(square_points):
    #             plt.annotate(ut_labels[idx], xy=(x, y), color='white', fontsize=5,
    #                         xytext=(x+5, y+5), textcoords='data')
    #     plt.title(f'Hi-C Square UT Points (corr={cc.statistic:.2f})')
    #     plt.savefig("output/test_value_positions_square_fromUT.png", dpi=300)
    #     plt.close()

    #     # --- Plot for the rotated rectangle ---
    #     plt.figure(figsize=(40, 5), layout="constrained")
    #     # For plotting the rectangle, use np.flipud as before.
    #     plt.imshow(np.flipud(rect_blurred), cmap='viridis', origin='lower')
    #     if rect_points.size > 0:
    #         adjusted_rect_points = rect_points.copy()
    #         adjusted_rect_points[:, 1] = rect_blurred.shape[0] - rect_points[:, 1] - 1
    #         plt.scatter(adjusted_rect_points[:, 0], adjusted_rect_points[:, 1],
    #                     c='red', marker='x', s=10, label='Converted UT Points')
    #         for idx, (x, y) in enumerate(adjusted_rect_points):
    #             plt.annotate(ut_labels[idx], xy=(x, y), color='white', fontsize=5,
    #                         xytext=(x+5, y+5), textcoords='data')
    #     plt.title(f'Hi-C Rotated Rectangle UT Points (corr={cc.statistic:.2f})')
    #     plt.savefig("output/test_value_positions_rect_fromUT.png", dpi=300)
    #     plt.close()

    def test_value_positions_square_to_rect_vectorized(self):
        """
        For random indices in the Hi-C rectangle, convert the rectangle coordinates
        into square coordinates, log the conversion details, and overlay the points with
        labels on interactive plots for visual confirmation.
        """

        hic_file = "/nfs/turbo/umms-minjilab/mingjiay/GSE199059_wt_selected_30_new.hic"
        resolution = int(50e3)
        window_size_bin = int(np.ceil(6e6 / resolution))
        
        # Read in the full square matrix and the extracted rectangle.
        square = read_hic_file(hic_file, "chr1", resolution, "all", "oe", "KR", verbose=False)
        rect = read_hic_rectangle(hic_file, "chr1", resolution, window_size_bin,
                                "oe", "KR", rotate_mode="mirror", cval=0,
                                handle_zero_sum=None, verbose=False)
        
        logging.info("Square shape: %s", square.shape)
        logging.info("Rectangle shape: %s", rect.shape)
        
        # Apply Gaussian blurring to both matrices.
        square_blurred = gaussian_filter(square, sigma=4)
        rect_blurred = gaussian_filter(rect, sigma=4)
        
        M, N_rect = rect_blurred.shape  # dimensions of the rectangle
        N = square_blurred.shape[0]     # dimensions of the square (N x N)

        # Lists to collect points and labels.
        square_points = []   # (col, row) positions in the rectangle
        square_values = []

        # Perform 10 random comparisons.
        for epoch in range(100):
            # Sample a random UT coordinate from the square.
            ut_coords = generate_ut_coords(N, window_size_bin)

            ut_coords = np.array(ut_coords)

            i, j = random.choice(list(ut_coords.T))       

            value_square = square_blurred[i, j]

            # Save the points
            square_points.append((i, j))
            square_values.append(value_square)

        square_points = np.array(square_points)
                    
        # Convert rectangle index to the corresponding square coordinates.
        rect_points = square_to_rect(N, window_size_bin, square_points)

        rect_values = []
        for epoch, (i, j) in enumerate(rect_points):
            i_idx = int(round(i))
            j_idx = int(round(j))

            value_rect = rect_blurred[i_idx, j_idx]

            r_rect, c_rect = square_points[epoch, :]
            value_rect = square_values[epoch]

            # logging.info(f"Epoch {epoch}: rect[{r_rect}, {c_rect}] (value {value_rect}) -> square[{i_idx}, {j_idx}] (value {value_square})")

            rect_values.append(value_rect)
    
        cc = stats.spearmanr(rect_values, square_values)
        logging.info(f"Correlation (higher the better): {cc.statistic:.2f}")


        

if __name__ == '__main__':
    unittest.main(buffer=False)