import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import os 
from utils.plotting import convert_imagej_coord_to_numpy, genomic_labels, set_genomic_ticks
from utils.scale_space import extract_line_scale_space, create_maxima_set, round_line_scale_space, extract_angle_scale_space
from scipy.signal import argrelmax
from utils.processing import read_hic_rectangle
import os
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm




def rank_true_ridges(df_agg, df_features, f_true_cns, ranking, chromosome, resolution, save_path, parameter_str,
                      contour_label="Contour Number", x_label="X_(px)_orig"):
    # Assume required modules (os, numpy as np, pandas as pd) and the function genomic_labels are already imported/defined.
    
    # Work on a copy to preserve the original dataframe
    df_agg = df_agg.copy()
    df_features = df_features.copy()

    df_agg["s_imagej"] = df_agg["s_imagej"].round(2)
    df_features["s_imagej"] = df_features["s_imagej"].round(2)

    
    # Read and filter the true ridges file (assumes no header)
    tp = pd.read_csv(f_true_cns, sep="\t", usecols=[0, 1, 2], names=["chrom", contour_label, "s_imagej"])
    tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
    
    # Order df_agg by the ranking column (descending) and round the s_imagej values to 2 decimals
    df_agg = df_agg.sort_values(ranking, ascending=False, ignore_index=True)

    # Add a 1-indexed rank column
    df_agg['rank'] = df_agg.index + 1
    total_ridges = len(df_agg)
    
    # Compute the median x position for each ridge from df_features, which may have multiple rows per ridge.
    median_x_df = df_features.groupby([contour_label, "s_imagej"])[x_label].median().reset_index().rename(columns={x_label: "median_x"})
    
    # Merge the median x positions into the ranked aggregated dataframe (using df_features info)
    df_ranked = pd.merge(df_agg, median_x_df, on=[contour_label, "s_imagej"], how="left")
    
    # Merge true ridges (tp) with the df_ranked dataframe to obtain ranking and median x values
    tp_merged = pd.merge(tp, df_ranked[[contour_label, "s_imagej", "rank", "median_x"]], on=[contour_label, "s_imagej"], how="left")
    
    tp_merged['genomic start'] = tp_merged['median_x'].apply(
        lambda row: pd.Series(genomic_labels(row * resolution / np.sqrt(2), N=2))
    )

    tp_merged['genomic end'] = tp_merged['median_x'].apply(
        lambda row: pd.Series(genomic_labels(row * resolution / np.sqrt(2), N=2))
    )
    
    # Create a "position" column: if "rank" is not NaN, convert it to an integer, otherwise set as "NA"
    tp_merged["position"] = tp_merged["rank"].apply(lambda x: int(x) if not pd.isna(x) else "NA")

    # Create a "TOTAL" column, which is constant for all rows
    tp_merged["TOTAL"] = total_ridges

    
    # Assemble the final dataframe with the desired columns and order
    final_df = tp_merged[["chrom", contour_label, "s_imagej", "genomic start", "genomic end", "position", "TOTAL"]]
    
    # Save the output to a text file, with two header lines: one for a "root" (using chromosome as a proxy) and one for the parameter string.
    save_name = os.path.join(save_path, f"{chromosome}_true_ranks.txt")
    with open(save_name, "w") as f:
        f.write(f"# {parameter_str}\n")
        final_df.to_csv(f, index=False, sep="\t")






def plot_distribution_diagnostic(
    df_agg,
    df_features,
    im,
    ranking,
    resolution,
    save_path,
    contour_label="Contour Number",
    x_label="X_(px)_unmap",
    y_label="Y_(px)_unmap",
    smooth_sigma=0
):
    """
    Diagnostic plot function (single plot, no chunking).
    Creates a figure with three stacked panels:
      1) The entire image
      2) The same image plus a heatmap overlay of ridge strengths
      3) The ridge-strength distribution (mu2) over the x-dimension.

    Uses log scaling & percentile-based vmax in the heatmap for clarity.
    Also applies an optional Gaussian smoothing to the 2D grid.

    Parameters
    ----------
    df_agg : pd.DataFrame
        Aggregated results, containing a 'ranking' column, etc.
    df_features : pd.DataFrame
        Must contain columns [contour_label, 's_imagej', x_label, y_label]
    im : np.ndarray
        2D image array (height x width)
    ranking : str
        Column name in df_agg to rank by (e.g. 'peak_strength')
    resolution : float
        Resolution in "bp per pixel" (often scaled by sqrt(2))
    save_path : str
        Directory for saving the resulting PNG
    contour_label : str, optional
        Column name labeling each contour
    x_label : str, optional
        X coordinate column in df_features
    y_label : str, optional
        Y coordinate column in df_features
    smooth_sigma : float, optional
        Sigma for the Gaussian blur on the 2D grid (default=1.0)

    Returns
    -------
    None
        Saves a single figure with 3 subplots to save_path.
    """
    height, width = im.shape[:2]

    # 1) Merge input data
    df_agg_sorted = df_agg.sort_values(ranking, ascending=False, ignore_index=True)
    df_agg_sorted = df_agg_sorted.merge(
        df_features[[contour_label, "s_imagej", x_label, y_label]],
        how="inner",
        on=[contour_label, "s_imagej"]
    )

    # 2) Accumulate the per-x ridge strengths (for the bottom panel),
    #    and also gather (x,y,ranking) points for a 2D heatmap
    ridge_strength_prob = np.zeros(width, dtype=float)
    x_points = []
    y_points = []
    surface_values = []

    grouped = df_agg_sorted.groupby([contour_label, "s_imagej"])
    for _, df_ridge in grouped:
        # Convert min/max x from pixel to “genomic” dimension
        df_min = df_ridge[x_label].min() * resolution / np.sqrt(2)
        df_max = df_ridge[x_label].max() * resolution / np.sqrt(2)
        start  = max(df_min, 0)
        end    = min(df_max, width * resolution / np.sqrt(2))

        start_bin = int(round(start / (resolution / np.sqrt(2))))
        end_bin   = int(round(end   / (resolution / np.sqrt(2))))

        # Convert ImageJ coords to NumPy coords
        ridge_coords = convert_imagej_coord_to_numpy(
            df_ridge[[x_label, y_label]].values,
            height,
            flip_y=False,
            start_bin=0
        )

        # Accumulate scattered data
        x_points.extend(ridge_coords[:, 0])
        y_points.extend(ridge_coords[:, 1])
        surface_values.extend(df_ridge[ranking].values)

        # Accumulate the 1D distribution for the bottom panel
        val_ranking = df_ridge.iloc[0][ranking]
        if 0 <= start_bin < end_bin <= width:
            ridge_strength_prob[start_bin:end_bin] += val_ranking

    total_strength = ridge_strength_prob.sum()
    mu2 = ridge_strength_prob / total_strength if total_strength > 0 else np.zeros_like(ridge_strength_prob)

    # 3) Build the 2D heatmap grid
    # Choose number of bins in x,y
    num_x, num_y = width, height
    grid_x = np.linspace(0, width, num_x)
    grid_y = np.linspace(0, height, num_y)
    grid_Z = np.zeros((num_y, num_x), dtype=float)  # shape = (rows, cols) = (y, x)

    # Accumulate ridge strengths into bins
    for x_val, y_val, val in zip(x_points, y_points, surface_values):
        # Skip out-of-bounds
        if not (0 <= x_val < width and 0 <= y_val < height):
            continue
        ix = np.searchsorted(grid_x, x_val) - 1
        iy = np.searchsorted(grid_y, y_val) - 1
        if 0 <= ix < num_x and 0 <= iy < num_y:
            grid_Z[iy, ix] += val

    # Optional: smooth the grid to avoid single-bin spikes
    if smooth_sigma > 0:
        grid_Z = gaussian_filter(grid_Z, sigma=smooth_sigma)

    # 4) Decide on a color range (e.g., 99th percentile among nonzero bins)
    nonzero_vals = grid_Z[grid_Z > 0]
    if nonzero_vals.size > 0:
        z_vmax = np.percentile(nonzero_vals, 99)  # or 95, or 90, etc.
        z_vmin = max(1e-6, nonzero_vals.min())
    else:
        z_vmax = 1.0
        z_vmin = 1e-6

    # 5) Make the figure
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(width / 55.0, 7), layout="constrained")
    fig.suptitle(f"Ranking by {ranking}")

    # -- Panel 1: raw image
    ax[0].imshow(
        im,
        cmap="Reds",
        interpolation="none",
        aspect="equal",
        extent=[0, width, height, 0]  # left, right, top, bottom
    )
    ax[0].set_ylabel("Image row (px)")

    # -- Panel 2: image + heatmap using imshow
    ax[1].imshow(
        im,
        cmap="Reds",
        interpolation="none",
        aspect="equal",
        extent=[0, width, height, 0]
    )

    # Overplot the heatmap with partial transparency
    # We invert Y because grid_Z’s row 0 is at the top, row (num_y-1) at the bottom
    cax = ax[1].imshow(
        grid_Z,           
        extent=[0, width, height, 0],  
        cmap="viridis",
        alpha=0.5,
        norm=LogNorm(vmin=z_vmin, vmax=z_vmax),  # Log scale
        interpolation="none",
        aspect="auto",
    )
    ax[1].set_ylabel("Image row (px)")
    cb = fig.colorbar(cax, ax=ax[1], label=f"{ranking} (log scale)")

    # -- Panel 3: 1D distribution
    ax[2].plot(np.arange(width), mu2, lw=1)
    ax[2].set_title("Cumulative sum of ridge strength (normalized)")
    ax[2].set_ylabel("Normalized ridge strength")
    ax[2].set_xlabel("Genomic position (bp, approx)")

    # 6) Set genomic ticks
    num_ticks = 10
    x_tick_positions = np.linspace(0, width, num_ticks).astype(int)
    x_tick_labels = [
        f"{genomic_labels(x_px * resolution / np.sqrt(2))}"
        for x_px in x_tick_positions
    ]
    for axis in ax:
        axis.set_xticks(x_tick_positions)
        axis.set_xticklabels(x_tick_labels, fontsize=8, rotation=45)

    # 7) Save
    out_name = os.path.join(save_path, "diagnostic_distribution_single.png")

    fig = plt.gcf()  # get the current figure
    fig_width = fig.get_size_inches()[0]
    max_allowed_dpi = 65536 / fig_width
    dpi = min(300, max_allowed_dpi / 2) 

    plt.savefig(out_name, dpi=dpi)
    plt.close()



from utils.plotting import plot_p_value_basics, plot_p_value_observed_null
from miajet.compute_p_value import compute_test_statistic, compute_test_statistic_quantities


def plot_top_k_diagnostic(df_agg, df_features, K, im, im_p_value, corr_im_p_value, I, D, A, W1, W2, R, C, ranking, resolution, chromosome, scale_range, window_size, save_path,
                          num_bins, bin_size, points_min, points_max, plot_unique, f_true_bed, tolerance, angle_range, f_true_cns,
                          contour_label="Contour Number",
                          x_label="X_(px)_unmap", 
                          y_label="Y_(px)_unmap"):
    

    def plot_summary_image():
        # NOTE: changed 03/17/25 from using the "s_imagej" to "scale_assigned"
        scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["scale_assigned"]))[0]
        if len(scale_assigned_set) == 0: 
            # then don't round to 3 
            # this issue crops when saving and reading the files (numeric columns are saved to 3 d.p)
            selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["scale_assigned"]))[0][0]
        else:
            selected_scale_idx = scale_assigned_set[0]

        selected_scale = scale_range[selected_scale_idx]

        A_bool = np.logical_and(A[selected_scale_idx, :, start_bin:end_bin] >= angle_range[0], A[selected_scale_idx, :, start_bin:end_bin] <= angle_range[1])
        
        forplot_im = [
            im[:, start_bin:end_bin],
            I[selected_scale_idx, :, start_bin:end_bin], 
            D[selected_scale_idx, :, start_bin:end_bin], 
            A[selected_scale_idx, :, start_bin:end_bin], 
            A_bool,
            W1[selected_scale_idx, :, start_bin:end_bin], 
            W2[selected_scale_idx, :, start_bin:end_bin], 
            R[selected_scale_idx, :, start_bin:end_bin], 
            C[selected_scale_idx, :, start_bin:end_bin]
        ]
        # im, I, D, A, W1, W2, R, C

        titles = [
            f"Original (mean contact frequency: {np.mean(im_curves[0]):.2g})",
            f"Blurred s={selected_scale:.2f}",
            f'Ridge strength {df_ridge["ridge_strength_mean"].iloc[0]:.2f}',
            f'Angle mean: {df_ridge["angle_mean"].iloc[0]:.2f} {df_ridge["angle_unwrapped_mean"].iloc[0]:.2f} {df_ridge["angle_imagej_mean"].iloc[0]:.2f}\nderiv: {df_ridge["angle_deriv_max"].iloc[0]:.2f}',
            f'Angle mean: {df_ridge["angle_mean"].iloc[0]:.2f} {df_ridge["angle_unwrapped_mean"].iloc[0]:.2f} {df_ridge["angle_imagej_mean"].iloc[0]:.2f}\nderiv: {df_ridge["angle_deriv_max"].iloc[0]:.2f}',
            f'Eig1 {df_ridge["eig1_mean"].iloc[0]:.2f}',
            f'Eig2 {df_ridge["eig2_mean"].iloc[0]:.2f}',
            f"Ridge condition",
            f'Corner condition {df_ridge["corner_cond_fraction"].iloc[0]:.2f}',
        ]

        vmaxes = [
            np.percentile(im, q=98), # im
            np.percentile(I[selected_scale_idx], q=98), # I
            np.percentile(D[selected_scale_idx], q=98), # D
            None, # A
            None, # A
            np.percentile(W1[selected_scale_idx], q=98), # W1
            None, # W2
            None, # R
            None #C
        ]

        cmaps = [
            plt.cm.Reds, # im
            plt.cm.Reds, # I
            plt.cm.Reds, # D
            plt.cm.twilight_shifted, # A
            plt.cm.binary, # A bool
            plt.cm.bwr, # W1
            plt.cm.bwr, # W2
            plt.cm.binary, # R
            plt.cm.binary # C
        ]

        vcenters = [
            None,
            None,
            None,
            90,
            None,
            0,
            0,
            None,
            None
        ]


        lines = np.array([ridge_coords])

        forplot_curves = [
            im_curves[0],
            I_curves[selected_scale_idx],
            D_curves[selected_scale_idx],
            A_curves[selected_scale_idx],
            A_bool_curves[selected_scale_idx],
            W1_curves[selected_scale_idx],
            W2_curves[selected_scale_idx],
            R_curves[selected_scale_idx],
            C_curves[selected_scale_idx],
        ]


        fig, ax = plt.subplots(2, 9, layout="constrained", figsize=(30, 10))

        fig.suptitle(ridge_title)

        for i in range(9): # Plot 9 images
            # Plot images
            if vcenters[i] is not None:
                im0 = ax[0, i].imshow(np.flipud(forplot_im[i]), cmap=cmaps[i], origin="lower", interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenters[i], vmax=vmaxes[i]))
            else:
                im0 = ax[0, i].imshow(np.flipud(forplot_im[i]), cmap=cmaps[i], origin="lower", interpolation="none", vmax=vmaxes[i])
            ax[0, i].set_title(titles[i])
            set_genomic_ticks(ax[0, i], num_ticks=[5, 8], resolution=resolution, mat_shape=forplot_im[i].shape, genomic_shift=[0, start], dp=1)

            cbar = fig.colorbar(im0, ax=ax[0, i])
            
            for j, line in enumerate(lines):
                ax[0, i].plot(line[:, 0], line[:, 1], linewidth=1.2, label=None, alpha=0.7)

            # Plot curves 
            ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), forplot_curves[i], color="blue", marker="o", markersize=4)
            if i == 3: # the 3rd plot is the angle plot
                # We should try to plot the other angle values in `df_ridge`
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle"], label="angle", alpha=0.5)
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle_unwrapped"], label="angle unwrapped", alpha=0.5)
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle_imagej"], label="imageJ angle", alpha=0.5)
                ax[1, i].legend(loc="best")

                # ASSERT
                # assert np.allclose(np.round(df_ridge["angle"].values, 2), np.round(forplot_curves[i], 2))

            num_ticks = 10
            def_xticks = np.arange(0, forplot_curves[i].shape[0], np.ceil(forplot_curves[i].shape[0] / num_ticks).astype(int))
            ticks_bp_x = [f"{genomic_labels(window_size / 2 - x[1] * resolution / np.sqrt(2))}" for x in df_ridge[[x_label, y_label]].values[def_xticks]]

            ax[1, i].set_xticks(def_xticks)
            ax[1, i].set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)

        # Start of main plotting function
        if true_given: #"merge-true_"
            save_name = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_summary.png") 
        else:
            save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_summary.png") 

        plt.savefig(save_name, dpi=400)

        plt.close()



    def plot_scale_space_heatmap():

        scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["scale_assigned"]))[0]
        if len(scale_assigned_set) == 0: 
            # then don't round to 3 
            # this issue crops when saving and reading the files (numeric columns are saved to 3 d.p)
            selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["scale_assigned"]))[0][0]
        else:
            selected_scale_idx = scale_assigned_set[0]

        selected_scale = scale_range[selected_scale_idx]

        fig, axes = plt.subplots(4, 5, figsize=(24, 10), layout="constrained")

        # main plot
        axes[0,0].imshow(D_curves, aspect="auto", cmap="viridis", vmax=np.percentile(D_curves, 95))

        scale_idx_argmax, pos_argmax = argrelmax(D_curves, axis=0)
        axes[0,0].scatter(pos_argmax, scale_idx_argmax, color="cyan", marker="x", s=15)
        axes[0,0].scatter(np.arange(D_curves.shape[1]), 
                        np.argmax(D_curves, axis=0), 
                        color="red", s=15, marker="o")

        axes[0,0].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,0].invert_yaxis()
        axes[0,0].set_ylabel("Scale")
        axes[0,0].set_xlabel("Ridge position")
        axes[0,0].set_title("Ridge Strength Signature (global maxima in red)")

        # angle signature
        t0 = axes[0,1].imshow(A_curves, aspect="auto", cmap="twilight_shifted", norm=colors.TwoSlopeNorm(vcenter=90))
        axes[0,1].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,1].invert_yaxis()
        axes[0,1].set_ylabel("Scale")
        axes[0,1].set_xlabel("Ridge position")
        axes[0,1].set_title("Angle Signature")
        cbar = plt.colorbar(t0, ax=axes[0,1])

        # angle bool condition signature
        t0 = axes[0,2].imshow(A_bool_curves, aspect="auto", cmap="binary")
        axes[0,2].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,2].invert_yaxis()
        axes[0,2].set_ylabel("Scale")
        axes[0,2].set_xlabel("Ridge position")
        axes[0,2].set_title("Angle Bool Signature")
        cbar = plt.colorbar(t0, ax=axes[0,2])

        # ridge condition signature
        t0 = axes[0,3].imshow(R_curves, aspect="auto", cmap="binary", vmax=np.percentile(R_curves, 95))
        axes[0,3].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,3].invert_yaxis()
        axes[0,3].set_ylabel("Scale")
        axes[0,3].set_xlabel("Ridge position")
        axes[0,3].set_title("Ridge Condition Signature")
        cbar = plt.colorbar(t0, ax=axes[0,3])

        # corner condition signature
        t0 = axes[0,4].imshow(C_curves, aspect="auto", cmap="binary")
        axes[0,4].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,4].invert_yaxis()
        axes[0,4].set_ylabel("Scale")
        axes[0,4].set_xlabel("Ridge position")
        axes[0,4].set_title("Corner Condition Signature")
        cbar = plt.colorbar(t0, ax=axes[0,4])


        for i in range(5):
            # Add rectangle around the selected row
            rect = patches.Rectangle(
                (0 - 0.5, selected_scale_idx - 0.5),  # (x, y) of bottom-left corner
                width=D_curves.shape[1],                      # full width
                height=1,                            # height of 1 row
                edgecolor='red',                     # box color
                facecolor='none',                    # no fill
                linewidth=1
            )
            axes[0, i].add_patch(rect)

    
        # column-wise expected scale and fitted polynomial + statistics
        # col_vars = np.var(D_curves, axis=0)
        # col_vars_scale = np.var(D_curves[:selected_scale_idx+1, :], axis=0)
        # col_mean_diff = df_ridge["col_mean_diff"].values
        expected_scales = df_ridge["expected_scale"].values
        coeffs = df_ridge["coeffs"].values[0]
        coeffs_str = ",".join(f"{c:.3f}" for c in coeffs)
        y_fit_ev = np.polyval(coeffs, np.arange(len(expected_scales)))
        axes[1,0].scatter(range(len(expected_scales)), expected_scales, label="all scales", s=10)
        axes[1,0].plot(y_fit_ev, label="Cubic")
        axes[1,0].set_title(f"Expected scales \nparam: {coeffs_str} | norm rmse: {df_ridge['rmse'].iloc[0]:.4f} | rmse: {df_ridge['rmse'].iloc[0] * len(expected_scales):.3f}")
        axes[1,0].set_xlabel("Ridge position")

        # column-wise mean of ridge strength
        col_mean = np.mean(D_curves, axis=0)
        # col_mean_scale = np.mean(D_curves[:selected_scale_idx+1, :], axis=0)
        axes[2,0].scatter(range(len(col_mean)), col_mean, label="all scales", s=10)
        # axes[2,0].scatter(range(len(col_mean_scale)), col_mean_scale, label="≤ selected scale")
        axes[2,0].set_title("Column-wise mean")
        axes[2,0].set_xlabel("Ridge position")
        # axes[2,0].legend() 

        # column-wise variance of angle
        col_vars = np.var(A_curves, axis=0)
        col_vars_scale = np.var(A_curves[:selected_scale_idx+1, :], axis=0)
        axes[1,1].scatter(range(len(col_vars)), col_vars, label="all scales", s=10)
        axes[1,1].scatter(range(len(col_vars_scale)), col_vars_scale, label="≤ selected scale", s=10)
        axes[1,1].set_title("Column-wise variance")
        axes[1,1].set_xlabel("Ridge position")
        axes[1,1].legend() 
        # column-wise mean of ridge strength
        col_mean = np.mean(A_curves, axis=0)
        col_mean_scale = np.mean(A_curves[:selected_scale_idx+1, :], axis=0)
        axes[2,1].scatter(range(len(col_mean)), col_mean, label="all scales", s=10)
        axes[2,1].scatter(range(len(col_mean_scale)), col_mean_scale, label="≤ selected scale", s=10)
        axes[2,1].set_title("Column-wise mean")
        axes[2,1].set_xlabel("Ridge position")
        axes[2,1].legend() 


        # Entropy Histogram 1
        edges = df_ridge["edges"].iloc[0]
        axes[3,0].stairs(values=df_ridge["pmf"].iloc[0], edges=edges, 
                         fill=True, edgecolor="black", facecolor="blue")
        axes[3,0].set_xticks(edges)
        axes[3, 0].set_xticklabels(edges, rotation=45)
        axes[3,0].tick_params(labelbottom=True)
        axes[3,0].set_xlabel("Column-wise mean values")
        axes[3,0].set_ylabel("Probability")
        ent_str = f'Normalized Entropy: {df_ridge["entropy"].iloc[0]:.3f}'
        if num_bins is not None:
            axes[3,0].set_title(f"Histogram (range: [{points_min}, {points_max}] num bins: {num_bins})\n{ent_str}")
        else:
            axes[3,0].set_title(f"Histogram (range: [{points_min}, {points_max}] bin size: {bin_size})\n{ent_str}")


        # Width Histogram
        axes[3,1].hist(df_ridge["width"], bins=10, color="blue")
        axes[3,1].set_xlabel("Ridge width (px)")
        axes[3,1].set_ylabel("Frequency")
        axes[3,1].set_title(f"Width Histogram (mean: {df_ridge['width_mean'].iloc[0]:.2f}, median: {df_ridge['width'].median():.2f}, length={len(df_ridge)})")


        for ax in axes.flatten():
            if not ax.has_data():
                ax.axis("off")

        fig.suptitle(ridge_title)

        if true_given: #"merge-true_"
            save_name = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_scale_signature.png")
        else:
            save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_scale_signature.png")

        plt.savefig(save_name, dpi=400)
        plt.close()

    def plot_p_value_diagnostic():

        for factor_lr in [1]: 

            if true_given: #"merge-true_"
                save_name_mean = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_mean_lr{factor_lr}-p_value.png")
                save_name_med = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_med_lr{factor_lr}-p_value.png")
            else:
                save_name_mean = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_mean_lr{factor_lr}-p_value.png")
                save_name_med = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_med_lr{factor_lr}-p_value.png")
            
            left_means = []
            right_means = []
            center_means = []
            left_vals = []
            right_vals = []
            center_vals = []
            mean_CR_subtract = []
            mean_CR_ratio = []
            mean_C2R_ratio = []
            left_meds = []
            right_meds = []
            center_meds = []
            med_CR_subtract = []
            med_CR_ratio = []
            med_C2R_ratio = []

            for _, im_each in enumerate([im_p_value, corr_im_p_value]):

                (
                    left_means_each,
                    right_means_each,
                    center_means_each,
                    left_box_coords,
                    right_box_coords,
                    center_box_coords,
                    left_num_points,
                    right_num_points,
                    center_num_points,
                    left_vals_each,
                    right_vals_each,
                    center_vals_each, 
                    left_med_each,
                    right_med_each,
                    center_med_each,
                ) = compute_test_statistic_quantities(im=im_each, 
                                                    ridge_points=ridge_coords_curve,
                                                    ridge_angles=-df_ridge["angle_imagej"].values - 90, # -1 * ANGLE - 90 (negate and make normal)
                                                    width_in=df_ridge["width"].values,
                                                    height=1,
                                                    im_shape=im_each.shape, 
                                                    factor_lr=factor_lr # Left, right box width is width * factor_lr
                                                    )
                

                mean_CR_subtract_each, mean_CR_ratio_each, mean_C2R_ratio_each = compute_test_statistic(left_means_each, right_means_each, center_means_each)
                med_CR_subtract_each, med_CR_ratio_each, med_C2R_ratio_each = compute_test_statistic(left_med_each, right_med_each, center_med_each)

                left_means.append(left_means_each)
                right_means.append(right_means_each)
                center_means.append(center_means_each)
                left_vals.append(left_vals_each)
                right_vals.append(right_vals_each)
                center_vals.append(center_vals_each)
                mean_CR_subtract.append(mean_CR_subtract_each)
                mean_CR_ratio.append(mean_CR_ratio_each)
                mean_C2R_ratio.append(mean_C2R_ratio_each)
                left_meds.append(left_med_each)
                right_meds.append(right_med_each)
                center_meds.append(center_med_each)
                med_CR_subtract.append(med_CR_subtract_each)
                med_CR_ratio.append(med_CR_ratio_each)
                med_C2R_ratio.append(med_C2R_ratio_each)

            # Plot one for mean
            plot_p_value_observed_null(im_p_value=im_p_value[:, start_bin:end_bin],
                                       corr_im_p_value=corr_im_p_value[:, start_bin:end_bin],
                                       ridge_points=ridge_points, 
                                       ridge_widths=df_ridge["width"].values, 
                                       center_box_coords=center_box_coords,
                                       right_box_coords=right_box_coords,
                                       left_box_coords=left_box_coords,
                                       center_num_points=center_num_points,
                                       right_num_points=right_num_points,
                                       left_num_points=left_num_points,
                                       center_vals=center_vals,
                                       right_vals=right_vals,
                                       left_vals=left_vals,
                                       center_means=center_means,
                                       right_means=right_means,
                                       left_means=left_means,
                                       mean_CR_subtract=mean_CR_subtract,
                                       mean_CR_ratio=mean_CR_ratio,
                                       mean_C2R_ratio=mean_C2R_ratio,
                                       fig_suptitle=ridge_title,
                                       save_path=save_name_mean, 
                                       start_bin=start_bin
                                       )
            
            # Plot one for median
            plot_p_value_observed_null(im_p_value=im_p_value[:, start_bin:end_bin],
                                        corr_im_p_value=corr_im_p_value[:, start_bin:end_bin],
                                        ridge_points=ridge_points, 
                                        ridge_widths=df_ridge["width"].values, 
                                        center_box_coords=center_box_coords,
                                        right_box_coords=right_box_coords,
                                        left_box_coords=left_box_coords,
                                        center_num_points=center_num_points,
                                        right_num_points=right_num_points,
                                        left_num_points=left_num_points,
                                        center_vals=center_vals,
                                        right_vals=right_vals,
                                        left_vals=left_vals,
                                        center_means=center_meds,
                                        right_means=right_meds,
                                        left_means=left_meds,
                                        mean_CR_subtract=med_CR_subtract,
                                        mean_CR_ratio=med_CR_ratio,
                                        mean_C2R_ratio=med_C2R_ratio,
                                        fig_suptitle=ridge_title,
                                        save_path=save_name_med, 
                                        start_bin=start_bin
                                        )


    # Start of main plotting function

    if f_true_bed is not None:
        # load in true etc. 
        tp = pd.read_csv(f_true_bed, sep="\t", usecols=[0, 1, 2], names=["chrom", "start", "end"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
        tp["start"] -= tolerance # don't care about out of chromosome issues
        tp["end"] += tolerance # don't care about out of chromosome issues

    if f_true_cns is not None:
        # load in the true file, assuming 
        # second col is Contour Number  
        # third col is s_imagej rounded to 2 dp
        tp = pd.read_csv(f_true_cns, sep="\t", usecols=[0, 1, 2], names=["chrom", contour_label, "s_imagej"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)

    if f_true_bed is not None and f_true_cns is not None:
        print("Cannot specify both true bed file AND true contour number / s_imagej file")
        raise ValueError
    
    true_given = False
    if f_true_bed is not None or f_true_cns is not None:
        true_given = True

    # Sort by ranking
    df_agg = df_agg.sort_values(ranking, ascending=False, ignore_index=True)

    # Top K
    # df_agg_topK = df_agg.iloc[:K]
    df_agg_topK = df_agg.copy() # instead of selecting K here, do it in the for loop so we always generate K unique plots

    # Get X and Y positiosn for plotting
    df_agg_topK = df_agg_topK.merge(df_features, how="inner", on=[contour_label, "s_imagej"])


    # Groupby for iterating through each ridge and plotting
    gb = df_agg_topK.groupby([contour_label, "s_imagej"], sort=False)

    unique_regions = []

    count_plots = 0
    for rank, (indexer, df_ridge) in enumerate(gb):


        # if indexer[0] == 18 and np.round(indexer[1], 3) == 17.086:
        #     pass

        # Plot each ridge separately
        df_min = df_ridge[x_label].min() * resolution / np.sqrt(2)
        df_max = df_ridge[x_label].max() * resolution / np.sqrt(2)
        start = max(df_min - 1000e3, 0)
        end = min(df_max + 1000e3, im.shape[1] * resolution / np.sqrt(2))

        df_min = df_ridge["X_(px)_orig"].min() * resolution / np.sqrt(2)
        df_max = df_ridge["X_(px)_orig"].max() * resolution / np.sqrt(2)
        start_correct = max(df_min - 500e3, 0)
        end_correct = min(df_max + 500e3, im.shape[1] * resolution / np.sqrt(2))

        condition_bool = False
        if f_true_bed is not None:
            # intersect with true here!
            genomic_pos_max = df_ridge["X_(px)_orig"].max() * resolution / np.sqrt(2)
            genomic_pos_min = df_ridge["X_(px)_orig"].min() * resolution / np.sqrt(2)

            condition_bool = (genomic_pos_max >= tp["start"].values) & (genomic_pos_min <= tp["end"].values)

        if f_true_cns is not None:
            condition_bool = (indexer[0] == tp[contour_label].values) & (np.round(indexer[1], 2) == tp["s_imagej"].values)

        no_true_file_specified = f_true_bed is None and f_true_cns is None

        # if true file is not given, then assume no merging with true (in this case, plot 100%)
        if np.any(condition_bool) or no_true_file_specified:

            if plot_unique:
                # then check for uniquness
                # otherwise, simply continue
                if f"{genomic_labels(start_correct, N=1)}-{genomic_labels(end_correct, N=1)}" in unique_regions:
                    print(f"\tRidge at {chromosome}:{genomic_labels(start_correct, N=2)}-{genomic_labels(end_correct, N=2)} already generated. Skipping plot...")
                    continue
                else:
                    unique_regions.append(f"{genomic_labels(start_correct, N=1)}-{genomic_labels(end_correct, N=1)}")

            if K != "all" and count_plots > K:
                break # stop plotting

            count_plots += 1 # plotting 

            ridge_title = f"{ranking}: {df_ridge.iloc[0][ranking]:.2g} (top {rank + 1} / {len(df_agg)} | p={df_ridge['p-val_corr'].iloc[0]:.3f})" 
            ridge_title += f"\n{chromosome}:{genomic_labels(start_correct, N=1)}-{genomic_labels(end_correct, N=1)} ({genomic_labels(resolution)})"
            ridge_title += f"\nContour Number: {indexer[0]} | s_imagej: {indexer[1]}"
            if np.any(condition_bool): 
                # then assume that we plot because it is merged with true
                ridge_title += " | TRUE"


            start_bin = np.round(start / resolution * np.sqrt(2)).astype(int)
            end_bin = np.round(end / resolution * np.sqrt(2)).astype(int)

            ridge_points = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=start_bin)
            ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=True, start_bin=start_bin)
            ridge_coords_curve = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0) # global
            
            # for curve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL (i.e. start_bin=0)
            im_curves, I_curves, D_curves, W1_curves, W2_curves, R_curves, C_curves = extract_line_scale_space(ridge_coords_curve,
                                                                                                                        scale_space_container=[
                                                                                                                            np.expand_dims(im, 0),
                                                                                                                            I, 
                                                                                                                            D, 
                                                                                                                            W1, 
                                                                                                                            W2, 
                                                                                                                            R, 
                                                                                                                            C
                                                                                                                            ])
            
            # TEMPORARY SAVE NUMPY D_CURVES
            # print("\t DEBUG: Saving D_curves to temp path")
            # temp_path = os.path.join(save_path, f"temp/D_curves_{indexer[0]}_{np.round(indexer[1], 2)}.npy")
            # os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            # np.save(temp_path, D_curves)
            
            A_curves = round_line_scale_space(ridge_coords_curve, scale_space_container=[A]) # round
            # A_curves = extract_angle_scale_space(ridge_coords_curve, A) # interpolate

            A_bool_curves = np.logical_and((A_curves >= angle_range[0]), (A_curves <= angle_range[1]))

            plot_summary_image()

            # plot_max_scale() 
            plot_scale_space_heatmap()

            plot_p_value_diagnostic()


from multiprocessing import Pool
from functools import partial


def _process_ridge(
    args,
    # 1) your tuple
    # 2) **all** external names this function uses:
    im, im_p_value, corr_im_p_value,
    I, D, A, W1, W2, R, C,
    resolution, scale_range, angle_range,
    window_size,               # <-- was missing
    save_path,
    num_bins, bin_size, points_min, points_max,
    f_true_bed, tolerance, f_true_cns,
    ranking, chromosome,
    contour_label, x_label, y_label,
    df_agg_len,                # <-- use this instead of len(df_agg)
    true_given
):
    rank, indexer, df_ridge = args

    # Reconstruct any “true”‐file test
    if f_true_bed is not None:
        tp = pd.read_csv(f_true_bed, sep="\t", usecols=[0,1,2],
                         names=["chrom","start","end"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
        tp["start"] -= tolerance
        tp["end"]   += tolerance
        condition_bool = (
            (df_ridge["X_(px)_orig"].max() * resolution/np.sqrt(2) >= tp["start"].values) &
            (df_ridge["X_(px)_orig"].min() * resolution/np.sqrt(2) <= tp["end"].values)
        )
    elif f_true_cns is not None:
        tp = pd.read_csv(f_true_cns, sep="\t", usecols=[0,1,2],
                         names=["chrom", contour_label, "s_imagej"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
        condition_bool = (
            (indexer[0] == tp[contour_label].values) &
            (np.round(indexer[1], 2) == tp["s_imagej"].values)
        )
    else:
        condition_bool = True

    if not np.any(condition_bool):
        return


    def plot_summary_image():
        # NOTE: changed 03/17/25 from using the "s_imagej" to "scale_assigned"
        scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["scale_assigned"]))[0]
        if len(scale_assigned_set) == 0: 
            # then don't round to 3 
            # this issue crops when saving and reading the files (numeric columns are saved to 3 d.p)
            selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["scale_assigned"]))[0][0]
        else:
            selected_scale_idx = scale_assigned_set[0]

        selected_scale = scale_range[selected_scale_idx]

        A_bool = np.logical_and(A[selected_scale_idx, :, start_bin:end_bin] >= angle_range[0], A[selected_scale_idx, :, start_bin:end_bin] <= angle_range[1])
        
        forplot_im = [
            im[:, start_bin:end_bin],
            I[selected_scale_idx, :, start_bin:end_bin], 
            D[selected_scale_idx, :, start_bin:end_bin], 
            A[selected_scale_idx, :, start_bin:end_bin], 
            A_bool,
            W1[selected_scale_idx, :, start_bin:end_bin], 
            W2[selected_scale_idx, :, start_bin:end_bin], 
            R[selected_scale_idx, :, start_bin:end_bin], 
            C[selected_scale_idx, :, start_bin:end_bin]
        ]
        # im, I, D, A, W1, W2, R, C

        titles = [
            f"Original (mean contact frequency: {np.mean(im_curves[0]):.2g})",
            f"Blurred s={selected_scale:.2f}",
            f'Ridge strength {df_ridge["ridge_strength_mean"].iloc[0]:.2f}',
            f'Angle mean: {df_ridge["angle_mean"].iloc[0]:.2f} {df_ridge["angle_unwrapped_mean"].iloc[0]:.2f} {df_ridge["angle_imagej_mean"].iloc[0]:.2f}\nderiv: {df_ridge["angle_deriv_max"].iloc[0]:.2f}',
            f'Angle mean: {df_ridge["angle_mean"].iloc[0]:.2f} {df_ridge["angle_unwrapped_mean"].iloc[0]:.2f} {df_ridge["angle_imagej_mean"].iloc[0]:.2f}\nderiv: {df_ridge["angle_deriv_max"].iloc[0]:.2f}',
            f'Eig1 {df_ridge["eig1_mean"].iloc[0]:.2f}',
            f'Eig2 {df_ridge["eig2_mean"].iloc[0]:.2f}',
            f"Ridge condition",
            f'Corner condition {df_ridge["corner_cond_fraction"].iloc[0]:.2f}',
        ]

        vmaxes = [
            np.percentile(im, q=98), # im
            np.percentile(I[selected_scale_idx], q=98), # I
            np.percentile(D[selected_scale_idx], q=98), # D
            None, # A
            None, # A
            np.percentile(W1[selected_scale_idx], q=98), # W1
            None, # W2
            None, # R
            None #C
        ]

        cmaps = [
            plt.cm.Reds, # im
            plt.cm.Reds, # I
            plt.cm.Reds, # D
            plt.cm.twilight_shifted, # A
            plt.cm.binary, # A bool
            plt.cm.bwr, # W1
            plt.cm.bwr, # W2
            plt.cm.binary, # R
            plt.cm.binary # C
        ]

        vcenters = [
            None,
            None,
            None,
            90,
            None,
            0,
            0,
            None,
            None
        ]


        lines = np.array([ridge_coords])

        forplot_curves = [
            im_curves[0],
            I_curves[selected_scale_idx],
            D_curves[selected_scale_idx],
            A_curves[selected_scale_idx],
            A_bool_curves[selected_scale_idx],
            W1_curves[selected_scale_idx],
            W2_curves[selected_scale_idx],
            R_curves[selected_scale_idx],
            C_curves[selected_scale_idx],
        ]


        fig, ax = plt.subplots(2, 9, layout="constrained", figsize=(30, 10))

        fig.suptitle(ridge_title)

        for i in range(9): # Plot 9 images
            # Plot images
            if vcenters[i] is not None:
                im0 = ax[0, i].imshow(np.flipud(forplot_im[i]), cmap=cmaps[i], origin="lower", interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenters[i], vmax=vmaxes[i]))
            else:
                im0 = ax[0, i].imshow(np.flipud(forplot_im[i]), cmap=cmaps[i], origin="lower", interpolation="none", vmax=vmaxes[i])
            ax[0, i].set_title(titles[i])
            set_genomic_ticks(ax[0, i], num_ticks=[5, 8], resolution=resolution, mat_shape=forplot_im[i].shape, genomic_shift=[0, start], dp=1)

            cbar = fig.colorbar(im0, ax=ax[0, i])
            
            for j, line in enumerate(lines):
                ax[0, i].plot(line[:, 0], line[:, 1], linewidth=1.2, label=None, alpha=0.7)

            # Plot curves 
            ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), forplot_curves[i], color="blue", marker="o", markersize=4)
            if i == 3: # the 3rd plot is the angle plot
                # We should try to plot the other angle values in `df_ridge`
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle"], label="angle", alpha=0.5)
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle_unwrapped"], label="angle unwrapped", alpha=0.5)
                ax[1, i].plot(np.arange(forplot_curves[i].shape[0]), df_ridge["angle_imagej"], label="imageJ angle", alpha=0.5)
                ax[1, i].legend(loc="best")

                # ASSERT
                # assert np.allclose(np.round(df_ridge["angle"].values, 2), np.round(forplot_curves[i], 2))

            num_ticks = 10
            def_xticks = np.arange(0, forplot_curves[i].shape[0], np.ceil(forplot_curves[i].shape[0] / num_ticks).astype(int))
            ticks_bp_x = [f"{genomic_labels(window_size / 2 - x[1] * resolution / np.sqrt(2))}" for x in df_ridge[[x_label, y_label]].values[def_xticks]]

            ax[1, i].set_xticks(def_xticks)
            ax[1, i].set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)

        # Start of main plotting function
        if true_given: #"merge-true_"
            save_name = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_summary.png") 
        else:
            save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_summary.png") 

        plt.savefig(save_name, dpi=400)

        plt.close()



    def plot_scale_space_heatmap():

        scale_assigned_set = np.where(np.isclose(np.round(scale_range, 3), df_ridge.iloc[0]["scale_assigned"]))[0]
        if len(scale_assigned_set) == 0: 
            # then don't round to 3 
            # this issue crops when saving and reading the files (numeric columns are saved to 3 d.p)
            selected_scale_idx = np.where(np.isclose(scale_range, df_ridge.iloc[0]["scale_assigned"]))[0][0]
        else:
            selected_scale_idx = scale_assigned_set[0]

        selected_scale = scale_range[selected_scale_idx]

        fig, axes = plt.subplots(4, 5, figsize=(24, 10), layout="constrained")

        # main plot
        axes[0,0].imshow(D_curves, aspect="auto", cmap="viridis", vmax=np.percentile(D_curves, 95))

        scale_idx_argmax, pos_argmax = argrelmax(D_curves, axis=0)
        axes[0,0].scatter(pos_argmax, scale_idx_argmax, color="cyan", marker="x", s=15)
        axes[0,0].scatter(np.arange(D_curves.shape[1]), 
                        np.argmax(D_curves, axis=0), 
                        color="red", s=15, marker="o")

        axes[0,0].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,0].invert_yaxis()
        axes[0,0].set_ylabel("Scale")
        axes[0,0].set_xlabel("Ridge position")
        axes[0,0].set_title("Ridge Strength Signature (global maxima in red)")

        # angle signature
        t0 = axes[0,1].imshow(A_curves, aspect="auto", cmap="twilight_shifted", norm=colors.TwoSlopeNorm(vcenter=90))
        axes[0,1].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,1].invert_yaxis()
        axes[0,1].set_ylabel("Scale")
        axes[0,1].set_xlabel("Ridge position")
        axes[0,1].set_title("Angle Signature")
        cbar = plt.colorbar(t0, ax=axes[0,1])

        # angle bool condition signature
        t0 = axes[0,2].imshow(A_bool_curves, aspect="auto", cmap="binary")
        axes[0,2].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,2].invert_yaxis()
        axes[0,2].set_ylabel("Scale")
        axes[0,2].set_xlabel("Ridge position")
        axes[0,2].set_title("Angle Bool Signature")
        cbar = plt.colorbar(t0, ax=axes[0,2])

        # ridge condition signature
        t0 = axes[0,3].imshow(R_curves, aspect="auto", cmap="binary", vmax=np.percentile(R_curves, 95))
        axes[0,3].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,3].invert_yaxis()
        axes[0,3].set_ylabel("Scale")
        axes[0,3].set_xlabel("Ridge position")
        axes[0,3].set_title("Ridge Condition Signature")
        cbar = plt.colorbar(t0, ax=axes[0,3])

        # corner condition signature
        t0 = axes[0,4].imshow(C_curves, aspect="auto", cmap="binary")
        axes[0,4].set_yticks(np.arange(len(scale_range)), [f"{x:.2f}" for x in scale_range])
        axes[0,4].invert_yaxis()
        axes[0,4].set_ylabel("Scale")
        axes[0,4].set_xlabel("Ridge position")
        axes[0,4].set_title("Corner Condition Signature")
        cbar = plt.colorbar(t0, ax=axes[0,4])


        for i in range(5):
            # Add rectangle around the selected row
            rect = patches.Rectangle(
                (0 - 0.5, selected_scale_idx - 0.5),  # (x, y) of bottom-left corner
                width=D_curves.shape[1],                      # full width
                height=1,                            # height of 1 row
                edgecolor='red',                     # box color
                facecolor='none',                    # no fill
                linewidth=1
            )
            axes[0, i].add_patch(rect)

    
        # column-wise expected scale and fitted polynomial + statistics
        # col_vars = np.var(D_curves, axis=0)
        # col_vars_scale = np.var(D_curves[:selected_scale_idx+1, :], axis=0)
        # col_mean_diff = df_ridge["col_mean_diff"].values
        expected_scales = df_ridge["expected_scale"].values
        coeffs = df_ridge["coeffs"].values[0]
        coeffs_str = ",".join(f"{c:.3f}" for c in coeffs)
        y_fit_ev = np.polyval(coeffs, np.arange(len(expected_scales)))
        axes[1,0].scatter(range(len(expected_scales)), expected_scales, label="all scales", s=10)
        axes[1,0].plot(y_fit_ev, label="Cubic")
        axes[1,0].set_title(f"Expected scales \nparam: {coeffs_str} | norm rmse: {df_ridge['rmse'].iloc[0]:.4f} | rmse: {df_ridge['rmse'].iloc[0] * len(expected_scales):.3f}")
        axes[1,0].set_xlabel("Ridge position")

        # column-wise mean of ridge strength
        col_mean = np.mean(D_curves, axis=0)
        # col_mean_scale = np.mean(D_curves[:selected_scale_idx+1, :], axis=0)
        axes[2,0].scatter(range(len(col_mean)), col_mean, label="all scales", s=10)
        # axes[2,0].scatter(range(len(col_mean_scale)), col_mean_scale, label="≤ selected scale")
        axes[2,0].set_title("Column-wise mean")
        axes[2,0].set_xlabel("Ridge position")
        # axes[2,0].legend() 

        # column-wise variance of angle
        col_vars = np.var(A_curves, axis=0)
        col_vars_scale = np.var(A_curves[:selected_scale_idx+1, :], axis=0)
        axes[1,1].scatter(range(len(col_vars)), col_vars, label="all scales", s=10)
        axes[1,1].scatter(range(len(col_vars_scale)), col_vars_scale, label="≤ selected scale", s=10)
        axes[1,1].set_title("Column-wise variance")
        axes[1,1].set_xlabel("Ridge position")
        axes[1,1].legend() 
        # column-wise mean of ridge strength
        col_mean = np.mean(A_curves, axis=0)
        col_mean_scale = np.mean(A_curves[:selected_scale_idx+1, :], axis=0)
        axes[2,1].scatter(range(len(col_mean)), col_mean, label="all scales", s=10)
        axes[2,1].scatter(range(len(col_mean_scale)), col_mean_scale, label="≤ selected scale", s=10)
        axes[2,1].set_title("Column-wise mean")
        axes[2,1].set_xlabel("Ridge position")
        axes[2,1].legend() 


        # Entropy Histogram 1
        edges = df_ridge["edges"].iloc[0]
        axes[3,0].stairs(values=df_ridge["pmf"].iloc[0], edges=edges, 
                        fill=True, edgecolor="black", facecolor="blue")
        axes[3,0].set_xticks(edges)
        axes[3, 0].set_xticklabels(edges, rotation=45)
        axes[3,0].tick_params(labelbottom=True)
        axes[3,0].set_xlabel("Column-wise mean values")
        axes[3,0].set_ylabel("Probability")
        ent_str = f'Normalized Entropy: {df_ridge["entropy"].iloc[0]:.3f}'
        if num_bins is not None:
            axes[3,0].set_title(f"Histogram (range: [{points_min}, {points_max}] num bins: {num_bins})\n{ent_str}")
        else:
            axes[3,0].set_title(f"Histogram (range: [{points_min}, {points_max}] bin size: {bin_size})\n{ent_str}")


        # Width Histogram
        axes[3,1].hist(df_ridge["width"], bins=10, color="blue")
        axes[3,1].set_xlabel("Ridge width (px)")
        axes[3,1].set_ylabel("Frequency")
        axes[3,1].set_title(f"Width Histogram (mean: {df_ridge['width_mean'].iloc[0]:.2f}, median: {df_ridge['width'].median():.2f}, length={len(df_ridge)})")


        for ax in axes.flatten():
            if not ax.has_data():
                ax.axis("off")

        fig.suptitle(ridge_title)

        if true_given: #"merge-true_"
            save_name = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_scale_signature.png")
        else:
            save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_scale_signature.png")

        plt.savefig(save_name, dpi=400)
        plt.close()

    def plot_p_value_diagnostic():

        for factor_lr in [1]: 

            if true_given: #"merge-true_"
                save_name_mean = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_mean_lr{factor_lr}-p_value.png")
                save_name_med = os.path.join(save_path, f"merge-true_rank-{rank + 1}_{genomic_labels(start_correct)}_med_lr{factor_lr}-p_value.png")
            else:
                save_name_mean = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_mean_lr{factor_lr}-p_value.png")
                save_name_med = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_med_lr{factor_lr}-p_value.png")
            
            left_means = []
            right_means = []
            center_means = []
            left_vals = []
            right_vals = []
            center_vals = []
            mean_CR_subtract = []
            mean_CR_ratio = []
            mean_C2R_ratio = []
            left_meds = []
            right_meds = []
            center_meds = []
            med_CR_subtract = []
            med_CR_ratio = []
            med_C2R_ratio = []

            for _, im_each in enumerate([im_p_value, corr_im_p_value]):

                (
                    left_means_each,
                    right_means_each,
                    center_means_each,
                    left_box_coords,
                    right_box_coords,
                    center_box_coords,
                    left_num_points,
                    right_num_points,
                    center_num_points,
                    left_vals_each,
                    right_vals_each,
                    center_vals_each, 
                    left_med_each,
                    right_med_each,
                    center_med_each,
                ) = compute_test_statistic_quantities(im=im_each, 
                                                    ridge_points=ridge_coords_curve,
                                                    ridge_angles=-df_ridge["angle_imagej"].values - 90, # -1 * ANGLE - 90 (negate and make normal)
                                                    width_in=df_ridge["width"].values,
                                                    height=1,
                                                    im_shape=im_each.shape, 
                                                    factor_lr=factor_lr # Left, right box width is width * factor_lr
                                                    )
                

                mean_CR_subtract_each, mean_CR_ratio_each, mean_C2R_ratio_each = compute_test_statistic(left_means_each, right_means_each, center_means_each)
                med_CR_subtract_each, med_CR_ratio_each, med_C2R_ratio_each = compute_test_statistic(left_med_each, right_med_each, center_med_each)

                left_means.append(left_means_each)
                right_means.append(right_means_each)
                center_means.append(center_means_each)
                left_vals.append(left_vals_each)
                right_vals.append(right_vals_each)
                center_vals.append(center_vals_each)
                mean_CR_subtract.append(mean_CR_subtract_each)
                mean_CR_ratio.append(mean_CR_ratio_each)
                mean_C2R_ratio.append(mean_C2R_ratio_each)
                left_meds.append(left_med_each)
                right_meds.append(right_med_each)
                center_meds.append(center_med_each)
                med_CR_subtract.append(med_CR_subtract_each)
                med_CR_ratio.append(med_CR_ratio_each)
                med_C2R_ratio.append(med_C2R_ratio_each)

            # Plot one for mean
            plot_p_value_observed_null(im_p_value=im_p_value[:, start_bin:end_bin],
                                    corr_im_p_value=corr_im_p_value[:, start_bin:end_bin],
                                    ridge_points=ridge_points, 
                                    ridge_widths=df_ridge["width"].values, 
                                    center_box_coords=center_box_coords,
                                    right_box_coords=right_box_coords,
                                    left_box_coords=left_box_coords,
                                    center_num_points=center_num_points,
                                    right_num_points=right_num_points,
                                    left_num_points=left_num_points,
                                    center_vals=center_vals,
                                    right_vals=right_vals,
                                    left_vals=left_vals,
                                    center_means=center_means,
                                    right_means=right_means,
                                    left_means=left_means,
                                    mean_CR_subtract=mean_CR_subtract,
                                    mean_CR_ratio=mean_CR_ratio,
                                    mean_C2R_ratio=mean_C2R_ratio,
                                    fig_suptitle=ridge_title,
                                    save_path=save_name_mean, 
                                    start_bin=start_bin
                                    )
            
            # Plot one for median
            plot_p_value_observed_null(im_p_value=im_p_value[:, start_bin:end_bin],
                                        corr_im_p_value=corr_im_p_value[:, start_bin:end_bin],
                                        ridge_points=ridge_points, 
                                        ridge_widths=df_ridge["width"].values, 
                                        center_box_coords=center_box_coords,
                                        right_box_coords=right_box_coords,
                                        left_box_coords=left_box_coords,
                                        center_num_points=center_num_points,
                                        right_num_points=right_num_points,
                                        left_num_points=left_num_points,
                                        center_vals=center_vals,
                                        right_vals=right_vals,
                                        left_vals=left_vals,
                                        center_means=center_meds,
                                        right_means=right_meds,
                                        left_means=left_meds,
                                        mean_CR_subtract=med_CR_subtract,
                                        mean_CR_ratio=med_CR_ratio,
                                        mean_C2R_ratio=med_C2R_ratio,
                                        fig_suptitle=ridge_title,
                                        save_path=save_name_med, 
                                        start_bin=start_bin
                                        )
            
    # if indexer[0] == 18 and np.round(indexer[1], 3) == 17.086:
    #     pass

    # Plot each ridge separately
    df_min = df_ridge[x_label].min() * resolution / np.sqrt(2)
    df_max = df_ridge[x_label].max() * resolution / np.sqrt(2)
    start = max(df_min - 1000e3, 0)
    end = min(df_max + 1000e3, im.shape[1] * resolution / np.sqrt(2))

    df_min = df_ridge["X_(px)_orig"].min() * resolution / np.sqrt(2)
    df_max = df_ridge["X_(px)_orig"].max() * resolution / np.sqrt(2)
    start_correct = max(df_min - 500e3, 0)
    end_correct = min(df_max + 500e3, im.shape[1] * resolution / np.sqrt(2))

    condition_bool = False
    if f_true_bed is not None:
        # intersect with true here!
        genomic_pos_max = df_ridge["X_(px)_orig"].max() * resolution / np.sqrt(2)
        genomic_pos_min = df_ridge["X_(px)_orig"].min() * resolution / np.sqrt(2)

        condition_bool = (genomic_pos_max >= tp["start"].values) & (genomic_pos_min <= tp["end"].values)

    if f_true_cns is not None:
        condition_bool = (indexer[0] == tp[contour_label].values) & (np.round(indexer[1], 2) == tp["s_imagej"].values)

    no_true_file_specified = f_true_bed is None and f_true_cns is None

    # if true file is not given, then assume no merging with true (in this case, plot 100%)
    if np.any(condition_bool) or no_true_file_specified:

        ridge_title = f"{ranking}: {df_ridge.iloc[0][ranking]:.2g} (top {rank + 1} / {df_agg_len} | p={df_ridge['p-val_corr'].iloc[0]:.3f})" 
        ridge_title += f"\n{chromosome}:{genomic_labels(start_correct, N=1)}-{genomic_labels(end_correct, N=1)} ({genomic_labels(resolution)})"
        ridge_title += f"\nContour Number: {indexer[0]} | s_imagej: {indexer[1]}"
        if np.any(condition_bool): 
            # then assume that we plot because it is merged with true
            ridge_title += " | TRUE"


        start_bin = np.round(start / resolution * np.sqrt(2)).astype(int)
        end_bin = np.round(end / resolution * np.sqrt(2)).astype(int)

        ridge_points = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=start_bin)
        ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=True, start_bin=start_bin)
        ridge_coords_curve = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=False, start_bin=0) # global
        
        # for curve extraction, do NOT flip y axis AND ensure coordinates are GLOBAL (i.e. start_bin=0)
        im_curves, I_curves, D_curves, W1_curves, W2_curves, R_curves, C_curves = extract_line_scale_space(ridge_coords_curve,
                                                                                                                    scale_space_container=[
                                                                                                                        np.expand_dims(im, 0),
                                                                                                                        I, 
                                                                                                                        D, 
                                                                                                                        W1, 
                                                                                                                        W2, 
                                                                                                                        R, 
                                                                                                                        C
                                                                                                                        ])
        
        # TEMPORARY SAVE NUMPY D_CURVES
        # print("\t DEBUG: Saving D_curves to temp path")
        # temp_path = os.path.join(save_path, f"temp/D_curves_{indexer[0]}_{np.round(indexer[1], 2)}.npy")
        # os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        # np.save(temp_path, D_curves)
        
        A_curves = round_line_scale_space(ridge_coords_curve, scale_space_container=[A]) # round
        # A_curves = extract_angle_scale_space(ridge_coords_curve, A) # interpolate

        A_bool_curves = np.logical_and((A_curves >= angle_range[0]), (A_curves <= angle_range[1]))

        plot_summary_image()

        # plot_max_scale() 
        plot_scale_space_heatmap()

        plot_p_value_diagnostic()
        
    

def plot_top_k_diagnostic_parallel(
    df_agg, df_features, K, im, im_p_value, corr_im_p_value,
    I, D, A, W1, W2, R, C, ranking, resolution, chromosome,
    scale_range, window_size, save_path, num_bins, bin_size,
    points_min, points_max, num_cores,
    f_true_bed, tolerance, angle_range,
    f_true_cns, verbose, contour_label="Contour Number",
    x_label="X_(px)_unmap", y_label="Y_(px)_unmap"
):

    if f_true_bed is not None:
        tp = pd.read_csv(f_true_bed, sep="\t", usecols=[0,1,2],
                         names=["chrom","start","end"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)
        tp["start"] -= tolerance
        tp["end"]   += tolerance

    if f_true_cns is not None:
        tp = pd.read_csv(f_true_cns, sep="\t", usecols=[0,1,2],
                         names=["chrom", contour_label, "s_imagej"])
        tp = tp.loc[tp["chrom"] == chromosome].reset_index(drop=True)

    if f_true_bed is not None and f_true_cns is not None:
        raise ValueError("Cannot specify both true bed file AND true contour number / s_imagej file")

    true_given = (f_true_bed is not None) or (f_true_cns is not None)

    # 1) sort & merge as before
    df_agg = df_agg.sort_values(ranking, ascending=False, ignore_index=True)
    df_agg_topK = df_agg.merge(df_features, on=[contour_label, "s_imagej"])

    items = list(df_agg_topK.groupby([contour_label, "s_imagej"], sort=False))
    if K != "all":
        items = items[:K]

    # 2) build the little (rank, indexer, df_ridge) tuples
    pool_args = [(rank, idx, df) for rank, (idx, df) in enumerate(items)]

    # 3) bind everything else into our worker
    worker = partial(
        _process_ridge,
        # note: args tuple stays unbound, partial binds everything else:
        im=im, 
        im_p_value=im_p_value,               # <-- newly bound
        corr_im_p_value=corr_im_p_value,     # <-- newly bound
        I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
        resolution=resolution, 
        scale_range=scale_range, 
        angle_range=angle_range,
        window_size=window_size,             # <-- already in your partial but missing above
        save_path=save_path,
        num_bins=num_bins, 
        bin_size=bin_size, 
        points_min=points_min, 
        points_max=points_max,
        f_true_bed=f_true_bed, 
        tolerance=tolerance, 
        f_true_cns=f_true_cns,
        ranking=ranking, 
        chromosome=chromosome,
        contour_label=contour_label, 
        x_label=x_label, 
        y_label=y_label,
        df_agg_len=len(df_agg),              # <-- pass the total count
        true_given=true_given,
    )

    # 4) launch the pool
    if num_cores is None:
        num_cores = max(1, os.cpu_count() - 1)

    with Pool(processes=num_cores) as pool:
        if verbose:
            for _ in tqdm(
                pool.imap_unordered(worker, pool_args),
                total=len(pool_args),
                desc="Plotting ridges"
            ):
                pass
        else:
            pool.map(worker, pool_args)



def plot_entropy_distribution(df_agg, num_bins, bin_size, points_min, points_max, save_path):
    # Filter out zeros from the entropy values
    data = df_agg["entropy"].values
    data = data[data != 0]

    if np.isnan(data).all():
        print("All entropy values are NaN after filtering out zeros.")
        return

    # Compute statistics on the non-zero data
    N = len(data)
    mean_val = np.nanmean(data)
    median_val = np.nanmedian(data)
    p25 = np.nanpercentile(data, 25)
    p75 = np.nanpercentile(data, 75)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Create the histogram using the filtered data
    plt.hist(data, bins=40)
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Frequency')

    # Create a title that includes data range, mode info, and statistics
    if num_bins is not None:
        title = f'Data range [{points_min}, {points_max}]\nNumber of bins mode: {num_bins}'
    else:
        title = f'Data range [{points_min}, {points_max}]\nBin size mode: {bin_size}'

    title += (f'\nMean: {mean_val:.2f}, Median: {median_val:.2f}, '
              f'25th: {p25:.2f}, 75th: {p75:.2f}, N: {N}')
    plt.title(title)

    # Save the figure
    save_name = os.path.join(save_path, "entropy_histogram.png")
    plt.savefig(save_name, dpi=400)
    plt.close()



def plot_saliency_distribution(df_agg, ranking, q, save_path, bins=30):
    # Extract raw data
    data = df_agg[ranking].values
    nonzero_data = data[~np.isclose(data, 0)]

    # Compute the bin edges up front
    bin_edges = np.histogram_bin_edges(data, bins=bins)

    # Compute stats for full data
    N_all      = len(data)
    mean_all   = np.nanmean(data)
    median_all = np.nanpercentile(data, q=q)
    p25_all    = np.nanpercentile(data, 25)
    p75_all    = np.nanpercentile(data, 75)

    # Compute stats for “zero‐bin”‐filtered data
    N_nz       = len(nonzero_data)
    mean_nz    = np.nanmean(nonzero_data)
    median_nz  = np.nanpercentile(nonzero_data, q=q)
    p25_nz     = np.nanpercentile(nonzero_data, 25)
    p75_nz     = np.nanpercentile(nonzero_data, 75)

    # Create two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")

    # Left: all data
    ax[0].hist(data, bins=bins)
    ax[0].axvline(median_all, color='red', linestyle=':', label=f'{q}-Percentile={median_all:.2f}')
    ax[0].legend()
    ax[0].set_xlabel('Saliency')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title(
        f'Including zeros\n'
        f'Range: [{data.min():.2f}, {data.max():.2f}]\n'
        f'Mean: {mean_all:.2f}, Median: {median_all:.2f}\n'
        f'25th: {p25_all:.2f}, 75th: {p75_all:.2f}, N={N_all}'
    )

    # Right: with the zero‐bin removed
    ax[1].hist(nonzero_data, bins=bins)
    ax[1].axvline(median_all, color='red', linestyle=':', label=f'{q}-Percentile={median_all:.2f}')
    ax[1].legend()
    ax[1].set_xlabel('Saliency')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title(
        f'Zero-bin removed\n'
        f'Range: [{nonzero_data.min():.2f}, {nonzero_data.max():.2f}], '
        f'Mean: {mean_nz:.2f}, Median: {median_nz:.2f}\n'
        f'25th: {p25_nz:.2f}, 75th: {p75_nz:.2f}, N={N_nz}'
    )

    fig.suptitle(f'Distribution of Jet Saliency (for saliency thresholding diagnostics)')

    # Save and close
    save_name = os.path.join(save_path, "saliency_histogram.png")
    plt.savefig(save_name, dpi=400)
    plt.close()




from utils.plotting import plot_n_rect
from .expanded_table import rect_to_square
import random
import cv2 as cv


def plot_top_k(df_agg, df_features, K, ranking, hic_file, chromosome, resolution, window_size, normalization, rotation_padding, save_path, 
               root, parameter_str, im_vmin, im_vmax,
               contour_label="Contour Number",
               x_label="X_(px)_orig", 
               y_label="Y_(px)_orig"):
    """
    The output plot for users
    """

    window_size_bin = np.ceil(window_size / resolution).astype(int)

    im = read_hic_rectangle(hic_file, chromosome, resolution, window_size_bin, data_type="observed", normalization=normalization, whiten=None,
                            rotate_mode=rotation_padding, cval=0, handle_zero_sum=None, verbose=False)
    
    im = np.log10(im + 1)
    im = cv.normalize(im, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)

    save_name = os.path.join(save_path, f"{root}_plot_{K}.png") 

    # first, subset the correct results
    # Sort by ranking
    df_agg.sort_values(ranking, inplace=True, ascending=False, ignore_index=True)

    # Top K
    if K == "all":
        # then do not subset, but use all
        df_agg_topK = df_agg.copy()
    else:
        df_agg_topK = df_agg.iloc[:K]

    # Get X and Y positiosn for plotting
    df_agg_topK = df_agg_topK.merge(df_features[[contour_label, "s_imagej", x_label, y_label, "width"]], how="inner", on=[contour_label, "s_imagej"])

    # Groupby for iterating through each ridge and plotting
    gb = df_agg_topK.groupby([contour_label, "s_imagej"])

    # Collect lines
    lines = []
    line_widths = []
    for rank, (indexer, df_ridge) in enumerate(gb):

        ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=True, start_bin=0)

        lines.append(ridge_coords)
        line_widths.append(df_ridge["width"].values)

    # Just plot im with the lines overlaid
    if lines:
        colors = [plt.cm.winter(x) for x in np.linspace(0, 1, len(lines[0]))]
        random.shuffle(colors)
        plot_n_rect([im, im, im], ["", "", ""], f"{root}", resolution,
                    savepath=save_name, lines=[None, lines, lines], line_widths=[None, None, line_widths], line_colors=colors, cmap="Reds", num_ticks=[5, 100],
                    vmax=[np.percentile(im, im_vmax), np.percentile(im, im_vmax), np.percentile(im, im_vmax)])
    else:
        plot_n_rect([im, im, im], ["", "", ""], f"{root}", resolution,
                    savepath=save_name, cmap="Reds", num_ticks=[5, 100],
                    vmax=[np.percentile(im, im_vmax), np.percentile(im, im_vmax), np.percentile(im, im_vmax)])
                

def plot_corner_diagnostic(df_agg, df_features, K, ranking, im, im_corner, corner_type, C, scale_range, resolution, save_path, root,               
                           contour_label="Contour Number",
                           x_label="X_(px)_unmap", 
                           y_label="Y_(px)_unmap"):
    
    s_idx = np.linspace(0, C.shape[0] - 1, 5, dtype=int)

    save_name = os.path.join(save_path, f"{root}_diagnostic_corner_top{K}.png") 
    # save_name = save_path + f"{root}_diagnostic_corner_top{K}.png"

    # first, subset the correct results
    # Sort by ranking
    df_agg.sort_values(ranking, inplace=True, ascending=False, ignore_index=True)

    # Top K
    df_agg_topK = df_agg.iloc[:K]

    # Get X and Y positiosn for plotting
    df_agg_topK = df_agg_topK.merge(df_features[[contour_label, "s_imagej", x_label, y_label]], how="inner", on=[contour_label, "s_imagej"])

    # Groupby for iterating through each ridge and plotting
    gb = df_agg_topK.groupby([contour_label, "s_imagej"])

    # Collect lines
    lines = []
    for rank, (indexer, df_ridge) in enumerate(gb):

        ridge_coords = convert_imagej_coord_to_numpy(df_ridge[[x_label, y_label]].values, im.shape[0], flip_y=True, start_bin=0)

        lines.append(ridge_coords)

    # Just plot im with the lines overlaid
    colors = [plt.cm.winter(x) for x in np.linspace(0, 1, len(lines[0]))]
    random.shuffle(colors)
    plot_n_rect([im, im, im_corner] + list(C[s_idx]), ["Original", "Original", corner_type] + [f"Corner s={x:.2f}" for x in scale_range[s_idx]], 
                f"{root}",
                resolution, 
                lines=[None, lines, lines] + [lines] * len(s_idx), 
                line_colors=colors, 
                cmap=["Reds", "Reds", "Reds"] + ["binary"] * len(s_idx), 
                vmax=[np.percentile(im, 98), np.percentile(im, 98), np.percentile(im_corner, 98)] + [None] * len(s_idx),
                num_ticks=[5, 100],
                savepath=save_name)





def make_bed(df_pos_all, N, chromosome, window_size, resolution, x_label, y_label, save_name):

    df_pos_all["chr1"] = chromosome
    df_pos_all["chr2"] = chromosome
    
    window_size_bin = np.ceil(window_size / resolution).astype(int)
    coords = rect_to_square(N, window_size_bin, df_pos_all[[y_label, x_label]].values)
    rows, cols = coords[:, 0], coords[:, 1]

    # define the columns
    df_pos_all["y1"] = np.ceil(cols * resolution).astype(int)
    df_pos_all["y2"] = np.ceil(df_pos_all["y1"] + resolution / 2).astype(int)
    
    # define the rows
    df_pos_all["x1"] = np.ceil(rows * resolution).astype(int)
    df_pos_all["x2"] = np.ceil(df_pos_all["x1"] + resolution / 2).astype(int)
        
    df_pos_all[["chr1", "x1", "x2", "chr2", "y1", "y2"]].to_csv(save_name, sep="\t", header=None, index=False)

from utils.file_io import save_csv

def format_summary_table(df_agg_in, df_features_in, chromosome, resolution, ranking, scale_range,
                         contour_label="Contour Number", x_label="X_(px)_orig"):
    """
    Format summary table for results
    * Combines chromosome, Contour Number and s_imagej into a single unique identifier
    * Adds new columns: chrom, start, end, length
    * Keeps: angle_mean, input_mean, `ranking`, ks, p-val
    """
    df_agg = df_agg_in.copy()
    df_features = df_features_in.copy()

    # convert s_imagej to index accoridng to scale_range for memory constraints
    scale_index_agg = np.digitize(df_agg["s_imagej"].values, scale_range) - 1
    scale_index_features = np.digitize(df_features["s_imagej"].values, scale_range) - 1

    # new unique identifier
    df_agg["unique_id"] = chromosome + "_" + df_agg[contour_label].astype(str) + "_" + scale_index_agg.astype(str)
    df_features["unique_id"] = chromosome + "_" + df_features[contour_label].astype(str) + "_" + scale_index_features.astype(str)

    df_agg["chrom"] = chromosome
    df_agg["p-val_raw"] = df_agg["p-val"]

    factor = resolution / np.sqrt(2)
    agg = df_features.groupby("unique_id").agg(start=(x_label, lambda x : np.min(x) * factor),
                                               end=(x_label, lambda x : np.max(x) * factor),
                                               length=(x_label, lambda x : x.count() * factor)).reset_index()

    # ensure that other columns are kept in the summary table (e.g. ks)
    df_agg = df_agg.merge(agg, on="unique_id", how="left")

    keep = ["unique_id", "chrom", "start", "end", "length", "input_mean", "angle_mean", "width_mean", ranking, "ks", "p-val_raw", "p-val_corr"]

    # sort by ranking
    df_agg.sort_values(ranking, inplace=True, ascending=False, ignore_index=True)

    return df_agg[keep].copy()

def format_expanded_table(df_features, chromosome, resolution, scale_range, window_size, N, 
                          x_label="X_(px)_orig", y_label="Y_(px)_orig", contour_label="Contour Number"):
    """
    Format expanded table for results
    * Combines chromosome, Contour Number and s_imagej into a single unique identifier
    * Adds new columns: chrom, x (bp), y (bp)
    * Keeps: width, angle_imagej, ridge_strength
    """
    df = df_features.copy()

    scale_index = np.digitize(df["s_imagej"].values, scale_range) - 1

    # new unique identifier
    df["unique_id"] = chromosome + "_" + df[contour_label].astype(str) + "_" + scale_index.astype(str)

    # rotate coordinates from rectangle to square then to genomic coordinates
    window_size_bin = np.ceil(window_size / resolution).astype(int)
    coords = rect_to_square(N, window_size_bin, df[[y_label, x_label]].values)
    rows, cols = coords[:, 0], coords[:, 1]
    df["x (bp)"] = cols * resolution
    df["y (bp)"] = rows * resolution 

    # for pixels, simply have them in terms of rectangle coordinates
    df["x (pixels)"] = df[x_label]
    df["y (pixels)"] = df[y_label]
    df["chrom"] = chromosome

    keep = ["unique_id", "chrom", "x (bp)", "y (bp)", "x (pixels)", "y (pixels)", "width", "angle_imagej", "ridge_strength"]

    return df[keep].copy()


def save_results(df_agg, df_features, K, ranking, save_path, chromosome, N_removed, 
                 rm_idx, window_size, resolution, root, parameter_str, scale_range,
                 hic_file, normalization, rotation_padding, im_vmax, im_vmin, plot,
                 contour_label="Contour Number",
                 x_label="X_(px)_orig", 
                 y_label="Y_(px)_orig"
               ):
    """
    Saves 3 results for the top K ridges (or all if K == "all"):
    1. Expanded table 
    2. Summary table 
    3. Bedpe visualization 
    4. Plots ridges 

    The tables are ranked by the `ranking` column (i.e. jet saliency)

    Parameters
    ----------
    df_agg : pd.DataFrame
        Summary dataframe
    df_features : pd.DataFrame
        Expanded dataframe
    K : int or str
        Number of top ridges to save, or "all" for all ridges
    ranking : str
        Column name to rank the ridges by 
    save_path : str
        Path to save the results
    chromosome : str
        Chromosome name (e.g. "chr1")
    N_removed : int
        NxN square size after removing zero sum rows (or columns)
    rm_idx : list
        List of indices that were removed from the original contact map 
    window_size : int
        Window size in base pairs
    resolution : int
        Resolution in base pairs
    root : str
        Root name for the saved files
    parameter_str : str
        String containing formatted parameters of this run
    scale_range : np.ndarray
        Scales generated

    hic_file : str, optional        *(added)*
        Path to the .hic file for plotting contact maps.
    normalization : str, optional   *(added)*
        Normalization method to pass to `plot_top_k`.
    rotation_padding : int, optional*(added)*
        Padding (in pixels) used by `plot_top_k` when rotating.
    im_vmax : float, optional      *(added)*
        Upper‐bound intensity for image color‐scaling.
    im_vmin : float, optional      *(added)*
        Lower‐bound intensity for image color‐scaling.
    plot : bool, default=False      *(added)*
        If True, calls `plot_top_k` after saving tables.
    
    Returns
    -------
    None, but saves the following files:
    1. `{root}_expanded_table.csv` - Expanded table with features for the top K ridges
    2. `{root}_summary_table.csv` - Summary table with aggregated features for the top K ridges
    3. `{root}_juicer-visualize.bedpe` - Bedpe file for visualization in Juicer
    4. `{root}_plot_{K}.png` - Plot of the top K ridges
    """
    # Sort by ranking
    df_agg.sort_values(ranking, inplace=True, ascending=False, ignore_index=True)

    # Subset top K
    if K == "all":
        df_agg_topK = df_agg.copy()
    else:
        df_agg_topK = df_agg.iloc[:K]

    # Get X and Y positions
    keep = [contour_label, "s_imagej", x_label, y_label, "width", "angle_imagej", "ridge_strength"]
    df_features_topK = df_agg_topK.merge(df_features[keep], how="inner", on=[contour_label, "s_imagej"])

    # Save bedpe visualization
    save_name = os.path.join(save_path, f"{root}_juicer-visualize.bedpe") 
    N = N_removed + len(rm_idx)
    make_bed(df_features_topK, N, chromosome, window_size, resolution, x_label, y_label, save_name)
    
    # Save expanded table
    save_name = os.path.join(save_path, f"{root}_expanded_table.csv") 
    df_features_topK = format_expanded_table(df_features_topK, chromosome, resolution, scale_range, window_size, N,
                                             x_label=x_label, y_label=y_label, contour_label=contour_label)
    save_csv(df_features_topK, save_name, root, parameter_str, dp=3, exclude_rounding=[]) # round everything 

    # Save summary table
    save_name = os.path.join(save_path, f"{root}_summary_table.csv")
    df_agg_topK = format_summary_table(df_agg_topK, df_features, chromosome, resolution, ranking, scale_range,
                                       contour_label=contour_label, x_label=x_label)
    save_csv(df_agg_topK, save_name, root, parameter_str, dp=3)

    # Plot top K
    if plot:
        plot_top_k(df_agg, df_features, K, ranking, hic_file, chromosome, resolution, window_size, normalization, rotation_padding,
                   save_path, im_vmax=im_vmax, im_vmin=im_vmin, root=root, parameter_str=parameter_str)






    # def plot_curves_transpose(overlay_ridge_condition=False):
    #     D_curves_transpose = D_curves.T
    #     R_curves_transpose = R_curves_neighborhood.T > 0
    #     num_rows = np.ceil(np.sqrt(len(df_ridge))).astype(int)
    #     fig, axs = plt.subplots(num_rows, num_rows, figsize=(3.5 * num_rows, 3.5 * num_rows), layout='constrained', sharey=True)

    #     titles = [f"({genomic_labels(x[0] * resolution / np.sqrt(2), N=1)}, {genomic_labels(window_size / 2 - x[1] * resolution / np.sqrt(2), N=1)})" for x in df_ridge[[x_label, y_label]].values]

    #     for i, ax in enumerate(axs.flat):
    #         if i < len(D_curves_transpose):

    #             # reverse direction
    #             j = i

    #             # Overlay ridge condition color
    #             if overlay_ridge_condition:
    #                 ax.plot(np.arange(D_curves_transpose.shape[1]), D_curves_transpose[j], color='gray', alpha=0.6)
    #                 sc = ax.scatter(np.arange(D_curves_transpose.shape[1]), D_curves_transpose[j], c=R_curves_transpose[j]) # for the colors
    #                 plt.colorbar(sc, ax=ax)
                    
    #             else:
    #                 ax.plot(np.arange(D_curves_transpose.shape[1]), D_curves_transpose[j], color='blue', marker="o")
                
    #             # Find the index of the maximum value
    #             max_index = np.argmax(D_curves_transpose[j])
    #             local_max_indices = argrelmax(D_curves_transpose[j])[0]

    #             for local_max_index in local_max_indices:
                
    #                 # Plot the local maximum point in orange
    #                 ax.plot(local_max_index, D_curves_transpose[j, local_max_index], color='orange', marker="x", markersize=8)

    #             # Plot the global maximum point in red
    #             ax.plot(max_index, D_curves_transpose[j, max_index], color='red', marker="x", markersize=8)
                

    #             # Set title and x-axis ticks
    #             ax.set_title(titles[j], fontsize=10)
    #             ax.set_xticks(np.arange(len(scale_range)))
    #             ax.set_xticklabels([f"{x:.2f}" for x in scale_range], rotation=50)

    #         else:  # Hide any unused subplots
    #             ax.axis('off')

    #     fig.suptitle(ridge_title, fontsize=12.5)

    #     save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_scale_curves.png") 

    #     plt.savefig(save_name, dpi=300)

    #     plt.close()


    # def plot_max_scale(step=3):
    #     fig, ax = plt.subplots(1, local_max.shape[1] + 2, figsize=(4 * (local_max.shape[1] + 2), 5), layout="constrained", sharex=True, sharey=True)

    #     global_colors = ["red" if x else "blue" for x in global_max_ridge_cond]
    #     ax[0].bar(np.arange(D_curves.shape[1]), global_max, facecolor='none', edgecolor=global_colors, width=0.8)
    #     ax[0].set_title("Global max")

    #     # then multiple local bar plots
    #     for j in range(local_max.shape[1]):

    #         global_colors = ["red" if x else "blue" for x in global_max_ridge_cond]
    #         ax[j + 1].bar(np.arange(D_curves.shape[1]), global_max, facecolor='none', edgecolor=global_colors, width=0.8)

            
    #         local_colors = ["red" if x else "blue" for x in local_max_ridge_cond[:, j]]
    #         ax[j + 1].bar(np.arange(D_curves.shape[1]), local_max[:, j], color=local_colors, alpha=0.8, width=0.5)     

    #         ax[j + 1].set_title(f"Local maxima {j + 1}")


    #     assigned_colors = ["red" if x else "blue" for x in assigned_s_ridge_cond]
    #     ax[-1].bar(np.arange(D_curves.shape[1]), assigned_s, color=assigned_colors, width=0.8)
    #     ax[-1].set_title("Assigned scales")    
            
    #     # Generate xticks as before
    #     xticks = [f"{genomic_labels(window_size / 2 - x[1] * resolution / np.sqrt(2), N=1)}" for x in df_ridge[[x_label, y_label]].values]
        
    #     # Create positions and labels for every 3rd tick
    #     positions = np.arange(0, D_curves.shape[1], step)
    #     xticks_filtered = xticks[::step]  # Select every 3rd label
        
    #     # Set xticks and labels
    #     ax[0].set_xticks(positions)
    #     for a in ax:
    #         a.set_xticklabels(xticks_filtered, rotation=45)

    #     ax[0].set_ylabel("Scale")
    #     ax[0].set_xlabel("Position")
    #     fig.suptitle(f"Max scale at each position | median of max: {np.median(global_max):.2f} (median of assigned: {np.nanpercentile(assigned_s, 50, method='lower'):.2f})\n{ridge_title}")

    #     save_name = os.path.join(save_path, f"rank-{rank + 1}_{genomic_labels(start_correct)}_assigned_scales.png") 

    #     plt.savefig(save_name, dpi=400)

    #     plt.close()