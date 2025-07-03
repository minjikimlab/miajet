import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from cycler import cycler
import os



def save_histogram(im, save_path, vmax_perc, vmin_perc, file_name):
    """
    Save the histogram of the image

    Parameters:
    -----------
    im : np.ndarray
        The image data for the histogram
    save_path : str
        The directory where the histogram image will be saved
    vmax_perc : float
        The upper percentile of the data shown on the histogram as a vertical line
    vmin_perc : float
        The lower percentile of the data shown on the histogram as a vertical line
    file_name : str
        The name of the file to save the histogram as

    Returns:
    --------
    None, but saves the histogram plot to the specified path
    """
    plt.figure(figsize=(10, 5))

    plt.hist(im.ravel(), bins=256, color='gray', alpha=0.7)
    plt.axvline(np.percentile(im, vmin_perc), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(im, vmax_perc), color='blue', linestyle='dashed', linewidth=1)
    plt.text(np.percentile(im, vmin_perc), 100, f'vmin: {np.percentile(im, vmin_perc):.2f}', color='red')
    plt.text(np.percentile(im, vmax_perc), 100, f'vmax: {np.percentile(im, vmax_perc):.2f}', color='blue')

    plt.title(f"Histogram of Image Intensity Values (lb: {vmin_perc}% ub: {vmax_perc}%)")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()


    
def boxplot_statistics(values):
    return f"N={len(values)} | 25th: {np.percentile(values, 25):.3g} | 50th: {np.percentile(values, 50):.3g} | 75th: {np.percentile(values, 75):.3g} | mean: {np.mean(values):.3g}"

def plot_n_signal(X, titles, suptitle, resolution, ppr=5, savepath=None, show=False, supxlabel=None, supylabel=None, figsize=None, dpi=100, num_ticks=5, show_legend=False, label=None, genomic_shift=0, **kwargs):
    """
    Plots n signals (initially defined for signals extracted over Hi-C data), but should be generalizable to any collection of signals 
        that are the same dimension
    """
    
    if isinstance(X, np.ndarray):
        if X.ndim == 2:  
            X = list(X)  
        else:
            raise ValueError("Input data X must be 2-dimensional with shape (number of signals, length)")
            
    if isinstance(X, list):
        closest_multiple = ppr - len(X) % ppr if len(X) % ppr != 0 else 0
        
        num_rows = (len(X) + closest_multiple) // ppr 
        num_cols = min(len(X), ppr)
            
        if figsize is not None:
            
            fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, layout='constrained',
                                    sharex='all', dpi=dpi)
        else:

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 3 * num_rows + 1), layout='constrained',
                                    sharex='all', dpi=dpi)

        for i, ax in enumerate(axs.flat):
            if i < len(X):
                
                if show_legend:
                    im = ax.plot(np.arange(X[i].shape[0]), X[i], label=label[i], **kwargs)
                else:
                    im = ax.plot(np.arange(X[i].shape[0]), X[i], **kwargs)

                if show_legend:
                    ax.legend()
                    
                ax.set_title(titles[i], fontsize=10)
                                
                def_xticks = np.arange(0, X[i].shape[0], np.ceil(X[i].shape[0] / num_ticks).astype(int))
                
                if isinstance(genomic_shift, list) or isinstance(genomic_shift, np.ndarray):
                    ticks_bp_x = [genomic_labels(x) for x in list(genomic_shift[i] + def_xticks * resolution)]
                else:
                    ticks_bp_x = [genomic_labels(x) for x in list(genomic_shift + def_xticks * resolution)]
                
                
                ax.set_xticks(def_xticks)
                ax.set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)
                
            else:  # Hide any unused subplots
                ax.axis('off')

        fig.suptitle(suptitle, fontsize=12.5)

        if supxlabel:
            fig.supxlabel(supxlabel)
        if supylabel:
            fig.supylabel(supylabel)

        if savepath is not None:
            plt.savefig(savepath, dpi=400)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("Please put variables `X`, `titles` into a list for one plot case")


def genomic_label_int(pos):
    """
    Convert a genomic position into an integer value after scaling:
      - If the absolute position is at least 1e6, divide by 1e6 and round.
      - Else if at least 1e3, divide by 1e3 and round.
      - Otherwise, just round the position.
      
    Returns the integer result
    """
    if np.abs(pos) >= 1e6:
        return int(round(pos / 1e6))
    elif np.abs(pos) >= 1e3:
        return int(round(pos / 1e3))
    else:
        return int(round(pos))

def genomic_labels(pos, N=0):
    if np.abs(pos) >= 1e6:
        return f"{pos / 1e6:.{N}f}Mb".rstrip("0").rstrip(".")
    elif np.abs(pos) >= 1e3:
        return f"{pos / 1e3:.{N}f}Kb".rstrip("0").rstrip(".")
    else:
        return f"{pos:.{N}f}bp".rstrip("0").rstrip(".")

def chr_num_to_string(x):
    return "chr{}".format(int(x))
    
    
def plot_hic(A, title, resolution, genomic_shift=0, savepath=None, show=True, log=False, cbar=True, figsize=(6, 5), vcenter=None, **kwargs):
    # print("Plotting Hi-C...")
    plt.close("all")
    
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    
    if log:
        if vcenter is None:
            im = ax.imshow(np.log10(A + 1), interpolation="none", **kwargs)
        else:
            im = ax.imshow(np.log10(A + 1), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter), **kwargs)

    else:
        if vcenter is None:
            im = ax.imshow(A, interpolation="none", **kwargs)
        else:
            im = ax.imshow(A, interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter), **kwargs)
        
    if cbar:
        plt.colorbar(im, ax=ax)

    if resolution is not None:
            
        def_xticks = plt.gca().get_xticks()
        def_yticks = plt.gca().get_yticks()
        
        def_xticks = np.array([x for x in def_xticks if x >= 0 and x < A.shape[0]])
        def_yticks = np.array([x for x in def_yticks if x >= 0 and x < A.shape[1]])
        
        # Add lower bound to both x and y because matrix is naturally 0-indexed
        ticks_bp_x = [genomic_labels(x) for x in list(genomic_shift + def_xticks * resolution)]
        ticks_bp_y = [genomic_labels(x) for x in list(genomic_shift + def_yticks * resolution)]
        
        plt.xticks(ticks=def_xticks, labels=ticks_bp_x, fontsize=8, rotation=30)
        plt.yticks(ticks=def_yticks, labels=ticks_bp_y, fontsize=8)

    plt.title(title, fontsize=10)
    
    if savepath is not None:
        # print("Saved to {}")
        plt.savefig(savepath, dpi=400)
    
    if show:
        plt.show()
    else:
        plt.close("all")
        
        
        

        

        
# def plot_n_hic(H, titles, suptitle, resolution, ppr=5, genomic_shift=0, savepath=None, show=False, supxlabel=None, supylabel=None, figsize=None, cmap_label="Intensity", dpi=100, num_ticks=5, show_cbar=True, vcenter=None, cmap="viridis", vmax=None, **kwargs):
#     """
#     Plots n Hi-C plots with max plots per row `ppr`, each with its own colorbar.
#         genomic_shift : starting genomic coordinates of Hi-C region,
#         either a number or a list of the number of Hi-C plots
    
#     """
    
#     if isinstance(H, np.ndarray):
#         if H.ndim == 3:  
#             H = list(H)  
#         else:
#             raise ValueError("Numpy array H must be 3-dimensional with shape (num_matrices, dim1, dim2)")
            
#     if isinstance(H, list):
#         closest_multiple = ppr - len(H) % ppr if len(H) % ppr != 0 else 0
        
#         num_rows = (len(H) + closest_multiple) // ppr 
#         num_cols = min(len(H), ppr)
            
#         if figsize is not None:
            
#             if isinstance(genomic_shift, list) or isinstance(genomic_shift, np.ndarray):
#                 fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, layout='constrained', dpi=dpi)
#             else:
#                 fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, layout='constrained',
#                                         sharex='all', sharey='all', dpi=dpi)
#         else:
#             if isinstance(genomic_shift, list) or isinstance(genomic_shift, np.ndarray):
#                 fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 3 * num_rows + 1), layout='constrained', dpi=dpi)
#             else:
#                 fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 3 * num_rows + 1), layout='constrained',
#                                         sharex='all', sharey='all', dpi=dpi)

#         for i, ax in enumerate(axs.flat):
#             if i < len(H):
                
#                 if isinstance(vmax, np.ndarray) or isinstance(vmax, list):
                    
#                     if isinstance(cmap, np.ndarray) or isinstance(cmap, list):
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], vmax=vmax[i], **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=vmax[i]), cmap=cmap[i], **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], vmax=vmax[i], **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=vmax[i]), cmap=cmap[i], **kwargs)
#                     else:
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap,vmax=vmax[i],  **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=vmax[i]), cmap=cmap, **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap, vmax=vmax[i], **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=vmax[i]), cmap=cmap, **kwargs)                    
                    
#                 elif vmax is not None:
#                     # should be a number
#                     # let's NOT interpret it as a percentile

#                     if isinstance(cmap, np.ndarray) or isinstance(cmap, list):
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], vmax=vmax, **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=vmax), cmap=cmap[i], **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], vmax=vmax, **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=vmax), cmap=cmap[i], **kwargs)
#                     else:
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap,vmax=vmax,  **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=vmax), cmap=cmap, **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap, vmax=vmax, **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=vmax), cmap=cmap, **kwargs)
                        
#                 else:


#                     if isinstance(cmap, np.ndarray) or isinstance(cmap, list):
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i]), cmap=cmap[i], **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap[i], **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter), cmap=cmap[i], **kwargs)
#                     else:
#                         if vcenter is None:
#                             im = ax.imshow(H[i], interpolation="none", cmap=cmap, **kwargs)
#                         elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
#                             if vcenter[i] is not None:
#                                 im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i]), cmap=cmap, **kwargs)
#                             else:
#                                 im = ax.imshow(H[i], interpolation="none", cmap=cmap, **kwargs)
#                         else:
#                             im = ax.imshow(H[i], interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter), cmap=cmap, **kwargs)
                    
                
#                 if show_cbar:
#                     cbar = fig.colorbar(im, ax=ax)  # Add a colorbar to each subplot
#                     cbar.set_label(cmap_label, rotation=270)
                
#                 ax.set_title(titles[i], fontsize=10)
                                
#                 def_xticks = np.arange(0, H[i].shape[1], np.ceil(H[i].shape[1] / num_ticks).astype(int))
#                 def_yticks = np.arange(0, H[i].shape[0], np.ceil(H[i].shape[0] / num_ticks).astype(int))
                
#                 if isinstance(genomic_shift, list) or isinstance(genomic_shift, np.ndarray):
#                     ticks_bp_x = [genomic_labels(x, N=1) for x in list(genomic_shift[i] + def_xticks * resolution)]
#                     ticks_bp_y = [genomic_labels(x, N=1) for x in list(genomic_shift[i] + def_yticks * resolution)]
#                 else:
#                     ticks_bp_x = [genomic_labels(x, N=1) for x in list(genomic_shift + def_xticks * resolution)]
#                     ticks_bp_y = [genomic_labels(x, N=1) for x in list(genomic_shift + def_yticks * resolution)]
                
                
#                 ax.set_xticks(def_xticks)
#                 ax.set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)
                
#                 ax.set_yticks(def_yticks)
#                 ax.set_yticklabels(ticks_bp_y, fontsize=8)
                
#             else:  # Hide any unused subplots
#                 ax.axis('off')

#         fig.suptitle(suptitle, fontsize=12.5)

#         if supxlabel:
#             fig.supxlabel(supxlabel)
#         if supylabel:
#             fig.supylabel(supylabel)

#         if savepath is not None:
#             plt.savefig(savepath, dpi=400)
        
#         if show:
#             plt.show()
#         else:
#             plt.close(fig)
#     else:
#         print("Please put variables `H`, `titles` into a list for one plot case. Or use `plot_hic`")


def plot_n_hic(
    H,
    titles,
    suptitle,
    resolution,
    ppr=5,
    genomic_shift=0,
    savepath=None,
    show=False,
    supxlabel=None,
    supylabel=None,
    figsize=None,
    cmap_label="Intensity",
    dpi=100,
    num_ticks=5,
    show_cbar=True,
    vcenter=None,
    cmap="Reds",
    vmax=None,
    standardize_cbar=False,
    **kwargs,
):
    """Plot *n* Hi-C matrices in a compact grid.

    If ``standardize_cbar`` is *True*, all panels share a single colour scale
    (common *vmin*/*vmax*) **and** a single colourbar.  When a non-``None``
    ``vcenter`` is supplied, the shared scale is implemented with
    :class:`matplotlib.colors.TwoSlopeNorm`, so the zero (or other specified
    centre) is aligned and symmetric across every image and the global
    colourbar.

    The public API is unchanged apart from the new ``standardize_cbar`` flag; a
    value of *False* reproduces the original behaviour exactly.
    """

    # ------------------------------------------------------------------
    # Validate / normalise input array(s)
    # ------------------------------------------------------------------
    if isinstance(H, np.ndarray):
        if H.ndim == 3:
            H = list(H)
        else:
            raise ValueError("Numpy array H must be 3-dim with shape (n, m, m)")

    if not isinstance(H, list):
        raise TypeError("H must be a list or 3-D numpy array")

    n_mats = len(H)

    # ------------------------------------------------------------------
    # Determine global limits when a shared colourbar is requested
    # ------------------------------------------------------------------
    global_vmin = kwargs.get("vmin", None)
    global_vmax = None

    if standardize_cbar:
        # -- vmax --------------------------------------------------------
        if vmax is None:
            global_vmax = max(np.nanmax(h) for h in H)
        else:
            global_vmax = max(vmax) if isinstance(vmax, (list, np.ndarray)) else vmax
        # -- vmin --------------------------------------------------------
        if global_vmin is None:
            global_vmin = min(np.nanmin(h) for h in H)
    # (If standardize_cbar is False we leave vmin/vmax handling to per-panel logic)

    # ------------------------------------------------------------------
    # Figure / grid layout
    # ------------------------------------------------------------------
    closest_multiple = ppr - n_mats % ppr if n_mats % ppr else 0
    num_rows = (n_mats + closest_multiple) // ppr
    num_cols = min(n_mats, ppr)

    share_xy = not isinstance(genomic_shift, (list, np.ndarray))
    fig_size = figsize if figsize is not None else (4.2 * num_cols, 3 * num_rows + 1)

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=fig_size,
        layout="constrained",
        sharex="all" if share_xy else False,
        sharey="all" if share_xy else False,
        dpi=dpi,
        squeeze=False,
    )

    first_im = None  # reference image for a potential global colourbar

    # ------------------------------------------------------------------
    # Helper for kwargs without vmin/vmax (to avoid conflicts with `norm`)
    # ------------------------------------------------------------------
    def _kwargs_without_limits(src):
        return {k: v for k, v in src.items() if k not in {"vmin", "vmax"}}

    # ------------------------------------------------------------------
    # Main plotting loop
    # ------------------------------------------------------------------
    for i, ax in enumerate(axs.flat):
        if i >= n_mats:
            ax.axis("off")
            continue

        # Panel-specific parameters (scalar or sequence allowed)
        res_i = resolution[i] if isinstance(resolution, (list, np.ndarray)) else resolution
        cmap_i = cmap[i] if isinstance(cmap, (list, np.ndarray)) else cmap
        vmax_i = vmax[i] if isinstance(vmax, (list, np.ndarray)) else vmax
        vcenter_i = vcenter[i] if isinstance(vcenter, (list, np.ndarray)) else vcenter

        # If using a shared scale, override per-panel limits
        if standardize_cbar:
            vmax_i = global_vmax
            vmin_i = global_vmin
        else:
            vmin_i = kwargs.get("vmin", None)

        # ------------------------------------------------------------
        # Draw image (handling centred vs. linear colour scales)
        # ------------------------------------------------------------
        if vcenter_i is None:
            im = ax.imshow(
                H[i],
                interpolation="none",
                cmap=cmap_i,
                vmax=vmax_i,
                vmin=vmin_i,
                **_kwargs_without_limits(kwargs),
            )
        else:
            # Use TwoSlopeNorm for a symmetric / centred scale.  Remove any
            # vmin/vmax from kwargs to avoid conflicts with `norm`.
            norm = colors.TwoSlopeNorm(
                vcenter=vcenter_i,
                vmax=vmax_i,
                vmin=vmin_i,
            )
            im = ax.imshow(
                H[i],
                interpolation="none",
                cmap=cmap_i,
                norm=norm,
                **_kwargs_without_limits(kwargs),
            )

        # Keep first image handle for a potential global colourbar
        if first_im is None:
            first_im = im

        # Individual colourbar unless a shared one is requested
        if show_cbar and not standardize_cbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cmap_label, rotation=270)

        # ------------------------------------------------------------
        # Axis labelling
        # ------------------------------------------------------------
        ax.set_title(titles[i], fontsize=10)

        def_xticks = np.arange(0, H[i].shape[1], np.ceil(H[i].shape[1] / num_ticks).astype(int))
        def_yticks = np.arange(0, H[i].shape[0], np.ceil(H[i].shape[0] / num_ticks).astype(int))

        gshift = genomic_shift[i] if isinstance(genomic_shift, (list, np.ndarray)) else genomic_shift

        if resolution is not None:
            ticks_bp_x = [genomic_labels(gshift + x * res_i, N=1) for x in def_xticks]
            ticks_bp_y = [genomic_labels(gshift + y * res_i, N=1) for y in def_yticks]

            ax.set_xticks(def_xticks)
            ax.set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)
            ax.set_yticks(def_yticks)
            ax.set_yticklabels(ticks_bp_y, fontsize=8)

    # ------------------------------------------------------------------
    # Draw a single shared colourbar if requested
    # ------------------------------------------------------------------
    if show_cbar and standardize_cbar and first_im is not None:
        cbar = fig.colorbar(first_im, ax=axs.ravel().tolist(), shrink=0.6)
        cbar.set_label(cmap_label, rotation=270)

    # ------------------------------------------------------------------
    # Figure-level titles / labels & I/O
    # ------------------------------------------------------------------
    fig.suptitle(suptitle, fontsize=12.5)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    if savepath:
        plt.savefig(savepath, dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)






def plot_n_rect_chunks(H, titles, suptitle, resolution, savepath=None, show=False, supxlabel=None, supylabel=None, figsize=None, cmap_label="Intensity", dpi=100, num_ticks=[5, 50],
                show_cbar=True, vcenter=None, cmap="viridis", vmax=None, genomic_shift=0, lines=None, line_colors=["blue", "cyan", "lime"], line_labels=None, show_legend=False, ppr=1, dp=1, chunk_size=None, **kwargs):
    """
    Plots n rectangle Hi-C plots with 1 plot per row, supporting chunking along the columns.

    Parameters:
    - H: list or numpy array of 2D arrays to plot.
    - titles: list of titles for each image.
    - suptitle: overall title for the figure.
    - resolution: resolution of the data.
    - savepath: base path to save the figures.
    - chunk_size: number of columns per chunk.
    - Other parameters are similar to the original function.
    """

    # Ensure H is a list of images
    if isinstance(H, np.ndarray):
        if H.ndim == 3:
            H = list(H)
        else:
            raise ValueError("Numpy array H must be 3-dimensional with shape (num_matrices, dim1, dim2)")

    for i, H_i in enumerate(H):
        num_cols = H_i.shape[1]

        # Determine chunking
        if chunk_size is None or chunk_size >= num_cols:
            num_chunks = 1
            chunks = [(0, num_cols)]
        else:
            num_chunks = int(np.ceil(num_cols / chunk_size))
            chunks = []
            for chunk_num in range(num_chunks):
                start_col = chunk_num * chunk_size
                end_col = min(start_col + chunk_size, num_cols)
                chunks.append((start_col, end_col))

        # Get parameters per image
        vmax_i = vmax[i] if isinstance(vmax, (list, np.ndarray)) else vmax
        cmap_i = cmap[i] if isinstance(cmap, (list, np.ndarray)) else cmap
        vcenter_i = vcenter[i] if isinstance(vcenter, (list, np.ndarray)) else vcenter
        genomic_shift_i = genomic_shift[i] if isinstance(genomic_shift, (list, np.ndarray)) else genomic_shift
        lines_i = lines[i] if isinstance(lines, list) and isinstance(lines[0], list) else lines
        line_labels_i = line_labels[i] if line_labels is not None and isinstance(line_labels[0], list) else line_labels

        for chunk_num, (start_col, end_col) in enumerate(chunks):
            H_chunk = H_i[:, start_col:end_col]

            # Adjust genomic_shift for the chunk
            genomic_shift_chunk = genomic_shift_i + (start_col * resolution / np.sqrt(2))

            # Adjust lines for the chunk
            lines_chunk = []
            if lines_i is not None:
                for j, line in enumerate(lines_i):
                    adjusted_line = line.copy()
                    adjusted_line[:, 0] -= start_col
                    x_in_chunk = np.logical_and(adjusted_line[:, 0] >= 0, adjusted_line[:, 0] <= (end_col - start_col))
                    if np.any(x_in_chunk):
                        adjusted_line = adjusted_line[x_in_chunk]
                        lines_chunk.append(adjusted_line)
            else:
                lines_chunk = None

            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

            # Plot H_chunk
            if vcenter_i is None:
                im = ax.imshow(np.flipud(H_chunk), interpolation="none", cmap=cmap_i, origin="lower", vmax=vmax_i, **kwargs)
            else:
                im = ax.imshow(np.flipud(H_chunk), interpolation="none",
                               norm=colors.TwoSlopeNorm(vcenter=vcenter_i, vmin=None, vmax=vmax_i),
                               cmap=cmap_i, origin="lower", **kwargs)

            # Add colorbar
            if show_cbar:
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(cmap_label, rotation=270)

            # Plot adjusted lines
            if lines_chunk:
                ax.set_prop_cycle(cycler('color', line_colors))
                for j, line in enumerate(lines_chunk):
                    label = line_labels_i[j] if line_labels_i is not None else None
                    ax.plot(line[:, 0], line[:, 1], linewidth=1.2, label=label, alpha=0.7)
                if show_legend:
                    ax.legend(fontsize=6)

            # Set title and ticks
            if chunk_size is not None:
                ax.set_title(f"{titles[i]} - Chunk {chunk_num + 1}", fontsize=10)
            else:
                ax.set_title(f"{titles[i]}", fontsize=10)
                
            set_genomic_ticks(ax, num_ticks, resolution / np.sqrt(2), H_chunk.shape, [0, genomic_shift_chunk], dp)
            ax.autoscale(False)

            # Set super titles and labels
            fig.suptitle(suptitle, fontsize=12.5)
            if supxlabel:
                fig.supxlabel(supxlabel)
            if supylabel:
                fig.supylabel(supylabel)

            # Adjust savepath
            if savepath is not None:
                base, ext = os.path.splitext(savepath)
                savepath_chunk = f"{base}-image-{i + 1}-chunk-{chunk_num + 1}{ext}"
                plt.savefig(savepath_chunk, dpi=400, bbox_inches='tight', pad_inches=0)

            if show:
                plt.show()
            else:
                plt.close(fig)


from matplotlib.collections import LineCollection


def plot_n_rect(H, titles, suptitle, resolution, savepath=None, show=False, supxlabel=None, supylabel=None, figsize=None, cmap_label="Intensity", dpi=100, num_ticks=[5, 50], 
                show_cbar=True, vcenter=None, cmap="viridis", vmax=None, genomic_shift=0, lines=None, line_colors=["blue", "cyan", "lime"], line_labels=None, line_widths=None, 
                show_legend=False, ppr=1, dp=1, **kwargs):
    """
    Plots n rectagnle Hi-C plots with 1 plot per row
    """
    
    if isinstance(H, np.ndarray):
        if H.ndim == 3:  
            H = list(H)  
        else:
            raise ValueError("Numpy array H must be 3-dimensional with shape (num_matrices, dim1, dim2)")
            
    if isinstance(H, list):
            
        if figsize is not None:
            if genomic_shift != 0 and not isinstance(genomic_shift[0], (int, float)):
                fig, axs = plt.subplots(np.ceil(len(H) / ppr).astype(int), ppr, figsize=figsize, layout='constrained',
                                        sharey='all', dpi=dpi)
            else:
                fig, axs = plt.subplots(np.ceil(len(H) / ppr).astype(int), ppr, figsize=figsize, layout='constrained',
                                        sharex='all', sharey='all', dpi=dpi)
        else:
            if genomic_shift != 0 and not isinstance(genomic_shift[0], (int, float)):
                fig, axs = plt.subplots(np.ceil(len(H) / ppr).astype(int), ppr, figsize=(36, 2 * len(H)), layout='constrained',
                                        sharey='all', dpi=dpi)
            else:
                fig, axs = plt.subplots(np.ceil(len(H) / ppr).astype(int), ppr, figsize=(36, 2 * len(H)), layout='constrained',
                                        sharex='all', sharey='all', dpi=dpi)
                
        # Ensure axs is iterable by wrapping a single Axes into an array
        if not hasattr(axs, "flat"):
            axs = np.array([axs])

        for i, ax in enumerate(axs.flat):
            
            ax.set_prop_cycle(cycler('color', line_colors))

            if i < len(H):
                
                if isinstance(vmax, np.ndarray) or isinstance(vmax, list):
                    
                    if isinstance(cmap, np.ndarray) or isinstance(cmap, list):
                        if vcenter is None:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap[i], origin="lower", vmax=vmax[i], **kwargs)
                        elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
                            if vcenter[i] is not None:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmin=None, vmax=vmax[i]), cmap=cmap[i], origin="lower",**kwargs)
                            else:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap[i], origin="lower",  vmax=vmax[i], **kwargs)
                        else:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmin=None, vmax=vmax[i]), cmap=cmap[i], origin="lower",**kwargs)
                    else:
                        if vcenter is None:
                            im =ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap, origin="lower", vmax=vmax[i], **kwargs)
                        elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
                            if vcenter[i] is not None:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmin=None, vmax=vmax[i]), cmap=cmap, origin="lower",**kwargs)
                            else:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap, origin="lower",  vmax=vmax[i],**kwargs)
                        else:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmin=None, vmax=vmax[i]), cmap=cmap, origin="lower",**kwargs)                    
                    
                    
                else:

                    if vmax is None:
                        vmax_in = 100
                    else:
                        vmax_in = vmax
                

                    if isinstance(cmap, np.ndarray) or isinstance(cmap, list):
                        if vcenter is None:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap[i], origin="lower", vmax=np.percentile(H[i], q=vmax_in), **kwargs)
                        elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
                            if vcenter[i] is not None:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=np.percentile(H[i], q=vmax_in)), cmap=cmap[i], origin="lower", **kwargs)
                            else:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap[i], origin="lower", vmax=np.percentile(H[i], q=vmax_in), **kwargs)
                        else:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=np.percentile(H[i], q=vmax_in)), cmap=cmap[i], origin="lower", **kwargs)
                    else:
                        if vcenter is None:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap, origin="lower", vmax=np.percentile(H[i], q=vmax_in), **kwargs)
                        elif isinstance(vcenter, np.ndarray) or isinstance(vcenter, list):
                            if vcenter[i] is not None:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter[i], vmax=np.percentile(H[i], q=vmax_in)), cmap=cmap, origin="lower", **kwargs)
                            else:
                                im = ax.imshow(np.flipud(H[i]), interpolation="none", cmap=cmap, origin="lower", vmax=np.percentile(H[i], q=vmax_in), **kwargs)
                        else:
                            im = ax.imshow(np.flipud(H[i]), interpolation="none", norm=colors.TwoSlopeNorm(vcenter=vcenter, vmax=np.percentile(H[i], q=vmax_in)), cmap=cmap, origin="lower", **kwargs)
                    
                
                if show_cbar:
                    cbar = fig.colorbar(im, ax=ax)  # Add a colorbar to each subplot
                    cbar.set_label(cmap_label, rotation=270)
                    # cbar.ax.tick_params(rotation=45)
                    
                if lines is not None:
                    
                    if isinstance(lines, list):
                        
                        if lines[i] is not None:
                            
                            for j, line in enumerate(lines[i]):
                                label = line_labels[i][j] if line_labels is not None and line_labels[i] is not None else None
                                # Plot the primary line with a fixed, thin linewidth and default color cycle
                                ax.plot(line[:, 0], line[:, 1], linewidth=1, label=label, alpha=0.7)
                                # If variable widths are provided, overlay a LineCollection with the variable widths
                                if line_widths is not None:
                                    if isinstance(line_widths, list) and line_widths[i] is not None:
                                        points = line[::2].reshape(-1, 1, 2) # sample every other point and reshape
                                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                                        # Assume line_widths[i][j] is a vector of widths for each segment;
                                        # if not provided for a segment, default to 1 for that segment.
                                        if line_widths[i][j] is not None:
                                            lw_1_min = np.clip(line_widths[i][j], a_min=1, a_max=None) # minimum is 1
                                            lc = LineCollection(segments, linewidths=lw_1_min, color='gray', alpha=0.5)
                                            ax.add_collection(lc)
                                
                    else:
                        for j, line in enumerate(lines):
                            label = line_labels[j] if line_labels is not None else None
                            ax.plot(line[:, 0], line[:, 1], linewidth=1, label=label, alpha=0.7)
                            if line_widths is not None:
                                points = line[::2].reshape(-1, 1, 2)
                                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                                if line_widths[j] is not None:
                                    lw_1_min = np.clip(line_widths[j], a_min=1, a_max=None)
                                    lc = LineCollection(segments, linewidths=lw_1_min, color='gray', alpha=0.5)
                                    ax.add_collection(lc)
                        
                    if line_labels is not None:
                        if line_labels[i] is not None:
                            if show_legend:
                                ax.legend(fontsize=6)
                
                ax.set_title(titles[i], fontsize=10)
                                
                if genomic_shift != 0 and not isinstance(genomic_shift[0], (int, float)):
                    
                    set_genomic_ticks(ax, num_ticks, resolution / np.sqrt(2), H[0].shape, genomic_shift[i], dp)
                else:
                    set_genomic_ticks(ax, num_ticks, resolution / np.sqrt(2), H[0].shape, genomic_shift, dp)
                
                ax.autoscale(False)                
                
            else:  # Hide any unused subplots
                ax.axis('off')

        fig.suptitle(suptitle, fontsize=12.5, )

        if supxlabel:
            fig.supxlabel(supxlabel)
        if supylabel:
            fig.supylabel(supylabel)

        if savepath is not None:
            fig = plt.gcf()  # get the current figure
            fig_width = fig.get_size_inches()[0]
            max_allowed_dpi = 65536 / fig_width
            dpi = min(300, max_allowed_dpi / 2) 
            plt.savefig(savepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("Please put variables `H`, `titles` into a list for one plot case. Or use `plot_hic`") 
        
        
# helper functions for plotting
from matplotlib.ticker import EngFormatter
bp_formatter = EngFormatter('b')
def format_ticks(ax, x=True, y=True, rotate=True):
    """
    Format ticks with genomic coordinates as human readable
    From cooltools
    """
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)


        
def set_genomic_ticks(ax, num_ticks, resolution, mat_shape, genomic_shift, dp=1):
    
    if isinstance(num_ticks, np.ndarray) or isinstance(num_ticks, list):
        step_size_x = np.ceil(mat_shape[1] / num_ticks[1]).astype(int)
        step_size_y = np.ceil(mat_shape[0] / num_ticks[0]).astype(int)
        
        def_xticks = np.arange(0, mat_shape[1] + step_size_x, step_size_x)
        def_yticks = np.arange(0, mat_shape[0] + step_size_y, step_size_y)        
    else:
        step_size_x = np.ceil(mat_shape[1] / num_ticks).astype(int)
        step_size_y = np.ceil(mat_shape[0] / num_ticks).astype(int)
        
        def_xticks = np.arange(0, mat_shape[1] + step_size_x, step_size_x)
        def_yticks = np.arange(0, mat_shape[0] + step_size_y, step_size_y)

    if isinstance(resolution, np.ndarray) or isinstance(resolution, list):
        
        if isinstance(genomic_shift, np.ndarray) or isinstance(genomic_shift, list):
            ticks_bp_x = [genomic_labels(x, dp) for x in list(genomic_shift[1] + def_xticks * resolution[1])]
            ticks_bp_y = [genomic_labels(x, dp) for x in list(genomic_shift[0] + def_yticks * resolution[0])]
        else:
            ticks_bp_x = [genomic_labels(x, dp) for x in list(genomic_shift + def_xticks * resolution[1])]
            ticks_bp_y = [genomic_labels(x, dp) for x in list(genomic_shift + def_yticks * resolution[0])]
            
    else:
        if isinstance(genomic_shift, np.ndarray) or isinstance(genomic_shift, list):
            ticks_bp_x = [genomic_labels(x, dp) for x in list(genomic_shift[1] + def_xticks * resolution)]
            ticks_bp_y = [genomic_labels(x, dp) for x in list(genomic_shift[0] + def_yticks * resolution)]
        else:
            ticks_bp_x = [genomic_labels(x, dp) for x in list(genomic_shift + def_xticks * resolution)]
            ticks_bp_y = [genomic_labels(x, dp) for x in list(genomic_shift + def_yticks * resolution)]
        
    ax.set_xticks(def_xticks)
    ax.set_xticklabels(ticks_bp_x, fontsize=8, rotation=45)
    ax.set_yticks(def_yticks)
    ax.set_yticklabels(ticks_bp_y, fontsize=8)

    return 
   

def convert_imagej_coord_to_numpy(coords, window_size_bin, flip_y, start_bin=0):
    """
    Converts coordinates of rotated contact map (i.e. coordinates provided by ImageJ)
    to indices of numpy array representation of the contact map

    First it ensures that coordinates are within bounds of the contact map
    Then it flips the y-coordinates if `flip_y` is set to True.
    This is used in plotting situations when the origin is in the bottom left corner

    However, for indexing purposes (i.e. indexing scale space tensors with `coords`)
    keep `flip_y` as False.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates in the format [[x1, y1], [x2, y2], ...]
        where x is the horizontal axis and y is the vertical axis
    window_size_bin : int
        Binned window size (i.e. height of the rotated contact map)
    flip_y : bool
        Whether to flip the y-coordinates (i.e. invert the vertical axis)
        Set this to True if plotting ridge lines `coords` on a contact map 
        Set this to False if indexing scale space tensors with `coords`
    start_bin : int, optional
        Offset for the x-coordinates, by default 0
        This is useful when plotting chunks of the contact map.
        For example, if you are plotting a chunk starting from `start_bin`, 
        then you don't want the coords to be global coordinates, but rather local to the chunk

    Returns
    -------
    ridge_coords : np.ndarray
        Coordinates in the format [[x1, y1], [x2, y2], ...]
        where x is the horizontal axis and y is the vertical axis
    """
    ridge_coords = np.copy(coords)
    # Fix out of bounds
    ridge_coords[:, 1][ridge_coords[:, 1] >= window_size_bin] = window_size_bin - 1
    
    if flip_y:
        # Invert
        ridge_coords[:, 1] = window_size_bin - 1 - ridge_coords[:, 1]
    
    if start_bin:
        # Offset x position
        ridge_coords[:, 0] -= start_bin # need to offset x coordinates since we're plotting chunks
    
    return ridge_coords

from scipy.stats import ks_2samp, wilcoxon, kstest, ttest_rel


def plot_p_value_observed_null(im_p_value, corr_im_p_value, ridge_points, ridge_widths,
                     center_box_coords, right_box_coords, left_box_coords,
                     center_num_points, right_num_points, left_num_points,
                     center_vals, right_vals, left_vals,
                     center_means, right_means, left_means,
                     mean_CR_subtract, mean_CR_ratio, mean_C2R_ratio,
                     fig_suptitle="", save_path=None, start_bin=0):
    
    # Create figure and axes
    fig, ax = plt.subplots(5, 6, figsize=(33, 16), layout="constrained", height_ratios=[3, 1, 1, 1, 1])
    fig.suptitle(fig_suptitle)    
    
    # --- First row: Original image and ridge points ---

    # OBSERVED
    imcm = ax[0, 0].imshow(im_p_value, cmap='Reds')
    ax[0, 0].set_title("Original Log Obs")
    plt.colorbar(imcm, ax=ax[0, 0])

    # Image with ridge points
    imcm = ax[0, 1].imshow(im_p_value, cmap='Reds')
    ax[0, 1].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    ax[0, 1].set_title("With Ridge Points")
    ax[0, 1].legend()
    plt.colorbar(imcm, ax=ax[0, 1])

    # Image with ridge points and boxes
    imcm = ax[0, 2].imshow(im_p_value, cmap='Reds')
    ax[0, 2].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    plt.colorbar(imcm, ax=ax[0, 2])


    # NULL MODEL
    imcm = ax[0, 3].imshow(corr_im_p_value, cmap='Reds')
    ax[0, 3].set_title("Correlation Log Obs")
    plt.colorbar(imcm, ax=ax[0, 3])

    # Image with ridge points
    imcm = ax[0, 4].imshow(corr_im_p_value, cmap='Reds')
    ax[0, 4].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    ax[0, 4].set_title("With Ridge Points")
    ax[0, 4].legend()
    plt.colorbar(imcm, ax=ax[0, 4])

    # Image with ridge points and boxes
    imcm = ax[0, 5].imshow(corr_im_p_value, cmap='Reds')
    ax[0, 5].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    plt.colorbar(imcm, ax=ax[0, 5])        



    # Center boxes
    for i, coords in enumerate(center_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75,
                          marker=None, label='Center Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75, marker=None)
    
    # Right boxes
    for i, coords in enumerate(right_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Right Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    # Left boxes
    for i, coords in enumerate(left_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Left Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    ax[0, 2].set_title("With Ridge Points and Boxes")
    ax[0, 2].legend()


    # Center boxes
    for i, coords in enumerate(center_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75,
                          marker=None, label='Center Box')
        else:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75, marker=None)
    
    # Right boxes
    for i, coords in enumerate(right_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Right Box')
        else:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    # Left boxes
    for i, coords in enumerate(left_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Left Box')
        else:
            ax[0, 5].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    ax[0, 5].set_title("With Ridge Points and Boxes")
    ax[0, 5].legend()

    # --- Second row: Boxplot, histogram and ECDF of intensity values ---
    # Boxplot
    ax[1, 0].plot(ridge_widths, "-o")
    ax[1, 0].set_title("Ridge Widths")
    ax[1, 0].set_ylabel("Intensity")
    ax[1, 0].set_xlabel("Ridge Point Index")
    # Boxplot
    ax[1, 3].plot(ridge_widths, "-o")
    ax[1, 3].set_title("Ridge Widths")
    ax[1, 3].set_ylabel("Intensity")
    ax[1, 3].set_xlabel("Ridge Point Index")

    # Histogram (PMF)
    ax[1, 1].hist(np.concatenate(center_vals[0]), bins=20, color="cyan", alpha=0.5, label="Center Box")
    ax[1, 1].hist(np.concatenate(right_vals[0]), bins=20, color="green", alpha=0.5, label="Right Box")
    ax[1, 1].hist(np.concatenate(left_vals[0]), bins=20, color="blue", alpha=0.5, label="Left Box")
    ax[1, 1].legend()
    ax[1, 1].set_title("Histogram of Intensity Values")
    ax[1, 1].set_ylabel("Frequency")
    ax[1, 1].set_xlabel("Intensity") 

    # Histogram (PMF)
    ax[1, 4].hist(np.concatenate(center_vals[1]), bins=20, color="cyan", alpha=0.5, label="Center Box")
    ax[1, 4].hist(np.concatenate(right_vals[1]), bins=20, color="green", alpha=0.5, label="Right Box")
    ax[1, 4].hist(np.concatenate(left_vals[1]), bins=20, color="blue", alpha=0.5, label="Left Box")
    ax[1, 4].legend()
    ax[1, 4].set_title("Histogram of Intensity Values")
    ax[1, 4].set_ylabel("Frequency")
    ax[1, 4].set_xlabel("Intensity") 

    # Cumulative distribution (ECDF)
    ax[1, 2].ecdf(np.concatenate(center_vals[0]), color="cyan", label="Center Box")
    ax[1, 2].ecdf(np.concatenate(right_vals[0]), color="green", label="Right Box")
    ax[1, 2].ecdf(np.concatenate(left_vals[0]), color="blue", label="Left Box")
    ax[1, 2].legend()
    ax[1, 2].set_title("Cumulative Histogram of Intensity Values")
    ax[1, 2].set_ylabel("Cumulative Frequency")
    ax[1, 2].set_xlabel("Intensity")

    # Cumulative distribution (ECDF)
    ax[1, 5].ecdf(np.concatenate(center_vals[1]), color="cyan", label="Center Box")
    ax[1, 5].ecdf(np.concatenate(right_vals[1]), color="green", label="Right Box")
    ax[1, 5].ecdf(np.concatenate(left_vals[1]), color="blue", label="Left Box")
    ax[1, 5].legend()
    ax[1, 5].set_title("Cumulative Histogram of Intensity Values")
    ax[1, 5].set_ylabel("Cumulative Frequency")
    ax[1, 5].set_xlabel("Intensity")

    # --- Third row:  Mean or Median intensity values  ---
    for i, offset in enumerate([0, 3]):
        ax[2, 0+offset].plot(center_means[i], '-o', color="cyan", label="Center")
        ax[2, 0+offset].plot(left_means[i], '-o', color="blue", label="Left")
        ax[2, 0+offset].plot(right_means[i], '-o', color="green", label="Right")
        ax[2, 0+offset].set_title("Comparison of Mean (or Median) Intensity")
        ax[2, 0+offset].set_ylabel("Intensity")
        ax[2, 0+offset].set_xlabel("Ridge Point Index")
        ax[2, 0+offset].legend()

        ax[2, 1+offset].hist(center_means[i], bins=20, color="cyan", alpha=0.5, label="Center Box")
        ax[2, 1+offset].hist(right_means[i], bins=20, color="green", alpha=0.5, label="Right Box")
        ax[2, 1+offset].hist(left_means[i], bins=20, color="blue", alpha=0.5, label="Left Box")
        ax[2, 1+offset].legend()
        ax[2, 1+offset].set_title("Histogram of Mean (or Median) Intensity Values")
        ax[2, 1+offset].set_ylabel("Frequency")
        ax[2, 1+offset].set_xlabel("Intensity")
        
        ax[2, 2+offset].ecdf(center_means[i], color="cyan", label="Center Box")
        ax[2, 2+offset].ecdf(right_means[i], color="green", label="Right Box")
        ax[2, 2+offset].ecdf(left_means[i], color="blue", label="Left Box")
        ax[2, 2+offset].legend()
        ax[2, 2+offset].set_title("Cumulative Histogram of Mean (or Median) Intensity Values")
        ax[2, 2+offset].set_ylabel("Cumulative Frequency")
        ax[2, 2+offset].set_xlabel("Intensity")


    # --- Fourth row: Number of Points in boxes and Mean test statistic plots ---
    ax[3, 1].plot(center_num_points, '-o', color="cyan")
    ax[3, 1].set_title("Center Box Number of Points")
    ax[3, 1].set_ylabel("Number of Points")
    ax[3, 1].set_xlabel("Ridge Point Index")
    
    ax[3, 2].plot(right_num_points, '-o', color="green")
    ax[3, 2].set_title("Right Box Number of Points")
    ax[3, 2].set_ylabel("Number of Points")
    ax[3, 2].set_xlabel("Ridge Point Index")
    
    ax[3, 0].plot(left_num_points, '-o', color="blue")
    ax[3, 0].set_title("Left Box Number of Points")
    ax[3, 0].set_ylabel("Number of Points")
    ax[3, 0].set_xlabel("Ridge Point Index")

    ax[3, 3].plot(mean_CR_subtract[0], '-o', color="magenta", label="Observed")
    ax[3, 3].plot(mean_CR_subtract[1], '-o', color="black", label="Null")
    ax[3, 3].legend()
    ax[3, 3].set_title("C - avg(L, R)")
    ax[3, 3].set_xlabel("Ridge Point Index")
    
    ax[3, 4].plot(mean_C2R_ratio[0], '-o', color="magenta", label="Observed")
    ax[3, 4].plot(mean_C2R_ratio[1], '-o', color="black", label="Null")
    ax[3, 4].legend()
    ax[3, 4].set_title("C^2 / avg(L, R)")
    ax[3, 4].set_xlabel("Ridge Point Index")
    
    ax[3, 5].plot(mean_CR_ratio[0], '-o', color="magenta", label="Observed")
    ax[3, 5].plot(mean_CR_ratio[1], '-o', color="black", label="Null")
    ax[3, 5].legend()
    ax[3, 5].set_title("C / avg(L, R)")
    ax[3, 5].set_xlabel("Ridge Point Index")


    # --- Fifth row: Ratio of observed and null test statistic and ECDF ---
    ax[4, 0].plot(mean_CR_subtract[0] / mean_CR_subtract[1], '-o')
    ax[4, 0].set_title("C - avg(L, R) Ratio")
    ax[4, 0].set_xlabel("Ridge Point Index")
    ax[4, 1].plot(mean_C2R_ratio[0] / mean_C2R_ratio[1], '-o')
    ax[4, 1].set_title("C^2 / avg(L, R) Ratio")
    ax[4, 1].set_xlabel("Ridge Point Index")
    ax[4, 2].plot(mean_CR_ratio[0] / mean_CR_ratio[1], '-o')
    ax[4, 2].set_title("C / avg(L, R) Ratio")

    ks_stat, ks_p_value = ks_2samp(mean_CR_subtract[0], mean_CR_subtract[1], nan_policy='omit', alternative='less')
    _, w_p_value = wilcoxon(mean_CR_subtract[0] - mean_CR_subtract[1], nan_policy='omit', alternative='greater')
    # _, ks1_p_value = kstest(mean_CR_subtract[0] - mean_CR_subtract[1], lambda x : np.where(x < 0, 0.0, 1.0), alternative='less', nan_policy='omit')
    _, t_p_val = ttest_rel(mean_CR_subtract[0], mean_CR_subtract[1], alternative='greater', nan_policy='omit') 
    ax[4, 3].ecdf(mean_CR_subtract[0], color="magenta", label="C-avg(L,R) (Observed)")
    ax[4, 3].ecdf(mean_CR_subtract[1], color="black", label="C-avg(L,R) (Null)")
    ax[4, 3].legend()
    ax[4, 3].set_title(f"C - avg(L, R) (2-KS:{ks_stat:.2f} ({ks_p_value:.3f}) T:{t_p_val:.3f} W:{w_p_value:.3f})")
    ks_stat, ks_p_value = ks_2samp(mean_C2R_ratio[0], mean_C2R_ratio[1], nan_policy='omit', alternative='less')
    _, w_p_value = wilcoxon(mean_C2R_ratio[0] - mean_C2R_ratio[1], nan_policy='omit', alternative='greater')
    # _, ks1_p_value = kstest(mean_C2R_ratio[0] - mean_C2R_ratio[1], lambda x : np.where(x < 0, 0.0, 1.0), alternative='less', nan_policy='omit')
    _, t_p_val = ttest_rel(mean_C2R_ratio[0], mean_C2R_ratio[1], alternative='greater', nan_policy='omit')
    # remove nan values
    mean_C2R_ratio[0] = mean_C2R_ratio[0][~np.isnan(mean_C2R_ratio[0])]
    mean_C2R_ratio[1] = mean_C2R_ratio[1][~np.isnan(mean_C2R_ratio[1])]
    ax[4, 4].ecdf(mean_C2R_ratio[0], color="magenta", label="C^2/avg(L,R) (Observed)")
    ax[4, 4].ecdf(mean_C2R_ratio[1], color="black", label="C^2/avg(L,R) (Null)")
    ax[4, 4].legend()
    ax[4, 4].set_title(f"C^2 / avg(L, R) (2-KS:{ks_stat:.2f} ({ks_p_value:.3f}) T:{t_p_val:.3f} W:{w_p_value:.3f})")
    ks_stat, ks_p_value = ks_2samp(mean_CR_ratio[0], mean_CR_ratio[1], nan_policy='omit', alternative='less')
    _, w_p_value = wilcoxon(mean_CR_ratio[0] - mean_CR_ratio[1], nan_policy='omit', alternative='greater')
    # _, ks1_p_value = kstest(mean_CR_ratio[0] - mean_CR_ratio[1], lambda x : np.where(x < 0, 0.0, 1.0), alternative='less', nan_policy='omit')
    _, t_p_val = ttest_rel(mean_CR_ratio[0], mean_CR_ratio[1], alternative='greater', nan_policy='omit')
    # remove nan values
    mean_CR_ratio[0] = mean_CR_ratio[0][~np.isnan(mean_CR_ratio[0])]
    mean_CR_ratio[1] = mean_CR_ratio[1][~np.isnan(mean_CR_ratio[1])]
    ax[4, 5].ecdf(mean_CR_ratio[0], color="magenta", label="C/avg(L,R) (Observed)")
    ax[4, 5].ecdf(mean_CR_ratio[1], color="black", label="C/avg(L,R) (Null)")
    ax[4, 5].legend()
    ax[4, 5].set_title(f"C / avg(L, R) (2-KS:{ks_stat:.2f} ({ks_p_value:.3f})) T:{t_p_val:.3f} W:{w_p_value:.3f})")

    # Hide any axes that did not receive data
    for a in ax.flat:
        if not a.has_data():
            a.set_visible(False)
    
    # Save figure if a save_path is provided; otherwise display the plot
    if save_path is not None:
        fig.savefig(save_path, dpi=400)
        plt.close(fig)
    else:
        plt.show()        




def plot_p_value_basics(im, ridge_points, ridge_widths,
                     center_box_coords, right_box_coords, left_box_coords,
                     center_num_points, right_num_points, left_num_points,
                     center_vals, right_vals, left_vals,
                     center_means, right_means, left_means,
                     center_medians, right_medians, left_medians,
                     mean_CR_subtract, mean_CR_ratio, mean_C2R_ratio,
                     med_CR_subtract, med_CR_ratio, med_C2R_ratio,
                     fig_suptitle="", save_path=None, start_bin=0):
    """
    Create a 6x3 grid of plots showing various aspects of the ridge analysis.
    
    Parameters:
      im                : 2D image array
      ridge_points      : Nx2 array of ridge point coordinates
      center_box_coords : List of arrays/coordinates for the center boxes
      right_box_coords  : List of arrays/coordinates for the right boxes
      left_box_coords   : List of arrays/coordinates for the left boxes
      center_num_points : List of numbers for the center box counts
      right_num_points  : List of numbers for the right box counts
      left_num_points   : List of numbers for the left box counts
      center_vals       : List of intensity value arrays from the center boxes
      right_vals        : List of intensity value arrays from the right boxes
      left_vals         : List of intensity value arrays from the left boxes
      center_means      : Array or list of mean intensity values from the center boxes
      right_means       : Array or list of mean intensity values from the right boxes
      left_means        : Array or list of mean intensity values from the left boxes
      center_medians    : Array or list of median intensity values from the center boxes.
      right_medians     : Array or list of median intensity values from the right boxes.
      left_medians      : Array or list of median intensity values from the left boxes.
      CR_subtract       : Array of differences between center and average of left/right.
      CR_ratio          : Array of center divided by average of left/right.
      C2R_ratio         : Array of (center^2) divided by average of left/right
      fig_suptitle      : String for the figure suptitle
      save_path         : If provided, the figure will be saved to os.path.join(save_path, "p_value.png")
    """
    # Create figure and axes
    fig, ax = plt.subplots(7, 3, figsize=(20, 22), layout="constrained", height_ratios=[3, 1, 1, 1, 1, 1, 1])
    fig.suptitle(fig_suptitle)
    
    # --- First row: Original image and ridge points ---
    # Original image
    imcm = ax[0, 0].imshow(im, cmap='Reds')
    ax[0, 0].set_title("Original")
    plt.colorbar(imcm, ax=ax[0, 0])
    
    # Image with ridge points
    imcm = ax[0, 1].imshow(im, cmap='Reds')
    ax[0, 1].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    ax[0, 1].set_title("Original with Ridge Points")
    ax[0, 1].legend()
    plt.colorbar(imcm, ax=ax[0, 1])

    # Image with ridge points and boxes
    imcm = ax[0, 2].imshow(im, cmap='Reds')
    ax[0, 2].scatter(ridge_points[:, 0], ridge_points[:, 1], color='cyan', marker='x', s=5, label='Ridge Points')
    plt.colorbar(imcm, ax=ax[0, 2])
    
    # Center boxes
    for i, coords in enumerate(center_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75,
                          marker=None, label='Center Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='cyan', linestyle='-', linewidth=1, alpha=0.75, marker=None)
    
    # Right boxes
    for i, coords in enumerate(right_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Right Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='green', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    # Left boxes
    for i, coords in enumerate(left_box_coords):
        coords_arr = np.array(coords)
        if i == 0:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5,
                          marker=None, label='Left Box')
        else:
            ax[0, 2].plot(coords_arr[:, 0] - start_bin, coords_arr[:, 1],
                          color='blue', linestyle='-', linewidth=0.5, alpha=0.5, marker=None)
    
    ax[0, 2].set_title("Original with Ridge Points and Boxes")
    ax[0, 2].legend()
    
    # --- Second row: Number of points in boxes ---
    ax[1, 1].plot(center_num_points, '-o', color="cyan")
    ax[1, 1].set_title("Center Box Number of Points")
    ax[1, 1].set_ylabel("Number of Points")
    ax[1, 1].set_xlabel("Ridge Point Index")
    
    ax[1, 2].plot(right_num_points, '-o', color="green")
    ax[1, 2].set_title("Right Box Number of Points")
    ax[1, 2].set_ylabel("Number of Points")
    ax[1, 2].set_xlabel("Ridge Point Index")
    
    ax[1, 0].plot(left_num_points, '-o', color="blue")
    ax[1, 0].set_title("Left Box Number of Points")
    ax[1, 0].set_ylabel("Number of Points")
    ax[1, 0].set_xlabel("Ridge Point Index")
    
    # --- Third row: Boxplot, histogram and ECDF of intensity values ---
    # Boxplot
    ax[2, 0].plot(ridge_widths, "-o")
    ax[2, 0].set_title("Ridge Widths")
    ax[2, 0].set_ylabel("Intensity")
    ax[2, 0].set_xlabel("Ridge Point Index")
    
    # Histogram (PMF)
    ax[2, 1].hist(np.concatenate(center_vals), bins=20, color="cyan", alpha=0.5, label="Center Box")
    ax[2, 1].hist(np.concatenate(right_vals), bins=20, color="green", alpha=0.5, label="Right Box")
    ax[2, 1].hist(np.concatenate(left_vals), bins=20, color="blue", alpha=0.5, label="Left Box")
    ax[2, 1].legend()
    ax[2, 1].set_title("Histogram of Intensity Values")
    ax[2, 1].set_ylabel("Frequency")
    ax[2, 1].set_xlabel("Intensity")
    
    # Cumulative distribution (ECDF)
    ax[2, 2].ecdf(np.concatenate(center_vals), color="cyan", label="Center Box")
    ax[2, 2].ecdf(np.concatenate(right_vals), color="green", label="Right Box")
    ax[2, 2].ecdf(np.concatenate(left_vals), color="blue", label="Left Box")
    ax[2, 2].legend()
    ax[2, 2].set_title("Cumulative Histogram of Intensity Values")
    ax[2, 2].set_ylabel("Cumulative Frequency")
    ax[2, 2].set_xlabel("Intensity")
    
    # --- Fourth row: Mean intensity values ---
    ax[3, 0].plot(center_means, '-o', color="cyan", label="Center")
    ax[3, 0].plot(left_means, '-o', color="blue", label="Left")
    ax[3, 0].plot(right_means, '-o', color="green", label="Right")
    ax[3, 0].set_title("Comparison of Mean Intensity")
    ax[3, 0].set_ylabel("Mean Intensity")
    ax[3, 0].set_xlabel("Ridge Point Index")
    ax[3, 0].legend()
    
    ax[3, 1].hist(center_means, bins=20, color="cyan", alpha=0.5, label="Center Box")
    ax[3, 1].hist(right_means, bins=20, color="green", alpha=0.5, label="Right Box")
    ax[3, 1].hist(left_means, bins=20, color="blue", alpha=0.5, label="Left Box")
    ax[3, 1].legend()
    ax[3, 1].set_title("Histogram of Mean Intensity Values")
    ax[3, 1].set_ylabel("Frequency")
    ax[3, 1].set_xlabel("Mean Intensity")
    
    ax[3, 2].ecdf(center_means, color="cyan", label="Center Box")
    ax[3, 2].ecdf(right_means, color="green", label="Right Box")
    ax[3, 2].ecdf(left_means, color="blue", label="Left Box")
    ax[3, 2].legend()
    ax[3, 2].set_title("Cumulative Histogram of Mean Intensity Values")
    ax[3, 2].set_ylabel("Cumulative Frequency")
    ax[3, 2].set_xlabel("Mean Intensity")
    
    # --- Fifth row: Median intensity values ---
    ax[4, 0].plot(center_medians, '-o', color="cyan", label="Center")
    ax[4, 0].plot(left_medians, '-o', color="blue", label="Left")
    ax[4, 0].plot(right_medians, '-o', color="green", label="Right")
    ax[4, 0].set_title("Comparison of Median Intensity")
    ax[4, 0].set_ylabel("Median Intensity")
    ax[4, 0].set_xlabel("Ridge Point Index")
    ax[4, 0].legend()
    
    ax[4, 1].hist(center_medians, bins=20, color="cyan", alpha=0.5, label="Center Box")
    ax[4, 1].hist(right_medians, bins=20, color="green", alpha=0.5, label="Right Box")
    ax[4, 1].hist(left_medians, bins=20, color="blue", alpha=0.5, label="Left Box")
    ax[4, 1].legend()
    ax[4, 1].set_title("Histogram of Median Intensity Values")
    ax[4, 1].set_ylabel("Frequency")
    ax[4, 1].set_xlabel("Median Intensity")
    
    ax[4, 2].ecdf(center_medians, color="cyan", label="Center Box")
    ax[4, 2].ecdf(right_medians, color="green", label="Right Box")
    ax[4, 2].ecdf(left_medians, color="blue", label="Left Box")
    ax[4, 2].legend()
    ax[4, 2].set_title("Cumulative Histogram of Median Intensity Values")
    ax[4, 2].set_ylabel("Cumulative Frequency")
    ax[4, 2].set_xlabel("Median Intensity")
    
    # --- Sixth row: Mean test statistic plots ---
    ax[5, 0].plot(mean_CR_subtract, '-o')
    ax[5, 0].set_title("Mean C - avg(L, R)")
    ax[5, 0].set_xlabel("Ridge Point Index")
    
    ax[5, 1].plot(mean_C2R_ratio, '-o')
    ax[5, 1].set_title("Mean C^2 / avg(L, R)")
    ax[5, 1].set_xlabel("Ridge Point Index")
    
    ax[5, 2].plot(mean_CR_ratio, '-o')
    ax[5, 2].set_title("Mean C / avg(L, R)")
    ax[5, 2].set_xlabel("Ridge Point Index")


    # --- Seventh row: Median test statistic plots ---
    ax[6, 0].plot(med_CR_subtract, '-o')
    ax[6, 0].set_title("Median C - avg(L, R)")
    ax[6, 0].set_xlabel("Ridge Point Index")
    
    ax[6, 1].plot(med_C2R_ratio, '-o')
    ax[6, 1].set_title("Median C^2 / avg(L, R)")
    ax[6, 1].set_xlabel("Ridge Point Index")
    
    ax[6, 2].plot(med_CR_ratio, '-o')
    ax[6, 2].set_title("Median C / avg(L, R)")
    ax[6, 2].set_xlabel("Ridge Point Index")
    
    # Hide any axes that did not receive data
    for a in ax.flat:
        if not a.has_data():
            a.set_visible(False)
    
    # Save figure if a save_path is provided; otherwise display the plot
    if save_path is not None:
        fig.savefig(save_path, dpi=400)
        plt.close(fig)
    else:
        plt.show()
    
