import re
import bioframe as bf
import hicstraw
from tqdm import tqdm
import numpy as np
import cv2 as cv
import pandas as pd

from utils.processing import read_hic_file, whiten_matrix
from miajet.hic_image import compute_rm_idx

def get_pileups(hic_file, bed_df_in, resolution, chrom_sizes,
                chromosomes='all', window_range=(None, None),
                data_type="observed", normalization="KR", sort=False):
    """
    Processes Hi-C data to generate pileups for genomic regions specified in a BED-format DataFrame

    Returns:
    --------
    pileups : list of numpy arrays
        Each array is a pileup matrix of Hi-C interaction data.
    bed_df  : pandas DataFrame
        Possibly sorted and trimmed bed DataFrame used for pileups.
    """
    bed_df = bed_df_in.copy()

    # Optional sort on natural chromosome order
    if sort:
        # Define a key function for numeric and special chromosomes
        def chrom_key(c):
            m = re.search(r"(\d+)$", c)
            if m:
                return int(m.group(1))
            cl = c.lower()
            if cl.endswith('x'):
                return 23
            if cl.endswith('y'):
                return 24
            if cl.endswith(('m', 'mt')):
                return 25
            return 100

        bed_df['sort_key'] = bed_df['chrom'].map(chrom_key)
        bed_df = bed_df.sort_values(['sort_key', 'start', 'end'])
        bed_df = bed_df.drop(columns=['sort_key']).reset_index(drop=True)

    # Handle custom window around midpoints
    win_up, win_down = window_range
    if win_up is not None or win_down is not None:
        bed_df['midpoint'] = ((bed_df['start'] + bed_df['end']) // 2)
        # default missing values
        win_up = win_up or 0
        win_down = win_down or 0
        bed_df['start'] = bed_df['midpoint'] - win_up
        bed_df['end'] = bed_df['midpoint'] + win_down
        bed_df = bed_df.drop(columns=['midpoint'])
        # Trim out-of-bounds
        # bioframe.trim expects a 'name' column on chrom_sizes
        chrom_sizes['name'] = chrom_sizes['chrom'] + '-valid'
        bed_df[['start', 'end']] = bed_df[['start', 'end']].astype(int)
        bed_df = bf.trim(bed_df, chrom_sizes)
        bed_df = bed_df.dropna().reset_index(drop=True)
        bed_df[['start', 'end']] = bed_df[['start', 'end']].astype(int)

    # Determine which chromosomes to include
    if chromosomes == 'all':
        chrom_set = list(bed_df['chrom'].unique())
    elif isinstance(chromosomes, (list, np.ndarray)):
        chrom_set = list(set(bed_df['chrom'].unique()) & set(chromosomes))
    elif isinstance(chromosomes, str):
        chrom_set = [chromosomes]
    else:
        print(f"Warning: 'chromosomes' argument improperly formatted: {chromosomes}")
        chrom_set = list(bed_df['chrom'].unique())

    # Filter bed_df by chrom_set in both branches
    bed_df = bed_df[bed_df['chrom'].isin(chrom_set)].reset_index(drop=True)

    # Open Hi-C file and detect prefix usage
    hic = hicstraw.HiCFile(hic_file)
    names = [c.name for c in hic.getChromosomes()]
    no_chr_prefix = not any(n.startswith('chr') for n in names)

    # Build pileups
    pileups = []
    if sort:
        for chrom in chrom_set:
            key = chrom[3:] if no_chr_prefix and chrom.startswith('chr') else chrom
            mzd = hic.getMatrixZoomData(key, key, data_type, normalization,
                                         'BP', int(resolution))
            sub = bed_df[bed_df['chrom'] == chrom]
            for _, row in tqdm(sub.iterrows(), total=len(sub)):
                mat = mzd.getRecordsAsMatrix(int(row['start']), int(row['end']),
                                             int(row['start']), int(row['end']))
                pileups.append(mat)
    else:
        for _, row in tqdm(bed_df.iterrows(), total=len(bed_df)):
            chrom = row['chrom']
            key = chrom[3:] if no_chr_prefix and chrom.startswith('chr') else chrom
            mzd = hic.getMatrixZoomData(key, key, data_type, normalization,
                                         'BP', int(resolution))
            mat = mzd.getRecordsAsMatrix(int(row['start'] - resolution), int(row['end']),
                                         int(row['start'] - resolution), int(row['end']))
            
            pileups.append(mat[1:, 1:])

    return pileups, bed_df


def remove_stack_centromeres(stack, stack_positions, expected_stack_size):
    '''
    Essentially removes Hi-C windows in the stack that are not size `expected_stack_size`
    Returns modified stack, stack_positions
    '''
    # process centromeres
    stack_uniform = []
    problem = []
    for i, each in enumerate(stack):
        if each.shape[0] != expected_stack_size:
            problem.append(i)
        else:
            stack_uniform.append(each)

    stack = np.array(stack_uniform)
    stack_positions = stack_positions.drop(problem, axis=0).reset_index(drop=True)
    assert stack.shape[0] == len(stack_positions)
    return stack, stack_positions

from scipy import ndimage

def remove_and_resize_square_stacks(stack, stack_positions, expected_stack_size):
    """
    Filters out any arrays in `stack` that aren't square, then
    resizes the remaining square arrays to (expected_stack_size, expected_stack_size).

    Parameters
    ----------
    stack : Sequence of 2D numpy arrays
        Each array should represent a Hi-C window.
    stack_positions : pandas.DataFrame
        Positions corresponding to each entry in `stack`.
    expected_stack_size : int
        The desired width and height for all retained windows.

    Returns
    -------
    (np.ndarray, pandas.DataFrame)
        - stack_resized: Array of shape (n_retained, expected_stack_size, expected_stack_size)
        - stack_positions_filtered: DataFrame of length n_retained
    """
    if stack_positions.empty:
        return stack, stack_positions

    stack_uniform = []
    bad_indices = []

    for i, arr in enumerate(stack):
        # check it's 2D and square
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
            bad_indices.append(i)
            continue

        # resize square array to expected_stack_size × expected_stack_size
        resized = cv.resize(
            arr,
            (expected_stack_size, expected_stack_size),
            interpolation=cv.INTER_AREA
        )

        # # Use scipy ndimage
        # zoom_factor = expected_stack_size / arr.shape[0]
        # resized = ndimage.zoom(arr,
        #                        zoom=zoom_factor, 
        #                        order=1, 
        #                        mode="reflect")

        stack_uniform.append(resized)

    # build numpy array of resized windows
    stack_resized = np.array(stack_uniform)

    # drop bad rows from positions, reset index
    stack_positions_filtered = (
        stack_positions
        .drop(index=bad_indices, errors='ignore')
        .reset_index(drop=True)
    )

    # sanity check
    assert stack_resized.shape[0] == len(stack_positions_filtered), (
        f"Number of retained stacks ({stack_resized.shape[0]}) "
        f"does not match positions ({len(stack_positions_filtered)})"
    )

    return stack_resized, stack_positions_filtered





def get_pileups_dynamic_resolution(
    hic_file,
    bed_df_in,
    expected_stack_size,
    chrom_sizes,
    chromosomes='all',
    window_range=(None, None),
    data_type="observed",
    normalization="KR",
    sort=False,
    verbose=False
):
    """
    Generates Hi-C pileups for each region in bed_df_in, choosing for each
    region the resolution (from hic.getResolutions()) that makes
    (region_length / resolution) as close as possible to expected_stack_size.

    Parameters
    ----------
    hic_file : str
        Path to your Hi-C .hic file.
    bed_df_in : pandas.DataFrame
        Must contain columns ['chrom', 'start', 'end'].
    expected_stack_size : int
        Desired number of bins per side of your square pileup.
    chrom_sizes : pandas.DataFrame
        Columns ['chrom', 'length'] (or 'name','length' after trimming).
    chromosomes : 'all' | list of str | str
        Which chroms to include.
    window_range : tuple(int|None, int|None)
        (upstream, downstream) around the midpoint to override bed_df_in.
    data_type : str
        e.g. "observed", "oe", etc.
    normalization : str
        e.g. "KR", "VC", ...
    sort : bool
        If True, sorts bed_df naturally by chrom, start, end.

    Returns
    -------
    pileups : list of 2D np.ndarray
        Each is a square matrix of Hi-C contacts at the chosen resolution.
    bed_df_out : pandas.DataFrame
        The (possibly trimmed, sorted) DataFrame actually used.
    """

    if bed_df_in.empty:
        print("Warning: Empty bed_df_in provided. Returning empty results.")
        return [np.zeros((expected_stack_size, expected_stack_size))], pd.DataFrame(), []

    # 1) Copy & optional sort
    bed_df = bed_df_in.copy()
    if sort:
        def chrom_key(c):
            m = re.search(r"(\d+)$", c)
            if m:
                return int(m.group(1))
            cl = c.lower()
            return {'x':23,'y':24,'m':25,'mt':25}.get(cl[-2:] if len(cl)>1 else cl[-1],100)
        bed_df['_ck'] = bed_df['chrom'].map(chrom_key)
        bed_df = bed_df.sort_values(['_ck','start','end']).drop(columns=['_ck']).reset_index(drop=True)

    # 2) apply window_range if given
    win_up, win_down = window_range
    if win_up is not None or win_down is not None:
        bed_df['mid'] = ((bed_df['start'] + bed_df['end'])//2)
        win_up = win_up or 0
        win_down = win_down or 0
        bed_df['start'] = bed_df['mid'] - win_up
        bed_df['end']   = bed_df['mid'] + win_down
        bed_df = bed_df.drop(columns=['mid'])
        # bioframe.trim wants a 'name' column on chrom_sizes
        chrom_sizes = chrom_sizes.copy()
        chrom_sizes['name'] = chrom_sizes['chrom'] + '-valid'
        bed_df[['start','end']] = bed_df[['start','end']].astype(int)
        bed_df = bf.trim(bed_df, chrom_sizes).dropna().reset_index(drop=True)
        bed_df[['start','end']] = bed_df[['start','end']].astype(int)

    # 3) restrict to desired chromosomes
    if chromosomes == 'all':
        chrom_set = bed_df['chrom'].unique().tolist()
    elif isinstance(chromosomes, str):
        chrom_set = [chromosomes]
    else:
        chrom_set = list(set(bed_df['chrom']).intersection(chromosomes))
    bed_df = bed_df[bed_df['chrom'].isin(chrom_set)].reset_index(drop=True)

    # open hic & fetch available resolutions
    hic = hicstraw.HiCFile(hic_file)
    avail_res = sorted(hic.getResolutions())

    # determine whether 'chr' prefix is used in the file
    names = [c.name for c in hic.getChromosomes()]
    no_chr_prefix = not any(n.startswith('chr') for n in names)

    # 5) build pileups
    pileups = []
    selected_resolutions = []
    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Retrieving pileups", disable=not verbose):

        # if row['unique_id'] == "chrX_9_5.933_t-1":
        #     pass # debug 


        chrom = row['chrom']
        key = chrom[3:] if no_chr_prefix and chrom.startswith('chr') else chrom

        # compute jet length
        length = int(row['end']) - int(row['start'])

        # 1) find all resolutions that yield ≥ target bins
        #    i.e. length / r >= target -> r <= length / target
        candidates = [r for r in avail_res if length / r >= expected_stack_size]

        if candidates:
            # select the largest resolution that still gives you enough pixels
            best_res = max(candidates)
        else:
            # if none can give you that many pixels, fall back
            #     to the closest in absolute terms 
            best_res = min(avail_res, key=lambda r: abs((length / r) - expected_stack_size))
            print(f"\tWarning: No resolution that guarantees the matrix size to be {expected_stack_size}")
            print(f"\tThe closest resolution is {best_res} yielding a {int(length / best_res)} size matrix")

        selected_resolutions.append(best_res)

        # fetch matrix zoom data at that resolution
        mzd = hic.getMatrixZoomData(
            key, key,
            data_type, 
            normalization,
            'BP',
            best_res
        )

        # extract the pileup
        mat = mzd.getRecordsAsMatrix(
            int(row['start'] - best_res), int(row['end']),
            int(row['start'] - best_res), int(row['end'])
        )
        # NEW extract one pixel beyond the start as the first row and col is always 0
        # Then index to get correct size
        pileups.append(mat[1:, 1:])

    return pileups, bed_df, selected_resolutions




def get_pileups_dynamic_resolution_white(
    hic_file,
    bed_df_in,
    expected_stack_size,
    chrom_sizes,
    chromosomes='all',
    window_range=(None, None),
    data_type="observed",  # retained for API compatibility; whitening uses OE
    normalization="KR",
    sort=False,
    verbose=False
):
    """
    Generates whitened Hi-C pileups for each region in bed_df_in, choosing for each
    region the resolution that makes (region_length / resolution) as close as possible
    to expected_stack_size.

    Processes one chromosome at a time to avoid caching whole-chromosome matrices.
    """

    if bed_df_in.empty:
        print("Warning: Empty bed_df_in provided. Returning empty results.")
        return [np.zeros((expected_stack_size, expected_stack_size))], pd.DataFrame(), []

    # 1) Copy & optional sort
    bed_df = bed_df_in.copy()
    if sort:
        def chrom_key(c):
            m = re.search(r"(\d+)$", c)
            if m:
                return int(m.group(1))
            cl = c.lower()
            return {'x': 23, 'y': 24, 'm': 25, 'mt': 25}.get(
                cl[-2:] if len(cl) > 1 else cl[-1], 100
            )

        bed_df['_ck'] = bed_df['chrom'].map(chrom_key)
        bed_df = (
            bed_df.sort_values(['_ck', 'start', 'end'])
            .drop(columns=['_ck'])
            .reset_index(drop=True)
        )

    # 2) apply window_range if given
    win_up, win_down = window_range
    if win_up is not None or win_down is not None:
        bed_df['mid'] = ((bed_df['start'] + bed_df['end']) // 2)
        win_up = win_up or 0
        win_down = win_down or 0
        bed_df['start'] = bed_df['mid'] - win_up
        bed_df['end'] = bed_df['mid'] + win_down
        bed_df = bed_df.drop(columns=['mid'])

        chrom_sizes = chrom_sizes.copy()
        chrom_sizes['name'] = chrom_sizes['chrom'] + '-valid'
        bed_df[['start', 'end']] = bed_df[['start', 'end']].astype(int)
        bed_df = bf.trim(bed_df, chrom_sizes).dropna().reset_index(drop=True)
        bed_df[['start', 'end']] = bed_df[['start', 'end']].astype(int)

    # 3) restrict to desired chromosomes
    if chromosomes == 'all':
        chrom_set = bed_df['chrom'].unique().tolist()
    elif isinstance(chromosomes, str):
        chrom_set = [chromosomes]
    else:
        chrom_set = list(set(bed_df['chrom']).intersection(chromosomes))

    bed_df = bed_df[bed_df['chrom'].isin(chrom_set)].reset_index(drop=True)

    # open hic & fetch available resolutions
    hic = hicstraw.HiCFile(hic_file)
    avail_res = sorted(hic.getResolutions())

    # determine whether 'chr' prefix is used in the file
    names = [c.name for c in hic.getChromosomes()]
    no_chr_prefix = not any(n.startswith('chr') for n in names)

    def hic_key(chrom):
        return chrom[3:] if no_chr_prefix and chrom.startswith('chr') else chrom

    def choose_resolution(length):
        candidates = [r for r in avail_res if length / r >= expected_stack_size]

        if candidates:
            return max(candidates)

        best_res = min(avail_res, key=lambda r: abs((length / r) - expected_stack_size))
        print(f"\tWarning: No resolution that guarantees the matrix size to be {expected_stack_size}")
        print(f"\tThe closest resolution is {best_res} yielding a {int(length / best_res)} size matrix")
        return best_res

    def whiten_chrom_matrix(chromosome, resolution):
        mat_oe = read_hic_file(
            filename=hic_file,
            chrom=chromosome,
            resolution=resolution,
            positions="all",
            data_type="oe",
            normalization=normalization,
            verbose=verbose
        )

        mat_oe = mat_oe.astype(np.float32)
        mat_oe[np.isnan(mat_oe)] = 0

        rm_idx = np.asarray(
            compute_rm_idx(mat_oe, expected_stack_size, verbose=verbose),
            dtype=int
        )

        keep_idx = np.setdiff1d(np.arange(mat_oe.shape[0]), rm_idx)
        mat_oe_sub = mat_oe[np.ix_(keep_idx, keep_idx)]

        coe_sq = np.log10(mat_oe_sub + 1)
        coe_sq = cv.normalize(
            coe_sq,
            None,
            alpha=0,
            beta=1,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_32F
        )

        coe_sq_white, _ = whiten_matrix(
            A_for_corr=coe_sq,
            A_for_whiten=coe_sq,
            epsilon=1e-5
        )

        old_to_new = np.full(mat_oe.shape[0], -1, dtype=int)
        old_to_new[keep_idx] = np.arange(len(keep_idx))

        return coe_sq_white.astype(np.float32, copy=False), old_to_new

    # 4) precompute resolutions / chrom keys while preserving output order
    bed_df['_pileup_idx'] = np.arange(len(bed_df))
    bed_df['_hic_chrom'] = bed_df['chrom'].map(hic_key)
    bed_df['_resolution'] = [
        choose_resolution(int(row['end']) - int(row['start']))
        for _, row in bed_df.iterrows()
    ]

    selected_resolutions = bed_df['_resolution'].tolist()
    pileups = [None] * len(bed_df)

    # 5) process one chromosome at a time, and one resolution within that chromosome
    chrom_iter = bed_df.groupby('_hic_chrom', sort=False)
    for chromosome, chrom_df in tqdm(
        chrom_iter,
        total=bed_df['_hic_chrom'].nunique(),
        desc="Retrieving whitened pileups",
        disable=not verbose
    ):
        for resolution, res_df in chrom_df.groupby('_resolution', sort=False):
            coe_sq_white, old_to_new = whiten_chrom_matrix(chromosome, resolution)

            for _, row in res_df.iterrows():
                start_bin = int(row['start']) // resolution
                end_bin = int(np.ceil(int(row['end']) / resolution))

                bins = np.arange(start_bin, end_bin)
                bins = bins[(bins >= 0) & (bins < len(old_to_new))]

                idx = old_to_new[bins]
                idx = idx[idx >= 0]

                pileups[int(row['_pileup_idx'])] = coe_sq_white[np.ix_(idx, idx)]

            del coe_sq_white, old_to_new

    bed_df = bed_df.drop(columns=['_pileup_idx', '_hic_chrom', '_resolution'])

    return pileups, bed_df, selected_resolutions



def assign_start_end(row):
    """ Assigns the start and end of a jet based on the maximum extrusion point"""
    if row["x (bp)"].min() < row["y (bp)"].max():
        start = row["x (bp)"].min()
        end = row["y (bp)"].max()
    else:
        start = row["y (bp)"].min()
        end = row["x (bp)"].max()
    
    return pd.Series({"start": start, "end": end})



def assign_extrusion(row):
    """
    Assigns the extrusion point of a jet to be 
    mp x = max(x (bp))
    mp y = min(y (bp))
    """
    return pd.Series({"extrusion x": row["x (bp)"].max(), "extrusion y": row["y (bp)"].min()})

def assign_midpoint(row):
    """ 
    Assigns the midpoint to be the point which is closest to the main diagonal
    Computationally, this turns out to be the minimizer of the projection distance

    arg min_i |x_i - y_i|
    """
    x = row["x (bp)"].values
    y = row["y (bp)"].values

    i = np.argmin(np.abs(x - y))

    return pd.Series({"mp x": x[i], "mp y": y[i]})

def assign_midpoint_main_diagonal(row):
    """
    Assigns midpoint to be the point on the main diagonal that is the closest
    """
    x = row["x (bp)"].values
    y = row["y (bp)"].values

    i = np.argmin(np.abs(x - y))

    common_point = (x[i] + y[i]) / 2

    return pd.Series({"mp x": common_point, "mp y": common_point})


def generate_bed_2(df_summary, df_expanded, eps, fraction, project_midpoint_to_diag=False):
    """
    Identical API and invariants to `generate_bed_df` but computes the start and end in a different way

    For more details, refer to ipad notes
    """

    if df_summary.empty or df_expanded.empty:
        return pd.DataFrame()
    
    df_plot_extrusion = df_expanded.groupby('unique_id').apply(assign_extrusion, include_groups=False).reset_index()

    if project_midpoint_to_diag:
        # New: project the midpoint (i.e. the point closest of jet to the main diagonal) TO THE MAIN DIAGONAL
        # I.e. assign it as the midpoint
        df_plot_midpoint = df_expanded.groupby('unique_id').apply(assign_midpoint_main_diagonal, include_groups=False).reset_index()
    else:
        # This is the assignment we've been using for individual plots so far
        df_plot_midpoint = df_expanded.groupby('unique_id').apply(assign_midpoint, include_groups=False).reset_index()

    # simply concatenate
    df_plot_summary = pd.merge(left=df_plot_extrusion, right=df_plot_midpoint, on="unique_id", how="inner")

    # compute virtual jet length
    df_plot_summary["delta x"] = np.abs(df_plot_summary["extrusion x"] - df_plot_summary["mp x"])
    df_plot_summary["delta y"] = np.abs(df_plot_summary["extrusion y"] - df_plot_summary["mp y"])

    df_plot_summary["virtual jet length"] = df_plot_summary.apply(lambda x : max(x["delta x"], x["delta y"]), axis=1)

    df_plot_summary["start"] = df_plot_summary["mp y"] - df_plot_summary["virtual jet length"] * fraction - eps
    df_plot_summary["end"] = df_plot_summary["mp x"] + df_plot_summary["virtual jet length"] * fraction + eps

    # just keep the essentials for merging
    df_plot_summary = df_plot_summary[['unique_id', 'start', 'end']]
    df_summary_copy = df_summary.copy()
    # Drop start and end columns of the old summary dataframe (this is generated from the miajet program)
    df_summary_copy = df_summary_copy.drop(columns=['start', 'end'])
    # Join on "unique_id" with df_plot_summary
    df_summary_copy = df_summary_copy.merge(df_plot_summary, on='unique_id', how='inner')

    return df_summary_copy



def generate_bed_3(df_summary, df_expanded, project_midpoint_to_diag=False, eps=0):
    """
    Identical API and invariants to `generate_bed_df` but computes the start and end in a different way

    Uses the midpoint as simply the start, end
    """

    if df_summary.empty or df_expanded.empty:
        return pd.DataFrame()
    
    df_plot_extrusion = df_expanded.groupby('unique_id').apply(assign_extrusion, include_groups=False).reset_index()

    if project_midpoint_to_diag:
        # New: project the midpoint (i.e. the point closest of jet to the main diagonal) TO THE MAIN DIAGONAL
        # I.e. assign it as the midpoint
        df_plot_midpoint = df_expanded.groupby('unique_id').apply(assign_midpoint_main_diagonal, include_groups=False).reset_index()
    else:
        # This is the assignment we've been using for individual plots so far
        df_plot_midpoint = df_expanded.groupby('unique_id').apply(assign_midpoint, include_groups=False).reset_index()

    # simply concatenate
    df_plot_summary = pd.merge(left=df_plot_extrusion, right=df_plot_midpoint, on="unique_id", how="inner")

    # compute virtual jet length
    # df_plot_summary["delta x"] = np.abs(df_plot_summary["extrusion x"] - df_plot_summary["mp x"])
    # df_plot_summary["delta y"] = np.abs(df_plot_summary["extrusion y"] - df_plot_summary["mp y"])

    # df_plot_summary["virtual jet length"] = df_plot_summary.apply(lambda x : max(x["delta x"], x["delta y"]), axis=1)

    df_plot_summary["start"] = np.minimum(df_plot_summary["mp y"], df_plot_summary["mp x"]) - eps
    df_plot_summary["end"] = np.maximum(df_plot_summary["mp y"], df_plot_summary["mp x"]) + eps

    # just keep the essentials for merging
    df_plot_summary = df_plot_summary[['unique_id', 'start', 'end']]
    df_summary_copy = df_summary.copy()
    # Drop start and end columns of the old summary dataframe (this is generated from the miajet program)
    df_summary_copy = df_summary_copy.drop(columns=['start', 'end'])
    # Join on "unique_id" with df_plot_summary
    df_summary_copy = df_summary_copy.merge(df_plot_summary, on='unique_id', how='inner')

    return df_summary_copy
