#!/usr/bin/env python
# coding: utf-8

# # Plot Venn Diagrams between 2 experiments
# 
# Experiments
# * Handpicked 37 jets (Guo et al. 2022)
# * Fontanka (Galitsyna et al., 2023) 
# * Fun (Liu et al., 2024) 
# * MIA-Jet 

# In[1]:


import sys
sys.path.append('..')
from utils.plotting import plot_hic, plot_n_hic, genomic_labels, format_ticks
from utils.processing import read_hic_file

import os
import pandas as pd
import itertools
import pybedtools
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import bioframe as bf
import numpy as np
import re
from shapely.geometry import LineString, Point
from collections import defaultdict
from shapely.geometry import MultiPoint
from tqdm import tqdm
from pyBedGraph import BedGraph
import subprocess
import yaml


import networkx as nx      


# In[2]:


def generate_positions(df, resolution):
    """
    For each row in df:
      1. Build p1=(extrusion_x, extrusion_y) and p2=(root, root)
      2. Compute the convex hull (a LineString) between p1 and p2
      3. Measure its length, decide how many points to sample (at least 2)
      4. Interpolate that many evenly spaced points along the hull
      5. Emit one row per interpolated point with columns: unique_id, x (bp), y (bp)
    """
    rows = []

    for _, row in df.iterrows():
        # define extrusion point of jet
        p1 = (row["extrusion_x"], row["extrusion_y"]) 
        # define root point of jet (on main diagonal)
        p2 = (row["root"], row["root"])

        # construct convex hull between poitns
        hull = MultiPoint([p1, p2]).convex_hull  
        
        # compute the number of points to sample along the hull
        # this is dependent on the resolution
        distance = hull.length        
        num_points = np.ceil(distance / resolution).astype(int)
        num_points = max(num_points, 2)  # Ensure at least two points
        
        # extract coordinates
        alpha = np.linspace(0, 1, num_points)
        coords = [hull.interpolate(a, normalized=True).coords[0] for a in alpha]
        
        for x_bp, y_bp in coords:
            rows.append({
                "unique_id": row["unique_id"],
                "chrom": row["chrom"],
                "x (bp)":    x_bp,
                "y (bp)":    y_bp
            })

    return pd.DataFrame.from_records(rows)



def plot_overlap_diagnostic(hic_file, plot_chrom, resolution, data_type, normalization,
                            A_name, B_name, 
                            df_pos_A, df_pos_B,
                            df_pos_intersection, 
                            df_pos_diff_A, df_pos_diff_B, 
                            save_path, data_name):
    """
    Plot the diagnostic Hi-C plots for the overlap of jets
    """
    H = read_hic_file(hic_file, chrom=plot_chrom, resolution=resolution, positions="all", 
                      data_type=data_type, normalization=normalization, verbose=False)

    if data_type == "observed":
        H = np.log10(H + 1)  # log transform for better visualization

    # Select the postiions for the chromosomes
    df_pos_A_chrom = df_pos_A[df_pos_A["chrom"] == plot_chrom].copy()
    # Bin
    df_pos_A_chrom["x_bin"] = np.ceil(df_pos_A_chrom["x (bp)"] / resolution).astype(int)
    df_pos_A_chrom["y_bin"] = np.ceil(df_pos_A_chrom["y (bp)"] / resolution).astype(int)

    # Repeat for each dataframe (A, B, A - intersection, B - intersection, intersection)
    df_pos_B_chrom = df_pos_B[df_pos_B["chrom"] == plot_chrom].copy()
    df_pos_B_chrom["x_bin"] = np.ceil(df_pos_B_chrom["x (bp)"] / resolution).astype(int)
    df_pos_B_chrom["y_bin"] = np.ceil(df_pos_B_chrom["y (bp)"] / resolution).astype(int)

    df_pos_diff_A_chrom = df_pos_diff_A[df_pos_diff_A["chrom"] == plot_chrom].copy()
    df_pos_diff_A_chrom["x_bin"] = np.ceil(df_pos_diff_A_chrom["x (bp)"] / resolution).astype(int)
    df_pos_diff_A_chrom["y_bin"] = np.ceil(df_pos_diff_A_chrom["y (bp)"] / resolution).astype(int)
    df_pos_diff_B_chrom = df_pos_diff_B[df_pos_diff_B["chrom"] == plot_chrom].copy()
    df_pos_diff_B_chrom["x_bin"] = np.ceil(df_pos_diff_B_chrom["x (bp)"] / resolution).astype(int)
    df_pos_diff_B_chrom["y_bin"] = np.ceil(df_pos_diff_B_chrom["y (bp)"] / resolution).astype(int)

    df_pos_intersection_chrom = df_pos_intersection[df_pos_intersection["chrom"] == plot_chrom].copy()
    df_pos_intersection_chrom["x_bin"] = np.ceil(df_pos_intersection_chrom["x (bp)"] / resolution).astype(int)
    df_pos_intersection_chrom["y_bin"] = np.ceil(df_pos_intersection_chrom["y (bp)"] / resolution).astype(int)

    fig, ax = plt.subplots(figsize=(20, 20), layout="constrained", dpi=400)

    im = ax.imshow(H, cmap="Reds", interpolation="none", vmax=np.percentile(H, 98))
    ax.scatter(df_pos_A_chrom["x_bin"], df_pos_A_chrom["y_bin"], s=0.3, c="blue", label=A_name)
    ax.scatter(df_pos_B_chrom["x_bin"], df_pos_B_chrom["y_bin"], s=0.3, c="green", label=B_name)
    ax.scatter(df_pos_intersection_chrom["y_bin"], df_pos_intersection_chrom["x_bin"], s=0.3, c="cyan", marker="o", label=f"{A_name} AND {B_name}")
    # ax.scatter(df_pos_diff_A_chrom["x_bin"], df_pos_diff_A_chrom["y_bin"], s=3, c="blue", marker="x", label=f"{A_name} - {B_name}", alpha=0.5)
    # ax.scatter(df_pos_diff_B_chrom["x_bin"], df_pos_diff_B_chrom["y_bin"], s=3, c="green", marker="x", label=f"{B_name} - {A_name}", alpha=0.5)

    ax.set_title(f"{data_name} {plot_chrom} Comparison of Jet Called", fontsize=16)

    ax.legend(loc="upper right", fontsize=12)

    plt.savefig(f"{save_path}/{data_name}_diagnostic_{plot_chrom}_{n1}_{n2}_jet_comparison.png", dpi=400)

    plt.close()




def match_by_iou(dfA: pd.DataFrame, dfB: pd.DataFrame, buffer_radius=1.0, iou_threshold=0.0, verbose=False):
    """
    For each unique_id in dfA, build a buffered geometry from its (x (bp), y (bp)) coords,
    then compare to every unique_id in dfB (also buffered), computing intersection-over-union.
    Record the dfB unique_id with the highest IoU > iou_threshold (and non-empty intersection).

    Parameters
    ----------
    dfA, dfB : pd.DataFrame
        Must have columns ["unique_id", "x (bp)", "y (bp)"].
    buffer_radius : float
        How much to buffer each LineString/Point before computing areas.
    iou_threshold : float
        Only record matches whose IoU exceeds this (default 0.0, i.e. any non-empty overlap).

    Returns
    -------
    List of (unique_id_A, unique_id_B) pairs.
    """
    # Precompute buffered geometries for dfB
    geomsB = dict()
    for uid_b, grp_b in dfB.groupby("unique_id"):
        coords_b = list(zip(grp_b["x (bp)"], grp_b["y (bp)"]))
        if len(coords_b) < 2:
            geom_b = Point(coords_b[0]).buffer(buffer_radius)
        else:
            geom_b = LineString(coords_b).buffer(buffer_radius)
        geomsB[uid_b] = geom_b

    matches = []

    # Now for each unique_id in dfA, find best‐matching unique_id in dfB
    gb = dfA.groupby("unique_id")
    for uid_a, grp_a in tqdm(gb, total=len(gb), disable=not verbose):
        coords_a = list(zip(grp_a["x (bp)"], grp_a["y (bp)"]))
        if len(coords_a) < 2:
            geom_a = Point(coords_a[0]).buffer(buffer_radius)
        else:
            geom_a = LineString(coords_a).buffer(buffer_radius)

        best_iou = 0.0
        best_uid_b = None

        for uid_b, geom_b in geomsB.items():
            inter = geom_a.intersection(geom_b).area
            if inter == 0:
                continue
            union = geom_a.union(geom_b).area
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_uid_b = uid_b

        if best_uid_b is not None and best_iou > iou_threshold:
            matches.append((uid_a, best_uid_b, best_iou))

    return matches



from typing import List, Tuple, Hashable
PairT = Tuple[Hashable, Hashable, float]

def unique_pairs(pairs_a2b: List[PairT], pairs_b2a: List[PairT], method="optimal") -> List[Tuple[Hashable, Hashable]]:
    """
    Combine two directed match lists and return a list of (A_ID, B_ID)
    such that no ID appears more than once.  If *use_optimal* is True,
    solve the maximum-weight matching; otherwise use a greedy heuristic
    """

    # put every edge in the same orientation
    edges: List[PairT] = []
    for a, b, w in pairs_a2b:
        edges.append((a, b, w))          
    for b, a, w in pairs_b2a:
        edges.append((a, b, w))          # flip 

    if method == "optimal":
        # max_weight_matching method
        G = nx.Graph()
        for a, b, w in edges:
            G.add_edge(f"a.{a}", f"b.{b}", weight=w)  # add tags "a." and "b." to avoid uid collisions
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
        # nx returns unordered 2-tuples; recover original IDs and orientation
        result = []
        for u, v in matching:
            if u.startswith("a."):      # u is left side
                a_id = u[2:]            # strip "a."
                b_id = v[2:]            # strip "b."
            else:
                a_id = v[2:]
                b_id = u[2:]
            result.append((a_id, b_id))
        return result

    else:
        # Greedy method
        # highest IOU first
        edges.sort(key=lambda t: t[2], # sort by weight i.e. IOU
                   reverse=True)   
        used_a = set()
        used_b = set()
        result = []

        for a, b, w in edges:
            if a not in used_a and b not in used_b:
                result.append((a, b))
                used_a.add(a)
                used_b.add(b)
        return result


import hicstraw
import cv2 as cv

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

    # 4) open hic & fetch available resolutions
    hic = hicstraw.HiCFile(hic_file)
    avail_res = sorted(hic.getResolutions())  # e.g. [500,1000,5000,...]

    # determine whether 'chr' prefix is used in the file
    names = [c.name for c in hic.getChromosomes()]
    no_chr_prefix = not any(n.startswith('chr') for n in names)

    # 5) build pileups
    pileups = []
    selected_resolutions = []
    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Retrieving pileups", disable=not verbose):
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
            int(row['start']), int(row['end']),
            int(row['start']), int(row['end'])
        )
        pileups.append(mat)

    return pileups, bed_df, selected_resolutions



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
            mat = mzd.getRecordsAsMatrix(int(row['start']), int(row['end']),
                                         int(row['start']), int(row['end']))
            pileups.append(mat)

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


def remove_and_resize_square_stacks(stack, stack_positions, expected_stack_size):
    """
    Filters out any arrays in `stack` that aren’t square, then
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
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            bad_indices.append(i)
            continue

        # resize square array to expected_stack_size × expected_stack_size
        resized = cv.resize(
            arr,
            (expected_stack_size, expected_stack_size),
            interpolation=cv.INTER_AREA
        )
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


def assign_start_end(row):
    """ Assigns the start and end of a jet based on the maximum extrusion point"""
    if row["x (bp)"].min() < row["y (bp)"].max():
        start = row["x (bp)"].min()
        end = row["y (bp)"].max()
    else:
        start = row["y (bp)"].min()
        end = row["x (bp)"].max()
    
    return pd.Series({"start": start, "end": end})




# def generate_bed_df(df_summary, df_expanded, eps, fraction):
#     """
#     Generate a summary dataframe (bed file) that contains 
#     "chrom", "start", "end", in addition to all other columns in df_summary

#     The "start" and "end" is recomputed to be the maximum extrusion point of the jet,
#     where "maximum" is defined relative to the main diagonal

#     The `eps` and `fraction` parameters controls the additional margin around the maximum extrusion point
#     * `eps` is the fixed margin in basepairs
#     * `fraction` is the fraction increase of the window size around the maximum extrusion point
#     """
#     if df_summary.empty or df_expanded.empty:
#         return pd.DataFrame()

#     # Need to make new columns for the window boundaries of aggregate plotting
#     df_plot_summary = df_expanded.groupby('unique_id').apply(assign_start_end, include_groups=False).reset_index()

#     assert np.all(df_plot_summary['end'] >= df_plot_summary['start'])

#     # Epsilon margin around the maximum extrusion point
#     curr_window_size = df_plot_summary['end'] - df_plot_summary['start']
#     df_plot_summary['start'] -= curr_window_size * fraction + eps
#     df_plot_summary['end'] += curr_window_size * fraction + eps
    
#     # just keep the essentials for merging 
#     df_plot_summary = df_plot_summary[['unique_id','start','end']]

#     df_summary_copy = df_summary.copy()

#     # Drop start and end columns of the old summary dataframe (this is generated from the miajet program)
#     df_summary_copy = df_summary_copy.drop(columns=['start', 'end'])

#     # Join on "unique_id" with df_plot_summary
#     df_summary_copy = df_summary_copy.merge(df_plot_summary, on='unique_id', how='inner')

#     return df_summary_copy



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


def generate_bed_2(df_summary, df_expanded, eps, fraction):
    """
    Identical API and invariants to `generate_bed_df` but computes the start and end in a different way

    For more details, refer to ipad notes
    """

    if df_summary.empty or df_expanded.empty:
        return pd.DataFrame()
    
    df_plot_extrusion = df_expanded.groupby('unique_id').apply(assign_extrusion, include_groups=False).reset_index()
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

def safe_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def extract_chipseq_values(chip_files, intervals, f_chrom_sizes, chromosomes, names):
    # List of list where each sublist corresponds to a chip-seq experiment
    # Each chip-seq experiment contains a list of dictionaries corresponding to each jet caller
    chipseq_values = []

    for i, f_chip in enumerate(chip_files):
        # loop through each chip-seq experiment

        bg = BedGraph(f_chrom_sizes, f_chip)

        chip_val = []
        for j, inter in enumerate(intervals):
            # loop through each jet caller method

            # genome wide
            unique_ids = []
            values = []
            for chr in chromosomes:

                # must do one chromosome at a time
                try:
                    bg.load_chrom_data(chr)
                except KeyError:
                    try:
                        # Strip away the "chr" prefix
                        bg.load_chrom_data(chr.replace("chr", ""))  
                    except KeyError:
                        print(f"Chromosome {chr} not found in {f_chip}. Skipping...")
                        # Skip this chromosome if it doesn't exist
                        continue

                inter_chrom = inter.loc[inter["chrom"] == chr].reset_index(drop=True)

                if inter_chrom.empty:
                    print(f"No jets called for chromosome {chr} by {names[j]}")
                    continue

                v = bg.stats(stat=stat, intervals=inter_chrom[["chrom", "start", "end"]].values)
                u = inter_chrom["unique_id"].values

                values.extend(list(v))
                unique_ids.extend(list(u))
            
            chip_val.append(dict(zip(unique_ids, values)))

        chipseq_values.append(chip_val)

    return chipseq_values




import itertools
from scipy.stats import ranksums

def boxplot_statistics(boxplot_data):
    """
    Computes statistics of boxplot data assuming that `boxplot_data` is a single boxplot
    Returns a python string that can be put into the boxplot title
    """
    mean = np.mean(boxplot_data)
    median = np.median(boxplot_data)
    std = np.std(boxplot_data)
    N = len(boxplot_data)

    return f"median={median:.3g} | N={N}"

sig_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

def format_sig(p):
    # find the star code (or empty string)
    stars = next((s for thr, s in sig_levels if p < thr), '')
    # always show the p-value too
    return f"{stars} (p={p:.2g})"

def add_side_stats(ax, stats, xpos=-0.35, ystart=0.95, dy=0.05, fontsize=10):
    """
    Writes one line per group down the left edge of the axes.
    `stats` is a list of strings – one per group.
    """
    for i, txt in enumerate(stats):
        ax.text(xpos, ystart - i*dy, txt,
                transform=ax.transAxes, fontsize=fontsize,
                ha='left', va='top')

def add_pairwise_sig(ax, positions, groups, y_pad=0.05):
    """
    Draws significance bars for every pair of groups.
    For crowded plots keep only selected pairs 
    """
    pairs = list(itertools.combinations(range(len(groups)), 2))
    y_max = max(max(g) for g in groups)          # top of the tallest box
    h = (max(max(g) for g in groups) - min(min(g) for g in groups)) * y_pad

    for k, (i, j) in enumerate(pairs):
        p = ranksums(groups[i], groups[j]).pvalue
        x1, x2 = positions[i], positions[j]
        y = y_max + h*(k+1)
        ax.plot([x1, x1, x2, x2], [y, y+h/3, y+h/3, y],
                lw=1, c='k')
        ax.text((x1+x2)/2, y+h/2, format_sig(p),
                ha='center', va='bottom', fontsize=8)
        

def title_boxplot(ax_title, side_by_side_titles, side_by_side_data):
    """
    Returns the axis title (str) for the boxplot with side-by-side boxplot statistics
    """
    stats = [boxplot_statistics(data) for data in side_by_side_data]
    title = f"{ax_title}\n" + "\n".join(f"{name}: {stat}" for name, stat in zip(side_by_side_titles, stats))
    return title




# # DP thymocytes (Guo et al. 2022)
# 
# The reason why we separate DP thymocyte and other cell lines is because 
# 1. Guo et al. has manual annotations which we consider as another "jet caller" 
# 2. DP thymocytes has ChIP-seq tracks which we wish to plot also

# In[3]:
hic_file = "/nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic"
# mod = "-root-2"
mod = "-stripiness-1-p-enrich-0.4"
# mod = "-normalized-2"
data_name = f"GSE199059_CD69negDPWTR1R2R3R4_merged{mod}"

# FUN
f_pred_fun_bedpe = "/nfs/turbo/umms-minjilab/sionkim/jet_pred/FUN_GSE199059_CD69negDPWTR1R2R3R4_merged_mm9/FUN-pred_50000_1.3.bedpe"
f_pred_fun_tab = "/nfs/turbo/umms-minjilab/sionkim/jet_pred/FUN_GSE199059_CD69negDPWTR1R2R3R4_merged_mm9/FUN-pred_50000_1.3.tab"

# FONTANKA
f_pred_fontanka = "/nfs/turbo/umms-minjilab/sionkim/jet_pred/FONTANKA_GSE199059_CD69negDPWTR1R2R3R4_merged.50000.predicted.fountains.thresholded.tsv"

# MIA-JET
result_type = "saliency-90-p-0.1" # MIA-Jet
# result_type = "p-0.1"  # MIA-Jet
# f_pred_miajet_summary = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/GSE199059_CD69negDPWTR1R2R3R4_merged_chr_combined_50Kb/GSE199059_CD69negDPWTR1R2R3R4_merged_chr_combined_{result_type}_50Kb_summary_table.csv"
# f_pred_miajet_expanded = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/GSE199059_CD69negDPWTR1R2R3R4_merged_chr_combined_50Kb/GSE199059_CD69negDPWTR1R2R3R4_merged_chr_combined_{result_type}_50Kb_expanded_table.csv"
f_pred_miajet_summary = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/GSE199059_CD69negDPWTR1R2R3R4_merged{mod}_chr_combined_50Kb/GSE199059_CD69negDPWTR1R2R3R4_merged{mod}_chr_combined_{result_type}_50Kb_summary_table.csv"
f_pred_miajet_expanded = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/GSE199059_CD69negDPWTR1R2R3R4_merged{mod}_chr_combined_50Kb/GSE199059_CD69negDPWTR1R2R3R4_merged{mod}_chr_combined_{result_type}_50Kb_expanded_table.csv"

# HAND-PICKED (Guo et al. 2022 DP thymocytes ONLY)
f_guo37 = "/nfs/turbo/umms-minjilab/sionkim/jet_pred/GUO-37_mm9.tsv"

# AUXILIAR DATA (ChIP-seq)
chip_files = [
    "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_input-R1_chipseq_Guo-2022_GSM5963432_mm9.bedGraph",
    # "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_input-R2_chipseq_Guo-2022_GSM5963433_mm9.bedGraph",
    "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_H3K27ac-R1_chipseq_Guo-2022_GSM5963434_mm9.bedGraph",
    # "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_H3K27ac-R2_chipseq_Guo-2022_GSM5963435_mm9.bedGraph",
    "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_RAD21-R1_chipseq_Guo-2022_GSM5963436_mm9.bedGraph",
    # "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_RAD21-R2_chipseq_Guo-2022_GSM5963437_mm9.bedGraph",
    "/nfs/turbo/umms-minjilab/downloaded_data/dp_thymocytes_CTCF_chipseq_Guo-2022_GSM5963438_mm9.bedGraph",
    # "/nfs/turbo/umms-minjilab/processing/results/SRR931713/dp_thymocytes_NIPBL_SRR931713_miseq.q30.nr.sorted.bedgraph",
    "/nfs/turbo/umms-minjilab/processing/results/SRR931714/dp_thymocytes_NIPBL_SRR931714_miseq.q30.nr.sorted.bedgraph",
]

chip_names = [
    "ChIP-seq input control GSM5963432 (R1)",
    # "ChIP-seq input control GSM5963433 (R2)",
    "ChIP-seq H3K27ac GSM5963434 (R1)",
    # "ChIP-seq H3K27ac GSM5963435 (R2)",
    "ChIP-seq RAD21 GSM5963436 (R1)",
    # "ChIP-seq RAD21 GSM5963437 (R2)",
    "ChIP-seq CTCF GSM5963438",
    # "ChIP-seq NIPBL SRR931713 (R1)",
    "ChIP-seq NIPBL SRR931714 (R2)",
]

chip_names_abrev = ["Input", "H3K27ac", "RAD21", "CTCF", "NIPBL"]

genome = "mm9"
f_chrom_sizes = "/nfs/turbo/umms-minjilab/processing/genomes/mm9/mm9.chrom.sizes"

# Common parameters
save_path = "/nfs/turbo/umms-minjilab/sionkim/miajet_analysis"

# Should be the resolution of the miajet program
# This should be standardized across all methods
# This parameter is used to define the sampling frequency of the jet lines 
resolution = 50000 # 50Kb resolution

# Pileup parameters
expected_stack_size = 100
data_type = "oe"
normalization = "KR"


# In[ ]:


guo37_table = pd.read_csv(f_guo37, sep="\t")


# In[5]:


# For the other jet callers, we just copy the code from ##Other cell lines

# Load it in
fun_bedpe = pd.read_csv(f_pred_fun_bedpe, sep="\t")
fun_table = pd.read_csv(f_pred_fun_tab, sep="\t")

fontanka_table = pd.read_csv(f_pred_fontanka, sep="\t", index_col=0)
fontanka_table.dropna(inplace=True, how="any")
fontanka_table.reset_index(inplace=True, drop=True)

miajet_positions = pd.read_csv(f_pred_miajet_expanded, comment="#")
miajet_table = pd.read_csv(f_pred_miajet_summary, comment="#")

# Common chromosome determination
# Lets not include Guo et al. because they didn't find any in chrX
# For a fair comparison of overlaps, we will only consider the common chromosomes

chromsizes = bf.fetch_chromsizes(db=genome, as_bed=True)
common_chroms = chromsizes["chrom"].tolist()
# common_chroms = set(fun_table["chrom"].unique()) & set(fontanka_table["chrom"].unique()) & set(miajet_table["chrom"].unique()) 
# common_chroms = list(common_chroms)

miajet_table = miajet_table[miajet_table["chrom"].isin(common_chroms)]
fun_table = fun_table[fun_table["chrom"].isin(common_chroms)]
fontanka_table = fontanka_table[fontanka_table["chrom"].isin(common_chroms)]

print(f"Common chromosomes: {common_chroms}")


# ## Preprocessing

# In[6]:


# Flip because we wish to minimize these quantities
# miajet_table["rmse"] *= -1
# miajet_table["entropy"] *= -1

# miajet_table["stripiness"] *= -1
miajet_table


# In[7]:


# midpoint
fun_bedpe["extrusion_x"] = (fun_bedpe["y1"] + fun_bedpe["y2"]) / 2
fun_bedpe["extrusion_y"] = (fun_bedpe["x1"] + fun_bedpe["x2"]) / 2

# guarantees that the extrusion point is in the upper diagonal
# doesn't matter except for convention but should be standardized for all points
assert np.all(fun_bedpe["extrusion_x"] >= fun_bedpe["extrusion_y"])

# the main table hosts the "start" and "end" positions, which is the main diagonal start and end
# to get a single point for the jet, we can use the midpoint of the start and end positions
fun_table["root"] = (fun_table["start"] + fun_table["end"]) / 2
# Note that "root" position should technically be "root_x" and "root_y" but since 
# this is a point on the main diagonal, they are the same and we can just use one value

# Lets add a unique identifier now
fun_table["unique_id"] = fun_table.index

# Lets combine them
fun_minimal = fun_bedpe.copy()

fun_minimal["root"] = fun_table["root"]

# Lets add a unique identifier now
fun_minimal["unique_id"] = fun_minimal.index
fun_minimal["chrom"] = fun_table["chrom"]

fun_minimal = fun_minimal[["unique_id", "chrom", "root", "extrusion_x", "extrusion_y"]]

fun_positions = generate_positions(fun_minimal, resolution)
fun_positions["unique_id"] = fun_positions["unique_id"].astype(int)

fun_positions


# In[8]:


fontanka_table = fontanka_table.reset_index(drop=True)
fontanka_table["root"] = (fontanka_table["end"] + fontanka_table["start"]) / 2
fontanka_table["unique_id"] = fontanka_table.index
fontanka_table["extrusion_x"] = fontanka_table["window_end"]
fontanka_table["extrusion_y"] = fontanka_table["window_start"]

# guarantees that the extrusion point is in the upper diagonal
# doesn't matter except for convention but should be standardized for all points
assert np.all(fontanka_table["extrusion_x"] >= fontanka_table["extrusion_y"])

fontanka_minimal = fontanka_table.copy()

fontanka_minimal = fontanka_minimal[["unique_id", "chrom", "root", "extrusion_x", "extrusion_y"]]

fontanka_positions = generate_positions(fontanka_minimal, resolution)
fontanka_positions["unique_id"] = fontanka_positions["unique_id"].astype(int)

fontanka_positions


# In[9]:


guo37_table = guo37_table.reset_index(drop=True)

guo37_table["extrusion_x"] = (guo37_table["Pos_kb"] + guo37_table["WT_d2_kb"]) * 1000
guo37_table["extrusion_y"] = (guo37_table["Pos_kb"] - guo37_table["WT_d1_kb"]) * 1000

assert np.all(guo37_table["extrusion_x"] >= guo37_table["extrusion_y"])

guo37_table["root"] = guo37_table["Pos_kb"] * 1000

guo37_table["chrom"] = "chr" + guo37_table["Chr"].astype(str)

guo37_table["unique_id"] = guo37_table.index

guo37_table["start"] = (guo37_table["Pos_kb"] - 1) * 1000
guo37_table["end"] = (guo37_table["Pos_kb"] + 1) * 1000

guo37_minimal = guo37_table.copy()
guo37_minimal = guo37_minimal[["unique_id", "chrom", "root", "extrusion_x", "extrusion_y"]]
guo37_positions = generate_positions(guo37_minimal, resolution)


guo37_positions


# In[10]:


# Expanded
positions = [
    miajet_positions,
    fun_positions,
    fontanka_positions, 
    guo37_positions
]

# Summary
tables = [
    miajet_table,
    fun_table,
    fontanka_table,
    guo37_table
]

names = [
    f"MIA-Jet {result_type}",
    "Fun",
    "Fontanka",
    "Guo et al. 2022"
]

ranking_col = [
    "stripiness", # changed! 
    "SoN", 
    "FS_peaks", 
    "Strength_au_WT"
]

position_dict = dict(zip(names, positions))
table_dict = dict(zip(names, tables))


# In[11]:


bed_tables = []
for i, (s, e) in enumerate(zip(tables, positions)):
    # bed_tables.append(generate_bed_df(s, e, eps=500e3, fraction=0.5))
    bed_tables.append(generate_bed_2(s, e, eps=500e3, fraction=1.5))

chrom_sizes = bf.fetch_chromsizes(genome, as_bed=True)


# In[12]:


bed_tables[0]


# ## N=MIA-Jet aggregate plots

# In[ ]:


N = len(bed_tables[0]) # miajet determines the number of jets to plot


# In[14]:


# Now get the top N jets according to respective ranking column
top_bed_tables = []
for i, bed_df in enumerate(bed_tables):
    bed_df.sort_values(by=ranking_col[i], ascending=False, inplace=True)
    top_bed_tables.append(bed_df.head(N).reset_index(drop=True))

top_tables = []
for i, t in enumerate(tables):
    t.sort_values(by=ranking_col[i], ascending=False, inplace=True)
    top_tables.append(t.head(N).reset_index(drop=True))


# In[15]:


agg_map = []
vmaxes = []
stacks = []
df_stack = []
resolutions = []
chipseq_stacks = []
# for each 

# resolutions = []
for bed_df in top_bed_tables:
    # Get pileups for each bed_df
    s, d, r = get_pileups_dynamic_resolution(
        hic_file=hic_file,
        bed_df_in=bed_df,
        expected_stack_size=expected_stack_size,
        chrom_sizes=chrom_sizes,
        chromosomes='all',
        window_range=(None, None),
        data_type=data_type,
        normalization=normalization,
        sort=True,
        verbose=True
    )

    # Remove centromeres and resize square stacks
    s, d = remove_and_resize_square_stacks(s, d, expected_stack_size)

    if data_type == "observed":
        s = np.log10(s + 1) # log transform for visualization

    a = np.mean(s, axis=0)
    vmaxes.append(np.percentile(a, 99))  # store the 99th percentile for color scaling
    agg_map.append(a)  # average over all stacks
    stacks.append(s)  # store the stacks for later use
    df_stack.append(d)  # store the DataFrame for later use
    resolutions.append(r)  # store the resolution for later use


# In[16]:


titles = [f"{names[i]} ranked by '{ranking_col[i]}'" for i in range(len(names))]

plot_n_hic(
    agg_map,
    suptitle=f"{data_name} top {N} aggregate contact maps",
    resolution=None,
    cmap_label=None,
    titles=titles,
    cmap='Reds',
    show=True, 
    vmax=vmaxes,
    ppr=4,
    savepath=f"{save_path}/{data_name}_{result_type}_topN-miajet_agg_map-{data_type}.png"
)


# In[17]:


# Make intervals (i.e. bed tables) for ChIP-seq data extraction
stat = "max"
window_plus_minus = int(10e3) # 10 kb
chipseq_resolution = 10000 # 10kb resolution

intervals = []

for t in top_bed_tables:
    df_interval = t[["unique_id", "chrom"]].copy()
    df_interval["midpoint"] = (t["start"] + t["end"]) / 2
    df_interval["start"] = df_interval["midpoint"] - window_plus_minus
    df_interval["end"] = df_interval["midpoint"] + window_plus_minus

    intervals.append(df_interval.reset_index(drop=True))


# In[18]:


chipseq_values = extract_chipseq_values(chip_files, intervals, f_chrom_sizes, common_chroms, names)


# In[19]:


n_exp = len(chip_names)
n_callers = len(names)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1, n_exp, layout='constrained', figsize=(5 * len(chip_names), 6.5))

for i in range(len(chip_names)):
    chip_name = chip_names_abrev[i]

    datum = chipseq_values[i]
    boxplot_data = [list(x.values()) for x in datum]

    clean_data = [np.array(vals)[~np.isnan(vals)] for vals in boxplot_data]

    # log transform
    clean_data = [np.clip(vals, 0, None) for vals in clean_data]  # clip negative values to zero
    clean_data = [np.log10(vals + 1) for vals in clean_data]
    exp_counts = [len(vals) for vals in clean_data]
    positions = np.arange(1, n_callers + 1)

    # make boxplot titles
    title_boxstat = title_boxplot(chip_names[i], names, clean_data)

    # add pairwise significance 
    add_pairwise_sig(ax[i], positions, clean_data)

    bp = ax[i].boxplot(clean_data, patch_artist=True, positions=positions, widths=0.6)

    # color each box
    for patch, col in zip(bp['boxes'], colors[:n_callers]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    # jittered scatter on top
    for j, vals in enumerate(clean_data):
        y = np.array(vals)
        x = np.full_like(y, positions[j], dtype=float) + np.random.normal(0, 0.08, size=len(y))
        ax[i].scatter(x, y, alpha=0.3, s=10, color='k', edgecolor='none')

    # x-tick labels
    ax[i].set_xticks(positions)
    ax[i].set_xticklabels(names, rotation=0)

    ax[i].set_ylabel("log10(x + 1) transformed")
    
    ax[i].set_title(title_boxstat)
    ax[i].grid(axis='y', linestyle='--', alpha=0.4)


fig.suptitle(f"Jet Caller ChIP-seq Comparison (Top N) for {data_name}\n"
             f"stat: {stat} | center ±{genomic_labels(window_plus_minus)} | chipseq-resolution: {genomic_labels(chipseq_resolution)}", fontsize=16)

plt.savefig(f"{save_path}/{data_name}_{result_type}_chipseq_comparison-topN-miajet.png", dpi=300)

plt.show()


# ## Top N aggregate plots

# In[20]:


N = 100 # the top jets to plot

# First determine if N is appropriate
min_set = np.array([len(bed_df) for bed_df in bed_tables])
if np.any(N < min_set):
    print("Current N is too large for some jet callers.")
    print(f"Adjusting N from {N} to {np.min(min_set)}")
    N = np.min(min_set)


# Now get the top N jets according to respective ranking column
top_bed_tables = []
for i, bed_df in enumerate(bed_tables):
    bed_df.sort_values(by=ranking_col[i], ascending=False, inplace=True)
    top_bed_tables.append(bed_df.head(N).reset_index(drop=True))

top_tables = []
for i, t in enumerate(tables):
    t.sort_values(by=ranking_col[i], ascending=False, inplace=True)
    top_tables.append(t.head(N).reset_index(drop=True))


# In[21]:


agg_map = []
vmaxes = []
stacks = []
df_stack = []
resolutions = []
chipseq_stacks = []
# for each 

# resolutions = []
for bed_df in top_bed_tables:
    # Get pileups for each bed_df
    s, d, r = get_pileups_dynamic_resolution(
        hic_file=hic_file,
        bed_df_in=bed_df,
        expected_stack_size=expected_stack_size,
        chrom_sizes=chrom_sizes,
        chromosomes='all',
        window_range=(None, None),
        data_type=data_type,
        normalization=normalization,
        sort=True,
        verbose=True
    )

    # Remove centromeres and resize square stacks
    s, d = remove_and_resize_square_stacks(s, d, expected_stack_size)

    if data_type == "observed":
        s = np.log10(s + 1) # log transform for visualization

    a = np.mean(s, axis=0)
    vmaxes.append(np.percentile(a, 99))  # store the 99th percentile for color scaling
    agg_map.append(a)  # average over all stacks
    stacks.append(s)  # store the stacks for later use
    df_stack.append(d)  # store the DataFrame for later use
    resolutions.append(r)  # store the resolution for later use


# In[22]:


titles = [f"{names[i]} ranked by '{ranking_col[i]}'" for i in range(len(names))]

plot_n_hic(
    agg_map,
    suptitle=f"{data_name} top {N} aggregate contact maps",
    resolution=None,
    cmap_label=None,
    titles=titles,
    cmap='Reds',
    show=True, 
    vmax=vmaxes,
    ppr=4,
    savepath=f"{save_path}/{data_name}_{result_type}_topN_agg_map-{data_type}.png"
)


# In[23]:


stat = "max"
window_plus_minus = int(10e3) # 10 kb
chipseq_resolution = 10000 # 10kb resolution

intervals = []

for t in top_bed_tables:
    df_interval = t[["unique_id", "chrom"]].copy()
    df_interval["midpoint"] = (t["start"] + t["end"]) / 2
    df_interval["start"] = df_interval["midpoint"] - window_plus_minus
    df_interval["end"] = df_interval["midpoint"] + window_plus_minus

    intervals.append(df_interval.reset_index(drop=True))


# In[24]:


chipseq_values = extract_chipseq_values(chip_files, intervals, f_chrom_sizes, common_chroms, names)


# In[25]:


n_exp = len(chip_names)
n_callers = len(names)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1, n_exp, layout='constrained', figsize=(5 * len(chip_names), 6.5))

for i in range(len(chip_names)):
    chip_name = chip_names_abrev[i]

    datum = chipseq_values[i]
    boxplot_data = [list(x.values()) for x in datum]

    clean_data = [np.array(vals)[~np.isnan(vals)] for vals in boxplot_data]

    # log transform
    clean_data = [np.clip(vals, 0, None) for vals in clean_data]  # clip negative values to zero
    clean_data = [np.log10(vals + 1) for vals in clean_data]
    exp_counts = [len(vals) for vals in clean_data]
    positions = np.arange(1, n_callers + 1)

    # make boxplot titles
    title_boxstat = title_boxplot(chip_names[i], names, clean_data)

    # add pairwise significance 
    add_pairwise_sig(ax[i], positions, clean_data)

    bp = ax[i].boxplot(clean_data, patch_artist=True, positions=positions, widths=0.6)

    # color each box
    for patch, col in zip(bp['boxes'], colors[:n_callers]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    # jittered scatter on top
    for j, vals in enumerate(clean_data):
        y = np.array(vals)
        x = np.full_like(y, positions[j], dtype=float) + np.random.normal(0, 0.08, size=len(y))
        ax[i].scatter(x, y, alpha=0.3, s=10, color='k', edgecolor='none')

    # x-tick labels
    ax[i].set_xticks(positions)
    ax[i].set_xticklabels(names, rotation=0)

    ax[i].set_ylabel("log10(x + 1) transformed")
    
    ax[i].set_title(title_boxstat)
    ax[i].grid(axis='y', linestyle='--', alpha=0.4)


fig.suptitle(f"Jet Caller ChIP-seq Comparison (Top N) for {data_name}\n"
             f"stat: {stat} | center ±{genomic_labels(window_plus_minus)} | chipseq-resolution: {genomic_labels(chipseq_resolution)}", fontsize=16)

plt.savefig(f"{save_path}/{data_name}_{result_type}_chipseq_comparison-topN.png", dpi=300)

plt.show()


# In[26]:


method_names = [f"{names[i]} ranked by '{ranking_col[i]}'" for i in range(len(names))]

for idx in range(len(names)):
    
    suptitle = method_names[idx]

    # Select top jets according to "jet_saliency" column
    top_n = N  
    # 1) get the original row‐indices of the top N
    top_idx = df_stack[idx].nlargest(top_n, ranking_col[idx]).index

    # 2) subset the DataFrame (and only then reset if you like)
    sampled_df_stack = df_stack[idx].loc[top_idx].reset_index(drop=True)

    # 3) subset the array with those SAME indices
    sampled_stack = stacks[idx][top_idx]

    titles = sampled_df_stack["chrom"].astype(str) + ":" + sampled_df_stack["start"].apply(lambda x : genomic_labels(x, N=1)) + "-" + sampled_df_stack["end"].apply(lambda x : genomic_labels(x, N=1))
    titles += f" ({ranking_col[idx]}: " + sampled_df_stack[ranking_col[idx]].map(lambda x : f"{x:.3g}") + ")"
    titles += f"\nResolution: " + pd.Series([genomic_labels(r, N=1) for r in resolutions[idx]]).astype(str) 
    titles = titles.tolist()

    genomic_shift = sampled_df_stack["start"].to_numpy()

    vmaxes = [np.percentile(s, 99) for s in sampled_stack]

    plot_n_hic(sampled_stack, 
            titles=titles, 
            resolution=resolutions[idx],
            suptitle=f"{suptitle} N={top_n}\n{data_type} {normalization}", 
            show=False, 
            genomic_shift=genomic_shift, 
            cmap_label=None, 
            vmax=vmaxes,
            ppr=6,
            savepath=f"{save_path}/{data_name}_{result_type}_individual-topN_{data_type}_{names[idx]}_negate-{ranking_col[idx]}.png",
            cmap="Reds")


# ## ALL Aggregate plots

# In[27]:


agg_map = []
vmaxes = []
stacks = []
df_stack = []
resolutions = []
chipseq_stacks = []
# for each 

# resolutions = []
for bed_df in bed_tables:
    # Get pileups for each bed_df
    s, d, r = get_pileups_dynamic_resolution(
        hic_file=hic_file,
        bed_df_in=bed_df,
        expected_stack_size=expected_stack_size,
        chrom_sizes=chrom_sizes,
        chromosomes='all',
        window_range=(None, None),
        data_type=data_type,
        normalization=normalization,
        sort=True,
        verbose=True
    )

    # Remove centromeres and resize square stacks
    s, d = remove_and_resize_square_stacks(s, d, expected_stack_size)

    if data_type == "observed":
        s = np.log10(s + 1) # log transform for visualization

    a = np.mean(s, axis=0)
    vmaxes.append(np.percentile(a, 99))  # store the 99th percentile for color scaling
    agg_map.append(a)  # average over all stacks
    stacks.append(s)  # store the stacks for later use
    df_stack.append(d)  # store the DataFrame for later use
    resolutions.append(r)  # store the resolution for later use


# In[28]:


titles = [f"{names[i]} (N={len(df_stack[i])})" for i in range(len(names))]

plot_n_hic(
    agg_map,
    suptitle=f"{data_name} All Called aggregate contact maps",
    resolution=None,
    cmap_label=None,
    titles=titles,
    cmap='Reds',
    show=True, 
    vmax=vmaxes,
    ppr=4,
    savepath=f"{save_path}/{data_name}_all_agg_map-{data_type}.png"
)


# In[29]:


stat = "max"
window_plus_minus = int(10e3) # 10 kb
chipseq_resolution = 10000 # 10kb resolution

intervals = []

for t in bed_tables:
    df_interval = t[["unique_id", "chrom"]].copy()
    df_interval["midpoint"] = (t["start"] + t["end"]) / 2
    df_interval["start"] = df_interval["midpoint"] - window_plus_minus
    df_interval["end"] = df_interval["midpoint"] + window_plus_minus

    intervals.append(df_interval.reset_index(drop=True))


# In[30]:


chipseq_values = extract_chipseq_values(chip_files, intervals, f_chrom_sizes, common_chroms, names)


# In[31]:


n_exp = len(chip_names)
n_callers = len(names)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1, n_exp, layout='constrained', figsize=(5 * len(chip_names), 6.5))

for i in range(len(chip_names)):
    chip_name = chip_names_abrev[i]

    datum = chipseq_values[i]
    boxplot_data = [list(x.values()) for x in datum]

    clean_data = [np.array(vals)[~np.isnan(vals)] for vals in boxplot_data]

    # log transform
    clean_data = [np.clip(vals, 0, None) for vals in clean_data]  # clip negative values to zero
    clean_data = [np.log10(vals + 1) for vals in clean_data]
    exp_counts = [len(vals) for vals in clean_data]
    positions = np.arange(1, n_callers + 1)

    # make boxplot titles
    title_boxstat = title_boxplot(chip_names[i], names, clean_data)

    # add pairwise significance 
    add_pairwise_sig(ax[i], positions, clean_data)

    bp = ax[i].boxplot(clean_data, patch_artist=True, positions=positions, widths=0.6)

    # color each box
    for patch, col in zip(bp['boxes'], colors[:n_callers]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    # jittered scatter on top
    for j, vals in enumerate(clean_data):
        y = np.array(vals)
        x = np.full_like(y, positions[j], dtype=float) + np.random.normal(0, 0.08, size=len(y))
        ax[i].scatter(x, y, alpha=0.3, s=10, color='k', edgecolor='none')

    # x-tick labels
    ax[i].set_xticks(positions)
    ax[i].set_xticklabels(names, rotation=0)

    ax[i].set_ylabel("log10(x + 1) transformed")
    
    ax[i].set_title(title_boxstat)
    ax[i].grid(axis='y', linestyle='--', alpha=0.4)


fig.suptitle(f"Jet Caller ChIP-seq Comparison for {data_name}\n"
             f"stat: {stat} | center ±{genomic_labels(window_plus_minus)} | chipseq-resolution: {genomic_labels(chipseq_resolution)}", fontsize=16)

plt.savefig(f"{save_path}/{data_name}_{result_type}_chipseq_comparison-all.png", dpi=300)

plt.show()


# ## Confirm NIPBL ChIP-seq mapping

# In[32]:


# chip_files[-1]


# In[33]:


# # First, lets confirm that the ChIP-seq mapping was successful
# bg = BedGraph(f_chrom_sizes, chip_files[-1])


# In[34]:


# # Region in paper is chr17:
# region_start = 67500e3 # # 67.5 Mb or 67,000 kb
# region_end = 71500e3 # 71.5 Mb or 71,000 kb
# bg.load_chrom_data('chr17')    


# In[35]:


# starts = np.arange(region_start, region_end + int(10e3), int(10e3))
# ends = starts + int(10e3)

# intervals = pd.DataFrame({
#     "chrom": "chr17",
#     "start": starts,
#     "end": ends
# }).values.tolist()


# In[36]:


# region_max = bg.stats(stat="max", intervals=intervals)
# region_mean = bg.stats(stat="mean", intervals=intervals)
# # region_cov = bg.stats(stat="coverage", intervals=intervals)


# In[37]:


# fig, ax = plt.subplots(2, 1, figsize=(12, 2), layout='constrained', sharex=True)

# ax[0].bar(starts, region_max, width=int(10e3), label="max", color='red')
# ax[1].bar(starts, region_mean, width=int(10e3), label="mean", color='blue')
# # ax[2].bar(starts, region_cov, width=chipseq_resolution, label="coverage")

# # show common legend NOW:
# fig.legend(loc='upper right')

# fig.suptitle(f"{chip_names[-1]} {int(10e3)}bp chr17:67.5mb-71.5mb", fontsize=16)

# format_ticks(ax[0], y=False)

# plt.show()


# ## Reading BedGraph for Venn Diagrams

# In[38]:


stat = "max"
window_plus_minus = int(10e3) # 10 kb
chipseq_resolution = 10000 # 10kb resolution

intervals = []

for t in bed_tables:
    df_interval = t[["unique_id", "chrom"]].copy()
    df_interval["midpoint"] = (t["start"] + t["end"]) / 2
    df_interval["start"] = df_interval["midpoint"] - window_plus_minus
    df_interval["end"] = df_interval["midpoint"] + window_plus_minus

    intervals.append(df_interval.reset_index(drop=True))


# In[39]:


# # Confirm it has chrX data!
# temp = pd.read_csv("/nfs/turbo/umms-minjilab/processing/results/SRR931714/dp_thymocytes_NIPBL_SRR931714_miseq.q30.nr.sorted.bedgraph", names=["chrom", "start", "end", "value"], sep="\t")
# temp = temp.loc[temp["chrom"] == "chrX"].reset_index(drop=True)
# temp


# In[40]:


chipseq_values = extract_chipseq_values(chip_files, intervals, f_chrom_sizes, common_chroms, names)


# In[41]:


chip_names


# In[42]:


len(chipseq_values) # number of chip-seq files


# In[43]:


len(chipseq_values[0]) # number of jet callers


# In[44]:


chipseq_values[0][0] # a dictionary of values


# ## Call AB compartments
# 
# Update: Difficult to plot A/B compartments because we may need to flip the eigenvector based on open chromatin

# In[45]:


# from cooltools.api.eigdecomp import cis_eig


# In[46]:


# eigvecs = []
# for c in tqdm(common_chroms):

#     A = read_hic_file(hic_file, chrom=c, resolution=chipseq_resolution, positions="all", 
#                       normalization=normalization, data_type=data_type, verbose=False)
    
#     _, v = cis_eig(A, n_eigs=1)

#     eigvecs.append(v[0, :])


# ## Intersection
# 
# For each partition
# 1. Aggregate plots (all cell lines)
# 2. Boxplots (DP thymocytes only)
# 3. Correlation (all cell lines, for only intersection partition)

# In[47]:


buffer_radius = 3 * resolution  # 5 bins
iou_threshold = 0  # ANY overlap


# In[48]:


results = pd.DataFrame(index=names, columns=names, dtype=int).fillna(0)
unique_identifiers = []
name_pairs = []
# fill main diagonal with the total number of jets from each method
for n in names:
    results.loc[n, n] = len(table_dict[n])

for n1, n2 in itertools.combinations(names, 2):

    print("-" * 20)
    print(f"Comparing {n1} and {n2}...")

    genome_wide_overlap = 0
    identifiers = []

    for chrom in common_chroms:
        # IOU method is per-chromosome so filter each table by chromosome
        t1 = position_dict[n1].loc[position_dict[n1]["chrom"] == chrom]
        t2 = position_dict[n2].loc[position_dict[n2]["chrom"] == chrom]

        # Summary tables to get the number of jets called by each method
        s1 = table_dict[n1].loc[table_dict[n1]["chrom"] == chrom]
        s2 = table_dict[n2].loc[table_dict[n2]["chrom"] == chrom]

        # Compute the pairs of jets in each direction
        pairs12 = match_by_iou(t1, t2, buffer_radius, iou_threshold)
        pairs21 = match_by_iou(t2, t1, buffer_radius, iou_threshold)

        # Construct graph and find unique pairs
        pairs = unique_pairs(pairs12, pairs21, method="optimal")

        # print(f"* {chrom}: {len(pairs)} pairs between {n1} ({len(s1)}) and {n2} ({len(s2)})")

        genome_wide_overlap += len(pairs)
        identifiers += pairs # extend the list

    # Update reuslts table
    results.loc[n1, n2] = genome_wide_overlap
    results.loc[n2, n1] = genome_wide_overlap  # symmetric

    unique_identifiers.append(identifiers)
    name_pairs.append((n1, n2))
    
print("Genome-wide overlaps:")
results


# In[ ]:


# Collect plot data
agg_map_pairs = [] # aggregate contact map data
jet_strength_pairs = [] # jet strength correlation data
boxplot_pairs = [] # boxplot data
for (n1, n2), pairs in zip(name_pairs, unique_identifiers):

    # vectorize the function (so it applies element-wise)
    safe_convert = np.vectorize(safe_int)

    # Construct bed dataframes from the unique identifier pairs 
    df_intersections = []

    uid1 = np.array(pairs)[:, 0]  # unique identifiers from n1
    uid2 = np.array(pairs)[:, 1]  # unique identifiers from n2

    uid1 = safe_convert(uid1)
    uid2 = safe_convert(uid2)

    df1 = pd.DataFrame({"unique_id" : uid1,})
    df2 = pd.DataFrame({"unique_id" : uid2,})

    # Lets do a merge of the summary table and the unique identifiers
    # We may now use any of df1 OR df2 (any – shouldn't matter) to construct the bed dataframe
    # to plot the aggregate jet positions

    # Use n1
    A_name = n1
    B_name = n2
    # Summary dataframe
    df_A = table_dict[n1]
    df_B = table_dict[n2]
    # IDs from the returned unique pairs
    # These will be used to merge to extract the relevant positions
    df_id = df1
    df_id_alt = df2
    # Expanded dataframe 
    df_pos_A = position_dict[n1].copy()
    df_pos_B = position_dict[n2].copy()

    # These are the "main" summary dataframes
    # That should be sufficien to plot the aggregate contact maps and boxplots
    df_intersection = df_A[df_A["unique_id"].isin(df_id["unique_id"])].reset_index(drop=True)
    df_diff_A = df_A[~df_A["unique_id"].isin(df_id["unique_id"])].reset_index(drop=True)
    df_diff_B = df_B[~df_B["unique_id"].isin(df_id_alt["unique_id"])].reset_index(drop=True)

    # Assertions to ensure that the intersection and difference cover the whole table
    assert len(df_intersection) + len(df_diff_A) == len(df_A), "The intersection and difference should cover the whole table"
    assert len(df_intersection) + len(df_diff_B) == len(df_B), "The intersection and difference should cover the whole table"

    # BOXPLOT
    table_frames = [df_diff_A, df_intersection, df_diff_B] 
    table_names = [f"{A_name} only", f"{A_name} & {B_name}", f"{B_name} only"]
    # pick A_idx for the first two groups, B_idx for the last one
    # this is the convention established above (i.e. if intersection, then n1 takes preference)
    # source means jet caller
    source_idxs  = [names.index(n1), names.index(n1), names.index(n2)]


    boxplot_vals = []
    for table_df, src in zip(table_frames, source_idxs):
        boxplot_exp = []
        # now loop over each chipseq file (outer dim)
        for chip_vals in chipseq_values:
            # pick the dict for this caller (jet caller)
            chip_exp = chip_vals[src]
            # extract in order values for these unique_ids
            vals = table_df["unique_id"].apply(lambda uid: chip_exp.get(uid, np.nan)).values
            boxplot_exp.append(vals)
        boxplot_vals.append(boxplot_exp)

    boxplot_pairs.append(boxplot_vals)

    # CORRELATION 
    # Lets get the jet strength correlation between different jet callers, which is only possible for the intersection
    # Since these are the jets that are called by both methods, enabling correlation plots
    df_intersection_A = df_A[df_A["unique_id"].isin(df_id["unique_id"])].reset_index(drop=True)
    df_intersection_B = df_B[df_B["unique_id"].isin(df_id_alt["unique_id"])].reset_index(drop=True)

    A_idx = names.index(n1)
    B_idx = names.index(n2)
    # Construct dictionaries for jet strengths
    strength_A = df_A.set_index("unique_id")[ranking_col[A_idx]].to_dict()
    strength_B = df_B.set_index("unique_id")[ranking_col[B_idx]].to_dict()

    # extract in the same order
    js_A = [strength_A.get(u, np.nan) for u in uid1]
    js_B = [strength_B.get(u, np.nan) for u in uid2]

    jet_strength_pairs.append(np.array([js_A, js_B]))

    # AGGREGATE CONTACT MAPS
    # Lets get the positions, which is better to plot the individual Hi-C diagnostic plots
    # These diagnostic plots are meant to confirm that the overlapping parameters are good
    df_pos_intersection = df_pos_A[df_pos_A["unique_id"].isin(df_id["unique_id"])].reset_index(drop=True)
    df_pos_diff_A = df_pos_A[~df_pos_A["unique_id"].isin(df_id["unique_id"])].reset_index(drop=True)
    df_pos_diff_B = df_pos_B[~df_pos_B["unique_id"].isin(df_id_alt["unique_id"])].reset_index(drop=True)

    plot_overlap_diagnostic(
        hic_file=hic_file,
        plot_chrom=common_chroms[0],  
        resolution=resolution,
        data_type="observed",
        normalization="VC_SQRT",
        A_name=A_name,
        B_name=B_name,
        df_pos_A=df_pos_A,
        df_pos_B=df_pos_B,
        df_pos_intersection=df_pos_intersection,
        df_pos_diff_A=df_pos_diff_A,
        df_pos_diff_B=df_pos_diff_B,
        save_path=save_path,
        data_name=data_name
    )

    # Make bed files to get stacks
    bed_intersection = generate_bed_2(df_intersection, df_pos_intersection, eps=500e3, fraction=1.5)
    bed_diff_A = generate_bed_2(df_diff_A, df_pos_diff_A, eps=500e3, fraction=1.5)
    bed_diff_B = generate_bed_2(df_diff_B, df_pos_diff_B, eps=500e3, fraction=1.5)

    chrom_sizes = bf.fetch_chromsizes(genome, as_bed=True)

    bed_frames = [bed_diff_A, bed_intersection, bed_diff_B]
    bed_names = [f"{A_name} only", f"{A_name} & {B_name}", f"{B_name} only"]

    agg_map = []
    # resolutions = []
    for bed_df, name in zip(bed_frames, bed_names):
        # Get pileups for each bed_df
        s, d, r = get_pileups_dynamic_resolution(
            hic_file=hic_file,
            bed_df_in=bed_df,
            expected_stack_size=expected_stack_size,
            chrom_sizes=chrom_sizes,
            chromosomes='all',
            window_range=(None, None),
            data_type=data_type,
            normalization=normalization,
            sort=True,
            verbose=True
        )

        # Remove centromeres and resize square stacks
        s, d = remove_and_resize_square_stacks(s, d, expected_stack_size)

        if data_type == "observed":
            s = np.log10(s + 1) # log transform for visualization

        agg_map.append(np.mean(s, axis=0))  # average over all stacks
        # resolutions.append(r)

    agg_map_pairs.append(agg_map)


# In[ ]:


agg_map_pairs = np.array(agg_map_pairs)

agg_map_pairs.shape 
# number of pair combinations, (A only, A&B, B only), stack size, stack size


# In[ ]:


# every pairwise combination of table names
pairs = list(itertools.combinations(names, 2))
num_pairs = len(pairs)

fig, axes = plt.subplots(num_pairs, 4, figsize=(5*4, 5*num_pairs), layout='constrained', width_ratios=[2, 1, 1, 1])

for row, (A, B) in enumerate(pairs):
    # compute overlaps as before
    total_A = results.loc[A, A]
    total_B = results.loc[B, B]
    inter   = results.loc[A, B]
    only_A  = total_A - inter
    only_B  = total_B - inter

    # column 0: the Venn
    ax = axes[row, 0]
    venn2(subsets=(int(only_A), int(only_B), int(inter)),
          set_labels=(f"{A} (N={int(total_A)})", f"{B} (N={int(total_B)})"),
          ax=ax)
    ax.set_title(f"{A} vs {B}")

    # columns 1–3: the three imshows
    # agg_map_pairs.shape == (n_pairs, 3, H, W)
    titles = [f"{A} only", f"{A} & {B}", f"{B} only"]
    counts = [int(only_A), int(inter), int(only_B)]
    row_imgs = agg_map_pairs[row]
    row_vmax = max(np.percentile(img, 99) for img in row_imgs)
    for col_idx in range(3):
        ax = axes[row, col_idx+1]
        img = agg_map_pairs[row, col_idx]
        im = ax.imshow(img, cmap="Reds", interpolation="none", vmax=np.percentile(img, 99))
        ax.set_title(f"{titles[col_idx]} (N={counts[col_idx]})")
        ax.axis('off')
    fig.colorbar(im, ax=axes[row, 1:], fraction=0.02, pad=0.04, label=f"Mean of '{data_type}'")


fig.suptitle(f"Jet Caller Comparison for {data_name}\nOverlap radius: {int(buffer_radius)} bp | Hi-C resolution: {resolution}bp", fontsize=16)

plt.savefig(f"{save_path}/{data_name}_{result_type}_venn_diagram-agg.png", dpi=300)

plt.show()


# In[ ]:


from scipy.stats import linregress

pairs = list(itertools.combinations(names, 2))
num_pairs = len(pairs)

fig, axes = plt.subplots(num_pairs, 2, figsize=(14, 5 * num_pairs), constrained_layout=True, width_ratios=[2, 1])

for row, (A, B) in enumerate(pairs):
    # compute overlap counts
    total_A = results.loc[A, A]
    total_B = results.loc[B, B]
    inter   = results.loc[A, B]
    only_A  = total_A - inter
    only_B  = total_B - inter

    # Venn Diagram
    ax0 = axes[row, 0] if num_pairs > 1 else axes[0]
    venn2(
        subsets=(int(only_A), int(only_B), int(inter)),
        set_labels=(f"{A} (N={int(total_A)})", f"{B} (N={int(total_B)})"),
        ax=ax0
    )
    ax0.set_title(f"{A} vs {B}")

    # Correlation plots
    ax1 = axes[row, 1] if num_pairs > 1 else axes[1]
    js = jet_strength_pairs[row]      # shape (2, N)
    x  = np.array(js[0], dtype=float)  # A strengths
    y  = np.array(js[1], dtype=float)  # B strengths

    # fit line and compute R2
    result = linregress(x, y)
    y_fit = result.intercept + result.slope * x
    r2 = result.rvalue ** 2

    # plot
    ax1.scatter(x, y, alpha=1, s=10)
    ax1.plot(x, y_fit, color='k', linewidth=2, )
    ax1.set_box_aspect(1)
    ax1.set_xlabel(f"{A} jet strength")
    ax1.set_ylabel(f"{B} jet strength")
    ax1.set_title(f"R2 = {r2:.3f}")

# overall title
fig.suptitle(f"Jet Strength Correlation for {data_name}\n" 
             f"Overlap radius: {int(buffer_radius)} bp | Hi-C resolution: {resolution} bp", 
             fontsize=16)

plt.savefig(f"{save_path}/{data_name}_{result_type}_venn_diagram-line.png", dpi=300)

plt.show()


# In[ ]:


len(boxplot_pairs) # for the number of pairs


# In[ ]:


name_pairs


# In[ ]:


len(boxplot_pairs[0]) # for the A only, A&B, B Only


# In[ ]:


len(boxplot_pairs[0][0]) # for the number of chip-seq experiments


# In[ ]:


chip_names_abrev


# In[ ]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

pairs = list(itertools.combinations(names, 2))
num_pairs = len(pairs)
n_exp = len(chip_files)

fig, axes = plt.subplots(num_pairs, n_exp + 1, figsize=(8 * (n_exp + 1), 7 * num_pairs), layout='constrained', width_ratios=[2] + [1] * n_exp)

for row, (A, B) in enumerate(pairs):
    # 1) Venn diagram in col 0
    total_A = results.loc[A, A]
    total_B = results.loc[B, B]
    inter   = results.loc[A, B]
    only_A  = total_A - inter
    only_B  = total_B - inter

    titles = [f"{A} only", f"{A} & {B}", f"{B} only"]

    ax = axes[row, 0]
    venn2((int(only_A), int(only_B), int(inter)), set_labels=(f"{A} (N={int(total_A)})", f"{B} (N={int(total_B)})"), ax=ax)
    ax.set_title(f"{A} vs {B}")

    row_boxes = boxplot_pairs[row]     

    # counts per category (for the titles below)
    counts = [int(only_A), int(inter), int(only_B)]

    # 2) one box‐axis per experiment
    for exp_idx in range(n_exp):
        ax = axes[row, exp_idx+1]

        # gather the 3 arrays (one per category) for this experiment
        data = [row_boxes[cat][exp_idx] for cat in range(3)]
        # drop NaNs
        clean = [np.array(d)[~np.isnan(d)] for d in data]
        clean = [np.clip(vals, 0, None) for vals in clean]  # clip negative values to zero
        clean = [np.log10(vals + 1) for vals in clean]  # log transform

        positions = np.arange(1, 4) # A, A&B, B 
        bp = ax.boxplot(clean, patch_artist=True,
                        positions=positions, widths=0.6)
        
        add_pairwise_sig(ax, positions, clean)

        # color by category
        for patch, col in zip(bp['boxes'], colors[:3]):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)

        # overplot jittered points
        for i, vals in enumerate(clean):
            x = np.full_like(vals, positions[i], dtype=float)
            x += np.random.normal(0, 0.08, size=len(vals))
            ax.scatter(x, vals, alpha=0.3, s=10, color='k', edgecolor='none')

        ax.set_xticks(positions)
        ax.set_xticklabels(titles, rotation=15, ha='right')

        title_boxstat = title_boxplot(chip_names[exp_idx], titles, clean)
        ax.set_title(title_boxstat)

        ax.set_ylabel("log10(x + 1) transformed")
        ax.grid(axis='y', linestyle='--', alpha=0.4)

# overall title
fig.suptitle(
    f"Jet Caller Comparison for {data_name}\n"
    f"Overlap radius: {int(buffer_radius)} bp | Hi-C resolution: {resolution} bp",
    fontsize=16
)

plt.savefig(f"{save_path}/{data_name}_{result_type}_venn_diagram-box.png", dpi=400)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




