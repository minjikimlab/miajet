from shapely.geometry import MultiPoint, LineString, Point
import pandas as pd
import numpy as np
from utils.processing import read_hic_file
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx      

from typing import List, Tuple, Hashable
PairT = Tuple[Hashable, Hashable, float]


def generate_positions(df, resolution):
    """
    For each row in df:
      1. Build p1=(extrusion_x, extrusion_y) and p2=(root, root)
      2. Compute the convex hull (a LineString) between p1 and p2
      3. Measure its length, decide how many points to sample (at least 2)
      4. Interpolate that many evenly spaced points along the hull
      5. Emit one row per interpolated point with columns: unique_id, x (bp), y (bp)
    """
    if df.empty:
        return pd.DataFrame(columns=["unique_id", "chrom", "x (bp)", "y (bp)"])

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

    plt.savefig(f"{save_path}/{data_name}_diagnostic_{plot_chrom}_{A_name}_{B_name}_jet_comparison.png", dpi=400)

    plt.close()




def match_by_iou(dfA: pd.DataFrame, dfB: pd.DataFrame, 
                 x_label, y_label,
                 buffer_radius=1.0, iou_threshold=0.0, verbose=False):
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
        coords_b = list(zip(grp_b[x_label], grp_b[y_label]))
        if len(coords_b) < 2:
            geom_b = Point(coords_b[0]).buffer(buffer_radius)
        else:
            geom_b = LineString(coords_b).buffer(buffer_radius)
        geomsB[uid_b] = geom_b

    matches = []

    # Now for each unique_id in dfA, find best‐matching unique_id in dfB
    gb = dfA.groupby("unique_id")
    for uid_a, grp_a in tqdm(gb, total=len(gb), disable=not verbose):
        coords_a = list(zip(grp_a[x_label], grp_a[y_label]))
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



