import numpy as np
from utils.plotting import convert_imagej_coord_to_numpy
from utils.scale_space import extract_line_scale_space, round_line_scale_space
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from tqdm import tqdm

def peaks_with_borders(mat):
    """
    Adds relative maxima for end-points too 
    """
    n_rows, n_cols = mat.shape
    out = []
    for j in range(n_cols):
        col = mat[:, j]
        peaks = list(find_peaks(col)[0])
        # manual border checks:
        if n_rows > 1:
            if col[0] > col[1]:
                peaks.insert(0, 0)
            if col[-1] > col[-2]:
                peaks.append(n_rows - 1)
        out.append(np.array(peaks, dtype=int))
    return out



def construct_ridge_datum(df, df_pos, im, D, A, comp_binary, angle_range, compartment,
                          x_label="X_(px)_unmap", y_label="Y_(px)_unmap"):
    """
    Constructs a dictionary mapping each ridge to data
    Key: "unique_id" 
    Value: dict with keys
        "s_idx" : scale index
        "s_imagej" : scale of ridge detected by imagej
        "im_curves" : numpy.ndarray of shape (length of ridge,) 
            containing the image values along the ridge
        "D_curves" : numpy.ndarray of shape (num_scales, length of ridge) 
            containing the scale space ridge strength values along ridge
        "A_curves" : numpy.ndarray of shape (num_scales, length of ridge) 
            containing the scale space angle values along ridge
        "A_bool_curves" : numpy.ndarray of shape (num_scales, length of ridge)
            containing the scale space angle boolean values along ridge

    Updates df_pos to contain two new columns (and hence gb object):
        "comp_binary" : binary values of compartment image along ridge
        "input" : input image values along ridge
    """

    if df is None or df_pos is None:
        print("\tSkipping ridge dictionary construction...")
        return None, None, None

    ridge_datum = {}

    ridge_coords = convert_imagej_coord_to_numpy(
        df_pos[[x_label, y_label]].values,
        im.shape[0],
        flip_y=False,
        start_bin=0
    )

    if compartment:
        extracted_lines = extract_line_scale_space(ridge_coords, [comp_binary[None, ...], im[None, ...]]) 
        df_pos['comp_binary'] = extracted_lines[0].squeeze()
        df_pos['input'] = extracted_lines[1].squeeze()
    else:
        extracted_lines = extract_line_scale_space(ridge_coords, [im[None, ...]]) 
        df_pos['input'] = extracted_lines.squeeze()


    gb = df_pos.groupby('unique_id')

    n_dropped = 0
    for uid, df_ridge in tqdm(gb):

        s_idx = df_ridge["s_idx"].iloc[0]

        ridge_coords_curve = convert_imagej_coord_to_numpy(
            df_ridge[[x_label, y_label]].values,
            im.shape[0],
            flip_y=False,
            start_bin=0
        )

        # Extract the curves first
        D_curves = extract_line_scale_space(ridge_coords_curve, scale_space_container=[D])

        # For the ridge curves, we will specifically find local maxima w.r.t scale space using dbscan
        max_idx = peaks_with_borders(D_curves)

        if len(max_idx) == 0:
            n_dropped += 1
            continue

        dbscan_s_idx, _ = assign_scales_via_dbscan(max_idx, s_idx=s_idx, eps=2.5, min_samples=2)

        if len(dbscan_s_idx) == 0:
            n_dropped += 1
            continue

        A_curves = round_line_scale_space(ridge_coords_curve, scale_space_container=[A])
        A_bool_curves = np.logical_and(A_curves >= angle_range[0], A_curves <= angle_range[1])

        ridge_datum[uid] = {
            "s_idx": s_idx,
            "s_imagej": df_ridge["s_imagej"].iloc[0],
            "length": len(df_ridge),
            "D_curves": D_curves,
            "A_curves": A_curves,
            "A_bool_curves": A_bool_curves,
            "dbscan_s_idx": np.asarray(dbscan_s_idx, dtype=int), 
        }

    return ridge_datum, gb









def assign_scales_via_dbscan(local_maxima, s_idx, eps=2.5, min_samples=2):
    """
    NOTE: Currently falls back if there is no intersection; unsure if this is wanted behavior
    Use DBSCAN to cluster the 2D point cloud of local maxima across ridge positions,
    then select the cluster that intersects y = s_idx (earliest in x if ties).

    For each ridge position in that cluster, the scale index closest to s_idx is
    returned. Positions not covered by the chosen cluster fall back to the
    overall closest-to-s_idx local maximum.

    Parameters
    ----------
    local_maxima : list of np.ndarray
        Output of `peaks_with_borders`. local_maxima[i] is an array of scale
        indices where column i has a local maximum.
    s_idx : int
        The reference scale index (e.g. rec["s_idx"]).
    eps : float
        DBSCAN neighbourhood radius in the (x=position, y=scale_index) space.
    min_samples : int
        DBSCAN minimum cluster size.

    Returns
    -------
    closest_s : list of int
        One scale index per *non-empty* column, chosen from the winning cluster
        (or fallback).
    """
    flag = 0

    # ---- 1. Build 2D point cloud (x=ridge_position, y=scale_index) ----
    points = []          # (x, y) pairs
    col_indices = []     # which column each point came from
    for i, maxima in enumerate(local_maxima):
        if len(maxima) == 0:
            continue
        for s in maxima:
            points.append((i, s))
            col_indices.append(i)

    if len(points) == 0:
        return [], flag

    points = np.array(points, dtype=float)       # shape (N, 2)
    col_indices = np.array(col_indices, dtype=int)

    # ---- 2. Cluster with DBSCAN ----
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    unique_labels = set(labels)
    unique_labels.discard(-1)  # drop noise label

    # ---- 3. Find clusters that intersect y = s_idx ----
    intersecting = []  # list of (earliest_x_at_s_idx, label)
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_pts = points[mask]
        at_scale = cluster_pts[cluster_pts[:, 1] == s_idx]
        if len(at_scale) > 0:
            earliest_x = at_scale[:, 0].min()
            intersecting.append((earliest_x, lbl))

    if len(intersecting) > 0:
        # Pick the cluster whose intersection with y=s_idx is earliest in x
        intersecting.sort(key=lambda t: t[0])
        chosen_label = intersecting[0][1]
        flag = 1
    else:
        # Fallback: pick the cluster whose points are closest to s_idx in scale,
        # breaking ties by earliest x
        fallback = []
        for lbl in unique_labels:
            mask = labels == lbl
            cluster_pts = points[mask]
            min_scale_dist = np.min(np.abs(cluster_pts[:, 1] - s_idx))
            earliest_x = cluster_pts[np.abs(cluster_pts[:, 1] - s_idx) == min_scale_dist, 0].min()
            fallback.append((min_scale_dist, earliest_x, lbl))
        if fallback:
            fallback.sort(key=lambda t: (t[0], t[1]))
            chosen_label = fallback[0][2]
        else:
            chosen_label = None

    # ---- 4. Build per-column lookup from the chosen cluster ----
    chosen_by_col = {}  # col -> list of scale indices in the cluster
    if chosen_label is not None:
        chosen_mask = labels == chosen_label
        for (x, y) in points[chosen_mask]:
            chosen_by_col.setdefault(int(x), []).append(int(y))

    # ---- 5. Populate closest_s ----
    closest_s = []
    for i, maxima in enumerate(local_maxima):
        if len(maxima) == 0:
            # If no maxima at all, fall back to s_idx
            closest_s.append(s_idx)
            continue
        if i in chosen_by_col:
            candidates = np.array(chosen_by_col[i])
            closest_s.append(int(candidates[np.argmin(np.abs(candidates - s_idx))]))
        else:
            # Position not in the chosen cluster — fall back to nearest maximum
            closest_s.append(int(maxima[np.argmin(np.abs(maxima - s_idx))]))

    return closest_s, flag