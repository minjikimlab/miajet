from shapely.geometry import LineString, Point
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
from tqdm import tqdm


def find_and_remove_overlaps(df_agg, df_features, iou_threshold=0.6, verbose=False,                         
                             contour_label="Contour Number",
                             x_label="X_(px)_unmap",
                             y_label="Y_(px)_unmap",):
    """
    Find and remove overlaps by selecting the ridge that minimizes the p-value

    Method:
    1. Build a KDTree from the ridge x and y coordinates
    2. For each ridge, 
        * Construct a buffered geometry object (Point if single point, LineString otherwise)
        * Query KD tree for any intersections with the geometry object and any other ridge(s)
        * For any neighboring ridges, define "overlapping" if the IOU is above `iou_threshold`
        * Among the overlapping ridges, select the one with the minimum p-value    
        * Remove all other overlapping ridges from the final output
    
    Parameters:
    df_agg : pd.DataFrame
        Summary dataframe containing "p-val" column
    df_features : pd.DataFrame
        Expanded dataframe containing features
    iou_threshold : float
        IOU threshold to consider two ridges as overlapping
    
    Returns:
    df_agg_filtered : pd.DataFrame
        Filtered summary dataframe with reduced overlapping ridges
    """
    df_agg_merge = df_agg.copy()

    df_agg_merge = df_agg_merge.merge(df_features[[contour_label, "s_imagej", x_label, y_label, "width"]],
                                       how="inner", on=[contour_label, "s_imagej"])    

    kd_tree = KDTree(df_agg_merge[[x_label, y_label]].values)
    
    gb = df_agg_merge.groupby([contour_label, "s_imagej"], sort=False)

    final_indexers = set()
    removed_indexers = set()

    # print("\tWARNING: DEBUG MODE")
    for rank, (indexer, df_ridge) in tqdm(enumerate(gb), total=len(gb)):

        if indexer in removed_indexers:
            # skip if the ridge has already been removed
            continue

        group_indexes = []
        group_p_vals = []

        # add the current ridge to the group
        group_indexes.append(indexer)
        group_p_vals.append(df_ridge["p-val"].values[0])

        # build geometry objects of each ridge 
        # width_radius = max(df_ridge["width"].mean() / 2, 3.5)
        width_radius = 7
        if len(df_ridge) < 2:
            ridge_obj = Point(df_ridge[x_label].values[0], df_ridge[y_label].values[0]).buffer(width_radius)
        else:
            ridge_obj = LineString(df_ridge[[x_label, y_label]].values).buffer(width_radius)

        indices = kd_tree.query_radius(df_ridge[[x_label, y_label]].values, r=width_radius)
        indices = np.unique(np.concatenate(indices))

        overlapping_points = df_agg_merge.iloc[indices].reset_index(drop=True)
        overlapping_groups = overlapping_points.groupby([contour_label, "s_imagej"])

        for neigh_indexer, df_neigh in overlapping_groups: 

            if neigh_indexer == indexer:
                # skip if the same ridge
                continue

            # build geometry objects of each ridge 
            # width_radius_neigh = max(df_neigh["width"].mean() / 2, 3.5)
            width_radius_neigh = 7
            if len(df_neigh) < 2:
                neigh_obj = Point(df_neigh[x_label].values[0], df_neigh[y_label].values[0]).buffer(width_radius_neigh)
            else:
                neigh_obj = LineString(df_neigh[[x_label, y_label]].values).buffer(width_radius_neigh)

            # compute IOU
            intersection = ridge_obj.intersection(neigh_obj).area
            union = ridge_obj.union(neigh_obj).area
            iou = intersection / union if union > 0 else 0

            # if IOU is above threshold, consider the ridge as overlapping
            if iou > iou_threshold:
                group_indexes.append(neigh_indexer)
                group_p_vals.append(df_neigh["p-val"].values[0])


        # now select the ridge with the minimum p-value
        group_p_vals = np.array(group_p_vals)
        min_p_val_index = np.argmin(group_p_vals)
        min_p_val_indexer = group_indexes[min_p_val_index]

        final_indexers.add(min_p_val_indexer)
        # add all OTHER indexers
        for each_indexer in group_indexes:
            if each_indexer != min_p_val_indexer:
                removed_indexers.add(each_indexer)

    keep_df = pd.DataFrame(list(final_indexers), columns=[contour_label, "s_imagej"])
    df_agg_filtered = df_agg.merge(keep_df, how="inner", on=[contour_label, "s_imagej"]).reset_index(drop=True)

    if verbose:
        print(f"\tNumber of unique (iou: {iou_threshold}) ridges: {len(df_agg_filtered)} out of {len(df_agg)}...")

    return df_agg_filtered

    
        
        
        














