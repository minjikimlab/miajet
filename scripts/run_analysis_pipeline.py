#!/usr/bin/env python
import os
import yaml
import numpy as np
import pandas as pd
import bioframe as bf
import itertools
import matplotlib.pyplot as plt
from skimage.filters import threshold_li
from matplotlib_venn import venn2

import sys
sys.path.append('..')
from notebooks.func_analysis import *
from notebooks.func_plotting import *
from notebooks.func_iou import *
from notebooks.func_pileup import *
from utils.plotting import plot_hic, plot_n_hic, genomic_labels, format_ticks


def safe_int(x):
    try:
        return int(x)
    except ValueError:
        return x
    


import argparse

def main(key):
    # Load config
    with open("../submit_all_config.yaml") as cf:
        config = yaml.safe_load(cf)
    samples = config["samples"]

    # Analysis parameters
    save_path = "/nfs/turbo/umms-minjilab/sionkim/miajet_analysis"
    result_type = "saliency-90-p-0.1" # MIA-Jet

    # Pileup parameters
    data_type = "oe"
    expected_stack_size = 100
    N = 100 # the top jets to plot

    # Overlap parameters
    iou_threshold = 0  # ANY overlap

    # single‐sample
    print(f"Processing experiment: {key}")

    attributes = samples[key]

    hic_file = attributes["file"]
    mcool_file = attributes["mcool"]

    basename_hic = os.path.basename(hic_file)
    data_name_hic = os.path.splitext(basename_hic)[0]

    basename_mcool = os.path.basename(mcool_file)
    if "ice" in basename_mcool:
        data_name_mcool = basename_mcool.strip("ice.mcool") # due to ice.matrix
    else:
        data_name_mcool = os.path.splitext(basename_mcool)[0] 

    data_name = data_name_hic

    resolution = int(attributes["res"])
    normalization = attributes["norm"]
    genome = attributes["genome"]

    # Standardize normalization to "VC_SQRT" due to hicstraw reading in issues
    # However, if "NONE" then keep it as is
    print("\tWarning: standardizing normalization to 'VC_SQRT' (if not 'NONE') for hicstraw compatibility")
    if normalization != "NONE":
        normalization = "VC_SQRT"

    # Overlap parameter is relative to resolution
    buffer_radius = 3 * resolution  # 5 bins

    # FUN
    f_pred_fun_bedpe = f"/nfs/turbo/umms-minjilab/sionkim/jet_pred/FUN_{data_name_hic}/FUN-pred_{resolution}_1.3.bedpe"
    f_pred_fun_tab = f"/nfs/turbo/umms-minjilab/sionkim/jet_pred/FUN_{data_name_hic}/FUN-pred_{resolution}_1.3.tab"

    # FONTANKA
    if data_name == "Repli-HiC_K562_WT_totalS":
        # If Repli Hi-C, don't apply the stringent filters
        f_pred_fontanka = f"/nfs/turbo/umms-minjilab/sionkim/jet_pred/FONTANKA_{data_name_hic}.{resolution}.predicted.fountains.thresholded-weak.tsv"
    else:
        f_pred_fontanka = f"/nfs/turbo/umms-minjilab/sionkim/jet_pred/FONTANKA_{data_name_hic}.{resolution}.predicted.fountains.thresholded.tsv"

    # MIA-JET
    f_pred_miajet_summary = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/{data_name_hic}_chr_combined_{genomic_labels(resolution)}/{data_name_hic}_chr_combined_{result_type}_{genomic_labels(resolution)}_summary_table.csv"
    f_pred_miajet_expanded = f"/nfs/turbo/umms-minjilab/sionkim/miajet_output/{data_name_hic}_chr_combined_{genomic_labels(resolution)}/{data_name_hic}_chr_combined_{result_type}_{genomic_labels(resolution)}_expanded_table.csv"


    # Load it in
    fun_bedpe = pd.read_csv(f_pred_fun_bedpe, sep="\t")
    fun_table = pd.read_csv(f_pred_fun_tab, sep="\t")

    fontanka_table = pd.read_csv(f_pred_fontanka, sep="\t", index_col=0)
    fontanka_table.dropna(inplace=True, how="any")
    fontanka_table.reset_index(inplace=True, drop=True)

    miajet_positions = pd.read_csv(f_pred_miajet_expanded, comment="#")
    miajet_table = pd.read_csv(f_pred_miajet_summary, comment="#")

    chrom_sizes = bf.fetch_chromsizes(genome, as_bed=True)
    common_chroms = chrom_sizes["chrom"].tolist()
    
    miajet_table = miajet_table[miajet_table["chrom"].isin(common_chroms)]
    fun_table = fun_table[fun_table["chrom"].isin(common_chroms)]
    fontanka_table = fontanka_table[fontanka_table["chrom"].isin(common_chroms)]

    print(f"Common chromosomes: {common_chroms}")


    # Process FUN
    fun_bedpe["extrusion_x"] = (fun_bedpe["y1"] + fun_bedpe["y2"]) / 2
    fun_bedpe["extrusion_y"] = (fun_bedpe["x1"] + fun_bedpe["x2"]) / 2
    assert np.all(fun_bedpe["extrusion_x"] >= fun_bedpe["extrusion_y"])


    fun_table["root"] = (fun_table["start"] + fun_table["end"]) / 2
    fun_table["unique_id"] = fun_table.index

    fun_minimal = fun_bedpe.copy()
    fun_minimal["root"] = fun_table["root"]

    fun_minimal["unique_id"] = fun_minimal.index
    fun_minimal["chrom"] = fun_table["chrom"]

    fun_minimal = fun_minimal[["unique_id", "chrom", "root", "extrusion_x", "extrusion_y"]]

    fun_positions = generate_positions(fun_minimal, resolution)
    fun_positions["unique_id"] = fun_positions["unique_id"].astype(int)
    
    if not fun_table.empty:
        lengths = fun_positions.groupby('unique_id').apply(lambda x : compute_length(x, p=2), include_groups=False)
        fun_table["length"] = lengths
    else:
        fun_table["length"] = None

    # Process Fontanka
    fontanka_table = fontanka_table.reset_index(drop=True)
    fontanka_table["root"] = (fontanka_table["end"] + fontanka_table["start"]) / 2
    fontanka_table["unique_id"] = fontanka_table.index
    fontanka_table["extrusion_x"] = fontanka_table["window_end"]
    fontanka_table["extrusion_y"] = fontanka_table["window_start"]
    assert np.all(fontanka_table["extrusion_x"] >= fontanka_table["extrusion_y"])

    fontanka_table["length"] = -1

    fontanka_minimal = fontanka_table.copy()
    fontanka_minimal = fontanka_minimal[["unique_id", "chrom", "root", "extrusion_x", "extrusion_y"]]

    fontanka_positions = generate_positions(fontanka_minimal, resolution)
    fontanka_positions["unique_id"] = fontanka_positions["unique_id"].astype(int)

    # Combine
    positions = [miajet_positions, fun_positions, fontanka_positions]
    tables = [miajet_table, fun_table, fontanka_table]
    names = [f"MIA-Jet {result_type}", "Fun","Fontanka"]
    ranking_col = ["jet_saliency", "SoN", "FS_peaks"]

    position_dict = dict(zip(names, positions))
    table_dict = dict(zip(names, tables)) 

    bed_tables = []
    for i, (s, e) in enumerate(zip(tables, positions)):
        # bed_tables.append(generate_bed_df(s, e, eps=500e3, fraction=0.5))
        bed_tables.append(generate_bed_2(s, e, eps=500e3, fraction=1.5))

    # First determine if N is appropriate
    min_set = np.array([len(bed_df) for bed_df in bed_tables])
    if np.any(N < min_set):
        print("Current N is too large for some jet callers.")
        print(f"Adjusting N from {N} to {np.min(min_set[min_set > 0])}")
        N = np.min(min_set[min_set > 0])

    # Now get the top N jets according to respective ranking column
    top_bed_tables = []
    for i, bed_df in enumerate(bed_tables):
        if bed_df.empty:
            top_bed_tables.append(pd.DataFrame(columns=["chrom", "start", "end", "unique_id", "length"]))
        else:
            bed_df.sort_values(by=ranking_col[i], ascending=False, inplace=True)
            top_bed_tables.append(bed_df.head(N).reset_index(drop=True))

    top_tables = []
    for i, t in enumerate(tables):
        t.sort_values(by=ranking_col[i], ascending=False, inplace=True)
        top_tables.append(t.head(N).reset_index(drop=True))


    # PLOT LENGTH DISTRIBUTION OF ALL CALLED
    for i, s in enumerate(tables):

        lengths = s["length"].values

        if len(lengths) == 0 or lengths[0] == -1:
            print(f"Lengths for {names[i]} are not available, skipping length histogram.")
            continue

        plot_length_histogram(lengths, 
                            title=f"{data_name}\nDistribution of lengths called by '{names[i]}' (N={len(lengths)})",
                            bins=50,
                            show=False, 
                            save_name=f"{save_path}/{data_name}_length_distribution_{names[i]}.png")


    # PLOT TOP N
    agg_map = []
    vmaxes = []
    stacks = []
    df_stack = []
    resolutions = []
    chipseq_stacks = []
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

    titles = [f"{names[i]} ranked by '{ranking_col[i]}'" for i in range(len(names))]

    if data_type == "observed":
        cmap = "Reds"
        vcenter = None
    else:
        cmap = "bwr"
        vcenter = 1

    plot_n_hic(
        agg_map,
        suptitle=f"{data_name} top {N} aggregate contact maps",
        resolution=None,
        cmap_label=None,
        titles=titles,
        cmap=cmap,
        vcenter=vcenter,
        standardize_cbar=True,
        show=False,
        vmax=vmaxes,
        ppr=4,
        savepath=f"{save_path}/{data_name}_topN_agg_map-{data_type}.png"
    )
    

        
    # PLOT ALL CALLED
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

    titles = [f"{names[i]} (N={len(df_stack[i])})" for i in range(len(names))]

    plot_n_hic(
        agg_map,
        suptitle=f"{data_name} All Called aggregate contact maps",
        resolution=None,
        cmap_label=None,
        titles=titles,
        cmap=cmap,
        vcenter=vcenter,
        standardize_cbar=True,
        show=False,
        vmax=vmaxes,
        ppr=4,
        savepath=f"{save_path}/{data_name}_all_agg_map-{data_type}.png"
    )


    # PLOT DIAGNOSTIC OF ALL CALLED
    method_names = [f"{names[i]} ranked by '{ranking_col[i]}'" for i in range(len(names))]

    for idx in range(len(names)):
        
        suptitle = method_names[idx]

        # Select top jets according to "jet_saliency" column
        top_n = 42  

        if len(df_stack[idx]) == 0:
            print(f"No jets called by {names[idx]}, skipping top N plot.")
            continue

        # get the original row‐indices of the top N
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
                savepath=f"{save_path}/{data_name}_individual_{data_type}_{names[idx]}_{ranking_col[idx]}.png",
                cmap="Reds")


    # COMPUTE OVERLAPS
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
            pairs12 = match_by_iou(t1, t2, buffer_radius=buffer_radius, iou_threshold=iou_threshold, x_label="x (bp)", y_label="y (bp)")
            pairs21 = match_by_iou(t2, t1, buffer_radius=buffer_radius, iou_threshold=iou_threshold, x_label="x (bp)", y_label="y (bp)")

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
    print(results)


    # COLLECT AGG DATA / PLOT DIAGNOSTIC OVERLAPS
    agg_map_pairs = [] # aggregate contact map data
    jet_strength_pairs = [] # jet strength correlation data
    for (n1, n2), pairs in zip(name_pairs, unique_identifiers):
        safe_convert = np.vectorize(safe_int)
        df_intersections = []

        if len(pairs) == 0:
            uid1 = np.array([])  
            uid2 = np.array([])  
        else:
            uid1 = np.array(pairs)[:, 0]  # unique identifiers from n1
            uid2 = np.array(pairs)[:, 1]  # unique identifiers from n2

            uid1 = safe_convert(uid1)
            uid2 = safe_convert(uid2)

        df1 = pd.DataFrame({"unique_id" : uid1,})
        df2 = pd.DataFrame({"unique_id" : uid2,})
        # Use n1
        A_name = n1
        B_name = n2
        # Summary dataframe
        df_A = table_dict[n1]
        df_B = table_dict[n2]
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
            normalization=normalization,
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

    agg_map_pairs = np.array(agg_map_pairs)

    # PLOT VENN DIAGRAM
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

    plt.savefig(f"{save_path}/{data_name}_venn_diagram-agg.png", dpi=300)

    plt.close()



    # PLOT VENN DIAGRAM + CORRELATION
    from scipy.stats import linregress

    pairs = list(itertools.combinations(names, 2))
    num_pairs = len(pairs)

    fig, axes = plt.subplots(num_pairs, 2, figsize=(14, 5 * num_pairs), constrained_layout=True, width_ratios=[2, 1])

    for row, (A, B) in enumerate(pairs):
        # compute overlap counts
        total_A = results.loc[A, A]
        total_B = results.loc[B, B]
        inter = results.loc[A, B]
        only_A = total_A - inter
        only_B = total_B - inter

        # Venn Diagram
        ax0 = axes[row, 0] if num_pairs > 1 else axes[0]
        venn2(subsets=(int(only_A), int(only_B), int(inter)),
              set_labels=(f"{A} (N={int(total_A)})", f"{B} (N={int(total_B)})"), ax=ax0)
        ax0.set_title(f"{A} vs {B}")

        # Correlation plots
        ax1 = axes[row, 1] if num_pairs > 1 else axes[1]

        if inter == 0:
            # Delete the axis 
            ax1.axis('off')
        else:
            js = jet_strength_pairs[row] # shape (2, N)
            x  = np.array(js[0], dtype=float) # A strengths
            y  = np.array(js[1], dtype=float) # B strengths

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
    
    plt.savefig(f"{save_path}/{data_name}_venn_diagram-line.png", dpi=300)


    plt.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    main(args.sample)
