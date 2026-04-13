# from miajet._version import __version__

from .cli import parse_args
from .config import assign_defaults, process_args, print_parameters
from .hic_image import generate_hic_bundle, generate_contact_maps, save_image_chunks
from .call_imagej import call_imagej_scale_space, run_imagej_on_chunks
from .process_imagej import load_imagej_results, process_imagej_results, split_ridges, load_imagej_results_chunked
from .ridge_datum import construct_ridge_datum
from .expanded_table import generate_expanded_table, save_expanded_table, insert_unmapped_regions, intersect_with_true
from .rank_ridges import filter_dist_diag, generate_summary_table, filter_ridges, simulate_filter_ridges, compute_saliency, \
    compute_saliency_parallel, filter_dist_diag
from .analyze_ridges import plot_distribution_diagnostic, plot_top_k_diagnostic, temp_plot, diagnostic_filter_plot, plot_top_k, \
    save_results, plot_entropy_distribution, plot_corner_diagnostic, rank_true_ridges, plot_saliency_distribution, save_scale_space
from .compute_p_value import compute_significance, correct_significance, threshold_significance
from .overlaps import find_and_remove_overlaps
# from .stripiness import compute_stripiness
from .threshold_saliency import threshold_saliency_q
from utils.scale_space import clip_scale_range_and_update_thresholds, construct_scale_space
import time


def main():
    """
    The main function for MIA-Jet
    """
    total_time = 0 # Set timer
    # HANDLE PARAMETERS
    args = parse_args() # parse arguments
    args = assign_defaults(args) # assign based on experiment type
    config = process_args(args) # make config dictionary to be used in the rest of the code
    print_parameters(config) # print parameters if verbose
    
    # GENERATE HI-C IMAGE
    if config.verbose: print("Generating Hi-C image...")
    t0 = time.time()
    im, im_orig, im_p_value, white_im_p_value, comp_binary, rm_idx, image_path, square_size = generate_contact_maps(
        hic_file=config.hic_file,
        chromosome=config.chrom,
        resolution=config.resolution,
        window_size=config.window_size,
        data_type=config.data_type,
        normalization=config.normalization,
        rotation_padding=config.rotation_padding,
        null_model_whiten=config.compartment,
        root_within_comp=config.root_within_comp,
        save_path=config.save_dir,
        verbose=config.verbose,
        root=config.root
    )
    # Update the thresholds 
    config = clip_scale_range_and_update_thresholds(im, config, b_vmax=75, b_vmin=25) 
    total_time += time.time() - t0
    if config.verbose: print(f"Generating Hi-C image... {time.time() - t0:.0f}s Done")

    # CHUNK IMAGE
    chunks = save_image_chunks(im, config.save_dir, config.root, config.verbose)

    # RUN IMAGEJ
    if config.verbose: print("Running ImageJ...")
    t0 = time.time()
    # call_imagej_scale_space(scale_range=config.scale_range, lt=config.thresholds[0], ut=config.thresholds[1], root=config.root,
    #                         image_path=image_path, save_path=config.save_dir, num_cores=config.num_cores, verbose=config.verbose)
    run_imagej_on_chunks(chunks, config)
    total_time += time.time() - t0
    if config.verbose: print(f"Running ImageJ... {time.time() - t0:.0f}s Done")

    # PROCESS IMAGEJ
    if config.verbose: print("Processing ImageJ...")
    t0 = time.time()
    # df, df_pos, config = load_imagej_results(config, save_path=config.save_dir, scale_range=config.scale_range, root=config.root, 
    #                                  verbose=config.verbose)
    df, df_pos, config = load_imagej_results_chunked(config, chunks, save_path=config.save_dir, scale_range=config.scale_range, 
                                                     verbose=config.verbose)
    df, df_pos = process_imagej_results(df=df, df_pos=df_pos, window_size=config.window_size, N=square_size, 
                                        resolution=config.resolution, remove_kth_strata=config.rem_k_strata, remove_min_size=1, 
                                        root_within=config.root_within, num_cores=config.num_cores, verbose=config.verbose) 
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)", y_label="Y_(px)", im=im, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_1_process_imagej")
    total_time += time.time() - t0
    if config.verbose: print(f"Processing ImageJ... {time.time() - t0:.0f}s Done")


    # GENERATE SCALE SPACE FEATURES
    if config.verbose: print("Generating scale space features...")
    t0 = time.time()
    D, A = construct_scale_space(im, config.scale_range, config.gamma, config.ridge_method,
                                 filter_mode=config.convolution_padding, num_pools=config.num_cores)
    total_time += time.time() - t0
    if config.verbose: print(f"Generating scale space features... {time.time() - t0:.0f}s Done")

    # INSERT UNMAPPED REGIONS
    if config.verbose: print("Inserting unmapped regions into ridges...")
    t0 = time.time()
    df_pos = insert_unmapped_regions(df_pos, im_orig, rm_idx, square_size, config.window_size,
                                     config.resolution, config.verbose, num_cores=config.num_cores)
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)_orig", y_label="Y_(px)_orig", im=im_orig, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_2_insert_unmapped_regions")
    total_time += time.time() - t0
    if config.verbose: print(f"Inserting unmapped regions into ridges... {time.time() - t0:.0f}s Done")


    # DATA STRUCTURE CONSTRUCTION
    if config.verbose: print("Constructing data structures...")
    t0 = time.time()
    ridge_datum, gb = construct_ridge_datum(df, df_pos, im, D, A, comp_binary, 
                                            compartment=config.compartment, angle_range=config.angle_range)
    total_time += time.time() - t0
    if config.verbose: print(f"Constructing data structures... {time.time() - t0:.0f}s Done")

    # SPLITTING
    if config.verbose: print("Splitting ridges based on scale space features...")
    t0 = time.time()
    df, df_pos, ridge_datum = split_ridges(df, df_pos, gb, ridge_datum, scale_range=config.scale_range, remove_min_size=1,
                                angle_trim=config.angle_trim,
                                scale_trim=config.scale_trim,
                                scale_trim_thresh=config.scale_trim_thresh,
                                scale_trim_window=config.scale_trim_window,
                                comp_trim=config.comp_trim,
                                scale_dec_trim=config.scale_dec_trim,
                                scale_dec_thresh_trim=config.scale_dec_thresh_trim,
                                verbose=config.verbose)
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)_orig", y_label="Y_(px)_orig", im=im_orig, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_3_splitting")
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)_unmap", y_label="Y_(px)_unmap", im=comp_binary, 
      save_path=f"{config.save_sub_dir}/", im_vmax=None, resolution=config.resolution, cmap="binary_r",
        root=f"diagnostic_3_splitting_compartments")
    # Filter out ridges that are far form the diagonal after splitting
    df_pos = filter_dist_diag(df, df_pos, root_within=config.root_within, window_size=config.window_size, 
                              resolution=config.resolution, square_size_original=square_size + len(rm_idx), verbose=config.verbose)
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)_orig", y_label="Y_(px)_orig", im=im_orig, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_4_splitting_and_filter_root_within")
    temp_plot(df_pos, config.diagnostic_plots, x_label="X_(px)_unmap", y_label="Y_(px)_unmap", im=comp_binary, 
      save_path=f"{config.save_sub_dir}/", im_vmax=None, resolution=config.resolution, cmap="binary_r",
        root=f"diagnostic_4_splitting_compartments_and_filter_root_within")
    total_time += time.time() - t0
    if config.verbose: print(f"Splitting ridges based on scale space features... {time.time() - t0:.0f}s Done") 



    # COMPUTE SALIENCY 
    if config.verbose: print("Computing saliency...")
    t0 = time.time()
    df_agg, df_features = compute_saliency_parallel(df_pos, ridge_datum, im_p_val=im_p_value, im_p_val2=white_im_p_value,
                                           agg_pval="mean", scale_range=config.scale_range, 
                                           adj_nondec=config.adj_nondec, ang_frac=config.ang_frac, num_cores=config.num_cores)
    total_time += time.time() - t0
    if config.verbose: print(f"Computing saliency... {time.time() - t0:.0f}s Done")


    # FILTER RIDGES
    if config.verbose: print("Filtering ridges...")
    t0 = time.time()
    df_agg_thresholded, df_features_thresholded, individual_masks = filter_ridges(df_agg, df_features, config.resolution,
                                                                length=config.length,
                                                                angle_turbulence=config.angle_turbulence,
                                                                blobness=config.blobness,
                                                                consistency=config.consistency,
                                                                sum_consistency=config.sum_consistency,
                                                                sum_consistency_im=config.sum_consistency_im,
                                                                ridge_strength_turbulence=config.ridge_strength_turbulence,
                                                                angle_satisfied=config.angle_satisfied,
                                                                verbose=config.verbose)
    diagnostic_filter_plot(df_agg, df_features, individual_masks, config.diagnostic_plots, im_orig,
                           save_path=f"{config.save_sub_dir}/", im_vmax=99.5,
                           root="diagnostic_5_thresholded_ridges",
                           resolution=config.resolution)
    total_time += time.time() - t0
    if config.verbose: print(f"Filtering... {time.time() - t0:.0f}s Done")


    # REMOVE OVERLAPS
    if config.verbose: print("Removing overlaps...")
    t0 = time.time()
    df_agg_thresholded, df_features_thresholded = find_and_remove_overlaps(df_agg_thresholded, df_features_thresholded, 
                                                  iou_threshold=0.1, verbose=config.verbose, 
                                                  resolve_conflict=config.resolve_conflict) # Will be maximized if not p-val or p-val_white
    temp_plot(df_features_thresholded, config.diagnostic_plots, x_label="X_(px)_orig", y_label="Y_(px)_orig", im=im_orig, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_6_remove_overlaps")
    total_time += time.time() - t0
    if config.verbose: print(f"Removing overlaps... {time.time() - t0:.0f}s Done")


    # CORRECT AND THRESHOLD JETS
    if config.verbose: print("Correcting and thresholding jets...")
    t0 = time.time()
    df_agg_thresholded = correct_significance(df_agg_thresholded, method="fdr_bh")
    df_agg = correct_significance(df_agg, method="fdr_bh")
    df_agg_alpha, df_features_alpha = threshold_significance(df_agg_thresholded, df_features_thresholded, 
                                                             q_val=config.q_val, q_val_white=config.q_val_white, 
                                                             compartment=config.compartment,verbose=config.verbose)
    temp_plot(df_features_alpha, config.diagnostic_plots,x_label="X_(px)_orig", y_label="Y_(px)_orig", im=im_orig, 
              save_path=f"{config.save_sub_dir}/", im_vmax=99.5, resolution=config.resolution, 
              root=f"diagnostic_7_significant_jets")
    total_time += time.time() - t0
    if config.verbose: print(f"Correcting and thresholding jets... {time.time() - t0:.0f}s Done")
    
    
    # SAVE RESULTS (tables + genomic wide plots)
    if config.verbose: print("Saving results...")
    t0 = time.time()
    save_results(df_agg, # Save all
                 df_features, ridge_datum, "all", "sum_consistency_im", config.save_sub_dir, config.chrom, square_size,
                 rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                 root=config.root, parameter_str=config.parameter_str,
                 im=im_orig, plot=True, im_vmax=99.5)

    save_results(df_agg_thresholded, # Save filtered
                 df_features_thresholded, ridge_datum, "all", "sum_consistency_im", config.dir_thresholded, config.chrom, square_size,
                 rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                 root=config.root, parameter_str=config.parameter_str,
                im=im_orig, plot=True, im_vmax=99.5)

    save_results(df_agg_alpha, # Save filtered and significant
                df_features_alpha, ridge_datum, "all", "sum_consistency_im", config.dir_alpha, config.chrom, square_size,
                rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                root=config.root, parameter_str=config.parameter_str,
                im=im_orig, plot=True, im_vmax=99.5)

    # plot_saliency_distribution(df_agg, q=config.saliency_thresh, save_path=config.save_sub_dir)

    total_time += time.time() - t0
    if config.verbose: print(f"Saving results... {time.time() - t0:.0f}s Done")
    if config.verbose: print(f"Total time elapsed: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    

if __name__ == "__main__":
    main()
