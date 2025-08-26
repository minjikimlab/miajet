from miajet._version import __version__

from .cli import parse_args
from .config import assign_defaults, process_args, print_parameters
from .hic_image import generate_hic_image, generate_hic_corr_image, check_im_vmin_vmax, check_im_corner_vmin_vmax, compute_edge_strength, generate_hic_bundle
from .call_imagej import call_imagej_scale_space
from .process_imagej import load_imagej_results, process_imagej_results, \
    trim_imagej_results_corner, trim_imagej_results_angle, trim_imagej_results_eig2, trim_imagej_results_all
from .expanded_table import generate_expanded_table, save_expanded_table, insert_unmapped_regions, intersect_with_true
from .rank_ridges import generate_summary_table, filter_ridges, simulate_filter_ridges
from .analyze_ridges import plot_distribution_diagnostic, plot_top_k_diagnostic, plot_top_k_diagnostic_parallel, plot_top_k, \
    save_results, plot_entropy_distribution, plot_corner_diagnostic, rank_true_ridges, plot_saliency_distribution, save_scale_space
from .compute_p_value import compute_significance, correct_significance, threshold_significance
from .overlaps import find_and_remove_overlaps
from .stripiness import compute_stripiness
from .threshold_saliency import threshold_saliency_q
from .tee import set_logging_file
from utils.scale_space import construct_scale_space_pair, clip_scale_range_and_update_thresholds
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
    set_logging_file(config) # log output to unique file 
    print(f"miajet version: {__version__}") # print versions
    print_parameters(config) # print parameters if verbose
    
    # GENERATE HI-C IMAGE
    if config.verbose: print("Generating Hi-C image...")
    t0 = time.time()
    im, im_orig, im_p_value, im_corner, corr_im_p_value, rm_idx, image_path, square_size = generate_hic_bundle(
        hic_file=config.hic_file,
        chromosome=config.chrom,
        resolution=config.resolution,
        window_size=config.window_size,
        data_type=config.data_type,
        normalization=config.normalization,
        rotation_padding=config.rotation_padding,
        whiten=config.whiten,
        save_path=config.save_dir,
        verbose=config.verbose,
        root=config.root,
        im_vmax_perc=config.im_vmax,
        im_vmin_perc=config.im_vmin,
        corner_vmax_perc=config.im_corner_vmax,
        corner_vmin_perc=config.im_corner_vmin
    )
    # config = check_im_vmin_vmax(im, config) # Remove this because we no longer reference config.vmin/config.vmax
    # Update the thresholds 
    config = clip_scale_range_and_update_thresholds(im, config, b_vmax=90, b_vmin=25) 
    total_time += time.time() - t0
    if config.verbose: print(f"Generating Hi-C image... {time.time() - t0:.0f}s Done")

    # RUN IMAGEJ
    if config.verbose: print("Running ImageJ...")
    t0 = time.time()
    call_imagej_scale_space(scale_range=config.scale_range, lt=config.thresholds[0], ut=config.thresholds[1], root=config.root,
                            image_path=image_path, save_path=config.save_dir, num_cores=config.num_cores, verbose=config.verbose)
    total_time += time.time() - t0
    if config.verbose: print(f"Running ImageJ... {time.time() - t0:.0f}s Done")

    # PROCESS IMAGEJ
    if config.verbose: print("Processing ImageJ...")
    t0 = time.time()
    df, df_pos = load_imagej_results(save_path=config.save_dir, scale_range=config.scale_range, root=config.root, verbose=config.verbose)
    df, df_pos = process_imagej_results(df=df, df_pos=df_pos, window_size=config.window_size, N=square_size, 
                                        resolution=config.resolution, remove_kth_strata=config.rem_k_strata, remove_min_size=1, 
                                        root_within=config.root_within, num_cores=config.num_cores, verbose=config.verbose) 
    total_time += time.time() - t0
    if config.verbose: print(f"Processing ImageJ... {time.time() - t0:.0f}s Done")

    # GENERATE SCALE SPACE FEATURES
    if config.verbose: print("Generating scale space features...")
    t0 = time.time()
    I, D, W1, W2, A, R, C = construct_scale_space_pair(im, im_corner, config.scale_range, config.gamma, config.ridge_method,
        filter_mode=config.convolution_padding,
        eps_r=config.eps_r, eps_c1=config.eps_c1, eps_c2=config.eps_c2,
        zc_method=2, zc_ks=5, num_pools=config.num_cores)
    total_time += time.time() - t0
    if config.verbose: print(f"Generating scale space features... {time.time() - t0:.0f}s Done")

    # TRIMMING 
    if config.verbose: print("Trimming ridges based on scale space features...")
    t0 = time.time()
    # df, df_pos = trim_imagej_results_all(df, df_pos, im_shape_0=im.shape[0], scale_range=config.scale_range,
    #                                      C=C, corner_trim=config.corner_trim,
    #                                      A=A, angle_trim=config.angle_trim, angle_range=config.angle_range, 
    #                                      W2=W2, eig2_trim=config.eig2_trim,
    #                                      remove_min_size=1,
    #                                      num_cores=config.num_cores,
    #                                      verbose=config.verbose)   # NO SORT
    df, df_pos = trim_imagej_results_corner(df, df_pos, C=C, im_shape_0=im.shape[0], min_trim_size_in=config.corner_trim, 
                                            remove_min_size=1, num_cores=config.num_cores, verbose=config.verbose)
    df, df_pos = trim_imagej_results_angle(df, df_pos, A=A, scale_range=config.scale_range, angle_range=config.angle_range, 
                                           min_trim_size_in=config.angle_trim, im_shape_0=im.shape[0], remove_min_size=1,
                                           num_cores=config.num_cores, verbose=config.verbose)
    df, df_pos = trim_imagej_results_eig2(df, df_pos, W2=W2, scale_range=config.scale_range, 
                                          min_trim_size_in=config.eig2_trim, im_shape_0=im.shape[0], remove_min_size=1,
                                          num_cores=config.num_cores, verbose=config.verbose)    
    total_time += time.time() - t0
    if config.verbose: print(f"Trimming ridges based on scale space features... {time.time() - t0:.0f}s Done")    

    # GENERATE EXPANDED TABLE
    if config.verbose: print("Generating expanded table...")
    t0 = time.time()
    df_features = generate_expanded_table(im=im, df=df, df_pos=df_pos, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
                                          scale_range=config.scale_range, angle_range=config.angle_range,
                                          num_cores=config.num_cores, verbose=config.verbose, save_path=config.save_dir, root=config.root)
    df_features = insert_unmapped_regions(df_features, im_orig, rm_idx, square_size, config.window_size,
                                          config.resolution, config.verbose, num_cores=config.num_cores)
    total_time += time.time() - t0
    if config.verbose: print(f"Generating expanded table... {time.time() - t0:.0f}s Done")

    # RANK RIDGES
    if config.verbose: print("Ranking ridges...")
    t0 = time.time()
    df_agg = generate_summary_table(df_features, 
                                    ranking=config.ranking,
                                    angle_label="angle_imagej", # hard-code (alternatively, "angle_unwrapped")
                                    angle_range=config.angle_range,
                                    noise_consec=config.noise_consec, 
                                    noise_alt=config.noise_alt,
                                    sum_cond=config.sum_cond, 
                                    agg=config.agg,
                                    save_path=config.save_sub_dir, 
                                    root=config.root, 
                                    parameter_str=config.parameter_str,
                                    num_bins=config.num_bins, # entropy
                                    bin_size=config.bin_size, # entropy
                                    points_min=config.points_min, # entropy
                                    points_max=config.points_max, # entropy
                                    ang_frac=config.ang_frac, # angle fraction
                                    scale_range=config.scale_range, # scale range
                                    verbose=config.verbose)
    total_time += time.time() - t0
    if config.verbose: print(f"Ranking ridges... {time.time() - t0:.0f}s Done")

    # PLOT ENTROPY DISRIBUTION BEFORE FILTERING
    # plot_entropy_distribution(df_agg, num_bins=config.num_bins, bin_size=config.bin_size, 
    #                           points_min=config.points_min, points_max=config.points_max, save_path=config.save_sub_dir)


    # DIAGNOSE RIDGES
    # if config.verbose: print("\tWARNING: Diagnosing ridges...")
    # t0 = time.time()
    # save_scale_space(df_agg, df_features,
    #                 K="all", im=im, edge_strength=edge_strength,
    #                 I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                 ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                 scale_range=config.scale_range, 
    #                 save_path=config.save_sub_dir, 
    #                 angle_range=config.angle_range, verbose=config.verbose)
    # simulate_filter_ridges(df_agg, df_features, 
    #                        rmse=0.2,
    #                        entropy_thresh=config.entropy_thresh,
    #                        c0_filter=None, # Greater than 0 (i.e. select ridges whose scale is increasing only)
    #                        exp_scale_deriv=None, # first derivative threshold
    #                        exp_scale_deriv2=None, # first derivative threshold
    #                        col_mean_diff_std=None, # 0.025
    #                        ridge_cond_type=None, # frac_zeros, num_zeros or None
    #                        ridge_cond_val=None, # higher, more stringent
    #                        angle_mean_type=None, # or None
    #                        angle_range=None,
    #                        angle_deriv_thresh=None, # lower, more stringent
    #                        verbose=config.verbose,
    #                        save_path=config.save_sub_dir,
    #                        config=config, # for the plot top K variables
    #                        )
    # total_time += time.time() - t0
    # if config.verbose: print(f"Diagnosing ridges... {time.time() - t0:.0f}s Done")

    # FILTER RIDGES
    if config.verbose: print("Filtering ridges...")
    t0 = time.time()
    df_agg = filter_ridges(df_agg, 
                           rmse=config.rmse, 
                           entropy_thresh=config.entropy_thresh,
                           c0_filter=None, # Greater than 0 (i.e. select ridges whose scale is increasing only)
                           exp_scale_deriv=None, # first derivative threshold
                           exp_scale_deriv2=None, # second derivative threshold
                           col_mean_diff_std=None, # 0.025
                           ridge_cond_type=None, # frac_zeros, num_zeros or None
                           ridge_cond_val=None, # higher, more stringent
                           angle_mean_type=None, # or None
                           angle_range=None, 
                           angle_deriv_thresh=None, # lower, more stringent
                           verbose=config.verbose)
    total_time += time.time() - t0
    if config.verbose: print(f"Filtering... {time.time() - t0:.0f}s Done")

    # OUTPUT RANK OF TRUE RIDGES
    # if config.data_type == "observed":
    #     f_true_cns = "/nfs/turbo/umms-minjilab/sionkim/output/data/jets/true_set/032825_Guo_true_jets_50Kb.txt" # log observed
    # elif config.data_type == "oe":
    #     f_true_cns = "/nfs/turbo/umms-minjilab/sionkim/output/data/jets/true_set/041025_Guo_OE_true_jets_50Kb.txt" # OE
    # rank_true_ridges(df_agg, df_features, f_true_cns=f_true_cns, ranking=config.ranking, chromosome=config.chrom, resolution=config.resolution, 
    #                  save_path=config.save_sub_dir, parameter_str=config.parameter_str)

    # COMPUTE P-VALUES
    if config.verbose: print("Computing p-values...")
    t0 = time.time()
    df_agg = compute_significance(df_agg=df_agg, df_features=df_features, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, 
                                  agg="mean", statistic=1, factor_lr=1, # hard-code aggregation function and statistic
                                  num_cores=config.num_cores, verbose=config.verbose)     
    total_time += time.time() - t0
    if config.verbose: print(f"Computing p-values... {time.time() - t0:.0f}s Done")

    # REMOVE OVERLAPS
    if config.verbose: print("Removing overlaps...")
    t0 = time.time()
    df_agg = find_and_remove_overlaps(df_agg, df_features, iou_threshold=0.05, verbose=config.verbose, 
                                    #   resolve_conflict=config.ranking, 
                                      resolve_conflict="p-val", 
                                      )
    total_time += time.time() - t0
    if config.verbose: print(f"Removing overlaps... {time.time() - t0:.0f}s Done")

    # DIAGNOSE AFTER OVERLAPS
    # if config.verbose: print("\tWARNING: Diagnosing ridges after removing overlaps...")
    # t0 = time.time()
    # save_scale_space(df_agg, df_features,
    #                 K="all", im=im, edge_strength=edge_strength,
    #                 I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                 ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                 scale_range=config.scale_range, 
    #                 save_path=config.save_sub_dir, 
    #                 angle_range=config.angle_range, verbose=config.verbose)
    # simulate_filter_ridges(df_agg, df_features, 
    #                     rmse=None,
    #                     entropy_thresh=None,
    #                     c0_filter=None, # Greater than 0 (i.e. select ridges whose scale is increasing only)
    #                     exp_scale_deriv=None, # first derivative threshold
    #                     exp_scale_deriv2=None, # first derivative threshold
    #                     col_mean_diff_std=None, # 0.025
    #                     ridge_cond_type=None, # frac_zeros, num_zeros or None
    #                     ridge_cond_val=None, # higher, more stringent
    #                     angle_mean_type=None, # or None
    #                     angle_range=None,
    #                     angle_deriv_thresh=None, # lower, more stringent
    #                     verbose=config.verbose,
    #                     save_path=config.save_dir,
    #                     config=config, # for the plot top K variables
    #                     )
    # total_time += time.time() - t0
    # if config.verbose: print(f"Diagnosing ridges after removing overlaps... {time.time() - t0:.0f}s Done")

    # COMPUTE STRIPINESS 
    # if config.verbose: print("Computing stripiness...")
    # t0 = time.time()
    # max_width_bin = None # hard-code 
    # p_norm = 1 # hard-code 
    # df_agg = compute_stripiness(df_agg, df_features, im_oe=im_oe, stripiness_factor_lr=1, p_norm=p_norm, max_width_bin=max_width_bin)
    # total_time += time.time() - t0
    # if config.verbose: print(f"Computing stripiness... {time.time() - t0:.0f}s Done")

    # CORRECT AND THRESHOLD JETS
    if config.verbose: print("Correcting and thresholding jets...")
    t0 = time.time()
    df_agg = correct_significance(df_agg, method="fdr_bh")
    # threshold saliency first (based on results before p-value thresholding)
    df_agg_saliency = threshold_saliency_q(df_agg, ranking=config.ranking, q=config.saliency_thresh, verbose=config.verbose)
    df_agg_alpha = threshold_significance(df_agg, alpha_range=config.alpha, enrich_thresh=1, verbose=config.verbose)
    df_agg_alpha_sal = threshold_significance(df_agg_saliency, alpha_range=config.alpha, enrich_thresh=1, verbose=config.verbose)
    total_time += time.time() - t0
    if config.verbose: print(f"Correcting and thresholding jets... {time.time() - t0:.0f}s Done")

    # SAVE RESULTS (tables + genomic wide plots)
    if config.verbose: print("Saving results...")
    t0 = time.time()
    save_results(df_agg, # Save all
                 df_features, "all", config.ranking, config.save_sub_dir, config.chrom, square_size,
                 rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                 root=config.root, parameter_str=config.parameter_str,
                 hic_file=config.hic_file, normalization=config.normalization, plot=True,
                 rotation_padding=config.rotation_padding, im_vmax=config.im_vmax, im_vmin=config.im_vmin)
    plot_saliency_distribution(df_agg, ranking=config.ranking, q=config.saliency_thresh, save_path=config.save_sub_dir)
    save_results(df_agg_saliency, # Save saliency thresholded
                 df_features, "all", config.ranking, config.saliency_dir, config.chrom, square_size,
                 rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                 root=config.root, parameter_str=config.parameter_str,
                 hic_file=config.hic_file, normalization=config.normalization, plot=True,
                 rotation_padding=config.rotation_padding, im_vmax=config.im_vmax, im_vmin=config.im_vmin)
    for i, df_agg_each in enumerate(df_agg_alpha_sal):
        save_results(df_agg_each, # Save alpha and saliency thresholded
                     df_features, "all", config.ranking, config.alphasal_dir[i], config.chrom, square_size,
                     rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                     root=config.root, parameter_str=config.parameter_str,
                    hic_file=config.hic_file, normalization=config.normalization, plot=True,
                    rotation_padding=config.rotation_padding, im_vmax=config.im_vmax, im_vmin=config.im_vmin)
    for i, df_agg_each in enumerate(df_agg_alpha):
        save_results(df_agg_each, 
                     df_features, "all", config.ranking, config.alpha_dir[i], config.chrom, square_size, 
                     rm_idx, config.window_size, config.resolution, scale_range=config.scale_range, 
                     root=config.root, parameter_str=config.parameter_str,
                    hic_file=config.hic_file, normalization=config.normalization, plot=True,
                    rotation_padding=config.rotation_padding, im_vmax=config.im_vmax, im_vmin=config.im_vmin)
            # For the saliency thresholded only, plot diagnostic plots
    # print("\tWarning: diagnostic plots being generated! Comment out once finalized")
    # for i, df_agg_each in enumerate(df_agg_alpha):
    #     plot_top_k_diagnostic_parallel(df_agg_each, df_features,
    #                     K="all", im=im, im_oe=im_oe, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                     max_width_bin=max_width_bin, p_norm=p_norm, ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                     scale_range=config.scale_range, window_size=config.window_size, 
    #                     save_path=config.alpha_dir[i], num_cores=config.num_cores,
    #                     num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                     f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range, verbose=config.verbose)
    # For the saliency thresholded only, plot diagnostic plots
    # print("\tWarning: diagnostic plots being generated! Comment out once finalized")
    # plot_top_k_diagnostic_parallel(df_agg_saliency, df_features,
    #                 K=32, im=im, im_oe=im_oe, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                 max_width_bin=max_width_bin, p_norm=p_norm, ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                 scale_range=config.scale_range, window_size=config.window_size, 
    #                 save_path=config.saliency_dir, num_cores=config.num_cores,
    #                 num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                 f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range, verbose=config.verbose)
    total_time += time.time() - t0
    if config.verbose: print(f"Saving results... {time.time() - t0:.0f}s Done")
    if config.verbose: print(f"Total time elapsed: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    
    # PLOT RIDGES
    # if config.verbose: print("Plotting individual jets...")
    # if config.verbose: print("\tWarning: diagnostic plots do not have unmapped regions inserted! Coordinates will be slightly offset") 
    # if config.verbose: print("\tWarning: this may take a while; disable plotting option if needed")       
    # plot_top_k_diagnostic(df_agg, df_features,
    #                 K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                 ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                 scale_range=config.scale_range, window_size=config.window_size, 
    #                 save_path=config.save_sub_dir, plot_unique=False, 
    #                 num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                 f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range,
    #                 )
    # plot_top_k_diagnostic(df_agg_saliency, df_features,
    #                 K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                 ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                 scale_range=config.scale_range, window_size=config.window_size, 
    #                 save_path=config.saliency_dir, plot_unique=False, 
    #                 num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                 f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range,
    #                 )
    # for i, df_agg_each in enumerate(df_agg_alpha_sal):
    #     plot_top_k_diagnostic(df_agg_each, df_features,
    #                     K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                     ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                     scale_range=config.scale_range, window_size=config.window_size, 
    #                     save_path=config.alphasal_dir[i], plot_unique=False, 
    #                     num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                     f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range,
    #                     )
    # for i, df_agg_each in enumerate(df_agg_alpha):
    #     plot_top_k_diagnostic(df_agg_each, df_features,
    #                     K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #                     ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #                     scale_range=config.scale_range, window_size=config.window_size, 
    #                     save_path=config.alpha_dir[i], plot_unique=False, 
    #                     num_bins=config.num_bins, bin_size=config.bin_size, points_min=config.points_min, points_max=config.points_max, 
    #                     f_true_bed=None, tolerance=None, f_true_cns=None, angle_range=config.angle_range,
    #                     )
    # total_time += time.time() - t0
    # if config.verbose: print(f"Plotting... {time.time() - t0:.0f}s Done")


    # plot_distribution_diagnostic(df_agg, df_features, im, config.ranking, config.resolution, config.save_sub_dir) 
    # plot_corner_diagnostic(df_agg, df_features, K=config.top_k, ranking=config.ranking, im=im, 
    #                     im_corner=im_corner, corner_type=config.corner_type, C=C, 
    #                     scale_range=config.scale_range, resolution=config.resolution, 
    #                     save_path=config.save_sub_dir, root=config.root)
    # # Plot p-value AND saliency thresholded ridges
    # for i, df_agg_each in enumerate(df_agg_alpha_sal):
    #     plot_distribution_diagnostic(df_agg_each, df_features, im, config.ranking, config.resolution, config.alphasal_dir[i]) 
    #     plot_corner_diagnostic(df_agg_each, df_features, K=config.top_k, ranking=config.ranking, im=im, 
    #                         im_corner=im_corner, corner_type=config.corner_type, C=C, 
    #                         scale_range=config.scale_range, resolution=config.resolution, 
    #                         save_path=config.alphasal_dir[i], root=config.root)
    #     plot_top_k(df_agg_each, df_features, config.top_k, config.ranking, config.hic_file, config.chrom, config.resolution, 
    #             config.window_size, config.normalization, config.rotation_padding, config.alphasal_dir[i], 
    #             root=config.root, parameter_str=config.parameter_str)
    # # Plot p-value thresholded ridges
    # for i, df_agg_each in enumerate(df_agg_alpha):
    #     plot_distribution_diagnostic(df_agg_each, df_features, im, config.ranking, config.resolution, config.alpha_dir[i]) 
    #     plot_corner_diagnostic(df_agg_each, df_features, K=config.top_k, ranking=config.ranking, im=im, 
    #                         im_corner=im_corner, corner_type=config.corner_type, C=C, 
    #                         scale_range=config.scale_range, resolution=config.resolution, 
    #                         save_path=config.alpha_dir[i], root=config.root)
    # # Plot saliency thresholded ridges
    # plot_distribution_diagnostic(df_agg_saliency, df_features, im, config.ranking, config.resolution, config.saliency_dir) 
    # plot_corner_diagnostic(df_agg_saliency, df_features, K=config.top_k, ranking=config.ranking, im=im, 
    #                     im_corner=im_corner, corner_type=config.corner_type, C=C, 
    #                     scale_range=config.scale_range, resolution=config.resolution, 
    #                     save_path=config.saliency_dir, root=config.root)

    # # # Plot true EXACT 
    # plot_top_k_diagnostic(df_agg, df_features,
    #         K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    #         ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    #         scale_range=config.scale_range, window_size=config.window_size, 
    #         save_path=config.save_sub_dir, 
    #         plot_unique=False,
    #         num_bins=config.num_bins, bin_size=config.bin_size, 
    #         points_min=config.points_min, points_max=config.points_max,
    #         f_true_bed=None, tolerance=None, 
    #         f_true_cns=f_true_cns,  # EXACT
    #         angle_range=config.angle_range,
    #         )
    # # # Plot true APPROXIMATE 
    # # # plot_top_k_diagnostic(df_agg, df_features,
    # # #         K=config.top_k, im=im, im_p_value=im_p_value, corr_im_p_value=corr_im_p_value, I=I, D=D, A=A, W1=W1, W2=W2, R=R, C=C,
    # # #         ranking=config.ranking, resolution=config.resolution, chromosome=config.chrom,
    # # #         scale_range=config.scale_range, window_size=config.window_size, 
    # # #         save_path=config.save_sub_dir, 
    # # #         plot_unique=False,
    # # #         num_bins=config.num_bins, bin_size=config.bin_size, 
    # # #         points_min=config.points_min, points_max=config.points_max,
    # # #         f_true_bed=config.f_true, tolerance=500e3,                  # APPROXIMATE 
    # # #         f_true_cns=None, 
    # # #         angle_range=config.angle_range,
    # # #         )

if __name__ == "__main__":
    main()
