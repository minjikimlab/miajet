import argparse

_MISSING = argparse.SUPPRESS

def none_or_float(x):
    if isinstance(x, str) and x.lower() == 'none':
        return None
    try:
        return float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {x}. Use 'None' for no value.")

def none_or_int(x):
    if isinstance(x, str) and x.lower() == 'none':
        return None
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: {x}. Use 'None' for no value.")
 
def none_or_str(x):
    if isinstance(x, str) and x.lower() == 'none':
        return None
    return str(x)

def parse_args():

    # Instantiate the argument parser
    parser = argparse.ArgumentParser(prog="miajet", description="Find jets in Hi-C or Repli Hi-C data")  


    # Command line interface option
    # Inputs
    parser.add_argument("hic_file", type=str, help="Path to Hi-C data file (.hic or .mcool)")

    parser.add_argument("--exp_type", type=str, required=True, choices=["hic", "replihic"],
                        help="Experiment type. 'hic' for Hi-C or 'replihic' for Repli Hi-C")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome (e.g. 'chr1')")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Hi-C resolution in base pairs (e.g. 50000 for 50 kbp)")
    parser.add_argument("--save_dir_root", type=str, required=True, default=None,
                        help="Absolute path to directory where results will be saved")
    parser.add_argument("--window_size", type=int, required=True, default=None,
                        help="Distance from main diagonal (see recommended window_size in README across resolutions)") 
    parser.add_argument("--compartment", type=str, choices=["True", "False"], default=_MISSING,
                        help="Whether the data contains A/B compartments or not. Either 'True' or 'False'."
                        "(default: 'True' if exp_type='hic', 'False' if exp_type='replihic)")
    
    parser.add_argument("--q_val", type=float, required=False, default=_MISSING,
                        help="Threshold for corrected p-value cutoffs on Hi-C data. "
                        "(default: 0.1 if exp_type='hic', 0.2 if exp_type='replihic'). ") 
    parser.add_argument("--q_val_white", type=float, required=False, default=0.95,
                    help="Threshold for corrected q-value cutoffs on Hi-C data after removing A/B compartments. "
                        "(default: 0.95 if compartment='True', feature disabled if compartment='False'). ") 
    parser.add_argument("--normalization", type=str, required=False, default=_MISSING,
                        help="Hi-C normalization method (e.g. 'KR', 'VC_SQRT', 'NONE'). "
                        "(default: 'KR' if exp_type='hic', 'VC_SQRT' if exp_type='replihic). ") 
    parser.add_argument("--data_type", type=str, required=False, choices=["observed", "oe"], default=_MISSING,
                        help="Hi-C data type either 'observed' or 'oe' (observed/expected). "
                        "(default: 'oe' if exp_type='hic', 'observed' if exp_type='replihic). ") 
    parser.add_argument("--thresholds", nargs="+", type=float, required=False, default=None,
                        help="The lower and upper thresholds for ImageJ Curve Tracing plugin. "
                        "If None, automatically generates suggested thresholds based on scale_range or jet_widths. "
                        "(default: None)") 
    parser.add_argument("--angle_range", nargs="+", required=False, type=float, default=_MISSING,
                        help="Angle lower and upper bound of jets in degrees with 90˚ being a typical jet and 45˚ or 135˚ being a stripe. "
                        "(default: 60 120 if exp_type='hic', 80 100 if exp_type='replihic')") # 
    parser.add_argument("--jet_widths", nargs="+", required=False, type=float, default=None,
                        help="The lower and upper bound of jet widths to be detected (unit is in pixels of the image i.e., bins). "
                        "This parameter is alternative to scale_range, and if specified, will override scale_range. "
                        "If not specified, a default scale range will be used: logspace 1.5^1 to 1.5^7 with 24 increments") 
    parser.add_argument("--root_within", type=float, required=False, default=_MISSING,
                        help="Enforce the closest point of the jet to the main diagonal to be within a certain genomic distance. "
                        "Otherwise, the jet is filtered out.\n" 
                        "* If root_within = 1, then all jets are kept regardless of their distance to the main diagonal.\n"
                        "* If root_within < 1, then it is fraction of the window size.\n"
                        "* If root_within >= 1, then it is the number of bins. \n"
                        "(default: 12 if exp_type='hic', 0.5 if exp_type='replihic). ")
    parser.add_argument("--root_within_comp", type=none_or_float, required=False, default=None,
                        help="Does not trim jets that go across A/B compartment that are ≤ certain number of bins to main diagonal. "
                        "This is to prevent some real jets that are close to the main diagonal from being trimmed. "
                        "Similarly to root_within, if root_within_comp <= 1 then its interpreted as a fraction of the window size. "
                        "* If root_within >= 1, then it is the number of bins. \n"
                        "(default: identical to `root_within` if compartment='True', feature disabled if compartment='False'). ")
    parser.add_argument("--folder_name", type=none_or_str, required=False, default=None,
                        help="Folder name to store generated files. Defaults to the Hi-C file name without extension.")
    parser.add_argument("--num_cores", type=int, required=False, default=1,
                        help="Number of CPU cores available (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Print details")
    parser.add_argument("--diagnostic_plots", action="store_true", help="Print diagnostic plots at every major step.")

    parser.add_argument("--scale_range", nargs="+", required=False, type=float, default=None,
                        help="Standard deviations of Gaussian blurs in scale space (list e.g., 1,1.5,2,3,5)"
                        "This parameter is alternative to jet_widths. It is recommended that scales are in logspace. "
                        "(default: logspace 1.5^1 to 1.5^7 with 24 increments)")
    parser.add_argument("--gamma", type=float, required=False, default=0.75,
                        help="Gamma for scale space between 0 and 1. (default: 0.75)")
    parser.add_argument("--ridge_method", type=int, required=False, choices=[1, 2, 3, 4, 5, 6, 7], default=1,
                        help="Ridge strength method (1, 2, 3). (default: 1)")
    parser.add_argument("--rotation_padding", type=str, required=False,
                        choices=["reflect", "grid-mirror", "constant", "grid-constant", "nearest", "mirror", "grid-wrap", "wrap"],
                        default="nearest", help="Padding method for scipy.ndimage.rotate. (default: 'nearest')")
    parser.add_argument("--convolution_padding", type=str, required=False, choices=["reflect", "constant", "nearest", "mirror", "wrap"],
                        default="nearest", help="Padding method for scipy.ndimage.correlate convolution. (default: 'nearest')")
    parser.add_argument("--resolve_conflict", type=str, required=False, default="combined", 
                        choices=["length", "p-val", "p-val_white", "saliency", "avg_width", "sum_consistency", "sum_consistency_im", "blobness", "angle_turbulence", "combined"],
                    help="The jet statistic to minimize or maximize among overlapping jets. Needs to be a column in the summary dataframe. "
                        "(default: 'blobness')")     
    parser.add_argument("--rem_k_strata", type=int, required=False, default=1,
                        help="Removes positions of jets within k-th off diagonal strata. (default: 1)")
    parser.add_argument("--angle_trim", type=none_or_float, required=False, default=_MISSING,
                        help="Splits ridges if angle deviates from the range specified by angle_range parameter\n"
                            "If None, then no splitting is performed\n"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length\n"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified\n"
                            "(default: 3 if exp_type='hic', 0.5 if exp_type='replihic). ")
    parser.add_argument("--scale_dec_trim", type=none_or_float, required=False, default=_MISSING,
                        help="Splits ridges if there is a large decrease in scale values along ridge\n"
                            "If None, then no splitting is performed\n"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length\n"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified\n"
                            "(default: 3 if exp_type='hic', None if exp_type='replihic). ")
    parser.add_argument("--scale_dec_thresh_trim", type=int, required=False, default=10,
                    help="Number of scales that must decrease for a ridge to be split. "
                        "Must be less than the number of scales in scale_range\n"
                        "(default: 10 if exp_type='hic', feature disabled if exp_type='replihic). "
                        )
    parser.add_argument("--scale_trim", type=none_or_float, required=False, default=_MISSING,
                        help="Splits ridges if there is a large deviation in scale in a window specified by scale_trim_window param\n"
                            "If None, then no splitting is performed\n"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length\n"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified\n"
                            "(default: 0.25 if exp_type='hic', None if exp_type='replihic). ")
    parser.add_argument("--scale_trim_thresh", type=float, required=False, default=3,
                        help="The threshold for scale split in units of scales. "
                            "It is the standard deviation in a window specified by scale_trim_window param\n"
                            "(default: 3 if exp_type='hic', feature disabled if exp_type='replihic). ")
    parser.add_argument("--scale_trim_window", type=int, required=False, default=5,
                        help="The window size for scale trim. "
                        "(default: 5 if exp_type='hic', feature disabled if exp_type='replihic).")    
    parser.add_argument("--comp_trim", type=none_or_float, required=False, default=0.25,
                        help="Splits ridges to prevent ridges from going *through* A/B compartments\n"
                            "If None, then no splitting is performed\n"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length\n"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified\n"
                            "(default: 3 if compartment='True', feature disabled if compartment='False'). "
                            )
    parser.add_argument("--ang_frac", type=str, default=_MISSING, choices=["True", "False"],
                        help="Whether to turn off the angle fraction multipliers to the saliency. "
                        "(default: 'True' if exp_type='hic', 'False' if exp_type='replihic). "
                        )
    parser.add_argument("--adj_nondec", type=str, default=_MISSING, choices=["True", "False"],
                        help="Whether to turn off the adjacent non-decreasing criteria (True if unspecified)"
                        "(default: 'True' if exp_type='hic', 'False' if exp_type='replihic). ")    
    
    # Filter parameters
    parser.add_argument("--angle_turbulence", type=none_or_float, required=False, default=_MISSING,
                        help="Filters according to the coefficient of variation of the jet 'angle' values"
                        "(default: 0.325 if exp_type='hic', feature disabled if exp_type='replihic'). ")
    parser.add_argument("--blobness", type=none_or_float, required=False, default=2.0,  
                        help="Filters according to blobness, which is the ratio between the maximum width and length of jet."
                        "(default: 2.0) i.e., widths can be at most 2 times the length. This is to filter out blobs that are not jet-like. ")
    parser.add_argument("--consistency", type=none_or_float, required=False, default=_MISSING,  
                        help="Filters according to consistency. (default: 0.6 if exp_type='hic', feature disabled if exp_type='replihic). ")
    parser.add_argument("--sum_consistency", type=none_or_float, required=False, default=None,
                        help="Filters according to sum_consistency. (default: None)")
    parser.add_argument("--sum_consistency_im", type=str, default=_MISSING, choices=["True", "False"],
                        help="Filters according to sum_consistency_im. "
                        "This is to filter out false positive jets that are in sparse, noisy regions."
                        "(default: 'False' if exp_type='hic', 'True' if exp_type='replihic). ")
    parser.add_argument("--ridge_strength_turbulence", type=none_or_float, required=False, default=0.9, 
                        help="Filters according to the coefficient of variation of the jet 'ridge_strength' values."
                        "(default: 0.9)")
    parser.add_argument("--angle_satisfied", type=none_or_float, required=False, default=0.3, 
                        help="Filters according to the fraction of points in jet that lie in the angle_range specified. "
                        "(default: 0.3)")
    parser.add_argument("--length", type=none_or_float, required=False, default=5,
                        help="Filters according to minimum length of jet. (default: 5)")


    # Debug parameters
    parser.add_argument("--chip_file", type=str, required=False, default=None,
                        help="Path to the chip file for debugging purposes.")
    parser.add_argument("--chrom_size_file", type=str, required=False, default=None,
                        help="Path to the chromosome size file for ChIP-seq.")

    
    return parser.parse_args()


