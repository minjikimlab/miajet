import argparse


def parse_args():

    # Instantiate the argument parser
    parser = argparse.ArgumentParser(prog="MIA_jet", description="Find jets in Hi-C or Repli Hi-C data")  


    # Command line interface option
    # Inputs
    parser.add_argument("hic_file", type=str, help="Path to Hi-C data file (.hic or .mcool)")

    # Required Parameters
    parser.add_argument("--exp_type", type=str, required=True, choices=["hic", "replihic"],
                        help="Experiment type. 'hic' for Hi-C or 'replihic' for Repli Hi-C")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome (e.g. 'chr1')")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Hi-C resolution in base pairs (e.g. 50000 for 50 kbp)")
    parser.add_argument("--save_dir_root", type=str, required=True, default=None,
                        help="Absolute path to directory where results will be saved")

    # Extended Parameters
    parser.add_argument("--alpha", nargs="+", type=float, required=False, default=[0.1, 0.05],
                        help="Alpha or a list of alpha values for p-value cutoffs (default: 0.1 0.05)")
    parser.add_argument("--window_size", type=int, required=False, default=6_000_000,
                        help="Distance from main diagonal (default: 6_000_000 for 6 Mbp)")
    parser.add_argument("--normalization", type=str, required=False, default=None,
                        help="Hi-C normalization method (e.g. 'KR', 'VC_SQRT')")
    parser.add_argument("--data_type", type=str, required=False, choices=["observed", "oe"], default=None,
                        help="Hi-C data type (default: 'observed')")
    parser.add_argument("--thresholds", nargs="+", type=float, required=False, default=[0.01, 0.05],
                        help="The lower and upper thresholds for ImageJ Curve Tracing plugin (default: 0.01 0.05)")
    parser.add_argument("--angle_range", nargs="+", required=False, type=float, default=[80, 100],
                        help="Angle lower and upper bound of jets in degrees with 90˚ being the secondary diagonal of contact map (default: 80 100)")
    parser.add_argument("--saliency_thresh", default=90, type=float,
                        help="Percentile for saliency thresholding. Percentile is computed from non-zero saliency values only (default: 90)")
    parser.add_argument("--jet_widths", nargs="+", required=False, type=float, default=None,
                        help="The lower and upper bound of widths in pixels of jets to be detected"
                        "If not specified, a default scale range will be used: logspace 1.5^1 to 1.5^7 with 24 increments")
    parser.add_argument("--root_within", type=int, required=False, default=None,
                        help="Enforce the root of any ridge to be ≤ certain number of bins to main diagonal")
    parser.add_argument("--folder_name", type=str, required=False, default=None,
                        help="Folder name to store generated files. Defaults to the Hi-C file name without extension.")
    parser.add_argument("--num_cores", type=int, required=False, default=None,
                        help="Number of CPU cores available (default: 1)")
    parser.add_argument("--verbose", action="store_true", default=None, help="Print details")

    # Optional Parameters
    parser.add_argument("--scale_range", nargs="+", required=False, type=float, default=None,
                        help="Standard deviations of Gaussian blurs in scale space (list)"
                        "This parameter is alternative to jet_widths, and if specified, will override jet_widths."
                        "It is recommended that scales are in logspace")
    # Fixed Parameters
    parser.add_argument("--gamma", type=float, required=False, default=0.75,
                        help="Gamma for scale space between 0 and 1 (default: 0.75)")
    parser.add_argument("--ridge_method", type=int, required=False, choices=[1, 2, 3, 4, 5, 6, 7], default=1,
                        help="Ridge strength method (1 (D1), 2 (D2), 3 (D3), 5 (D5), 6 (D6)) (default: 1)")
    parser.add_argument("--rotation_padding", type=str, required=False,
                        choices=["reflect", "grid-mirror", "constant", "grid-constant", "nearest", "mirror", "grid-wrap", "wrap"],
                        default="nearest", help="Padding method for scipy.ndimage.rotate (default: 'nearest')")
    parser.add_argument("--convolution_padding", type=str, required=False, choices=["reflect", "constant", "nearest", "mirror", "wrap"],
                        default="nearest", help="Padding method for scipy.ndimage.correlate convolution (default: 'nearest')")
    parser.add_argument("--sum_cond", required=False, type=str, default="a-r",
                        choices=["a", "r", "c", "a-r", "a-c", "r-c", "a-r-c"],
                        help=("Which conditions to sum for the saliency score (default: 'a-r'). 'a' (angle only), "
                            "'r' (ridge only), 'a-r' (combined angle & ridge) "
                            "'a-r-c' (combined angle & ridge & corner). "))
    parser.add_argument("--noise_consec", required=False, type=str, default="",
                        help=("Noise adjustment for consecutive true (default: ''). Format: 'INTEGER-TYPE', where INTEGER is the "
                            "number of consecutive True values needed and TYPE is one of: 'a' (angle only), "
                            "'r' (ridge only), or 'a-r' (combined angle & ridge)."))
    parser.add_argument("--noise_alt", required=False, type=str, default="",
                        choices=["", "a", "r", "c", "a-r", "a-c", "r-c", "a-r-c"],
                        help=("Noise adjustment for alternating 01 normalization (default: ''). Selects the conditions for normalization: "
                            "'a' (angle only), 'r' (ridge only), or 'a-r' (combined angle & ridge)."))
    parser.add_argument("--agg", required=False, type=str, default="sum", choices=["sum", "mean"],
                        help="Aggregation function of computing jet salinecy score (default: 'sum')")
    parser.add_argument("--rem_k_strata", type=int, required=False, default=1,
                        help="Removes positions of jets within k-th off diagonal strata (default: 1)")
    parser.add_argument("--num_bins", type=int, required=False, default=10,
                        help="Number of bins to use for entropy histogram (default: 10). If not specified, None")
    parser.add_argument("--bin_size", type=float, required=False, default=None,
                        help="Bin size for entropy histogram; if not specified, None")
    parser.add_argument("--points_min", type=float, required=False, default=0,
                        help="Minimum data range for entropy histogram (default: 0)")
    parser.add_argument("--points_max", type=float, required=False, default=0.04,
                        help="Maximum data range for entropy histogram (default: 0.04)")
    parser.add_argument("--entropy_thresh", type=float, required=False, default=None,
                        help="Normalized entropy threshold (default: None)")
    parser.add_argument("--eps_r", type=float, required=False, default=0.0005,
                        help="Epsilon value for ridge (default: 0.0005)")
    parser.add_argument("--eps_c1", type=float, required=False, default=0.1,
                        help="Epsilon value for condition 1 (default: 0.1)")
    parser.add_argument("--eps_c2", type=float, required=False, default=1e-5,
                        help="Epsilon value for condition 2 (default: 1e-5)")
    parser.add_argument("--whiten", type=float, required=False, default=None,
                        help="Whether to whiten image, effectively removing correlation of the image map (default: None)"
                            "If not None, give the float of the epsilon of the ZCA whitening (typically 1e-5 but may require adjustment)")
    parser.add_argument("--im_vmax", type=float, required=False, default=None,
                        help="The percentile (0-100 scale) for the maximum intensity range of Hi-C image")
    parser.add_argument("--im_vmin", type=float, required=False, default=None,
                        help="The percentile (0-100 scale) for the minimum intensity range of Hi-C image")
    parser.add_argument("--im_corner_vmax", type=float, required=False, default=None,
                        help="The percentile (0-100 scale) for the maximum intensity range of corner image")
    parser.add_argument("--im_corner_vmin", type=float, required=False, default=None,
                        help="The percentile (0-100 scale) for the minimum intensity range of corner image")
    parser.add_argument("--angle_trim", type=float, required=False, default=None,
                        help="Angle trim for ridges according to range specified by angle_range option"
                            "If None, then no trimming is performed"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified"
                            )
    parser.add_argument("--corner_trim", type=float, required=False, default=None,
                        help="Corner trim for ridges (note: corner must be identified at ALL scales in order to be trimmed at the location)"
                            "If None, then no trimming is performed"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified"
                            )
    parser.add_argument("--eig2_trim", type=float, required=False, default=None,
                        help="Eigenvalue 2 trim for ridges"
                            "If None, then no trimming is performed"
                            "If a float (0.0-1.0), then the minimum possible length of ridge is the fraction specified of the original length"
                            "If an integer (>=1), then the minimum possible length of ridge is the integer specified"
                            )
    parser.add_argument("--ang_frac", action="store_true", default=None,
                        help="Whether to use the angle fraction multipliers to the saliency (default: False)")
    parser.add_argument("--rmse", type=float, required=False, default=None,
                        help="Normalized RMSE threshold (default: None)")
    parser.add_argument("--f_true", type=str, required=False, default=None,
                        help="True bed file to merge")
    
    return parser.parse_args()


