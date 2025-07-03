import argparse
import os
import io
import contextlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from utils.scale_space import generate_scales_from_widths
from utils.plotting import genomic_labels


# Config class to hold all parameters (dictionary)
@dataclass
class Config:
    # Input
    hic_file: str

    # Required parameters
    exp_type: str 
    chrom: str
    resolution: int
    alpha: List[str]

    # Extended parameters
    window_size: int
    data_type: str
    normalization: Optional[str] = None
    jet_widths: Optional[List[float]] = None
    angle_range: Optional[List[float]] = None
    saliency_thresh: Optional[float] = None
    thresholds: Optional[List[float]] = None # ImageJ hysteresis thresholds
    root_within: Optional[int] = None # Number of bins from diagonal for root of jet
    save_dir_root: Optional[str] = None
    folder_name: Optional[str] = None
    num_cores: Optional[int] = None
    verbose: Optional[bool] = None
    
    # Optional parameters
    scale_range: Optional[List[float]] = None

    # Internal/Fixed parameters
    gamma: Optional[float] = None
    ridge_method: Optional[int] = None
    rotation_padding: Optional[str] = None
    convolution_padding: Optional[str] = None
    noise_consec: Optional[str] = None
    noise_alt: Optional[str] = None
    sum_cond: Optional[str] = None
    agg: Optional[str] = None
    rem_k_strata: Optional[int] = None

    num_bins: Optional[int] = None # Entropy parameters
    bin_size: Optional[float] = None # Entropy parameters
    points_min: Optional[float] = None # Entropy parameters
    points_max: Optional[float] = None # Entropy parameters
    entropy_thresh: Optional[float] = None # Entropy parameters

    eps_r: Optional[float] = None # Corner parameters
    eps_c1: Optional[float] = None # Corner parameters
    eps_c2: Optional[float] = None # Corner parameters

    whiten: bool = False

    im_vmin: Optional[float] = None
    im_vmax: Optional[float] = None
    im_corner_vmin: Optional[float] = None
    im_corner_vmax: Optional[float] = None
    angle_trim: Optional[float] = None
    corner_trim: Optional[float] = None
    eig2_trim: Optional[float] = None
    ang_frac: bool = False
    rmse: Optional[float] = None

    f_true: Optional[str] = None
    save_dir: Optional[str] = None
    save_sub_dir: Optional[str] = None
    saliency_dir: Optional[str] = None
    alpha_dir: List[str] = field(default_factory=list)
    alphasal_dir: List[str] = field(default_factory=list)
    parameter_str: str = field(default_factory=str)
    root: Optional[str] = None

def assign_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Assigns default values to the arguments based on the experiment type
    Defaults are assigned first but then overwritten if user explicitly specifies them
    Args:
        args (argparse.Namespace): Parsed command line arguments
    Returns:
        argparse.Namespace: Updated arguments with defaults assigned
    """
    exp_defaults = {
        "hic": {
            "normalization":    "KR",
            "data_type":        "oe",
            "root_within":      10,
            "entropy_thresh":   0.5,
            "angle_trim":       0.5,
            "corner_trim":      0,
            "ang_frac":         True,
            "rmse":             0.01,
            "eig2_trim":        0.0,
            "im_vmin":          0,
            "im_vmax":          99,
            "im_corner_vmin":   0,
            "im_corner_vmax":   98,
        },
        "replihic": {
            "normalization":    "VC_SQRT",
            "data_type":        "observed",
            "root_within":      None,
            "entropy_thresh":   None,
            "angle_trim":       0.5,
            "corner_trim":      None,
            "ang_frac":         True,
            "rmse":             None,
            "eig2_trim":        None,
            "im_vmin":          0,
            "im_vmax":          100,
            "im_corner_vmin":   0,
            "im_corner_vmax":   100,
        }
    }
    # Assign defaults
    defaults = exp_defaults[args.exp_type]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    return args

def process_args(args: argparse.Namespace) -> Config:
    """
    Processes arguments and creates a Config object with all parameters
    Args:
        args (argparse.Namespace): Parsed command line arguments
    Returns:
        Config: Configuration object with all parameters
    """
    config = Config(
        exp_type=args.exp_type,
        hic_file=args.hic_file,
        chrom=args.chrom,
        normalization=args.normalization,
        resolution=args.resolution,
        data_type=args.data_type,
        thresholds=args.thresholds,
        window_size=args.window_size,
        scale_range=args.scale_range,
        gamma=args.gamma,
        ridge_method=args.ridge_method,
        rotation_padding=args.rotation_padding,
        convolution_padding=args.convolution_padding,
        angle_range=args.angle_range,   
        noise_consec=args.noise_consec,   
        noise_alt=args.noise_alt,   
        sum_cond=args.sum_cond,
        agg=args.agg,   
        folder_name=args.folder_name,
        save_dir_root=args.save_dir_root,
        num_cores=args.num_cores,
        verbose=args.verbose,
        rem_k_strata=args.rem_k_strata,
        root_within=args.root_within,
        num_bins=args.num_bins,
        bin_size=args.bin_size,
        points_min=args.points_min,
        points_max=args.points_max,
        entropy_thresh=args.entropy_thresh,
        eps_r=args.eps_r,
        eps_c1=args.eps_c1,
        eps_c2=args.eps_c2,
        whiten=args.whiten,
        im_vmax=args.im_vmax,
        im_vmin=args.im_vmin,
        im_corner_vmax=args.im_corner_vmax,
        im_corner_vmin=args.im_corner_vmin,
        angle_trim=args.angle_trim,
        corner_trim=args.corner_trim,
        eig2_trim=args.eig2_trim,
        ang_frac=args.ang_frac,
        alpha=args.alpha,
        saliency_thresh=args.saliency_thresh,
        rmse=args.rmse,
        f_true=args.f_true
    )

    if config.data_type == "observed" and config.exp_type == "hic":
        # if Hi-C and observed, then deactivate eig2 trimming, which is only suited for OE
        config.eig2_trim = None

    if config.jet_widths is not None:
        # Compute scale range from jet widths
        config.scale_range = generate_scales_from_widths(w0=config.jet_widths[0], w1=config.jet_widths[1], base=1.5, scale_resolution=0.25)
        
    elif config.scale_range is not None:
        # Directly take in the scale range specified
        config.scale_range = np.array(args.scale_range)
    else:
        # Both are none
        # Then use the development mode scale_range 
        config.scale_range = np.logspace(1, 7, num=24, base=1.5)
        print(f"Warning: jet_widths or scale_range is not specified, using development mode scale_range: {config.scale_range}")

    # Main direcotry (level 1): ImageJ parameters
    hic_file_name = os.path.basename(config.hic_file)
    # hic_file_name = hic_file_name.split(".")[0] # bug if hic_file_name has multiple dots, e.g. "file.name.hic"
    hic_file_name = os.path.splitext(hic_file_name)[0]  # Remove file extension

    # generate save name root to append to every level 1 files
    config.root = f"{hic_file_name}_{config.chrom}_{genomic_labels(config.resolution)}"

    if config.folder_name is None:
        # If not specified, use the Hi-C file name for the folder name + chromosome and resolution
        config.folder_name = hic_file_name + f"_{config.chrom}_{genomic_labels(config.resolution)}"
    else:
        # Otherwise, just append the chromosome and resolution
        config.folder_name += f"_{config.chrom}_{genomic_labels(config.resolution)}"
    
    config.save_dir = os.path.join(config.save_dir_root, config.folder_name)

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    else:
        print(f"WARNING: Save directory {config.save_dir} already exists")
        print("If you changed the parameters, change the `folder_name` or `save_root_dir` parameter")
        # sys.exit(1) 

    # Sub directory (level 2): all results
    config.save_sub_dir = os.path.join(config.save_dir, f"{config.root}_results_all")

    if not os.path.exists(config.save_sub_dir):
        os.makedirs(config.save_sub_dir)

    # Sub directory (level 2): results after saliency thresholding
    sub_dir = os.path.join(config.save_dir, f"{config.root}_results_saliency-{config.saliency_thresh}")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    config.saliency_dir = sub_dir

    # Sub directories (level 2): results for each alpha and alpha AND saliency thresholding
    # Notably, saliency thresholding is applied before alpha thresholding
    if config.alpha is not None:
        for a in config.alpha:
            sub_dir = os.path.join(config.save_dir, f"{config.root}_results_p-{a}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            config.alpha_dir.append(sub_dir)

            sub_dir = os.path.join(config.save_dir, f"{config.root}_results_saliency-{config.saliency_thresh}-p-{a}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            config.alphasal_dir.append(sub_dir)

    # This is the column name of the jet saliency score in the summary table
    config.ranking = "jet_saliency"

    # Generate parameter string from print_parameters function output
    # The parameter string is inserted into summary and expanded tables as a comment
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_parameters(config)
    config.parameter_str = buf.getvalue()

    return config



def print_parameters(config: Config):
    """Print the configuration parameters"""
    if config.verbose:
        # Inputs
        print("Inputs")
        print("* Hi-C file (.hic):", config.hic_file)

        # Required Parameters
        print("\nRequired Parameters")
        print("* Experiment type:", config.exp_type)
        print("* Chromosome:", config.chrom)
        print("* Resolution:", config.resolution)
        print("* Save directory root:", config.save_dir_root)

        # Extended Parameters
        print("\nExtended Parameters")
        print("* Significance threshold(s):", config.alpha)
        print("* Window size:", config.window_size)                          
        print("* Normalization:", config.normalization)                      
        print("* Data type:", config.data_type)    
        print("* Jet widths (if specified):", config.jet_widths)                     
        print("* Angle range:", config.angle_range)                          
        print("* Saliency threshold on zero removed:", config.saliency_thresh) 
        print("* Hysteresis thresholding parameters:", config.thresholds)     
        print("* Root within:", config.root_within)                          
        print("* Folder name:", config.folder_name)                    
        print("* Save directory:", config.save_dir)
        print("* Number of cores:", config.num_cores)                         
        print("* Verbose:", config.verbose)                                   

        # Optional Parameters
        print("\nOptional Parameters")
        print("* Scale range:", config.scale_range)  
        # Fixed Parameters
        print("\nFixed Parameters")
        print("* Entropy threshold:", config.entropy_thresh)    
        print("* Angle trim:", config.angle_trim)               
        print("* Corner trim:", config.corner_trim)             
        print("* Angle fraction:", config.ang_frac)             
        print("* Normalized RMSE:", config.rmse)                
        print("* Eig2 trim:", config.eig2_trim)                 
        print("* Whiten:", config.whiten)                       
        print("* Gamma:", config.gamma)                         
        print("* Ridge method:", config.ridge_method)           
        print("* Sum condition:", config.sum_cond)              
        print("* Rotation padding:", config.rotation_padding)   
        print("* Convolution padding:", config.convolution_padding)  
        print("* Noise consecutive:", config.noise_consec)       
        print("* Noise alternating:", config.noise_alt)          
        print("* Aggregation function:", config.agg)             
        print("* Remove k-th strata:", config.rem_k_strata)      
        print("* Corner condition ridge (eps_r):", config.eps_r)      
        print("* Corner condition 1 (eps_c1):", config.eps_c1)  
        print("* Corner condition 2 (eps_c2):", config.eps_c2)  
        print("* Bin size (entropy):", config.bin_size)         
        print("* Number of bins (entropy):", config.num_bins)   
        print("* Points minimum (entropy):", config.points_min) 
        print("* Points maximum (entropy):", config.points_max) 
        print("* Image vmin, vmax:", (config.im_vmin, config.im_vmax))               
        print("* Corner image vmin, vmax:", (config.im_corner_vmin, config.im_corner_vmax))  