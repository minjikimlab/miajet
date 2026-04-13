import argparse
import os
import io
import contextlib
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from utils.scale_space import generate_scales_from_widths, print_scale_width_conversion
from utils.plotting import genomic_labels
from .tee import set_logging_file
from miajet._version import __version__

_MISSING = argparse.SUPPRESS

@dataclass
class Config:
    # Input
    hic_file: str

    # Required parameters
    exp_type: str
    chrom: str
    resolution: int
    save_dir_root: Optional[str] = None
    compartment: Optional[str] = None

    # Extended parameters
    q_val: Optional[float] = None
    q_val_white: Optional[float] = None
    window_size: Optional[int] = None
    data_type: Optional[str] = None
    normalization: Optional[str] = None
    jet_widths: Optional[List[float]] = None
    angle_range: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None
    root_within: Optional[float] = None
    root_within_comp: Optional[float] = None
    folder_name: Optional[str] = None
    num_cores: Optional[int] = None
    verbose: Optional[bool] = None
    diagnostic_plots: Optional[bool] = None

    # Optional parameters
    scale_range: Optional[List[float]] = None

    # Fixed parameters
    gamma: Optional[float] = None
    ridge_method: Optional[int] = None
    rotation_padding: Optional[str] = None
    convolution_padding: Optional[str] = None
    rem_k_strata: Optional[int] = None
    resolve_conflict: Optional[str] = None

    # Trim parameters
    angle_trim: Optional[float] = None
    scale_dec_trim: Optional[float] = None
    scale_dec_thresh_trim: Optional[int] = None
    scale_trim: Optional[float] = None
    scale_trim_thresh: Optional[float] = None
    scale_trim_window: Optional[int] = None
    comp_trim: Optional[float] = None
    ang_frac: Optional[str] = None
    adj_nondec: Optional[str] = None

    # Internal/derived (not in parser, but used at runtime)
    save_dir: Optional[str] = None
    save_sub_dir: Optional[str] = None
    dir_thresholded: Optional[str] = None
    dir_alpha: Optional[str] = None
    parameter_str: str = field(default_factory=str)
    root: Optional[str] = None

    angle_turbulence: Optional[float] = None  # TODO: confirm type
    blobness: Optional[float] = None  # TODO: confirm type
    consistency: Optional[float] = None  # TODO: confirm type
    sum_consistency: Optional[float] = None  # TODO: confirm type
    sum_consistency_im: Optional[str] = None  # TODO: confirm type

    ridge_strength_turbulence: Optional[float] = None  # TODO: confirm type
    angle_satisfied: Optional[float] = None  # TODO: confirm type
    length: Optional[float] = None  # TODO: confirm type

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
            "normalization": "KR",
            "data_type": "oe",
            "compartment": "True",
            "angle_range": [60, 120],
            "q_val": 0.1,
            "root_within": 12,
            "angle_trim": 3, # 3 bins
            "adj_nondec": "True", # Consistency measure with Guo et al. style jets
            "ang_frac": "True", 
            "scale_trim": 0.25,
            "scale_dec_trim": 3,
            "angle_turbulence": 0.325, # Filter param
            "consistency": 0.6,
            "sum_consistency_im": "False",
        },
        "replihic": {
            "normalization": "VC_SQRT",
            "data_type": "observed",
            "compartment": "False",
            "angle_range": [80, 100],
            "q_val": 0.2,
            "root_within": 0.5,
            "angle_trim": 0.5,
            "adj_nondec": "False", 
            "ang_frac": "False",
            "scale_trim": None,
            "scale_dec_trim": None,
            "angle_turbulence": None, # Filter param
            "consistency": None,
            "sum_consistency_im": "True",
        }
    }
    # Assign defaults
    defaults = exp_defaults[args.exp_type]
    for key, val in defaults.items():
        # if not hasattr(args, key):
        #     setattr(args, key, val)

        if getattr(args, key, _MISSING) is _MISSING:
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
        jet_widths=args.jet_widths,
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
        folder_name=args.folder_name,
        save_dir_root=args.save_dir_root,
        num_cores=args.num_cores,
        verbose=args.verbose,
        diagnostic_plots=args.diagnostic_plots,
        compartment=args.compartment,
        root_within=args.root_within,
        root_within_comp=args.root_within_comp,
        angle_trim=args.angle_trim,
        scale_dec_trim=args.scale_dec_trim,
        scale_dec_thresh_trim=args.scale_dec_thresh_trim,
        scale_trim=args.scale_trim,
        scale_trim_thresh=args.scale_trim_thresh,
        scale_trim_window=args.scale_trim_window,
        comp_trim=args.comp_trim,
        ang_frac=args.ang_frac,
        resolve_conflict=args.resolve_conflict,
        adj_nondec=args.adj_nondec,
        q_val=args.q_val,
        q_val_white=args.q_val_white,
        rem_k_strata=args.rem_k_strata,
        angle_turbulence=args.angle_turbulence,
        blobness=args.blobness,
        consistency=args.consistency,
        sum_consistency=args.sum_consistency,
        sum_consistency_im=args.sum_consistency_im,
        ridge_strength_turbulence=args.ridge_strength_turbulence,
        angle_satisfied=args.angle_satisfied,
        length=args.length,
    )

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

    # Logging file requires config.save_dir, config.root
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        set_logging_file(config) # log output to unique file 
    else:
        set_logging_file(config) # log output to unique file 
        print(f"WARNING: Save directory {config.save_dir} already exists")
        print("If you changed the parameters, change the `folder_name` or `save_root_dir` parameter")

    print("*" * 40)
    print(f"miajet version: {__version__}") # print versions
    
    if config.verbose:
        print("*" * 40)
        print("Processing arguments and assigning defaults if not specified...")

    if config.root_within == 0:
        # All jets will be trimmed. Exit program.
        print("WARNING: root_within=0, all jets will be trimmed. Please set root_within to a positive value to keep some jets. Exiting program.")
        sys.exit(1)
    elif config.root_within <= 1:
        # Interpret as fraction of the window size
        window_size_bins = np.ceil((config.window_size / config.resolution) / np.sqrt(2)).astype(int)
        root_within_in = config.root_within
        config.root_within = np.ceil(config.root_within * window_size_bins).astype(int)
        if config.verbose:
            print(f"Interpreting root_within={root_within_in} as fraction of window size={genomic_labels(config.window_size)}, "
                  f"which corresponds to {genomic_labels(root_within_in * config.window_size)} or {config.root_within} bins.")
    else:
        # Interpret as bins but convert to genomic distance for diagnostics
        # config.root_within = np.ceil(config.root_within / config.resolution / np.sqrt(2)).astype(int) # converting genomic distance to bins
        config.root_within = int(config.root_within) # interpret directly as bins
        root_within_bp = config.root_within * config.resolution * np.sqrt(2) # convert bins to genomic distance 
        if config.verbose:
            print(f"Interpreting root_within={config.root_within} as bins, "
                  f"which corresponds to {genomic_labels(root_within_bp)} or {root_within_bp / config.window_size * 100:.3g}% of window size={genomic_labels(config.window_size)}.")


    if args.compartment.lower() == "true":
        config.compartment = True

        if config.root_within_comp is None:
            # The default is the same as root_within if in compartment mode
            # unless it is explicitly specified
            config.root_within_comp = config.root_within

        # If numeric then interpret as user-specified
        elif config.root_within_comp <= 1:
            # Interpret as fraction of the window size
            window_size_bins = np.ceil((config.window_size / config.resolution) / np.sqrt(2)).astype(int)
            root_within_comp_in = config.root_within_comp
            config.root_within_comp = np.ceil(config.root_within_comp * window_size_bins).astype(int)
            if config.verbose:
                print(f"Interpreting root_within_comp={root_within_comp_in} as fraction of window size={genomic_labels(config.window_size)}, "
                      f"which corresponds to {genomic_labels(root_within_comp_in * config.window_size)} or {config.root_within_comp} bins.")
        else:
            # Interpret as bins but convert to genomic distance for diagnostics
            # config.root_within_comp = np.ceil(config.root_within_comp / config.resolution / np.sqrt(2)).astype(int) # converting genomic distance to bins
            config.root_within_comp = int(config.root_within_comp) # interpret directly as bins
            root_within_comp_bp = config.root_within_comp * config.resolution * np.sqrt(2) # convert bins to genomic distance 
            if config.verbose:
                print(f"Interpreting root_within_comp={config.root_within_comp} as bins, "
                      f"which corresponds to {genomic_labels(root_within_comp_bp)} or "
                      f"{root_within_comp_bp / config.window_size * 100:.3g}% of window size={genomic_labels(config.window_size)}.")

    elif args.compartment.lower() == "false":
        config.compartment = False
        config.comp_trim = None # Must be disabled if not in compartment mode

        if config.resolve_conflict == "p-val_white":
            # Then lets default to the "blobness"
            if config.verbose:
                print("resolve_conflict is set to p-val_white, but p-val_white is not available when compartment='False'." 
                      "Defaulting resolve_conflict to 'length'.")
            config.resolve_conflict = "length" 
    else:
        raise ValueError("compartment parameter must be 'True' or 'False'")
    

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
        if config.verbose:
            print(f"jet_widths or scale_range is not specified, using default scale_range: {config.scale_range}")
    
    print_scale_width_conversion(config.scale_range)

    if args.sum_consistency_im.lower() == "true":
        config.sum_consistency_im = True
    elif args.sum_consistency_im.lower() == "false":
        config.sum_consistency_im = False
    else:        
        raise ValueError("sum_consistency_im parameter must be 'True' or 'False'")
    
    if args.adj_nondec.lower() == "true":
        config.adj_nondec = True
    elif args.adj_nondec.lower() == "false":
        config.adj_nondec = False
    else:        
        raise ValueError("adj_nondec parameter must be 'True' or 'False'")

    if args.ang_frac.lower() == "true":
        config.ang_frac = True
    elif args.ang_frac.lower() == "false":
        config.ang_frac = False
    else:        
        raise ValueError("ang_frac parameter must be 'True' or 'False'")    

    # Sub directory (level 2): all results
    config.save_sub_dir = os.path.join(config.save_dir, f"{config.root}_results_all")
    if not os.path.exists(config.save_sub_dir):
        os.makedirs(config.save_sub_dir)

    # Sub directories (level 2): results for just q_val and both q_val_white thresholding
    sub_dir = os.path.join(config.save_dir, f"{config.root}_results_thresholded")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    config.dir_thresholded = sub_dir

    q_value_name = f"q_val_{config.q_val}" if not config.compartment else f"q_val_{config.q_val}-{config.q_val_white}"
    sub_dir = os.path.join(config.save_dir, f"{config.root}_results_{q_value_name}")
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    config.dir_alpha = sub_dir

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
        print("MIA-Jet version:", __version__)

        # Inputs
        print("Inputs")
        print("* Hi-C file (.hic):", config.hic_file)

        # Required Parameters
        print("\nRequired Parameters")
        print("* Experiment type:", config.exp_type)
        print("* Compartment:", config.compartment)
        print("* Chromosome:", config.chrom)
        print("* Resolution:", config.resolution)
        print("* Window size:", config.window_size)
        print("* Save directory root:", config.save_dir_root)

        # Extended Parameters
        print("\nExtended Parameters")
        print("* Significance threshold q_val:", config.q_val)
        print("* Significance threshold q_val_white:", config.q_val_white)
        print("* Jet widths (if specified):", config.jet_widths)
        print("* Scale range (if specified):", config.scale_range)
        print("* Angle range:", config.angle_range)
        print("* Root within:", config.root_within)
        print("* Root within comp:", config.root_within_comp)
        print("* Folder name:", config.folder_name)
        print("* Save directory:", config.save_dir)
        print("* Number of cores:", config.num_cores)
        print("* Verbose:", config.verbose)
        print("* Diagnostic plots:", config.diagnostic_plots)

        # Trim Parameters
        print("\nTrim Parameters")
        print("* Angle trim:", config.angle_trim)
        print("* Scale dec trim:", config.scale_dec_trim)
        print("* Scale trim:", config.scale_trim)
        print("* Comp trim:", config.comp_trim)

        # Filter parameters
        print("\nFilter Parameters")
        print("* Angle turbulence:", config.angle_turbulence)
        print("* Blobness:", config.blobness)
        print("* Consistency:", config.consistency)
        print("* Sum consistency:", config.sum_consistency)
        print("* Sum consistency im:", config.sum_consistency_im)
        print("* Ridge strength turbulence:", config.ridge_strength_turbulence)
        print("* Angle satisfied:", config.angle_satisfied)
        print("* Length:", config.length)

        # Fixed/Advanced Parameters
        print("\nFixed/Advanced Parameters")
        print("* Scale dec thresh trim:", config.scale_dec_thresh_trim) # Too low level
        print("* Scale trim thresh:", config.scale_trim_thresh) # Too low level
        print("* Scale trim window:", config.scale_trim_window) # Too low level
        print("* Angle fraction:", config.ang_frac) # Saliency (too low level)
        print("* Adjacent non-decreasing:", config.adj_nondec) # Saliency (too low level)
        print("* Data type:", config.data_type)
        print("* Normalization:", config.normalization)
        print("* Hysteresis thresholding parameters:", config.thresholds)
        print("* Gamma:", config.gamma)
        print("* Ridge method:", config.ridge_method)
        print("* Rotation padding:", config.rotation_padding)
        print("* Convolution padding:", config.convolution_padding)
        print("* Remove k strata:", config.rem_k_strata)
        print("* Resolve conflict:", config.resolve_conflict)

        

