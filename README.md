# MIA-Jet: Multiscale Identification Algorithm for Chromatin Jets
![Latest tag](https://img.shields.io/github/v/tag/sion23/miajet?sort=semver)


## Release notes
* There are several improvements to version v1.0.20, namely in efficiency and optimizations for higher resolutions (e.g. 25 kb)
* Recommended parameters have been slightly adjusted (see below)

## Overview

At its most basic input, MIA-Jet requires only 4 parameters (Required). However, MIA-Jet also offers advanced customization for various types of 3C data via the extended and advanced parameters. 

### Input

* `hic_file` (str): Path to Hi-C data file (`.hic` or `.mcool`).

### Parameters 


#### Required

* `--exp_type` (`"hic"` | `"replihic"`): Experiment type. This influences the parameter values that are set. We denote the default for <span style="color:red">"hic"</span> (red) and <span style="color:blue">"replihic"</span> (blue).
* `--chrom` (str): Chromosome (e.g. `"chr1"`).
* `--resolution` (int): Hi-C resolution in base pairs (e.g. `50000` for 50 kbp).
* `--save_dir_root` (str): Absolute path to directory where results will be saved.


#### Extended

* `--alpha` (float, or multiple floats; default: `0.4 0.3 0.2`): One or more α values for p-value cutoffs.
* `--window_size` (int; default: `6000000`): Max distance from main diagonal in which jets are expected (e.g. `6000000` for 6 Mbp).
* `--normalization` (str; optional): Hi-C normalization method (e.g. `"KR"`, `"VC_SQRT"`, `"NONE"`). If omitted, uses dataset/tooling defaults.
    * <span style="color:red">"hic"</span>: "KR"
    * <span style="color:blue">"replihic"</span>: "VC_SQRT"
* `--data_type` (`"observed"` | `"oe"`; default: `"observed"`): Hi-C data type.
    * <span style="color:red">"hic"</span>: "oe"
    * <span style="color:blue">"replihic"</span>: "observed"
* `--thresholds` (float float; default: `None`): Lower/upper thresholds for ImageJ Curve Tracing. If `None`, thresholds are suggested automatically from `scale_range` or `jet_widths`.
* `--angle_range` (float float; default: `80 100`): Angle bounds (degrees) with 90° being the perpendicular (uncurved) jet
* `--saliency_thresh` (float; default: `90`): Percentile (computed from non-zero saliency) used for saliency thresholding.
* `--jet_widths` (float,float; default: `None`): Lower/upper bounds of jet widths _in pixels_ to detect. It is important to ensure this parameter is set accurately, as this determines the scales considered. Nevertheless, if omitted, a default log-spaced scale range is used (≈ $1.5^1$ … $1.5^7$ with 24 steps). 
* `--root_within` (int; optional): Enforce ridge root ≤ this many bins from the main diagonal. This parameter is used to ensure that we see jets "connected" to the main diagonal
    * <span style="color:red">"hic"</span>: 3
    * <span style="color:blue">"replihic"</span>: None
* `--folder_name` (str; default: `None`): Output subfolder name. If `None`, defaults to the Hi-C file’s stem.
* `--num_cores` (int; default: `1`): Number of CPU cores to use.
* `--verbose` (flag; default: off): Print debug/details.
* `--rmse` (`None`|float; default: `None`): Normalized RMSE threshold.
    * <span style="color:red">"hic"</span>: 0.01
    * <span style="color:blue">"replihic"</span>: None
* `--entropy_thresh` (`None`|float; default: `None`): Normalized entropy threshold.
    * <span style="color:red">"hic"</span>: 0.5
    * <span style="color:blue">"replihic"</span>: None

#### Optional

* `--scale_range` (float, or multiple floats; default: `None`): Standard deviations of Gaussian blurs used in scale space (list). **Alternative to** `jet_widths`; if given, overrides `jet_widths`. Recommended to be log-spaced. With v1.0.25, it is now recommended to use the `--jet_widths` parameter.

#### Advanced

* `--gamma` (float; default: `0.75`): Scale space parameter $\gamma$ in $[0,1]$ (0.75 recommended for ridges; 1.0 for edges).
* `--ridge_method` (int; choices: `1,2,3,4,5,6,7`; default: `1`): Ridge strength/saliency formulation. Option 1 is the recommended.
    * 1: D1: $\lambda_1$, where $\lambda_1$ is the largest eigenvalue of the Hessian matrix $H$
    * 2: D2: $(\lambda_1^2 - \lambda_2^2)^2$
    * 3: D3: $(\lambda_1 - \lambda_2)^2$
* `--rotation_padding` (choice; default: `"nearest"`): Padding for `scipy.ndimage.rotate`. Choices: `"reflect"`, `"grid-mirror"`, `"constant"`, `"grid-constant"`, `"nearest"`, `"mirror"`, `"grid-wrap"`, `"wrap"`.
* `--convolution_padding` (choice; default: `"nearest"`): Padding for `scipy.ndimage.correlate`/`correlate1d`. Choices: `"reflect"`, `"constant"`, `"nearest"`, `"mirror"`, `"wrap"`.
* `--sum_cond` (choice; default: `"a-r"`): Which condition masks to **sum** (or **average** if `--agg "mean"`) into the saliency score. Choices: `"a"`, `"r"`, `"c"`, `"a-r"`, `"a-c"`, `"r-c"`, `"a-r-c"`, where 
    * `"a"` denotes angle boolean condition (i.e. sum current pixel's ridge strength if it is within the specified `angle_range`), 
    * `"r"` denotes the ridge condition (i.e. sum current pixel's ridge strength if it is a ridge or not), 
    * `"c"` denotes the corner condition (i.e. sum current pixels' ridge strength if it is not a corner). 
* `--noise_consec` (str; default: `""`): Consecutive-True noise adjustment. Format: `"INTEGER-TYPE"`, where `INTEGER` is the min run length and `TYPE` ∈ `{ "a", "r", "a-r" }`. Empty string disables.
* `--noise_alt` (choice; default: `""`): Alternating 0/1 normalization selector. Choices: `""`, `"a"`, `"r"`, `"c"`, `"a-r"`, `"a-c"`, `"r-c"`, `"a-r-c"`.
* `--agg` (`"sum"` | `"mean"`; default: `"sum"`): Aggregation for final jet saliency score.
* `--rem_k_strata` (int; default: `1`): Remove jets located within the k-th off-diagonal strata.
* **Entropy histogram settings**

  * `--num_bins` (int; default: `10`)
  * `--bin_size` (float; default: `None`)
  * `--points_min` (float; default: `0`)
  * `--points_max` (`None`|float; default: `0.05`)
* **Epsilon thresholds**

  * `--eps_r` (`None`|float; default: `0.0005`)
  * `--eps_c1` (`None`|float; default: `0.1`)
  * `--eps_c2` (`None`|float; default: `1e-5`)
* `--whiten` (`None`|float; default: `None`): Enable ZCA whitening of the image; value is the epsilon (e.g. `1e-5`).
* **Intensity percentiles (0–100)**

  * `--im_vmax` (`None`|float): Max intensity percentile for Hi-C image.
    * <span style="color:red">"hic"</span>: 99
    * <span style="color:blue">"replihic"</span>: 100
  * `--im_vmin` (`None`|float): Min intensity percentile for Hi-C image.
  * `--im_corner_vmax` (`None`|float): Max intensity percentile for corner image.
      * <span style="color:red">"hic"</span>: 98
    * <span style="color:blue">"replihic"</span>: 100
  * `--im_corner_vmin` (`None`|float): Min intensity percentile for corner image.
* **Trim controls** (each accepts `None`, a **float** in `[0,1]` as a fraction of original length, or an **int ≥1\`** as an absolute minimum length in pixels). Note that a value of `0` means that trimming can occur anywhere, `0.5` means that trimming cannot cause the ridge to be less than half of its original length etc.
  * `--angle_trim` (default 0.5)
  * `--corner_trim`
    * <span style="color:red">"hic"</span>: 0
    * <span style="color:blue">"replihic"</span>: None
  * `--eig2_trim`
    * <span style="color:red">"hic"</span>: 0
    * <span style="color:blue">"replihic"</span>: None
* `--ang_frac` (flag): **Disable** angle-fraction multipliers in saliency. (By default it is **on**; passing this flag turns them off.)

---

**Notes**

* Anywhere a parameter says “`None`|float”, the CLI accepts the literal string `"None"` (case-insensitive) to mean Python `None`, or a numeric value.




**Examples**
```
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic \
  --chrom "${CHROM}" \
  --exp_type "hic" \
  --resolution 25000 \
  --alpha 0.4 0.3 0.2 \
  --saliency_thresh 80 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_output_v1.0.25" \
  --num_cores 4 \
  --verbose \
  --root_within 10 \
```
```
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/Repli-HiC_K562_WT_totalS.hic \
  --chrom "${CHROM}" \
  --exp_type "replihic" \
  --resolution 25000 \
  --alpha 0.4 0.3 0.2 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_debug" \
  --num_cores 4 \
  --verbose \
  --folder_name "test" \
```







