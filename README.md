# MIA-Jet: Multiscale Identification Algorithm for Chromatin Jets
![Latest tag](https://img.shields.io/github/v/tag/sion23/miajet?sort=semver)


## Release notes
* Major changes to the program with v1.1.x. 
* The current program is optimized for linux or macOS systems (HPC clusters)
* The [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672730v1) paper uses results (`saliency-90-p-0.1`) from an earlier version of the program (v1.0.19), for which the processed data is available to download via [this link](https://www.dropbox.com/scl/fi/rp8sooa9wm0pp3qdry3pb/miajet_output_v1.0.19_paper_data_chr_combined.zip?rlkey=vbrlg3m3ijkgu2jnsbffvq658&st=luo39mcz&dl=0).

## Overview

At its most basic input, MIA-Jet requires a Hi-C data file and a small set of [required parameters](#required). MIA-Jet also offers extensive customization for various types of 3C data via [extended](#extended), [trim](#trim-parameters), [filter](#filter-parameters), and [fixed parameters](#fixed-parameters). Setting `--exp_type` automatically assigns sensible defaults for many parameters; directly specifying any parameter will take precedence over these automatic assignments. We highlight the automatic defaults for `--exp_type "hic"` and `--exp_type "replihic"`. 

### Example output

TODO: Make a figure that showcases all the parameters 

## Input

* `hic_file` (str): Path to Hi-C data file (`.hic` or `.mcool`).

## Parameters


### Required

* `--exp_type` (`"hic"` | `"replihic"`): Experiment type. Setting this automatically assigns defaults for many other parameters. Unless the experiment type is Repli-HiC, it is recommended to set this parameter to `"hic"` for most cases. 
* `--compartment` (`"True"` | `"False"`): Whether the data contains A/B compartments. When `"True"`, jets that go through compartments will be trimmed (see [comp_trim](#trim-parameters)). When `"False"`, `--comp_trim` is disabled and `--root_within_comp` / `--q_val_white` have no effect.
    * Experiment type is "hic": `"True"`
    * Experiment type is "replihic": `"False"`
* `--chrom` (str): Chromosome (e.g. `"chr1"`).
* `--resolution` (int): Hi-C resolution in base pairs (e.g. `25000` for 25 kbp).
* `--window_size` (int): Distance from main diagonal (e.g. `6000000` for 6 Mbp). For computational efficiency, if the resolution is high (≤ 10 kb) it is recommended to set window_size to be < 3 Mb. If resolution is > 10 kb, it is recommended to set window_size ≥ 3 Mb. 
* `--save_dir_root` (str): Absolute path to directory where results will be saved.


---


### Extended

* `--q_val` (float): Threshold for corrected p-value cutoffs on Hi-C data.
    * Experiment type is "hic":  `0.01`
    * Experiment type is "replihic":  `0.2`
* `--q_val_white` (float; default: `0.95`): Threshold for corrected q-value cutoffs on the Hi-C map after regressing out A/B compartments. Only applies when `--compartment "True"`; feature is disabled when `--compartment "False"`.
* `--jet_widths` (float float; default: `None`): Lower/upper bounds of jet widths _in pixels_. If this parameter is specified, then the set of scales (i.e., `scale_range`) is automatically generated. If omitted, a default log-spaced scale range is used (≈ $1.5^1$ … $1.5^7$ with 24 steps). **Alternative to** `--scale_range`; if given, overrides `--scale_range`.
* `--scale_range` (float, or multiple floats; default: `None`): Standard deviations of Gaussian blurs used in scale space (list). **Alternative to** `--jet_widths`; if `--jet_widths` is given, `--jet_widths` takes precedence. Recommended to be log-spaced. 
* `--angle_range` (float float): Angle lower and upper bounds (degrees) with 90° being a typical jet and 45° or 135° being a stripe (horizontal or vertical).
    * Experiment type is "hic": `60 120`
    * Experiment type is "replihic": `80 100`
* `--root_within` (float): Enforce the closest point of the jet to the main diagonal to be within a certain distance. Jets that do not satisfy this are filtered out.
    * If `root_within` ≤ 1, it is interpreted as a **fraction of the window size**.
    * If `root_within` > 1, it is interpreted as a **number of bins** directly.
    * A value of `0` causes all jets to be trimmed and the program will exit.
    * Experiment type is "hic": `12` (bins)
    * Experiment type is "replihic": `0.5` (fraction of window size)
* `--root_within_comp` (float; default: `None`): Prevents trimming of jets that cross A/B compartment boundaries if they are ≤ this many bins from the main diagonal. Only applies when `--compartment "True"`; feature is disabled when `--compartment "False"`. Interpretation follows the same rules as `--root_within` (≤ 1 = fraction, > 1 = bins). If not specified, defaults to the value of `--root_within` when in compartment mode.
* `--folder_name` (str; default: `None`): Output subfolder name. If `None`, defaults to the Hi-C file stem appended with chromosome and resolution. If specified, chromosome and resolution are still appended.
* `--num_cores` (int; default: `1`): Number of CPU cores to use.
* `--verbose` (flag; default: off): Print debug/details.
* `--diagnostic_plots` (flag; default: off): Print diagnostic plots at every major step.


---


### Trim Parameters

Trim parameters control how detected ridges are split. Each trim parameter accepts:
* `"None"` — no splitting is performed.
* A **float** in `[0, 1]` — minimum allowed length after trimming as a **fraction** of the original ridge length. This is to prevent jets from being split into very small segments in some extreme cases.
* An **int** ≥ 1 — minimum allowed length after trimming in **pixels** (bins). This is to prevent jets from being split into very small segments in some extreme cases.

These are advanced parameters and we recommend users to either disable them if necessary (specify "None") or enable with a fraction (e.g. 0.3). 

| Parameter | Description | <span style="color:red">hic</span> default | <span style="color:blue">replihic</span> default |
|---|---|---|---|
| `--angle_trim` | Splits ridges where the angle deviates from `--angle_range` | `3` (bins) | `0.5` (fraction) |
| `--scale_dec_trim` | Splits ridges where there is a large decrease in scale values along the ridge | `3` (bins) | `"None"` (disabled) |
| `--scale_trim` | Splits ridges where there is a large deviation in scale within a window specified by `--scale_trim_window` | `0.25` (fraction) | Feature disabled |
| `--comp_trim` | Splits ridges to prevent them from going through A/B compartments. Only applies when `--compartment "True"`; automatically disabled when `--compartment "False"` | `3` (bins) | Feature disabled |

---


### Filter Parameters

Filter parameters for post-detection filtering of jets. Setting a filter to `"None"` disables it.

| Parameter | Description | Default |
|---|---|---|
| `--angle_turbulence` | Filters by the coefficient of variation of the jet's angle values. Idea is that a jet's angle shouldn't vary too much. | Experiment type is "hic": < `0.325`; Experiment type is "replihic": `"None"` (disabled) |
| `--blobness` | Filters by the ratio of the maximum width and the length of the jet (removes blob-like, small focal enrichments e.g., loops) | < `2.0` |
| `--consistency` | Filters by jet consistency, which is the fraction of points in jet where the scale is non-decreasing. A non-decreasing scale is characteristic of jets observed in Guo et al. 2022. Disable this option if jets are not diffuse (see example output). | Experiment type is "hic": > `0.6`; Experiment type is "replihic": `"None"` (disabled) |
| `--sum_consistency_im` | Filters by sum consistency on the image, which is the sum of the observed Hi-C pixels along points where the scale is non-decreasing. This helps to remove false positive jets in sparse/noisy regions. The threshold value itself cannot be specified and if `"True"` then the Yen threshold is applied. | Experiment type is "hic": `"False"` (disabled); Experiment type is "replihic": `"True"` |
| `--ridge_strength_turbulence` | Filters by the coefficient of variation of the jet's ridge strength values. A jet should have relatively stable ridge strength values across jet. | < `1.0` |
| `--angle_satisfied` | Filters by the fraction of points in the jet that lie within `--angle_range` | > `0.3` |
| `--length` | Filters by minimum length of jet (in pixels/bins) | > `5` |

---


### Fixed Parameters

These parameters control low-level algorithmic behavior and typically do not need to be changed.

* `--scale_dec_thresh_trim` (int): Number of scales that must decrease for a ridge to be split. Related to `scale_dec-trim`. Must be less than the number of scales in scale_range.
    * Experiment type is "hic": `10`
    * Experiment type is "replihic": Feature disabled
* `--scale_trim_thresh` (float): Threshold for the scale_trim parameter (in units of standard deviation of scales). Related to `scale_trim` and `--scale_trim_window`. 
    * Experiment type is "hic": `3`
    * Experiment type is "replihic": Feature disabled
* `--scale_trim_window` (int): Kernel size (odd integer) for the scale_trim parameter. Specifically it is the kernel size for the convolution operation where we compute standard deviation of scales. Related to `scale_trim` and `--scale_trim_thresh`. 
    * Experiment type is "hic": `5`
    * Experiment type is "replihic": Feature disabled
* `--ang_frac` (`"True"` | `"False"`): Whether to apply angle-fraction multipliers to the saliency score.
    * Experiment type is "hic": `"True"`
    * Experiment type is "replihic": `"False"`
* `--adj_nondec` (`"True"` | `"False"`): Whether to apply the adjacent non-decreasing criteria (consistency measure) to the saliency score. Note that if set to `"False"`, then `consistency` becomes 1 and `sum_consistency_im` is simply the sum of  the observed Hi-C pixels along all points of the ridge.
    * Experiment type is "hic": `"True"`
    * Experiment type is "replihic": `"False"`
* `--data_type` (`"observed"` | `"oe"`): Hi-C data type.
    * Experiment type is "hic": `"oe"`
    * Experiment type is "replihic": `"observed"`
* `--normalization` (str): Hi-C normalization method (e.g. `"KR"`, `"VC_SQRT"`, `"NONE"`).
    * Experiment type is "hic": `"KR"`
    * Experiment type is "replihic": `"VC_SQRT"`
* `--thresholds` (float float; default: `None`): Lower/upper thresholds for ImageJ Curve Tracing. If `None` (default), thresholds are suggested automatically based on `--scale_range` or `--jet_widths`.
* `--gamma` (float; default: `0.75`): Scale space parameter $\gamma$ in $[0,1]$ (0.75 recommended for ridges; 1.0 for edges).
* `--ridge_method` (int; choices: `1,2,3`; default: `1`): Ridge strength/saliency formulation. Option 1 is recommended.
    * 1: $\lambda_1$, where $\lambda_1$ is the largest eigenvalue of the image Hessian matrix $H$
    * 2: $(\lambda_1^2 - \lambda_2^2)^2$
    * 3: $(\lambda_1 - \lambda_2)^2$
* `--rotation_padding` (str; default: `"nearest"`): Padding for `scipy.ndimage.rotate`. Choices: `"reflect"`, `"grid-mirror"`, `"constant"`, `"grid-constant"`, `"nearest"`, `"mirror"`, `"grid-wrap"`, `"wrap"`.
* `--convolution_padding` (str; default: `"nearest"`): Padding for `scipy.ndimage.correlate`/`correlate1d`. Choices: `"reflect"`, `"constant"`, `"nearest"`, `"mirror"`, `"wrap"`.
* `--rem_k_strata` (int; default: `1`): Remove jets located within the k-th off-diagonal strata. 
* `--resolve_conflict` (str; default: `"blobness"`): The jet statistic to maximize among overlapping jets. Needs to be a column in the summary dataframe. Choices: `"length"`, `"p-val"`, `"p-val_white"`, `"sum_consistency"`, `"sum_consistency_im"`, `"saliency"`, `"blobness"`.


### Notes
* For different resolutions, the `--root_within` may need to be adjusted.
* When `--compartment "False"` is set, compartment-dependent features (`--comp_trim`, `--root_within_comp`, `--q_val_white`) are automatically disabled regardless of their specified values.

### Running across chromosomes
* See `submit_DP_thymocyte.sh` and corresponding `job_DP_thymocyte.sbat` to see how to call MIA-Jet across chromosomes for one cell-line
* See `./notebooks/combine_results.ipynb` to see how to combine MIA-Jet results (once they finish generating)


## Examples
```bash
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic \
  --chrom "chr3" \
  --exp_type "hic" \
  --resolution 25000 \
  --window_size 6000000 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output" \
  --num_cores 4 \
  --verbose \
  --diagnostic_plots
```
```bash
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/Repli-HiC_K562_WT_totalS.hic \
  --chrom "chr22" \
  --exp_type "replihic" \
  --resolution 25000 \
  --window_size 6000000 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output" \
  --num_cores 4 \
  --verbose \
  --diagnostic_plots
```
```bash
python -m miajet "test1cii_s-23_hic_003.hic" \
  --chrom "chrS" \
  --exp_type "hic" \
  --resolution "25000" \
  --compartment "False" \ # There are no A/B compartments in this chromosome
  --normalization "NONE" \ # Set Hi-C normalization to "NONE" as there are no "KR" vectors
  --window_size "5000000" \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_revision/miajet/output" \
  --num_cores 4 \
  --verbose \
  --diagnostic_plots
```

## Installing MIA-Jet
1. Clone directory
2. Create conda environment: `conda env create -f environment.yml` (default name is `jet-env`)
3. Activate environment: `conda activate jet-env`
4. Build: `python -m pip install -e .` 
5. Run examples

## Output Table
There are 3 key outputs of the MIA-Jet program: 
1. `*_expanded_table.csv`
2. `*_summary_table.csv`
3. `*_juicer-visualize.bedpe`

The `*_juicer-visualize.bedpe` can be loaded into the [Juicebox program](https://github.com/aidenlab/Juicebox) as a 2D annotation for viewing. Note that the juicer visualized outputs do not contain outputs such as the width or angles, and is intended for simple visualization. The expanded and summary tables are formatted as follows:

### Expanded table (`*_expanded_table.csv`)

<div style="overflow-x: auto;">

| unique_id | chrom | x (bp) | y (bp) | x (pixels) | y (pixels) | scale | angle | ridge_strength | width | input | Asymmetry | Contrast |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1151_1.5 | chrS | 5720000 | 5650000 | 322 | 139 | 1.5 | 85.7 | 0.0666 | 3.46 | 0.234 | 0.0149 | 51.1 |
| 1151_1.5 | chrS | 5740000 | 5640000 | 322 | 138 | 1.5 | 86.6 | 0.0853 | 3.46 | 0.336 | -0.00111 | 58.4 |
| 1151_1.5 | chrS | 5760000 | 5620000 | 322 | 137 | 1.5 | 87.5 | 0.0986 | 3.46 | 0.354 | -0.0171 | 69.6 |
| … | … | … | … | … | … | … | … | … | … | … | … | … |
| 2_2.546 | chrS | 7770000 | 7740000 | 439 | 140 | 1.5 | 86.8 | 0.0479 | 3.46 | 0.166 | -0.0451 | 24.8 |
| 2_2.546 | chrS | 7790000 | 7720000 | 439 | 139 | 1.5 | 87.4 | 0.0711 | 3.46 | 0.267 | -0.00323 | 89.8 |
| 2_2.546 | chrS | 7810000 | 7700000 | 439 | 138 | 1.5 | 89.7 | 0.0903 | 3.46 | 0.322 | -0.0042 | 85 |
| … | … | … | … | … | … | … | … | … | … | … | … | … |

</div>

The expanded table contains the most detailed information about each jet. Each jet is uniquely identifiable with the `unique_id` column. The columns `chrom`, `x (bp)`, and `y (bp)` holds the genomic location for each point of a jet, where `x (bp)` is the genomic x coordinate of the Hi-C map and `y (bp)` is the genomic y coordinate of the Hi-C map. The `x (pixel)`, and `y (pixel)` are the binned coordinates (indices) relative to the generated image matrix `*_contact_map.jpg`. Each jet point is associated with a scale, width, angle at which the jet is pointing towards, ridge strength, and input, which is the value of the normalized Hi-C data at that jet point. Asymmetry and Contrast are columns returned from the [CurveTracing ImageJ plugin](https://github.com/UU-cellbiology/CurveTrace). The scale is approximately equal to width and conversion can obtained with the following equation [3]:

$$
s = \frac{w}{2 \sqrt{3}} + 0.5
$$

where $s$ is the scale and $w$ is the width. 


### Summary table (`*_summary_table.csv`)

<div style="overflow-x: auto;">

| unique_id | Label | chrom | start | end | s_imagej | length | saliency | ridge_strength_turbulence | angle_turbulence | angle_satisfied | consistency | sum_consistency | sum_consistency_im | blobness | p-val | q-val_white | q-val |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1151_1.5 | contour_f_1_c_1151 | chrS | 5720000 | 5650000 | 1.5 | 64 | 6.46 | 0.245 | 0.0868 | 0.938 | 0.844 | 54 | 34.1 | 0.183 | 7.26e-20 |  | 2.18e-19 |
| 3_1.853 | contour_f_1_c_3 | chrS | 8320000 | 8280000 | 1.8533820793094375 | 14 | 0.849 | 0.178 | 0.223 | 1 | 1 | 14 | 5.95 | 0.289 | 2.49e-08 |  | 3.74e-08 |

</div>

The summary table summarizes each jet into a single row uniquely identified by `unique_id` column (the `Label` column is the original unique jet identifier from the ImageJ .csv files). Each column quantifies different properties of the jet. The columns `chrom`, `start`, `end` specify the genomic locations of the jet loading site. `s_imagej` is the scale at which the jet was found in the CurveTracing ImageJ program. The `length` is the length in number of bins of the jet. The `saliency` quantifies the overall 'ridgeness' of the jet as a measure of the second derivative curvature (see paper method for more details). 

The columns `ridge_strength_turbulence`, `angle_turbulence`, `angle_satisfied`, and `blobness` are primarily used for filtering jets. `ridge_strength_turbulence` captures the variation of the ridge strength of the jet; false jets that connect two different structures (e.g. TAD and a loop) typically have different curvatures and therefore high variability in ridge strength. Similarly `angle_turbulence` captures the variation of the angle in the jet; true jets tend to have low variation in angle. `angle_satisfied` captures the proportion of points whose angle is in the specified angle range. `blobness` was designed to filter out very small jets whose width were significantly larger than their length, characteristic of loops.

The columns `consistency`, `sum_consistency`, `sum_consistency_im` are used to quantify characteristics specific to the jets observed in Guo et al. 2022, which are noticeable more 'diffuse' (i.e., a narrow loading site followed by a gradual increase in width as it portrudes). The `consistency` measure is the proportion of points where adjacent scales are non-decreasing, capturing the aspect of a jet gradually increasing in scale (width). While `consistency`  doesn't consider the length of the jet, `sum_consistency` captures the length – it is the *number* of points where adjacent scales are non-decreasing. `sum_consistency_im` was developed for Replhic data specifically, where many false jets in sparse regions were being identified. The `sum_consistency_im` simply sums the observed Hi-C pixel values at only the points where adjacent scales are non-decreasing. Therefore, `sum_consistency_im` is an appropriate column to rank jets for both Hi-C and Replihic data. 

The `p-val` is the uncorrected p-value, `q-val` is the FDR BH corrected p-values. The `q-val_white` is the corrected p-value using the Hi-C map with A/B compartments regressed out. 


## References

1. Guo, Ya, et al. "Chromatin jets define the properties of cohesin-driven in vivo loop extrusion." Molecular cell 82.20 (2022): 3769-3780.
2. https://github.com/UU-cellbiology/CurveTrace
3. https://imagej.net/plugins/ridge-detection  
4. https://github.com/aidenlab/Juicebox

## Citation
"MIA-Jet: Multi-scale Identification Algorithm of Chromatin Jets" by Sion Kim and Minji Kim. [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672730v1) (2025), 672730.

For questions or bug reports, contact Sion (sionkim@umich.edu) or visit the "Issues" page.
