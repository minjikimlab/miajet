# MIA-Jet: Multiscale Identification Algorithm for Chromatin Jets
![Latest tag](https://img.shields.io/github/v/tag/sion23/miajet?sort=semver)


## Release notes
* There are several improvements to version v1.0.26 (over v1.0.20), namely in efficiency and optimizations for higher resolutions (e.g. 25 kb)
* Recommended parameters have been slightly adjusted (see below)
* The current program is optimized for linux or macOS systems (HPC clusters)
* The [biorXiv](https://www.biorxiv.org/content/10.1101/2025.08.27.672730v1) paper uses results from an earlier version of the program (v1.0.19), for which the processed data is available to download via [this link](https://www.dropbox.com/scl/fi/rp8sooa9wm0pp3qdry3pb/miajet_output_v1.0.19_paper_data_chr_combined.zip?rlkey=vbrlg3m3ijkgu2jnsbffvq658&st=luo39mcz&dl=0).

## Overview

At its most basic input, MIA-Jet requires only 4 parameters (Required). However, MIA-Jet also offers advanced customization for various types of 3C data via the extended and advanced parameters. 

## Input

* `hic_file` (str): Path to Hi-C data file (`.hic` or `.mcool`).

## Parameters 


#### Required

* `--exp_type` (`"hic"` | `"replihic"`): Experiment type. Setting this automatically sets some of the parameters. We denote <span style="color:red">"hic"</span> (red) and <span style="color:blue">"replihic"</span> (blue) for this automatic assignment. Nevertheless, directly specifying any one of these parameters will take precedence over automatic assignment. 
* `--chrom` (str): Chromosome (e.g. `"chr1"`).
* `--resolution` (int): Hi-C resolution in base pairs (e.g. `50000` for 50 kbp).
* `--save_dir_root` (str): Absolute path to directory where results will be saved.


---


#### Extended

* `--alpha` (float, or multiple floats; default: `0.2, 0.1, 0.05`): One or more α values for p-value cutoffs.
* `--window_size` (int; default: `6000000`): Max distance from main diagonal in which jets are expected (e.g. `6000000` for 6 Mbp).
* `--normalization` (str; optional): Hi-C normalization method (e.g. `"KR"`, `"VC_SQRT"`, `"NONE"`). If omitted, uses dataset/tooling defaults.
    * <span style="color:red">"hic"</span>: "KR"
    * <span style="color:blue">"replihic"</span>: "VC_SQRT"
* `--data_type` (`"observed"` | `"oe"`; default: `"observed"`): Hi-C data type.
    * <span style="color:red">"hic"</span>: "oe"
    * <span style="color:blue">"replihic"</span>: "observed"
* `--thresholds` (float float; default: `None`): Lower/upper thresholds for ImageJ Curve Tracing. If `None`, thresholds are suggested automatically from `scale_range` or `jet_widths`.
* `--angle_range` (float float; default: `80 100`): Angle bounds (degrees) with 90° being the perpendicular (uncurved) jet
* `--saliency_thresh` (float; default: `80`): Percentile (computed from non-zero saliency) used for saliency thresholding.
* `--jet_widths` (float,float; default: `None`): Lower/upper bounds of jet widths _in pixels_ to detect. It is important to ensure this parameter is set accurately, as this determines the scales considered. Nevertheless, if omitted, a default log-spaced scale range is used (≈ $1.5^1$ … $1.5^7$ with 24 steps). 
* `--root_within` (int; optional): Enforce ridge root ≤ this many bins from the main diagonal. This parameter is used to ensure that we see jets "connected" to the main diagonal
    * <span style="color:red">"hic"</span>: 10
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


### Notes
* “`None`|float” means that the CLI accepts the literal string `"None"` (case-insensitive) to mean Python `None`, or a numeric value.
* For different resolutions, the `root_within` may need to be adjusted (see examples).

### Running across chromosomes
* See `submit_DP_thymocyte_50Kb.sh` and corresponding `job_DP_thymocyte_50Kb.sbat` to see how to call MIA-Jet across chromosomes for one cell-line
* See `./notebooks/combine_results.ipynb` to see how to combine MIA-Jet results (once they finish generating)


## Examples
```
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic \
  --chrom "chr3" \
  --exp_type "hic" \
  --resolution 25000 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_output_v1.0.25" \
  --num_cores 4 \
  --verbose \
  --root_within 10 \
```
```
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/GSE199059_CD69negDPWTR1R2R3R4_merged.hic \
  --chrom "chr3" \
  --exp_type "hic" \
  --resolution 50000 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_output_v1.0.25" \
  --num_cores 4 \
  --verbose \
  --root_within 5 \
```
```
python -m miajet /nfs/turbo/umms-minjilab/downloaded_data/Repli-HiC_K562_WT_totalS.hic \
  --chrom "chr3" \
  --exp_type "replihic" \
  --resolution 50000 \
  --alpha 0.1 0.05 0.01 \
  --save_dir_root "/nfs/turbo/umms-minjilab/sionkim/miajet_output_v1.0.25" \
  --num_cores 4 \
  --verbose \
```

## Installing MIA-Jet
1. Clone directory
2. Create conda environment: `conda env create -f environment.yml` (default name is `jet-env`)
3. Activate environment: `conda activate jet-env`
4. Run examples

## Output Table
There are 3 key outputs of the MIA-Jet program: 
1. `*_expanded_table.csv`
2. `*_summary_table.csv`
3. `*_juicer-visualize.bedpe`

The `*_juicer-visualize.bedpe` can be loaded into the [Juicebox program](https://github.com/aidenlab/Juicebox) as a 2D annotation for viewing. Note that the juicer visualized outputs do not contain outputs such as the width or angles, and is intended for simple visualization. The expanded and summary tables are formatted as follows:

### Summary table (`*_summary_table.csv`)

| unique_id     | chrom | start       | end         | length | input_mean | angle_mean | width_mean | jet_saliency | ks | p-val_raw | p-val_corr | stripiness |
|----------------|--------|-------------|-------------|---------|-------------|-------------|-------------|---------------|----|------------|-------------|-------------|
| chr1_1493_2    | chr1  | 52096283.31 | 52145898.04 | 100000  | 0.092       | 150.749     | 2.418       | 0             | 0  | 1          | 1           | 0           |
| chr1_7908_0    | chr1  | 16042274.7  | 16088778.77 | 100000  | 0.077       | 42.156      | 1.057       | 0             | 1  | 0.167      | 0.294       | 0           |


The summary table summarizes each jet into a single row, with metrics detailing the location of the jet (`chrom`, `start`, `end` i.e. genomic coordinates) and summaries of the jet, such as `length`, `input_mean`, `angle_mean`. An important column is the `jet_saliency` column, which is the final ranking metric for the jets, which can be interpreted as a weighted aggregate sum of the jet strength. We also include the significance statistics, namely the `ks` `p-val_raw` (uncorrected p-value) and `p-val_corr` (corrected p-value).



### Expanded table (`*_expanded_table.csv`)

| unique_id     | chrom | x (bp)       | y (bp)       | x (pixels) | y (pixels) | width  | angle_imagej | ridge_strength |
|----------------|--------|--------------|--------------|-------------|-------------|--------|---------------|----------------|
| chr1_237_16    | chr1  | 10020173.92  | 9917977.853  | 282.382     | 82.7        | 11.535 | 89.349        | 0.001          |
| chr1_237_16    | chr1  | 10055521.32  | 9883785.156  | 282.399     | 81.717      | 11.535 | 88.876        | 0.001          |

The expanded table "expands" each jet into multiple rows, where each row corresponds to a single point of a jet. The `unique_id` column serves as a linker between the summary table and the expanded table (i.e. each `unique_id` corresponds to a single jet). Each row contains the genomic coordinates of each jet point in basepairs (`x (bp)`, `y (bp)`) and also in pixel coordinates with respect to the rotated image (`x (pixels)`, `y (pixels)`). Additionally, each jet point has an associated width `width` and angle that it is heading in `angle_imagej` as well as the ridge strength `ridge_strength`. 






