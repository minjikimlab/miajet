# MIA-Jet
![GitHub release (latest by date)](https://img.shields.io/github/v/release/OWNER/REPO)

Multiscale Identification Algorithm for Chromatin Jets

## Note: README is not up to date


1. **Generating expanded table**
    * **Input: Hi-C data, chromosome (and other parameters)**
    * Converts Hi-C data into rectangular images
    * Runs Curve Tracing ImageJ Plugin at various scales and gets ridge positions
    * Runs scale-space on Hi-C image to generate features at the ridge positions
    * **Output: an *expanded table*, defined by each row being a position of a ridge with features as columns**
2. **Ranking ridges**
    * **Input: _expanded table_, aggregation function $f$, threshold $\epsilon$**
    * Collapses the *expanded table* to a *summary table* using the aggregation function $f$
    * Plots features of all ridges that have score > threshold $\epsilon$. 
    * **Output: a _summary table_, defined by each row being a ridge with associated score** 


## Overview
### Inputs
* Hi-C data (`.hic`)

### Parameters
#### Generating expanded table
* `chrom`: chromosome (e.g. `"chr1"`)
* `normalization`: Hi-C normalization method (e.g. `"KR"`, `"VC_SQRT"`)
* `data_type=["observed", "oe"]`: Hi-C data type 
* `resolution`: Hi-C resolution (e.g. `50000` for 50 kbp)
* TODO: add thresholds
* `window_size`: distance from main diagonal, in which every jet is expected to be within (e.g. `6000000` for 6 mbp)
* `data_type=["observed", "oe"]`: Hi-C data-type (default: "observed")
* `scale_range`: standard deviation of Gaussian blurs considered in scale space. This can either be specified using the same parameters as [numpy.logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) or as a custom list of values
    * `(start, stop, num, base)` corresponding to `scale_range_mode="logspace"`. Please refer to [numpy.logspace](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) for a description of these parameters (default: `1 7 24 1.5`). 
    * `(s0, s1, ...)`, where `s0` is the first standard deviation, `s1` is the second etc. corresponding to `scale_range_mode="custom"`.
* `scale_range_mode=["logspace", "custom"]`: Specify whether `scale_range` supplied is log space or a list of standard deviation values (default: `"logspace"`).
* `gamma`: The $\gamma$ parameter in scale space between 0 and 1 (default: `0.75`). A value of 0.75 is recommended for ridges and a value of 1 is recommended for edges. 
* `ridge_method=[1, 2, 3, 5, 6, 7]`: the ridge strength measure to be used as the saliency measure (default: `1`)
    * 1: D1: $\lambda_1$, where $\lambda_1$ is the largest eigenvalue of the Hessian matrix $H$
    * 2: D2: $(\lambda_1^2 - \lambda_2^2)^2$
    * 3: D3: $(\lambda_1 - \lambda_2)^2$
    * 5: D5: $\lambda_1 + sign(det(H)) * \sqrt(|det(H)|)$
    * There are other possible options: such as (6) $\lambda_1 - \sqrt(|det(M)|)$ or (7) $(\lambda_1^2 - \lambda_2^2)^{1/2} - \sqrt(|det(M)|)$, although these are less tested than the others
* `rotation_padding`: the padding method for filling in the trapezoid corners in the [scipy.ndimage.rotate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html) rotation step in step (1) (default: `"mirror"`)
* `convolution_padding`: the padding method for [scipy.ndimage.correlate1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate1d.html), [scipy.ndimage.correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate.html) convolution operation in step (3) (default: `"nearest"`)
* `scale_selection=["local", "min_scale"]`: The method to select a single scale $s$ for a position $(x, y)$ in an image (default: `"local"`). 
    * Local: Selects scale for an $(x, y)$ position that is (1) global maxima of ridge strength in scale space AND passes 1st, 2nd deriv tests (2) If no global maxima exists but local maxima exists, then select the local maxima that maximizes ridge strength (3) Otherwise, does not assign a scale to this position.
    * Min scale: Selects scale for an $(x, y)$ position that is simply the minimum scale out of all global and local maxima in scale space. 
    * Note: this is in development and am testing DBSCAN to link scales appropriately

#### Ranking ridges
# Parameter Descriptions

## noise_consec
* `noise_consec` is a string in the format `"INTEGER-TYPE"` (default: `"2-a"`). Here, `INTEGER` specifies the minimum number of consecutive True values required for filtering, and `TYPE` determines which mask(s) to use. Allowed `TYPE` values are:
    - **"a"** – Use only the **angle** conditions.
    - **"r"** – Use only the **ridge** conditions.
    - **"c"** – Use only the **corner** conditions.
    - **"a-r"** – Use the intersection of the **angle** and **ridge** conditions.
    - **"a-c"** – Use the intersection of the **angle** and **corner** conditions.
    - **"r-c"** – Use the intersection of the **ridge** and **corner** conditions.
    - **"a-r-c"** – Use the intersection of **angle**, **ridge**, and **corner** conditions.
* `noise_alt` is a string that specifies the boolean mask for alternating normalization (i.e., the final score is divided by the number of alternating True/False transitions). The default value is `"a"`. Allowed values are:
    - **"a"** – Use only the **angle** conditions.
    - **"r"** – Use only the **ridge** conditions.
    - **"c"** – Use only the **corner** conditions.
    - **"a-r"** – Use the intersection of the **angle** and **ridge** conditions.
    - **"a-c"** – Use the intersection of the **angle** and **corner** conditions.
    - **"r-c"** – Use the intersection of the **ridge** and **corner** conditions.
    - **"a-r-c"** – Use the intersection of **angle**, **ridge**, and **corner** conditions.
* `sum_cond` is a string that specifies the baseline mask for final aggregation. This mask determines which conditions are summed (or averaged if `agg="mean"`) when calculating the final score. Allowed values are:
    - **"a"** – Use only the **angle** conditions.
    - **"r"** – Use only the **ridge** conditions.
    - **"c"** – Use only the **corner** conditions.
    - **"a-r"** – Use the intersection of the **angle** and **ridge** conditions.
    - **"a-c"** – Use the intersection of the **angle** and **corner** conditions.
    - **"r-c"** – Use the intersection of the **ridge** and **corner** conditions.
    - **"a-r-c"** – Use the intersection of **angle**, **ridge**, and **corner** conditions.
* `angle_range`: imposes angle lower and upper bound in degrees to each pixel (e.g. `[80, 100]`).
* `agg=["sum", "mean"]`: aggregation function
* `top_k`: the percentile threshold [0, 1] to plot the ridges with score above this cutoff 

#### General parameters
* `folder_name`: the folder name to store generated files (default: `"debug"`). If `"debug"` then all the parameter values are combined. The folder is located in `./output/<folder_name>`. If you specify it yourself, then this value should be distinct from all other file/folder names in `./output/` in order to prevent conflicts (unless you want to overwrite the results).
* `save_dir_root`: Root directory to store generated files (default: `"./output"` in the project directory `finding_jets/`)
* `num_cores`: Number of CPU cores available for parallelizing. 
* `verbose`: Whether debug statements are used (default: True) 


### Steps
#### Generating expanded table
1. **Generate Hi-C Image**: reads in Hi-C data, removes *unmapped regions* (i.e. columns and rows whose sum is $0$), extracts the upper-diagonal trapezoid, with the height determined by `window_size`. 
    * The Hi-C image is saved in `output/<folder_name>/`.
2. **Run ImageJ**: runs [CurveTracing Plugin](https://github.com/ekatrukha/CurveTrace/wiki/Source-Steger%27s-Algorithm) at all scales specified by `scale_range`. ImageJ returns the set of ridge positions as pixel coordinates of the Hi-C image. 
3. **Process ImageJ**: combines the results of ImageJ and performs preliminary filtering, namely
    - Remove ridges within `k=3` off-diagonal
    - Remove ridges with size less than or equal to `1`
    - Reorients ridges based on a simple check of the ridge end-points (i.e. is one end of the ridge closer to the main diagonal?)
3. **Generate scale space features**: generates scale space features (e.g. eigenvalues, eigenvectors of the image Hessian) using python packages, resulting in a tensor of Hi-C features. For reference, steps 2 and 3 are independent.
4. **Generate expanded table** generates an *expanded table*, where each row is a position of a ridge with features as columns.
    * Note that the scale space tensors and the ridge positions from ImageJ have unmapped regions removed. This means that we may generate the expanded table without inserting the 0 regions, where simple indexing should work. 
    * The output should be a large table containing multiple ridges, with each pixel of a ridge having an x-y position on the Hi-C image and features of (3). 
    * Once the table has been generated, we insert the unmapped regions by an indexing scheme. 
        > Note to developer: none of the objects like `I, D, W1, W2, A, R` or `df`, `df_pos` have the unmapped regions inserted, so they should be discarded and not used for downstream analysis. 
    * The output table is stored in `output/<folder_name>/`.

#### Ranking ridges
1. For each ridge in the _expanded table_,
    1. Generates a boolean mask of **ridge condition** and/or **angle constraints**. If noise parameters are not specified, this mask is all true. 
    2. Aggregates the ridge only where the boolean mask is true, using either sum or mean aggregation and the values determined by ridge strength. The output of this step is a single number.
    3. Noise adjustment is applied if specified. The output of this step is called the ridge score. 
2. The ridge scores for all ridges are then collected in a table and saved
    * The *summary table* is saved in `output/<folder_name>/`
3. The top K ridges in terms of ridge scores has its features plotted (e.g. ridge strength map, eigenvector, ridge conditions, angle constraints etc.).
    * The feature plots are saved in `output/<folder_name>/`


**Examples**
```
python -m generate_features /nfs/turbo/umms-minjilab/mingjiay/GSE199059_wt_selected_30_new.hic \
    --chrom "chr1" --normalization "KR" --resolution 50000 --window_size 6000000 \
    --data_type "observed" --thresholds 0.01 0.05 --rem_k_strata 3 \
    --scale_range 1 7 24 1.5 --scale_range_mode "logspace" --gamma 0.75 --ridge_method 1 \
    --rotation_padding "nearest" --convolution_padding "nearest" --scale_selection "min_scale" \
    --angle_range 80 100 --noise_consec "" --noise_alt "" --sum_cond "a-r" --agg "sum" --top_k 30 \
    --num_cores 6 --verbose true
```
```
python -m generate_features /nfs/turbo/umms-minjilab/downloaded_data/Repli-HiC_K562_WT_totalS.hic \
    --chrom "chr1" --normalization "VC_SQRT" --resolution 25000 --window_size 6000000 \
    --data_type "observed" --thresholds 0.01 0.05 --rem_k_strata 3 \
    --scale_range 1 7 24 1.5 --scale_range_mode "logspace" --gamma 0.75 --ridge_method 1 \
    --rotation_padding "mirror" --convolution_padding "nearest" --scale_selection "min_scale" \
    --angle_range 80 100 --noise_consec "" --noise_alt "" --sum_cond "a-r" --agg "sum" --top_k 10 \
    --num_cores 6 --verbose true
```







