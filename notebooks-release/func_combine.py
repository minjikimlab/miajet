import re
import os
import pandas as pd



def _chrom_order(chrom):
    """
    Map chr names to an integer order
    """
    m = re.match(r'^chr(\d+)$', chrom)
    if m:
        return int(m.group(1))
    if chrom in ('chrX', 'X'):
        return 23
    if chrom in ('chrY', 'Y'):
        return 24
    if chrom in ('chrM', 'chrMT', 'MT'):
        return 25
    # put everything else at the end
    return float('inf')


def extract_comments(csv_path: str) -> list[str]:
    """
    Read the leading comments “#” from a csv and return them 
    Stops as soon as it hits a non-# line
    """
    comments = []
    with open(csv_path, 'r') as fh:
        for line in fh:
            if line.startswith('#'):
                # strip newline, then replace chr<number> with chr*
                txt = line.rstrip('\n')
                txt = re.sub(r'chr\d+', 'chr*', txt)
                comments.append(txt)
            else:
                break
    return comments


def _parse_star_pattern(pattern: str) -> tuple[str, str]:
    """Split a single-* pattern into prefix/suffix."""
    if pattern.count('*') != 1:
        raise ValueError(f"Pattern must contain exactly one '*': {pattern}")
    return pattern.split('*')


def _resolve_base_name(folder: str, folder_pattern: str, exp_name=None) -> str:
    """
    Resolve the expected base run name for a chromosome folder.
    """
    if exp_name is None:
        return os.path.basename(folder)

    prefix, suffix = _parse_star_pattern(folder_pattern)
    exp_prefix, exp_suffix = _parse_star_pattern(exp_name)
    folder_regex = re.compile(rf"^{re.escape(prefix)}(.+?){re.escape(suffix)}$")

    folder_name = os.path.basename(folder)
    m = folder_regex.match(folder_name)
    if not m:
        raise ValueError(
            f"Folder name '{folder_name}' did not match pattern '{folder_pattern}'"
        )

    wild = m.group(1)
    return f"{exp_prefix}{wild}{exp_suffix}"


def _list_matched_folders(results_dir: str, folder_pattern: str) -> list[str]:
    """
    List folders matching `folder_pattern` under `results_dir`, excluding combined outputs.
    """
    search_pattern = os.path.join(results_dir, folder_pattern)
    prefix, suffix = _parse_star_pattern(folder_pattern)

    # If this notebook is already run, then do not include the (already combined) combined folder
    exclude_name = f"{prefix}_combined{suffix}"

    pattern = re.compile(
        rf"^{re.escape(prefix)}"
        r".+"  # supports non-numeric chromosomes (e.g. roman numerals)
        rf"{re.escape(suffix)}$"
    )

    matched_folders = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if pattern.match(d) and os.path.isdir(os.path.join(results_dir, d))
    ]

    matched_folders = [
        f for f in matched_folders
        if os.path.basename(f) != exclude_name
    ]

    if not matched_folders:
        raise FileNotFoundError(f"No folders found matching {search_pattern}")

    return sorted(matched_folders)


def _result_type_sort_key(result_type: str) -> tuple[int, str]:
    """Keep common result names in a human-friendly order."""
    if result_type == "all":
        return (0, result_type)
    if result_type.startswith("q_val"):
        return (1, result_type)
    if result_type == "thresholded":
        return (2, result_type)
    return (3, result_type)


def discover_result_subdirs(results_dir: str, folder_pattern: str, exp_name=None) -> list[str]:
    """
    Discover available `_results_<type>` subdirectories across matching chromosome folders.
    """
    matched_folders = _list_matched_folders(results_dir, folder_pattern)

    discovered = set()
    for folder in matched_folders:
        base = _resolve_base_name(folder, folder_pattern, exp_name=exp_name)
        prefix = f"{base}_results_"

        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if not os.path.isdir(full_path):
                continue
            if not entry.startswith(prefix):
                continue

            result_type = entry[len(prefix):]
            if result_type:
                discovered.add(result_type)

    if not discovered:
        raise FileNotFoundError(
            f"No result subdirectories found for folders matching {os.path.join(results_dir, folder_pattern)}"
        )

    return sorted(discovered, key=_result_type_sort_key)



def combine_results(results_dir, folder_pattern, result_type, exp_name=None, enforce_all_chroms=True):
    """
    Search for folders under `results_dir` matching `folder_pattern` (which may
    contain a glob-style asterisk), then for each folder:
      - load <folder>/<folder>_results_<result_type>/<folder>_summary_table.csv
      - load <folder>/<folder>_results_<result_type>/<folder>_expanded_table.csv
    and concatenate all summaries and all expanded tables into two DataFrames.

    Parameters
    ----------
    results_dir : str
        Path to the directory containing all result-folders
    folder_pattern : str
        A glob style pattern (e.g. "splenic-B-cell_*_50Kb") used to match subfolders
    result_type : str
        The suffix after "_results" in each folder (e.g. "all", "p-0.01", etc)

    Returns
    -------
    (combined_summary, combined_expanded) : tuple of pd.DataFrame
        The concatenated summary and expanded tables
    """
    matched_folders = _list_matched_folders(results_dir, folder_pattern)
    print(matched_folders)

    summary_frames = []
    expanded_frames = []
    bedpe_frames = []
    agg_frames = []
    feature_frames = []
    no_comment_found = True
    successful_chroms = []
    for i, folder in enumerate(matched_folders):

        base = _resolve_base_name(folder, folder_pattern, exp_name=exp_name)

        # path to the results-type subdirectory
        res_subdir = os.path.join(folder, f"{base}_results_{result_type}")

        # csv paths
        summary_csv = os.path.join(res_subdir, f"{base}_summary_table.csv")
        expanded_csv = os.path.join(res_subdir, f"{base}_expanded_table.csv")
        juicer_bedpe = os.path.join(res_subdir, f"{base}_juicer-visualize.bedpe")
        meta_subdir = os.path.join(folder, f"{base}_results_all") # metafiles (df_agg and df_features only exist in 'all' result_type)
        df_agg_csv = os.path.join(meta_subdir, "df_agg.csv")
        df_features_csv = os.path.join(meta_subdir, "df_features.csv")

        # Load and collect
        if not os.path.isfile(summary_csv):
            if not enforce_all_chroms:
                print(f"Warning: Expected file not found: {summary_csv}. Skipping this file.")
                continue
            raise FileNotFoundError(f"Expected file not found: {summary_csv}")
        if not os.path.isfile(expanded_csv):
            if not enforce_all_chroms:
                print(f"Warning: Expected file not found: {expanded_csv}. Skipping this file.")
                continue
            raise FileNotFoundError(f"Expected file not found: {expanded_csv}")
        if not os.path.isfile(juicer_bedpe):
            if not enforce_all_chroms:
                print(f"Warning: Expected file not found: {juicer_bedpe}. Skipping this file.")
                continue
            raise FileNotFoundError(f"Expected file not found: {juicer_bedpe}")
        
        # If at this stage, then chromosome ran successfully
        successful_chroms.append(folder)
        
        # extract the comments from a single summary csv only
        if no_comment_found:
            comments = extract_comments(summary_csv)
            for comment in comments:
                print(comment)
            no_comment_found = False

        summary_frames.append(pd.read_csv(summary_csv, comment='#'))
        expanded_frames.append(pd.read_csv(expanded_csv, comment='#'))
        bedpe_frames.append(pd.read_csv(juicer_bedpe, comment='#', sep='\t', header=None, index_col=False, 
                                        names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']))
        
        if os.path.isfile(df_agg_csv):
            agg_frames.append(pd.read_csv(df_agg_csv, comment='#'))
        if os.path.isfile(df_features_csv):
            feature_frames.append(pd.read_csv(df_features_csv, comment='#'))
        
    if len(summary_frames) == 0:
        print(f"No results found at all for {folder_pattern} with result type {result_type}")
        # return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
        raise ValueError
    
    print("Successfully read in:")
    for f in successful_chroms:
        print(f"  * {f}")

    # Concatenate all and reset the index
    combined_summary = pd.concat(summary_frames,  ignore_index=True)
    combined_expanded = pd.concat(expanded_frames, ignore_index=True)
    combined_bedpe = pd.concat(bedpe_frames, ignore_index=True)
    combine_df_agg = pd.concat(agg_frames, ignore_index=True) if agg_frames else None
    combine_df_features = pd.concat(feature_frames, ignore_index=True) if feature_frames else None

    # Sort 
    # First convert chromosomes to a numerical value for sorting
    combined_summary["chrom_order"] = combined_summary["chrom"].apply(_chrom_order)
    combined_expanded["chrom_order"] = combined_expanded["chrom"].apply(_chrom_order)
    combined_bedpe["chrom1_order"] = combined_bedpe["chrom1"].apply(_chrom_order)
    combined_bedpe["chrom2_order"] = combined_bedpe["chrom2"].apply(_chrom_order)
    # Then simply sort by chromosome order and position
    combined_summary.sort_values(by=["chrom_order"], inplace=True)
    combined_expanded.sort_values(by=["chrom_order"], inplace=True)
    combined_bedpe.sort_values(by=["chrom1_order", "start1", "chrom2_order", "start2"], inplace=True)
    # Drop the temporary chrom_order column 
    combined_summary.drop(columns=["chrom_order"], inplace=True)
    combined_expanded.drop(columns=["chrom_order"], inplace=True)
    combined_bedpe.drop(columns=["chrom1_order", "chrom2_order"], inplace=True)

    # Reset index 
    combined_summary.reset_index(drop=True, inplace=True)
    combined_expanded.reset_index(drop=True, inplace=True)
    combined_bedpe.reset_index(drop=True, inplace=True)

    return combined_summary, combined_expanded, combined_bedpe, comments, combine_df_agg, combine_df_features

# Save the combined tables
def save_csv(df, save_dir, parameter_str_comment):
    """
    Save a dataframe to a csv file 

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to save
    save_dir : str
        The name of the file to save the dataframe to
    parameter_str : list
        A list containing parameters to be included in the beginning of the csv file as comments
        Each element in the list is a string that must start with a special character (e.g '#')
    
    Returns
    -------
    None, but saves the dataframe to a csv file
    """
    with open(save_dir, "w") as f:
        f.write("\n".join(parameter_str_comment) + "\n")
        df.to_csv(f, index=False)