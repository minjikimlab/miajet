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



def combine_results(results_dir, folder_pattern, result_type, exp_name=None):
    """
    Search for folders under `results_dir` matching `folder_pattern` (which may
    contain a glob-style asterisk), then for each folder:
      - load <folder>/<folder>_results_<result_type>/<folder>_summary_table.csv
      - load <folder>/<folder>_results_<result_type>/<folder>_expanded_table.csv
    and concatenate all summaries and all expanded tables into two DataFrames.

    Parameters
    ----------
    results_dir : str
        Path to the directory containing all result‐folders
    folder_pattern : str
        A glob style pattern (e.g. "splenic-B-cell_*_50Kb") used to match subfolders
    result_type : str
        The suffix after "_results" in each folder (e.g. "all", "p-0.01", etc)

    Returns
    -------
    (combined_summary, combined_expanded) : tuple of pd.DataFrame
        The concatenated summary and expanded tables
    """
    # build the full glob pattern
    search_pattern = os.path.join(results_dir, folder_pattern)

    prefix, suffix = folder_pattern.split('*')

    print(prefix, suffix)

    if exp_name is not None:
        exp_prefix, exp_suffix = exp_name.split('*')
        folder_regex = re.compile(
            rf"^{re.escape(prefix)}(.+?){re.escape(suffix)}$"
        )

    # If this notebook is already run, then do not include the (already combined) combined folder
    exclude_name = f"{prefix}_combined{suffix}"
    print(exclude_name)

    pattern = re.compile(
        rf"^{re.escape(prefix)}"      # “Repli-HiC_K562_WT_totalS_chr”
        r".+" # Changed to match any character because ce10 genome is roman numerals
        rf"{re.escape(suffix)}$"      # “_50Kb”
    )

    matched_folders = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if pattern.match(d)
    ]

    print(matched_folders)

    # Exclude the combined folder if it exists
    matched_folders = [
        f for f in matched_folders
        if os.path.basename(f) != exclude_name
    ]

    if not matched_folders:
        raise FileNotFoundError(f"No folders found matching {search_pattern}")

    summary_frames = []
    expanded_frames = []
    bedpe_frames = []
    for i, folder in enumerate(matched_folders):

        if exp_name is not None:
            m = folder_regex.match(os.path.basename(folder))
            if not m:
                raise ValueError(
                    f"Folder name '{os.path.basename(folder)}' "
                    f"did not match pattern '{folder_pattern}'"
                )
            wild = m.group(1)
            base = f"{exp_prefix}{wild}{exp_suffix}"
        else:
            # basename is the actual folder name without the path
            base = os.path.basename(folder)

        # path to the results‐type subdirectory
        res_subdir = os.path.join(folder, f"{base}_results_{result_type}")

        # csv paths
        summary_csv = os.path.join(res_subdir, f"{base}_summary_table.csv")
        expanded_csv = os.path.join(res_subdir, f"{base}_expanded_table.csv")
        juicer_bedpe = os.path.join(res_subdir, f"{base}_juicer-visualize.bedpe")

        # Load and collect
        if not os.path.isfile(summary_csv):
            raise FileNotFoundError(f"Expected file not found: {summary_csv}")
        if not os.path.isfile(expanded_csv):
            raise FileNotFoundError(f"Expected file not found: {expanded_csv}")
        if not os.path.isfile(juicer_bedpe):
            raise FileNotFoundError(f"Expected file not found: {juicer_bedpe}")
        
        # extract the comments from a single summary csv only
        if i == 0:
            comments = extract_comments(summary_csv)
            for comment in comments:
                print(comment)

        summary_frames.append(pd.read_csv(summary_csv, comment='#'))
        expanded_frames.append(pd.read_csv(expanded_csv, comment='#'))
        bedpe_frames.append(pd.read_csv(juicer_bedpe, comment='#', sep='\t', header=None, index_col=False, 
                                        names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']))

    # Concatenate all and reset the index
    combined_summary = pd.concat(summary_frames,  ignore_index=True)
    combined_expanded = pd.concat(expanded_frames, ignore_index=True)
    combined_bedpe = pd.concat(bedpe_frames, ignore_index=True)

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

    return combined_summary, combined_expanded, combined_bedpe, comments

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