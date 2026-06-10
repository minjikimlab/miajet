import numpy as np
from pyBedGraph import BedGraph


def compute_length(group, p=2):
    """
    Compute the length of each jet in a group by summing the 
    adjacent Euclidean distances between each point in the jet

    Assumes that the points of the jet is in order  
    """

    points = group[['x (bp)', 'y (bp)']].values
    if len(points) < 2:
        return 0.0  # If there's only one point, length is zero
    distances = np.sum(np.diff(points, axis=0)**p, axis=1) ** (1 / p)
    return np.sum(distances)


def extract_chipseq_values(chip_files, intervals, f_chrom_sizes, chromosomes, names, stat):
    # List of list where each sublist corresponds to a chip-seq experiment
    # Each chip-seq experiment contains a list of dictionaries corresponding to each jet caller
    chrom_sizes = {}
    with open(f_chrom_sizes) as chrom_size_file:
        for line in chrom_size_file:
            fields = line.split()
            if len(fields) < 2:
                continue
            chrom_sizes[fields[0]] = int(fields[1])

    chipseq_values = []

    for i, f_chip in enumerate(chip_files):
        # loop through each chip-seq experiment

        # bg = BedGraph(f_chrom_sizes, f_chip, min_value=-1e10)
        bg = BedGraph(f_chrom_sizes, f_chip, chroms_to_load=list(chromosomes), min_value=-1e10) # CHANGED 05/11/26

        chip_val = []
        for j, inter in enumerate(intervals):
            # loop through each jet caller method

            required_cols = {"chrom", "start", "end", "unique_id"}

            if inter is None or inter.empty:
                print(f"No intervals for {names[j]}")
                chip_val.append({})
                continue

            missing_cols = required_cols.difference(inter.columns)
            if missing_cols:
                raise KeyError(f"intervals[{j}] is missing required columns: {sorted(missing_cols)}")

            # genome wide
            unique_ids = []
            values = []
            for chrom in chromosomes:

                aliases = [chrom]
                if str(chrom).startswith("chr"):
                    stripped = str(chrom).replace("chr", "", 1)
                    if stripped:
                        aliases.append(stripped)
                else:
                    aliases.append(f"chr{chrom}")

                inter_chrom = inter.loc[inter["chrom"].isin(aliases)].copy().reset_index(drop=True)

                if inter_chrom.empty:
                    print(f"No jets called for chromosome {chrom} by {names[j]}")
                    continue

                bg_chrom = None
                for alias in aliases:
                    if bg.has_chrom(alias):
                        bg_chrom = alias
                        break

                if bg_chrom is None:
                    print(f"Chromosome {chrom} not found in {f_chip}. Skipping...")
                    continue

                # must do one chromosome at a time
                bg.load_chrom_data(bg_chrom)

                chrom_size = None
                for alias in aliases:
                    if alias in chrom_sizes:
                        chrom_size = chrom_sizes[alias]
                        break

                if chrom_size is None or chrom_size <= 1:
                    print(f"Chromosome size not found for {chrom}. Skipping...")
                    continue

                start = np.floor(inter_chrom["start"].to_numpy(dtype=float)).astype(np.int64)
                end = np.ceil(inter_chrom["end"].to_numpy(dtype=float)).astype(np.int64)

                start = np.clip(start, 0, chrom_size - 1)
                end = np.clip(end, 1, chrom_size)

                valid = end > start
                if not np.any(valid):
                    continue

                inter_chrom = inter_chrom.loc[valid].reset_index(drop=True)
                start = start[valid].astype(np.int32)
                end = end[valid].astype(np.int32)

                v = bg.stats(stat=stat, start_list=start, end_list=end, chrom_name=bg_chrom)
                u = inter_chrom["unique_id"].values

                values.extend(list(v))
                unique_ids.extend(list(u))
            
            chip_val.append(dict(zip(unique_ids, values)))

        chipseq_values.append(chip_val)

    return chipseq_values


