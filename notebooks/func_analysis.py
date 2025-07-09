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
    chipseq_values = []

    for i, f_chip in enumerate(chip_files):
        # loop through each chip-seq experiment

        bg = BedGraph(f_chrom_sizes, f_chip)

        chip_val = []
        for j, inter in enumerate(intervals):
            # loop through each jet caller method

            # genome wide
            unique_ids = []
            values = []
            for chr in chromosomes:

                # must do one chromosome at a time
                try:
                    bg.load_chrom_data(chr)
                except KeyError:
                    try:
                        # Strip away the "chr" prefix
                        bg.load_chrom_data(chr.replace("chr", ""))  
                    except KeyError:
                        print(f"Chromosome {chr} not found in {f_chip}. Skipping...")
                        # Skip this chromosome if it doesn't exist
                        continue

                inter_chrom = inter.loc[inter["chrom"] == chr].reset_index(drop=True)

                if inter_chrom.empty:
                    print(f"No jets called for chromosome {chr} by {names[j]}")
                    continue

                v = bg.stats(stat=stat, intervals=inter_chrom[["chrom", "start", "end"]].values)
                u = inter_chrom["unique_id"].values

                values.extend(list(v))
                unique_ids.extend(list(u))
            
            chip_val.append(dict(zip(unique_ids, values)))

        chipseq_values.append(chip_val)

    return chipseq_values


