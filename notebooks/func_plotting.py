import itertools
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import numpy as np


from utils.plotting import format_ticks


def plot_length_histogram(lengths, title, bins=50, show=False, save_name=None):
    plt.figure(figsize=(10, 6))

    unique_vals = np.unique(lengths.round(decimals=0))
    if unique_vals.size == 1:
        v = unique_vals[0]
        bins_to_use = [v - 100e3, v + 100e3]
    else:
        bins_to_use = bins

    plt.hist(lengths, bins=bins_to_use, linewidth=0.5, edgecolor='white', color="#dddddd")
    sup_title = f"\nN={len(lengths)} Median={np.median(lengths):.2f} Mean={np.mean(lengths):.2f}"
    plt.title(title + sup_title)
    plt.xlabel('Genomic Length')

    ax = plt.gca()

    format_ticks(ax, x=True, y=False, rotate=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    if unique_vals.size == 1:
        ax.set_xlim(0, bins_to_use[-1] + 2e6)
    else:
        ax.set_xlim(left=0)

    plt.ylabel('Frequency')
    if save_name:
        plt.savefig(save_name)

    if show:
        plt.show()

    plt.close()




def boxplot_statistics(boxplot_data):
    """
    Computes statistics of boxplot data assuming that `boxplot_data` is a single boxplot
    Returns a python string that can be put into the boxplot title
    """
    mean = np.mean(boxplot_data)
    median = np.median(boxplot_data)
    std = np.std(boxplot_data)
    N = len(boxplot_data)

    return f"median={median:.3g} | N={N}"

sig_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

def format_sig(p):
    # find the star code (or empty string)
    stars = next((s for thr, s in sig_levels if p < thr), '')
    # always show the p-value too
    return f"{stars} (p={p:.2g})"

def add_side_stats(ax, stats, xpos=-0.35, ystart=0.95, dy=0.05, fontsize=10):
    """
    Writes one line per group down the left edge of the axes.
    `stats` is a list of strings – one per group.
    """
    for i, txt in enumerate(stats):
        ax.text(xpos, ystart - i*dy, txt,
                transform=ax.transAxes, fontsize=fontsize,
                ha='left', va='top')

def add_pairwise_sig(ax, positions, groups, y_pad=0.05):
    """
    Draws significance bars for every pair of groups.
    For crowded plots keep only selected pairs 
    """
    # Only consider groups that actually have data for y‐scale
    non_empty = [g for g in groups if len(g) > 0]
    if not non_empty:
        # nothing to plot
        return

    y_max = max(max(g) for g in non_empty)          # top of the tallest box
    y_min = min(min(g) for g in non_empty)
    h = (y_max - y_min) * y_pad


    pairs = list(itertools.combinations(range(len(groups)), 2))
    for k, (i, j) in enumerate(pairs):

        if len(groups[i]) == 0 or len(groups[j]) == 0:
            continue

        # Test
        p = ranksums(groups[i], groups[j]).pvalue

        x1, x2 = positions[i], positions[j]
        y = y_max + h*(k+1)

        
        ax.plot([x1, x1, x2, x2], [y, y+h/3, y+h/3, y],
                lw=1, c='k')
        
        ax.text((x1+x2)/2, y+h/2, format_sig(p),
                ha='center', va='bottom', fontsize=8)
        

def title_boxplot(ax_title, side_by_side_titles, side_by_side_data):
    """
    Returns the axis title (str) for the boxplot with side-by-side boxplot statistics
    """
    stats = [boxplot_statistics(data) for data in side_by_side_data]
    title = f"{ax_title}\n" + "\n".join(f"{name}: {stat}" for name, stat in zip(side_by_side_titles, stats))
    return title