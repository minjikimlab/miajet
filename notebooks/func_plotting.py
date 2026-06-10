import itertools
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import numpy as np


from utils.plotting import format_ticks

# def plot_n_length_histograms(lengths, titles, bins=50, suptitle=None, show=False, save_name=None, sharex=True, sharey=True):
#     fig, ax = plt.subplots(1, len(lengths), figsize=(len(lengths) * 10, 6), layout="constrained", sharex=sharex, sharey=sharey)

#     ax = ax.flatten()

#     for i, l in enumerate(lengths):
#         unique_vals = np.unique(l.round(decimals=0))
#         if unique_vals.size == 1:
#             v = unique_vals[0]
#             bins_to_use = [v - 100e3, v + 100e3]
#         else:
#             bins_to_use = bins

#         ax[i].hist(l, bins=bins_to_use, linewidth=0.5, edgecolor='white', color="black")
#         sup_title = f"\nN={len(l)} Median={np.median(l):.2f} Mean={np.mean(l):.2f}"
#         ax[i].set_title(titles[i] + sup_title)
#         ax[i].set_xlabel('Genomic Length')

#         format_ticks(ax[i], x=True, y=False, rotate=False)

#         for spine in ax[i].spines.values():
#             spine.set_visible(False)
#         ax[i].tick_params(left=False, bottom=False)
#         ax[i].yaxis.grid(True, linestyle='--', alpha=0.4)

#         if unique_vals.size == 1:
#             ax[i].set_xlim(0, bins_to_use[-1] + 2e6)
#         else:
#             ax[i].set_xlim(left=0)

#         if not sharey:
#             ax[i].set_ylabel('Frequency')

#     if sharey:
#         ax[0].set_ylabel('Frequency')

#     if suptitle:
#         fig.suptitle(suptitle)

#     if save_name:
#         plt.savefig(save_name)
#     if show:
#         plt.show()
#     plt.close()

from matplotlib.ticker import EngFormatter

bp_formatter = EngFormatter(unit='b', places=1)

def plot_n_length_histograms(lengths, titles, bins=50, suptitle=None, show=False, save_name=None, sharex=True, sharey=True):
    fig, ax = plt.subplots(1, len(lengths), figsize=(len(lengths) * 10, 6), layout="constrained", sharex=sharex, sharey=sharey)

    if len(lengths) > 1:
        ax = ax.flatten()

    for i, l in enumerate(lengths):
        unique_vals = np.unique(l.round(decimals=0))
        if unique_vals.size == 1:
            v = unique_vals[0]
            bins_to_use = [v - 100e3, v + 100e3]
        else:
            bins_to_use = bins

        ax[i].hist(l, bins=bins_to_use, linewidth=0.5, edgecolor='white', color="black")
        sup_title = f"\nN={len(l)} Median={bp_formatter(np.median(l))} Mean={bp_formatter(np.mean(l))}"
        ax[i].set_title(titles[i] + sup_title, fontsize=18)
        ax[i].set_xlabel('Genomic Length', fontsize=15)

        format_ticks(ax[i], x=True, y=False, rotate=False)

        for spine in ax[i].spines.values():
            spine.set_visible(False)
        ax[i].tick_params(left=False, bottom=False, labelsize=16)
        ax[i].yaxis.grid(True, linestyle='--', alpha=0.4)

        if unique_vals.size == 1:
            ax[i].set_xlim(0, bins_to_use[-1] + 2e6)
        else:
            ax[i].set_xlim(left=0)

        if not sharey:
            ax[i].set_ylabel('Frequency', fontsize=15)

    if sharey:
        ax[0].set_ylabel('Frequency', fontsize=15)

    if suptitle:
        fig.suptitle(suptitle, fontsize=18)

    if save_name:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.close()



def plot_length_histogram(lengths, title, bins=50, show=False, save_name=None):
    plt.figure(figsize=(10, 6))

    unique_vals = np.unique(lengths.round(decimals=0))
    if unique_vals.size == 1:
        v = unique_vals[0]
        bins_to_use = [v - 100e3, v + 100e3]
    else:
        bins_to_use = bins

    plt.hist(lengths, bins=bins_to_use, linewidth=0.5, edgecolor='white', color="black")
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

def add_pairwise_sig(ax, positions, groups, y_pad=0.05, fontsize=8):
    """
    Draws significance bars for every pair of groups.
    For crowded plots keep only selected pairs 
    """
    # Identify which groups actually have data
    non_empty_idx = [i for i, g in enumerate(groups) if len(g) > 0]

    # If fewer than two non-empty groups, nothing to do
    if len(non_empty_idx) < 2:
        return

    # Compute min/max on just the non-empty slices
    y_max = max(max(groups[i]) for i in non_empty_idx)
    y_min = min(min(groups[i]) for i in non_empty_idx)
    h = (y_max - y_min) * y_pad

    # Only compare non-empty groups
    pairs = list(itertools.combinations(non_empty_idx, 2))

    for k, (i, j) in enumerate(pairs):
        # now i and j are guaranteed non-empty indexes
        p = ranksums(groups[i], groups[j]).pvalue

        x1, x2 = positions[i], positions[j]
        y = y_max + h * (k + 1)

        ax.plot([x1, x1, x2, x2],
                [y,   y + h/3, y + h/3, y],
                lw=1, c='k')
        ax.text((x1 + x2) / 2,
                y + h/2,
                format_sig(p),
                ha='center', va='bottom', fontsize=fontsize)
        

def title_boxplot(ax_title, side_by_side_titles, side_by_side_data):
    """
    Returns the axis title (str) for the boxplot with side-by-side boxplot statistics
    """
    stats = [boxplot_statistics(data) for data in side_by_side_data]
    title = f"{ax_title}\n" + "\n".join(f"{name}: {stat}" for name, stat in zip(side_by_side_titles, stats))
    return title