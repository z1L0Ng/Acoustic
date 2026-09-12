"""Render the final Figure 1 from the accompanying frozen summary tables.

Run with Python, NumPy, pandas and Matplotlib. Inputs and outputs are resolved
relative to this file, so the complete Figure/ package can be moved unchanged.
No audio, model, PCA or bootstrap computation is performed by this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / 'figure1_source_data'
STEM = 'figure1_dataset_characteristics'
ORDER = ['ICBHI', 'SPRSound', 'HF_Lung', 'KAUH']
CLASSES = ['Normal', 'Crackle', 'Wheeze', 'Both']

# Coordinate with Figure 2's Arial, slate text, and muted blue/green/purple/tan.
INK = '#213343'
RULE = '#d4dce3'
COLORS = {
    'ICBHI': '#58768f',
    'SPRSound': '#548271',
    'HF_Lung': '#947ba4',
    'KAUH': '#a29578',
}


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=9.0, fontweight='bold', pad=9)
    ax.set_ylabel(ylabel, fontsize=8.2, labelpad=5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.7)
    ax.tick_params(axis='both', length=2.5, width=.6, pad=3)
    ax.grid(axis='y', color=RULE, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)


def plot_overview(ax, groups, support):
    values = [groups.loc[groups.dataset == ds, 'band80_centroid_hz'].to_numpy()
              for ds in ORDER]
    boxes = ax.boxplot(
        values, positions=np.arange(4), widths=.53, patch_artist=True,
        showfliers=False, whis=1.5,
        medianprops={'color': INK, 'linewidth': 1.0},
        whiskerprops={'color': INK, 'linewidth': .7},
        capprops={'color': INK, 'linewidth': .7},
        boxprops={'linewidth': .8},
    )
    for box, ds in zip(boxes['boxes'], ORDER):
        box.set_edgecolor(COLORS[ds])
        box.set_facecolor(to_rgba(COLORS[ds], .27))
    names = ['ICBHI', 'SPRSound', 'HF Lung', 'KAUH']
    counts = support.set_index('dataset')['groups']
    ax.set_xticks(np.arange(4), [f'{name}\n{int(counts[ds])}'
                                for name, ds in zip(names, ORDER)])
    ax.tick_params(axis='x', labelsize=7.0)
    ax.set_xlim(-.38, 3.38)
    ax.set_ylim(80, 410)
    ax.set_yticks([100, 200, 300, 400])
    style_axis(ax, '(a) Dataset overview', 'Spectral centroid (Hz)')


def plot_classes(ax, summary, feature, title, ylabel):
    for ds, marker, offset in [('ICBHI', 'o', -.12), ('SPRSound', 's', .12)]:
        selected = summary[(summary.dataset == ds) & (summary.feature == feature)]
        selected = selected.set_index('compatible_label').loc[CLASSES]
        median = selected['median'].to_numpy()
        errors = np.vstack([median - selected.ci95_low.to_numpy(),
                            selected.ci95_high.to_numpy() - median])
        ax.errorbar(
            np.arange(4) + offset, median, yerr=errors, fmt=marker,
            markersize=4.0, markeredgewidth=.65, color=COLORS[ds],
            elinewidth=.85, capsize=2.2, capthick=.8, zorder=3,
        )
    ax.set_xticks(np.arange(4), CLASSES)
    ax.set_xlim(-.52, 3.52)
    ax.tick_params(axis='x', labelsize=7.0)
    style_axis(ax, title, ylabel)


def main():
    groups = pd.read_csv(DATA / 'overview_group_centroids.csv', dtype={'group_id': str})
    support = pd.read_csv(DATA / 'overview_support.csv')
    summary = pd.read_csv(DATA / 'class_matched_summary.csv')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.5,
        'text.color': INK,
        'axes.labelcolor': INK,
        'axes.edgecolor': INK,
        'axes.titlecolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
        'savefig.facecolor': 'white',
    })
    # Exact 7-inch page width; no tight bounding-box resizing on export.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.26))
    fig.subplots_adjust(left=.071, right=.992, bottom=.245, top=.84, wspace=.41)
    plot_overview(axes[0], groups, support)
    plot_classes(axes[1], summary, 'dbfs_dc', '(b) Class-matched level', 'Level (dBFS)')
    axes[1].set_ylim(-56, -10)
    axes[1].set_yticks([-50, -40, -30, -20, -10])
    plot_classes(axes[2], summary, 'band80_centroid_hz',
                 '(c) Class-matched spectrum', 'Spectral centroid (Hz)')
    axes[2].set_ylim(125, 235)
    axes[2].set_yticks([140, 160, 180, 200, 220])
    fig.text(.217, .047, 'Number of groups', ha='center',
             fontsize=6.7, color='#4e6172')
    handles = [Line2D([], [], linestyle='none', marker=marker, color=COLORS[ds],
                      markersize=4.2, label=ds)
               for ds, marker in [('ICBHI', 'o'), ('SPRSound', 's')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.682, .012),
               ncol=2, frameon=False, fontsize=8, handletextpad=.4,
               columnspacing=1.8, borderaxespad=0)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(HERE / f'{STEM}.{ext}', dpi=400)
    plt.close(fig)
    print(f'Created {STEM}.pdf/.svg/.png at 7.00 x 2.26 inches.')


if __name__ == '__main__':
    main()
