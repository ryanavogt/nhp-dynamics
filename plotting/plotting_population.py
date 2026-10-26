"""
plotting_population.py — Population-level figures.

Produces:
  • Figure 1: Neuron count table.
  • Figure 4: Epoch-of-max-discharge proportions (Panel A) + Grasp heatmaps (Panel B).

Figures saved under Results/Population/figures/:
    monkeyNeurons.png
    EpochMaxFiring_merged{n}.png

Data loaded from Results/Population/data/.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pandas.plotting import table

from config import (
    POPULATION_DIR,
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
    WINDOW_RANGE,
)
from utils.plot_utils import apply_theme
from plot_config import (
    FIGURE_WIDTH,
    DPI_DRAFT, DPI_FINAL,
    PANEL_LABEL_SIZE, REGION_TITLE_SIZE,
    LABEL_SIZE, TICK_SIZE,
    ANNOT_SIZE, ANNOT_SMALL,
    apply_rcparams,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()
apply_rcparams()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
POPULATION_DATA = os.path.join(POPULATION_DIR, 'data')
POPULATION_FIGS = os.path.join(POPULATION_DIR, 'figures')
os.makedirs(POPULATION_FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _panel_label(ax, letter, x=-0.12, y=1.05):
    """Bold panel label at top-left of *ax* in axes coordinates."""
    ax.text(
        x, y, r'$\mathbf{' + letter + r'}$',
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE, fontweight='bold',
        ha='left', va='bottom',
        clip_on=False,
    )

def _subfig_panel_label(subfig, letter):
    """Bold panel label anchored to the top-left of a SubFigure."""
    subfig.text(
        0, 1.0, r'$\mathbf{' + letter + r'}$',
        transform=subfig.transSubfigure,
        fontsize=PANEL_LABEL_SIZE, fontweight='bold',
        ha='left', va='bottom',
        clip_on=False,
    )

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n_monkeys = len(monkey_name_map)
regions   = ['M1', 'PMd', 'PMv']

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
neuron_csv_filename = os.path.join(POPULATION_DATA, 'monkeyNeurons.csv')
neuron_df           = pd.read_csv(neuron_csv_filename, index_col=0)
n_neurons           = neuron_df.sum(axis=1).to_dict()

neuron_peaks_filename   = os.path.join(POPULATION_DATA, 'neuron_peaks.p')
all_orient_sdf_filename = os.path.join(
    POPULATION_DATA,
    f'merged_sdfDict_bin{binsize}_k{kernel_width}_orientMerged.p')
max_rate_filename       = os.path.join(
    POPULATION_DATA,
    f'sdfMaxRate_bin{binsize}_k{kernel_width}_orientMerged.p')
peak_order_filename     = os.path.join(
    POPULATION_DATA,
    f'peakOrderDict_bin{binsize}_k{kernel_width}_orientMerged.p')
heatmap_corr_filename   = os.path.join(
    POPULATION_DATA,
    f'heatmapCorr_bin{binsize}_k{kernel_width}_orientMerged.p')

with open(neuron_peaks_filename, 'rb') as f:
    event_neuron_max = pkl.load(f)
with open(all_orient_sdf_filename, 'rb') as f:
    all_orient_sdf_dict = pkl.load(f)
with open(max_rate_filename, 'rb') as f:
    area_max_rate = pkl.load(f)
with open(peak_order_filename, 'rb') as f:
    peak_order_dict = pkl.load(f)
with open(heatmap_corr_filename, 'rb') as f:
    heatmap_corr = pkl.load(f)

print('Data loaded.')

# ---------------------------------------------------------------------------
# Figure 1: Neuron count table
# ---------------------------------------------------------------------------
fig_dims   = (5, 1.5)
neuron_fig = plt.figure(figsize=fig_dims)
ax         = plt.subplot(111, frame_on=False)
ax.set_xticks([])
ax.xaxis.set_label_position('top')
ax.set_yticks([])
table(ax, neuron_df, loc='center')
plt.title('Monkey Region Neurons')
out1 = os.path.join(POPULATION_FIGS, 'monkeyNeurons.png')
plt.savefig(out1, dpi=DPI_DRAFT, bbox_inches='tight')
plt.close(neuron_fig)
print(f'Saved {out1}')

# ---------------------------------------------------------------------------
# Figure 4: Epoch-of-max proportions (Panel A) + Grasp heatmaps (Panel B)
# ---------------------------------------------------------------------------
grasp_event    = epoch_window_map['Grasp']['event']
active_epochs4 = [ep for ep in epochs if ep != 'Pre-Cue']
x_epochs4      = np.arange(len(active_epochs4))
heatmap_window = np.array(WINDOW_RANGE)

side_keys4   = ['ipsilateral', 'contralateral']
side_labels4 = ['Ipsilateral', 'Contralateral']
side_colors4 = ['#8B7DC8', '#E8927C']
bar_width4   = 0.35

bar_max4 = max(
    vals.max() * 100
    for region in regions
    for vals in event_neuron_max[region]['proportions'].values()
)
bar_ylim4 = bar_max4 * 1.3

fig4      = plt.figure(figsize=(11, 6), layout='constrained')
fig4_cols = fig4.subfigures(1, 3, wspace=0.06)

for reg_idx, region in enumerate(regions):
    col = fig4_cols[reg_idx]

    sf_bar4, sf_heat4 = col.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)

    # ── Panel A: epoch-of-max-discharge bar chart ────────────────────────────
    ax_bar4     = sf_bar4.subplots(1)
    n_neurons_r = event_neuron_max[region]['neurons']

    for s_idx, (side_key, side_label, color) in enumerate(
            zip(side_keys4, side_labels4, side_colors4)):
        prop = event_neuron_max[region]['proportions'][side_key]
        ax_bar4.bar(x_epochs4 + s_idx * bar_width4,
                    prop * 100, bar_width4,
                    label=side_label, color=color)

    ax_bar4.set_xticks(x_epochs4 + bar_width4 / 2)
    ax_bar4.set_xticklabels([ep.split(' ')[0] for ep in active_epochs4])
    ax_bar4.set_ylim([0, bar_ylim4])
    ax_bar4.text(0.98, 0.96, f'n = {n_neurons_r}',
                 transform=ax_bar4.transAxes, ha='right', va='top', fontsize=9)
    ax_bar4.spines[['top', 'right']].set_visible(False)

    if reg_idx == 0:
        ax_bar4.set_ylabel('Neurons at max (%)')
        ax_bar4.legend(frameon=False, fontsize=9, loc='upper left',
                       handlelength=1, handleheight=0.8)
        _subfig_panel_label(sf_bar4, 'A')

    col.text(0.5, 1.01, region,
             transform=col.transSubfigure,
             fontsize=REGION_TITLE_SIZE, fontweight='bold',
             ha='center', va='bottom', clip_on=False)

    # ── Panel B: Grasp heatmaps (Contralateral | Ipsilateral) ────────────────
    ax_contra4, ax_ipsi4 = sf_heat4.subplots(1, 2, sharey=True)
    lat_heat_axes4 = {'c': ax_contra4, 'i': ax_ipsi4}

    max_rate4   = area_max_rate[f'c{region}']
    peak_order4 = peak_order_dict[f'c{region}'][grasp_event]
    n_neurons_h = len(peak_order4)

    y_grid4, x_grid4 = np.mgrid[
        1:n_neurons_h + 2:1,
        heatmap_window[0] / 1000:heatmap_window[1] / 1000 + 2 * binsize / 1000:binsize / 1000
    ]

    for lat in ['c', 'i']:
        ax     = lat_heat_axes4[lat]
        sdf    = all_orient_sdf_dict[f'{lat}{region}'][grasp_event]
        a      = sdf.T
        scaled = np.divide(a, max_rate4[:, np.newaxis],
                           where=max_rate4[:, np.newaxis] > 0,
                           out=np.zeros_like(a))
        ax.pcolor(x_grid4, y_grid4, scaled[peak_order4],
                  cmap='inferno', norm=mpl.colors.Normalize(vmin=0, vmax=1.5))
        ax.axvline(x=0, color='w', linestyle=':', lw=0.8)
        lat_name = 'Contralateral' if lat == 'c' else 'Ipsilateral'
        ax.set_title(lat_name, fontsize=10)
        ax.set_xlabel('Time from Grasp (s)', fontsize=9)
        if lat == 'i':
            ax.set_yticks([])
        elif reg_idx == 0:
            ax.set_ylabel('Neuron #')

    ax_contra4.invert_yaxis()

    r_val = heatmap_corr[region][grasp_event]['r_peak_timing']
    p_val = heatmap_corr[region][grasp_event]['p_peak_timing']
    ax_contra4.text(
        0.04, 0.04, f'r = {r_val:.2f}',
        transform=ax_contra4.transAxes,
        color='white', fontsize=8.5, ha='left', va='bottom',
        fontweight='bold',
    )

    if reg_idx == 0:
        _subfig_panel_label(sf_heat4, 'B')

out4 = os.path.join(POPULATION_FIGS, f'EpochMaxFiring_merged{n_monkeys}.png')
fig4.savefig(out4, bbox_inches='tight', dpi=DPI_DRAFT)
plt.close(fig4)
print(f'Saved {out4}')

print('\nplotting_population.py complete.')
