"""
plotting_selectivity.py — Neuron-selectivity figure.

Produces:
  • Figure 2: Joint scatter / AUC / ANOVA figure (hand vs orientation selectivity).

Figures saved under Results/Selectivity/figures/:
    neuronSelection_merged{n}_bin{b}_k{k}_joint.png

Data loaded from Results/Selectivity/data/ (scores, ANOVA) and
Results/Population/data/ (monkeyNeurons.csv).
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
from matplotlib import cm
import seaborn as sns
import scipy as sp

from config import (
    POPULATION_DIR,
    SELECTIVITY_DIR,
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
)
from utils.plot_utils import scatter_hist, apply_theme
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
POPULATION_DATA  = os.path.join(POPULATION_DIR,  'data')
SELECTIVITY_DATA = os.path.join(SELECTIVITY_DIR, 'data')
SELECTIVITY_FIGS = os.path.join(SELECTIVITY_DIR, 'figures')
os.makedirs(SELECTIVITY_FIGS, exist_ok=True)

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
n_monkeys    = len(monkey_name_map)
regions      = ['M1', 'PMd', 'PMv']
SCATTER_CMAP = 'rocket'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
neuron_csv_filename = os.path.join(POPULATION_DATA, 'monkeyNeurons.csv')
anova_file_name     = os.path.join(
    SELECTIVITY_DATA, f'anova_results_merged{n_monkeys}.p')

neuron_df = pd.read_csv(neuron_csv_filename, index_col=0)
n_neurons = neuron_df.sum(axis=1).to_dict()

score_data = {}
for region in regions:
    score_filename = os.path.join(
        SELECTIVITY_DATA,
        f'neuronSelection_scores_merged{n_monkeys}_bin{binsize}_k{kernel_width}_{region}.p')
    with open(score_filename, 'rb') as f:
        score_data[region] = pkl.load(f)

with open(anova_file_name, 'rb') as f:
    anova_data = pkl.load(f)

print('Data loaded.')

# ---------------------------------------------------------------------------
# Figure 2: Joint scatter / AUC / ANOVA figure
# ---------------------------------------------------------------------------
anova_significance = anova_data['significance']
all_tr      = np.concatenate([score_data[r]['total_rates'] for r in regions])
global_norm = mpl.colors.LogNorm(
    vmin=float(np.nanmin(all_tr[all_tr > 0])),
    vmax=float(np.nanmax(all_tr)),
)

joint_fig   = plt.figure(figsize=(11, 8.5), layout='constrained')
outer_cols  = joint_fig.subfigures(1, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.03)
cbar_subfig = outer_cols[3]
anova_axes  = {}
col_subfigs = {}

for reg_idx, region in enumerate(regions):
    d = score_data[region]
    hemisphere_score  = d['hemi_score']
    orientation_score = d['orient_score']
    tr                = d['total_rates']

    column = outer_cols[reg_idx].subfigures(3, 1, height_ratios=[1.5, 1, 1])
    col_subfigs[region] = column

    # ── Scatter plot ───────────────────────────────────────────────────────
    upper_mosaic = [['histx', '.'], ['scatter', 'histy']]
    axs          = column[0].subplot_mosaic(upper_mosaic,
                                            width_ratios=(5, 1.2),
                                            height_ratios=(1, 5))
    ax           = axs['scatter']
    last_scatter = scatter_hist(ax=axs['scatter'], ax_histx=axs['histx'],
                                ax_histy=axs['histy'],
                                y=orientation_score, x=hemisphere_score,
                                c=tr, norm=global_norm, cmap=SCATTER_CMAP)

    xmed = np.nanmean(hemisphere_score)
    ymed = np.nanmean(orientation_score)
    t_hemi, p_hemi = sp.stats.ttest_1samp(hemisphere_score, popmean=0, nan_policy='omit')
    t_ori,  p_ori  = sp.stats.ttest_1samp(orientation_score, popmean=0, nan_policy='omit')
    print(f'{region}: Hand: Mean = {hemisphere_score.mean():.3f} p = {p_hemi:.3f}, '
          f'Orient: Mean = {orientation_score.mean():.3f} p = {p_ori:.3f}')
    arrow_color_hem = 'black' if p_hemi < 0.01 else 'gray'
    arrow_color_ori = 'black' if p_ori  < 0.01 else 'gray'
    _, top_histx    = axs['histx'].get_ylim()
    _, right_histy  = axs['histy'].get_xlim()
    axs['histx'].plot(xmed, top_histx, 'v', color=arrow_color_hem, markersize=9,
                      clip_on=False, zorder=5)
    axs['histy'].plot(right_histy, ymed, '<', color=arrow_color_ori, markersize=9,
                      clip_on=False, zorder=5)
    if reg_idx == 0:
        ax.set_ylabel('Orientation Score (V - H)')
        _subfig_panel_label(column[0], 'A')
    ax.set_xlabel('Hand Score (C - I)')
    ax.set_xlim([-1.03, 1.03])
    ax.set_ylim([-1.03, 1.03])
    outer_cols[reg_idx].text(
        0.5, 1.01, region,
        transform=outer_cols[reg_idx].transSubfigure,
        fontsize=REGION_TITLE_SIZE, fontweight='bold',
        ha='center', va='bottom',
        clip_on=False,
    )

    # ── ECDF / AUC plot ────────────────────────────────────────────────────
    ax_cum = column[1].subplots(1)
    res_hem = sp.stats.ecdf(np.abs(hemisphere_score))
    res_ori = sp.stats.ecdf(np.abs(orientation_score))
    q_hem   = res_hem.cdf.quantiles
    q_ori   = res_ori.cdf.quantiles
    area_hem = ((q_hem[1:] - q_hem[:-1]) * np.arange(1, q_hem.shape[0])).sum() / q_hem.shape[0]
    area_ori = ((q_ori[1:] - q_ori[:-1]) * np.arange(1, q_ori.shape[0])).sum() / q_ori.shape[0]
    ax_cum.ecdf(np.abs(hemisphere_score), label=f'Hand AUC: {area_hem:.2f}', c='b')
    ax_cum.ecdf(np.abs(orientation_score), label=f'Orient AUC: {area_ori:.2f}', c='r')
    ax_cum.plot([0, 1], [0, 1], 'k--')
    ax_cum.set_ylabel('Cumulative Fraction')
    ax_cum.set_xlabel('abs Hand and Orient Index')
    ax_cum.set_xlim([0, 1])
    ax_cum.set_ylim([0, 1])
    ax_cum.legend()
    if reg_idx == 0:
        _subfig_panel_label(column[1], 'B')

    anova_axes[region] = column[2]

# Shared colorbar
SCATTER_TOP    = 0.88
SCATTER_BOTTOM = 0.65
cax = cbar_subfig.add_axes([0.2, SCATTER_BOTTOM, 0.35, SCATTER_TOP - SCATTER_BOTTOM])
cbar = cbar_subfig.colorbar(
    cm.ScalarMappable(norm=global_norm, cmap=SCATTER_CMAP),
    cax=cax,
)
cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Mean firing rate (spikes/s)')

# ── ANOVA bar chart row ────────────────────────────────────────────────────
ANOVA_COLORS  = ['tab:blue', 'tab:red', 'tab:gray']
ANOVA_EFFECTS = ['hand', 'orientation', 'interaction']
ANOVA_LABELS  = ['Hand', 'Orientation', 'Hand × Ori.']
EPOCH_LIST    = list(epochs)
bw   = 0.22
xpos = np.arange(len(EPOCH_LIST))
offs = np.array([-bw, 0, bw])

anova_sig_rates = {region: {epoch: [] for epoch in epochs} for region in regions}
for region in regions:
    for epoch in epochs:
        anova_sig_rates[region][epoch] = anova_significance[region][epoch].mean(axis=0)

for reg_idx, region in enumerate(regions):
    ax_an = anova_axes[region].subplots(1)
    for fi, (effect, color, label) in enumerate(
            zip(ANOVA_EFFECTS, ANOVA_COLORS, ANOVA_LABELS)):
        vals = [anova_sig_rates[region][ep][fi] for ep in EPOCH_LIST]
        ax_an.bar(xpos + offs[fi], vals, width=bw * 0.9, color=color, label=label)
    ax_an.set_xticks(xpos)
    ax_an.set_xticklabels(EPOCH_LIST)
    ax_an.set_ylim([0, 0.8])
    ax_an.axhline(0, color='#c3c2b7', lw=1)
    ax_an.tick_params(axis='y')
    ax_an.text(0.98, 0.97, f'n = {n_neurons[region]}',
               transform=ax_an.transAxes, ha='right', va='top')
    if reg_idx == 0:
        ax_an.set_ylabel('Prop. sig. neurons')
        ax_an.legend(fontsize=10, frameon=False, loc='upper left',
                     handlelength=1, handleheight=0.8)
        _subfig_panel_label(col_subfigs['M1'][2], 'C')
    else:
        ax_an.set_yticklabels([])
    ax_an.spines[['top', 'right']].set_visible(False)

out2 = os.path.join(
    SELECTIVITY_FIGS,
    f'neuronSelection_merged{n_monkeys}_bin{binsize}_k{kernel_width}_joint.png')
joint_fig.savefig(out2, bbox_inches='tight', dpi=DPI_DRAFT)
plt.close(joint_fig)
print(f'Saved {out2}')

print('\nplotting_selectivity.py complete.')
