"""
pca_plot_neurons.py — Neuron PC-contribution figures.

Loads the allMonkeyDict and MergedModDataFrame, computes eigenvalue-weighted
neuron contribution weights, and produces two sets of figures:

  1. neuronMonkeyCont.png
       Per-cortex bar chart of contribution weights (coloured by monkey) and
       empirical CDF with KS-uniformity test.

  2. neuronModTypeCont{Cue|Grasp}_orientMerged.png
       Per-cortex bar chart of contribution weights coloured by modulation type
       (orientations merged) and empirical CDF, for each task epoch.

Figures saved under Results/PCA/figures/.
Data loaded from Results/PCA/data/PCA_b{b}_k{k}/ and Results/Modulation/data/.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from utils.plot_utils import apply_theme

from config import (
    MONKEY_NAME_MAP as monkey_name_map,
    MODULATION_DIR,
    PCA_RESULTS_DIR,
    BIN_SIZE        as binsize,
    KERNEL_WIDTH    as kernel_width,
)
from plot_config import DPI_DRAFT, apply_rcparams

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()
apply_rcparams()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORTEX_LIST = ['M1', 'PMd', 'PMv']

PCA_DATA        = os.path.join(PCA_RESULTS_DIR, 'data')
PCA_FIG_DIR     = os.path.join(PCA_RESULTS_DIR, 'figures')
MODULATION_DATA = os.path.join(MODULATION_DIR,  'data')
pca_dir         = os.path.join(PCA_DATA, f'PCA_b{binsize}_k{kernel_width}')
os.makedirs(PCA_FIG_DIR, exist_ok=True)

color_map = {'R': 'tab:red', 'G': 'tab:green', 'Y': 'tab:orange', 'B': 'tab:blue'}
mod_color_map = {
    'ipsi':       'tab:green',
    'both:ipsi':  'tab:blue',
    'both:equal': 'tab:brown',
    'both:opp':   'tab:olive',
    'both:contra':'tab:orange',
    'contra':     'tab:red',
    'none':       'tab:gray',
    'both:all':   'tab:brown',
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading data ...')
monkey_dict_file = os.path.join(pca_dir, 'allMonkeyDict.p')
mod_df_file      = os.path.join(MODULATION_DATA, 'MergedModDataFrame.p')

with open(monkey_dict_file, 'rb') as f:
    monkeys = pkl.load(f)
with open(mod_df_file, 'rb') as f:
    mod_df = pkl.load(f)

merged_monkey = monkeys['All']
print('Data loaded.')

# ---------------------------------------------------------------------------
# Figure 1 — Per-monkey neuron contribution
# ---------------------------------------------------------------------------
print('\nPlotting per-monkey neuron contribution ...')
joint_fig, joint_axes = plt.subplots(
    nrows=2, ncols=3, figsize=(12, 6),
    gridspec_kw={'height_ratios': [1.5, 1]}
)

for cort_idx, cortex in enumerate(CORTEX_LIST):
    cort_ax = joint_axes[0][cort_idx]
    cdf_ax  = joint_axes[1][cort_idx]

    # Eigenvalue-weighted contribution weight
    merged_S  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['S']
    merged_V  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
    eigvals   = merged_S[:3] ** 2
    eig_total = (merged_S ** 2).sum()
    V_plot, V_indices = (
        (eigvals * merged_V[:, :3] ** 2).sum(dim=1) / eig_total
    ).sqrt().sort(descending=True, dim=0)

    n_neurons   = merged_monkey.cortices[cortex]['neurons']
    neuron_list = np.arange(n_neurons)
    cont_total  = V_plot.sum()

    for m_name, monkey in monkeys.items():
        if m_name == 'All':
            continue
        m_indices      = monkey.monkey_indices[cortex]
        monkey_neurons = monkey.cortices[cortex]['neurons']
        monkey_x       = np.where(
            (V_indices <= m_indices[-1]) & (V_indices >= m_indices[0])
        )
        mod_ecdf = stats.ecdf(neuron_list[monkey_x])
        ks_stat  = stats.ks_1samp(neuron_list[monkey_x] / n_neurons,
                                   stats.uniform.cdf)
        p_val = ks_stat[1]
        mod_ecdf.cdf.plot(cdf_ax, label=f'{m_name}, p:{p_val:.2f}',
                          c=color_map[m_name])
        monkey_V = V_plot[monkey_x].sum()
        cort_ax.bar(
            x=monkey_x[0], width=0.85, height=V_plot[monkey_x[0]],
            label=(f'{m_name}, n:{monkey_neurons}, '
                   f'T:{monkey_V/cont_total:.2f}, '
                   f'Avg:{(monkey_V/cont_total)/monkey_x[0].shape[0]*100:.2f}'),
            color=color_map[m_name],
        )

    cdf_ax.plot([0, n_neurons], [0, 1], 'k--')
    cdf_ax.set(
        ylabel='Cumulative Fraction', xlabel='Neuron Index',
        xlim=[0, n_neurons], ylim=[0, 1],
    )
    cdf_ax.legend(prop={'size': 6})

    cort_ax.set(
        xlim=[-1, n_neurons],
        title=f'{cortex} V weights, {n_neurons} neurons',
        ylabel='Contribution Weight',
    )
    cort_ax.legend(title='Monkey')

joint_fig.suptitle('Abs Neuron Loading Value Sum for First 3 PCs')
joint_fig.tight_layout()
monkey_out = os.path.join(PCA_FIG_DIR, 'neuronMonkeyCont.png')
joint_fig.savefig(monkey_out, bbox_inches='tight', dpi=DPI_DRAFT)
plt.close(joint_fig)
print(f'Saved {monkey_out}')

# ---------------------------------------------------------------------------
# Figure 2 — Per-epoch modulation-type neuron contribution
# ---------------------------------------------------------------------------
print('\nPlotting per-epoch mod-type neuron contribution ...')
for epoch in ['Cue', 'Grasp']:
    joint_fig, joint_axes = plt.subplots(
        nrows=2, ncols=3, figsize=(11, 4),
        gridspec_kw={'height_ratios': [1.5, 1]}
    )
    for cort_idx, cortex in enumerate(CORTEX_LIST):
        cort_ax = joint_axes[0][cort_idx]
        cdf_ax  = joint_axes[1][cort_idx]

        # Eigenvalue-weighted contribution weight
        merged_S  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['S']
        merged_V  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
        eigvals   = merged_S[:3] ** 2
        eig_total = (merged_S ** 2).sum()
        V_plot, V_indices = (
            (eigvals * merged_V[:, :3] ** 2).sum(dim=1) / eig_total
        ).sqrt().sort(descending=True, dim=0)

        n_neurons   = merged_monkey.cortices[cortex]['neurons']
        neuron_list = np.arange(n_neurons)

        for mod_type, color in mod_color_map.items():
            # Union of neurons modulated in either orientation
            mod_indices = mod_df[
                (mod_df['region'] == cortex) &
                (mod_df['mod_type'] == mod_type) &
                (mod_df['event'] == epoch)
            ]['n_idc'].unique()

            mod_x = np.array([False] * n_neurons)
            for idx, V_index in enumerate(V_indices):
                if V_index in mod_indices:
                    mod_x[idx] = True
            n_mod = mod_x.sum()
            if n_mod == 0:
                continue   # skip empty mod types (ecdf undefined on empty set)

            mod_ecdf = stats.ecdf(neuron_list[mod_x])
            ks_stat  = stats.ks_1samp(neuron_list[mod_x] / n_neurons,
                                       stats.uniform.cdf)
            p_val     = ks_stat[1]
            sig_label = (r'$\ast\!\ast$' if p_val < 0.01
                         else r'$\,\ast\ $' if p_val < 0.05
                         else ' ')
            mod_ecdf.cdf.plot(cdf_ax, label=sig_label, c=color)
            cort_ax.bar(
                x=neuron_list[mod_x], width=1.0, linewidth=0,
                height=V_plot[mod_x],
                label=f'{mod_type}, n:{n_mod}', color=color,
            )

        cdf_ax.plot([0, n_neurons], [0, 1], 'k--')
        cdf_ax.set_ylabel('Cumulative Fraction')
        cdf_ax.set_xlabel('Neuron Index')
        cdf_ax.set_xlim([0, n_neurons])
        cdf_ax.set_ylim([0, 1])
        leg = cdf_ax.legend(prop={'size': 6}, handlelength=0.6, handleheight=0.8,
                            borderpad=0.3, labelspacing=0.1, loc='lower right')
        for h in leg.legend_handles:
            h.set_linewidth(4.5)

        cort_ax.set_xlim([-1, n_neurons])
        cort_ax.legend(title='Mod Type', prop={'size': 7}, handlelength=0.8,
                       handleheight=0.8, borderpad=0.3, labelspacing=0.2,
                       loc='upper right')
        cort_ax.set_ylabel('Contribution')

    joint_fig.tight_layout()
    epoch_out = os.path.join(PCA_FIG_DIR,
                             f'neuronModTypeCont{epoch}_orientMerged.png')
    joint_fig.savefig(epoch_out, bbox_inches='tight', dpi=DPI_DRAFT)
    plt.close(joint_fig)
    print(f'Saved {epoch_out}')

# ---------------------------------------------------------------------------
# [Commented-out] Correlation / slope analysis
# ---------------------------------------------------------------------------
# corr_fig = plt.figure(layout='constrained', figsize=(12,8))
# subfigs = corr_fig.subfigures(1,3, wspace=0.05)
# slope_fig, slope_axs = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
# for cort_idx, cortex in enumerate(CORTEX_LIST):
#     score_filename = (f'{pca_dir}/neuronSelection_scores_merged'
#                       f'{len(monkey_name_map)}_bin{binsize}_k{kernel_width}_{cortex}.p')
#     with open(score_filename, 'rb') as score_file:
#         score_dict = pkl.load(score_file)
#     merged_V = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
#     V_sum = (merged_V[:, :3] ** 2).sqrt().sum(dim=1)
#     V_plot, V_indices = V_sum.sort(descending=True, dim=0)
#     hemisphere_score  = score_dict['hemi_score']
#     orientation_score = score_dict['orient_score']
#     event_score       = score_dict['event_score']
#     nzero_mask        = score_dict['nzero_mask']
#     event_nzero_mask  = score_dict['event_nzero_mask']
#     subfigs[cort_idx].suptitle(f'{cortex}')
#     cort_axs = subfigs[cort_idx].subplots(2, 1)
#     cort_axs[0].set_xlabel('PC Contribution Weight')
#     cort_axs[1].set_xlabel('PC Contribution Index')
#     cortexSlope_df = mod_df[(mod_df['region'] == cortex)]
#     slope_ax = slope_axs[cort_idx]
#     for direction, minmax, color in zip(['up','down'], ['max','min'],
#                                         ['tab:blue', 'tab:red']):
#         idx        = cortexSlope_df[cortexSlope_df['up-down'] == minmax]
#         slope_idx  = idx['t_idc']
#         weight_val = V_sum[list(idx['n_idc'])]
#         slope_reg  = stats.linregress(slope_idx, weight_val)
#         slope_ax.scatter(slope_idx, weight_val, color=color, alpha=0.4)
#         slope_ax.plot(slope_idx,
#                       slope_reg.intercept + slope_reg.slope * slope_idx,
#                       color=color,
#                       label=f'{direction.capitalize()}:{slope_reg.rvalue**2:.3f}')
#     slope_ax.legend(title='Mod Direction')
#     for cort_ax, x in zip(cort_axs, [V_sum, V_indices]):
#         cort_ax.set_ylabel('Selection Score')
#         cort_ax.set_title('Selection vs. PC Correlation')
#         hemi_reg  = stats.linregress(x[nzero_mask], hemisphere_score)
#         orien_reg = stats.linregress(x[nzero_mask], orientation_score)
#         event_reg = stats.linregress(x[event_nzero_mask], event_score)
#         cort_ax.scatter(x[nzero_mask], hemisphere_score,    alpha=0.5, color='tab:blue')
#         cort_ax.scatter(x[nzero_mask], orientation_score,   alpha=0.5, color='tab:red')
#         cort_ax.scatter(x[event_nzero_mask], event_score,   alpha=0.5, color='tab:green')
#         cort_ax.plot(x, hemi_reg.intercept  + hemi_reg.slope  * x, color='tab:blue',
#                      label=f'Hemi (C-I) R-sq:{hemi_reg.rvalue**2:.3f}')
#         cort_ax.plot(x, orien_reg.intercept + orien_reg.slope * x, color='tab:red',
#                      label=f'Orient (V-H) R-sq:{orien_reg.rvalue**2:.3f}')
#         cort_ax.plot(x, event_reg.intercept + event_reg.slope * x, color='tab:green',
#                      label=f'Event (G-C) R-sq:{event_reg.rvalue**2:.3f}')
#         cort_ax.legend()
# corr_fig.savefig(f'{PCA_FIG_DIR}/SelectionPC.png',   bbox_inches='tight', dpi=600)
# slope_fig.savefig(f'{PCA_FIG_DIR}/SlopeTimingPC.png', bbox_inches='tight', dpi=600)

print('\npca_plot_neurons.py complete.')