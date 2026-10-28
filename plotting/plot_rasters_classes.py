"""
plot_rasters_classes.py — Raster plots for top PC-loading neurons with cluster labels.

Reads modulation data, cluster assignments, and spike lists, then plots raster/SDF
figures for the neurons with the highest PC loadings in each region.

Fix applied: line 94 bug — replaced `np.range()` (non-existent) with the correct
    `pc_indices[region][:plot_neurons]`.

Figures saved under Results/PCA/figures/:
    Raster_PCs_{region}.png
    Event_Baseline_Correlation.png

Data loaded from Results/Population/data/, Results/Modulation/data/,
Results/Selectivity/data/, and Results/PCA/data/PCA_b{b}_k{k}/.
"""

import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.plot_utils import raster_plot, apply_theme

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()

from config import (
    POPULATION_DIR, MODULATION_DIR, SELECTIVITY_DIR, PCA_RESULTS_DIR,
    EPOCH_WINDOW_MAP_PLOT, TOTAL_TIME_PLOT, MONKEY_NAME_MAP, BIN_SIZE, KERNEL_WIDTH,
)
from plot_config import (
    DPI_DRAFT,
    DPI_FINAL,
    REGION_TITLE_SIZE,
    apply_rcparams,
)
apply_rcparams()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION_LIST = ['M1', 'PMd', 'PMv']

MERGED_COUNT    = len(MONKEY_NAME_MAP)
PLOT_NEURONS  = 10     # top N neurons per region to plot
PLOT_COLUMNS  = 5
USE_CONDITIONS = True  # if True, colour-sort rasters by condition
# Define epochs and cumulative time axis
EPOCH_WINDOW = [0, TOTAL_TIME_PLOT]



POPULATION_DATA  = os.path.join(POPULATION_DIR,  'data')
MODULATION_DATA  = os.path.join(MODULATION_DIR,  'data')
SELECTIVITY_DATA = os.path.join(SELECTIVITY_DIR, 'data')
PCA_DATA_DIR     = os.path.join(PCA_RESULTS_DIR, 'data', f'PCA_b{BIN_SIZE}_k{KERNEL_WIDTH}')
PCA_FIG_DIR      = os.path.join(PCA_RESULTS_DIR, 'figures')
os.makedirs(PCA_FIG_DIR, exist_ok=True)

# Input file paths
AREA_DICT_FILE  = f'{POPULATION_DATA}/spike_summary_merged{MERGED_COUNT}.p'
MOD_DF_FILE     = f'{MODULATION_DATA}/MergedModDataFrame.p'
# CLUSTER_FILE    = f'{POPULATION_DATA}/Clusters.p'
SPIKES_FILE     = f'{POPULATION_DATA}/spike_list_merged{MERGED_COUNT}.p'
INDICES_FILE    = f'{PCA_DATA_DIR}/pcIndicesJoint_merged{MERGED_COUNT}.p'
EVENT_FILE      = f'{POPULATION_DATA}/event_list_merged{MERGED_COUNT}.p'
CONDITION_FILE  = f'{POPULATION_DATA}/condition_list_merged{MERGED_COUNT}.p'
MERGED_MOD_FILE = f'{MODULATION_DATA}/MergedMods.p'
ANOVA_FILE      = f'{SELECTIVITY_DATA}/anova_results_merged{MERGED_COUNT}.p'


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading data …')
with open(AREA_DICT_FILE,  'rb') as f:
    area_summary_dict = pkl.load(f)
with open(MOD_DF_FILE,     'rb') as f:
    merged_mod_df = pkl.load(f)
with open(ANOVA_FILE, 'rb') as f:
    anova_data = pkl.load(f)
with open(SPIKES_FILE,     'rb') as f:
    spike_dict = pkl.load(f)
with open(INDICES_FILE,    'rb') as f:
    pc_indices = pkl.load(f)
with open(EVENT_FILE,      'rb') as f:
    all_events = pkl.load(f)
with open(CONDITION_FILE,  'rb') as f:
    all_conditions = pkl.load(f)
with open(MERGED_MOD_FILE, 'rb') as f:
    merged_mods = pkl.load(f)
with open(SPIKES_FILE,     'rb') as f:
    all_spikes = pkl.load(f)
print('Data loaded.')

# ---------------------------------------------------------------------------
# Scatter figure: event mean vs baseline mean
# ---------------------------------------------------------------------------
scatter_fig = plt.figure()
sc_ax = scatter_fig.gca()
sc_ax.set_xlabel('Base SDF Mean')
sc_ax.set_ylabel('Grasp SDF Mean')

condition_string = '_conds' if USE_CONDITIONS else ''
anova_significance = anova_data['significance']
anova_pvals = anova_data['pvalues']
anova_fr_masks = anova_data['fr_mask']

# Before the loop
combined_fig = plt.figure(figsize=(11, 3 * len(REGION_LIST)))
subfigs = combined_fig.subfigures(len(REGION_LIST), 1, hspace=0.2)
trialStart_time = EPOCH_WINDOW_MAP_PLOT['Cue']['time']

# Compute once before the loop, after trialStart_time is defined
cue_time = EPOCH_WINDOW_MAP_PLOT['Cue']['time']
grasp_time = EPOCH_WINDOW_MAP_PLOT['Grasp']['time']
total_time = TOTAL_TIME_PLOT

mod_abbrev = {
    'both:contra': 'B:C', 'both:ipsi': 'B:I',
    'both:opp': 'B:O', 'both:equal': 'B:E',
    'contra': 'C', 'ipsi': 'I', 'none': 'N'
}
dir_abbrev = {'up': '↑', 'down': '↓'}

# Explicit ticks at meaningful points: start, cue, grasp, end
tick_positions = [0, cue_time, grasp_time, total_time]
tick_labels = [int(t - cue_time) for t in tick_positions]
for letter, (subfig, region) in zip('ABC', zip(subfigs, REGION_LIST)):
    subfig.text(
        0.0, 1.0, rf'$\mathbf{{{letter}}}$-{region}',
        transform=subfig.transSubfigure,
        fontsize=REGION_TITLE_SIZE, fontweight='bold',
        ha='left', va='bottom', clip_on=False,
    )

    fr_mask = anova_fr_masks[region]
    all_event_means    = merged_mod_df[merged_mod_df['region'] == region]['event_mean']
    all_baseline_means = merged_mod_df[merged_mod_df['region'] == region]['baseline_mean']
    sc_ax.scatter(all_baseline_means, all_event_means, label=region, alpha=0.3)

    idx_list = pc_indices[region][:PLOT_NEURONS]

    mod_list       = []
    condition_list = []

    for idx in idx_list:
        mod_vals = merged_mod_df[
            (merged_mod_df['n_idc'] == int(idx)) &
            (merged_mod_df['region'] == region) &
            (merged_mod_df['event'] == 'Grasp')
            ][['mod_type', 'tVal', 'side', 'event_mean', 'baseline_mean']]

        t_vals = list(mod_vals['tVal'])
        mod_type = list(mod_vals['mod_type'])[-1] if mod_vals['mod_type'].any() else 'none'
        if mod_type in ('both:ipsi', 'ipsi'):
            t_ref = t_vals[0]
        else:
            t_ref = t_vals[1] if len(t_vals) > 1 else t_vals[0]
        up_down = 'up' if t_ref > 0 else 'down'
        mod_list.append(f'{mod_type} {up_down}')
        condition_list.append(all_conditions[region][idx])

    conditions = condition_list if USE_CONDITIONS else None
    region_spikes = [
        np.hstack(
            [np.vstack([
                all_spikes[region][neuron][i],
                np.ones_like(all_spikes[region][neuron][i]) * (i + 1)])
             for i in range(len(all_spikes[region][neuron]))]).T
        for neuron in range(len(all_spikes[region]))]

    _, neuron_axes = raster_plot(
        idx_list, region_spikes, all_events[region],
        EPOCH_WINDOW_MAP_PLOT,
        n_cols=PLOT_COLUMNS,
        sdf=True,
        window=EPOCH_WINDOW,
        conditions=conditions,
        # fig_size=(11, 2.5),
        fig=subfig
    )

    for i, (idx, mod_val, ax) in enumerate(zip(idx_list, mod_list, neuron_axes.flatten())):
        if not fr_mask[idx]:
            sel_str = '[low FR]'
            print(f'Low FR for neuron {idx + 1}')
        else:
            anova_flags = anova_significance[region]['Grasp'][idx]
            selectivity_parts = []
            if anova_flags[0]: selectivity_parts.append('H')
            if anova_flags[1]: selectivity_parts.append('O')
            if anova_flags[2]: selectivity_parts.append('HxO')
            sel_str = f'[{", ".join(selectivity_parts) if selectivity_parts else "none"}]'
        parts = mod_val.split(' ')
        mod_short = mod_abbrev.get(parts[0], parts[0]) + dir_abbrev.get(parts[1], parts[1])
        ax.set_title(f'{mod_short} | {sel_str}', fontsize=10)
        # ax.set_title(f'{mod_val}\n{sel_str}', fontsize=8)
        ax.set_xticks(tick_positions)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if i < PLOT_COLUMNS:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(tick_labels)
    subfig.supxlabel('Time (ms)', fontsize=12, y=0.01, va='top')
    subfig.supylabel('Trial', fontsize=12, x=0.00, ha='left')

combined_fig.subplots_adjust(left=0.04)
combined_fig.savefig(f'{PCA_FIG_DIR}/Raster_PCs_combined{condition_string}.png',
                     bbox_inches='tight', dpi=DPI_FINAL)
plt.close(combined_fig)


sc_ax.legend()
scatter_out = f'{PCA_FIG_DIR}/Event_Baseline_Correlation.png'
scatter_fig.savefig(scatter_out, bbox_inches='tight', dpi=DPI_DRAFT)
plt.close(scatter_fig)
print(f'Saved {scatter_out}')

print('\nplot_rasters_classes.py complete.')