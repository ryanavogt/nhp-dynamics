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
from pandas.plotting import table

from config import (
    SUMMARY_DIR,
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    MOD_COLOR_MAP          as mod_color_map,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
    WINDOW_RANGE,
)
from utils.plot_utils import scale_lightness, scatter_hist, apply_theme

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sns.set()
sns.set_style('ticks')
mpl.rcParams['axes.linewidth'] = 0.4
mpl.rcParams['font.family'] = 'Arial'

# ---------------------------------------------------------------------------
# Style constants (match PCA figure)
# ---------------------------------------------------------------------------
PANEL_LABEL_SIZE   = 14    # capital-letter panel labels (A, B, C …)
REGION_TITLE_SIZE  = 16    # M1 / PMd / PMv column headers (larger than panel labels)

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
    """Bold panel label anchored to the top-left of a SubFigure.

    Uses transSubfigure so the label sits just outside the subfigure's
    own top-left corner, clear of any suptitle or axes content.
    """
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
summary_dir  = SUMMARY_DIR
n_monkeys    = len(monkey_name_map)
regions      = ['M1', 'PMd', 'PMv']
SCATTER_CMAP = 'rocket'
FR_THRESHOLD_HZ = 1.0

# ---------------------------------------------------------------------------
# Load saved analysis outputs
# ---------------------------------------------------------------------------
merged_df_filename    = f'{summary_dir}/MergedModDataFrame.p'
hand_use_mod_filename = f'{summary_dir}/HandUseMod.p'
neuron_csv_filename   = os.path.join(summary_dir, 'monkeyNeurons.csv')
anova_file_name       = os.path.join(summary_dir, f'anova_results_merged{n_monkeys}.p')

merged_mod_df = pd.read_pickle(merged_df_filename)
with open(hand_use_mod_filename, 'rb') as f:
    hand_use_mod = pkl.load(f)

neuron_df = pd.read_csv(neuron_csv_filename, index_col=0)
n_neurons = neuron_df.sum(axis=1).to_dict()

score_data = {}
for region in regions:
    score_filename = os.path.join(
        summary_dir,
        f'neuronSelection_scores_merged{n_monkeys}_bin{binsize}_k{kernel_width}_{region}.p')
    with open(score_filename, 'rb') as f:
        score_data[region] = pkl.load(f)

with open(anova_file_name, 'rb') as f:
    anova_data = pkl.load(f)

print('All analysis outputs loaded')

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
plt.savefig(os.path.join(summary_dir, 'monkeyNeurons.png'), dpi=300, bbox_inches='tight')
# plt.show()
print('Neuron count table saved')

# ---------------------------------------------------------------------------
# Figure 2: Joint scatter / AUC / ANOVA figure
# ---------------------------------------------------------------------------

# Global colour scale for the shared colorbar (computed once, before the loop)
anova_significance = anova_data['significance']
all_tr      = np.concatenate([score_data[r]['total_rates'] for r in regions])
global_norm = mpl.colors.LogNorm(
    vmin=float(np.nanmin(all_tr[all_tr > 0])),
    vmax=float(np.nanmax(all_tr)),
)

joint_fig   = plt.figure(figsize=(11, 8.5), layout='constrained')
# 4 outer columns: M1 | PMd | PMv | narrow colorbar strip
outer_cols  = joint_fig.subfigures(1, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.03)
cbar_subfig = outer_cols[3]
anova_axes  = {}
# Keep per-region column subfigures so we can label rows B and C after the loop
col_subfigs = {}

for reg_idx, region in enumerate(regions):
    d = score_data[region]
    hemisphere_score  = d['hemi_score']
    orientation_score = d['orient_score']
    tr                = d['total_rates']   # already in Hz

    # Each data column: scatter (top) + AUC (middle) + ANOVA (bottom)
    column = outer_cols[reg_idx].subfigures(3, 1, height_ratios=[1.5, 1, 1])
    col_subfigs[region] = column   # save for row-B / row-C labels below

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

    # Median arrow markers on marginal histograms
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
        # Panel label A — anchored to the scatter subfigure top-left,
        # clear of the region suptitle (which is centred).
        _subfig_panel_label(column[0], 'A')
    ax.set_xlabel('Hand Score (C - I)')
    ax.set_xlim([-1.03, 1.03])
    ax.set_ylim([-1.03, 1.03])
    # Region title — placed above the entire outer column so it sits above
    # the panel label (which anchors at y=1.0 of column[0] transSubfigure).
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
        # Panel label B — anchored to the AUC subfigure top-left
        _subfig_panel_label(column[1], 'B')

    # Store ANOVA axes reference for the loop below
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
EPOCH_LIST    = list(epochs)   # ['Cue', 'Grasp']
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
        # Panel label C — anchored to the ANOVA subfigure top-left
        _subfig_panel_label(col_subfigs['M1'][2], 'C')
    else:
        ax_an.set_yticklabels([])
    ax_an.spines[['top', 'right']].set_visible(False)

joint_fig.savefig(
    os.path.join(summary_dir,
                 f'neuronSelection_merged{n_monkeys}_bin{binsize}_k{kernel_width}_joint.png'),
    bbox_inches='tight', dpi=200)
# plt.show()
print('Joint scatter/AUC/ANOVA figure saved')

# ---------------------------------------------------------------------------
# Figure 3: Modulation bar chart + both-modulated pie charts
# ---------------------------------------------------------------------------
layout = [
    ['main_M1', 'main_M1', 'main_PMd', 'main_PMd', 'main_PMv', 'main_PMv'],
    ['M1_pie_Cue', 'M1_pie_Grasp', 'PMd_pie_Cue', 'PMd_pie_Grasp', 'PMv_pie_Cue', 'PMv_pie_Grasp'],
]
f, axs = plt.subplot_mosaic(layout, figsize=(11, 4), sharey=False,
                            gridspec_kw={'height_ratios': [1, 1.8]})

width_main   = 0.2
width_shadow = 0.1
step         = width_main + width_shadow
rgb          = mpl.colors.ColorConverter.to_rgb
x            = np.arange(len(list(epoch_window_map.keys())[1:]))  # Cue, Grasp
ud_bar_types = ['both:all', 'ipsi', 'contra']
mod_type_map = {'ipsi':'Ipsi', 'contra':'Contra', 'both:all':'Both'}

for region in hand_use_mod.keys():
    ax         = axs[f'main_{region}']
    bar_base   = 0
    max_height = 100
    region_df  = merged_mod_df[merged_mod_df['region'] == region]

    for e_i, (event, mod_dictionary) in enumerate(hand_use_mod[region].items()):
        multiplier    = 0
        event_df      = region_df[region_df['event'] == event]
        both_subtypes = ['both:equal', 'both:opp', 'both:ipsi', 'both:contra']

        for mod_type, modulation in mod_dictionary.items():
            if mod_type not in ud_bar_types:
                continue
            hand_mod_perc = np.average(modulation)
            n_neurons_bar = len(modulation)
            offset        = step * multiplier

            # Primary bar: % of neurons modulated
            bar_kw = dict(width=width_main, color=mod_color_map[mod_type], edgecolor='none')
            if e_i == 0:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, label=mod_type_map[mod_type], **bar_kw)
            else:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, **bar_kw)
            bar_base += hand_mod_perc * 100

            # Shadow bar: up/down breakdown
            if mod_type == 'ipsi':
                tval_series = event_df[
                    (event_df['side'] == 'ipsilateral') &
                    (event_df['mod_type'] == 'ipsi')
                ]['tVal']
            elif mod_type == 'contra':
                tval_series = event_df[
                    (event_df['side'] == 'contralateral') &
                    (event_df['mod_type'] == 'contra')
                ]['tVal']
            else:  # both:all
                tval_series = event_df[
                    (event_df['side'] == 'contralateral') &
                    (event_df['mod_type'].isin(both_subtypes))
                ]['tVal']

            up_perc     = (tval_series > 0).sum() / n_neurons_bar * 100
            down_perc   = (tval_series < 0).sum() / n_neurons_bar * 100
            color_full  = mod_color_map[mod_type]
            color_light = scale_lightness(rgb(color_full), 1.7)
            color_dark  = scale_lightness(rgb(color_full), 1.2)
            shadow_x    = x[e_i] + offset + width_main / 2 + width_shadow / 2
            ax.bar(shadow_x, down_perc, width_shadow, color=color_light, edgecolor='none')
            ax.bar(shadow_x, up_perc,   width_shadow, bottom=down_perc,
                   color=color_dark, edgecolor='none')

            multiplier += 1

    ax.set_ylim([0, max_height])
    # ax.set_xlabel('Epochs')
    # Region title — styled to match PCA figure
    ax.set_title(f'{region}', fontsize=REGION_TITLE_SIZE, fontweight='bold')
    ax.set_xticks(
        x + step * multiplier / 2,
        [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]],
    )
    if region != 'M1':
        ax.set_yticks([])

handles, labels = ax.get_legend_handles_labels()
axs['main_PMd'].legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0), ncols=3)

# Pie charts: both-modulated subtypes, split by up/down direction
both_mod_list = ['both:ipsi', 'both:contra', 'both:equal', 'both:opp']
size          = 0.4

for region in hand_use_mod.keys():
    region_df       = merged_mod_df[merged_mod_df['region'] == region]
    both_mod_counts = {}

    for event, mod_dictionary in hand_use_mod[region].items():
        both_mod_counts[event] = []
        mod_order = []
        for mod_type, mod_bools in mod_dictionary.items():
            if mod_type not in both_mod_list:
                continue
            mod_order.append(mod_type)
            subset = region_df[
                (region_df['event']    == event) &
                (region_df['side']     == 'contralateral') &
                (region_df['mod_type'] == mod_type)
            ]['tVal']
            both_mod_counts[event].append([(subset < 0).sum(), (subset > 0).sum()])

    both_mod_counts['Cue']   = np.stack(both_mod_counts['Cue'])
    both_mod_counts['Grasp'] = np.stack(both_mod_counts['Grasp'])

    for epoch in ['Cue', 'Grasp']:
        pie_ax       = axs[f'{region}_pie_{epoch}']
        labels_pie   = [mod.split(':')[1] for mod in mod_order]
        counts       = both_mod_counts[epoch]
        outer_colors = [mod_color_map[mod] for mod in mod_order]
        inner_colors = []
        for color in outer_colors:
            inner_colors += [scale_lightness(rgb(color), 1.7),
                             scale_lightness(rgb(color), 1.2)]

        pie_ax.pie(
            counts.sum(axis=1), radius=1, colors=outer_colors, startangle=90,
            wedgeprops=dict(width=size, edgecolor='w'),
            labels=None, autopct='%1.f', labeldistance=1.1, pctdistance=0.8,
        )
        pie_ax.pie(
            counts.flatten(), radius=1 - size, colors=inner_colors, startangle=90,
            wedgeprops=dict(width=size, edgecolor='w'),
        )
        pie_ax.set_title(f'{epoch}', va='center')
        # if region == 'M1' and epoch == 'Cue':
        #     pie_ax.set_ylabel('Both Mod Breakdown (%)')

axs['main_M1'].set_ylabel('Modulated neurons (%)')

from matplotlib.patches import Patch
both_legend_labels = ['ipsi-dominant', 'contra-dominant', 'equal', 'opposite']
both_legend_colors = [mod_color_map[m] for m in both_mod_list]  # both_mod_list order: ipsi, contra, equal, opp
legend_handles = [Patch(facecolor=c, label=l)
                  for c, l in zip(both_legend_colors, both_legend_labels)]
f.legend(handles=legend_handles, loc='lower center',
        title='Both-hand modulation subtypes', title_fontsize=9,
         ncols=4, frameon=False, fontsize=8,
         bbox_to_anchor=(0.5, 0.01))

# suptitle removed — title is given in the figure caption
plt.tight_layout()
sns.despine()

# Panel labels A and B — placed after tight_layout() so get_position() is reliable.
# Both share the same figure-level x, computed from main_M1's left edge minus a
# fixed offset that clears the rotated y-axis label.
_pos_A   = axs['main_M1'].get_position()
_pos_B   = axs['M1_pie_Cue'].get_position()
# x shared by both labels — enough left of x0 to clear the rotated y-axis label
_x_label = _pos_A.x0 - 0.02
# A: just above the bar-chart row
f.text(_x_label, _pos_A.y1 + 0.04,
       r'$\mathbf{A}$',
       fontsize=PANEL_LABEL_SIZE, fontweight='bold',
       ha='left', va='bottom', clip_on=False)
# B: just above the pie-chart row (below the bar row's x-axis label)
f.text(_x_label, _pos_B.y1 + 0.01,
       r'$\mathbf{B}$',
       fontsize=PANEL_LABEL_SIZE, fontweight='bold',
       ha='left', va='bottom', clip_on=False)
plt.savefig(
    f'{summary_dir}/HandModulation_merged{n_monkeys}.png',
    bbox_inches='tight', dpi=400,
)
# plt.show()
print('Modulation figure saved')

# ---------------------------------------------------------------------------
# Figure 4: Epoch-of-max proportions (Panel A) + Grasp heatmaps (Panel B)
# ---------------------------------------------------------------------------

# ── Load Figure 4 pipeline outputs ──────────────────────────────────────────
neuron_peaks_filename   = f'{summary_dir}/neuron_peaks.p'
all_orient_sdf_filename = (f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')
max_rate_filename       = (f'{summary_dir}/sdfMaxRate_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')
peak_order_filename     = (f'{summary_dir}/peakOrderDict_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')

with open(neuron_peaks_filename, 'rb') as f:
    event_neuron_max = pkl.load(f)
with open(all_orient_sdf_filename, 'rb') as f:
    all_orient_sdf_dict = pkl.load(f)
with open(max_rate_filename, 'rb') as f:
    area_max_rate = pkl.load(f)
with open(peak_order_filename, 'rb') as f:
    peak_order_dict = pkl.load(f)

heatmap_corr_filename = (f'{summary_dir}/heatmapCorr_bin{binsize}_k{kernel_width}'
                         f'_orientMerged.p')
with open(heatmap_corr_filename, 'rb') as f:
    heatmap_corr = pkl.load(f)

print('Figure 4 data loaded')

# ── Figure 4 parameters ─────────────────────────────────────────────────────
grasp_event    = epoch_window_map['Grasp']['event']             # raw event key, e.g. 'trialGraspOn'
active_epochs4 = [ep for ep in epochs if ep != 'Pre-Cue']      # ['Cue', 'Grasp']
x_epochs4      = np.arange(len(active_epochs4))
heatmap_window = np.array(WINDOW_RANGE)                         # full-trial window in ms

# Bar-chart side colours (contra = salmon, ipsi = purple; matching setup figure hand icons) and labels
side_keys4    = ['ipsilateral', 'contralateral']
side_labels4  = ['Ipsilateral', 'Contralateral']
side_colors4  = ['#8B7DC8', '#E8927C']   # ipsi=purple, contra=salmon (matches setup figure)
bar_width4    = 0.35

# Shared y-limit for all three bar charts (computed once across all regions/sides)
bar_max4 = max(
    vals.max() * 100
    for region in regions
    for vals in event_neuron_max[region]['proportions'].values()
)
bar_ylim4 = bar_max4 * 1.3

# ── Figure layout: 3 outer column subfigures ─────────────────────────────────
fig4      = plt.figure(figsize=(11, 6), layout='constrained')
fig4_cols = fig4.subfigures(1, 3, wspace=0.06)

for reg_idx, region in enumerate(regions):
    col = fig4_cols[reg_idx]

    # Each outer column: bar chart sub (top) + heatmap sub (bottom)
    sf_bar4, sf_heat4 = col.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)

    # ── Panel A: epoch-of-max-discharge bar chart ────────────────────────────
    ax_bar4     = sf_bar4.subplots(1)
    n_neurons_r = event_neuron_max[region]['neurons']

    for s_idx, (side_key, side_label, color) in enumerate(
            zip(side_keys4, side_labels4, side_colors4)):
        prop = event_neuron_max[region]['proportions'][side_key]   # shape (n_active_epochs,)
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

    # Region title — above the entire outer column (same style as Figure 2)
    col.text(0.5, 1.01, region,
             transform=col.transSubfigure,
             fontsize=REGION_TITLE_SIZE, fontweight='bold',
             ha='center', va='bottom', clip_on=False)

    # ── Panel B: Grasp heatmaps (Contralateral | Ipsilateral) ────────────────
    ax_contra4, ax_ipsi4 = sf_heat4.subplots(1, 2, sharey=True)
    lat_heat_axes4 = {'c': ax_contra4, 'i': ax_ipsi4}

    # Normalise both sides by the contralateral area's max rate per neuron
    max_rate4   = area_max_rate[f'c{region}']           # (n_neurons,)
    # Sort neurons by contralateral Grasp peak time (applied to both sides)
    peak_order4 = peak_order_dict[f'c{region}'][grasp_event]
    n_neurons_h = len(peak_order4)

    # pcolor coordinate grid: y = neuron edges, x = time edges (same convention as
    # area_peak_plots_mergedOrients.py: y, x = np.mgrid[neurons, time])
    y_grid4, x_grid4 = np.mgrid[
        1:n_neurons_h + 2:1,
        heatmap_window[0] / 1000:heatmap_window[1] / 1000 + 2 * binsize / 1000:binsize / 1000
    ]

    for lat in ['c', 'i']:
        ax      = lat_heat_axes4[lat]
        sdf     = all_orient_sdf_dict[f'{lat}{region}'][grasp_event]   # (n_bins, n_neurons)
        a       = sdf.T                                                  # (n_neurons, n_bins)
        scaled  = np.divide(a, max_rate4[:, np.newaxis],
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

    # Invert once after both pcolor calls — sharey=True means calling it inside
    # the loop inverts twice (once per axis), which cancels out.
    ax_contra4.invert_yaxis()

    # Peak-timing Pearson r annotation: contra vs ipsi sequence similarity.
    # Placed in the contralateral panel (left), bottom-left corner, white text
    # so it reads against the dark inferno background.
    r_val = heatmap_corr[region][grasp_event]['r_peak_timing']
    p_val = heatmap_corr[region][grasp_event]['p_peak_timing']
    p_str = f'p = {p_val:.2e}' if p_val >= 1e-99 else 'p < 10⁻⁹⁹'
    ax_contra4.text(
        0.04, 0.04, f'r = {r_val:.2f}',
        transform=ax_contra4.transAxes,
        color='white', fontsize=8.5, ha='left', va='bottom',
        fontweight='bold',
    )

    if reg_idx == 0:
        _subfig_panel_label(sf_heat4, 'B')

fig4.savefig(
    os.path.join(summary_dir,
                 f'EpochMaxFiring_merged{n_monkeys}.png'),
    bbox_inches='tight', dpi=200)
# plt.show()
print('Figure 4 (epoch-max proportions + Grasp heatmaps) saved')
