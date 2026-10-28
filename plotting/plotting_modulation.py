"""
plotting_modulation.py — Hand-modulation figure.

Produces:
  • Figure 3: Modulation bar charts + both-modulated pie charts.

Figures saved under Results/Modulation/figures/:
    HandModulation_merged{n}.png

Data loaded from Results/Modulation/data/.
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
from matplotlib.patches import Patch
import seaborn as sns

from config import (
    MODULATION_DIR,
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    MOD_COLOR_MAP          as mod_color_map,
)
from utils.plot_utils import scale_lightness, apply_theme
from plot_config import (
    FIGURE_WIDTH,
    DPI_FINAL,
    PANEL_LABEL_SIZE, REGION_TITLE_SIZE,
    apply_rcparams,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()
apply_rcparams()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODULATION_DATA = os.path.join(MODULATION_DIR, 'data')
MODULATION_FIGS = os.path.join(MODULATION_DIR, 'figures')
os.makedirs(MODULATION_FIGS, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n_monkeys = len(monkey_name_map)
rgb       = mpl.colors.ColorConverter.to_rgb

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
merged_df_filename    = os.path.join(MODULATION_DATA, 'MergedModDataFrame.p')
hand_use_mod_filename = os.path.join(MODULATION_DATA, 'HandUseMod.p')

merged_mod_df = pd.read_pickle(merged_df_filename)
with open(hand_use_mod_filename, 'rb') as f:
    hand_use_mod = pkl.load(f)

print('Data loaded.')

# ---------------------------------------------------------------------------
# Figure 3: Modulation bar chart + both-modulated pie charts
# ---------------------------------------------------------------------------
layout = [
    ['main_M1', 'main_M1', 'main_PMd', 'main_PMd', 'main_PMv', 'main_PMv'],
    ['M1_pie_Cue', 'M1_pie_Grasp', 'PMd_pie_Cue', 'PMd_pie_Grasp', 'PMv_pie_Cue', 'PMv_pie_Grasp'],
]
f, axs = plt.subplot_mosaic(layout, figsize=(FIGURE_WIDTH, 4), sharey=False,
                            gridspec_kw={'height_ratios': [1, 1.8]})

width_main   = 0.2
width_shadow = 0.1
step         = width_main + width_shadow
x            = np.arange(len(list(epoch_window_map.keys())[1:]))  # Cue, Grasp
ud_bar_types = ['both:all', 'ipsi', 'contra']
mod_type_map = {'ipsi': 'Ipsi', 'contra': 'Contra', 'both:all': 'Both'}

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

            bar_kw = dict(width=width_main, color=mod_color_map[mod_type], edgecolor='none')
            if e_i == 0:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, label=mod_type_map[mod_type], **bar_kw)
            else:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, **bar_kw)
            bar_base += hand_mod_perc * 100

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
    ax.set_title(f'{region}', fontsize=REGION_TITLE_SIZE, fontweight='bold', pad=15)
    # ax.set_xticks(
    #     x + step * multiplier / 2,
    #     [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]],
    # )
    ax.set_xticks([])
    if region != 'M1':
        ax.set_yticks([])

epoch_names   = [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]]
n_bar_types   = len(ud_bar_types)
cluster_width = step * (n_bar_types - 1) + width_main
for region in hand_use_mod.keys():
    ax = axs[f'main_{region}']
    for e_i, epoch_name in enumerate(epoch_names):
        ax.text(
            x[e_i] + cluster_width / 2,
            max_height * 0.98,
            epoch_name,
            ha='center', va='bottom',
            fontsize=REGION_TITLE_SIZE,
            color='gray',
            transform=ax.transData,
            clip_on=False,
        )

handles, labels = ax.get_legend_handles_labels()
axs['main_PMd'].legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.5, 1.0), ncols=3)

# Pie charts
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
        # pie_ax.set_title(f'{epoch}', va='center')

axs['main_M1'].set_ylabel('Modulated neurons (%)')

both_legend_labels = ['ipsi-dominant', 'contra-dominant', 'equal', 'opposite']
both_legend_colors = [mod_color_map[m] for m in both_mod_list]
legend_handles = [Patch(facecolor=c, label=l)
                  for c, l in zip(both_legend_colors, both_legend_labels)]
f.legend(handles=legend_handles, loc='lower center',
         title='Both-hand modulation subtypes', title_fontsize=9,
         ncols=4, frameon=False, fontsize=8,
         bbox_to_anchor=(0.5, 0.01))

plt.tight_layout()
sns.despine()

# Panel labels A and B
_pos_A   = axs['main_M1'].get_position()
_pos_B   = axs['M1_pie_Cue'].get_position()
_x_label = _pos_A.x0 - 0.02
f.text(_x_label, _pos_A.y1 + 0.04,
       r'$\mathbf{A}$',
       fontsize=PANEL_LABEL_SIZE, fontweight='bold',
       ha='left', va='bottom', clip_on=False)
f.text(_x_label, _pos_B.y1 - 0.05,
       r'$\mathbf{B}$',
       fontsize=PANEL_LABEL_SIZE, fontweight='bold',
       ha='left', va='bottom', clip_on=False)

out3 = os.path.join(MODULATION_FIGS, f'HandModulation_merged{n_monkeys}.png')
plt.savefig(out3, bbox_inches='tight', dpi=DPI_FINAL)
plt.close(f)
print(f'Saved {out3}')

print('\nplotting_modulation.py complete.')
