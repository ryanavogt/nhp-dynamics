"""
pca_plot_angles.py — Principal angle difference bar charts.

Loads AngleMatrix files produced by pca_compute.py and plots the pairwise
difference in leading principal angles across all condition pairs,
with error bars propagated from the across-subset SD.

Figures saved under Data/Processed/Summary/PCA_b{b}_k{k}/:
    PrincipalAngleDiff_monkey{m}_{diff_mode}.png
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MONKEY_NAME_MAP as monkey_name_map,
    SUMMARY_DIR     as summary_dir,
    BIN_SIZE        as binsize,
    KERNEL_WIDTH    as kernel_width,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sns.set_theme(); sns.set_style(style='white')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_DIMS    = 5           # Must match pca_compute.py
DIFF_MODE = 'absolute'  # 'absolute' or 'relative'

pca_dir          = os.path.join(summary_dir, f'PCA_b{binsize}_k{kernel_width}')
monkey_dict_file = f'{pca_dir}/allMonkeyDict.p'

with open(monkey_dict_file, 'rb') as f:
    monkeys = pkl.load(f)

cortex_list = ['M1', 'PMd', 'PMv']

# ---------------------------------------------------------------------------
# Bar charts: pairwise angle differences with error bars
# ---------------------------------------------------------------------------
for m_name in monkeys.keys():
    comb_fig, comb_axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    for cort_idx, cortex in enumerate(cortex_list):
        cort_ax    = comb_axs[cort_idx]
        angle_file = os.path.join(
            pca_dir, f'AngleMatrix_{cortex}_monkey{m_name}_PCdims{N_DIMS}.p'
        )
        if not os.path.exists(angle_file):
            print(f'  Missing {angle_file}, skipping.')
            continue

        with open(angle_file, 'rb') as f:
            angle_data = pkl.load(f)

        mean_matrix = angle_data['mean']
        std_matrix  = angle_data['std']
        cond_matrix = angle_data['cond_matrix']
        cond_list   = angle_data['cond_list']

        # Lower-triangle pairs only
        tril_mask  = np.tril(np.ones_like(mean_matrix, dtype=bool), k=-1)
        mean_vals  = mean_matrix[tril_mask]
        std_vals   = std_matrix[tril_mask]
        cond_combs = cond_matrix[tril_mask]

        # Build diff relative to each reference pair
        comb_dict = {}
        for c_i, start_comb in enumerate(cond_combs):
            label      = f'{start_comb[:3]}:{start_comb.split(chr(58))[1][:3]}'
            diffs, errs = [], []
            for c_j in range(len(cond_combs)):
                mean_diff = mean_vals[c_j] - mean_vals[c_i]
                if DIFF_MODE == 'absolute':
                    diffs.append(mean_diff)
                else:
                    diffs.append(mean_diff / mean_vals[c_i])
                # Propagate SD in quadrature (independent subsets)
                combined_std = np.sqrt(std_vals[c_j] ** 2 + std_vals[c_i] ** 2)
                if DIFF_MODE == 'relative':
                    combined_std /= mean_vals[c_i]
                errs.append(combined_std)
            comb_dict[label] = {'diffs': diffs, 'errs': errs}

        width      = 0.14
        x          = np.arange(len(cond_combs))
        multiplier = 0
        for comb, data in comb_dict.items():
            offset = width * multiplier
            rects  = cort_ax.bar(
                x + offset, data['diffs'], width,
                yerr=data['errs'], capsize=3, label=comb
            )
            cort_ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=6)
            multiplier += 1
            cort_ax.axvline(x=multiplier - width, color='grey')

        cort_ax.axhline(y=0, color='black')
        cort_ax.set_ylabel(f'{DIFF_MODE.capitalize()} Difference')
        cort_ax.set_title(cortex)
        cort_ax.set_xticks(x + len(cond_combs) / 2 * width, comb_dict.keys())
        cort_ax.legend(ncols=3)
        cort_ax.set_ylim([-.38, .38])
        cort_ax.set_xlim([-width, len(cond_combs) - width])

    comb_fig.tight_layout()
    out = f'{pca_dir}/PrincipalAngleDiff_monkey{m_name}_{DIFF_MODE}.png'
    comb_fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(comb_fig)
    print(f'Saved {out}')

print('\npca_plot_angles.py complete.')
