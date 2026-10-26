"""
sdf_visualization.py
--------------------
Plots mean SDFs per condition for a given monkey and cortical area,
before they enter PCA. Useful for verifying smoothing bandwidth and
checking that conditions separate sensibly.

Requires pop_dict_*.p files saved by pca_compute.py in SUMMARY_DIR.

Usage:
    python diagnostics/sdf_visualization.py --monkey 0 --area M1R
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import glob

from OLD.config import SUMMARY_DIR, EPOCH_WINDOW_MAP_MERGE, TOTAL_TIME_MERGE

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--monkey', type=int, default=0, help='Monkey index (0-based)')
parser.add_argument('--area',   default='M1R', help='Cortical area label, e.g. M1R, PMd, PMv')
parser.add_argument('--max_neurons', type=int, default=20,
                    help='Max neurons to show individual traces for')
args = parser.parse_args()

# ── Find pop_dict file ────────────────────────────────────────────────────────
pattern = os.path.join(SUMMARY_DIR, f'pop_dict_{args.area}_monkey{args.monkey}*.p')
matches = glob.glob(pattern)
if not matches:
    print(f'No pop_dict file found matching: {pattern}')
    print(f'Files in SUMMARY_DIR: {os.listdir(SUMMARY_DIR)}')
    sys.exit(1)

pop_dict_path = matches[0]
print(f'Loading: {pop_dict_path}')
with open(pop_dict_path, 'rb') as f:
    pop_dict = pkl.load(f)   # expected: dict cond_label → SDF array (n_neurons, n_times)

conditions = list(pop_dict.keys())
n_conds    = len(conditions)
t          = np.arange(TOTAL_TIME_MERGE)

# ── Event marker positions ────────────────────────────────────────────────────
event_times = {epoch: info['time'] for epoch, info in EPOCH_WINDOW_MAP_MERGE.items()
               if 'time' in info}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5), sharey=True)
if n_conds == 1:
    axes = [axes]

cmap = plt.cm.tab10

for ax, cond in zip(axes, conditions):
    sdf = pop_dict[cond]  # (n_neurons, n_times)
    n_neurons = sdf.shape[0]

    # Individual neuron traces (subsampled)
    step = max(1, n_neurons // args.max_neurons)
    for i in range(0, n_neurons, step):
        ax.plot(t, sdf[i], lw=0.5, alpha=0.3, color='steelblue')

    # Population mean
    ax.plot(t, sdf.mean(axis=0), lw=2, color='k', label='Mean')

    ax.set_title(f'Cond: {cond}\n(n={n_neurons})')
    ax.set_xlabel('Time (ms)')

    for epoch, te in event_times.items():
        ax.axvline(te, color='tab:red', lw=0.8, ls='--', alpha=0.7)
        ax.text(te, ax.get_ylim()[1] * 0.98, epoch, fontsize=7,
                rotation=90, va='top', ha='right', color='tab:red')

axes[0].set_ylabel('Firing rate (spk/s)')
fig.suptitle(f'Population SDFs — {args.area}  monkey {args.monkey}', fontsize=12)
plt.tight_layout()
plt.show()
