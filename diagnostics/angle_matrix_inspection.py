"""
angle_matrix_inspection.py
--------------------------
Loads saved AngleMatrix files and visualises the subsampled principal
angles between condition subspaces.

For each file found, plots:
  - Mean ± std of principal angles across subsets (line + shaded band)
  - Strip plot of raw per-subset angles (to see spread)
  - Flags dimension pairs where std is unusually high (> HIGH_STD_THRESH)

Usage:
    python diagnostics/angle_matrix_inspection.py [--dims 3] [--cortex M1R]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle as pkl
import glob
import numpy as np
import matplotlib.pyplot as plt

from OLD.config import SUMMARY_DIR

HIGH_STD_THRESH = 10.0   # degrees — flag if std exceeds this

parser = argparse.ArgumentParser()
parser.add_argument('--dims',   type=int, default=None, help='Filter by PC dims (e.g. 3)')
parser.add_argument('--cortex', default=None,            help='Filter by cortex label (e.g. M1R)')
args = parser.parse_args()

# ── Find AngleMatrix files ────────────────────────────────────────────────────
pattern = os.path.join(SUMMARY_DIR, 'AngleMatrix_*.p')
files   = sorted(glob.glob(pattern))

if not files:
    print(f'No AngleMatrix files found in {SUMMARY_DIR}'); sys.exit(1)

# Apply filters
if args.cortex:
    files = [f for f in files if f'_{args.cortex}_' in os.path.basename(f)]
if args.dims:
    files = [f for f in files if f'PCdims{args.dims}' in os.path.basename(f)]

if not files:
    print(f'No files match the specified filters.'); sys.exit(1)

print(f'Found {len(files)} AngleMatrix file(s):')
for f in files:
    print(f'  {os.path.basename(f)}')

# ── Plot each file ────────────────────────────────────────────────────────────
for fpath in files:
    with open(fpath, 'rb') as fh:
        am = pkl.load(fh)   # dict: mean, std, raw_angles, cond_matrix, cond_list, n_dims, n_subsets, trial_frac

    fname      = os.path.basename(fpath).replace('.p', '')
    mean_ang   = np.array(am['mean'])       # shape depends on implementation
    std_ang    = np.array(am['std'])
    raw_angles = np.array(am['raw_angles']) # (n_subsets, n_comparisons) or similar
    cond_list  = am.get('cond_list', [])
    n_dims     = am.get('n_dims', '?')
    n_subsets  = am.get('n_subsets', raw_angles.shape[0] if raw_angles.ndim > 0 else '?')

    # Flatten to 1-D if multi-dimensional
    mean_1d = mean_ang.flatten()
    std_1d  = std_ang.flatten()
    raw_2d  = raw_angles.reshape(int(n_subsets) if str(n_subsets).isdigit() else -1, len(mean_1d)) \
              if raw_angles.size == int(n_subsets if str(n_subsets).isdigit() else 0) * len(mean_1d) \
              else raw_angles

    n_pairs = len(mean_1d)
    x       = np.arange(n_pairs)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_pairs * 0.6), 8))
    fig.suptitle(fname, fontsize=10)

    # ── Top: mean ± std ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, mean_1d, 'k-o', ms=4, lw=1.5, label='Mean')
    ax.fill_between(x, mean_1d - std_1d, mean_1d + std_1d,
                    alpha=0.25, color='steelblue', label='±1 SD')
    ax.axhline(90, ls='--', color='gray', lw=0.8, label='90° (random)')
    ax.set_ylabel('Principal angle (°)')
    ax.set_title(f'Mean ± SD  ({n_subsets} subsets, {n_dims} dims)')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_list if len(cond_list) == n_pairs else [str(i) for i in x],
                       rotation=45, ha='right', fontsize=7)

    # Flag high-variance pairs
    high_var = np.where(std_1d > HIGH_STD_THRESH)[0]
    for idx in high_var:
        ax.annotate(f'SD={std_1d[idx]:.1f}°', (idx, mean_1d[idx]),
                    textcoords='offset points', xytext=(0, 8),
                    fontsize=7, color='tab:red', ha='center')

    # ── Bottom: raw subset angles ─────────────────────────────────────────────
    ax2 = axes[1]
    if raw_2d.ndim == 2 and raw_2d.shape[1] == n_pairs:
        jitter = (np.random.rand(*raw_2d.shape) - 0.5) * 0.3
        for si in range(raw_2d.shape[0]):
            ax2.scatter(x + jitter[si], raw_2d[si], s=15, alpha=0.5, color='steelblue')
        ax2.plot(x, mean_1d, 'k-o', ms=4, lw=1.5, zorder=5)
    else:
        ax2.text(0.5, 0.5, 'raw_angles shape not compatible with strip plot',
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.axhline(90, ls='--', color='gray', lw=0.8)
    ax2.set_ylabel('Principal angle (°)')
    ax2.set_title('Raw angles per subset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_list if len(cond_list) == n_pairs else [str(i) for i in x],
                        rotation=45, ha='right', fontsize=7)

    if high_var.size > 0:
        print(f'\n⚠  {fname}: high std (>{HIGH_STD_THRESH}°) at indices {high_var.tolist()}')

    plt.tight_layout()

plt.show()
