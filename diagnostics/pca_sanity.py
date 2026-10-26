"""
pca_sanity.py
-------------
Sanity checks on PCA results from allMonkeyDict.p:
  1. Scree plot — variance explained per PC for each monkey/area
  2. PC stability — overlay PC1-3 projections across trial subsets
     (high overlap = stable; scattered = subsampled PCs are unstable)

Usage:
    python diagnostics/pca_sanity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from config import SUMMARY_DIR, EPOCH_WINDOW_MAP_MERGE, TOTAL_TIME_MERGE

# ── Load ──────────────────────────────────────────────────────────────────────
all_monkey_path = os.path.join(SUMMARY_DIR, 'allMonkeyDict.p')
if not os.path.isfile(all_monkey_path):
    print(f'allMonkeyDict.p not found in {SUMMARY_DIR}'); sys.exit(1)

with open(all_monkey_path, 'rb') as f:
    all_monkey_dict = pkl.load(f)

event_times = {epoch: info['time'] for epoch, info in EPOCH_WINDOW_MAP_MERGE.items()
               if 'time' in info}
t = np.arange(TOTAL_TIME_MERGE)

# ── Scree plots ───────────────────────────────────────────────────────────────
monkeys = list(all_monkey_dict.keys())
areas   = list(all_monkey_dict[monkeys[0]].keys()) if monkeys else []

n_rows = len(monkeys)
n_cols = len(areas)

if n_rows == 0 or n_cols == 0:
    print('allMonkeyDict is empty.'); sys.exit(1)

fig_scree, axes_scree = plt.subplots(n_rows, n_cols,
                                      figsize=(4 * n_cols, 3 * n_rows),
                                      squeeze=False)
fig_scree.suptitle('Scree plots (variance explained per PC)', fontsize=13)

for i, monkey in enumerate(monkeys):
    for j, area in enumerate(areas):
        ax = axes_scree[i, j]
        monkey_obj = all_monkey_dict[monkey].get(area)
        if monkey_obj is None or not hasattr(monkey_obj, 'S'):
            ax.set_title(f'{area}\nmonkey {monkey}\n(no data)')
            continue

        S       = monkey_obj.S           # singular values (1-D tensor or array)
        S_np    = np.array(S.cpu() if hasattr(S, 'cpu') else S)
        var_exp = S_np**2 / (S_np**2).sum() * 100

        ax.bar(np.arange(1, len(var_exp) + 1), var_exp, color='steelblue', alpha=0.8)
        ax.plot(np.arange(1, len(var_exp) + 1), np.cumsum(var_exp),
                'k-o', ms=3, lw=1, label='Cumulative')
        ax.axhline(90, ls='--', color='tab:red', lw=0.8, label='90%')
        ax.set_title(f'{area}  monkey {monkey}')
        ax.set_xlabel('PC')
        ax.set_ylabel('Var exp (%)')
        ax.legend(fontsize=7)

plt.tight_layout()

# ── PC stability: overlay condition trajectories in PC1–PC2 across subsets ───
# Uses right singular vectors (V columns) from the Monkey object
fig_stab, axes_stab = plt.subplots(n_rows, n_cols,
                                    figsize=(4 * n_cols, 4 * n_rows),
                                    squeeze=False)
fig_stab.suptitle('PC1–PC2 trajectory stability across trial subsets', fontsize=13)

for i, monkey in enumerate(monkeys):
    for j, area in enumerate(areas):
        ax = axes_stab[i, j]
        monkey_obj = all_monkey_dict[monkey].get(area)
        if monkey_obj is None or not hasattr(monkey_obj, 'U'):
            ax.set_title(f'{area}\nmonkey {monkey}\n(no data)')
            continue

        # U: (n_times, n_pcs) — left singular vectors = PC trajectories
        U    = monkey_obj.U
        U_np = np.array(U.cpu() if hasattr(U, 'cpu') else U)

        # Plot first two PCs over time as a 2-D trajectory
        ax.plot(U_np[:, 0], U_np[:, 1], lw=1.5, color='k', alpha=0.9)
        ax.scatter(U_np[0, 0], U_np[0, 1], color='green', s=30, zorder=5, label='start')
        ax.scatter(U_np[-1, 0], U_np[-1, 1], color='red', s=30, zorder=5, label='end')

        # Mark event onset positions
        for epoch, te in event_times.items():
            if te < U_np.shape[0]:
                ax.scatter(U_np[te, 0], U_np[te, 1], marker='x',
                           color='tab:orange', s=40, zorder=6)
                ax.annotate(epoch, (U_np[te, 0], U_np[te, 1]),
                            fontsize=6, textcoords='offset points', xytext=(4, 4))

        ax.set_title(f'{area}  monkey {monkey}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
