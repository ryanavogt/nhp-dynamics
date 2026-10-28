"""
pca_plot_coupled.py — Per-neuron SDF and coupled PC trace figures.

Loads the Monkey objects produced by pca_compute.py and plots:
  • Per-cortex × per-monkey SDF figure:
      - Mean SDF ± SD of the top neurons loading on each of the first 3 PCs
        for each condition (cond_neuron_plot).
  • Coupled PC trace figures (one per monkey, position and velocity):
      - All three cortices overlaid on a grid of n_pcs × n_conditions panels.
      - Each row is a PC; each column is a condition.
      - Figures saved as CoupledPCs_monkey{m}.png and CoupledPCsVelocity_monkey{m}.png

Figures saved under Results/PCA/figures/:
    Monkey{m}/SDF_{cortex}_monkey{m}.png
    CoupledPCs_monkey{m}.png
    CoupledPCsVelocity_monkey{m}.png

Data loaded from Results/PCA/data/PCA_b{b}_k{k}/.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import pickle as pkl
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from utils.sig_proc import velocity
from utils.plot_utils import apply_theme
from utils.pca_utils import cond_neuron_plot, project_sdf, align_sign
from config import (
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    PCA_RESULTS_DIR,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
)
from plot_config import (
    DPI_DRAFT,
    LABEL_SIZE,
    TICK_SIZE,
    ANNOT_SIZE,
    CORTEX_COLORS,
    apply_rcparams,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()
apply_rcparams()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_PCS         = 2   # Number of PC rows in the CoupledPCs grid
MEAN_NEURONS  = 5   # Top-N neurons to average in the SDF figure

PCA_DATA         = os.path.join(PCA_RESULTS_DIR, 'data')
PCA_FIG_DIR      = os.path.join(PCA_RESULTS_DIR, 'figures')
pca_dir          = os.path.join(PCA_DATA, f'PCA_b{binsize}_k{kernel_width}')
os.makedirs(PCA_FIG_DIR, exist_ok=True)
monkey_dict_file = os.path.join(pca_dir, 'allMonkeyDict.p')
pop_filename     = (f'{pca_dir}/pop_dict_b{binsize}_k{kernel_width}'
                    f'_merged{len(monkey_name_map)}.p')

# ---------------------------------------------------------------------------
# Load compute outputs
# ---------------------------------------------------------------------------
print('Loading compute outputs ...')
with open(monkey_dict_file, 'rb') as f: monkeys         = pkl.load(f)
with open(pop_filename,     'rb') as f: merged_pop_dict = pkl.load(f)
print('Loaded.')

cortex_map    = merged_pop_dict['cortex_map']
condition_map = merged_pop_dict['condition_map']


# ---------------------------------------------------------------------------
# Per-cortex × per-monkey SDF figures
# ---------------------------------------------------------------------------
for cort_idx, cortex in enumerate(cortex_map.keys()):
    cond_list = list(condition_map[cortex].keys())

    for m_name, monkey_obj in monkeys.items():
        if cortex not in monkey_obj.cortices:
            continue

        # Reconstruct neuron_dict from saved monkey data
        neuron_dict = {}
        for cond in cond_list:
            cond_sdf  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]['sdf']  # (n_neurons, n_time)
            V_cond    = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]['V']    # (n_neurons, n_pcs)
            pc_sort   = V_cond[:, :3].abs().sort(dim=0, descending=True)
            neuron_dict[cond] = {
                'sdf':     cond_sdf,
                'indices': pc_sort.indices,   # (n_neurons, 3)
            }

        sdf_fig = cond_neuron_plot(neuron_dict, epoch_window_map, plot_neurons=MEAN_NEURONS, mean_plot=True, binsize=binsize)
        sdf_fig.suptitle(
            f'SDF for {cortex}, Mean over {MEAN_NEURONS} Neurons, Monkey {m_name}',
            size='xx-large')
        sdf_fig.tight_layout()

        fig_dir = os.path.join(PCA_FIG_DIR, f'Monkey{m_name}')
        os.makedirs(fig_dir, exist_ok=True)
        out = f'{fig_dir}/SDF_{cortex}_monkey{m_name}.png'
        sdf_fig.savefig(out, dpi=DPI_DRAFT, bbox_inches='tight')
        plt.close(sdf_fig)
        print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Coupled PC trace figures — all cortices overlaid, grid: n_pcs × n_conditions
# ---------------------------------------------------------------------------
# Use the 'All' monkey to determine condition order (same for every monkey).
base_monkey = monkeys['All']
# All cortices share the same conditions (by design); take from the first.
first_cortex  = next(iter(cortex_map.keys()))
cond_list_all = list(base_monkey.condition_map[first_cortex].keys())
n_conds       = len(cond_list_all)

for vel_mode in [False, True]:
    for m_name, monkey_obj in monkeys.items():
        pc_fig, pc_axs = plt.subplots(
            nrows=N_PCS, ncols=n_conds, figsize=(3 * n_conds, 3 * N_PCS))
        # Ensure 2-D indexing even for N_PCS == 1
        if N_PCS == 1:
            pc_axs = [pc_axs]

        for cortex in cortex_map.keys():
            if cortex not in monkey_obj.cortices:
                continue
            monkey_pcs = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['V']

            for cond_idx, condition in enumerate(cond_list_all):
                if condition not in monkey_obj.cortices[cortex]['SVD']['Merged']:
                    continue
                cond_sdf  = monkey_obj.cortices[cortex]['SVD']['Merged'][condition]['sdf']
                cond_proj = project_sdf(cond_sdf, monkey_pcs)   # (n_time, n_pcs)

                for pc in range(N_PCS):
                    ax    = pc_axs[pc][cond_idx]
                    y_traj = cond_proj.T[pc].numpy()
                    if vel_mode:
                        y    = y_traj[1:] - y_traj[:-1]
                        name = 'Velocity'
                    else:
                        y    = y_traj
                        name = ''
                    x = np.arange(len(y)) * binsize
                    # Orient so the first meaningful time-point is positive
                    y = align_sign(y, ref_idx=5)
                    ax.plot(x, y, color=CORTEX_COLORS[cortex], label=cortex)

                    # Decorate only on the last cortex pass (avoids duplicate lines)
                    if cortex == list(cortex_map.keys())[-1]:
                        for epoch in epochs:
                            t_start = epoch_window_map[epoch]['time']
                            t_end   = t_start + epoch_window_map[epoch]['window'][1]
                            ax.axvline(t_start, color='black', alpha=0.5, linewidth=0.8)
                            ax.axvline(t_end,   color='gray',  alpha=0.5, linewidth=0.8)
                        ax.axhline(0, color='black', alpha=0.7, linewidth=0.8)
                        ax.legend(title='Cortex', fontsize=TICK_SIZE)
                        ax.set_title(f'{condition}, PC {pc + 1}', fontsize=ANNOT_SIZE)
                        ax.set_xlabel('Time (ms)')
                        ax.set_ylabel(f'PC {name}')

        label = 'Velocity' if vel_mode else ''
        pc_fig.suptitle(
            f'First {N_PCS} PCs {label} for each Cortex, Monkey {m_name}')
        pc_fig.tight_layout()
        out = f'{PCA_FIG_DIR}/CoupledPCs{label}_monkey{m_name}.png'
        pc_fig.savefig(out, dpi=DPI_DRAFT, bbox_inches='tight')
        plt.close(pc_fig)
        print(f'Saved {out}')

print('\npca_plot_coupled.py complete.')
