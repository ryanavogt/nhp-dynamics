"""
pca_compute.py — PCA computation and principal angle statistics.

Builds the population SDF dictionary, runs SVD for each cortex/monkey,
and computes principal angles between condition subspaces using a
subsampling approach: N_SUBSETS random draws of TRIAL_FRACTION of trials,
yielding mean and SD of the leading principal angle across subsets.

For within-condition (diagonal) pairs: each subset draws TRIAL_FRACTION of
trials and splits them into two non-overlapping halves; angles are computed
between the two half-subset subspaces.

For between-condition (off-diagonal) pairs: each subset independently draws
TRIAL_FRACTION of each condition's trials; angles are computed between the
two resulting subspaces.

Outputs saved under Results/PCA/data/PCA_b{b}_k{k}/:
    pop_dict_b{b}_k{k}_merged{n}.p          — population SDF + metadata
    AngleMatrix_{cortex}_monkey{m}_PCdims{d}.p  — angle statistics (mean/std/raw)
    event_marker_idx.p                       — event timing indices for plotting
    allMonkeyDict.p                          — all Monkey objects
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import pickle as pkl
import numpy as np
import torch

from utils.Monkey import Monkey
from utils.pca_utils import compute_subspace

from config import (
    MONKEY_NAME_MAP  as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    TOTAL_TIME_MERGE as total_time,
    POPULATION_DIR,
    PCA_RESULTS_DIR,
    BIN_SIZE         as binsize,
    KERNEL_WIDTH     as kernel_width,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SUBSETS      = 5      # Number of random trial subsamples
TRIAL_FRACTION = 0.80   # Fraction of each condition's trials per subset
N_DIMS         = 5      # Subspace dimensionality for principal angles
NEW_POPDICT    = True   # Set True to rebuild the population SDF dict from scratch
PCA_OVERWRITE  = True   # Set True to rerun SVD even if a saved PCA file exists

all_x = np.arange(0, total_time, binsize)

POPULATION_DATA = os.path.join(POPULATION_DIR, 'data')
PCA_DATA        = os.path.join(PCA_RESULTS_DIR, 'data')
PCA_FIGS        = os.path.join(PCA_RESULTS_DIR, 'figures')
pca_dir         = os.path.join(PCA_DATA, f'PCA_b{binsize}_k{kernel_width}')
for _d in [pca_dir, PCA_FIGS]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Load input files
# ---------------------------------------------------------------------------
print('Loading input files ...')
durs_filename    = f'{POPULATION_DATA}/durs_list_merged{len(monkey_name_map)}.p'
sdf_filename     = f'{POPULATION_DATA}/merged_sdfDict_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map)}.p'
psth_filename    = f'{POPULATION_DATA}/trialPSTH_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map)}.p'
monkey_filename  = f'{POPULATION_DATA}/monkey_indices_merged{len(monkey_name_map)}.p'

with open(durs_filename,   'rb') as f: all_durs         = pkl.load(f)
with open(sdf_filename,    'rb') as f: all_sdf_dict     = pkl.load(f)
with open(psth_filename,   'rb') as f: all_psth_dict    = pkl.load(f)
with open(monkey_filename, 'rb') as f: all_monkey_indices = pkl.load(f)
print('Input files loaded.')

# ---------------------------------------------------------------------------
# Event timing indices (saved for use by plotting scripts)
# ---------------------------------------------------------------------------
# all_durs stores inter-event durations (matching data_prep_merge.py):
#   trialGraspOn  – cue → grasp onset            (reference; grasp-relative = 0)
#   trialReachOn  – cue → reach onset             (cue-relative)
#   trialGraspOff – grasp duration (off − on)     (grasp-relative)
#
# The concatenated timeline is NOT linear real time — it splices epoch windows —
# so we anchor every event to the known grasp position on the timeline:
#   cue-relative events:   delta = event_mean − mean_grasp_from_cue
#   grasp-relative events: delta = event_mean          (already offset from grasp)
#   tl_pos = grasp_timeline_pos + delta
grasp_timeline_pos  = epoch_window_map['Grasp']['time']
mean_grasp_from_cue = np.mean(all_durs['trialGraspOn'])

# Map each event key to its reference frame so the delta is computed correctly.
# 'cue'   → subtract mean_grasp_from_cue to get offset from grasp onset
# 'grasp' → use directly as offset from grasp onset
_event_ref = {
    'trialReachOn':  'cue',
    'trialGraspOff': 'grasp',
}

event_marker_idx = {}
for event, vals in all_durs.items():
    if event == 'trialGraspOn':
        continue   # already shown as the Grasp epoch marker in the plot; omit here
    ref        = _event_ref.get(event, 'grasp')    # default to grasp-relative
    event_mean = np.mean(vals)
    event_std  = np.std(vals)
    delta      = (event_mean - mean_grasp_from_cue) if ref == 'cue' else event_mean
    tl_pos     = grasp_timeline_pos + delta         # position on concatenated timeline
    event_marker_idx[event] = {
        'mean': [np.argmin(np.abs(all_x - tl_pos))],
        'std':  [np.argmin(np.abs(all_x - (tl_pos - event_std))),
                 np.argmin(np.abs(all_x - (tl_pos + event_std)))],
    }
event_idx_file = os.path.join(pca_dir, 'event_marker_idx.p')
with open(event_idx_file, 'wb') as f:
    pkl.dump(event_marker_idx, f)
print(f'Event marker indices saved → {event_idx_file}')

# ---------------------------------------------------------------------------
# Build or load the merged population SDF dictionary
# ---------------------------------------------------------------------------
pop_filename = (f'{pca_dir}/pop_dict_b{binsize}_k{kernel_width}'
                f'_merged{len(monkey_name_map)}.p')

if os.path.exists(pop_filename) and not NEW_POPDICT:
    with open(pop_filename, 'rb') as f:
        merged_pop_dict = pkl.load(f)
    print('Population dictionary loaded.')
else:
    print('Building population dictionary ...')
    merged_population_dict = {}
    region_map    = {'idx': 0}
    condition_map = {}
    cortex_map    = {'idx': 0}
    merged_pop_dict = {'sdf': [], 'psth': [], 'conditions': []}

    for area in all_sdf_dict.keys():
        region, orientation = area.split('_')
        side, cortex = region[0], region[1:]
        if orientation not in merged_population_dict:
            merged_population_dict[orientation] = []
        if cortex not in condition_map:
            condition_map[cortex] = {'idx': 0}

        sdf = all_sdf_dict[area][:, 0].T                     # (n_neurons, n_time)
        psth_epochs = all_psth_dict[area]
        psth = [np.vstack([i, j]).T
                for i, j in zip(psth_epochs[0], psth_epochs[1])]  # list[(n_trials, n_time)]

        sdf_max = sdf.max(axis=1) + 5 * binsize / 1000
        sdf     = sdf / np.expand_dims(sdf_max, 1)

        merged_population_dict[orientation].append(sdf)

        if cortex not in cortex_map:
            cortex_map[cortex] = [cortex_map['idx'], cortex_map['idx']]
        cortex_map[cortex][1] += sdf.shape[0]
        cortex_map['idx']     += sdf.shape[0]

        if region not in region_map:
            region_map[region] = [region_map['idx'], region_map['idx'] + sdf.shape[0]]
            region_map['idx'] += sdf.shape[0]

        or_key = f'{side}_{orientation}'
        if or_key not in condition_map[cortex]:
            condition_map[cortex][or_key] = [
                condition_map[cortex]['idx'],
                condition_map[cortex]['idx'] + sdf.shape[0],
            ]
            condition_map[cortex]['idx'] += sdf.shape[0]

        merged_pop_dict['sdf'].append(sdf)
        merged_pop_dict['psth'].append(psth)
        merged_pop_dict['conditions'].append(area)

    for orientation, sdf_list in merged_population_dict.items():
        merged_population_dict[orientation] = np.vstack(sdf_list)

    del region_map['idx']
    for cor in condition_map:
        del condition_map[cor]['idx']
    del cortex_map['idx']

    merged_pop_sdf = torch.Tensor(np.vstack(merged_pop_dict['sdf']))
    merged_pop_dict['sdf']           = merged_pop_sdf
    merged_pop_dict['region_map']    = region_map
    merged_pop_dict['cortex_map']    = cortex_map
    merged_pop_dict['condition_map'] = condition_map

    with open(pop_filename, 'wb') as f:
        pkl.dump(merged_pop_dict, f)
    print(f'Population dictionary saved → {pop_filename}')

# Unpack commonly used references
cortex_map      = merged_pop_dict['cortex_map']
merged_pop_sdf  = merged_pop_dict['sdf']
condition_map   = merged_pop_dict['condition_map']
condition_order = merged_pop_dict['conditions']

# ---------------------------------------------------------------------------
# Main compute loop: SVD + trial PSTH + subsampled principal angles
# ---------------------------------------------------------------------------
monkeys = {}

for cort_idx, cortex in enumerate(cortex_map.keys()):
    print(f'\n=== {cortex} ===')
    cort           = cortex_map[cortex]
    sdf            = merged_pop_sdf[cort[0]:cort[1]]
    monkey_indices = all_monkey_indices[cortex]
    del monkey_indices['count']
    monkey_indices['All'] = np.arange(
        0, max(e[-1] for e in monkey_indices.values()) + 1
    )

    for monkey in monkey_indices.keys():
        print(f'  Monkey {monkey} ...')
        if monkey not in monkeys:
            monkeys[monkey] = Monkey(monkey, kernel_width, binsize, epoch_window_map)
        monkey_obj = monkeys[monkey]
        monkey_obj.load_indices(monkey_indices, cortex)
        monkey_obj.set_condition_map(condition_map)
        monkey_obj.set_save_dir(pca_dir)

        # SVD and trial-averaged PSTH (writes into monkey_obj.cortices)
        monkey_obj.compute_sdf(sdf, cortex)
        monkey_obj.compute_svd(cortex)
        monkey_obj.compute_trial_psth(merged_pop_dict['psth'], cortex, condition_order)

        cond_list = list(condition_map[cortex].keys())
        n_conds   = len(cond_list)

        # -----------------------------------------------------------------------
        # Subsampled principal angles
        #
        # For each of N_SUBSETS random draws:
        #   Off-diagonal (A vs B): sample TRIAL_FRACTION of A and TRIAL_FRACTION
        #     of B independently → compute subspace for each → principal angles.
        #   Diagonal (A vs A): sample TRIAL_FRACTION of A, split the sample into
        #     two non-overlapping halves → compute a subspace from each half →
        #     principal angles.  This is the within-condition reliability estimate.
        #
        # Aggregate: mean and SD of the leading principal angle across N_SUBSETS.
        # -----------------------------------------------------------------------
        # raw_angles[cond_i][cond_j] is a list of length N_SUBSETS,
        # each element a numpy array of shape (N_DIMS,) — cosines of the
        # N_DIMS principal angles for that subset.
        raw_angles = {c: {o: [] for o in cond_list} for c in cond_list}

        for k in range(N_SUBSETS):
            # --- sample one subspace per condition for off-diagonal pairs ---
            # Each neuron independently draws TRIAL_FRACTION of its own trials,
            # so neurons with fewer trials are not used to cap those with more.
            subset_Vs = {}
            for cond in cond_list:
                cond_psth      = monkey_obj.cortices[cortex]['psth']['Merged'][cond]
                idx_per_neuron = [
                    np.random.choice(p.shape[0],
                                     max(2, int(round(p.shape[0] * TRIAL_FRACTION))),
                                     replace=False)
                    for p in cond_psth
                ]
                subset_Vs[cond] = compute_subspace(cond_psth, idx_per_neuron, N_DIMS,
                                                   kernel_width, binsize)

            for cond in cond_list:
                for o_cond in cond_list:
                    if cond == o_cond:
                        # --- diagonal: per-neuron split-within-subsample ---
                        # For each neuron, draw TRIAL_FRACTION of its trials, then split
                        # into two non-overlapping halves → no shared trials between V1 and V2.
                        cond_psth  = monkey_obj.cortices[cortex]['psth']['Merged'][cond]
                        idx_pairs  = []
                        for p in cond_psth:
                            n      = p.shape[0]
                            half_n = max(1, int(round(n * TRIAL_FRACTION / 2)))
                            pool   = np.random.choice(n, 2 * half_n, replace=False)
                            idx_pairs.append((pool[:half_n], pool[half_n:]))
                        V1 = compute_subspace(cond_psth, [pair[0] for pair in idx_pairs],
                                              N_DIMS, kernel_width, binsize)
                        V2 = compute_subspace(cond_psth, [pair[1] for pair in idx_pairs],
                                              N_DIMS, kernel_width, binsize)
                        G  = V1.T @ V2
                    else:
                        G  = subset_Vs[cond].T @ subset_Vs[o_cond]

                    cosines = torch.linalg.svdvals(G).numpy()  # (N_DIMS,), descending
                    raw_angles[cond][o_cond].append(cosines)

        # --- aggregate across subsets ---
        mean_matrix = np.zeros((n_conds, n_conds))
        std_matrix  = np.zeros((n_conds, n_conds))
        for ci, cond in enumerate(cond_list):
            for cj, o_cond in enumerate(cond_list):
                stacked = np.stack(raw_angles[cond][o_cond], axis=0)  # (K, N_DIMS)
                # Summarise using the first (leading) principal angle
                mean_matrix[ci, cj] = stacked[:, 0].mean()
                std_matrix[ci, cj]  = stacked[:, 0].std(ddof=1)

        cond_matrix = np.empty((n_conds, n_conds), dtype=object)
        for ci, cond in enumerate(cond_list):
            for cj, o_cond in enumerate(cond_list):
                cond_matrix[ci, cj] = f'{cond}:{o_cond}'

        angle_file = os.path.join(
            pca_dir,
            f'AngleMatrix_{cortex}_monkey{monkey}_PCdims{N_DIMS}.p'
        )
        with open(angle_file, 'wb') as f:
            pkl.dump({
                'mean':        mean_matrix,
                'std':         std_matrix,
                'raw_angles':  raw_angles,      # full K × N_DIMS data for each pair
                'cond_matrix': cond_matrix,
                'cond_list':   cond_list,
                'n_dims':      N_DIMS,
                'n_subsets':   N_SUBSETS,
                'trial_frac':  TRIAL_FRACTION,
            }, f)
        print(f'    Angle matrix saved → {angle_file}')

# ---------------------------------------------------------------------------
# Save all Monkey objects
# ---------------------------------------------------------------------------
for m_name, monkey in monkeys.items():
    monkey.save_monkey()
    print(f'Saved Monkey {m_name}')

monkey_dict_file = os.path.join(pca_dir, 'allMonkeyDict.p')
with open(monkey_dict_file, 'wb') as f:
    pkl.dump(monkeys, f)
print(f'\nMonkey dict saved → {monkey_dict_file}')

# ---------------------------------------------------------------------------
# PC neuron indices — eigenvalue-weighted contribution (sorted descending)
#
# For each cortex, each neuron's contribution weight is computed as:
#   w_i = sqrt( sum_j( λ_j * V_ij² ) / sum_j( λ_j ) )   for j in {1, 2, 3}
# where λ_j = S_j² are the eigenvalues of the first 3 PCs and V_ij is
# neuron i's loading on PC j from the merged ('All') monkey SVD.
# Neurons are sorted by w_i descending; the resulting index array is used
# by plot_rasters_classes.py and pca_plot_neurons.py.
#
# Output: pcIndicesJoint_merged{n}.p
# ---------------------------------------------------------------------------
print('\nComputing PC neuron indices (eigenvalue-weighted contribution) ...')
merged_monkey = monkeys['All']
joint_indices = {}

for cortex in cortex_map.keys():
    merged_S  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['S']
    merged_V  = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
    eigvals   = merged_S[:3] ** 2
    eig_total = (merged_S ** 2).sum()
    _, V_indices = (
        (eigvals * merged_V[:, :3] ** 2).sum(dim=1) / eig_total
    ).sqrt().sort(descending=True, dim=0)
    joint_indices[cortex] = V_indices
    print(f'  {cortex}: indices computed ({merged_monkey.cortices[cortex]["neurons"]} neurons)')

joint_indices_file = os.path.join(
    pca_dir, f'pcIndicesJoint_merged{len(monkey_name_map)}.p'
)
with open(joint_indices_file, 'wb') as f:
    pkl.dump(joint_indices, f)
print(f'PC neuron indices saved → {joint_indices_file}')

print('\npca_compute.py complete.')
