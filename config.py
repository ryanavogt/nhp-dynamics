"""
config.py — Single source of truth for shared parameters across the NHP Data Analysis project.

Import from here instead of re-declaring constants in each script:
    from config import EPOCH_WINDOW_MAP, BIN_SIZE, KERNEL_WIDTH, MONKEY_NAME_MAP
"""

# ── Monkeys ───────────────────────────────────────────────────────────────────
MONKEY_LIST       = ['G', 'R']
MONKEY_NAME_MAP   = {'G': 'Green', 'R': 'Red', 'Y': 'Yellow', 'B': 'Blue'}
M1_HEMISPHERE_MAP = {'Green': 'R', 'Red': 'R', 'Blue': 'R', 'Yellow': 'R'}

# Format groups used by data_prep_merge.py.
# Format A: legacy MATLAB layout — angle+isRightHandUsed columns; times need ×1000 to reach ms.
# Format B: current MATLAB layout — handOrien column; includes trialReachOn and trialGraspOff.
MONKEY_LIST_FORMAT_A = ['Y', 'B']
MONKEY_LIST_FORMAT_B = ['R', 'G']
# All monkeys: spike times stored in seconds and must be scaled to ms.
SCALE_MONKEYS        = ['R', 'G', 'Y', 'B']

# ── Signal processing ─────────────────────────────────────────────────────────
BIN_SIZE     = 5   # ms — width of histogram bins
KERNEL_WIDTH = 25  # ms — Gaussian kernel width for SDF
N_DIMS       = 5   # number of principal dimensions for PCA (must match pca_compute.py)

# ── Epoch definitions ─────────────────────────────────────────────────────────
# Each script uses a different epoch set; all are defined here so there
# is one place to update windows if they ever change.
BASELINE_EPOCH = 'Pre-Cue'

# Standard 4-epoch map: used by data_process.py, plot_utils.py, and analysis scripts.
EPOCH_WINDOW_MAP = {
    'Cue':       {'event': 'trialRewardDrop', 'window': [-200,  100]},
    'Reach':     {'event': 'trialReachOn',    'window': [-100,   60]},
    'Grasp On':  {'event': 'trialGraspOn',    'window': [ -60,  100]},
    'Grasp Off': {'event': 'trialGraspOff',   'window': [-100,  100]},
}

# 3-epoch map used by data_process_merge.py and all modulation/clustering/plotting scripts.
# Grasp window [-100, 300] ms relative to grasp onset.
EPOCH_WINDOW_MAP_MERGE = {
    'Pre-Cue': {'event': 'trialRewardDrop', 'window': [-300, -100]},
    'Cue':     {'event': 'trialRewardDrop', 'window': [   0,  200]},
    'Grasp':   {'event': 'trialGraspOn',    'window': [-100,  500]},
}

# 2-epoch map used by data_process_earlyLate.py.
EPOCH_WINDOW_MAP_EARLY_LATE = {
    'Early': {'event': 'trialRewardDrop', 'window': [-500, 300]},
    'Late':  {'event': 'trialGraspOn',    'window': [-500, 300]},
}

EPOCH_WINDOW_MAP_PLOT = {
    'Cue':   {'event': 'trialRewardDrop', 'window': [-300, 200]},
    'Grasp': {'event': 'trialGraspOn',    'window': [-100, 600]},
}

def _compute_epoch_times(epoch_map):
    """
    Add a cumulative 'time' offset key to each epoch in the map.
    'time' is the position of the event within the concatenated timeline (ms).
    Returns (mutated map, total duration, list of active epoch names).
    """
    t = 0
    active_epochs = []
    for epoch in epoch_map:
        if epoch == BASELINE_EPOCH:
            continue
        epoch_map[epoch]['time'] = t - epoch_map[epoch]['window'][0]
        t = epoch_map[epoch]['time'] + epoch_map[epoch]['window'][1]
        active_epochs.append(epoch)
    return epoch_map, t, active_epochs


EPOCH_WINDOW_MAP,            TOTAL_TIME,            EPOCHS            = _compute_epoch_times(EPOCH_WINDOW_MAP)
EPOCH_WINDOW_MAP_MERGE,      TOTAL_TIME_MERGE,      EPOCHS_MERGE      = _compute_epoch_times(EPOCH_WINDOW_MAP_MERGE)
EPOCH_WINDOW_MAP_EARLY_LATE, TOTAL_TIME_EARLY_LATE, EPOCHS_EARLY_LATE = _compute_epoch_times(EPOCH_WINDOW_MAP_EARLY_LATE)
EPOCH_WINDOW_MAP_PLOT,       TOTAL_TIME_PLOT,       EPOCHS_PLOT       = _compute_epoch_times(EPOCH_WINDOW_MAP_PLOT)

# Derived: event onset position within the concatenated timeline, keyed by epoch name.
EVENT_TIMES            = {e: EPOCH_WINDOW_MAP[e]['time']            for e in EPOCHS}
EVENT_TIMES_MERGE      = {e: EPOCH_WINDOW_MAP_MERGE[e]['time']      for e in EPOCHS_MERGE}
EVENT_TIMES_EARLY_LATE = {e: EPOCH_WINDOW_MAP_EARLY_LATE[e]['time'] for e in EPOCHS_EARLY_LATE}

# ── Event name mappings ───────────────────────────────────────────────────────
# Map raw MATLAB field names to display labels.
EVENT_MAP = {
    'trialRewardDrop': 'Cue',
    'trialReachOn':    'Reach',
    'trialGraspOn':    'Grasp On',
    'trialGraspOff':   'Grasp Off',
}
EVENT_MAP_MERGE = {
    'trialRewardDrop': 'Cue',
    'trialGraspOn':    'Grasp',
}
EVENT_MAP_EARLY_LATE = {
    'trialRewardDrop': 'Early',
    'trialGraspOn':    'Late',
}
EVENT_MAP_UNALIGNED = {
    'trialReachOn':    'Reach',
    'trialGraspOff':   'Grasp Off'
}

# Ordered list of raw event keys for the merged analysis (Cue then Grasp).
EVENTS_MERGE = list(EVENT_MAP_MERGE.keys())

# ── Hemisphere / condition labels ─────────────────────────────────────────────
LAT_MAP   = {'c': 'contralateral', 'i': 'ipsilateral'}
HAND_LIST = ['R', 'L']

# ── Analysis parameters ───────────────────────────────────────────────────────
P_SCORE      = 0.05            # FDR threshold for modulation t-tests
WINDOW_RANGE = (-1000, 1000)   # ms — full SDF window used by peak_analysis.py

# Inter-event duration bounds (ms) for trial-quality filtering in data_prep_merge.py.
# trialRewardDrop is excluded: the cue delay is machine-controlled (not a behavioral
# marker), so all delay values are equally valid regardless of duration.
EVENT_THRESHOLDS_PREP = {
    'trialGraspOn': [300, 1000],
}

EVENT_THRESHOLDS_DURS = {
    'trialGraspOff': (100, 2000),   # grasp duration (ms) — p5≈47, p95≈1700
    'trialReachOn':  (150,  500),   # cue→reach latency (ms) — p5≈246, p95≈410
}

# ── Modulation color map ──────────────────────────────────────────────────────
MOD_COLOR_MAP = {
    'ipsi':        'tab:green',
    'both:ipsi':   'tab:blue',
    'both:equal':  'tab:brown',
    'both:opp':    'tab:olive',
    'both:contra': 'tab:orange',
    'contra':      'tab:red',
    'none':        'tab:gray',
    'both:all':    'tab:cyan',
}

# ── Directory layout ──────────────────────────────────────────────────────────
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT     = os.path.join(_ROOT, 'Data')
SORTED_DIR    = os.path.join(DATA_ROOT, 'Sorted_Inactivation')
MATLAB_DIR    = os.path.join(SORTED_DIR, 'matlabFiles')
SORTING_DIR   = os.path.join(SORTED_DIR, 'sortingNotes')
PROCESSED_DIR = os.path.join(DATA_ROOT, 'Processed')
SUMMARY_DIR      = os.path.join(PROCESSED_DIR, 'Summary')

RESULTS_DIR      = os.path.join(_ROOT, 'Results')
POPULATION_DIR   = os.path.join(RESULTS_DIR, 'Population')
MODULATION_DIR   = os.path.join(RESULTS_DIR, 'Modulation')
SELECTIVITY_DIR  = os.path.join(RESULTS_DIR, 'Selectivity')
CLUSTERING_DIR   = os.path.join(RESULTS_DIR, 'Clustering')
PCA_RESULTS_DIR  = os.path.join(RESULTS_DIR, 'PCA')
DMD_DIR          = os.path.join(RESULTS_DIR, 'DMD')
