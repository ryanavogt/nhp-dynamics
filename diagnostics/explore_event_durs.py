"""
explore_event_durs.py — Distribution of inter-event durations.

Two rows of plots:
  Row 1  Format B monkeys (Green, Red)  — 4 events
           trialRewardDrop  pre-mask   (from trial_data.p)
           trialGraspOn     post-mask  (from durs_list_merged)
           trialReachOn     post-mask  (from durs_list_merged)
           trialGraspOff    post-mask  (from durs_list_merged)

  Row 2  Format A monkeys (Yellow, Blue) — 2 events
           trialRewardDrop  pre-mask   (from trial_data.p)
           trialGraspOn     post-mask  (trialGraspOn − trialRewardDrop from eventTimes.p)

Red shading marks the rejection region for every EVENT_THRESHOLDS_PREP bound
(trialRewardDrop, trialGraspOn) and every EVENT_THRESHOLDS_DURS bound
(trialGraspOff, trialReachOn).

Console prints how many trials each filter removed.

Run from the project root after data_prep_merge and the merge step.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import glob
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

from config import (
    MONKEY_NAME_MAP         as monkey_name_map,
    MONKEY_LIST_FORMAT_A    as m_list_a,
    MONKEY_LIST_FORMAT_B    as m_list_b,
    SUMMARY_DIR             as summary_dir,
    PROCESSED_DIR           as processed_dir,
    EVENT_THRESHOLDS_PREP   as thresholds_prep,
    EVENT_THRESHOLDS_DURS   as thresholds_durs,
)

sns.set_theme(); sns.set_style('white')

# Merge both threshold dicts for easy lookup
ALL_THRESHOLDS = {**thresholds_prep, **thresholds_durs}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session_dirs_for(monkey_labels):
    dirs = []
    for ml in monkey_labels:
        name = monkey_name_map[ml]
        pattern = os.path.join(processed_dir, f'Monkey_{name}', '????_??_??')
        dirs.extend(sorted(glob.glob(pattern)))
    return dirs


def load_reward_drop(monkey_labels):
    """Load trialRewardDrop PRE-mask from trial_data.p for given monkey labels."""
    vals = []
    for sdir in session_dirs_for(monkey_labels):
        fpath = os.path.join(sdir, 'trial_data.p')
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'rb') as f:
            td = pkl.load(f)
        rd = td.get('trialRewardDrop')
        if rd is not None:
            arr = np.array(rd).astype(float).ravel()
            vals.append(arr[np.isfinite(arr)])
    return np.concatenate(vals) if vals else np.array([])


def load_grasp_on_dur_A(monkey_labels):
    """
    Load trialGraspOn duration POST-mask for Format A monkeys.
    Computed as trialGraspOn − trialRewardDrop from eventTimes.p.
    """
    vals = []
    for sdir in session_dirs_for(monkey_labels):
        fpath = os.path.join(sdir, 'eventTimes.p')
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'rb') as f:
            et = pkl.load(f)
        go = et.get('trialGraspOn')
        rd = et.get('trialRewardDrop')
        if go is not None and rd is not None:
            arr = np.array(go).astype(float) - np.array(rd).astype(float)
            vals.append(arr[np.isfinite(arr)])
    return np.concatenate(vals) if vals else np.array([])


# ---------------------------------------------------------------------------
# Load Format B merged durations (post-mask)
# ---------------------------------------------------------------------------
durs_filename = os.path.join(summary_dir, f'durs_list_merged{len(monkey_name_map)}.p')
with open(durs_filename, 'rb') as f:
    all_durs_b = pkl.load(f)

# ---------------------------------------------------------------------------
# Assemble data for each row
# ---------------------------------------------------------------------------
FORMAT_B_DATA = {
    'trialRewardDrop': load_reward_drop(m_list_b),
    **{k: np.array(v).astype(float) for k, v in all_durs_b.items()},
}
FORMAT_A_DATA = {
    'trialRewardDrop': load_reward_drop(m_list_a),
    'trialGraspOn':    load_grasp_on_dur_A(m_list_a),
}

ROWS = [
    ('Format B — Green, Red  (pre-mask: trialRewardDrop;  post-mask: rest)',
     FORMAT_B_DATA,
     ['trialRewardDrop', 'trialGraspOn', 'trialReachOn', 'trialGraspOff']),
    ('Format A — Yellow, Blue  (pre-mask: trialRewardDrop;  post-mask: trialGraspOn)',
     FORMAT_A_DATA,
     ['trialRewardDrop', 'trialGraspOn']),
]

LABELS = {
    'trialRewardDrop': 'Cue delay\n(Trial start → Cue)',
    'trialGraspOn':    'Cue → Grasp onset',
    'trialReachOn':    'Cue → Reach onset',
    'trialGraspOff':   'Grasp duration\n(Grasp off − on)',
}
COLORS = {
    'trialRewardDrop': 'tab:purple',
    'trialGraspOn':    'tab:blue',
    'trialReachOn':    'tab:orange',
    'trialGraspOff':   'tab:green',
}

PERCENTILES = [5, 25, 50, 75, 95]

# ---------------------------------------------------------------------------
# Figure with subfigures (different column counts per row)
# ---------------------------------------------------------------------------
n_cols_max = max(len(events) for _, _, events in ROWS)
fig_w = 4.5 * n_cols_max
fig = plt.figure(figsize=(fig_w, 14))
subfigs = fig.subfigures(2, 1, hspace=0.10)

for sf, (row_title, data_dict, events) in zip(subfigs, ROWS):
    n_cols = len(events)
    axes = sf.subplots(
        nrows=2, ncols=n_cols,
        gridspec_kw={'height_ratios': [4, 1]},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    sf.suptitle(row_title, fontsize=10, y=1.01)

    for col, event in enumerate(events):
        vals  = data_dict.get(event, np.array([]))
        if vals.size == 0:
            for row in range(2):
                axes[row, col].set_visible(False)
            continue

        vals  = vals[np.isfinite(vals)]
        color = COLORS.get(event, 'tab:gray')
        label = LABELS.get(event, event)
        pcts  = np.percentile(vals, PERCENTILES)
        ax_h  = axes[0, col]
        ax_s  = axes[1, col]
        thr   = ALL_THRESHOLDS.get(event)   # (lo, hi) or None

        # Histogram + KDE
        ax_h.hist(vals, bins=40, color=color, alpha=0.35, density=True)
        x_kde = np.linspace(vals.min(), vals.max(), 400)
        kde   = gaussian_kde(vals, bw_method='scott')
        ax_h.plot(x_kde, kde(x_kde), color=color, lw=2)
        ax_h.set_xlabel('Duration (ms)')
        ax_h.set_ylabel('Density' if col == 0 else '')
        ax_h.set_title(label)

        # Threshold shading (red = rejected region)
        if thr is not None:
            lo, hi = thr
            ylims = ax_h.get_ylim()
            ax_h.axvspan(vals.min(), lo, color='tab:red', alpha=0.12, zorder=0)
            ax_h.axvspan(hi, vals.max(), color='tab:red', alpha=0.12, zorder=0)
            ax_h.axvline(lo, color='tab:red', lw=1.2, ls='--',
                         label=f'thresh [{lo}, {hi}]')
            ax_h.axvline(hi, color='tab:red', lw=1.2, ls='--')
            ax_h.legend(fontsize=7, loc='upper right')
            ax_h.set_ylim(ylims)   # restore after axvspan

            # Console removal stats
            n_total = len(vals)
            n_out   = int(np.sum((vals < lo) | (vals > hi)))
            print(f'  {label.replace(chr(10), " ")}: '
                  f'{n_out} / {n_total} removed ({100*n_out/n_total:.1f}%)')

        # Percentile lines
        for p, v in zip(PERCENTILES, pcts):
            ls = '--' if p in (5, 95) else ':'
            ax_h.axvline(v, color='k', lw=0.8, ls=ls, alpha=0.6)
            ax_h.text(v, ax_h.get_ylim()[1] * 0.97, f'p{p}',
                      ha='center', va='top', fontsize=7, color='k')

        # Strip plot
        jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(vals))
        ax_s.scatter(vals, jitter, s=2, alpha=0.15, color=color, linewidths=0)
        if thr is not None:
            ax_s.axvline(thr[0], color='tab:red', lw=1.2, ls='--')
            ax_s.axvline(thr[1], color='tab:red', lw=1.2, ls='--')
        for v in pcts:
            ax_s.axvline(v, color='k', lw=0.8, ls=':', alpha=0.6)
        ax_s.set_yticks([])
        ax_s.set_xlabel('Duration (ms)')
        ax_s.set_xlim(ax_h.get_xlim())

        # Summary stats to console
        print(f'\n{label.replace(chr(10), " ")}')
        print(f'  n      = {len(vals)}')
        print(f'  mean   = {vals.mean():.1f} ms')
        print(f'  median = {np.median(vals):.1f} ms')
        print(f'  SD     = {vals.std():.1f} ms')
        for p, v in zip(PERCENTILES, pcts):
            print(f'  p{p:<2}    = {v:.1f} ms')

out = os.path.join(summary_dir, 'event_dur_distributions.png')
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {out}')
