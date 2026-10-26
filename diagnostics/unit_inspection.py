"""
unit_inspection.py
------------------
Plots spike rasters for a single channel, session, and area.
Useful for visually verifying that the spike splitter worked correctly.

Usage:
    python diagnostics/unit_inspection.py --monkey Green --date 2022_03_14 --area M1R --unit 1

Or edit the DEFAULTS below and run without arguments.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from OLD.config import PROCESSED_DIR, EPOCH_WINDOW_MAP_MERGE

# ── Defaults (edit here for quick runs) ──────────────────────────────────────
DEFAULTS = dict(monkey='Green', date='2022_03_14', area='M1R', unit=1, channel=None)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--monkey',  default=DEFAULTS['monkey'])
parser.add_argument('--date',    default=DEFAULTS['date'])
parser.add_argument('--area',    default=DEFAULTS['area'])
parser.add_argument('--unit',    type=int, default=DEFAULTS['unit'])
parser.add_argument('--channel', type=int, default=DEFAULTS['channel'],
                    help='Specific channel to plot (default: first channel in area)')
args = parser.parse_args()

session_dir = os.path.join(PROCESSED_DIR, f'Monkey_{args.monkey}', args.date)
if not os.path.isdir(session_dir):
    print(f'Session directory not found: {session_dir}'); sys.exit(1)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(os.path.join(session_dir, f'spikeTimes_U{args.unit}_{args.area}.p'), 'rb') as f:
    spike_times = pkl.load(f)  # dict: channel_sig → array (n_spikes, 2) [trial_idx, spike_time_ms]

with open(os.path.join(session_dir, 'trial_data.p'), 'rb') as f:
    trial_data = pkl.load(f)

with open(os.path.join(session_dir, 'eventMasks.p'), 'rb') as f:
    mask = pkl.load(f)

n_trials = mask.sum()
hand_orien = trial_data['handOrien'].values if hasattr(trial_data['handOrien'], 'values') else np.array(trial_data['handOrien'])
conditions = np.unique(hand_orien)

# ── Select channel ────────────────────────────────────────────────────────────
channels = sorted(spike_times.keys())
if args.channel is not None and args.channel in channels:
    channel = args.channel
else:
    channel = channels[0]
    if args.channel is not None:
        print(f'Channel {args.channel} not found; using {channel}. Available: {channels}')

spikes = spike_times[channel]  # shape (n_spikes, 2): col0=trial_idx, col1=spike_time_ms

# ── Plot raster ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 6), sharey=True)
if len(conditions) == 1:
    axes = [axes]

for ax, cond in zip(axes, conditions):
    cond_trials = np.where((hand_orien == cond) & mask)[0]
    for row_idx, trial_idx in enumerate(cond_trials):
        trial_spikes = spikes[spikes[:, 0] == trial_idx, 1]
        ax.scatter(trial_spikes, np.full_like(trial_spikes, row_idx),
                   s=1, c='k', linewidths=0)
    ax.set_title(f'Cond {cond}')
    ax.set_xlabel('Time from trial start (ms)')

    # Event markers
    for epoch, info in EPOCH_WINDOW_MAP_MERGE.items():
        t = info.get('time')
        if t is not None:
            ax.axvline(t, color='tab:red', lw=0.8, ls='--', alpha=0.7)
            ax.text(t, ax.get_ylim()[1] * 0.98, epoch, fontsize=7,
                    rotation=90, va='top', ha='right', color='tab:red')

axes[0].set_ylabel('Trial')
fig.suptitle(f'{args.monkey}  {args.date}  —  {args.area}  U{args.unit}  Ch{channel}', fontsize=11)
plt.tight_layout()
plt.show()
