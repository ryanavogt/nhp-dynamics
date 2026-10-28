"""
pipeline_validation.py
----------------------
Validates pipeline outputs across all processed sessions.

Checks for:
  - Presence of expected files per session directory
  - Keep proportions from eventMasks.p
  - Neuron counts per area
  - Trial counts in trial_data.p

Run from the project root:
    python diagnostics/pipeline_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle as pkl
import glob
from collections import defaultdict

from OLD.config import PROCESSED_DIR

# ── Files expected in every processed session directory ──────────────────────
REQUIRED_FILES = [
    'trial_data.p',
    'eventMasks.p',
    'eventTimes.p',
    'Neurons_U1.p',
    'Neurons_U2.p',
]

# ── Walk every Monkey_*/date session directory ────────────────────────────────
session_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'Monkey_*', '????_??_??')))

if not session_dirs:
    print(f'No session directories found under {PROCESSED_DIR}')
    sys.exit(1)

print(f'Found {len(session_dirs)} session(s)\n')
print(f'{"Session":<30} {"Trials":>8} {"Keep":>6}  {"Missing files / area neuron counts"}')
print('-' * 90)

area_neuron_totals = defaultdict(int)
monkey_session_counts = defaultdict(int)
issues = []

for sdir in session_dirs:
    parts   = sdir.replace('\\', '/').split('/')
    date    = parts[-1]
    monkey  = parts[-2].replace('Monkey_', '')
    label   = f'{monkey}  {date}'
    monkey_session_counts[monkey] += 1

    # ── Check required files ──────────────────────────────────────────────
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(sdir, f))]

    # ── Trial count and keep proportion ──────────────────────────────────
    try:
        with open(os.path.join(sdir, 'eventMasks.p'), 'rb') as f:
            mask = pkl.load(f)
        n_trials = len(mask)
        keep_pct = mask.sum() / n_trials
    except Exception:
        n_trials, keep_pct = -1, float('nan')

    # ── Spike file presence and neuron counts ─────────────────────────────
    area_counts = {}
    for unit in [1, 2]:
        spike_files = glob.glob(os.path.join(sdir, f'spikeTimes_U{unit}_*.p'))
        for sf in spike_files:
            area = sf.split(f'spikeTimes_U{unit}_')[-1].replace('.p', '')
            try:
                with open(os.path.join(sdir, f'Neurons_U{unit}.p'), 'rb') as f:
                    neurons = pkl.load(f)
                n = sum(c for _, c in neurons.get(area, []))
                area_counts[f'U{unit}:{area}'] = n
                area_neuron_totals[area] += n
            except Exception:
                area_counts[f'U{unit}:{area}'] = '?'

    area_str = '  '.join(f'{k}={v}' for k, v in sorted(area_counts.items()))
    miss_str = f'MISSING: {missing}' if missing else area_str

    flag = '⚠' if missing or keep_pct < 0.5 else ' '
    print(f'{flag} {label:<28} {n_trials:>8} {keep_pct:>5.0%}  {miss_str}')

    if missing or keep_pct < 0.5:
        issues.append((label, missing, keep_pct))

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n── Session counts per monkey ────────────────────────────────')
for monkey, count in sorted(monkey_session_counts.items()):
    print(f'  {monkey}: {count} session(s)')

print('\n── Total neurons per area (across all sessions, Unit 1+2) ───')
for area, total in sorted(area_neuron_totals.items()):
    print(f'  {area}: {total}')

if issues:
    print(f'\n⚠  {len(issues)} session(s) with issues (missing files or keep < 50%):')
    for label, missing, keep in issues:
        print(f'   {label}  keep={keep:.0%}  missing={missing}')
else:
    print('\n✓  All sessions passed validation.')
