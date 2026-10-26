"""
session_summary.py
------------------
Aggregates metadata across all processed sessions and prints a table
suitable for copying into the Methods section or a supplemental table.

Prints:
  - Per-monkey: session count, date range, total neurons per area
  - Per-session: trial count, keep proportion, neuron counts
  - Grand totals

Usage:
    python diagnostics/session_summary.py [--csv]

With --csv, also writes session_summary.csv to the current directory.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle as pkl
import glob
import csv
from collections import defaultdict

from OLD.config import PROCESSED_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--csv', action='store_true', help='Also save CSV')
args = parser.parse_args()

session_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'Monkey_*', '????_??_??')))
if not session_dirs:
    print(f'No sessions found under {PROCESSED_DIR}'); sys.exit(1)

rows = []
monkey_data = defaultdict(lambda: {
    'sessions': 0, 'dates': [], 'areas': defaultdict(int),
    'trials_pre': 0, 'trials_post': 0,
})

for sdir in session_dirs:
    parts  = sdir.replace('\\', '/').split('/')
    date   = parts[-1]
    monkey = parts[-2].replace('Monkey_', '')

    # Trial counts
    try:
        with open(os.path.join(sdir, 'trial_data.p'), 'rb') as f:
            td = pkl.load(f)
        n_pre = len(td['trialStartTimes'])
    except Exception:
        n_pre = -1

    try:
        with open(os.path.join(sdir, 'eventMasks.p'), 'rb') as f:
            mask = pkl.load(f)
        n_post = int(mask.sum())
        keep   = n_post / len(mask) if len(mask) > 0 else float('nan')
    except Exception:
        n_post, keep = -1, float('nan')

    # Neuron counts per area
    area_counts = {}
    for unit in [1, 2]:
        npath = os.path.join(sdir, f'Neurons_U{unit}.p')
        if not os.path.isfile(npath):
            continue
        with open(npath, 'rb') as f:
            neurons = pkl.load(f)
        for area, cell_list in neurons.items():
            n = sum(c for _, c in cell_list)
            area_counts[area] = area_counts.get(area, 0) + n
            monkey_data[monkey]['areas'][area] += n

    monkey_data[monkey]['sessions'] += 1
    monkey_data[monkey]['dates'].append(date)
    monkey_data[monkey]['trials_pre']  += max(n_pre, 0)
    monkey_data[monkey]['trials_post'] += max(n_post, 0)

    rows.append({
        'monkey':   monkey,
        'date':     date,
        'trials_pre':  n_pre,
        'trials_post': n_post,
        'keep_pct': keep,
        **{f'neurons_{a}': c for a, c in area_counts.items()},
    })

# ── Per-monkey summary ────────────────────────────────────────────────────────
print('─' * 70)
print('PER-MONKEY SUMMARY')
print('─' * 70)
all_areas = sorted({a for d in monkey_data.values() for a in d['areas']})
header = f'  {"Monkey":<10} {"Sessions":>9} {"Date range":<24} ' + \
         ''.join(f'{a:>10}' for a in all_areas) + f'  {"Pre-mask":>9} {"Post-mask":>9}'
print(header)
print('  ' + '-' * (len(header) - 2))

grand_neurons = defaultdict(int)
grand_pre, grand_post = 0, 0

for monkey, d in sorted(monkey_data.items()):
    dates     = sorted(d['dates'])
    date_rng  = f'{dates[0]} – {dates[-1]}'
    neuron_str = ''.join(f'{d["areas"].get(a, 0):>10}' for a in all_areas)
    print(f'  {monkey:<10} {d["sessions"]:>9} {date_rng:<24}{neuron_str}  '
          f'{d["trials_pre"]:>9} {d["trials_post"]:>9}')
    for a in all_areas:
        grand_neurons[a] += d['areas'].get(a, 0)
    grand_pre  += d['trials_pre']
    grand_post += d['trials_post']

print('  ' + '-' * (len(header) - 2))
grand_neuron_str = ''.join(f'{grand_neurons.get(a, 0):>10}' for a in all_areas)
print(f'  {"TOTAL":<10} {sum(d["sessions"] for d in monkey_data.values()):>9} '
      f'{"":24}{grand_neuron_str}  {grand_pre:>9} {grand_post:>9}')

# ── Per-session table ─────────────────────────────────────────────────────────
print('\n─' * 70)
print('PER-SESSION DETAIL')
print('─' * 70)
all_area_cols = sorted({k for r in rows for k in r if k.startswith('neurons_')})
hdr = f'  {"Monkey":<10} {"Date":<14} {"Pre":>7} {"Post":>7} {"Keep":>6}  ' + \
      '  '.join(f'{c.replace("neurons_",""):>6}' for c in all_area_cols)
print(hdr)
print('  ' + '-' * (len(hdr) - 2))

for r in rows:
    neuron_str = '  '.join(f'{r.get(c, 0):>6}' for c in all_area_cols)
    keep_str   = f'{r["keep_pct"]:.0%}' if r["keep_pct"] == r["keep_pct"] else 'N/A'
    print(f'  {r["monkey"]:<10} {r["date"]:<14} {r["trials_pre"]:>7} '
          f'{r["trials_post"]:>7} {keep_str:>6}  {neuron_str}')

# ── Optional CSV ──────────────────────────────────────────────────────────────
if args.csv:
    out_path = 'session_summary.csv'
    fieldnames = ['monkey', 'date', 'trials_pre', 'trials_post', 'keep_pct'] + all_area_cols
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved to: {os.path.abspath(out_path)}')
