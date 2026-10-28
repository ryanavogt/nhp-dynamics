import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os

import numpy as np
import pandas as pd
import pingouin as pg

from config import (
    MONKEY_NAME_MAP        as monkey_name_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    SELECTIVITY_DIR,
    POPULATION_DIR,
    LAT_MAP                as lat_map,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

POPULATION_DATA  = os.path.join(POPULATION_DIR,  'data')
SELECTIVITY_DATA = os.path.join(SELECTIVITY_DIR, 'data')
SELECTIVITY_FIGS = os.path.join(SELECTIVITY_DIR, 'figures')
for _d in [SELECTIVITY_DATA, SELECTIVITY_FIGS]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n_monkeys       = len(monkey_name_map)
regions         = ['M1', 'PMd', 'PMv']
FR_THRESHOLD_HZ = 1.0
N_CONDITIONS    = 4
fr_scaling      = (binsize / 1000.0) * N_CONDITIONS
fr_threshold    = FR_THRESHOLD_HZ * fr_scaling
load_anova      = False

# ---------------------------------------------------------------------------
# Load pipeline outputs
# ---------------------------------------------------------------------------
area_mean_filename = (f'{POPULATION_DATA}/areaMeanRates_bin{binsize}_k{kernel_width}'
                      f'_merged{n_monkeys}.p')
all_psth_filename  = (f'{POPULATION_DATA}/trialPSTH_bin{binsize}_k{kernel_width}'
                      f'_merged{n_monkeys}.p')
neuron_csv_filename = os.path.join(POPULATION_DATA, 'monkeyNeurons.csv')

with open(area_mean_filename, 'rb') as f:
    area_mean_rate, event_mean_rate = pkl.load(f)
with open(all_psth_filename, 'rb') as f:
    all_psth_dict = pkl.load(f)

neuron_df = pd.read_csv(neuron_csv_filename, index_col=0)
n_neurons = neuron_df.sum(axis=1).to_dict()

print('Pipeline outputs loaded')

# ---------------------------------------------------------------------------
# Rate computation and FR masking
# ---------------------------------------------------------------------------
cond_types        = ['horizontal', 'vertical', 'ipsilateral', 'contralateral']
area_rates        = {region: {cond: [] for cond in cond_types} for region in regions}
event_rates       = {region: {epoch: {}  for epoch in epochs}  for region in regions}
total_rates       = {region: []                                 for region in regions}
event_total_rates = {region: {epoch: {}  for epoch in epochs}  for region in regions}

nzero_mask       = {}
event_nzero_mask = {}

for region_key in area_mean_rate.keys():
    lat            = lat_map[region_key[0]]
    region, orient = region_key[1:].split('_')
    area_mean      = area_mean_rate[region_key]
    area_rates[region][orient].append(area_mean)
    area_rates[region][lat].append(area_mean)
    for event in event_rates[region].keys():
        event_mean = event_mean_rate[region_key][epoch_window_map[event]['event']]
        event_rates[region][event][lat]       = event_mean
        event_total_rates[region][event][lat] = area_mean
    total_rates[region].append(area_mean)

for region in total_rates.keys():
    total_rates[region]     = np.vstack(total_rates[region]).sum(axis=0)
    nzero_mask[region]      = (total_rates[region] > 0)
    for event in event_rates[region].keys():
        e_r = event_rates[region][event]
        event_rates[region][event]       = np.mean([e for e in e_r.values()], axis=0)
        event_total_rates[region][event] = np.sum(
            [i for i in event_total_rates[region][event].values()], axis=0)
    event_total_rates[region] = np.sum(list(event_total_rates[region].values()), axis=0)
    event_nzero_mask[region]  = (event_total_rates[region] > 0)

# ---------------------------------------------------------------------------
# Score computation
# Output: neuronSelection_scores_merged{n}_bin{b}_k{k}_{region}.p
# ---------------------------------------------------------------------------
fr_masks = {}

for region in area_rates.keys():
    for cond in area_rates[region].keys():
        area_rates[region][cond] = np.vstack(area_rates[region][cond]).sum(axis=0)
    a_r     = area_rates[region]
    e_r     = event_rates[region]
    fr_mask = total_rates[region] >= fr_threshold
    fr_masks[region] = fr_mask
    n_before = fr_mask.size
    n_after  = int(fr_mask.sum())
    print(f'  {region}: {n_before} → {n_after} neurons after ≥{FR_THRESHOLD_HZ} Hz '
          f'threshold ({n_before - n_after} excluded)')
    total_rates[region] = total_rates[region][fr_mask]
    for cond in a_r.keys():
        a_r[cond] = a_r[cond][fr_mask]
    for event in e_r.keys():
        e_r[event] = e_r[event][fr_mask]
    orientation_score = (a_r['vertical'] - a_r['horizontal']) / total_rates[region]
    event_score       = (e_r['Grasp'] - e_r['Cue']) / event_total_rates[region][fr_mask]
    hemisphere_score  = (a_r['contralateral'] - a_r['ipsilateral']) / total_rates[region]
    tr                = total_rates[region] / fr_scaling   # convert to Hz
    score_filename    = os.path.join(
        SELECTIVITY_DATA,
        f'neuronSelection_scores_merged{n_monkeys}_bin{binsize}_k{kernel_width}_{region}.p')
    with open(score_filename, 'wb') as f:
        pkl.dump({
            'hemi_score':       hemisphere_score,
            'orient_score':     orientation_score,
            'event_score':      event_score,
            'nzero_mask':       nzero_mask[region],
            'event_nzero_mask': event_nzero_mask[region],
            'fr_mask':          fr_mask,
            'fr_threshold_hz':  FR_THRESHOLD_HZ,
            'total_rates':      tr,
        }, f)
    print(f'  Scores saved for {region}')

# ---------------------------------------------------------------------------
# ANOVA computation
# Output: anova_results_merged{n}.p
# ---------------------------------------------------------------------------
anova_file_name = os.path.join(SELECTIVITY_DATA, f'anova_results_merged{n_monkeys}.p')
orients         = ['horizontal', 'vertical']
lats            = ['c', 'i']
anova_p         = 0.01

if load_anova:
    with open(anova_file_name, 'rb') as f:
        anova_results      = pkl.load(f)
        anova_significance = anova_results['significance']
        anova_pvalues      = anova_results['pvalues']
else:
    anova_significance = {region: {epoch: [] for epoch in epochs} for region in regions}
    anova_pvalues      = {region: {epoch: [] for epoch in epochs} for region in regions}
    for region in regions:
        print(f'Starting ANOVA for {region}')
        fr_mask = fr_masks[region]
        for neuron_idx in range(n_neurons[region]):
            if not fr_mask[neuron_idx]:
                for epoch in epochs:
                    anova_significance[region][epoch].append([False, False, False])
                    anova_pvalues[region][epoch].append([np.nan, np.nan, np.nan])
                continue
            for e_idx, epoch in enumerate(epochs):
                trial_rates_all = []
                hand_labels_all = []
                ori_labels_all  = []
                for lat in lats:
                    for orientation in orients:
                        key      = f'{lat}{region}_{orientation}'
                        psth     = all_psth_dict[key][e_idx][neuron_idx]
                        trial_rates_all.append(psth.mean(axis=0))
                        n_trials = psth.shape[1]
                        hand_labels_all.append([lat] * n_trials)
                        ori_labels_all.append([orientation[0]] * n_trials)
                df = pd.DataFrame({
                    'rate':        np.concatenate(trial_rates_all),
                    'hand':        np.concatenate(hand_labels_all),
                    'orientation': np.concatenate(ori_labels_all),
                })
                if df['rate'].var() == 0 or df['rate'].isna().all():
                    anova_significance[region][epoch].append([False, False, False])
                    anova_pvalues[region][epoch].append([np.nan, np.nan, np.nan])
                    continue
                aov   = pg.anova(data=df, dv='rate', between=['hand', 'orientation'],
                                 detailed=True)
                anova = {
                    'p_hand': aov.loc[aov['Source'] == 'hand',              'p_unc'].values[0],
                    'p_ori':  aov.loc[aov['Source'] == 'orientation',        'p_unc'].values[0],
                    'p_int':  aov.loc[aov['Source'] == 'hand * orientation', 'p_unc'].values[0],
                }
                anova_significance[region][epoch].append(
                    [anova[k] < anova_p for k in anova.keys()])
                anova_pvalues[region][epoch].append(
                    [anova['p_hand'], anova['p_ori'], anova['p_int']])
    for region in regions:
        for epoch in epochs:
            anova_significance[region][epoch] = np.array(anova_significance[region][epoch])
            anova_pvalues[region][epoch]      = np.array(anova_pvalues[region][epoch])
    with open(anova_file_name, 'wb') as f:
        pkl.dump({
            'significance': anova_significance,
            'pvalues':      anova_pvalues,
            'fr_mask':      fr_masks,
        }, f)
    print(f'ANOVA results saved → {anova_file_name}')
