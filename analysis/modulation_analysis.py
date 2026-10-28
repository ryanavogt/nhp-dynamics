import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os

import numpy as np
import pandas as pd

from config import (
    MODULATION_DIR,
    POPULATION_DIR,
    MONKEY_NAME_MAP,
    EVENT_MAP_MERGE      as event_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    LAT_MAP              as lat_map,
    EVENTS_MERGE         as events,
    BIN_SIZE             as binsize,
    KERNEL_WIDTH         as kernel_width,
    BASELINE_EPOCH,
    P_SCORE              as p_score,
)
from utils.sig_proc import epoch_trial_rates, laterality_ttest
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
POPULATION_DATA = os.path.join(POPULATION_DIR, 'data')
MODULATION_DATA = os.path.join(MODULATION_DIR, 'data')
MODULATION_FIGS = os.path.join(MODULATION_DIR, 'figures')
for _d in [MODULATION_DATA, MODULATION_FIGS]:
    os.makedirs(_d, exist_ok=True)

load_override = False

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
spike_file_name     = f'{POPULATION_DATA}/spike_list_merged{len(MONKEY_NAME_MAP)}.p'
event_file_name     = f'{POPULATION_DATA}/event_list_merged{len(MONKEY_NAME_MAP)}.p'
condition_file_name = f'{POPULATION_DATA}/condition_list_merged{len(MONKEY_NAME_MAP)}.p'
all_sdf_filename    = f'{POPULATION_DATA}/merged_sdfDict_bin{binsize}_k{kernel_width}_orientMerged.p'
lat_ttest_filename  = f'{MODULATION_DATA}/laterality_ttest_dict.p'
merged_mods_filename  = f'{MODULATION_DATA}/MergedMods.p'
merged_df_filename    = f'{MODULATION_DATA}/MergedModDataFrame.p'
hand_use_mod_filename = f'{MODULATION_DATA}/HandUseMod.p'

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
with open(spike_file_name, 'rb') as f:
    all_spikes = pkl.load(f)
with open(event_file_name, 'rb') as f:
    all_events = pkl.load(f)
with open(condition_file_name, 'rb') as f:
    all_conditions = pkl.load(f)

# Normalise all_events keys to match event_map values ('Cue', 'Grasp')
for _area in all_events:
    if 'Grasp On' in all_events[_area]:
        all_events[_area]['Grasp'] = all_events[_area].pop('Grasp On')

with open(all_sdf_filename, 'rb') as f:
    all_sdf_dict = pkl.load(f)

# ---------------------------------------------------------------------------
# Loop 1: Per-trial laterality t-test (cached)
# ---------------------------------------------------------------------------
if os.path.exists(lat_ttest_filename) and not load_override:
    with open(lat_ttest_filename, 'rb') as f:
        laterality_ttest_dict = pkl.load(f)
    print('Laterality t-test dict loaded')
else:
    laterality_ttest_dict = {}
    for area in ['M1', 'PMd', 'PMv']:
        laterality_ttest_dict[area] = {}
        for event in events:
            event_label = event_map[event]
            epoch_ms    = epoch_window_map[event_label]['window']
            n       = len(all_spikes[area])
            t_stats = np.full(n, np.nan)
            p_vals  = np.full(n, np.nan)
            for neuron_idx in range(n):
                t_stats[neuron_idx], p_vals[neuron_idx] = laterality_ttest(
                    area, neuron_idx, event, epoch_ms,
                    all_spikes, all_conditions, all_events, event_map)
            valid  = ~np.isnan(p_vals)
            reject = np.zeros(n, dtype=bool)
            if valid.sum() > 1:
                reject[valid], _, _, _ = multipletests(
                    p_vals[valid], alpha=p_score, method='fdr_bh')
            laterality_ttest_dict[area][event_label] = {
                't':         t_stats,
                'p':         p_vals,
                'reject':    reject,
                'direction': np.sign(t_stats),
            }
    with open(lat_ttest_filename, 'wb') as f:
        pkl.dump(laterality_ttest_dict, f)
    print('Laterality t-test dict saved')

# ---------------------------------------------------------------------------
# Loop 2: Per-trial modulation t-tests (baseline vs event epoch)
# ---------------------------------------------------------------------------
full_window   = np.arange(-1000, 1000 + binsize, binsize)
base_window   = epoch_window_map[BASELINE_EPOCH]['window']
baseline_mask = (full_window > base_window[0]) * (full_window < base_window[1])

event_masks = {}
slope_masks = {}
for event_label in ['Cue', 'Grasp']:
    event_window             = epoch_window_map[event_label]['window']
    event_masks[event_label] = (full_window > event_window[0]) * (full_window < event_window[1])
    slope_masks[event_label] = event_masks[event_label]

baseline_raw_event   = epoch_window_map[BASELINE_EPOCH]['event']   # 'trialRewardDrop'
baseline_event_label = event_map[baseline_raw_event]               # 'Cue'
baseline_epoch_ms    = base_window                                  # [-300, -100]

modulation_dict    = {}
up_down_modulation = {}
tVals_dict         = {}
mod_sdfs           = {}

for region_key in all_sdf_dict.keys():   # 'cM1', 'iM1', 'cPMd', ...
    lat       = region_key[0]
    area      = region_key[1:]
    lat_label = lat_map[lat]

    if area not in modulation_dict:
        modulation_dict[area]    = {}
        up_down_modulation[area] = {}
        tVals_dict[area]         = {}
        mod_sdfs[area]           = {}

    baseline_sdf = all_sdf_dict[region_key][baseline_raw_event][baseline_mask]
    n_neurons    = len(all_spikes[area])

    event_modulations = []
    event_tVals       = []
    event_max_slopes  = []
    event_sdfs_out    = []

    for raw_event in events:
        event_label    = event_map[raw_event]
        event_epoch_ms = epoch_window_map[event_label]['window']

        event_sdf    = all_sdf_dict[region_key][raw_event][event_masks[event_label]]
        slope_sdf    = all_sdf_dict[region_key][raw_event][slope_masks[event_label]]
        event_slopes = slope_sdf[1:] - slope_sdf[:-1]

        event_max_slopes.append({
            'max':      (event_slopes.max(axis=0),  event_slopes.argmax(axis=0)),
            'min':      (event_slopes.min(axis=0),  event_slopes.argmin(axis=0)),
            'slopes':   event_slopes,
            'sdf':      event_sdf,
            'baseline': baseline_sdf,
        })

        t_stats = np.full(n_neurons, np.nan)
        p_vals  = np.full(n_neurons, np.nan)

        for neuron_idx in range(n_neurons):
            lat_labels_n    = all_conditions[area][neuron_idx][:, 0]
            spikes_by_trial = all_spikes[area][neuron_idx]
            baseline_times  = np.array(all_events[area][baseline_event_label][neuron_idx])
            event_times_n   = np.array(all_events[area][event_label][neuron_idx])

            baseline_rates = epoch_trial_rates(
                spikes_by_trial, baseline_times, baseline_epoch_ms, lat_labels_n, lat)
            event_rates    = epoch_trial_rates(
                spikes_by_trial, event_times_n, event_epoch_ms, lat_labels_n, lat)

            n_b, n_e = len(baseline_rates), len(event_rates)
            if n_b == n_e and n_b > 1:
                t_stats[neuron_idx], p_vals[neuron_idx] = ttest_rel(event_rates, baseline_rates)
            elif n_b > 1 and n_e > 1:
                t_stats[neuron_idx], p_vals[neuron_idx] = ttest_ind(
                    event_rates, baseline_rates, equal_var=False)

        valid  = ~np.isnan(p_vals)
        reject = np.zeros(n_neurons, dtype=bool)
        if valid.sum() > 1:
            reject[valid], _, _, _ = multipletests(
                p_vals[valid], alpha=p_score, method='fdr_bh')

        event_modulations.append(reject)
        event_tVals.append(t_stats)
        event_sdfs_out.append(event_sdf)

    event_modulations = np.vstack(event_modulations)
    event_tVals       = np.vstack(event_tVals)

    modulation_dict[area][lat_label] = {
        'modulations':  event_modulations,
        'tVals':        event_tVals,
        'event_slopes': event_max_slopes,
        'event_sdfs':   event_sdfs_out,
    }
    up_down_modulation[area][lat_label] = np.stack([
        event_modulations * event_tVals < 0,
        event_modulations * event_tVals > 0,
    ])
    tVals_dict[area][lat_label] = event_tVals
    mod_sdfs[area][lat_label]   = event_sdfs_out

# ---------------------------------------------------------------------------
# Build merged_mods — flat view keyed by area > lat_label > event_label
# ---------------------------------------------------------------------------
merged_mods = {}
for area in modulation_dict.keys():
    merged_mods[area] = {}
    for lat_label in modulation_dict[area].keys():
        slopes = modulation_dict[area][lat_label]['event_slopes']
        merged_mods[area][lat_label] = {
            event_map[raw_event]: {
                'modulations': modulation_dict[area][lat_label]['modulations'][i],
                'tVals':       modulation_dict[area][lat_label]['tVals'][i],
                'event_sdfs':  modulation_dict[area][lat_label]['event_sdfs'][i],
                'base_sdfs':   slopes[i]['baseline'],
            }
            for i, raw_event in enumerate(events)
        }

# ---------------------------------------------------------------------------
# Laterality-preference subtype classification (merged_mod_types)
# ---------------------------------------------------------------------------
merged_mod_types = {}
for area in merged_mods.keys():
    merged_mod_types[area] = {}
    lats          = list(merged_mods[area].keys())
    area_mod_dict = merged_mods[area]
    event_mod_dict = {}

    for l_idx, lat in enumerate(lats):
        merged_mod_types[area][lat] = {}
        rev_lat_map    = {v: k for k, v in lat_map.items()}
        lat_char       = rev_lat_map[lat]
        other_lat_char = rev_lat_map[lats[1 - l_idx]]

        for event in merged_mods[area][lat].keys():
            event_mods  = area_mod_dict[lat][event]['modulations']
            both_mods   = area_mod_dict[lats[1 - l_idx]][event]['modulations'] * event_mods
            event_tVals = area_mod_dict[lat][event]['tVals']
            other_tVals = area_mod_dict[lats[1 - l_idx]][event]['tVals']
            same_sign   = np.equal(np.sign(event_tVals), np.sign(other_tVals))
            up_down     = np.stack([event_mods * event_tVals < 0, event_mods * event_tVals > 0])
            event_epoch_ms = epoch_window_map[event]['window']

            mod_vals = np.zeros_like(up_down[0]).astype(bool)
            for m_idx, mod in enumerate(['down', 'up']):
                mod_mask       = up_down[m_idx]
                masked_indices = np.where(mod_mask)[0]
                n_masked       = len(masked_indices)

                t_stats_m = np.full(n_masked, np.nan)
                p_vals_m  = np.full(n_masked, np.nan)

                for i, neuron_idx in enumerate(masked_indices):
                    lat_labels_n    = all_conditions[area][neuron_idx][:, 0]
                    spikes_by_trial = all_spikes[area][neuron_idx]
                    event_times_n   = np.array(all_events[area][event][neuron_idx])

                    lat_rates       = epoch_trial_rates(
                        spikes_by_trial, event_times_n, event_epoch_ms, lat_labels_n, lat_char)
                    other_lat_rates = epoch_trial_rates(
                        spikes_by_trial, event_times_n, event_epoch_ms, lat_labels_n, other_lat_char)

                    n_l, n_o = len(lat_rates), len(other_lat_rates)
                    if n_l == n_o and n_l > 1:
                        t_stats_m[i], p_vals_m[i] = ttest_rel(lat_rates, other_lat_rates)
                    elif n_l > 1 and n_o > 1:
                        t_stats_m[i], p_vals_m[i] = ttest_ind(
                            lat_rates, other_lat_rates, equal_var=False)

                valid  = ~np.isnan(p_vals_m)
                reject = np.zeros(n_masked, dtype=bool)
                if valid.sum() > 1:
                    reject[valid], _, _, _ = multipletests(
                        p_vals_m[valid], alpha=p_score, method='fdr_bh')
                # Require t > 0: for the ipsilateral pass (l_idx=0) this means
                # ipsi_rates > contra_rates (ipsi-preferring); for the contralateral
                # pass (l_idx=1) it means contra_rates > ipsi_rates (contra-preferring).
                # Without this guard the two passes produce identical reject masks
                # (p-values are symmetric) so both:ipsi and both:contra select the
                # same neurons, doubling their count in the pie chart.
                mod_vals[mod_mask] = reject & (t_stats_m > 0)

            merged_mod_types[area][lat][event] = mod_vals * both_mods * same_sign
            if l_idx == 0:
                event_mod_dict[event] = {'both': both_mods, 'sign': same_sign}

    merged_mod_types[area]['opp'] = {}
    for event in merged_mods[area][lat].keys():
        both_mods = event_mod_dict[event]['both']
        same_sign = event_mod_dict[event]['sign']
        merged_mod_types[area]['opp'][event] = both_mods * (~same_sign)

# ---------------------------------------------------------------------------
# Build MergedModDataFrame and hand_use_mod
# ---------------------------------------------------------------------------
hand_use_mod = {}
all_mod_dict = {
    'tVal': [], 'n_idc': [], 'region': [], 'event': [],
    'side': [], 'mod_type': [], 'event_mean': [], 'baseline_mean': [],
}

for region in merged_mods.keys():
    n_neurons = len(all_spikes[region])
    hand_use_mod[region] = {}

    for event in ['Cue', 'Grasp']:
        hand_mod_list  = []
        mod_tVals      = []
        sdf_means      = []
        baseline_means = []

        for side, mod_vals in merged_mods[region].items():
            hand_mod_list.append(mod_vals[event]['modulations'])
            mod_tVals.append(mod_vals[event]['tVals'])
            # NOTE: event_mean / baseline_mean are SDF-derived (mean over time bins).
            # Replace with per-trial firing rate means if epoch_trial_rates output is stored.
            sdf_means.append(mod_vals[event]['event_sdfs'].mean(axis=0))
            baseline_means.append(mod_vals[event]['base_sdfs'].mean(axis=0))

        neuron_idcs      = np.arange(n_neurons)
        ipsi_mod         = hand_mod_list[0]
        contra_mod       = hand_mod_list[1]
        hand_nonSpecific = ~np.bitwise_xor(ipsi_mod, contra_mod)
        both_mod         = ipsi_mod * contra_mod
        equal_mod        = ipsi_mod * contra_mod
        none_mod         = ~ipsi_mod * ~contra_mod
        ipsi_only        = ipsi_mod * ~hand_nonSpecific
        contra_only      = contra_mod * ~hand_nonSpecific

        hand_use_mod[region][event] = {'ipsi': ipsi_only, 'contra': contra_only}

        for lat in ['ipsilateral', 'contralateral', 'opp']:
            hand_mod   = merged_mod_types[region][lat][event]
            key_string = {'ipsilateral': 'ipsi', 'contralateral': 'contra', 'opp': 'opp'}[lat]
            hand_use_mod[region][event][f'both:{key_string}'] = hand_mod
            equal_mod[hand_mod] = False
            if lat == 'ipsilateral':
                ipsi_more    = hand_mod
            elif lat == 'contralateral':
                contra_more  = hand_mod
            elif lat == 'opp':
                opposite_mod = hand_mod

        hand_use_mod[region][event]['both:equal'] = equal_mod
        hand_use_mod[region][event]['both:all']   = both_mod

        all_mods = {
            'ipsi':        ipsi_only,
            'contra':      contra_only,
            'both:equal':  equal_mod,
            'both:opp':    opposite_mod,
            'both:ipsi':   ipsi_more,
            'both:contra': contra_more,
            'none':        none_mod,
        }

        for side_idx, (side, mod_vals) in enumerate(merged_mods[region].items()):
            hand_tVals         = mod_tVals[side_idx]
            baseline_mean_list = baseline_means[side_idx]
            sdf_mean_list      = sdf_means[side_idx]
            for mod_type, mod_mask in all_mods.items():
                hand_tVal     = hand_tVals[mod_mask]
                mod_nIdcs     = neuron_idcs[mod_mask]
                sdf_mean      = sdf_mean_list[mod_mask]
                baseline_mean = baseline_mean_list[mod_mask]
                n_mods        = mod_mask.sum()
                all_mod_dict['side']          += [side]     * n_mods
                all_mod_dict['tVal']          += list(hand_tVal)
                all_mod_dict['n_idc']         += list(mod_nIdcs)
                all_mod_dict['event_mean']    += list(sdf_mean)
                all_mod_dict['baseline_mean'] += list(baseline_mean)
                all_mod_dict['region']        += [region]   * n_mods
                all_mod_dict['mod_type']      += [mod_type] * n_mods
                all_mod_dict['event']         += [event]    * n_mods

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
merged_mod_df = pd.DataFrame(all_mod_dict)
merged_mod_df.to_pickle(merged_df_filename)

with open(hand_use_mod_filename, 'wb') as f:
    pkl.dump(hand_use_mod, f)

with open(merged_mods_filename, 'wb') as f:
    pkl.dump(merged_mods, f)

print('MergedModDataFrame, HandUseMod, and MergedMods saved')
