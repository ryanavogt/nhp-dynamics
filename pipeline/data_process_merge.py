import pickle as pkl                # Saving/loading data (built-in)
import os                           # Directory creation and verification (built-in)
import sys
import glob

# Allow imports from the parent directory when this script is run from a subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# The following packages need to be installed in your virtual environment (using conda or pip)
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.sig_proc import gen_psth, gen_sdf, trial_psth

from config import (
    MONKEY_NAME_MAP         as monkey_name_map,
    EVENT_MAP_MERGE         as event_map,
    EPOCH_WINDOW_MAP_MERGE  as epoch_window_map,
    EPOCHS_MERGE            as epochs,
    EVENTS_MERGE            as events,
    SORTING_DIR             as sorting_dir,
    POPULATION_DIR,
    PROCESSED_DIR           as processed_dir,
    LAT_MAP                 as lat_map,
    HAND_LIST               as hand_list,
    BIN_SIZE                as binsize,
    KERNEL_WIDTH            as kernel_width,
    WINDOW_RANGE,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

POPULATION_DATA = os.path.join(POPULATION_DIR, 'data')
POPULATION_FIGS = os.path.join(POPULATION_DIR, 'figures')
for _d in [POPULATION_DATA, POPULATION_FIGS]:
    os.makedirs(_d, exist_ok=True)

trial_count              = 100
load_override_preprocess = False
load_override            = False

# ---------------------------------------------------------------------------
# Extract session list from sorting notes
# ---------------------------------------------------------------------------
file_list = []
for m in monkey_name_map.keys():
    file_list += [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*{m}.xlsx')]
file_names       = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]
file_names_split = [n.split('_') for n in file_names]

date_strings  = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])
    monkey_labels.append(name[2])

# ---------------------------------------------------------------------------
# BLOCK A: Spike / event / condition aggregation
# Outputs: spike_summary_merged{n}.p, spike_list_merged{n}.p,
#          event_list_merged{n}.p, condition_list_merged{n}.p,
#          durs_list_merged{n}.p, monkey_indices_merged{n}.p
# ---------------------------------------------------------------------------
area_summary_dict  = {}
all_areas          = []
all_spikes         = {}
all_conditions     = {}
all_durs           = {}
all_monkey_indices = {'M1': {'count': 0}, 'PMd': {'count': 0}, 'PMv': {'count': 0}}
all_events         = {
    'M1':  {'Cue': [], 'Grasp': []},
    'PMd': {'Cue': [], 'Grasp': []},
    'PMv': {'Cue': [], 'Grasp': []},
}

n_monkeys           = len(monkey_name_map.keys())
summary_file_name   = f'{POPULATION_DATA}/spike_summary_merged{n_monkeys}.p'
spike_file_name     = f'{POPULATION_DATA}/spike_list_merged{n_monkeys}.p'
event_file_name     = f'{POPULATION_DATA}/event_list_merged{n_monkeys}.p'
condition_file_name = f'{POPULATION_DATA}/condition_list_merged{n_monkeys}.p'
durs_file_name      = f'{POPULATION_DATA}/durs_list_merged{n_monkeys}.p'
monkey_file_name    = f'{POPULATION_DATA}/monkey_indices_merged{n_monkeys}.p'

if os.path.exists(summary_file_name) and not load_override_preprocess:
    with open(summary_file_name, 'rb') as f:
        area_summary_dict = pkl.load(f)
    with open(spike_file_name, 'rb') as f:
        all_spikes = pkl.load(f)
    with open(condition_file_name, 'rb') as f:
        all_conditions = pkl.load(f)
    with open(monkey_file_name, 'rb') as f:
        all_monkey_indices = pkl.load(f)
    print('Spike File Loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        print(date)
        monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
        date_folder   = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
        trial_dir     = os.path.join(processed_dir, monkey_folder, date_folder)
        with open(f'{trial_dir}/trial_data.p', 'rb') as f:
            trial_data = pkl.load(f)
        with open(f'{trial_dir}/eventMasks.p', 'rb') as f:
            full_mask = pkl.load(f)
        if monkey in ['R', 'G']:
            with open(f'{trial_dir}/eventDurs.p', 'rb') as f:
                eventDurs = pkl.load(f)
            for event in eventDurs.keys():
                if event not in all_durs:
                    all_durs[event] = [eventDurs[event]]
                else:
                    all_durs[event].append(eventDurs[event])
        area_list = [f for f in glob.glob(f'{trial_dir}/spikeTimes_*')]
        for area in area_list:
            area_name = area.split('_')[-1].split('.')[0]
            if area_name == 'S1':
                continue
            with open(area, 'rb') as f:
                spike_times = pkl.load(f)
            area_label, area_hemisphere = area_name[:-1], area_name[-1]
            area_indices      = all_monkey_indices[area_label]
            if monkey not in area_indices.keys():
                area_indices[monkey] = []
            area_neuron_index = area_indices['count']
            if area_label not in all_areas:
                all_areas.append(area_label)
                region_spike_list     = []
                region_condition_list = []
            else:
                region_spike_list     = all_spikes[area_label]
                region_condition_list = all_conditions[area_label]
            for channel in spike_times:
                channel_spikes = spike_times[channel]
                if channel_spikes.shape[0] > 0:
                    channel_neurons    = channel_spikes[:, 0].max()
                    channel_conditions = np.empty(
                        (len(trial_data['handOrien']), 2), dtype='str_')
                    for neuron in range(1, channel_neurons + 1):
                        neuron_spike_list = []
                        neuron_spikes     = channel_spikes[channel_spikes[:, 0] == neuron]
                        for trial in range(1, channel_conditions.shape[0] + 1):
                            neuron_spike_list.append(
                                neuron_spikes[neuron_spikes[:, -1] == trial, 1])
                        region_spike_list.append(neuron_spike_list)
                        for event in events:
                            all_events[area_label][event_map[event]].append(
                                list(trial_data[event]))
                    for o_idx, orient in enumerate(['horizontal', 'vertical']):
                        orientation_mask = (trial_data['handOrien'] - 1) // 2 == o_idx
                        for mod in [0, 1]:
                            hand_mask  = trial_data['handOrien'] % 2 == mod
                            spike_mask = np.isin(
                                channel_spikes[:, -1] - 1,
                                np.where(hand_mask * orientation_mask * full_mask))
                            hand_label    = hand_list[mod]
                            lateral_label = 'i' if area_hemisphere == hand_label else 'c'
                            channel_conditions[hand_mask * orientation_mask] = np.array(
                                [lateral_label, orient[0]])
                            region_key = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in area_summary_dict:
                                area_summary_dict[region_key] = {}
                            for event in events:
                                if event not in area_summary_dict[region_key]:
                                    area_summary_dict[region_key][event] = {
                                        'spikes': [], 'neurons': 0, 'trial_idcs': []}
                                event_times  = trial_data[event][
                                    channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([
                                    channel_spikes[spike_mask, 0]
                                    + area_summary_dict[region_key][event]['neurons'],
                                    channel_spikes[spike_mask, 1]
                                    - event_times[spike_mask].astype(float),
                                ])
                                area_summary_dict[region_key][event]['spikes'].append(
                                    event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += (
                                    channel_neurons)
                                area_summary_dict[region_key][event]['trial_idcs'].append(
                                    channel_spikes[spike_mask, -1])
                    for neuron in range(channel_neurons):
                        region_condition_list.append(channel_conditions)
                        area_indices[monkey].append(area_neuron_index)
                        area_neuron_index += 1
            all_monkey_indices[area_label]['count'] = area_neuron_index
            all_spikes[area_label]     = region_spike_list
            all_conditions[area_label] = region_condition_list
    with open(summary_file_name, 'wb') as f:
        pkl.dump(area_summary_dict, f)
    with open(spike_file_name, 'wb') as f:
        pkl.dump(all_spikes, f)
    with open(event_file_name, 'wb') as f:
        pkl.dump(all_events, f)
    with open(condition_file_name, 'wb') as f:
        pkl.dump(all_conditions, f)
    with open(durs_file_name, 'wb') as f:
        for event in all_durs.keys():
            all_durs[event] = np.hstack(all_durs[event])
        pkl.dump(all_durs, f)
    with open(monkey_file_name, 'wb') as f:
        pkl.dump(all_monkey_indices, f)
    print('Spike File Saved')

# ---------------------------------------------------------------------------
# Neuron count table — CSV only (figure lives in plotting.py)
# ---------------------------------------------------------------------------
regions = list(all_monkey_indices.keys())
monkeys = list(monkey_name_map.keys())
region_neurons = {}
for region in regions:
    region_neurons[region] = {}
    for monkey in monkeys:
        region_neurons[region][monkey] = len(all_monkey_indices[region][monkey])
neuron_df = pd.DataFrame.from_dict(region_neurons, orient='index')
neuron_df.to_csv(os.path.join(POPULATION_DATA, 'monkeyNeurons.csv'))

# ---------------------------------------------------------------------------
# BLOCK B: SDF / PSTH computation (epoch-windowed, used for scores and ANOVA)
# Outputs: merged_sdfDict_bin{b}_k{k}_merged{n}.p,
#          areaMeanRates_...p, trialPSTH_...p
# ---------------------------------------------------------------------------
area_mean_rate  = {}
event_mean_rate = {}
all_sdf_filename   = (f'{POPULATION_DATA}/merged_sdfDict_bin{binsize}_k{kernel_width}'
                      f'_merged{n_monkeys}.p')
area_mean_filename = (f'{POPULATION_DATA}/areaMeanRates_bin{binsize}_k{kernel_width}'
                      f'_merged{n_monkeys}.p')
all_psth_filename  = (f'{POPULATION_DATA}/trialPSTH_bin{binsize}_k{kernel_width}'
                      f'_merged{n_monkeys}.p')

if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as f:
        all_sdf_dict = pkl.load(f)
    with open(area_mean_filename, 'rb') as f:
        area_mean_rate, event_mean_rate = pkl.load(f)
    with open(all_psth_filename, 'rb') as f:
        all_psth_dict = pkl.load(f)
    print(f'SDF Dictionary Loaded (Bin: {binsize}, Kernel: {kernel_width})')
else:
    all_sdf_dict  = {}
    all_psth_dict = {}
    for area_label in area_summary_dict.keys():
        region_key      = area_label
        latArea, orient = region_key.split('_')
        lat, region     = latArea[0], latArea[1:]
        print(region_key)
        lat = region_key[0]
        all_psth       = []
        all_trial_psth = []
        for event in events:
            window_range = np.array(epoch_window_map[event_map[event]]['window'])
            num_bins     = int(1.0 * (window_range.max() - window_range.min()) / binsize)
            if region_key in area_summary_dict:
                area_summary_dict[region_key][event]['spikes'] = np.concatenate(
                    area_summary_dict[region_key][event]['spikes'], axis=1)
                area_summary_dict[region_key][event]['trial_idcs'] = np.concatenate(
                    area_summary_dict[region_key][event]['trial_idcs'])
            neuron_count    = area_summary_dict[region_key][event]['neurons'].astype(int)
            psth_list       = np.zeros((num_bins, neuron_count + 1))
            trial_psth_list = []
            for neuron_idx in range(neuron_count):
                neuron_mask   = (area_summary_dict[region_key][event]['spikes'][0, :]
                                 == neuron_idx + 1)
                neuron_spikes = area_summary_dict[region_key][event]['spikes'][
                    :, neuron_mask]
                if neuron_spikes.shape[1] == 0:
                    print(f'No spikes for neuron {neuron_idx + 1}')
                neuron_trials     = area_summary_dict[region_key][event][
                    'trial_idcs'][neuron_mask]
                conds             = all_conditions[region][neuron_idx]
                neuron_trial_list = np.where(
                    (conds[:, 0] == lat) & (conds[:, 1] == orient[0]))[0]
                neuron_spikes[0, :] = 1
                psth_trials = trial_psth(
                    neuron_spikes.T, neuron_trials, binsize=binsize,
                    window=window_range, neurons=1, trial_nos=neuron_trial_list)
                psth_trials = np.stack(psth_trials, axis=-1)
                neuron_psth = gen_psth(
                    neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
                if neuron_idx == 0:
                    psth_list[:, 0] = neuron_psth[1:, 0]
                psth_list[:, neuron_idx + 1] = neuron_psth[1:, 1]
                trial_psth_list.append(psth_trials)
            all_psth.append(psth_list)
            if region_key not in event_mean_rate.keys():
                event_mean_rate[region_key] = {}
            event_mean_rate[region_key][event] = psth_list[:, 1:].mean(axis=0)
            all_trial_psth.append(trial_psth_list)
        all_psth = np.vstack(all_psth)
        area_sdf, _ = gen_sdf(all_psth[:, 1:], w=kernel_width, bin_size=binsize,
                               ftype='Gauss', multi_unit=True)
        all_sdf_dict[region_key]   = area_sdf.T
        all_psth_dict[region_key]  = all_trial_psth
        area_mean_rate[region_key] = all_psth[:, 1:].mean(axis=0)
    with open(all_sdf_filename, 'wb') as f:
        pkl.dump(all_sdf_dict, f)
    with open(area_mean_filename, 'wb') as f:
        pkl.dump((area_mean_rate, event_mean_rate), f)
    with open(all_psth_filename, 'wb') as f:
        pkl.dump(all_psth_dict, f)

# ---------------------------------------------------------------------------
# BLOCK C: OrientMerged spike aggregation (no full_mask filter; no -1 offset)
# Output: spike_summary.p
# ---------------------------------------------------------------------------
orient_area_dict  = {}
orient_areas      = []
orient_spike_name = f'{POPULATION_DATA}/spike_summary.p'

if os.path.exists(orient_spike_name) and not load_override_preprocess:
    with open(orient_spike_name, 'rb') as f:
        orient_area_dict = pkl.load(f)
    print('OrientMerged spike summary loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        print(date)
        monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
        date_folder   = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
        trial_dir     = os.path.join(processed_dir, monkey_folder, date_folder)
        with open(f'{trial_dir}/trial_data.p', 'rb') as f:
            trial_data = pkl.load(f)
        area_list = glob.glob(f'{trial_dir}/spikeTimes_*')
        for area in area_list:
            area_name = area.split('_')[-1].split('.')[0]
            if area_name == 'S1':
                continue
            with open(area, 'rb') as f:
                spike_times = pkl.load(f)
            area_label, area_hemisphere = area_name[:-1], area_name[-1]
            if area_label not in orient_areas:
                orient_areas.append(area_label)
            for channel in spike_times:
                channel_spikes = spike_times[channel]
                if channel_spikes.shape[0] > 0:
                    channel_neurons = channel_spikes[:, 0].max()
                    for o_idx, orient in enumerate(['horizontal', 'vertical']):
                        orientation_mask = (trial_data['handOrien'] - 1) // 2 == o_idx
                        for mod in [0, 1]:
                            hand_mask = trial_data['handOrien'] % 2 == mod
                            # No full_mask filter; trial indices are not offset by -1
                            spike_mask    = np.isin(channel_spikes[:, -1],
                                                    np.where(hand_mask * orientation_mask))
                            hand_label    = hand_list[mod]
                            lateral_label = 'i' if area_hemisphere == hand_label else 'c'
                            region_key    = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in orient_area_dict:
                                orient_area_dict[region_key] = {}
                            for event in events:
                                if event not in orient_area_dict[region_key]:
                                    orient_area_dict[region_key][event] = {
                                        'spikes': [], 'neurons': 0, 'event_times': []}
                                event_times  = trial_data[event][
                                    channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([
                                    channel_spikes[spike_mask, 0]
                                    + orient_area_dict[region_key][event]['neurons'],
                                    channel_spikes[spike_mask, 1]
                                    - event_times[spike_mask].astype(float),
                                ])
                                orient_area_dict[region_key][event]['spikes'].append(
                                    event_spikes)
                                orient_area_dict[region_key][event]['neurons'] += (
                                    channel_neurons)
                                orient_area_dict[region_key][event]['event_times'].append(
                                    event_times)
    with open(orient_spike_name, 'wb') as f:
        pkl.dump(orient_area_dict, f)
    print('OrientMerged spike summary saved')

# ---------------------------------------------------------------------------
# Merge orientations into merged_summary_dict
# Output: mergedSpikeDict.p
# ---------------------------------------------------------------------------
merged_summary_dict  = {}
merged_dict_filename = f'{POPULATION_DATA}/mergedSpikeDict.p'

for region_key in orient_area_dict.keys():
    latArea, orient = region_key.split('_')
    if latArea not in merged_summary_dict:
        merged_summary_dict[latArea] = {}
    for event in orient_area_dict[region_key].keys():
        if event not in merged_summary_dict[latArea]:
            merged_summary_dict[latArea][event] = {'spikes': []}
        merged_summary_dict[latArea][event]['spikes'].append(
            np.hstack(orient_area_dict[region_key][event]['spikes']))

for latArea in merged_summary_dict.keys():
    for event in merged_summary_dict[latArea].keys():
        merged_summary_dict[latArea][event]['spikes'] = np.hstack(
            merged_summary_dict[latArea][event]['spikes'])
        merged_summary_dict[latArea][event]['neurons'] = int(
            np.max(merged_summary_dict[latArea][event]['spikes'][0]))

with open(merged_dict_filename, 'wb') as f:
    pkl.dump(merged_summary_dict, f)

# ---------------------------------------------------------------------------
# BLOCK D: OrientMerged SDF — per-neuron, full WINDOW_RANGE
# Outputs: merged_sdfDict_bin{b}_k{k}_orientMerged.p,
#          sdfMaxRate_...p, peakOrderDict_...p
# ---------------------------------------------------------------------------
window_range            = np.array(WINDOW_RANGE)
area_max_rate           = {}
all_orient_sdf_filename = (f'{POPULATION_DATA}/merged_sdfDict_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')
max_rate_filename       = (f'{POPULATION_DATA}/sdfMaxRate_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')
peak_order_filename     = (f'{POPULATION_DATA}/peakOrderDict_bin{binsize}_k{kernel_width}'
                           f'_orientMerged.p')

if os.path.exists(all_orient_sdf_filename) and not load_override:
    with open(all_orient_sdf_filename, 'rb') as f:
        all_orient_sdf_dict = pkl.load(f)
    with open(max_rate_filename, 'rb') as f:
        area_max_rate = pkl.load(f)
    with open(peak_order_filename, 'rb') as f:
        peak_order_dict = pkl.load(f)
    print(f'OrientMerged SDF loaded (bin={binsize}, kernel={kernel_width})')
else:
    all_orient_sdf_dict = {}
    peak_order_dict     = {}
    for latArea in merged_summary_dict.keys():
        peak_order_dict[latArea]  = {}
        area_max_rate[latArea]    = np.zeros(
            merged_summary_dict[latArea][events[0]]['neurons'])
        print(latArea)
        all_orient_sdf_dict[latArea] = {}
        for event in events:
            sdf_list          = []
            neuron_count      = merged_summary_dict[latArea][event]['neurons']
            neuron_peak_times = np.zeros(neuron_count)
            for neuron_idx in range(neuron_count):
                neuron_mask   = (merged_summary_dict[latArea][event]['spikes'][0, :]
                                 == neuron_idx + 1)
                neuron_spikes = merged_summary_dict[latArea][event]['spikes'][
                    :, neuron_mask].copy()
                neuron_spikes[0, :] = 1
                neuron_psth   = gen_psth(neuron_spikes.T, binsize=binsize,
                                         window=window_range, neurons=1)
                neuron_sdf, _ = gen_sdf(neuron_psth, w=kernel_width, bin_size=binsize,
                                        ftype='Gauss', multi_unit=False)
                neuron_peak_times[neuron_idx] = neuron_sdf[:].argmax()
                sdf_list.append(neuron_sdf[:, 0].T)
            peak_order_dict[latArea][event] = np.argsort(neuron_peak_times)
            area_sdf = np.hstack(sdf_list)
            all_orient_sdf_dict[latArea][event] = area_sdf
            event_max_rate = np.max(area_sdf, axis=0)
            area_max_rate[latArea] = np.maximum(event_max_rate, area_max_rate[latArea])
    with open(all_orient_sdf_filename, 'wb') as f:
        pkl.dump(all_orient_sdf_dict, f)
    with open(max_rate_filename, 'wb') as f:
        pkl.dump(area_max_rate, f)
    with open(peak_order_filename, 'wb') as f:
        pkl.dump(peak_order_dict, f)
    print('OrientMerged SDF saved')

# ---------------------------------------------------------------------------
# Neuron peaks — identify epoch of maximal firing per neuron
# Output: neuron_peaks.p
# ---------------------------------------------------------------------------
neuron_peaks_filename = f'{POPULATION_DATA}/neuron_peaks.p'
full_window   = np.arange(WINDOW_RANGE[0], WINDOW_RANGE[1] + binsize, binsize)
active_epochs = [ep for ep in epochs if ep != 'Pre-Cue']

area_merged_rates = {region: None for region in regions}
for area_label in area_max_rate.keys():
    lat, region = area_label[0], area_label[1:]
    area_max    = area_max_rate[area_label]
    if area_merged_rates[region] is None:
        area_merged_rates[region] = np.zeros_like(area_max)
    area_merged_rates[region] = np.max(
        np.stack([area_max, area_merged_rates[region]]), axis=0)

event_spike_maxes = {}
event_neuron_max  = {}
for area_label in merged_summary_dict.keys():
    lat, region = area_label[0], area_label[1:]
    area_max    = area_max_rate[area_label]
    event_spike_maxes[area_label] = {}
    if region not in event_neuron_max:
        event_neuron_max[region] = {'proportions': {}, 'neurons': 0}
    for epoch in active_epochs:
        event        = epoch_window_map[epoch]['event']
        event_window = epoch_window_map[epoch]['window']
        event_mask   = ((full_window > event_window[0]) *
                        (full_window <= event_window[1]))
        area_sdf     = all_orient_sdf_dict[area_label][event][event_mask]
        event_spike_maxes[area_label][epoch] = area_sdf.max(axis=0)
    event_neuron_peaks     = np.vstack(list(event_spike_maxes[area_label].values()))
    event_peaks            = (event_neuron_peaks == area_merged_rates[region])
    event_peak_proportions = np.average(event_peaks, axis=1)
    event_neuron_max[region]['neurons']                   = event_neuron_peaks.shape[1]
    event_neuron_max[region]['proportions'][lat_map[lat]] = event_peak_proportions

with open(neuron_peaks_filename, 'wb') as f:
    pkl.dump(event_neuron_max, f)
print('Neuron peaks saved')

# ---------------------------------------------------------------------------
# BLOCK E: Heatmap similarity — two complementary Pearson r metrics
#
# Metric 1 — Matrix r:
#   Correlates the full normalised SDF matrices (contra vs. ipsi), flattened.
#   Captures overall activity similarity but is inflated by the shared
#   low-firing baseline (most bins are near zero for both sides).
#
# Metric 2 — Peak-timing r:
#   Correlates only the per-neuron peak bin index between contra and ipsi.
#   Isolates sequential structure: does the order of who fires when
#   map across sides? Unaffected by amplitude or baseline.
#
# Both use the full WINDOW_RANGE SDF (not epoch-windowed) so peak timing
# reflects the neuron's true preferred moment across the whole trial.
#
# Output: heatmapCorr_bin{b}_k{k}_orientMerged.p
#   {region: {event: {'r_matrix':      float,
#                     'p_matrix':      float,
#                     'r_peak_timing': float,
#                     'p_peak_timing': float,
#                     'n_neurons':     int   }}}
# ---------------------------------------------------------------------------
grasp_event           = epoch_window_map['Grasp']['event']
heatmap_corr          = {}
heatmap_corr_filename = (f'{POPULATION_DATA}/heatmapCorr_bin{binsize}_k{kernel_width}'
                         f'_orientMerged.p')

for region in regions:
    heatmap_corr[region] = {}
    contra_key = f'c{region}'
    ipsi_key   = f'i{region}'

    # Per-neuron max rate from the contralateral side — shape (n_neurons,)
    max_rate   = area_max_rate[contra_key]
    # Neuron order sorted by contralateral Grasp peak time (ascending)
    peak_order = peak_order_dict[contra_key][grasp_event]

    for event in events:
        # all_orient_sdf_dict[latArea][event] has shape (n_bins, n_neurons)
        # Transpose → (n_neurons, n_bins) to match the heatmap row convention
        contra_sdf = all_orient_sdf_dict[contra_key][event].T   # (n_neurons, n_bins)
        ipsi_sdf   = all_orient_sdf_dict[ipsi_key][event].T

        # ── Metric 1: matrix r ───────────────────────────────────────────────
        # Normalise both sides by contralateral max rate (per neuron)
        denom = max_rate[:, np.newaxis]                          # (n_neurons, 1)
        contra_norm = np.divide(contra_sdf, denom,
                                where=denom > 0,
                                out=np.zeros_like(contra_sdf))
        ipsi_norm   = np.divide(ipsi_sdf,   denom,
                                where=denom > 0,
                                out=np.zeros_like(ipsi_sdf))

        # Sort rows by contralateral Grasp peak order (same ordering as heatmap)
        contra_sorted = contra_norm[peak_order]
        ipsi_sorted   = ipsi_norm[peak_order]

        r_mat, p_mat = sp_stats.pearsonr(contra_sorted.ravel(), ipsi_sorted.ravel())

        # ── Metric 2: peak-timing r ──────────────────────────────────────────
        # For each neuron, find the time bin of peak firing on each side.
        # argmax over axis=1 (bins), giving shape (n_neurons,).
        # This strips out amplitude and baseline — pure sequence comparison.
        contra_peak_bins = np.argmax(contra_sdf, axis=1)   # (n_neurons,)
        ipsi_peak_bins   = np.argmax(ipsi_sdf,   axis=1)

        r_timing, p_timing = sp_stats.pearsonr(contra_peak_bins, ipsi_peak_bins)

        heatmap_corr[region][event] = {
            'r_matrix':      r_mat,
            'p_matrix':      p_mat,
            'r_peak_timing': r_timing,
            'p_peak_timing': p_timing,
            'n_neurons':     len(peak_order),
        }
        print(f'  {region} | {event}:  '
              f'matrix r = {r_mat:.3f}  |  '
              f'peak-timing r = {r_timing:.3f}, p = {p_timing:.3e}')

with open(heatmap_corr_filename, 'wb') as f:
    pkl.dump(heatmap_corr, f)
print(f'Heatmap correlation saved → {heatmap_corr_filename}')