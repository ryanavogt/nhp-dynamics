import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os
import glob

import numpy as np

from config import (
    SUMMARY_DIR,
    SORTING_DIR,
    MONKEY_NAME_MAP      as monkey_name_map,
    EVENT_MAP_MERGE      as event_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as active_epochs,
    LAT_MAP              as lat_map,
    HAND_LIST            as hand_list,
    EVENTS_MERGE         as events,
    BIN_SIZE             as binsize,
    KERNEL_WIDTH         as kernel_width,
    WINDOW_RANGE,
)
from utils.sig_proc import gen_psth, gen_sdf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
summary_dir  = SUMMARY_DIR
window_range = np.array(WINDOW_RANGE)

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

load_override_preprocess = False
load_override            = False

# ---------------------------------------------------------------------------
# Extract session list from sorting notes
# ---------------------------------------------------------------------------
file_list        = [f for f in glob.glob(f'{SORTING_DIR}/SortingNotes_*.xlsx')]
file_names       = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]
file_names_split = [n.split('_') for n in file_names]

date_strings  = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])
    monkey_labels.append(name[2])

# ---------------------------------------------------------------------------
# Build area_summary_dict — one pass over raw session files
# ---------------------------------------------------------------------------
area_summary_dict = {}
all_areas         = []

spikes_file_name = f'{summary_dir}/spike_summary.p'
if os.path.exists(spikes_file_name) and not load_override_preprocess:
    with open(spikes_file_name, 'rb') as spike_file:
        area_summary_dict = pkl.load(spike_file)
    print('Spike summary loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        print(date)
        monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
        date_folder   = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
        trial_dir     = f'Data/Processed/{monkey_folder}/{date_folder}'
        with open(f'{trial_dir}/trial_data.p', 'rb') as trial_file:
            trial_data = pkl.load(trial_file)
        area_list = glob.glob(f'{trial_dir}/spikeTimes_*')
        for area in area_list:
            area_name = area.split('_')[-1].split('.')[0]
            if area_name == 'S1':
                continue
            with open(area, 'rb') as area_file:
                spike_times = pkl.load(area_file)
            area_label, area_hemisphere = area_name[:-1], area_name[-1]
            if area_label not in all_areas:
                all_areas.append(area_label)
            for channel in spike_times:
                channel_spikes = spike_times[channel]
                if channel_spikes.shape[0] > 0:
                    channel_neurons = channel_spikes[:, 0].max()
                    for o_idx, orient in enumerate(['horizontal', 'vertical']):
                        orientation_mask = (trial_data['handOrien'] - 1) // 2 == o_idx
                        for mod in [0, 1]:
                            hand_mask     = trial_data['handOrien'] % 2 == mod
                            spike_mask    = np.isin(channel_spikes[:, -1],
                                                    np.where(hand_mask * orientation_mask))
                            hand_label    = hand_list[mod]
                            lateral_label = 'i' if area_hemisphere == hand_label else 'c'
                            region_key    = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in area_summary_dict:
                                area_summary_dict[region_key] = {}
                            for event in events:
                                if event not in area_summary_dict[region_key]:
                                    area_summary_dict[region_key][event] = {
                                        'spikes': [], 'neurons': 0, 'event_times': []}
                                event_times = trial_data[event][
                                    channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([
                                    channel_spikes[spike_mask, 0]
                                    + area_summary_dict[region_key][event]['neurons'],
                                    channel_spikes[spike_mask, 1]
                                    - event_times[spike_mask].astype(float),
                                ])
                                area_summary_dict[region_key][event]['spikes'].append(event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += channel_neurons
                                area_summary_dict[region_key][event]['event_times'].append(event_times)
    with open(spikes_file_name, 'wb') as spike_file:
        pkl.dump(area_summary_dict, spike_file)

# ---------------------------------------------------------------------------
# Merge orientations into merged_summary_dict
# ---------------------------------------------------------------------------
merged_summary_dict  = {}
merged_dict_filename = f'{summary_dir}/mergedSpikeDict.p'
for region_key in area_summary_dict.keys():
    latArea, orient = region_key.split('_')
    if latArea not in merged_summary_dict:
        merged_summary_dict[latArea] = {}
    for event in area_summary_dict[region_key].keys():
        if event not in merged_summary_dict[latArea]:
            merged_summary_dict[latArea][event] = {'spikes': []}
        event_spikes = area_summary_dict[region_key][event]['spikes']
        merged_summary_dict[latArea][event]['spikes'].append(np.hstack(event_spikes))
for latArea in merged_summary_dict.keys():
    for event in merged_summary_dict[latArea].keys():
        merged_summary_dict[latArea][event]['spikes'] = np.hstack(
            merged_summary_dict[latArea][event]['spikes'])
        merged_summary_dict[latArea][event]['neurons'] = int(
            np.max(merged_summary_dict[latArea][event]['spikes'][0]))
with open(merged_dict_filename, 'wb') as merged_file:
    pkl.dump(merged_summary_dict, merged_file)

# ---------------------------------------------------------------------------
# SDF computation — per-neuron, orientation-merged
# ---------------------------------------------------------------------------
area_max_rate       = {}
all_sdf_filename    = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_orientMerged.p'
max_rate_filename   = f'{summary_dir}/sdfMaxRate_bin{binsize}_k{kernel_width}_orientMerged.p'
peak_order_filename = f'{summary_dir}/peakOrderDict_bin{binsize}_k{kernel_width}_orientMerged.p'

if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as sdf_file:
        all_sdf_dict = pkl.load(sdf_file)
    with open(max_rate_filename, 'rb') as max_rate_file:
        area_max_rate = pkl.load(max_rate_file)
    with open(peak_order_filename, 'rb') as peak_order_file:
        peak_order_dict = pkl.load(peak_order_file)
    print(f'SDF dictionary loaded (bin={binsize}, kernel={kernel_width})')
else:
    all_sdf_dict    = {}
    peak_order_dict = {}
    for latArea in merged_summary_dict.keys():
        peak_order_dict[latArea] = {}
        area_max_rate[latArea] = np.zeros(
            merged_summary_dict[latArea][events[0]]['neurons'])
        print(latArea)
        all_sdf_dict[latArea] = {}
        for event in events:
            sdf_list          = []
            neuron_count      = merged_summary_dict[latArea][event]['neurons']
            neuron_peak_times = np.zeros(neuron_count)
            for neuron_idx in range(neuron_count):
                neuron_mask   = (merged_summary_dict[latArea][event]['spikes'][0, :]
                                 == neuron_idx + 1)
                neuron_spikes = merged_summary_dict[latArea][event]['spikes'][:, neuron_mask].copy()
                neuron_spikes[0, :] = 1
                neuron_psth   = gen_psth(neuron_spikes.T, binsize=binsize,
                                         window=window_range, neurons=1)
                neuron_sdf, _ = gen_sdf(neuron_psth, w=kernel_width, bin_size=binsize,
                                        ftype='Gauss', multi_unit=False)
                neuron_peak_times[neuron_idx] = neuron_sdf[:].argmax()
                sdf_list.append(neuron_sdf[:, 0].T)
            peak_order_dict[latArea][event] = np.argsort(neuron_peak_times)
            area_sdf = np.hstack(sdf_list)
            all_sdf_dict[latArea][event] = area_sdf
            event_max_rate = np.max(area_sdf, axis=0)
            area_max_rate[latArea] = np.maximum(event_max_rate, area_max_rate[latArea])

    with open(all_sdf_filename, 'wb') as sdf_file:
        pkl.dump(all_sdf_dict, sdf_file)
    with open(max_rate_filename, 'wb') as max_rate_file:
        pkl.dump(area_max_rate, max_rate_file)
    with open(peak_order_filename, 'wb') as peak_order_file:
        pkl.dump(peak_order_dict, peak_order_file)
    print('SDF dictionary saved')

# ---------------------------------------------------------------------------
# Neuron peaks — identify epoch of maximal firing rate per neuron
# ---------------------------------------------------------------------------
neuron_peaks_filename = f'{summary_dir}/neuron_peaks.p'
full_window = np.arange(WINDOW_RANGE[0], WINDOW_RANGE[1] + binsize, binsize)

# Merge max rates across c/i for each bare region
area_merged_rates = {'M1': None, 'PMd': None, 'PMv': None}
for area_label in area_max_rate.keys():
    lat, region = area_label[0], area_label[1:]
    area_max = area_max_rate[area_label]
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
        event_mask   = (full_window > event_window[0]) * (full_window <= event_window[1])
        area_sdf     = all_sdf_dict[area_label][event][event_mask]
        event_spike_maxes[area_label][epoch] = area_sdf.max(axis=0)
    event_neuron_peaks     = np.vstack(list(event_spike_maxes[area_label].values()))
    event_peaks            = (event_neuron_peaks == area_merged_rates[region])
    event_peak_proportions = np.average(event_peaks, axis=1)
    event_neuron_max[region]['neurons']                   = event_neuron_peaks.shape[1]
    event_neuron_max[region]['proportions'][lat_map[lat]] = event_peak_proportions

with open(neuron_peaks_filename, 'wb') as neuron_peaks_file:
    pkl.dump(event_neuron_max, neuron_peaks_file)
print('Neuron peaks saved')
