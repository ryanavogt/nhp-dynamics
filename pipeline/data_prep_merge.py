import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os
import glob

import pandas as pd
import scipy.io as sio
import numpy as np

from utils.sig_proc import trial_splitter

from config import (
    MONKEY_NAME_MAP       as monkey_name_map,
    M1_HEMISPHERE_MAP     as m1_hemisphere_map,
    MONKEY_LIST_FORMAT_A  as m_list_1,
    MONKEY_LIST_FORMAT_B  as m_list_2,
    SCALE_MONKEYS         as scale_monkeys,
    MATLAB_DIR            as matlab_dir,
    SORTING_DIR           as sorting_dir,
    PROCESSED_DIR         as processed_dir,
    EVENTS_MERGE          as event_list,
    EVENT_THRESHOLDS_PREP as event_thresholds,
)

load_override = True

# ---------------------------------------------------------------------------
# Identify all sessions from sorting notes
# ---------------------------------------------------------------------------
file_list_1, file_list_2 = [], []
for m1, m2 in zip(m_list_1, m_list_2):
    file_list_1 += glob.glob(f'{sorting_dir}/*ortingNotes_*{m1}.xlsx')
    file_list_2 += glob.glob(f'{sorting_dir}/*ortingNotes_*{m2}.xlsx')

for file in file_list_1 + file_list_2:
    file_name  = file.split('\\')[-1]
    _, date_flat, name_label = file_name.split('_')
    name_label  = name_label[0]
    monkey_name = monkey_name_map[name_label]
    year, month, day = date_flat[0:4], date_flat[4:6], date_flat[6:]
    date        = f'{year}_{month}_{day}'
    print(date)

    # Output directory
    save_dir = os.path.join(processed_dir, f'Monkey_{monkey_name}', date)
    os.makedirs(save_dir, exist_ok=True)

    if not load_override and os.path.isfile(os.path.join(save_dir, 'Neurons_U1.p')):
        print(f'  Skipping {monkey_name} {date} (already processed)')
        continue

    # ── Load trial event times ──────────────────────────────────────────────
    rt              = sio.loadmat(
        f'{sorting_dir}/relTrialTimes_{year}{month}{day}_{name_label}.mat'
    )['relTrialTimes'][0, 0]
    mat_event_cols  = rt[-1]
    mat_event_names = rt[0][0].split(' | ')
    event_df        = pd.DataFrame(mat_event_cols, columns=mat_event_names)

    if file in file_list_1:
        event_df = event_df.astype({'newTrialNumbers': int})
    else:
        event_df = event_df.astype({'trialNumbers': int})

    # ── Load sorting notes (neuron metadata) ────────────────────────────────
    sorting_file = os.path.join(sorting_dir, f'SortingNotes_{year}{month}{day}_{name_label}.xlsx')
    df           = pd.read_excel(sorting_file)

    neurons = {1: {}, 2: {}}
    for unit, channel, cell_no, area in zip(df['Unit'], df['Channel'], df['n cells'], df['Area']):
        if cell_no > 0:
            if area == 'M1':
                area = f'M1{m1_hemisphere_map[monkey_name]}'
            if area in neurons[unit]:
                neurons[unit][area].append((channel, cell_no))
            else:
                neurons[unit][area] = [(channel, cell_no)]

    for unit in [1, 2]:
        with open(os.path.join(save_dir, f'Neurons_U{unit}.p'), 'wb') as f:
            pkl.dump(neurons[unit], f)

    # ── Build trial_data ────────────────────────────────────────────────────
    trial_data = {}
    event_durs = {}
    track_durs = True

    if file in file_list_1:
        trial_data['handOrien']       = (event_df['angle'] + event_df['isRightHandUsed']).astype(int)
        trial_data['trialStartTimes'] = (event_df['trialbaseline'] * 1000).astype(int)
        trial_data['trialRewardDrop'] = (event_df['trialGoCue'] * 1000).astype(int) - trial_data['trialStartTimes']
        trial_data['trialGraspOn']    = (event_df['trialGraspOn'] * 1000).astype(int) - trial_data['trialStartTimes']
        trial_data['trialEnd']        = (event_df['trialGraspOn'] * 1000).astype(int) + 800

    if file in file_list_2:
        track_durs = True
        trial_data['handOrien']       = event_df['handOrien']
        trial_data['trialStartTimes'] = event_df['trialStartTimes']
        trial_data['trialEnd']        = event_df['trialGraspOn'] + 800
        trial_data['trialGraspOn']    = event_df['trialGraspOn']    - trial_data['trialStartTimes']
        trial_data['trialRewardDrop'] = event_df['trialRewardDrop'] - trial_data['trialStartTimes']
        event_durs['trialGraspOff']   = (event_df['trialGraspOff'].astype(int)
                                         - trial_data['trialGraspOn']
                                         - trial_data['trialStartTimes'])
        event_durs['trialReachOn']    = (event_df['trialReachOn'].astype(int)
                                         - trial_data['trialRewardDrop']
                                         - trial_data['trialStartTimes'])
        event_durs['trialGraspOn']    = (trial_data['trialGraspOn']
                                         - trial_data['trialRewardDrop'])

    with open(os.path.join(save_dir, 'trial_data.p'), 'wb') as f:
        pkl.dump(trial_data, f)

    # ── Split spikes into trials ────────────────────────────────────────────
    matlab_monkey_dir = os.path.join(matlab_dir, f'monk{monkey_name}2024Sorting_SQ')
    scaling           = name_label in scale_monkeys

    for matlab_file_name in glob.glob(f'{matlab_monkey_dir}/{year}_{month}_{day}*'):
        unit = int(matlab_file_name.split('.')[0][-1])
        data = sio.loadmat(matlab_file_name)
        area_spike_counts = {}

        for area in neurons[unit].keys():
            rel_spike_times   = {}
            area_neuron_count = 0
            area_spike_count  = 0
            for sig, no_neurons in neurons[unit][area]:
                trial_windows        = np.vstack([trial_data['trialStartTimes'], trial_data['trialEnd']]).T
                rel_spike_times[sig] = trial_splitter(data, trial_windows, sig, scaling=scaling)
                area_neuron_count   += no_neurons
                area_spike_count    += rel_spike_times[sig].shape[0]
            with open(os.path.join(save_dir, f'spikeTimes_U{unit}_{area}.p'), 'wb') as f:
                pkl.dump(rel_spike_times, f)
            area_spike_counts[area] = area_spike_count
            print(f'  Area: {area}, Neurons: {area_neuron_count}, Spikes: {area_spike_count}')

        with open(os.path.join(save_dir, f'spikeCounts_U{unit}.p'), 'wb') as f:
            pkl.dump(area_spike_counts, f)

    # ── Compute and save trial event mask ───────────────────────────────────
    full_mask  = np.ones(len(trial_data['trialRewardDrop']), dtype=bool)
    prev_times = 0
    for event in event_list:
        event_times_arr = trial_data[event].values
        event_duration  = event_times_arr - prev_times
        if event in event_thresholds:          # skip events with no threshold defined
            lo, hi      = event_thresholds[event]
            full_mask  &= (event_duration > lo) & (event_duration < hi)
        prev_times      = event_times_arr

    rel_event_times = {event: trial_data[event].values[full_mask] for event in event_list}
    print(f'  Keep proportion: {full_mask.sum() / full_mask.shape[0]:.2f}\n')

    with open(os.path.join(save_dir, 'eventTimes.p'), 'wb') as f:
        pkl.dump(rel_event_times, f)
    with open(os.path.join(save_dir, 'eventMasks.p'), 'wb') as f:
        pkl.dump(full_mask, f)

    if track_durs:
        durs = {event: event_durs[event][full_mask] for event in event_durs}
        with open(os.path.join(save_dir, 'eventDurs.p'), 'wb') as f:
            pkl.dump(durs, f)
