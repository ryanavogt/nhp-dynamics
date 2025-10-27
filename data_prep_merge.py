# The following packages are built-in for python builds
import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (using conda or pip)
import pandas as pd                 #Loading tables
import scipy.io as sio              #Loading matlab files
import numpy as np

from sig_proc import *


m_list_1 = ['Y', 'B']
m_list_2 = ['R','G']
scale_monkeys = ['R', 'G', 'Y', 'B'] #Which monkeys have time recorded in seconds
monkey_names_full = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
monkey_name_dict = {}
for monkey in m_list_1+m_list_2:
    monkey_name_dict[monkey] = monkey_names_full[monkey]
monkey_M1_hemisphere = {'Green': 'R', 'Red': 'R', 'Blue':'R', 'Yellow':'R'}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'

#Identify all data sessions
file_list_1 = []
file_list_2 = []
for m1, m2 in zip(m_list_1,m_list_2):
    file_list_1 += [f for f in glob.glob(f'{sorting_dir}/*ortingNotes_*{m1}.xlsx')]
    file_list_2 += [f for f in glob.glob(f'{sorting_dir}/*ortingNotes_*{m2}.xlsx')]

for file in file_list_1+file_list_2:
    file_name = file.split('\\')[-1]
    _, date_flat, name_label = file_name.split('_')
    name_label = name_label[0]
    monkey_name = monkey_name_dict[name_label]
    year, month, day = date_flat[0:4], date_flat[4:6], date_flat[6:]
    date = f'{year}_{month}_{day}'
    print(date)

    # Load trial times for the session
    """
    Contents:
    trialNumbers:           Index for trial (clean)             - int
    originalTrialNumber:    Trial index before cleaning         - int
    isRightHandUsed:        (error?) Right hand used? T(R)/F(L) - boolean (0/1)
    handOrien               Orientation of target, hand         - 1:left horizontal, 2:right horizontal, 3:left vertical, 4:right vertical
    trialStartTimes         Start time of trial                 - int (ms)
    trialRewardDrop         Time of pellet drop ("Go cue")      - int (ms)
    trialReachOn            Time monkey starts to reach         - int (ms)
    trialGraspOn            Time monkey starts grasp            - int (ms)
    trialGraspOff           Time monkey ends grasp              - int (ms)
    """
    rt = sio.loadmat(f'{sorting_dir}/relTrialTimes_{year}{month}{day}_{name_label}.mat')['relTrialTimes'][0,0]

    # Split absolute start times by event
    event_times = rt[-1]
    events = rt[0][0].split(' | ')
    event_df = pd.DataFrame(event_times, columns=events)
    if file in file_list_1:
        event_df = event_df.astype({'newTrialNumbers':int})
    else:
        event_df = event_df.astype({'trialNumbers':int})
    # if monkey_name == 'Red':
    #     print(event_df)
    # Load sorting notes for session neuron metadata
    """
    Contents:
    Unit                    Machine unit of collection              - int (1 or 2)
    Channel                 Channel/electrode                       - int (1 - 128)
    n cells                 Number of cells identified              - int (0 - 4)
    Sort code [1, 2, 3, 4]  Quality of neurons identified           - int : 1 Well-isolated, 2: Mixed, 3: Multi-unit
    Notes                   Notes                                   - str
    Connector               Array Connector Number                  - int
    Area                    Region of the brain                     - str (M1, PMd(L or R), PMv(L or R))
    Hemis                   Hemisphere of brain (relative to ???)   - str (contra or ipsi)
    """
    sorting_file = f'{sorting_dir}/SortingNotes_{year}{month}{day}_{name_label}.xlsx'
    df = pd.read_excel(sorting_file)

    # Sort neurons by region - Assign channel and number of cells per channel to region
    neurons = {1: {}, 2: {}}
    neurons_other = {}
    sig_nos = {1: [], 2: []}
    sig_nos_other = []

    for unit, channel, cell_no, area in zip(df['Unit'], df['Channel'], df['n cells'], df['Area']):
        if cell_no>0: #Ignore channels with no neurons
            sig_no = sig_nos[unit]
            neurons_temp = neurons[unit]
            sig_no.append(channel)
            if area == 'M1':
                area = f'M1{monkey_M1_hemisphere[monkey_name]}'
            if area in neurons_temp.keys():
                neurons_temp[area].append((channel, cell_no))
            else:
                neurons_temp[area] = [(channel, cell_no)]

    monkey_save_dir = f'Data/Processed/Monkey_{monkey_name}'
    if not os.path.isdir(monkey_save_dir):
        os.mkdir(monkey_save_dir)

    save_dir = f'{monkey_save_dir}/{date}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for unit in [1,2]:
        with open(f'{save_dir}/Neurons_U{unit}.p', 'wb') as neuron_file:
            pkl.dump(neurons[unit], neuron_file)

    trial_data = {}
    event_durs = {}
    track_durs = False
    if file in file_list_1:
        # Define time window for each trial
        trial_data['handOrien'] = (event_df['angle'] + event_df['isRightHandUsed']).astype(int)
        trial_data['trialStartTimes'] = (event_df['trialbaseline']*1000).astype(int)
        trial_data['trialRewardDrop'] = (event_df['trialGoCue']*1000).astype(int)- trial_data['trialStartTimes']
        trial_data['trialGraspOn'] = (event_df['trialGraspOn']*1000).astype(int) - trial_data['trialStartTimes']
        trial_data['trialEnd'] = (event_df['trialGraspOn']*1000).astype(int) + 800

    if file in file_list_2:
        track_durs = True
        trial_data['handOrien'] = event_df['handOrien']
        trial_data['trialStartTimes'] = event_df['trialStartTimes']
        trial_data['trialEnd'] = event_df['trialGraspOn'] + 800
        trial_data['trialGraspOn'] = event_df['trialGraspOn'] - trial_data['trialStartTimes']
        event_durs['trialGraspOff'] = ((event_df['trialGraspOff']).astype(int) - trial_data['trialGraspOn']
                                       - trial_data['trialStartTimes'])
        # trial_data['trialReachOn'] = event_df['trialReachOn'] - trial_data['trialStartTimes']
        trial_data['trialRewardDrop'] = event_df['trialRewardDrop'] - trial_data['trialStartTimes']
        event_durs['trialReachOn'] = ((event_df['trialReachOn']).astype(int) - trial_data['trialRewardDrop']
                                      - trial_data['trialStartTimes'])
        # trial_data['trialGraspOff'] = event_df['trialGraspOff'] - trial_data['trialStartTimes']

    with open(f'{save_dir}/trial_data.p', 'wb') as f:
        pkl.dump(trial_data, f)

    area_spike_counts = {}
    area_neuron_counts = {}
    monkey_dir = f'{matlab_dir}/monk{monkey_name}2024Sorting_SQ'
    matlab_file_list = [f for f in glob.glob(f'{monkey_dir}/{year}_{month}_{day}*')]
    for matlab_file_name in matlab_file_list:
        unit = int(matlab_file_name.split('.')[0][-1])
        data = sio.loadmat(matlab_file_name)
        area_spike_counts[unit] = {}
        area_neuron_counts[unit] = {}
        if name_label in scale_monkeys:
            print()
        for area in neurons[unit].keys():
            area_spike_count = 0
            area_neuron_count = 0
            rel_spike_times = {}
            for sig, no_neurons in neurons[unit][area]:
                trial_windows = np.vstack([trial_data['trialStartTimes'], trial_data['trialEnd']]).T
                scaling = name_label in scale_monkeys
                rel_spike_times[sig] = trial_splitter(data, trial_windows, sig, scaling=scaling)
                area_neuron_count += no_neurons
                area_spike_count += rel_spike_times[sig].shape[0]
            with open(f'{save_dir}/spikeTimes_U{unit}_{area}.p', 'wb') as spike_time_file:
                pkl.dump(rel_spike_times, spike_time_file)
            area_spike_counts[unit][area] = area_spike_count
            area_neuron_counts[unit][area] = area_neuron_count
            print(f'Area: {area}, Neurons: {area_neuron_count}, Spikes: {area_spike_count}')
        with open(f'{save_dir}/spikeCounts_U{unit}.p', 'wb') as spike_count_file:
            pkl.dump(area_spike_counts[unit], spike_count_file)

        # Record relative event times for each trial
        event_list = ['trialRewardDrop', 'trialGraspOn']
        event_thresholds = {'trialRewardDrop':[200, 2000], 'trialGraspOn':[300, 1000]}
        rel_event_times = {}
        prev_times = 0
        event_masks = []
        full_mask = True
        for event in event_list:
            event_times = trial_data[event].values
            rel_event_times[event] = event_times
            event_duration = rel_event_times[event] - prev_times
            event_mask = (event_duration > event_thresholds[event][0])*(event_duration < event_thresholds[event][1])
            prev_times = rel_event_times[event]
            event_masks.append(event_mask)
            full_mask = full_mask*event_mask

        for event in event_list:
            rel_event_times[event] = rel_event_times[event][full_mask]
        print(f'Keep Proportion: {full_mask.sum()/full_mask.shape[0]}\n')
        durs = {}
        if track_durs:
            for event in event_durs.keys():
                durs[event] = event_durs[event][full_mask]
            with open(f'{save_dir}/eventDurs.p', 'wb') as event_dur_file:
                pkl.dump(durs, event_dur_file)
        with open(f'{save_dir}/eventTimes.p', 'wb') as event_time_file:
            pkl.dump(rel_event_times, event_time_file)
        with open(f'{save_dir}/eventMasks.p', 'wb') as event_mask_file:
            pkl.dump(full_mask, event_mask_file)