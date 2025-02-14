# The following packages are built-in for python builds
import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (using conda or pip)
import pandas as pd                 #Loading tables
import scipy.io as sio              #Loading matlab files

# Local module(s)
from  sig_proc import *             #Local module - for processing signal data


monkey_name_dict = {'G': 'Green', 'R': 'Red'}
monkey_M1_hemisphere = {'Green': 'R', 'Red': 'R'}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'

#Identify all data sessions
file_list = [f for f in glob.glob(f'{sorting_dir}/*ortingNotes_*')]
for file in file_list:
    file_name = file.split('\\')[-1]
    _, date_flat, name_label = file_name.split('_')
    name_label = name_label[0]
    monkey_name = monkey_name_dict[name_label]
    year, month, day = date_flat[0:4], date_flat[4:6], date_flat[6:]
    date = f'{year}_{month}_{day}'
    print(date)
    # How many units for this session
    # units = [1, 2]

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
    event_times = rt[2]
    events = rt[0][0].split(' | ')
    event_df = pd.DataFrame(event_times, columns=events)
    event_df = event_df.astype({'trialNumbers':int})

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
    df = pd.read_excel(f'{sorting_dir}/SortingNotes_{year}{month}{day}_{name_label}.xlsx')


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

    # Define time window for each trial
    trial_data = {}
    trial_data['handOrien'] = event_df['handOrien']
    trial_data['trialStartTimes'] = event_df['trialStartTimes']
    trial_data['trialEnd'] = event_df['trialGraspOff'] + 500
    trial_data['trialGraspOn'] = event_df['trialGraspOn'] - trial_data['trialStartTimes']
    trial_data['trialReachOn'] = event_df['trialReachOn'] - trial_data['trialStartTimes']
    trial_data['trialRewardDrop'] = event_df['trialRewardDrop'] - trial_data['trialStartTimes']


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
        for area in neurons[unit].keys():
            area_spike_count = 0
            area_neuron_count = 0
            rel_spike_times = {}
            for sig, no_neurons in neurons[unit][area]:
                trial_windows = np.vstack([trial_data['trialStartTimes'], trial_data['trialEnd']]).T
                rel_spike_times[sig] = trial_splitter(data, trial_windows, sig)
                area_neuron_count += no_neurons
                area_spike_count += rel_spike_times[sig].shape[0]
            with open(f'{save_dir}/spikeTimes_U{unit}_{area}.p', 'wb') as spike_time_file:
                pkl.dump(rel_spike_times, spike_time_file)
            area_spike_counts[unit][area] = area_spike_count
            area_neuron_counts[unit][area] = area_neuron_count
            print(f'Area: {area}, Neurons: {area_neuron_count}')
        with open(f'{save_dir}/spikeCounts_U{unit}.p', 'wb') as spike_count_file:
            pkl.dump(area_spike_counts[unit], spike_count_file)

        # Record relative event times for each trial
        events = ['trialRewardDrop', 'trialReachOn', 'trialGraspOn', 'trialGraspOff']
        rel_event_times = {}
        for event in events:
            event_times = event_df[event].values
            rel_event_times[event] = event_times - trial_data['trialStartTimes']

        with open(f'{save_dir}/eventTimes.p', 'wb') as event_time_file:
            pkl.dump(rel_event_times, event_time_file)

        # #Generate psth across all neurons in each area for a single trial
        # kernel_type = 'gpfa'
        # kernel_width = 50
        # bin_size = 10
        # trial_psth_dir = f'{save_dir}/trial_psth_{bin_size}'
        # trial_sdf_dir = f'{save_dir}/trial_sdf_{kernel_type}_{kernel_width}'
        # total_spike_counts = {}
        #
        # if not os.path.isdir(trial_psth_dir):
        #     os.mkdir(trial_psth_dir)
        # if not os.path.isdir(trial_sdf_dir):
        #     os.mkdir(trial_sdf_dir)
        # for trial, (t_start, t_end) in enumerate(trial_windows[:, 2:4]):
        #     trial += 1
        #     window = np.array([0, t_end-t_start])
        #     trial_psth_dict = {}
        #     trial_sdf_dict = {}
        #     for unit in units:
        #         for area in neurons[unit]:
        #             with open(f'{save_dir}/spikeTimes_U{unit}_{area}.p', 'rb') as spike_time_file:
        #                 spike_times = pkl.load(spike_time_file)
        #             area_neurons = len(neurons[unit][area])
        #             psth_full_bins = np.zeros(math.ceil(window[-1]/bin_size) + 1)
        #             # psth_full_counts = np.zeros((math.ceil(window[-1]/bin_size) + 1, area_neurons))
        #             psth_full_counts = []
        #             for idx, (signal, no_neurons) in enumerate(neurons[unit][area]):
        #                 trial_mask = (spike_times[signal][:, -1] == trial)
        #                 psth = gen_psth(spike_times[signal][trial_mask], binsize = bin_size, window = window, neurons = int(no_neurons))
        #                 psth_bins, psth_counts = psth[:, 0], psth[:, 1:]
        #                 # psth_full_counts[:, idx] = psth_counts
        #                 psth_full_counts.append(psth_counts)
        #                 if idx == 0:
        #                     psth_full_bins[:] = psth_bins
        #             psth_full_counts = np.hstack(psth_full_counts)
        #             trial_psth_dict[area] = (psth_full_bins, psth_full_counts)
        #             sdf, kernel = gen_sdf(psth_full_counts, kernel_type, w=kernel_width, bin_size=bin_size)
        #             trial_sdf_dict[area] = (kernel, sdf)
        #     with open(f'{trial_psth_dir}/PSTH_trial{trial}.p', 'wb') as psth_trial_file:
        #         pkl.dump(trial_psth_dict, psth_trial_file)
        #     with open(f'{trial_sdf_dir}/SDF_trial{trial}.p', 'wb') as sdf_trial_file:
        #         pkl.dump(trial_sdf_dict, sdf_trial_file)
        #
        #
        #     print(f'Trial {trial}')