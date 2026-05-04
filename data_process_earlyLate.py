import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (usig conda or pip)
import matplotlib.pyplot as plt     #Generating plots
from sig_proc import *
import pandas as pd
import matplotlib as mpl
from matplotlib import cm
import scipy as sp
import numpy as np

from sig_proc import *
from plot_utils import *
from pandas.plotting import table

import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sns.set_style(style='white')

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Early', 'trialGraspOn':'Late'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Early':        {'event': 'trialRewardDrop', 'window': [-500,   300]},
                    'Late':         {'event': 'trialGraspOn',    'window': [-500,   300]}}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary/earlyLate'

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
events = ['trialRewardDrop', 'trialGraspOn']
binsize = 5
kernel_width = 25
trial_count = 100
load_override_preprocess = False
load_override = True

# Extract All Sessions from their Sorting Notes
file_list = []
for m in monkey_name_map.keys():
    file_list += [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*{m}.xlsx')]
file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])    #Date of Session
    monkey_labels.append(name[2])   #Monkey (G or R)

"""
Aggregate all neurons across all sessions within a single region, separated by lateral 
relation to hand used (contra or ipsi) and orientation of target (horizontal or vertical)
"""

area_summary_dict = {}
all_areas = []
all_spikes = {}
all_conditions = {}
all_monkey_indices = {'M1': {'count':0}, 'PMd':{'count':0}, 'PMv':{'count':0}}
all_events =    {'M1': {'Early':[], 'Late':[]},
                 'PMd':{'Early':[], 'Late':[]},
                 'PMv':{'Early':[], 'Late':[]}}
summary_file_name = f'{summary_dir}/spike_summary_earlyLate.p'
spike_file_name = f'{summary_dir}/spike_list_earlyLate.p'
event_file_name = f'{summary_dir}/event_list_earlyLate.p'
condition_file_name = f'{summary_dir}/condition_list_earlyLate.p'
durs_file_name = f'{summary_dir}/durs_list_earlyLate.p'
monkey_file_name = f'{summary_dir}/monkey_indices_earlyLate.p'
if os.path.exists(summary_file_name) and not load_override_preprocess:
    with open(summary_file_name, 'rb') as summary_file:
        area_summary_dict = pkl.load(summary_file)
    with open(spike_file_name, 'rb') as spike_file:
        all_spikes = pkl.load(spike_file)
    with open(condition_file_name, 'rb') as condition_file:
        all_conditions = pkl.load(condition_file)
    with open(monkey_file_name, 'rb') as monkey_file:
        all_monkey_indices = pkl.load(monkey_file)
    print('Spike File Loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
        date_folder = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
        trial_dir = f'Data/Processed/{monkey_folder}/{date_folder}' #Folder where to find the session data
        with open(f'{trial_dir}/trial_data.p', 'rb') as trial_file:
            trial_data = pkl.load(trial_file)
        with open(f'{trial_dir}/eventMasks.p', 'rb') as event_mask_file:
            full_mask = pkl.load(event_mask_file)
        area_list = [f for f in glob.glob(f'{trial_dir}/spikeTimes_*')] #All brain regions in folder (e.g. M1R, PMdR, PMvL, etc.)
        for area in area_list:
            area_name = area.split('_')[-1].split('.')[0]
            if area_name == 'S1': #Ignore S1 area
                continue
            with open(area, 'rb') as area_file:
                spike_times = pkl.load(area_file) #Load the spike timing file
            area_label, area_hemisphere = area_name[:-1], area_name[-1] #Final letter of file name indicates side (R or L)
            area_indices = all_monkey_indices[area_label]
            if monkey not in area_indices.keys():
                area_indices[monkey]= []
            area_neuron_index = area_indices['count']
            if area_label not in all_areas:
                all_areas.append(area_label)
                region_spike_list = []
                region_condition_list = []
            else:
                region_spike_list = all_spikes[area_label]
                region_condition_list = all_conditions[area_label]
            for channel in spike_times:
                channel_spikes = spike_times[channel] #Extract spike times for a single channel
                if channel_spikes.shape[0]>0:
                    channel_neurons = channel_spikes[:, 0].max()
                    channel_conditions = np.empty((len(trial_data['handOrien']), 2), dtype='str_')
                    for neuron in range(1, channel_neurons+1):
                        neuron_spike_list = []
                        neuron_spikes = channel_spikes[channel_spikes[:,0]==neuron]
                        for trial in range(1, channel_conditions.shape[0]+1):
                            neuron_spike_list.append(neuron_spikes[neuron_spikes[:,-1]==trial, 1])
                        region_spike_list.append(neuron_spike_list)
                        for event in events:
                            all_events[area_label][event_map[event]].append(list(trial_data[event]))
                    for o_idx, orient in enumerate(['horizontal', 'vertical']):
                        orientation_mask = (trial_data['handOrien']-1)//2 == o_idx
                        for mod in [0,1]:
                            hand_mask = trial_data['handOrien']%2 == mod
                            spike_mask = np.isin(channel_spikes[:, -1], np.where(hand_mask*orientation_mask*full_mask))
                            hand_label = hand_list[mod]
                            if area_hemisphere == hand_label:
                                lateral_label = 'i'
                            else:
                                lateral_label = 'c'
                            condition_label = f'{lateral_label}{orient[0]}'
                            channel_conditions[hand_mask*orientation_mask] = np.array([lateral_label, orient[0]])
                            region_key = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in area_summary_dict:
                                area_summary_dict[region_key] = {}
                            for event in events:
                                if event not in area_summary_dict[region_key]:
                                    area_summary_dict[region_key][event] = {'spikes':[], 'neurons': 0, 'trial_idcs':[]}
                                event_times = trial_data[event][channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([channel_spikes[spike_mask, 0] + area_summary_dict[region_key][event]['neurons'],
                                                  channel_spikes[spike_mask, 1] - event_times[spike_mask].astype(float)])
                                area_summary_dict[region_key][event]['spikes'].append(event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += channel_neurons
                                area_summary_dict[region_key][event]['trial_idcs'].append(channel_spikes[spike_mask, -1])
                    for neuron in range(channel_neurons):
                        region_condition_list.append(channel_conditions)
                        area_indices[monkey].append(area_neuron_index)
                        area_neuron_index += 1
            all_monkey_indices[area_label]['count'] = area_neuron_index
            all_spikes[area_label]=region_spike_list
            all_conditions[area_label] = region_condition_list
    with open(summary_file_name, 'wb') as summary_file:
        pkl.dump(area_summary_dict, summary_file)
    with open(spike_file_name, 'wb') as spike_file:
        pkl.dump(all_spikes, spike_file)
    with open(event_file_name, 'wb') as event_file:
        pkl.dump(all_events, event_file)
    with open(condition_file_name, 'wb') as condition_file:
        pkl.dump(all_conditions, condition_file)
    with open(monkey_file_name, 'wb') as monkey_file:
        pkl.dump(all_monkey_indices, monkey_file)
    print('Spike File Saved')

"""
Generate plots of the max spiking rates for each area, side, and orientation.
Scale the rate to the max rate for contralateral side. Apply that scale to ipsilateral side.
"""
skip = False
area_mean_rate = {}
event_mean_rate = {}
all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_earlyLate.p'
area_mean_filename = f'{summary_dir}/areaMeanRates_bin{binsize}_k{kernel_width}_earlyLate.p'
all_psth_filename = f'{summary_dir}/trialPSTH_bin{binsize}_k{kernel_width}_earlyLate.p'
if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as sdf_file:
        all_sdf_dict = pkl.load(sdf_file)
    print(f'SDF Dictionary Loaded (Bin: {binsize}, Kernel: {kernel_width})')
else:
    all_sdf_dict = {}
    all_psth_dict = {}
    for area_label in area_summary_dict.keys():
        if skip:
            break
        region_key = f'{area_label}'
        print(region_key)
        lat = region_key[0]
        all_psth = []
        all_trial_psth = []
        all_spikes = []
        all_sdf_dict[region_key] = {}
        all_psth_dict[region_key] = {}
        area_mean_rate[region_key] = {}
        for event in events:
            window_range = np.array(epoch_window_map[event_map[event]]['window'])
            num_bins = int(1.0*(window_range.max() - window_range.min())/binsize)
            if region_key in area_summary_dict:
                area_summary_dict[region_key][event]['spikes'] = np.concatenate(area_summary_dict[region_key][event]['spikes'], axis=1)
                area_summary_dict[region_key][event]['trial_idcs'] = np.concatenate(area_summary_dict[region_key][event]['trial_idcs'])
            sdf_list = []
            neuron_count = area_summary_dict[region_key][event]['neurons'].astype(int)
            psth_list = np.zeros((num_bins, neuron_count+1))
            trial_psth_list = []
            for neuron_idx in range(neuron_count):
                neuron_mask = area_summary_dict[region_key][event]['spikes'][0, :] == neuron_idx+1
                neuron_spikes = area_summary_dict[region_key][event]['spikes'][:, neuron_mask]
                if neuron_spikes.shape[1] == 0:
                    print(f'No spikes for neuron {neuron_idx+1}')
                neuron_trials = area_summary_dict[region_key][event]['trial_idcs'][neuron_mask]
                neuron_spikes[0, :] = 1
                psth_trials = trial_psth(neuron_spikes.T, neuron_trials, binsize=binsize, window=window_range, neurons=1)
                psth_trials = np.stack(psth_trials, axis=-1)
                neuron_psth = gen_psth(neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
                if neuron_idx == 0:
                    psth_list[:, 0] = neuron_psth[1:, 0]
                psth_list[:, neuron_idx+1] = neuron_psth[1:, 1]
                trial_psth_list.append(psth_trials)
            if region_key not in event_mean_rate.keys():
                event_mean_rate[region_key] = {}
            event_mean_rate[region_key][event] = psth_list[:,1:].mean(axis=0)
            area_sdf, _ = gen_sdf(psth_list[:, 1:], w=kernel_width, bin_size=binsize, ftype='Gauss', multi_unit=True)
            all_sdf_dict[region_key][event_map[event]] = area_sdf.T
            all_psth_dict[region_key][event_map[event]] = np.stack(trial_psth_list)
    with open(all_sdf_filename, 'wb') as sdf_file:
        print('Saving SDF Dictionary')
        pkl.dump(all_sdf_dict, sdf_file)
    with open(all_psth_filename, 'wb') as psth_file:
        pkl.dump(all_psth_dict, psth_file)