import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import math                         #Ceil, floor, and related functions (built-in)
import glob

# The following packages need to be installed in your virtual environment (usig conda or pip)
import pandas as pd                 #Loading tables
import scipy.io as sio              #Loading matlab files
import numpy as np                  #Array operations
import matplotlib.pyplot as plt     #Generating plots
from sig_proc import *
import matplotlib as mpl

monkey_name_map = {'R': 'Red', 'G': 'Green'}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

hand_list = ['R', 'L']

file_list = [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*.xlsx')]
file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])
    monkey_labels.append(name[2])

area_summary_dict = {}
all_areas = []
for date, monkey in zip(date_strings, monkey_labels):
    monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
    date_folder = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
    trial_dir = f'Data/Processed/{monkey_folder}/{date_folder}'
    with open(f'{trial_dir}/trial_data.p', 'rb') as trial_file:
        trial_windows = pkl.load(trial_file)
    area_list = [f for f in glob.glob(f'{trial_dir}/spikeTimes_*')]
    for area in area_list:
        area_name = area.split('_')[-1].split('.')[0]
        if area_name[-1]=='1':
            print(f'{date}: {area_name}')
        if area_name == 'S1':
            continue
        with open(area, 'rb') as area_file:
            spike_times = pkl.load(area_file)
        area_label, area_hemisphere = area_name[:-1], area_name[-1]
        if area_label not in all_areas:
            all_areas.append(area_label)
        for channel in spike_times:
            channel_spikes = spike_times[channel]
            grasp_times = trial_windows[:, -1][channel_spikes[:, -1].astype(int)-1]
            if channel_spikes.shape[0]>0:
                channel_neurons = channel_spikes[:, 0].max()
                for mod in [0,1]:
                    hand_mask = trial_windows[:, 1]%2 == mod
                    spike_mask = np.in1d(channel_spikes[:, -1], np.where(hand_mask))
                    hand_label = hand_list[mod]
                    if area_hemisphere == hand_label:
                        lateral_label = 'i'
                    else:
                        lateral_label = 'c'
                    region_key = f'{lateral_label}{area_label}'
                    if region_key not in area_summary_dict:
                        area_summary_dict[region_key] = {'spikes':[], 'neurons': 0}
                    grasp_spikes = np.vstack([channel_spikes[spike_mask, 0] + area_summary_dict[region_key]['neurons'],
                                      channel_spikes[spike_mask, 1] - grasp_times[spike_mask].astype(float)])
                    area_summary_dict[region_key]['spikes'].append(grasp_spikes)
                    area_summary_dict[region_key]['neurons'] += channel_neurons

for area_label in all_areas:
    for lat in ['c', 'i']:
        region_key = f'{lat}{area_label}'
        if region_key in area_summary_dict:
            area_summary_dict[region_key]['spikes'] = np.hstack(area_summary_dict[f'{lat}{area_label}']['spikes'])
        psth_list = []
        neuron_count = area_summary_dict[region_key]['neurons'].astype(int)
        window_range = np.array([-1000, 1000])
        binsize = 20
        if lat == 'c':
            neuron_peak_times = np.zeros(neuron_count)
            neuron_scales = np.zeros(neuron_count)
        for neuron_idx in range(neuron_count):
            neuron_mask = area_summary_dict[region_key]['spikes'][0, :] == neuron_idx+1
            neuron_spikes = area_summary_dict[region_key]['spikes'][:, neuron_mask]
            neuron_spikes[0, :] = 1
            neuron_psth = gen_psth(neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
            neuron_scales[neuron_idx] = neuron_psth[:,1].max()
            peak_location = neuron_psth[:,1].argmax()
            if lat == 'c':
                neuron_peak_times[neuron_idx] = peak_location
            time_scale = neuron_psth[:,0]
            psth_list.append(neuron_psth[:,1]/neuron_scales[neuron_idx])
        peak_order = np.argsort(neuron_peak_times)
        area_scale_psth = np.vstack(psth_list)
        y, x = np.mgrid[1:neuron_count+2:1, window_range[0]:window_range[1]+2*binsize:binsize]
        plt.figure(figsize = (3, 8))
        plt.pcolor(x, y, area_scale_psth[peak_order],
                   cmap='inferno', norm = mpl.colors.Normalize(vmin=0, vmax=1.5))
        plt.axvline(x=0, color='w', linestyle=':')
        plt.gca().invert_yaxis()
        plt.title(f'{region_key}')
        plt.ylabel('Neuron No.')
        plt.xlabel('Time from Grasp Onset (ms)')
        plt.savefig(f'{summary_dir}/NeuronSpikes_{region_key}.png', bbox_inches='tight')