import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
from matplotlib.lines import Line2D
import torch
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

from plot_utils import pc_subplot, epoch_window_map

from sig_proc import *

import seaborn as sns

def raster_plot(neuron_list, region_spikes, event_list, epoch_window_map):
    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(12, 7))
    epoch_times = [epoch_window_map[event]['time'] for event in epoch_window_map.keys()]
    for idx, neuron in enumerate(neuron_list):
        ax = axs[idx//3][idx%3]
        spikes = region_spikes[neuron-1]
        neuron_spikes = [[]] * len(spikes)
        for event in event_list.keys():
            event_window = epoch_window_map[event]['window']
            event_times = np.array(event_list[event][neuron - 1])
            e_starts = event_times + event_window[0]
            e_ends = event_times + event_window[1]
            e_times = [0]*len(spikes)
            for trial, spike_list in enumerate(spikes):
                if trial == len(event_times):
                    break
                event_start = e_starts[trial]
                event_end = e_ends[trial]
                e_time = e_times[trial] + event_start
                event_mask = (spike_list>event_start)*(spike_list<event_end)
                neuron_spikes[trial] = neuron_spikes[trial]+list(spike_list[event_mask]-e_time +
                                                                 epoch_window_map[event]['time'])
                e_times[trial] += (event_end-event_start)
        for trial, trial_spike_list in enumerate(neuron_spikes):
            trial_spikes = np.array(trial_spike_list)-epoch_times[0]
            neuron_spikes[trial] = trial_spikes
        ax.set_title(f'Neuron {neuron}')
        ax.set_ylabel(f'Trial')
        ax.set_xlabel(f'Time (ms)')
        for epoch_time in epoch_times:
            ax.axvline(epoch_time, alpha=.8, color='k')
        ax.invert_yaxis()
        ax.eventplot(neuron_spikes)
    return fig

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

binsize = 5
kernel_width = 25
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'

epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   100]},
                   'Reach':     {'event': 'trialReachOn',    'window': [-100,    60]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-60,    100]},
                   'Grasp Off': {'event': 'trialGraspOff',   'window': [-100,   100]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]

summary_file_name = f'{summary_dir}/spike_summary.p'
spike_file_name = f'{summary_dir}/spike_list.p'
event_file_name = f'{summary_dir}/event_list.p'

selected_neurons = {'M1': [[63, 11, 39, 73, 71, 115], [42, 115, 147, 61, 149, 74], [61, 21, 85, 149, 53, 65], [81, 70, 37, 61, 80, 65]],
                    'PMd':[[178, 2, 166, 88, 68, 175], [178, 2, 49, 48, 102, 88], [78, 181, 111, 175, 146,118],[78, 77,181, 178,22, 6]],
                    'PMv':[[225,177, 70, 4, 145, 68],[245,110, 225, 215, 49,250], [171, 50, 27,162, 245, 98], [1, 110, 4, 181,162,131]]}

with open(summary_file_name, 'rb') as summary_file:
    area_summary_dict = pkl.load(summary_file)
with open(spike_file_name, 'rb') as spike_file:
    all_spikes = pkl.load(spike_file)
with open(event_file_name, 'rb') as event_file:
    all_events = pkl.load(event_file)

cond_map = {0:'ipsi-hori', 1:'ipsi-vert', 2:'contra-hori', 3:'contra-vert'}

for region in selected_neurons.keys():
    for i in range(4):
        fig = raster_plot(selected_neurons[region][i], all_spikes[region], all_events[region], epoch_window_map)
        fig.suptitle(f'Raster plots for {region}, Top {cond_map[i]} Neurons')
        plt.tight_layout()
        fig.savefig(f'{pca_dir}/Raster_{region}{cond_map[i]}.png', bbox_inches='tight', dpi=200)