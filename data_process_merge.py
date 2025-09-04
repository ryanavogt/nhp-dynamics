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

import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sns.set()
sns.set_style(style='white')

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
# monkey_name_map = {'R': 'Red', 'G': 'Green'}
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp On'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   200]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-100,   500]}}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
events = ['trialRewardDrop', 'trialGraspOn']
binsize = 5
kernel_width = 25
trial_count = 100
load_override_preprocess = True
load_override = True

# Extract All Sessions from their Sorting Notes
file_list = [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*.xlsx')]
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
all_events =    {'M1': {'Cue':[], 'Grasp On':[]},
                 'PMd':{'Cue':[], 'Grasp On':[]},
                 'PMv':{'Cue':[], 'Grasp On':[]}
              }
summary_file_name = f'{summary_dir}/spike_summary_merged{len(monkey_name_map.keys())}.p'
spike_file_name = f'{summary_dir}/spike_list_merged{len(monkey_name_map.keys())}.p'
event_file_name = f'{summary_dir}/event_list_merged{len(monkey_name_map.keys())}.p'
condition_file_name = f'{summary_dir}/condition_list_merged{len(monkey_name_map.keys())}.p'
if os.path.exists(summary_file_name) and not load_override_preprocess:
    with open(summary_file_name, 'rb') as summary_file:
        area_summary_dict = pkl.load(summary_file)
    with open(spike_file_name, 'rb') as spike_file:
        all_spikes = pkl.load(spike_file)
    with open(condition_file_name, 'rb') as condition_file:
        all_conditions = pkl.load(condition_file)
    print('Spike File Loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        print(date)
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
            if area_label not in all_areas:
                all_areas.append(area_label)
                region_spike_list = []
                # all_conditions[area_label] = [trial_data['handOrien'][full_mask]]
                region_condition_list = []
            else:
                region_spike_list = all_spikes[area_label]
                # all_conditions[area_label].append(trial_data['handOrien'][full_mask])
                region_condition_list = all_conditions[area_label]
            for channel in spike_times:
                channel_spikes = spike_times[channel] #Extract spike times for a single channel
                # grasp_times = trial_data['trialGraspOn'][channel_spikes[:, -1].astype(int)-1] #Set Grasp onset time to 0
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
                                    area_summary_dict[region_key][event] = {'spikes':[], 'neurons': 0}
                                event_times = trial_data[event][channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([channel_spikes[spike_mask, 0] + area_summary_dict[region_key][event]['neurons'],
                                                  channel_spikes[spike_mask, 1] - event_times[spike_mask].astype(float)])
                                area_summary_dict[region_key][event]['spikes'].append(event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += channel_neurons
                    for neuron in range(channel_neurons):
                        # region_condition_list.append(channel_conditions[full_mask])
                        region_condition_list.append(channel_conditions)
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
    print('Spike File Saved')

"""
Generate plots of the max spiking rates for each area, side, and orientation.
Scale the rate to the max rate for contralateral side. Apply that scale to ipsilateral side.
"""
# window_range = np.array([-1000, 1000])
skip = False
area_mean_rate = {}
all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map.keys())}.p'
area_mean_filename = f'{summary_dir}/areaMeanRates_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map.keys())}.p'
if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as sdf_file:
        all_sdf_dict = pkl.load(sdf_file)
    with open(area_mean_filename, 'rb') as area_mean_file:
        area_mean_rate = pkl.load(area_mean_file)
    print(f'SDF Dictionary Loaded (Bin: {binsize}, Kernel: {kernel_width})')
else:
    all_sdf_dict = {}
    for area_label in area_summary_dict.keys():
        if skip:
            break
        region_key = f'{area_label}'
        print(region_key)
        lat = region_key[0]
        all_psth = []
        for event in events:
            window_range = np.array(epoch_window_map[event_map[event]]['window'])
            num_bins = int(1.0*(window_range.max() - window_range.min())/binsize)
            if region_key in area_summary_dict:
                area_summary_dict[region_key][event]['spikes'] = np.concatenate(area_summary_dict[region_key][event]['spikes'], axis=1)
            sdf_list = []
            neuron_count = area_summary_dict[region_key][event]['neurons'].astype(int)
            psth_list = np.zeros((num_bins, neuron_count+1))
            for neuron_idx in range(neuron_count):
                neuron_mask = area_summary_dict[region_key][event]['spikes'][0, :] == neuron_idx+1
                neuron_spikes = area_summary_dict[region_key][event]['spikes'][:, neuron_mask]
                neuron_spikes[0, :] = 1
                neuron_psth = gen_psth(neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
                if neuron_idx == 0:
                    psth_list[:, 0] = neuron_psth[1:, 0]
                psth_list[:, neuron_idx+1] = neuron_psth[1:, 1]
            all_psth.append(psth_list)
        all_psth = np.vstack(all_psth)
        area_sdf, _ = gen_sdf(all_psth[:, 1:], w=kernel_width, bin_size=binsize, ftype='Gauss', multi_unit=True)
        all_sdf_dict[region_key] = area_sdf.T
        area_mean_rate[region_key] = all_psth[:, 1:].mean(axis=0)
    with open(all_sdf_filename, 'wb') as sdf_file:
        pkl.dump(all_sdf_dict, sdf_file)
    with open(area_mean_filename, 'wb') as area_mean_file:
        pkl.dump(area_mean_rate, area_mean_file)
area_rates = {'M1': {'horizontal': [], 'vertical': [], 'ipsilateral': [], 'contralateral': []},
              'PMd':{'horizontal': [], 'vertical': [], 'ipsilateral': [], 'contralateral': []},
              'PMv':{'horizontal': [], 'vertical': [], 'ipsilateral': [], 'contralateral': []}}
total_rates = {'M1': [], 'PMd': [], 'PMv': []}
nzero_mask = {}
for region_key in area_mean_rate.keys():
    lat = lat_map[region_key[0]]
    region, orient = region_key[1:].split('_')
    area_mean = area_mean_rate[region_key]
    area_rates[region][orient].append(area_mean)
    area_rates[region][lat].append(area_mean)
    total_rates[region].append(area_mean)
for region in total_rates.keys():
    total_rates[region] = np.vstack(total_rates[region]).sum(axis=0)
    nzero_mask[region] = (total_rates[region]>0)
    total_rates[region] = total_rates[region][nzero_mask[region]]
for region in area_rates.keys():
    print(region)
    for cond in area_rates[region].keys():
        rates = area_rates[region][cond]
        area_rates[region][cond] = np.vstack(rates).sum(axis=0)[nzero_mask[region]]
    a_r = area_rates[region]
    orientation_score = (a_r['vertical'] - a_r['horizontal'])/total_rates[region]
    hemisphere_score = (a_r['contralateral'] - a_r['ipsilateral']) / total_rates[region]
    upper_mosaic = [['histx', '.'],['scatter', 'histy']]
    fig, axs = plt.subplot_mosaic(upper_mosaic,
                                  figsize=(6, 6),
                                  width_ratios=(4, 1.5), height_ratios=(1, 4),
                                  layout='constrained')
    ax = axs['scatter']
    tr = total_rates[region]
    norm = mpl.colors.LogNorm(vmin = np.nanmin(total_rates[region]), vmax = np.nanmax(total_rates[region]))
    scatter_hist(ax=axs['scatter'], ax_histx=axs['histx'], ax_histy=axs['histy'], y=orientation_score,
                 x = hemisphere_score, c=tr, norm='log')
    plt.colorbar(cm.ScalarMappable(norm=norm), label='Total Rate', ax= axs['histy'])
    plt.suptitle(f'Neuron Scores for {region}')
    plt.gcf().text(1, 0.9, in_layout=False, ha='right', va='top',
                   s= f'x mean = {hemisphere_score.mean():.3f}, x std = {hemisphere_score.std():.3f}\ny mean = {orientation_score.mean():.3f}, y std = {orientation_score.std():.3f}')
    ax.set_ylabel(f'Orientation Score (Vertical - Horizontal)')
    ax.set_xlabel(f'Hemisphere Score (Contra - Ipsi)')
    ax.set_xlim([-1.03,1.03])
    ax.set_ylim([-1.03,1.03])

    plt.savefig(f'{summary_dir}/neuronSelection_merged{len(monkey_name_map.keys())}_bin{binsize}_k{kernel_width}_{region}.png',
                bbox_inches='tight', dpi=200)
    f_cum, ax_cum = plt.subplots(figsize=(6,6))
    # ax_cum = axs['cumulative']
    res_hem = sp.stats.ecdf(np.abs(hemisphere_score))
    res_ori = sp.stats.ecdf(np.abs(orientation_score))
    q_hem = res_hem.cdf.quantiles
    q_ori = res_ori.cdf.quantiles
    area_hem = ((q_hem[1:]-q_hem[:-1])*np.arange(1,q_hem.shape[0])).sum()/q_hem.shape[0]
    area_ori = ((q_ori[1:]-q_ori[:-1])*np.arange(1,q_ori.shape[0])).sum()/q_ori.shape[0]
    res_hem.cdf.plot(ax_cum, label=f'abs Hemi, AUC: {area_hem:.3f}', c='b')
    res_ori.cdf.plot(ax_cum, label=f'abs Orient, AUC: {area_ori:.3f}', c ='r')
    # ax_cum.ecdf(np.abs(hemisphere_score), label='abs Hemi Index', c='b')
    # ax_cum.ecdf(np.abs(orientation_score), label='abs Orient Index', c ='r')
    ax_cum.plot([0,1], [0,1],'k--')
    ax_cum.set_ylabel('Cumulative Fraction')
    ax_cum.set_xlabel('abs Hemi and Orient Index')
    ax_cum.set_title(f'{region} Selection Ratio')
    ax_cum.set_xlim([0,1])
    ax_cum.set_ylim([0,1])
    ax_cum.legend()
    plt.savefig(f'{summary_dir}/neuronSelectionCumulative_merged{len(monkey_name_map.keys())}_bin{binsize}_k{kernel_width}_{region}.png',
                bbox_inches='tight', dpi=200)