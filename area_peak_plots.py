import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (usig conda or pip)
import matplotlib.pyplot as plt     #Generating plots
from sig_proc import *
import pandas as pd
import matplotlib as mpl

from sig_proc import *

import seaborn as sns


sns.set()
sns.set_style(style='white')

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
monkey_name_map = {'R': 'Red', 'G': 'Green'}
event_map = {'trialRewardDrop': 'Cue', 'trialReachOn':'Reach', 'trialGraspOn':'Grasp'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Pre-cue':  {'event': 'trialRewardDrop', 'window': [-700,    -100]},
                   'Post-cue': {'event': 'trialRewardDrop', 'window': [0,       100]},
                   'Reach':    {'event': 'trialReachOn',    'window': [-700,    -100]},
                   'Grasp On': {'event': 'trialGraspOn',    'window': [-700,    -100]},
                   'Grasp Off':{'event': 'trialEnd',        'window': [-700,    -400]}}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
events = ['trialRewardDrop', 'trialReachOn', 'trialGraspOn']
binsize = 10
kernel_width = 100
load_override_preprocess = False
load_override = False

# Extract All Sessions from their Sorting Notes
file_list = [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*.xlsx')]
file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1]) #Date of Session
    monkey_labels.append(name[2])#Monkey (G or R)

"""
Aggregate all neurons across all sessions within a single region, separated by lateral 
relation to hand used (contra or ipsi) and orientation of target (horizontal or vertical)
"""

area_summary_dict = {}
all_areas = []


spikes_file_name = f'{summary_dir}/spike_summary.p'
if os.path.exists(spikes_file_name) and not load_override_preprocess:
    with open(spikes_file_name, 'rb') as spike_file:
        area_summary_dict = pkl.load(spike_file)
    print('Spike File Loaded')
else:
    for date, monkey in zip(date_strings, monkey_labels):
        print(date)
        monkey_folder = f'Monkey_{monkey_name_map[monkey]}'
        date_folder = f'{date[0:4]}_{date[4:6]}_{date[6:]}'
        trial_dir = f'Data/Processed/{monkey_folder}/{date_folder}' #Folder where to find the session data
        with open(f'{trial_dir}/trial_data.p', 'rb') as trial_file:
            trial_data = pkl.load(trial_file)
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
            for channel in spike_times:
                channel_spikes = spike_times[channel] #Extract spike times for a single channel
                # grasp_times = trial_data['trialGraspOn'][channel_spikes[:, -1].astype(int)-1] #Set Grasp onset time to 0
                if channel_spikes.shape[0]>0:
                    channel_neurons = channel_spikes[:, 0].max()
                    for o_idx, orient in enumerate(['horizontal', 'vertical']):
                        orientation_mask = (trial_data['handOrien']-1)//2 == o_idx
                        for mod in [0,1]:
                            hand_mask = trial_data['handOrien']%2 == mod
                            spike_mask = np.in1d(channel_spikes[:, -1], np.where(hand_mask*orientation_mask))
                            hand_label = hand_list[mod]
                            if area_hemisphere == hand_label:
                                lateral_label = 'i'
                            else:
                                lateral_label = 'c'
                            region_key = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in area_summary_dict:
                                area_summary_dict[region_key] = {}
                            for event in ['trialRewardDrop', 'trialReachOn', 'trialGraspOn']:
                                if event not in area_summary_dict[region_key]:
                                    area_summary_dict[region_key][event] = {'spikes':[], 'neurons': 0}
                                event_times = trial_data[event][channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([channel_spikes[spike_mask, 0] + area_summary_dict[region_key][event]['neurons'],
                                                  channel_spikes[spike_mask, 1] - event_times[spike_mask].astype(float)])
                                area_summary_dict[region_key][event]['spikes'].append(event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += channel_neurons
    with open(spikes_file_name, 'wb') as spike_file:
        pkl.dump(area_summary_dict, spike_file)
"""
Generate plots of the max spiking rates for each area, side, and orientation.
Scale the rate to the max rate for contralateral side. Apply that scale to ipsilateral side.
"""
window_range = np.array([-1000, 1000])
skip = False
area_max_rate = {}
all_sdf_filename = f'{summary_dir}/sdfDict_bin{binsize}_k{kernel_width}.p'
max_rate_filename = f'{summary_dir}/sdfMaxRate_bin{binsize}_k{kernel_width}.p'
peak_order_filename = f'{summary_dir}/peakOrderDict_bin{binsize}_k{kernel_width}.p'
if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as sdf_file:
        all_sdf_dict = pkl.load(sdf_file)
    with open(max_rate_filename, 'rb') as max_rate_file:
        area_max_rate = pkl.load(max_rate_file)
    with open(peak_order_filename, 'rb') as peak_order_file:
        peak_order_dict = pkl.load(peak_order_file)
    print(f'SDF Dictionary Loaded (Bin: {binsize}, Kernel: {kernel_width})')
else:
    all_sdf_dict = {}
    peak_order_dict = {}
    for area_label in area_summary_dict.keys():
        if skip:
            break
        peak_order_dict[area_label] = {}
        area_max_rate[area_label] = np.zeros(area_summary_dict[area_label][events[0]]['neurons'].astype(int))
        region_key = f'{area_label}'
        print(region_key)
        lat = region_key[0]
        all_sdf_dict[region_key] = {}
        for event in events:
            if region_key in area_summary_dict:
                area_summary_dict[region_key][event]['spikes'] = np.concatenate(area_summary_dict[region_key][event]['spikes'], axis=1)
            sdf_list = []
            # sdf_scaled_list = []
            neuron_count = area_summary_dict[region_key][event]['neurons'].astype(int)
            neuron_peak_times = np.zeros(neuron_count)
            # neuron_scales = np.zeros(neuron_count)
            for neuron_idx in range(neuron_count):
                neuron_mask = area_summary_dict[region_key][event]['spikes'][0, :] == neuron_idx+1
                neuron_spikes = area_summary_dict[region_key][event]['spikes'][:, neuron_mask]
                neuron_spikes[0, :] = 1
                neuron_psth = gen_psth(neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
                neuron_sdf, _ = gen_sdf(neuron_psth, w=kernel_width, bin_size=binsize, ftype='Gauss', multi_unit=False)
                # neuron_scales[neuron_idx] = neuron_sdf[:].max()
                peak_location = neuron_sdf[:].argmax()
                neuron_peak_times[neuron_idx] = peak_location
                time_scale = neuron_psth[:,0]
                # sdf_scaled_list.append(neuron_sdf[:, 0].T/neuron_scales[neuron_idx])
                sdf_list.append(neuron_sdf[:, 0].T)
            peak_order = np.argsort(neuron_peak_times)
            peak_order_dict[area_label][event] = peak_order
            # area_scale_sdf = np.vstack(sdf_scaled_list)
            area_sdf = np.vstack(sdf_list)
            all_sdf_dict[region_key][event] = area_sdf
            event_max_rate = np.max(area_sdf, axis=1)
            area_max_rate[area_label] = np.maximum(event_max_rate, area_max_rate[area_label])

            # Area Relative Spike Density Plots
            # y, x = np.mgrid[1:neuron_count+2:1, window_range[0]:window_range[1]+2*binsize:binsize]
            # plt.figure(figsize = (3, 8))
            # plt.pcolor(x, y, area_scale_sdf[peak_order],
            #            cmap='inferno', norm = mpl.colors.Normalize(vmin=0, vmax=1.5))
            # plt.axvline(x=0, color='w', linestyle=':')
            # plt.gca().invert_yaxis()
            #
            # area_title, orientation = region_key.split('_')
            # plt.title(f'{area_title}, {orientation}, {event}')
            # plt.ylabel('Neuron No.')
            # plt.xlabel(f'Time from Event (ms)')
            # plt.savefig(f'{summary_dir}/NeuronSpikes_{region_key}_event{event}.png', bbox_inches='tight')
            # plt.close()

    with open(all_sdf_filename, 'wb') as sdf_file:
        pkl.dump(all_sdf_dict, sdf_file)
    with open(max_rate_filename, 'wb') as max_rate_file:
        pkl.dump(area_max_rate, max_rate_file)
    with open(peak_order_filename, 'wb') as peak_order_file:
        pkl.dump(peak_order_dict, peak_order_file)

"""
Generate plots after loading data
"""
plot_dict = {}
max_dict = {}
for area_label in area_summary_dict.keys():
    area_key, orientation = area_label.split('_')
    lat, region = area_key[0], area_key[1:]
    peak_order = peak_order_dict[area_label]
    if orientation not in plot_dict.keys():
        plot_dict[orientation] = {}
        max_dict[orientation] = {}
    if region not in plot_dict[orientation].keys():
        plot_dict[orientation][region] = {}
        max_dict[orientation][region] = {}
    for event in events:
        if event not in plot_dict[orientation][region]:
            plot_dict[orientation][region][event] = {}
            max_dict[orientation][region][event] = {}
        neuron_count = area_summary_dict[area_label][event]['neurons']
        area_sdf = all_sdf_dict[area_label][event]
        plot_dict[orientation][region][event][lat] = area_sdf
        max_dict[orientation][region][event][lat] = {'max':area_sdf.max(axis=1), 'order': peak_order[event]}

skip_plots = True
neuron_plot_dir = f'{summary_dir}/Neuron Plots'
if not os.path.exists(neuron_plot_dir):
    os.mkdir(neuron_plot_dir)
for orientation in plot_dict.keys():
    if skip_plots:
        break
    for region in plot_dict[orientation].keys():
        neuron_count = area_summary_dict[f'c{region}_{orientation}']['trialGraspOn']['neurons'].astype(int)
        for event in plot_dict[orientation][region].keys():
            peak_order = max_dict[orientation][region][event]['c']['order']
            max_rate = max_dict[orientation][region][event]['c']['max']
            fig, axs = plt.subplots(1, 2, figsize=(8, 8))
            for idx, lat in enumerate(['c', 'i']):
                a = plot_dict[orientation][region][event][lat].T
                scaled_sdf = np.divide(a, max_rate, where= max_rate>0)
                y, x = np.mgrid[1:neuron_count + 2:1,
                       window_range[0]/1000:window_range[1]/1000 + 2 * binsize/1000:binsize/1000]
                # plt.figure(figsize = (3, 8))
                axs[idx].pcolor(x, y, scaled_sdf.T[peak_order],
                           cmap='inferno', norm = mpl.colors.Normalize(vmin=0, vmax=1.5))
                axs[idx].axvline(x=0, color='w', linestyle=':')
                axs[idx].invert_yaxis()
                axs[idx].title.set_text(lat_map[lat].capitalize())
                axs[idx].set_xlabel(f'Time from Event (s)')
            fig.suptitle(f'{region}, {orientation.capitalize()}, {event_map[event].capitalize()}')
            axs[0].set_ylabel('Neuron No.')
            plt.savefig(f'{neuron_plot_dir}/NeuronSpikes_{region}_{orientation}_{event_map[event]}.png', bbox_inches='tight')
            plt.close()


"""
Identify epoch of maximal spike rate and find proportion of neurons with peak in each epoch.
"""
epoch_names = ['trialRewardDrop', 'trialReachOn', 'trialGraspOn']
epoch_windows = {'trialRewardDrop': [0, 100], 'trialReachOn': [-100, 200], 'trialGraspOn':[-100, 500]}
epoch_windows = {'trialRewardDrop': [0, 100], 'trialReachOn': [-100, 200], 'trialGraspOn':[-100, 500]}
# events = ['trialRewardDrop', 'trialReachOn', 'trialGraspOn', 'trialGraspOff']
event_spike_maxes = {}
event_neuron_max = {}
neuron_peaks_filename = f'{summary_dir}/neuron_peaks.p'

with open(all_sdf_filename, 'rb') as all_sdf_filename:
    all_sdf_dict = pkl.load(all_sdf_filename)
with open(max_rate_filename, 'rb') as max_rate_file:
    area_max_rate = pkl.load(max_rate_file)

for area_label in area_summary_dict.keys():
    max_rate_mask = {}
    full_window = np.arange(-1000, 1000+binsize, binsize)
    area_max = area_max_rate[area_label]
    region_key = f'{area_label}'
    region, orient = area_label.split('_')
    lat, region = region[0], region[1:]
    event_spike_maxes[region_key] = {}
    if orient not in event_neuron_max.keys():
        event_neuron_max[orient] = {}
    if region not in event_neuron_max[orient].keys():
        event_neuron_max[orient][region] = {'proportions': {}, 'neurons':0}
    for event in events:
        event_window = epoch_windows[event]
        event_mask = (full_window>event_window[0]) * (full_window<event_window[1])
        area_sdf = all_sdf_dict[region_key][event][:, event_mask]
        event_spike_maxes[region_key][event] = area_sdf.max(axis=1)
    event_neuron_peaks = np.vstack(event_spike_maxes[region_key].values())
    # event_neuron_max[region_key] = event_neuron_peaks == area_max_rate[region_key]
    # for idx, event in enumerate(events):
    #     event_peak_proportions[event] = np.average(event_neuron_max[region_key]==idx)
    event_peak_proportions = np.average(event_neuron_peaks == area_max_rate[region_key], axis=1)
    event_neuron_max[orient][region]['neurons'] = event_neuron_peaks.shape[1]
    event_neuron_max[orient][region]['proportions'][lat_map[lat]]= event_peak_proportions

for orientation in event_neuron_max.keys():
    # plt.figure(figsize=(5,5))
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharey='row')
    x = np.arange(len(events))
    max_height = 0
    for idx, region in enumerate(event_neuron_max[orientation].keys()):
        ax = axs[idx]
        neurons = event_neuron_max[orientation][region]['neurons']
        width = 0.4
        multiplier = 0
        for side, proportion in event_neuron_max[orientation][region]['proportions'].items():
            offset = width*multiplier
            max_height = max(proportion.max()*100, max_height)
            rects = ax.bar(x+offset, proportion*100, width, label=side)
            # ax.bar_label(rects, padding=3)
            multiplier+= 1
        # ax.set_ylabel('Neurons(%)')
        ax.set_xlabel('Epochs')
        ax.set_title(f'{region}')
        ax.set_xticks(x+width*(multiplier-1)/2, [event_map[e] for e in events])
        ax.text(x=0 - width / 2, y=max_height, s=f'n={neurons}', fontsize=15, color='black',
                 bbox=dict(facecolor='white', alpha=0.5))
    handles, labels = plt.gca().get_legend_handles_labels()
    f.legend(handles, labels, loc=(0.5, 0), ncols=2)
    # ax.legend(loc='upper left', ncols=3)
    sns.move_legend(f, "lower center", bbox_to_anchor=(.5, -.06), ncol=3)
    axs[0].set_ylabel('Neurons (%)')
    f.suptitle(f'Epoch of Max Discharge, {orientation.capitalize()}')
    sns.despine()
    plt.savefig(f'{summary_dir}/MaxDischargeProportions_{orientation}.png', bbox_inches='tight')

with open(neuron_peaks_filename, 'wb') as neuron_peaks_file:
    pkl.dump(event_neuron_max, neuron_peaks_file)

"""
Perform t-tests to determine if neurons were modulated 
"""
full_window = np.arange(-1000, 1000+binsize, binsize)
baseline_epoch = 'Pre-cue'
base_window= epoch_window_map[baseline_epoch]['window']
baseline_mask = (full_window>base_window[0]) * (full_window<base_window[1])
p_score = 0.05
modulation_dict = {}
for region in all_sdf_dict.keys():
    area, orientation = region.split('_')
    lat, area = area[0], area[1:]
    if orientation not in modulation_dict.keys():
        modulation_dict[orientation] = {}
    baseline_sdf = all_sdf_dict[region][epoch_window_map[baseline_epoch]['event']][:, baseline_mask]
    event_modulations = []
    for event in all_sdf_dict[region].keys():
        mod_tuple = t_test(baseline_sdf, all_sdf_dict[region][event], q=p_score/2)
        event_modulations.append(mod_tuple[0])
    event_modulations=np.vstack(event_modulations)
    if area not in modulation_dict[orientation].keys():
        modulation_dict[orientation][area] = {}
    modulation_dict[orientation][area][lat_map[lat]] = event_modulations

hand_use_mod = {}
for orientation in modulation_dict.keys():
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharey='row')
    hand_use_mod[orientation] = {}
    for idx, region in enumerate(modulation_dict[orientation].keys()):
        hand_use_mod[orientation][region] = {}
        # plt.figure(figsize=(5, 5))
        ax = axs[idx]
        multiplier = 0
        hand_mod = []
        for side, modulation in modulation_dict[orientation][region].items():
            hand_mod.append(modulation)
            offset = width*multiplier
            mod_perc = np.average(modulation, axis=1)
            # max_height = max(mod_perc.max()*100, max_height)
            max_height = 100
            rects = ax.bar(x+offset, mod_perc*100, width, label=side)
            # ax.bar_label(rects, padding=3)
            multiplier+= 1
        # Compute the share of modulated neurons across each hand
        hand_nonSpecific = ~np.bitwise_xor(hand_mod[0], hand_mod[1])
        ipsi_only = hand_mod[0] * ~hand_nonSpecific
        contra_only = hand_mod[1] * ~hand_nonSpecific
        hand_use_mod[orientation][region] = {'Ipsilateral': ipsi_only, 'Contralateral': contra_only,
                                             'Hand non-specific': hand_nonSpecific}
        ax.set_ylabel('Neurons(%)')
        ax.set_ylim([0, max_height])
        ax.set_xlabel('Epochs')
        ax.set_title(f'{region}')
        ax.set_xticks(x + width * (multiplier - 1) / 2, [event_map[e] for e in events])
        # ax.legend(loc='upper left', ncols=3)
    handles, labels = plt.gca().get_legend_handles_labels()
    f.legend(handles, labels, loc = (0.5, 0), ncols=3)
    sns.move_legend(f, "lower center", bbox_to_anchor=(.5, -.06), ncol=3)
    sns.despine()
    f.suptitle(f'Modulation by Epoch, {orientation.capitalize()}')
    plt.savefig(f'{summary_dir}/Modulation_{orientation}.png', bbox_inches='tight')


width = 0.3
for orientation in modulation_dict.keys():
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,5), sharey='row')
    for idx, region in enumerate(hand_use_mod[orientation].keys()):
        # plt.figure(figsize=(5, 5))
        ax = axs[idx]
        multiplier = 0
        for side, modulation in hand_use_mod[orientation][region].items():
            offset = width*multiplier
            hand_mod_perc = np.average(modulation, axis=1)
            # max_height = max(mod_perc.max()*100, max_height)
            max_height = 100
            rects = ax.bar(x+offset, hand_mod_perc*100, width, label=side)
            # ax.bar_label(rects, padding=3)
            multiplier+= 1
        # ax.set_ylabel('Neurons(%)')
        ax.set_ylim([0, max_height])
        ax.set_xlabel('Epochs')
        ax.set_title(f'{region}')
        ax.set_xticks(x + width * (multiplier - 1) / 2, [event_map[e] for e in events])
    handles, labels = plt.gca().get_legend_handles_labels()
    f.legend(handles, labels, loc=(0.5, 0), ncols=3)
    sns.move_legend(f, "lower center", bbox_to_anchor=(.5, -.06), ncol=3)
    axs[0].set_ylabel('Epoch Modulated Neurons (%)')
    f.suptitle(f'Modulation by Hand Used, {orientation.capitalize()}')
    sns.despine()
    plt.savefig(f'{summary_dir}/HandModulation_{orientation}.png', bbox_inches='tight')