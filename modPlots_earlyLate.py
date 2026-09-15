import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob
import matplotlib as mpl

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import numpy as np

from sig_proc import *
import pandas as pd
import matplotlib as mpl
from plot_utils import scale_lightness, add_headers
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from scipy.spatial.distance import pdist
from math import ceil

from sig_proc import *

import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sns.set()
sns.set_style(style='white')

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp On'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Pre-cue':  {'event': 'trialRewardDrop', 'window': [-300,   -100]},
                    'Cue':      {'event': 'trialRewardDrop', 'window': [0,   200]},
                    'Grasp On': {'event': 'trialGraspOn',    'window': [-100,   300]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary/earlyLate'
save_dir = summary_dir
# save_dir=f'{summary_dir}/earlyLate'

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
events = ['trialRewardDrop', 'trialGraspOn']
binsize = 5
kernel_width = 25
load_override_preprocess = False
load_override = False

# Extract All Sessions from their Sorting Notes
file_list = [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*.xlsx')]
file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])    #Date of Session
    monkey_labels.append(name[2])   #Monkey (G, R, Y, or B)

"""
Aggregate all neurons across all sessions within a single region, separated by lateral 
relation to hand used (contra or ipsi) and orientation of target (horizontal or vertical)
"""

area_summary_dict = {}
all_areas = []

spike_dict_filename = f'{summary_dir}/spike_list_earlyLate.p'
with open(spike_dict_filename, 'rb') as spike_dict_file:
    spike_dict = pkl.load(spike_dict_file)

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
                            spike_mask = np.isin(channel_spikes[:, -1], np.where(hand_mask*orientation_mask))
                            hand_label = hand_list[mod]
                            if area_hemisphere == hand_label:
                                lateral_label = 'i'
                            else:
                                lateral_label = 'c'
                            region_key = f'{lateral_label}{area_label}_{orient}'
                            if region_key not in area_summary_dict:
                                area_summary_dict[region_key] = {}
                            for event in events:
                                if event not in area_summary_dict[region_key]:
                                    area_summary_dict[region_key][event] = {'spikes':[], 'neurons': 0, 'event_times':[]}
                                event_times = trial_data[event][channel_spikes[:, -1].astype(int) - 1]
                                event_spikes = np.vstack([channel_spikes[spike_mask, 0] + area_summary_dict[region_key][event]['neurons'],
                                                  channel_spikes[spike_mask, 1] - event_times[spike_mask].astype(float)])
                                area_summary_dict[region_key][event]['spikes'].append(event_spikes)
                                area_summary_dict[region_key][event]['neurons'] += channel_neurons
                                area_summary_dict[region_key][event]['event_times'].append(event_times)
    with open(spikes_file_name, 'wb') as spike_file:
        pkl.dump(area_summary_dict, spike_file)
"""
Generate plots of the max spiking rates for each area, side, and orientation.
Scale the rate to the max rate for contralateral side. Apply that scale to ipsilateral side.
"""
window_range = np.array([-1000, 1000])
area_kernel = 25
skip = False
area_max_rate = {}
all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_earlyLate.p'
max_rate_filename = f'{summary_dir}/sdfMaxRate_bin{binsize}_k{kernel_width}_earlyLate.p'
peak_order_filename = f'{summary_dir}/peakOrderDict_bin{binsize}_k{kernel_width}_earlyLate.p'
windowed_spike_filename = f'{summary_dir}/windowedSpikes_earlyLate.p'
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
            # spike_list = []
            sdf_list = []
            psth_list = []
            neuron_count = area_summary_dict[region_key][event]['neurons'].astype(int)
            neuron_peak_times = np.zeros(neuron_count)
            for neuron_idx in range(neuron_count):
                neuron_mask = area_summary_dict[region_key][event]['spikes'][0, :] == neuron_idx+1
                neuron_spikes = area_summary_dict[region_key][event]['spikes'][:, neuron_mask]
                neuron_spikes[0, :] = 1
                neuron_psth = gen_psth(neuron_spikes.T, binsize=binsize, window=window_range, neurons=1)
                psth_list.append(neuron_psth)
                neuron_sdf, _ = gen_sdf(neuron_psth, w=area_kernel, bin_size=binsize, ftype='Gauss', multi_unit=False)
                peak_location = neuron_sdf[:].argmax()
                neuron_peak_times[neuron_idx] = peak_location
                time_scale = neuron_psth[:,0]
                sdf_list.append(neuron_sdf[:, 0].T)
            peak_order = np.argsort(neuron_peak_times)
            peak_order_dict[area_label][event] = peak_order
            area_sdf = np.hstack(sdf_list)
            all_sdf_dict[region_key][event] = area_sdf
            event_max_rate = np.max(area_sdf, axis=0)
            area_max_rate[area_label] = np.maximum(event_max_rate, area_max_rate[area_label])
    with open(all_sdf_filename, 'wb') as sdf_file:
        pkl.dump(all_sdf_dict, sdf_file)
    with open(max_rate_filename, 'wb') as max_rate_file:
        pkl.dump(area_max_rate, max_rate_file)
    with open(peak_order_filename, 'wb') as peak_order_file:
        pkl.dump(peak_order_dict, peak_order_file)
    # with open(windowed_spike_filename, 'wb') as windowed_spike_file:
    #     pkl.dump(peak_order_dict, peak_order_file)

"""
Generate plots after loading data
"""
plot_dict = {}
max_dict = {}
clustering_dict = {}
sns.set_style("ticks")
# sns.set_style("whitegrid")
plots = {}
for region in ['M1', 'PMd', 'PMv']:
    plots[region] = {}
for area_label in area_summary_dict.keys():
    area_key, orientation = area_label.split('_')
    lat, region = area_key[0], area_key[1:]
    peak_order = peak_order_dict[area_label]
    if region not in plot_dict.keys():
        plot_dict[region]= {}
        max_dict[region] = {}
        clustering_dict[region]={}
    if orientation not in plot_dict[region].keys():
        plot_dict[region][orientation] = {}
        max_dict[region][orientation] = {}
        clustering_dict[region][orientation] = {}
        plots[region][orientation] = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    # dend_fig, dend_axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
    dend_fig, dend_axs = plots[region][orientation]
    if lat == 'c':
        lat_idx=0
        y_label=True
    else:
        lat_idx=1
        y_label=False
    for event, dend_ax in zip(events, dend_axs):
        if event not in plot_dict[region][orientation]:
            plot_dict[region][orientation][event] = {}
            max_dict[region][orientation][event] = {}
            clustering_dict[region][orientation][event] = {}
        neuron_count = area_summary_dict[area_label][event]['neurons']
        area_sdf = all_sdf_dict[area_label][event]
        norm_sdf = area_sdf.T/(np.expand_dims(area_sdf.max(axis=0), 1).repeat(area_sdf.shape[0], 1)+.000001)
        y = pdist(norm_sdf)
        Z = ward(y)
        if event == 'trialGraspOn':
            dend_ax = dend_axs[0][lat_idx]
            clust_ax = dend_axs[1][lat_idx]
            dists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
            clusters = []
            for dist in dists:
                dist_cluster = fcluster(Z, dist, criterion='distance')
                clusters.append(len(np.unique(fcluster(Z, dist, criterion='distance'))))
            clust_ax.plot(dists, clusters)
            clust_ax.set_yscale('log')
            clust_ax.set_yticks([1, 5, 10, 20, 50, 100, 200])
            clust_ax.set_xlabel('Distance')
            dendrogram(Z, ax=dend_ax)
            dend_ax.set_title(f'{lat_map[lat].capitalize()}')
            if y_label:
                dend_ax.set_ylabel('Distance')
                clust_ax.set_ylabel('No. of Clusters')
        plot_dict[region][orientation][event][lat] = area_sdf
        max_dict[region][orientation][event][lat] = {'max':area_sdf.max(axis=0), 'order': peak_order[event]}
        clustering_dict[region][orientation][event][lat] = {'dist':y, 'Z':Z}
        # dend_ax.set_xticks(dists)
for region in plots.keys():
    for orientation in plots[region].keys():
        dend_fig, _ = plots[region][orientation]
        # dend_fig.tight_layout()
        dend_fig.suptitle(f'Clustering for {region}, Grasp Onset\n{orientation.capitalize()} Orientation')
        dend_fig.savefig(f'{summary_dir}/NumberOfClusters_All.png', dpi=200)
        plt.close()

joint_sdf_dict= {}
wide_sdf_dict = {}
cluster_dict = {'M1':{}, 'PMd':{}, 'PMv':{}}
dend_fig, dend_axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
epoch_windows = {'trialRewardDrop': [-100, 500], 'trialGraspOn':[-300, 300]}
for reg_idx, region in enumerate(plot_dict.keys()):
    joint_sdfs =[]
    wide_sdfs = []
    for lat in ['c', 'i']:
        for orientation in ['horizontal', 'vertical']:
            window_idcs = ((epoch_window_map['Grasp On']['window'] - window_range[0]) / binsize).astype(np.int_)
            wide_window_idcs = ((np.array([-500, 500])-window_range[0])/binsize).astype(np.int_)
            joint_sdfs.append(plot_dict[region][orientation]['trialGraspOn'][lat][window_idcs[0]:window_idcs[1]])
            wide_sdfs.append(plot_dict[region][orientation]['trialGraspOn'][lat][wide_window_idcs[0]:wide_window_idcs[1]])
    joint_sdf=np.vstack(joint_sdfs)
    sdf_max = joint_sdf.max(axis=0) + .000001
    wide_sdf = np.vstack([sdf for sdf in wide_sdfs]).T
    wide_sdf_max = wide_sdf.max(axis=1) + .000001
    wide_sdf = wide_sdf/(np.expand_dims(wide_sdf_max, 1).repeat(wide_sdf.shape[1], 1))
    norm_sdf = joint_sdf.T/(np.expand_dims(joint_sdf.max(axis=0), 1).repeat(joint_sdf.shape[0], 1)+.000001)
    joint_sdf_dict[region]=norm_sdf
    wide_sdf_dict[region] =wide_sdf
    y = pdist(norm_sdf)
    Z = ward(y)
    dend_ax = dend_axs[0][reg_idx]
    clust_ax = dend_axs[1][reg_idx]
    dists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22.5, 25, 27.5, 30, 35, 40]
    clusters = []
    for dist in dists:
        dist_cluster = fcluster(Z, dist, criterion='distance')
        n_clust = len(np.unique(dist_cluster))
        clusters.append(n_clust)
    for num_clust in n_clusters:
        cluster_dict[region][num_clust] = fcluster(Z, num_clust, criterion='maxclust')
    clust_ax.plot(dists, clusters)
    clust_ax.set_yscale('log')
    clust_ax.set_yticks([1, 5, 10, 20, 50, 100, 200])
    clust_ax.set_xlabel('Distance')
    dendrogram(Z, ax=dend_ax)
add_headers(dend_fig, col_headers=list(plot_dict.keys()), row_headers=['Distance', 'No. Clusters'])
dend_fig.suptitle(f'Grasp Clustering')
dend_fig.savefig(f'{summary_dir}/AllClusters_{region}.png', dpi=200)

cluster_file_name = f'{summary_dir}/Clusters.p'
with open(cluster_file_name, 'wb') as cluster_file:
    pkl.dump(cluster_dict, cluster_file)
cluster = True
for n_clust in n_clusters:
    if not cluster:
        continue
    clust_fig, clust_axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    cond_order = ['c_h', 'c_v', 'i_h', 'i_v']
    for reg_idx, region in enumerate(plot_dict.keys()):
        clust_ax = clust_axs[reg_idx]
        clust_ax.set_title(region, y=1.05)
        joint_sdf = wide_sdf_dict[region]
        if n_clust not in cluster_dict[region].keys():
            continue
        clusters = cluster_dict[region][n_clust]
        plot_sdf = np.zeros_like(joint_sdf)
        start_idx = 0
        boundaries = [0]
        cond_boundaries = [0]
        time_steps = plot_sdf.shape[1]
        n_neurons = plot_sdf.shape[0]
        cond_len = int(joint_sdf.shape[1]/4)*binsize
        x = np.arange(time_steps)*binsize
        y = np.arange(n_neurons)
        xv, yv = np.meshgrid(x,y)
        for i in np.unique(clusters):
            idcs = (clusters == i)
            end_idx = start_idx + sum(idcs)
            plot_sdf[start_idx:end_idx] = joint_sdf[idcs]
            boundaries.append(end_idx)
            start_idx=end_idx
        clust_ax.pcolor(xv,yv, plot_sdf)
        for i_cond, cond in enumerate(cond_order):
            clust_ax.text(cond_len*(i_cond +1/2), n_neurons+3, cond, horizontalalignment='center')
            cond_boundaries.append((i_cond+1)*cond_len)
            clust_ax.axvline(cond_boundaries[-1], ls= '--', c='white', alpha=0.5)
            clust_ax.axvline((cond_boundaries[-2]+cond_boundaries[-1])/2, c='white', alpha=0.5)
        for loc in boundaries:
            clust_ax.axhline(loc, c='white')
        clust_ax.set_ylim([0, n_neurons])
        clust_ax.set_xlim([0, x.max()])
        clust_ax.set_xlabel('Time (ms)')
        clust_ax.set_xticks(cond_boundaries, labels = [])
        # clust_ax.hlines([boundaries], c='white')

    clust_fig.suptitle(f'Neuron Clusters in each Region, {n_clust} Clusters')
    clust_fig.savefig(f'{summary_dir}/NeuronClusters_{n_clust}.png', dpi=200, bbox_inches='tight')

"""
Identify epoch of maximal spike rate and find proportion of neurons with peak in each epoch.
"""
epoch_names = ['trialRewardDrop', 'trialGraspOn']
epoch_windows = {'trialRewardDrop': [-100, 500], 'trialGraspOn':[-400, 400]}
event_spike_maxes = {}
event_neuron_max = {}
neuron_peaks_filename = f'{summary_dir}/neuron_peaks_earlyLate.p'

with open(all_sdf_filename, 'rb') as all_sdf_file:
    all_sdf_dict = pkl.load(all_sdf_file)
with open(max_rate_filename, 'rb') as max_rate_file:
    area_max_rate = pkl.load(max_rate_file)

area_merged_rates = {'M1':None, 'PMd':None, 'PMv':None}
for area_label in area_max_rate.keys():
    region, orient = area_label.split('_')
    lat, region = region[0], region[1:]
    area_max = area_max_rate[area_label]
    if area_merged_rates[region] is None:
        area_merged_rates[region] = np.zeros_like(area_max)
    new_maxes = np.max(np.stack([area_max, area_merged_rates[region]]), axis=0)
    area_merged_rates[region] = new_maxes

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
    for epoch in epoch_window_map.keys():
        if epoch == 'Pre-cue':
            continue
        event = epoch_window_map[epoch]['event']
        event_window = epoch_window_map[epoch]['window']
        event_mask = (full_window>event_window[0]) * (full_window<=event_window[1])
        area_sdf = all_sdf_dict[region_key][event][event_mask]
        event_spike_maxes[region_key][epoch] = area_sdf.max(axis=0)
    event_neuron_peaks = np.vstack(list(event_spike_maxes[region_key].values()))
    # event_neuron_max[region_key] = event_neuron_peaks == area_max_rate[region_key]
    # for idx, event in enumerate(events):
    #     event_peak_proportions[event] = np.average(event_neuron_max[region_key]==idx)
    event_peaks = (event_neuron_peaks == area_merged_rates[region])
    # event_peaks[-2] = np.logical_or(event_peaks[-2], event_peaks[-1])
    event_peak_proportions = np.average(event_peaks, axis=1)
    event_neuron_max[orient][region]['neurons'] = event_neuron_peaks.shape[1]
    event_neuron_max[orient][region]['proportions'][lat_map[lat]]= event_peak_proportions


with open(neuron_peaks_filename, 'wb') as neuron_peaks_file:
    pkl.dump(event_neuron_max, neuron_peaks_file)

# event_padding = 250 #ms to add on either end of each event window
"""
Perform t-tests to determine if neurons were modulated 
"""
full_window = np.arange(-1000, 1000+binsize, binsize)
baseline_epoch = 'Pre-cue'
base_window= epoch_window_map[baseline_epoch]['window']
baseline_mask = (full_window>base_window[0]) * (full_window<base_window[1])
event_masks = {}
slope_masks = {}
event_times = {}
for event in ['Cue', 'Grasp On']:
    event_window = epoch_window_map[event]['window']
    event_masks[event] = (full_window>event_window[0]) * (full_window<event_window[1])
    slope_masks[event] = (full_window>event_window[0]) * (full_window<event_window[1]) #expanded range for slope computation
    event_times[event] = int(-1*(event_window[0])/binsize) #time at which event occurs in this frame
p_score = 0.05
modulation_dict = {}
up_down_modulation = {}
tVals_dict = {}
mod_sdfs = {}
for region in all_sdf_dict.keys():
    area, orientation = region.split('_')
    lat, area = area[0], area[1:]
    if orientation not in modulation_dict.keys():
        modulation_dict[orientation] = {}
        up_down_modulation[orientation] = {}
        tVals_dict[orientation] = {}
        mod_sdfs[orientation] = {}
    mod_sdfs[orientation][region] = {}
    baseline_sdf = all_sdf_dict[region][epoch_window_map[baseline_epoch]['event']][baseline_mask]
    event_modulations = []
    event_tVals = []
    event_max_slopes = []
    event_sdfs = []
    for event in all_sdf_dict[region].keys():
        event_sdf = all_sdf_dict[region][event][event_masks[event_map[event]]]
        slope_sdf = all_sdf_dict[region][event][slope_masks[event_map[event]]]
        event_slopes = slope_sdf[1:]-slope_sdf[:-1]
        event_max_slopes.append({'max': (event_slopes.max(axis=0), event_slopes.argmax(axis=0)),
                                    'min': (event_slopes.min(axis=0), event_slopes.argmin(axis=0)),
                                    'slopes':event_slopes, 'sdf':event_sdf, 'baseline': baseline_sdf})
        mod_tuple = t_test(baseline_sdf.T, event_sdf.T, q=p_score/2, paired=True)
        event_modulations.append(mod_tuple[0])
        event_tVals.append(mod_tuple[1])
        event_sdfs.append(event_sdf)
    event_modulations=np.vstack(event_modulations)
    event_tVals = np.vstack(event_tVals)
    #Add storage for up-modulation and down-modulation
    if area not in modulation_dict[orientation].keys():
        modulation_dict[orientation][area] = {}
        up_down_modulation[orientation][area] = {}
        tVals_dict[orientation][area] = {}
        mod_sdfs[orientation][area] = {}

    modulation_dict[orientation][area][lat_map[lat]] = {'modulations':event_modulations, 'tVals':event_tVals,
                                                        'event_slopes':event_max_slopes, 'event_sdfs': event_sdfs}
    up_down_modulation[orientation][area][lat_map[lat]] = np.stack([event_modulations*event_tVals<0, event_modulations*event_tVals >0])
    tVals_dict[orientation][area][lat_map[lat]] = event_tVals
    mod_sdfs[orientation][area][lat_map[lat]] = event_sdfs

merged_mods = {}
orients = ['horizontal', 'vertical']
for area in modulation_dict['horizontal'].keys():
    if area not in merged_mods.keys():
        merged_mods[area] = {}
    for lat in modulation_dict['horizontal'][area].keys():
        if lat not in merged_mods[area].keys():
            merged_mods[area][lat] = {}
        joint_baseline_sdf = np.vstack([modulation_dict[orient][area][lat]['event_slopes'][1]['baseline'] for
                                        orient in orients])
        merged_mods[area][lat] = {}
        for i, event in enumerate(['Cue', 'Grasp']):
            joint_event_sdf = np.vstack([modulation_dict[orient][area][lat]['event_slopes'][i]['sdf'] for
                                            orient in orients])
            # joint_grasp_sdf = np.vstack([modulation_dict[orient][area][lat]['event_slopes'][1]['sdf'] for
            #                            orient in orients])
            joint_mod_tuple = t_test(joint_baseline_sdf.T, joint_event_sdf.T, q=p_score/2, paired=True)
            merged_mods[area][lat][event] = {'modulations':joint_mod_tuple[0], 'tVals':joint_mod_tuple[1],
                                             'event_sdfs': joint_event_sdf, 'base_sdfs': joint_baseline_sdf}

n_cols = 10
lat_colors = {'contralateral': 'tab:blue', 'ipsilateral':'tab:orange'}
plot_sdfs = False
if plot_sdfs:
    for area in merged_mods.keys():
        n_neurons = merged_mods[area]['contralateral']['Grasp']['base_sdfs'].shape[1]
        n_rows = ceil(n_neurons/n_cols)
        sdf_fig, sdf_axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 4*n_rows))
        sdf_axs = sdf_axs.flatten()
        for neuron in np.arange(n_neurons):
            sdf_ax = sdf_axs[neuron]
            for lat in merged_mods[area].keys():
                event_sdf = merged_mods[area][lat]['Grasp']['event_sdfs']
                sdf_ax.set_title(f'Neuron {neuron+1}')
                sdf_ax.plot(event_sdf[:,neuron], color=lat_colors[lat], label=lat)
                sdf_ax.axhline(event_sdf[:,neuron].mean(), c=lat_colors[lat], ls='--', alpha=0.5)
        sdf_fig.suptitle(f'SDF by Lat for {area}')
        sdf_fig.tight_layout()
        sdf_fig.savefig(f'{save_dir}/GraspSDFs_{area}.png', dpi=200, bbox_inches='tight')

merged_mod_types = {}
p = 0.05
for area in merged_mods.keys():
    merged_mod_types[area]= {}
    lats = list(merged_mods[area].keys())
    area_mod_dict = merged_mods[area]
    event_mod_dict = {}
    for l_idx, lat in enumerate(lats):
        merged_mod_types[area][lat] = {}
        for event in merged_mods[area][lat].keys():
            event_mods = area_mod_dict[lat][event]['modulations'] #Which neurons modulated for this hand
            both_mods  = area_mod_dict[lats[1-l_idx]][event]['modulations']*event_mods #Neurons which also modulated for other hand
            event_tVals= area_mod_dict[lat][event]['tVals']
            other_tVals= area_mod_dict[lats[1-l_idx]][event]['tVals']
            same_sign = np.equal(np.sign(event_tVals), np.sign(other_tVals)) #Neurons that change in same direction
            up_down = np.stack([event_mods * event_tVals < 0, event_mods * event_tVals > 0]) #Track upward/downward direction
            event_sdf = area_mod_dict[lat][event]['event_sdfs']
            other_sdf = area_mod_dict[lats[1 - l_idx]][event]['event_sdfs']
            mod_vals = np.zeros_like(up_down[0]).astype(bool) #Track
            for m_idx, mod in enumerate(['down', 'up']):
                mod_mask = up_down[m_idx]
                if mod == 'down':
                    modulation, t_vals = t_test(other_sdf[:, mod_mask].T, event_sdf[:, mod_mask].T, type='less', q=p)
                else:
                    modulation, t_vals = t_test(other_sdf[:, mod_mask].T, event_sdf[:, mod_mask].T, type='greater', q=p)
                mod_vals[mod_mask] = modulation
            merged_mod_types[area][lat][event] = mod_vals * both_mods * same_sign
            if l_idx ==0:
                event_mod_dict[event] = {'both': both_mods,'sign': same_sign}
    merged_mod_types[area]['opp'] = {}
    for event in merged_mods[area][lat].keys():
        both_mods = event_mod_dict[event]['both']
        same_sign = event_mod_dict[event]['sign']
        merged_mod_types[area]['opp'][event] = both_mods * (~same_sign)

hand_use_mod = {}
epoch_indices = {}
start_time=0
event_padding=0
event_names = ['Cue', 'Grasp On']
for i, event in enumerate(event_names):
    epoch_indices[event] = {'times': {'start': start_time,
                            'end': start_time - epoch_window_map[event]['window'][0] + epoch_window_map[event]['window'][1] + 2*event_padding,
                            'event_time': start_time - epoch_window_map[event]['window'][0] + event_padding}
                            }
    start_time += (-epoch_window_map[event]['window'][0] + epoch_window_map[event]['window'][1] + 2*event_padding)
    epoch_indices[event]['indices'] = {}
    for key, val in epoch_indices[event]['times'].items():
        epoch_indices[event]['indices'][key] = int(val/binsize)

width = 0.4
event_list = ['Cue', 'Grasp On']
x = np.arange(len(list(epoch_window_map.keys())))
all_mod_dict = {'tVal':[], 'n_idc':[], 'region':[], 'event':[], 'side':[], 'mod_type':[], 'event_mean': [],
                'baseline_mean': []}
merged_df_filename = f'{summary_dir}/MergedModDataFrame.p'
merged_mods_filename = f'{summary_dir}/MergedMods.p'
load_df = False
sdf_fig, sdf_axs = plt.subplots(nrows=1, ncols=3, figsize=(10,6), layout='constrained')
mod_color_map = {'ipsi':'tab:purple', 'contra':'tab:red', 'both:equal':'tab:olive', 'both:opp':'tab:brown',
                         'both:ipsi':'tab:blue', 'both:contra':'tab:orange', 'none':'tab:gray', 'both:all':'tab:green'}
cond_order = ['c_h', 'c_v', 'i_h', 'i_v']
if os.path.exists(merged_df_filename) and load_df:
    merged_mod_df = pd.read_pickle(merged_df_filename)
else:
    sdf_fig.suptitle(f'SDFs by Merged Mod types')
    tval_f, tval_axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey='row')
    t_max = 0
    t_min = 0
    for idx, region in enumerate(merged_mods.keys()):
        mod_ax = sdf_axs[idx]
        mod_ax.set_title(region)
        joint_sdf = wide_sdf_dict[region]
        n_neurons = joint_sdf.shape[0]
        hand_use_mod[region] = {}
        tval_ax = tval_axs[idx]
        multiplier = 0
        for event in ['Cue', 'Grasp']:
            hand_mod_list = []
            mod_tVals = []
            sdf_means = []
            baseline_means = []

            for side_idx, (side, mod_vals) in enumerate(merged_mods[region].items()):
                hand_mod_list.append(mod_vals[event]['modulations'])
                mod_tVals.append(mod_vals[event]['tVals'])
                sdf_means.append(mod_vals[event]['event_sdfs'].mean(axis=0))
                baseline_means.append(mod_vals[event]['base_sdfs'].mean(axis=0))
            # Compute the share of modulated neurons across each hand
            neuron_idcs= np.arange(n_neurons)

            ipsi_mod  =  hand_mod_list[0]
            contra_mod = hand_mod_list[1]
            hand_nonSpecific = ~np.bitwise_xor(ipsi_mod, contra_mod)
            both_mod = ipsi_mod*contra_mod
            equal_mod = ipsi_mod*contra_mod
            none_mod = ~ipsi_mod*~contra_mod
            ipsi_only = ipsi_mod * ~hand_nonSpecific
            contra_only = contra_mod * ~hand_nonSpecific
            hand_use_mod[region][event] = {'ipsi': ipsi_only, 'contra': contra_only}
                                                 # 'Hand non-specific': both_mod}
            for lat in ['ipsilateral', 'contralateral', 'opp']:
                hand_mod = merged_mod_types[region][lat][event]
                if lat== 'ipsilateral':
                    key_string = 'ipsi'
                elif lat == 'contralateral':
                    key_string = 'contra'
                elif lat == 'opp':
                    key_string = 'opp'
                hand_use_mod[region][event][f'both:{key_string}'] = hand_mod
                equal_mod[hand_mod]=False
                if lat == 'ipsilateral':
                    ipsi_more = hand_mod
                elif lat == 'contralateral':
                    contra_more = hand_mod
                elif lat == 'opp':
                    opposite_mod = hand_mod
            hand_use_mod[region][event]['both:equal'] = equal_mod
            hand_use_mod[region][event]['both:all'] = both_mod
            merged_mod_vals = {'ipsi':ipsi_only, 'contra':contra_only, 'both':both_mod}
            all_mods = {'ipsi':ipsi_only, 'contra':contra_only, 'both:equal':equal_mod, 'both:opp':opposite_mod,
                        'both:ipsi':ipsi_more, 'both:contra':contra_more, 'none':none_mod}

            for side_idx, (side, mod_vals) in enumerate(merged_mods[region].items()):
                hand_tVals = mod_tVals[side_idx]
                baseline_mean_list = baseline_means[side_idx]
                sdf_mean_list = sdf_means[side_idx]
                for k, mod_type in enumerate(all_mods.keys()):
                    mod_mask = all_mods[mod_type]
                    hand_tVal = hand_tVals[mod_mask]
                    mod_nIdcs = neuron_idcs[mod_mask]
                    sdf_mean = sdf_mean_list[mod_mask]
                    baseline_mean = baseline_mean_list[mod_mask]
                    n_mods = mod_mask.sum()
                    all_mod_dict['side'] += [side] * n_mods
                    all_mod_dict['tVal'] += list(hand_tVal)
                    all_mod_dict['n_idc'] += list(mod_nIdcs)
                    all_mod_dict['event_mean'] += list(sdf_mean)
                    all_mod_dict['baseline_mean'] += list(baseline_mean)
                    all_mod_dict['region'] += [region] * n_mods
                    all_mod_dict['mod_type'] += [mod_type] * n_mods
                    all_mod_dict['event'] += [event] * n_mods

        tick_list = []
        tick_label_list = []
        x_max = 0
        # Plot SDFs for each modulation Type
        start_idx = 0
        plot_sdf = np.zeros_like(joint_sdf)
        cond_boundaries= [0]
        mod_boundaries = [0]
        label_locations = {}
        time_steps = plot_sdf.shape[1]
        n_neurons = plot_sdf.shape[0]
        cond_len = int(joint_sdf.shape[1] / 4) * binsize
        x = np.arange(time_steps) * binsize
        y = np.arange(n_neurons)
        xv, yv = np.meshgrid(x, y)
        for mod_idx, mod_type in enumerate(all_mods.keys()):
            mod_neurons = all_mods[mod_type]
            end_idx = start_idx + sum(mod_neurons)
            plot_sdf[start_idx:end_idx] = joint_sdf[mod_neurons]
            mod_boundaries.append(end_idx)
            label_locations[mod_type] = (start_idx+end_idx)/2
            start_idx=end_idx
        mod_ax.pcolor(xv, yv, plot_sdf)
        for i_cond, cond in enumerate(cond_order):
            mod_ax.text(cond_len * (i_cond + 1 / 2), n_neurons*(-.02), cond, horizontalalignment='center',
                        verticalalignment='top')
            cond_boundaries.append((i_cond + 1) * cond_len)
            mod_ax.axvline(cond_boundaries[-1], ls='-', c='black', alpha=0.8)
            mod_ax.axvline((cond_boundaries[-2] + cond_boundaries[-1]) / 2, c='white', alpha=0.5)
        for mod_label, loc in label_locations.items():
            mod_ax.text(-150, loc, mod_label, horizontalalignment='right', verticalalignment='center',
                        color=mod_color_map[mod_label], fontsize='small')
        mod_ax.set_ylim([0, n_neurons])
        mod_ax.set_xlim([0, x.max()])
        mod_ax.set_xlabel('Condition', labelpad=10)
        mod_ax.set_xticks(cond_boundaries, labels=[])
        mod_ax.set_yticks(list(label_locations.values()), labels=[])
        for loc in mod_boundaries:
            mod_ax.axhline(loc, c='white')
    sdf_fig.savefig(f'{save_dir}/modSDFs_{event}.png', dpi=200, bbox_inches='tight')
    merged_mod_df = pd.DataFrame(all_mod_dict)
    merged_mod_df.to_pickle(merged_df_filename)

# mod_types = {}
# p = 0.05
# for orientation in modulation_dict.keys():
#     mod_types[orientation] = {}
#     for area in modulation_dict[orientation].keys():
#         mod_types[orientation][area] = {}
#         area_sdfs = []
#         # up_down = []
#         lats = list(modulation_dict[orientation][area].keys())
#         for l_idx, lat in enumerate(lats):
#             event_mods = modulation_dict[orientation][area][lat]['modulations']
#             both_mods =  modulation_dict[orientation][area][lats[1-l_idx]]['modulations']*event_mods
#             event_tVals = modulation_dict[orientation][area][lat]['tVals']
#             other_tVals = modulation_dict[orientation][area][lats[1-l_idx]]['tVals']
#             same_sign = np.equal(np.sign(event_tVals), np.sign(other_tVals))
#             up_down = np.stack([event_mods * event_tVals < 0, event_mods * event_tVals > 0])
#             event_sdfs = modulation_dict[orientation][area][lat]['event_sdfs']
#             other_sdfs = modulation_dict[orientation][area][lats[1-l_idx]]['event_sdfs']
#             mod_vals = np.zeros_like(up_down[0]).astype(bool)
#             for e_idx, event in enumerate(all_sdf_dict[region].keys()):
#                 event_sdf = event_sdfs[e_idx]
#                 other_sdf = other_sdfs[e_idx]
#                 for m_idx, mod in enumerate(['down', 'up']):
#                     mod_mask = up_down[m_idx, e_idx]
#                     if mod == 'down':
#                         modulation, t_vals = t_test(other_sdf[:, mod_mask].T, event_sdf[:, mod_mask].T, type='less', q=p)
#                     else:
#                         modulation, t_vals = t_test(other_sdf[:, mod_mask].T, event_sdf[:, mod_mask].T, type='greater', q=p)
#                     mod_vals[e_idx][mod_mask] = modulation
#             mod_types[orientation][area][lat] = mod_vals * both_mods*same_sign
#         mod_types[orientation][area]['opp'] = both_mods*(~same_sign)

"""
Plot modulation by epoch and compute modulation by hand used
"""
# hand_use_mod = {}
# epoch_indices = {}
# start_time=0
# event_padding=0
# event_names = ['Cue', 'Grasp On']
# for i, event in enumerate(event_names):
#     epoch_indices[event] = {'times': {'start': start_time,
#                             'end': start_time - epoch_window_map[event]['window'][0] + epoch_window_map[event]['window'][1] + 2*event_padding,
#                             'event_time': start_time - epoch_window_map[event]['window'][0] + event_padding}
#                             }
#     start_time += (-epoch_window_map[event]['window'][0] + epoch_window_map[event]['window'][1] + 2*event_padding)
#     epoch_indices[event]['indices'] = {}
#     for key, val in epoch_indices[event]['times'].items():
#         epoch_indices[event]['indices'][key] = int(val/binsize)
#
# width = 0.4
# event_list = ['Cue', 'Grasp On']
# x = np.arange(len(list(epoch_window_map.keys())))
# all_mod_dict = {'tVal':[], 't_idc':[], 'n_idc':[], 'region':[], 'orient':[], 'event':[], 'side':[], 'mod_type':[], 'up-down':[]}
# df_filename = f'{summary_dir}/ModDataFrame.p'
# load_df = False
# plots_dict = {'horizontal':plt.subplots(nrows=1, ncols=3, figsize=(10,6), layout='constrained'),
#               'vertical':  plt.subplots(nrows=1, ncols=3, figsize=(10,6), layout='constrained')}
# cond_order = ['c_h', 'c_v', 'i_h', 'i_v']
# if os.path.exists(df_filename) and load_df:
#     mod_df = pd.read_pickle(df_filename)
# else:
#     for orientation in modulation_dict.keys():
#         orient_fig, orient_axs = plots_dict[orientation]
#         orient_fig.suptitle(f'SDFs for Mod types with {orientation.capitalize()} Orientation')
#         tval_f, tval_axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey='row')
#         hand_use_mod[orientation] = {}
#         t_max = 0
#         t_min = 0
#         for idx, region in enumerate(up_down_modulation[orientation].keys()):
#             mod_ax = orient_axs[idx]
#             mod_ax.set_title(region)
#             joint_sdf = joint_sdf_dict[region]
#             hand_use_mod[orientation][region] = {}
#             # ax = axs[idx]
#             # tval_subfig = tval_subfigs[idx]
#             # tval_subfig.suptitle(f'{region}')
#             # tval_subfig.supxlabel('Epoch Max Slope Timing')
#             # tval_axs = tval_subfig.subplots(nrows=2, ncols=1)
#             tval_ax = tval_axs[idx]
#             multiplier = 0
#             hand_mod = []
#             tVal_list = []
#             for side_idx, (side, mod_tuple) in enumerate(up_down_modulation[orientation][region].items()):
#                 bar_base = 0
#                 if side_idx == 0:
#                     n_neurons = mod_tuple.shape[-1]
#                 for i, mod_val in enumerate(['down', 'up']):
#                     modulation = mod_tuple[i]
#                     hand_mod.append(modulation)
#                     # offset = width*multiplier
#                     # mod_perc = np.average(modulation, axis=1)
#                     # max_height = 100
#                     # rects = ax.bar(x+offset, mod_perc*100, width, bottom=bar_base, label=f'{side}, {mod_val}')
#                     # bar_base = mod_perc*100
#                 if t_max < np.nanmax(tVals_dict[orientation][region][side]):
#                     t_max = np.nanmax(tVals_dict[orientation][region][side])
#                 if t_min > np.nanmin(tVals_dict[orientation][region][side]):
#                     t_min = np.nanmin(tVals_dict[orientation][region][side])
#                 tVal_list.append(tVals_dict[orientation][region][side])
#                 multiplier+= 1
#
#             # Compute the share of modulated neurons across each hand
#             neuron_idcs= np.arange(n_neurons)
#             ipsi_mod = np.bitwise_or(hand_mod[0], hand_mod[1]) #Merge up and down modulation for Ipsi
#             contra_mod = np.bitwise_or(hand_mod[2], hand_mod[3]) #Merge up and down modulation for Contra
#             hand_nonSpecific = ~np.bitwise_xor(ipsi_mod, contra_mod)
#             both_mod = ipsi_mod*contra_mod
#             equal_mod = ipsi_mod*contra_mod
#             none_mod = ~ipsi_mod*~contra_mod
#             ipsi_only = ipsi_mod * ~hand_nonSpecific
#             contra_only = contra_mod * ~hand_nonSpecific
#             hand_use_mod[orientation][region] = {'ipsi': ipsi_only, 'contra': contra_only}
#                                                  # 'Hand non-specific': both_mod}
#             for lat in ['ipsilateral', 'contralateral', 'opp']:
#                 hand_mod = mod_types[orientation][region][lat]
#                 if lat== 'ipsilateral':
#                     key_string = 'ipsi'
#                 elif lat == 'contralateral':
#                     key_string = 'contra'
#                 elif lat == 'opp':
#                     key_string = 'opp'
#                 hand_use_mod[orientation][region][f'both:{key_string}'] = hand_mod
#                 equal_mod[hand_mod]=False
#                 if lat == 'ipsilateral':
#                     ipsi_more = hand_mod
#                 elif lat == 'contralateral':
#                     contra_more = hand_mod
#                 elif lat == 'opp':
#                     opposite_mod = hand_mod
#             hand_use_mod[orientation][region]['both:equal'] = equal_mod
#             hand_use_mod[orientation][region]['both:all'] = both_mod
#             merged_mods = {'ipsi':ipsi_only, 'contra':contra_only, 'both':both_mod}
#             all_mods = {'ipsi':ipsi_only, 'contra':contra_only, 'both:equal':equal_mod, 'both:opp':opposite_mod,
#                         'both:ipsi':ipsi_more, 'both:contra':contra_more, 'none':none_mod}
#             mod_color_map = {'ipsi':'tab:purple', 'contra':'tab:red', 'both:equal':'tab:olive', 'both:opp':'tab:brown',
#                              'both:ipsi':'tab:blue', 'both:contra':'tab:orange', 'none':'tab:gray', 'both:all':'tab:green'}
#             tick_list = []
#             tick_label_list = []
#             x_max = 0
#             # Plot SDFs for each modulation Type
#             start_idx = 0
#             plot_sdf = np.zeros_like(joint_sdf)
#             cond_boundaries= [0]
#             mod_boundaries = [0]
#             label_locations = {}
#             time_steps = plot_sdf.shape[1]
#             n_neurons = plot_sdf.shape[0]
#             cond_len = int(joint_sdf.shape[1] / 4) * binsize
#             x = np.arange(time_steps) * binsize
#             y = np.arange(n_neurons)
#             xv, yv = np.meshgrid(x, y)
#             for mod_idx, mod_type in enumerate(all_mods.keys()):
#                 mod_neurons = all_mods[mod_type][1]
#                 end_idx = start_idx + sum(mod_neurons)
#                 plot_sdf[start_idx:end_idx] = joint_sdf[mod_neurons]
#                 mod_boundaries.append(end_idx)
#                 label_locations[mod_type] = (start_idx+end_idx)/2
#                 start_idx=end_idx
#             mod_ax.pcolor(xv, yv, plot_sdf)
#             for i_cond, cond in enumerate(cond_order):
#                 mod_ax.text(cond_len * (i_cond + 1 / 2), n_neurons*(-.02), cond, horizontalalignment='center',
#                             verticalalignment='top')
#                 cond_boundaries.append((i_cond + 1) * cond_len)
#                 mod_ax.axvline(cond_boundaries[-1], ls='-', c='black', alpha=0.8)
#                 mod_ax.axvline((cond_boundaries[-2] + cond_boundaries[-1]) / 2, c='white', alpha=0.5)
#             for mod_label, loc in label_locations.items():
#                 mod_ax.text(-150, loc, mod_label, horizontalalignment='right', verticalalignment='center',
#                             color=mod_color_map[mod_label], fontsize='small')
#             mod_ax.set_ylim([0, n_neurons])
#             mod_ax.set_xlim([0, x.max()])
#             mod_ax.set_xlabel('Condition', labelpad=10)
#             mod_ax.set_xticks(cond_boundaries, labels=[])
#             mod_ax.set_yticks(list(label_locations.values()), labels=[])
#             for loc in mod_boundaries:
#                 mod_ax.axhline(loc, c='white')

    #         # Plot slopes of
    #         for i, event in enumerate(event_list):
    #             start_idx = epoch_indices[event]['indices']['start']
    #             event_idx = epoch_indices[event]['indices']['event_time']
    #             end_idx = epoch_indices[event]['indices']['end']
    #             tick_list.append(event_idx)
    #             tick_label_list.append(event)
    #             x_max = max(x_max, end_idx)
    #             for j, side in enumerate(up_down_modulation[orientation][region].keys()):
    #                 # tval_ax = tval_axs[j]
    #                 tval_ax.axvline(x=start_idx, color='grey', linestyle=':')
    #                 tval_ax.axvline(x=event_idx, color='black')
    #                 mod_dict = modulation_dict[orientation][region][side]
    #                 for k, mod_type in enumerate(all_mods.keys()):
    #                     mod_mask = all_mods[mod_type][i]
    #                     event_mod = tVal_list[j][i]
    #                     if not (mod_type[:4] == side[:4] or mod_type[:4] == 'both' or mod_type == 'none'):
    #                         continue
    #                     for direction in mod_dict['event_slopes'][i].keys():
    #                         dir_idcs = mod_dict['event_slopes'][i][direction][1]
    #                         if direction == 'max':
    #                             dir_mask = event_mod>0
    #                         elif direction == 'min':
    #                             dir_mask = event_mod<0
    #                         else: continue
    #                         full_mask = dir_mask*mod_mask
    #                         n_mods = full_mask.sum()
    #                         mod_tIdcs = dir_idcs[full_mask]
    #                         mod_tVals= event_mod[full_mask]
    #                         mod_nIdcs = neuron_idcs[full_mask]
    #                         all_mod_dict['tVal'] += [mod_tVals]
    #                         all_mod_dict['side'] += [side] * n_mods
    #                         # all_mod_dict['t_idc'] += [mod_tIdcs + start_idx]
    #                         all_mod_dict['t_idc'] += [mod_tIdcs]
    #                         all_mod_dict['n_idc'] += [mod_nIdcs]
    #                         all_mod_dict['orient'] += [orientation] * n_mods
    #                         all_mod_dict['region'] += [region] * n_mods
    #                         all_mod_dict['event'] += [event] * n_mods
    #                         all_mod_dict['mod_type'] += [mod_type] * n_mods
    #                         all_mod_dict['up-down'] += [direction] * n_mods
    #
    #                         # print(f'Plotting {mod_type}, {side}')
    #                         tval_ax.scatter(mod_tIdcs+start_idx, mod_tVals, label=mod_type,
    #                                         c=mod_color_map[mod_type], alpha=0.5)
    #                 tval_ax.set_xticks(tick_list, labels=tick_label_list)#, rotation='vertical'
    #                 tval_ax.set_ylabel('t Value')
    #                 tval_ax.set_title(f'{region}')
    #                 tval_ax.axhline(y=0, color='grey')
    #                 tval_ax.set_xlim([0, x_max+2])
    #                 tval_ax.set_ylim([t_min-2, t_max+2])
    #                 tval_ax.set_yscale('symlog')
    #                 tval_ax.set_xlabel('Epoch Max Slope Timing')
    #         # ax.set_ylabel('Neurons(%)')
    #         # ax.set_ylim([0, max_height])
    #         # ax.set_xlabel('Epochs')
    #         # ax.set_title(f'{region}')
    #         # ax.set_xticks(x + width * (multiplier - 1) / 2, [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]])
    #     # handles, labels = plt.figure(f).get_axes()[-1].get_legend_handles_labels()
    #     # f.legend(handles, labels, loc = (0.5, -0.05), ncols=4)
    #     # plt.tight_layout()
    #     # sns.move_legend(f, "lower center", bbox_to_anchor=(.5, -.1), ncol=4)
    #     # sns.despine()
    #     # f.suptitle(f'Modulation by Epoch, {orientation.capitalize()}')
    #     # plt.savefig(f'{save_dir}/Modulation_merged{len(monkey_name_map.keys())}_{orientation}.png', bbox_inches='tight')
    #
    #     plt.figure(tval_f)
    #     t_handles, t_labels = tval_f.get_axes()[-1].get_legend_handles_labels()
    #     plt.tight_layout()
    #     sns.despine()
    #     # f.suptitle(f'T Value by Epoch, {orientation.capitalize()}')
    #     plt.savefig(f'{save_dir}/tVals_merged{len(monkey_name_map.keys())}_{orientation}.png', bbox_inches='tight')
    #     orient_fig.savefig(f'{save_dir}/modSDFs_{orientation}.png', dpi=200, bbox_inches='tight')
    # for key, val in all_mod_dict.items():
    #     all_mod_dict[key] = np.hstack(val)
    # mod_df = pd.DataFrame(all_mod_dict)
    # mod_df.to_pickle(f'{summary_dir}/ModDataFrame.p')

# width = 0.3
# for orientation in modulation_dict.keys():
#     f_hist = plt.figure(layout='constrained', figsize=(8,10))
#     sub_figs = f_hist.subfigures(3,1, wspace=0.01)
#     # f_hist, hist_axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
#     for reg_idx, region in enumerate(modulation_dict[orientation].keys()):
#         reg_fig = sub_figs[reg_idx]
#         hist_axs = reg_fig.subplots(nrows=1, ncols=2,sharey=True)
#         reg_fig.suptitle(region)
#         legend_elements = [mpl.lines.Line2D([0], [0], color=mod_color_map[mod], label = mod) for mod in mod_color_map.keys()]
#         for i, event in enumerate(event_list):
#             # if event == 'Grasp On'
#             mod_data = merged_mod_df[(merged_mod_df['orient']==orientation)&(merged_mod_df['region']==region)&(merged_mod_df['event']==event)]
#             pos_ax = hist_axs[i]
#             neg_ax = pos_ax.twinx()
#             pos_ax.set_xlabel('Time (ms)')
#             n_bins= 30
#             sns.histplot(data=mod_data[mod_data['up-down'] == 'max'], x="t_idc", hue="mod_type", shrink=1, alpha=.8,
#                          legend=False, ax=pos_ax, multiple='stack', bins=n_bins,
#                          hue_order = ['ipsi', 'both:ipsi', 'both:equal' , 'both:opp', 'both:contra', 'contra'])
#             sns.histplot(data=mod_data[mod_data['up-down'] == 'min'], x="t_idc", hue="mod_type", shrink=1, alpha=.8,
#                          legend=False, ax=neg_ax, multiple='stack', bins=n_bins,
#                          hue_order = ['ipsi', 'both:ipsi', 'both:equal' , 'both:opp', 'both:contra', 'contra'])
#             neg_ax.set_ylabel('')
#             max_val = pos_ax.get_ylim()[1]
#             min_val = neg_ax.get_ylim()[1]
#             y_lim = max(max_val, min_val)
#             start_idx = epoch_indices[event]['indices']['start']
#             event_idx = epoch_indices[event]['indices']['event_time']
#             end_idx = epoch_indices[event]['indices']['end']
#             tick_list = (np.array([start_idx, event_idx, end_idx])-start_idx)
#             tick_label_list = ((tick_list-event_idx+start_idx)*binsize).astype(np.int_)
#             neg_ax.axvline(x=0, color='grey', linestyle=':')
#             neg_ax.axvline(x=event_idx-start_idx, color='black')
#             pos_ax.set_title(f'{event}')
#             pos_ax.set_xticks(tick_list, labels=tick_label_list)
#             pos_ax.set_ylim([-y_lim, y_lim])
#             if i==0:
#                 pos_ax.set_ylabel('Count')
#             y_ticks = pos_ax.get_yticks()
#             pos_ax.set_yticks(ticks = y_ticks, labels=np.abs(y_ticks).astype(np.int_))
#             neg_ax.axhline(y=0, color='black')
#             neg_ax.set_yticks([])
#             neg_ax.set_ylim([-y_lim, y_lim])
#             neg_ax.invert_yaxis()
#
#     reg_fig.legend(legend_elements, mod_color_map.keys(), loc='lower center', bbox_to_anchor=(0.5, -.2), ncol=4)
#     f_hist.suptitle(f'T Value Timing, {orientation.capitalize()}')
#     # f_hist.tight_layout()
#     f_hist.savefig(f'{summary_dir}/tValsHist_earlyLate_{orientation}.png', bbox_inches='tight')

layout = [
    ['main_M1', 'main_M1', 'main_PMd', 'main_PMd', 'main_PMv', 'main_PMv'],
    # ['main_M1', 'main_PMd', 'main_PMv'],
    # ['M1_pie_Cue',  'PMd_pie_Cue',  'PMv_pie_Cue'],
    # ['M1_pie_Grasp', 'PMd_pie_Grasp', 'PMv_pie_Grasp']
    ['M1_pie_Cue', 'M1_pie_Grasp', 'PMd_pie_Cue', 'PMd_pie_Grasp', 'PMv_pie_Cue', 'PMv_pie_Grasp'],
]
f, axs = plt.subplot_mosaic(layout, figsize=(16,5.5), sharey = False)
width = 0.3
x = np.arange(len(list(epoch_window_map.keys())[1:]))
for idx, region in enumerate(hand_use_mod.keys()):
    ax = axs[f'main_{region}']
    # multiplier = 0
    bar_base = 0
    max_height = 100
    for e_i, (event, mod_dictionary) in enumerate(hand_use_mod[region].items()):
        multiplier = 0
        for mod_type, modulation in mod_dictionary.items():
            hand_mod_perc = np.average(modulation)
            # hand_mod_perc = np.average(modulation, axis=1)
            if mod_type in ['both:all', 'ipsi', 'contra']:
                offset = width * multiplier
                if e_i == 0:
                    rects = ax.bar(x[e_i]+offset, hand_mod_perc*100, width, label=mod_type, color=mod_color_map[mod_type])
                else:
                    rects = ax.bar(x[e_i] + offset, hand_mod_perc * 100, width, color=mod_color_map[mod_type])
                bar_base = hand_mod_perc*100 + bar_base
                multiplier += 1
            else:
                continue
    ax.set_ylim([0, max_height])
    ax.set_xlabel('Epochs')
    ax.set_title(f'{region}')
    ax.set_xticks(x + width * (multiplier - 1) / 2, [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]])
    if region != 'M1':
        ax.set_yticks([])
handles, labels = ax.get_legend_handles_labels()
axs['main_PMd'].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=3)

for idx, region in enumerate(hand_use_mod.keys()):
    multiplier = 0
    bar_base = 0
    region_df = merged_mod_df[(merged_mod_df['region']==region)]
    both_mod_list = ['both:ipsi', 'both:contra', 'both:equal', 'both:opp']
    both_mod_counts = {}
    for event, mod_dictionary in hand_use_mod[region].items():
        both_mod_counts[event] = []
        event_df = region_df[region_df['event']==event]
        mod_order=[]
        for mod_type, mod_bools in mod_dictionary.items():
            if mod_type in both_mod_list:
                mod_order.append(mod_type)
                mods = hand_use_mod[region][event][mod_type]
                up_mod   = mod_bools * event_df[(event_df['side']=='contralateral')]['tVal'] > 0
                down_mod = mod_bools * event_df[(event_df['side']=='contralateral')]['tVal'] < 0
                split_mod = np.stack([down_mod, up_mod])
                both_mod_counts[event].append(split_mod.sum(axis=1))
                # both_mod_counts['Grasp'].append(split_mod[:,1].sum(axis=1))
        else:
            continue

    both_mod_counts['Cue'] = np.stack(both_mod_counts['Cue'])
    both_mod_counts['Grasp'] = np.stack(both_mod_counts['Grasp'])
    size=0.4
    rgb = mpl.colors.ColorConverter.to_rgb
    for epoch in ['Cue', 'Grasp']:
        pie_ax = axs[f'{region}_pie_{epoch}']
        labels = []
        labels += [f'{mod.split(':')[1]}' for mod in mod_order]
        counts = both_mod_counts[epoch]
        outer_colors = [mod_color_map[mod] for mod in mod_order]
        inner_colors = []
        for color in outer_colors:
            inner_colors += [scale_lightness(rgb(color), 1.7)]
            inner_colors += [scale_lightness(rgb(color), 1.2)]
        pie_ax.pie(counts.sum(axis=1), radius=1, colors = outer_colors, startangle=90,
                   wedgeprops=dict(width=size, edgecolor='w'), labels=labels, autopct='%1.f', labeldistance=1.1,
                   pctdistance=0.8)
        pie_ax.pie(counts.flatten(), radius = 1-size, colors=inner_colors, startangle=90,
                   wedgeprops=dict(width=size, edgecolor='w'))
        pie_ax.set_title(f'{epoch}', va='center')
        if region == 'M1' and epoch == 'Cue':
            pie_ax.set_ylabel(f'Both Mod Breakdown (%)')
with open(merged_mods_filename, 'wb') as merged_mod_file:
    pkl.dump(merged_mods, merged_mod_file)
axs['main_M1'].set_ylabel('Epoch Modulated Neurons (%)')
f.suptitle(f'Modulation by Hand Used')
plt.tight_layout()
sns.despine()
plt.savefig(f'{summary_dir}/HandModulation_merged{len(monkey_name_map.keys())}.png',
            bbox_inches='tight')