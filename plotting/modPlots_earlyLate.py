import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob
import matplotlib as mpl
from config import SORTED_DIR, SUMMARY_DIR, BIN_SIZE, KERNEL_WIDTH

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import numpy as np

from utils.sig_proc import *
import pandas as pd
import matplotlib as mpl
from utils.plot_utils import scale_lightness, add_headers
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from scipy.spatial.distance import pdist
from math import ceil

from utils.sig_proc import *

import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from config import (
    MONKEY_NAME_MAP as monkey_name_map,
    EVENT_MAP_MERGE as event_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    SORTED_DIR as data_dir,
    MATLAB_DIR as matlab_dir,
    SORTING_DIR as sorting_dir,
    SUMMARY_DIR as summary_dir,
    LAT_MAP as lat_map,
    HAND_LIST as hand_list,
)


sns.set()
sns.set_style(style='white')

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Pre-cue':  {'event': 'trialRewardDrop', 'window': [-300,   -100]},
                    'Cue':      {'event': 'trialRewardDrop', 'window': [0,   200]},
                    'Grasp': {'event': 'trialGraspOn',    'window': [-100,   300]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]

#Define directories of data
summary_dir = os.path.join(SUMMARY_DIR, 'earlyLate')
save_dir = summary_dir

if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
events = ['trialRewardDrop', 'trialGraspOn']
binsize = 5
kernel_width = 25
p_score = 0.05
load_override_preprocess = False
load_override = False

# Spike/event/condition files live in SUMMARY_DIR (not the earlyLate subdirectory)
spike_file_name     = f'{SUMMARY_DIR}/spike_list_merged{len(monkey_name_map.keys())}.p'
event_file_name     = f'{SUMMARY_DIR}/event_list_merged{len(monkey_name_map.keys())}.p'
condition_file_name = f'{SUMMARY_DIR}/condition_list_merged{len(monkey_name_map.keys())}.p'

with open(spike_file_name, 'rb') as spike_file:
    all_spikes = pkl.load(spike_file)
with open(event_file_name, 'rb') as event_file:
    all_events = pkl.load(event_file)
with open(condition_file_name, 'rb') as condition_file:
    all_conditions = pkl.load(condition_file)

# Normalise all_events keys to match event_map values ('Cue', 'Grasp')
for _area in all_events:
    if 'Grasp On' in all_events[_area]:
        all_events[_area]['Grasp'] = all_events[_area].pop('Grasp On')

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
Create a dictionary of trials merging orientation
"""
merged_summary_dict = {}
merged_dict_filename = f'{summary_dir}/mergedSpikeDict.p'
for region_key in area_summary_dict.keys():
    latArea, orient = region_key.split('_')
    merged_key = latArea
    if merged_key not in merged_summary_dict.keys():
        merged_summary_dict[merged_key] = {}
    for event in area_summary_dict[region_key].keys():
        if event not in merged_summary_dict[latArea].keys():
            merged_summary_dict[latArea][event] = {'spikes': []}
        event_spikes = area_summary_dict[region_key][event]['spikes']
        merged_summary_dict[latArea][event]['spikes'].append(np.hstack(event_spikes))
for latArea in merged_summary_dict.keys():
    for event in merged_summary_dict[latArea].keys():
        merged_summary_dict[latArea][event]['spikes'] = np.hstack(merged_summary_dict[latArea][event]['spikes'])
        merged_summary_dict[latArea][event]['neurons'] = (np.max(merged_summary_dict[latArea][event]['spikes'][0])).astype(int)
with open(merged_dict_filename, 'wb') as merged_file:
    pkl.dump(merged_summary_dict, merged_file)

"""
Generate plots of the max spiking rates for each area, side, and orientation.
Scale the rate to the max rate for contralateral side. Apply that scale to ipsilateral side.
"""
window_range = np.array([-1000, 1000])
area_kernel = 25
skip = False
area_max_rate = {}
all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_orientMerged.p'
max_rate_filename = f'{summary_dir}/sdfMaxRate_bin{binsize}_k{kernel_width}_orientMerged.p'
peak_order_filename = f'{summary_dir}/peakOrderDict_bin{binsize}_k{kernel_width}_orientMerged.p'
windowed_spike_filename = f'{summary_dir}/windowedSpikes_orientMerged.p'
trial_avg_filename = f'{summary_dir}/trialAverages_orientMerged.p'
lat_ttest_filename = f'{summary_dir}/laterality_ttest_dict.p'

if os.path.exists(all_sdf_filename) and not load_override:
    with open(all_sdf_filename, 'rb') as sdf_file:
        all_sdf_dict = pkl.load(sdf_file)
    with open(max_rate_filename, 'rb') as max_rate_file:
        area_max_rate = pkl.load(max_rate_file)
    with open(peak_order_filename, 'rb') as peak_order_file:
        peak_order_dict = pkl.load(peak_order_file)
    print(f'SDF Dictionary Loaded (Bin: {binsize}, Kernel: {kernel_width})')
else:
    # ------------------------------------------------------------------
    # Loop 1: SDF computation — iterates over latArea ('cM1', 'iM1', …)
    # ------------------------------------------------------------------
    all_sdf_dict = {}
    peak_order_dict = {}
    for latArea in merged_summary_dict.keys():
        area_label = latArea
        if skip:
            break
        peak_order_dict[area_label] = {}
        area_max_rate[area_label] = np.zeros(merged_summary_dict[area_label][events[0]]['neurons'])
        region_key = f'{area_label}'
        print(region_key)
        lat = region_key[0]
        all_sdf_dict[region_key] = {}
        for event in events:
            sdf_list = []
            psth_list = []
            neuron_count = merged_summary_dict[region_key][event]['neurons']
            neuron_peak_times = np.zeros(neuron_count)
            for neuron_idx in range(neuron_count):
                neuron_mask = merged_summary_dict[region_key][event]['spikes'][0, :] == neuron_idx + 1
                neuron_spikes = merged_summary_dict[region_key][event]['spikes'][:, neuron_mask]
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

# ------------------------------------------------------------------
# Loop 2: per-trial laterality t-test — one pass per area × event.
# Uses all_spikes / all_conditions / all_events keyed by bare area
# ('M1', 'PMd', 'PMv'), so no c/i duplication.
# ------------------------------------------------------------------
if os.path.exists(lat_ttest_filename) and not load_override:
    with open(lat_ttest_filename, 'rb') as lat_ttest_file:
        laterality_ttest_dict = pkl.load(lat_ttest_file)
    print('Laterality t-test dict loaded')
else:
    laterality_ttest_dict = {}
    for area in ['M1', 'PMd', 'PMv']:
        laterality_ttest_dict[area] = {}
        for event in events:
            event_label = event_map[event]                        # 'Cue' or 'Grasp'
            epoch_ms    = epoch_window_map[event_label]['window'] # e.g. [0, 200]

            n       = len(all_spikes[area])
            t_stats = np.full(n, np.nan)
            p_vals  = np.full(n, np.nan)

            for neuron_idx in range(n):
                t_stats[neuron_idx], p_vals[neuron_idx] = laterality_ttest(
                    area, neuron_idx, event, epoch_ms,
                    all_spikes, all_conditions, all_events, event_map)

            valid  = ~np.isnan(p_vals)
            reject = np.zeros(n, dtype=bool)
            if valid.sum() > 1:
                reject[valid], _, _, _ = multipletests(
                    p_vals[valid], alpha=p_score, method='fdr_bh')

            laterality_ttest_dict[area][event_label] = {
                't':         t_stats,
                'p':         p_vals,
                'reject':    reject,
                'direction': np.sign(t_stats),
            }
    with open(lat_ttest_filename, 'wb') as lat_ttest_file:
        pkl.dump(laterality_ttest_dict, lat_ttest_file)
    print('Laterality t-test dict saved')

"""
Generate plots after loading data
"""
plot_dict = {}
max_dict = {}
clustering_dict = {}
sns.set_style("ticks")
plots = {}
for region in ['M1', 'PMd', 'PMv']:
    plots[region] = {}
for area_label in merged_summary_dict.keys():
    lat, region = area_label[0], area_label[1:]
    peak_order = peak_order_dict[area_label]
    if region not in plot_dict.keys():
        plot_dict[region]= {}
        max_dict[region] = {}
        clustering_dict[region]={}
        plots[region] = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    dend_fig, dend_axs = plots[region]
    if lat == 'c':
        lat_idx=0
        y_label=True
    else:
        lat_idx=1
        y_label=False
    for event, dend_ax in zip(events, dend_axs):
        if event not in plot_dict[region]:
            plot_dict[region][event] = {}
            max_dict[region][event] = {}
            clustering_dict[region][event] = {}
        neuron_count = merged_summary_dict[area_label][event]['neurons']
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
        plot_dict[region][event][lat] = area_sdf
        max_dict[region][event][lat] = {'max':area_sdf.max(axis=0), 'order': peak_order[event]}
        clustering_dict[region][event][lat] = {'dist':y, 'Z':Z}
for region in plots.keys():
    dend_fig, _ = plots[region]
    dend_fig.suptitle(f'Clustering for {region}, Grasp Onset')
    dend_fig.savefig(f'{summary_dir}/NumberOfClusters_All_mergedOrient.png', dpi=200)
    plt.close()

joint_sdf_dict= {}
wide_sdf_dict = {}
cluster_dict = {'M1':{}, 'PMd':{}, 'PMv':{}}
dend_fig, dend_axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
epoch_windows = {'trialRewardDrop': [-100, 500], 'trialGraspOn':[-300, 300]}
for reg_idx, region in enumerate(cluster_dict.keys()):
    joint_sdfs =[]
    wide_sdfs = []
    for lat in ['c', 'i']:
        latArea = f'{lat}{region}'
        window_idcs = ((epoch_window_map['Grasp']['window'] - window_range[0]) / binsize).astype(np.int_)
        wide_window_idcs = ((np.array([-500, 500])-window_range[0])/binsize).astype(np.int_)
        joint_sdfs.append(all_sdf_dict[latArea]['trialGraspOn'][window_idcs[0]:window_idcs[1]])
        wide_sdfs.append(all_sdf_dict[latArea]['trialGraspOn'][wide_window_idcs[0]:wide_window_idcs[1]])
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
    lat, region = area_label[0], area_label[1:]
    area_max = area_max_rate[area_label]
    if area_merged_rates[region] is None:
        area_merged_rates[region] = np.zeros_like(area_max)
    new_maxes = np.max(np.stack([area_max, area_merged_rates[region]]), axis=0)
    area_merged_rates[region] = new_maxes

for area_label in merged_summary_dict.keys():
    max_rate_mask = {}
    full_window = np.arange(-1000, 1000+binsize, binsize)
    area_max = area_max_rate[area_label]
    lat, region = area_label[0], area_label[1:]
    event_spike_maxes[area_label] = {}
    if region not in event_neuron_max.keys():
        event_neuron_max[region] = {'proportions': {}, 'neurons':0}
    for epoch in epoch_window_map.keys():
        if epoch == 'Pre-cue':
            continue
        event = epoch_window_map[epoch]['event']
        event_window = epoch_window_map[epoch]['window']
        event_mask = (full_window>event_window[0]) * (full_window<=event_window[1])
        area_sdf = all_sdf_dict[area_label][event][event_mask]
        event_spike_maxes[area_label][epoch] = area_sdf.max(axis=0)
    event_neuron_peaks = np.vstack(list(event_spike_maxes[area_label].values()))
    event_peaks = (event_neuron_peaks == area_merged_rates[region])
    event_peak_proportions = np.average(event_peaks, axis=1)
    event_neuron_max[region]['neurons'] = event_neuron_peaks.shape[1]
    event_neuron_max[region]['proportions'][lat_map[lat]]= event_peak_proportions


with open(neuron_peaks_filename, 'wb') as neuron_peaks_file:
    pkl.dump(event_neuron_max, neuron_peaks_file)

"""
Perform t-tests to determine if neurons were modulated (per-trial spike rates vs baseline)
"""
full_window = np.arange(-1000, 1000 + binsize, binsize)
baseline_epoch = 'Pre-cue'
base_window    = epoch_window_map[baseline_epoch]['window']
baseline_mask  = (full_window > base_window[0]) * (full_window < base_window[1])

event_masks = {}
slope_masks = {}
event_times = {}
for event_label in ['Cue', 'Grasp']:
    event_window = epoch_window_map[event_label]['window']
    event_masks[event_label] = (full_window > event_window[0]) * (full_window < event_window[1])
    slope_masks[event_label] = (full_window > event_window[0]) * (full_window < event_window[1])
    event_times[event_label] = int(-1 * (event_window[0]) / binsize)

modulation_dict = {}
up_down_modulation = {}
tVals_dict = {}
mod_sdfs = {}

# raw event whose aligned SDF defines the baseline window (trialRewardDrop → Pre-cue epoch)
baseline_raw_event   = epoch_window_map[baseline_epoch]['event']     # 'trialRewardDrop'
baseline_event_label = event_map[baseline_raw_event]                 # 'Cue'
baseline_epoch_ms    = base_window                                   # [-300, -100]

for region_key in all_sdf_dict.keys():   # 'cM1', 'iM1', 'cPMd', ...
    lat       = region_key[0]            # 'c' or 'i'
    area      = region_key[1:]           # 'M1', 'PMd', 'PMv'
    lat_label = lat_map[lat]             # 'contralateral' or 'ipsilateral'

    if area not in modulation_dict:
        modulation_dict[area]      = {}
        up_down_modulation[area]   = {}
        tVals_dict[area]           = {}
        mod_sdfs[area]             = {}

    # Keep the baseline SDF slice for slope bookkeeping / plotting downstream
    baseline_sdf = all_sdf_dict[region_key][baseline_raw_event][baseline_mask]

    # Use the SDF neuron count so modulation arrays align with joint_sdf downstream
    n_neurons = all_sdf_dict[region_key][events[0]].shape[1]

    event_modulations = []
    event_tVals       = []
    event_max_slopes  = []
    event_sdfs_out    = []

    for raw_event in events:                         # ['trialRewardDrop', 'trialGraspOn']
        event_label    = event_map[raw_event]        # 'Cue' or 'Grasp'
        event_epoch_ms = epoch_window_map[event_label]['window']

        # SDF slices kept for slope computation and downstream plotting
        event_sdf  = all_sdf_dict[region_key][raw_event][event_masks[event_label]]
        slope_sdf  = all_sdf_dict[region_key][raw_event][slope_masks[event_label]]
        event_slopes = slope_sdf[1:] - slope_sdf[:-1]

        event_max_slopes.append({
            'max':      (event_slopes.max(axis=0),  event_slopes.argmax(axis=0)),
            'min':      (event_slopes.min(axis=0),  event_slopes.argmin(axis=0)),
            'slopes':   event_slopes,
            'sdf':      event_sdf,
            'baseline': baseline_sdf,
        })

        # --- Per-trial spike rate comparison: baseline epoch vs event epoch ---
        t_stats = np.full(n_neurons, np.nan)
        p_vals  = np.full(n_neurons, np.nan)

        for neuron_idx in range(n_neurons):
            lat_labels_n    = all_conditions[area][neuron_idx][:, 0]
            spikes_by_trial = all_spikes[area][neuron_idx]
            baseline_times  = np.array(all_events[area][baseline_event_label][neuron_idx])
            event_times_n   = np.array(all_events[area][event_label][neuron_idx])

            baseline_rates = epoch_trial_rates(
                spikes_by_trial, baseline_times, baseline_epoch_ms, lat_labels_n, lat)
            event_rates    = epoch_trial_rates(
                spikes_by_trial, event_times_n,  event_epoch_ms,   lat_labels_n, lat)

            n_b, n_e = len(baseline_rates), len(event_rates)
            if n_b == n_e and n_b > 1:
                t_stats[neuron_idx], p_vals[neuron_idx] = ttest_rel(event_rates, baseline_rates)
            elif n_b > 1 and n_e > 1:
                t_stats[neuron_idx], p_vals[neuron_idx] = ttest_ind(
                    event_rates, baseline_rates, equal_var=False)

        # FDR correction across all neurons for this region × event
        valid  = ~np.isnan(p_vals)
        reject = np.zeros(n_neurons, dtype=bool)
        if valid.sum() > 1:
            reject[valid], _, _, _ = multipletests(
                p_vals[valid], alpha=p_score, method='fdr_bh')

        event_modulations.append(reject)
        event_tVals.append(t_stats)
        event_sdfs_out.append(event_sdf)

    event_modulations = np.vstack(event_modulations)
    event_tVals       = np.vstack(event_tVals)

    modulation_dict[area][lat_label] = {
        'modulations':  event_modulations,
        'tVals':        event_tVals,
        'event_slopes': event_max_slopes,
        'event_sdfs':   event_sdfs_out,
    }
    up_down_modulation[area][lat_label] = np.stack([
        event_modulations * event_tVals < 0,
        event_modulations * event_tVals > 0,
    ])
    tVals_dict[area][lat_label] = event_tVals
    mod_sdfs[area][lat_label]   = event_sdfs_out

merged_mods = {}
for area in modulation_dict.keys():
    merged_mods[area] = {}
    for lat_label in modulation_dict[area].keys():
        slopes = modulation_dict[area][lat_label]['event_slopes']
        merged_mods[area][lat_label] = {
            event_map[raw_event]: {
                'modulations': modulation_dict[area][lat_label]['modulations'][i],
                'tVals':       modulation_dict[area][lat_label]['tVals'][i],
                'event_sdfs':  modulation_dict[area][lat_label]['event_sdfs'][i],
                'base_sdfs':   slopes[i]['baseline'],
            }
            for i, raw_event in enumerate(events)
        }

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
            rev_lat_map    = {v: k for k, v in lat_map.items()}   # 'contralateral'->'c', etc.
            lat_char       = rev_lat_map[lat]                       # 'c' or 'i'
            other_lat_char = rev_lat_map[lats[1 - l_idx]]
            event_epoch_ms = epoch_window_map[event]['window']

            mod_vals = np.zeros_like(up_down[0]).astype(bool)
            for m_idx, mod in enumerate(['down', 'up']):
                mod_mask       = up_down[m_idx]
                masked_indices = np.where(mod_mask)[0]
                n_masked       = len(masked_indices)

                t_stats_m = np.full(n_masked, np.nan)
                p_vals_m  = np.full(n_masked, np.nan)

                for i, neuron_idx in enumerate(masked_indices):
                    lat_labels_n    = all_conditions[area][neuron_idx][:, 0]
                    spikes_by_trial = all_spikes[area][neuron_idx]
                    event_times_n   = np.array(all_events[area][event][neuron_idx])

                    lat_rates       = epoch_trial_rates(
                        spikes_by_trial, event_times_n, event_epoch_ms, lat_labels_n, lat_char)
                    other_lat_rates = epoch_trial_rates(
                        spikes_by_trial, event_times_n, event_epoch_ms, lat_labels_n, other_lat_char)

                    n_l, n_o = len(lat_rates), len(other_lat_rates)
                    if n_l == n_o and n_l > 1:
                        t_stats_m[i], p_vals_m[i] = ttest_rel(lat_rates, other_lat_rates)
                    elif n_l > 1 and n_o > 1:
                        t_stats_m[i], p_vals_m[i] = ttest_ind(
                            lat_rates, other_lat_rates, equal_var=False)

                valid  = ~np.isnan(p_vals_m)
                reject = np.zeros(n_masked, dtype=bool)
                if valid.sum() > 1:
                    reject[valid], _, _, _ = multipletests(
                        p_vals_m[valid], alpha=p_score, method='fdr_bh')

                mod_vals[mod_mask] = reject & (t_stats_m > 0)
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
event_names = ['Cue', 'Grasp']
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
event_list = ['Cue', 'Grasp']
x = np.arange(len(list(epoch_window_map.keys())))
all_mod_dict = {'tVal':[], 'n_idc':[], 'region':[], 'event':[], 'side':[], 'mod_type':[], 'event_mean': [],
                'baseline_mean': []}
merged_df_filename = f'{summary_dir}/MergedModDataFrame.p'
merged_mods_filename = f'{summary_dir}/MergedMods.p'
hand_use_mod_filename = f'{summary_dir}/HandUseMod.p'
load_df = False
sdf_fig, sdf_axs = plt.subplots(nrows=1, ncols=3, figsize=(10,6), layout='constrained')
mod_color_map = {'ipsi':'tab:green', 'both:ipsi':'tab:blue', 'both:equal':'tab:brown', 'both:opp':'tab:olive',
                          'both:contra':'tab:orange', 'contra':'tab:red', 'none':'tab:gray', 'both:all':'tab:cyan'}
cond_order = ['c_h', 'c_v', 'i_h', 'i_v']
if os.path.exists(merged_df_filename) and load_df:
    merged_mod_df = pd.read_pickle(merged_df_filename)
    hand_use_mod = pkl.load(open(hand_use_mod_filename, 'rb'))
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

        plot_sdfs = False
        if plot_sdfs:
            tick_list = []
            tick_label_list = []
            x_max = 0
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
    with open(hand_use_mod_filename, 'wb') as hum_file:
        pkl.dump(hand_use_mod, hum_file)

"""
Plot modulation by epoch and compute modulation by hand used
"""
layout = [
    ['main_M1', 'main_M1', 'main_PMd', 'main_PMd', 'main_PMv', 'main_PMv'],
    ['M1_pie_Cue', 'M1_pie_Grasp', 'PMd_pie_Cue', 'PMd_pie_Grasp', 'PMv_pie_Cue', 'PMv_pie_Grasp'],
]
f, axs = plt.subplot_mosaic(layout, figsize=(16,5.5), sharey = False)
width_main = 0.2   # width of the primary (% modulated) bar
width_shadow = 0.1 # same width — shadow bar sits flush against main bar
step = width_main + width_shadow  # total footprint per mod_type group (0.3)
rgb = mpl.colors.ColorConverter.to_rgb
x = np.arange(len(list(epoch_window_map.keys())[1:]))
ud_bar_types = ['both:all', 'ipsi', 'contra']
for idx, region in enumerate(hand_use_mod.keys()):
    ax = axs[f'main_{region}']
    bar_base = 0
    max_height = 100
    region_df = merged_mod_df[merged_mod_df['region'] == region]
    for e_i, (event, mod_dictionary) in enumerate(hand_use_mod[region].items()):
        multiplier = 0
        event_df = region_df[region_df['event'] == event]
        both_subtypes = ['both:equal', 'both:opp', 'both:ipsi', 'both:contra']
        for mod_type, modulation in mod_dictionary.items():
            if mod_type not in ud_bar_types:
                continue
            hand_mod_perc = np.average(modulation)
            n_neurons = len(modulation)
            offset = step * multiplier

            # --- primary bar: % of neurons modulated ---
            if e_i == 0:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, width_main,
                       label=mod_type, color=mod_color_map[mod_type], edgecolor='none')
            else:
                ax.bar(x[e_i] + offset, hand_mod_perc * 100, width_main,
                       color=mod_color_map[mod_type], edgecolor='none')
            bar_base = hand_mod_perc * 100 + bar_base

            # --- shadow bar: up (↑) / down (↓) breakdown ---
            if mod_type == 'ipsi':
                tval_series = event_df[(event_df['side'] == 'ipsilateral') &
                                       (event_df['mod_type'] == 'ipsi')]['tVal']
            elif mod_type == 'contra':
                tval_series = event_df[(event_df['side'] == 'contralateral') &
                                       (event_df['mod_type'] == 'contra')]['tVal']
            else:  # both:all — aggregate all both:* subtypes from contralateral side
                tval_series = event_df[(event_df['side'] == 'contralateral') &
                                       (event_df['mod_type'].isin(both_subtypes))]['tVal']
            up_perc   = (tval_series > 0).sum() / n_neurons * 100
            down_perc = (tval_series < 0).sum() / n_neurons * 100
            color_full  = mod_color_map[mod_type]
            color_light = scale_lightness(rgb(color_full), 1.7)
            color_dark = scale_lightness(rgb(color_full), 1.2)
            shadow_x = x[e_i] + offset + width_main/2 + width_shadow/2
            ax.bar(shadow_x, down_perc, width_shadow, color=color_light, edgecolor='none')
            ax.bar(shadow_x, up_perc,   width_shadow, bottom=down_perc, color=color_dark, edgecolor='none')

            multiplier += 1
    ax.set_ylim([0, max_height])
    ax.set_xlabel('Epochs')
    ax.set_title(f'{region}')
    ax.set_xticks(x + step * multiplier / 2,
                  [e.split(' ')[0] for e in list(epoch_window_map.keys())[1:]])
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
            bbox_inches='tight', dpi=400)
