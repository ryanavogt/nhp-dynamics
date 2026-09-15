import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import matplotlib as mpl

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import numpy as np

from sig_proc import *
import pandas as pd
import matplotlib as mpl
from plot_utils import scale_lightness, add_headers, raster_plot
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from scipy.spatial.distance import pdist
from operator import itemgetter

from sig_proc import *

import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sns.set()
sns.set_style(style='white')
binsize=5
kernel_width=25

"""
Define maps for reference:
monkey_name_map: Map labels to full names   - Name label (R, G) -> Full name (Red, Green)
event_map: Name of events for plotting      - Raw event name -> Shortened event name (for labels on plots)
"""
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
merged_count = len(monkey_name_map.keys())
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp On'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-300,   200]},
                    'Grasp On': {'event': 'trialGraspOn',    'window': [-100,   300]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]
epoch_window = [0, current_time]
#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary/earlyLate'
pca_dir = f'Data/Processed/Summary/PCA_b{binsize}_k{kernel_width}'
save_dir = summary_dir

area_summary_dict_filename = f'{summary_dir}/spike_summary_earlyLate.p'
merged_mod_filename = f'{summary_dir}/MergedModDataFrame.p'
cluster_filename = f'{summary_dir}/Clusters.p'
spikes_filename = f'{summary_dir}/spike_list_earlyLate.p'
indices_filename = f'{pca_dir}/pcIndicesJoint_merged{merged_count}.p'
event_file_name = f'Data/Processed/Summary/event_list_merged{merged_count}.p'
condition_file_name = f'Data/Processed/Summary/condition_list_merged{merged_count}.p'
merged_filename = f'{summary_dir}/MergedMods.p'
spike_file_name = f'{summary_dir}/spike_list_earlyLate.p'

with open(area_summary_dict_filename, 'rb') as area_dict_file:
    area_summary_dict = pkl.load(area_dict_file)
with open(merged_mod_filename, 'rb') as merged_mod_file:
    merged_mod_df = pkl.load(merged_mod_file)
with open(cluster_filename, 'rb') as cluster_file:
    cluster_dict = pkl.load(cluster_file)
with open(spikes_filename, 'rb') as spikes_file:
    spike_dict = pkl.load(spikes_file)
with open(indices_filename, 'rb') as indices_file:
    pc_indices = pkl.load(indices_file)
with open(event_file_name, 'rb') as event_file:
    all_events = pkl.load(event_file)
with open(condition_file_name, 'rb') as condition_file:
    all_conditions = pkl.load(condition_file)
with open(merged_filename, 'rb') as merged_file:
    merged_mods = pkl.load(merged_file)
with open(spike_file_name, 'rb') as spike_file:
    all_spikes = pkl.load(spike_file)

region_list = ['M1', 'PMd', 'PMv']
plot_neurons = 10
plot_columns = 5
scatter_fig= plt.figure()
sc_ax = scatter_fig.gca()
sc_ax.set_xlabel('Base SDF Mean')
sc_ax.set_ylabel('Grasp SDF Mean')
use_conditions = False

for region in region_list:
    sdf_fig, sdf_axs = plt.subplots()
    all_event_means = merged_mod_df[merged_mod_df['region']==region]['event_mean']
    all_baseline_means = merged_mod_df[merged_mod_df['region']==region]['baseline_mean']
    sc_ax.scatter(all_baseline_means, all_event_means, label=region, alpha = 0.3)
    idx_list = np.range()
    # idx_list = pc_indices[region][:plot_neurons]
    # if region == 'M1':
    #     idx_list = [150, 156, 191, 196, 201, 206, 212, 216, 221, 226, 240, 241, 242, 243, 244]
    cluster_list = []
    mod_list = []
    condition_list = []
    event_means = []
    base_means = []
    for idx in idx_list:
        mod_vals = merged_mod_df[(merged_mod_df['n_idc']==int(idx))&(merged_mod_df['region']==region)&
                                 (merged_mod_df['event']=='Grasp')][['mod_type','tVal','side','event_mean','baseline_mean']]
        if list(mod_vals['tVal'])[-1] > 0:
            up_down = 'up'
        else: up_down = 'down'
        mod_list.append(f'{list(mod_vals['mod_type'])[-1]} {up_down}')
        cluster_list.append(cluster_dict[region][10][idx])
        condition_list.append(all_conditions[region][idx])
        event_means.append((mod_vals['side'], mod_vals['event_mean']))
        base_means.append((mod_vals['side'], mod_vals['baseline_mean']))
    if use_conditions:
        conditions = condition_list
        condition_string = '_conds'
    else:
        conditions = None
        condition_string = ''
    neuron_fig, neuron_axes = raster_plot(idx_list, all_spikes[region], all_events[region], epoch_window_map, n_cols=plot_columns,
                                          sdf=True, window=epoch_window, conditions=conditions, event_means = event_means,
                                          baseline_means = base_means)
    for idx, mod_val, cluster_val, ax in zip(idx_list, mod_list, cluster_list, neuron_axes.flatten()):
        ax.set_title(f'Neuron {idx+1}: {mod_val}, Group {cluster_val}')
    neuron_fig.suptitle(f'Raster Plot for top PC Neurons in {region}')
    neuron_fig.savefig(f'{pca_dir}/Raster_PCs_{region}{condition_string}.png', bbox_inches='tight', dpi=200)
sc_ax.legend()
# sc_ax.set_xscale('log')
# sc_ax.set_yscale('log')
scatter_fig.savefig(f'{pca_dir}/Event_Baseline_Correlation.png', bbox_inches='tight', dpi=200)
