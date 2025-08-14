import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
import matplotlib.colors as colors

from plot_utils import pc_subplot, epoch_window_map

from sig_proc import *

import seaborn as sns

condition_colors = {'cont-hori': 'orange', 'cont-vert': 'red', 'ipsi-hori': 'blue', 'ipsi-vert': 'green'}
binsize = 5
kernel_width = 25


def raster_plot(neuron_list, region_spikes, event_list, epoch_window_map, conditions=None, n_cols=4, indices=None,
                sdf = False, window = [0,820]):
    total_neurons = np.array(neuron_list).shape[0]
    n_rows = math.ceil(total_neurons/n_cols)
    fig, axs = plt.subplots(nrows=math.ceil(total_neurons/n_cols),ncols=n_cols, figsize=(4*n_cols, 3.5*n_rows),
                            constrained_layout=True)
    epoch_times = [epoch_window_map[event]['time'] for event in epoch_window_map.keys()]
    for idx, neuron in enumerate(neuron_list):
        idx_string = ''
        f_weight = 'normal'
        ax = axs[idx // n_cols][idx % n_cols]
        spikes = region_spikes[neuron - 1]
        neuron_spikes = [[]] * len(spikes)
        if idx%50 == 0:
            print(f'Neuron {neuron}')
        if indices is not None:
            neuron_idx = np.where(indices == neuron-1)[1]
            idx_string += f', ch:{neuron_idx[0]+1}, cv:{neuron_idx[1]+1}, ih:{neuron_idx[2]+1}, iv:{neuron_idx[3]+1}'
            sig_neuron = neuron_idx < 10
            if sig_neuron.sum()>0:
                f_weight = 'bold'
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(3)
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
        ax.set_title(f'Neuron {neuron}{idx_string}',fontweight=f_weight)
        ax.set_ylabel(f'Trial')
        ax.set_xlabel(f'Time (ms)')
        max_time = epoch_window_map['Grasp Off']['time'] + epoch_window_map['Grasp Off']['window'][1]
        ax.set_xlim([0, max_time])
        ax.set_ylim([0,len(spikes)])
        if conditions is not None:
            for condition in condition_colors.keys():
                cond_inds = np.where((conditions[idx][:,0]==condition[0]) * (conditions[idx][:,1]==condition.split('-')[1][0]))
                cond_max, cond_min = cond_inds[0].max(), cond_inds[0].min()
                cond_rect = mpl.patches.Rectangle((0, cond_min), width=max_time, height=cond_max-cond_min,
                                                  color=condition_colors[condition], alpha=0.1)
                ax.add_patch(cond_rect)
                # ax.annotate(xy=(1, cond_max), text=condition, color=condition_colors[condition], alpha = 0.6, fontsize=9)
        for epoch_time in epoch_times:
            ax.axvline(epoch_time, alpha=.8, color='k')
        ax.invert_yaxis()
        ax.eventplot(neuron_spikes, color='k')
        if conditions is not None:
            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_mask = (conditions[idx][:,0]==condition[0]) * (conditions[idx][:,1]==condition.split('-')[1][0])
                cond_inds = np.where(cond_mask)
                cond_max, cond_min = cond_inds[0].max(), cond_inds[0].min()
                if sig_neuron[cond_i]:
                    bbox = dict(boxstyle="square", fc = 'none', ec="black", lw=2, pad=0.1)
                else:
                    bbox = None
                ax.annotate(xy=(window[1]-5, cond_min+0.5), rotation=270, ha='right', va='top', text=condition,
                            color=condition_colors[condition],
                            alpha=0.8, fontsize=9.5, bbox = bbox)
                if sdf:
                    cond_spike_list = []
                    for i in range(cond_mask.shape[0]):
                        if cond_mask[i]:
                            cond_spike_list.append(neuron_spikes[i])
                    spike_input = np.concatenate(cond_spike_list)
                    psth_input = np.ones((spike_input.shape[0],2))
                    psth_input[:,1] = spike_input
                    cond_psth = gen_psth(psth_input, binsize=binsize, window=np.array(window))
                    cond_sdf, _ = gen_sdf(cond_psth[:, 1:], w=kernel_width, bin_size=binsize, ftype='Gauss',
                                          multi_unit=False)
                    sdf_scaled = -cond_sdf[:,0,0]*(cond_max-cond_min)*2 + cond_max# - (cond_max-cond_min)/8
                    ax.plot(cond_psth[:,0],sdf_scaled,color=condition_colors[condition])
    return fig

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'


pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
indices_filename = f'{pca_dir}_pcIndices.p'

epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   100]},
                   'Reach':     {'event': 'trialReachOn',    'window': [-100,    60]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-60,    100]},
                   'Grasp Off': {'event': 'trialGraspOff',   'window': [-100,   100]}}
current_time = 0
epoch_window = [0,0]
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_min = min(current_time - epoch_window_map[epoch]['window'][0], epoch_window[0])
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]
    current_max = current_time
    if epoch != 'Grasp Off':
        current_max = current_max + epoch_window_map[epoch]['window'][1]
    epoch_window = [current_min, current_max]

summary_file_name = f'{summary_dir}/spike_summary.p'
spike_file_name = f'{summary_dir}/spike_list.p'
event_file_name = f'{summary_dir}/event_list.p'
condition_file_name = f'{summary_dir}/condition_list.p'

selected_neurons = {'M1': [[63, 11, 39, 73, 71, 115], [42, 115, 147, 61, 149, 74], [61, 21, 85, 149, 53, 65], [81, 70, 37, 61, 80, 65]],
                    'PMd':[[178, 2, 166, 88, 68, 175], [178, 2, 49, 48, 102, 88], [78, 181, 111, 175, 146,118],[78, 77,181, 178,22, 6]],
                    'PMv':[[225,177, 70, 4, 145, 68],[245,110, 225, 215, 49,250], [171, 50, 27,162, 245, 98], [1, 110, 4, 181,162,131]]}

with open(summary_file_name, 'rb') as summary_file:
    area_summary_dict = pkl.load(summary_file)
with open(spike_file_name, 'rb') as spike_file:
    all_spikes = pkl.load(spike_file)
with open(event_file_name, 'rb') as event_file:
    all_events = pkl.load(event_file)
with open(condition_file_name, 'rb') as condition_file:
    all_conditions = pkl.load(condition_file)
with open(indices_filename, 'rb') as indices_file:
    all_indices = pkl.load(indices_file)

cond_map = {0:'ipsi-hori', 1:'ipsi-vert', 2:'contra-hori', 3:'contra-vert'}

for region in selected_neurons.keys():
    print(f'{region}')
    region_conditions = all_conditions[region]
    region_indices = all_indices[region]
    n_neurons = region_indices.shape[1]
    # n_neurons = 27
    full_fig = raster_plot(np.arange(1,n_neurons+1), all_spikes[region], all_events[region],
                           epoch_window_map, all_conditions[region], n_cols=9, indices = region_indices, sdf=True,
                           window=epoch_window)
    conditions_string = ''
    full_fig.suptitle(f'Neurons in {region}')
    plt.savefig(f'{pca_dir}/Raster_{region}all.png', bbox_inches='tight', dpi=200)
    print(f'Figure Saved for {region}')
    # for i in range(4):
    #     fig = raster_plot(selected_neurons[region][i], all_spikes[region], all_events[region], epoch_window_map, n_cols=3)
    #     fig.suptitle(f'Raster plots for {region}, Top {cond_map[i]} Neurons')
    #     plt.tight_layout()
    #     fig.savefig(f'{pca_dir}/Raster_{region}{cond_map[i]}.png', bbox_inches='tight', dpi=200)