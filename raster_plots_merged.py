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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        if n_cols < total_neurons:
            ax = axs[idx // n_cols][idx % n_cols]
        else:
            ax = axs[idx]
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
        max_time = epoch_window_map['Grasp On']['time'] + epoch_window_map['Grasp On']['window'][1]
        ax.set_xlim([0, max_time])
        ax.set_ylim([0,len(spikes)])
        cond_order = np.arange(len(conditions[idx]))
        cond_masks = {}
        cond_idx = 0
        if conditions is not None:
            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_mask = (conditions[idx][:, 0] == condition[0]) * (
                            conditions[idx][:, 1] == condition.split('-')[1][0])
                cond_masks[condition] = cond_mask
                cond_inds = np.where(cond_mask)
                trial_count = len(cond_inds[0])
                cond_min = cond_idx
                cond_max = cond_idx+trial_count
                cond_order[cond_min:cond_max] = cond_inds[0]
                # cond_max, cond_min = cond_inds[0].max(), cond_inds[0].min()
                cond_rect = mpl.patches.Rectangle((0, cond_min), width=max_time, height=cond_max-cond_min,
                                                  color=condition_colors[condition], alpha=0.1)
                ax.add_patch(cond_rect)
                cond_idx = cond_max
        ax.invert_yaxis()
        ordered_spikes = [neuron_spikes[i] for i in cond_order]
        ax.eventplot(ordered_spikes, color='k')
        if conditions is not None:
            cond_idx = 0
            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_mask = cond_masks[condition]
                trial_count = len(np.where(cond_mask)[0])
                cond_min = cond_idx
                cond_max = cond_min + trial_count
                cond_idx = cond_max
                if sdf:
                    cond_spike_list = []
                    for i in range(cond_mask.shape[0]):
                        if cond_mask[i]:
                            cond_spike_list.append(ordered_spikes[i])
                    spike_input = np.concatenate(cond_spike_list)
                    psth_input = np.ones((spike_input.shape[0],2))
                    psth_input[:,1] = spike_input
                    cond_psth = gen_psth(psth_input, binsize=binsize, window=np.array(window))
                    cond_sdf, _ = gen_sdf(cond_psth[:, 1:], w=kernel_width, bin_size=binsize, ftype='Gauss',
                                          multi_unit=False)
                    sdf_scaled = -cond_sdf[:,0,0]*(cond_max-cond_min)*2 + cond_max# - (cond_max-cond_min)/8
                    ax.plot(cond_psth[:,0],sdf_scaled,color=condition_colors[condition])
                if indices is not None:
                    if sig_neuron[cond_i]:
                        bbox = dict(boxstyle="square", fc = 'none', ec="black", lw=2, pad=0.1)
                else:
                    bbox = None
                ax.annotate(xy=(window[1] - 5, cond_min + 0.5), rotation=270, ha='right', va='top', text=condition,
                            color=condition_colors[condition],
                            alpha=0.8, fontsize=9.5, bbox=bbox)
        for epoch_time in epoch_times:
            ax.axvline(epoch_time, alpha=.8, color='k')

    return fig

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
merged_count = len(monkey_name_map.keys())

pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
indices_filename = f'{pca_dir}_pcIndices_merged{merged_count}.p'

epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   200]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-100,   500]}}
current_time = 0
epoch_window = [0,0]
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_min = min(current_time - epoch_window_map[epoch]['window'][0], epoch_window[0])
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]
    current_max = current_time
    if epoch != 'Grasp On':
        current_max = current_max + epoch_window_map[epoch]['window'][1]
    epoch_window = [current_min, current_max]

summary_file_name = f'{summary_dir}/spike_summary_merged{merged_count}.p'
spike_file_name = f'{summary_dir}/spike_list_merged{merged_count}.p'
event_file_name = f'{summary_dir}/event_list_merged{merged_count}.p'
condition_file_name = f'{summary_dir}/condition_list_merged{merged_count}.p'

selected_neurons = {'M1': {'Contra-Down':[153, 236, 200, 32], 'Contra-Up':[17, 211, 128, 35], 'Ipsi-Down':[114, 194, 87, 22], 'Ipsi-Up':[69, 162, 183, 41]},
                    'PMd':{'Contra-Down':[153, 236, 200, 32], 'Contra-Up':[17, 211, 128, 35], 'Ipsi-Down':[114, 194, 87, 22], 'Ipsi-Up':[69, 162, 183, 41]},
                    'PMv':{'Contra-Down':[153, 236, 200, 32], 'Contra-Up':[17, 211, 128, 35], 'Ipsi-Down':[114, 194, 87, 22], 'Ipsi-Up':[69, 162, 183, 41]}}

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
make_full_fig = False
for region in selected_neurons.keys():
    print(f'{region}')
    region_conditions = all_conditions[region]
    region_indices = all_indices[region]
    n_neurons = region_indices.shape[1]
    # n_neurons = 27
    if make_full_fig:
        full_fig = raster_plot(np.arange(1,n_neurons+1), all_spikes[region], all_events[region],
                               epoch_window_map, all_conditions[region], n_cols=12, indices = region_indices, sdf=True,
                               window=epoch_window)
        conditions_string = ''
        full_fig.suptitle(f'Neurons in {region}')
        plt.savefig(f'{pca_dir}/Raster_{region}all_merged{merged_count}.png', bbox_inches='tight', dpi=200)
        print(f'Figure Saved for {region}')
    mod_labels = ['Cue, Grasp', 'Cue, Not Grasp', 'Not Cue, Grasp', 'Not Cue, Not Grasp']
    for modulation, neurons in selected_neurons[region].items():
        neuron_conditions = [all_conditions[region][i] for i in neurons]
        fig = raster_plot(neurons, all_spikes[region], all_events[region], epoch_window_map, conditions = neuron_conditions,
                          n_cols=4, indices = None, sdf=True, window=epoch_window)
        for neuron, mod_label, ax in zip(neurons, mod_labels, fig.get_axes()):
            ax.set_title(f'{mod_label}, Neuron {neuron}')
        fig.suptitle(f'Raster plots for {region}, Horizontal {modulation} Modulation')
        plt.tight_layout()
        fig.savefig(f'{pca_dir}/Raster_{region}{modulation}.png', bbox_inches='tight', dpi=200)
        print(f'Figure Saved for {region}, {modulation}')