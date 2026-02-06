import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas
import torch
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from plot_utils import pc_subplot, epoch_window_map
from DSA import DSA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS
import pandas as pd
from DSA.stats import *
from Monkey import *

from sig_proc import *

import seaborn as sns
sns.set_theme()
sns.set_style(style='white')

def region_pca(region_map, pop_sdf, region_name):
    # print('Computing PCA')
    pca_sdf = []
    for key in region_map.keys():
        if region_name in key:
            pca_ind = region_map[key]
            pca_sdf.append(pop_sdf[pca_ind[0]:pca_ind[1]])
    pca_sdf = torch.vstack(pca_sdf)
    # U, S, V = torch.pca_lowrank(pca_sdf, center=True, q=q)
    cov = torch.cov(pca_sdf)
    sdf_square = center_sdf(pca_sdf)['square']
    # U, S, V = torch.linalg.svd(cov, full_matrices = False)
    U, S, V = torch.linalg.svd(sdf_square, full_matrices=False)

    return U, S, V.T, cov

def hemi_variance(pc_dict, region_map, baseline_region, target_regions, orientation='vertical', pc_dims=10,
                  all_regions=False, plot_cov = False, fig = plt.figure(), indices=0):
    #First, define relevant variables from parameters
    var_plot_dict = {}
    hemi_color = {'c': 'b', 'i': 'r'}
    hemi_map = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_hemi, base_area = baseline_region[0], baseline_region[1:]
    base_pca = pc_dict[orientation][baseline_region]
    base_cov = base_pca['cov']
    if pc_dims == 'all':
        pc_dims = base_cov.shape[0]
    base_pcs = base_pca['V']
    base_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@base_cov@base_pcs), dim=0)
    base_var_trace = base_var_sum.max()
    if plot_cov:
        f_cov = plt.figure(figsize=(8,8))
        plt.imshow(base_cov, norm=mpl.colors.SymLogNorm(linthresh=.001, linscale=.01), cmap='RdBu_r')
        plt.title(f'Covariance Matrix for {baseline_region}, {orientation}')
        plt.colorbar()
        var_plot_dict['Covariance'] = f_cov
    plt.figure(fig)
    ax_var = fig.axes[indices]
    ax_var.scatter(np.arange(1, pc_dims+1), base_var_sum[:pc_dims]/base_var_trace,
                label = hemi_map[base_hemi], c = hemi_color[base_hemi])
    for target_region in target_regions:
        target_hemi, target_area = target_region[0], target_region[1:]
        if target_region == baseline_region:
            continue
        if target_area != base_area and not all_regions:
            # print(f'Skipping {target_area}')
            continue
        target_pca = pc_dict[orientation][target_region]
        target_cov = target_pca['cov']
        target_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@target_cov@base_pcs), dim=0)
        ax_var.scatter(np.arange(1, pc_dims+1), target_var_sum[:pc_dims]/target_var_sum.max(),
                       label=hemi_map[target_hemi], c=hemi_color[target_hemi])
    ax_var.set_ylim([0, 1.03])
    ax_var.set_title(f'{orientation.capitalize()}, {hemi_map[base_hemi].capitalize()} PCs')
    if indices % 2 == 0:
        ax_var.set_ylabel('Variance Explained')
    # f_var.legend()
    var_plot_dict['Variance'] = fig
    return var_plot_dict

def orientation_variance(pc_dict, base_orientation, baseline_region, pc_dims=10,
                  all_regions=False, plot_cov = False, fig = plt.figure(), indices=0):

    #First, define relevant variables from parameters
    orient_plot_dict = {}
    target_orientation = ['horizontal', 'vertical']
    target_orientation.remove(orientation)
    target_orientation = target_orientation[0]
    orient_color = {'vertical': 'r', 'horizontal': 'b'}
    hemi_map = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_hemi, base_area = baseline_region[0], baseline_region[1:]
    base_pca = pc_dict[base_orientation][baseline_region]
    base_cov = base_pca['cov']
    if pc_dims == 'all':
        pc_dims = base_cov.shape[0]
    base_pcs = base_pca['V']
    base_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@base_cov@base_pcs), dim=0)
    base_var_trace = base_var_sum.max()
    plt.figure(fig)
    ax_var = fig.axes[indices]
    ax_var.scatter(np.arange(1, pc_dims+1), base_var_sum[:pc_dims]/base_var_trace,
                label = base_orientation, c = orient_color[base_orientation])
    #Now, compute target orientation variance explained by base orientation PCs
    target_pca = pc_dict[target_orientation][baseline_region]
    target_cov = target_pca['cov']
    target_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@target_cov@base_pcs), dim=0)
    ax_var.scatter(np.arange(1, pc_dims+1), target_var_sum[:pc_dims]/target_var_sum.max(),
                   label=target_orientation, c=orient_color[target_orientation])
    ax_var.set_ylim([0, 1.03])
    ax_var.set_title(f'{hemi_map[base_hemi].capitalize()}, {orientation.capitalize()} PCs')
    if indices % 2 == 0:
        ax_var.set_ylabel('Variance Explained')
    # f_var.legend()
    orient_plot_dict['Variance'] = fig
    return orient_plot_dict

def region_projection(all_sdf, pc_dict, region_map, data_region, pc_region, pc_dims=3, orientation='vertical',
                      fig = plt.figure(), indices=0, bin_size=5, epoch_windows=epoch_window_map):
    hemi_map = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_pca = pc_dict[orientation][pc_region]
    base_pcs = base_pca['V']
    data_ind = region_map[data_region]
    data = all_sdf[data_ind[0]:data_ind[1]]
    projected = data.T @ base_pcs
    plt.figure(fig)
    ax_proj = fig.axes[indices]
    x_proj = torch.arange(1, data.shape[1]*bin_size, step=bin_size)
    for pc in range(pc_dims):
        ax_proj.plot(x_proj, projected[:, pc], label = f'PC {pc+1}')
    epoch_lines = []
    for epoch in epoch_windows.keys():
        epoch_lines.append(epoch_windows[epoch]['time'])
    ax_proj.vlines(epoch_lines, ymin = projected[:, :pc].min(), ymax = projected[:, :pc].max(), colors='k')
    ax_proj.set_xticks(epoch_lines, labels=epoch_windows.keys())
    ax_proj.set_title(f'{orientation.capitalize()}, {hemi_map[pc_region[0]]}')
    plot_dict = {'Projection': fig}
    return plot_dict

def cond_neuron_plot(neuron_dict, epoch_window_map, plot_neurons = 4, fig_size = (9, 12), legend_elements=[],
                     cond_shapes = {'c_vertical': '+', 'c_horizontal': 'x', 'i_vertical': '|', 'i_horizontal': '.'},
                     mean_plot = False):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    clr_list = prop_cycle.by_key()['color']
    n_rows = len(neuron_dict.keys())
    cond_fig, cond_axs = plt.subplots(nrows= n_rows, ncols=1, figsize=fig_size)
    for cond_idx, condition in enumerate(neuron_dict.keys()):
        condition_sdf = neuron_dict[condition]['sdf']
        condition_ax = cond_axs[cond_idx]
        if not mean_plot:
            for o_condition in neuron_dict.keys():
                if o_condition == condition:
                    continue
                o_cond_idx = neuron_dict[o_condition]['indices']
                for o_idx in range(3):
                    color = 'gray'
                    condition_ax.plot(torch.arange(1, cond_sdf.shape[1] * binsize + 1, binsize),
                                condition_sdf[o_cond_idx[:plot_neurons, o_idx]].T,
                                      marker=cond_shapes[o_condition], markerfacecolor='gray', markevery= 5,
                                       ms = 10, label=f'PC {o_idx + 1}', alpha =0.6,
                                       color=color, linestyle=(0, (5, max(1, 5*o_idx))))
        for pc_idx in range(3):
            color=clr_list[pc_idx]
            if not mean_plot:
                condition_ax.plot(torch.arange(1, cond_sdf.shape[1] * binsize + 1, binsize),
                                            condition_sdf[neuron_dict[condition]['indices'][:plot_neurons, pc_idx]].T,
                                            color=color, linewidth = 2, linestyle=(0, (5, max(0, 2*pc_idx*0))))
            else:
                sdf_mean = condition_sdf[neuron_dict[condition]['indices'][:plot_neurons, pc_idx]].mean(dim=0)
                sdf_error = condition_sdf[neuron_dict[condition]['indices'][:plot_neurons, pc_idx]].std(dim=0)
                condition_ax.plot(torch.arange(1, cond_sdf.shape[1] * binsize + 1, binsize),
                                  sdf_mean, color=color, linewidth=2, linestyle=(0, (5, max(0, 2 * pc_idx * 0))))
                condition_ax.fill_between(torch.arange(1, cond_sdf.shape[1] * binsize + 1, binsize),
                                  sdf_mean-sdf_error, sdf_mean+sdf_error, facecolor=color, alpha=0.5)
            if (cond_idx == 0):
                if mean_plot:
                    legend_elements.append(Line2D([0], [0], color=color,
                                                  label=f'PC {pc_idx + 1}'))
                else:
                    legend_elements.append(Line2D([0], [0], color=color, linestyle=(0, (5, max(1, 3*pc_idx))),
                                              label=f'PC {pc_idx+1}'))
        condition_ax.set_title(f'{condition}')
        times = [epoch_window_map[epoch]['time'] for epoch in epoch_window_map.keys()]
        condition_ax.vlines([epoch_window_map[epoch]['time'] for epoch in epoch_window_map.keys()],
                      cond_sdf.min() * 1.1, cond_sdf.max() * 1.1, color='k')
        condition_ax.set_xticks(times, labels = [epoch for epoch in epoch_window_map.keys()])
        ymin = cond_sdf.min()
        ymax = cond_sdf.max()
        condition_ax.set_ylim([ymin, ymax])
    leg = cond_axs[0].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(.5, 1.1), ncols = 4)
    return cond_fig

# monkey_name_map = {'G':'Green', 'R':'Red'}
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp On', 'trialGraspOff':'Grasp Off', 'trialReachOn':'Reach'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   200]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-100,   500]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
binsize = 5
all_x = np.arange(0, current_time, binsize)
kernel_width = 25
full_window = np.arange(-1000, 1000+binsize, binsize)
new_popdict = True
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'

durs_file_name = f'{summary_dir}/durs_list_merged{len(monkey_name_map.keys())}.p'
with open(durs_file_name, 'rb') as durs_file:
    all_durs = pkl.load(durs_file)

if not os.path.exists(pca_dir):
    os.mkdir(pca_dir)

all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map.keys())}.p'
all_durs_filename = f'{summary_dir}/durs_list_merged{len(monkey_name_map.keys())}.p'
all_monkeys_filename = f'{summary_dir}/monkey_indices_merged{len(monkey_name_map.keys())}.p'
all_psth_filename = f'{summary_dir}/trialPSTH_bin{binsize}_k{kernel_width}_merged{len(monkey_name_map.keys())}.p'

with open(all_monkeys_filename, 'rb') as monkey_idcs_file:
    all_monkey_indices = pkl.load(monkey_idcs_file)
with open(all_sdf_filename, 'rb') as sdf_file:
    all_sdf_dict = pkl.load(sdf_file)
with open(all_psth_filename, 'rb') as psth_file:
    all_psth_dict = pkl.load(psth_file)

pop_filename = f'{pca_dir}/pop_dict_b{binsize}_k{kernel_width}_merged{len(monkey_name_map.keys())}.p'
merged_pop_dict = {'sdf':[], 'idx':0, 'psth':[]}
if os.path.exists(pop_filename) and not new_popdict:
    with open(pop_filename, 'rb') as pop_file:
        pop_tuple = pkl.load(pop_file)
        merged_pop_dict= pop_tuple
        print('Population Dictionary Loaded')
else:
    merged_population_dict = {}
    merged_psth_dict = {}
    region_map = {'idx':0}
    condition_map  = {}
    cortex_map = {'idx':0}
    for area in all_sdf_dict.keys():
        region, orientation = area.split('_')
        side, cortex = region[0], region[1:]
        if orientation not in merged_population_dict.keys():
            merged_population_dict[orientation] = []
            merged_psth_dict[orientation] = []
        if cortex not in condition_map.keys():
            condition_map[cortex] = {'idx':0}
        sdf = all_sdf_dict[area]
        sdf = sdf[:,0].T
        psth = all_psth_dict[area].transpose(2,1,0)
        merged_psth_dict[orientation].append(psth)
        sdf_max = sdf.max(axis=1)+5*binsize/1000
        # zero_mask = sdf.max(axis=1) > 0
        sdf = sdf/np.repeat(np.expand_dims(sdf_max, 1), sdf.shape[1], axis=1)
        merged_population_dict[orientation].append(sdf)
        if cortex not in cortex_map.keys():
            cortex_map[cortex] = [cortex_map['idx'], cortex_map['idx']]
        cortex_map[cortex][1]+= sdf.shape[0]
        cortex_map['idx'] += sdf.shape[0]
        if region not in region_map.keys():
            region_boundaries = [region_map['idx'], region_map['idx'] + sdf.shape[0]]
            region_map[region] = region_boundaries
            region_map['idx'] += sdf.shape[0]
        or_key = f'{side}_{orientation}'
        if or_key not in condition_map[cortex].keys():
            region_boundaries = [condition_map[cortex]['idx'], condition_map[cortex]['idx'] + sdf.shape[0]]
            condition_map[cortex][or_key] = region_boundaries
            condition_map[cortex]['idx'] += sdf.shape[0]
        merged_pop_dict['sdf'].append(sdf)
        merged_pop_dict['psth'].append(psth)
    for orientation, sdf_list in merged_population_dict.items():
        merged_population_dict[orientation] = np.vstack(sdf_list)
    for orientation, psth_list in merged_psth_dict.items():
        merged_psth_dict[orientation] = np.vstack(psth_list)
    del region_map['idx']
    for cor in condition_map.keys():
        del condition_map[cor]['idx']
    del cortex_map['idx']
    del merged_pop_dict['idx']
    # cortex_map['All'] = max([e[1] for e in list(cortex_map.values())])
    merged_pop_sdf = torch.Tensor(np.vstack(merged_pop_dict['sdf']))
    merged_pop_psth = torch.Tensor(np.vstack(merged_pop_dict['psth']))
    merged_pop_dict['sdf'] = merged_pop_sdf
    merged_pop_dict['psth'] = merged_pop_psth
    pop_sdf_dict = center_sdf(merged_pop_sdf)
    pop_pca_filename = f'{summary_dir}/merged{len(monkey_name_map.keys())}_sdfPCA_bin{binsize}_k{kernel_width}.p'
    pop_psth_filename = f'{summary_dir}/merged{len(monkey_name_map.keys())}_psth_bin{binsize}_k{kernel_width}.p'
    merged_pop_dict['region_map'] = region_map
    merged_pop_dict['cortex_map'] = cortex_map
    merged_pop_dict['condition_map'] = condition_map
    with open(pop_filename, 'wb') as pop_file:
        pkl.dump(merged_pop_dict, pop_file)
    # with open(pop_psth_filename, 'wb') as psth_file:
    #     pkl.dump(pop_pca_vals, psth_file)

pca_overwrite = True
cortex_map = merged_pop_dict['cortex_map']
merged_pop_sdf = merged_pop_dict['sdf']
condition_map = merged_pop_dict['condition_map']
n_angles = 5
full_angles = {}
n_cort = len(cortex_map.keys())-1
cortex_colors = {'M1':'b', 'PMd':'r', 'PMv':'g'}
epoch_shapes = {'Cue': 'o', 'Grasp On': '^'}
event_shapes = {'Reach': 's', 'Grasp Off': 'v'}
legend_elements = [Line2D([0], [0], color='w', marker=epoch_shapes[e],
                          markerfacecolor='k', label=e) for e in epoch_shapes.keys()]
legend_elements+= [Line2D([0], [0], color='w', marker=event_shapes[e],
                          markerfacecolor='k', label=e, alpha=0.4) for e in event_shapes.keys()]
cond_shapes = {'c_vertical': "+", 'c_horizontal': "x", 'i_vertical': "1", 'i_horizontal': "."}
region_var_explained = {}

event_marker_idx = {}
for event in all_durs.keys():
    event_mean = np.mean(all_durs[event])
    event_std = np.std(all_durs[event])
    mean_idx = [np.argmin(np.abs(all_x - event_mean))]
    std_idcs = [np.argmin(np.abs(all_x-(event_mean-event_std))), np.argmin(np.abs(all_x+event_std))]
    event_marker_idx[event] = {'mean':mean_idx, 'std':std_idcs}

monkeys = {}
cond_data = {}
cond_neuron_dict = {}
cortex_angles = {}
for cortex in cortex_map.keys():
    monkey_indices = all_monkey_indices[cortex]
    del monkey_indices['count']
    monkey_indices['All'] = np.arange(0, max([e[-1] for e in list(monkey_indices.values())])+1)
    pca_filename = f'{pca_dir}/PCA_merged{len(monkey_name_map.keys())}_{cortex}_b{binsize}_k{kernel_width}.p'
    cort = cortex_map[cortex]
    sdf = merged_pop_sdf[cort[0]:cort[1]] #cortex sdf
    for monkey in monkey_indices.keys():
        if monkey == 'count':
            continue
        if monkey not in monkeys.keys():
            monkeys[monkey] = Monkey(monkey, kernel_width, binsize, epoch_window_map)
        monkey_obj = monkeys[monkey]
        monkey_obj.get_indices(monkey_indices, cortex)
        monkey_obj.set_condition_map(condition_map)
        monkey_obj.get_savedir(summary_dir)
        cortex_sdf = monkey_obj.get_sdf(sdf, cortex)
        cortex_svd = monkey_obj.get_svd(cortex)
        monkey_obj.get_trial_psth(sdf, cortex)

        V_cort = cortex_svd['Full']['V']
        S_cort = cortex_svd['Full']['S']
        fig = plt.figure(cortex, figsize=(7, 10))
        ax2 = fig.add_subplot(2, 2, 1)
        ax2a = fig.add_subplot(2, 2, 4)
        ax3 = fig.add_subplot(2, 2, 2, projection='3d')
        ax_var = fig.add_subplot(2, 2, 3)
        cond_neuron_dict[monkey] = {}
        cond_data[monkey] = []
        cortex_angles[monkey] = {}
        monkey_index = np.array(monkey_indices[monkey])
        if monkey != 'All':
            m_neurons = len(monkey_index)
        else:
            m_neurons = max(cort)
        for idx, (cond, c_ind) in enumerate(condition_map[cortex].items()):
            cond_sdf = sdf[c_ind[0]:c_ind[1]][monkey_index]
            cond_svd = cortex_svd[cond]
            V_cond = cond_svd['V']
            cond_pc_neurons = V_cond[:, :3].abs().sort(dim=0, descending=True)
            cond_proj = cond_sdf.T @ V_cort #Use full-cortex vectors for projections

            cond_data[monkey].append(cond_sdf.T)
            cond_event_means = {}
            cond_event_stds = {}
            # (cond_pc_neurons.values ** 2 / ((V[c_ind[0]:c_ind[1], :3] ** 2).sum(dim=0)))
            cond_neuron_dict[monkey][cond] = {'sdf': cond_sdf, 'indices': cond_pc_neurons.indices}
            # cond_ax = axs[idx]
            prop_cycle = plt.rcParams['axes.prop_cycle']
            clrs = prop_cycle.by_key()['color']

            var_plot_pcs = 'all'
            cond_cov = cortex_svd[cond]['cov']
            cond_var_sum = torch.cumsum(torch.diagonal(V_cond.T @ cond_cov @ V_cond), dim=0)
            cond_var_explained = cond_var_sum / cond_var_sum.max()
            cond_var_len = cond_var_explained.shape[0]
            if var_plot_pcs == 'all':
                var_plot_pcs = cond_var_len
            ax_var.scatter(np.arange(1, var_plot_pcs + 1), cond_var_explained[:var_plot_pcs], label=cond[:6])
            for event in event_marker_idx.keys():
                mean_idx = event_marker_idx[event]['mean']
                std_idx = event_marker_idx[event]['std']
                cond_event_means[event] = cond_proj[mean_idx, :3].T
                cond_event_stds[event] = [cond_proj[std_idcs[0], :3].T, cond_sdf[std_idcs[1], :3].T]
            x, y, z = cond_proj[:, :3].T
            ax3.plot(x, y, z, label=cond)
            ax2.plot(x, y, label=cond)
            for epoch in epoch_window_map:
                epoch_time = epoch_window_map[epoch]['time']
                epoch_idx = epoch_time // binsize
                ax3.scatter(x[epoch_idx], y[epoch_idx], z[epoch_idx], marker=epoch_shapes[epoch], c='k')
                ax2.scatter(x[epoch_idx], y[epoch_idx], marker=epoch_shapes[epoch], c='k')
            for event in event_marker_idx.keys():
                em_x, em_y, em_z = cond_event_means[event]
                ax3.scatter(em_x, em_y, em_z, marker=event_shapes[event_map[event]], c='k', alpha=0.4, s=100)
                ax2.scatter(em_x, em_y, marker=event_shapes[event_map[event]], c='k', alpha=0.4, s=100)
            # ax2a.plot(y, z, label=cond)
            angles = {}
            cortex_angles[monkey][cond] = {'v': V_cond, 'angles': angles}
        mean_plot = True
        mean_neurons = 5
        if not mean_plot:
            cond_legend_elements = [Line2D([0], [0], color='gray', marker=cond_shapes[c], markerfacecolor='gray',
                                           ms=15, label=c) for c in condition_map[cortex].keys()]
            mean_string = ''
        else:
            cond_legend_elements = []
            mean_string = f', Mean over {mean_neurons} Neurons'
        cort_sdf_fig = cond_neuron_plot(cond_neuron_dict[monkey], epoch_window_map, plot_neurons=mean_neurons,
                                        legend_elements=cond_legend_elements, cond_shapes=cond_shapes, mean_plot=True)
        # cort_sdf_fig.suptitle(f'SDF for {cortex}{mean_string}, {len(monkey_name_map.keys())} Monkeys', size='xx-large')
        cort_sdf_fig.suptitle(f'SDF for {cortex}{mean_string}, Monkey {monkey}', size='xx-large')
        plot_mean = ''
        if mean_plot:
            plot_mean += f'_mean{mean_neurons}'
        cort_sdf_fig.tight_layout()
        # cort_sdf_fig.savefig(f'{pca_dir}/{cortex}_merged{len(monkey_name_map.keys())}_{plot_mean}_SDF.png', bbox_inches='tight', dpi=200)
        # cort_sdf_fig.savefig(f'{pca_dir}/{cortex}_monkey{monkey}{plot_mean}_SDF.png',
        #                      bbox_inches='tight', dpi=200)
        plt.close(cort_sdf_fig)
        leg2 = ax3.legend(handles=legend_elements, labels=list(epoch_shapes.keys()) + list(event_shapes.keys()),
                          ncols=2, loc='lower center')
        sns.move_legend(ax3, 'upper center', ncols=2, bbox_to_anchor=(.5, -.1))
        conds = len(cortex_angles[monkey].keys()) - 1
        angle_matrix = torch.empty(conds, conds * n_angles)
        for c_i, cond in enumerate(cortex_angles[monkey].keys()):
            if c_i == 0:
                continue
            for c_j, o_cond in enumerate(cortex_angles[monkey][cond]['angles'].keys()):
                angle_matrix[c_i - 1, c_j * min(n_angles, m_neurons):(c_j + 1) * min(n_angles, m_neurons)] = (
                    torch.Tensor(cortex_angles[monkey][cond]['angles'][o_cond][:n_angles]))
                # print(angle_matrix)
        # angle_fig = plt.figure()
        im = ax2a.pcolor(angle_matrix, cmap=mpl.colormaps['magma_r'],
                         norm=colors.Normalize(vmin=0.2, vmax=1, clip=True))  # norm=colors.LogNorm(vmin=0.4, vmax = 1)
        div = make_axes_locatable(ax2a)
        cax = div.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax2a.set_yticks(ticks=[0.5, 1.5, 2.5], labels=['ch', 'iv', 'cv'])
        ax2a.set_xticks(ticks=[n_angles / 2, 3 * n_angles / 2, 5 * n_angles / 2], labels=['ih', 'ch', 'iv'])
        ax2a.vlines([n_angles, 2 * n_angles], 0, 3, colors='w')
        ax2a.hlines([1, 2], 0, conds * n_angles, colors='w')
        # ax2a.colorbar(label='Cos Sim.')
        ax2a.set_title(f'Principal Angles Between First {n_angles} PCs')
        # plt.savefig(f'{pca_dir}/AngleMatrix_{cortex}.png', dpi = 300, bbox_inches= 'tight')

        var = S_cort.square()
        var_sum = torch.cumsum(var, 0)
        ax_var.scatter(range(1, var_plot_pcs + 1), var_sum[:var_plot_pcs] / var_sum[-1], label='All', color='k')
        ax_var.set_xlabel('PC index')
        ax_var.set_ylabel('Variance Explained')
        ax_var.set_ylim([0, 1.05])
        ax_var.legend(ncol=2, loc='right')

        ax2.set_title(f'Conditions plotted onto first 2 {cortex} PCs')
        # ax2a.set_title(f'Conditions plotted onto second 2 {cortex} PCs')
        ax3.set_title(f'Conditions plotted onto first 3 {cortex} PCs')
        ax3.set_xlabel('PC 1')
        ax3.set_ylabel('PC 2')
        ax3.set_xlabel('PC 1')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        # ax2a.set_xlabel('PC 2')
        # ax2a.set_ylabel('PC 3')
        ax2.legend()
        # fig.savefig(f'{pca_dir}/PCTraj3D_{cortex}_merged{len(monkey_name_map.keys())}.png', dpi = 300, bbox_inches= 'tight')
        fig.savefig(f'{pca_dir}/PCTraj3D_{cortex}_monkey{monkey}_NEW.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

for m_name, monkey in monkeys.items():
    monkey.save_monkey()
    print(f'Saving Monkey {m_name}')