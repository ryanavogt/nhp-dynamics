import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
import torch
from mpl_toolkits.mplot3d.axes3d import get_test_data
from plot_utils import pc_subplot

from sig_proc import *

import seaborn as sns
sns.set_theme()
sns.set_style(style='white')


def region_pca(region_map, pop_sdf, region_name):
    print('Computing PCA')
    pca_sdf = []
    for key in region_map.keys():
        if region_name in key:
            pca_ind = region_map[key]
            pca_sdf.append(pop_sdf[pca_ind[0]:pca_ind[1]])
    pca_sdf = torch.vstack(pca_sdf)
    # U, S, V = torch.pca_lowrank(pca_sdf, center=True, q=q)
    cov = torch.cov(pca_sdf)
    sdf_mean = pca_sdf.mean(dim=1, keepdim=True)
    sdf_centered = pca_sdf - sdf_mean.repeat(1, pca_sdf.shape[1])
    sdf_square = sdf_centered@sdf_centered.T
    # U, S, V = torch.linalg.svd(cov, full_matrices = False)
    U, S, V = torch.linalg.svd(sdf_square, full_matrices=False)

    return U, S, V.T, cov

def hemi_variance(pc_dict, region_map, baseline_region, target_regions, orientation='vertical', pc_dims=10,
                  all_regions=False, plot_cov = False, fig = plt.figure(), indices=0):
    """

    :param pc_dict:
    :param region_map:
    :param baseline_region:
    :param target_regions:
    :param orientation:
    :param pc_dims:
    :param all_regions:
    :param plot_cov:
    :param fig:
    :param indices:
    :return:
    """
    #First, define relevant variables from parameters
    var_plot_dict = {}
    hemi_color = {'c': 'b', 'i': 'r'}
    hemi_map = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_hemi, base_area = baseline_region[0], baseline_region[1:]
    base_pca = pc_dict[orientation][baseline_region]
    base_cov = base_pca['cov']
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
        ax_var.scatter(np.arange(1, pc_dims+1), target_var_sum[:pc_dims]/base_var_trace,
                       label=hemi_map[target_hemi], c=hemi_color[target_hemi])
    ax_var.set_ylim([0, 1])
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
    ax_var.scatter(np.arange(1, pc_dims+1), target_var_sum[:pc_dims]/base_var_trace,
                   label=target_orientation, c=orient_color[target_orientation])
    ax_var.set_ylim([0, 1])
    ax_var.set_title(f'{hemi_map[base_hemi].capitalize()}, {orientation.capitalize()} PCs')
    if indices % 2 == 0:
        ax_var.set_ylabel('Variance Explained')
    # f_var.legend()
    orient_plot_dict['Variance'] = fig
    return orient_plot_dict


monkey_name_map = {'R': 'Red', 'G': 'Green'}
event_map = {'trialRewardDrop': 'Cue', 'trialReachOn':'Reach', 'trialGraspOn':'GraspOn', 'trialEnd':'GraspOff'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   100]},
                   'Reach':     {'event': 'trialReachOn',    'window': [-100,    60]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-60,    100]},
                   'Grasp Off': {'event': 'trialGraspOff',   'window': [-100,   100]}}
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
kernel_width = 25
full_window = np.arange(-1000, 1000+binsize, binsize)
new_popdict = True
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'

pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
if not os.path.exists(pca_dir):
    os.mkdir(pca_dir)

all_sdf_filename = f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}.p'
with open(all_sdf_filename, 'rb') as sdf_file:
    all_sdf_dict = pkl.load(sdf_file)

pop_filename = f'{pca_dir}/pop_dict_merged_b{binsize}_k{kernel_width}.p'
if os.path.exists(pop_filename) and not new_popdict:
    with open(pop_filename, 'rb') as pop_file:
        pop_tuple = pkl.load(pop_file)
        merged_population_dict, region_map = pop_tuple
        print('Population Dictionary Loaded')
else:
    merged_population_dict = {}
    region_map = {'idx':0}
    for area in all_sdf_dict.keys():
        region, orientation = area.split('_')
        side, cortex = region[0], region[1:]
        if orientation not in merged_population_dict.keys():
            merged_population_dict[orientation] = []
        sdf = all_sdf_dict[area]
        sdf = sdf[:,0].T
        sdf_max = sdf.max(axis=1)+5*binsize/1000
        sdf = sdf/np.repeat(np.expand_dims(sdf_max, 1), sdf.shape[1], axis=1)
        merged_population_dict[orientation].append(sdf)
        if region not in region_map.keys():
            region_boundaries = [region_map['idx'], region_map['idx'] + sdf.shape[0]]
            region_map[region] = region_boundaries
            region_map['idx'] += sdf.shape[0]
    for orientation, sdf_list in merged_population_dict.items():
        merged_population_dict[orientation] = np.vstack(sdf_list)
    del region_map['idx']


pca_overwrite = False
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sides = ['i', 'c']
n_plot = 4
q = 20 # Estimate for number of PCs to use
pca_regions = ['cM1', 'iM1','cPMv', 'iPMv','cPMd', 'iPMd']
# projection_regions = ['PMd', 'PMv']
full_pca_dict = {}
all_area_plots = {}
for pca_region in pca_regions:
    hemisphere, region = pca_region[0], pca_region[1:]
    f = plt.figure(region, figsize = (15, 6))
    pca_filename = f'{pca_dir}/MergedPCA_{pca_region}_b{binsize}_k{kernel_width}.p'
    if os.path.exists(pca_filename) and not pca_overwrite:
        with open(pca_filename, 'rb') as pca_file:
            pca_dict = pkl.load(pca_file)
        new_pca = False
    else:
        pca_dict = {}
        new_pca = True
    for idx, orientation in enumerate(merged_population_dict.keys()):
        plot_idx = 2*(hemisphere=='i')+idx + 1
        ax = f.add_subplot(220+plot_idx)
        if orientation not in full_pca_dict.keys():
            full_pca_dict[orientation] = {}
        if new_pca:
            pca_dict[orientation] = {}
        pop_sdf = torch.Tensor(merged_population_dict[orientation]).detach()
        if new_pca:
            U,S,V, cov= region_pca(region_map, pop_sdf, pca_region)
            pca_vals = {'U': U, 'S': S, 'V': V, 'cov':cov}
            pca_dict[orientation] = pca_vals
        else:
            pca_vals = pca_dict[orientation]
            U, S, V, cov = pca_vals['U'], pca_vals['S'], pca_vals['V'], pca_vals['cov']
        full_pca_dict[orientation][pca_region] = pca_dict[orientation]
        bottoms = np.zeros(V.shape[0])
        for pc in range(3):
            ax.bar(np.arange(V.shape[0]), V[:, pc].abs(), bottom = bottoms, label = f'PC{pc+1}')
            bottoms = V[:, pc].abs() + bottoms
        if region == 'PMv':
            print(f'{region} sum: {bottoms.sum()}')
        ax.set_title(f'{pca_region}, {orientation.capitalize()}')
        ax.set_xlabel('Neuron No.')
        ax.set_ylabel('PC Value')

    handles, labels = ax.get_legend_handles_labels()
    f.suptitle(f'Neuron Map onto First PCs for {region}')
    f.legend(handles, labels, ncols=2)
    f.savefig(f'{pca_dir}/NeuronPCMap_{region}.png', dpi = 300)
    if new_pca:
        with open(pca_filename, 'wb') as pca_file:
            pkl.dump(pca_dict, pca_file)

# base_region = 'cM1'
pc_plot_dims = 10
var_plot_dict = {'M1':plt.subplots(2,2), 'PMd':plt.subplots(2,2), 'PMv': plt.subplots(2,2)}
orient_plot_dict = {'M1':plt.subplots(2,2), 'PMd':plt.subplots(2,2), 'PMv': plt.subplots(2,2)}
for orientation in merged_population_dict.keys():
    orient_idx = orientation == 'vertical'
    pop_sdf = torch.Tensor(merged_population_dict[orientation]).detach()
    for base_region in pca_regions:
        hemi, region = base_region[0], base_region[1:]
        hemi_idx = hemi=='i'
        hemi_plot_dict= hemi_variance(full_pca_dict, region_map, base_region, pca_regions,
                                 orientation, pc_dims=pc_plot_dims, fig=var_plot_dict[region][0],
                                 indices=orient_idx+2*hemi_idx)
        orient_dict = orientation_variance(full_pca_dict, orientation, base_region,
                                                pc_dims=pc_plot_dims, fig=orient_plot_dict[region][0],
                                       indices=orient_idx + 2 * hemi_idx)
        var_plot_dict[region] = hemi_plot_dict['Variance'], hemi_plot_dict['Variance'].axes
        orient_plot_dict[region] = orient_dict['Variance'], orient_dict['Variance'].axes

for region in var_plot_dict.keys():
    fig_var, var_axes = var_plot_dict[region]
    handles, labels = var_axes[0].get_legend_handles_labels()
    fig_var.legend(handles, labels, ncols=2, loc='lower center')
    fig_var.set_size_inches(7, 10)
    fig_var.text(0.5, 0.06, 'Principal Component Index', ha='center')
    fig_var.suptitle(f'Variance Explained by First {pc_plot_dims} PCs of {region}\nAcross Hemisphere')
    sns.despine(fig=fig_var)
    print('Hemi Plot')
    fig_var.savefig(f'{pca_dir}/varExplained_{region}.png', bbox_inches='tight', dpi = 200)

    fig_orient, orient_axes = orient_plot_dict[region]
    handles, labels = orient_axes[0].get_legend_handles_labels()
    fig_orient.legend(handles, labels, ncols=2, loc='lower center')
    fig_orient.set_size_inches(7, 10)
    fig_orient.text(0.5, 0.06, 'Principal Component Index', ha='center')
    fig_orient.suptitle(f'Variance Explained by First {pc_plot_dims} PCs of {region}\nAcross Orientation')
    sns.despine(fig=fig_orient)
    print('Orientation Plot')
    fig_orient.savefig(f'{pca_dir}/varExplainedOrientation_{region}.png', bbox_inches='tight', dpi=200)

#     rows = int(len(region_map.keys())/2)
#     plot_signals = 1
#     fig, axs = plt.subplots(rows, 2, figsize=(15, rows*4))
#     event_times = {}
#     for epoch in epoch_window_map.keys():
#         event_times[epoch] = epoch_window_map[epoch]['time']
#     for i, (region, window) in enumerate(region_map.items()):
#         axs = pc_subplot(pca_vals, axs, i, window = window)
#     fig.suptitle(f'PCs by region over trial, {orientation.capitalize()}')
#     handles, labels = axs[0][0].get_legend_handles_labels()
#     fig.legend(handles, labels, ncols=2)
#     fig.savefig(f'{pca_dir}/PCANeurons_merged_{orientation}_PC{pca_region}.png', bbox_inches='tight', dpi=300)
#
# if new_pca:
#     with open(pca_filename, 'wb') as pca_file:
#         pkl.dump(pca_dict, pca_file)
