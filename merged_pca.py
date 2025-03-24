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

def region_variance(pc_dict, pop_sdf, region_map, baseline_region, target_regions, orientation='vertical', pc_dims=10):
    """

    :param baseline_region: Region onto which data/variance is projected
    :param target_regions: Region(s) from which data is projected, variance is measured
    :param orientation: Orientation of hand (hyperparameter)
    :param pc_dims: Number of principal components to display
    :return:
    """
    base_pca = pc_dict[orientation][baseline_region]
    base_cov = base_pca['cov']
    base_pcs = base_pca['V']
    base_S = base_pca['S']
    base_window = region_map[baseline_region]
    base_pc_proj = pop_sdf[base_window[0]:base_window[1]].T@base_pcs
    # base_var_sum = torch.cumsum(torch.diagonal(base_pc_proj.T@base_cov@base_pc_proj), dim=0)
    base_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@base_cov@base_pcs), dim=0)
    base_var_trace = base_var_sum.max()
    f_cov = plt.figure(figsize=(8,8))
    plt.imshow(base_cov, norm=mpl.colors.SymLogNorm(linthresh=.001, linscale=.01), cmap='RdBu_r')
    plt.title(f'Covariance Matrix for {baseline_region}, {orientation}')
    plt.colorbar()
    f_var = plt.figure(figsize=(3, 5))
    plt.scatter(np.arange(1, pc_dims+1), base_var_sum[:pc_dims]/base_var_trace, label = baseline_region)
    plt.title(f'Variance Captured by First {pc_dims} {baseline_region} PCs')
    # f_S = plt.figure()
    # plt.scatter(np.arange(1, pc_dims+1), torch.cumsum(base_S, dim=0)[:pc_dims]/base_S.sum(), label = baseline_region)
    # plt.title(f'Variance Captured by First {pc_dims} {baseline_region} PCS, S Matrix')

    # f_pcs, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': '3d'})
    # plot_pca = base_pc_proj[:, :3]
    # Z = plot_pca[:, 2]
    # ax.scatter(xs = plot_pca[:, 0], ys = plot_pca[:, 1], zs = Z, label = baseline_region)
    # ax.set_title(f'Projections onto {baseline_region} PCs')
    for target_region in target_regions:
        if target_region == baseline_region:
            continue
        r_idx = region_map[target_region]
        target_sdf = pop_sdf[r_idx[0]:r_idx[1]]
        target_pca = pc_dict[orientation][target_region]
        target_pcs = target_pca['V']
        target_cov = target_pca['cov']
        target_S = target_pca['S']
        # cos_sim = torch.nn.functional.cosine_similarity(base_pcs, target_pcs)
        # counts, bins = np.histogram(cos_sim, range = (-1,1), bins = 50)
        # align_fig = plt.figure(figsize=(4, 1.5))
        # plt.hist(bins[:-1], bins, weights = counts)
        # plt.title(f'Cosine Similarity of PCs of {baseline_region} and {target_region}, {orientation}')
        # align_fig.savefig(f'{pca_dir}/CosSim_{orientation}_{base_region}_{target_region}.png', bbox_inches='tight', dpi=200)
        # plt.figure(f_S)
        # plt.scatter(np.arange(1, pc_dims+1), torch.cumsum(target_S, dim=0)[:pc_dims] / target_S.sum(), label=target_region)
        target_var_sum = torch.cumsum(torch.diagonal(base_pcs.T@target_cov@base_pcs), dim=0)
        # print(target_var_sum[:pc_dims])
        plt.figure(f_var)
        plt.scatter(np.arange(1, pc_dims+1), target_var_sum[:pc_dims]/base_var_trace, label=target_region)
        # plt.scatter(np.arange(1, pc_dims+1), (target_var_sum / base_var_sum)[:pc_dims], label=f'{target_region}/{base_region}')
        # target_window = region_map[target_region]
    #     plot_pca = pop_sdf[target_window[0]:target_window[1]].T@base_pcs[:, :3]
    #     Z = plot_pca[:, 2]
    #     ax.scatter(xs=plot_pca[:, 0], ys=plot_pca[:, 1], zs=Z, label=target_region)
    # ax.legend()
    f_var.legend()
    # f_S.legend()
    return f_var, f_cov#, f_S#, f_pcs

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


pca_overwrite = True
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sides = ['i', 'c']
n_plot = 4
q = 20 # Estimate for number of PCs to use
pca_regions = ['cPMv', 'iPMv']#, ['cM1', 'iM1'], ['cPMd', 'iPMd']
# projection_regions = ['PMd', 'PMv']
full_pca_dict = {}
for pca_region in pca_regions:
    pca_filename = f'{pca_dir}/MergedPCA_{pca_region}_b{binsize}_k{kernel_width}.p'
    if os.path.exists(pca_filename) and not pca_overwrite:
        with open(pca_filename, 'rb') as pca_file:
            pca_dict = pkl.load(pca_file)
        new_pca = False
    else:
        pca_dict = {}
        new_pca = True

    for orientation in merged_population_dict.keys():
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


# base_region = 'cM1'
for orientation in merged_population_dict.keys():
    pop_sdf = torch.Tensor(merged_population_dict[orientation]).detach()
    for base_region in pca_regions:
        fig_var, fig_cov= region_variance(full_pca_dict, pop_sdf, region_map, base_region, pca_regions, orientation, pc_dims=20)
        fig_var.savefig(f'{pca_dir}/varExplained_{orientation}_{base_region}.png', bbox_inches='tight', dpi = 200)
        fig_cov.savefig(f'{pca_dir}/covMat_{orientation}_{base_region}.png', bbox_inches='tight', dpi=200)
        # fig_S.savefig(f'{pca_dir}/sVariance_{orientation}_{base_region}.png', bbox_inches='tight', dpi=200)
        # fig_pcs.savefig(f'{pca_dir}/pcProjections_{orientation}_{base_region}.png', bbox_inches='tight', dpi=200)



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
#     # Plot the variance from each eigenvalue
#     plt.figure(figsize=(4,3))
#     scale = S.sum()
#     plt.scatter(np.arange(1, S.shape[0] + 1), torch.cumsum(S / scale, dim=0), label=epoch)
#     plt.xlabel('PC index')
#     plt.ylabel('Variance Proportion')
#     plt.suptitle(f'Variance Explained by PCs, {orientation.capitalize()}, PCA Region: {pca_region}')
#     plt.savefig(f'{pca_dir}/PCAVariance_merged_{orientation}_PC{pca_region}.png', bbox_inches='tight', dpi=300)
#
#
#
# if new_pca:
#     with open(pca_filename, 'wb') as pca_file:
#         pkl.dump(pca_dict, pca_file)
