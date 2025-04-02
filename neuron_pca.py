import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
from mpl_toolkits.mplot3d.axes3d import get_test_data

from sig_proc import *
import pandas as pd
import matplotlib as mpl
import torch

from sig_proc import *

import seaborn as sns

sns.set()
sns.set_style(style='white')

monkey_name_map = {'R': 'Red', 'G': 'Green'}
event_map = {'trialRewardDrop': 'Cue', 'trialReachOn':'Reach', 'trialGraspOn':'GraspOn', 'trialEnd':'GraspOff'}
# Define the reference events and time window defining each epoch
epoch_window_map = {'Pre-cue':  {'event': 'trialRewardDrop','window': [-700,    -100]},
                   'Post-cue': {'event': 'trialRewardDrop', 'window': [0,       100]},
                   'Reach':    {'event': 'trialReachOn',    'window': [-100,    200]},
                   'Grasp On': {'event': 'trialGraspOn',    'window': [-100,    500]},
                   'Grasp Off':{'event': 'trialGraspOff',   'window': [-200,    -100]}}

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

pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
if not os.path.exists(pca_dir):
    os.mkdir(pca_dir)

all_sdf_filename = f'{summary_dir}/sdfDict_bin{binsize}_k{kernel_width}.p'
with open(all_sdf_filename, 'rb') as sdf_file:
    all_sdf_dict = pkl.load(sdf_file)

pop_filename = f'{pca_dir}/pop_dict_merged_b{binsize}_k{kernel_width}.p'
if os.path.exists(pop_filename):
    with open(pop_filename, 'rb') as pop_file:
        pop_tuple = pkl.load(pop_file)
        population_dict, region_map = pop_tuple
        print('Population Dictionary Loaded')
else:
    population_dict = {}
    region_map = {'idx':0}
    for area in all_sdf_dict.keys():
        region, orientation = area.split('_')
        side, cortex = region[0], region[1:]
        if orientation not in population_dict.keys():
            population_dict[orientation] = {}
        for epoch in epoch_window_map.keys():
            event_name = epoch_window_map[epoch]['event']
            event_window = epoch_window_map[epoch]['window']
            event_mask = (full_window > event_window[0]) * (full_window <= event_window[1])
            sdf = all_sdf_dict[area][event_name][:, event_mask]
            if epoch not in population_dict[orientation]:
                population_dict[orientation][epoch] = []
            population_dict[orientation][epoch].append(sdf)
            if region not in region_map.keys():
                region_boundaries  = [region_map['idx'], region_map['idx'] + sdf.shape[0]]
                region_map[region] = region_boundaries
                region_map['idx'] += sdf.shape[0]

    with open(pop_filename, 'wb') as pop_file:
        pkl.dump((population_dict, region_map), pop_file)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sides = ['i', 'c']
n_plot = 4
q = 10 # Estimate for number of PCs to use
pca_filename = f'{pca_dir}/TotalPCA_dict_b{binsize}_k{kernel_width}.p'
if os.path.exists(pca_filename):
    with open(pca_filename, 'rb') as pca_file:
        pca_dict = pkl.load(pca_file)
    new_pca = False
else:
    pca_dict = {}
    new_pca = True

for orientation in population_dict.keys():
    if new_pca:
        pca_dict[orientation] = {}
    rows = 2
    columns = math.ceil((len(epoch_window_map.keys())-1)/rows)
    fig, axs = plt.subplots(rows, columns, figsize=(8, 8), subplot_kw={'projection': '3d'})
    fig2, axs2 = plt.subplots(rows, columns, figsize=(8, 8))
    fig3, axs3 = plt.subplots(1, 1, figsize=(4, 4))
    for idx, epoch in enumerate(population_dict[orientation].keys()):
        # Compute PCs
        if epoch == 'Grasp Off':
            continue
        col = idx//rows
        row = idx%rows
        ax = axs[row][col]
        ax2= axs2[row][col]
        pop_sdf = torch.Tensor(np.vstack(population_dict[orientation][epoch])).detach()
        if new_pca:
            U, S, V = torch.pca_lowrank(pop_sdf, center=True, q=q)
            population_dict[orientation][epoch] = pop_sdf
            pca_dict[orientation][epoch] = {'U': U, 'S': S, 'V': V}
        else:
            pca_vals = pca_dict[orientation][epoch]
            U, S, V = pca_vals['U'], pca_vals['S'], pca_vals['V']

        # Plot PCs in 2D and color points by region
        plt.figure(fig)
        for region, boundary in region_map.items():
            if region == 'idx':
                continue
            plot_pca = torch.matmul(pop_sdf[boundary[0]:boundary[1]], V[:, :3])
            # X, Y = np.meshgrid(plot_pca[:, 0], plot_pca[:, 1])
            Z = plot_pca[:, 2]
            ax.scatter(xs=plot_pca[:, 0], ys=plot_pca[:, 1], zs=Z, label = region, alpha = 0.8)
        ax.title.set_text(epoch)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        # Plot first 2 PCs as neurons
        plt.figure(fig2)
        for i in range(n_plot):
            event_window = epoch_window_map[epoch]['window']
            plot_x = np.arange(event_window[0], event_window[1], binsize)
            ax2.plot(plot_x, V[:, i], label = f'PC{i+1}')
        ax2.title.set_text(epoch)

        if row == rows-1:
            ax2.set_xlabel('Time from Event (ms)')
        if col == 0:
            ax2.set_ylabel('PC Value')

        # Plot the variance from each eigenvalue
        plt.figure(fig3)
        scale = S.sum()
        plt.scatter(np.arange(1, S.shape[0]+1), S/scale, label=epoch)
        plt.xlabel('PC index')
        plt.ylabel('Variance Proportion')

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.5, 0), ncols=3)
    fig.suptitle(f'First Three PC Dimensions for Total Population, {orientation.capitalize()}')
    fig.savefig(f'{pca_dir}/PopPCA_{orientation}.png', bbox_inches='tight', dpi = 300)

    handles2, labels2 = axs2[0][0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc=(0.5, 0), ncols=3)
    fig2.suptitle(f'First {n_plot} PC Dimensions Over Time, {orientation.capitalize()}')
    fig2.savefig(f'{pca_dir}/PCANeurons_{orientation}.png', bbox_inches='tight', dpi=300)

    handles3, labels3 = axs3.get_legend_handles_labels()
    fig3.legend(handles3, labels3, loc=(0.28, 0.75), ncols=2)
    fig3.suptitle(f'Variance Explained by PCs, {orientation.capitalize()}')
    fig3.savefig(f'{pca_dir}/PCAVariance_{orientation}.png', bbox_inches='tight', dpi=300)

if new_pca:
    with open(pca_filename, 'wb') as pca_file:
        pkl.dump(pca_dict, pca_file)