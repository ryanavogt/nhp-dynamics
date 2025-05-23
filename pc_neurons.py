import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
from matplotlib.lines import Line2D
import torch

from sig_proc import *

import seaborn as sns

binsize = 5
kernel_width = 25

zoomed_count = 20

summary_dir = f'Data/Processed/Summary'
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'


cortex_list = ['M1', 'PMd', 'PMv']
for cortex in cortex_list:
    neuron_fig, neuron_axes = plt.subplots(2,2, figsize=(20, 10))
    neuron_fig_zoomed, neuron_axes_zoomed = plt.subplots(2, 2, figsize=(20, 10))
    pca_filename = f'{pca_dir}/PCA_{cortex}_b{binsize}_k{kernel_width}.p'
    with open(pca_filename, 'rb') as pca_file:
        cortex_pca_vals = pkl.load(pca_file)
    V = cortex_pca_vals['V']
    full_fig = plt.figure(figsize=(10, 6))
    V_plot, V_indices = (V[:,:3]**2).sqrt().sum(dim=1).sort(descending=True, dim=0)
    plt.bar(np.arange(zoomed_count), V_plot[:zoomed_count])
    plt.xticks(np.arange(zoomed_count), (V_indices[:zoomed_count]+1).tolist())
    full_fig.savefig(f'{pca_dir}/neuronCompFull_{cortex}')
    cond_map = cortex_pca_vals['cond_map']
    for c_idx, condition in enumerate(cond_map):
        h_ax = neuron_axes[c_idx%2][c_idx//2]
        h_ax_z = neuron_axes_zoomed[c_idx%2][c_idx//2]
        v_cond = V[cond_map[condition][0]:cond_map[condition][1]]
        v_norm = v_cond.norm(dim=0)
        v_squared = v_cond**2
        v_sorted, v_sortidcs = v_squared.sort(descending=True, dim=0)
        v_squaredsum = (v_sorted).cumsum(dim=0)
        v_contribution = v_squaredsum.sqrt()/v_norm
        h_ax.bar(np.arange(1, v_cond.shape[0]+1), (v_squared[:, :3].sqrt()).sum(dim=1).sort(dim=0, descending=True)[0])
        # h_ax.bar(np.arange(1, v_cond.shape[0] + 1),
                 # (v_squared[:, :3]).sum(dim=1).sort(dim=0, descending=True)[0])
        h_ax.set_title(f'{condition}')
        h_ax.set_ylabel(f'Abs. PC Val')
        h_ax.set_xlabel(f'Neuron Idx')
        v_plot, v_idcs = (v_squared[:, :3].sqrt()).sum(dim=1).sort(dim=0, descending=True)
        h_ax_z.bar(np.arange(zoomed_count), v_plot[:zoomed_count])
        h_ax_z.set_xticks(ticks=np.arange(zoomed_count), labels = (v_idcs[:zoomed_count]+1).tolist())
        h_ax_z.set_title(f'{condition}')
        h_ax_z.set_ylabel(f'Summed Neuron Value')
    neuron_fig.suptitle(f'PC Contribution for First 3 PCs in {cortex}')
    neuron_fig.savefig(f'{pca_dir}/neuronComp_{cortex}.png', bbox_inches='tight', dpi=300)
    neuron_fig_zoomed.suptitle(f'Top {zoomed_count} PC Contribution for First 3 PCs in {cortex}')
    neuron_fig_zoomed.savefig(f'{pca_dir}/neuronCompZoomed_{cortex}.png', bbox_inches='tight', dpi=300)

