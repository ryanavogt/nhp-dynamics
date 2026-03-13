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
n_pcs =3
n_bins = 30

os.environ['KMP_DUPLICATE_LIB_OK']='True'
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
merged_count = len(monkey_name_map.keys())

summary_dir = f'Data/Processed/Summary'
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
indices_filename = f'{pca_dir}_pcIndices_merged{merged_count}.p'
joint_indices_filename = f'{pca_dir}_pcIndicesJoint_merged{merged_count}.p'
monkey_dict_filename = f'{pca_dir}/allMonkeyDict.p'

cortex_list = ['M1', 'PMd', 'PMv']
cont_prop = [.2, .3, .4, .5, .6]
cort_inds = {}
all_indices = {}
joint_indices = {}
n_neurons = {}
joint_fig, joint_axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 15))

skip_old = True
for cort_idx, cortex in enumerate(cortex_list):
    if skip_old:
        break
    neuron_fig, neuron_axes = plt.subplots(2,2, figsize=(20, 10))
    neuron_fig_zoomed, neuron_axes_zoomed = plt.subplots(2, 2, figsize=(20, 10))
    pca_filename = f'{pca_dir}/PCA_merged{merged_count}_{cortex}_b{binsize}_k{kernel_width}.p'
    with open(pca_filename, 'rb') as pca_file:
        cortex_pca_vals = pkl.load(pca_file)
    V = cortex_pca_vals['V']
    full_fig = plt.figure(figsize=(10, 6))
    V_plot, V_indices = (V[:,:3]**2).sqrt().sum(dim=1).sort(descending=True, dim=0)
    plt.bar(np.arange(V_plot.shape[0]), V_plot)
    # plt.xticks(np.arange(zoomed_count), (V_indices[:zoomed_count]+1).tolist())
    full_fig.savefig(f'{pca_dir}/neuronCompFull_{cortex}_merged{merged_count}.png')
    cond_map = cortex_pca_vals['cond_map']
    n_neurons = cond_map[list(cond_map.keys())[0]][1] - cond_map[list(cond_map.keys())[0]][0]
    hist_fig, hist_axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 15), constrained_layout=True)
    joint_indices[cortex] = V_indices
    inds_list = []
    j_ax = joint_axes[cort_idx]
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
        for pc in range(n_pcs):
            hist_ax = hist_axes[c_idx][pc]
            hist_ax.hist(v_cond[:, pc], bins=n_bins, color='k', alpha=0.9)
            hist_ax.set_title(f'{condition}, PC{pc+1}')
            hist_ax.set_ylabel('Count')
            hist_ax.set_xlim([-.11, .11])
            hist_ax.set_ylim([0, 50])
        # h_ax.bar(np.arange(1, v_cond.shape[0] + 1),
                 # (v_squared[:, :3]).sum(dim=1).sort(dim=0, descending=True)[0])
        h_ax.set_title(f'{condition}')
        h_ax.set_ylabel(f'Abs. PC Val')
        h_ax.set_xlabel(f'Neuron Idx')
        v_plot, v_idcs = (v_squared[:, :3].sqrt()).sum(dim=1).sort(dim=0, descending=True)
        inds_list.append(v_idcs)
        h_ax_z.bar(np.arange(zoomed_count), v_plot[:zoomed_count])
        h_ax_z.set_xticks(ticks=np.arange(zoomed_count), labels = (v_idcs[:zoomed_count]+1).tolist())
        h_ax_z.set_title(f'{condition}')
        h_ax_z.set_ylabel(f'Summed Neuron Value')
        # h_ax_z.set_title(f'Zoomed Neuron Contributions for {cortex}')

        cond_x = np.where((V_indices<cond_map[condition][1])*(V_indices>=cond_map[condition][0]))
        j_ax.bar(x=cond_x[0], width=.85, height=V_plot[cond_x[0]], label=condition)
    j_ax.set_xlim([-2, 4*n_neurons])
    j_ax.legend()
    j_ax.set_title(f'{cortex} V weights')
    j_ax.set_ylabel(f'Summed Neuron Value')
    inds_list = np.vstack(inds_list)
    all_indices[cortex] = inds_list
    vals, counts = np.unique(inds_list[:, :zoomed_count]+1, return_counts= True)
    neuron_fig.suptitle(f'PC Contribution for First 3 PCs in {cortex}')
    neuron_fig.savefig(f'{pca_dir}/neuronComp_{cortex}.png', bbox_inches='tight', dpi=300)
    neuron_fig_zoomed.suptitle(f'Top {zoomed_count} PC Contribution for First 3 PCs in {cortex}')
    neuron_fig_zoomed.savefig(f'{pca_dir}/neuronCompZoomed_{cortex}.png', bbox_inches='tight', dpi=300)
    hist_fig.suptitle(f'Histogram of Neuron PC Values Across Conditions, {cortex}')
    hist_fig.savefig(f'{pca_dir}/neuronWeightHist_{cortex}.png', bbox_inches='tight', dpi=300)
    cort_inds[cortex] = [vals, counts]

joint_fig.suptitle(f'Abs Neuron Loading Value Sum for First 3 PCs')
joint_fig.tight_layout()
joint_fig.savefig(f'{pca_dir}/neuronJointCont.png', bbox_inches='tight', dpi=600)
with open(indices_filename, 'wb') as indices_file:
    pkl.dump(all_indices, indices_file)
with open(joint_indices_filename, 'wb') as joint_indices_file:
    pkl.dump(joint_indices, joint_indices_file)
# print(joint_indices)

monkeys = pkl.load(open(monkey_dict_filename, 'rb'))
color_map = {'R': 'tab:red', 'G': 'tab:green', 'Y': 'tab:orange', 'B': 'tab:blue'}
merged_monkey = monkeys['All']
joint_fig, joint_axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
for cort_idx, cortex in enumerate(cortex_list):
    cort_ax = joint_axes[cort_idx]
    merged_V = merged_monkey.cortices[cortex]['SVD']['Full']['V']
    V_plot, V_indices = (merged_V[:,:3]**2).sqrt().sum(dim=1).sort(descending=True, dim=0)
    n_neurons = merged_monkey.cortices[cortex]['neurons']
    title_string = f'{cortex} V weights, {n_neurons} neurons'
    for m_name, monkey in monkeys.items():
        if m_name == 'All':
            continue
        m_indices = monkey.monkey_indices[cortex]
        monkey_neurons = monkey.cortices[cortex]['neurons']
        title_string += f', {m_name}: {monkey_neurons}'
        monkey_x = np.where((V_indices<=m_indices[-1])*(V_indices>=m_indices[0]))
        cort_ax.bar(x=monkey_x[0], width=.85, height=V_plot[monkey_x[0]], label=m_name, color=color_map[m_name])
    cort_ax.set_xlim([-1, n_neurons])
    cort_ax.legend(title='Monkey')
    cort_ax.set_title(title_string)
    cort_ax.set_ylabel(f'Summed Neuron Value')
joint_fig.suptitle(f'Abs Neuron Loading Value Sum for First 3 PCs')
joint_fig.tight_layout()
joint_fig.savefig(f'{pca_dir}/neuronMonkeyCont.png', bbox_inches='tight', dpi=600)

mod_df = pkl.load(open(f'{summary_dir}/ModDataFrame.p', 'rb'))
mod_color_map = {'ipsi':'tab:purple', 'contra':'tab:red', 'both:equal':'tab:green', 'both:opp':'tab:brown',
                 'both:ipsi':'tab:blue', 'both:contra':'tab:orange', 'none':'tab:gray'}
merged_monkey = monkeys['All']
for orientation in ['horizontal', 'vertical']:
    for epoch in ['Cue', 'Grasp On']:
        joint_fig, joint_axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        for cort_idx, cortex in enumerate(cortex_list):
            cort_ax = joint_axes[cort_idx]
            merged_V = merged_monkey.cortices[cortex]['SVD']['Full']['V']
            V_plot, V_indices = (merged_V[:,:3]**2).sqrt().sum(dim=1).sort(descending=True, dim=0)
            n_neurons = merged_monkey.cortices[cortex]['neurons']
            neuron_list = np.arange(n_neurons)
            title_string = f'{cortex} V weights, {n_neurons} neurons'
            for mod_type, color in mod_color_map.items():
                # m_indices = monkey.monkey_indices[cortex]
                monkey_neurons = monkey.cortices[cortex]['neurons']
                mod_indices = mod_df[(mod_df['orient']==orientation)&(mod_df['region']==cortex)
                                     &(mod_df['mod_type']==mod_type)&(mod_df['event']==epoch)]['n_idc']
                mod_x = [False]*n_neurons
                for idx, V_index in enumerate(V_indices):
                    if V_index in list(mod_indices):
                        mod_x[idx] = True
                # title_string += f', {mod_type}: {len(mod_indices)}'
                cort_ax.bar(x=neuron_list[mod_x], width=.85, height=V_plot[mod_x], label=mod_type, color=color)
            cort_ax.set_xlim([-1, n_neurons])
            cort_ax.legend(title='Monkey')
            cort_ax.set_title(title_string)
            cort_ax.set_ylabel(f'Summed Neuron Value')
        joint_fig.suptitle(f'Abs Neuron Loading Value Sum for First 3 PCs by Mod Type in {epoch}')
        joint_fig.tight_layout()
        joint_fig.savefig(f'{pca_dir}/neuronModTypeCont{epoch}_{orientation}.png', bbox_inches='tight', dpi=600)