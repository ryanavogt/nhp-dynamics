import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
import torch
import scipy.stats as stats
import pandas as pd

from utils.sig_proc import *

import seaborn as sns

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

binsize = 5
kernel_width = 25

zoomed_count = 20
n_pcs =3
n_bins = 30

os.environ['KMP_DUPLICATE_LIB_OK']='True'
monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
merged_count = len(monkey_name_map.keys())

# summary_dir = f'Data/Processed/Summary'
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'
indices_filename = f'{pca_dir}_pcIndices_merged{merged_count}.p'
joint_indices_filename = f'{pca_dir}/pcIndicesJoint_merged{merged_count}.p'
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
        if skip_old:
            continue
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
    plt.close()
    # inds_list = np.vstack(inds_list)
    # all_indices[cortex] = inds_list
    # vals, counts = np.unique(inds_list[:, :zoomed_count]+1, return_counts= True)
    # neuron_fig.suptitle(f'PC Contribution for First 3 PCs in {cortex}')
    # neuron_fig.savefig(f'{pca_dir}/neuronComp_{cortex}.png', bbox_inches='tight', dpi=300)
    # neuron_fig_zoomed.suptitle(f'Top {zoomed_count} PC Contribution for First 3 PCs in {cortex}')
    # neuron_fig_zoomed.savefig(f'{pca_dir}/neuronCompZoomed_{cortex}.png', bbox_inches='tight', dpi=300)
    # hist_fig.suptitle(f'Histogram of Neuron PC Values Across Conditions, {cortex}')
    # hist_fig.savefig(f'{pca_dir}/neuronWeightHist_{cortex}.png', bbox_inches='tight', dpi=300)
    # cort_inds[cortex] = [vals, counts]

# joint_fig.suptitle(f'Abs Neuron Loading Value Sum for First 3 PCs')
# joint_fig.tight_layout()
# joint_fig.savefig(f'{pca_dir}/neuronJointCont.png', bbox_inches='tight', dpi=600)
# with open(indices_filename, 'wb') as indices_file:
#     pkl.dump(all_indices, indices_file)
with open(joint_indices_filename, 'wb') as joint_indices_file:
    pkl.dump(joint_indices, joint_indices_file)
# print(joint_indices)

monkeys = pkl.load(open(monkey_dict_filename, 'rb'))
color_map = {'R': 'tab:red', 'G': 'tab:green', 'Y': 'tab:orange', 'B': 'tab:blue'}
merged_monkey = monkeys['All']
joint_fig, joint_axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), gridspec_kw={'height_ratios':[1.5,1]})
for cort_idx, cortex in enumerate(cortex_list):
    cort_ax = joint_axes[0][cort_idx]
    cdf_ax = joint_axes[1][cort_idx]
    merged_V = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
    V_plot, V_indices = (merged_V[:,:3]**2).sqrt().sum(dim=1).sort(descending=True, dim=0)
    n_neurons = merged_monkey.cortices[cortex]['neurons']
    neuron_list = np.arange(n_neurons)
    title_string = f'{cortex} V weights, {n_neurons} neurons'
    cont_total = V_plot.sum()
    for m_name, monkey in monkeys.items():
        if m_name == 'All':
            continue
        m_indices = monkey.monkey_indices[cortex]
        monkey_neurons = monkey.cortices[cortex]['neurons']
        # title_string += f', {m_name}: {monkey_neurons}'
        monkey_x = np.where((V_indices<=m_indices[-1])*(V_indices>=m_indices[0]))
        mod_ecdf = stats.ecdf(neuron_list[monkey_x])
        ks_stat = stats.ks_1samp(neuron_list[monkey_x] / n_neurons, stats.uniform.cdf)
        p_val = ks_stat[1]
        mod_ecdf.cdf.plot(cdf_ax, label=f'{m_name}, p:{p_val:.2f}', c=color_map[m_name])
        monkey_V = V_plot[monkey_x].sum()
        cort_ax.bar(x=monkey_x[0], width=.85, height=V_plot[monkey_x[0]],
                    label=f'{m_name}, n:{monkey_neurons}, T:{monkey_V/cont_total:.2f}, Avg:{(monkey_V/cont_total)/monkey_x[0].shape[0]*100:.2f}',
                    color=color_map[m_name])
    cdf_ax.plot([0, n_neurons], [0, 1], 'k--')
    cdf_ax.set_ylabel('Cumulative Fraction')
    cdf_ax.set_xlabel('Neuron Index')
    cdf_ax.set_xlim([0, n_neurons])
    cdf_ax.set_ylim([0, 1])
    cdf_ax.legend(prop={'size':6})
    cort_ax.set_xlim([-1, n_neurons])
    cort_ax.legend(title='Monkey')
    cort_ax.set_title(title_string)
    cort_ax.set_ylabel(f'Summed Neuron Value')
joint_fig.suptitle(f'Abs Neuron Loading Value Sum for First 3 PCs')
joint_fig.tight_layout()
joint_fig.savefig(f'{pca_dir}/neuronMonkeyCont.png', bbox_inches='tight', dpi=600)

mod_df = pkl.load(open(os.path.join(summary_dir, 'earlyLate', 'MergedModDataFrame.p'), 'rb'))
mod_color_map = {'ipsi':'tab:green', 'both:ipsi':'tab:blue', 'both:equal':'tab:brown', 'both:opp':'tab:olive',
                 'both:contra':'tab:orange', 'contra':'tab:red', 'none':'tab:gray', 'both:all':'tab:brown'}
merged_monkey = monkeys['All']

# ── Orientation-merged modulation plots ───────────────────────────────────────
# The outer orientation loop is removed. For each mod_type, we take the UNION
# of neuron indices modulated in either orientation (.unique() prevents
# double-counting neurons present in both horizontal and vertical conditions).
for epoch in ['Cue', 'Grasp']:
    joint_fig, joint_axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 4), gridspec_kw={'height_ratios':[1.5,1]})
    # cum_fig, cum_axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    for cort_idx, cortex in enumerate(cortex_list):
        cort_ax = joint_axes[0][cort_idx]
        cdf_ax = joint_axes[1][cort_idx]
        # cum_ax = cum_axes[cort_idx]
        merged_S = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['S']
        merged_V = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
        eigvals = merged_S[:3]**2
        eig_total = (merged_S**2).sum()
        V_plot, V_indices = ((eigvals*merged_V[:,:3]**2).sum(dim=1)/eig_total).sqrt().sort(descending=True, dim=0)
        n_neurons = merged_monkey.cortices[cortex]['neurons']
        neuron_list = np.arange(n_neurons)
        V_cum = V_plot.cumsum(dim=0)
        # cum_ax.bar(x=neuron_list, width=0.85, height=V_cum/V_cum[-1])
        # cum_ax.set_xlim([0, 1])
        # cum_ax.set_title(f'{cortex}')
        # cum_ax.set_ylabel(f'3-PC Contribution Cumulative Sum')
        # cum_ax.set_xlabel(f'Index')
        # title_string = f'{cortex}, {n_neurons} neurons'
        for mod_type, color in mod_color_map.items():
            # Union of neurons modulated in either orientation
            mod_indices = mod_df[
                (mod_df['region'] == cortex) &
                (mod_df['mod_type'] == mod_type) &
                (mod_df['event'] == epoch)
            ]['n_idc'].unique()
            mod_x = np.array([False] * n_neurons)
            for idx, V_index in enumerate(V_indices):
                if V_index in mod_indices:
                    mod_x[idx] = True
            n_mod = mod_x.sum()
            if n_mod == 0:
                continue  # skip empty mod types (ecdf undefined on empty set)
            mod_ecdf = stats.ecdf(neuron_list[mod_x])
            ks_stat = stats.ks_1samp(neuron_list[mod_x] / n_neurons, stats.uniform.cdf)
            p_val = ks_stat[1]
            sig_label = r'$\ast\!\ast$' if p_val < 0.01 else r'$\,\ast\ $' if p_val < 0.05 else ' '
            mod_ecdf.cdf.plot(cdf_ax, label=sig_label, c=color)
            # mod_ecdf.cdf.plot(cdf_ax, label=f'p:{p_val:.2f}', c=color)
            cort_ax.bar(x=neuron_list[mod_x], width=1.0, linewidth=0, height=V_plot[mod_x],
                        label=f'{mod_type}, n:{n_mod}', color=color)
        cdf_ax.plot([0, n_neurons], [0, 1], 'k--')
        cdf_ax.set_ylabel('Cumulative Fraction')
        cdf_ax.set_xlabel('Neuron Index')
        cdf_ax.set_xlim([0, n_neurons])
        cdf_ax.set_ylim([0, 1])
        leg = cdf_ax.legend(prop={'size': 6}, handlelength=0.6, handleheight=0.8, borderpad=0.3, labelspacing=0.1, loc='lower right')
        for h in leg.legend_handles:
            h.set_linewidth(4.5)
        cort_ax.set_xlim([-1, n_neurons])
        cort_ax.legend(title='Mod Type', prop={'size':7}, handlelength=0.8, handleheight=0.8, borderpad=0.3, labelspacing=0.2, loc='upper right')
        # cort_ax.set_title(title_string)
        cort_ax.set_ylabel(f'Contribution')
    # joint_fig.suptitle(f'Cumulative Neuron Contribution for First 3 PCs by Mod Type in {epoch} (Orientations Merged)')
    joint_fig.tight_layout()
    joint_fig.savefig(os.path.join(pca_dir, f'neuronModTypeCont{epoch}_orientMerged.png'), bbox_inches='tight', dpi=300)
    plt.close(joint_fig)
    # cum_fig.suptitle(f'Cumulative PC Sum for First 3 PCs in {epoch} (Orientations Merged)')
    # cum_fig.tight_layout()
    # cum_fig.savefig(os.path.join(pca_dir, f'UPDATED_neuronModTypeCumulative{epoch}_orientMerged.png'), bbox_inches='tight', dpi=300)
    # plt.close(cum_fig)
# ─────────────────────────────────────────────────────────────────────────────

# corr_fig = plt.figure(layout='constrained', figsize=(12,8))
# subfigs = corr_fig.subfigures(1,3, wspace=0.05)
# slope_fig, slope_axs = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
# for cort_idx, cortex in enumerate(cortex_list):
#     score_filename = f'{summary_dir}/neuronSelection_scores_merged{len(monkey_name_map.keys())}_bin{binsize}_k{kernel_width}_{cortex}.p'
#     with open(score_filename, 'rb') as score_file:
#         score_dict = pkl.load(score_file)
#     merged_V = merged_monkey.cortices[cortex]['SVD']['Merged']['Full']['V']
#     V_sum = (merged_V[:, :3] ** 2).sqrt().sum(dim=1)
#     V_plot, V_indices = V_sum.sort(descending=True, dim=0)
#     hemisphere_score=score_dict['hemi_score']
#     orientation_score=score_dict['orient_score']
#     event_score=score_dict['event_score']
#     nzero_mask=score_dict['nzero_mask']
#     event_nzero_mask=score_dict['event_nzero_mask']
#     subfigs[cort_idx].suptitle(f'{cortex}')
#     cort_axs = subfigs[cort_idx].subplots(2,1)
#     cort_axs[0].set_xlabel(f'PC Contribution Weight')
#     cort_axs[1].set_xlabel(f'PC Contribution Index')
#     cortexSlope_df = mod_df[(mod_df['region']==cortex)]
#     slope_ax = slope_axs[cort_idx]
#     for direction, minmax, color in zip(['up','down'], ['max','min'], ['tab:blue', 'tab:red']):
#         idx = cortexSlope_df[cortexSlope_df['up-down']==minmax]
#         slope_idx = idx['t_idc']
#         weight_val = V_sum[list(idx['n_idc'])]
#         slope_reg = stats.linregress(slope_idx, weight_val)
#         slope_ax.scatter(slope_idx, weight_val, color=color, alpha = 0.4)
#         slope_ax.plot(slope_idx, slope_reg.intercept + slope_reg.slope * slope_idx, color=color,
#                      label=f'{direction.capitalize()}:{slope_reg.rvalue ** 2:.3f}')
#
#     slope_ax.legend(title='Mod Direction')
#     for cort_ax, x in zip(cort_axs, [V_sum, V_indices]):
#         cort_ax.set_ylabel(f'Selection Score')
#         cort_ax.set_title(f'Selection vs. PC Correlation')
#         hemi_reg = stats.linregress(x[nzero_mask], hemisphere_score)
#         orien_reg = stats.linregress(x[nzero_mask], orientation_score)
#         event_reg = stats.linregress(x[event_nzero_mask], event_score)
#         cort_ax.scatter(x[nzero_mask], hemisphere_score, alpha=0.5, color='tab:blue')
#         cort_ax.scatter(x[nzero_mask], orientation_score, alpha=0.5, color='tab:red')
#         cort_ax.scatter(x[event_nzero_mask], event_score, alpha=0.5, color='tab:green')
#         cort_ax.plot(x, hemi_reg.intercept+hemi_reg.slope*x, color='tab:blue',
#                      label = f'Hemi (C-I) R-sq:{hemi_reg.rvalue**2:.3f}')
#         cort_ax.plot(x, orien_reg.intercept + orien_reg.slope * x, color='tab:red',
#                      label=f'Orient (V-H) R-sq:{orien_reg.rvalue**2:.3f}')
#         cort_ax.plot(x, event_reg.intercept + event_reg.slope * x, color='tab:green',
#                      label=f'Event (G-C) R-sq:{event_reg.rvalue**2:.3f}')
#         cort_ax.legend()
# corr_fig.savefig(f'{pca_dir}/SelectionPC.png', bbox_inches='tight', dpi=600)
# slope_fig.savefig(f'{pca_dir}/SlopeTimingPC.png', bbox_inches='tight', dpi=600)
