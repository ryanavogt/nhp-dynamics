import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
from mpl_toolkits.mplot3d.axes3d import get_test_data

from sig_proc import *

import seaborn as sns

sns.set_theme()
sns.set_style(style='white')

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
        merged_population_dict[orientation].append(sdf[:,0].T)
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
q = 10 # Estimate for number of PCs to use
pca_filename = f'{pca_dir}/MergedPCA_dict_b{binsize}_k{kernel_width}.p'
if os.path.exists(pca_filename) and not pca_overwrite:
    with open(pca_filename, 'rb') as pca_file:
        pca_dict = pkl.load(pca_file)
    new_pca = False
else:
    pca_dict = {}
    new_pca = True

for orientation in merged_population_dict.keys():
    if new_pca:
        pca_dict[orientation] = {}
    pop_sdf = torch.Tensor(merged_population_dict[orientation]).detach()
    if new_pca:
        print('Computing PCA')
        U, S, V = torch.pca_lowrank(pop_sdf, center=False, q=q)
        merged_population_dict[orientation]= pop_sdf
        pca_dict[orientation] = {'U': U, 'S': S, 'V': V}
    else:
        pca_vals = pca_dict[orientation]
        U, S, V = pca_vals['U'], pca_vals['S'], pca_vals['V']

    rows = int(len(region_map.keys())/2)
    plot_signals = 1
    fig, axs = plt.subplots(rows, 2, figsize=(15, rows*4))
    event_times = {}
    for epoch in epoch_window_map.keys():
        event_times[epoch] = epoch_window_map[epoch]['time']
    for i, (region, window) in enumerate(region_map.items()):
        y_max = -1000
        y_min = 1000
        row = i//2
        column = i%2
        for j in range(1, n_plot+1):
            plot_x = np.arange(0, current_time, binsize)
            plot_y = (U[window[0]:window[0]+plot_signals, j-1:j]@np.diag(S[j-1:j])@V[:,j-1:j].T).T
            axs[row][column].plot(plot_x, plot_y, label = f'PC{j}')
            y_max = max(plot_y.max(), y_max)
            y_min = min(plot_y.min(), y_min)
        axs[row][column].vlines(event_times.values(), ymin = y_min, ymax = y_max,
                      colors='k', linestyles = 'dashed')
        axs[row][column].set_xticks(list(event_times.values()), labels = event_times.keys())
        axs[row][column].set_title(f'{region}')
    fig.suptitle(f'PCs by region over trial, {orientation.capitalize()}')
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2)
    fig.savefig(f'{pca_dir}/PCANeurons_merged_{orientation}.png', bbox_inches='tight', dpi=300)

    # Plot the variance from each eigenvalue
    plt.figure(figsize=(4,3))
    scale = S.sum()
    plt.scatter(np.arange(1, S.shape[0] + 1), S / scale, label=epoch)
    plt.xlabel('PC index')
    plt.ylabel('Variance Proportion')
    plt.suptitle(f'Variance Explained by PCs, {orientation.capitalize()}')
    plt.savefig(f'{pca_dir}/PCAVariance_merged_{orientation}.png', bbox_inches='tight', dpi=300)

if new_pca:
    with open(pca_filename, 'wb') as pca_file:
        pkl.dump(pca_dict, pca_file)