import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (usig conda or pip)
import pandas as pd                 #Loading tables
import numpy as np                  #Array operations
import matplotlib.pyplot as plt     #Generating plots
from pandas.plotting import table   #Plotting Tables


#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'

file_list = [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*.xlsx')]

file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])
    monkey_labels.append(name[2])

data_summary = {}
spikes_summary = {}
for _, date, monkey in file_names_split:
    print(date)
    if not monkey in data_summary.keys():
        data_summary[monkey] = {}
    if not monkey in spikes_summary.keys():
        spikes_summary[monkey] = {}
    if monkey == 'G':
        processed_dir = f'Data/Processed/Monkey_Green/{date[:4]}_{date[4:6]}_{date[6:]}'
    else:
        processed_dir = f'Data/Processed/Monkey_Red/{date[:4]}_{date[4:6]}_{date[6:]}'
    if not os.path.isdir(processed_dir):
        print(f'Data not processed for Monkey {monkey} on {date[:4]}_{date[4:6]}_{date[6:]}')
        continue
    else:
        area_spikes = {}
        area_neurons = {}
        for unit in [1,2]:
            spike_file_name = f'{processed_dir}/spikeCounts_U{unit}.p'
            if not os.path.isfile(spike_file_name):
                print(f'No spike count file for Monkey {monkey} on {date[:4]}_{date[4:6]}_{date[6:]}')
                continue
            with open(f'{processed_dir}/Neurons_U{unit}.p', 'rb') as neuron_file:
                neurons = pkl.load(neuron_file)
            with open(spike_file_name, 'rb') as spike_count_file:
                spike_counts = pkl.load(spike_count_file)
            for area in neurons:
                area_spikes[area] = spike_counts[area]
                area_sum = np.array([n[1] for n in neurons[area]]).sum()
                area_neurons[area] = area_sum
            print(f'Unit {unit}: {spike_counts}')
    print(f'Neurons: {area_neurons}')
    data_summary[monkey][f'{date[:4]}/{date[4:6]}/{date[6:]}'] = area_neurons
    spikes_summary[monkey][f'{date[:4]}/{date[4:6]}/{date[6:]}'] = area_spikes

for monkey in ['G', 'R']:
    fig_dims = (5, 2.5)
    area_df= pd.DataFrame(data_summary[monkey]).fillna(0).transpose()
    spikes_df = pd.DataFrame(spikes_summary[monkey]).fillna(0).transpose()
    f = plt.figure(figsize=fig_dims)
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.set_xticks([])
    ax.xaxis.set_label_position('top')
    ax.set_yticks([])

    table(ax, area_df.astype('Int64'), loc='center')

    plt.title(f'Neuron Count for Monkey {monkey}')
    plt.savefig(f'Data/dataSummary_neuronCounts_{monkey}.png', dpi = 300, bbox_inches = 'tight')

    f2 = plt.figure(figsize=fig_dims)
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.set_xticks([])
    ax.xaxis.set_label_position('top')
    ax.set_yticks([])

    table(ax, spikes_df.astype('Int64'), loc='center')

    plt.title(f'Spike Count for Monkey {monkey}')
    plt.savefig(f'Data/dataSummary_spikeCounts_{monkey}.png', dpi=300, bbox_inches='tight')
# plt.show()


