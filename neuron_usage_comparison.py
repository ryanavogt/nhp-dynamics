# The following packages are built-in for python builds
import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob

# The following packages need to be installed in your virtual environment (using conda or pip)
import pandas as pd                 #Loading tables
import scipy.io as sio              #Loading matlab files
import mat73
from pandas.plotting import table
import matplotlib.pyplot as plt
import numpy as np

monkey_name_dict = {'G': 'Green', 'R': 'Red'}
monkey_M1_hemisphere = {'Green': 'R', 'Red': 'R'}
hemisphere_map= {'i': 'L', 'c':'R'}

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'

all_ian_neurons = {}
all_new_neurons = {}
monkey_list = []
date_list = []
channel_list = []
area_list = []
ian_neuron_list = []
new_neuron_list = []
file_list = glob.glob(f'{data_dir}/*PreInact.mat')
for file in file_list:
    ian_neurons = mat73.loadmat(file)['Neurons']
    for neuron in ian_neurons:
        date_unit, region, _, sig_neuron = neuron[0].split('_')
        monkey, year, month, day, unit = date_unit[0], date_unit[1:5], date_unit[5:7], date_unit[7:9], date_unit[-1]
        sig, neuron = int(sig_neuron[3:6]), int(sig_neuron[-1])
        monkey_load_dir = f'Data/Processed/Monkey_{monkey_name_dict[monkey]}'
        load_dir = f'{monkey_load_dir}/{year}_{month}_{day}'
        if region == 'M1':
            lat, area = 'c', region
        else:
            lat, area = region[0], region[1:]
        id_key = f'{monkey}_{year}{month}{day}_{unit}_{area}{hemisphere_map[lat]}_{sig}'
        if id_key not in all_ian_neurons.keys():
            all_ian_neurons[id_key] = 1
        else:
            all_ian_neurons[id_key] += 1

file_list = [f for f in glob.glob(f'Data/Processed/**/Neurons_U*', recursive=True)]
for file in file_list:
    monkey_name, date, unit = file.split('\\')[1:]
    monkey = monkey_name.split('_')[1]
    year, month, day = date.split('_')
    unit = unit.split('.')[0][-1]
    with open(file, 'rb') as neuron_file:
        neuron_dict = pkl.load(neuron_file)
    for area in neuron_dict:
        # if 'PMv' in area:
        id_key_short = f'{monkey[0]}_{year}{month}{day}_{unit}_{area}'
        all_new_neurons[id_key] = {}
        for sig, neurons in neuron_dict[area]:
            id_key = f'{id_key_short}_{sig}'
            all_new_neurons[f'{id_key}'] = int(neurons)
            monkey_list.append(monkey[0])
            date_list.append(f'{year}{month}{day}')
            channel_list.append(f'U{unit}, Sig{sig}')
            area_list.append(area)
            new_neuron_list.append(int(neurons))
            if id_key in all_ian_neurons.keys():
                ian_neuron_list.append(all_ian_neurons[id_key])
            else:
                ian_neuron_list.append(0)

for id_key in all_ian_neurons.keys():
    if id_key not in all_new_neurons:
        monkey_name, date, unit, area, signal = id_key.split('_')
        monkey_list.append(monkey_name)
        date_list.append(date)
        channel_list.append(f'U{unit}, Sig{sig}')
        area_list.append(area)
        ian_neuron_list.append(all_ian_neurons[id_key])
        new_neuron_list.append(0)
neuron_diff = np.array(ian_neuron_list) - np.array(new_neuron_list)
neuron_df = pd.DataFrame.from_dict({'Monkey':monkey_list, 'Date':date_list, 'Region':area_list, 'Channel':channel_list,
                             'Ian Neurons': ian_neuron_list, 'New Neurons': new_neuron_list,
                             'Difference':neuron_diff}
                           )
cortex_list = []
shared_list = []
ian_only_list = []
new_only_list = []
for area in ['PMvL', 'PMvR', 'M1R', 'PMdL', 'PMdR']:
    cortex_list.append(area)
    region_filter = neuron_df['Region']==area
    # region_filter = neuron_df['New Neurons'] > -1000
    equal_counts    = (neuron_df['New Neurons'][(neuron_df['Difference']== 0) & region_filter]).sum()
    ian_more        = (neuron_df['New Neurons'][(neuron_df['Difference'] > 0) & region_filter]).sum()
    new_more        = (neuron_df['Ian Neurons'][(neuron_df['Difference'] < 0) & region_filter]).sum()
    shared_neurons = (equal_counts + ian_more + new_more)
    print([equal_counts, ian_more, new_more])
    ian_exclusive_neurons = neuron_df['Ian Neurons'][region_filter].sum() - shared_neurons
    new_exclusive_neurons = neuron_df['New Neurons'][region_filter].sum() - shared_neurons
    shared_list.append(shared_neurons)
    ian_only_list.append(ian_exclusive_neurons)
    new_only_list.append(new_exclusive_neurons)

summary_df = pd.DataFrame.from_dict({'Cortex': cortex_list, 'Shared Neurons': shared_list,
                                     'Ian-only': ian_only_list, 'New Only': new_only_list})
summary_df.style.hide(axis='index')

fig_dims = (5, 1.5)
f = plt.figure(figsize=fig_dims)
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.set_xticks([])
ax.xaxis.set_label_position('top')
ax.set_yticks([])
table(ax, summary_df, loc='center')

plt.title(f'Neuron Differences')
plt.savefig(f'Data/Processed/Summary/neuronSplits.png', dpi = 300, bbox_inches = 'tight')

neuron_df.to_csv(f'Data/Processed/Summary/neuronComparison_PMv.csv')

print(neuron_df)