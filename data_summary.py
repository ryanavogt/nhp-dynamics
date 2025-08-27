import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)
import glob
import math

# The following packages need to be installed in your virtual environment (usig conda or pip)
import pandas as pd                 #Loading tables
import numpy as np                  #Array operations
import matplotlib.pyplot as plt     #Generating plots
from pandas.plotting import table   #Plotting Tables

import seaborn as sns
sns.set()
sns.set_style(style='white')

#Define directories of data
data_dir = 'Data/Sorted_Inactivation'
matlab_dir = f'{data_dir}/matlabFiles'
sorting_dir = f'{data_dir}/sortingNotes'
summary_dir = f'Data/Processed/Summary'

file_list = []
# m_list = ['G', 'R']
m_list = ['G', 'R', 'Y', 'B']
for m_label in m_list:
    file_list += [f for f in glob.glob(f'{sorting_dir}/SortingNotes_*{m_label}*.xlsx')]

file_names = [f.split('\\')[-1].split('.xlsx')[0] for f in file_list]

file_names_split = [n.split('_') for n in file_names]
date_strings = []
monkey_labels = []
for name in file_names_split:
    date_strings.append(name[1])
    monkey_labels.append(name[2])

data_summary = {}
spikes_summary = {}
eventTimes_summary = {}

monkey_colors_full = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
monkey_colors = {}
for monkey in m_list:
    monkey_colors[monkey] = monkey_colors_full[monkey]
for _, date, monkey in file_names_split:
    print(date)
    if not monkey in data_summary.keys():
        data_summary[monkey] = {}
    if not monkey in spikes_summary.keys():
        spikes_summary[monkey] = {}
    if not monkey in eventTimes_summary.keys():
        eventTimes_summary[monkey] = {}
    processed_dir = f'Data/Processed/Monkey_{monkey_colors[monkey]}/{date[:4]}_{date[4:6]}_{date[6:]}'
    if not os.path.isdir(processed_dir):
        print(f'Data not processed for Monkey {monkey} on {date[:4]}_{date[4:6]}_{date[6:]}')
        continue
    else:
        event_file_name = f'{processed_dir}/eventTimes.p'
        with open(event_file_name, 'rb') as event_file:
            rel_event_times = pkl.load(event_file)
        for event in rel_event_times.keys():
            print(event)
            if event not in eventTimes_summary[monkey]:
                eventTimes_summary[monkey][event] = []
            eventTimes_summary[monkey][event].append(rel_event_times[event])
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

event_lengths = {}
nrows = len(m_list)//2
fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(16,5*nrows), sharey='row', sharex='row')
all_text_vals = []
max_length, max_counts = 0, 0
bin_width = 20
u_lim = 1000
for idx, monkey in enumerate(eventTimes_summary.keys()):
    if nrows > 1:
        ax = axs[idx//2][idx%2]
    else:
        ax = axs[idx]
    event_lengths[monkey] = {}
    temp_time = 0
    text_vals=[]
    all_lengths = []
    for event in eventTimes_summary[monkey]:
        eventTimes_summary[monkey][event]=np.hstack(eventTimes_summary[monkey][event])
        e_lengths = eventTimes_summary[monkey][event] - temp_time
        event_lengths[monkey][event] = e_lengths
        temp_time = eventTimes_summary[monkey][event]
        all_lengths.append(e_lengths)
    all_lengths = np.hstack(e_lengths)
    e_max = np.max(all_lengths)
    bin_number = math.ceil(u_lim/bin_width)
    for event in eventTimes_summary[monkey]:
        event_label = event.split('trial')[-1]
        e_lengths = event_lengths[monkey][event]
        counts, bins = np.histogram(e_lengths[e_lengths<u_lim], bin_number)
        ax.stairs(counts, bins, label = event_label)
        e_max = np.max(e_lengths)
        e_mean = np.mean(e_lengths)
        e_min = np.min(e_lengths)
        e_std = np.std(e_lengths)
        max_length = max(max_length, e_max)
        max_counts = max(np.max(counts), max_counts)
        text_vals.append(f'{event_label}: Min:{e_min:d}, Max:{e_max}, Mean:{e_mean:.0f}, Std:{e_std:.0f}')
    all_text_vals.append(text_vals)
    ax.set_title(f'Monkey {monkey}')
    ax.set_xlabel(f'Event Duration (ms)')
    ax.set_ylabel(f'Count')
    ax.set_xlim([0, u_lim])

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
all_text_vals = np.vstack(all_text_vals)
for k, (text_vals, color) in enumerate(zip(all_text_vals.T, colors)):
    for i in range(len(text_vals)):
        text = text_vals[i]
        print(text)
        axs[i].text(x = 0.6*u_lim, y = (.9-0.1*k)*max_counts, s = text, size = 'x-small', color=color)
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.5, -.01), ncols=4)
fig.suptitle('Inter-Event Durations')
sns.despine()
plt.savefig(f'{summary_dir}/EventDurations_max{u_lim}.png', bbox_inches='tight', dpi=200)

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


