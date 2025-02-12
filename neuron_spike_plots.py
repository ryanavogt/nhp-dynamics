import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os


#Define Experimental Parameters (date, subject, other file identifiers)
year = '2014'
month = '10'
day = '14'
date = f'{year}_{month}_{day}'
monkey_name = 'Green'
name_label = monkey_name[0]
m_number = 1

#Define Trial Plotting Parameters
trial_no = 5
kernel_size = 50
bin_size = 1
kernel_type = 'Gauss'


# How many units for this session
units = [1, 2]

#Define directories and load relevant data
monkey_save_dir = f'Data/Processed/Monkey_{monkey_name}'
save_dir = f'{monkey_save_dir}/{date}'
sdf_dir = f'{save_dir}/trial_sdf_{kernel_type}_{kernel_size}'
figure_dir = f'{save_dir}/Figures'
if not os.path.isdir(figure_dir):
    os.mkdir(figure_dir)
kernel_figure_dir = f'{figure_dir}/{kernel_size}'
if not os.path.isdir(kernel_figure_dir):
    os.mkdir(kernel_figure_dir)

trial_file = f'{save_dir}/trial_data.p'
trial_windows = pkl.load(open(trial_file, 'rb'))
trial_orientation = trial_windows[:, 1]
event_time_file = f'{save_dir}/eventTimes.p'
event_times = pkl.load(open(event_time_file, 'rb'))

orient_dict = {1: ('left', 'horizontal'),
               2: ('right', 'horizontal'),
               3: ('left', 'vertical'),
               4: ('right', 'vertical')}

for trial_no in range(1, trial_windows.shape[0], 5):
    print(f'Trial {trial_no}')
    hand_used, hand_orient = orient_dict[int(trial_orientation[trial_no-1])]
    sdf_trial_file = f'{sdf_dir}/SDF_trial{trial_no}.p'
    sdf_trial_dict = pkl.load(open(sdf_trial_file, 'rb'))
    rewardDrop_time = event_times['trialRewardDrop'][trial_no-1]/bin_size
    graspOff_time = event_times['trialGraspOff'][trial_no-1]/bin_size
    x_plot_limits = [rewardDrop_time - 500/bin_size, graspOff_time + 100/bin_size]

    x_labels = list(event_times.keys())
    x_labels = [x[5:] for x in x_labels]
    x_ticks = []
    for event in event_times:
        x_ticks.append(event_times[event][trial_no-1]/bin_size)

    for area in sdf_trial_dict:
        area_sdf = sdf_trial_dict[area][1]
        print(f'Spike sum for {area} : {area_sdf.sum()}')
        fig, ax = plt.subplots(figsize = (8,3))
        ax.set_xlim(x_plot_limits)
        if area_sdf[:, 0].shape[0]>1:
            x = area_sdf[:, 0].T
        else:
            x = area_sdf[:, 0].T
        ax.plot(x)
        plt.title(f'SDF of {area}, Trial {trial_no}, Kernel Width = {kernel_size}, \nHand: {hand_used}, Orient: {hand_orient}, Neurons = {area_sdf.shape[0]}')
        ax.set_xticks(x_ticks, x_labels, rotation = 30, ha='right', rotation_mode='anchor')
        ax.set_xticklabels(x_labels, )
        # plt.show()
        plt.savefig(f'{kernel_figure_dir}/{area}_{trial_no}_{kernel_size}_sdf.png', dpi=300, bbox_inches='tight')
        plt.close()