import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from plot_utils import pc_subplot
from Monkey import Monkey

monkey_name_map = {'G':'Green', 'R':'Red', 'Y':'Yellow', 'B':'Blue'}
event_map = {'trialRewardDrop': 'Cue', 'trialGraspOn':'Grasp On', 'trialGraspOff':'Grasp Off', 'trialReachOn':'Reach'}

#Define directories of data
lat_map = {'c':'contralateral', 'i':'ipsilateral'}
hand_list = ['R', 'L']
binsize = 5
kernel_width = 25
summary_dir = f'Data/Processed/Summary'
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'

#DSA/DMD hyperparameters
ranks = [3, 5, 10, 20]
n_delays = [5]
k_folds = 5

for monkey_name in monkey_name_map.keys():
    monkey_filename = f'{pca_dir}/Monkey{monkey_name}/monkey{monkey_name}.p'
    with open(monkey_filename, 'rb') as monkey_file:
        monkey = pkl.load(monkey_file)
    dmd_dir = f'{monkey.save_dir}/DMD'
    if not os.path.exists(dmd_dir):
        os.mkdir(dmd_dir)
    for cortex in monkey.cortices.keys():
        for rank in ranks:
            for n_delay in n_delays:
                dmd_filename = f'{dmd_dir}/dmd_{cortex}_rank{rank}_delays{n_delay}_kfold{k_folds}.p'
                test_losses, train_losses, dmds = monkey.get_dmds(cortex, k_folds = k_folds, n_delays=n_delay,
                                                                            rank=rank, delay_interval=1, device = 'cpu')
                print(f'Monkey {monkey.name}, {cortex}, Rank {rank}, {n_delay} Delays DMD Test Losses: '
                      f'{[f'{cond}: {loss:.5f}' for cond, loss in test_losses.items()]}')
                dmd_dict = {'test_loss':test_losses, 'train_loss':train_losses, 'dmds':dmds}
                with open(dmd_filename, 'wb') as dmd_file:
                    pkl.dump(dmd_dict, dmd_file)