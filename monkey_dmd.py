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
from DSA.simdist import SimilarityTransformDist
from matplotlib.patches import Rectangle

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
n_delays = [1, 3]
k_folds = 5

LOAD = True
all_dmd_filename = f'{pca_dir}/allDMDs.p'
all_dmd_list_filename = f'{pca_dir}/allDMDLists.p'
all_dmd_lists = {}
if os.path.exists(all_dmd_filename) and LOAD:
    all_dmds = pkl.load(open(all_dmd_filename, 'rb'))
    all_dmd_lists = pkl.load(open(all_dmd_filename, 'rb'))
    print('Loading DMD Dictionary')
else:
    all_dmds = {}
    for monkey_name in monkey_name_map.keys():
        monkey_filename = f'{pca_dir}/Monkey{monkey_name}/monkey{monkey_name}.p'
        if monkey_name not in all_dmds.keys():
            all_dmds[monkey_name] = {}
        with open(monkey_filename, 'rb') as monkey_file:
            monkey = pkl.load(monkey_file)
        dmd_dir = f'{monkey.save_dir}/DMD'
        if not os.path.exists(dmd_dir):
            os.mkdir(dmd_dir)
        for cortex in monkey.cortices.keys():
            if cortex not in all_dmds[monkey_name].keys():
                all_dmds[monkey_name][cortex] = {}
            for rank in ranks:
                if rank not in all_dmd_lists.keys():
                    all_dmd_lists[rank] = {}
                if rank not in all_dmds[monkey_name][cortex].keys():
                    all_dmds[monkey_name][cortex][rank] = {}
                rank_dir = f'{dmd_dir}/Rank{rank}'
                if not os.path.exists(rank_dir):
                    os.mkdir(rank_dir)
                for n_delay in n_delays:
                    if n_delay not in all_dmd_lists[rank].keys():
                        all_dmd_lists[rank][n_delay] = []
                    delay_dir = f'{rank_dir}/Delays{n_delay}'
                    if not os.path.exists(delay_dir):
                        os.mkdir(delay_dir)
                    dmd_filename = f'{delay_dir}/dmd_{cortex}_rank{rank}_delays{n_delay}_kfold{k_folds}.p'
                    test_losses, train_losses, dmds = monkey.get_dmds(cortex, k_folds = k_folds, n_delays=n_delay,
                                                                                rank=rank, delay_interval=1, device = 'cpu')
                    print(f'Monkey {monkey.name}, {cortex}, Rank {rank}, {n_delay} Delays DMD Test Losses: '
                          f'{[f'{cond}: {loss:.5f}' for cond, loss in test_losses.items()]}')
                    dmd_dict = {'test_loss':test_losses, 'train_loss':train_losses, 'dmds':dmds}
                    all_dmds[monkey_name][cortex][rank][n_delay] = dmds
                    all_dmd_lists[rank][n_delay].append(dmds)
                    with open(dmd_filename, 'wb') as dmd_file:
                        pkl.dump(dmd_dict, dmd_file)
    with open(all_dmd_filename, 'wb') as all_dmd_file:
        pkl.dump(all_dmds, all_dmd_file)
    with open(all_dmd_list_filename, 'wb') as all_dmd_list_file:
        pkl.dump(all_dmd_lists, all_dmd_list_file)

n_delay=3
rank=5
sim_dict = {}
simdist = SimilarityTransformDist(iters = 1000)
sim_filename = f'{pca_dir}/AllSims.p'
if LOAD and os.path.exists(sim_filename):
    sim_dict = pkl.load(open(sim_filename, 'rb'))
    print("Loading Similarity Measures")
else:
    for monkey in all_dmds.keys():
        for cortex in all_dmds[monkey].keys():
            for cond in all_dmds[monkey][cortex][rank][n_delay]:
                base_dmds = all_dmds[monkey][cortex][rank][n_delay][cond]
                base_data = [dmd.data for dmd in base_dmds]
                sim_dict[f'{monkey}_{cortex}_{cond}'] = {}
                for o_monkey in all_dmds.keys():
                    # for o_cortex in all_dmds[o_monkey].keys():
                    for o_cond in all_dmds[o_monkey][cortex][rank][n_delay]:
                        # simdist = SimilarityTransformDist(iters=5000, lr=1e-3)
                        o_dmds = all_dmds[o_monkey][cortex][rank][n_delay][o_cond]
                        o_data = [dmd.data for dmd in o_dmds]
                        sim_dists = []
                        for base_dmd, o_dmd in zip(base_data, o_data):
                            sim_dists.append(simdist.fit_score(base_dmd.A_v, o_dmd.A_v))
                            # dsa = DSA(base_data, o_data, n_delays=n_delay, rank=rank, iters=500)
                            # sims = dsa.fit_score()
                        sim_dists = np.vstack(sim_dists)
                        sim_mean, sim_std = sim_dists.mean(), sim_dists.std()
                        sim_dict[f'{monkey}_{cortex}_{cond}'][f'{o_monkey}_{cortex}_{o_cond}'] = [sim_mean, sim_std]
                        print(f'{monkey}_{cortex}_{cond}:{o_monkey}_{cortex}_{o_cond} : {sim_mean:.6f}+-{sim_std:.6f}')
    with open(sim_filename, 'wb') as sim_file:
        pkl.dump(sim_dict, sim_file)

lat_map = {'c':'con', 'i':'ips'}
for cortex in ['M1', 'PMv', 'PMd']:
    tick_list = []
    monkey_list= []
    similarity_matrix = []
    for condition in sim_dict.keys():
        if cortex not in condition:
            continue
        sims = np.vstack([d[0] for d in list(sim_dict[condition].values())])
        similarity_matrix.append(sims)
        monkey, cort, lateral, orient = condition.split('_')
        tick_label = f'{lat_map[lateral]}-{orient[:4]}'
        tick_list.append(tick_label)
        monkey_list.append(monkey)
    monkey_list = np.vstack(monkey_list)
    similarity_matrix = np.hstack(similarity_matrix)
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    im = plt.imshow(similarity_matrix)
    for monkey in np.unique(monkey_list):
        monkey_idcs = np.where(monkey_list==monkey)[0]
        plt.text(-3, np.average(monkey_idcs)+1, f'Monkey {monkey}',
                 rotation='vertical', horizontalalignment='center', verticalalignment='bottom')
        plt.text(np.average(monkey_idcs)+1, similarity_matrix.shape[0]+2, f'Monkey {monkey}',
                 rotation='horizontal', horizontalalignment='center', verticalalignment='top')
        box_length = max(monkey_idcs)-min(monkey_idcs)+1
        box = Rectangle((min(monkey_idcs)-.5,min(monkey_idcs)-.5), width=box_length, height=box_length,
                             edgecolor=monkey.lower(), facecolor='none', linewidth=3)
        ax.add_patch(box)
    ax.set_xticks(np.arange(0, len(tick_list)))
    ax.set_yticks(np.arange(0, len(tick_list)))
    ax.set_title(f'{cortex} Similiarity Matrix')
    ax.set_xticklabels(labels = tick_list, rotation=90, ha='right')
    ax.set_yticklabels(labels = tick_list)
    plt.colorbar(im, ax=ax)
    plt.savefig(f'{pca_dir}/SimMatrix_{cortex}_delay{n_delay}_rank{rank}.png', dpi=150, bbox_inches='tight')