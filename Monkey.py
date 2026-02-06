from DSA import DSA,DMD
from DSA.stats import mse
import numpy as np
import os
import pickle as pkl
import random
import torch

from sig_proc import *


class Monkey:
    def __init__(self, name, kernel_width, bin_size, epoch_window_map, verbose=False, seed=31):
        self.save_dir = None
        self.name = name
        self.kernel_width = kernel_width
        self.bin_size = bin_size
        self.epoch_window_map = epoch_window_map
        self.monkey_indices = {}
        self.cortices = {}
        self.condition_map = {}
        self.verbose=verbose
        self.seed = random.seed(seed)


    def save_monkey(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        monkey_save_name = f'{self.save_dir}/monkey{self.name}.p'
        with open(monkey_save_name, 'wb') as monkey_file:
            pkl.dump(self, monkey_file)

    def check_cortex(self, cortex):
        if cortex not in self.cortices:
            self.cortices[cortex] = {}

    def set_condition_map(self, condition_map):
        self.condition_map = condition_map

    def get_indices(self, monkey_indices, cortex):
        """
        Sets the object's indices for this cortex, needs to be repeated for all cortices
        :param monkey_indices: The set of all monkey indices corresponding to cortex (dict)
        :param cortex: The cortex of the monkey indices (string)
        """
        self.monkey_indices[cortex] = monkey_indices[self.name]
        return self.monkey_indices[cortex]

    def get_savedir(self, root_dir):
        """
        Sets the save directory for this monkey, where all related files (and subdirectories) will be saved
        :param root_dir: String of base directory
        """
        self.save_dir = f'{root_dir}/PCA_b{self.bin_size}_k{self.kernel_width}/Monkey{self.name}'
        return self.save_dir

    def get_sdf(self, cort_sdf, cortex):
        """
        Split the sdf for the neurons of this monkey by cortex and by condition
        :param cort_sdf: Full sdf of this entire cortex (not split by monkey)
        :param condition_map: Dictionary of indices for conditions in this cortex
        :param cortex: Cortex for the SDF and condition_map
        :return: Dictionary of SDF for this cortex, keys: conditions, values: SDF
        """
        self.check_cortex(cortex)
        cortex_sdf = {}
        for (cond, cond_idcs) in self.condition_map[cortex].items():
            cortex_sdf[cond] = cort_sdf[cond_idcs[0]:cond_idcs[1]][self.monkey_indices[cortex]]
        self.cortices[cortex]['sdf'] = cortex_sdf
        return cortex_sdf

    def get_trial_psth(self, cort_psth, cortex):
        self.check_cortex(cortex)
        cortex_psth = {}
        for (cond, cond_idcs) in self.condition_map[cortex].items():
            cortex_psth[cond] = cort_psth[cond_idcs[0]:cond_idcs[1]][self.monkey_indices[cortex]]
        self.cortices[cortex]['psth'] = cortex_psth
        return self.cortices[cortex]['psth']

    def get_svd(self, cortex):
        """
        Split principal vectors (V) by condition for this cortex
        :param condition_map: Dictionary of indices for conditions in this cortex
        :param cortex: Cortex for the V and condition_map
        :return: Dictionary of principal vectors (V) for this cortex
        """
        self.check_cortex(cortex)
        sdf = self.cortices[cortex]['sdf']
        cortex_svd = {}
        pca_sdf = []
        pca_conds = []
        for condition, ind in self.condition_map[cortex].items():
            pca_sdf.append(sdf[condition])
            pca_conds.append(condition)
        cort_sdf = torch.Tensor(np.hstack(pca_sdf))
        cort_cov = torch.cov(cort_sdf)
        cort_dict_sdf = center_sdf(cort_sdf)
        U, S, Vh = torch.linalg.svd(cort_dict_sdf['square'], full_matrices=False)
        V = Vh.T
        cortex_svd['Full'] = {'U': U, 'S': S, 'V': V, 'sdf': cort_dict_sdf['centered'], 'cov': cort_cov}
        for (cond, cond_idcs) in self.condition_map[cortex].items():
            cond_sdf = self.cortices[cortex]['sdf'][cond]
            cond_cov = torch.cov(cond_sdf)
            dict_sdf = center_sdf(cond_sdf)
            U, S, Vh = torch.linalg.svd(dict_sdf['square'], full_matrices=False)
            V = Vh.T
            cortex_svd[cond] = {'U':U, 'S':S, 'V':V, 'sdf': dict_sdf['centered'], 'cov':cond_cov}
        self.cortices[cortex]['SVD'] = cortex_svd
        return cortex_svd

    def get_dmds(self, cortex, k_folds=5, verbose=False, permute = False, **dsa_kwargs):
        split_data_dict = {}
        for cond, psth in self.cortices[cortex]['psth'].items():
            k_folds = min(k_folds, psth.shape[1])
            # permutation = np.arange(psth.shape[1])
            # part_size = sdf.shape[0] // k_folds
            # if permute:
            #     permutation = np.random.permutation(permutation)
            # inv_perm = np.argsort(permutation)
            # split_data = np.array_split(sdf[permutation][:part_size*k_folds], k_folds)
            # part_permutation = permutation[:part_size * k_folds]
            # part_inv_permutation = inv_perm[:part_size * k_folds]
            split_psth = np.array_split(psth, k_folds, axis=1)
            split_data_dict[cond] = split_psth
        test_losses={}
        train_losses = {}
        dmds = {}
        sim_list = []
        for i in range(k_folds):
            test_data = []
            train_data = []
            for cond in self.cortices[cortex]['sdf'].keys():
                if cond not in train_losses.keys():
                    train_losses[cond] = []
                    test_losses[cond]  = []
                    dmds[cond] = []
                cond_test_psth = split_data_dict[cond][i]
                test_data_sdf = gen_sdf(cond_test_psth.sum(axis=1), ftype='Gauss', w=self.kernel_width, bin_size=self.bin_size, multi_unit=True)[
                     0].squeeze()
                # cond_test_sdf = np.swapaxes(test_data_sdf, -2, -1)
                cond_test_sdf = test_data_sdf
                test_data.append(cond_test_sdf)
                train_list = split_data_dict[cond].copy()
                train_list.pop(i)
                cond_train_psth =  np.swapaxes(np.stack(train_list), -2,-1)
                cond_train_sdf = np.stack([gen_sdf(p.sum(axis=1), ftype='Gauss', w=self.kernel_width, bin_size=self.bin_size,
                                  multi_unit=True)[0].squeeze() for p in train_list], axis=0)
                train_data.append(np.swapaxes(cond_train_sdf, -2,-1))
                dmd_cond = DMD(cond_train_sdf, **dsa_kwargs)
                dmd_cond.fit(rank=dsa_kwargs['rank'])
                train_preds = dmd_cond.predict()
                test_preds = dmd_cond.predict(cond_test_sdf)
                train_mse = mse(cond_train_sdf, train_preds)
                test_mse = mse(cond_test_sdf, test_preds)
                train_losses[cond].append(train_mse)
                test_losses[cond].append(test_mse)
                dmds[cond].append(dmd_cond)
            # dsa = DSA(train_data)
            # sim_list.append(dsa.fit_score())
        # sim_list = np.array(sim_list).mean(axis=0)
        for cond in test_losses.keys():
            test_losses[cond] = np.mean(test_losses[cond])
            train_losses[cond] = np.mean(train_losses[cond])
        return test_losses, train_losses, dmds