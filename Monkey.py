from DSA import DSA,DMD
from DSA.stats import mse
import numpy as np

class Monkey:
    def __init__(self, name, kernel_width, bin_size, epoch_window_map, all_x, verbose=False):
        self.save_dir = None
        self.name = name
        self.kernel_width = kernel_width
        self.bin_size = bin_size
        self.epoch_window_map = epoch_window_map
        self.x = all_x
        self.monkey_indices = {}
        self.sdf = {}
        self.V_dict = {}
        self.verbose=verbose

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
        self.save_dir = f'{root_dir}/PCA_b{self.bin_size}_k{self.kernel_width}'
        return self.save_dir

    def get_sdf(self, cort_sdf, condition_map, cortex):
        """
        Split the sdf for the neurons of this monkey by cortex and by condition
        :param cort_sdf: Full sdf of this entire cortex
        :param condition_map: Dictionary of indices for conditions in this cortex
        :param cortex: Cortex for the SDF and condition_map
        :return: Dictionary of SDF for this cortex, keys: conditions, values: SDF
        """
        cortex_sdf = {}
        for (cond, cond_idcs) in condition_map.items():
            cortex_sdf[cond] = cort_sdf[cond_idcs[0]:cond_idcs[1]][self.monkey_indices[cortex]]
        self.sdf[cortex] = cortex_sdf
        return self.sdf[cortex]

    def get_svd(self, cort_V, condition_map, cortex):
        """
        Split principal vectors (V) by condition for this cortex
        :param cort_V: Full PC vector matrix (V) for this cortex
        :param condition_map: Dictionary of indices for conditions in this cortex
        :param cortex: Cortex for the V and condition_map
        :return: Dictionary of principal vectors (V) for this cortex
        """
        cortex_V = {}
        for (cond, cond_idcs) in condition_map.items():
            cortex_V[cond] = cort_V[cond_idcs[0]:cond_idcs[1]][self.monkey_indices[cortex]]
        self.V_dict[cortex] = cortex_V
        return self.V_dict[cortex]

    def get_dmds(self, cortex, k_folds=5, verbose=False, **dsa_kwargs):
        split_data_dict = {}
        for cond, sdf in self.sdf[cortex].items():
            split_data = np.split(sdf, k_folds)
            split_data_dict[cond] = split_data
        test_losses={}
        train_losses = {}
        dmds = {}
        sim_list = []
        for i in range(k_folds):
            test_data = []
            train_data = []
            for cond in self.sdf[cortex].keys():
                if cond not in train_losses.keys():
                    train_losses[cond] = []
                    test_losses[cond]  = []
                    dmds[cond] = []
                test_data.append(split_data_dict[cond][i])
                train_data.append(np.vstack(split_data_dict[cond].copy().pop(i)))
                dmd_cond = DMD(train_data, **dsa_kwargs)
                dmd_cond.fit(rank=dsa_kwargs['rank'])
                train_preds = dmd_cond.predict()
                test_preds = dmd_cond.predict(test_data)
                train_mse = mse(train_data, train_preds)
                test_mse = mse(test_data, test_preds)
                train_losses[cond].append(train_mse)
                test_losses[cond].append(test_mse)
                dmds[cond].append(dmd_cond)
            dsa = DSA(train_data)
            sim_list.append(dsa.fit_score())
        sim_list = np.array(sim_list).mean(axis=0)
        for cond in test_losses.keys():
            test_losses[cond] = np.mean(test_losses[cond])
            train_losses[cond] = np.mean(train_losses[cond])
        return test_losses, train_losses, sim_list, dmds


