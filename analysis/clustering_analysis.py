import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import pickle as pkl
import os

import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

from config import (
    SUMMARY_DIR,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    BIN_SIZE             as binsize,
    KERNEL_WIDTH         as kernel_width,
    WINDOW_RANGE,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
summary_dir  = SUMMARY_DIR
window_range = np.array(WINDOW_RANGE)
n_clusters   = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# ---------------------------------------------------------------------------
# Load SDF dictionary (produced by peak_analysis.py)
# ---------------------------------------------------------------------------
all_sdf_filename = (
    f'{summary_dir}/merged_sdfDict_bin{binsize}_k{kernel_width}_orientMerged.p'
)
with open(all_sdf_filename, 'rb') as f:
    all_sdf_dict = pkl.load(f)
print('SDF dictionary loaded')

# ---------------------------------------------------------------------------
# Compute joint SDFs, wide SDFs, and hierarchical cluster assignments
# ---------------------------------------------------------------------------
joint_sdf_dict = {}
wide_sdf_dict  = {}
cluster_dict   = {'M1': {}, 'PMd': {}, 'PMv': {}}

for region in cluster_dict.keys():
    joint_sdfs = []
    wide_sdfs  = []
    for lat in ['c', 'i']:
        latArea = f'{lat}{region}'
        window_idcs = (
            (np.array(epoch_window_map['Grasp']['window']) - window_range[0]) / binsize
        ).astype(np.int_)
        wide_window_idcs = (
            (np.array([-500, 500]) - window_range[0]) / binsize
        ).astype(np.int_)
        joint_sdfs.append(
            all_sdf_dict[latArea]['trialGraspOn'][window_idcs[0]:window_idcs[1]]
        )
        wide_sdfs.append(
            all_sdf_dict[latArea]['trialGraspOn'][wide_window_idcs[0]:wide_window_idcs[1]]
        )

    joint_sdf = np.vstack(joint_sdfs)
    wide_sdf  = np.vstack(wide_sdfs).T
    wide_sdf_max = wide_sdf.max(axis=1) + 1e-6
    wide_sdf = wide_sdf / np.expand_dims(wide_sdf_max, 1).repeat(wide_sdf.shape[1], 1)

    norm_sdf = (
        joint_sdf.T
        / (np.expand_dims(joint_sdf.max(axis=0), 1).repeat(joint_sdf.shape[0], 1) + 1e-6)
    )

    joint_sdf_dict[region] = norm_sdf
    wide_sdf_dict[region]  = wide_sdf

    y = pdist(norm_sdf)
    Z = ward(y)
    for num_clust in n_clusters:
        cluster_dict[region][num_clust] = fcluster(Z, num_clust, criterion='maxclust')

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
cluster_file_name = f'{summary_dir}/Clusters.p'
with open(cluster_file_name, 'wb') as f:
    pkl.dump(cluster_dict, f)

wide_sdf_filename = f'{summary_dir}/wide_sdf_dict.p'
with open(wide_sdf_filename, 'wb') as f:
    pkl.dump(wide_sdf_dict, f)

print('Clusters.p and wide_sdf_dict.p saved')
