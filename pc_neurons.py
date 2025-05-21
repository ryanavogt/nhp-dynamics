import pickle as pkl                #Saving/loading data (built-in)
import os                           #Directory Creation and Verification (built-in)

# The following packages need to be installed in your virtual environment (using conda or pip)
import matplotlib.pyplot as plt     #Generating plots
import matplotlib as mpl
from matplotlib.lines import Line2D
import torch

from sig_proc import *

import seaborn as sns

binsize = 5
kernel_width = 25

summary_dir = f'Data/Processed/Summary'
pca_dir = f'{summary_dir}/PCA_b{binsize}_k{kernel_width}'

cortex_list = ['M1', 'PMd', 'PMv']
for cortex in cortex_list:
    pca_filename = f'{pca_dir}/PCA_{cortex}_b{binsize}_k{kernel_width}.p'
    with open(pca_filename, 'rb') as pca_file:
        cortex_pca_vals = pkl.load(pca_file)
    V = cortex_pca_vals['V']
