"""
pca_utils.py — Shared PCA helper functions for the NHP analysis pipeline.

New pipeline utilities (added current session):
    compute_subspace         — build a PC subspace from a trial subset
    project_sdf              — project an SDF matrix onto a PC space
    align_sign               — enforce positive-onset sign convention
    project_onto_joint_space — project per-monkey SDFs into the joint PC space

Legacy utilities (from spliced_pca / updated_pca era):
    region_pca, hemi_variance, orientation_variance,
    region_projection, cond_neuron_plot
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))           # Claude outputs/
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # project root
del _sys, _os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.sig_proc import gen_sdf, center_sdf


# ===========================================================================
# New pipeline utilities
# ===========================================================================

def compute_subspace(cond_psth, trial_indices, n_dims, kernel_width, binsize):
    """
    Compute an n_dims-dimensional PC subspace from a subset of trials.

    For each neuron, sums the selected trial spike counts, smooths into an
    SDF, stacks the SDFs into a matrix, centres, runs SVD, and returns the
    first n_dims right singular vectors in time space.

    Parameters
    ----------
    cond_psth     : list of np.ndarray, each (n_trials_i, n_time)
        Per-neuron trial arrays for a single condition.  Different neurons
        may have different numbers of trials.
    trial_indices : array-like of int  OR  list of array-like of int
        Which trials to include.  Pass a single 1-D array to use the same
        indices for every neuron, or a list of arrays (one per neuron) to
        use per-neuron indices — the preferred form when neurons have
        differing trial counts.
    n_dims        : int
        Number of PC columns to return.
    kernel_width  : int or float
        Gaussian kernel width (ms) for gen_sdf.
    binsize       : int or float
        Bin size in ms.

    Returns
    -------
    V : torch.Tensor, shape (n_time, n_dims)
        First n_dims right singular vectors (in time space).
    """
    per_neuron = isinstance(trial_indices, list)
    sdfs = []
    for i, neuron_psth in enumerate(cond_psth):
        idx    = trial_indices[i] if per_neuron else trial_indices
        subset = neuron_psth[idx]                                # (n_subset, n_time)
        sdf = gen_sdf(
            subset.sum(axis=0, keepdims=True),                   # (1, n_time) sum over trials
            w=kernel_width, bin_size=binsize,
            ftype='Gauss', multi_unit=True
        )[0].squeeze()                                           # (n_time,)
        sdfs.append(sdf)
    sdf_mat    = np.stack(sdfs, axis=0)                          # (n_neurons, n_time)
    sdf_square = center_sdf(torch.Tensor(sdf_mat))['square']
    _, _, Vh   = torch.linalg.svd(sdf_square, full_matrices=False)
    return Vh.T[:, :n_dims]                                      # (n_time, n_dims)


def project_sdf(sdf, V):
    """
    Project an SDF matrix onto a PC loading space.

    Parameters
    ----------
    sdf : torch.Tensor, shape (n_neurons, n_time)
    V   : torch.Tensor, shape (n_neurons, n_pcs)

    Returns
    -------
    torch.Tensor, shape (n_time, n_pcs)
    """
    return sdf.T @ V


def align_sign(traj, ref_idx=5):
    """
    Flip a 1-D trajectory so that its value at ref_idx is positive.

    Enforces a consistent sign convention: the first meaningful post-onset
    time-point (default index 5) should be positive, avoiding arbitrary
    sign flips that differ across conditions or monkeys.

    Parameters
    ----------
    traj    : array-like or torch.Tensor, 1-D
    ref_idx : int

    Returns
    -------
    Same type as input, possibly negated.
    """
    if traj[ref_idx] < 0:
        return -traj
    return traj


def project_onto_joint_space(monkeys, cortex_map, joint_monkey='All',
                             normalize=True):
    """
    Project each monkey's condition SDFs into the joint monkey's full-data
    PC space.

    For the joint monkey itself the projection is a straightforward matrix
    product.  For individual monkeys, only their neuron rows are filled into
    a zeros tensor the size of the joint SDF, then projected.  With
    normalize=True the result is divided by each monkey's per-PC contribution
    norm squared, placing individual monkeys on a comparable scale despite
    contributing different neuron counts.

    Parameters
    ----------
    monkeys      : dict[str, Monkey]
    cortex_map   : dict   keys are cortex names ('M1', 'PMd', 'PMv')
    joint_monkey : str    key for the reference PC space (default 'All')
    normalize    : bool   if True, divide by the squared per-PC norm of each
                          monkey's neuron rows in the joint loading matrix
                          (recommended for cross-monkey comparisons)

    Returns
    -------
    dict[monkey_name][cortex][condition] -> torch.Tensor (n_time, n_pcs)
        Projected trajectory for every monkey × cortex × condition.
        'Full' condition is skipped.
    """
    joint_obj = monkeys[joint_monkey]
    result = {}

    for m_name, monkey_obj in monkeys.items():
        result[m_name] = {}
        for cortex in cortex_map.keys():
            if cortex not in monkey_obj.cortices:
                continue
            if cortex not in joint_obj.cortices:
                continue

            joint_V = joint_obj.cortices[cortex]['SVD']['Merged']['Full']['V']
            result[m_name][cortex] = {}

            for condition, cond_data in monkey_obj.cortices[cortex]['SVD']['Merged'].items():
                if condition == 'Full':
                    continue
                cond_sdf = cond_data['sdf']          # (n_neurons_monkey, n_time)

                if m_name == joint_monkey:
                    proj = project_sdf(cond_sdf, joint_V)
                else:
                    # Fill monkey's neurons into a zeros tensor sized to the
                    # joint population, then project into the joint PC space
                    base_sdf   = joint_obj.cortices[cortex]['SVD']['Merged'][condition]['sdf']
                    monkey_sdf = torch.zeros_like(base_sdf)
                    m_idx      = monkey_obj.monkey_indices[cortex]
                    monkey_sdf[m_idx] = cond_sdf
                    if normalize:
                        pc_norm = joint_V[m_idx].norm(dim=0)        # (n_pcs,)
                        proj    = project_sdf(monkey_sdf, joint_V) / pc_norm ** 2
                    else:
                        proj = project_sdf(monkey_sdf, joint_V)

                result[m_name][cortex][condition] = proj

    return result


# ===========================================================================
# Legacy utilities (from spliced_pca / updated_pca era)
# ===========================================================================

def region_pca(region_map, pop_sdf, region_name):
    """
    Compute SVD-based PCA for a named region from a population SDF matrix.

    Parameters
    ----------
    region_map  : dict  maps region labels to [start, end] row indices
    pop_sdf     : torch.Tensor  (n_neurons, n_timepoints)
    region_name : str  e.g. 'cM1', 'iM1'

    Returns
    -------
    U   : torch.Tensor  (n_timepoints, k)
    S   : torch.Tensor  (k,)
    Vt  : torch.Tensor  (k, n_neurons)   — V transposed
    cov : torch.Tensor  (n_neurons, n_neurons)
    """
    pca_sdf = []
    for key in region_map.keys():
        if region_name in key:
            pca_ind = region_map[key]
            pca_sdf.append(pop_sdf[pca_ind[0]:pca_ind[1]])
    pca_sdf    = torch.vstack(pca_sdf)
    cov        = torch.cov(pca_sdf)
    sdf_square = center_sdf(pca_sdf)['square']
    U, S, V    = torch.linalg.svd(sdf_square, full_matrices=False)
    return U, S, V.T, cov


def hemi_variance(pc_dict, region_map, baseline_region, target_regions,
                  orientation='vertical', pc_dims=10, all_regions=False,
                  plot_cov=False, fig=None, indices=0):
    """
    Plot cumulative variance explained in hemisphere PCs.

    Parameters
    ----------
    pc_dict          : dict  nested as pc_dict[orientation][region]
    region_map       : dict
    baseline_region  : str   e.g. 'cM1'
    target_regions   : list[str]
    orientation      : str   'vertical' | 'horizontal'
    pc_dims          : int or 'all'
    all_regions      : bool  if True, include different-area targets
    plot_cov         : bool  if True, also plot covariance matrix
    fig              : matplotlib.figure.Figure  existing figure
    indices          : int   which axis in fig.axes to plot on

    Returns
    -------
    dict with keys 'Variance' (and optionally 'Covariance').
    """
    if fig is None:
        fig = plt.figure()

    var_plot_dict = {}
    hemi_color = {'c': 'b', 'i': 'r'}
    hemi_map   = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_hemi  = baseline_region[0]
    base_pca   = pc_dict[orientation][baseline_region]
    base_cov   = base_pca['cov']

    if pc_dims == 'all':
        pc_dims = base_cov.shape[0]

    base_pcs       = base_pca['V']
    base_var_sum   = torch.cumsum(
        torch.diagonal(base_pcs.T @ base_cov @ base_pcs), dim=0)
    base_var_trace = base_var_sum.max()

    if plot_cov:
        import matplotlib as mpl
        f_cov = plt.figure(figsize=(8, 8))
        plt.imshow(base_cov,
                   norm=mpl.colors.SymLogNorm(linthresh=.001, linscale=.01),
                   cmap='RdBu_r')
        plt.title(f'Covariance Matrix for {baseline_region}, {orientation}')
        plt.colorbar()
        var_plot_dict['Covariance'] = f_cov

    ax_var = fig.axes[indices]
    ax_var.scatter(np.arange(1, pc_dims + 1),
                   base_var_sum[:pc_dims] / base_var_trace,
                   label=hemi_map[base_hemi], c=hemi_color[base_hemi])

    for target_region in target_regions:
        target_hemi = target_region[0]
        target_area = target_region[1:]
        base_area   = baseline_region[1:]
        if target_region == baseline_region:
            continue
        if target_area != base_area and not all_regions:
            continue
        target_pca     = pc_dict[orientation][target_region]
        target_cov     = target_pca['cov']
        target_var_sum = torch.cumsum(
            torch.diagonal(base_pcs.T @ target_cov @ base_pcs), dim=0)
        ax_var.scatter(np.arange(1, pc_dims + 1),
                       target_var_sum[:pc_dims] / target_var_sum.max(),
                       label=hemi_map[target_hemi], c=hemi_color[target_hemi])

    ax_var.set_ylim([0, 1.03])
    ax_var.set_title(
        f'{orientation.capitalize()}, {hemi_map[base_hemi].capitalize()} PCs')
    if indices % 2 == 0:
        ax_var.set_ylabel('Variance Explained')
    var_plot_dict['Variance'] = fig
    return var_plot_dict


def orientation_variance(pc_dict, base_orientation, baseline_region,
                         pc_dims=10, all_regions=False,
                         plot_cov=False, fig=None, indices=0):
    """
    Plot cumulative variance explained by orientation PCs (cross-orientation
    transfer).

    Parameters
    ----------
    pc_dict           : dict  nested as pc_dict[orientation][region]
    base_orientation  : str   'vertical' | 'horizontal'
    baseline_region   : str
    pc_dims           : int or 'all'
    all_regions       : bool
    plot_cov          : bool
    fig               : matplotlib.figure.Figure
    indices           : int

    Returns
    -------
    dict with key 'Variance'.
    """
    if fig is None:
        fig = plt.figure()

    orient_plot_dict   = {}
    target_list        = ['horizontal', 'vertical']
    target_list.remove(base_orientation)
    target_orientation = target_list[0]

    orient_color = {'vertical': 'r', 'horizontal': 'b'}
    hemi_map     = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_hemi    = baseline_region[0]

    base_pca     = pc_dict[base_orientation][baseline_region]
    base_cov     = base_pca['cov']
    if pc_dims == 'all':
        pc_dims = base_cov.shape[0]
    base_pcs       = base_pca['V']
    base_var_sum   = torch.cumsum(
        torch.diagonal(base_pcs.T @ base_cov @ base_pcs), dim=0)
    base_var_trace = base_var_sum.max()

    ax_var = fig.axes[indices]
    ax_var.scatter(np.arange(1, pc_dims + 1),
                   base_var_sum[:pc_dims] / base_var_trace,
                   label=base_orientation,
                   c=orient_color[base_orientation])

    target_pca     = pc_dict[target_orientation][baseline_region]
    target_cov     = target_pca['cov']
    target_var_sum = torch.cumsum(
        torch.diagonal(base_pcs.T @ target_cov @ base_pcs), dim=0)
    ax_var.scatter(np.arange(1, pc_dims + 1),
                   target_var_sum[:pc_dims] / target_var_sum.max(),
                   label=target_orientation,
                   c=orient_color[target_orientation])

    ax_var.set_ylim([0, 1.03])
    ax_var.set_title(
        f'{hemi_map[base_hemi].capitalize()}, {base_orientation.capitalize()} PCs')
    if indices % 2 == 0:
        ax_var.set_ylabel('Variance Explained')
    orient_plot_dict['Variance'] = fig
    return orient_plot_dict


def region_projection(all_sdf, pc_dict, region_map, data_region, pc_region,
                      pc_dims=3, orientation='vertical', fig=None, indices=0,
                      bin_size=5, epoch_windows=None):
    """
    Project a region's SDF onto another region's PC subspace and plot.

    Parameters
    ----------
    all_sdf       : torch.Tensor  (n_neurons, n_timepoints)
    pc_dict       : dict
    region_map    : dict
    data_region   : str   region whose SDF to project
    pc_region     : str   region whose PCs to use
    pc_dims       : int   how many PCs to plot
    orientation   : str
    fig           : matplotlib.figure.Figure
    indices       : int   axis index
    bin_size      : int   ms per bin
    epoch_windows : dict  epoch_window_map (for tick marks)

    Returns
    -------
    dict with key 'Projection'.
    """
    if fig is None:
        fig = plt.figure()
    if epoch_windows is None:
        epoch_windows = {}

    hemi_map  = {'c': 'contralateral', 'i': 'ipsilateral'}
    base_pcs  = pc_dict[orientation][pc_region]['V']
    data_ind  = region_map[data_region]
    data      = all_sdf[data_ind[0]:data_ind[1]]
    projected = project_sdf(data, base_pcs)

    ax_proj = fig.axes[indices]
    x_proj  = torch.arange(1, data.shape[1] * bin_size, step=bin_size)
    for pc in range(pc_dims):
        ax_proj.plot(x_proj, projected[:, pc], label=f'PC {pc + 1}')

    epoch_lines = [epoch_windows[e]['time'] for e in epoch_windows]
    ax_proj.vlines(epoch_lines,
                   ymin=projected[:, :pc_dims].min(),
                   ymax=projected[:, :pc_dims].max(),
                   colors='k')
    ax_proj.set_xticks(epoch_lines, labels=list(epoch_windows.keys()))
    ax_proj.set_title(f'{orientation.capitalize()}, {hemi_map[pc_region[0]]}')
    return {'Projection': fig}


def cond_neuron_plot(neuron_dict, epoch_window_map, plot_neurons=4,
                     fig_size=(9, 12), legend_elements=None,
                     cond_shapes=None, mean_plot=False, binsize=5):
    """
    Plot per-condition SDF traces for the top PC-loading neurons.

    Parameters
    ----------
    neuron_dict      : dict
        Keys are condition labels; values are dicts with:
            'sdf'     : torch.Tensor  (n_neurons, n_timepoints)
            'indices' : torch.Tensor  (n_neurons, 3) — sort indices per PC,
                        descending by absolute loading magnitude
    epoch_window_map : dict
    plot_neurons     : int   number of top neurons to show per PC
    fig_size         : tuple
    legend_elements  : list  extra matplotlib handles for the legend
    cond_shapes      : dict  condition → marker string (for individual traces)
    mean_plot        : bool  if True, plot mean ± SD; if False, individual traces
    binsize          : int   ms per bin

    Returns
    -------
    matplotlib.figure.Figure
    """
    if legend_elements is None:
        legend_elements = []
    if cond_shapes is None:
        cond_shapes = {'c_vertical': '+', 'c_horizontal': 'x',
                       'i_vertical': '|',  'i_horizontal': '.'}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    clr_list   = prop_cycle.by_key()['color']
    n_rows     = len(neuron_dict)
    fig, axs   = plt.subplots(nrows=n_rows, ncols=1, figsize=fig_size)
    if n_rows == 1:
        axs = [axs]

    # Consistent y-axis bounds across all conditions
    all_sdfs   = [neuron_dict[c]['sdf'] for c in neuron_dict]
    global_min = min(s.min().item() for s in all_sdfs)
    global_max = max(s.max().item() for s in all_sdfs)

    for cond_idx, condition in enumerate(neuron_dict):
        condition_sdf = neuron_dict[condition]['sdf']   # (n_neurons, n_time)
        ax            = axs[cond_idx]
        n_time        = condition_sdf.shape[1]
        x_axis        = torch.arange(1, n_time * binsize + 1, binsize)

        if not mean_plot:
            # Background: top neurons from every *other* condition, in gray
            for o_cond in neuron_dict:
                if o_cond == condition:
                    continue
                o_idx = neuron_dict[o_cond]['indices']
                for o_pc in range(3):
                    ax.plot(x_axis,
                            condition_sdf[o_idx[:plot_neurons, o_pc]].T,
                            marker=cond_shapes.get(o_cond, '.'),
                            markerfacecolor='gray', markevery=5,
                            ms=10, alpha=0.6, color='gray',
                            linestyle=(0, (5, max(1, 5 * o_pc))))

        for pc_idx in range(3):
            color    = clr_list[pc_idx]
            top_rows = neuron_dict[condition]['indices'][:plot_neurons, pc_idx]
            if not mean_plot:
                ax.plot(x_axis, condition_sdf[top_rows].T,
                        color=color, linewidth=2)
            else:
                sdf_mean = condition_sdf[top_rows].mean(dim=0)
                sdf_std  = condition_sdf[top_rows].std(dim=0)
                ax.plot(x_axis, sdf_mean, color=color, linewidth=2)
                ax.fill_between(x_axis,
                                sdf_mean - sdf_std,
                                sdf_mean + sdf_std,
                                facecolor=color, alpha=0.5)
            if cond_idx == 0:
                legend_elements.append(
                    Line2D([0], [0], color=color,
                           linestyle=(0, (5, max(1, 3 * pc_idx))),
                           label=f'PC {pc_idx + 1}'))

        times = [epoch_window_map[ep]['time'] for ep in epoch_window_map]
        ax.vlines(times, global_min * 1.1, global_max * 1.1,
                  color='k', linewidth=0.8)
        ax.set_xticks(times, labels=list(epoch_window_map.keys()))
        ax.set_ylim([global_min, global_max])
        ax.set_title(condition)

    axs[0].legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, 1.1), ncols=4)
    return fig
