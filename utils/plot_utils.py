from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import colorsys
import math
from utils.sig_proc import gen_psth, gen_sdf
import matplotlib as mpl

# Import shared constants from central config — re-exported under the original names
# so that existing callers (`from plot_utils import epoch_window_map`) still work.
from config import (
    EPOCH_WINDOW_MAP as epoch_window_map,
    EVENT_TIMES as event_times,
    TOTAL_TIME as current_time,
)


def apply_theme(ticks=True):
    """Apply seaborn theme.  Call once at the start of a script, not on import."""
    sns.set_theme()
    sns.set_style(style='ticks' if ticks else 'white')

def pc_subplot(pca, axs, i, region='', plot_signals=1, window=[0], binsize=5, n_plot =4, projection = True,
               event_times=event_times):
    """
    :param event_times:
    :param pca: Dictionary containing U, S, V from PCA
    :param axs: List of 2D subplot axes
    :param i: Index of subplot
    :param region: Region of data projected onto PCs
    :param plot_signals: Number of trials to plot (default: 1)
    :param window: Indices of U corresponding to region
    :param binsize: Width of bins used in the PSTH
    :param n_plot: Number of PCs to plot
    :param projection: boolean: if True, plot the projection of data on PCs. Otherwise, plot PC directly
    :return: Updated Axes list with subplot complete
    """

    U, S, V = pca['U'], pca['S'], pca['V']
    y_max = -1000
    y_min = 1000
    row = i // 2
    column = i % 2
    for j in range(1, n_plot + 1):
        plot_x = np.arange(0, current_time, binsize)
        if projection: # Plot projection of data onto principal components
            plot_y = (U[window[0]:window[0] + plot_signals, j - 1:j] @ np.diag(S[j - 1:j]) @ V[:,
                                                j - 1:j].T).T * 1000 / binsize
        else: # Plot principal components directly
            plot_y = V[:, j-1:j]
        axs[row][column].plot(plot_x, plot_y, label=f'PC{j}')
        y_max = max(plot_y.max(), y_max)
        y_min = min(plot_y.min(), y_min)
    axs[row][column].vlines(event_times.values(), ymin=y_min, ymax=y_max,
                            colors='k', linestyles='dashed')
    axs[row][column].set_xticks(list(event_times.values()), labels=event_times.keys())
    axs[row][column].set_title(f'{region}')
    # axs[row][column].set_ylabel('Firing Rate (Hz)')
    return axs

def scatter_hist(x, y, ax, ax_histx, ax_histy, binwidth=0.1, **scatter_kwargs):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.plot([-1.1, 1.1], [0, 0], 'k--', alpha=0.25)
    ax.plot([0, 0], [-1.1, 1.1], 'k--', alpha=0.25)

    scatter_artist = ax.scatter(x, y, **scatter_kwargs)

    xymax = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='b')
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='r')
    return scatter_artist

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def add_headers(fig, *, row_headers=None, col_headers=None, row_pad=1, col_pad=5,
                rotate_row_headers=True, **text_kwargs):
    axes = fig.get_axes()
    for ax in axes:
        sbs = ax.get_subplotspec()
        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(col_headers[sbs.colspan.start], xy=(0.5, 1), xytext=(0, col_pad), xycoords="axes fraction",
                        textcoords="offset points", ha="center", va="baseline", **text_kwargs,)
        # Putti          ng headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(row_headers[sbs.rowspan.start], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - row_pad, 0),
                        xycoords=ax.yaxis.label, textcoords="offset points", ha="right", va="center",
                        rotation=rotate_row_headers * 90, **text_kwargs,)

def raster_plot(neuron_list, region_spikes, event_list, epoch_window_map, conditions=None, n_cols=4, indices=None,
                sdf = False, window = [0,820], binsize=5, kernel_width=25,
                condition_colors={'c-h':'orange','c-v': 'red','i-h':'blue','i-v':'green'},
                event_means= None, baseline_means = None, fig_size = None, fig = None):
    total_neurons = np.array(neuron_list).shape[0]
    n_rows = math.ceil(total_neurons/n_cols)
    if fig_size == None:
        fig_size = (4*n_cols, 2.5*n_rows)
    if fig is None:
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
                                constrained_layout=True)
    else:
        axs = np.array(fig.subplots(nrows=n_rows, ncols=n_cols)).reshape(n_rows, n_cols)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
    epoch_times = [epoch_window_map[event]['time'] for event in epoch_window_map.keys()]
    for idx, neuron in enumerate(neuron_list):
        idx_string = ''
        f_weight = 'normal'
        if event_means is not None:
            sdf_mean = event_means[idx]
        if baseline_means is not None:
            base_mean = baseline_means[idx]
        if n_cols < total_neurons:
            ax = axs[idx // n_cols][idx % n_cols]
        else:
            ax = axs[idx]
        spikes = region_spikes[neuron]
        if conditions is not None:
            no_trials = len(conditions[idx])
        else:
            no_trials = spikes[:,1].max()
        neuron_spikes = [[] for _ in range(no_trials)]
        neuron_trials = [[] for _ in range(no_trials)]
        ordered_spikes = []
        if idx%50 == 0:
            print(f'Neuron {neuron}')
        if indices is not None:  
            neuron_idx = np.where(indices == neuron)[1]
            idx_string += f', ch:{neuron_idx[0]+1}, cv:{neuron_idx[1]+1}, ih:{neuron_idx[2]+1}, iv:{neuron_idx[3]+1}'
            sig_neuron = neuron_idx < 10
            if sig_neuron.sum()>0:
                f_weight = 'bold'
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(3)
        for event in event_list.keys():
            event_window = epoch_window_map[event]['window']
            event_times = np.array(event_list[event][neuron])
            e_starts = event_times + event_window[0]
            e_ends = event_times + event_window[1]
            e_times = [0] * no_trials
            for trial in np.arange(1, no_trials+1):
                if trial == len(event_times):
                    break

                event_start = e_starts[trial]
                event_end = e_ends[trial]
                # e_time = e_times[trial] + event_start
                e_time = event_start
                trial_spikes = spikes[spikes[:,1]==trial, 0]
                trials = spikes[spikes[:,1]==trial, 1]
                event_mask = (trial_spikes>event_start)*(trial_spikes<event_end)
                # neuron_spikes[trial] = neuron_spikes[trial]+list(spike_list[event_mask].astype(np.int16)-event_start+
                #                                                  epoch_window_map[event]['time' ]+epoch_window_map[event]['window'][0])
                neuron_spikes[trial-1]=neuron_spikes[trial-1]+list(trial_spikes[event_mask].astype(np.int16)-event_start
                                                                   + epoch_window_map[event]['time' ]
                                                                   + epoch_window_map[event]['window'][0])
                neuron_trials[trial-1]=neuron_trials[trial-1] + list(trials[event_mask])
                e_times[trial-1] += (event_end-event_start)
        for trial, trial_spike_list in enumerate(neuron_spikes):
            trial_spikes = np.array(trial_spike_list)
            neuron_spikes[trial] = trial_spikes
        neuron_trials_flat = np.concatenate(neuron_trials)
        neuron_spikes_flat = np.concatenate(neuron_spikes)
        ax.set_title(f'Neuron {neuron}{idx_string}',fontweight=f_weight)
        ax.set_ylabel(f'Trial')
        ax.set_xlabel(f'Time (ms)')
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True))
        ax.tick_params(axis='y', labelsize=8, pad=0.5, left=True)
        ax.tick_params(axis='x', labelsize=8)
        max_time = epoch_window_map['Grasp']['time'] + epoch_window_map['Grasp']['window'][1]
        ax.set_xlim([0, max_time])
        ax.set_ylim([0,no_trials])
        if conditions is not None:
            cond_order = np.arange(no_trials)
            cond_masks = {}
            cond_idx = 0
            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_mask = (conditions[idx][:, 0] == condition[0]) * (
                            conditions[idx][:, 1] == condition.split('-')[1][0])
                cond_masks[condition] = cond_mask
                cond_inds = np.where(cond_mask)[0]
                trial_count = len(cond_inds)
                cond_min = cond_idx
                cond_max = cond_idx+trial_count
                cond_order[cond_min:cond_max] = cond_inds
                # cond_spikes_mask = [neuron_trials_flat==i for i in np.where(cond_mask)[0]]
                cond_spikes_mask = [neuron_trials_flat == (i + 1) for i in np.where(cond_mask)[0]]
                cond_spikes = [neuron_spikes_flat[mask] for mask in cond_spikes_mask]
                # cond_max, cond_min = cond_inds[0].max(), cond_inds[0].min()
                cond_rect = mpl.patches.Rectangle((0, cond_min), width=max_time, height=cond_max-cond_min,
                                                  color=condition_colors[condition], alpha=0.1)
                ax.add_patch(cond_rect)
                cond_idx = cond_max
                ordered_spikes+=cond_spikes
                # print(f'{condition}: {cond_inds}')
        else:
            ordered_spikes = neuron_spikes
        ax.invert_yaxis()
        # if conditions is not None:
        #     ordered_spikes = [neuron_spikes[i] for i in cond_order]
        # else:
        #     ordered_spikes = neuron_spikes
        ax.eventplot(ordered_spikes, color='k')
        if conditions is not None:
            cond_idx = 0
            max_sdf = 0
            cond_sdfs = {}
            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_mask = cond_masks[condition]
                trial_count = len(np.where(cond_mask)[0])
                cond_min = cond_idx
                cond_max = cond_min + trial_count
                cond_idx = cond_max
                # print(f'{condition}: {cond_min}:{cond_max}')
                if sdf:
                    cond_spike_list = []
                    # for i in range(cond_mask.shape[0]):
                    #     if cond_mask[i]:
                    for i in np.arange(cond_min,cond_max):
                        cond_spike_list.append(ordered_spikes[i])
                    spike_input = np.concatenate(cond_spike_list)
                    psth_input = np.ones((spike_input.shape[0],2))
                    psth_input[:, 1] = spike_input  # spike times in col 1 (gen_psth convention)
                    # col 1 stays 1 (single pseudo-trial for pooled condition spikes)
                    cond_psth = gen_psth(psth_input, binsize=binsize, window=np.array(window))
                    cond_sdf, _ = gen_sdf(cond_psth[:, 1:], w=kernel_width, bin_size=binsize, ftype='Gauss',
                                          multi_unit=False)
                    cond_sdf = cond_sdf.squeeze()
                    if max_sdf<cond_sdf.max():
                        max_sdf = cond_sdf.max()
                    cond_sdfs[condition] = {'sdf': cond_sdf, 'cond_max': cond_max, 'cond_min': cond_min}

            for cond_i, condition in enumerate(condition_colors.keys()):
                cond_vals = cond_sdfs[condition]
                cond_sdf, cond_max, cond_min = cond_vals['sdf'], cond_vals['cond_max'], cond_vals['cond_min']
                if event_means is not None and max_sdf > 0:
                    sides, event_mean = sdf_mean
                    sdf_mean_val = event_mean[[side[:4]==condition[:4] for side in sides]]
                    if len(sdf_mean_val) > 0:
                        plot_mean_val = -float(sdf_mean_val.iloc[0])*(cond_max - cond_min)*0.9 / max_sdf + cond_max
                        ax.axhline(y=plot_mean_val, ls='--', alpha=0.6, c='black')
                if baseline_means is not None and max_sdf > 0:
                    sides, baseline_mean = base_mean
                    base_mean_val = baseline_mean[[side[:4]==condition[:4] for side in sides]]
                    if len(base_mean_val) > 0:
                        plot_base_val = -float(base_mean_val.iloc[0])*(cond_max - cond_min)*0.9 / max_sdf + cond_max
                        ax.axhline(y=plot_base_val, ls='--', alpha=0.4, c='grey')
                sdf_scaled = -cond_sdf * (cond_max - cond_min)*0.9 / max_sdf + cond_max
                ax.plot(cond_psth[:, 0], sdf_scaled, color=condition_colors[condition])
                bbox= None
                if indices is not None:
                    if sig_neuron[cond_i]:
                        bbox = dict(boxstyle="square", fc = 'none', ec="black", lw=2, pad=0.1)
                ax.annotate(xy=(window[1] - 15, cond_min + 0.5), rotation=270, ha='right', va='top', text=condition,
                            color=condition_colors[condition],
                            alpha=0.8, fontsize=9.5, bbox=bbox)
        for epoch_time in epoch_times:
            ax.axvline(epoch_time, alpha=.8, color='k')

    return fig, axs