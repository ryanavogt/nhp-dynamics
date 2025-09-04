from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()
sns.set_style(style='white')

epoch_window_map = {'Cue':      {'event': 'trialRewardDrop', 'window': [-200,   100]},
                   'Reach':     {'event': 'trialReachOn',    'window': [-100,    60]},
                   'Grasp On':  {'event': 'trialGraspOn',    'window': [-60,    100]},
                   'Grasp Off': {'event': 'trialGraspOff',   'window': [-100,   100]}}
current_time = 0
for epoch in epoch_window_map:
    epoch_window_map[epoch]['time'] = current_time - epoch_window_map[epoch]['window'][0]
    current_time = epoch_window_map[epoch]['time'] + epoch_window_map[epoch]['window'][1]
event_times = {}
for epoch in epoch_window_map.keys():
    event_times[epoch] = epoch_window_map[epoch]['time']

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

    ax.scatter(x, y, **scatter_kwargs)

    xymax = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='b')
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='r')