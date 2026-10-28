"""
pca_plot_trajectories.py — PC trajectory and variance-explained figures.

Loads the Monkey objects produced by pca_compute.py and plots:
  • Per-cortex × per-monkey figure (2×2 layout):
      - Variance explained by condition (ax_var)
      - 2-D PC trajectory (ax2)
      - 3-D PC trajectory (ax3)
      - Principal angle heatmap with mean ± SD annotations (ax_angle)
  • Population-level velocity figure (PCVelocity.png)
  • Joint monkey PC trajectory figure (Joint_PCTrajectories.png)
  • Combined all-region figure (Combined_PCTrajectories_All.png)
      - Width 11, 2×6 panel grid (2 rows × 2 cols per region) + narrow colorbar col
      - Panel labels a_i … a_iv, b_i … b_iv, c_i … c_iv
      - Centered region titles via figure coordinates
      - Single shared colorbar for all heatmaps

Figures saved under Results/PCA/figures/:
    Monkey{m}/PCTraj3D_{cortex}_monkey{m}_PCdims{d}.png
    PCVelocity.png
    Joint_PCTrajectories.png
    Combined_PCTrajectories_All.png

Data loaded from Results/PCA/data/PCA_b{b}_k{k}/.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
del _sys, _os

import os
import pickle as pkl
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from utils.pca_utils import project_sdf, project_onto_joint_space
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
import matplotlib.font_manager as fm
# fm._load_fontmanager(try_read_cache=False)

from utils.sig_proc import velocity
from utils.plot_utils import apply_theme
from config import (
    MONKEY_NAME_MAP        as monkey_name_map,
    EVENT_MAP_MERGE        as event_map_aligned,
    EVENT_MAP              as event_map,
    EPOCH_WINDOW_MAP_MERGE as epoch_window_map,
    EPOCHS_MERGE           as epochs,
    PCA_RESULTS_DIR,
    BIN_SIZE               as binsize,
    KERNEL_WIDTH           as kernel_width,
    N_DIMS,
)
from plot_config import (
    FIGURE_WIDTH,
    DPI_DRAFT, DPI_FINAL,
    PANEL_LABEL_SIZE,
    REGION_TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    ANNOT_SMALL,
    apply_rcparams,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
apply_theme()
apply_rcparams()
print(fm.findfont('Arial'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PCA_DATA    = os.path.join(PCA_RESULTS_DIR, 'data')
PCA_FIG_DIR = os.path.join(PCA_RESULTS_DIR, 'figures')
pca_dir     = os.path.join(PCA_DATA, f'PCA_b{binsize}_k{kernel_width}')
os.makedirs(PCA_FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load compute outputs
# ---------------------------------------------------------------------------
print('Loading compute outputs ...')
pop_filename     = (f'{pca_dir}/pop_dict_b{binsize}_k{kernel_width}'
                    f'_merged{len(monkey_name_map)}.p')
monkey_dict_file = f'{pca_dir}/allMonkeyDict.p'
event_idx_file   = f'{pca_dir}/event_marker_idx.p'

with open(pop_filename,     'rb') as f: merged_pop_dict  = pkl.load(f)
with open(monkey_dict_file, 'rb') as f: monkeys          = pkl.load(f)
with open(event_idx_file,   'rb') as f: event_marker_idx = pkl.load(f)
print('Loaded.')

cortex_map      = merged_pop_dict['cortex_map']
merged_pop_sdf  = merged_pop_dict['sdf']
condition_map   = merged_pop_dict['condition_map']
condition_order = merged_pop_dict['conditions']

epoch_shapes = {'Cue': 'o', 'Grasp On': '^'}
event_shapes = {'Reach': 's', 'Grasp Off': 'v'}

legend_elements = (
    [Line2D([0], [0], color='w', marker=epoch_shapes[e],
            markerfacecolor='k', label=e) for e in epoch_shapes]
    + [Line2D([0], [0], color='w', marker=event_shapes[e],
              markerfacecolor='k', label=e, alpha=0.4) for e in event_shapes]
)

# ---------------------------------------------------------------------------
# Velocity figure
# ---------------------------------------------------------------------------
individual_plots = False

vel_fig, vel_axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

for cort_idx, cortex in enumerate(cortex_map.keys()):
    if not individual_plots:
        break
    monkey_obj = monkeys['All']
    cond_list  = list(condition_map[cortex].keys())
    vel_ax     = vel_axs[cort_idx]
    for cond in cond_list:
        cond_sdf  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]['sdf']
        V_full    = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['V']
        cond_proj = project_sdf(cond_sdf, V_full)
        cond_vel  = velocity(cond_proj, dims=3, dt=binsize)
        x = np.arange(cond_vel.shape[0]) * binsize
        vel_ax.plot(x, cond_vel / cond_vel.max(),                      color='tab:red')
        vel_ax.plot(x, cond_sdf[:, :-1].mean(dim=0) /
                       cond_sdf[:, :-1].mean(dim=0).max(),             color='tab:blue')
    for epoch in epochs:
        vel_ax.axvline(epoch_window_map[epoch]['time'], color='black')
        vel_ax.axvline(epoch_window_map[epoch]['time']
                       + epoch_window_map[epoch]['window'][1], color='gray')
    vel_ax.set_title(cortex)
    vel_ax.set_xlabel('Time (ms)')
    vel_ax.set_ylabel('Relative Magnitude')

legend_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                Line2D([0], [0], color='tab:red',  lw=4)]
vel_axs[0].legend(legend_lines, ['Mean SDF', 'Velocity'])
vel_fig.tight_layout()
vel_fig.savefig(f'{PCA_FIG_DIR}/PCVelocity.png', dpi=DPI_DRAFT, bbox_inches='tight')
plt.close(vel_fig)
print('Saved PCVelocity.png')

# ---------------------------------------------------------------------------
# Per-cortex × per-monkey figure
# ---------------------------------------------------------------------------
for cort_idx, cortex in enumerate(cortex_map.keys()):
    if not individual_plots:
        break
    cond_list = list(condition_map[cortex].keys())
    n_conds   = len(cond_list)

    for m_name, monkey_obj in monkeys.items():
        if cortex not in monkey_obj.cortices:
            continue
        cortex_neurons = monkey_obj.cortices[cortex]['neurons']

        fig      = plt.figure(figsize=(7, 8))
        ax2      = fig.add_subplot(2, 2, 1)
        ax3      = fig.add_subplot(2, 2, 2, projection='3d')
        ax_var   = fig.add_subplot(2, 2, 3)
        ax_angle = fig.add_subplot(2, 2, 4)

        V_full = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['V']
        S_full = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['S']
        pcs_plotted = min(cortex_neurons, 20)

        for cond in cond_list:
            cond_sdf  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]['sdf']
            cond_svd  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]
            V_cond    = cond_svd['V']
            cond_cov  = cond_svd['cov']
            cond_proj = project_sdf(cond_sdf, V_full)

            # Variance explained
            cond_var = torch.cumsum(
                torch.diagonal(V_cond.T @ cond_cov @ V_cond), dim=0)
            ax_var.scatter(np.arange(1, pcs_plotted + 1),
                           (cond_var / cond_var.max())[:pcs_plotted],
                           label=cond[:6])

            # Trajectories
            x, y, z = cond_proj[:, :3].T
            ax3.plot(x, y, z, label=cond)
            ax2.plot(x, y)
            for epoch in epochs:
                new_epoch = 'Grasp On' if epoch == 'Grasp' else epoch
                ei        = epoch_window_map[epoch]['time'] // binsize
                ax3.scatter(x[ei], y[ei], z[ei],
                            marker=epoch_shapes[new_epoch], c='k')
                ax2.scatter(x[ei], y[ei],
                            marker=epoch_shapes[new_epoch], c='k')
            for event in event_marker_idx:
                if event not in event_map:
                    continue
                mi = event_marker_idx[event]['mean'][0]
                em_x, em_y, em_z = cond_proj[mi, :3]
                if event in event_map_aligned:
                    # Aligned event: epoch marker shape, moderate alpha
                    mkr, alpha, size = epoch_shapes[event_map[event]], 0.4, 100
                else:
                    # Unaligned event (trialReachOn, trialGraspOff): event marker shape, very light
                    mkr, alpha, size = event_shapes[event_map[event]], 0.15, 100
                ax3.scatter(em_x.item(), em_y.item(), em_z.item(),
                            marker=mkr, c='k', alpha=alpha, s=size)
                ax2.scatter(em_x.item(), em_y.item(),
                            marker=mkr, c='k', alpha=alpha, s=size)

        # Full-population variance
        var     = S_full.square()
        var_sum = torch.cumsum(var, 0)
        ax_var.scatter(range(1, pcs_plotted + 1),
                       (var_sum / var_sum[-1])[:pcs_plotted], label='All', color='k')
        ax_var.set_xlabel('PC index'); ax_var.set_ylabel('Variance Explained')
        ax_var.set_ylim([0, 1.05]); ax_var.legend(ncol=2, loc='lower right')

        # Principal angle heatmap with mean ± SD in each cell
        angle_file = os.path.join(
            pca_dir, f'AngleMatrix_{cortex}_monkey{m_name}_PCdims{N_DIMS}.p'
        )
        if os.path.exists(angle_file):
            with open(angle_file, 'rb') as f:
                angle_data = pkl.load(f)
            mean_mat = angle_data['mean']
            std_mat  = angle_data['std']
            im = ax_angle.pcolor(
                mean_mat, cmap=mpl.colormaps['magma_r'],
                norm=colors.Normalize(vmin=0.4, vmax=1, clip=True))
            div = make_axes_locatable(ax_angle)
            cax = div.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            bbox_props = dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.1)
            for ci in range(n_conds):
                for cj in range(n_conds):
                    col = 'yellow' if ci == cj else 'white'
                    ax_angle.text(
                        x=cj + 0.5, y=ci + 0.5,
                        s=f'{mean_mat[ci, cj]:.3f}\n±{std_mat[ci, cj]:.3f}',
                        color=col, va='center', ha='center',
                        fontsize=ANNOT_SMALL, bbox=bbox_props,
                    )
            ax_angle.set_yticks(np.arange(n_conds) + .5, [c[:3] for c in cond_list])
            ax_angle.set_xticks(np.arange(n_conds) + .5, [c[:3] for c in cond_list])
            ax_angle.vlines(np.arange(n_conds)[1:], 0, n_conds, colors='w')
            ax_angle.hlines(np.arange(n_conds)[1:], 0, n_conds, colors='w')
            ax_angle.invert_yaxis()
            ax_angle.set_title('Principal Angles (mean ± SD)')

        ax3.set_xlabel('PC 1'); ax3.set_ylabel('PC 2')#; ax3.set_zlabel('PC 3')
        ax2.set_xlabel('PC 1'); ax2.set_ylabel('PC 2')
        ax3.legend(handles=legend_elements, ncols=2,
                   loc='upper center', bbox_to_anchor=(.5, -.1))
        fig.tight_layout()

        fig_dir = os.path.join(PCA_FIG_DIR, f'Monkey{m_name}')
        os.makedirs(fig_dir, exist_ok=True)
        out = f'{fig_dir}/PCTraj3D_{cortex}_monkey{m_name}_PCdims{N_DIMS}.png'
        fig.savefig(out, dpi=DPI_DRAFT, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out}')

# ---------------------------------------------------------------------------
# Joint monkey PC trajectory figure
# ---------------------------------------------------------------------------
color_map   = {'R': 'tab:red', 'G': 'tab:green', 'Y': 'tab:orange', 'B': 'tab:blue'}
base_monkey = monkeys['All']
pc_fig, pc_axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

# Pre-compute all projections into the joint PC space
all_projs = project_onto_joint_space(monkeys, cortex_map)

for c_idx, cortex in enumerate(cortex_map.keys()):
    reg_ax   = pc_axs[c_idx]
    handles, labels = [], []
    new_handle = True
    for condition in base_monkey.condition_map[cortex]:
        proj     = all_projs['All'][cortex][condition]
        x, y, _  = proj[:, :3].T
        if new_handle:
            line, = reg_ax.plot(x, y, color='black')
            handles.append(line); labels.append('All'); new_handle = False
        else:
            reg_ax.plot(x, y, color='black')
    reg_ax.set_title(cortex); reg_ax.set_xlabel('PC 1')
    if c_idx == 0:
        reg_ax.set_ylabel('PC 2')

    for m_name, monkey_obj in monkeys.items():
        if m_name == 'All':
            continue
        if cortex not in all_projs[m_name]:
            continue
        if monkey_obj.cortices[cortex]['neurons'] <= 5:
            continue
        new_handle = True
        for condition in base_monkey.condition_map[cortex]:
            if condition not in all_projs[m_name][cortex]:
                continue
            proj     = all_projs[m_name][cortex][condition]
            x, y, _ = proj[:, :3].T
            if new_handle:
                line, = reg_ax.plot(x, y, label=m_name,
                                    color=color_map[m_name], alpha=0.5)
                handles.append(line); labels.append(m_name); new_handle = False
            else:
                reg_ax.plot(x, y, color=color_map[m_name], alpha=0.5)

pc_fig.tight_layout()
pc_fig.savefig(f'{PCA_FIG_DIR}/Joint_PCTrajectories.png', dpi=DPI_FINAL, bbox_inches='tight')
plt.close(pc_fig)
print('Saved Joint_PCTrajectories.png')

# ---------------------------------------------------------------------------
# Combined figure — all regions, 'All' monkey, width=11
#
# Layout (2 rows × 6 data cols + 1 narrow colorbar col):
#   Row 0:  2D traj  | 3D traj  || 2D traj  | 3D traj  || 2D traj  | 3D traj  | (cbar)
#   Row 1:  var expl | PA heat  || var expl  | PA heat  || var expl | PA heat  | (cbar)
#            ←── M1 ──→               ←── PMd ──→              ←── PMv ──→
#
# Fixes applied:
#   1. Region labels centered over each pair via figure-coordinate bbox
#   2. Trajectory legend placed between rows; condition legend below bottom row
#   3. Heatmap annotation fontsize = ANNOT_SMALL
#   4. Single shared colorbar in dedicated narrow 7th column, label Cos(θ₁)
#   5. Panel labels use GridSpec cell bbox (figure coords) → consistent height
#      for _i/_ii; _iv label shifted left to avoid crowding heatmap title
#   6. All fonts reduced by 1 pt
# ---------------------------------------------------------------------------
REGION_LABELS  = ['A', 'B', 'C']              # letter prefix per cortex
PANEL_SUBLABEL = [['i', 'ii'], ['iii', 'iv']] # [row][col within region]

m_name     = 'All'
monkey_obj = monkeys[m_name]
cortex_list = list(cortex_map.keys())

# Shared colormap / norm for all heatmaps → single colorbar
angle_norm = colors.Normalize(vmin=0.4, vmax=1, clip=True)
angle_cmap = mpl.colormaps['magma_r']

fig = plt.figure(figsize=(FIGURE_WIDTH, 5))

# 9 columns: [M1_i, M1_ii, SPACE, PMd_i, PMd_ii, SPACE, PMv_i, PMv_ii, cbar]
# Spacer cols (width 0.05) + wspace=0.25 create larger inter-region gaps;
# wspace=0.25 alone sets the smaller intra-region gap (i↔ii, iii↔iv).
gs = GridSpec(
    2, 9,
    figure=fig,
    width_ratios=[1, 1, 0.08, 1, 1, 0.08, 1, 1, 0.06],
    hspace=0.32,
    wspace=0.28,
)
# Reserve margins: bottom reduced now that condition legend sits between rows
fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.12)

# Colorbar placeholder — will be repositioned tight against c_iv after loop
cbar_ax = fig.add_subplot(gs[1, 8])

# Will be populated from the last region's variance axis
cond_handles_fig = None
cond_labels_fig  = None

for cort_idx, cortex in enumerate(cortex_list):
    if cortex not in monkey_obj.cortices:
        continue

    cond_list      = list(condition_map[cortex].keys())
    n_conds        = len(cond_list)
    cortex_neurons = monkey_obj.cortices[cortex]['neurons']
    pcs_plotted    = min(cortex_neurons, 20)

    V_full = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['V']
    S_full = monkey_obj.cortices[cortex]['SVD']['Merged']['Full']['S']

    col_base = cort_idx * 3   # cols 0, 3, 6 — each region skips a spacer col

    ax2      = fig.add_subplot(gs[0, col_base])
    ax3      = fig.add_subplot(gs[0, col_base + 1], projection='3d')
    ax_var   = fig.add_subplot(gs[1, col_base])
    ax_angle = fig.add_subplot(gs[1, col_base + 1])

    # Shift each 3D axes upward within its GridSpec cell, freeing a strip at
    # the bottom where the shared legend will sit (under bii only).
    _cell = gs[0, col_base + 1].get_position(fig)
    _shift = 0.04  # fraction of figure height
    ax3.set_position([_cell.x0, _cell.y0 + _shift,
                      _cell.width, _cell.height - _shift])

    prefix = REGION_LABELS[cort_idx]

    # ── Panel labels via GridSpec cell bbox (figure coordinates) ─────────────
    # Using the cell's figure-coordinate bbox guarantees that _i and _ii in
    # the same row share the same y, and that _iv is offset left of the cell
    # (not left of the heatmap's own origin, which can crowd the title).
    gs_cells = [[gs[0, col_base],     gs[0, col_base + 1]],
                [gs[1, col_base],     gs[1, col_base + 1]]]
    for r in range(2):
        for c in range(2):
            cell_bbox = gs_cells[r][c].get_position(fig)
            lbl = (r'$\mathbf{' + prefix + r'}_{\mathbf{'
                   + PANEL_SUBLABEL[r][c] + r'}}$')
            # All labels shifted left uniformly (matching _iv treatment)
            x_off = -0.022
            fig.text(cell_bbox.x0 + x_off, cell_bbox.y1 + 0.006,
                     lbl, ha='left', va='bottom',
                     fontsize=PANEL_LABEL_SIZE, fontweight='bold', clip_on=False)

    # ── Region title centered above the two top panels ───────────────────────
    region_bbox = gs[0, col_base:col_base + 2].get_position(fig)
    title_x = (region_bbox.x0 + region_bbox.x1) / 2
    title_y = region_bbox.y1 + 0.055
    fig.text(title_x, title_y, cortex,
             ha='center', va='bottom', fontsize=REGION_TITLE_SIZE, fontweight='bold')

    # ── Trajectories & event markers ─────────────────────────────────────────
    # Store full (x, y) arrays so we can find each trajectory's point of
    # maximum separation from the others for label placement.
    ax2_traj_data = []   # [(x_arr, y_arr, color, short_label), ...]
    for cond in cond_list:
        cond_sdf  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]['sdf']
        cond_svd  = monkey_obj.cortices[cortex]['SVD']['Merged'][cond]
        V_cond    = cond_svd['V']
        cond_cov  = cond_svd['cov']
        cond_proj = project_sdf(cond_sdf, V_full)

        # Variance explained
        cond_var = torch.cumsum(
            torch.diagonal(V_cond.T @ cond_cov @ V_cond), dim=0)
        ax_var.scatter(np.arange(1, pcs_plotted + 1),
                       (cond_var / cond_var.max())[:pcs_plotted],
                       label=cond[:6], s=18)

        x, y, z = cond_proj[:, :3].T
        ax3.plot(x, y, z, label=cond)
        line2d, = ax2.plot(x, y)
        ax2_traj_data.append((x.numpy(), y.numpy(),
                               line2d.get_color(), cond[:3]))

        for epoch in epochs:
            new_epoch = 'Grasp On' if epoch == 'Grasp' else epoch
            ei = epoch_window_map[epoch]['time'] // binsize
            ax3.scatter(x[ei], y[ei], z[ei],
                        marker=epoch_shapes[new_epoch], c='k', s=18)
            ax2.scatter(x[ei], y[ei],
                        marker=epoch_shapes[new_epoch], c='k', s=18)

        for event in event_marker_idx:
            if event not in event_map:
                continue
            mi = event_marker_idx[event]['mean'][0]
            em_x, em_y, em_z = cond_proj[mi, :3]
            if event in event_map_aligned:
                mkr, alpha, size = epoch_shapes[event_map[event]], 0.7, 25
            else:
                mkr, alpha, size = event_shapes[event_map[event]], 0.3, 25
            ax3.scatter(em_x.item(), em_y.item(), em_z.item(),
                        marker=mkr, c='k', alpha=alpha, s=size)
            ax2.scatter(em_x.item(), em_y.item(),
                        marker=mkr, c='k', alpha=alpha, s=size)

    # ── Full-population variance ──────────────────────────────────────────────
    var     = S_full.square()
    var_sum = torch.cumsum(var, 0)
    ax_var.scatter(range(1, pcs_plotted + 1),
                   (var_sum / var_sum[-1])[:pcs_plotted],
                   label='All', color='k', s=18)
    ax_var.set_xlabel('PC index', fontsize=LABEL_SIZE)
    ax_var.set_ylim([0, 1.05])
    ax_var.tick_params(labelsize=TICK_SIZE, pad=0)
    # Variance Explained ylabel only on leftmost column
    if cort_idx == 0:
        ax_var.set_ylabel('Variance Explained', fontsize=LABEL_SIZE)
    # ── Adaptive line labels on ax2 ──────────────────────────────────────────
    # For each trajectory find the time index where that trajectory passes
    # through its most isolated region — i.e. the point whose minimum distance
    # to *any* point on *any* other trajectory (across all time steps) is
    # greatest. This finds open corners / gaps rather than the moment of peak
    # synchronous spread, and naturally seats labels in unoccupied space.
    xy_arrays = np.stack(
        [np.column_stack([xd, yd]) for xd, yd, _, _ in ax2_traj_data], axis=0
    )  # shape (n_conds, T, 2)

    ax2.margins(0.25)  # give labels at the periphery room inside the frame
    ax2_texts = []
    for i, (xi, yi, lc, lbl) in enumerate(ax2_traj_data):
        pts_i = xy_arrays[i]                              # (T, 2)
        # All points from other trajectories, flattened across time
        other_pts = np.delete(xy_arrays, i, axis=0).reshape(-1, 2)  # ((n-1)*T, 2)
        # For each point on traj i, minimum distance to any point on any other traj
        dists = np.linalg.norm(
            pts_i[:, None, :] - other_pts[None, :, :], axis=2
        )  # (T, (n-1)*T)
        min_dist = dists.min(axis=1)                      # (T,)
        best_t = int(np.argmax(min_dist))
        lx, ly = xi[best_t], yi[best_t]
        # clip_on=False: text whose box slightly crosses the axes edge still
        # renders fully. adjust_text's ensure_inside_axes keeps anchors within
        # the (now padded) frame, so labels won't wander far outside.
        t = ax2.text(lx, ly, lbl, fontsize=ANNOT_SMALL, color=lc,
                     fontweight='bold', ha='center', va='center',
                     clip_on=False)
        ax2_texts.append(t)

    # Fine-tune with adjustText to resolve any residual overlaps.
    ax2_lines = [obj for obj in ax2.get_lines() if len(obj.get_xdata()) > 1]
    adjust_text(
        ax2_texts,
        ax=ax2,
        objects=ax2_lines,
        expand=(1.3, 1.5),
        force_text=(0.4, 0.6),
        force_objects=(0.3, 0.5),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        min_arrow_len=4,
        ensure_inside_axes=True,
    )

    # ── 2D / 3D axis labels ───────────────────────────────────────────────────
    ax2.set_xlabel('PC 1', fontsize=LABEL_SIZE, labelpad=2)
    ax2.set_ylabel('PC 2', fontsize=LABEL_SIZE, labelpad=2)
    ax2.tick_params(labelsize=TICK_SIZE, pad=1)
    ax3.set_xlabel('PC 1', fontsize=ANNOT_SMALL, labelpad=-10)
    ax3.set_ylabel('PC 2', fontsize=ANNOT_SMALL, labelpad=-10)
    # ax3.set_zlabel('PC 3', fontsize=ANNOT_SMALL, labelpad=-10)  # omitted — adds clutter

    # Remove filled panes — clears dead grey/white space around the trajectories
    for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('lightgray')

    # Compress z so the cube doesn't waste height; widen x to match PC1 range
    ax3.set_box_aspect([1.3, 1.0, 0.75])

    # Viewing angle: low elevation shows the z-separation; azimuth chosen so
    # PC1 runs left-right and PC3 rises in the foreground.
    # Adjust per-region below if needed.
    view_angles = {'M1': (20, -55), 'PMd': (22, -50), 'PMv': (20, -60)}
    elev, azim = view_angles.get(cortex, (20, -55))
    ax3.view_init(elev=elev, azim=azim)


    # 3D tick labels: negative pad pulls numbers flush against the tick marks.
    ax3.tick_params(labelsize=ANNOT_SMALL, pad=-6)

    # ── Principal angle heatmap — shared cmap/norm, no per-panel colorbar ────
    angle_file = os.path.join(
        pca_dir, f'AngleMatrix_{cortex}_monkey{m_name}_PCdims{N_DIMS}.p')
    if os.path.exists(angle_file):
        with open(angle_file, 'rb') as f:
            angle_data = pkl.load(f)
        mean_mat = angle_data['mean']
        std_mat  = angle_data['std']
        ax_angle.pcolor(mean_mat, cmap=angle_cmap, norm=angle_norm)
        bbox_props = dict(facecolor='black', alpha=0.4, edgecolor='none', pad=0.1)
        for ci in range(n_conds):
            for cj in range(n_conds):
                col = 'yellow' if ci == cj else 'white'
                ax_angle.text(
                    x=cj + 0.5, y=ci + 0.5,
                    s=f'{mean_mat[ci, cj]:.2f}\n±{std_mat[ci, cj]:.2f}',
                    color=col, va='center', ha='center',
                    fontsize=ANNOT_SMALL, bbox=bbox_props,
                )
        ax_angle.set_yticks(np.arange(n_conds) + .5,
                            [c[:3] for c in cond_list], fontsize=TICK_SIZE)
        ax_angle.set_xticks(np.arange(n_conds) + .5,
                            [c[:3] for c in cond_list], fontsize=TICK_SIZE)
        ax_angle.tick_params(axis='both', pad=1, length=0)
        ax_angle.vlines(np.arange(n_conds)[1:], 0, n_conds, colors='w')
        ax_angle.hlines(np.arange(n_conds)[1:], 0, n_conds, colors='w')
        ax_angle.invert_yaxis()
        ax_angle.set_title('Principal Angles', fontsize=LABEL_SIZE)

# ── Single shared colorbar for all heatmaps ──────────────────────────────────
sm = mpl.cm.ScalarMappable(cmap=angle_cmap, norm=angle_norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label(r'$\cos(\theta_1)$', fontsize=LABEL_SIZE)
cbar.ax.tick_params(labelsize=ANNOT_SMALL, pad=1)

# Reposition colorbar tight against the c_iv heatmap (col_base + 1 = 7 for PMv)
last_angle_bbox = gs[1, col_base + 1].get_position(fig)
cbar_pos = cbar_ax.get_position()
cbar_ax.set_position([last_angle_bbox.x1 + 0.010,
                      cbar_pos.y0, cbar_pos.width, cbar_pos.height])

# ── Epoch/event marker legend — centred below bii (PMd 3D panel) ─────────────
# PMd 3D panel is at GridSpec column 4 (col_base=3, col_base+1=4).
# We anchor fig.legend just below that cell's bottom edge, in the strip freed
# by shifting ax3 upward.
_pmd_cell = gs[0, 4].get_position(fig)
fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(_pmd_cell.x0 + _pmd_cell.width / 2, _pmd_cell.y0 + 0.05),
    ncols=2,
    fontsize=ANNOT_SMALL,
    frameon=False,
    markerscale=0.9,
    handletextpad=0.4,
    columnspacing=0.8,
)

combined_path = f'{PCA_FIG_DIR}/Combined_PCTrajectories_All.png'
fig.savefig(combined_path, dpi=DPI_DRAFT, bbox_inches='tight')
plt.close(fig)
print(f'Saved {combined_path}')

print('\npca_plot_trajectories.py complete.')
