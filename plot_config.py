"""
plot_config.py
Shared style parameters for all plotting scripts.

Usage in each plotting script:
    from plot_config import *
    from utils.plot_utils import apply_theme
    apply_theme()
"""

import matplotlib as mpl

# ── Figure dimensions ──────────────────────────────────────────────────────────
FIGURE_WIDTH = 11          # inches; heights remain per-figure

# ── Resolution ────────────────────────────────────────────────────────────────
DPI_DRAFT  = 200           # quick iteration / review
DPI_FINAL  = 400           # publication-ready output

# ── Font ──────────────────────────────────────────────────────────────────────
FONT_FAMILY = 'Arial'

# ── Axes ──────────────────────────────────────────────────────────────────────
AXES_LINEWIDTH = 0.4

# ── Type scale ────────────────────────────────────────────────────────────────
PANEL_LABEL_SIZE  = 13     # A / B / C panel labels
REGION_TITLE_SIZE = 16     # M1 / PMd / PMv region headings
LABEL_SIZE        = 11     # axis labels
TICK_SIZE         = 10     # tick labels
ANNOT_SIZE        = 9      # statistical summaries (n counts, r values)
ANNOT_SMALL       = 7      # geometry-constrained text (heatmap cells, bar values, trajectory labels)
TICK_LENGTH       = 4
TICK_WIDTH        = 0.8

# ── Colours ───────────────────────────────────────────────────────────────────
CORTEX_COLORS = {
    'M1':  'tab:blue',
    'PMd': 'tab:orange',
    'PMv': 'tab:green',
}


def apply_rcparams():
    """Apply global rcParams. Call after apply_theme() from utils.plot_utils."""
    mpl.rcParams['font.family']      = FONT_FAMILY
    mpl.rcParams['axes.linewidth']   = AXES_LINEWIDTH
    mpl.rcParams['xtick.labelsize']  = TICK_SIZE
    mpl.rcParams['ytick.labelsize']  = TICK_SIZE
    mpl.rcParams['axes.labelsize']   = LABEL_SIZE
    mpl.rcParams['legend.fontsize']  = ANNOT_SMALL
    mpl.rcParams['xtick.major.size'] = TICK_LENGTH
    mpl.rcParams['ytick.major.size'] = TICK_LENGTH
    mpl.rcParams['xtick.major.width']= TICK_WIDTH
    mpl.rcParams['ytick.major.width']= TICK_WIDTH
    mpl.rcParams['xtick.direction']  = 'out'
    mpl.rcParams['ytick.direction']  = 'out'
