#!/usr/bin/env bash
# run_figures.sh — Generate all figures in dependency order.
#
# Analysis scripts (run_analysis.sh) must have been run first so that the
# Results/ pickle files required by each plotting script are available.
#
# Usage (from the project root):
#   bash run_figures.sh

set -e

# Initialize conda if not already done (e.g. when run directly, not via run_pipeline.sh)
if [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    for candidate in \
        "/c/ProgramData/anaconda3" \
        "/c/ProgramData/miniconda3" \
        "$HOME/anaconda3" \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/mambaforge"
    do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi

PYTHON="conda run -n NHP_Data_Analysis python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    $PYTHON "$ROOT/$1"
}

run plotting/plotting_population.py
run plotting/plotting_modulation.py
run plotting/plotting_selectivity.py
run plotting/rasters_classes.py
run plotting/pca_plot_neurons.py
run plotting/pca_plot_trajectories.py
run plotting/pca_plot_angles.py
run plotting/pca_plot_coupled.py

echo ""
echo "=========================================="
echo "  Figures complete."
echo "=========================================="
