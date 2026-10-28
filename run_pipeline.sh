#!/usr/bin/env bash
# run_pipeline.sh — Run the full pipeline end-to-end.
#
# Calls the three sub-scripts in dependency order:
#   1. run_preprocess.sh  — load MATLAB files, build processed data
#   2. run_analysis.sh    — modulation, selectivity, PCA, clustering
#   3. run_figures.sh     — all plotting scripts
#
# Pass --skip-prep to skip data_prep_merge.py (the slow per-session step)
# if the per-session pickles in Data/Processed/ already exist.
#
# Usage (from the project root):
#   bash run_pipeline.sh             # full pipeline including data prep
#   bash run_pipeline.sh --skip-prep # skip prep, re-run processing onward

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize conda — search common install locations
CONDA_BASE=""

# Search candidate paths in order
for candidate in \
    "/c/ProgramData/anaconda3" \
    "/c/ProgramData/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/miniconda3" \
    "$HOME/miniforge3" \
    "$HOME/mambaforge"
do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
export CONDA_BASE

# Create environment if it doesn't exist
if ! conda env list | grep -q "NHP_Data_Analysis"; then
    echo ""
    echo "=========================================="
    echo "  Creating conda environment from environment.yml..."
    echo "=========================================="
    conda env create -f "$ROOT/environment.yml"
fi

echo ""
echo "=========================================="
echo "  Starting full pipeline"
echo "=========================================="

bash "$ROOT/run_preprocess.sh" "$@"
bash "$ROOT/run_analysis.sh"
bash "$ROOT/run_figures.sh"

echo ""
echo "=========================================="
echo "  Full pipeline complete."
echo "=========================================="
