#!/usr/bin/env bash
# run_analysis.sh — Run all analysis scripts in dependency order.
#
# Each script has its own load_override flag — set it to True inside the script
# to skip recomputation and load existing results instead.
#
# Usage (from the project root):
#   bash run_analysis.sh

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

run analysis/modulation_analysis.py
run analysis/selectivity_analysis.py
run analysis/pca_compute.py
run analysis/clustering_analysis.py

echo ""
echo "=========================================="
echo "  Analysis complete."
echo "=========================================="
