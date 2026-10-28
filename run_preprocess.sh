#!/usr/bin/env bash
# run_preprocess.sh — Load raw MATLAB files and build the processed data directory.
#
# Step 1 (data_prep_merge.py) only needs to run once per dataset — skip it by
# passing --skip-prep if the per-session pickles in Data/Processed/ already exist.
#
# Usage (from the project root):
#   bash run_preprocess.sh            # run both prep and processing
#   bash run_preprocess.sh --skip-prep  # skip prep, re-run processing only

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
SKIP_PREP=false

for arg in "$@"; do
    case $arg in
        --skip-prep) SKIP_PREP=true ;;
    esac
done

run() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    $PYTHON "$ROOT/$1"
}

if [ "$SKIP_PREP" = false ]; then
    run pipeline/data_prep_merge.py
fi

run pipeline/data_process_merge.py

echo ""
echo "=========================================="
echo "  Preprocessing complete."
echo "=========================================="
