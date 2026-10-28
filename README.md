# NHP Data Analysis

Analysis pipeline for neural population dynamics in non-human primates (NHP), examining activity in primary motor cortex (M1), dorsal premotor cortex (PMd), and ventral premotor cortex (PMv) during a reach-to-grasp task.

---

## Overview

Electrophysiological recordings from four monkeys (Green, Red, Yellow, Blue) were collected during a reach-to-grasp task in which the object orientation (vertical / horizontal) and reach hand (contra- / ipsilateral) were varied. The pipeline preprocesses trial-aligned spike data, computes spike density functions, and runs a suite of population-level analyses including modulation classification, selectivity scoring, dimensionality reduction (PCA), and clustering.

---

## Repository Structure

```
NHP Data Analysis/
├── config.py              # Single source of truth for all shared parameters
├── plot_config.py         # Figure styling constants and rcParams setup
├── pipeline/              # Data preparation and preprocessing
│   ├── data_prep_merge.py
│   └── data_process_merge.py
├── analysis/              # Core statistical analyses
│   ├── modulation_analysis.py
│   ├── selectivity_analysis.py
│   ├── pca_compute.py
│   └── clustering_analysis.py
├── plotting/              # Figure generation
│   ├── plotting_population.py
│   ├── plotting_modulation.py
│   ├── plotting_selectivity.py
│   ├── rasters_classes.py
│   ├── pca_plot_neurons.py
│   ├── pca_plot_trajectories.py
│   ├── pca_plot_angles.py
│   └── pca_plot_coupled.py
├── utils/                 # Shared utility functions
├── Data/                  # Raw and processed data (not tracked)
│   ├── Sorted_Inactivation/
│   └── Processed/
└── Results/               # Analysis outputs (not tracked)
    ├── Population/
    ├── Modulation/
    ├── Selectivity/
    ├── Clustering/
    ├── PCA/
    └── DMD/
```

---

## Task and Epochs

Monkeys performed a delayed reach-to-grasp task. Three epochs are used in the merged analysis:

| Epoch    | Alignment event    | Window (ms)  |
|----------|--------------------|--------------|
| Pre-Cue  | trialRewardDrop    | −300 to −100 |
| Cue      | trialRewardDrop    | 0 to 200     |
| Grasp    | trialGraspOn       | −100 to 500  |

---

## Signal Processing

Spike trains are binned and convolved with a Gaussian kernel to produce spike density functions (SDFs).

| Parameter    | Value |
|--------------|-------|
| Bin size     | 5 ms  |
| Kernel width | 25 ms |
| PCA dims     | 5     |

---

## Analysis Pipeline

Scripts must be run in order. The full pipeline is automated in `run_pipeline.sh` (project root).

### 1. Preprocessing — `pipeline/`

**`data_prep_merge.py`** — Loads raw MATLAB files and sorting notes, splits spike trains into trial-aligned segments, applies event-duration quality filters, and writes per-session pickles to `Data/Processed/`. Handles two legacy MATLAB format variants (Format A: monkeys Y/B; Format B: monkeys G/R).

**`data_process_merge.py`** — Reads the per-session pickles, builds PSTHs and Gaussian SDFs across all neurons, applies a mean firing-rate threshold, and saves the merged population dictionary (`Data/Processed/Summary/`) along with per-region neuron counts.

### 2. Analysis — `analysis/`

**`modulation_analysis.py`** — Classifies each neuron's modulation type (contra, ipsi, both, none) for Cue and Grasp epochs using FDR-corrected t-tests (α = 0.05). Saves `MergedModDataFrame.p`.

**`selectivity_analysis.py`** — Computes hand (C − I) and orientation (V − H) selectivity scores per neuron using AUC-based methods. Runs a two-way ANOVA (hand × orientation). Saves scores and ANOVA results.

**`pca_compute.py`** — Performs SVD on the merged population SDF matrix for each cortex and monkey. Computes principal angles between condition subspaces using a subsampling approach (N=5 random draws of 80% of trials per condition), yielding mean ± SD of the leading angle across subsets. Saves Monkey objects (`allMonkeyDict.p`) and per-region angle matrices.

**`clustering_analysis.py`** — Clusters neurons based on their activity profiles.

### 3. Plotting — `plotting/`

**`plotting_population.py`** — Population summary figures (neuron counts, recording yield).

**`plotting_modulation.py`** — Bar charts and proportions of modulation types across cortices and epochs.

**`plotting_selectivity.py`** — Joint scatter / ECDF / ANOVA figure showing hand and orientation selectivity (Figure 2).

**`rasters_classes.py`** — Trial-averaged rasters sorted by PC contribution weight.

**`pca_plot_neurons.py`** — Per-neuron PC contribution weight bar charts and CDFs, coloured by monkey and by modulation type.

**`pca_plot_trajectories.py`** — Population trajectory plots in PC space across task epochs.

**`pca_plot_angles.py`** — Principal angle analyses between cortices and conditions.

**`pca_plot_coupled.py`** — Coupled subspace analyses.

---

## Setup

### 1. Install Conda

If you don't already have conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).

### 2. Create the environment

From the project root, run this once before anything else:

```bash
conda env create -f environment.yml
```

This installs all required dependencies into a conda environment called `NHP_Data_Analysis`. You only need to do this once.

### 3. (Maintainers) Updating the environment file

If you add or remove packages, re-export the environment so others can stay in sync:

```bash
conda env export --from-history > environment.yml
```

Commit `environment.yml` to the repository so others can recreate it.

---

## Running the Pipeline

### Full pipeline

```bash
# From the project root
bash run_pipeline.sh
```

### Individual scripts

> **Note:** The `NHP_Data_Analysis` conda environment must exist before running individual scripts. If you haven't already, run `conda env create -f environment.yml` first (see [Setup](#setup) above). The full `run_pipeline.sh` handles this automatically.

```bash
conda run -n NHP_Data_Analysis python pipeline/data_prep_merge.py
conda run -n NHP_Data_Analysis python pipeline/data_process_merge.py
conda run -n NHP_Data_Analysis python analysis/modulation_analysis.py
conda run -n NHP_Data_Analysis python analysis/selectivity_analysis.py
conda run -n NHP_Data_Analysis python analysis/pca_compute.py
conda run -n NHP_Data_Analysis python analysis/clustering_analysis.py
```

---

## Configuration

All shared parameters are defined in `config.py` — import from there rather than re-declaring constants in individual scripts:

```python
from config import BIN_SIZE, KERNEL_WIDTH, MONKEY_NAME_MAP, EPOCH_WINDOW_MAP_MERGE
```

Figure styling constants (font sizes, DPI, rcParams) are defined in `plot_config.py`:

```python
from plot_config import DPI_DRAFT, PANEL_LABEL_SIZE, apply_rcparams
```

---

## Environment

Key dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `torch` (SVD tensor operations), `pingouin` (ANOVA), `adjustText`, `DSA`.

---

## Monkeys

| Key    | Name   | M1 Hemisphere |
|--------|--------|---------------|
| G      | Green  | Right         |
| R      | Red    | Right         |
| Y      | Yellow | Right         |
| B      | Blue   | Right         |
