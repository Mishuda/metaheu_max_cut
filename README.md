# metaheu_max_cut

Tabu Search implementation for the **weighted Max-Cut** problem in Python.

## Contents
- `maxcut_tabu.py`: runnable script with problem setup, instance generator, Tabu Search, baselines, experiments, and plots.
- `maxcut_tabu_search.ipynb`: full, runnable notebook with problem setup, instance generator, Tabu Search, baselines, experiments, and plots.

## Quick start (script)
1) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2) Run the script:
   ```bash
   python maxcut_tabu.py
   ```
3) Outputs:
   - `results_maxcut_tabu.csv` (experiment results with graph statistics)
   - `tabu_mean_curve.png`
   - `best_F_boxplot.png`
   - `runtime_vs_n.png`
   - `tabu_performance_vs_tenure.png`

## Quick start (notebook)
1) Open the notebook.
2) Run cells top-to-bottom.

Dependencies: `numpy`, `networkx`, `pandas`, `matplotlib`.
