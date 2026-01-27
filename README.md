# Max-Cut via Tabu Search (Metaheuristics Project)

## Overview
This project solves the **Max-Cut problem**, a classical **NP-complete combinatorial optimization problem**, using **Tabu Search**, a local-search metaheuristic with memory.  
The goal is to find a partition of the vertices of a graph into two sets such that the total weight of edges crossing the partition is maximized.

The implementation is written in **Python** and includes:
- a reproducible random instance generator,
- a full Tabu Search implementation with incremental gain updates,
- baseline heuristics (random search and greedy hill climbing),
- a systematic experimental evaluation with plots and saved results.

---

## Problem Definition: Max-Cut
Given a (weighted) graph $G = (V, E)$ with edge weights $w(u,v) \ge 0$, find a partition of the vertices into two disjoint sets $A$ and $B$ that maximizes:

$$
\text{Cut}(A,B) = \sum_{(u,v)\in E} w(u,v)\,\mathbf{1}[u \in A, v \in B]
$$

### Encoding
A candidate solution is encoded as a **bitstring**:

$$
x \in \{0,1\}^{|V|}
$$

- `x[v] = 0` → vertex $v$ is in set $A$
- `x[v] = 1` → vertex $v$ is in set $B$

This encoding is symmetric: flipping all bits yields the same cut.

---

## Metaheuristic: Tabu Search

### Neighborhood
- A move consists of **flipping one vertex** to the opposite side.
- The neighborhood of a solution contains exactly $|V|$ such moves.

### Objective Gain (Δ-evaluation)
For each vertex $v$, the gain of flipping it is:

$$
\Delta(v) = \sum_{u \in N(v)} w(v,u)
\begin{cases}
+1 & \text{if } x_u = x_v \\
-1 & \text{if } x_u \ne x_v
\end{cases}
$$

This allows **incremental updates** in $O(\deg(v))$ time instead of recomputing the full cut value.

### Tabu Mechanism
- Recently flipped vertices are marked **tabu** for a fixed or randomized tenure.
- A tabu move is allowed if it satisfies an **aspiration criterion**:
  - it yields a solution at least as good as the best seen so far.

### Termination
- Maximum number of iterations, or
- no improvement for a fixed number of iterations (“patience”).

---

## Baseline Methods
To contextualize Tabu Search performance, two baselines are included:

1. **Random Search**  
   Samples random partitions and keeps the best result.

2. **Greedy Hill Climbing**  
   Iteratively applies the best improving single-vertex flip until no improvement exists.

These baselines help demonstrate the added value of tabu memory.

---

## Experimental Setup

### Instance Generation
- Random Erdős–Rényi graphs $G(n,p)$
- Integer edge weights sampled uniformly in $[1,10]$
- Fully reproducible via controlled random seeds

### Parameters Explored
- Number of vertices: `n ∈ {50, 100, 200}`
- Edge probability: `p ∈ {0.1, 0.3}`
- Tabu tenure: `{5, 10, 20}`
- 10 independent runs per configuration

### Metrics Recorded
- Best cut value found
- Final cut value
- Runtime
- Number of iterations
- Graph size statistics (edges, total weight)

All results are saved to `results_maxcut_tabu.csv`.

---

## Output and Visualizations
The script automatically generates and saves:
- **Mean best-fitness curve** with variability band
- **Boxplots** comparing methods
- **Runtime scaling** vs graph size
- **Tabu performance vs tenure**

All plots are saved as PNG files with fixed DPI for report inclusion.

---

## How to Run

### Requirements
```bash
pip install -r requirements.txt
```

### Run the script
```bash
python maxcut_tabu.py
```

### Outputs
- `results_maxcut_tabu.csv` (experiment results with graph statistics)
- `tabu_mean_curve.png`
- `best_F_boxplot.png`
- `runtime_vs_n.png`
- `tabu_performance_vs_tenure.png`
