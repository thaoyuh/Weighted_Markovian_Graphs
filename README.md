# Random Walks with Traversal Costs

Companion code for:

> **Le, T.P., van der Burg, R., Heidergott, B.F., Lindner, I.D., & Zocca, A.** (2025). *Random Walks with Traversal Costs: Variance-Aware Performance Analysis and Network Optimization.* Working paper.

This repository contains the numerical experiments for the paper. Currently, the **surveillance network application** (Section 6.1) is implemented. The traffic network application (Section 6.2) ...

---

## Overview

The paper introduces a framework for weighted random walks on graphs that explicitly incorporates the moments of edge weights into the performance analysis. The code implements the surveillance application, which optimizes a *surprise index*, a scale-invariant measure of patrol unpredictability:

$$
\mathcal{S}(\mathbf{P}, \mathcal{W}) = \frac{\sqrt{V_{\mathcal{W}}(\mathbf{P})}}{K_{\mathcal{W}}(\mathbf{P})}
$$

where $K_{\mathcal{W}}$ is the weighted Kemeny constant and $V_{\mathcal{W}}$ is the aggregate weighted variance of first-passage times. The optimization finds a transition matrix $\mathbf{P}$ that maximizes $\mathcal{S}$ subject to a hard stationary distribution constraint, using the SPSA algorithm with null-space perturbations from [Franssen et al. (2025)](#references).

The framework supports:
- **Deterministic and stochastic edge weights** (only first two moments required)
- **Uniform and non-uniform target distributions** $\boldsymbol{\mu}$
- **Obstacle configurations** with LP-based feasibility verification
- **Hard stationary distribution constraints** via Dykstra projection onto the null space

---

## Repository Structure

```
├── Main_new.py                 # Main experiment script (publication results)
├── problem_instance.py         # Problem formulation: constraints, projections, objective
├── network_stochastic.py       # Markov chain computations (M, V, K_W, S) with stochastic weights
├── optimization.py             # SPSA solver with null-space perturbations
├── grid_generation.py          # Grid network generation with stochastic weights
├── utils.py                    # Utility functions (edge matrices, row normalization)
├── feasibility.py              # LP-based feasibility checking
├── network.py                  # Markov chain class (deterministic weights)
├── Optimize_K_W.py             # Kemeny constant optimization (baseline comparison)
└── Figures/                    # Output directory for policy visualizations
```

### Key modules

| File | Purpose |
|------|---------|
| `Main_new.py` | Runs all surveillance experiments: 4×4 min-variance/max-surprise, 8×8 with obstacles and non-uniform $\boldsymbol{\mu}$ |
| `problem_instance.py` | Builds the constraint system (row-stochastic + stationary distribution), computes null-space basis, Dykstra projection |
| `network_stochastic.py` | Computes $\pi$, $\pi_{\mathcal{W}}$, fundamental matrix $\mathbf{Z}$, mean FPT matrix $\mathbf{M}$, variance matrix $\mathbf{V}$, $K_{\mathcal{W}}$, $V_{\mathcal{W}}$, and $\mathcal{S}$ |
| `optimization.py` | SPSA with null-space perturbations to maintain hard constraints at every iterate |
| `grid_generation.py` | Generates $n \times n$ grid networks with obstacles, stochastic weights (parameterized by CV), and target distributions |

---

## Prerequisites

- **Python** 3.10+ (tested on 3.12)
- Required packages:

```
numpy
scipy
matplotlib
sympy
```

Install with pip:

```bash
pip install numpy scipy matplotlib sympy
```

Or with conda:

```bash
conda install numpy scipy matplotlib sympy
```

No GPU or special hardware is required. All experiments run on standard hardware (the 8×8 grid with 60 nodes completes in minutes).

---

## Running the Experiments

Run the surveillance network experiments (Section 6.1 of the paper):

```bash
python Main_new.py
```

This will:

1. **4×4 grid, uniform $\boldsymbol{\mu}$, deterministic weights:**
   - Compute the uniform baseline ($\mathcal{S} \approx 1.05$)
   - Minimize variance → Hamiltonian cycle ($\mathcal{S} \approx 0.21$)
   - Maximize surprise index ($\mathcal{S}^* \approx 1.41$)

2. **8×8 grid, non-uniform $\boldsymbol{\mu}$, stochastic weights:**
   - 4 obstacle nodes, 12 priority nodes with 2:1 coverage ratio
   - Feasibility check via LP
   - Maximize surprise index ($\mathcal{S}^* \approx 1.26$)

3. **Generate outputs:**
   - Policy visualization figures (`fig*.png`)
   - Convergence plots (`supp_*.png`)
   - Numerical report (`optimization_report.txt`)

---

## Configuration

Key parameters are set in `Main_new.py`:

```python
SEED = 42                # Random seed for reproducibility
ETA_SMALL = 1e-4         # Minimum transition probability (4×4 grid)
ETA_BIG   = 1e-8         # Minimum transition probability (8×8 grid)
```

### SPSA parameters (in `run_opt`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 8000–25000 | Number of SPSA iterations |
| `a` | 0.0005–0.001 | Step size (fixed) |
| `eta` | 1e-4 or 1e-8 | Lower bound on transition probabilities |

### Obstacle and priority configuration (8×8 grid):

```python
obs8 = [(2, 2), (2, 3), (5, 4), (5, 5)]   # Horizontal obstacle pairs
priority8 = {(r,c): 2.0 for ...}            # 2× weight on obstacle-adjacent nodes
```

---

## Output

### Figures

| File | Description |
|------|-------------|
| `fig1a_uniform.png` | Uniform baseline policy (4×4) |
| `fig1b_min_variance.png` | Min-variance policy / Hamiltonian cycle (4×4) |
| `fig2_max_surprise_det.png` | Max-surprise policy, deterministic weights (4×4) |
| `fig5_8x8_stoch.png` | Max-surprise policy, stochastic weights (8×8) |

### Report

`optimization_report.txt` contains a summary table with $K_{\mathcal{W}}$, $\sqrt{V_{\mathcal{W}}}$, $\mathcal{S}$, and $\|\pi - \hat{\mu}\|$ for all experiments.

---

## References

- **Le, T.P., van der Burg, R., Heidergott, B.F., Lindner, I.D, & Zocca, A.** (2025). Random Walks with Traversal Costs: Variance-Aware Performance Analysis and Network Optimization. Working paper.

---

## License

This project is licensed under the...

---

## Contact

For questions or issues, contact: Thao Le — t.p.t.le@vu.nl
