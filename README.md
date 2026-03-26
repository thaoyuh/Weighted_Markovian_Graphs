# Weighted Markovian Graphs

Companion code for:

> **Le, T.P., van der Burg, R., Heidergott, B.F., Lindner, I.D., & Zocca, A.** (2026). *Random Walks with Traversal Costs: Variance-Aware Performance Analysis and Network Optimization.* Working paper.

This repository contains the numerical experiments for both applications in the paper:

- **Surveillance Network** (Section 6.1) — Maximizing patrol unpredictability via the surprise index
- **Traffic Network** (Section 6.2) — Preserving network resilience under sequential edge failures

---

## Overview

The paper introduces a framework for weighted random walks on graphs that explicitly incorporates the moments of edge weights into the performance analysis. Two key quantities are derived in closed form: the weighted Kemeny constant $K_{\mathcal{W}}$ (mean connectivity) and its second-order counterpart $V_{\mathcal{W}}$ (variance of first-passage times). These enable variance-aware network optimization through two complementary applications.

---

## Repository Structure
```
├── README.md
├── Surveillance_Network/           # Section 6.1: Surprise index optimization
│   ├── Main_new.py                 # Main experiment script
│   ├── problem_instance.py         # Constraints, projections, objective
│   ├── network_stochastic.py       # Markov chain computations (M, V, K_W, S)
│   ├── optimization.py             # SPSA solver with null-space perturbations
│   ├── grid_generation.py          # Grid networks with stochastic weights
│   ├── utils.py                    # Utility functions
│   ├── feasibility.py              # LP-based feasibility checking
│   └── Results/                    # Output figures and reports
│
└── Traffic_Network/                # Section 6.2: Resilience under edge failures
    ├── main.ipynb                  # Main experiment notebook
    ├── functions.py                # Core computations (MFPT, variance, Kemeny)
    ├── projections.py              # Dykstra projection (hard pi constraint)
    ├── naive_proj.py               # Naive row-normalization projection
    ├── hybrid_projection.py        # Hybrid projection (locked destinations)
    ├── minimal_optimization.py     # Constrained weight optimization (SLSQP)
    └── Figures/                    # Output figures
```

---

## Application 1: Surveillance Network (Section 6.1)

Optimizes a *surprise index* $\mathcal{S}(\mathbf{P}) = \sqrt{V_{\mathcal{W}}} / K_{\mathcal{W}}$, a scale-invariant measure of patrol unpredictability. The optimization finds a transition matrix $\mathbf{P}$ that maximizes $\mathcal{S}$ subject to a hard stationary distribution constraint, using SPSA with null-space perturbations from Franssen et al. (2025).

**Features:** deterministic and stochastic edge weights, uniform and non-uniform target distributions, obstacle configurations with LP-based feasibility verification.

**Configuration** (set in `Main_new.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ETA_SMALL` / `ETA_BIG` | `1e-4` / `1e-8` | Minimum transition probability (4×4 / 8×8) |
| `max_iter` | 8000–25000 | SPSA iterations |
| `a` | 0.0005–0.001 | SPSA step size |
| `obs8` | `[(2,2),(2,3),(5,4),(5,5)]` | Obstacle positions (8×8 grid) |
| `priority8` | `2.0` on adjacent nodes | Coverage weight multiplier |
```bash
cd Surveillance_Network
python Main_new.py
```

**Outputs** (saved to `Results/`):

| File | Description |
|------|-------------|
| `optimization_report.txt` | Summary table: $K_{\mathcal{W}}$, $\sqrt{V_{\mathcal{W}}}$, $\mathcal{S}$, $\|\pi - \hat{\mu}\|$ |
| `fig1a_uniform.png` | Uniform baseline policy (4×4) |
| `fig1b_min_variance.png` | Min-variance / Hamiltonian cycle (4×4) |
| `fig2_max_surprise_det.png` | Max-surprise policy (4×4) |
| `fig5_8x8_stoch.png` | Max-surprise policy (8×8, stochastic) |
| `supp_*.png` | Convergence plots (supplementary) |

---

## Application 2: Traffic Network (Section 6.2)

Performs a sequential $N{-}k$ failure analysis to quantify network resilience. When edges fail (road closures), the framework re-projects the transition matrix and optimizes edge weights (speed limits) to preserve connectivity. Three projection strategies are implemented:

- **Naive** (`naive_proj.py`) — row renormalization (short-term response)
- **Stationary-preserving** (`projections.py`) — Dykstra projection to maintain $\pi^*$ (long-term response)
- **Hybrid** (`hybrid_projection.py`) — locks destination nodes to $\pi^*$, scales intersections to natural flow

The weight optimization (`minimal_optimization.py`) solves a constrained minimal-intervention program: find the smallest change in weights such that $K_{\mathcal{W}}$ and $V_{\mathcal{W}}$ do not deteriorate.

**Configuration** (set in `main.ipynb`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_nodes` | 10 | Number of nodes in the network |
| `connection_prob` | 0.5 | Edge density (geometric graph radius) |
| `dest_set` | `[1, 4, 8]` | Destination nodes (locked in hybrid projection) |
| `W_min` / `W_max` | 5 / 50 | Speed limit bounds (weight range) |
| `k` | 1–10 | Number of sequential edge failures |
```bash
cd Traffic_Network
jupyter notebook main.ipynb
```

**Outputs** (saved to `Figures/`):

| File | Description |
|------|-------------|
| `boxplot_colorpat_n10_deg5_geom_150reps_k6.png` | Improvement in $K_{\mathcal{W}}$ and $V_{\mathcal{W}}$ after minimal intervention (boxplots) |
| `statdist_boxplot_colorpat_n10_deg5_geom_150reps_k6.png` | Stationary distribution preservation across projection strategies |
| `res_boxplot_colorpat_n10_deg5_geom_150reps_k6.rtf` | Numerical summary: mean improvements and $\Delta\pi$ statistics |

---

## Prerequisites

- **Python** 3.10+ (tested on 3.12)
- Required packages:
```
numpy
scipy
matplotlib
networkx
sympy
jupyter
```

Install with pip:
```bash
pip install numpy scipy matplotlib networkx sympy jupyter
```

---

## References

- **Le, T.P., van der Burg, R., Heidergott, B.F., Lindner, I.D., & Zocca, A.** (2026). Random Walks with Traversal Costs: Variance-Aware Performance Analysis and Network Optimization. Working paper.

- **Franssen, C.P., Zocca, A., & Heidergott, B.F.** (2025). A First-Order Gradient Approach for the Connectivity Optimization of Markov Chains. *IEEE Transactions on Automatic Control*.

---

## Acknowledgements

A. Zocca and R. van der Burg are partially supported by the NWO Vidi project *Power Network Optimization in the Age of Climate Extremes* ([https://doi.org/10.61686/GOOEL09973](https://doi.org/10.61686/GOOEL09973)).

---

## License

This project is licensed under the [MIT License](LICENSE.md).

---

## Contact

For questions or issues, contact: Thao Le — t.p.t.le@vu.nl
