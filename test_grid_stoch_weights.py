"""
Grid Network Surveillance Optimization with Stochastic Weights.

This script extends the Markov chain optimization to handle STOCHASTIC edge weights,
where the weight W(i,j) on each edge is a random variable with:
    - Mean: μ(i,j) = E[W(i,j)]
    - Second moment: μ²(i,j) = E[W²(i,j)]
    - Variance: Var[W(i,j)] = μ²(i,j) - μ(i,j)²
    - Coefficient of Variation: CV(i,j) = σ(i,j) / μ(i,j)

Key difference from deterministic case:
    - Deterministic: W2[i,j] = W[i,j]² (variance = 0)
    - Stochastic: W2[i,j] = W[i,j]² * (1 + CV²) where CV is the coefficient of variation

This allows modeling scenarios where:
    - Some edges have predictable travel times (CV < 1)
    - Some edges have highly variable travel times (CV > 1, e.g., traffic, weather)
"""

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import null_space
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def row_normalize(M):
    row_sums = np.sum(M, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums

def create_edge_matrix(mA):
    indices = np.nonzero(mA)
    num_edges = indices[0].shape[0]
    E = np.zeros((num_edges, 2), dtype=int)
    E[:, 0] = indices[0]
    E[:, 1] = indices[1]
    return E

def build_neighborhoods(edge_matrix, N):
    neighborhoods = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edge_matrix):
        neighborhoods[i].append(idx)
    return neighborhoods

def x_to_matrix(x, N, edge_matrix, bUndirected):
    P = np.zeros((N, N))
    P[edge_matrix[:, 0], edge_matrix[:, 1]] = x
    if bUndirected:
        return P + P.T
    return P

def proj_c_simplex(v, c=1, tol=1e-8):
    N = len(v)
    vU = np.sort(v)[::-1]
    cssv = np.cumsum(vU)
    l = [k+1 for k in range(N) if (cssv[k] - c) / (k + 1) < vU[k]]
    
    if not l:
        v_proj = np.maximum(v, 0)
        if np.sum(v_proj) > 0:
            return v_proj * c / np.sum(v_proj)
        return np.ones(N) * c / N
    
    K = max(l)
    tau = (cssv[K - 1] - c) / K
    return np.maximum(v - tau, 0)

def projection_markov(x_to_proj, eta, neighborhoods, mA):
    x = x_to_proj - eta
    x_proj = []
    for i, subset in enumerate(neighborhoods):
        if subset:
            n_edges = len(subset)
            c = 1 - n_edges * eta
            x_proj.extend(proj_c_simplex(x[subset], c=c).tolist())
    return np.array(x_proj) + eta


# ============================================================================
# GRID NETWORK GENERATION WITH STOCHASTIC WEIGHTS
# ============================================================================

def generate_grid_network_stochastic(n, obstacles=None, diagonal=False,
                                      cv_low=0.3, cv_high=1.5, 
                                      high_cv_fraction=0.5,
                                      seed=42):
    """
    Generate an n×n grid network with STOCHASTIC edge weights.
    
    Parameters:
        n: Grid size (n×n nodes, total N = n² nodes)
        obstacles: List of (row, col) tuples indicating obstacle positions
        diagonal: If True, include diagonal neighbors (8-connectivity)
        cv_low: Coefficient of variation for "reliable" edges (CV < 1)
        cv_high: Coefficient of variation for "unreliable" edges (CV > 1)
        high_cv_fraction: Fraction of edges that have high CV
        seed: Random seed for reproducibility
    
    Returns:
        mA: Adjacency matrix (N×N where N = n²)
        W: Mean weight matrix μ(i,j) (travel times mean, based on distance)
        W2: Second moment matrix E[W²] (NOT the square of W!)
        CV_matrix: Matrix of coefficient of variations for each edge
        obstacle_mask: Boolean array of size N, True for obstacle nodes
        grid_positions: Dict mapping node index to (row, col) position
    """
    np.random.seed(seed)
    
    N = n * n  # Total number of nodes
    mA = np.zeros((N, N))
    W = np.zeros((N, N))      # Mean weights μ
    W2 = np.zeros((N, N))     # Second moment E[W²]
    CV_matrix = np.zeros((N, N))  # Coefficient of variation for each edge
    
    # Convert obstacles to set for fast lookup
    if obstacles is None:
        obstacles = []
    obstacle_set = set(obstacles)
    obstacle_mask = np.zeros(N, dtype=bool)
    
    # Create mapping from (row, col) to node index
    def pos_to_idx(row, col):
        return row * n + col
    
    def idx_to_pos(idx):
        return (idx // n, idx % n)
    
    grid_positions = {i: idx_to_pos(i) for i in range(N)}
    
    # Mark obstacle nodes
    for (r, c) in obstacle_set:
        if 0 <= r < n and 0 <= c < n:
            obstacle_mask[pos_to_idx(r, c)] = True
    
    # Define neighbor offsets
    if diagonal:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # First pass: count edges and build adjacency
    edge_list = []
    for i in range(N):
        ri, ci = idx_to_pos(i)
        if obstacle_mask[i]:
            continue
        for dr, dc in offsets:
            rj, cj = ri + dr, ci + dc
            if 0 <= rj < n and 0 <= cj < n:
                j = pos_to_idx(rj, cj)
                if not obstacle_mask[j]:
                    edge_list.append((i, j, np.sqrt(dr**2 + dc**2)))
                    mA[i, j] = 1
    
    num_edges = len(edge_list)
    
    # Randomly assign which edges have high CV
    num_high_cv = int(num_edges * high_cv_fraction)
    high_cv_indices = np.random.choice(num_edges, size=num_high_cv, replace=False)
    high_cv_set = set(high_cv_indices)
    
    # Second pass: assign weights with varying CVs
    for idx, (i, j, dist) in enumerate(edge_list):
        # Base mean weight is distance with some variation
        mu = dist * (1 + 0.2 * np.random.rand())
        
        # Assign CV based on whether this is a high-CV edge
        if idx in high_cv_set:
            cv = cv_high + 0.2 * np.random.rand()  # Some variation in high CV
        else:
            cv = cv_low + 0.1 * np.random.rand()   # Some variation in low CV
        
        # Compute second moment from CV
        # CV = σ/μ, so σ = CV * μ
        # Var = σ² = CV² * μ²
        # E[W²] = Var + μ² = μ² * (1 + CV²)
        variance = (cv * mu) ** 2
        mu2 = mu**2 + variance  # E[W²] = E[W]² + Var[W]
        
        W[i, j] = mu
        W2[i, j] = mu2
        CV_matrix[i, j] = cv
    
    return mA, W, W2, CV_matrix, obstacle_mask, grid_positions


def create_grid_target_distribution(n, obstacle_mask, priority_positions=None):
    """Create a target stationary distribution for the grid."""
    N = n * n
    weights = np.ones(N)
    weights[obstacle_mask] = 0
    
    if priority_positions is not None:
        for (r, c), priority in priority_positions.items():
            idx = r * n + c
            if not obstacle_mask[idx]:
                weights[idx] = priority
    
    pi_hat = weights / weights.sum()
    return pi_hat


# ============================================================================
# MARKOV CHAIN CLASS WITH STOCHASTIC WEIGHTS
# ============================================================================

class MarkovChainStochastic:
    """
    Markov chain class that supports STOCHASTIC edge weights.
    
    Key difference: W2 (second moment) is stored independently from W (mean),
    allowing for non-zero variance in the edge weights.
    """
    
    def __init__(self, mA, x=None, W=None, W2=None, bUndirected=False):
        """
        Parameters:
            mA: Adjacency matrix
            x: Edge weight vector for transition probabilities
            W: Mean weight matrix μ(i,j) = E[W(i,j)]
            W2: Second moment matrix μ²(i,j) = E[W²(i,j)]
                If None, defaults to W² (deterministic case)
            bUndirected: Whether the graph is undirected
        """
        self.n = mA.shape[0]
        self.bUndirected = bUndirected
        self.mA = mA
        
        # Initialize mean weights
        if W is not None:
            self.W = W
        else:
            self.W = np.ones((self.n, self.n))
        
        # Initialize second moments
        # CRITICAL: If W2 is not provided, use W² (deterministic case)
        # If W2 IS provided, use it directly (stochastic case)
        if W2 is not None:
            self.W2 = W2
        else:
            # Deterministic case: E[W²] = E[W]²
            self.W2 = self.W ** 2
        
        # Compute and store the edge variance matrix
        self.W_var = self.W2 - self.W ** 2
        
        # Cached computations
        self.pi = None
        self.Pi = None
        self.pi_W = None
        self.Z = None
        self.M = None
        self.V = None
        self.K_W = None
        self.Net_Var = None
        self.Eff_Idx = None
        
        self.edge_matrix = create_edge_matrix(mA)
        
        if x is not None:
            self.x = x
    
    @property
    def P(self):
        return x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected)
    
    @staticmethod
    def P_to_x(P, mA, bUndirected):
        N, _ = mA.shape
        if not bUndirected:
            return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1])
        else:
            return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1 and i <= j])
    
    def compute_pi(self):
        """Compute stationary distribution of embedded chain."""
        eigenvalues, eigenvectors = eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = np.real(eigenvectors[:, idx])
        self.pi = pi / np.sum(pi)
        return self.pi
    
    def compute_Pi(self):
        if self.pi is None:
            self.compute_pi()
        self.Pi = np.outer(np.ones(self.n), self.pi)
        return self.Pi
    
    def compute_pi_W(self):
        """Compute time-weighted stationary distribution."""
        if self.pi is None:
            self.compute_pi()
        P_hadamard_W = np.multiply(self.P, self.W)
        U_bar = np.sum(P_hadamard_W, axis=1)
        numerator = self.pi * U_bar
        denominator = np.sum(numerator)
        if denominator < 1e-12:
            self.pi_W = self.pi.copy()
        else:
            self.pi_W = numerator / denominator
        return self.pi_W
    
    def compute_Z(self):
        """Compute fundamental matrix."""
        if self.Pi is None:
            self.compute_Pi()
        I = np.eye(self.n)
        self.Z = inv(I - self.P + self.Pi)
        return self.Z
    
    def compute_M(self):
        """Compute mean first-passage time matrix."""
        if self.Z is None:
            self.compute_Z()
        
        I = np.eye(self.n)
        Ones = np.ones((self.n, self.n))
        P_dot_W = np.multiply(self.P, self.W)  # Uses MEAN weights
        Xi_inv = np.diag(1.0 / (self.pi + 1e-12))
        
        scalar_term = np.dot(self.pi, np.sum(P_dot_W, axis=1))
        Term1 = self.Z @ P_dot_W @ self.Pi
        Term2 = Ones @ np.diag(np.diag(Term1))
        Z_dg_matrix = np.diag(np.diag(self.Z))
        Term3_inner = I - self.Z + (Ones @ Z_dg_matrix)
        
        Bracket = Term1 - Term2 + (scalar_term * Term3_inner)
        self.M = Bracket @ Xi_inv
        return self.M
    
    def compute_V(self):
        """
        Compute variance of first-passage times.
        
        CRITICAL: This uses W2 (second moment), NOT W² (square of mean).
        This is where stochastic weights affect the variance!
        """
        if self.M is None:
            self.compute_M()
        
        I = np.eye(self.n)
        Ones = np.ones((self.n, self.n))
        P_dot_W = np.multiply(self.P, self.W)    # P ⊙ μ
        P_dot_W2 = np.multiply(self.P, self.W2)  # P ⊙ μ² (SECOND MOMENT, not μ²!)
        
        val_A = np.dot(self.pi, np.sum(P_dot_W2, axis=1))
        M_off_diag = self.M - np.diag(np.diag(self.M))
        vec_B = 2 * self.pi @ (P_dot_W @ M_off_diag)
        M2_diag_vals = (val_A + vec_B) / (self.pi + 1e-12)
        M2_dg = np.diag(M2_diag_vals)

        Sum_PW2 = np.sum(P_dot_W2, axis=1, keepdims=True)
        Term1 = self.Z @ Sum_PW2 @ np.ones((1, self.n))
        Term2 = Ones @ np.diag(np.diag(Term1))
        
        Matrix_A = self.Z @ P_dot_W @ M_off_diag
        Term3 = 2 * (Matrix_A - (Ones @ np.diag(np.diag(Matrix_A))))
        
        Z_dg = np.diag(np.diag(self.Z))
        Prefactor = I - self.Z + (Ones @ Z_dg)
        Term4 = Prefactor @ M2_dg
        
        M2 = Term1 - Term2 + Term3 + Term4
        self.V = M2 - (self.M ** 2)
        return self.V
    
    def compute_kemeny_W(self):
        """Compute weighted Kemeny constant."""
        if self.pi_W is None:
            self.compute_pi_W()
        if self.M is None:
            self.compute_M()
        self.K_W = self.pi_W @ self.M @ self.pi_W.T
        return self.K_W
    
    def compute_network_variance(self):
        """Compute network-level variance."""
        if self.pi_W is None:
            self.compute_pi_W()
        if self.V is None:
            self.compute_V()
        self.Net_Var = self.pi_W @ self.V @ self.pi_W.T
        return self.Net_Var
    
    def compute_efficiency_index(self):
        """Compute efficiency index = Variance / Mean."""
        if self.K_W is None:
            self.compute_kemeny_W()
        if self.Net_Var is None:
            self.compute_network_variance()
        if self.K_W == 0:
            return np.inf
        self.Eff_Idx = self.Net_Var / self.K_W
        return self.Eff_Idx
    
    def get_weight_statistics(self):
        """Return statistics about edge weights."""
        mask = self.mA > 0
        return {
            'mean_weights': self.W[mask],
            'second_moments': self.W2[mask],
            'variances': self.W_var[mask],
            'cv': np.sqrt(self.W_var[mask]) / (self.W[mask] + 1e-12)
        }
    
    def clear_cache(self):
        self.pi = None
        self.Pi = None
        self.pi_W = None
        self.Z = None
        self.M = None
        self.V = None
        self.K_W = None
        self.Net_Var = None
        self.Eff_Idx = None


# ============================================================================
# PROBLEM INSTANCE FOR STOCHASTIC WEIGHTS
# ============================================================================

class EfficiencyProblemInstanceStochastic:
    """
    Problem instance for efficiency index optimization with STOCHASTIC weights.
    """
    
    def __init__(self, mA, W, W2, eta=1e-4, pi_hat=None, 
                 objective_type='maximize_efficiency',
                 pi_penalty_weight=1e4):
        """
        Parameters:
            mA: Adjacency matrix
            W: Mean weight matrix μ(i,j)
            W2: Second moment matrix E[W²(i,j)] - NOT the square of W!
            eta: Small constant for numerical stability
            pi_hat: Target stationary distribution
            objective_type: 'maximize_efficiency' or 'minimize_efficiency'
            pi_penalty_weight: Penalty weight for distribution constraint
        """
        self.mA = mA
        self.W = W
        self.W2 = W2
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
        self.objective_type = objective_type
        self.pi_penalty_weight = pi_penalty_weight
        
        if pi_hat is None:
            self.pi_hat = np.ones(self.N) / self.N
        else:
            self.pi_hat = np.array(pi_hat)
        
        self.edge_matrix = create_edge_matrix(mA)
        self.d = len(self.edge_matrix)
        self.neighborhoods = build_neighborhoods(self.edge_matrix, self.N)
        
        self._build_constraint_matrices()
        self.proj_type = 'Markov'
    
    def _build_constraint_matrices(self):
        N = self.N
        d = self.d
        A_row = np.zeros((N, d))
        for idx, (i, j) in enumerate(self.edge_matrix):
            A_row[i, idx] = 1
        b_row = np.ones(N)
        valid_rows = np.sum(A_row, axis=1) > 0
        self.A_eq = A_row[valid_rows]
        self.b_eq = b_row[valid_rows]
    
    def evaluate_metrics(self, P):
        """Evaluate all metrics for a given transition matrix."""
        # Create chain with STOCHASTIC weights
        mc = MarkovChainStochastic(self.mA, W=self.W, W2=self.W2)
        mc.x = MarkovChainStochastic.P_to_x(P, self.mA, self.bUndirected)
        
        mc.compute_pi()
        mc.compute_pi_W()
        mc.compute_M()
        mc.compute_V()
        mc.compute_kemeny_W()
        mc.compute_network_variance()
        mc.compute_efficiency_index()
        
        pi_error = np.linalg.norm(mc.pi_W - self.pi_hat)
        
        return {
            'pi': mc.pi,
            'pi_W': mc.pi_W,
            'M': mc.M,
            'V': mc.V,
            'K_W': mc.K_W,
            'Net_Var': mc.Net_Var,
            'Eff_Idx': mc.Eff_Idx,
            'pi_error': pi_error
        }
    
    def objective(self, x):
        """Compute objective function."""
        P = x_to_matrix(x, self.N, self.edge_matrix, self.bUndirected)
        metrics = self.evaluate_metrics(P)
        
        pi_error = metrics['pi_error']
        penalty = self.pi_penalty_weight * (pi_error ** 2)
        
        if self.objective_type == 'maximize_efficiency':
            return -metrics['Eff_Idx'] + penalty
        else:
            return metrics['Eff_Idx'] + penalty
    
    def project(self, x):
        return projection_markov(x, self.eta, self.neighborhoods, self.mA)


# ============================================================================
# SPSA OPTIMIZATION
# ============================================================================

def solve_spsa_efficiency(problem, x_init, max_iter=5000, 
                          a=0.05, a_eps=100, e=1e-3, r_nu=0.101,
                          obj_interval=100, verbose=True,
                          max_obj_value=1e10):
    """SPSA optimization for efficiency index with numerical safeguards."""
    
    x = problem.project(x_init.copy())
    best_x = x.copy()
    best_obj = float('inf')
    
    # Compute initial objective
    try:
        init_obj = problem.objective(x)
        if np.isfinite(init_obj):
            best_obj = init_obj
    except:
        pass
    
    iter_hist = []
    eff_hist = []
    kw_hist = []
    var_hist = []
    
    stagnation_count = 0
    prev_best = best_obj
    
    for k in range(max_iter):
        c_k = e / ((k + 1) ** r_nu)
        a_k = a / (k + a_eps + 1)
        
        delta = 2 * (np.random.rand(problem.d) > 0.5).astype(float) - 1
        
        x_plus = problem.project(x + c_k * delta)
        x_minus = problem.project(x - c_k * delta)
        
        try:
            f_plus = problem.objective(x_plus)
            f_minus = problem.objective(x_minus)
        except:
            continue
        
        # Skip if objectives are invalid
        if not np.isfinite(f_plus) or not np.isfinite(f_minus):
            continue
        if abs(f_plus) > max_obj_value or abs(f_minus) > max_obj_value:
            continue
        
        grad_estimate = (f_plus - f_minus) / (2 * c_k * delta + 1e-12)
        
        # Clip gradient to prevent explosion
        grad_estimate = np.clip(grad_estimate, -100, 100)
        
        x_new = x - a_k * grad_estimate
        x = problem.project(x_new)
        
        if np.isfinite(f_plus) and f_plus < best_obj:
            best_obj = f_plus
            best_x = x_plus.copy()
        if np.isfinite(f_minus) and f_minus < best_obj:
            best_obj = f_minus
            best_x = x_minus.copy()
        
        if (k + 1) % obj_interval == 0 or k == 0:
            P = x_to_matrix(best_x, problem.N, problem.edge_matrix, problem.bUndirected)
            try:
                metrics = problem.evaluate_metrics(P)
                
                # Only record if metrics are reasonable
                if np.isfinite(metrics['Eff_Idx']) and abs(metrics['Eff_Idx']) < 1e6:
                    iter_hist.append(k + 1)
                    eff_hist.append(metrics['Eff_Idx'])
                    kw_hist.append(metrics['K_W'])
                    var_hist.append(metrics['Net_Var'])
                    
                    if verbose:
                        print(f"Iter {k+1:5d}: Eff={metrics['Eff_Idx']:.4f}, "
                              f"K_W={metrics['K_W']:.4f}, Var={metrics['Net_Var']:.4f}, "
                              f"π_err={metrics['pi_error']:.4f}")
                    
                    # Check for stagnation
                    if abs(best_obj - prev_best) < 1e-8:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                    prev_best = best_obj
                    
                    # Early stopping if stagnated
                    if stagnation_count > 10:
                        if verbose:
                            print(f"Converged at iteration {k+1}")
                        break
            except:
                continue
    
    return iter_hist, eff_hist, kw_hist, var_hist, best_x, best_obj


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cv_distribution(CV_matrix, mA, title="Coefficient of Variation Distribution"):
    """Plot histogram of CVs across all edges."""
    mask = mA > 0
    cvs = CV_matrix[mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if cv < 1 else 'red' for cv in cvs]
    ax.hist(cvs[cvs < 1], bins=20, alpha=0.7, color='green', label=f'CV < 1 (n={np.sum(cvs < 1)})')
    ax.hist(cvs[cvs >= 1], bins=20, alpha=0.7, color='red', label=f'CV ≥ 1 (n={np.sum(cvs >= 1)})')
    
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='CV = 1')
    ax.set_xlabel('Coefficient of Variation (CV = σ/μ)')
    ax.set_ylabel('Number of Edges')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_optimization_results_single(iter_hist, eff_hist, kw_hist, var_hist,
                                      initial_metrics, final_metrics,
                                      n, case_name, filename):
    """
    Plot optimization convergence for a single case (deterministic or stochastic).
    Matches the original grid_optimization_results.png format exactly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Efficiency Index
    axes[0, 0].plot(iter_hist, eff_hist, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].axhline(y=initial_metrics['Eff_Idx'], color='g', linestyle='--', 
                       label=f"Initial = {initial_metrics['Eff_Idx']:.4f}")
    axes[0, 0].axhline(y=final_metrics['Eff_Idx'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['Eff_Idx']:.4f}")
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(r'Efficiency Index $\lambda$')
    axes[0, 0].set_title(r'Efficiency Index ($\uparrow$ higher is better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Patrol Time (K_W)
    axes[0, 1].plot(iter_hist, kw_hist, 'orange', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=initial_metrics['K_W'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['K_W']:.4f}")
    axes[0, 1].axhline(y=final_metrics['K_W'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['K_W']:.4f}")
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[0, 1].set_title('Mean Patrol Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Path Variance
    axes[1, 0].plot(iter_hist, var_hist, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(y=initial_metrics['Net_Var'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['Net_Var']:.4f}")
    axes[1, 0].axhline(y=final_metrics['Net_Var'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['Net_Var']:.4f}")
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 0].set_title(r'Path Variance ($\uparrow$ more unpredictable)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mean-Variance tradeoff
    scatter = axes[1, 1].scatter(kw_hist, var_hist, c=iter_hist, cmap='viridis', 
                                  s=50, alpha=0.7)
    axes[1, 1].scatter(initial_metrics['K_W'], initial_metrics['Net_Var'], 
                      color='green', s=300, marker='*', label='Initial', zorder=5, edgecolor='black')
    axes[1, 1].scatter(final_metrics['K_W'], final_metrics['Net_Var'],
                      color='red', s=300, marker='*', label='Final', zorder=5, edgecolor='black')
    axes[1, 1].set_xlabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[1, 1].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 1].set_title('Mean-Variance Tradeoff')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Iteration')
    
    plt.suptitle(f'{n}×{n} Grid Network - {case_name} - Optimization Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n✓ Optimization plot saved to '{filename}'")
    plt.show()
    
    return fig


def plot_grid_network_stochastic(n, mA, W, CV_matrix, obstacle_mask, grid_positions,
                                  P=None, pi=None, title="Grid Network", filename=None):
    """
    Plot grid network showing optimal surveillance policy.
    
    - Edge THICKNESS indicates transition probability P(i,j): thicker = higher probability
    - Edge COLOR indicates reliability: green (CV < 1, reliable) vs red (CV > 1, unreliable)
    - Node color intensity indicates stationary distribution π
    - Curved arrows for better visibility of direction
    """
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw edges colored by CV, thickness by P
    edge_list = np.argwhere(mA > 0)
    
    # Determine max P for scaling line width
    if P is not None:
        max_P = np.max(P[mA > 0]) if np.any(mA > 0) else 1.0
        min_P = np.min(P[mA > 0]) if np.any(mA > 0) else 0.0
    else:
        max_P, min_P = 1.0, 0.0
    
    for i, j in edge_list:
        ri, ci = grid_positions[i]
        rj, cj = grid_positions[j]
        cv = CV_matrix[i, j]
        
        # Color by CV: green (reliable, CV < 1) vs red (unreliable, CV >= 1)
        if cv < 1:
            color = 'green'
            alpha = 0.6 + 0.4 * (1 - cv)  # More opaque for lower CV
        else:
            color = 'red'
            alpha = min(0.6 + 0.2 * (cv - 1), 0.95)  # More opaque for higher CV
        
        # Line width and normalized P
        if P is not None and P[i, j] > 0.001:
            if max_P > min_P:
                normalized_P = (P[i, j] - min_P) / (max_P - min_P)
            else:
                normalized_P = 0.5
            linewidth = 0.5 + 5.5 * normalized_P
        else:
            normalized_P = 0.0
            linewidth = 0.5  # Very thin for near-zero P
        
        # Calculate curve direction
        dx = cj - ci
        dy = rj - ri
        
        # Shrink start and end points to not overlap with nodes
        shrink = 0.35
        start = (ci + shrink * dx, ri + shrink * dy)
        end = (cj - shrink * dx, rj - shrink * dy)
        
        # Curved arrow with connectionstyle
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0.2",  # Curved arrow
            arrowstyle='-|>',
            mutation_scale=10 + 5 * normalized_P,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2
        )
        ax.add_patch(arrow)
    
    # Draw nodes
    for i in range(n * n):
        r, c = grid_positions[i]
        if obstacle_mask[i]:
            ax.add_patch(plt.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                        color='black', zorder=3))
            ax.text(c, r, 'X', ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold', zorder=4)
        else:
            # Node color intensity by stationary distribution
            node_color = 'lightblue'
            if pi is not None:
                intensity = min(pi[i] / (np.max(pi) + 1e-12), 1.0)
                node_color = plt.cm.Blues(0.3 + 0.7 * intensity)
            
            circle = plt.Circle((c, r), 0.3, color=node_color, ec='black', 
                                linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=4)
    
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    green_line = Line2D([0], [0], color='green', linewidth=3, label='Reliable (CV < 1)')
    red_line = Line2D([0], [0], color='red', linewidth=3, label='Unreliable (CV ≥ 1)')
    thick_line = Line2D([0], [0], color='gray', linewidth=5, label='High P(i,j)')
    thin_line = Line2D([0], [0], color='gray', linewidth=1, label='Low P(i,j)')
    
    ax.legend(handles=[green_line, red_line, thick_line, thin_line], 
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to '{filename}'")
    
    plt.show()
    return fig


def plot_optimal_policy_comparison(n, mA, CV_matrix, obstacle_mask, grid_positions,
                                    P_det, P_stoch, pi_det, pi_stoch,
                                    filename='optimal_policy_comparison.png'):
    """
    Plot side-by-side comparison of optimal policies for deterministic vs stochastic weights.
    - Deterministic: all edges gray (no CV variation)
    - Stochastic: edges colored by CV (green=reliable, red=unreliable)
    - Edge thickness indicates P(i,j)
    - Curved arrows for better visibility of direction
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax_idx, (P, pi, case_title, use_cv_color) in enumerate([
        (P_det, pi_det, 'Deterministic Weights (CV=0)', False),  # No CV coloring
        (P_stoch, pi_stoch, 'Stochastic Weights (Mixed CV)', True)  # CV coloring
    ]):
        ax = axes[ax_idx]
        
        # Draw edges
        edge_list = np.argwhere(mA > 0)
        max_P = np.max(P[mA > 0]) if np.any(mA > 0) else 1.0
        min_P = np.min(P[mA > 0]) if np.any(mA > 0) else 0.0
        
        for i, j in edge_list:
            ri, ci = grid_positions[i]
            rj, cj = grid_positions[j]
            cv = CV_matrix[i, j]
            
            # Color by CV only for stochastic case
            if use_cv_color:
                if cv < 1:
                    color = 'green'
                    alpha = 0.6 + 0.4 * (1 - cv)
                else:
                    color = 'red'
                    alpha = min(0.6 + 0.2 * (cv - 1), 0.95)
            else:
                # Deterministic: all edges gray
                color = 'dimgray'
                alpha = 0.7
            
            # Line width by P
            if P[i, j] > 0.001:
                if max_P > min_P:
                    normalized_P = (P[i, j] - min_P) / (max_P - min_P)
                else:
                    normalized_P = 0.5
                linewidth = 0.5 + 5.5 * normalized_P
            else:
                linewidth = 0.3
            
            # Use curved arrows (FancyArrowPatch) for better visibility
            from matplotlib.patches import FancyArrowPatch
            
            # Calculate curve direction (perpendicular offset)
            dx = cj - ci
            dy = rj - ri
            
            # Shrink start and end points to not overlap with nodes
            shrink = 0.35
            start = (ci + shrink * dx, ri + shrink * dy)
            end = (cj - shrink * dx, rj - shrink * dy)
            
            # Curved arrow with connectionstyle
            arrow = FancyArrowPatch(
                start, end,
                connectionstyle="arc3,rad=0.2",  # Curved arrow
                arrowstyle='-|>',
                mutation_scale=10 + 5 * normalized_P if P[i, j] > 0.001 else 8,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for i in range(n * n):
            r, c = grid_positions[i]
            if obstacle_mask[i]:
                ax.add_patch(plt.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                            color='black', zorder=3))
                ax.text(c, r, 'X', ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold', zorder=4)
            else:
                node_color = 'lightblue'
                if pi is not None:
                    intensity = min(pi[i] / (np.max(pi) + 1e-12), 1.0)
                    node_color = plt.cm.Blues(0.3 + 0.7 * intensity)
                
                circle = plt.Circle((c, r), 0.3, color=node_color, ec='black', 
                                    linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=4)
        
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(n - 0.3, -0.7)
        ax.set_aspect('equal')
        ax.set_title(case_title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add shared legend
    from matplotlib.lines import Line2D
    gray_line = Line2D([0], [0], color='dimgray', linewidth=3, label='Deterministic (no CV)')
    green_line = Line2D([0], [0], color='green', linewidth=3, label='Reliable (CV < 1)')
    red_line = Line2D([0], [0], color='red', linewidth=3, label='Unreliable (CV ≥ 1)')
    thick_line = Line2D([0], [0], color='gray', linewidth=5, label='High P(i,j)')
    thin_line = Line2D([0], [0], color='gray', linewidth=1, label='Low P(i,j)')
    
    fig.legend(handles=[gray_line, green_line, red_line, thick_line, thin_line], 
               loc='lower center', ncol=5, fontsize=11)
    
    plt.suptitle('Optimal Surveillance Policy Comparison\n(Edge thickness = P, Curved arrows show direction)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Policy comparison saved to '{filename}'")
    
    plt.show()
    return fig


# ============================================================================
# MAIN TEST: COMPARISON DETERMINISTIC VS STOCHASTIC
# ============================================================================

def analyze_policy_vs_cv(P, CV_matrix, mA, title="Policy vs CV Analysis"):
    """
    Analyze whether the optimal policy favors high-CV or low-CV edges.
    This helps understand if the optimizer is meaningfully using edge uncertainty.
    """
    mask = mA > 0
    
    # Get P values and CV values for all edges
    P_values = P[mask]
    CV_values = CV_matrix[mask]
    
    # Separate into low-CV and high-CV edges
    low_cv_mask = CV_values < 1
    high_cv_mask = CV_values >= 1
    
    avg_P_low_cv = np.mean(P_values[low_cv_mask]) if np.any(low_cv_mask) else 0
    avg_P_high_cv = np.mean(P_values[high_cv_mask]) if np.any(high_cv_mask) else 0
    
    # Correlation between P and CV
    correlation = np.corrcoef(P_values, CV_values)[0, 1]
    
    print(f"\n{title}")
    print("-" * 50)
    print(f"Average P for low-CV edges (CV < 1):  {avg_P_low_cv:.4f}")
    print(f"Average P for high-CV edges (CV ≥ 1): {avg_P_high_cv:.4f}")
    print(f"Correlation(P, CV): {correlation:.4f}")
    
    if avg_P_high_cv > avg_P_low_cv:
        print("→ Policy FAVORS unreliable edges (exploiting variance)")
    else:
        print("→ Policy FAVORS reliable edges (avoiding variance)")
    
    return {
        'avg_P_low_cv': avg_P_low_cv,
        'avg_P_high_cv': avg_P_high_cv,
        'correlation': correlation
    }


def test_deterministic_vs_stochastic():
    """
    Compare optimization results between:
    1. Deterministic weights (CV = 0)
    2. Stochastic weights (mixed CVs)
    """
    
    print("="*80)
    print("TEST: DETERMINISTIC vs STOCHASTIC WEIGHT COMPARISON")
    print("="*80)
    
    # Grid parameters
    n = 5
    obstacles = [(2, 2)]  # Center obstacle
    
    print(f"\nGrid size: {n}×{n} = {n*n} nodes")
    print(f"Obstacle positions: {obstacles}")
    
    # =========== CASE 1: DETERMINISTIC WEIGHTS ===========
    print("\n" + "="*60)
    print("CASE 1: DETERMINISTIC WEIGHTS (CV = 0)")
    print("="*60)
    
    # Generate network with stochastic setup, then override to deterministic
    mA, W_mean, W2_stoch, CV_matrix, obstacle_mask, grid_positions = \
        generate_grid_network_stochastic(n, obstacles=obstacles, cv_low=0.3, cv_high=1.5)
    
    # For deterministic: W2 = W²
    W2_det = W_mean ** 2
    
    # Create target distribution
    pi_hat = create_grid_target_distribution(n, obstacle_mask)
    
    # Setup problem with DETERMINISTIC weights
    problem_det = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W_mean, W2=W2_det,  # W2 = W²
        eta=1e-4, pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        pi_penalty_weight=1e3
    )
    
    # Initial uniform policy
    x_init = np.zeros(problem_det.d)
    for i, subset in enumerate(problem_det.neighborhoods):
        if len(subset) > 0:
            for idx in subset:
                x_init[idx] = 1.0 / len(subset)
    
    P_init = x_to_matrix(x_init, problem_det.N, problem_det.edge_matrix, False)
    metrics_det_init = problem_det.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics (Deterministic W2 = W²):")
    print(f"  Efficiency Index: {metrics_det_init['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_det_init['K_W']:.6f}")
    print(f"  Variance: {metrics_det_init['Net_Var']:.6f}")
    
    # Compute edge weight variance (should be 0 for deterministic)
    det_edge_var = W2_det - W_mean**2
    print(f"  Edge weight variance (max): {np.max(det_edge_var[mA > 0]):.6f}")
    
    # Optimize
    print("\nRunning SPSA optimization (Deterministic)...")
    iter_det, eff_hist_det, kw_det, var_det, best_x_det, _ = solve_spsa_efficiency(
        problem_det, x_init, max_iter=5000, a=0.02, a_eps=50, 
        obj_interval=200, verbose=True
    )
    
    P_det = x_to_matrix(best_x_det, problem_det.N, problem_det.edge_matrix, False)
    metrics_det_final = problem_det.evaluate_metrics(P_det)
    
    print(f"\nFinal metrics (Deterministic):")
    print(f"  Efficiency Index: {metrics_det_final['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_det_final['K_W']:.6f}")
    print(f"  Variance: {metrics_det_final['Net_Var']:.6f}")
    
    # =========== CASE 2: STOCHASTIC WEIGHTS ===========
    print("\n" + "="*60)
    print("CASE 2: STOCHASTIC WEIGHTS (Mixed CVs)")
    print("="*60)
    
    # Show CV distribution
    mask = mA > 0
    cvs = CV_matrix[mask]
    print(f"\nCoefficient of Variation Statistics:")
    print(f"  Min CV: {cvs.min():.3f}")
    print(f"  Max CV: {cvs.max():.3f}")
    print(f"  Mean CV: {cvs.mean():.3f}")
    print(f"  Edges with CV < 1: {np.sum(cvs < 1)} ({100*np.sum(cvs < 1)/len(cvs):.1f}%)")
    print(f"  Edges with CV > 1: {np.sum(cvs > 1)} ({100*np.sum(cvs > 1)/len(cvs):.1f}%)")
    
    # Compute edge weight variance
    stoch_edge_var = W2_stoch - W_mean**2
    print(f"  Edge weight variance (mean): {np.mean(stoch_edge_var[mA > 0]):.6f}")
    print(f"  Edge weight variance (max): {np.max(stoch_edge_var[mA > 0]):.6f}")
    
    # Setup problem with STOCHASTIC weights
    problem_stoch = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W_mean, W2=W2_stoch,  # W2 ≠ W² (stochastic!)
        eta=1e-4, pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        pi_penalty_weight=1e3
    )
    
    metrics_stoch_init = problem_stoch.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics (Stochastic E[W²] ≠ E[W]²):")
    print(f"  Efficiency Index: {metrics_stoch_init['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_stoch_init['K_W']:.6f}")
    print(f"  Variance: {metrics_stoch_init['Net_Var']:.6f}")
    
    # Optimize
    print("\nRunning SPSA optimization (Stochastic)...")
    iter_stoch, eff_hist_stoch, kw_stoch, var_stoch, best_x_stoch, _ = solve_spsa_efficiency(
        problem_stoch, x_init, max_iter=5000, a=0.02, a_eps=50,
        obj_interval=200, verbose=True
    )
    
    P_stoch = x_to_matrix(best_x_stoch, problem_stoch.N, problem_stoch.edge_matrix, False)
    metrics_stoch_final = problem_stoch.evaluate_metrics(P_stoch)
    
    print(f"\nFinal metrics (Stochastic):")
    print(f"  Efficiency Index: {metrics_stoch_final['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_stoch_final['K_W']:.6f}")
    print(f"  Variance: {metrics_stoch_final['Net_Var']:.6f}")
    
    # =========== COMPARISON ===========
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Deterministic':<20} {'Stochastic':<20} {'Difference':<15}")
    print("-" * 85)
    
    diff_eff_init = metrics_stoch_init['Eff_Idx'] - metrics_det_init['Eff_Idx']
    diff_var_init = metrics_stoch_init['Net_Var'] - metrics_det_init['Net_Var']
    
    print(f"{'INITIAL Efficiency Index':<30} {metrics_det_init['Eff_Idx']:<20.4f} {metrics_stoch_init['Eff_Idx']:<20.4f} {diff_eff_init:+.4f}")
    print(f"{'INITIAL K_W (Mean)':<30} {metrics_det_init['K_W']:<20.4f} {metrics_stoch_init['K_W']:<20.4f} {0:+.4f}")
    print(f"{'INITIAL Variance':<30} {metrics_det_init['Net_Var']:<20.4f} {metrics_stoch_init['Net_Var']:<20.4f} {diff_var_init:+.4f}")
    
    if len(eff_hist_det) > 0 and len(eff_hist_stoch) > 0:
        diff_eff_final = metrics_stoch_final['Eff_Idx'] - metrics_det_final['Eff_Idx']
        diff_kw_final = metrics_stoch_final['K_W'] - metrics_det_final['K_W']
        diff_var_final = metrics_stoch_final['Net_Var'] - metrics_det_final['Net_Var']
        
        print(f"\n{'FINAL Efficiency Index':<30} {metrics_det_final['Eff_Idx']:<20.4f} {metrics_stoch_final['Eff_Idx']:<20.4f} {diff_eff_final:+.4f}")
        print(f"{'FINAL K_W (Mean)':<30} {metrics_det_final['K_W']:<20.4f} {metrics_stoch_final['K_W']:<20.4f} {diff_kw_final:+.4f}")
        print(f"{'FINAL Variance':<30} {metrics_det_final['Net_Var']:<20.4f} {metrics_stoch_final['Net_Var']:<20.4f} {diff_var_final:+.4f}")
    
    print(f"\n" + "="*80)
    print("★ KEY INSIGHT:")
    print("="*80)
    print(f"""
Even with the SAME MEAN weights (W), the stochastic case has HIGHER variance
because the edge weight variance contributes to the path variance:

  Var[path] = Var[from P transitions] + Var[from W stochasticity]

Initial variance increase due to stochastic W: {diff_var_init:.4f}
This is {100*diff_var_init/metrics_det_init['Net_Var']:.2f}% additional variance.

The Efficiency Index (Var/Mean) also increases because:
  - Mean (K_W) stays the same (depends only on E[W])
  - Variance increases (depends on E[W²])
""")
    
    # Analyze how optimal policies relate to edge CV
    print("\n" + "="*80)
    print("POLICY vs EDGE RELIABILITY ANALYSIS")
    print("="*80)
    
    analyze_policy_vs_cv(P_det, CV_matrix, mA, "Deterministic Case (optimal P)")
    analyze_policy_vs_cv(P_stoch, CV_matrix, mA, "Stochastic Case (optimal P)")
    
    # Compare initial policy
    analyze_policy_vs_cv(P_init, CV_matrix, mA, "Initial Uniform Policy")
    
    # Plot P vs CV scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mask = mA > 0
    cv_vals = CV_matrix[mask]
    
    for ax, (P, title) in zip(axes, [(P_det, 'Deterministic'), (P_stoch, 'Stochastic')]):
        p_vals = P[mask]
        colors = ['green' if cv < 1 else 'red' for cv in cv_vals]
        ax.scatter(cv_vals, p_vals, c=colors, alpha=0.6, s=50)
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coefficient of Variation (CV)')
        ax.set_ylabel('Transition Probability P(i,j)')
        ax.set_title(f'{title} Case: P vs CV')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(cv_vals, p_vals, 1)
        p_trend = np.poly1d(z)
        cv_range = np.linspace(cv_vals.min(), cv_vals.max(), 100)
        ax.plot(cv_range, p_trend(cv_range), 'b--', alpha=0.7, label=f'Trend (slope={z[0]:.3f})')
        ax.legend()
    
    plt.suptitle('Does Optimal Policy Favor High-CV Edges?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('policy_vs_cv_analysis.png', dpi=150)
    print("\n✓ Policy vs CV analysis saved to 'policy_vs_cv_analysis.png'")
    plt.show()
    
    # Plot comparison
    if len(eff_hist_det) > 0 and len(eff_hist_stoch) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Efficiency convergence
        axes[0].plot(iter_det, eff_hist_det, 'b-', label='Deterministic', linewidth=2, marker='o')
        axes[0].plot(iter_stoch, eff_hist_stoch, 'r-', label='Stochastic', linewidth=2, marker='s')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Efficiency Index')
        axes[0].set_title('Efficiency Index Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # CV distribution
        axes[1].hist(cvs[cvs < 1], bins=15, alpha=0.7, color='green', label='CV < 1 (reliable)')
        axes[1].hist(cvs[cvs >= 1], bins=15, alpha=0.7, color='red', label='CV ≥ 1 (variable)')
        axes[1].axvline(x=1, color='black', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Coefficient of Variation')
        axes[1].set_ylabel('Number of Edges')
        axes[1].set_title('CV Distribution Across Edges')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Bar comparison - Initial values
        metrics_names = ['Efficiency\nIndex', 'Mean\n(K_W)', 'Variance']
        det_vals = [metrics_det_init['Eff_Idx'], metrics_det_init['K_W'], metrics_det_init['Net_Var']]
        stoch_vals = [metrics_stoch_init['Eff_Idx'], metrics_stoch_init['K_W'], metrics_stoch_init['Net_Var']]
        
        x_pos = np.arange(len(metrics_names))
        width = 0.35
        
        axes[2].bar(x_pos - width/2, det_vals, width, label='Deterministic', color='blue', alpha=0.7)
        axes[2].bar(x_pos + width/2, stoch_vals, width, label='Stochastic', color='red', alpha=0.7)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(metrics_names)
        axes[2].set_ylabel('Value')
        axes[2].set_title('Initial Metrics Comparison\n(Same P, Different W variance)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Deterministic vs Stochastic Weights Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('deterministic_vs_stochastic_comparison.png', dpi=150)
        print("\n✓ Comparison plot saved to 'deterministic_vs_stochastic_comparison.png'")
        plt.show()
        
        # Plot SEPARATE optimization results (like original grid_optimization_results.png)
        # One for deterministic
        plot_optimization_results_single(
            iter_det, eff_hist_det, kw_det, var_det,
            metrics_det_init, metrics_det_final,
            n=n, case_name='Deterministic Weights',
            filename='grid_optimization_results_deterministic.png'
        )
        
        # One for stochastic
        plot_optimization_results_single(
            iter_stoch, eff_hist_stoch, kw_stoch, var_stoch,
            metrics_stoch_init, metrics_stoch_final,
            n=n, case_name='Stochastic Weights',
            filename='grid_optimization_results_stochastic.png'
        )
    
    # Plot grid with CV coloring
    plot_grid_network_stochastic(n, mA, W_mean, CV_matrix, obstacle_mask, grid_positions,
                                  P=P_stoch, pi=metrics_stoch_final['pi_W'],
                                  title="Optimal Policy (Stochastic Weights)\nThick=High P, Green=Reliable, Red=Unreliable",
                                  filename='grid_stochastic_policy.png')
    
    # Plot side-by-side comparison of optimal policies
    plot_optimal_policy_comparison(n, mA, CV_matrix, obstacle_mask, grid_positions,
                                    P_det, P_stoch,
                                    metrics_det_final['pi_W'], metrics_stoch_final['pi_W'],
                                    filename='optimal_policy_comparison.png')
    
    return {
        'deterministic_init': metrics_det_init,
        'stochastic_init': metrics_stoch_init,
        'deterministic_final': metrics_det_final,
        'stochastic_final': metrics_stoch_final,
        'CV_matrix': CV_matrix
    }


def test_varying_cv_levels():
    """
    Test how different CV levels affect the optimization.
    Shows that increasing edge weight variance increases path variance.
    """
    print("\n" + "="*80)
    print("TEST: IMPACT OF CV LEVELS ON NETWORK METRICS")
    print("="*80)
    
    n = 5
    obstacles = [(2, 2)]
    
    # Different CV configurations
    cv_configs = [
        {'cv_low': 0.0, 'cv_high': 0.0, 'high_frac': 0.0, 'label': 'Deterministic (CV=0)'},
        {'cv_low': 0.2, 'cv_high': 0.3, 'high_frac': 0.0, 'label': 'Low CV (0.2-0.3)'},
        {'cv_low': 0.5, 'cv_high': 0.7, 'high_frac': 0.0, 'label': 'Medium CV (0.5-0.7)'},
        {'cv_low': 0.3, 'cv_high': 1.2, 'high_frac': 0.4, 'label': 'Mixed CV (0.3-1.2, 40% high)'},
        {'cv_low': 0.5, 'cv_high': 2.0, 'high_frac': 0.5, 'label': 'High CV (0.5-2.0, 50% high)'},
    ]
    
    results = []
    
    print(f"\n{'Configuration':<35} {'Mean CV':<12} {'Eff_Idx':<15} {'K_W':<15} {'Variance':<15}")
    print("-" * 92)
    
    for config in cv_configs:
        mA, W, W2, CV_matrix, obstacle_mask, grid_positions = \
            generate_grid_network_stochastic(
                n, obstacles=obstacles,
                cv_low=config['cv_low'], 
                cv_high=config['cv_high'],
                high_cv_fraction=config['high_frac']
            )
        
        if config['cv_low'] == 0.0 and config['cv_high'] == 0.0:
            W2 = W ** 2  # Force deterministic
            mean_cv = 0.0
        else:
            mask = mA > 0
            mean_cv = np.mean(CV_matrix[mask])
        
        pi_hat = create_grid_target_distribution(n, obstacle_mask)
        
        problem = EfficiencyProblemInstanceStochastic(
            mA=mA, W=W, W2=W2,
            eta=1e-4, pi_hat=pi_hat,
            objective_type='maximize_efficiency',
            pi_penalty_weight=1e2
        )
        
        # Use uniform initial policy (no optimization needed to show effect)
        x_init = np.zeros(problem.d)
        for i, subset in enumerate(problem.neighborhoods):
            if len(subset) > 0:
                for idx in subset:
                    x_init[idx] = 1.0 / len(subset)
        
        P = x_to_matrix(x_init, problem.N, problem.edge_matrix, False)
        metrics = problem.evaluate_metrics(P)
        
        results.append({
            'config': config['label'],
            'mean_cv': mean_cv,
            'Eff_Idx': metrics['Eff_Idx'],
            'K_W': metrics['K_W'],
            'Net_Var': metrics['Net_Var']
        })
        
        print(f"{config['label']:<35} {mean_cv:<12.3f} {metrics['Eff_Idx']:<15.4f} {metrics['K_W']:<15.4f} {metrics['Net_Var']:<15.4f}")
    
    # Compute variance increase relative to deterministic
    base_var = results[0]['Net_Var']
    print(f"\n{'Configuration':<35} {'Variance Increase':<20} {'% Increase':<15}")
    print("-" * 70)
    for r in results:
        var_increase = r['Net_Var'] - base_var
        pct_increase = 100 * var_increase / base_var if base_var > 0 else 0
        print(f"{r['config']:<35} {var_increase:<20.4f} {pct_increase:+.2f}%")
    
    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = [r['config'] for r in results]
    mean_cvs = [r['mean_cv'] for r in results]
    efficiencies = [r['Eff_Idx'] for r in results]
    variances = [r['Net_Var'] for r in results]
    
    # Plot 1: CV vs Variance
    axes[0].scatter(mean_cvs, variances, s=100, c='blue', alpha=0.7)
    for i, label in enumerate(labels):
        axes[0].annotate(label.split('(')[0].strip(), (mean_cvs[i], variances[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    axes[0].set_xlabel('Mean Coefficient of Variation')
    axes[0].set_ylabel('Network Variance')
    axes[0].set_title('Edge Weight CV vs Path Variance')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Bar comparison
    x_pos = np.arange(len(labels))
    width = 0.35
    
    ax2 = axes[1]
    bars1 = ax2.bar(x_pos - width/2, efficiencies, width, label='Efficiency Index', color='blue', alpha=0.7)
    ax2.set_ylabel('Efficiency Index', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x_pos + width/2, variances, width, label='Variance', color='orange', alpha=0.7)
    ax2_twin.set_ylabel('Variance', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=20, ha='right')
    ax2.set_title('Impact of CV on Metrics')
    
    fig.legend([bars1, bars2], ['Efficiency', 'Variance'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('cv_level_comparison.png', dpi=150)
    print("\n✓ CV level comparison saved to 'cv_level_comparison.png'")
    plt.show()
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("GRID NETWORK SURVEILLANCE WITH STOCHASTIC WEIGHTS")
    print("="*80)
    print("""
This test demonstrates the difference between:
- DETERMINISTIC weights: W2 = W² (zero variance on edges)
- STOCHASTIC weights: W2 = W²(1 + CV²) (non-zero variance)

The coefficient of variation (CV = σ/μ) controls edge uncertainty:
- CV < 1: Reliable edges (low variability)  
- CV > 1: Unreliable edges (high variability, e.g., traffic)
""")
    
    # Run main comparison
    results = test_deterministic_vs_stochastic()
    
    # Run CV level sensitivity analysis
    cv_results = test_varying_cv_levels()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)