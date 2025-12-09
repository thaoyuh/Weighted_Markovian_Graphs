"""
Grid Network Surveillance Optimization with Obstacles.

This script demonstrates Markov chain optimization on a grid network
where some cells (nodes) are obstacles that cannot be traversed.
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
# GRID NETWORK GENERATION WITH OBSTACLES
# ============================================================================

def generate_grid_network(n, obstacles=None, diagonal=False):
    """
    Generate an n×n grid network adjacency and weight matrices.
    
    Parameters:
        n: Grid size (n×n nodes, total N = n² nodes)
        obstacles: List of (row, col) tuples indicating obstacle positions.
                   Obstacles cannot be traversed - all edges to/from them are removed.
        diagonal: If True, include diagonal neighbors (8-connectivity)
                  If False, only 4-directional neighbors (up/down/left/right)
    
    Returns:
        mA: Adjacency matrix (N×N where N = n²)
        W: Weight matrix (travel times, based on distance)
        obstacle_mask: Boolean array of size N, True for obstacle nodes
        grid_positions: Dict mapping node index to (row, col) position
    """
    N = n * n  # Total number of nodes
    mA = np.zeros((N, N))
    W = np.zeros((N, N))
    
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
        # 8-connectivity: all 8 neighbors
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4-dir
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonals
    else:
        # 4-connectivity: up, down, left, right
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Build adjacency matrix
    for i in range(N):
        ri, ci = idx_to_pos(i)
        
        # Skip if current node is an obstacle
        if obstacle_mask[i]:
            continue
        
        for dr, dc in offsets:
            rj, cj = ri + dr, ci + dc
            
            # Check bounds
            if 0 <= rj < n and 0 <= cj < n:
                j = pos_to_idx(rj, cj)
                
                # Skip if neighbor is an obstacle
                if obstacle_mask[j]:
                    continue
                
                mA[i, j] = 1
                
                # Weight = Euclidean distance (1.0 for 4-dir, sqrt(2) for diagonal)
                dist = np.sqrt(dr**2 + dc**2)
                W[i, j] = dist
    
    # Add some random variation to weights (travel time variability)
    np.random.seed(42)
    W = W * (1 + 0.2 * np.random.rand(N, N))  # ±20% variation
    W = W * mA  # Zero out non-edges
    
    return mA, W, obstacle_mask, grid_positions


def create_grid_target_distribution(n, obstacle_mask, priority_positions=None):
    """
    Create a target stationary distribution for the grid.
    
    Parameters:
        n: Grid size
        obstacle_mask: Boolean array, True for obstacle nodes
        priority_positions: Dict mapping (row, col) to priority weight (higher = more coverage)
    
    Returns:
        pi_hat: Target stationary distribution (normalized, zeros for obstacles)
    """
    N = n * n
    weights = np.ones(N)
    
    # Zero out obstacles
    weights[obstacle_mask] = 0
    
    # Apply priority positions
    if priority_positions is not None:
        for (r, c), priority in priority_positions.items():
            idx = r * n + c
            if not obstacle_mask[idx]:
                weights[idx] = priority
    
    # Normalize
    pi_hat = weights / weights.sum()
    return pi_hat


# ============================================================================
# MARKOV CHAIN CLASS
# ============================================================================

class MarkovChain:
    def __init__(self, mA, x=None, W=None, bUndirected=False):
        self.n = mA.shape[0]
        self.bUndirected = bUndirected
        self.mA = mA
        
        if W is not None:
            self.W = W
            self.W2 = W ** 2
        else:
            self.W = np.ones((self.n, self.n))
            self.W2 = np.ones((self.n, self.n))
        
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
        if self.Pi is None:
            self.compute_Pi()
        I = np.eye(self.n)
        self.Z = inv(I - self.P + self.Pi)
        return self.Z
    
    def compute_M(self):
        if self.Z is None:
            self.compute_Z()
        
        I = np.eye(self.n)
        Ones = np.ones((self.n, self.n))
        P_dot_W = np.multiply(self.P, self.W)
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
        if self.M is None:
            self.compute_M()
        
        I = np.eye(self.n)
        Ones = np.ones((self.n, self.n))
        P_dot_W = np.multiply(self.P, self.W)
        P_dot_W2 = np.multiply(self.P, self.W2)
        
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
        if self.pi_W is None:
            self.compute_pi_W()
        if self.M is None:
            self.compute_M()
        self.K_W = self.pi_W @ self.M @ self.pi_W.T
        return self.K_W
    
    def compute_network_variance(self):
        if self.pi_W is None:
            self.compute_pi_W()
        if self.V is None:
            self.compute_V()
        self.Net_Var = self.pi_W @ self.V @ self.pi_W.T
        return self.Net_Var
    
    def compute_efficiency_index(self):
        if self.K_W is None:
            self.compute_kemeny_W()
        if self.Net_Var is None:
            self.compute_network_variance()
        if self.K_W == 0:
            return np.inf
        self.Eff_Idx = self.Net_Var / self.K_W
        return self.Eff_Idx
    
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
# PROBLEM INSTANCE
# ============================================================================

class EfficiencyProblemInstance:
    def __init__(self, mA, W, eta=1e-4, pi_hat=None, 
                 objective_type='maximize_efficiency',
                 efficiency_target=None,
                 pi_penalty_weight=1e3):
        """
        Problem instance for efficiency index optimization.
        """
        self.mA = mA
        self.W = W
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
        self.objective_type = objective_type
        self.efficiency_target = efficiency_target
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
        """Build constraint matrices for row-stochastic constraint only."""
        N = self.N
        d = self.d
        
        # Only row-sum constraints
        A_row = np.zeros((N, d))
        for idx, (i, j) in enumerate(self.edge_matrix):
            A_row[i, idx] = 1
        b_row = np.ones(N)
        
        # Remove rows for nodes with no edges (obstacles)
        valid_rows = np.sum(A_row, axis=1) > 0
        A_row = A_row[valid_rows]
        b_row = b_row[valid_rows]
        
        self.A = A_row
        self.b = b_row
        self.m2 = A_row.shape[0]
        
        self.C = null_space(self.A).T
        if self.C.size == 0:
            self.C = np.eye(d) * 1e-10
            print("Warning: Problem is fully constrained.")
        
        print(f"Variables: {d}, Hard constraints: {self.m2}, Free dimensions: {self.C.shape[0]}")
        
        try:
            A_pinv = self.A.T @ np.linalg.inv(self.A @ self.A.T + 1e-10 * np.eye(len(self.A)))
            self.A_pinv_b = A_pinv @ self.b
            self.C__C_T_C_inv__C_T = self.C.T @ self.C
        except np.linalg.LinAlgError:
            self.A_pinv_b = np.zeros(d)
            self.C__C_T_C_inv__C_T = np.eye(d)
    
    def objective(self, P, mA_sample=None):
        """Compute objective: maximize efficiency - penalty for π deviation."""
        mc = MarkovChain(mA=self.mA, W=self.W, bUndirected=False)
        mc.x = MarkovChain.P_to_x(P, self.mA, self.bUndirected)
        
        try:
            Eff = mc.compute_efficiency_index()
            
            pi = mc.pi
            # Only penalize non-obstacle nodes
            valid = self.pi_hat > 0
            pi_error = np.sum((pi[valid] - self.pi_hat[valid]) ** 2)
            penalty = self.pi_penalty_weight * pi_error
            
            if self.objective_type == 'maximize_efficiency':
                return -Eff + penalty
            elif self.objective_type == 'target_efficiency':
                return (Eff - self.efficiency_target) ** 2 + penalty
            else:
                raise ValueError(f"Unknown objective_type: {self.objective_type}")
        
        except (np.linalg.LinAlgError, RuntimeWarning, ValueError):
            return 1e10
    
    def evaluate_metrics(self, P):
        """Evaluate all metrics for analysis."""
        mc = MarkovChain(mA=self.mA, W=self.W, bUndirected=False)
        mc.x = MarkovChain.P_to_x(P, self.mA, self.bUndirected)
        
        mc.compute_efficiency_index()
        
        valid = self.pi_hat > 0
        pi_error = np.sqrt(np.sum((mc.pi[valid] - self.pi_hat[valid]) ** 2))
        
        return {
            'pi': mc.pi,
            'pi_W': mc.pi_W,
            'K_W': mc.K_W,
            'Net_Var': mc.Net_Var,
            'Eff_Idx': mc.Eff_Idx,
            'M_diag': np.diag(mc.M),
            'pi_error': pi_error,
            'pi_max_error': np.max(np.abs(mc.pi[valid] - self.pi_hat[valid])),
        }


# ============================================================================
# SPSA OPTIMIZATION
# ============================================================================

def g_spsa_efficiency(problem, x, eta_perturb):
    d_free = problem.C.shape[0]
    if d_free == 0:
        return np.zeros(problem.d)
    
    Delta = np.random.choice([-1, 1], size=d_free)
    direction = problem.C.T @ Delta
    
    x_plus = np.clip(x + eta_perturb * direction, problem.eta, 1 - problem.eta)
    x_min = np.clip(x - eta_perturb * direction, problem.eta, 1 - problem.eta)
    
    P_plus = x_to_matrix(x_plus, problem.N, problem.edge_matrix, problem.bUndirected)
    P_min = x_to_matrix(x_min, problem.N, problem.edge_matrix, problem.bUndirected)
    
    J_plus = problem.objective(P_plus)
    J_min = problem.objective(P_min)
    
    gradient_est = (J_plus - J_min) / (2 * eta_perturb * Delta)
    return problem.C.T @ gradient_est


def projection(x_to_proj, problem, eta):
    return projection_markov(x_to_proj, eta=eta, neighborhoods=problem.neighborhoods, mA=problem.mA)


def solve_spsa_efficiency(problem, x_init, max_iter=10000,
                          a=0.01, a_eps=100, r_epsilon=0.602,
                          e=1e-6, r_nu=0.101,
                          obj_interval=100, verbose=True):
    
    x = x_init.copy()
    
    eff_hist = []
    kw_hist = []
    var_hist = []
    iter_hist = []
    
    x_sum = np.zeros_like(x)
    
    P_init = x_to_matrix(x, problem.N, problem.edge_matrix, problem.bUndirected)
    initial_metrics = problem.evaluate_metrics(P_init)
    
    eff_hist.append(initial_metrics['Eff_Idx'])
    kw_hist.append(initial_metrics['K_W'])
    var_hist.append(initial_metrics['Net_Var'])
    iter_hist.append(0)
    
    if verbose:
        print(f"Initial Efficiency: {initial_metrics['Eff_Idx']:.6f}")
        print(f"Initial K_W: {initial_metrics['K_W']:.6f}")
        print(f"Initial Variance: {initial_metrics['Net_Var']:.6f}")
        print(f"\nStarting SPSA optimization...")
        print(f"{'Iter':<8} {'Efficiency':<12} {'K_W':<12} {'Variance':<12} {'Status':<15}")
        print("-" * 70)
    
    best_eff = initial_metrics['Eff_Idx']
    best_x = x.copy()
    
    for k in range(1, max_iter):
        alpha_k = a / (a_eps + k) ** r_epsilon
        eta_k = e / k ** r_nu
        
        grad = g_spsa_efficiency(problem, x, eta_k)
        x_new = x - alpha_k * grad
        
        if np.any(x_new < problem.eta) or np.any(x_new > 1 - problem.eta):
            x_new = projection(x_new, problem, problem.eta)
        
        x = x_new
        x_sum += x
        
        if k % obj_interval == 0:
            x_avg = x_sum / k
            
            P_avg = x_to_matrix(x_avg, problem.N, problem.edge_matrix, problem.bUndirected)
            metrics = problem.evaluate_metrics(P_avg)
            
            eff_hist.append(metrics['Eff_Idx'])
            kw_hist.append(metrics['K_W'])
            var_hist.append(metrics['Net_Var'])
            iter_hist.append(k)
            
            if problem.objective_type == 'maximize_efficiency':
                if metrics['Eff_Idx'] > best_eff:
                    best_eff = metrics['Eff_Idx']
                    best_x = x_avg.copy()
                    status = "✓ New best!"
                else:
                    status = ""
            else:
                current_error = abs(metrics['Eff_Idx'] - problem.efficiency_target)
                best_error = abs(best_eff - problem.efficiency_target)
                if current_error < best_error:
                    best_eff = metrics['Eff_Idx']
                    best_x = x_avg.copy()
                    status = "✓ Closer!"
                else:
                    status = ""
            
            if verbose:
                print(f"{k:<8} {metrics['Eff_Idx']:<12.6f} {metrics['K_W']:<12.6f} {metrics['Net_Var']:<12.6f} {status:<15}")
    
    x_final = x_sum / max_iter
    P_final = x_to_matrix(x_final, problem.N, problem.edge_matrix, problem.bUndirected)
    final_metrics = problem.evaluate_metrics(P_final)
    
    if verbose:
        print("-" * 70)
        print(f"Final Efficiency: {final_metrics['Eff_Idx']:.6f}")
        print(f"Best Efficiency found: {best_eff:.6f}")
        if problem.objective_type == 'maximize_efficiency':
            improvement = best_eff - eff_hist[0]
            pct_change = (best_eff / eff_hist[0] - 1) * 100
            print(f"Improvement: {improvement:.6f} ({pct_change:.2f}% increase)")
    
    return (np.array(iter_hist), np.array(eff_hist), np.array(kw_hist), 
            np.array(var_hist), best_x, initial_metrics)


# ============================================================================
# GRID NETWORK PLOTTING
# ============================================================================

def plot_grid_network(n, mA, W, obstacle_mask, grid_positions, P=None, pi=None, 
                      title="Grid Network", filename=None):
    """
    Plot the grid network.
    - Nodes are smaller to reduce clutter.
    - Edges use exponential scaling for high contrast between low/high probabilities.
    - Curvature added to clear show directions.
    """
    N = n * n
    fig, ax = plt.subplots(figsize=(10, 10)) # Slightly smaller figure relative to grid
    
    # Node positions (flip y so (0,0) is top-left like matrix indexing)
    pos = {}
    for i in range(N):
        r, c = grid_positions[i]
        pos[i] = (c, n - 1 - r)
    
    # -------------------------------------------------------
    # 1. DRAW EDGES (The Transitions)
    # -------------------------------------------------------
    # Node radius definition (used for drawing and for edge gaps)
    NODE_RADIUS = 0.2
    
    for i in range(N):
        for j in range(N):
            if mA[i, j] > 0:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                
                # Get probability
                prob = P[i, j] if P is not None else 0.25
                
                # --- MODIFICATION: Extreme Contrast Calculation ---
                if P is not None:
                    # 1. Thickness: Exponential scaling (prob^2). 
                    # This suppresses low values and exaggerates high values.
                    linewidth = 0.5 + (prob**2) * 8.0 
                    
                    # 2. Darkness (Alpha): Linear scaling
                    # Low prob = very faint (0.1), High prob = solid (1.0)
                    alpha = 0.15 + (prob * 0.85)
                    
                    # 3. Arrow Head Size: Scale slightly with prob so tiny lines don't have huge heads
                    arrow_size = 10 + (prob * 15)
                    
                else:
                    linewidth = 1.0
                    alpha = 0.5
                    arrow_size = 15
                
                # Draw arrow
                # connectionstyle 'arc3,rad=0.2' creates a distinct curve
                # allowing us to see i->j and j->i separately
                ax.annotate(
                    '', 
                    xy=(x2, y2), 
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='-|>',             # Sharp arrow head
                        mutation_scale=arrow_size,    # Size of arrow head
                        connectionstyle='arc3,rad=0.2', # Curve amount
                        color='black',                # Edge color (alpha controls darkness)
                        lw=linewidth,
                        alpha=alpha,
                        shrinkA=12, # Gap from start node (calibrated for small radius)
                        shrinkB=12, # Gap from end node
                    )
                )
    
    # -------------------------------------------------------
    # 2. DRAW NODES (Smaller size, colored by coverage)
    # -------------------------------------------------------
    for i in range(N):
        x, y = pos[i]
        
        if obstacle_mask[i]:
            # Obstacle
            circle = plt.Circle((x, y), NODE_RADIUS, color='#303030', ec='black', lw=1, zorder=10)
            ax.add_patch(circle)
            # Smaller text for smaller nodes
            ax.text(x, y, '✕', fontsize=10, ha='center', va='center', 
                   color='#ff4444', fontweight='bold', zorder=11)
        else:
            # Color calculation
            if pi is not None:
                valid_pi = pi[~obstacle_mask]
                pi_min, pi_max = valid_pi.min(), valid_pi.max()
                denom = pi_max - pi_min
                pi_norm = (pi[i] - pi_min) / denom if denom > 0 else 0.5
                
                # Darker blue for high coverage
                node_color = plt.cm.Blues(0.2 + (pi_norm * 0.8))
                text_color = 'white' if pi_norm > 0.6 else 'black'
            else:
                node_color = 'skyblue'
                text_color = 'black'
            
            # Draw Node
            circle = plt.Circle((x, y), NODE_RADIUS, color=node_color, ec='black', lw=1, zorder=10)
            ax.add_patch(circle)
            
            # Node ID (Smaller font)
            ax.text(x, y, str(i), fontsize=7, ha='center', va='center', 
                   fontweight='bold', color=text_color, zorder=11)
            
            # Pi value label (moved slightly closer due to smaller nodes)
            if pi is not None and pi[i] > 0:
                ax.text(x, y - 0.25, f'{pi[i]:.2f}', fontsize=6, ha='center', 
                       va='top', color='black', fontweight='bold', zorder=12)
    
    # Grid lines
    for i in range(n):
        ax.axhline(y=i, color='lightgray', linestyle=':', alpha=0.4, zorder=1)
        ax.axvline(x=i, color='lightgray', linestyle=':', alpha=0.4, zorder=1)
    
    # Setup Axes
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#303030', edgecolor='black', label='Obstacle'),
        mpatches.Patch(facecolor=plt.cm.Blues(0.2), edgecolor='black', label='Low Coverage'),
        mpatches.Patch(facecolor=plt.cm.Blues(0.9), edgecolor='black', label='High Coverage'),
    ]
    if P is not None:
        # Custom lines for legend to show thickness difference
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=0.5, alpha=0.3, label='Low Prob P(j|i)'))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=4, alpha=1.0, label='High Prob P(j|i)'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to '{filename}'")
    
    plt.show()
    return fig

# ============================================================================
# MAIN TEST: 5×5 GRID WITH 1 OBSTACLE
# ============================================================================

def test_grid_with_obstacle():
    """Test surveillance optimization on a 5×5 grid with 1 obstacle."""
    
    print("="*80)
    print("TEST: 5×5 GRID NETWORK WITH OBSTACLE")
    print("="*80)
    
    # Grid parameters
    n = 6  # 5×5 grid
    
    # Place obstacle in the middle of the grid
    obstacles = [(1, 1), (3,4)]  # Center cell
    
    print(f"\nGrid size: {n}×{n} = {n*n} nodes")
    print(f"Obstacle positions: {obstacles}")
    
    # Generate network
    mA, W, obstacle_mask, grid_positions = generate_grid_network(n, obstacles=obstacles)
    
    num_edges = int(mA.sum())
    num_valid_nodes = (~obstacle_mask).sum()
    print(f"Valid (non-obstacle) nodes: {num_valid_nodes}")
    print(f"Total edges: {num_edges}")
    
    # Create target distribution - higher priority for corner nodes (entry points)
    # and the center region around the obstacle
    priority_positions = {
        (0, 0): 2.0,  # Corners
        (0, 4): 2.0,
        (4, 0): 2.0,
        (4, 4): 2.0,
        (1, 2): 1.5,  # Around obstacle
        (2, 1): 1.5,
        (2, 3): 1.5,
        (3, 2): 1.5,
    }
    
    pi_hat = create_grid_target_distribution(n, obstacle_mask, priority_positions)
    
    print(f"\nTarget π (reshaped to grid):")
    pi_grid = pi_hat.reshape(n, n)
    for r in range(n):
        row_str = " ".join([f"{pi_grid[r,c]:.3f}" if not obstacle_mask[r*n+c] else " OBS " 
                           for c in range(n)])
        print(f"  {row_str}")
    
    # Problem setup
    eta = 1e-4
    problem = EfficiencyProblemInstance(
        mA=mA, 
        W=W, 
        eta=eta,
        pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        pi_penalty_weight=5e2,
    )
    
    # Compute initial uniform transition probabilities for each node
    # Each node distributes probability equally among its neighbors
    x_init = np.zeros(problem.d)
    for i, subset in enumerate(problem.neighborhoods):
        if len(subset) > 0:
            for idx in subset:
                x_init[idx] = 1.0 / len(subset)
    
    print(f"\nInitial policy: uniform transitions to neighbors")
    
    # Plot initial network structure
    print("\n" + "-"*40)
    print("Plotting grid network structure...")
    
    plot_grid_network(n, mA, W, obstacle_mask, grid_positions,
                      P=None, pi=None,
                      title=f"{n}×{n} Grid Network with Obstacle (Structure)",
                      filename='grid_structure.png')
    
    # Evaluate initial metrics
    P_init = x_to_matrix(x_init, problem.N, problem.edge_matrix, problem.bUndirected)
    initial_metrics = problem.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics:")
    print(f"  Efficiency Index: {initial_metrics['Eff_Idx']:.6f}")
    print(f"  K_W (Mean Patrol Time): {initial_metrics['K_W']:.6f}")
    print(f"  Variance: {initial_metrics['Net_Var']:.6f}")
    print(f"  π error (L2): {initial_metrics['pi_error']:.6f}")
    
    # Plot initial policy
    plot_grid_network(n, mA, W, obstacle_mask, grid_positions,
                      P=P_init, pi=initial_metrics['pi'],
                      title="Initial Policy (Uniform Transitions)",
                      filename='grid_initial_policy.png')
    
    # Run SPSA optimization
    print("\n" + "="*80)
    print("Running SPSA Optimization...")
    print("="*80)
    
    iter_hist, eff_hist, kw_hist, var_hist, best_x, _ = solve_spsa_efficiency(
        problem=problem,
        x_init=x_init,
        max_iter=8000,
        a=0.1,         # <--- Step size numerator
        a_eps=500,     # <--- Step size denominator constant
        e=1e-4,        # <--- Perturbation size numerator
        r_nu=0.101,    # <--- Perturbation size exponent
        obj_interval=200,
        verbose=True
    )
    
    # Final analysis
    P_final = x_to_matrix(best_x, problem.N, problem.edge_matrix, problem.bUndirected)
    final_metrics = problem.evaluate_metrics(P_final)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nMetrics Comparison:")
    print(f"{'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Efficiency Index λ':<25} {initial_metrics['Eff_Idx']:<15.6f} {final_metrics['Eff_Idx']:<15.6f} {final_metrics['Eff_Idx'] - initial_metrics['Eff_Idx']:+.6f}")
    print(f"{'Mean Patrol Time K_W':<25} {initial_metrics['K_W']:<15.6f} {final_metrics['K_W']:<15.6f} {final_metrics['K_W'] - initial_metrics['K_W']:+.6f}")
    print(f"{'Path Variance V_W':<25} {initial_metrics['Net_Var']:<15.6f} {final_metrics['Net_Var']:<15.6f} {final_metrics['Net_Var'] - initial_metrics['Net_Var']:+.6f}")
    print(f"{'Coverage Error ||π-π₀||':<25} {initial_metrics['pi_error']:<15.6f} {final_metrics['pi_error']:<15.6f} {final_metrics['pi_error'] - initial_metrics['pi_error']:+.6f}")
    
    pct_improvement = (final_metrics['Eff_Idx'] / initial_metrics['Eff_Idx'] - 1) * 100
    print(f"\nEfficiency improvement: {pct_improvement:.2f}%")
    
    # Plot final policy
    print("\n" + "-"*40)
    print("Plotting optimal policy...")
    
    plot_grid_network(n, mA, W, obstacle_mask, grid_positions,
                      P=P_final, pi=final_metrics['pi'],
                      title="Optimal Surveillance Policy",
                      filename='grid_optimal_policy.png')
    
    # Plot optimization convergence
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
    
    plt.suptitle(f'{n}×{n} Grid Network with Obstacle - Optimization Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grid_optimization_results.png', dpi=150)
    print("\n✓ Optimization plot saved to 'grid_optimization_results.png'")
    plt.show()
    
    return best_x, final_metrics


if __name__ == "__main__":
    np.random.seed(42)
    print("Running 5×5 Grid Network Test with Obstacle\n")
    best_x, metrics = test_grid_with_obstacle()
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)