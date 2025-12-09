"""
Grid Network Surveillance Optimization

Creates an n×n grid network with obstacles and optimizes surveillance policy.
Following the paper's approach for handling fixed vs. optimizable edges.
"""

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product

# ============================================================================
# GRID NETWORK GENERATION
# ============================================================================

def generate_grid_network(n, obstacles=None):
    """
    Generate an n×n grid network with optional obstacles.
    
    Parameters:
        n: Grid size (n×n nodes)
        obstacles: List of node indices that are obstacles (removed from network)
                   Node indices are 0 to n²-1, row-major order
                   e.g., for 5×5 grid, center node is index 12
    
    Returns:
        mA: Adjacency matrix (N×N where N = n² - len(obstacles))
        node_positions: Dict mapping node index to (row, col) position
        original_to_new: Dict mapping original node index to new index
        new_to_original: Dict mapping new node index to original index
    """
    if obstacles is None:
        obstacles = []
    
    total_nodes = n * n
    
    # Create full grid adjacency (4-connectivity: up, down, left, right)
    mA_full = np.zeros((total_nodes, total_nodes))
    
    for i in range(total_nodes):
        row, col = i // n, i % n
        
        # Up neighbor
        if row > 0:
            mA_full[i, i - n] = 1
        # Down neighbor
        if row < n - 1:
            mA_full[i, i + n] = 1
        # Left neighbor
        if col > 0:
            mA_full[i, i - 1] = 1
        # Right neighbor
        if col < n - 1:
            mA_full[i, i + 1] = 1
    
    # Remove obstacle nodes
    keep_nodes = [i for i in range(total_nodes) if i not in obstacles]
    N = len(keep_nodes)
    
    # Create mapping
    original_to_new = {orig: new for new, orig in enumerate(keep_nodes)}
    new_to_original = {new: orig for new, orig in enumerate(keep_nodes)}
    
    # Extract submatrix
    mA = np.zeros((N, N))
    for i_new, i_orig in enumerate(keep_nodes):
        for j_new, j_orig in enumerate(keep_nodes):
            mA[i_new, j_new] = mA_full[i_orig, j_orig]
    
    # Node positions (row, col) in grid coordinates
    node_positions = {}
    for new_idx, orig_idx in new_to_original.items():
        row, col = orig_idx // n, orig_idx % n
        node_positions[new_idx] = (row, col)
    
    return mA, node_positions, original_to_new, new_to_original, n, obstacles


def generate_travel_times(mA, node_positions, seed=42):
    """
    Generate travel time matrix W based on grid distances.
    Adds some randomness to make it asymmetric.
    
    Parameters:
        mA: Adjacency matrix
        node_positions: Dict mapping node index to (row, col)
        seed: Random seed
    
    Returns:
        W: Travel time matrix (same shape as mA)
    """
    np.random.seed(seed)
    N = mA.shape[0]
    W = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if mA[i, j] > 0:
                # Base travel time = 1, with random variation
                W[i, j] = 1.0 + np.random.uniform(0, 1)
    
    return W


# ============================================================================
# MARKOV CHAIN CLASS (same as before)
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
        
        self.edge_matrix = self._create_edge_matrix(mA)
        
        if x is not None:
            self.x = x
    
    @staticmethod
    def _create_edge_matrix(mA):
        indices = np.nonzero(mA)
        num_edges = indices[0].shape[0]
        E = np.zeros((num_edges, 2), dtype=int)
        E[:, 0] = indices[0]
        E[:, 1] = indices[1]
        return E
    
    @property
    def P(self):
        P_matrix = np.zeros((self.n, self.n))
        P_matrix[self.edge_matrix[:, 0], self.edge_matrix[:, 1]] = self.x
        return P_matrix
    
    @staticmethod
    def P_to_x(P, mA, bUndirected=False):
        N, _ = mA.shape
        return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1])
    
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
        Xi_inv = np.diag(1.0 / self.pi)
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
        M2_diag_vals = (val_A + vec_B) / self.pi
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
# HELPER FUNCTIONS
# ============================================================================

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


def x_to_matrix(x, N, edge_matrix, bUndirected=False):
    P = np.zeros((N, N))
    P[edge_matrix[:, 0], edge_matrix[:, 1]] = x
    return P


# ============================================================================
# PROBLEM INSTANCE FOR GRID SURVEILLANCE
# ============================================================================

class GridSurveillanceProblem:
    def __init__(self, mA, W, eta=1e-4, pi_hat=None, pi_penalty_weight=1e3):
        """
        Problem instance for grid surveillance optimization.
        
        Parameters:
            mA: Adjacency matrix
            W: Travel time matrix
            eta: Lower bound for probabilities
            pi_hat: Target stationary distribution
            pi_penalty_weight: Penalty weight for π deviation
        """
        self.mA = mA
        self.W = W
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
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
        
        self.A = A_row
        self.b = b_row
        self.m2 = A_row.shape[0]
        
        self.C = null_space(self.A).T
        if self.C.size == 0:
            self.C = np.eye(d) * 1e-10
        
        print(f"Grid network: {N} nodes, {d} edges")
        print(f"Variables: {d}, Hard constraints: {self.m2}, Free dimensions: {self.C.shape[0]}")
        
        try:
            A_pinv = self.A.T @ np.linalg.inv(self.A @ self.A.T + 1e-10 * np.eye(len(self.A)))
            self.A_pinv_b = A_pinv @ self.b
            self.C__C_T_C_inv__C_T = self.C.T @ self.C
        except np.linalg.LinAlgError:
            self.A_pinv_b = np.zeros(d)
            self.C__C_T_C_inv__C_T = np.eye(d)
    
    def objective(self, P, mA_sample=None):
        mc = MarkovChain(mA=self.mA, W=self.W, bUndirected=False)
        mc.x = MarkovChain.P_to_x(P, self.mA, self.bUndirected)
        
        try:
            Eff = mc.compute_efficiency_index()
            pi = mc.pi
            pi_error = np.sum((pi - self.pi_hat) ** 2)
            penalty = self.pi_penalty_weight * pi_error
            return -Eff + penalty
        except (np.linalg.LinAlgError, RuntimeWarning, ValueError):
            return 1e10
    
    def evaluate_metrics(self, P):
        mc = MarkovChain(mA=self.mA, W=self.W, bUndirected=False)
        mc.x = MarkovChain.P_to_x(P, self.mA, self.bUndirected)
        mc.compute_efficiency_index()
        pi_error = np.sqrt(np.sum((mc.pi - self.pi_hat) ** 2))
        
        return {
            'pi': mc.pi,
            'pi_W': mc.pi_W,
            'K_W': mc.K_W,
            'Net_Var': mc.Net_Var,
            'Eff_Idx': mc.Eff_Idx,
            'M_diag': np.diag(mc.M),
            'pi_error': pi_error,
            'pi_max_error': np.max(np.abs(mc.pi - self.pi_hat)),
        }


# ============================================================================
# SPSA OPTIMIZATION
# ============================================================================

def proj_simplex(v, c=1, tol=1e-8):
    N = len(v)
    vU = np.sort(v)[::-1]
    cssv = np.cumsum(vU)
    l = [k+1 for k in range(N) if (cssv[k] - c) / (k + 1) < vU[k]]
    if not l:
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
            x_proj.extend(proj_simplex(x[subset], c=c).tolist())
    return np.array(x_proj) + eta


def g_spsa(problem, x, eta_perturb):
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


def solve_spsa(problem, x_init, max_iter=50000, a=0.5, a_eps=1000, 
               r_epsilon=0.602, e=1e-4, r_nu=0.101, obj_interval=2000, verbose=True):
    
    x = x_init.copy()
    x_hist = np.zeros((max_iter, len(x)))
    x_hist[0] = x
    
    iter_hist = [0]
    eff_hist = []
    kw_hist = []
    var_hist = []
    
    # Initial evaluation
    P_init = x_to_matrix(x, problem.N, problem.edge_matrix, problem.bUndirected)
    initial_metrics = problem.evaluate_metrics(P_init)
    eff_hist.append(initial_metrics['Eff_Idx'])
    kw_hist.append(initial_metrics['K_W'])
    var_hist.append(initial_metrics['Net_Var'])
    
    if verbose:
        print(f"Initial Efficiency: {initial_metrics['Eff_Idx']:.6f}")
        print(f"Initial K_W: {initial_metrics['K_W']:.6f}")
        print(f"Initial Variance: {initial_metrics['Net_Var']:.6f}")
        print(f"\nStarting SPSA optimization...")
        print(f"{'Iter':<8} {'Efficiency':<12} {'K_W':<12} {'Variance':<12} {'Status':<15}")
        print("-" * 70)
    
    best_eff = initial_metrics['Eff_Idx']
    best_x = x.copy()
    
    for k in range(max_iter - 1):
        alpha_k = a / (a_eps + k + 1) ** r_epsilon
        eta_k = e / (k + 1) ** r_nu
        
        grad = g_spsa(problem, x, eta_k)
        x_new = x - alpha_k * grad
        
        if np.any(x_new < problem.eta) or np.any(x_new > 1 - problem.eta):
            x_new = projection_markov(x_new, problem.eta, problem.neighborhoods, problem.mA)
        
        x = x_new
        x_hist[k + 1] = x
        
        if (k + 1) % obj_interval == 0:
            start_avg = max(0, int(0.5 * (k + 1)))
            x_avg = np.mean(x_hist[start_avg:k+2], axis=0)
            
            P_avg = x_to_matrix(x_avg, problem.N, problem.edge_matrix, problem.bUndirected)
            metrics = problem.evaluate_metrics(P_avg)
            
            iter_hist.append(k + 1)
            eff_hist.append(metrics['Eff_Idx'])
            kw_hist.append(metrics['K_W'])
            var_hist.append(metrics['Net_Var'])
            
            if metrics['Eff_Idx'] > best_eff:
                best_eff = metrics['Eff_Idx']
                best_x = x_avg.copy()
                status = "✓ New best!"
            else:
                status = ""
            
            if verbose:
                print(f"{k+1:<8} {metrics['Eff_Idx']:<12.6f} {metrics['K_W']:<12.6f} {metrics['Net_Var']:<12.6f} {status:<15}")
    
    if verbose:
        print("-" * 70)
        print(f"Final Efficiency: {eff_hist[-1]:.6f}")
        print(f"Best Efficiency found: {best_eff:.6f}")
        improvement = best_eff - initial_metrics['Eff_Idx']
        pct = (best_eff / initial_metrics['Eff_Idx'] - 1) * 100
        print(f"Improvement: {improvement:.6f} ({pct:.2f}% increase)")
    
    return np.array(iter_hist), np.array(eff_hist), np.array(kw_hist), np.array(var_hist), best_x, initial_metrics


# ============================================================================
# GRID NETWORK PLOTTING
# ============================================================================

def plot_grid_network(n, obstacles, node_positions, new_to_original, mA, W, 
                      P=None, pi=None, title="Grid Network", filename=None):
    """
    Plot the grid network with obstacles.
    
    Parameters:
        n: Grid size
        obstacles: List of obstacle node indices (in original coordinates)
        node_positions: Dict mapping new node index to (row, col)
        new_to_original: Dict mapping new index to original index
        mA: Adjacency matrix
        W: Travel time matrix
        P: Transition matrix (optional)
        pi: Stationary distribution (optional)
        title: Plot title
        filename: Save filename (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Grid spacing
    spacing = 1.0
    
    # Draw grid cells (background)
    for row in range(n):
        for col in range(n):
            orig_idx = row * n + col
            x_pos = col * spacing
            y_pos = (n - 1 - row) * spacing  # Flip y so row 0 is at top
            
            if orig_idx in obstacles:
                # Draw obstacle as dark gray square
                rect = patches.Rectangle((x_pos - 0.4, y_pos - 0.4), 0.8, 0.8,
                                         facecolor='#404040', edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_pos, y_pos, 'X', fontsize=16, ha='center', va='center', 
                       color='white', fontweight='bold')
    
    # Draw edges
    N = mA.shape[0]
    for i in range(N):
        for j in range(N):
            if mA[i, j] > 0:
                row_i, col_i = node_positions[i]
                row_j, col_j = node_positions[j]
                
                x1 = col_i * spacing
                y1 = (n - 1 - row_i) * spacing
                x2 = col_j * spacing
                y2 = (n - 1 - row_j) * spacing
                
                # Edge properties
                if P is not None:
                    linewidth = 1 + P[i, j] * 8
                    alpha = 0.3 + P[i, j] * 0.7
                else:
                    linewidth = 1.5
                    alpha = 0.5
                
                # Color by travel time
                w_normalized = (W[i, j] - W[W > 0].min()) / (W[W > 0].max() - W[W > 0].min() + 1e-6)
                color = plt.cm.Reds(0.3 + w_normalized * 0.7)
                
                # Draw arrow
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Shorten arrow to not overlap with nodes
                    shrink = 0.25
                    x1_adj = x1 + dx * shrink / length
                    y1_adj = y1 + dy * shrink / length
                    x2_adj = x2 - dx * shrink / length
                    y2_adj = y2 - dy * shrink / length
                    
                    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                               arrowprops=dict(arrowstyle='->', color=color,
                                             lw=linewidth, alpha=alpha,
                                             connectionstyle='arc3,rad=0.1'))
    
    # Draw nodes
    for new_idx in range(N):
        row, col = node_positions[new_idx]
        x_pos = col * spacing
        y_pos = (n - 1 - row) * spacing
        
        # Node color based on pi
        if pi is not None:
            pi_normalized = (pi[new_idx] - pi.min()) / (pi.max() - pi.min() + 1e-6)
            node_color = plt.cm.Blues(0.3 + pi_normalized * 0.7)
        else:
            node_color = '#6699cc'
        
        circle = plt.Circle((x_pos, y_pos), 0.2, facecolor=node_color, 
                            edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Node label (original index)
        orig_idx = new_to_original[new_idx]
        ax.text(x_pos, y_pos, str(orig_idx), fontsize=10, ha='center', va='center',
               fontweight='bold', zorder=11)
        
        # Show pi value below node
        if pi is not None:
            ax.text(x_pos, y_pos - 0.35, f'{pi[new_idx]:.3f}', fontsize=8, 
                   ha='center', va='top', color='darkblue')
    
    # Legend
    legend_text = "Node labels = Original index\n"
    if P is not None:
        legend_text += "Edge thickness = Transition prob.\n"
    legend_text += "Edge color = Travel time (darker = longer)"
    if pi is not None:
        legend_text += "\nNode color = Coverage π (darker = higher)"
    legend_text += "\nGray square = Obstacle"
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(-0.7, (n - 1) * spacing + 0.7)
    ax.set_ylim(-0.7, (n - 1) * spacing + 0.7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Grid plot saved to '{filename}'")
    
    plt.show()
    return fig


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_grid_surveillance():
    """Test surveillance optimization on 5x5 grid with center obstacle."""
    
    print("="*80)
    print("GRID NETWORK SURVEILLANCE OPTIMIZATION")
    print("="*80)
    
    # Generate 5x5 grid with center obstacle (node 12)
    n = 5
    center_node = (n * n) // 2  # Node 12 for 5x5 grid
    obstacles = [center_node]
    
    print(f"\nGrid size: {n}×{n} = {n*n} nodes")
    print(f"Obstacle at node: {center_node} (center)")
    print(f"Active nodes: {n*n - len(obstacles)}")
    
    # Generate network
    mA, node_positions, orig_to_new, new_to_orig, grid_size, obs = generate_grid_network(n, obstacles)
    N = mA.shape[0]
    
    print(f"\nNetwork structure:")
    print(f"  Nodes: {N}")
    print(f"  Edges: {int(mA.sum())}")
    
    # Generate travel times
    W = generate_travel_times(mA, node_positions, seed=42)
    
    # Target stationary distribution (uniform for now)
    pi_hat = np.ones(N) / N
    
    print(f"\nTarget stationary distribution: uniform (π₀ = 1/{N} = {1/N:.4f})")
    
    # Create problem instance
    problem = GridSurveillanceProblem(
        mA=mA,
        W=W,
        eta=1e-4,
        pi_hat=pi_hat,
        pi_penalty_weight=1e3
    )
    
    # Initial policy: uniform transitions
    x_init = np.zeros(problem.d)
    for i, subset in enumerate(problem.neighborhoods):
        if subset:
            for idx in subset:
                x_init[idx] = 1.0 / len(subset)
    
    # Plot initial network
    print("\n" + "-"*40)
    print("Plotting initial grid network...")
    P_init = x_to_matrix(x_init, N, problem.edge_matrix, problem.bUndirected)
    init_metrics = problem.evaluate_metrics(P_init)
    
    plot_grid_network(n, obstacles, node_positions, new_to_orig, mA, W,
                      P=None, pi=None,
                      title=f"{n}×{n} Grid Network (Obstacle at Center)",
                      filename='grid_network_structure.png')
    
    # Run optimization
    print("\n" + "="*80)
    iter_hist, eff_hist, kw_hist, var_hist, best_x, initial_metrics = solve_spsa(
        problem=problem,
        x_init=x_init,
        max_iter=50000,
        a=0.5,
        a_eps=1000,
        e=1e-4,
        r_nu=0.101,
        obj_interval=2500,
        verbose=True
    )
    
    # Final results
    P_final = x_to_matrix(best_x, N, problem.edge_matrix, problem.bUndirected)
    final_metrics = problem.evaluate_metrics(P_final)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
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
    
    # Plot optimal policy on grid
    print("\n" + "-"*40)
    print("Plotting optimal surveillance policy...")
    plot_grid_network(n, obstacles, node_positions, new_to_orig, mA, W,
                      P=P_final, pi=final_metrics['pi'],
                      title="Optimal Surveillance Policy on Grid",
                      filename='grid_optimal_policy.png')
    
    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Efficiency
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
    
    # Plot 2: Mean Patrol Time
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
    axes[1, 0].set_title(r'Path Variance ($\uparrow$ higher = more unpredictable)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mean-Variance Tradeoff
    scatter = axes[1, 1].scatter(kw_hist, var_hist, c=iter_hist, cmap='viridis', s=50, alpha=0.7)
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
    
    plt.tight_layout()
    plt.savefig('grid_surveillance_results.png', dpi=150)
    print("\n✓ Convergence plot saved to 'grid_surveillance_results.png'")
    plt.show()
    
    return best_x, final_metrics


if __name__ == "__main__":
    np.random.seed(42)
    print("Running Grid Surveillance Optimization\n")
    best_x, metrics = test_grid_surveillance()
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)