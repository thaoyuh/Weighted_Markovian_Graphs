"""
Improved test with 5-node network for more optimization freedom.
Also fixes the plotting bug.
"""

import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import null_space, qr
from itertools import product
import matplotlib.pyplot as plt


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
# PROBLEM INSTANCE (WITHOUT stationary distribution constraint)
# ============================================================================

class EfficiencyProblemInstance:
    def __init__(self, mA, W, eta=1e-4, pi_hat=None, 
                 objective_type='maximize_efficiency',
                 efficiency_target=None,
                 pi_penalty_weight=1e3):  # Penalty weight for π deviation
        """
        Problem instance for efficiency index optimization.
        
        The stationary distribution π = π₀ is enforced as a SOFT constraint
        via penalty in the objective function. This allows more optimization
        freedom while still pushing toward the target distribution.
        
        Parameters:
            mA: Adjacency matrix
            W: Edge weight matrix (costs/travel times)
            eta: Lower bound for probabilities
            pi_hat: Target stationary distribution
            objective_type: 'maximize_efficiency' or 'target_efficiency'
            efficiency_target: Target efficiency value (for target mode)
            pi_penalty_weight: Weight for penalizing deviation from π₀
        """
        self.mA = mA
        self.W = W
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
        self.objective_type = objective_type
        self.efficiency_target = efficiency_target
        self.pi_penalty_weight = pi_penalty_weight
        
        # Target stationary distribution
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
        
        # Only row-sum constraints (soft constraint for π)
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
            print("Warning: Problem is fully constrained.")
        
        print(f"Variables: {d}, Hard constraints: {self.m2}, Free dimensions: {self.C.shape[0]}")
        print(f"Stationary distribution π = π₀ enforced as SOFT constraint (penalty weight = {self.pi_penalty_weight:.0e})")
        
        try:
            A_pinv = self.A.T @ np.linalg.inv(self.A @ self.A.T + 1e-10 * np.eye(len(self.A)))
            self.A_pinv_b = A_pinv @ self.b
            self.C__C_T_C_inv__C_T = self.C.T @ self.C
        except np.linalg.LinAlgError:
            self.A_pinv_b = np.zeros(d)
            self.C__C_T_C_inv__C_T = np.eye(d)
    
    def objective(self, P, mA_sample=None):
        """
        Compute objective: maximize efficiency - penalty for π deviation.
        
        Objective = -Efficiency + λ * ||π - π₀||²
        
        We minimize this, so maximizing efficiency and minimizing π deviation.
        """
        mc = MarkovChain(mA=self.mA, W=self.W, bUndirected=False)
        mc.x = MarkovChain.P_to_x(P, self.mA, self.bUndirected)
        
        try:
            # Compute efficiency
            Eff = mc.compute_efficiency_index()
            
            # Compute penalty for deviation from target π
            pi = mc.pi  # Already computed
            pi_error = np.sum((pi - self.pi_hat) ** 2)  # L2 squared error
            penalty = self.pi_penalty_weight * pi_error
            
            if self.objective_type == 'maximize_efficiency':
                return -Eff + penalty  # Minimize negative efficiency + penalty
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
        
        mc.compute_efficiency_index()  # Computes all intermediate values
        
        pi_error = np.sqrt(np.sum((mc.pi - self.pi_hat) ** 2))  # L2 error
        
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


def create_feasible_P(mA, pi_hat, W=None):
    """
    Create a feasible transition matrix P that satisfies:
    - Row-stochastic: P @ 1 = 1
    - Stationary distribution: pi^T @ P = pi^T
    
    Uses the Metropolis-Hastings construction for reversible chains.
    
    Parameters:
        mA: Adjacency matrix
        pi_hat: Target stationary distribution
        W: Edge weights (optional, used to bias towards lower-cost edges)
    
    Returns:
        P: Feasible transition matrix
    """
    n = mA.shape[0]
    P = np.zeros((n, n))
    
    # For each pair (i,j) with an edge, set acceptance probability
    # using Metropolis-Hastings style construction
    for i in range(n):
        neighbors = np.where(mA[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        
        # Proposal: uniform over neighbors (or weighted by 1/W for lower cost preference)
        if W is not None:
            # Prefer lower cost edges
            weights = 1.0 / (W[i, neighbors] + 0.1)
            q = weights / weights.sum()
        else:
            q = np.ones(len(neighbors)) / len(neighbors)
        
        for idx, j in enumerate(neighbors):
            # Metropolis acceptance ratio
            acceptance = min(1.0, pi_hat[j] / pi_hat[i])
            P[i, j] = q[idx] * acceptance
        
        # Self-loop to make row sum to 1
        P[i, i] = 1.0 - P[i, neighbors].sum()
        
        # If negative self-loop, renormalize
        if P[i, i] < 0:
            P[i, neighbors] = P[i, neighbors] / P[i, neighbors].sum()
            P[i, i] = 0
    
    # For non-reversible chains, we need a different approach
    # Let's use a simpler heuristic: scale outgoing probabilities by target pi
    P2 = np.zeros((n, n))
    for i in range(n):
        neighbors = np.where(mA[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        
        # Weight by pi_j (prefer going to high-pi nodes if we're at low-pi node)
        weights = pi_hat[neighbors]
        P2[i, neighbors] = weights / weights.sum()
    
    # Check which one is closer to target pi
    # Compute stationary distribution of P2
    eigenvalues, eigenvectors = eig(P2.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi_result = np.real(eigenvectors[:, idx])
    pi_result = pi_result / np.sum(pi_result)
    
    return P2, pi_result


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
    
    # Store evaluation points only (not every iteration)
    eval_iters = list(range(0, max_iter, obj_interval))
    n_evals = len(eval_iters)
    
    eff_hist = []
    kw_hist = []
    var_hist = []
    iter_hist = []
    
    # For Polyak-Ruppert averaging
    x_sum = np.zeros_like(x)
    
    # Initial evaluation
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
        
        # Periodic evaluation
        if k % obj_interval == 0:
            # Use averaged x
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
    
    # Final evaluation
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
# TEST: 5-NODE NETWORK (MORE FREEDOM)
# ============================================================================

def plot_network(mA, W, P=None, pi=None, title="Network Structure", filename=None):
    """
    Plot the network with edge weights and optionally transition probabilities.
    
    Parameters:
        mA: Adjacency matrix
        W: Edge weight matrix (costs)
        P: Transition probability matrix (optional, shown as edge thickness)
        pi: Stationary distribution (optional, shown as node color intensity)
        title: Plot title
        filename: If provided, save to this file
    """
    n = mA.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Position nodes in a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Start from top (90 degrees) and go clockwise
    angles = np.pi/2 - angles
    radius = 1.0
    pos = {i: (radius * np.cos(angles[i]), radius * np.sin(angles[i])) for i in range(n)}
    
    # Draw edges
    for i in range(n):
        for j in range(n):
            if mA[i, j] > 0:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                
                # Edge properties
                weight = W[i, j]
                prob = P[i, j] if P is not None else 0.25
                
                # Curve the edges slightly to show both directions
                # Calculate perpendicular offset for curved arrows
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                
                # Perpendicular direction (for curving)
                px, py = -dy/dist * 0.08, dx/dist * 0.08
                
                # Control point for quadratic bezier (midpoint + offset)
                mid_x, mid_y = (x1 + x2) / 2 + px, (y1 + y2) / 2 + py
                
                # Arrow properties based on transition probability
                if P is not None:
                    linewidth = 1 + prob * 8  # Scale line width by probability
                    alpha = 0.4 + prob * 0.6
                else:
                    linewidth = 2
                    alpha = 0.7
                
                # Color based on weight (cost) - darker = higher cost
                color_intensity = (weight - W[mA > 0].min()) / (W[mA > 0].max() - W[mA > 0].min() + 1e-6)
                color = plt.cm.Reds(0.3 + 0.6 * color_intensity)
                
                # Draw curved arrow
                arrow = plt.annotate(
                    '', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        connectionstyle=f'arc3,rad=0.15',
                        color=color,
                        lw=linewidth,
                        alpha=alpha,
                        shrinkA=15,
                        shrinkB=15,
                    )
                )
                
                # Add weight label on edge
                label_x = mid_x + px * 1.5
                label_y = mid_y + py * 1.5
                ax.text(label_x, label_y, f'{weight:.1f}', fontsize=8, 
                       ha='center', va='center', color='darkred', alpha=0.8)
    
    # Draw nodes - color by stationary distribution if provided
    if pi is not None:
        # Normalize pi for coloring (higher pi = darker blue)
        pi_normalized = (pi - pi.min()) / (pi.max() - pi.min() + 1e-6)
        node_colors = [plt.cm.Blues(0.4 + 0.5 * pi_normalized[i]) for i in range(n)]
    else:
        # Uniform color when no stationary distribution
        node_colors = [plt.cm.Blues(0.6)] * n
    
    for i in range(n):
        x, y = pos[i]
        circle = plt.Circle((x, y), 0.12, color=node_colors[i], ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(i), fontsize=14, ha='center', va='center', 
               fontweight='bold', color='white', zorder=11)
        
        # Add pi value below node if provided
        if pi is not None:
            ax.text(x, y - 0.22, f'π={pi[i]:.3f}', fontsize=9, ha='center', va='top',
                   color='darkblue', fontweight='bold')
    
    # Add legend for interpretation
    legend_text = ""
    if P is not None:
        legend_text += "Edge thickness = Transition probability\n"
    legend_text += "Edge color = Travel time (darker = longer)"
    if pi is not None:
        legend_text += "\nNode color = Coverage π (darker = higher)"
    
    ax.text(0.02, 0.98, legend_text, 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Network plot saved to '{filename}'")
    
    plt.show()
    return fig


def test_5node_network():
    """Test with 5-node surveillance network with soft π constraint."""
    
    print("="*80)
    print("TEST: 5-NODE SURVEILLANCE NETWORK")
    print("="*80)
    
    # 5-node complete graph
    n = 5
    mA = np.ones((n, n)) - np.eye(n)  # Complete graph (no self-loops)
    
    # Asymmetric edge weights (travel times)
    np.random.seed(42)
    W = np.random.uniform(1, 5, (n, n))
    W = W * mA  # Zero out non-edges
    
    # Target stationary distribution (non-uniform - some nodes need more coverage)
    # Node 0 is high-priority (30%), node 4 is low-priority (10%)
    pi_hat = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    
    print(f"\nNetwork: {n}-node complete graph")
    print(f"Number of edges: {int(mA.sum())}")
    print(f"\nEdge Weight Matrix W (travel times):")
    print(np.round(W, 2))
    print(f"\nTarget stationary distribution π₀: {pi_hat}")
    print("(Higher values = nodes that need more frequent visits)")
    
    # Problem setup with SOFT π constraint
    eta = 1e-4
    problem = EfficiencyProblemInstance(
        mA=mA, 
        W=W, 
        eta=eta,
        pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        pi_penalty_weight=1e3,  # Soft constraint penalty
    )
    
    # Initial: uniform transitions
    x_init = np.ones(problem.d) / (n - 1)
    
    print(f"\nInitial policy: uniform transitions (1/{n-1} = {1/(n-1):.4f})")
    
    # Evaluate initial metrics
    P_init_check = x_to_matrix(x_init, problem.N, problem.edge_matrix, problem.bUndirected)
    init_metrics_check = problem.evaluate_metrics(P_init_check)
    print(f"Initial π: {np.round(init_metrics_check['pi'], 4)}")
    print(f"Initial π error (L2): {init_metrics_check['pi_error']:.4f}")
    print(f"Initial return times M(i,i): {np.round(init_metrics_check['M_diag'], 2)}")
    
    # Run SPSA
    print("\n" + "="*80)
    iter_hist, eff_hist, kw_hist, var_hist, best_x, initial_metrics = solve_spsa_efficiency(
        problem=problem,
        x_init=x_init,
        max_iter=1000,   # Run longer until convergence
        a=0.5,
        a_eps=1000,
        e=1e-4,
        r_nu=0.101,
        obj_interval=5000,  # Less frequent evaluation
        verbose=True
    )
    
    # Final analysis
    P_final = x_to_matrix(best_x, problem.N, problem.edge_matrix, problem.bUndirected)
    final_metrics = problem.evaluate_metrics(P_final)
    
    # Also get initial P for comparison plotting
    P_init = x_to_matrix(x_init, problem.N, problem.edge_matrix, problem.bUndirected)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\nOptimal Transition Matrix P:")
    print(np.round(P_final, 3))
    
    print(f"\nStationary Distribution Analysis:")
    print(f"{'Node':<8} {'π (achieved)':<15} {'π₀ (target)':<15} {'Error':<15}")
    print("-" * 53)
    for i in range(problem.N):
        error = final_metrics['pi'][i] - problem.pi_hat[i]
        print(f"{i:<8} {final_metrics['pi'][i]:<15.4f} {problem.pi_hat[i]:<15.4f} {error:+.4f}")
    print(f"\nπ constraint error (L2): {final_metrics['pi_error']:.4f}")
    print(f"π constraint error (max): {final_metrics['pi_max_error']:.4f}")
    
    print(f"\nReturn times M(i,i) for each node:")
    for i in range(problem.N):
        print(f"  Node {i}: M({i},{i}) = {final_metrics['M_diag'][i]:.2f}")
    
    print(f"\nMetrics Comparison:")
    print(f"{'Metric':<25} {'Initial':<15} {'Final':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Efficiency Index λ':<25} {initial_metrics['Eff_Idx']:<15.6f} {final_metrics['Eff_Idx']:<15.6f} {final_metrics['Eff_Idx'] - initial_metrics['Eff_Idx']:+.6f}")
    print(f"{'Mean Patrol Time K_W':<25} {initial_metrics['K_W']:<15.6f} {final_metrics['K_W']:<15.6f} {final_metrics['K_W'] - initial_metrics['K_W']:+.6f}")
    print(f"{'Path Variance V_W':<25} {initial_metrics['Net_Var']:<15.6f} {final_metrics['Net_Var']:<15.6f} {final_metrics['Net_Var'] - initial_metrics['Net_Var']:+.6f}")
    print(f"{'Coverage Error ||π-π₀||':<25} {initial_metrics['pi_error']:<15.6f} {final_metrics['pi_error']:<15.6f} {final_metrics['pi_error'] - initial_metrics['pi_error']:+.6f}")
    
    pct_improvement = (final_metrics['Eff_Idx'] / initial_metrics['Eff_Idx'] - 1) * 100
    print(f"\nEfficiency improvement: {pct_improvement:.2f}%")
    
    # Plot the network structure with travel times
    print("\n" + "-"*40)
    print("Plotting network structure...")
    plot_network(mA, W, P=None, pi=None,
                 title="Network Structure (Edge Labels = Travel Times)", 
                 filename='network_structure.png')
    
    # Plot the initial (uniform) policy
    print("\nPlotting initial (uniform) policy...")
    plot_network(mA, W, P=P_init, pi=initial_metrics['pi'],
                 title="Initial Policy (Uniform Transitions)", 
                 filename='network_initial_policy.png')
    
    # Plot the network with optimal transition probabilities and resulting pi
    print("\nPlotting optimal policy on network...")
    plot_network(mA, W, P=P_final, pi=final_metrics['pi'],
                 title="Optimal Surveillance Policy", 
                 filename='network_optimal_policy.png')
    
    # Plot results - include all iterations from 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Include all iterations (starting from 0)
    iter_plot = iter_hist
    eff_plot = eff_hist
    kw_plot = kw_hist
    var_plot = var_hist
    
    # Plot 1: Efficiency Index
    axes[0, 0].plot(iter_plot, eff_plot, 'b-', linewidth=2, marker='o', markersize=4)
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
    axes[0, 1].plot(iter_plot, kw_plot, 'orange', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=initial_metrics['K_W'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['K_W']:.4f}")
    axes[0, 1].axhline(y=final_metrics['K_W'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['K_W']:.4f}")
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[0, 1].set_title('Mean Patrol Time (expected time to reach target)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Path Variance (unpredictability)
    axes[1, 0].plot(iter_plot, var_plot, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(y=initial_metrics['Net_Var'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['Net_Var']:.4f}")
    axes[1, 0].axhline(y=final_metrics['Net_Var'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['Net_Var']:.4f}")
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 0].set_title(r'Path Variance ($\uparrow$ higher = more unpredictable)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mean-Variance tradeoff
    scatter = axes[1, 1].scatter(kw_plot, var_plot, c=iter_plot, cmap='viridis', 
                                  s=50, alpha=0.7)
    axes[1, 1].scatter(initial_metrics['K_W'], initial_metrics['Net_Var'], 
                      color='green', s=300, marker='*', label='Initial', zorder=5, edgecolor='black')
    axes[1, 1].scatter(final_metrics['K_W'], final_metrics['Net_Var'],
                      color='red', s=300, marker='*', label='Final', zorder=5, edgecolor='black')
    axes[1, 1].set_xlabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[1, 1].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 1].set_title('Mean-Variance Tradeoff (optimization trajectory)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Iteration')
    
    plt.tight_layout()
    plt.savefig('efficiency_5node_results.png', dpi=150)
    print("\n✓ Plot saved to 'efficiency_5node_results.png'")
    plt.show()
    
    return best_x, final_metrics


if __name__ == "__main__":
    np.random.seed(42)
    print("Running 5-Node Network Test\n")
    best_x, metrics = test_5node_network()
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)