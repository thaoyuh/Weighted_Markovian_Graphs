import numpy as np
from numpy.linalg import eig, inv
from itertools import product
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# ============================================================================
# PART 1: MarkovChain Class (with weighted metrics)
# ============================================================================

class MarkovChain:
    def __init__(self, mA, x=None, W=None, bUndirected=False):
        """MarkovChain with weighted Kemeny constant support."""
        self.n = mA.shape[0]
        self.bUndirected = bUndirected
        self.mA = mA
        
        # Edge weights
        if W is not None:
            if W.shape != (self.n, self.n):
                raise ValueError(f"W must be {self.n}x{self.n}, got {W.shape}")
            self.W = W
            self.W2 = W ** 2
        else:
            self.W = np.ones((self.n, self.n))
            self.W2 = np.ones((self.n, self.n))
        
        # Cached properties
        self.pi = None
        self.Pi = None
        self.pi_W = None
        self.Z = None
        self.M = None
        self.K_W = None
        
        # Edge matrix
        self.edge_matrix = self._create_edge_matrix(mA)
        
        if x is not None:
            self.x = x
    
    @staticmethod
    def _create_edge_matrix(mA):
        """Create list of edges from adjacency matrix."""
        indices = np.nonzero(mA)
        num_edges = indices[0].shape[0]
        E = np.zeros((num_edges, 2), dtype=int)
        E[:, 0] = indices[0]
        E[:, 1] = indices[1]
        return E
    
    @property
    def P(self):
        """Convert x vector to transition matrix P."""
        P_matrix = np.zeros((self.n, self.n))
        P_matrix[self.edge_matrix[:, 0], self.edge_matrix[:, 1]] = self.x
        return P_matrix
    
    @staticmethod
    def P_to_x(P, mA):
        """Extract x vector from P matrix."""
        N, _ = mA.shape
        return np.array([P[i, j] for i, j in product(range(N), range(N)) if mA[i, j] == 1])
    
    def compute_pi(self):
        """Compute stationary distribution."""
        eigenvalues, eigenvectors = eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = np.real(eigenvectors[:, idx])
        self.pi = pi / np.sum(pi)
        return self.pi
    
    def compute_Pi(self):
        """Matrix where each row is pi."""
        if self.pi is None:
            self.compute_pi()
        self.Pi = np.outer(np.ones(self.n), self.pi)
        return self.Pi
    
    def compute_pi_W(self):
        """Weighted stationary distribution."""
        if self.pi is None:
            self.compute_pi()
        
        P_hadamard_W = np.multiply(self.P, self.W)
        U_bar = np.sum(P_hadamard_W, axis=1)
        
        numerator = self.pi * U_bar
        denominator = np.sum(numerator)
        self.pi_W = numerator / denominator
        return self.pi_W
    
    def compute_Z(self):
        """Group inverse Z = (I - P + Pi)^-1."""
        if self.Pi is None:
            self.compute_Pi()
        I = np.eye(self.n)
        self.Z = inv(I - self.P + self.Pi)
        return self.Z
    
    def compute_M(self):
        """Mean first-entrance path weights matrix."""
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
    
    def compute_kemeny_W(self):
        """Weighted Kemeny constant."""
        if self.pi_W is None:
            self.compute_pi_W()
        if self.M is None:
            self.compute_M()
        
        self.K_W = self.pi_W @ self.M @ self.pi_W.T
        return self.K_W
    
    def clear_cache(self):
        """Clear all cached computations."""
        self.pi = None
        self.Pi = None
        self.pi_W = None
        self.Z = None
        self.M = None
        self.K_W = None


# ============================================================================
# PART 2: Problem Instance with Weighted Objective
# ============================================================================

class WeightedProblemInstance:
    def __init__(self, mA, W, eta=1e-4, pi_hat=None):
        """
        Problem instance for minimizing weighted Kemeny constant.
        
        Parameters:
            mA: Adjacency matrix (binary)
            W: Edge weight matrix
            eta: Lower bound for probabilities
            pi_hat: Target stationary distribution (optional)
        """
        self.mA = mA
        self.W = W
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
        
        # Stationary distribution constraint
        if pi_hat is not None:
            self.pi_hat = pi_hat
            self.bPi_constraint = True
        else:
            self.bPi_constraint = False
        
        # Count edges
        self.edge_matrix = self._create_edge_matrix(mA)
        self.d = len(self.edge_matrix)  # Number of decision variables
        
        # Build constraint matrices
        self._build_constraint_matrices()
        
        # For projections
        self.neighborhoods = self._build_neighborhoods()
        self.proj_type = 'Markov'  # Use simple Markov projection
    
    @staticmethod
    def _create_edge_matrix(mA):
        """Create edge list."""
        indices = np.nonzero(mA)
        num_edges = indices[0].shape[0]
        E = np.zeros((num_edges, 2), dtype=int)
        E[:, 0] = indices[0]
        E[:, 1] = indices[1]
        return E
    
    def _build_neighborhoods(self):
        """Build neighborhoods for Markov projection."""
        neighborhoods = [[] for _ in range(self.N)]
        for idx, (i, j) in enumerate(self.edge_matrix):
            neighborhoods[i].append(idx)
        return neighborhoods
    
    def _build_constraint_matrices(self):
        """Build A, b, C matrices for constraints."""
        N = self.N
        d = self.d
        
        # Build row-sum constraint matrix A
        A = np.zeros((N, d))
        for idx, (i, j) in enumerate(self.edge_matrix):
            A[i, idx] = 1
        
        b = np.ones(N)
        
        # If stationary distribution constraint, add it
        if self.bPi_constraint:
            # Build A_pi: constraints for pi^T P = pi^T
            A_pi = np.zeros((N, d))
            for idx, (i, j) in enumerate(self.edge_matrix):
                A_pi[j, idx] = self.pi_hat[i]
            
            # Combine constraints
            A_combined = np.vstack([A, A_pi])
            b_combined = np.hstack([b, self.pi_hat])
            
            # Remove redundant constraints
            A, b = self._remove_redundant_constraints(A_combined, b_combined)
        
        self.A = A
        self.b = b
        self.m2 = A.shape[0]  # Number of constraints
        
        # Compute null space basis C
        self.C = null_space(A).T
        if self.C.size == 0:
            # Fully constrained - use small perturbations
            self.C = np.eye(d) * 1e-10
        
        # Compute A^+ b for projection
        A_pinv = A.T @ np.linalg.inv(A @ A.T + 1e-10 * np.eye(len(A)))
        self.A_pinv_b = A_pinv @ b
        self.C__C_T_C_inv__C_T = self.C.T @ self.C
    
    def _remove_redundant_constraints(self, A, b):
        """Remove linearly dependent constraint rows."""
        # Stack A and b
        Ab = np.hstack([A, b.reshape(-1, 1)])
        
        # Find independent rows
        _, inds = np.linalg.qr(Ab.T, mode='economic')
        inds = inds[:np.linalg.matrix_rank(Ab)]
        
        return A[inds], b[inds]
    
    def objective(self, P, mA_sample=None):
        """
        Compute weighted Kemeny constant.
        
        Parameters:
            P: Transition matrix
            mA_sample: Not used (for compatibility)
        
        Returns:
            Weighted Kemeny constant K_W
        """
        mc = MarkovChain(mA=self.mA, W=self.W)
        mc.x = MarkovChain.P_to_x(P, self.mA)
        
        try:
            K_W = mc.compute_kemeny_W()
            return K_W
        except np.linalg.LinAlgError:
            # Matrix is singular - return large penalty
            return 1e10
    
    def sample_mA(self):
        """Return full adjacency (no failures for this simple case)."""
        return self.mA


# ============================================================================
# PART 3: Projection Functions
# ============================================================================

def proj_c_simplex(v, c=1, tol=1e-8):
    """Project vector v onto simplex {x >= 0, sum(x) = c}."""
    N = len(v)
    vU = np.sort(v)[::-1]
    cssv = np.cumsum(vU)
    l = [k+1 for k in range(N) if (cssv[k] - c) / (k + 1) < vU[k]]
    
    if not l:
        raise ValueError("No valid projection index found")
    
    K = max(l)
    tau = (cssv[K - 1] - c) / K
    v_proj = np.maximum(v - tau, 0)
    
    if abs(np.sum(v_proj) - c) > tol:
        print(f"Warning: Projection error = {abs(np.sum(v_proj) - c)}")
    
    return v_proj


def projection_markov(x_to_proj, eta, neighborhoods, mA):
    """Project onto Markov constraints (row-stochastic)."""
    x = x_to_proj - eta
    
    x_proj = []
    for i, subset in enumerate(neighborhoods):
        if subset:
            # Number of outgoing edges from node i
            n_edges = len(subset)
            c = 1 - n_edges * eta
            x_proj.extend(proj_c_simplex(x[subset], c=c).tolist())
    
    return np.array(x_proj) + eta


def projection(x_to_proj, problem, eta):
    """Main projection function."""
    return projection_markov(
        x_to_proj,
        eta=eta,
        neighborhoods=problem.neighborhoods,
        mA=problem.mA
    )


# ============================================================================
# PART 4: SPSA Algorithm
# ============================================================================

def g_spsa(problem, x, eta_perturb, mc_plus, mc_min):
    """
    SPSA gradient estimator.
    
    Parameters:
        problem: Problem instance
        x: Current decision vector
        eta_perturb: Perturbation size
        mc_plus, mc_min: MarkovChain objects for evaluation
    
    Returns:
        Gradient estimate
    """
    d_free = problem.C.shape[0]  # Dimension of null space
    
    if d_free == 0:
        # No free dimensions - return zero gradient
        return np.zeros(problem.d)
    
    # Generate random perturbation in {-1, +1}
    Delta = np.random.choice([-1, 1], size=d_free)
    
    # Project to constraint manifold
    direction = problem.C.T @ Delta
    
    # Perturb
    x_plus = x + eta_perturb * direction
    x_min = x - eta_perturb * direction
    
    # Ensure feasibility (clip to bounds)
    x_plus = np.clip(x_plus, problem.eta, 1 - problem.eta)
    x_min = np.clip(x_min, problem.eta, 1 - problem.eta)
    
    # Evaluate objectives
    mc_plus.x = x_plus
    mc_min.x = x_min
    
    J_plus = problem.objective(mc_plus.P)
    J_min = problem.objective(mc_min.P)
    
    # Compute gradient estimate
    gradient_est = (J_plus - J_min) / (2 * eta_perturb * Delta)
    
    # Map back to full space
    return problem.C.T @ gradient_est


def solve_spsa_weighted(problem, mc_init, max_iter=10000, 
                        a=0.01, a_eps=100, r_epsilon=0.602,
                        e=1e-6, r_nu=0.101,
                        obj_interval=100, verbose=True):
    """
    SPSA optimization for weighted Kemeny constant.
    
    Parameters:
        problem: WeightedProblemInstance
        mc_init: Initial MarkovChain
        max_iter: Maximum iterations
        a, a_eps, r_epsilon: Step size parameters
        e, r_nu: Perturbation size parameters
        obj_interval: How often to evaluate objective
        verbose: Print progress
    
    Returns:
        x_hist: History of x vectors
        obj_hist: History of objective values
    """
    # Initialize
    x = mc_init.x.copy()
    x_hist = np.zeros((max_iter, len(x)))
    x_hist[0] = x
    obj_hist = np.zeros(max_iter)
    
    # Create MarkovChain objects for evaluations
    mc_plus = MarkovChain(mA=problem.mA, W=problem.W)
    mc_min = MarkovChain(mA=problem.mA, W=problem.W)
    mc_eval = MarkovChain(mA=problem.mA, W=problem.W)
    
    # Initial objective
    mc_eval.x = x
    obj_hist[0] = problem.objective(mc_eval.P)
    
    if verbose:
        print(f"Initial K_W: {obj_hist[0]:.6f}")
        print(f"Starting SPSA optimization...")
        print(f"{'Iter':<8} {'K_W':<12} {'Step Size':<12} {'Improvement':<12}")
        print("-" * 50)
    
    best_obj = obj_hist[0]
    best_x = x.copy()
    
    for k in range(max_iter - 1):
        # Compute step sizes
        alpha_k = a / (a_eps + k + 1) ** r_epsilon
        eta_k = e / (k + 1) ** r_nu
        
        # Estimate gradient
        grad = g_spsa(problem, x, eta_k, mc_plus, mc_min)
        
        # Update
        x_new = x - alpha_k * grad
        
        # Project to feasible set
        if np.any(x_new < problem.eta) or np.any(x_new > 1 - problem.eta):
            x_new = projection(x_new, problem, problem.eta)
        
        x = x_new
        x_hist[k + 1] = x
        
        # Evaluate objective periodically
        if k % obj_interval == 0:
            # Use Polyak-Ruppert averaging
            if k > 0:
                start_avg = max(0, int(0.5 * k))
                x_avg = np.mean(x_hist[start_avg:k+1], axis=0)
            else:
                x_avg = x
            
            mc_eval.x = x_avg
            mc_eval.clear_cache()
            obj_hist[k] = problem.objective(mc_eval.P)
            
            if obj_hist[k] < best_obj:
                best_obj = obj_hist[k]
                best_x = x_avg.copy()
                improvement = "✓ New best!"
            else:
                improvement = ""
            
            if verbose and k > 0:
                delta = obj_hist[k] - obj_hist[k - obj_interval]
                print(f"{k:<8} {obj_hist[k]:<12.6f} {alpha_k:<12.6e} {delta:<12.6f} {improvement}")
        else:
            obj_hist[k] = obj_hist[k - 1]
    
    # Final evaluation with averaging
    start_avg = max(0, int(0.5 * max_iter))
    x_final = np.mean(x_hist[start_avg:], axis=0)
    mc_eval.x = x_final
    mc_eval.clear_cache()
    obj_final = problem.objective(mc_eval.P)
    
    if verbose:
        print("-" * 50)
        print(f"Final K_W (averaged): {obj_final:.6f}")
        print(f"Best K_W found: {best_obj:.6f}")
        print(f"Improvement: {obj_hist[0] - best_obj:.6f} ({(1 - best_obj/obj_hist[0])*100:.2f}%)")
    
    return x_hist, obj_hist, best_x


# ============================================================================
# PART 5: Test with 3-Node Network
# ============================================================================

def test_spsa_3node():
    """Test SPSA optimization on 3-node network."""
    
    print("="*70)
    print("SPSA OPTIMIZATION: 3-Node Network with Weighted Kemeny Constant")
    print("="*70)
    
    # Network setup
    n = 3
    mA = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    # Edge weights (asymmetric - different costs)
    W = np.array([
        [0,   1.0, 3.0],  # 0->1 cheap, 0->2 expensive
        [1.0, 0,   2.0],  # 1->0 cheap, 1->2 medium
        [3.0, 2.0, 0]     # 2->0 expensive, 2->1 medium
    ])
    
    print("\nEdge Weight Matrix W:")
    print(W)
    print("\nKey: Cheaper edges are better (lower cost)")
    
    # Problem setup
    eta = 1e-4
    problem = WeightedProblemInstance(mA=mA, W=W, eta=eta)
    
    print(f"\nProblem dimensions:")
    print(f"  Nodes (N): {problem.N}")
    print(f"  Edges (d): {problem.d}")
    print(f"  Constraints (m): {problem.m2}")
    print(f"  Free dimensions: {problem.C.shape[0]}")
    
    # Initial solution (uniform)
    x_init = np.ones(problem.d) * 0.5
    mc_init = MarkovChain(mA=mA, x=x_init, W=W)
    
    print(f"\nInitial transition probabilities (uniform):")
    print(f"  x = {x_init}")
    print(f"  K_W = {problem.objective(mc_init.P):.6f}")
    
    # Run SPSA
    print("\n" + "="*70)
    print("Running SPSA Optimization...")
    print("="*70 + "\n")
    
    x_hist, obj_hist, best_x = solve_spsa_weighted(
        problem=problem,
        mc_init=mc_init,
        max_iter=5000,
        a=0.1,
        a_eps=500,
        e=1e-6,
        obj_interval=500,
        verbose=True
    )
    
    # Analyze results
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    mc_final = MarkovChain(mA=mA, W=W)
    mc_final.x = best_x
    
    print(f"\nOptimal transition probabilities:")
    print(f"  x_opt = {best_x}")
    print(f"\nOptimal transition matrix P:")
    print(mc_final.P)
    
    print(f"\nStationary distribution π:")
    pi = mc_final.compute_pi()
    print(f"  π = {pi}")
    
    print(f"\nWeighted stationary distribution π_W:")
    pi_W = mc_final.compute_pi_W()
    print(f"  π_W = {pi_W}")
    
    print(f"\nMean first-passage weights M:")
    M = mc_final.compute_M()
    print(M)
    
    print(f"\nInterpretation:")
    for i in range(n):
        for j in range(n):
            if i != j:
                print(f"  M[{i},{j}] = {M[i,j]:.4f} - Cost to reach node {j} from node {i}")
    
    # Compare with uniform strategy
    print("\n" + "-"*70)
    print("COMPARISON: Optimal vs Uniform Strategy")
    print("-"*70)
    
    mc_uniform = MarkovChain(mA=mA, x=np.ones(problem.d)*0.5, W=W)
    K_W_uniform = mc_uniform.compute_kemeny_W()
    K_W_optimal = mc_final.compute_kemeny_W()
    
    print(f"\nUniform strategy:")
    print(f"  P = {mc_uniform.P}")
    print(f"  K_W = {K_W_uniform:.6f}")
    
    print(f"\nOptimal strategy:")
    print(f"  P = {mc_final.P}")
    print(f"  K_W = {K_W_optimal:.6f}")
    
    print(f"\nImprovement:")
    print(f"  ΔK_W = {K_W_uniform - K_W_optimal:.6f}")
    print(f"  Reduction = {(1 - K_W_optimal/K_W_uniform)*100:.2f}%")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(obj_hist, linewidth=2)
    plt.axhline(y=K_W_optimal, color='r', linestyle='--', label=f'Optimal K_W = {K_W_optimal:.4f}')
    plt.axhline(y=K_W_uniform, color='g', linestyle='--', label=f'Uniform K_W = {K_W_uniform:.4f}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Weighted Kemeny Constant K_W', fontsize=12)
    plt.title('SPSA Convergence: Minimizing Weighted Kemeny Constant', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figures/spsa_convergence_3node.png', dpi=150)
    print("\n✓ Convergence plot saved to 'spsa_convergence_3node.png'")
    plt.show()


# ============================================================================
# PART 6: Test with 5-Node Network (More Interesting)
# ============================================================================

def test_spsa_5node():
    """Test SPSA on a 5-node ring network."""
    
    print("\n" + "="*70)
    print("SPSA OPTIMIZATION: 5-Node Ring Network")
    print("="*70)
    
    # Ring topology: 0->1->2->3->4->0
    n = 5
    mA = np.array([
        [0, 1, 0, 0, 1],  # 0 connects to 1, 4
        [1, 0, 1, 0, 0],  # 1 connects to 0, 2
        [0, 1, 0, 1, 0],  # 2 connects to 1, 3
        [0, 0, 1, 0, 1],  # 3 connects to 2, 4
        [1, 0, 0, 1, 0]   # 4 connects to 3, 0
    ])
    
    # Asymmetric weights (clockwise cheaper than counter-clockwise)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i+1) % n] = 1.0  # Clockwise: cheap
        W[i, (i-1) % n] = 3.0  # Counter-clockwise: expensive
    
    print("\nTopology: Ring (0-1-2-3-4-0)")
    print("\nEdge Weights:")
    print("  Clockwise edges (0→1, 1→2, etc.): cost 1.0")
    print("  Counter-clockwise edges: cost 3.0")
    print("\nOptimal strategy should favor clockwise movement")
    
    # Setup
    eta = 1e-4
    problem = WeightedProblemInstance(mA=mA, W=W, eta=eta)
    
    # Initial (uniform)
    x_init = np.ones(problem.d) * 0.5
    mc_init = MarkovChain(mA=mA, x=x_init, W=W)
    
    print(f"\nInitial K_W (uniform): {problem.objective(mc_init.P):.6f}")
    
    # Optimize
    x_hist, obj_hist, best_x = solve_spsa_weighted(
        problem=problem,
        mc_init=mc_init,
        max_iter=10000,
        a=0.1,
        a_eps=1000,
        e=1e-6,
        obj_interval=1000,
        verbose=True
    )
    
    # Results
    mc_final = MarkovChain(mA=mA, W=W)
    mc_final.x = best_x
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print("\nOptimal transition matrix P:")
    print(mc_final.P)
    print("\nInterpretation:")
    for i in range(n):
        cw = (i+1) % n
        ccw = (i-1) % n
        p_cw = mc_final.P[i, cw]
        p_ccw = mc_final.P[i, ccw]
        print(f"  Node {i}: {p_cw:.3f} clockwise (cheap), {p_ccw:.3f} counter-clockwise (expensive)")
    
    K_W_uniform = mc_init.compute_kemeny_W()
    K_W_optimal = mc_final.compute_kemeny_W()
    
    print(f"\nPerformance:")
    print(f"  Uniform K_W:  {K_W_uniform:.6f}")
    print(f"  Optimal K_W:  {K_W_optimal:.6f}")
    print(f"  Improvement:  {(1 - K_W_optimal/K_W_uniform)*100:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(obj_hist, linewidth=2, label='SPSA trajectory')
    plt.axhline(y=K_W_optimal, color='r', linestyle='--', label=f'Optimal = {K_W_optimal:.4f}')
    plt.axhline(y=K_W_uniform, color='g', linestyle='--', label=f'Uniform = {K_W_uniform:.4f}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Weighted Kemeny Constant K_W', fontsize=12)
    plt.title('SPSA Convergence: 5-Node Ring Network', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figures/spsa_convergence_5node.png', dpi=150)
    print("\n✓ Plot saved to 'spsa_convergence_5node.png'")
    plt.show()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Test 3-node network
    test_spsa_3node()
    
    # Test 5-node network
    test_spsa_5node()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)