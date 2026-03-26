"""
Problem instance for efficiency index optimization with stochastic weights.

Uses HARD CONSTRAINT for stationary distribution (basis transformation approach
from Franssen et al.) to ensure the optimization problem is well-posed (bounded).
"""

import numpy as np
from itertools import product, chain
import sympy as sp

from utils import create_edge_matrix, build_neighborhoods, x_to_matrix


# =============================================================================
# Linear Algebra Utilities for Constraint Handling
# =============================================================================

def build_row_sum_constraints(mA):
    """
    Build the row-sum constraint matrix Aâ‚.
    
    Constraint: âˆ‘â±¼ P(i,j) = 1 for all i
    
    Parameters:
        mA: Adjacency matrix
        
    Returns:
        A_row: Constraint matrix of shape (N, num_edges)
    """
    N = mA.shape[0]
    edge_list = list(zip(*np.where(mA > 0)))
    num_edges = len(edge_list)
    
    # Map edges to indices
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edge_list)}
    
    A_row = np.zeros((N, num_edges))
    for (i, j), idx in edge_to_idx.items():
        A_row[i, idx] = 1
    
    return A_row


def build_stationary_constraints(mA, pi_hat):
    """
    Build the stationary distribution constraint matrix Aâ‚‚.
    
    Constraint: Ï€Ì‚áµ€P = Ï€Ì‚áµ€  âŸº  âˆ‘áµ¢ Ï€Ì‚(i)P(i,j) = Ï€Ì‚(j) for all j
    
    Parameters:
        mA: Adjacency matrix
        pi_hat: Target stationary distribution
        
    Returns:
        A_pi: Constraint matrix of shape (N, num_edges)
    """
    N = mA.shape[0]
    edge_list = list(zip(*np.where(mA > 0)))
    num_edges = len(edge_list)
    
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edge_list)}
    
    A_pi = np.zeros((N, num_edges))
    for (i, j), idx in edge_to_idx.items():
        # Coefficient for P(i,j) in constraint for node j
        A_pi[j, idx] = pi_hat[i]
    
    return A_pi


def echelon_sympy(matrix):
    """
    Convert a matrix to row echelon form using Sympy's exact arithmetic.
    This avoids numerical issues that can arise with floating point.
    """
    sympy_matrix = sp.Matrix(matrix.tolist())
    echelon = sympy_matrix.echelon_form()
    return np.array(echelon.tolist(), dtype=np.float64)


def pivot_rows(echelon_matrix, tol=1e-10):
    """
    Identify pivot rows in a row echelon matrix.
    
    Returns:
        List of row indices containing pivots
    """
    pivots = []
    pivot_cols = set()
    
    for row_idx, row in enumerate(echelon_matrix):
        # Find first non-zero element
        nonzero_idx = np.where(np.abs(row) > tol)[0]
        if len(nonzero_idx) > 0:
            first_nonzero = nonzero_idx[0]
            if first_nonzero not in pivot_cols:
                pivots.append(row_idx)
                pivot_cols.add(first_nonzero)
    
    return pivots


def orthonormal_basis_nullspace(A):
    """
    Compute orthonormal basis for the null space of A using SVD.
    
    Returns:
        C: Matrix whose columns span null(A), shape (d, d-rank)
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Find rank (number of non-zero singular values)
    tol = max(A.shape) * np.finfo(float).eps * S.max() if S.max() > 0 else 1e-10
    rank = np.sum(S > tol)
    
    # Null space is spanned by rows of Vt corresponding to zero singular values
    # These are the last (n - rank) rows of Vt, or columns of V
    null_space_basis = Vt[rank:].T
    
    return null_space_basis


# =============================================================================
# Projection Functions
# =============================================================================

def projection_box(x, eta):
    """Project x onto box [eta, 1-eta]."""
    return np.clip(x, eta, 1 - eta)


def projection_affine(x, A_pinv_b, C_proj):
    """
    Project x onto affine subspace Ax = b.
    
    Formula: x_proj = Aâºb + CC^T(x - Aâºb)
    """
    y = x - A_pinv_b
    return A_pinv_b + C_proj @ y


def dykstra_projection(x0, A_pinv_b, C_proj, eta, tol=1e-10, max_iter=100000):
    """
    Dykstra's alternating projection algorithm.
    
    Projects onto intersection of:
    - Xâ‚: Affine subspace Ax = b (row-sum + stationary constraints)
    - Xâ‚‚: Box constraints [eta, 1-eta]
    
    Parameters:
        x0: Initial point
        A_pinv_b: Aâºb (particular solution)
        C_proj: CC^T projection matrix for null space
        eta: Lower/upper bound margin
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Projected point satisfying both constraints
    """
    x = x0.copy()
    p = np.zeros_like(x)  # Increment for box projection
    q = np.zeros_like(x)  # Increment for affine projection
    
    for k in range(max_iter):
        # Project onto box
        z = projection_box(x + p, eta)
        p = x + p - z
        
        # Project onto affine subspace
        x_new = projection_affine(z + q, A_pinv_b, C_proj)
        q = z + q - x_new
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        
        x = x_new
    
    # Final box projection to ensure bounds
    x = projection_box(x, eta)
    
    return x


def projection_markov_simple(x, eta, neighborhoods, mA):
    """
    Simple Markov projection (row-sum = 1, no stationary constraint).
    Fallback when Dykstra fails.
    """
    x_shifted = x - eta
    x_proj_list = []
    
    for i, subset in enumerate(neighborhoods):
        if len(subset) > 0:
            v = x_shifted[subset]
            c = 1 - np.sum(mA[i]) * eta
            
            # Project onto simplex
            N = len(v)
            if N == 0:
                continue
            vU = np.sort(v)[::-1]
            cssv = np.cumsum(vU)
            
            # Find rho
            rho_candidates = np.where((vU - (cssv - c) / np.arange(1, N + 1)) > 0)[0]
            if len(rho_candidates) > 0:
                rho = rho_candidates[-1] + 1
            else:
                rho = 1
            
            theta = (cssv[rho - 1] - c) / rho
            v_proj = np.maximum(v - theta, 0)
            
            x_proj_list.append(v_proj.tolist())
        else:
            x_proj_list.append([])
    
    return np.array(list(chain.from_iterable(x_proj_list))) + eta


# =============================================================================
# Problem Instance Class
# =============================================================================

class EfficiencyProblemInstanceStochastic:
    """
    Problem instance for efficiency index optimization with STOCHASTIC weights.
    
    Uses HARD CONSTRAINT for stationary distribution via basis transformation
    (Franssen et al. approach). This ensures:
    - Ï€(P) = Ï€Ì‚ exactly at every iteration
    - For uniform Ï€Ì‚, P is doubly stochastic
    - The optimization problem is BOUNDED (well-posed)
    
    The objective is pure efficiency index (no penalty term needed).
    """
    
    def __init__(self, mA, W, W2, eta=1e-4, pi_hat=None, 
                 objective_type='maximize_efficiency',
                 use_hard_constraint=False):
        """
        Parameters:
            mA: Adjacency matrix
            W: Mean weight matrix Î¼(i,j) = E[W(i,j)]
            W2: Second moment matrix E[WÂ²(i,j)] - NOT the square of W!
            eta: Small constant for numerical stability in constraints
            pi_hat: Target stationary distribution (enforced as HARD CONSTRAINT)
            objective_type: 'maximize_efficiency', 'minimize_efficiency',
                           'minimize_variance', 'maximize_variance'
            use_hard_constraint: If True, enforce Ï€ as hard constraint (recommended)
                                If False, use soft penalty (may be unbounded!)
        """
        self.mA = mA
        self.W = W
        self.W2 = W2
        self.eta = eta
        self.N = mA.shape[0]
        self.bUndirected = False
        self.objective_type = objective_type
        self.use_hard_constraint = use_hard_constraint
        
        # Target stationary distribution (hard constraint)
        if pi_hat is None:
            self.pi_hat = np.ones(self.N) / self.N
            print("Using uniform stationary distribution constraint.")
        else:
            self.pi_hat = np.array(pi_hat)
            # Normalize to ensure it sums to 1
            self.pi_hat = self.pi_hat / np.sum(self.pi_hat)
        
        # Edge structure
        self.edge_matrix = create_edge_matrix(mA)
        self.d = len(self.edge_matrix)
        self.neighborhoods = build_neighborhoods(self.edge_matrix, self.N)
        
        # Build constraint system
        if use_hard_constraint:
            self._build_constraint_system_with_pi()
            self.proj_type = 'subspace'
        else:
            self._build_simple_constraints()
            self.proj_type = 'Markov'
    
    def _build_simple_constraints(self):
        """Build simple row-stochastic constraints only (soft Ï€ penalty)."""
        N = self.N
        d = self.d
        A_row = np.zeros((N, d))
        for idx, (i, j) in enumerate(self.edge_matrix):
            A_row[i, idx] = 1
        b_row = np.ones(N)
        valid_rows = np.sum(A_row, axis=1) > 0
        self.A_eq = A_row[valid_rows]
        self.b_eq = b_row[valid_rows]
        self.bProjectionReady = False
        print("Using SOFT penalty for stationary distribution (may be unbounded!)")
    
    def _build_constraint_system_with_pi(self):
        """
        Build the combined constraint system with stationary distribution.
        
        Following Franssen et al. Section IV-E:
        [A_Ï€Ì‚ | b_Ï€Ì‚] = rref([Aâ‚ | 1Ì„] ; [Aâ‚‚ | Ï€Ì‚])
        
        Then compute null space basis B_Ï€Ì‚ for feasible directions.
        """
        N = self.N
        d = self.d
        
        # Build row-sum constraint matrix Aâ‚
        A_row = build_row_sum_constraints(self.mA)
        b_row = np.ones(N)
        
        # Build stationary distribution constraint matrix Aâ‚‚
        A_pi = build_stationary_constraints(self.mA, self.pi_hat)
        b_pi = self.pi_hat.copy()
        
        # Filter out obstacle/isolated nodes (no outgoing edges).
        # These have all-zero rows in A_row, creating infeasible
        # constraints (0*x = 1). Remove them before combining.
        active_mask = np.sum(self.mA, axis=1) > 0
        A_row = A_row[active_mask]
        b_row = b_row[active_mask]
        A_pi = A_pi[active_mask]
        b_pi = b_pi[active_mask]
        
        # Combine constraints
        A_combined = np.vstack([A_row, A_pi])
        b_combined = np.hstack([b_row, b_pi])
        
        # Augmented matrix [A | b]
        Ab = np.hstack([A_combined, b_combined.reshape(-1, 1)])
        
        # Row echelon form to remove redundant constraints
        try:
            Ab_ech = echelon_sympy(Ab)
        except Exception as e:
            # Fallback to numpy if sympy fails
            print(f"Warning: Sympy echelon failed ({e}), using numpy QR")
            Q, R = np.linalg.qr(Ab.T)
            Ab_ech = R.T
        
        # Extract A and b from echelon form
        A_ech = Ab_ech[:, :-1]
        b_ech = Ab_ech[:, -1]
        
        # Keep only pivot rows (independent constraints)
        piv_rows = pivot_rows(A_ech)
        
        if len(piv_rows) == 0:
            raise ValueError("No valid constraints found!")
        
        A = A_ech[piv_rows]
        b = b_ech[piv_rows]
        
        # Store constraint system
        self.A = A
        self.b = b
        self.num_constraints = len(piv_rows)
        
        print(f"Constraint system: {self.num_constraints} independent constraints, "
              f"{d} variables, {d - self.num_constraints} free dimensions")
        
        # Compute pseudo-inverse and null space
        try:
            # Aâº = Aáµ€(AAáµ€)â»Â¹
            AAt = A @ A.T
            AAt_inv = np.linalg.inv(AAt + 1e-12 * np.eye(len(AAt)))
            A_pinv = A.T @ AAt_inv
            self.A_pinv_b = A_pinv @ b
            
            # Null space basis
            self.C = orthonormal_basis_nullspace(A)
            
            if self.C.size == 0 or self.C.shape[1] == 0:
                print("Warning: Problem is fully constrained (no free dimensions).")
                self.C = np.zeros((d, 1))
                self.C_proj = np.zeros((d, d))
            else:
                # Projection matrix for affine subspace: CC^T
                self.C_proj = self.C @ self.C.T
                print(f"Null space dimension: {self.C.shape[1]}")
            
            self.bProjectionReady = True
            
        except np.linalg.LinAlgError as e:
            print(f"Warning: Constraint system may be ill-conditioned: {e}")
            # Fallback to simple projection
            self.A_pinv_b = np.zeros(d)
            self.C_proj = np.eye(d)
            self.bProjectionReady = False
    
    def evaluate_metrics(self, P):
        """
        Evaluate all metrics for a given transition matrix P.
        
        Returns dict with: pi, pi_W, M, V, K_W, Net_Var, Eff_Idx, pi_error
        """
        # Import here to avoid circular import
        from network_stochastic import MarkovChainStochastic
        
        mc = MarkovChainStochastic(self.mA, W=self.W, W2=self.W2)
        mc.x = MarkovChainStochastic.P_to_x(P, self.mA, self.bUndirected)
        
        mc.compute_pi()
        mc.compute_pi_W()
        mc.compute_M()
        mc.compute_V()
        mc.compute_kemeny_W()
        mc.compute_network_variance()
        mc.compute_efficiency_index()
        
        # This should be ~0 if hard constraint is satisfied
        pi_error = np.linalg.norm(mc.pi - self.pi_hat)
        
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
        """
        Compute objective function.
        
        With hard Ï€ constraint on the standard stationary distribution.
        Note: pi_W (weighted distribution) may still differ from pi_hat.
        
        For stochastic weights, we use bounded objectives to prevent explosion.
        """
        P = x_to_matrix(x, self.N, self.edge_matrix, self.bUndirected)
        
        try:
            metrics = self.evaluate_metrics(P)
        except (np.linalg.LinAlgError, ValueError):
            # Return large value for infeasible points
            return 1e10
        
        eff_idx = metrics['Eff_Idx']
        net_var = metrics['Net_Var']
        k_w = metrics['K_W']
        
        if not np.isfinite(eff_idx) or not np.isfinite(net_var):
            return 1e10
        
        if self.objective_type == 'maximize_efficiency':
            if eff_idx <= -1 or k_w <= 0 or net_var < 0:
                return 1e10
            return -np.log1p(eff_idx)
        elif self.objective_type == 'maximize_efficiency_unbounded':
            # Original unbounded version (use with caution!)
            return -eff_idx
        elif self.objective_type == 'minimize_efficiency':
            if eff_idx <= -1:
                return -1e10
            return np.log1p(eff_idx)
        elif self.objective_type == 'minimize_variance':
            return net_var
        elif self.objective_type == 'minimize_mean':
            return k_w
        elif self.objective_type == 'maximize_variance':
            if net_var <= -1:
                return 1e10
            return -np.log1p(net_var)
        elif self.objective_type == 'maximize_variance_unbounded':
            return -net_var
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")
    
    def project(self, x):
        """
        Project x onto the feasible set satisfying:
        - Row-sum constraints (Markov chain)
        - Stationary distribution constraint (Ï€Ì‚áµ€P = Ï€Ì‚áµ€) [if hard constraint]
        - Box constraints [eta, 1-eta]
        
        Uses Dykstra's alternating projection algorithm for hard constraint.
        """
        if self.use_hard_constraint and self.bProjectionReady:
            return dykstra_projection(
                x, 
                self.A_pinv_b, 
                self.C_proj, 
                self.eta,
                tol=1e-10,
                max_iter=50000
            )
        else:
            # Simple Markov projection (no stationary constraint)
            return projection_markov_simple(x, self.eta, self.neighborhoods, self.mA)
    
    def get_feasible_initial_point(self):
        """
        Get a feasible initial point satisfying all constraints.
        
        Returns:
            x_init: Initial decision vector
        """
        if self.use_hard_constraint and self.bProjectionReady:
            # Start with Aâºb (particular solution satisfying Ax = b)
            x_init = self.A_pinv_b.copy()
            # Project to ensure box constraints
            x_init = self.project(x_init)
        else:
            # Uniform distribution over neighbors
            x_init = np.zeros(self.d)
            for i, subset in enumerate(self.neighborhoods):
                if len(subset) > 0:
                    for idx in subset:
                        x_init[idx] = 1.0 / len(subset)
        
        return x_init
    
    def verify_constraints(self, x, tol=1e-6):
        """
        Verify that x satisfies all constraints.
        
        Returns:
            dict with constraint violations
        """
        P = x_to_matrix(x, self.N, self.edge_matrix, self.bUndirected)
        
        # Check row sums
        row_sums = np.sum(P, axis=1)
        row_sum_error = np.max(np.abs(row_sums - 1))
        
        # Check stationary distribution (if hard constraint)
        metrics = self.evaluate_metrics(P)
        pi_error = metrics['pi_error']
        
        # Check bounds
        bound_violations = np.sum(x < self.eta - tol) + np.sum(x > 1 - self.eta + tol)
        
        is_feasible = (row_sum_error < tol) and (bound_violations == 0)
        if self.use_hard_constraint:
            is_feasible = is_feasible and (pi_error < tol)
        
        return {
            'row_sum_error': row_sum_error,
            'pi_error': pi_error,
            'bound_violations': bound_violations,
            'is_feasible': is_feasible
        }