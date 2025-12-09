"""
Markov Chain class with support for stochastic edge weights.

Key difference from deterministic case:
- W stores mean weights μ(i,j) = E[W(i,j)]
- W2 stores second moments E[W²(i,j)], NOT the square of W
- This allows Var[W(i,j)] = E[W²] - E[W]² > 0
"""

import numpy as np
from numpy.linalg import eig, inv
from itertools import product

from utils import create_edge_matrix, x_to_matrix


class MarkovChainStochastic:
    """
    Markov chain class that supports STOCHASTIC edge weights.
    
    The variance formula uses E[W²] (second moment), which captures
    the stochasticity of edge weights through:
        E[W²] = E[W]² + Var[W] = μ² × (1 + CV²)
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
        """Transition probability matrix."""
        return x_to_matrix(self.x, self.n, self.edge_matrix, self.bUndirected)
    
    @staticmethod
    def P_to_x(P, mA, bUndirected):
        """Convert transition matrix P to edge weight vector x."""
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
        """Compute outer product matrix for stationary distribution."""
        if self.pi is None:
            self.compute_pi()
        self.Pi = np.outer(np.ones(self.n), self.pi)
        return self.Pi
    
    def compute_pi_W(self):
        """
        Compute time-weighted stationary distribution.
        
        π_W(i) = π(i) × Ū(i) / Σ_k π(k) × Ū(k)
        
        where Ū(i) = Σ_j P(i,j) × μ(i,j) is the expected weight leaving state i.
        """
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
        """Compute fundamental matrix Z = (I - P + Π)^(-1)."""
        if self.Pi is None:
            self.compute_Pi()
        I = np.eye(self.n)
        self.Z = inv(I - self.P + self.Pi)
        return self.Z
    
    def compute_M(self):
        """
        Compute mean first-passage time matrix M.
        
        M(i,j) = E[τ(i,j)] where τ(i,j) is the first passage time from i to j.
        """
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
        
        The variance includes contributions from both:
        1. Variability in the path taken (from P)
        2. Variability in edge weights (from E[W²] - E[W]²)
        """
        if self.M is None:
            self.compute_M()
        
        I = np.eye(self.n)
        Ones = np.ones((self.n, self.n))
        P_dot_W = np.multiply(self.P, self.W)    # P ⊙ μ
        P_dot_W2 = np.multiply(self.P, self.W2)  # P ⊙ E[W²] (SECOND MOMENT!)
        
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
        """
        Compute weighted Kemeny constant.
        
        K_W = π_W^T × M × π_W
        """
        if self.pi_W is None:
            self.compute_pi_W()
        if self.M is None:
            self.compute_M()
        self.K_W = self.pi_W @ self.M @ self.pi_W.T
        return self.K_W
    
    def compute_network_variance(self):
        """
        Compute network-level variance.
        
        Net_Var = π_W^T × V × π_W
        """
        if self.pi_W is None:
            self.compute_pi_W()
        if self.V is None:
            self.compute_V()
        self.Net_Var = self.pi_W @ self.V @ self.pi_W.T
        return self.Net_Var
    
    def compute_efficiency_index(self):
        """
        Compute efficiency index = Variance / Mean.
        
        Higher efficiency means more unpredictable patrol times.
        """
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
        """Clear all cached computations."""
        self.pi = None
        self.Pi = None
        self.pi_W = None
        self.Z = None
        self.M = None
        self.V = None
        self.K_W = None
        self.Net_Var = None
        self.Eff_Idx = None
