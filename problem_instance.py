"""
Problem instance for efficiency index optimization with stochastic weights.
"""

import numpy as np

from utils import create_edge_matrix, build_neighborhoods, x_to_matrix, projection_markov
from network_stochastic import MarkovChainStochastic


class EfficiencyProblemInstanceStochastic:
    """
    Problem instance for efficiency index optimization with STOCHASTIC weights.
    
    The optimization is with respect to P (transition probabilities).
    W and W2 are fixed parameters characterizing the stochastic edge weights.
    """
    
    def __init__(self, mA, W, W2, eta=1e-4, pi_hat=None, 
                 objective_type='maximize_efficiency',
                 pi_penalty_weight=1e3):
        """
        Parameters:
            mA: Adjacency matrix
            W: Mean weight matrix μ(i,j) = E[W(i,j)]
            W2: Second moment matrix E[W²(i,j)] - NOT the square of W!
            eta: Small constant for numerical stability in constraints
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
        """Build constraint matrices for row-stochastic constraint."""
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
        """
        Evaluate all metrics for a given transition matrix P.
        
        Returns dict with: pi, pi_W, M, V, K_W, Net_Var, Eff_Idx, pi_error
        """
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
        """
        Compute objective function.
        
        Supported objective types:
        - 'maximize_efficiency': Maximize Var/Mean (more unpredictable)
        - 'minimize_efficiency': Minimize Var/Mean
        - 'minimize_variance': Minimize variance directly (more consistent)
        - 'maximize_variance': Maximize variance directly
        
        All include penalty for deviating from target stationary distribution.
        """
        P = x_to_matrix(x, self.N, self.edge_matrix, self.bUndirected)
        metrics = self.evaluate_metrics(P)
        
        pi_error = metrics['pi_error']
        penalty = self.pi_penalty_weight * (pi_error ** 2)
        
        if self.objective_type == 'maximize_efficiency':
            return -metrics['Eff_Idx'] + penalty
        elif self.objective_type == 'minimize_efficiency':
            return metrics['Eff_Idx'] + penalty
        elif self.objective_type == 'minimize_variance':
            return metrics['Net_Var'] + penalty
        elif self.objective_type == 'maximize_variance':
            return -metrics['Net_Var'] + penalty
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")
    
    def project(self, x):
        """Project x onto the feasible set (valid Markov chain)."""
        return projection_markov(x, self.eta, self.neighborhoods, self.mA)
