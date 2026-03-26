"""
Grid network generation with stochastic weights.

Generates grid networks where edges have stochastic travel times
characterized by their coefficient of variation (CV).
"""

import numpy as np


def generate_grid_network_stochastic(n, obstacles=None, diagonal=False,
                                      cv_low=0.3, cv_high=1.5, 
                                      high_cv_fraction=0.3,
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
    """
    Create a target stationary distribution for the grid.
    
    Parameters:
        n: Grid size
        obstacle_mask: Boolean array, True for obstacle nodes
        priority_positions: Dict mapping (row, col) to priority weight
    
    Returns:
        pi_hat: Target stationary distribution (normalized, zeros for obstacles)
    """
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
