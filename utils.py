"""
Utility functions for Markov chain optimization.
"""

import numpy as np


def row_normalize(M):
    """Normalize rows of matrix M to sum to 1."""
    row_sums = np.sum(M, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


def create_edge_matrix(mA):
    """
    Create edge matrix from adjacency matrix.
    
    Returns:
        E: (num_edges, 2) array where E[k] = [i, j] for edge k
    """
    indices = np.nonzero(mA)
    num_edges = indices[0].shape[0]
    E = np.zeros((num_edges, 2), dtype=int)
    E[:, 0] = indices[0]
    E[:, 1] = indices[1]
    return E


def build_neighborhoods(edge_matrix, N):
    """
    Build list of edge indices for each node.
    
    Returns:
        neighborhoods: List where neighborhoods[i] contains indices of edges leaving node i
    """
    neighborhoods = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edge_matrix):
        neighborhoods[i].append(idx)
    return neighborhoods


def x_to_matrix(x, N, edge_matrix, bUndirected):
    """
    Convert edge weight vector x to transition matrix P.
    
    Parameters:
        x: Vector of edge weights
        N: Number of nodes
        edge_matrix: Edge index matrix
        bUndirected: If True, symmetrize the matrix
    
    Returns:
        P: N×N transition matrix
    """
    P = np.zeros((N, N))
    P[edge_matrix[:, 0], edge_matrix[:, 1]] = x
    if bUndirected:
        return P + P.T
    return P


def proj_c_simplex(v, c=1, tol=1e-8):
    """
    Project vector v onto the simplex {x : sum(x) = c, x >= 0}.
    """
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
    """
    Project x onto the set of valid Markov chain parameters.
    Each row must sum to 1, with minimum value eta.
    """
    x = x_to_proj - eta
    x_proj = []
    for i, subset in enumerate(neighborhoods):
        if subset:
            n_edges = len(subset)
            c = 1 - n_edges * eta
            x_proj.extend(proj_c_simplex(x[subset], c=c).tolist())
    return np.array(x_proj) + eta
