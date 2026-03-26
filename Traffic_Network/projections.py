"""
Project a (possibly invalid) transition matrix P_hat to a valid transition matrix
that keeps zeros fixed (only alters specified non-zero positions), and enforces:
  - row sums = 1
  - stationary distribution pi_hat (i.e., pi_hat @ P = pi_hat)
  - box constraints: epsilon <= P_ij <= 1-epsilon for free entries

Uses Dykstra's algorithm alternating between:
  X1 = { x : A x = b }   (linear equalities: row sums and stationarity)
  X2 = { x : epsilon <= x <= 1-epsilon }  (box)

Returns projected P matrix (same shape as P_hat).
"""

import numpy as np


def build_A_b_from_mask_and_pi(mask, pi_hat):
    """
    mask: boolean matrix (N x N). True for free (modifiable) entries; False for fixed zeros.
    pi_hat: row vector shape (N,) summing to 1.

    Returns:
      A (m x d) matrix and b (m,) where d = number of free entries and m is the number
      of linear constraints (N row-sum constraints + N stationarity constraints),
      but rows may be linearly dependent. We keep them all: projection uses pseudo-inverse.
    """
    N = mask.shape[0]
    idxs = np.argwhere(mask)  # list of (i,j) pairs
    d = idxs.shape[0]

    # Map (i,j) -> index in x
    pos2idx = {(int(i), int(j)): k for k, (i, j) in enumerate(idxs)}

    # Build A: first N rows for row-sum constraints
    A_rows = []
    b_rows = []

    # Row-sum constraints: for each i sum_j x_{i,j} = 1
    for i in range(N):
        row = np.zeros(d)
        for j in range(N):
            if mask[i, j]:
                row[pos2idx[(i, j)]] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)

    # Stationarity constraints: for each j, sum_i pi_i * P_{i,j} = pi_j
    # This is linear in the unknown P_{i,j}
    for j in range(N):
        row = np.zeros(d)
        for i in range(N):
            if mask[i, j]:
                row[pos2idx[(i, j)]] = float(pi_hat[i])
        A_rows.append(row)
        b_rows.append(float(pi_hat[j]))

    A = np.vstack(A_rows)  # shape (2N, d)
    b = np.array(b_rows)  # shape (2N,)
    return A, b, idxs, pos2idx


def proj_affine(y, A, b, regularize=1e-12):
    """
    Project y (d,) to { x : A x = b } in Euclidean norm.
    Formula: x = y - A^T @ (A A^T)^(-1) @ (A y - b)
    uses pseudo-inverse if needed.
    """
    # compute r = A y - b
    Ay_minus_b = A.dot(y) - b  # shape (m,)
    # Solve (A A^T) z = r
    AA_T = A.dot(A.T)
    # regularize to improve numeric stability if near-singular
    # use np.linalg.solve if well-conditioned, otherwise pinv fallback
    try:
        # small regularization
        z = np.linalg.solve(AA_T + regularize * np.eye(AA_T.shape[0]), Ay_minus_b)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(AA_T).dot(Ay_minus_b)
    x = y - A.T.dot(z)
    return x


def proj_box(y, eps):
    """Elementwise clipping to [eps, 1-eps]."""
    upper = 1.0 - eps
    return np.minimum(np.maximum(y, eps), upper)


def dykstra_projection(y0, A, b, eps=1e-9, tol=1e-9, max_iters=10000, verbose=False):
    """
    Dykstra algorithm to compute projection onto intersection X1 ∩ X2,
    where X1 = {x : A x = b} and X2 = [eps, 1-eps]^d.
    y0: initial vector (d,)
    Returns projected vector x
    """
    d = y0.size
    # Dykstra variables:
    x = y0.copy()
    p = np.zeros_like(x)  # for first set
    q = np.zeros_like(x)  # for second set

    for it in range(max_iters):
        # Project onto X1
        y = proj_affine(x + p, A, b)
        p = x + p - y

        # Project onto X2
        x_new = proj_box(y + q, eps)
        q = y + q - x_new

        # check convergence
        if np.linalg.norm(x_new - x, ord=2) <= tol:
            if verbose:
                print(f"dykstra converged at iter {it}")
            return x_new
        x = x_new

    if verbose:
        print("Dykstra did not fully converge within max_iters; returning last iterate")
    return x


def project_P(P_hat, pi_hat, eps=1e-9, tol=1e-9, max_iters=10000, verbose=False):
    """
    Main function.
    P_hat : (N,N) array (can have rows that don't sum to 1 because of removed edges)
    pi_hat: (N,) desired stationary row-vector (sums to 1)
    eps   : lower bound for free entries (small positive)
    Returns:
      P_proj (N,N) projected matrix (rows sum to 1; stationary pi_hat; zeros kept at original zero positions)
    """
    P_hat = np.array(P_hat, dtype=float)
    N = P_hat.shape[0]
    assert P_hat.shape == (N, N)
    pi_hat = np.array(pi_hat, dtype=float)
    assert pi_hat.shape == (N,)
    assert abs(pi_hat.sum() - 1.0) < 1e-2, "pi_hat must sum to 1"

    # Define mask of modifiable entries: True where P_hat was nonzero OR we allow keeping edges that were zero?
    # Here we follow your requirement: only alter non-zero values in hat P -> mask = P_hat > 0
    mask = (P_hat > 0).astype(bool)

    # If some rows have zero free entries (all edges removed), infeasible
    for i in range(N):
        if not np.any(mask[i, :]):
            if verbose:
                print(f"Row {i} has no available outgoing edges (cannot normalize).")
            return None

    # Build linear system using free variables only
    A, b, idxs, pos2idx = build_A_b_from_mask_and_pi(mask, pi_hat)
    d = A.shape[1]

    # initial y: vector of current free entries extracted from P_hat
    y0 = np.array([P_hat[int(i), int(j)] for (i, j) in idxs], dtype=float)

    # As a small safety, ensure initial y0 within [eps,1-eps]
    y0 = np.minimum(np.maximum(y0, eps), 1.0 - eps)

    # run Dykstra to project onto intersection
    x_proj = dykstra_projection(
        y0, A, b, eps=eps, tol=tol, max_iters=max_iters, verbose=verbose
    )

    # Map back to full matrix
    P_proj = np.zeros_like(P_hat)
    for k, (i, j) in enumerate(idxs):
        P_proj[int(i), int(j)] = float(x_proj[k])

    # final numerical fix: renormalize each row if tiny residual due to numerical error
    for i in range(N):
        s = P_proj[i, :].sum()
        if s <= 0:
            if verbose:
                print(f"Row {i} sums to non-positive value after projection (s={s})")
            return None
        P_proj[i, :] = P_proj[i, :] / s  # this preserves zeros and enforces row-sum=1

    # check stationarity residual
    res = np.linalg.norm(pi_hat.dot(P_proj) - pi_hat, ord=1)
    if res > 1e-6:
        if verbose:
            print(
                "Warning: projected P does not satisfy stationarity within tolerance; residual L1=",
                res,
            )
        return None
    if verbose:
        print("stationarity residual (L1):", res)
    return P_proj
