import numpy as np


def project_P(P_hat, pi_hat, eps=1e-5, tol=1e-5, max_iters=2000, verbose=False):
    """
    Naive projection of P_hat, where we simply divide each row by its sum to get a stochastic matrix,
    """
    P_proj = np.array(P_hat, dtype=float)
    N = P_proj.shape[0]
    for i in range(N):
        row_sum = P_proj[i, :].sum()
        if row_sum > 0:
            P_proj[i, :] = P_proj[i, :] / row_sum
        else:
            # if the row sums to zero, distribute uniformly
            P_proj[i, :] = 1.0 / N
    return P_proj
