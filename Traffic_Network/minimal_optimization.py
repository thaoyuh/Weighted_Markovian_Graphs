import numpy as np
import functions as fn
from scipy.optimize import minimize


def optimize_edge_weights(
    P_new,  # The current broken topology (projected)
    pi_star,  # The stationary distribution we must preserve
    W_current,  # Current weights (from step k-1) -> The anchor for minimization
    W_init,  # Original weights (step 0) -> Used to define K_target/V_target
    P_init,  # Original topology -> Used to define K_target/V_target
    W_min,
    W_max,  # Speed limit bounds
    epsilon=1e-2,  # Tolerance for solver
    printing=False,
):
    """
    Finds new weights W that minimize ||W - W_current||^2 (Minimal Intervention),
    subject to K(W) <= K(W_init) and V(W) <= V(W_init).
    """
    N = P_new.shape[0]

    # 1. OPTIMIZATION SETUP: VECTORIZATION
    # We only optimize edges that actually exist in P_new (active edges)
    active_mask = P_new > 0
    w_active_current = W_current[active_mask]
    w_active_min = W_min[active_mask]
    w_active_max = W_max[active_mask]

    # 2. DEFINE TARGETS (The "Not Worse" Thresholds)
    # We calculate what the network was capable of at N=0
    K_target = fn.kemeny_constant(P_init, pi_star, W_init)
    V_target = fn.variance_based_kemeny(P_init, pi_star, W_init, W_init**2)

    if printing:
        print(f"Targets -> Max K: {K_target:.4f}, Max V: {V_target:.4f}")

    # 3. PRE-CALCULATE SENSITIVITY (The "Linear trick")
    # Kemeny(W) is linear: K = c dot w
    # c_ij = tr(Z_new) * pi_i * p_new_ij
    Z_new = fn.get_fundamental_matrix(P_new, pi_star)
    tr_Z = np.trace(Z_new)

    # Create the sensitivity matrix
    sensitivity_matrix = tr_Z * (pi_star[:, np.newaxis] * P_new)
    c_active = sensitivity_matrix[active_mask]  # Only need 'c' for active edges

    # 4. OBJECTIVE: Minimize change from CURRENT state (Step k-1)
    def objective(w_active):
        # Min || w - w_current ||^2
        return np.sum((w_active - w_active_current) ** 2)

    def objective_jac(w_active):
        # Gradient is linear: 2(w - w_current)
        return 2 * (w_active - w_active_current)

    # 5. CONSTRAINTS

    # Constraint 1: Kemeny must be <= K_target
    # Rewritten as: K_target - K_new >= 0
    def constraint_kemeny(w_active):
        K_new = np.dot(c_active, w_active)
        return K_target - K_new

    # Jacobian for Kemeny is just -c (Constant!)
    # Providing this makes SLSQP incredibly stable
    def constraint_kemeny_jac(w_active):
        return -c_active

    # Constraint 2: Variance must be <= V_target
    # Rewritten as: V_target - V_new >= 0
    # Note: Variance is Non-Linear (depends on W and W^2), so this is the bottleneck
    def constraint_variance(w_active):
        # Rebuild full matrix for the heavy Variance calculation
        W_temp = np.zeros((N, N))
        W_temp[active_mask] = w_active
        # Note: Variance calculation needs the SQUARE of weights too
        V_new = fn.variance_based_kemeny(P_new, pi_star, W_temp, W_temp**2)
        return V_target - V_new

    # 6. RUN SOLVER
    bounds = list(zip(w_active_min, w_active_max))

    cons = [
        {"type": "ineq", "fun": constraint_kemeny, "jac": constraint_kemeny_jac},
        {"type": "ineq", "fun": constraint_variance},
    ]

    result = minimize(
        fun=objective,
        x0=w_active_current,  # Start from current weights
        jac=objective_jac,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
        options={"ftol": epsilon, "maxiter": 100, "disp": printing},
    )

    if result.success:
        if printing:
            print("Optimization Successful!")
        # Reconstruct the full N x N matrix
        W_opt = np.zeros((N, N))
        W_opt[active_mask] = result.x
        return W_opt
    else:
        if printing:
            print(f"Optimization Failed: {result.message}")
        return None
