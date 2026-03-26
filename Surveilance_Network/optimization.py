"""
Optimization algorithms for Markov chain optimization.

Implements SPSA with null space perturbation (Franssen et al.) for
hard-constrained optimization with guaranteed stationary distribution.
"""

import numpy as np

from utils import x_to_matrix


def solve_spsa_efficiency(problem, x_init, max_iter=20000, 
                          a=1e-2, a_eps=100, e=1e-2, r_nu=0.101,
                          obj_interval=1000, verbose=True,
                          max_obj_value=1e10,
                          fixed_step_size=True):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimization 
    for efficiency index.
    
    When problem.use_hard_constraint=True, uses null space perturbation
    (Franssen et al. Section IV-B) to stay on the constraint manifold:
    
        δ̃(x) = [J(x - ηBΔ) - J(x + ηBΔ)] / (2η) × BΔ
    
    where B spans the null space of the constraint matrix A.
    
    Parameters:
        problem: EfficiencyProblemInstanceStochastic instance
        x_init: Initial parameter vector
        max_iter: Maximum number of iterations
        a: Step size (constant if fixed_step_size=True, or numerator if decaying)
        a_eps: Step size denominator constant (ignored if fixed_step_size=True)
        e: Perturbation size numerator
        r_nu: Perturbation size exponent
        obj_interval: Interval for recording metrics
        verbose: If True, print progress
        max_obj_value: Maximum allowed objective value (for numerical stability)
        fixed_step_size: If True, use constant step size a; if False, use decaying a/(k+a_eps+1)
    
    Returns:
        iter_hist: List of iteration numbers
        eff_hist: List of efficiency index values
        kw_hist: List of K_W values
        var_hist: List of variance values
        pi_err_hist: List of pi error values
        best_x: Best parameter vector found
        best_obj: Best objective value found
    """
    
    # Check if we can use null space perturbation
    use_null_space = (hasattr(problem, 'use_hard_constraint') and 
                      problem.use_hard_constraint and
                      hasattr(problem, 'C') and 
                      problem.C is not None and
                      problem.C.size > 0)
    
    if use_null_space:
        null_dim = problem.C.shape[1]
        if verbose:
            print(f"Using null space perturbation (dim={null_dim})")
    else:
        null_dim = problem.d
        if verbose:
            print(f"Using full space perturbation (dim={null_dim})")
    
    x = problem.project(x_init.copy())
    best_x = x.copy()
    best_obj = float('inf')
    
    # Compute initial objective
    try:
        init_obj = problem.objective(x)
        if np.isfinite(init_obj):
            best_obj = init_obj
    except:
        pass
    
    iter_hist = []
    eff_hist = []
    kw_hist = []
    var_hist = []
    pi_err_hist = []
    
    stagnation_count = 0
    prev_best = best_obj
    
    for k in range(max_iter):
        # Perturbation size (always decays for stability)
        c_k = e / ((k + 1) ** r_nu)
        
        # Step size: fixed or decaying
        if fixed_step_size:
            a_k = a
        else:
            a_k = a / (k + a_eps + 1)
        
        if use_null_space:
            # === NULL SPACE PERTURBATION (Franssen et al.) ===
            # Random perturbation in null space coordinates
            delta_u = 2 * (np.random.rand(null_dim) > 0.5).astype(float) - 1
            
            # Map to x-space: δ = B × Δ where B = C (null space basis)
            delta_x = problem.C @ delta_u
            
            # Perturbed points (stay on constraint manifold)
            x_plus = problem.project(x + c_k * delta_x)
            x_minus = problem.project(x - c_k * delta_x)
            
            # Evaluate objectives
            try:
                f_plus = problem.objective(x_plus)
                f_minus = problem.objective(x_minus)
            except:
                continue
            
            # Skip if objectives are invalid
            if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                continue
            if abs(f_plus) > max_obj_value or abs(f_minus) > max_obj_value:
                continue
            
            # Gradient estimate in null space coordinates
            # g_u = [f(x+) - f(x-)] / (2c) × Δ_u (element-wise)
            grad_u = (f_plus - f_minus) / (2 * c_k * delta_u + 1e-12)
            grad_u = np.clip(grad_u, -100, 100)
            
            # Steepest feasible descent direction: δ = B × g_u
            descent_direction = problem.C @ grad_u
            
            # Update step
            x_new = x - a_k * descent_direction
            x = problem.project(x_new)
            
        else:
            # === FULL SPACE PERTURBATION (original approach) ===
            delta = 2 * (np.random.rand(problem.d) > 0.5).astype(float) - 1
            
            x_plus = problem.project(x + c_k * delta)
            x_minus = problem.project(x - c_k * delta)
            
            try:
                f_plus = problem.objective(x_plus)
                f_minus = problem.objective(x_minus)
            except:
                continue
            
            if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                continue
            if abs(f_plus) > max_obj_value or abs(f_minus) > max_obj_value:
                continue
            
            grad_estimate = (f_plus - f_minus) / (2 * c_k * delta + 1e-12)
            grad_estimate = np.clip(grad_estimate, -100, 100)
            
            x_new = x - a_k * grad_estimate
            x = problem.project(x_new)
        
        # Track best solution
        if np.isfinite(f_plus) and f_plus < best_obj:
            best_obj = f_plus
            best_x = x_plus.copy()
        if np.isfinite(f_minus) and f_minus < best_obj:
            best_obj = f_minus
            best_x = x_minus.copy()
        
        # Record metrics at intervals
        if (k + 1) % obj_interval == 0 or k == 0:
            P = x_to_matrix(best_x, problem.N, problem.edge_matrix, problem.bUndirected)
            try:
                metrics = problem.evaluate_metrics(P)
                
                # Only record if metrics are reasonable
                if np.isfinite(metrics['Eff_Idx']) and abs(metrics['Eff_Idx']) < 1e12:
                    iter_hist.append(k + 1)
                    eff_hist.append(metrics['Eff_Idx'])
                    kw_hist.append(metrics['K_W'])
                    var_hist.append(metrics['Net_Var'])
                    pi_err_hist.append(metrics['pi_error'])
                    
                    if verbose:
                        print(f"Iter {k+1:5d}: Eff={metrics['Eff_Idx']:.4f}, "
                              f"K_W={metrics['K_W']:.4f}, Var={metrics['Net_Var']:.4f}, "
                              f"π_err={metrics['pi_error']:.6f}")
                    
                    # Check for stagnation
                    if abs(best_obj - prev_best) < 1e-8:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                    prev_best = best_obj
                    
                    # # Early stopping if stagnated
                    # if stagnation_count > 10:
                    #     if verbose:
                    #         print(f"Converged at iteration {k+1}")
                    #     break
            except:
                continue
    
    return iter_hist, eff_hist, kw_hist, var_hist, pi_err_hist, best_x, best_obj