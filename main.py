"""
Main test script for Grid Network Surveillance with Stochastic Weights.

This script demonstrates the optimization of Markov chain transition probabilities
for surveillance on a grid network with stochastic edge weights.

Key concepts:
- Optimization is with respect to P (transition probabilities)
- W (mean weights) and W2 (second moments) are fixed parameters
- Stochastic weights: E[W²] = E[W]² × (1 + CV²), where CV is coefficient of variation
- CV < 1: reliable edges, CV > 1: unreliable edges
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from modular files
from utils import x_to_matrix, build_neighborhoods
from network_stochastic import MarkovChainStochastic
from problem_instance import EfficiencyProblemInstanceStochastic
from grid_generation import generate_grid_network_stochastic, create_grid_target_distribution
from optimization import solve_spsa_efficiency
from visualization import (
    plot_optimization_results,
    plot_optimization_comparison,
    plot_grid_network,
    plot_policy_comparison,
    plot_policy_vs_cv
)


def analyze_policy_vs_cv(P, CV_matrix, mA, title="Policy vs CV Analysis"):
    """Analyze whether the optimal policy favors high-CV or low-CV edges."""
    mask = mA > 0
    P_values = P[mask]
    CV_values = CV_matrix[mask]
    
    low_cv_mask = CV_values < 1
    high_cv_mask = CV_values >= 1
    
    avg_P_low_cv = np.mean(P_values[low_cv_mask]) if np.any(low_cv_mask) else 0
    avg_P_high_cv = np.mean(P_values[high_cv_mask]) if np.any(high_cv_mask) else 0
    correlation = np.corrcoef(P_values, CV_values)[0, 1]
    
    print(f"\n{title}")
    print("-" * 50)
    print(f"Average P for low-CV edges (CV < 1):  {avg_P_low_cv:.4f}")
    print(f"Average P for high-CV edges (CV ≥ 1): {avg_P_high_cv:.4f}")
    print(f"Correlation(P, CV): {correlation:.4f}")
    
    if avg_P_high_cv > avg_P_low_cv:
        print("→ Policy FAVORS unreliable edges (exploiting variance)")
    else:
        print("→ Policy FAVORS reliable edges (avoiding variance)")
    
    return {'avg_P_low_cv': avg_P_low_cv, 'avg_P_high_cv': avg_P_high_cv, 'correlation': correlation}


def test_deterministic_vs_stochastic(seed=42):
    """Compare optimization results between deterministic and stochastic weights."""
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    print("="*80)
    print("TEST: DETERMINISTIC vs STOCHASTIC WEIGHT COMPARISON")
    print("="*80)
    print("Using HARD CONSTRAINT for stationary distribution (well-posed problem)")
    print(f"Random seed: {seed}")
    
    # Grid parameters
    n = 5
    obstacles = [(2, 2)]
    
    print(f"\nGrid size: {n}×{n} = {n*n} nodes")
    print(f"Obstacle positions: {obstacles}")
    
    # Generate network with stochastic weights
    mA, W_mean, W2_stoch, CV_matrix, obstacle_mask, grid_positions = \
        generate_grid_network_stochastic(n, obstacles=obstacles, cv_low=0.3, cv_high=1.5)
    
    # For deterministic: W2 = W²
    W2_det = W_mean ** 2
    
    # Target distribution (uniform over non-obstacle nodes)
    pi_hat = create_grid_target_distribution(n, obstacle_mask)
    
    # =========== CASE 1: DETERMINISTIC ===========
    print("\n" + "="*60)
    print("CASE 1: DETERMINISTIC WEIGHTS (CV = 0)")
    print("="*60)
    
    problem_det = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W_mean, W2=W2_det,
        eta=1e-4, pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        use_hard_constraint=True  # HARD CONSTRAINT for well-posed problem
    )
    
    # Get feasible initial point from the problem instance
    x_init = problem_det.get_feasible_initial_point()
    
    P_init = x_to_matrix(x_init, problem_det.N, problem_det.edge_matrix, False)
    metrics_det_init = problem_det.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics (Deterministic):")
    print(f"  Efficiency Index: {metrics_det_init['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_det_init['K_W']:.6f}")
    print(f"  Variance: {metrics_det_init['Net_Var']:.6f}")
    print(f"  π error: {metrics_det_init['pi_error']:.6f}")
    
    # Reset seed for reproducible SPSA
    np.random.seed(seed + 1)
    print("\nRunning SPSA optimization (Deterministic)...")
    iter_det, eff_det, kw_det, var_det, pi_err_det, best_x_det, _ = solve_spsa_efficiency(
        problem_det, x_init, verbose=True,
        max_iter=10000
    )
    
    P_det = x_to_matrix(best_x_det, problem_det.N, problem_det.edge_matrix, False)
    metrics_det_final = problem_det.evaluate_metrics(P_det)
    
    print(f"\nFinal metrics (Deterministic):")
    print(f"  Efficiency Index: {metrics_det_final['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_det_final['K_W']:.6f}")
    print(f"  Variance: {metrics_det_final['Net_Var']:.6f}")
    print(f"  π error: {metrics_det_final['pi_error']:.6f}")
    
    # =========== CASE 2: STOCHASTIC ===========
    print("\n" + "="*60)
    print("CASE 2: STOCHASTIC WEIGHTS (Mixed CVs)")
    print("="*60)
    
    mask = mA > 0
    cvs = CV_matrix[mask]
    print(f"\nCV Statistics: Min={cvs.min():.3f}, Max={cvs.max():.3f}, Mean={cvs.mean():.3f}")
    print(f"Edges with CV < 1: {np.sum(cvs < 1)} ({100*np.sum(cvs < 1)/len(cvs):.1f}%)")
    print(f"Edges with CV > 1: {np.sum(cvs > 1)} ({100*np.sum(cvs > 1)/len(cvs):.1f}%)")
    
    problem_stoch = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W_mean, W2=W2_stoch,
        eta=1e-4, pi_hat=pi_hat,
        objective_type='maximize_efficiency',
        use_hard_constraint=True  # HARD CONSTRAINT for well-posed problem
    )
    
    # Get feasible initial point for stochastic problem (same as det since same constraints)
    x_init_stoch = problem_stoch.get_feasible_initial_point()
    
    metrics_stoch_init = problem_stoch.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics (Stochastic):")
    print(f"  Efficiency Index: {metrics_stoch_init['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_stoch_init['K_W']:.6f}")
    print(f"  Variance: {metrics_stoch_init['Net_Var']:.6f}")
    print(f"  π error: {metrics_stoch_init['pi_error']:.6f}")
    
    # Reset seed for reproducible SPSA (different seed than deterministic)
    np.random.seed(seed + 2)
    print("\nRunning SPSA optimization (Stochastic)...")
    iter_stoch, eff_stoch, kw_stoch, var_stoch, pi_err_stoch, best_x_stoch, _ = solve_spsa_efficiency(
        problem_stoch, x_init_stoch, verbose=True,
        max_iter=10000
    )
    
    P_stoch = x_to_matrix(best_x_stoch, problem_stoch.N, problem_stoch.edge_matrix, False)
    metrics_stoch_final = problem_stoch.evaluate_metrics(P_stoch)
    
    print(f"\nFinal metrics (Stochastic):")
    print(f"  Efficiency Index: {metrics_stoch_final['Eff_Idx']:.6f}")
    print(f"  K_W (Mean): {metrics_stoch_final['K_W']:.6f}")
    print(f"  Variance: {metrics_stoch_final['Net_Var']:.6f}")
    print(f"  π error: {metrics_stoch_final['pi_error']:.6f}")
    
    # =========== COMPARISON ===========
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    diff_var_init = metrics_stoch_init['Net_Var'] - metrics_det_init['Net_Var']
    
    print(f"\n{'Metric':<30} {'Deterministic':<20} {'Stochastic':<20}")
    print("-" * 70)
    print(f"{'INITIAL Efficiency Index':<30} {metrics_det_init['Eff_Idx']:<20.4f} {metrics_stoch_init['Eff_Idx']:<20.4f}")
    print(f"{'INITIAL Variance':<30} {metrics_det_init['Net_Var']:<20.4f} {metrics_stoch_init['Net_Var']:<20.4f}")
    print(f"{'INITIAL π error':<30} {metrics_det_init['pi_error']:<20.6f} {metrics_stoch_init['pi_error']:<20.6f}")
    print(f"{'FINAL Efficiency Index':<30} {metrics_det_final['Eff_Idx']:<20.4f} {metrics_stoch_final['Eff_Idx']:<20.4f}")
    print(f"{'FINAL Variance':<30} {metrics_det_final['Net_Var']:<20.4f} {metrics_stoch_final['Net_Var']:<20.4f}")
    print(f"{'FINAL π error':<30} {metrics_det_final['pi_error']:<20.6f} {metrics_stoch_final['pi_error']:<20.6f}")
    
    print(f"\nInitial variance increase from stochastic W: {diff_var_init:.4f} ({100*diff_var_init/metrics_det_init['Net_Var']:.2f}%)")
    
    # Report pi error changes
    print("\n" + "="*80)
    print("STATIONARY DISTRIBUTION DEVIATION ANALYSIS")
    print("="*80)
    print(f"\nTarget distribution: Uniform over {np.sum(~obstacle_mask)} non-obstacle nodes")
    print(f"\n{'Case':<20} {'Initial π error':<20} {'Final π error':<20} {'Change':<20}")
    print("-" * 80)
    det_pi_change = metrics_det_final['pi_error'] - metrics_det_init['pi_error']
    stoch_pi_change = metrics_stoch_final['pi_error'] - metrics_stoch_init['pi_error']
    print(f"{'Deterministic':<20} {metrics_det_init['pi_error']:<20.6f} {metrics_det_final['pi_error']:<20.6f} {det_pi_change:+.6f}")
    print(f"{'Stochastic':<20} {metrics_stoch_init['pi_error']:<20.6f} {metrics_stoch_final['pi_error']:<20.6f} {stoch_pi_change:+.6f}")
    
    if det_pi_change > stoch_pi_change:
        print(f"\n★ Deterministic optimization deviates MORE from target π ({det_pi_change:+.6f} vs {stoch_pi_change:+.6f})")
    else:
        print(f"\n★ Stochastic optimization deviates MORE from target π ({stoch_pi_change:+.6f} vs {det_pi_change:+.6f})")
    
    # Policy analysis
    print("\n" + "="*80)
    print("POLICY vs EDGE RELIABILITY ANALYSIS")
    print("="*80)
    
    analyze_policy_vs_cv(P_det, CV_matrix, mA, "Deterministic Case")
    analyze_policy_vs_cv(P_stoch, CV_matrix, mA, "Stochastic Case")
    analyze_policy_vs_cv(P_init, CV_matrix, mA, "Initial Uniform Policy")
    
    # =========== PLOTS ===========
    
    # Individual optimization plots
    if len(iter_det) > 0:
        plot_optimization_results(iter_det, eff_det, kw_det, var_det,
                                   metrics_det_init, metrics_det_final,
                                   n, case_name='Deterministic',
                                   filename='grid_optimization_deterministic.png')
    
    if len(iter_stoch) > 0:
        plot_optimization_results(iter_stoch, eff_stoch, kw_stoch, var_stoch,
                                   metrics_stoch_init, metrics_stoch_final,
                                   n, case_name='Stochastic',
                                   filename='grid_optimization_stochastic.png')
    
    # Comparison plot with pi_error
    if len(iter_det) > 0 and len(iter_stoch) > 0:
        plot_optimization_comparison(iter_det, eff_det, kw_det, var_det,
                                      iter_stoch, eff_stoch, kw_stoch, var_stoch,
                                      metrics_det_init, metrics_det_final,
                                      metrics_stoch_init, metrics_stoch_final,
                                      n, filename='optimization_comparison.png',
                                      pi_err_det=pi_err_det, pi_err_stoch=pi_err_stoch)
    
    # Policy comparison
    plot_policy_comparison(n, mA, CV_matrix, obstacle_mask, grid_positions,
                            P_det, P_stoch,
                            metrics_det_final['pi_W'], metrics_stoch_final['pi_W'],
                            filename='policy_comparison.png')
    
    # P vs CV scatter
    plot_policy_vs_cv(P_det, P_stoch, CV_matrix, mA, filename='policy_vs_cv.png')
    
    return {
        'det_init': metrics_det_init,
        'det_final': metrics_det_final,
        'stoch_init': metrics_stoch_init,
        'stoch_final': metrics_stoch_final,
        'P_det': P_det,
        'P_stoch': P_stoch,
        'CV_matrix': CV_matrix,
        'pi_err_det': pi_err_det,
        'pi_err_stoch': pi_err_stoch
    }


def test_cv_levels():
    """Test impact of different CV levels on network metrics."""
    
    print("\n" + "="*80)
    print("TEST: IMPACT OF CV LEVELS")
    print("="*80)
    
    n = 5
    obstacles = [(2, 2)]
    
    cv_configs = [
        {'cv_low': 0.0, 'cv_high': 0.0, 'high_frac': 0.0, 'label': 'Deterministic'},
        {'cv_low': 0.2, 'cv_high': 0.3, 'high_frac': 0.0, 'label': 'Low CV (0.2-0.3)'},
        {'cv_low': 0.5, 'cv_high': 0.7, 'high_frac': 0.0, 'label': 'Medium CV (0.5-0.7)'},
        {'cv_low': 0.3, 'cv_high': 1.2, 'high_frac': 0.4, 'label': 'Mixed CV'},
        {'cv_low': 0.5, 'cv_high': 2.0, 'high_frac': 0.5, 'label': 'High CV'},
    ]
    
    results = []
    
    print(f"\n{'Configuration':<25} {'Mean CV':<12} {'Eff_Idx':<15} {'Variance':<15}")
    print("-" * 67)
    
    for config in cv_configs:
        mA, W, W2, CV_matrix, obstacle_mask, _ = generate_grid_network_stochastic(
            n, obstacles=obstacles,
            cv_low=config['cv_low'], cv_high=config['cv_high'],
            high_cv_fraction=config['high_frac']
        )
        
        if config['cv_low'] == 0.0 and config['cv_high'] == 0.0:
            W2 = W ** 2
            mean_cv = 0.0
        else:
            mean_cv = np.mean(CV_matrix[mA > 0])
        
        pi_hat = create_grid_target_distribution(n, obstacle_mask)
        
        problem = EfficiencyProblemInstanceStochastic(
            mA=mA, W=W, W2=W2, eta=1e-4, pi_hat=pi_hat,
            objective_type='maximize_efficiency', 
            use_hard_constraint=True
        )
        
        # Get feasible initial point
        x_init = problem.get_feasible_initial_point()
        
        P = x_to_matrix(x_init, problem.N, problem.edge_matrix, False)
        metrics = problem.evaluate_metrics(P)
        
        results.append({
            'config': config['label'],
            'mean_cv': mean_cv,
            'Eff_Idx': metrics['Eff_Idx'],
            'Net_Var': metrics['Net_Var']
        })
        
        print(f"{config['label']:<25} {mean_cv:<12.3f} {metrics['Eff_Idx']:<15.4f} {metrics['Net_Var']:<15.4f}")
    
    # Variance increase
    base_var = results[0]['Net_Var']
    print(f"\n{'Configuration':<25} {'Variance Increase':<20}")
    print("-" * 45)
    for r in results:
        var_inc = r['Net_Var'] - base_var
        print(f"{r['config']:<25} {var_inc:+.4f} ({100*var_inc/base_var:+.2f}%)")
    
    return results


def test_minimize_variance():
    """
    Test scenario: Minimize network variance with deterministic weights.
    
    This represents a scenario where we want CONSISTENT patrol times,
    e.g., ensuring regular coverage intervals for security applications.
    
    Objective: Minimize Var[first-passage time]
    Constraint: Keep π close to target distribution
    Optimize: P (transition probabilities)
    Weights: Deterministic (W2 = W²)
    """
    
    print("\n" + "="*80)
    print("TEST: MINIMIZE VARIANCE (Deterministic Weights)")
    print("="*80)
    print("""
Scenario: Design a patrol policy with CONSISTENT patrol times.
- Objective: Minimize network variance
- Constraint: Penalize deviation from target π
- Weights: Deterministic (no edge time variability)

Application: Security patrol where regular, predictable coverage is desired.
""")
    
    # Grid parameters
    n = 10
    obstacles = [(2, 2), (5, 5), (7, 3)]
    
    print(f"Grid size: {n}×{n} = {n*n} nodes")
    print(f"Obstacle positions: {obstacles}")
    
    # Generate network with deterministic weights
    mA, W, _, CV_matrix, obstacle_mask, grid_positions = \
        generate_grid_network_stochastic(n, obstacles=obstacles, cv_low=0.0, cv_high=0.0)
    
    # Force deterministic: W2 = W²
    W2 = W ** 2
    
    # Create uniform CV matrix for visualization (all zeros for deterministic)
    CV_matrix = np.zeros_like(W)
    
    # Target distribution - uniform over non-obstacle nodes
    pi_hat = create_grid_target_distribution(n, obstacle_mask)
    
    # =========== BASELINE: Uniform Policy ===========
    print("\n" + "-"*60)
    print("BASELINE: Uniform Random Walk")
    print("-"*60)
    
    # Create problem for evaluation with HARD CONSTRAINT
    problem = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W, W2=W2,
        eta=1e-4, pi_hat=pi_hat,
        objective_type='minimize_variance',
        use_hard_constraint=True  # Ensures uniform coverage
    )
    
    # Get feasible initial point
    x_init = problem.get_feasible_initial_point()
    
    P_init = x_to_matrix(x_init, problem.N, problem.edge_matrix, False)
    metrics_init = problem.evaluate_metrics(P_init)
    
    print(f"\nInitial metrics (Uniform Policy):")
    print(f"  Variance:         {metrics_init['Net_Var']:.4f}")
    print(f"  Mean (K_W):       {metrics_init['K_W']:.4f}")
    print(f"  Efficiency Index: {metrics_init['Eff_Idx']:.4f}")
    print(f"  π error:          {metrics_init['pi_error']:.6f}")
    
    # =========== OPTIMIZATION: Minimize Variance ===========
    print("\n" + "-"*60)
    print("OPTIMIZATION: Minimize Variance")
    print("-"*60)
    
    print("\nRunning SPSA optimization to minimize variance...")
    iter_hist, eff_hist, kw_hist, var_hist, pi_err_hist, best_x, _ = solve_spsa_efficiency(
        problem, x_init, 
        verbose=True,
        max_iter=10000
    )
    
    P_opt = x_to_matrix(best_x, problem.N, problem.edge_matrix, False)
    metrics_opt = problem.evaluate_metrics(P_opt)
    
    print(f"\nOptimized metrics (Minimum Variance Policy):")
    print(f"  Variance:         {metrics_opt['Net_Var']:.4f}")
    print(f"  Mean (K_W):       {metrics_opt['K_W']:.4f}")
    print(f"  Efficiency Index: {metrics_opt['Eff_Idx']:.4f}")
    print(f"  π error:          {metrics_opt['pi_error']:.6f}")
    
    # =========== COMPARISON ===========
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    var_reduction = metrics_init['Net_Var'] - metrics_opt['Net_Var']
    var_reduction_pct = 100 * var_reduction / metrics_init['Net_Var']
    
    print(f"\n{'Metric':<25} {'Uniform':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Variance':<25} {metrics_init['Net_Var']:<15.4f} {metrics_opt['Net_Var']:<15.4f} {-var_reduction:+.4f}")
    print(f"{'Mean (K_W)':<25} {metrics_init['K_W']:<15.4f} {metrics_opt['K_W']:<15.4f} {metrics_opt['K_W']-metrics_init['K_W']:+.4f}")
    print(f"{'Efficiency Index':<25} {metrics_init['Eff_Idx']:<15.4f} {metrics_opt['Eff_Idx']:<15.4f} {metrics_opt['Eff_Idx']-metrics_init['Eff_Idx']:+.4f}")
    print(f"{'π error':<25} {metrics_init['pi_error']:<15.6f} {metrics_opt['pi_error']:<15.6f} {metrics_opt['pi_error']-metrics_init['pi_error']:+.6f}")
    
    print(f"\n★ Variance Reduction: {var_reduction:.4f} ({var_reduction_pct:.2f}%)")
    
    # =========== PLOTS ===========
    
    # Plot 1: Optimization convergence (2x3 grid with pi_error)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Variance convergence (main objective)
    axes[0, 0].plot(iter_hist, var_hist, 'purple', linewidth=2, marker='o', markersize=4)
    axes[0, 0].axhline(y=metrics_init['Net_Var'], color='g', linestyle='--', 
                       label=f"Initial = {metrics_init['Net_Var']:.2f}")
    axes[0, 0].axhline(y=metrics_opt['Net_Var'], color='r', linestyle='--',
                       label=f"Final = {metrics_opt['Net_Var']:.2f}")
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(r'Network Variance $V_{\mathcal{W}}$')
    axes[0, 0].set_title(r'Variance Convergence ($\downarrow$ lower is better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean (K_W)
    axes[0, 1].plot(iter_hist, kw_hist, 'orange', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=metrics_init['K_W'], color='g', linestyle='--',
                       label=f"Initial = {metrics_init['K_W']:.2f}")
    axes[0, 1].axhline(y=metrics_opt['K_W'], color='r', linestyle='--',
                       label=f"Final = {metrics_opt['K_W']:.2f}")
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[0, 1].set_title('Mean Patrol Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pi error convergence
    if len(pi_err_hist) > 0:
        axes[0, 2].plot(iter_hist, pi_err_hist, 'green', linewidth=2, marker='o', markersize=4)
        axes[0, 2].axhline(y=metrics_init['pi_error'], color='g', linestyle='--',
                           label=f"Initial = {metrics_init['pi_error']:.4f}")
        axes[0, 2].axhline(y=metrics_opt['pi_error'], color='r', linestyle='--',
                           label=f"Final = {metrics_opt['pi_error']:.4f}")
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel(r'$\|\pi_W - \hat{\pi}\|_2$')
        axes[0, 2].set_title(r'Distribution Error ($\downarrow$ lower is better)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Efficiency Index
    axes[1, 0].plot(iter_hist, eff_hist, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(y=metrics_init['Eff_Idx'], color='g', linestyle='--',
                       label=f"Initial = {metrics_init['Eff_Idx']:.2f}")
    axes[1, 0].axhline(y=metrics_opt['Eff_Idx'], color='r', linestyle='--',
                       label=f"Final = {metrics_opt['Eff_Idx']:.2f}")
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel(r'Efficiency Index $\lambda$')
    axes[1, 0].set_title('Efficiency Index (Var/Mean)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean-Variance tradeoff
    scatter = axes[1, 1].scatter(kw_hist, var_hist, c=iter_hist, cmap='viridis', 
                                  s=50, alpha=0.7)
    axes[1, 1].scatter(metrics_init['K_W'], metrics_init['Net_Var'], 
                      color='green', s=300, marker='*', label='Initial', zorder=5, edgecolor='black')
    axes[1, 1].scatter(metrics_opt['K_W'], metrics_opt['Net_Var'],
                      color='red', s=300, marker='*', label='Optimized', zorder=5, edgecolor='black')
    axes[1, 1].set_xlabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[1, 1].set_ylabel(r'Variance $V_{\mathcal{W}}$')
    axes[1, 1].set_title('Mean-Variance Tradeoff')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Iteration')
    
    # Variance vs Pi error tradeoff
    if len(pi_err_hist) > 0:
        scatter2 = axes[1, 2].scatter(pi_err_hist, var_hist, c=iter_hist, cmap='viridis', 
                                       s=50, alpha=0.7)
        axes[1, 2].scatter(metrics_init['pi_error'], metrics_init['Net_Var'], 
                          color='green', s=300, marker='*', label='Initial', zorder=5, edgecolor='black')
        axes[1, 2].scatter(metrics_opt['pi_error'], metrics_opt['Net_Var'],
                          color='red', s=300, marker='*', label='Optimized', zorder=5, edgecolor='black')
        axes[1, 2].set_xlabel(r'$\|\pi_W - \hat{\pi}\|_2$ (Distribution Error)')
        axes[1, 2].set_ylabel(r'Variance $V_{\mathcal{W}}$')
        axes[1, 2].set_title('Variance vs Distribution Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[1, 2])
        cbar2.set_label('Iteration')
    
    plt.suptitle(f'{n}×{n} Grid - Minimize Variance Optimization\n(Deterministic Weights)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('minimize_variance_convergence.png', dpi=150)
    print("\n✓ Convergence plot saved to 'minimize_variance_convergence.png'")
    plt.show()
    
    # Plot 2: Policy comparison (Initial vs Optimized)
    from matplotlib.patches import FancyArrowPatch
    
    NODE_RADIUS = 0.2
    SHRINK_FACTOR = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax_idx, (P, pi, metrics, case_title) in enumerate([
        (P_init, metrics_init['pi_W'], metrics_init, f'Initial Uniform Policy\nVar={metrics_init["Net_Var"]:.2f}'),
        (P_opt, metrics_opt['pi_W'], metrics_opt, f'Minimum Variance Policy\nVar={metrics_opt["Net_Var"]:.2f}')
    ]):
        ax = axes[ax_idx]
        edge_list = np.argwhere(mA > 0)
        max_P = np.max(P[mA > 0]) if np.any(mA > 0) else 1.0
        min_P = np.min(P[mA > 0]) if np.any(mA > 0) else 0.0
        
        for i, j in edge_list:
            ri, ci = grid_positions[i]
            rj, cj = grid_positions[j]
            
            # All edges are "reliable" (deterministic) - use blue color
            color = 'steelblue'
            alpha = 0.7
            
            # Line width by P
            if P[i, j] > 0.001:
                if max_P > min_P:
                    normalized_P = (P[i, j] - min_P) / (max_P - min_P)
                else:
                    normalized_P = 0.5
                linewidth = 0.5 + 5.5 * normalized_P
            else:
                linewidth = 0.3
                normalized_P = 0.0
            
            # --- Use explicit calculation for shrunk arrow start/end points ---
            # Calculate curve direction (perpendicular offset)
            dx = cj - ci
            dy = rj - ri
            
            # Shrink start and end points based on SHRINK_FACTOR
            shrink = SHRINK_FACTOR
            start = (ci + shrink * dx, ri + shrink * dy)
            end = (cj - shrink * dx, rj - shrink * dy)
            
            # Use curved FancyArrowPatch for better visibility
            arrow = FancyArrowPatch(
                start, end,  # Uses calculated shrunk points
                connectionstyle='arc3,rad=0.2',
                arrowstyle='-|>',
                # mutation_scale adjusted for dynamic size, using the original logic
                mutation_scale=10 + 5 * normalized_P if P[i, j] > 0.001 else 8, 
                color=color,
                lw=linewidth,
                alpha=alpha,
                zorder=1
                # shrinkA and shrinkB removed as start/end points are pre-shrunk
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for i in range(n * n):
            r, c = grid_positions[i]
            if obstacle_mask[i]:
                # Obstacle size is calculated relative to NODE_RADIUS (0.2 + 0.1) * 2 = 0.6
                OBSTACLE_OFFSET = NODE_RADIUS + 0.1
                OBSTACLE_SIZE = 2 * OBSTACLE_OFFSET
                ax.add_patch(plt.Rectangle((c - OBSTACLE_SIZE/2, r - OBSTACLE_SIZE/2), 
                                            OBSTACLE_SIZE, OBSTACLE_SIZE, 
                                            color='black', zorder=3))
                ax.text(c, r, 'X', ha='center', va='center', fontsize=10, 
                    color='white', fontweight='bold', zorder=4)
            else:
                node_color = 'lightblue'
                if pi is not None:
                    intensity = min(pi[i] / (np.max(pi) + 1e-12), 1.0)
                    node_color = plt.cm.Blues(0.3 + 0.7 * intensity)
                
                # Use NODE_RADIUS variable (0.2)
                circle = plt.Circle((c, r), NODE_RADIUS, color=node_color, ec='black', 
                                    linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=4)
        
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(n - 0.3, -0.7)
        ax.set_aspect('equal')
        ax.set_title(case_title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Shared legend
    thick_line = plt.Line2D([0], [0], color='steelblue', linewidth=5, label='High P(i,j)')
    thin_line = plt.Line2D([0], [0], color='steelblue', linewidth=1, label='Low P(i,j)')
    fig.legend(handles=[thick_line, thin_line], loc='lower center', ncol=2, fontsize=11)

    plt.suptitle('Minimize Variance: Policy Comparison\n(Edge thickness = Transition Probability P)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('minimize_variance_policy.png', dpi=150, bbox_inches='tight')
    print("✓ Policy comparison saved to 'minimize_variance_policy.png'")
    plt.show()
    
    # Print some statistics about the optimal P
    print("\n" + "-"*60)
    print("OPTIMAL POLICY ANALYSIS")
    print("-"*60)
    
    P_values = P_opt[mA > 0]
    print(f"\nTransition probability statistics:")
    print(f"  Min P:  {P_values.min():.4f}")
    print(f"  Max P:  {P_values.max():.4f}")
    print(f"  Mean P: {P_values.mean():.4f}")
    print(f"  Std P:  {P_values.std():.4f}")
    
    # Check how concentrated the policy is
    entropy = -np.sum(P_opt[mA > 0] * np.log(P_opt[mA > 0] + 1e-12))
    max_entropy = -np.sum(P_init[mA > 0] * np.log(P_init[mA > 0] + 1e-12))
    print(f"\nPolicy entropy: {entropy:.4f} (uniform: {max_entropy:.4f})")
    print(f"Entropy ratio: {entropy/max_entropy:.4f} (1.0 = uniform, lower = more concentrated)")
    
    return {
        'initial': metrics_init,
        'optimized': metrics_opt,
        'P_init': P_init,
        'P_opt': P_opt,
        'var_reduction': var_reduction,
        'var_reduction_pct': var_reduction_pct
    }


if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("GRID NETWORK SURVEILLANCE WITH STOCHASTIC WEIGHTS")
    print("="*80)
    print("""
Key concepts:
- Optimization is with respect to P (transition probabilities)
- W and W2 are FIXED parameters (stochastic edge characteristics)
- E[W²] = E[W]² × (1 + CV²) captures edge weight variance
- Higher CV = more unpredictable travel times
""")
    
    # Run main comparison (maximize efficiency)
    results = test_deterministic_vs_stochastic()
    
    # Run CV sensitivity analysis
    cv_results = test_cv_levels()
    
    # Run minimize variance test (NEW)
    min_var_results = test_minimize_variance()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)