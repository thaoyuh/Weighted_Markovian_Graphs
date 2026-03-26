"""
Main script for Grid Network Surveillance Optimization.

Organized into experiments for the paper:

  CASE 1: Deterministic weights (CV = 0)
    - 1A: Minimize variance   (motivating negative example — "cost of predictability")
    - 1B: Maximize surprise index lambda  (THE main result)

  CASE 2: Stochastic weights (mixed high-CV & low-CV edges)
    - Maximize surprise index lambda  (disentangles routing vs weight-induced variance)

Publication figures (all individual):
  - Fig 1a: Initial uniform policy
  - Fig 1b: Minimum-variance policy (near-Hamiltonian cycle)
  - Fig 2:  Max-Surprise policy (deterministic)
  - Fig 3:  Max-Surprise policy (stochastic, CV-colored edges)
  - Table:  Summary of all policies

Convergence plots are saved separately (e-companion / supplementary).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Import from project modules
from utils import x_to_matrix, build_neighborhoods
from network_stochastic import MarkovChainStochastic
from problem_instance import EfficiencyProblemInstanceStochastic
from grid_generation import generate_grid_network_stochastic, create_grid_target_distribution
from optimization import solve_spsa_efficiency


# ============================================================================
# PUBLICATION-QUALITY POLICY PLOT
# ============================================================================

def plot_policy_publication(n, mA, obstacle_mask, grid_positions, P, pi,
                            title, filename, cv_matrix=None,
                            figsize=(6, 6), show_node_labels=True):
    """
    Publication-quality single policy network plot.
    """
    NODE_RADIUS = 0.22
    SHRINK = 0.22

    fig, ax = plt.subplots(figsize=figsize)

    edge_list = np.argwhere(mA > 0)
    P_vals = P[mA > 0]
    max_P = np.max(P_vals) if len(P_vals) > 0 else 1.0
    min_P = np.min(P_vals) if len(P_vals) > 0 else 0.0
    range_P = max_P - min_P if max_P > min_P else 1.0

    for i, j in edge_list:
        ri, ci = grid_positions[i]
        rj, cj = grid_positions[j]

        if cv_matrix is not None and cv_matrix[i, j] >= 1.0:
            color = '#d62728'
            alpha = min(0.55 + 0.3 * (cv_matrix[i, j] - 1), 0.9)
        elif cv_matrix is not None:
            color = '#2ca02c'
            alpha = 0.5 + 0.4 * (1 - cv_matrix[i, j])
        else:
            color = '#555555'
            alpha = 0.65

        p_ij = P[i, j]
        norm_P = (p_ij - min_P) / range_P if p_ij > 0.001 else 0.0
        lw = 0.3 + 4.5 * norm_P if p_ij > 0.001 else 0.15

        dx, dy = cj - ci, rj - ri
        start = (ci + SHRINK * dx, ri + SHRINK * dy)
        end = (cj - SHRINK * dx, rj - SHRINK * dy)

        arrow = FancyArrowPatch(
            start, end,
            connectionstyle='arc3,rad=0.18',
            arrowstyle='-|>',
            mutation_scale=8 + 5 * norm_P if p_ij > 0.001 else 7,
            color=color, lw=lw, alpha=alpha, zorder=1
        )
        ax.add_patch(arrow)

    pi_max = np.max(pi[~obstacle_mask]) if pi is not None else 1.0
    for i in range(n * n):
        r, c = grid_positions[i]
        if obstacle_mask[i]:
            sz = 2 * (NODE_RADIUS + 0.08)
            ax.add_patch(plt.Rectangle((c - sz/2, r - sz/2), sz, sz,
                                        color='black', zorder=3))
            ax.text(c, r, u'\u00d7', ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold', zorder=4)
        else:
            if pi is not None:
                intensity = min(pi[i] / (pi_max + 1e-12), 1.0)
                nc = plt.cm.Blues(0.25 + 0.65 * intensity)
            else:
                nc = '#b0d4f1'
            circle = plt.Circle((c, r), NODE_RADIUS, color=nc, ec='black',
                                linewidth=1.2, zorder=3)
            ax.add_patch(circle)
            if show_node_labels:
                ax.text(c, r, str(i), ha='center', va='center',
                        fontsize=10, zorder=4, color='white')

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  -> {filename}")
    plt.close()


# ============================================================================
# SUMMARY TABLE FIGURE
# ============================================================================

def plot_summary_table(rows, filename='fig_table_deterministic.png'):
    fig, ax = plt.subplots(figsize=(8, 2.0 + 0.35 * len(rows)))
    ax.axis('off')

    col_labels = ['Policy', r'$K_{\mathcal{W}}$', r'$V_{\mathcal{W}}$',
                  r'$\lambda = V/K$', r'$\|\pi - \hat{\mu}\|$']
    cell_text = []
    for r in rows:
        cell_text.append([
            r['Policy'],
            f"{r['K_W']:.2f}",
            f"{r['V_W']:.2f}",
            f"{r['lambda']:.2f}",
            '0' if r['pi_err'] < 1e-8 else f"{r['pi_err']:.1e}"
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E2F3')
        cell.set_edgecolor('#999999')

    plt.title('Surveillance Policies on $5 \\times 5$ Grid (Deterministic Weights)',
              fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  -> {filename}")
    plt.close()


def plot_summary_table_full(rows_det, rows_stoch, filename='fig_table_full.png'):
    fig, ax = plt.subplots(figsize=(9, 2.5 + 0.35 * (len(rows_det) + len(rows_stoch))))
    ax.axis('off')

    col_labels = ['Setting', 'Policy', r'$K_{\mathcal{W}}$', r'$V_{\mathcal{W}}$',
                  r'$\lambda$', r'$\|\pi - \hat{\mu}\|$']
    cell_text = []
    for r in rows_det:
        cell_text.append([
            r.get('Setting', 'Determ.'), r['Policy'],
            f"{r['K_W']:.2f}", f"{r['V_W']:.2f}", f"{r['lambda']:.2f}",
            '0' if r['pi_err'] < 1e-8 else f"{r['pi_err']:.1e}"
        ])
    for r in rows_stoch:
        cell_text.append([
            r.get('Setting', 'Stoch.'), r['Policy'],
            f"{r['K_W']:.2f}", f"{r['V_W']:.2f}", f"{r['lambda']:.2f}",
            '0' if r['pi_err'] < 1e-8 else f"{r['pi_err']:.1e}"
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    n_det = len(rows_det)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
            continue
        if row <= n_det:
            cell.set_facecolor('#D9E2F3')
        else:
            cell.set_facecolor('#FDE9D9')
        cell.set_edgecolor('#999999')

    plt.title('Summary: Surveillance Policies on $5 \\times 5$ Grid',
              fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  -> {filename}")
    plt.close()


# ============================================================================
# CONVERGENCE PLOTS (supplementary)
# ============================================================================

def save_convergence_plots(iter_h, eff_h, kw_h, var_h, pierr_h,
                            m_init, m_opt, prefix, label):
    for vals, init_v, opt_v, ylabel, tit, col, suf in [
        (var_h, m_init['Net_Var'], m_opt['Net_Var'],
         r'$V_{\mathcal{W}}$', 'Variance', 'purple', 'var'),
        (kw_h, m_init['K_W'], m_opt['K_W'],
         r'$K_{\mathcal{W}}$', 'Mean', 'orange', 'mean'),
        (eff_h, m_init['Eff_Idx'], m_opt['Eff_Idx'],
         r'$\lambda$', 'Surprise Index', 'blue', 'eff'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iter_h, vals, color=col, lw=1.5)
        ax.axhline(init_v, color='green', ls='--', lw=1, label=f'Init = {init_v:.2f}')
        ax.axhline(opt_v, color='red', ls='--', lw=1, label=f'Opt = {opt_v:.2f}')
        ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
        ax.set_title(f'{label} — {tit} Convergence')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}_{suf}_conv.png', dpi=150, bbox_inches='tight')
        plt.close()

    if len(pierr_h) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iter_h, pierr_h, color='green', lw=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\|\pi - \hat{\mu}\|_2$')
        ax.set_title(f'{label} — Distribution Error')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}_pierr_conv.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  -> Supplementary: {prefix}_*.png")


# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

def run_optimization(mA, W, W2, pi_hat, objective_type, seed_offset,
                     label, max_iter=15000, base_seed=42, verbose=True):
    problem = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W, W2=W2,
        eta=1e-6, pi_hat=pi_hat,
        objective_type=objective_type,
        use_hard_constraint=True
    )
    x_init = problem.get_feasible_initial_point()
    P_init = x_to_matrix(x_init, problem.N, problem.edge_matrix, False)
    metrics_init = problem.evaluate_metrics(P_init)

    np.random.seed(base_seed + seed_offset)
    print(f"\n  Running SPSA: {label} (objective={objective_type})...")
    iter_h, eff_h, kw_h, var_h, pierr_h, best_x, best_obj = solve_spsa_efficiency(
        problem, x_init, verbose=verbose, max_iter=max_iter
    )

    P_opt = x_to_matrix(best_x, problem.N, problem.edge_matrix, False)
    metrics_opt = problem.evaluate_metrics(P_opt)

    return {
        'problem': problem, 'P_init': P_init, 'P_opt': P_opt,
        'metrics_init': metrics_init, 'metrics_opt': metrics_opt,
        'iter_h': iter_h, 'eff_h': eff_h, 'kw_h': kw_h,
        'var_h': var_h, 'pierr_h': pierr_h, 'best_x': best_x,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    n = 5
    obstacles = [(2,2)]

    print("=" * 80)
    print("SURVEILLANCE NETWORK OPTIMIZATION — PUBLICATION EXPERIMENTS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    print(f"\nGrid: {n}x{n}, Obstacle: {obstacles}")

    mA, W_mean, _, CV_det, obstacle_mask, grid_positions = \
        generate_grid_network_stochastic(n, obstacles=obstacles,
                                          cv_low=0.0, cv_high=0.0,
                                          high_cv_fraction=0.0, seed=SEED)
    W2_det = W_mean ** 2
    pi_hat = create_grid_target_distribution(n, obstacle_mask)

    # Baseline (uniform)
    problem_base = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W_mean, W2=W2_det, eta=1e-4, pi_hat=pi_hat,
        objective_type='minimize_variance', use_hard_constraint=True)
    x_base = problem_base.get_feasible_initial_point()
    P_uniform = x_to_matrix(x_base, problem_base.N, problem_base.edge_matrix, False)
    m_uniform = problem_base.evaluate_metrics(P_uniform)

    print(f"\nBaseline (Uniform):")
    print(f"  K_W={m_uniform['K_W']:.2f}, V_W={m_uniform['Net_Var']:.2f}, "
          f"lambda={m_uniform['Eff_Idx']:.2f}")

    # ------------------------------------------------------------------
    # CASE 1A: Min Variance
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CASE 1A: Minimize Variance (Deterministic)")
    print("=" * 80)
    res_1a = run_optimization(mA, W_mean, W2_det, pi_hat,
                               'minimize_variance', seed_offset=1,
                               label='1A: Min Var', max_iter=5000)
    m_1a = res_1a['metrics_opt']
    print(f"\n  V_W: {m_uniform['Net_Var']:.2f} -> {m_1a['Net_Var']:.2f}")

    # ------------------------------------------------------------------
    # CASE 1B: Max Surprise (MAIN RESULT)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CASE 1B: MAXIMIZE SURPRISE INDEX (Deterministic) — MAIN RESULT")
    print("=" * 80)
    res_1b = run_optimization(mA, W_mean, W2_det, pi_hat,
                               'maximize_efficiency', seed_offset=3,
                               label='1B: Max Surprise')
    m_1b = res_1b['metrics_opt']
    print(f"\n  lambda: {m_uniform['Eff_Idx']:.2f} -> {m_1b['Eff_Idx']:.2f}")
    print(f"  K_W={m_1b['K_W']:.2f}, V_W={m_1b['Net_Var']:.2f}")

    # ------------------------------------------------------------------
    # CASE 2: Stochastic Max Surprise
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CASE 2: MAXIMIZE SURPRISE INDEX (Stochastic Weights)")
    print("=" * 80)

    cv_low, cv_high, high_cv_frac = 0.3, 1.5, 0.4
    mA_s, W_s, W2_s, CV_s, obs_s, gpos_s = \
        generate_grid_network_stochastic(n, obstacles, cv_low=cv_low,
                                          cv_high=cv_high,
                                          high_cv_fraction=high_cv_frac,
                                          seed=SEED)
    pi_hat_s = create_grid_target_distribution(n, obs_s)

    prob_s_base = EfficiencyProblemInstanceStochastic(
        mA=mA_s, W=W_s, W2=W2_s, eta=1e-4, pi_hat=pi_hat_s,
        objective_type='maximize_efficiency', use_hard_constraint=True)
    x_s_base = prob_s_base.get_feasible_initial_point()
    P_s_uniform = x_to_matrix(x_s_base, prob_s_base.N, prob_s_base.edge_matrix, False)
    m_s_uniform = prob_s_base.evaluate_metrics(P_s_uniform)

    mask_edges = mA_s > 0
    cvs = CV_s[mask_edges]
    print(f"\n  CV stats: min={cvs.min():.2f}, max={cvs.max():.2f}, mean={cvs.mean():.2f}")

    res_2 = run_optimization(mA_s, W_s, W2_s, pi_hat_s,
                              'maximize_efficiency', seed_offset=10,
                              label='2: Max Surprise (Stoch)')
    m_2 = res_2['metrics_opt']
    print(f"\n  lambda: {m_s_uniform['Eff_Idx']:.2f} -> {m_2['Eff_Idx']:.2f}")

    P_opt_s = res_2['P_opt']
    P_values = P_opt_s[mask_edges]
    CV_values = CV_s[mask_edges]
    low_cv = CV_values < 1; high_cv = CV_values >= 1
    avg_P_low = np.mean(P_values[low_cv]) if np.any(low_cv) else 0
    avg_P_high = np.mean(P_values[high_cv]) if np.any(high_cv) else 0
    corr_PCV = np.corrcoef(P_values, CV_values)[0, 1]
    print(f"  P vs CV: avg_P(low)={avg_P_low:.4f}, avg_P(high)={avg_P_high:.4f}, "
          f"corr={corr_PCV:.4f}")

    # ==================================================================
    # PUBLICATION FIGURES
    # ==================================================================
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 80)

    # --- Fig 1a: Initial uniform policy ---
    plot_policy_publication(
        n, mA, obstacle_mask, grid_positions,
        P_uniform, m_uniform['pi_W'],
        title='Initial Uniform Policy',
        filename='fig1a_uniform_policy.png')

    # --- Fig 1b: Minimum-variance policy ---
    plot_policy_publication(
        n, mA, obstacle_mask, grid_positions,
        res_1a['P_opt'], m_1a['pi_W'],
        title='Minimum-Variance Policy',
        filename='fig1b_min_variance_policy.png')

    # --- Fig 2: Max-Surprise policy (deterministic) ---
    plot_policy_publication(
        n, mA, obstacle_mask, grid_positions,
        res_1b['P_opt'], m_1b['pi_W'],
        title='Maximum-Surprise Policy (Deterministic Weights)',
        filename='fig2_max_surprise_deterministic.png')

    # Fig 3
    plot_policy_publication(
        n, mA_s, obs_s, gpos_s,
        P_opt_s, m_2['pi_W'],
        title='Maximum-Surprise Policy (Stochastic Weights)',
        filename='fig3_max_surprise_stochastic.png',
        cv_matrix=CV_s)

    # Tables
    rows_det = [
        {'Policy': 'Uniform',           'K_W': m_uniform['K_W'],
         'V_W': m_uniform['Net_Var'],    'lambda': m_uniform['Eff_Idx'],
         'pi_err': m_uniform['pi_error']},
        {'Policy': 'Min-Variance',      'K_W': m_1a['K_W'],
         'V_W': m_1a['Net_Var'],         'lambda': m_1a['Eff_Idx'],
         'pi_err': m_1a['pi_error']},
        {'Policy': 'Max-Surprise',      'K_W': m_1b['K_W'],
         'V_W': m_1b['Net_Var'],         'lambda': m_1b['Eff_Idx'],
         'pi_err': m_1b['pi_error']},
    ]
    plot_summary_table(rows_det, filename='fig_table_deterministic.png')

    rows_det_full = [dict(r, Setting='Determ.') for r in rows_det]
    rows_stoch = [
        {'Setting': 'Stochastic', 'Policy': 'Uniform',
         'K_W': m_s_uniform['K_W'], 'V_W': m_s_uniform['Net_Var'],
         'lambda': m_s_uniform['Eff_Idx'], 'pi_err': m_s_uniform['pi_error']},
        {'Setting': 'Stochastic', 'Policy': 'Max-Surprise',
         'K_W': m_2['K_W'], 'V_W': m_2['Net_Var'],
         'lambda': m_2['Eff_Idx'], 'pi_err': m_2['pi_error']},
    ]
    plot_summary_table_full(rows_det_full, rows_stoch,
                             filename='fig_table_full.png')

    # --- Optimization report ---
    report_lines = []
    report_lines.append("SURVEILLANCE OPTIMIZATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Grid: {n}x{n},  Obstacle: {obstacles},  eta = 1e-4")
    report_lines.append(f"Target distribution: uniform (1/{n*n - len(obstacles)})")
    report_lines.append(f"SPSA iterations: 10,000 per case")
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("DETERMINISTIC WEIGHTS (CV = 0)")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Policy':<20} {'K_W':>10} {'V_W':>14} {'lambda':>12} {'pi_err':>10}")
    report_lines.append("-" * 60)
    for label, m in [('Uniform', m_uniform), ('Min-Variance', m_1a), ('Max-Surprise', m_1b)]:
        report_lines.append(
            f"{label:<20} {m['K_W']:>10.2f} {m['Net_Var']:>14.2f} "
            f"{m['Eff_Idx']:>12.2f} {m['pi_error']:>10.1e}")
    report_lines.append("")
    var_red = (1 - m_1a['Net_Var'] / m_uniform['Net_Var']) * 100
    lam_gain = m_1b['Eff_Idx'] / m_uniform['Eff_Idx']
    report_lines.append(f"Min-Var: V_W reduced by {var_red:.1f}%  (near-Hamiltonian cycle)")
    report_lines.append(f"Max-Surprise: lambda increased {lam_gain:.0f}x over uniform")
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("STOCHASTIC WEIGHTS")
    report_lines.append(f"  CV_low ~ 0.3, CV_high ~ 1.5, high-CV fraction: 40%")
    report_lines.append(f"  Edge CV stats: min={cvs.min():.2f}, max={cvs.max():.2f}, mean={cvs.mean():.2f}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Policy':<20} {'K_W':>10} {'V_W':>14} {'lambda':>12} {'pi_err':>10}")
    report_lines.append("-" * 60)
    for label, m in [('Uniform', m_s_uniform), ('Max-Surprise', m_2)]:
        report_lines.append(
            f"{label:<20} {m['K_W']:>10.2f} {m['Net_Var']:>14.2f} "
            f"{m['Eff_Idx']:>12.2f} {m['pi_error']:>10.1e}")
    report_lines.append("")
    lam_gain_s = m_2['Eff_Idx'] / m_s_uniform['Eff_Idx']
    report_lines.append(f"Max-Surprise (stoch): lambda increased {lam_gain_s:.0f}x over uniform")
    report_lines.append(f"Corr(P, CV) = {corr_PCV:.4f}")
    report_lines.append(f"  Avg P on low-CV edges:  {avg_P_low:.4f}")
    report_lines.append(f"  Avg P on high-CV edges: {avg_P_high:.4f}")
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("FIGURES")
    report_lines.append("-" * 60)
    report_lines.append("fig1a_uniform_policy.png        — Initial uniform policy")
    report_lines.append("fig1b_min_variance_policy.png   — Min-variance (cost of predictability)")
    report_lines.append("fig2_max_surprise_deterministic.png — Max-surprise, deterministic")
    report_lines.append("fig3_max_surprise_stochastic.png    — Max-surprise, stochastic (CV-colored)")
    report_lines.append("fig_table_deterministic.png     — Summary table (deterministic)")
    report_lines.append("fig_table_full.png              — Summary table (all)")
    report_lines.append("supp_*.png                      — Convergence plots (e-companion)")

    report_text = "\n".join(report_lines)
    with open('optimization_report.txt', 'w') as f:
        f.write(report_text)
    print(f"  -> optimization_report.txt")

    # Supplementary convergence plots
    print("\n  Supplementary convergence plots...")
    for res, m_init, pref, lab in [
        (res_1a, m_uniform, 'supp_1a', '1A: Min-Var'),
        (res_1b, m_uniform, 'supp_1b', '1B: Max-Surprise (Det)'),
        (res_2, m_s_uniform, 'supp_2', '2: Max-Surprise (Stoch)'),
    ]:
        m_opt = res['metrics_opt']
        save_convergence_plots(res['iter_h'], res['eff_h'], res['kw_h'],
                                res['var_h'], res['pierr_h'],
                                m_init, m_opt, pref, lab)

    # ==================================================================
    # CONSOLE SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Policy':<25} {'K_W':>10} {'V_W':>12} {'lambda':>10} {'pi_err':>10}")
    print("-" * 70)
    for label, m in [('Uniform', m_uniform), ('Min-Variance', m_1a),
                     ('Max-Surprise', m_1b)]:
        print(f"{label:<25} {m['K_W']:>10.2f} {m['Net_Var']:>12.2f} "
              f"{m['Eff_Idx']:>10.2f} {m['pi_error']:>10.1e}")

    print(f"\n{'Policy':<25} {'K_W':>10} {'V_W':>12} {'lambda':>10} {'pi_err':>10}")
    print("-" * 70)
    for label, m in [('Stoch. Uniform', m_s_uniform), ('Stoch. Max-Surprise', m_2)]:
        print(f"{label:<25} {m['K_W']:>10.2f} {m['Net_Var']:>12.2f} "
              f"{m['Eff_Idx']:>10.2f} {m['pi_error']:>10.1e}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)