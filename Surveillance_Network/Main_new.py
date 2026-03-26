"""
Surveillance Network Optimization — Publication Experiments.

Structure:
  Step 1:  4x4 uniform pi, min-variance  (motivating: cost of predictability)
  Step 2a: 4x4 uniform pi, max-surprise  (deterministic — main result)
  Step 2b: 4x4 uniform pi, max-surprise  (stochastic — Remark 6, table only)
#   Step 3a: 8x8 non-uniform pi, max-surprise (deterministic — scalability)
  Step 3b: 8x8 non-uniform pi, max-surprise (stochastic — full case)

Figures:
  fig1a — 4x4 uniform policy (baseline)
  fig1b — 4x4 min-variance policy (Hamiltonian cycle)
  fig2  — 4x4 max-surprise policy (deterministic)
#   fig4  — 8x8 max-surprise policy (deterministic, non-uniform pi)
  fig5  — 8x8 max-surprise policy (stochastic, non-uniform pi)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linprog

from utils import x_to_matrix
from problem_instance import (EfficiencyProblemInstanceStochastic,
                               build_row_sum_constraints,
                               build_stationary_constraints)
from grid_generation import generate_grid_network_stochastic, create_grid_target_distribution
from optimization import solve_spsa_efficiency


# ============================================================================
# FEASIBILITY CHECK
# ============================================================================

def check_feasibility(mA, pi_hat):
    """
    Check if there exists a row-stochastic P with P(i,j)>=0 on edges
    satisfying pi_hat^T P = pi_hat^T.
    Returns True if feasible.
    """
    N = mA.shape[0]
    A_row = build_row_sum_constraints(mA)
    b_row = np.ones(N)
    A_pi = build_stationary_constraints(mA, pi_hat)
    b_pi = pi_hat.copy()
    active = np.sum(mA, axis=1) > 0
    A_eq = np.vstack([A_row[active], A_pi[active]])
    b_eq = np.hstack([b_row[active], b_pi[active]])
    d = A_eq.shape[1]
    result = linprog(np.zeros(d), A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, 1)] * d, method='highs')
    return result.success


# ============================================================================
# POLICY PLOT (clean, no legend)
# ============================================================================

def plot_policy(n, mA, obstacle_mask, grid_positions, P, pi,
                title, filename, cv_matrix=None,
                figsize=(6, 6), show_node_labels=True):
    NODE_RADIUS = 0.22
    SHRINK = 0.22
    fig, ax = plt.subplots(figsize=figsize)

    edge_list = np.argwhere(mA > 0)
    P_vals = P[mA > 0]
    max_P, min_P = P_vals.max(), P_vals.min()
    range_P = max_P - min_P if max_P > min_P else 1.0

    for i, j in edge_list:
        ri, ci = grid_positions[i]
        rj, cj = grid_positions[j]
        p_ij = P[i, j]
        norm_P = (p_ij - min_P) / range_P if p_ij > 0.001 else 0.0
        lw = 0.8 + 4.0 * norm_P if p_ij > 0.001 else 0.5

        if cv_matrix is not None and cv_matrix[i, j] >= 1.0:
            color = '#d62728'
            alpha = min(0.55 + 0.3 * (cv_matrix[i, j] - 1), 0.9)
        elif cv_matrix is not None:
            color = '#2ca02c'
            alpha = 0.5 + 0.4 * (1 - cv_matrix[i, j])
        else:
            color, alpha = '#555555', 0.65

        dx, dy = cj - ci, rj - ri
        start = (ci + SHRINK * dx, ri + SHRINK * dy)
        end = (cj - SHRINK * dx, rj - SHRINK * dy)
        arrow = FancyArrowPatch(
            start, end, connectionstyle='arc3,rad=0.18', arrowstyle='-|>',
            mutation_scale=8 + 5 * norm_P if p_ij > 0.001 else 7,
            color=color, lw=lw, alpha=alpha, zorder=1)
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
            intensity = min(pi[i] / (pi_max + 1e-12), 1.0) if pi is not None else 0.5
            nc = plt.cm.Blues(0.25 + 0.70 * intensity)
            circle = plt.Circle((c, r), NODE_RADIUS, color=nc, ec='black',
                                linewidth=1.2, zorder=3)
            ax.add_patch(circle)
            if show_node_labels:
                ax.text(c, r, str(i), ha='center', va='center', fontsize=12, zorder=4, color='white', fontweight='bold')

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  -> {filename}")
    plt.close()


# ============================================================================
# CONVERGENCE PLOTS (supplementary)
# ============================================================================

def save_convergence_plots(iter_h, eff_h, kw_h, var_h, pierr_h,
                            m_init, m_opt, prefix, label):
    # Transform var history to sqrt(V) for plotting
    sqrtvar_h = [np.sqrt(max(v, 0)) for v in var_h]
    for vals, iv, ov, yl, tit, col, suf in [
        (sqrtvar_h, np.sqrt(max(m_init['Net_Var'],0)), np.sqrt(max(m_opt['Net_Var'],0)),
            r'$\sqrt{V_{\mathcal{W}}}$', 'Std Dev FPT', 'purple', 'var'),
        (kw_h,  m_init['K_W'],    m_opt['K_W'],    r'$K_{\mathcal{W}}$', 'Kemeny',    'orange', 'mean'),
        (eff_h, m_init['Eff_Idx'],m_opt['Eff_Idx'],r'$\mathcal{S}(\mathbf{P})$', 'Surprise Index', 'blue', 'eff'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iter_h, vals, color=col, lw=1.5)
        ax.axhline(iv, color='green', ls='--', lw=1, label=f'Init = {iv:.2f}')
        ax.axhline(ov, color='red',   ls='--', lw=1, label=f'Opt = {ov:.2f}')
        ax.set_xlabel('Iteration'); ax.set_ylabel(yl)
        ax.set_title(f'{label} — {tit}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}_{suf}.png', dpi=150, bbox_inches='tight')
        plt.close()
    if len(pierr_h) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iter_h, pierr_h, color='green', lw=1.5)
        ax.set_xlabel('Iteration'); ax.set_ylabel(r'$\|\pi - \hat{\mu}\|_2$')
        ax.set_title(f'{label} — Dist. Error'); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{prefix}_pierr.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"  -> Supp: {prefix}_*.png")


# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

def run_opt(mA, W, W2, pi_hat, obj, label, eta=1e-4, max_iter=5000,
            seed=42, a=0.01, verbose=True):
    prob = EfficiencyProblemInstanceStochastic(
        mA=mA, W=W, W2=W2, eta=eta, pi_hat=pi_hat,
        objective_type=obj, use_hard_constraint=True)
    x0 = prob.get_feasible_initial_point()
    P0 = x_to_matrix(x0, prob.N, prob.edge_matrix, False)
    m0 = prob.evaluate_metrics(P0)

    np.random.seed(seed)
    print(f"\n  SPSA: {label} (obj={obj}, eta={eta}, a={a}, iters={max_iter})")
    ih, eh, kh, vh, ph, bx, bo = solve_spsa_efficiency(
        prob, x0, verbose=verbose, max_iter=max_iter, a=a)
    Popt = x_to_matrix(bx, prob.N, prob.edge_matrix, False)
    mopt = prob.evaluate_metrics(Popt)
    return {
        'prob': prob, 'P0': P0, 'Popt': Popt, 'm0': m0, 'mopt': mopt,
        'ih': ih, 'eh': eh, 'kh': kh, 'vh': vh, 'ph': ph,
    }


# ============================================================================
# NON-UNIFORM TARGET DISTRIBUTION
# ============================================================================

def create_nonuniform_pi(n, obstacle_mask, priority_positions, base_weight=1.0):
    """Non-uniform target distribution. priority_positions: {(r,c): weight}."""
    N = n * n
    weights = np.full(N, base_weight)
    weights[obstacle_mask] = 0
    for (r, c), w in priority_positions.items():
        idx = r * n + c
        if idx < N and not obstacle_mask[idx]:
            weights[idx] = w
    return weights / weights.sum()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    ETA_SMALL = 1e-4    # 4x4
    ETA_BIG   = 1e-8    # 8x8

    print("=" * 80)
    print("SURVEILLANCE OPTIMIZATION — PUBLICATION EXPERIMENTS")
    print("=" * 80)

    # ==================================================================
    # STEP 1: 4x4 UNIFORM — COST OF PREDICTABILITY
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 1: 4x4 Grid, Uniform pi, Minimize Variance")
    print("=" * 80)

    n4 = 4
    mA4, W4, _, CV4, mask4, gpos4 = generate_grid_network_stochastic(
        n4, obstacles=[], cv_low=0.0, cv_high=0.0, high_cv_fraction=0.0, seed=SEED)
    W2_4 = W4 ** 2
    pi4 = create_grid_target_distribution(n4, mask4)
    print(f"Grid: {n4}x{n4}, no obstacles, {int(np.sum(~mask4))} nodes, {int(np.sum(mA4))} edges")
    assert check_feasibility(mA4, pi4), "4x4 uniform is infeasible!"

    # Baseline (uniform policy)
    prob_base4 = EfficiencyProblemInstanceStochastic(
        mA=mA4, W=W4, W2=W2_4, eta=ETA_SMALL, pi_hat=pi4,
        objective_type='minimize_variance', use_hard_constraint=True)
    x_base4 = prob_base4.get_feasible_initial_point()
    P_unif4 = x_to_matrix(x_base4, prob_base4.N, prob_base4.edge_matrix, False)
    m_unif4 = prob_base4.evaluate_metrics(P_unif4)
    print(f"Baseline: K_W={m_unif4['K_W']:.2f}, sqrtV_W={np.sqrt(m_unif4['Net_Var']):.2f}, "
          f"S={m_unif4['Eff_Idx']:.4f}")

    # Min-Variance optimization
    res_1 = run_opt(mA4, W4, W2_4, pi4, 'minimize_variance',
                    '1: Min-Var (4x4)', eta=ETA_SMALL, max_iter=5000, a=0.001, seed=SEED+1)
    m_1 = res_1['mopt']
    print(f"  sqrtV_W: {np.sqrt(m_unif4['Net_Var']):.2f} -> {np.sqrt(m_1['Net_Var']):.2f}")
    dom_edges = np.sum(res_1['Popt'][mA4 > 0] > 0.5)
    print(f"  Dominant edges (P>0.5): {dom_edges}/{int(np.sum(mA4))} "
          f"({'= n nodes: Hamiltonian cycle' if dom_edges == n4*n4 else ''})")

    # ==================================================================
    # STEP 2a: 4x4 MAX SURPRISE (DETERMINISTIC)
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 2a: 4x4 Grid, Uniform pi, Max Surprise (Deterministic)")
    print("=" * 80)

    res_2a = run_opt(mA4, W4, W2_4, pi4, 'maximize_efficiency',
                     '2a: Max-Surprise (4x4 Det)', eta=ETA_SMALL, max_iter=5000, seed=SEED+1)
    m_2a = res_2a['mopt']
    print(f"  S: {m_unif4['Eff_Idx']:.4f} -> {m_2a['Eff_Idx']:.4f}")

    # # ==================================================================
    # # STEP 2b: 4x4 MAX SURPRISE (STOCHASTIC)
    # # ==================================================================
    # print("\n" + "=" * 80)
    # print("STEP 2b: 4x4 Grid, Uniform pi, Max Surprise (Stochastic)")
    # print("=" * 80)

    # mA4s, W4s, W2_4s, CV4s, mask4s, gpos4s = generate_grid_network_stochastic(
    #     n4, obstacles=[], cv_low=0.3, cv_high=1.5, high_cv_fraction=0.4, seed=SEED)
    # pi4s = create_grid_target_distribution(n4, mask4s)

    # prob_s4 = EfficiencyProblemInstanceStochastic(
    #     mA=mA4s, W=W4s, W2=W2_4s, eta=ETA_SMALL, pi_hat=pi4s,
    #     objective_type='maximize_efficiency', use_hard_constraint=True)
    # x_s4 = prob_s4.get_feasible_initial_point()
    # P_s4_unif = x_to_matrix(x_s4, prob_s4.N, prob_s4.edge_matrix, False)
    # m_s4_unif = prob_s4.evaluate_metrics(P_s4_unif)

    # res_2b = run_opt(mA4s, W4s, W2_4s, pi4s, 'maximize_efficiency',
    #                  '2b: Max-Surprise (4x4 Stoch)', eta=ETA_SMALL, max_iter=15000, seed=SEED+2)
    # m_2b = res_2b['mopt']
    # mask_e4s = mA4s > 0
    # cvs4 = CV4s[mask_e4s]
    # corr4 = np.corrcoef(res_2b['Popt'][mask_e4s], CV4s[mask_e4s])[0, 1]
    # print(f"  S: {m_s4_unif['Eff_Idx']:.4f} -> {m_2b['Eff_Idx']:.4f}, Corr(P,CV)={corr4:.4f}")

    # ==================================================================
    # STEP 3: 8x8 NON-UNIFORM pi
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 3: 8x8 Grid, Non-uniform pi, Max Surprise")
    print("=" * 80)

    n8 = 8
    obs8 = [(2, 2), (2, 3), (5, 4), (5, 5)]

    # Priority: nodes adjacent to obstacles get 2x weight (high-surveillance zones)
    priority8 = {}
    for (r, c) in obs8:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n8 and 0 <= nc < n8 and (nr, nc) not in obs8:
                priority8[(nr, nc)] = 2.0

    # # --- 3a: Deterministic ---
    # mA8, W8, _, CV8d, mask8, gpos8 = generate_grid_network_stochastic(
    #     n8, obstacles=obs8, cv_low=0.0, cv_high=0.0, high_cv_fraction=0.0, seed=SEED)
    # W2_8d = W8 ** 2
    # pi8 = create_nonuniform_pi(n8, mask8, priority8, base_weight=1.0)
    # n_active8 = int(np.sum(~mask8))
    # n_edges8 = int(np.sum(mA8))
    # print(f"Grid: {n8}x{n8}, obstacles={obs8}")
    # print(f"  {n_active8} nodes, {n_edges8} edges")
    # print(f"  Priority: 2x on {len(priority8)} nodes adjacent to obstacles")
    # print(f"  pi range: [{pi8[~mask8].min():.4f}, {pi8[~mask8].max():.4f}]")

    # # Feasibility check
    # assert check_feasibility(mA8, pi8), \
    #     f"8x8 det non-uniform is INFEASIBLE! Change obstacles or reduce priority."
    # print("  Feasibility: OK")

    # # Deterministic baseline
    # prob_d8 = EfficiencyProblemInstanceStochastic(
    #     mA=mA8, W=W8, W2=W2_8d, eta=ETA_BIG, pi_hat=pi8,
    #     objective_type='maximize_efficiency', use_hard_constraint=True)
    # x_d8 = prob_d8.get_feasible_initial_point()
    # P_d8_unif = x_to_matrix(x_d8, prob_d8.N, prob_d8.edge_matrix, False)
    # m_d8_unif = prob_d8.evaluate_metrics(P_d8_unif)
    # print(f"  Det baseline: K_W={m_d8_unif['K_W']:.2f}, sqrtV_W={np.sqrt(m_d8_unif['Net_Var']):.2f}, "
    #       f"S={m_d8_unif['Eff_Idx']:.4f}, pi_err={m_d8_unif['pi_error']:.2e}")

    # res_3a = run_opt(mA8, W8, W2_8d, pi8, 'maximize_efficiency',
    #                  '3a: Max-Surprise (8x8 Det)', eta=ETA_BIG,
    #                  max_iter=23000, seed=SEED+3, a=0.01)
    # m_3a = res_3a['mopt']
    # print(f"  S: {m_d8_unif['Eff_Idx']:.4f} -> {m_3a['Eff_Idx']:.4f}")

    # --- 3b: Stochastic ---
    mA8s, W8s, W2_8s, CV8s, mask8s, gpos8s = generate_grid_network_stochastic(
        n8, obstacles=obs8, cv_low=0.3, cv_high=1.5, high_cv_fraction=0.4, seed=SEED)
    pi8s = create_nonuniform_pi(n8, mask8s, priority8, base_weight=1.0)

    assert check_feasibility(mA8s, pi8s), \
        f"8x8 stoch non-uniform is INFEASIBLE!"
    print("  Stoch feasibility: OK")

    prob_s8 = EfficiencyProblemInstanceStochastic(
        mA=mA8s, W=W8s, W2=W2_8s, eta=ETA_BIG, pi_hat=pi8s,
        objective_type='maximize_efficiency', use_hard_constraint=True)
    x_s8 = prob_s8.get_feasible_initial_point()
    P_s8_unif = x_to_matrix(x_s8, prob_s8.N, prob_s8.edge_matrix, False)
    m_s8_unif = prob_s8.evaluate_metrics(P_s8_unif)

    mask_e8s = mA8s > 0
    cvs8 = CV8s[mask_e8s]
    print(f"  Stoch baseline: S={m_s8_unif['Eff_Idx']:.4f}, "
          f"CV: min={cvs8.min():.2f}, max={cvs8.max():.2f}, mean={cvs8.mean():.2f}")

    res_3b = run_opt(mA8s, W8s, W2_8s, pi8s, 'maximize_efficiency',
                     '3b: Max-Surprise (8x8 Stoch)', eta=ETA_BIG,
                     max_iter=23000, seed=SEED+4, a=0.012)
    m_3b = res_3b['mopt']
    P_opt_8s = res_3b['Popt']
    corr8 = np.corrcoef(P_opt_8s[mask_e8s], CV8s[mask_e8s])[0, 1]
    print(f"  S: {m_s8_unif['Eff_Idx']:.4f} -> {m_3b['Eff_Idx']:.4f}, Corr(P,CV)={corr8:.4f}")

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_policy(n4, mA4, mask4, gpos4, P_unif4, m_unif4['pi_W'],
                'Initial Uniform Policy', 'fig1a_uniform.png')

    plot_policy(n4, mA4, mask4, gpos4, res_1['Popt'], m_1['pi_W'],
                'Minimum-Variance Policy', 'fig1b_min_variance.png')

    plot_policy(n4, mA4, mask4, gpos4, res_2a['Popt'], m_2a['pi_W'],
                'Maximum-Surprise Policy (Deterministic)', 'fig2_max_surprise_det.png')

    # plot_policy(n8, mA8, mask8, gpos8, res_3a['Popt'], m_3a['pi_W'],
    #             'Max-Surprise (8x8, Non-uniform, Deterministic)',
    #             'fig4_8x8_det.png', figsize=(9, 9))

    plot_policy(n8, mA8s, mask8s, gpos8s, P_opt_8s, m_3b['pi_W'],
                'Max-Surprise (8x8, Non-uniform, Stochastic)',
                'fig5_8x8_stoch.png', cv_matrix=CV8s, figsize=(9, 9))

    # ==================================================================
    # OPTIMIZATION REPORT
    # ==================================================================
    L = []
    L.append("SURVEILLANCE OPTIMIZATION REPORT")
    L.append("Surprise Index: S(P) = sqrt(V_W) / K_W  (dimensionless)")
    L.append("=" * 65)
    L.append(f"eta (4x4) = {ETA_SMALL},  eta (8x8) = {ETA_BIG}")
    L.append("")

    L.append("-" * 65)
    L.append("4x4 GRID — UNIFORM pi, DETERMINISTIC WEIGHTS")
    L.append(f"  {int(np.sum(~mask4))} nodes, {int(np.sum(mA4))} edges, no obstacles")
    L.append("-" * 65)
    L.append(f"{'Policy':<20} {'K_W':>10} {'sqrtV_W':>14} {'S(P)':>12} {'pi_err':>10}")
    L.append("-" * 65)
    for lbl, m in [('Uniform', m_unif4), ('Min-Variance', m_1), ('Max-Surprise', m_2a)]:
        L.append(f"{lbl:<20} {m['K_W']:>10.2f} {np.sqrt(m['Net_Var']):>14.4f} "
                 f"{m['Eff_Idx']:>12.4f} {m['pi_error']:>10.1e}")
    sqrtv_red = (1 - np.sqrt(m_1['Net_Var']) / np.sqrt(m_unif4['Net_Var'])) * 100
    s_gain = m_2a['Eff_Idx'] / m_unif4['Eff_Idx'] if m_unif4['Eff_Idx'] > 0 else float('inf')
    L.append(f"\nMin-Var: sqrtV_W reduced by {sqrtv_red:.1f}% (Hamiltonian cycle, {dom_edges} dominant edges)")
    L.append(f"Max-Surprise: S increased {s_gain:.1f}x over uniform")

    # L.append("")
    # L.append("-" * 65)
    # L.append("4x4 GRID — UNIFORM pi, STOCHASTIC WEIGHTS")
    # L.append(f"  CV: min={cvs4.min():.2f}, max={cvs4.max():.2f}, mean={cvs4.mean():.2f}")
    # L.append("-" * 65)
    # L.append(f"{'Policy':<20} {'K_W':>10} {'sqrtV_W':>14} {'S(P)':>12} {'pi_err':>10}")
    # L.append("-" * 65)
    # for lbl, m in [('Uniform', m_s4_unif), ('Max-Surprise', m_2b)]:
    #     L.append(f"{lbl:<20} {m['K_W']:>10.2f} {np.sqrt(m['Net_Var']):>14.4f} "
    #              f"{m['Eff_Idx']:>12.4f} {m['pi_error']:>10.1e}")
    # s_s4 = m_2b['Eff_Idx'] / m_s4_unif['Eff_Idx'] if m_s4_unif['Eff_Idx'] > 0 else float('inf')
    # L.append(f"\nMax-Surprise: S increased {s_s4:.1f}x, Corr(P,CV)={corr4:.4f}")

    # L.append("")
    # L.append("-" * 65)
    # L.append("8x8 GRID — NON-UNIFORM pi, DETERMINISTIC WEIGHTS")
    # L.append(f"  {n_active8} nodes, {n_edges8} edges, obstacles={obs8}")
    # L.append(f"  Priority: 2x on {len(priority8)} nodes adjacent to obstacles")
    # L.append(f"  pi range: [{pi8[~mask8].min():.4f}, {pi8[~mask8].max():.4f}]")
    # L.append("-" * 65)
    # L.append(f"{'Policy':<20} {'K_W':>10} {'sqrtV_W':>14} {'S(P)':>12} {'pi_err':>10}")
    # L.append("-" * 65)
    # for lbl, m in [('Uniform', m_d8_unif), ('Max-Surprise', m_3a)]:
    #     L.append(f"{lbl:<20} {m['K_W']:>10.2f} {np.sqrt(m['Net_Var']):>14.4f} "
    #              f"{m['Eff_Idx']:>12.4f} {m['pi_error']:>10.1e}")
    # s_8d = m_3a['Eff_Idx'] / m_d8_unif['Eff_Idx'] if m_d8_unif['Eff_Idx'] > 0 else float('inf')
    # L.append(f"\nMax-Surprise: S increased {s_8d:.1f}x over uniform")

    L.append("")
    L.append("-" * 65)
    L.append("8x8 GRID — NON-UNIFORM pi, STOCHASTIC WEIGHTS")
    L.append(f"  CV: min={cvs8.min():.2f}, max={cvs8.max():.2f}, mean={cvs8.mean():.2f}")
    L.append("-" * 65)
    L.append(f"{'Policy':<20} {'K_W':>10} {'sqrtV_W':>14} {'S(P)':>12} {'pi_err':>10}")
    L.append("-" * 65)
    for lbl, m in [('Uniform', m_s8_unif), ('Max-Surprise', m_3b)]:
        L.append(f"{lbl:<20} {m['K_W']:>10.2f} {np.sqrt(m['Net_Var']):>14.4f} "
                 f"{m['Eff_Idx']:>12.4f} {m['pi_error']:>10.1e}")
    s_8s = m_3b['Eff_Idx'] / m_s8_unif['Eff_Idx'] if m_s8_unif['Eff_Idx'] > 0 else float('inf')
    L.append(f"\nMax-Surprise: S increased {s_8s:.1f}x, Corr(P,CV)={corr8:.4f}")

    report = "\n".join(L)
    with open('optimization_report.txt', 'w') as f:
        f.write(report)
    print(f"\n  -> optimization_report.txt")
    print(report)

    # ==================================================================
    # SUPPLEMENTARY CONVERGENCE PLOTS
    # ==================================================================
    print("\n  Supplementary plots...")
    for res, mi, pf, lb in [
        (res_1,  m_unif4,   'supp_1',  '1: Min-Var (4x4)'),
        (res_2a, m_unif4,   'supp_2a', '2a: Max-Surp (4x4 Det)'),
        # (res_2b, m_s4_unif, 'supp_2b', '2b: Max-Surp (4x4 Stoch)'),
        # (res_3a, m_d8_unif, 'supp_3a', '3a: Max-Surp (8x8 Det)'),
        (res_3b, m_s8_unif, 'supp_3b', '3b: Max-Surp (8x8 Stoch)'),
    ]:
        save_convergence_plots(res['ih'], res['eh'], res['kh'],
                                res['vh'], res['ph'], mi, res['mopt'], pf, lb)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)