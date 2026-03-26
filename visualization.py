"""
Visualization functions for stochastic surveillance optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_optimization_results(iter_hist, eff_hist, kw_hist, var_hist,
                               initial_metrics, final_metrics,
                               n, case_name='',
                               filename=None):
    """
    Plot optimization convergence (2×2 grid like original).
    
    Parameters:
        iter_hist: List of iteration numbers
        eff_hist: List of efficiency index values
        kw_hist: List of K_W values
        var_hist: List of variance values
        initial_metrics: Dict with initial metrics
        final_metrics: Dict with final metrics
        n: Grid size (for title)
        case_name: Name of the case (e.g., 'Deterministic', 'Stochastic')
        filename: If provided, save plot to this file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Efficiency Index
    axes[0, 0].plot(iter_hist, eff_hist, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].axhline(y=initial_metrics['Eff_Idx'], color='g', linestyle='--', 
                       label=f"Initial = {initial_metrics['Eff_Idx']:.4f}")
    axes[0, 0].axhline(y=final_metrics['Eff_Idx'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['Eff_Idx']:.4f}")
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(r'Efficiency Index $\lambda$')
    axes[0, 0].set_title(r'Efficiency Index ($\uparrow$ higher is better)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Patrol Time (K_W)
    axes[0, 1].plot(iter_hist, kw_hist, 'orange', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=initial_metrics['K_W'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['K_W']:.4f}")
    axes[0, 1].axhline(y=final_metrics['K_W'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['K_W']:.4f}")
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[0, 1].set_title('Mean Patrol Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Path Variance
    axes[1, 0].plot(iter_hist, var_hist, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(y=initial_metrics['Net_Var'], color='g', linestyle='--',
                       label=f"Initial = {initial_metrics['Net_Var']:.4f}")
    axes[1, 0].axhline(y=final_metrics['Net_Var'], color='r', linestyle='--',
                       label=f"Final = {final_metrics['Net_Var']:.4f}")
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 0].set_title(r'Path Variance ($\uparrow$ more unpredictable)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mean-Variance tradeoff
    scatter = axes[1, 1].scatter(kw_hist, var_hist, c=iter_hist, cmap='viridis', 
                                  s=50, alpha=0.7)
    axes[1, 1].scatter(initial_metrics['K_W'], initial_metrics['Net_Var'], 
                      color='green', s=300, marker='*', label='Initial', zorder=5, edgecolor='black')
    axes[1, 1].scatter(final_metrics['K_W'], final_metrics['Net_Var'],
                      color='red', s=300, marker='*', label='Final', zorder=5, edgecolor='black')
    axes[1, 1].set_xlabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    axes[1, 1].set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    axes[1, 1].set_title('Mean-Variance Tradeoff')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Iteration')
    
    title = f'{n}×{n} Grid Network'
    if case_name:
        title += f' - {case_name}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"✓ Optimization plot saved to '{filename}'")
    
    plt.show()
    return fig


def plot_optimization_comparison(iter_det, eff_det, kw_det, var_det,
                                  iter_stoch, eff_stoch, kw_stoch, var_stoch,
                                  init_det, final_det, init_stoch, final_stoch,
                                  n, filename=None,
                                  pi_err_det=None, pi_err_stoch=None):
    """
    Plot optimization convergence comparing deterministic vs stochastic weights.
    Includes optional pi_error plot.
    """
    # Determine if we have pi_error data
    has_pi_err = pi_err_det is not None and pi_err_stoch is not None
    
    if has_pi_err:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Efficiency Index Convergence
    ax = axes[0, 0]
    if len(iter_det) > 0:
        ax.plot(iter_det, eff_det, 'b-', linewidth=2, marker='o', 
                        markersize=4, label='Deterministic')
    if len(iter_stoch) > 0:
        ax.plot(iter_stoch, eff_stoch, 'r-', linewidth=2, marker='s', 
                        markersize=4, label='Stochastic')
    ax.axhline(y=init_det['Eff_Idx'], color='blue', linestyle=':', alpha=0.5,
                       label=f"Det Initial = {init_det['Eff_Idx']:.2f}")
    ax.axhline(y=init_stoch['Eff_Idx'], color='red', linestyle=':', alpha=0.5,
                       label=f"Stoch Initial = {init_stoch['Eff_Idx']:.2f}")
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Efficiency Index $\lambda$')
    ax.set_title(r'Efficiency Index ($\uparrow$ higher is better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean Patrol Time (K_W)
    ax = axes[0, 1]
    if len(iter_det) > 0:
        ax.plot(iter_det, kw_det, 'b-', linewidth=2, marker='o', 
                        markersize=4, label='Deterministic')
    if len(iter_stoch) > 0:
        ax.plot(iter_stoch, kw_stoch, 'r-', linewidth=2, marker='s', 
                        markersize=4, label='Stochastic')
    ax.axhline(y=init_det['K_W'], color='green', linestyle='--', alpha=0.7,
                       label=f"Initial K_W = {init_det['K_W']:.2f}")
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    ax.set_title('Mean Patrol Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pi Error (if available) or Path Variance
    if has_pi_err:
        ax = axes[0, 2]
        if len(iter_det) > 0 and len(pi_err_det) > 0:
            ax.plot(iter_det, pi_err_det, 'b-', linewidth=2, marker='o', 
                    markersize=4, label='Deterministic')
        if len(iter_stoch) > 0 and len(pi_err_stoch) > 0:
            ax.plot(iter_stoch, pi_err_stoch, 'r-', linewidth=2, marker='s', 
                    markersize=4, label='Stochastic')
        ax.axhline(y=init_det['pi_error'], color='blue', linestyle=':', alpha=0.5,
                   label=f"Det Initial = {init_det['pi_error']:.4f}")
        ax.axhline(y=init_stoch['pi_error'], color='red', linestyle=':', alpha=0.5,
                   label=f"Stoch Initial = {init_stoch['pi_error']:.4f}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\|\pi - \hat{\pi}\|_2$')
        ax.set_title(r'Stationary Distribution Error ($\downarrow$ lower is better)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Path Variance
    ax = axes[1, 0]
    if len(iter_det) > 0:
        ax.plot(iter_det, var_det, 'b-', linewidth=2, marker='o', 
                        markersize=4, label='Deterministic')
    if len(iter_stoch) > 0:
        ax.plot(iter_stoch, var_stoch, 'r-', linewidth=2, marker='s', 
                        markersize=4, label='Stochastic')
    ax.axhline(y=init_det['Net_Var'], color='blue', linestyle=':', alpha=0.5,
                       label=f"Det Initial = {init_det['Net_Var']:.2f}")
    ax.axhline(y=init_stoch['Net_Var'], color='red', linestyle=':', alpha=0.5,
                       label=f"Stoch Initial = {init_stoch['Net_Var']:.2f}")
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    ax.set_title(r'Path Variance ($\uparrow$ more unpredictable)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Mean-Variance Tradeoff
    ax = axes[1, 1]
    scatter = None
    if len(kw_det) > 0 and len(var_det) > 0:
        scatter_det = ax.scatter(kw_det, var_det, c=iter_det, 
                                          cmap='viridis', s=50, alpha=0.7, marker='o')
        scatter = scatter_det
    if len(kw_stoch) > 0 and len(var_stoch) > 0:
        scatter_stoch = ax.scatter(kw_stoch, var_stoch, c=iter_stoch, 
                                            cmap='viridis', s=50, alpha=0.7, marker='s')
        if scatter is None:
            scatter = scatter_stoch
    
    # Mark initial and final points
    ax.scatter(init_det['K_W'], init_det['Net_Var'], 
                      color='green', s=300, marker='o', label='Det Initial', 
                      zorder=5, edgecolor='black', linewidth=2)
    ax.scatter(final_det['K_W'], final_det['Net_Var'],
                      color='red', s=300, marker='o', label='Det Final', 
                      zorder=5, edgecolor='black', linewidth=2)
    ax.scatter(init_stoch['K_W'], init_stoch['Net_Var'], 
                      color='green', s=300, marker='s', label='Stoch Initial', 
                      zorder=5, edgecolor='black', linewidth=2)
    ax.scatter(final_stoch['K_W'], final_stoch['Net_Var'],
                      color='red', s=300, marker='s', label='Stoch Final', 
                      zorder=5, edgecolor='black', linewidth=2)
    
    ax.set_xlabel(r'Mean Patrol Time $K_{\mathcal{W}}$')
    ax.set_ylabel(r'Path Variance $V_{\mathcal{W}}$')
    ax.set_title('Mean-Variance Tradeoff\n(○ = Deterministic, □ = Stochastic)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Iteration')
    
    # Plot 6: Pi Error vs Efficiency Tradeoff (if available)
    if has_pi_err:
        ax = axes[1, 2]
        if len(eff_det) > 0 and len(pi_err_det) > 0:
            scatter_det = ax.scatter(pi_err_det, eff_det, c=iter_det, 
                                     cmap='viridis', s=50, alpha=0.7, marker='o')
        if len(eff_stoch) > 0 and len(pi_err_stoch) > 0:
            scatter_stoch = ax.scatter(pi_err_stoch, eff_stoch, c=iter_stoch, 
                                       cmap='viridis', s=50, alpha=0.7, marker='s')
        
        ax.scatter(init_det['pi_error'], init_det['Eff_Idx'], 
                   color='green', s=300, marker='o', label='Det Initial', 
                   zorder=5, edgecolor='black', linewidth=2)
        ax.scatter(final_det['pi_error'], final_det['Eff_Idx'],
                   color='red', s=300, marker='o', label='Det Final', 
                   zorder=5, edgecolor='black', linewidth=2)
        ax.scatter(init_stoch['pi_error'], init_stoch['Eff_Idx'], 
                   color='green', s=300, marker='s', label='Stoch Initial', 
                   zorder=5, edgecolor='black', linewidth=2)
        ax.scatter(final_stoch['pi_error'], final_stoch['Eff_Idx'],
                   color='red', s=300, marker='s', label='Stoch Final', 
                   zorder=5, edgecolor='black', linewidth=2)
        
        ax.set_xlabel(r'$\|\pi_W - \hat{\pi}\|_2$ (Distribution Error)')
        ax.set_ylabel(r'Efficiency Index $\lambda$')
        ax.set_title('Efficiency vs Distribution Error Tradeoff')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{n}×{n} Grid Network - Deterministic vs Stochastic Weights', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"\n✓ Optimization comparison saved to '{filename}'")
    
    plt.show()
    return fig



def plot_grid_network(n, mA, CV_matrix, obstacle_mask, grid_positions,
                      P=None, pi=None, title="Grid Network", filename=None):
    """
    Plot grid network showing optimal surveillance policy.
    
    - Edge THICKNESS indicates transition probability P(i,j)
    - Edge COLOR indicates reliability: green (CV < 1) vs red (CV > 1)
    - Node color intensity indicates stationary distribution π
    - Curved arrows to distinguish bidirectional edges
    """
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    edge_list = np.argwhere(mA > 0)
    
    # Determine max P for scaling line width
    if P is not None:
        max_P = np.max(P[mA > 0]) if np.any(mA > 0) else 1.0
        min_P = np.min(P[mA > 0]) if np.any(mA > 0) else 0.0
    else:
        max_P, min_P = 1.0, 0.0
    
    for i, j in edge_list:
        ri, ci = grid_positions[i]
        rj, cj = grid_positions[j]
        cv = CV_matrix[i, j]
        
        # Color by CV: green (reliable) vs red (unreliable)
        if cv < 1:
            color = 'green'
            alpha = 0.6 + 0.4 * (1 - cv)
        else:
            color = 'red'
            alpha = min(0.6 + 0.2 * (cv - 1), 0.95)
        
        # Line width by P value
        if P is not None and P[i, j] > 0.001:
            if max_P > min_P:
                normalized_P = (P[i, j] - min_P) / (max_P - min_P)
            else:
                normalized_P = 0.5
            linewidth = 0.5 + 5.5 * normalized_P
        else:
            linewidth = 0.5
        
        # Use curved FancyArrowPatch for better visibility
        arrow = FancyArrowPatch(
            (ci, ri), (cj, rj),
            connectionstyle='arc3,rad=0.2',
            arrowstyle='-|>',
            mutation_scale=15,
            color=color,
            lw=linewidth,
            alpha=alpha,
            shrinkA=12,
            shrinkB=12,
            zorder=1
        )
        ax.add_patch(arrow)
    
    # Draw nodes
    for i in range(n * n):
        r, c = grid_positions[i]
        if obstacle_mask[i]:
            ax.add_patch(plt.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                        color='black', zorder=3))
            ax.text(c, r, 'X', ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold', zorder=4)
        else:
            node_color = 'lightblue'
            if pi is not None:
                intensity = min(pi[i] / (np.max(pi) + 1e-12), 1.0)
                node_color = plt.cm.Blues(0.3 + 0.7 * intensity)
            
            circle = plt.Circle((c, r), 0.3, color=node_color, ec='black', 
                                linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=4)
    
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend
    green_line = plt.Line2D([0], [0], color='green', linewidth=3, label='Reliable (CV < 1)')
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Unreliable (CV ≥ 1)')
    thick_line = plt.Line2D([0], [0], color='gray', linewidth=5, label='High P(i,j)')
    thin_line = plt.Line2D([0], [0], color='gray', linewidth=1, label='Low P(i,j)')
    ax.legend(handles=[green_line, red_line, thick_line, thin_line], 
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to '{filename}'")
    
    plt.show()
    return fig


def plot_policy_comparison(n, mA, CV_matrix, obstacle_mask, grid_positions,
                                    P_det, P_stoch, pi_det, pi_stoch,
                                    filename='Figures/optimal_policy_comparison.png'):
    """
    Plot side-by-side comparison of optimal policies for deterministic vs stochastic weights.
    - Deterministic: all edges gray (no CV variation)
    - Stochastic: edges colored by CV (green=reliable, red=unreliable)
    - Edge thickness indicates P(i,j)
    - Curved arrows for better visibility of direction
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax_idx, (P, pi, case_title, use_cv_color) in enumerate([
        (P_det, pi_det, 'Deterministic Weights (CV=0)', False),  # No CV coloring
        (P_stoch, pi_stoch, 'Stochastic Weights (Mixed CV)', True)  # CV coloring
    ]):
        ax = axes[ax_idx]
        
        # Draw edges
        edge_list = np.argwhere(mA > 0)
        max_P = np.max(P[mA > 0]) if np.any(mA > 0) else 1.0
        min_P = np.min(P[mA > 0]) if np.any(mA > 0) else 0.0
        
        for i, j in edge_list:
            ri, ci = grid_positions[i]
            rj, cj = grid_positions[j]
            cv = CV_matrix[i, j]
            
            # Color by CV only for stochastic case
            if use_cv_color:
                if cv < 1:
                    color = 'green'
                    alpha = 0.6 + 0.4 * (1 - cv)
                else:
                    color = 'red'
                    alpha = min(0.6 + 0.2 * (cv - 1), 0.95)
            else:
                # Deterministic: all edges gray
                color = 'dimgray'
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
            
            # Use curved arrows (FancyArrowPatch) for better visibility
            from matplotlib.patches import FancyArrowPatch
            
            # Calculate curve direction (perpendicular offset)
            dx = cj - ci
            dy = rj - ri
            
            # Shrink start and end points to not overlap with nodes
            shrink = 0.20
            start = (ci + shrink * dx, ri + shrink * dy)
            end = (cj - shrink * dx, rj - shrink * dy)
            
            # Curved arrow with connectionstyle
            arrow = FancyArrowPatch(
                start, end,
                connectionstyle="arc3,rad=0.2",  # Curved arrow
                arrowstyle='-|>',
                mutation_scale=10 + 5 * normalized_P if P[i, j] > 0.001 else 8,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for i in range(n * n):
            r, c = grid_positions[i]
            if obstacle_mask[i]:
                ax.add_patch(plt.Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, 
                                            color='black', zorder=3))
                ax.text(c, r, 'X', ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold', zorder=4)
            else:
                node_color = 'lightblue'
                if pi is not None:
                    intensity = min(pi[i] / (np.max(pi) + 1e-12), 1.0)
                    node_color = plt.cm.Blues(0.3 + 0.7 * intensity)
                
                circle = plt.Circle((c, r), 0.2, color=node_color, ec='black', 
                                    linewidth=1.5, zorder=3)
                ax.add_patch(circle)
                ax.text(c, r, str(i), ha='center', va='center', fontsize=8, zorder=4)
        
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(n - 0.3, -0.7)
        ax.set_aspect('equal')
        ax.set_title(case_title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add shared legend
    from matplotlib.lines import Line2D
    gray_line = Line2D([0], [0], color='dimgray', linewidth=3, label='Deterministic (no CV)')
    green_line = Line2D([0], [0], color='green', linewidth=3, label='Reliable (CV < 1)')
    red_line = Line2D([0], [0], color='red', linewidth=3, label='Unreliable (CV ≥ 1)')
    thick_line = Line2D([0], [0], color='gray', linewidth=5, label='High P(i,j)')
    thin_line = Line2D([0], [0], color='gray', linewidth=1, label='Low P(i,j)')
    
    fig.legend(handles=[gray_line, green_line, red_line, thick_line, thin_line], 
               loc='lower center', ncol=5, fontsize=11)
    
    plt.suptitle('Optimal Surveillance Policy Comparison\n(Edge thickness = P, Curved arrows show direction)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Policy comparison saved to '{filename}'")
    
    plt.show()
    return fig


def plot_policy_vs_cv(P_det, P_stoch, CV_matrix, mA, filename=None):
    """Plot scatter of P vs CV for both cases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mask = mA > 0
    cv_vals = CV_matrix[mask]
    
    for ax, (P, title) in zip(axes, [(P_det, 'Deterministic'), (P_stoch, 'Stochastic')]):
        p_vals = P[mask]
        colors = ['green' if cv < 1 else 'red' for cv in cv_vals]
        ax.scatter(cv_vals, p_vals, c=colors, alpha=0.6, s=50)
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coefficient of Variation (CV)')
        ax.set_ylabel('Transition Probability P(i,j)')
        ax.set_title(f'{title} Case: P vs CV')
        ax.grid(True, alpha=0.3)
        
        # Trend line
        z = np.polyfit(cv_vals, p_vals, 1)
        p_trend = np.poly1d(z)
        cv_range = np.linspace(cv_vals.min(), cv_vals.max(), 100)
        ax.plot(cv_range, p_trend(cv_range), 'b--', alpha=0.7, 
                label=f'Trend (slope={z[0]:.3f})')
        ax.legend()
    
    plt.suptitle('Does Optimal Policy Favor High-CV Edges?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"✓ Policy vs CV analysis saved to '{filename}'")
    
    plt.show()
    return fig
