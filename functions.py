from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy


def generate_transition_matrix(num_nodes, connection_prob, seed):
    """
    Generate a random transition matrix using geometric graph.

    Parameters:
    num_nodes (int): Number of nodes in the network.
    connection_prob (float): Probability of connection between nodes.

    Returns:
    np.ndarray: A transition matrix representing the network.
    """
    # Generate a random geometric graph
    G = nx.random_geometric_graph(num_nodes, connection_prob, seed=seed)
    # Get the adjacency matrix of the graph
    A = nx.to_numpy_array(G)
    # Normalize the adjacency matrix to get the transition matrix
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes
    row_sums[row_sums == 0] = 1
    transition_matrix = A / row_sums
    return transition_matrix


def plot_graph(P, W, dest_set, title, figsize=(6, 4), show=True):
    """
    Plot the graph represented by the transition matrix P and weights W.

    Parameters:
    P (np.ndarray): Transition matrix.
    W (np.ndarray): Weight matrix.
    title (str): Title of the plot.
    figsize (tuple): Matplotlib figure size.
    show (bool): Whether to display the plot immediately.
    dest_set (list or np.ndarray): Indices of destination nodes.
    """
    # create adjacency matrix from transition matrix
    A = 1 * (P > 0)

    # create directed graph from adjacency matrix
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)

    pos_shadow = copy.deepcopy(pos)
    shift_amount = 0.009
    for idx in pos_shadow:
        pos_shadow[idx][0] += shift_amount
        pos_shadow[idx][1] -= shift_amount

    fig = plt.figure(figsize=figsize, frameon=False)

    nx.draw_networkx_nodes(G, pos_shadow, node_color="k", alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color="#e49090", linewidths=1)
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos=pos)

    # nodes from dest_set in a different shape
    nx.draw_networkx_nodes(
        G, pos_shadow, nodelist=dest_set, node_color="k", alpha=0.5, node_shape="s"
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=dest_set, node_color="#90b6e4", node_shape="s", linewidths=1
    )

    plt.title(title)
    if show:
        plt.show()
    return G


def stationary_distribution(P):
    """
    Compute the stationary distribution of a Markov chain given its transition matrix.

    Parameters:
    P (np.ndarray): Transition matrix.

    Returns:
    np.ndarray: Stationary distribution vector.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    mask = np.isclose(eigvals, 1.0, atol=1e-8)

    if not np.any(mask):
        # If no eigenvalue close to 1 is found, use the largest magnitude eigenvalue
        idx = np.argmax(np.abs(eigvals))
        stat_dist = np.real(eigvecs[:, idx])
    else:
        stat_dist = np.real(eigvecs[:, mask])
        if stat_dist.ndim > 1:
            stat_dist = stat_dist[:, 0]

    stat_dist = np.abs(stat_dist)  # Ensure non-negative
    stat_dist = stat_dist / stat_dist.sum()
    return stat_dist.flatten()


def kemeny_constant(P, pi, W):
    """
    Function to compute the Kemeny constant of a Markov chain with transition matrix P and time-weight matrix W.
    """
    num_states = P.shape[0]
    pi = np.asarray(pi).flatten()  # Ensure pi is 1D

    if len(pi) != num_states:
        raise ValueError(
            f"Stationary distribution size {len(pi)} does not match P size {num_states}"
        )

    fundamental_matrix = np.linalg.inv(
        np.eye(num_states) - P + np.outer(np.ones(num_states), pi)
    )

    term1 = np.trace(fundamental_matrix)
    term2 = pi @ (P * W).sum(axis=1)

    return term1 * term2


def mfpt(P, W):
    """
    Function to calculate the mean first passage times from the formula of my thesis.
    """
    num_states = P.shape[0]
    pi = stationary_distribution(P)
    Pi = np.outer(np.ones(num_states), pi)
    fundamental_matrix = np.linalg.inv(np.eye(num_states) - P + Pi)
    scalar_term = pi @ (P * W).sum(axis=1)

    # matrix of all ones
    J = np.ones((num_states, num_states))

    # matrix with 1/pi_i on the i-th diagonal
    Xi_inv = np.diag(1 / pi)

    term1 = fundamental_matrix @ (P * W) @ Pi
    term2 = J @ np.diag(np.diag(term1))
    term3 = scalar_term * (
        np.eye(num_states)
        - fundamental_matrix
        + J @ np.diag(np.diag(fundamental_matrix))
    )
    return (term1 - term2 + term3) @ Xi_inv


def sec_moment_mfpt(P, W, sec_moment_W):
    """
    Function to calculate the second moment of first passage times from the formula of my thesis.
    """
    num_states = P.shape[0]
    pi = stationary_distribution(P)
    Pi = np.outer(np.ones(num_states), pi)
    fundamental_matrix = np.linalg.inv(np.eye(num_states) - P + Pi)
    scalar_term = pi @ (P * W).sum(axis=1)

    # matrix of all ones
    J = np.ones((num_states, num_states))

    MFPT = mfpt(P, W)

    D = np.zeros((num_states, num_states))
    for i in range(num_states):
        D[i, i] = (
            pi @ (P * sec_moment_W).sum(axis=1)
            + 2 * (pi @ (P * W) @ (MFPT - np.diag(np.diag(MFPT))))
        )[i] / pi[i]

    term1 = fundamental_matrix @ (P * sec_moment_W) @ J - J @ np.diag(
        np.diag(fundamental_matrix @ (P * sec_moment_W) @ J)
    )
    term2 = fundamental_matrix @ (P * W) @ (
        MFPT - np.diag(np.diag(MFPT))
    ) - J @ np.diag(
        np.diag(fundamental_matrix @ (P * W) @ (MFPT - np.diag(np.diag(MFPT))))
    )
    term3 = (
        np.eye(num_states)
        - fundamental_matrix
        + J @ np.diag(np.diag(fundamental_matrix))
    ) @ np.diag(np.diag(D))

    return term1 + 2 * term2 + term3


def variance(P, W, sec_moment_W):
    """
    Function to calculate the variance of first passage times from the formula of my thesis.
    """
    MFPT = mfpt(P, W)
    sec_moment_MFPT = sec_moment_mfpt(P, W, sec_moment_W)

    return sec_moment_MFPT - (MFPT * MFPT)


def variance_based_kemeny(P, pi, W, sec_moment_W):
    """
    Function to compute the variance-based Kemeny constant of a Markov chain with transition matrix P, time-weight matrix W,
    and second moment time-weight matrix sec_moment_W.
    """
    V = variance(P, W, sec_moment_W)

    return pi.T @ V @ pi


def get_fundamental_matrix(P, pi):
    """
    Function to compute the fundamental matrix of a Markov chain with transition matrix P and stationary distribution pi.
    """
    num_states = len(P)
    pi = np.asarray(pi).flatten()  # Ensure pi is 1D
    Pi = np.outer(np.ones(num_states), pi)
    fundamental_matrix = np.linalg.inv(np.eye(num_states) - P + Pi)
    return fundamental_matrix
