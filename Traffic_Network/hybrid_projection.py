import numpy as np


def get_hybrid_target(P_broken, pi_star, destinations):
    """
    Constructs a valid target distribution pi_hybrid where:
    1. Destinations are LOCKED to their original pi_star values.
    2. Intersections are SCALED versions of their naive (natural) values.
    """
    N = P_broken.shape[0]

    # --- Step 1: Calculate the Naive Drift (Natural Physics) ---
    # This tells us: "If we didn't force anything, where would traffic go?"
    # We row-normalize P_broken just to get the eigen-distribution
    P_naive = P_broken.copy()
    row_sums = P_naive.sum(axis=1)
    valid_rows = row_sums > 0
    P_naive[valid_rows] /= row_sums[valid_rows, np.newaxis]

    try:
        evals, evecs = np.linalg.eig(P_naive.T)
        idx = np.argmin(np.abs(evals - 1.0))
        pi_naive = evecs[:, idx].real
        pi_naive = np.abs(pi_naive)
        pi_naive /= np.sum(pi_naive)  # sums to 1
    except:
        print(
            "Warning: Failed to compute naive stationary distribution. Using uniform distribution instead."
        )
        # Fallback for disconnected graphs: uniform distribution
        pi_naive = np.ones(N) / N

    # --- Step 2: Separate the Nodes ---
    mask_dest = np.zeros(N, dtype=bool)
    mask_dest[destinations] = True
    intersections = np.where(~mask_dest)[0]

    # --- Step 3: Construct the Hybrid Vector ---
    pi_hybrid = np.zeros(N)

    # A. LOCK DESTINATIONS (The Strict Rule)
    # We take the values strictly from the original pi_star
    pi_hybrid[destinations] = pi_star[destinations]

    # Calculate how much 'probability budget' is used and what is left
    mass_used = np.sum(pi_hybrid[destinations])
    mass_remaining = 1.0 - mass_used

    if mass_remaining < 0:
        print("Error: Destinations alone exceed 100% probability!")
        return None

    # B. FILL INTERSECTIONS (The Flexible Rule)
    # We take the naive values for intersections...
    naive_mass_intersections = np.sum(pi_naive[intersections])

    # ...and SCALE them to fit exactly into 'mass_remaining'
    if naive_mass_intersections > 0:
        scaling_factor = mass_remaining / naive_mass_intersections
        pi_hybrid[intersections] = pi_naive[intersections] * scaling_factor
    else:
        # Rare case: Naive flow puts 0 mass on intersections
        if len(intersections) > 0:
            pi_hybrid[intersections] = mass_remaining / len(intersections)

    return pi_hybrid
