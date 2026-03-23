import itertools

import numpy as np


def l_from_p(p_change, n_states=4) -> float | np.ndarray:
    """
    Inverse formula to determine length from p_change (probability of changing *to any state*).
    Parameters
    ----------
    p_change: float or np.ndarray, probability of changing to any state
    n_states: int, number of copy number states

    Returns
    -------
    float, length parameter
    """
    eps = 1e-10 # to avoid log(0) and log(1) = -0.
    return - 1 / n_states * np.log(1 - n_states / (n_states - 1) * p_change + eps)

def p_from_l(l, n_states=4) -> float | np.ndarray:
    """
    Formula to determine p_change (probability of changing *to any state*) from length.
    Parameters
    ----------
    l: float or np.ndarray, length parameter
    n_states: int, number of copy number states

    Returns
    -------
    float, probability of changing to any state
    """
    return (1 - np.exp(-n_states * l)) * (n_states - 1) / n_states

def cn_changes_from_healthy(cnps: np.ndarray, healthy_state: int = 2) -> list:
    """
    Compute the changes in copy number states from a healthy state for a given profile.
    Parameters
    ----------
    cn_profile: np.ndarray, shape (n_nodes, n_sites), copy number profiles for each node
    healthy_state: int, the healthy copy number state (default: 2)

    Returns
    -------
    np.ndarray, shape (n_nodes,), number of changes from healthy state for each node
    """
    n_bins = cnps.shape[1]
    healthy_cn = np.full(n_bins, healthy_state)
    changes = compute_cn_changes(np.vstack([healthy_cn, cnps]), pairs=[(0, i + 1) for i in range(cnps.shape[0])])
    return changes

def compute_cn_changes_pair(cn1, cn2) -> int:
    """
    Compute the changes in copy number states between two profiles according to the CopyTree model.
    Parameters
    ----------
    cn1: np.ndarray, shape (n_sites,), copy number profile for node 1
    cn2: np.ndarray, shape (n_sites,), copy number profile for node 2

    Returns
    -------
    int, number of changes between the two profiles
    """
    changes = int(cn1[0] != cn2[0])
    changes += np.sum(np.diff(cn1) != np.diff(cn2)).item()
    return changes

def compute_cn_changes(cn_profile: np.ndarray, pairs: list = None) -> list:
    """
    Compute the changes in copy number states for a given profile according to the CopyTree model.
    Parameters
    ----------
    cn_profile: np.ndarray, shape (n_nodes, n_sites), copy number profiles for each node
    pairs: list of tuples, pairs of nodes to compare (default: [(0, 1)])

    Returns
    -------
    list of changes for each pair of nodes
    """
    if pairs is None:
        pairs = [(0, 1)]

    changes_list = []
    for u, v in pairs:
        changes = compute_cn_changes_pair(cn_profile[u], cn_profile[v])
        changes_list.append(changes)
    return changes_list

def get_expected_branch_lengths_from_cnps(cnps: np.ndarray, n_states: int, model='jcb'):
    """
    Estimate the expected branch lengths for CNP pairs from copy number profiles using the JCB or CopyTree model.
    Parameters
    ----------
    cnps: np.ndarray, shape (n_nodes, n_sites), copy number profiles for each node
    n_states: int, number of copy number states
    model: str, 'jcb' or 'copytree'

    Returns
    -------
    dict with keys as tuples of node pairs and values as expected branch lengths
    """
    n_nodes, n_sites = cnps.shape
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    changes_list = compute_cn_changes(cnps, pairs=pairs)

    expected_lengths = {}
    for (u, v), changes in zip(pairs, changes_list):
        # FIXME: why differentiating models here if the formula is the same?
        if model == 'jcb':
            p_change = changes / n_sites
        elif model == 'copytree':
            p_change = changes / n_sites
        else:
            raise ValueError("Model must be 'jcb' or 'copytree'")
        expected_length = l_from_p(p_change, n_states=n_states)
        expected_lengths[(u, v)] = float(expected_length)

    return expected_lengths


import numpy as np
from typing import Tuple, List


# --- Helpers (some are from the previous implementation) ---

def _get_log_probs(eps: float, K: int) -> Tuple[float, float]:
    """Helper to get log(1-eps) and log(eps/(K-1))"""
    C = K - 1
    if C == 0: return 0.0, -np.inf
    if eps == 0.0: return 0.0, -np.inf
    if eps == 1.0: return -np.inf, 0.0
    return np.log(1.0 - eps), np.log(eps / C)


def _flatten_state(l: int, p: int, o: int, K: int) -> int:
    """Converts a (l, p, o) state tuple to a single integer index."""
    return (l * K + p) * K + o


def _unflatten_state(flat_index: int, K: int) -> Tuple[int, int, int]:
    """Converts a flat state index back to a (l, p, o) tuple."""
    o = flat_index % K
    p = (flat_index // K) % K
    l = flat_index // (K * K)
    return l, p, o


def _build_log_A_t(
        K: int,
        Delta_r_t: int,
        log_probs_ru: Tuple[float, float],
        log_probs_uv: Tuple[float, float],
        log_probs_uw: Tuple[float, float]
) -> np.ndarray:
    """
    Vectorized O(K^6) function to build the full transition tensor.

    Returns:
        log_A_t (K, K, K, K, K, K): The tensor where
                                  log_A_t[k,m,n, l,p,o] is the transition prob.
    """
    # 1. Create 6D coordinate grids
    k, m, n, l, p, o = np.indices((K, K, K, K, K, K))

    # 2. Calculate deltas for all K^6 possibilities
    Delta_u = k - l
    Delta_v = m - p
    Delta_w = n - o

    # 3. Get log probs for each link
    log_p_ru_hi, log_p_ru_lo = log_probs_ru
    log_p_uv_hi, log_p_uv_lo = log_probs_uv
    log_p_uw_hi, log_p_uw_lo = log_probs_uw

    # 4. Build each log_P tensor using np.where
    log_P_ru = np.where(Delta_u == Delta_r_t, log_p_ru_hi, log_p_ru_lo)
    log_P_uv = np.where(Delta_v == Delta_u, log_p_uv_hi, log_p_uv_lo)
    log_P_uw = np.where(Delta_w == Delta_u, log_p_uw_hi, log_p_uw_lo)

    # 5. Sum them to get the final K^6 tensor
    log_A_t = log_P_ru + log_P_uv + log_P_uw

    return log_A_t


def viterbi_matrix_K6(
        log_emissions: np.ndarray,
        Z_r: np.ndarray,
        log_pi: np.ndarray,
        eps_ru: float,
        eps_uv: float,
        eps_uw: float
) -> Tuple[List[Tuple[int, int, int]], float]:
    """
    Performs the standard O(M*K^6) Viterbi algorithm using
    vectorized numpy operations instead of explicit for-loops.

    Returns:
        best_path: List of (l, p, o) tuples representing the most probable path
        max_log_prob: Log probability of the best path

    WARNING: This function is EXTREMELY memory-intensive.
    """
    M, K, _ = log_emissions.shape
    N = K ** 3  # Total number of composite states

    # --- 1. Get Log Probs for Transitions ---
    # print("EPS values passed to viterbi: ", (eps_ru, eps_uv, eps_uw))
    log_probs_ru = _get_log_probs(eps_ru, K)
    log_probs_uv = _get_log_probs(eps_uv, K)
    log_probs_uw = _get_log_probs(eps_uw, K)

    # --- 2. Initialization (t=0) ---
    log_delta = np.full((M, K, K, K), -np.inf)

    # psi stores the *flat index* of the best previous state
    psi = np.zeros((M, K, K, K), dtype=np.int32)

    log_emissions_t0 = log_emissions[0, None, :, :]  # Shape (1, K, K)
    log_delta[0] = log_pi + log_emissions_t0

    # --- 3. Recursion (t=1 to M-1) ---
    # This is the O(M * K^6) section

    for t in range(1, M):
        # --- This part replaces the 6-fold loop ---

        # 3a. Get the K^6 transition tensor
        Delta_r_t = Z_r[t - 1] - Z_r[t]
        log_A_t = _build_log_A_t(
            K, Delta_r_t, log_probs_ru, log_probs_uv, log_probs_uw
        )

        # 3b. Prepare log_delta[t-1] for broadcasting
        # Shape (k, m, n) -> (k, m, n, 1, 1, 1)
        log_delta_tm1_expanded = log_delta[t - 1][:, :, :, np.newaxis, np.newaxis, np.newaxis]

        # 3c. Compute the max-sum
        # This is log(delta_{t-1}(k,m,n)) + log(A_t(kmn, lpo))
        # Shape: (k, m, n, l, p, o)
        log_sum_tensor = log_delta_tm1_expanded + log_A_t

        # 3d. Flatten the source axes (k,m,n) to N = K^3
        # Shape: (K^3, K, K, K)
        flat_log_sum_tensor = log_sum_tensor.reshape((N, K, K, K))

        # 3e. Find the max log-prob and the argmax (backpointer)
        # We take the max over the source axis (axis=0)
        max_log_prob = np.max(flat_log_sum_tensor, axis=0)  # Shape (l, p, o)
        psi[t] = np.argmax(flat_log_sum_tensor, axis=0)  # Shape (l, p, o)

        # 3f. Store the final delta value
        # Shape (l, p, o) + (p, o) -> (l, p, o)
        log_delta[t] = max_log_prob + log_emissions[t, None, :, :]
        # --- End of loop replacement ---

    # --- 4. Termination ---
    max_log_prob = np.max(log_delta[M - 1])
    best_last_state_flat = np.argmax(log_delta[M - 1].reshape(N))

    # --- 5. Backtracking (same as before) ---
    best_path_flat = np.zeros(M, dtype=np.int32)
    best_path_flat[M - 1] = best_last_state_flat

    for t in range(M - 2, -1, -1):
        current_state_flat = best_path_flat[t + 1]
        l, p, o = _unflatten_state(current_state_flat, K)

        # Look up the flat index from the psi table
        prev_state_flat = psi[t + 1, l, p, o]
        best_path_flat[t] = prev_state_flat

    best_path = np.array([_unflatten_state(idx, K) for idx in best_path_flat])
    best_path_ordered = best_path[:, [1, 2, 0]]  # reorder to (uv, uw, ru)
    return best_path_ordered, max_log_prob


# --- New Helper for O(K) Optimized Max ---

def _get_special_max_and_argmax(
        log_delta_slice_1d: np.ndarray,
        K: int,
        dest_idx: int,
        delta: int,
        log_prob_hi: float,
        log_prob_lo: float
) -> Tuple[float, int]:
    """
    Computes max_k [ log_delta(k) + log P(k -> dest_idx) ] in O(K) time.

    This is the core optimization.

    Args:
        log_delta_slice_1d: A 1D array of previous log-probs (size K).
        K: Number of states.
        dest_idx: The destination state (e.g., 'l', 'p', or 'o').
        delta: The required change (e.g., Delta_r_t or Delta_u).
        log_prob_hi/lo: The transition log-probs.

    Returns:
        max_log_prob: The max log-probability for this destination.
        argmax_k: The index of the source state 'k' that gave the max.
    """

    # --- 1. High-Probability Path ---
    # Find the "special" source k* that leads to dest_idx
    source_k_star = dest_idx + delta

    log_p_high = -np.inf
    if 0 <= source_k_star < K:
        log_p_high = log_delta_slice_1d[source_k_star] + log_prob_hi

    # --- 2. Low-Probability Path ---
    # Find the best path from *all other* sources
    max_log_delta_low = -np.inf
    best_k_low = 0  # Default argmax

    for k in range(K):
        if k == source_k_star:
            continue

        if log_delta_slice_1d[k] > max_log_delta_low:
            max_log_delta_low = log_delta_slice_1d[k]
            best_k_low = k

    log_p_low = max_log_delta_low + log_prob_lo

    # --- 3. Compare and return ---
    if log_p_high > log_p_low:
        return log_p_high, source_k_star
    else:
        return log_p_low, best_k_low


# --- NEW: Vectorized O(K) Helper ---

def _get_special_max_and_argmax_vec(
        log_delta_slice_1d: np.ndarray,
        K: int,
        dest_idx: int,
        delta: int,
        log_prob_hi: float,
        log_prob_lo: float
) -> Tuple[float, int]:
    """
    Computes max_k [ log_delta(k) + log P(k -> dest_idx) ] in O(K) time
    using vectorized numpy operations.
    """

    # --- 1. High-Probability Path ---
    source_k_star = dest_idx + delta

    log_p_high = -np.inf
    if 0 <= source_k_star < K:
        log_p_high = log_delta_slice_1d[source_k_star] + log_prob_hi

    # --- 2. Low-Probability Path ---

    # Compute all low-prob paths at once
    log_p_low_paths = log_delta_slice_1d + log_prob_lo

    # Mask out the special path so it's not chosen
    if 0 <= source_k_star < K:
        log_p_low_paths[source_k_star] = -np.inf

    # Find the best of the low-prob paths
    best_k_low = np.argmax(log_p_low_paths)
    log_p_low = log_p_low_paths[best_k_low]

    # --- 3. Compare and return ---
    if log_p_high > log_p_low:
        return log_p_high, source_k_star
    else:
        return log_p_low, best_k_low

def viterbi_optimized_K5(
        log_emissions: np.ndarray,
        Z_r: np.ndarray,
        log_pi: np.ndarray,
        eps_ru: float,
        eps_uv: float,
        eps_uw: float
) -> Tuple[List[Tuple[int, int, int]], float]:
    """
    Performs the optimized O(M*K^5) Viterbi algorithm.

    This replaces the K^6 loop with three factored K^4 loops,
    each using an O(K) helper, resulting in O(K^5) per time step.

    Args:
        log_emissions (M, K, K): log P(Xv_t, Xw_t | Zv_t=p, Zw_t=o)
        Z_r (M): Known sequence
        log_pi (K, K, K): Initial state log-probabilities log P(l, p, o)
        eps_ru, eps_uv, eps_uw: Epsilon for each link

    Returns:
        best_path: A list of M tuples, where each tuple is the (l,p,o) state.
        max_log_prob: The log-probability of the best path.
    """
    M, K, _ = log_emissions.shape
    N = K ** 3

    # --- 1. Get Log Probs for Transitions ---
    log_probs_ru = _get_log_probs(eps_ru, K)
    log_probs_uv = _get_log_probs(eps_uv, K)
    log_probs_uw = _get_log_probs(eps_uw, K)

    # --- 2. Initialization (t=0) ---
    log_delta = np.full((M, K, K, K), -np.inf)
    psi = np.zeros((M, K, K, K), dtype=np.int32)

    log_emissions_t0 = log_emissions[0, None, :, :]
    log_delta[0] = log_pi + log_emissions_t0

    # --- 3. Recursion (t=1 to M-1) ---

    # Intermediate tables to store max-products and backpointers
    S1_max = np.full((K, K, K, K), -np.inf)  # (k, m, l, o)
    S1_psi = np.full((K, K, K, K), 0, dtype=np.int32)  # (k, m, l, o) -> stores n*

    S2_max = np.full((K, K, K, K), -np.inf)  # (k, l, p, o)
    S2_psi = np.full((K, K, K, K), 0, dtype=np.int32)  # (k, l, p, o) -> stores m*

    for t in range(1, M):
        Delta_r_t = Z_r[t - 1] - Z_r[t]
        log_delta_tm1 = log_delta[t - 1]  # (k, m, n)

        # 1. Innermost "max" (over n) - O(K^5)
        # S1[k,m,l,o] = max_n [ log_delta(k,m,n) + log P_uw(n->o | k-l) ]
        for k, m, l, o in itertools.product(range(K), repeat=4):
            log_slice_n = log_delta_tm1[k, m, :]  # 1D array [n]
            Delta_u = k - l

            max_n, argmax_n = _get_special_max_and_argmax_vec(
                log_slice_n, K, o, Delta_u, log_probs_uw[0], log_probs_uw[1]
            )
            S1_max[k, m, l, o] = max_n
            S1_psi[k, m, l, o] = argmax_n

        # 2. Middle "max" (over m) - O(K^5)
        # S2[k,l,p,o] = max_m [ S1_max(k,m,l,o) + log P_uv(m->p | k-l) ]
        for k, l, p, o in itertools.product(range(K), repeat=4):
            log_slice_m = S1_max[k, :, l, o]  # 1D array [m]
            Delta_u = k - l

            max_m, argmax_m = _get_special_max_and_argmax_vec(
                log_slice_m, K, p, Delta_u, log_probs_uv[0], log_probs_uv[1]
            )
            S2_max[k, l, p, o] = max_m
            S2_psi[k, l, p, o] = argmax_m

        # 3. Outer "max" (over k) - O(K^4)
        # log_delta[t,l,p,o] = log_emiss + max_k [ S2_max(k,l,p,o) + log P_ru(k->l | Dr_t) ]
        for l, p, o in itertools.product(range(K), repeat=3):
            log_slice_k = S2_max[:, l, p, o]  # 1D array [k]

            max_k, argmax_k = _get_special_max_and_argmax_vec(
                log_slice_k, K, l, Delta_r_t, log_probs_ru[0], log_probs_ru[1]
            )

            # Store final delta
            log_delta[t, l, p, o] = max_k + log_emissions[t, p, o]

            # --- Store final backpointer ---
            # We found the best k* (argmax_k)
            # Now we need to look up the m* and n* that led to it
            m_star = S2_psi[argmax_k, l, p, o]
            n_star = S1_psi[argmax_k, m_star, l, o]

            # Store the flat index of the best *previous* state (k*, m*, n*)
            psi[t, l, p, o] = _flatten_state(argmax_k, m_star, n_star, K)

    # --- 4. Termination ---
    max_log_prob = np.max(log_delta[M - 1])
    best_last_state_flat = np.argmax(log_delta[M - 1].reshape(N))

    # --- 5. Backtracking (now simple, same as K^6 version) ---
    best_path_flat = np.zeros(M, dtype=np.int32)
    best_path_flat[M - 1] = best_last_state_flat

    for t in range(M - 2, -1, -1):
        current_state_flat = best_path_flat[t + 1]
        l, p, o = _unflatten_state(current_state_flat, K)

        # Look up the flat index of the best state at time t
        prev_state_flat = psi[t + 1, l, p, o]
        best_path_flat[t] = prev_state_flat

    best_path = np.array([_unflatten_state(idx, K) for idx in best_path_flat])
    best_path_ordered = best_path[:, [1, 2, 0]]  # reorder to (uv, uw, ru)
    return best_path_ordered, max_log_prob

