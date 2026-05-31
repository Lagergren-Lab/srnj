"""
Orienting-leaf selection strategies for sparse_rnj.

Provides factory functions that produce selector callables compatible with the
``ort_selector`` parameter of :func:`sparsernj.sparse_rnj`.

Typical usage::

    from sparsernj import sparse_rnj, FixedDistanceProvider
    from sparsernj.ort_selection import selection_matrices_from_cn, matrix_selection_strategy

    mats = selection_matrices_from_cn(C, A, observed_cn, healthy_cn)
    selector = matrix_selection_strategy(mats["hamming"])
    tree = sparse_rnj(provider, ort_selector=selector)
"""
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from .distance_provider import get_lca_from_pairwise
    from .utils.algorithms.neighbor_joining import dlca_lm
except ImportError:
    from sparsernj.distance_provider import get_lca_from_pairwise
    from sparsernj.utils.algorithms.neighbor_joining import dlca_lm


# ── Direction helpers ─────────────────────────────────────────────────────────

def _cherry_to_dir(cherry: list) -> int:
    """Map a 3-taxon NJ cherry to insertion direction: 0=A-side, 1=B-side, 2=root."""
    if 2 in cherry:
        return cherry[0] if cherry[1] == 2 else cherry[1]
    return 2  # ortA and ortB joined → taxon goes toward root


def _lm_direction(pointers: list, C: np.ndarray, A: np.ndarray) -> int:
    """Run dlca_lm on [ortA, ortB, taxon] from full C/A (integer-indexed)."""
    idx = np.array(pointers)
    cherry = dlca_lm(C[np.ix_(idx, idx)], A[np.ix_(idx, idx)])[0, :2].astype(int).tolist()
    return _cherry_to_dir(cherry)


# ── Selector factory ──────────────────────────────────────────────────────────

def matrix_selection_strategy(
    selection_matrix: np.ndarray,
    oracle_matrix: Optional[np.ndarray] = None,
    stats: Optional[dict] = None,
    distance_matrices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Callable:
    """Return an orienting-leaf selector that picks leaves by argmin of a precomputed matrix.

    The selector resolves ``ort_selector`` (passed through from ``sparse_rnj``) by
    looking up scores from ``selection_matrix[taxon, leaf]`` rather than calling the
    distance provider.  This is the mechanism used by the cheap-proxy strategies
    (Hamming, NLL, SCONCE2) to avoid paying the full distance-estimation cost during
    orientation.

    Parameters
    ----------
    selection_matrix : ndarray, shape (n_taxa, n_taxa)
        Pre-computed per-pair scores.  The leaf with the **lowest** score relative to
        the new taxon is chosen.  For ``max_lca``-style selection, pass ``-C`` so that
        argmin gives argmax of LCA depth.
    oracle_matrix : ndarray, optional
        Ground-truth selection matrix (e.g. the ``gt_min`` matrix).  When provided
        alongside ``stats``, each selection decision is compared against the oracle and
        accuracy counters are incremented.
    stats : dict, optional
        Mutable dict with keys ``total_decisions``, ``correct_A``, ``correct_B``,
        ``correct_pair``.  If the dict also contains ``"correct_direction"`` and
        ``distance_matrices`` is supplied, the 3-way direction accuracy is tracked too.
    distance_matrices : tuple (C, A), optional
        Ground-truth LCA and asymmetric distance matrices (integer-indexed, shape n×n).
        Required for direction accuracy tracking.  Taxa labels must be integer indices
        into these matrices.

    Returns
    -------
    Callable
        A selector with signature ``(leavesA, leavesB, taxon, dist_matrix) -> (ortA, ortB)``.
    """
    C_mat = A_mat = None
    if distance_matrices is not None:
        C_mat, A_mat = distance_matrices

    def _select(leavesA, leavesB, taxon, dist_matrix) -> tuple:
        leavesA = sorted(list(leavesA))
        leavesB = sorted(list(leavesB))
        dA = selection_matrix[taxon, leavesA]
        dB = selection_matrix[taxon, leavesB]
        min_dA = np.min(dA)
        min_dB = np.min(dB)
        # break ties deterministically by picking the smallest taxon label
        cand_A = [leaf for leaf, d in zip(leavesA, dA) if d == min_dA]
        cand_B = [leaf for leaf, d in zip(leavesB, dB) if d == min_dB]
        ortA = min(cand_A)
        ortB = min(cand_B)

        if oracle_matrix is not None and stats is not None:
            ref_A = oracle_matrix[taxon, leavesA]
            ref_B = oracle_matrix[taxon, leavesB]
            ref_min_A = np.min(ref_A)
            ref_min_B = np.min(ref_B)
            ref_cand_A = {leaf for leaf, d in zip(leavesA, ref_A) if d == ref_min_A}
            ref_cand_B = {leaf for leaf, d in zip(leavesB, ref_B) if d == ref_min_B}
            a_ok = ortA in ref_cand_A
            b_ok = ortB in ref_cand_B
            stats["total_decisions"] += 1
            stats["correct_A"] += int(a_ok)
            stats["correct_B"] += int(b_ok)
            stats["correct_pair"] += int(a_ok and b_ok)

            if "correct_direction" in stats and C_mat is not None:
                ref_ortA = min(ref_cand_A)
                ref_ortB = min(ref_cand_B)
                oracle_dir = _lm_direction([ref_ortA, ref_ortB, taxon], C_mat, A_mat)
                strat_dir  = _lm_direction([ortA, ortB, taxon], C_mat, A_mat)
                stats["correct_direction"] += int(strat_dir == oracle_dir)

        return ortA, ortB

    return _select


# ── Selection matrix helpers ──────────────────────────────────────────────────

def _pairwise_hamming(states: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between integer copy-number profiles."""
    return (states[:, None, :] != states[None, :, :]).sum(axis=2).astype(float)


def _hamming_to_root(observed: np.ndarray, healthy: np.ndarray) -> np.ndarray:
    """Per-taxon Hamming distance to the healthy consensus profile (pseudo root distance)."""
    ref = np.round(healthy.mean(axis=0))
    return (observed != ref[None, :]).sum(axis=1).astype(float)


def _fit_gaussian_baseline(healthy: np.ndarray, eps: float = 1e-3):
    """Fit per-bin mean and std from healthy (normal) cells."""
    mu = healthy.mean(axis=0)
    sigma = np.maximum(healthy.std(axis=0), eps)
    return mu, sigma


def _gaussian_nll_profiles(states: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (states - mu) / sigma
    return 0.5 * (z ** 2 + np.log(2.0 * np.pi * sigma ** 2))


def _pairwise_mean_abs_diff(profiles: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(profiles[:, None, :] - profiles[None, :, :]), axis=2)


def _max_lca_matrix(D: np.ndarray, root_dist: np.ndarray) -> np.ndarray:
    """Convert pairwise D + root distances to -C (so argmin gives argmax LCA depth)."""
    C, _ = get_lca_from_pairwise(np.asarray(D, dtype=float), np.asarray(root_dist, dtype=float))
    return -C


def selection_matrices_from_cn(
    C: np.ndarray,
    A: np.ndarray,
    observed: np.ndarray,
    healthy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build selection matrices for all built-in cheap-proxy strategies.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        Ground-truth LCA distance matrix.
    A : ndarray, shape (n, n)
        Ground-truth asymmetric (leaf-to-LCA) distance matrix.
    observed : ndarray, shape (n, n_bins)
        Observed copy-number profiles for the n tumor taxa.
    healthy : ndarray, shape (m, n_bins)
        Copy-number profiles for m normal (healthy) cells used to fit baselines.

    Returns
    -------
    dict mapping strategy name to selection matrix (n × n ndarray).

    Strategy names
    --------------
    ``gt_min``
        Ground-truth leaf-to-leaf matrix ``A + A^T`` (oracle argmin).
    ``gt_max_lca``
        Ground-truth ``-C`` (oracle argmax LCA depth, stored negated so argmin works).
    ``hamming``
        Pairwise Hamming distance between observed copy profiles.
    ``hamming_max_lca``
        Pseudo-LCA depths derived from Hamming distances via ``get_lca_from_pairwise``.
    ``nll``
        Pairwise mean-absolute-difference of per-bin Gaussian NLL profiles.
    ``nll_max_lca``
        Pseudo-LCA depths derived from NLL distances.
    """
    if healthy.shape[0] == 0:
        raise ValueError("Need at least one healthy cell to fit baselines.")
    if healthy.shape[1] != observed.shape[1]:
        raise ValueError(
            f"Healthy baseline has {healthy.shape[1]} bins but observed has {observed.shape[1]}."
        )

    hamming = _pairwise_hamming(observed)
    hamming_root = _hamming_to_root(observed, healthy)

    mu, sigma = _fit_gaussian_baseline(healthy)
    nll_profiles = _gaussian_nll_profiles(observed, mu, sigma)
    nll_root = nll_profiles.mean(axis=1)
    nll = _pairwise_mean_abs_diff(nll_profiles)

    return {
        "gt_min": A + A.T,
        "gt_max_lca": -np.asarray(C, dtype=float),
        "hamming": hamming,
        "hamming_max_lca": _max_lca_matrix(hamming, hamming_root),
        "nll": nll,
        "nll_max_lca": _max_lca_matrix(nll, nll_root),
    }
