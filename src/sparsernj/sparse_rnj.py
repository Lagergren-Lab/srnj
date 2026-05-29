import random
from typing import Callable, Union

import networkx as nx
import numpy as np

try:
    from .utils.algorithms.neighbor_joining import dlca_nj, dlca_lm
except ImportError:
    # Fallback when running as a script with sys.path.insert(0, 'src')
    from sparsernj.utils.algorithms.neighbor_joining import dlca_nj, dlca_lm
from . import treenode
from . import distance_provider


# ── Built-in orienting-leaf selectors ────────────────────────────────────────
# Selector signature: (leavesA, leavesB, taxon, dist_matrix) -> (ortA, ortB)

def _min_D_selector(leavesA, leavesB, taxon, dist_matrix):
    # Select the leaf on each side with minimum leaf-to-leaf distance D = A + A^T
    # to the new taxon (same rationale as SNJ by Kurt et al. 2024).
    _, adm_ = dist_matrix.get_dms(leavesA + leavesB + [taxon])
    return _min_D_selection(adm_, leavesA, leavesB)


def _max_lca_selector(leavesA, leavesB, taxon, dist_matrix):
    # Select the leaf on each side with maximum LCA depth relative to the new taxon.
    lca_dm_, _ = dist_matrix.get_dms(leavesA + leavesB + [taxon])
    return _max_lca_selection(lca_dm_, leavesA, leavesB)


ORT_SELECTORS: dict[str, Callable] = {
    "min_D": _min_D_selector,
    "max_lca": _max_lca_selector,
}


def _resolve_selector(ort_selector: Union[str, Callable]) -> Callable:
    """Return a selector callable from a name or a user-provided callable."""
    if callable(ort_selector):
        return ort_selector
    if ort_selector not in ORT_SELECTORS:
        raise ValueError(
            f"Unknown ort_selector {ort_selector!r}. Valid built-in options: {list(ORT_SELECTORS)}"
        )
    return ORT_SELECTORS[ort_selector]


# Kept for backward compatibility (alva.py monkeypatches this function).
def select_ort_leaves(leavesA, leavesB, taxon, dist_matrix, ort_strategy):
    lca_dm_, adm_ = dist_matrix.get_dms(leavesA + leavesB + [taxon])
    if ort_strategy == 'max_lca':
        return _max_lca_selection(lca_dm_, leavesA, leavesB)
    elif ort_strategy == 'min_D':
        return _min_D_selection(adm_, leavesA, leavesB)
    else:
        raise ValueError(f"Unknown strategy: {ort_strategy}")


def _min_D_selection(adm_, leavesA, leavesB):
    nA = len(leavesA)
    D = adm_ + adm_.T
    ortA_idx = np.argmin(D[-1, :nA])
    ortB_idx = np.argmin(D[-1, nA:-1])
    assert ortA_idx < nA
    assert ortB_idx < len(leavesB)
    return leavesA[ortA_idx], leavesB[ortB_idx]


def _max_lca_selection(lca_dm_, leavesA, leavesB):
    nA = len(leavesA)
    ortA_idx = np.argmax(lca_dm_[-1, :nA])
    ortB_idx = np.argmax(lca_dm_[-1, nA:-1])
    assert ortA_idx < nA
    assert ortB_idx < len(leavesB)
    return leavesA[ortA_idx], leavesB[ortB_idx]


# ── Tree insertion ────────────────────────────────────────────────────────────

def insert_taxon_into_tree(
    tree: treenode.Tree,
    taxon,
    dp: distance_provider.DistanceProvider,
    k: int = 1,
    ort_selector: Union[str, Callable] = "min_D",
    ort_strategy: str = None,  # deprecated alias for ort_selector
) -> treenode.Tree:
    """Insert taxon into the tree at the appropriate position.

    k is the number of candidate orienting leaves per side of the centroid.
    ort_selector can be a built-in strategy name ('min_D' or 'max_lca') or any
    callable with signature (leavesA, leavesB, taxon, dist_matrix) -> (ortA, ortB).
    """
    if ort_strategy is not None:
        ort_selector = ort_strategy
    selector_fn = _resolve_selector(ort_selector)

    centroid = tree.get_centroid()
    (leavesA, leavesB), ort_nodes = tree.pick_AB_leaves(k=k)
    ortA, ortB = selector_fn(leavesA, leavesB, taxon, dp)
    pointers = [ortA, ortB, taxon]
    lca_dm_, adm_ = dp.get_dms(pointers)
    linkage_matrix = dlca_lm(lca_dm_, adm_)
    taxon_idx = 2
    cherry = linkage_matrix[0, :2].astype(int).tolist()
    if cherry[0] == taxon_idx:
        direction_node = ort_nodes[cherry[1]]
    elif cherry[1] == taxon_idx:
        direction_node = ort_nodes[cherry[0]]
    else:
        direction_node = ort_nodes[2]  # root direction

    if direction_node.is_tip or direction_node.barrier:
        tree.insert_taxon_between(taxon, centroid, direction_node)
        return tree
    else:
        tree.mark_centroid_barrier_from(direction_node)
        return insert_taxon_into_tree(tree, taxon, dp, k=k, ort_selector=ort_selector)


def pick_initial_taxa(dist_matrix, num_initial):
    return random.sample(dist_matrix.taxa, num_initial)


def sparse_rnj(
    dist_matrix: distance_provider.DistanceProvider,
    initial_taxa: list = None,
    insertion_order: list = None,
    root_label=None,
    k: int = None,
    ort_selector: Union[str, Callable] = "min_D",
    ort_strategy: str = None,  # deprecated alias for ort_selector
) -> nx.DiGraph:
    """Sparse Rooted Neighbor-Joining algorithm.

    Builds a rooted tree from a distance provider using O(n log n) distance calls.

    Parameters
    ----------
    dist_matrix : DistanceProvider
        Provides pairwise LCA and asymmetric distances.  Subclass DistanceProvider
        and implement _compute_triplet to plug in a custom estimator.
    k : int, optional
        Orienting leaves per side of the centroid (default: log2(n)).
    ort_selector : str or callable
        Leaf-selection strategy.  Built-in names: 'min_D' (default) or 'max_lca'.
        Pass a callable (leavesA, leavesB, taxon, dist_matrix) -> (ortA, ortB) for
        custom selection (see sparsernj.ort_selection.matrix_selection_strategy).
    """
    if ort_strategy is not None:
        ort_selector = ort_strategy

    n = len(dist_matrix.taxa)
    if k is None:
        k = round(np.log2(n))
    if initial_taxa is None:
        initial_taxa = pick_initial_taxa(dist_matrix, num_initial=round(np.sqrt(n * np.log2(n))))
        insertion_order = [t for t in dist_matrix.taxa if t not in initial_taxa]
        random.shuffle(insertion_order)
    if insertion_order is None:
        insertion_order = [t for t in dist_matrix.taxa if t not in initial_taxa]
        random.shuffle(insertion_order)
    if root_label is None:
        root_label = 'root'
    lca_dm_, adm_ = dist_matrix.get_dms(initial_taxa)
    nxtree = dlca_nj(lca_dm_, adm_, collapsed_root=False, taxa=initial_taxa, root_label=root_label)
    tree = treenode.Tree.from_networkx(nxtree)
    for taxon in insertion_order:
        tree = insert_taxon_into_tree(tree, taxon, dist_matrix, k=k, ort_selector=ort_selector)
        tree.expand()

    return tree.to_networkx()
