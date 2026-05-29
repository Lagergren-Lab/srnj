import random
import networkx as nx

import numpy as np

try:
    from ..utils.algorithms.neighbor_joining import dlca_nj, dlca_lm
except ImportError:
    # Fallback for direct script execution
    from utils.algorithms.neighbor_joining import dlca_nj, dlca_lm
from . import treenode
from . import distance_provider


def select_ort_leaves(leavesA, leavesB, taxon, dist_matrix, ort_strategy):
    lca_dm_, adm_ = dist_matrix.get_dms(leavesA + leavesB + [taxon])
    if ort_strategy == 'max_lca':
        ortA, ortB = _max_lca_selection(lca_dm_, leavesA, leavesB)
    elif ort_strategy == 'min_D':
        ortA, ortB = _min_D_selection(adm_, leavesA, leavesB)
    else:
        raise ValueError(f"Unknown strategy: {ort_strategy}")
    return ortA, ortB

def _min_D_selection(adm_, leavesA, leavesB):
    # get D by summing A + A.T
    # this approach is based on the idea that leaves that are
    # closer to each other give a more accurate estimate of the LCA distance (same approach as SNJ by Kurt et al. 2024)
    # D is the leaf to leaf distance and here it is computed by summing the distances
    # from leaves to the LCA in both directions
    nA = len(leavesA)
    D = adm_ + adm_.T
    ortA_idx = np.argmin(D[-1, :nA])
    ortB_idx = np.argmin(D[-1, nA:-1])
    assert ortA_idx < nA
    assert ortB_idx < len(leavesB)
    ortA, ortB = leavesA[ortA_idx], leavesB[ortB_idx]
    return ortA, ortB

def _max_lca_selection(lca_dm_, leavesA, leavesB):
    # pick argmax of lca_dm_[-1, :len(ort_leaves[0])] and lca_dm_[-1, len(ort_leaves[0]):-1]
    nA = len(leavesA)  # can be smaller than k if not enough leaves on one side
    ortA_idx = np.argmax(lca_dm_[-1, :nA])
    ortB_idx = np.argmax(lca_dm_[-1, nA:-1])
    assert ortA_idx < nA
    assert ortB_idx < len(leavesB)
    ortA, ortB = leavesA[ortA_idx], leavesB[ortB_idx]
    return ortA, ortB

def insert_taxon_into_tree(tree: treenode.Tree, taxon, dp: distance_provider.DistanceProvider, k: int = 1, ort_strategy ='min_D') -> treenode.Tree:
    """
    Insert taxon into the tree at the appropriate position. Returns the updated tree.
    K is the number of orienting leaves to consider on each side of the centroid, for which to compute the LCA distance matrix.
    """
    centroid = tree.get_centroid()  # necessary to update tips/leaves
    (leavesA, leavesB), ort_nodes = tree.pick_AB_leaves(k=k) # tuple of lists for both direction, and tuple with 3 nodes (left, right, toward root)
    ortA, ortB = select_ort_leaves(leavesA, leavesB, taxon, dp, ort_strategy)
    pointers = [ortA, ortB, taxon]
    # subset lca_dm_ and adm_ to only these two leaves + taxon
    lca_dm_, adm_ = dp.get_dms(pointers)
    linkage_matrix = dlca_lm(lca_dm_, adm_)  # with 3 leaves + root
    # check if taxon (idx=2 in the dm) is in the only cherry or not
    taxon_idx = 2
    cherry = linkage_matrix[0, :2].astype(int).tolist()
    # pick edge (neighbor of centroid) as direction to insert
    if cherry[0] == taxon_idx:
        direction_node = ort_nodes[cherry[1]]

    elif cherry[1] == taxon_idx:
        direction_node = ort_nodes[cherry[0]]
    else:
        direction_node = ort_nodes[2] # the root direction

    if direction_node.is_tip or direction_node.barrier:
        tree.insert_taxon_between(taxon, centroid, direction_node)
        return tree
    else:
        tree.mark_centroid_barrier_from(direction_node)
        return insert_taxon_into_tree(tree, taxon, dp, k=k)


def pick_initial_taxa(dist_matrix, num_initial):
    # randomly pick without replacement from dist_matrix.taxa
    initial_taxa = random.sample(dist_matrix.taxa, num_initial)
    return initial_taxa


def sparse_rnj(dist_matrix: distance_provider.DistanceProvider, initial_taxa: list[int] = None,
               insertion_order: list[int] = None, root_label=None, k: int | None = None, ort_strategy: str = 'min_D') -> nx.DiGraph:
    """
    Sparse Rooted Neighbor-Joining algorithm to build a rooted tree from a lazy distance matrix.
    k is the number of orienting leaves to consider on each side of the centroid, for which to compute the LCA distance matrix.
    """
    n = len(dist_matrix.taxa)
    if k is None:
        # set k to log n as default which keeps the overall complexity at O(n log n)
        k = round(np.log2(n))
    # init tree
    if initial_taxa is None:
        # randomly select sqrt(n log2 n) initial taxa
        initial_taxa = pick_initial_taxa(dist_matrix, num_initial=round(np.sqrt(n * np.log2(n))))
        insertion_order = [t for t in dist_matrix.taxa if t not in initial_taxa]
        random.shuffle(insertion_order)
    if insertion_order is None:
        # random insertion order of remaining taxa
        insertion_order = [t for t in dist_matrix.taxa if t not in initial_taxa]
        random.shuffle(insertion_order)
    # FIXME: maybe check that insertion order is consistent with initial taxa (i.e. no overlap)
    if root_label is None:
        root_label = 'root'
    lca_dm_, adm_ = dist_matrix.get_dms(initial_taxa)
    nxtree = dlca_nj(lca_dm_, adm_, collapsed_root=False, taxa=initial_taxa, root_label=root_label)  # nx.DiGraph
    tree = treenode.Tree.from_networkx(nxtree)
    # iteratively insert remaining taxa
    for taxon in insertion_order:
        tree = insert_taxon_into_tree(tree, taxon, dist_matrix, k=k, ort_strategy=ort_strategy)
        tree.expand()  # reset barriers and update tips/leaves

    nx_tree = tree.to_networkx()
    return nx_tree
