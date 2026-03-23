"""
Sparse Neighbor-Joining (SNJ) algorithm for unrooted phylogenetic tree reconstruction.

This module implements a simplified, memory-efficient version of the Neighbor-Joining
algorithm that uses lazy distance matrix computation. The tree is unrooted and supports
incremental taxon insertion.
"""

from typing import Optional, List
import random
import networkx as nx
import numpy as np

from . import distance_provider as snj
from . import treenode
try:
    from ..utils.algorithms.neighbor_joining import std_nj_lm, lm_to_tree
except ImportError:
    # Fallback for direct script execution
    from utils.algorithms.neighbor_joining import std_nj_lm, lm_to_tree


def select_ort_leaves(leavesA, leavesB, leavesC, taxon, dp: snj.DistanceProvider) -> tuple[str, str, str]:
    nA = len(leavesA)
    nB = len(leavesB)
    D = dp.get_dist_matrix(leavesA + leavesB + leavesC + [taxon])
    ortA_idx = np.argmin(D[-1, :nA])
    ortB_idx = np.argmin(D[-1, nA:nA + nB])
    ortC_idx = np.argmin(D[-1, nA + nB:-1])
    assert ortA_idx < nA
    assert ortB_idx < nB
    assert ortC_idx < len(leavesC)
    ortA, ortB, ortC = leavesA[ortA_idx], leavesB[ortB_idx], leavesC[ortC_idx]
    return ortA, ortB, ortC


def insert_taxon_into_tree(tree: treenode.UTree, taxon, dp: snj.DistanceProvider, k: int = 1) -> treenode.UTree:
    """
    Insert taxon into the tree at the appropriate position. Returns the updated tree.
    K is the number of orienting leaves to consider on each side of the centroid, for which to compute the LCA distance matrix.
    """
    # check that taxon is not already in the tree
    assert taxon not in tree.taxa_map, f"Taxon {taxon} is already in the tree"
    centroid = tree.get_centroid()  # necessary to update tips/leaves
    (leavesA, leavesB, leavesC), ort_nodes = tree.pick_ABC_leaves(k=k) # tuple of lists for both direction, and tuple with 3 nodes (left, right, toward root)
    ortA, ortB, ortC = select_ort_leaves(leavesA, leavesB, leavesC, taxon, dp)
    pointers = [ortA, ortB, ortC, taxon]
    # construct quadruplet and find taxon neighbor
    lm = std_nj_lm(dp.get_dist_matrix(pointers))
    # check if taxon (idx=2 in the dm) is in the only cherry or not
    taxon_idx = 3
    cherry = lm[0, :2].astype(int).tolist()
    # pick edge (neighbor of centroid) as direction to insert
    if cherry[0] == taxon_idx:
        direction_node = ort_nodes[cherry[1]]
    elif cherry[1] == taxon_idx:
        direction_node = ort_nodes[cherry[0]]
    else:
        missing_idx = next(i for i in range(3) if i not in cherry)
        direction_node = ort_nodes[missing_idx]

    if direction_node.is_tip or direction_node.barrier:
        tree.insert_taxon_between(taxon, centroid, direction_node)
        return tree
    else:
        tree.mark_centroid_barrier_from(direction_node)
        return insert_taxon_into_tree(tree, taxon, dp, k=k)


def _relabel_initial_tree(tree, taxa):
    N = len(taxa)
    if any(isinstance(t, int) for t in taxa):
        # relabel internal nodes to avoid conflict with leaf labels
        tree = nx.relabel_nodes(tree, {i: f"ancestor_{i}" for i in range(N, 2 * N)})
    tree = nx.relabel_nodes(tree, {i: taxa[i] for i in range(len(taxa))})
    return tree


def sparse_nj(
    dist_provider: snj.DistanceProvider,
    initial_taxa: Optional[List] = None,
    insertion_order: Optional[List] = None,
    root_label: Optional[str] = None,
    rooting: str = 'outgroup',
    k: Optional[int] = None
) -> nx.DiGraph:
    """
    Build a tree using Sparse Neighbor-Joining.

    Uses lazy distance computation for efficiency with large datasets.

    Parameters:
        dist_provider: Object with get_dist_matrix(taxa) -> distance matrix
        initial_taxa: Initial set of taxa to build the tree (if None, randomly picks sqrt(n log n))
        insertion_order: Order to insert remaining taxa (if None, random)
        root_label: if provided, will rename the root node
        rooting: 'outgroup' (default) ['midpoint', 'none'] - how to root the tree
        k: Number of orienting leaves for each insertion (default log n)
    Returns:
        NetworkX Graph representing the unrooted tree
    """
    if rooting != 'outgroup':
        raise NotImplementedError("Currently only 'outgroup' rooting is supported. Midpoint rooting and no rooting options are not implemented yet.")
    if root_label is None:
        root_label = 'root'
    all_taxa = dist_provider.taxa + [root_label]
    n = len(all_taxa)
    # print(f"Building tree with {n} taxa (including root)")
    if k is None:
        # set k to log n as default which keeps the overall complexity at O(n log n)
        k = max(1, round(np.log2(n)))
    if initial_taxa is None:
        num_initial = max(4, round(np.sqrt(n * np.log2(n))))
        initial_taxa = random.sample(all_taxa, num_initial)
    if insertion_order is None:
        insertion_order = [t for t in all_taxa if t not in initial_taxa]
        random.shuffle(insertion_order)
    else:
        # remove any taxa from insertion_order that are in initial_taxa
        insertion_order = [t for t in insertion_order if t not in initial_taxa]
    # print(f"Initial taxa: {initial_taxa}")
    # print(f"Insertion order: {insertion_order}")

    # Build initial tree from first taxa
    dm = dist_provider.get_dist_matrix(initial_taxa)
    tree_nx = lm_to_tree(std_nj_lm(dm), edge_attr='weight', rooted=False)
    tree_nx = _relabel_initial_tree(tree_nx, initial_taxa)

    # print('Initial tree built with taxa:', initial_taxa)
    # print(nx.write_network_text(tree_nx))
    tree = treenode.UTree.from_networkx(tree_nx)

    # Insert remaining taxa
    for taxon in insertion_order:
        # print(f"Inserting taxon: {taxon}")
        tree = insert_taxon_into_tree(tree, taxon, dist_provider, k=k)
        tree.expand()  # reset barriers and update tips/leaves

    root_node = tree.get_node_by_taxon(root_label) if root_label in tree.taxa_map else None
    return tree.to_networkx(root_node=root_node)

