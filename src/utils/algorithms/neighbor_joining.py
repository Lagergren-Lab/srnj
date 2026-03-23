import logging
import operator
import os
import random
import subprocess

import networkx as nx
from dendropy import Tree
from numba import njit
import numpy as np
from skbio import DistanceMatrix, nj
from skbio.tree._c_nj import nj_minq_cy

try:
    from ..tree_utils import label_tree, convert_networkx_to_dendropy
except ImportError:
    # Fallback for direct script execution
    from utils.tree_utils import label_tree, convert_networkx_to_dendropy


def _build_tree_rec(ctr: dict, ntc: dict, ntr: dict, otus: set, edges: set[tuple], idx=None, collapsed_root=True) -> set[tuple]:
    if len(otus) == 2 and collapsed_root:
        for c in otus:
            # add edge with length
            # edges.add(('r', c, ntr[c]))
            edges.add(('r', c, ctr[frozenset(otus)]))
    elif len(otus) < 2:
        # add root to single OTU edge
        edges.add(('r', list(otus)[0], ntr[list(otus)[0]]))
    else:
        vw, l = max(ctr.items(), key=operator.itemgetter(1))
        # remove pair and add common ancestor with averaged distance

        # Computing edge _lengths after merge
        # save node-to-root distance for edge computation later
        # removing the pair from the centroid to rood distances as they are merged
        v, w = vw

        # remove node-to-root for merged nodes
        ntr.pop(v)
        ntr.pop(w)

        # Update distances merging vw in one OTU
        vsw = v + '_' + w  # node with string showing merges v_w
        vsw = str(idx) if idx is not None else vsw  # node with int showing merges
        ntr[vsw] = ctr.pop(vw)  # save centroid to root as the new node-to-root distance (new OTU)
        new_otus = otus.difference({w, v})
        for c in new_otus:
            vc = frozenset({v, c})
            wc = frozenset({w, c})
            # new pairwise distances
            vsw_c = frozenset({vsw, c})
            # update ctr distance for new node
            ctr[vsw_c] = .5 * (ctr[vc] + ctr[wc])
            # update ntc distances for new node
            ntc[vsw, c] = ntr[vsw] - ctr[vsw_c]
            ntc[c, vsw] = .5 * (ntc[c, v] + ntc[c, w])
            # remove already merged nodes
            ctr.pop(vc)
            ctr.pop(wc)
            ntc.pop(c, v)
            ntc.pop(c, w)
            ntc.pop(v, c)
            ntc.pop(w, c)

        # add node/subtree as OTU
        new_otus.add(vsw)

        v_edge_length = ntc.pop((v, w))
        w_edge_length = ntc.pop((w, v))

        idx = None if idx is None else idx + 1
        edges = _build_tree_rec(ctr, ntc, ntr, new_otus, edges, idx, collapsed_root)
        # find edge with merged node and add subtrees
        for x, v_, l in edges:
            if v_ == vsw:
                # add edge with length checking for negative values
                if v_edge_length < 0:
                    v_edge_length = 0
                    logging.warning(f'negative edge length for {v} <- {vsw}')
                if w_edge_length < 0:
                    w_edge_length = 0
                    logging.warning(f'negative edge length for {w} <- {vsw}')
                edges = edges.union([(v_, v, v_edge_length), (v_, w, w_edge_length)])
                break

    return edges


def rooted_nj0(ctr_table: np.ndarray, edge_attr='length', internal_indexing=False, taxa=None, collapsed_root=True) -> nx.DiGraph:
    """
    Build a tree from a centroid-to-root distance table. The root of the tree is assumed to be the common progenitor of all OTUs,
    which means that the healthy state is excluded from the tree and intended to be an additional node connected to the root.
    The input table is a 3D numpy array where the first two dimensions represent pairs of operational taxonomic units (OTUs),
    and the third dimension contains three values: the centroid-to-centroid distance, the node-to-centroid distance for the first OTU,
    and the node-to-centroid distance for the second OTU.
    Parameters:
        ctr_table (np.ndarray): A 3D numpy array with shape (N, N, 3) representing the centroid-to-root distance table.
        edge_attr (str): The attribute name to use for edge lengths in the resulting tree.
        internal_indexing (bool): If True, internal nodes are indexed with integers starting from N.
        taxa (list): A list of the taxa names to use in the resulting tree.
        collapsed_root (bool): If True, collapses the root edges by re-rooting at the child of the root and adjusting edge lengths.
    Returns:
        nx.DiGraph: A rooted tree represented as a directed graph with edge lengths.
    """
    # operational taxonomic units, OTUs, init with cells
    otus = set(map(str, range(ctr_table.shape[0])))
    # at each iteration, contains the centroid to root distance for each pair of OTUs
    # OTU is a set of cells (frozenset) which consist of a non-modifiable subtree
    ctr = {}
    # node-to-centroid distances for each OTU (initially single-cells) wrt to each other (index order is important here
    # as opposed to ctr that is symmetric)
    ntc = {}  # dict (str,str) -> float
    # node-to-root distances for each OTU as average of node-to-centroid distances over all other OTUs
    ntr = {str(v): 0 for v in range(len(otus))}  # dict str -> float
    for v in range(len(otus)):
        v_str = str(v)
        for w in range(v + 1, len(otus)):
            w_str = str(w)
            vsw = frozenset({v_str, w_str})
            # init ctr distances
            ctr[vsw] = ctr_table[v, w, 0]

            # compute node to centroid distance of v wrt w
            ntc[v_str, w_str] = ctr_table[v, w, 1]
            # compute node to centroid distance of w wrt v
            ntc[w_str, v_str] = ctr_table[v, w, 2]

            # compute node to root distance of v
            ntr[v_str] += ntc[v_str, w_str] + ctr_table[v, w, 1]
            # compute node to root distance of w
            ntr[w_str] += ntc[w_str, v_str] + ctr_table[v, w, 2]

    # normalize node-to-root distances to get the average
    ntr = {str(v): ntr[str(v)] / (len(otus) - 1) for v in range(len(otus))}

    # build tree only using ctr distances
    idx = len(otus) if internal_indexing else None  # index for new internal nodes
    edges = _build_tree_rec(ctr, ntc, ntr, otus, set(), idx, collapsed_root)
    nj_tree = nx.DiGraph()
    # add edges with _lengths
    nj_tree.add_weighted_edges_from(edges, weight=edge_attr)
    # add_lengths(em_tree, ctr_table)

    if taxa is not None:
        leaves_map = {str(i): taxa[i] for i in range(len(taxa))}
        nj_tree = nx.relabel_nodes(nj_tree, leaves_map)
    return nj_tree


@njit
def rooted_nj_minq(C, A, C_sums, A_sums, n):
    """
    Find the minimum value in the Q-matrix for rooted NJ.
    Q_{ij} = - C[i,j] + (n-2) * (A[i,j] + A[j,i]) - sum_k (C[i,k] + A[i,k]) - sum_k (C[j,k] + A[j,k])
    Args:
        C (np.ndarray): Current root-to-median distance matrix.
        A (np.ndarray): Current leaf-to-median distance matrix (asymmetric).
        C_sums (np.ndarray): Sum of root-to-median distances for each taxon.
        A_sums (np.ndarray): Sum of leaf-to-median distances for each taxon.
        n (int): Current number of taxa.
    Returns:
        tuple: Indices (i, j) of the maximum Q value.
    """
    min_q = 1e10
    min_ij = (-1, -1)
    for i in range(n):
        for j in range(i + 1, n):
            q_ij = -C[i, j] + (n - 2) * (A[i, j] + A[j, i]) - (C_sums[i] + A_sums[i]) - (C_sums[j] + A_sums[j])
            if q_ij < min_q:
                min_q = q_ij
                min_ij = (i, j)
    return min_ij


def split_tdm(triplet_distance_matrix) -> tuple[np.ndarray, np.ndarray]:
    # for i in range(N):
    #     lca_dm[i, i] = np.sum(adm[i, :] + lca_dm[i, :]) - lca_dm[i, i] - adm[i, i]
    #     lca_dm[i, i] /= N - 1
    C = triplet_distance_matrix[:, :, 0]  # root to median
    A = triplet_distance_matrix[:, :, 1].copy()
    for i in range(triplet_distance_matrix.shape[0]):
        for j in range(i+1, triplet_distance_matrix.shape[1]):
            A[j, i] = triplet_distance_matrix[i, j, 2]  # leaf to median
        C[i, i] = np.sum(A[i, :] + C[i, :]) - C[i, i] - A[i, i]  # node to root distance as in Gronau and Moran, 2007
        C[i, i] /= triplet_distance_matrix.shape[0] - 1
    return C, A


def rooted_nj_lm(triplet_distance_matrix, validate_input=False):
    """
    Compute the linkage matrix for a rooted tree using a modified Neighbor-Joining algorithm
    The formula used to compute the Q-matrix is adapted to account for the triplet distances:
    If l_h(u,v) is the distance from the root to the median of u and v, l_v(u,v) is the distance from leaf v to the median,
    then the Q-matrix is computed as:
    $$
    Q_{ij} = - l_h(i,j) + (n-2) * (l_v(i,j) + l_v(j,i)) - sum_k (l_h(i,k) + l_v(i,k)) - sum_k (l_h(j,k) + l_v(j,k))
    $$
    The linkage matrix is then constructed based on the maximum values in the Q-matrix.
    The linkage matrix has the following format:
    [[idx1, idx2, length1, length2],
     [idx3, idx4, length3, length4],
     ...
    ]
    where idx1 and idx2 are the indices of the merged nodes, and length1 and length2 are the lengths of the edges from the new internal node to the merged nodes.
    The implementation works in-place on the input distance matrix for memory efficiency.
    """
    # validate input
    if validate_input:
        assert triplet_distance_matrix is not None
        assert len(triplet_distance_matrix.shape) == 3 and triplet_distance_matrix.shape[2] == 3, \
            "Input triplet distance matrix must be a 3D numpy array with shape (N, N, 3)."
        # all three matrices should be symmetric with zeros on the diagonal
        for k in range(3):
            assert np.allclose(triplet_distance_matrix[:, :, k], triplet_distance_matrix[:, :, k].T), \
                f"All slices of the triplet distance matrix must be symmetric. (slice {k} is not)"
            assert np.allclose(np.diag(triplet_distance_matrix[:, :, k]), 0), \
                f"All slices of the triplet distance matrix must have zeros on the diagonal. (slice {k} has non-zero diagonal)"

    N = n = triplet_distance_matrix.shape[0]  # dimension
    # sum of leaf distances (N, 2) - from root to median and from median to leaf
    C, A = split_tdm(triplet_distance_matrix)
    C_sums = C.sum(axis=1)  # sum of root to median distances
    A_sums = A.sum(axis=1)  # sum of leaf to median distances
    idxs = np.arange(N)  # cluster indices
    lm = np.empty((N - 1, 4))  # linkage matrix

    # Iteratively merge taxa until there are two left
    while n > 2:
        # Create memory views of currently relevant array areas.
        C_ = C[:n, :n]
        A_ = A[:n, :n]
        C_sums_ = C_sums[:n]
        A_sums_ = A_sums[:n]
        idxs_ = idxs[:n]

        # Find the minimum value of the Q-matrix and return its position (i, j).
        # Q_{ij} = - C[i,j] + (n-2) * (A[i,j] + A[j,i]) - sum_k (C[i,k] + A[i,k]) - sum_k (C[j,k] + A[j,k])
        i, j = rooted_nj_minq(C_, A_, C_sums_, A_sums_, n)

        # Compute branch lengths of taxa i and j - just the leaf distances to the median.
        L_i = A[i, j]
        L_j = A[j, i]
        # Index i will hold the new cluster, so we update distances at row/column i.
        C_[i] += C_[j]
        C_[i] /= 2
        A_[i, :] += A_[j, :] - (L_i + L_j)  # leaf to median distances adjusted for new cluster
        A_[i, :] /= 2
        A_[:, i] += A_[:, j]  # median to k leaf is just average of that with i and j
        A_[:, i] /= 2
        # Now remove row/column j by replacing it with the last row/column (n-1).
        n_1 = n - 1
        C_[j] = C_[n_1]
        C_[:, j] = C_[:, n_1]
        A_[j, :] = A_[n_1, :]
        A_[:, j] = A_[:, n_1]
        # Also move the last sum to j.
        C_sums_[j] = C_sums_[n_1]
        A_sums_[j] = A_sums_[n_1]
        # Then calculate the updated sum at i (now cluster {i, j})
        C_sums_[i] = C_[i, :n_1].sum()
        A_sums_[i] = A_[i, :n_1].sum()
        # Store the taxa and branch lengths to the linkage matrix.
        lm[N - n] = idxs_[i], idxs_[j], L_i, L_j
        # Update cluster indices. Specifically, position i will have the new cluster
        # index. Meanwhile, position j will be replaced with the last cluster.
        idxs_[i] = 2 * N - n
        idxs_[j] = idxs_[n_1]
        n -= 1
    # Perform final calculation on the two remaining taxa. They will become children
    # of the root node, and the entire tree is rooted.
    L_i = A[0, 1] + C[0, 1]
    L_j = A[1, 0] + C[0, 1]
    lm[-1] = idxs[0], idxs[1], L_i, L_j
    return lm

def lm_to_tree(lm, edge_attr, rooted=True) -> nx.DiGraph | nx.Graph:
    """
    Convert a linkage matrix to a tree represented as a directed graph. The root is the last internal node created.
    Args:
        lm (np.ndarray): Linkage matrix with shape (N-1, 4)
        edge_attr (str): The attribute name to use for edge lengths in the resulting tree.
    Returns:
        nx.DiGraph: A tree represented as a directed graph with edge lengths with N leaves and one root.
    """
    tree = nx.DiGraph() if rooted else nx.Graph()
    idx = lm.shape[0] + 1  # starting index for internal nodes
    for k in range(lm.shape[0] - 1):
        parent = idx
        child1 = int(lm[k, 0])
        child2 = int(lm[k, 1])
        length1 = lm[k, 2]
        length2 = lm[k, 3]
        tree.add_edge(parent, child1, **{edge_attr: length1})
        tree.add_edge(parent, child2, **{edge_attr: length2})
        idx += 1

    if rooted:
        # handle last row separately to add root node
        tree.add_edge(idx, int(lm[-1, 0]), **{edge_attr: lm[-1, 2]})
        tree.add_edge(idx, int(lm[-1, 1]), **{edge_attr: lm[-1, 3]})
    else:
        # connect children with sum of their edge lengths to root (unrooted tree)
        tree.add_edge(int(lm[-1, 0]), int(lm[-1, 1]), **{edge_attr: lm[-1, 2] + lm[-1, 3]})

    return tree


def lm_to_rooted_tree(lm, edge_attr, root) -> nx.DiGraph:
    """
    Convert a linkage matrix to a rooted tree represented as a directed graph.
    The last row of the linkage matrix corresponds to a fictitious root node which
    will be collapsed and the tree re-rooted at the specified root node.
    Args:
        lm (np.ndarray): Linkage matrix.
        edge_attr (str): The attribute name to use for edge lengths in the resulting tree.
        root (int): The node index to use as the root of the tree.
    Returns:
        nx.DiGraph: A rooted tree represented as a directed graph with edge lengths.
    """
    tree = nx.Graph()
    idx = root + 1  # starting index for internal nodes if root was part of leaves (outgroup)
    if root != lm.shape[0]:
        idx = lm.shape[0]  # root is not part of leaves, therefore the internal nodes start at N
    for k in range(lm.shape[0] - 1):
        parent = idx
        child1 = int(lm[k, 0])
        child2 = int(lm[k, 1])
        length1 = lm[k, 2]
        length2 = lm[k, 3]
        tree.add_edge(parent, child1, **{edge_attr: length1})
        tree.add_edge(parent, child2, **{edge_attr: length2})
        idx += 1
    # handle last row separately
    tree.add_edge(int(lm[-1, 0]), int(lm[-1, 1]), **{edge_attr: lm[-1, 2]})
    # re-root the tree at the specified root node and collapse root edges, keeping lengths
    tree = reroot_preserving_weights(tree, root)
    return tree


def reroot_preserving_weights(G: nx.Graph, root):
    """Return a directed tree rooted at `root` while preserving edge weights."""
    T = nx.DiGraph()
    for parent, child in nx.bfs_edges(G, root):
        # Copy all edge attributes (not just weight)
        T.add_edge(parent, child, **G[parent][child])
    return T


def rooted_nj(triplet_distance_matrix, edge_attr='weight', taxa=None, inplace: bool = False) -> nx.DiGraph:
    """
    Modified Neighbor-Joining algorithm to build a rooted tree from a triplet distance matrix.
    The formula used to compute the Q-matrix is adapted to account for the triplet distances.
    Args:
        triplet_distance_matrix (np.ndarray): A 3D numpy array where the first two dimensions represent pairs of OTUs,
            and the third dimension contains three values: the median distance to the root and the two leaves distances to the median.
        edge_attr (str): The attribute name to use for edge lengths in the resulting tree.
        taxa (list of str, optional): Names corresponding to leaf nodes. If None, indices are used.
        inplace (bool): If True, modifies the input distance matrix in place. Default is False.
    Returns:
        nx.DiGraph: A rooted tree represented as a directed graph with edge lengths.
    """
    if not inplace:
        triplet_distance_matrix = np.copy(triplet_distance_matrix)
    lm = rooted_nj_lm(triplet_distance_matrix)  # linkage matrix with root already included
    tree = lm_to_tree(lm, edge_attr)
    if taxa is not None:
        # relabel leaves
        tree = nx.relabel_nodes(tree, {i: taxa[i] for i in range(len(taxa))})
    return tree


def std_nj_root(distance_matrix: np.ndarray, root_dist: np.ndarray, edge_attr='weight', taxa=None,
                collapsed_root=True, inplace=False, root_label=None) -> nx.DiGraph:
    """
    Standard Neighbor-Joining algorithm to build an rooted tree from a distance matrix and distances to the root.
    Args:
        distance_matrix (np.ndarray): A 2D numpy array representing the pairwise distances between OTUs.
        root_dist (np.ndarray): An array of distances from each OTU to the root.
        edge_attr (str): The attribute name to use for edge lengths in the resulting tree.
        taxa (list of str, optional): Names corresponding to leaf nodes. If None, indices are used.
        collapsed_root (bool): If True, collapses the root edges by re-rooting at the child of the root.
        inplace (bool): If True, modifies the input distance matrix in place. Default is False.
        root_label (str, optional): Label for the root node. If None, the root will be labeled with its index.
    Returns:
        nx.DiGraph: An unrooted tree represented as a directed graph with edge lengths.
    """
    root_idx = distance_matrix.shape[0]
    if not inplace:
        distance_matrix = distance_matrix.copy()
        root_dist = root_dist.copy()
    distance_matrix = extend_dm(distance_matrix, root_dist)
    lm = std_nj_lm(distance_matrix)  # standard nj on leaves + root
    tree = lm_to_rooted_tree(lm, edge_attr=edge_attr, root=root_idx) # convert to rooted tree at root_idx
    if collapsed_root:
        # re-root the tree at the child of the root to collapse root edges and adjust edge lengths
        tree = collapse_root(tree, root=root_idx, edge_attr=edge_attr)
    if taxa is not None:
        # relabel leaves
        tree = nx.relabel_nodes(tree, {i: taxa[i] for i in range(len(taxa))})
    if root_label is not None:
        tree = nx.relabel_nodes(tree, {root_idx: root_label})
    return tree


def collapse_root(tree, root, edge_attr):
    r"""
    Collapse the root of the tree (when root is separated - leaf node) by re-rooting at its child and adjusting edge lengths.
    E.g. (ascii art):
                 root
                   | lr
                new_root
                /    \ l2
            child1  child2
    becomes:
            new_root
            /    \  lr+l2
        child1  child2

    Args:
        tree (nx.DiGraph): The input tree with a separated root.
        root (int): The index of the root node to collapse.
        edge_attr (str): The attribute name for edge lengths.
    Returns:
        nx.DiGraph: The tree re-rooted at the child of the original root with adjusted edge lengths.
    """
    new_root = next(tree.successors(root))
    rl = tree.edges[root, new_root][edge_attr]
    # remove root and re-root at new_root
    tree.remove_edge(root, new_root)
    tree.remove_node(root)
    # adjust lengths of edges from new_root to its children
    for child in tree.successors(new_root):
        tree.edges[new_root, child][edge_attr] += rl
    return tree


def extend_dm(distance_matrix, root_dist):
    # extend distance matrix to include root distances
    N = distance_matrix.shape[0]
    extended_dm = np.zeros((N + 1, N + 1))
    extended_dm[:N, :N] = distance_matrix
    extended_dm[N, :N] = root_dist
    extended_dm[:N, N] = root_dist
    distance_matrix = extended_dm
    return distance_matrix


def std_nj_lm(dm: np.ndarray) -> np.ndarray:
    r"""Perform neighbor joining (NJ) for phylogenetic reconstruction.

    Parameters
    ----------
    dm : (N, N) ndarray
        Input distance matrix.

    Returns
    -------
    ndarray of shape (N - 1, 4)
        Linkage matrix representing the tree.

    Notes
    -----
    This function manipulates the distance matrix in-place. Therefore, one should make
    a copy prior to running the function if the original distance matrix needs to be
    preserved.

    """

    # This function re-uses the original array space during iteration, without creating
    # additional intermediate arrays. Therefore, it is memory-efficient, and avoids the
    # time overhead of allocating new memory space.

    # This function only operates on arrays of numbers, therefore it can be further
    # Cythonized. However, Cythonization did not bring significant performance gain in
    # tests, likely because all operations already utilize NumPy APIs. That being said,
    # further optimization and testing should be convenient.

    N = n = dm.shape[0]  # dimension
    sums = dm.sum(axis=0)  # distance sums
    idxs = np.arange(N)  # cluster indices
    lm = np.empty((N - 1, 4))  # linkage matrix

    # Iteratively merge taxa until there are three left.
    while n > 3:
        # Create memory views of currently relevant array areas.
        dm_ = dm[:n, :n]
        sums_ = sums[:n]
        idxs_ = idxs[:n]

        # Find the minimum value of the Q-matrix and return its position (i, j).
        #   Q(i, j) = (n - 2) d(i, j) - \sum d(i) - \sum d(j)
        # The function call avoids constructing the entire Q-matrix, but instead
        # computes values and finds the minimum as the computation goes.
        i, j = nj_minq_cy(dm_, sums_)

        # Get half of the original distance at (i, j).
        d_ij_ = dm[i, j] / 2

        # Taxa i and j will be merged into a cluster {i, j}. The updated distance from
        # cluster to any other taxon k is:
        #   d({i, j}, k) = (d(i, k) + d(j, k) - d(i, j)) / 2
        # We first compute (d(i, k) + d(j, k)) / 2 and save the results in row i.
        dm_[i] += dm_[j]
        dm_[i] /= 2  # continues later to complete d({i, j}, k) ... (*)

        # Compute branch lengths of taxa i and j.
        #   \delta = (\sum d(i) - \sum d(j)) / (2(n - 2))
        #   L(i) = d(i, j) / 2 + \delta
        #   L(j) = d(i, j) / 2 - \delta
        delta_ = (sums_[i] - sums_[j]) / (2 * n - 4)
        L_i = d_ij_ + delta_
        L_j = d_ij_ - delta_

        # The previously calculated sums can be updated for re-use. Specifically, for
        # taxon k, there is:
        #   new sum = old sum - d(i, k) - d(j, k) + d({i,j}, k)
        #           = old sum - d(i, k) - d(j, k) + (d(i, k) + d(j, k) - d(i, j)) / 2
        #           = old sum - (d(i, k) + d(j, k)) / 2 - d(i, j) / 2
        # We already have (d(i, k) + d(j, k)) / 2 stored in row i, therefore:
        sums_[:] -= dm_[i]
        sums_[:] -= d_ij_

        # ... (*) Now complete the calculation of the updated distances d({i, j}, k).
        dm_[i] -= d_ij_

        # Update column i to match row i.
        dm_[:, i] = dm_[i]

        # Because two taxa have been merged into one cluster, we will shrink the
        # distance matrix from (n, n) to (n - 1, n - 1). Specifically, we will move
        # the last row/column (index: n - 1) to row/column j.
        n_1 = n - 1
        dm_[j] = dm_[n_1]
        dm_[:, j] = dm_[:, n_1]

        # Also move the last sum to j.
        sums_[j] = sums_[n_1]

        # Then calculate the updated sum at i (now cluster {i, j}), which is the sum
        # of the updated distances.
        sums_[i] = dm_[i, :n_1].sum()

        # Store the taxa and branch lengths to the linkage matrix.
        lm[N - n] = idxs_[i], idxs_[j], L_i, L_j

        # Update cluster indices. Specifically, position i will have the new cluster
        # index. Meanwhile, position j will be replaced with the last cluster.
        idxs_[i] = 2 * N - n  # new cluster index
        idxs_[j] = idxs_[n_1]  # replace with last cluster

        n -= 1

    # Perform final calculation on the three remaining taxa. They will become children
    # of the root node, and the entire tree is unrooted.
    L_0 = (dm[0, 1] + dm[0, 2] - dm[1, 2]) / 2
    lm[N - 3] = idxs[1], idxs[2], dm[0, 1] - L_0, dm[0, 2] - L_0
    lm[N - 2] = idxs[0], 2 * N - 3, L_0, 0

    return lm

def _pick_cherry(adm: np.ndarray) -> tuple[int, int]:
    """
    Algorithm 1 from Zhai et al. 2016 to pick a cherry from an asymmetric distance matrix.
    """
    q = 1
    i, j = np.unravel_index(adm.argmin(), adm.shape)  # true cherry under exact conditions
    while q < adm.shape[0]:
        if adm[j, i] == adm[j].min():
            break
        else:
            i = j
            j = np.argmin(adm[j])
        q += 1
    return i, j

def anj_lm(adm: np.ndarray) -> np.ndarray:
    """
    Asymmetric Neighbor-Joining algorithm to build a rooted tree from an asymmetric distance matrix. (Zhai et al. 2016)
    Args:
        adm (np.ndarray): A 2D numpy array representing the asymmetric pairwise distances between taxa. The distance
        d(i, j) represents the distance from taxon i to the LCA of i and j.
    Returns:
        np.ndarray: A linkage matrix representing the tree. Every row represents a merge of two nodes with the format:
        [idx1, idx2, length1, length2] where idx1 and idx2 are the indices of the merged nodes, and length1 and length2
        are the lengths of the edges from the new internal node to the merged nodes.
    """
    adm = adm.copy()  # work on a copy to avoid modifying input
    # set diagonal to large value to avoid picking self-loops
    np.fill_diagonal(adm, 1e10)
    N = n = adm.shape[0]
    idxs = np.arange(N)  # cluster indices
    lm = np.empty((N - 1, 4))  # linkage matrix
    while n > 2:
        adm_ = adm[:n, :n]
        # pick cherry (reduction)
        i, j = _pick_cherry(adm_)
        # compute lengths (estimation)
        L_i = adm[i, j]
        L_j = adm[j, i]
        # update distances (update)
        adm_[i, :] += adm_[j, :] - (L_i + L_j)
        adm_[i, :] /= 2
        adm_[:, i] += adm_[:, j]
        adm_[:, i] /= 2
        # remove j
        n_1 = n - 1
        adm_[j] = adm_[n_1]
        adm_[:, j] = adm_[:, n_1]
        # store lengths
        lm[N - n] = idxs[i], idxs[j], L_i, L_j
        # update indices
        idxs[i] = 2 * N - n
        idxs[j] = idxs[n_1]
        n -= 1
    # final merge
    L_i = adm[0, 1]
    L_j = adm[1, 0]
    lm[-1] = idxs[0], idxs[1], L_i, L_j
    return lm

def anj(adm: np.ndarray, edge_attr='weight', root_idx=None, collapsed_root=True, taxa=None) -> nx.DiGraph:
    N = adm.shape[0]
    lm = anj_lm(adm)
    tree = lm_to_tree(lm, edge_attr=edge_attr)
    if root_idx is not None:
        # join the last row into a single edge
        tot_l = lm[-1, 2] + lm[-1, 3]
        last_node_idx = 2 * N - 2
        tree.add_edge(int(lm[-1, 0]), int(lm[-1, 1]), **{edge_attr: tot_l})
        tree.remove_edges_from([(last_node_idx, int(lm[-1, 0])), (last_node_idx, int(lm[-1, 1]))])
        tree.remove_node(last_node_idx)
        # reroot at specified root
        tree = tree.to_undirected()
        tree = reroot_preserving_weights(tree, root_idx)
        # join last row
        if collapsed_root:
            # re-root the tree at the child of the root to collapse root edges and adjust edge lengths
            tree = collapse_root(tree, root=root_idx, edge_attr=edge_attr)
    # otherwise, the tree has an implicit root at the last internal node
    # ===========================
    if taxa is not None:
        # relabel leaves
        tree = nx.relabel_nodes(tree, {i: taxa[i] for i in range(len(taxa))})
    return tree

def _pick_cherry_max(lca: np.ndarray) -> tuple[int, int]:
    """
    Neighbor selection step for DLCA (analogous to _pick_cherry for ANJ) based on maximal entries.
    """
    q = 1
    lca_copy = lca.copy()
    np.fill_diagonal(lca_copy, -1e10)  # search only off-diagonal entries
    i, j = np.unravel_index(lca_copy.argmax(), lca_copy.shape)  # true cherry under exact conditions
    while q < lca_copy.shape[0]:
        if lca_copy[j, i] == lca_copy[j].max():
            break
        else:
            i = j
            j = np.argmax(lca_copy[j])
        q += 1
    return i, j

def dlca_lm(lca_dm: np.ndarray, adm: np.ndarray | None = None) -> np.ndarray:
    """
    Deepest Least Common Ancestor (DLCA) algorithm to build a rooted tree from a LCA distance matrix (Gronau and Moran 2007).
    Implements the mid-point reduction (alpha=0.5) version of the algorithm.
    If asymmetric distances are provided, they are used to estimate edge lengths more accurately.
    Otherwise, lengths will use the LCA diagonal entries as distances from leaves to root.
    Args:
        lca_dm (np.ndarray): A 2D numpy array of shape (N, N) representing the pairwise LCA distances between taxa
        adm (np.ndarray, optional): A 2D numpy array representing the asymmetric pairwise distances between taxa.
    Returns:
        np.ndarray: linkage matrix of shape (N, 4) representing the tree. Every row represents a merge of two nodes
        and the last row represents the final merge to keep track of root distance.
    """
    lca_dm = lca_dm.copy()
    adm = adm.copy() if adm is not None else None
    N = n = lca_dm.shape[0]
    idxs = np.arange(N)  # cluster indices
    lm = np.empty((N, 4))  # linkage matrix
    if adm is not None:
        # add diagonal values to lca
        for i in range(N):
            lca_dm[i, i] = np.sum(adm[i, :] + lca_dm[i, :]) - lca_dm[i, i] - adm[i, i]
            lca_dm[i, i] /= N - 1
    while n > 1:
        lca_dm_ = lca_dm[:n, :n]
        # pick cherry (reduction) (i, j) maximal off-diagonal entry in row i,j
        i, j = _pick_cherry_max(lca_dm_)
        n_1 = n - 1
        if adm is not None:
            # compute lengths via asymmetric distances
            adm_ = adm[:n, :n]
            L_i = adm_[i, j]
            L_j = adm_[j, i]
            # update adm distances
            adm_[i, :] += adm_[j, :] - (L_i + L_j)
            adm_[i, :] /= 2
            adm_[:, i] += adm_[:, j]
            adm_[:, i] /= 2
            # remove j
            adm_[j, :] = adm_[n - 1, :]
            adm_[:, j] = adm_[:, n - 1]
        else:
            L_i = max(0, lca_dm_[i, i] - lca_dm_[i, j])
            L_j = max(0, lca_dm_[j, j] - lca_dm_[i, j])
        # update distances (update)
        LCA_ij = lca_dm_[i, j]  # save before overwriting
        lca_dm_[i, :] += lca_dm_[j]
        lca_dm_[i, :] /= 2  # average LCA i,j with all others
        lca_dm_[:, i] = lca_dm_[i]  # make symmetric
        lca_dm_[i, i] = LCA_ij  # update diagonal (distance from new cluster to root) as LCA(i,j)
        # remove j
        lca_dm_[j] = lca_dm_[n_1]
        lca_dm_[:, j] = lca_dm_[:, n_1]
        lca_dm_[j, j] = lca_dm_[n_1, n_1]
        # store lengths
        lm[N - n] = idxs[i], idxs[j], L_i, L_j
        # update indices
        idxs[i] = 2 * N - n
        idxs[j] = idxs[n_1]
        n -= 1
    # final merge to keep track of root distance
    # print("Final LCA DM:\n", lca_dm)
    lm[N - 1] = idxs[0], 2 * N - 1, lca_dm_[0, 0], 0
    return lm

def dlca_nj(lca_dm: np.ndarray, adm: np.ndarray | None = None,
            edge_attr='weight', collapsed_root=True, taxa=None, root_label=None) -> nx.DiGraph:
    N = lca_dm.shape[0]  # number of leaves
    lm = dlca_lm(lca_dm, adm)
    root_idx = 2 * N - 1
    tree = lm_to_rooted_tree(lm, edge_attr=edge_attr, root=root_idx)
    # nx.write_network_text(tree)
    if collapsed_root:
        # re-root the tree at the child of the root to collapse root edges and adjust edge lengths
        tree = collapse_root(tree, root=root_idx, edge_attr=edge_attr)
    if taxa is not None:
        # relabel leaves
        # if taxa are int, they should be range(len(taxa)), otherwise they might conflict with internal node labels. In that case, internal nodes must be labeled with artificial labels.
        if isinstance(taxa[0], int):
            # relabel internal nodes to avoid conflict with leaf labels
            tree = nx.relabel_nodes(tree, {i: f"ancestor_{i}" for i in range(N, 2 * N)})
            root_idx = "ancestor_" + str(root_idx)
        tree = nx.relabel_nodes(tree, {i: taxa[i] for i in range(len(taxa))})
    if root_label is not None:
        tree = nx.relabel_nodes(tree, {root_idx: root_label})
    return tree

def get_root_dist_from_tripledist(dist_matrix: np.ndarray, agg_func: str = 'mean') -> np.ndarray:
    match agg_func:
        case 'mean':
            return get_root_dist_from_tripledist_mean(dist_matrix)
        case 'max':
            return get_root_dist_from_tripledist_max(dist_matrix)
        case _:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")

def get_root_dist_from_tripledist_max(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate distances from each leaf to the root based on the triplet distance matrix.
    The input distance matrix is a 3D triplet distance matrix where for each pair (i, j), i < j:
    - dist_matrix[i, j, 0] is the distance from the root to the median of i and j
    - dist_matrix[i, j, 1] is the distance from leaf i to the median
    - dist_matrix[i, j, 2] is the distance from leaf j to the median
    The output is a 1D array where each element corresponds to the distance from the respective leaf to the root.
    NOTE: This function assumes that the input distance matrices are symmetric and only uses the upper triangle, i.e.
    if i > j, dist_matrix[i, j, 1] is actually the distance from leaf j to the median and dist_matrix[i, j, 2] is the distance from leaf i to the median.
    Args:
        dist_matrix (np.ndarray): A 3D numpy array representing the triplet distance matrix.
    Returns:
        np.ndarray: A 1D array of distances from each leaf to the root.
    """
    # max but use only upper triangle
    n = dist_matrix.shape[0]
    root_dist = np.zeros(n)
    cnt_check = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            root_dist[i] = max(root_dist[i], dist_matrix[i, j, 0] + dist_matrix[i, j, 1])
            root_dist[j] = max(root_dist[j], dist_matrix[i, j, 0] + dist_matrix[i, j, 2])
    return root_dist

def get_root_dist_from_tripledist_mean(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate distances from each leaf to the root based on the triplet distance matrix.
    The input distance matrix is a 3D triplet distance matrix where for each pair (i, j), i < j:
    - dist_matrix[i, j, 0] is the distance from the root to the median of i and j
    - dist_matrix[i, j, 1] is the distance from leaf i to the median
    - dist_matrix[i, j, 2] is the distance from leaf j to the median
    The output is a 1D array where each element corresponds to the distance from the respective leaf to the root.
    NOTE: This function assumes that the input distance matrices are symmetric and only uses the upper triangle, i.e.
    if i > j, dist_matrix[i, j, 1] is actually the distance from leaf j to the median and dist_matrix[i, j, 2] is the distance from leaf i to the median.
    Args:
        dist_matrix (np.ndarray): A 3D numpy array representing the triplet distance matrix.
    Returns:
        np.ndarray: A 1D array of distances from each leaf to the root.
    """
    # average but use only upper triangle
    n = dist_matrix.shape[0]
    root_dist = np.zeros(n)
    cnt_check = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            root_dist[i] += dist_matrix[i, j, 0] + dist_matrix[i, j, 1]
            root_dist[j] += dist_matrix[i, j, 0] + dist_matrix[i, j, 2]
            cnt_check[i] += 1
            cnt_check[j] += 1
    assert np.all(cnt_check == (n - 1)), "Count check failed in root dist calculation"
    root_dist /= (n - 1)
    return root_dist


def neighbor_joining(dist_matrix, taxon_namespace):
    ids = [str(i) for i in range(dist_matrix.shape[0])]
    # use luv + luw distances
    dm = DistanceMatrix(dist_matrix[:, :, 1] + dist_matrix[:, :, 2], ids)
    nj_tree = nj(dm).root_at_midpoint()
    # convert to dendropy tree
    dpy_tree = Tree.get(data=str(nj_tree), schema="newick", taxon_namespace=taxon_namespace)
    label_tree(dpy_tree, method='int')
    dpy_tree.is_rooted = True
    # print("NJ LEAVES ", [(l.taxon.label, l.label) for l in dpy_tree.leaf_nodes()])
    return dpy_tree


def save_distmatrix(dist_matrix, file_name):
    dist_matrix = dist_matrix[:, :, 1] + dist_matrix[:, :, 2]
    # clip very small values to the minimum representable float in 5 decimal places
    n = dist_matrix.shape[0]
    dist_matrix = np.clip(dist_matrix, 0.00001, None)
    max_char = len(str(n - 1)) + 1
    with open(file_name, 'w+') as f:
        f.write(str(n) + '\n')
        for i in range(n):
            cell = str(i)
            x = np.array2string(dist_matrix[i], formatter={'float_kind': lambda x: "%.5f" % x})[1:-1].replace('\n', '')
            sep_char = ' ' * (max_char - len(cell))
            f.write(cell + sep_char + x + '\n')


def fast_me(dist_matrix, taxon_namespace, suffix=""):
    # run balanced minimum evolution (fast ME)
    # timestamp to make unique file names
    if suffix == "":
        suffix = f'_{random.randint(0,1000000)}'
    file_name = f'.dist_mat{suffix}.PHYLIP'
    save_distmatrix(dist_matrix, file_name)
    tree_prefix = f'.tree{suffix}.nwk'
    call = subprocess.run(['fastme', '-i', file_name, '-o', tree_prefix, '-m', 'B', '-s'], capture_output=True,
                          text=True)
    # wait for process to finish
    if call.returncode != 0:
        print("FASTME ERROR: ", call.stderr)
        raise RuntimeError("FASTME failed")
    # open in dpy
    newick_str = ''
    with open(tree_prefix, 'r') as f:
        newick_str = f.read().strip()
    os.remove(tree_prefix)
    dpy_tree = Tree.get(data=newick_str, schema="newick", taxon_namespace=taxon_namespace)
    label_tree(dpy_tree, method='int')
    dpy_tree.is_rooted = True
    # print("FASTME LEAVES ", [(l.taxon.label, l.label) for l in dpy_tree.leaf_nodes()])
    # clean up
    os.remove(file_name)
    return dpy_tree


def cellmates_infer(dist_matrix, taxon_namespace):
    nx_rec_tree = rooted_nj0(dist_matrix)
    cellmates_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length')
    return cellmates_tree


def cellmates_rnj_infer(dist_matrix, taxon_namespace):
    nx_rec_tree = rooted_nj(dist_matrix, edge_attr='length')
    rnj_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length')
    return rnj_tree


def std_njx(dist_matrix, taxon_namespace, root_dist=None, collapsed_root=True):
    # standard neighbor-joining with correction for rooting
    # root_dist = np.sum(dist_matrix[:, :, 0] + dist_matrix[:, :, 1] + dist_matrix[:, :, 2], axis=0) / (dist_matrix.shape[0] - 1)
    if root_dist is None:
        root_dist = get_root_dist_from_tripledist(dist_matrix, agg_func='max')
    d1d2 = dist_matrix[:, :, 1] + dist_matrix[:, :, 2]
    nx_rec_tree = std_nj_root(d1d2, root_dist=root_dist, edge_attr='length',
                              taxa=[str(i) for i in range(dist_matrix.shape[0])], collapsed_root=collapsed_root)
    std_nj_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length',
                                               internal_nodes_label='int')
    return std_nj_tree


def anj_dpy(trip_dist, taxon_namespace, collapsed_root=True):
    n = trip_dist.shape[0]
    adm = np.zeros((n + 1, n + 1))
    C, A = split_tdm(trip_dist)
    adm[:n, :n] = A
    # add root distances to distance matrix (zero distance to any other node, and C + A from node to root)
    adm[n, :] = 0
    # vectorized accumulation of root distances for each tip (exclude diagonal)
    total = C + A
    row_sums = total.sum(axis=1) - np.diag(total)
    adm[:n, n] = row_sums / (n - 1)
    tree = anj(adm, root_idx=n, edge_attr='length', collapsed_root=collapsed_root)
    dpy_tree = convert_networkx_to_dendropy(tree, taxon_namespace=taxon_namespace, edge_length='length')
    return dpy_tree


def dlca_nj_dpy(dist_matrix, taxon_namespace, collapsed_root=True):
    C, A = split_tdm(dist_matrix)
    nx_dlca = dlca_nj(C, A, edge_attr='length', collapsed_root=collapsed_root)
    dpy_tree = convert_networkx_to_dendropy(nx_dlca, taxon_namespace=taxon_namespace, edge_length='length')
    return dpy_tree


def build_tree(dist_matrix, taxon_namespace, method='rnj0'):
    match method:
        case 'nj-mid':
            return neighbor_joining(dist_matrix, taxon_namespace)
        case 'bme':
            return fast_me(dist_matrix, taxon_namespace)
        case 'rnj0':
            return cellmates_infer(dist_matrix, taxon_namespace)
        case 'rnj1':
            return cellmates_rnj_infer(dist_matrix, taxon_namespace)
        case 'nj-root':
            return std_njx(dist_matrix, taxon_namespace)
        case 'anj':
            return anj_dpy(dist_matrix, taxon_namespace)
        case 'dlca-nj':
            return dlca_nj_dpy(dist_matrix, taxon_namespace)
        case _:
            raise ValueError(f"Unknown method: {method}")
