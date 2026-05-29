import logging
import math
import random
import subprocess
import re
from io import StringIO
from itertools import combinations

import dendropy as dpy
from dendropy.calculate import treecompare
import skbio
import numpy as np
import networkx as nx
import scipy.stats as ss
from dendropy.calculate.phylogeneticdistance import NodeDistanceMatrix
from sklearn.metrics import f1_score
from collections import defaultdict

from . import math_utils


def nxtree_to_newick(
    g: nx.DiGraph,
    root=None,
    weight=None,
    is_internal_call=False,
    drop_internal=False
):
    assert nx.is_arborescence(g)

    if root is None:
        roots = [n for n, d in g.in_degree() if d == 0]
        assert len(roots) == 1
        root = roots[0]

    subgs = []
    child_list = sorted(g[root]) if all(isinstance(n, int) for n in g[root]) else list(g[root])

    for child in child_list:
        if len(g[child]) > 0:
            node_str = nxtree_to_newick(
                g, root=child, weight=weight,
                is_internal_call=True, drop_internal=drop_internal
            )
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])

        subgs.append(node_str)

    label = "" if (drop_internal and is_internal_call) else str(root)
    newick = "(" + ",".join(subgs) + ")" + label

    if not is_internal_call:
        newick += ";"

    return newick


def convert_networkx_to_dendropy(nx_tree, labels_mapping: dict = None,
                                 taxon_namespace=None, edge_length=None, internal_nodes_label='group') -> dpy.Tree:
    """
    Converts a NetworkX tree to a DendroPy tree through newick string.

    Args:
      nx_tree: The NetworkX tree to convert (rooted).
      labels_mapping: dict, mapping of taxa labels to new labels

    Returns:
      A DendroPy rooted tree.
    """
    if labels_mapping is not None:
        nx_tree = nx.relabel_nodes(nx_tree, labels_mapping, copy=True)
    newick = "[&R] " + nxtree_to_newick(nx_tree, weight=edge_length)  # rooted tree
    dendropy_tree = dpy.Tree.get(data=newick, schema='newick', taxon_namespace=taxon_namespace)
    outgroup_node_label = dendropy_tree.seed_node.label if dendropy_tree.is_rooted else None
    label_tree(dendropy_tree, method=internal_nodes_label, outgroup_node_label=outgroup_node_label)

    return dendropy_tree

def convert_dendropy_to_networkx(dendropy_tree: dpy.Tree, edge_attr='weight', unifurcating_root=False, root_label='root') -> nx.DiGraph:
    """
    Converts a DendroPy tree to a NetworkX tree directly.

    Args:
      dendropy_tree: The DendroPy tree to convert.
      edge_attr: str, edge attribute in which to store the weights
      unifurcating_root: bool, whether to add a unifurcating root
      root_label: str, label for the unifurcating root node

    Returns:
      A NetworkX tree.
    """
    # Create directed graph
    nx_tree = nx.DiGraph()
    
    # Track node names to ensure uniqueness
    node_counter = 0
    node_map = {}
    
    # Helper function to get or create node name
    def get_node_name(dendropy_node):
        nonlocal node_counter
        if dendropy_node in node_map:
            return node_map[dendropy_node]
        
        # First, try to use existing label from DendroPy node
        if dendropy_node.label and dendropy_node.label != "None":
            node_name = dendropy_node.label
        # For leaves, use taxon label if available
        elif dendropy_node.is_leaf() and dendropy_node.taxon and dendropy_node.taxon.label:
            node_name = dendropy_node.taxon.label
        # Fall back to numeric counter for unlabeled nodes
        else:
            node_name = node_counter
            node_counter += 1
        
        node_map[dendropy_node] = node_name
        return node_name
    
    # Add all nodes and edges
    for node in dendropy_tree.preorder_node_iter():
        node_name = get_node_name(node)
        nx_tree.add_node(node_name)
        
        # Add edges to children
        for child in node.child_node_iter():
            child_name = get_node_name(child)
            nx_tree.add_edge(node_name, child_name)
            
            # Add edge weight if available
            if edge_attr and child.edge_length is not None:
                nx_tree[node_name][child_name][edge_attr] = child.edge_length
    
    # Handle unifurcating root if requested
    if unifurcating_root:
        # Find current root node (node with in_degree == 0)
        root_nodes = [n for n, d in nx_tree.in_degree() if d == 0]
        if len(root_nodes) == 1:
            current_root = root_nodes[0]
            # Get root branch length from original tree
            root_length = dendropy_tree.seed_node.edge_length if dendropy_tree.seed_node.edge_length is not None else 0.0
            # Add new unifurcating root
            nx_tree.add_edge(root_label, current_root)
            if edge_attr:
                nx_tree[root_label][current_root][edge_attr] = root_length
    
    return nx_tree


def random_binary_tree(n: int, length_mean: float = None, seed=None, full_length: float = None)-> dpy.Tree:
    """
    Generate a random binary tree with n leaves using Dendropy.
    ref: https://dendropy.org/primer/treesims.html
    Args:
        n: Number of leaves.
        seed: Random seed.
        length_mean: Mean branch length.
        full_length: If provided, scale the tree to have this total length, ignoring length_mean.

    Returns:
        A Dendropy tree.
    """
    # set dendropy and scipy seed for reproducibility
    if seed is not None:
        dpy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if full_length is not None and length_mean is not None:
        raise ValueError("Cannot specify both full_length and length_mean")
    # create taxon namespace
    tns = dpy.TaxonNamespace([dpy.Taxon(str(i)) for i in range(n)], label='taxa')
    # tree = dpy.treesim.treesim.pure_kingman_tree(taxon_namespace=tns)
    tree = dpy.treesim.treesim.uniform_pure_birth_tree(tns, birth_rate=1.0)
    tree.is_rooted = True
    label_tree(tree)
    # traverse the tree and assign _lengths
    if full_length is not None:
        tree = make_ultrametric(tree, target_height=full_length)
    else:
        for edge in tree.preorder_edge_iter():
            if full_length is not None:
                length_mean = edge.length # use current length as mean and add some noise
            # scale = 1 / lambda
            edge.length = ss.expon(scale=length_mean).rvs()
    return tree

def make_ultrametric(tree: dpy.Tree, target_height):
    # compute current leaf distances
    max_height = tree.max_distance_from_root()

    scale_factor = target_height / max_height

    # scale every edge
    for node in tree.postorder_node_iter():
        if node.edge.length is not None:
            node.edge.length *= scale_factor

    return tree

def get_node2node_distance(tree: dpy.Tree, node1_label: str, node2_label: str):
    tree.calc_node_root_distances()
    node1 = tree.find_node_with_label(node1_label)
    node2 = tree.find_node_with_label(node2_label)
    if node1.root_distance < node2.root_distance:
        node1, node2 = node2, node1
    return node1.root_distance - node2.root_distance


def label_tree(tree, method='int', outgroup_node_label=None):
    """
    Assigns int labels to tree nodes. Leaves are assigned with cell ids, internal nodes with decremental numbers
    different from cell ids.
    Parameters
    ----------
    tree: dpy.Tree, input tree
    method: str, 'int' or 'group'
        'int': internal nodes labeled with decremental integers
        'group': internal nodes labeled with grouped taxa in the subtree
    outgroup_node: dpy.Node, node to use as outgroup (therefore not relabeled)
    """
    if method == 'int':
        rev_node_idx = len(tree.nodes()) - 1
        # FIXME: when outgroup_node_label is set, rev_node_idx may conflict with the root node
        for n in tree.nodes():
            if outgroup_node_label is not None and n.label == outgroup_node_label:
                continue
            elif n.is_leaf():
                n.label = str(n.taxon.label)
            else:
                n.label = str(rev_node_idx)
                rev_node_idx -= 1
    elif method == 'group':
        for n in tree.postorder_node_iter():
            if outgroup_node_label is not None and n.label == outgroup_node_label:
                continue
            elif n.is_leaf():
                n.label = str(n.taxon.label)
            else:
                # group the taxa in the subtree in a sorted string
                taxa = []
                for c in n.child_node_iter():
                    for t in c.label.split('_'):
                        taxa.append(t)
                n.label = '_'.join(sorted(taxa, key=lambda x: int(x)))
    else:
        raise ValueError(f"Unknown method {method}")

def newick_to_nx(nwk_str, edge_attr='weight', interior_node_names=None, unifurcating_root=False, root_label='root') -> nx.DiGraph:
    """
    Convert newick string to NetworkX DiGraph using DendroPy.
    
    Parameters
    ----------
    nwk_str: str, newick string
    edge_attr: str, edge attribute in which to store the weights
    interior_node_names: list, optional list of names for internal nodes
    unifurcating_root: bool, whether to add a unifurcating root
    root_label: str, label for the unifurcating root node

    Returns
    -------
    nx.DiGraph tree with nodes and weighted edges
    """
    # Parse newick string using DendroPy without rooted parameter
    tree = dpy.Tree.get(data=nwk_str, schema='newick')
    
    # Set as rooted after parsing
    tree.is_rooted = True
    
    # Convert to NetworkX using existing conversion function
    tree_nx = convert_dendropy_to_networkx(tree, edge_attr=edge_attr, unifurcating_root=False, root_label='temp_root')
    
    # Handle interior node names if provided
    if interior_node_names is not None:
        # Find internal nodes (non-leaf nodes with out_degree > 0)
        internal_nodes = [n for n in tree_nx.nodes() if tree_nx.out_degree(n) > 0]
        
        # Sort internal nodes for consistent naming
        internal_nodes = sorted(internal_nodes, key=lambda x: (isinstance(x, int), x))
        
        # Create mapping for interior node names
        name_mapping = {}
        name_idx = 0
        for node in internal_nodes:
            if name_idx < len(interior_node_names):
                name_mapping[node] = interior_node_names[name_idx]
                name_idx += 1
        
        if name_mapping:
            tree_nx = nx.relabel_nodes(tree_nx, name_mapping, copy=True)
    
    # Handle unifurcating root if requested
    if unifurcating_root:
        # Find current root node (node with in_degree == 0)
        root_nodes = [n for n, d in tree_nx.in_degree() if d == 0]
        if len(root_nodes) == 1:
            current_root = root_nodes[0]
            # Get root branch length from original tree
            root_length = tree.seed_node.edge_length if tree.seed_node.edge_length is not None else 0.0
            # Add new unifurcating root
            tree_nx.add_edge(root_label, current_root)
            if edge_attr:
                tree_nx[root_label][current_root][edge_attr] = root_length
    
    return tree_nx

def write_cells_to_tree(nx_tree, cell_names) -> nx.DiGraph:
    # relabel leaf nodes with cell ids from adata
    # add ancestor nodes names with breadth-first search
    assert nx.is_arborescence(nx_tree)
    root_node = list(filter(lambda p: p[1] == 0, nx_tree.in_degree()))[0][0]
    mapping = {root_node: "root"}
    count = 1
    for u, v in nx.bfs_edges(nx_tree, source=root_node):
        if nx_tree.out_degree(v) == 0:
            mapping[v] = cell_names[int(v)] # leaf node
        else:
            mapping[v] = "ancestor" + str(count) # internal node
            count += 1

    nx.relabel_nodes(nx_tree, mapping, copy=False)
    # save and plot inferred tree
    return nx_tree

def make_gt_tree_dist(ad, n_states, cell_names: list) -> tuple[dpy.Tree, np.ndarray]:
    # TODO: this function can be rewritten to re-use the code from function (D, Dp) = utils.testing.get_expected_changes(cnps, tree_nx, cell_pairs)
    #   together with utils.testing.get_expected_distances(D, Dp, n_states, cell_pairs)
    # traverse the tree, write lengths to branches and, for each pair, sum lengths between them
    n_sites = ad.n_vars
    nxtree = newick_to_nx(ad.uns['cell-tree-newick'])
    nxtree = nx.dfs_tree(nxtree, source='founder')  # make sure it's rooted
    # get copy number (ancestors) at each node and compute the length based on changes
    ancestor_idx = {a: i for i,a in enumerate(ad.uns['ancestral-names'])}  # names as in the tree (index for ancestral-cn)
    ancestor_cn = ad.uns['ancestral-cn'] # shape (n_ancestors, n_bins)
    for u, v in nxtree.edges():
        if not v.startswith('cell'):
            i,j = ancestor_idx[u], ancestor_idx[v]
            p = math_utils.compute_cn_changes(np.vstack([ancestor_cn[i], ancestor_cn[j]]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        else:
            v_cn = ad[v].layers['state']
            u_cn = ancestor_cn[ancestor_idx[u]]
            p = math_utils.compute_cn_changes(np.vstack([u_cn, v_cn]), pairs=[(0, 1)])[0] / n_sites
            target_length = math_utils.l_from_p(p, n_states)
            nxtree[u][v]['length'] = target_length
        # print(f"Edge {u}->{v} p {p}, length {nxtree[u][v]['length']}")
    # relabel nodes to integers
    nxtree = relabel_name_to_int(nxtree, cell_names)
    dpy_tree = convert_networkx_to_dendropy(nxtree, edge_length='length', internal_nodes_label='int')
    # print("DPY tree with lengths:", dpy_tree.as_string(schema='newick'))
    dist_matrix = get_ctr_table_int(dpy_tree)
    return dpy_tree, dist_matrix

def relabel_name_to_int_mapping(nxtree: nx.DiGraph, cell_names: list, ancestors_mapping=None) -> tuple[nx.DiGraph, dict]:
    """
    Give integer labels to nodes in the tree. Cell names (leaves) are labeled from 0 to n-1 in the order of cell_names
    and ancestors are labeled from n to n+m-1 where m is the number of ancestors.
    Returns the relabeled tree and the mapping from original names to integer labels.
    Parameters
    ----------
    nxtree: nx.DiGraph, input tree
    cell_names: list, list of cell names (leaves)
    ancestors_mapping: dict, mapping of ancestor names to integer labels (optional)
    Returns
    -------
    relabeled_tree: nx.DiGraph, tree with integer labels
    full_mapping: dict, mapping from original names to integer labels
    """
    cells_mapping = {name: i for i, name in enumerate(cell_names)}
    if ancestors_mapping is None:
        ancestors = set(nxtree.nodes()) - set(cell_names)
        ancestors = sorted(list(ancestors))
        n_cells = len(cell_names)
        ancestors_mapping = {n: i + n_cells for i, n in enumerate(ancestors)}
    full_mapping = {**cells_mapping, **ancestors_mapping}
    # check that all cell names are in the tree and they are leaves
    for name in cell_names:
        assert name in nxtree.nodes(), f"Cell name {name} not in tree nodes"
        assert nxtree.out_degree(name) == 0, f"Cell name {name} is not a leaf node"
    return nx.relabel_nodes(nxtree, full_mapping, copy=True), full_mapping

def relabel_name_to_int(nxtree: nx.DiGraph, cell_names: list, ancestors_mapping=None) -> nx.DiGraph:
    """
    Give integer labels to nodes in the tree. Cell names (leaves) are labeled from 0 to n-1 in the order of cell_names
    and ancestors are labeled from n to n+m-1 where m is the number of ancestors.
    """
    relabeled_tree, _ = relabel_name_to_int_mapping(nxtree, cell_names, ancestors_mapping)
    return relabeled_tree

def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table_int(tree: dpy.Tree, full=False) -> np.ndarray:
    """
    Get the centroid table for a given tree where leaves are labeled with integers.
    The centroid table is a 3D numpy array of shape (n_cells, n_cells, 3) where n_cells is the number of leaves in the tree.
    For each pair of cells (r, s) with r < s, the entry ctr_table[r, s] is a vector of 3 values:
        - ctr_table[r, s, 0]: distance from the centroid of r and s to the root
        - ctr_table[r, s, 1]: distance from the centroid of r and s to r
        - ctr_table[r, s, 2]: distance from the centroid of r and s to s
    The entries for r >= s are set to -1.
    The tree must be rooted and all leaves must be labeled with unique integers from 0 to n_cells - 1.
    Parameters
    ----------
    tree: dpy.Tree, the input tree with edge _lengths
    full: bool, if True, returns the full symmetric matrix, otherwise only upper triangular part is filled

    Returns
    -------
    ctr_table: np.ndarray, the centroid table
    """
    # if leaves are not integers, raise error
    if not all(leaf.label.isdigit() for leaf in tree.leaf_nodes()):
        raise ValueError("Leaves must be labeled with integers to use get_ctr_table_int, use get_ctr_table instead")
    ctr_table, _ = get_ctr_table(tree, full=full)
    return ctr_table

def get_ctr_table(tree: dpy.Tree, full: bool = False) -> tuple[np.ndarray, list]:
    """
    Get the CTR table for a given tree where leaves are labeled with integers.
    The CTR table is a 2D numpy array of shape (n_cells, n_cells) where n_cells is the number of leaves in the tree.
    For each pair of cells (r, s), the entry ctr_table[r, s] is the distance between r and s in the tree.
    The tree must be rooted and all leaves must be labeled with unique integers from 0 to n_cells - 1.
    Parameters
    ----------
    tree: dpy.Tree, the input tree with edge _lengths
    full: bool, if True, returns the full symmetric matrix, otherwise only upper triangular part is filled
    Returns
    -------
    tuple[np.ndarray, list], the triplet distance table and the list of taxa labels as ordered in the table. Leaf to root
    distances are in ctr_table[i, i, 0] as required by the DLCA NJ algorithm.
    """
    taxa = tree.taxon_namespace
    n_cells = len(tree.leaf_nodes())
    # find root, check until parent is None
    root_node = [n for n in tree.nodes() if n.parent_node is None][0]

    ndm = NodeDistanceMatrix.from_tree(tree)
    ctr_table = - np.ones((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        r_taxa = taxa[r].label
        s_taxa = taxa[s].label
        r_node = tree.find_node_with_taxon_label(r_taxa)
        s_node = tree.find_node_with_taxon_label(s_taxa)

        ctr_table[r, s, 0] = ndm.distance(ndm.mrca(r_node, s_node), root_node)
        ctr_table[r, s, 1] = ndm.distance(r_node, ndm.mrca(r_node, s_node))
        ctr_table[r, s, 2] = ndm.distance(s_node, ndm.mrca(r_node, s_node))

        if full:
            ctr_table[s, r, :] = ctr_table[r, s, :]

    # fill diagonal with node-to-root distances
    for i in range(n_cells):
        ctr_table[i, i, :] = ndm.distance(tree.find_node_with_taxon_label(taxa[i].label), root_node)
    return ctr_table, [taxa[i].label for i in range(n_cells)]

def f1_score_clades(tree: dpy.Tree, clone_assignment: list) -> float:
    """
    Metrics defined in DICE: for each clone, find the clade that maximizes the F1 score
    Return the average F1 score across all clones.
    """
    # get all clades in the tree
    clades = []
    for n in tree.postorder_node_iter():
        if not n.is_leaf():
            clade = set()
            for leaf in n.leaf_iter():
                clade.add(int(leaf.label))
            clades.append(clade)

    # group cells by clone assignment
    clone_to_cells = defaultdict(set)
    for cell, clone in enumerate(clone_assignment):
        clone_to_cells[clone].add(cell)

    f1_scores = []
    for clone, cells in clone_to_cells.items():
        best_f1 = 0
        for clade in clades:
            y_true = [1 if i in cells else 0 for i in range(len(clone_assignment))]
            y_pred = [1 if i in clade else 0 for i in range(len(clone_assignment))]
            score = f1_score(y_true, y_pred)
            if score > best_f1:
                best_f1 = score
        f1_scores.append(best_f1)

    return np.mean(f1_scores).item()

def normalized_rf_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> float:
    """
    Compute the normalized Robinson-Foulds distance between two (rooted) trees using DendroPy.
    The trees must have the same set of leaf labels.
    The normalized RF distance is the RF distance divided by the maximum possible RF distance.
    """
    rf = treecompare.symmetric_difference(tree1, tree2)
    n_leaves = len(tree1.leaf_nodes())
    max_rf = 2 * (n_leaves - 2)
    return rf / max_rf


def get_lowest_common_ancestor(tree_nx, node1, node2):
    """
    Get the index of the least common ancestor of two nodes in a directed tree.
    """
    return nx.lowest_common_ancestor(tree_nx, node1, node2)


def convert_skbio_to_networkx(tree_nj_skbio: skbio.TreeNode, interior_node_names=None)-> nx.DiGraph:
    """
    Convert a skbio TreeNode to a networkx DiGraph.
    Parameters
    ----------
    tree_nj_skbio

    Returns tree_nx: nx.DiGraph
    -------

    """
    newick = str(tree_nj_skbio)
    tree_nx = newick_to_nx(newick, edge_attr='weight', interior_node_names=interior_node_names)
    return tree_nx




def relabel_dendropy(tree: dpy.Tree, leaves_mapping: dict):
    """
    Relabels the leaves of a DendroPy tree according to a given mapping.
    ----------
    tree: dendropy.Tree
    leaves_mapping: dict, mapping from old labels to new labels
    """
    for n in tree.leaf_node_iter():
        if n.label is None:
            n.label = leaves_mapping[n.taxon.label]
            n.taxon.label = n.label
        else:
            n.label = leaves_mapping[n.label]

def normalized_quartet_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> float:
    n = len(tree1.leaf_nodes())
    # if unifurcating root, consider root as a leaf
    if tree1.is_rooted and len(tree1.seed_node.child_nodes()) == 1:
        n += 1
    out_dist = quartet_distance(tree1, tree2)
    if out_dist is None:
        return None
    out_dist /=  math.comb(n, 4)
    return out_dist

def quartet_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> int:
    """
    Compute the quartet distance between two trees using quartet_dist binary from the tqDist package (Sand et al. 2014).
    If quartet_dist is not installed, returns None.
    NOTE: trees with unifurcating root are supported. The root will be considered as a leaf, therefore
    the quartet distance will be computed on (N+1 choose 4) quartets where N is the number of taxa.
    """
    n = len(tree1.leaf_nodes())
    assert n == len(tree2.leaf_nodes()), "Trees must have the same number of leaves"
    # write to temp newick files (with rnd prefix to avoid parallel executions clash) and run quartet_dist
    rnd_prefix = str(random.randint(0, 100000))
    tree1_file = f"/tmp/{rnd_prefix}_tree1.nwk"
    tree2_file = f"/tmp/{rnd_prefix}_tree2.nwk"
    with open(tree1_file, 'w') as f:
        f.write(tree1.as_string(schema='newick', suppress_rooting=True))
    with open(tree2_file, 'w') as f:
        f.write(tree2.as_string(schema='newick', suppress_rooting=True))
    # run quartet_dist
    result = subprocess.run(['quartet_dist', tree1_file, tree2_file], capture_output=True)
    if result.returncode != 0:
        logging.warning("quartet_dist failed to run, returning None for quartet distance\n" + result.stderr.decode())
        return None
    out_dist = int(result.stdout)
    # clean up temp files
    subprocess.run(['rm', tree1_file, tree2_file])
    return out_dist

def normalized_triplet_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> float:
    n = len(tree1.leaf_nodes())
    # if unifurcating root, consider root as a leaf
    if tree1.is_rooted and len(tree1.seed_node.child_nodes()) == 1:
        n += 1
    out_dist = triplet_distance(tree1, tree2)
    if out_dist is None:
        return None
    out_dist /=  math.comb(n, 3)
    return out_dist

def triplet_distance(tree1: dpy.Tree, tree2: dpy.Tree) -> int:
    """
    Compute the triplet distance between two trees using triplet_dist binary from the tqDist package (Sand et al. 2014).
    If tqDist is not installed, returns None.
    """
    n = len(tree1.leaf_nodes())
    assert n == len(tree2.leaf_nodes()), "Trees must have the same number of leaves"
    # write to temp newick files (with rnd prefix to avoid parallel executions clash) and run quartet_dist
    rnd_prefix = str(random.randint(0, 100000))
    tree1_file = f"/tmp/{rnd_prefix}_tree1.nwk"
    tree2_file = f"/tmp/{rnd_prefix}_tree2.nwk"
    with open(tree1_file, 'w') as f:
        f.write(tree1.as_string(schema='newick', suppress_rooting=True))
    with open(tree2_file, 'w') as f:
        f.write(tree2.as_string(schema='newick', suppress_rooting=True))
    # run quartet_dist
    result = subprocess.run(['triplet_dist', tree1_file, tree2_file], capture_output=True)
    if result.returncode != 0:
        logging.warning("quartet_dist failed to run, returning None for quartet distance\n" + result.stderr.decode())
        return None
    out_dist = int(result.stdout)
    # clean up temp files
    subprocess.run(['rm', tree1_file, tree2_file])
    return out_dist

def get_booster_transfer_indices(tree_ref: dpy.Tree, tree_est: dpy.Tree) -> dpy.Tree:
    """Core method: Executes booster and returns tree_ref annotated with distances."""
    booster_exec = 'booster_macos64' if subprocess.run(['uname'], stdout=subprocess.PIPE).stdout.decode().strip() == 'Darwin' else 'booster_linux64'

    tree_ref.suppress_unifurcations()
    tree_est.suppress_unifurcations()

    for node in [*tree_ref, *tree_est]:
        # if node.edge_length is None: # NOTE: use this if supporting edge lengths
        node.edge_length = 1.0

    rnd = str(random.randint(0, 10000))
    ref_path, est_path, out_path = f"/tmp/.{rnd}_ref.nwk", f"/tmp/.{rnd}_est.nwk", f"/tmp/.{rnd}_out.nwk"

    with open(ref_path, 'w') as f:
        f.write(tree_ref.as_string(schema='newick', suppress_rooting=True))
    with open(est_path, 'w') as f:
        f.write(tree_est.as_string(schema='newick', suppress_rooting=True))

    res = subprocess.run([booster_exec, '-i', ref_path, '-b', est_path, '-d', '0', '-r', out_path], capture_output=True)

    if res.returncode != 0:
        # give info about the two trees
        logging.warning(f"Booster failed. tree1=[n_nodes:{len(tree_ref.nodes())}, n_leaves:{len(tree_ref.leaf_nodes())}], tree2=[n_nodes:{len(tree_est.nodes())}, n_leaves:{len(tree_est.leaf_nodes())}]\n");
        return None

    # Load annotated tree to maintain topological context
    annotated_tree = dpy.Tree.get(path=out_path, schema='newick')

    # Clean up
    subprocess.run(['rm', ref_path, est_path, out_path])
    # print("TRANSFER IDX:", annotated_tree)
    return annotated_tree


def transfer_distance(tree_ref: dpy.Tree, tree_est: dpy.Tree) -> float:
    """Aggregates transfer indices for all internal edges."""
    ann_tree = get_booster_transfer_indices(tree_ref, tree_est)
    if not ann_tree: return None

    # Regex extracts the 'avgdist' from 'id|avgdist|depth'
    values = [float(re.findall(r'\|([\d\.]+)\|', str(n.label))[0])
              for n in ann_tree.internal_nodes() if n.label and '|' in str(n.label)]

    return sum(values) / len(values) if values else 0.0


def root_split_distance(tree_ref: dpy.Tree, tree_est: dpy.Tree) -> float:
    """Restricts the transfer index to the primary bipartition induced by the root."""
    ann_tree = get_booster_transfer_indices(tree_ref, tree_est)
    if not ann_tree: return None

    # In a bifurcating root, the children of the seed_node define the root split.
    # Booster labels the branch leading to the node; root children labels contain the root-split index.
    root_children = ann_tree.seed_node.child_nodes()
    indices = []
    for child in root_children:
        if child.label and '|' in str(child.label):
            indices.append(float(re.findall(r'\|([\d\.]+)\|', str(child.label))[0]))

    # For a bifurcating root, both children represent the same split (same distance).
    return indices[0] if indices else 0.0


def normalized_root_split_distance(tree_ref: dpy.Tree, tree_est: dpy.Tree) -> float:
    """Returns the normalized transfer distance [0, 1] for the root split."""
    ann_tree = get_booster_transfer_indices(tree_ref, tree_est)
    if not ann_tree: return None

    # Get the size of the smaller clade at the root in the reference
    root_children_ref = tree_ref.seed_node.child_nodes()
    p = min(len(child.leaf_nodes()) for child in root_children_ref)

    # Extract raw distance from booster output
    root_children_est = ann_tree.seed_node.child_nodes()
    raw_dist = 0.0
    for child in root_children_est:
        if child.label and '|' in str(child.label):
            raw_dist = float(re.findall(r'\|([\d\.]+)\|', str(child.label))[0])
            break

    # Normalize by p - 1
    return raw_dist / (p - 1) if p > 1 else 0.0