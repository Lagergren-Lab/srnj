"""
Tree simulation utilities for phylogenetic analysis.
Provides functions for generating random trees and computing distance matrices.
"""
import numpy as np
import dendropy as dpy
from .algorithms.neighbor_joining import split_tdm


def simulate_tree(n_cells, branch_length_mean, tree_type='balanced', seed=None):
    """
    Simulate a random tree with copy number profiles.
    
    Args:
        n_cells: Number of leaf nodes (cells)
        branch_length_mean: Mean branch length 
        tree_type: 'balanced' or 'unbalanced'
        seed: Random seed
        
    Returns:
        tuple: (dendropy Tree, copy number profiles array)
    """
    if seed is not None:
        np.random.seed(seed)
        dpy.utility.GLOBAL_RNG.seed(seed)
    
    # Create a simple random tree
    taxon_namespace = dpy.TaxonNamespace([str(i) for i in range(n_cells)])
    
    if tree_type == 'balanced':
        # Create balanced tree using dendropy's birth-death process
        tree = dpy.simulate.treesim.birth_death_tree(
            birth_rate=1.0, 
            death_rate=0.0,
            num_extant_tips=n_cells,
            taxon_namespace=taxon_namespace
        )
    else:
        # Create unbalanced tree
        tree = dpy.simulate.treesim.birth_death_tree(
            birth_rate=2.0,
            death_rate=0.5, 
            num_extant_tips=n_cells,
            taxon_namespace=taxon_namespace
        )
    
    # Scale branch lengths
    for node in tree.preorder_node_iter():
        if node.edge_length is not None:
            node.edge_length *= branch_length_mean / 0.01  # Scale to desired mean
    
    # Generate simple copy number profiles (simulate evolution along tree)
    n_bins = 1000  # Default number of genomic bins
    cnp = np.random.randint(0, 5, size=(n_cells * 2, n_bins))  # Include internal nodes
    
    return tree, cnp


def compute_triplet_distance_matrix(tree, cnp, n_cells):
    """
    Compute triplet distance matrix from tree and copy number profiles.
    
    Args:
        tree: dendropy Tree object
        cnp: Copy number profiles array  
        n_cells: Number of cells
        
    Returns:
        np.ndarray: 3D triplet distance matrix (n_cells, n_cells, 3)
    """
    from .tree_utils import get_ctr_table
    
    # Use the existing get_ctr_table function to compute proper triplet distances
    ctr_table, _ = get_ctr_table(tree, full=True)
    
    return ctr_table