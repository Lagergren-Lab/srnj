"""
This script performs the real dataset experiment for the paper SparseRNJ.
Given sconce2 output on the 500 tumor cells selected from region E of the 10x breast cancer dataset, we first
 run SRNJ, DLCA-NJ, and NJ on 20 bootstrapped samples of 200 cells each and validate the trees with an orthogonal
 mutation matrix obtained from the 10x data (Zaccaria et al. 2021), by means of a parsimony score. Also,
 we take the clone assignments from the inferred trees and compute F1 scores for clades with respect to the clone assignments
 as described in the DICE paper (Weiner et al. 2025).
"""
import pandas as pd
import networkx as nx
from sklearn.metrics import f1_score
import numpy as np
import dendropy as dpy
import os
import sys

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from sparsernj import FixedDistanceProvider, sparse_nj, sparse_rnj
from utils.algorithms.neighbor_joining import dlca_nj, std_nj_root
from utils.evaluation.benchmarking import read_chisel_mutations, get_sconce2_split_dist, read_chisel_clones
from utils.tree_utils import nxtree_to_newick

sconce2_hmm_path = "/proj/sc_ml/shared/s0_10x/5M/sconce2_out/model.hmm"
chisel_mutations_path = "/proj/sc_ml/users/x_vitza/chisel-data/patientS0/snvs/cellmutations.tsv.gz"
chisel_clones_path = "/proj/sc_ml/users/x_vitza/chisel-data/patientS0/clones/sectionE/mapping.tsv.gz"
out_csv_path = "/home/x_vitza/Cellmates/output/reproducibility/experiments/breast_data_sconce2/results.csv"
out_trees_dir = "/home/x_vitza/Cellmates/output/reproducibility/experiments/breast_data_sconce2/trees"

def f1_score_clades(nxtree, clone_series: pd.Series) -> float:
    # Get clades as sets of barcodes
    newick = nxtree_to_newick(nxtree)
    tree = dpy.Tree.get(data=newick, schema='newick')
    clades = [{l.taxon.label for l in n.leaf_iter()} for n in tree.postorder_node_iter()]

    # Group barcodes by clone
    clone_to_cells = clone_series.groupby(clone_series).groups
    barcodes = clone_series.index.tolist()
    
    f1_scores = []
    for clone, cells in clone_to_cells.items():
        best_f1 = 0
        y_true = [1 if b in cells else 0 for b in barcodes]
        for clade in clades:
            y_pred = [1 if b in clade else 0 for b in barcodes]
            score = f1_score(y_true, y_pred)
            if score > best_f1: best_f1 = score
        f1_scores.append(best_f1)

    return np.mean(f1_scores).item()

def compute_parsimony_score(tree: nx.DiGraph, snv_df: pd.DataFrame):
    total_score = 0
    active_mutations = 0
    post_order = list(nx.dfs_postorder_nodes(tree))
    # Map all leaf barcodes in the tree to matrix row indices
    node_to_idx = {n: i for i, n in enumerate(snv_df.index)}
    root = [n for n, d in tree.in_degree() if d == 0][0]

    for m_idx in range(snv_df.shape[1]):
        col = snv_df.iloc[:, m_idx].values
        if not np.any(col == 1): continue
        active_mutations += 1
        
        sets = {}
        col_score = 0
        for node in post_order:
            if tree.out_degree(node) == 0:
                # Every leaf must have a state (0 or 1)
                val = col[node_to_idx[node]]
                sets[node] = {int(val)} if val in [0, 1] else {0, 1}
            else:
                child_sets = [sets[c] for c in tree.successors(node)]
                intersect = set.intersection(*child_sets)
                if intersect:
                    sets[node] = intersect
                else:
                    sets[node] = set.union(*child_sets)
                    col_score += 1
        
        # If the root must be 1, that's an additional gain event
        if 0 not in sets[root]: col_score += 1
        total_score += col_score

    if total_score < active_mutations:
        print(f"Warning: total parsimony score {total_score} is less than the number of active mutations {active_mutations}. This should not happen.")
    return total_score, total_score - active_mutations

def make_trees(dist_provider):

    # SRNJ
    srnj_nxtree = sparse_rnj(dist_provider)
    # DLCA-NJ
    C, A = dist_provider.get_dms(taxa=dist_provider.taxa)
    dlcanj_nxtree = dlca_nj(C, A, taxa=dist_provider.taxa)
    # NJ
    snj_nxtree = sparse_nj(dist_provider)
    # standard NJ
    D, rootdist = dist_provider.get_dist_matrix_with_root(dist_provider.taxa)
    nj_nxtree = std_nj_root(D, rootdist, taxa=dist_provider.taxa)

    return {
        'SRNJ': srnj_nxtree,
        'DLCA-NJ': dlcanj_nxtree,
        'SNJ': snj_nxtree,
        'NJ': nj_nxtree
    }

def make_full_trees(C, A, cell_names, out_trees_dir, sparse_snv_matrix, clone_assignments):
    # build the full trees on all 500 cells and save them in newick format for reference (not used in the main loop)
    dp = FixedDistanceProvider(C, A, cell_names)
    trees = make_trees(dp)
    for method, tree in trees.items():
        # convert to dendropy and save in newick format
        newick = nxtree_to_newick(tree)
        dtree = dpy.Tree.get(data=newick, schema='newick')
        dtree.write(path=f"{out_trees_dir}/{method}_full_tree.nwk", schema='newick')

        # compute parsimony scores and F1 scores for the full trees as well
        parsimony_score, parsimony_homoplasies = compute_parsimony_score(tree, sparse_snv_matrix)
        f1_score = f1_score_clades(tree, clone_assignments)
        with open(f"{out_trees_dir}/full_tree_metrics.txt", 'a') as f:
            f.write(f"{method} full tree: Parsimony score = {parsimony_score}, F1 score = {f1_score}\n")

        print(f"{method} full tree: Parsimony score = {parsimony_score} (with {parsimony_homoplasies} homoplasies), F1 score = {f1_score}")

    print("Full trees saved in newick format.")

def bootstrap_test(C, A, cell_names, sparse_snv_matrix, clone_assignments, out_csv_path,
        n_sample=200, n_iterations=50):

    for i in range(n_iterations):
    #   sample 200 cells without replacement from the 500 tumor cells in region E (list)
        cells_sample = pd.Series(cell_names).sample(n=n_sample, replace=False, random_state=i).tolist()
        _idx = [cell_names.index(c) for c in cells_sample]
        dp = FixedDistanceProvider(C[_idx][:, _idx], A[_idx][:, _idx], cells_sample)
    #   run SRNJ, DLCA-NJ, and NJ on the sampled data
        trees = make_trees(dp)
    #   compute the parsimony score of the inferred trees with respect to the 10x mutation matrix
    #   compute clone assignments with f1 score clades
        parsimony_score = {}
        f1_score = {}
        for method, tree in trees.items():
            parsimony_score[method], _ = compute_parsimony_score(tree, sparse_snv_matrix.loc[cells_sample])
            f1_score[method] = f1_score_clades(tree, clone_assignments.loc[cells_sample])
        print(f"Sample {i}: Parsimony scores: {parsimony_score}, F1 scores: {f1_score}")
        # store in a csv file
        if i == 0:
            with open(out_csv_path, 'w') as f:
                f.write("Sample,Method,ParsimonyScore,F1Score\n")
        with open(out_csv_path, 'a') as f:
            for method in trees.keys():
                f.write(f"{i},{method},{parsimony_score[method]},{f1_score[method]}\n")
    print(f"Results saved to {out_csv_path}")
    
def main():
    # load the sconce2 matrix
    C, A, cell_names = get_sconce2_split_dist(sconce2_hmm_path)
    cell_names = [c.split('-', 1)[0] for c in cell_names]  # remove region prefix for matching with CHISEL

    print(f"Loaded sconce2 distance matrix with {C.shape[0]} cells and {C.shape[1]} distances.")
    print(f"Cell names: {cell_names[:5]} ...")
    # load the 10x mutation matrix
    sparse_snv_matrix = read_chisel_mutations(chisel_mutations_path)
    print(f"Loaded CHISEL SNV matrix with {sparse_snv_matrix.shape[0]} cells and {sparse_snv_matrix.shape[1]} SNVs.")
    print(sparse_snv_matrix.head())
    # load the CHISEL clone assignments for region E
    clone_assignments = read_chisel_clones(chisel_clones_path)
    print(f"Loaded CHISEL clone assignments for {len(clone_assignments)} cells.")
    print(clone_assignments.head())

    # run the bootstrap test
    #bootstrap_test(C, A, cell_names, sparse_snv_matrix, clone_assignments, out_csv_path)

    # optionally, build the full trees on all 500 cells and save them in newick format for reference (not used in the main loop)
    if not os.path.exists(out_trees_dir):
        os.makedirs(out_trees_dir, exist_ok=True)
    make_full_trees(C, A, cell_names, out_trees_dir, sparse_snv_matrix, clone_assignments.loc[cell_names])

if __name__ == "__main__":
    main()
