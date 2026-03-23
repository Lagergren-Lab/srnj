"""
Takes the output of sconce2 and medicc2, builds trees and evaluates them against the ground truth.
"""
import argparse
import os
import sys
import subprocess
import random

import anndata
import networkx as nx
import numpy as np
import dendropy as dpy

from sparsernj import sparse_rnj, sparse_nj, get_lca_from_pairwise, get_pairwise_from_lca
from utils.evaluation.benchmarking import get_sconce2_split_dist, get_medicc2_dist
from utils.algorithms.neighbor_joining import dlca_nj, anj, std_nj_root
from sparsernj.distance_provider import FixedDistanceProvider
from utils.tree_utils import convert_networkx_to_dendropy, normalized_quartet_distance, \
    normalized_rf_distance, transfer_distance, normalized_triplet_distance, normalized_root_split_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCONCE2 and MEDICC2 trees against ground truth.")
    parser.add_argument("--medicc2-dist", type=str, required=True, help="Path to MEDICC2 pairwise distance TSV file")
    parser.add_argument("--sconce2-hmm", type=str, required=True, help="Path to SCONCE2 HMM output file")
    parser.add_argument("--input-ad", type=str, required=True, help="Path to input .h5ad file with ground truth")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file for evaluation results")
    return parser.parse_args()

def get_anj_dtree(A, C, taxa, taxon_namespace, root_label):
    n = A.shape[0]
    root_adm = np.zeros((n + 1, n + 1))
    # add root distances to distance matrix (zero distance to any other node, and C + A from node to root)
    root_adm[:n, :n] = A
    # accumulation of root distances for each tip (exclude diagonal)
    total = C + A
    row_sums = total.sum(axis=1) - np.diag(total)
    root_adm[:n, n] = row_sums / (n - 1)
    # run rooted_nj
    anjnx = anj(root_adm, taxa=taxa, collapsed_root=False, root_idx=n)
    nx.relabel_nodes(anjnx, {n: root_label}, copy=False)
    anj_dtree = convert_networkx_to_dendropy(
        anjnx,
        taxon_namespace=taxon_namespace, edge_length='weight', internal_nodes_label='int'
    )
    return anj_dtree

def _writeD(dist_matrix, file_name):
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

def _fast_me(dist_matrix):
    # run balanced minimum evolution (fast ME)
    # timestamp to make unique file names
    suffix = f'_{random.randint(0,1000000)}'
    file_name = f'/tmp/.dist_mat{suffix}.PHYLIP'
    _writeD(dist_matrix, file_name)
    tree_prefix = f'/tmp/.tree{suffix}.nwk'
    call = subprocess.run(['fastme', '-i', file_name, '-o', tree_prefix, '-m', 'B', '-s'], capture_output=True,
                          text=True)
    # wait for process to finish
    if call.returncode != 0:
        print("FASTME ERROR: ", call.stderr)
        # cleanup
        os.remove(file_name)
        raise RuntimeError("FASTME failed")
    # open in dpy
    newick_str = ''
    with open(tree_prefix, 'r') as f:
        newick_str = f.read().strip()
    # clean up
    os.remove(tree_prefix)
    os.remove(file_name)
    return newick_str

def get_fastme_dtree(D, rootdist, taxa, taxon_namespace, root_label):
    n = D.shape[0]
    root_D = np.zeros((n + 1, n + 1))
    root_D[:n, :n] = D
    root_D[:n, n] = rootdist
    root_D[n, :n] = rootdist

    fastme_nwk = _fast_me(root_D)
    mapping = {i: taxa[i] for i in range(n)}
    mapping[n] = root_label
    tmpdtree = dpy.Tree.get(data=fastme_nwk, schema='newick')
    # root is a leaf here
    for leaf in tmpdtree.leaf_nodes():
        leaf.taxon.label = mapping[int(leaf.taxon.label)]
    # the following lines transform a taxon outgroup into a rooted tree with unifurcation
    root_leaf = tmpdtree.find_node_with_taxon_label(root_label)
    tmpdtree.reroot_at_edge(root_leaf.edge, update_bipartitions=False)
    tmpdtree.seed_node.label = root_label
    tmpdtree.seed_node.remove_child(root_leaf)
    tmpnwk = tmpdtree.as_string(schema='newick', suppress_rooting=True).strip()

    fastme_dtree = dpy.Tree.get(data=tmpnwk, schema='newick', taxon_namespace=taxon_namespace)
    fastme_dtree.is_rooted = True
    return fastme_dtree

def main():
    print("Validating input...")
    args = parse_args()
    header = "dist_method,tree_method,quartet_distance,rf_distance,transfer_distance,triplet_distance,rootsplit_distance"
    # check none of the input files are empty, if so, write NaN to output and exit
    for file_path in [args.sconce2_hmm, args.medicc2_dist, args.input_ad]:
        if os.path.getsize(file_path) == 0:
            print(f"Input file {file_path} is empty. Writing NaN to output and exiting.")
            with open(args.output, 'w') as f:
                f.write(header + '\n')
                for method in ['sconce2', 'med2']:
                    for tree_method in ['nj', 'snj', 'rnj', 'srnj', 'srnj1', 'srnjmaxlca', 'anj', 'fastme']:
                        f.write(f"{method},{tree_method},NaN,NaN,NaN,NaN,NaN\n")
            return
    # load SCONCE2 distance matrix
    sconce2_C, sconce2_A, sconce2_taxa = get_sconce2_split_dist(args.sconce2_hmm)
    # load MEDICC2 distance matrix
    med2_D, med2_rootdist, med2_taxa, med2_rootlabel = get_medicc2_dist(args.medicc2_dist)
    # load ground truth from input .h5ad
    adata = anndata.read_h5ad(args.input_ad)
    gt_nwk = adata.uns['cell-tree-newick']
    gt_dtree = dpy.Tree.get(data=gt_nwk, schema='newick')
    # prepare true tree, reroot_at_node 'founder'
    gt_dtree.is_rooted = True
    normal_taxa = [x.taxon for x in gt_dtree.seed_node.child_nodes() if x.is_leaf()]
    gt_dtree.prune_taxa(normal_taxa, suppress_unifurcations=False)
    gt_dtree.purge_taxon_namespace()  # after removing normal cells, update the taxon namespace
    root_label = gt_dtree.seed_node.label

    gt_taxa = [leaf.taxon.label for leaf in gt_dtree.leaf_nodes()]

    # ensure taxa match
    assert set(sconce2_taxa) == set(gt_taxa), f"Some SCONCE2 taxa not in ground truth\n\tGT: {gt_taxa}\n\tSCONCE2: {sconce2_taxa}"
    assert set(med2_taxa) == set(gt_taxa), f"Some MEDICC2 taxa not in ground truth\n\tGT: {gt_taxa}\n\tMEDICC2: {med2_taxa}"
    print("Building distance matrices...")
    # make DistanceProviders
    sconce2_D, sconce2_rootdist = get_pairwise_from_lca(sconce2_C, sconce2_A)
    med2_C, med2_A = get_lca_from_pairwise(med2_D, med2_rootdist)
    sconce2_dp = FixedDistanceProvider(sconce2_C, sconce2_A, taxa=sconce2_taxa)
    med2_dp = FixedDistanceProvider(med2_C, med2_A, taxa=med2_taxa)

    print("Building NJ...")
    # build trees from distance matrices
    # NJ trees
    sconce2_nj_dtree = convert_networkx_to_dendropy(
        std_nj_root(sconce2_D, sconce2_rootdist, taxa=sconce2_taxa, collapsed_root=False, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, edge_length='weight', internal_nodes_label='int'
    )
    # std_nj_tree = convert_networkx_to_dendropy(nx_rec_tree, taxon_namespace=taxon_namespace, edge_length='length', internal_nodes_label='int')
    med2_nj_dtree = convert_networkx_to_dendropy(
        std_nj_root(med2_D, med2_rootdist, taxa=med2_taxa, collapsed_root=False, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, edge_length='weight', internal_nodes_label='int'
    )

    print("Building SNJ...")
    # SNJ trees
    sconce2_snj_dtree = convert_networkx_to_dendropy(
        sparse_nj(sconce2_dp, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )
    med2_snj_dtree = convert_networkx_to_dendropy(
        sparse_nj(med2_dp, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )

    print("Building RNJ...")
    # RNJ trees
    sconce2_rnj_dtree = convert_networkx_to_dendropy(
        dlca_nj(sconce2_C, sconce2_A, taxa=sconce2_taxa, collapsed_root=False, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, edge_length='weight', internal_nodes_label='int'
    )
    med2_rnj_dtree = convert_networkx_to_dendropy(
        dlca_nj(med2_C, med2_A, taxa=med2_taxa, collapsed_root=False, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, edge_length='weight', internal_nodes_label='int'
    )

    print("Building ANJ...")
    sconce2_anj_dtree = get_anj_dtree(sconce2_A, sconce2_C, sconce2_taxa, gt_dtree.taxon_namespace, root_label)
    med2_anj_dtree = get_anj_dtree(med2_A, med2_C, med2_taxa, gt_dtree.taxon_namespace, root_label)

    print("Building FASTME...")
    sconce2_fastme_dtree = get_fastme_dtree(sconce2_D, sconce2_rootdist, sconce2_taxa, gt_dtree.taxon_namespace, root_label)
    med2_fastme_dtree = get_fastme_dtree(med2_D, med2_rootdist, med2_taxa, gt_dtree.taxon_namespace, root_label)

    print("Building SRNJ1 (k=1)...")
    # SRNJ trees
    sconce2_dp.reset_call_count()
    med2_dp.reset_call_count()
    sconce2_srnj1_dtree = convert_networkx_to_dendropy(
        sparse_rnj(sconce2_dp, k=1, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )
    med2_srnj1_dtree = convert_networkx_to_dendropy(
        sparse_rnj(med2_dp, k=1, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )

    print("Building SRNJ...")
    # SRNJ trees
    sconce2_dp.reset_call_count()
    med2_dp.reset_call_count()
    sconce2_srnj_dtree = convert_networkx_to_dendropy(
        sparse_rnj(sconce2_dp, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )
    med2_srnj_dtree = convert_networkx_to_dendropy(
        sparse_rnj(med2_dp, root_label=root_label),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )

    print("Building SRNJ (max_lca)...")
    # SRNJ trees
    sconce2_dp.reset_call_count()
    med2_dp.reset_call_count()
    sconce2_srnj_maxlca_dtree = convert_networkx_to_dendropy(
        sparse_rnj(sconce2_dp, root_label=root_label, ort_strategy='max_lca'),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )
    med2_srnj_maxlca_dtree = convert_networkx_to_dendropy(
        sparse_rnj(med2_dp, root_label=root_label, ort_strategy='max_lca'),
        taxon_namespace=gt_dtree.taxon_namespace, internal_nodes_label='int'
    )

    # evaluate trees against ground truth
    eval_results = [header]
    gt_dtree_collapsed = dpy.Tree(gt_dtree)  # clone but taxon namespace is shared
    gt_dtree_collapsed.suppress_unifurcations()
    med2_nj_rf = None
    # make all combinations of method and tree type
    combos = [
        ('sconce2_nj', sconce2_nj_dtree), ('med2_nj', med2_nj_dtree),
        ('sconce2_snj', sconce2_snj_dtree), ('med2_snj', med2_snj_dtree),
        ('sconce2_rnj', sconce2_rnj_dtree), ('med2_rnj', med2_rnj_dtree),
        ('sconce2_anj', sconce2_anj_dtree), ('med2_anj', med2_anj_dtree),
        ('sconce2_fastme', sconce2_fastme_dtree), ('med2_fastme', med2_fastme_dtree),
        ('sconce2_srnj1', sconce2_srnj1_dtree), ('med2_srnj1', med2_srnj1_dtree),
        ('sconce2_srnj', sconce2_srnj_dtree), ('med2_srnj', med2_srnj_dtree),
        ('sconce2_srnjmaxlca', sconce2_srnj_maxlca_dtree), ('med2_srnjmaxlca', med2_srnj_maxlca_dtree),
    ]
    for method, dtree in combos:
        print(f"Evaluating {method} tree...")
        assert len(gt_dtree.leaf_nodes()) == len(dtree.leaf_nodes()), f"Number of leaves in GT and {method} tree do not match"
        # quartet
        qd = normalized_quartet_distance(gt_dtree, dtree)
        trip_d = normalized_triplet_distance(gt_dtree, dtree)
        # collapse root for rf distance (but preserve unifurcations for quartet distance)
        dtree.suppress_unifurcations()
        rfd = normalized_rf_distance(gt_dtree_collapsed, dtree)
        td = transfer_distance(gt_dtree_collapsed, dtree)
        # rootsplit distance (compare rooted splits after collapsing unifurcations)
        rsd = normalized_root_split_distance(gt_dtree_collapsed, dtree)
        if method == 'med2_nj':
            med2_nj_rf = rfd  # save for later comparison with medicc2 original tree
        dist_method, tree_method = method.split('_')
        line = f"{dist_method},{tree_method},{qd:.4f},{rfd:.4f},{td:.4f},{trip_d:.4f},{rsd:.4f}"
        eval_results.append(line)

    # get medicc2 original tree and compare rf distance with nj tree computed above
    try:
        medicc2_tree_file = args.medicc2_dist.replace('pairwise_distances.tsv', 'final_tree.new')
        medicc2_dtree = dpy.Tree.get(path=medicc2_tree_file, schema='newick', taxon_namespace=gt_dtree.taxon_namespace)
        medicc2_dtree.is_rooted = True
        medicc2_dtree.suppress_unifurcations()
        medicc2_rf = normalized_rf_distance(gt_dtree_collapsed, medicc2_dtree)
        with open(args.output.replace('.csv', '_medicc2_validation.csv'), 'w') as f:
            f.write("method,rf_distance\n")
            f.write(f"medicc2_original,{medicc2_rf:.4f}\n")
            f.write(f"medicc2_nj,{med2_nj_rf}\n")
    except Exception as e:
        print(f"Could not validate MEDICC2 original tree: {e}")

    # save results to output file
    with open(args.output, 'w') as f:
        f.write('\n'.join(eval_results))
    print(f"Evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
