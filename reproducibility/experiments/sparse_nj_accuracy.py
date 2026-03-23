"""
Use sparse neighbor-joining to reconstruct phylogenetic trees from distance matrices and compare with quartet distance, RF distance, and triplet distance
against ground truth trees.
"""
import os
import sys
import random
import argparse
import numpy as np
import dendropy as dpy
import time
import tqdm

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.algorithms.neighbor_joining import split_tdm, std_njx, dlca_nj
from utils import tree_utils
from utils import tree_simulation as tb_utils
from sparsernj import sparse_rnj, sparse_nj
from sparsernj import FixedDistanceProvider
from utils.algorithms.tree_methods import get_fastme_dtree, get_anj_dtree

TIME_PAIRWISE_DIST = 0.1  # emulated time per pairwise distance call in seconds, for time estimation based on number of calls


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse NJ accuracy evaluation")
    parser.add_argument("--demo", action="store_true", 
                       help="Run with reduced parameters for quick demo")
    return parser.parse_args()


def make_random_tree_and_CA(n_cells, seed=None):
    """
    Return (tree, cnp, C, A) for testing.
    - n_cells: number of leaves/cells
    Uses default settings: unbalanced tree and p_change=0.1.
    """
    if seed is not None:
        dpy.utility.GLOBAL_RNG.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    tree, cnp = tb_utils.simulate_tree(n_cells, 0.01, tree_type='balanced')
    # tree, cnp = tb_utils.simulate_tree(n_cells, 0.01, tree_type='unbalanced')
    # print("=== Generated tree ===")
    # print("CNP shape:", cnp.shape)
    tree.encode_bipartitions(suppress_unifurcations=False)  # important for weighted RF distance
    trip_dist = tb_utils.compute_triplet_distance_matrix(tree, cnp, n_cells)
    C, A = split_tdm(trip_dist)
    return tree, cnp, C, A, trip_dist

def plot_results(file_path, out_path=None):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    from itertools import combinations

    df = pd.read_csv(file_path)

    # apply ggplot theme/aesthetics
    plt.style.use('ggplot')
    sns.set_theme()

    methods = sorted(df["method"].unique())
    palette_colors = sns.color_palette("tab10", n_colors=max(1, len(methods)))
    palette = dict(zip(methods, palette_colors))

    g = sns.catplot(
        data=df,
        x="method",
        y="dist",
        col="n_cells",
        row="metric",
        kind="box",
        sharey=False,
        height=4,
        aspect=1.1,
        order=methods,
        palette=palette,
        boxprops=dict(alpha=0.7),
        fliersize=0,
    )

    # overlay black scatter dots on each facet (so points are visible in front of boxes)
    for (row_val, col_val), ax in g.axes_dict.items():
        sub = df[(df["metric"] == row_val) & (df["n_cells"] == col_val)]
        sns.stripplot(
            data=sub,
            x="method",
            y="dist",
            order=methods,
            color='black',    # black scatter dots per user request
            size=3.5,
            jitter=True,
            dodge=False,
            linewidth=0,
            alpha=0.85,
            ax=ax,
            zorder=10,
        )

    g.set_axis_labels("Method", "Distance")
    g.set_titles(row_template="{row_name}", col_template="n_cells={col_name}")
    plt.tight_layout()

    if out_path is None:
        out_path = file_path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=300)

    print("\n=== Mann-Whitney U p-values ===")
    for metric in sorted(df["metric"].unique()):
        for n in sorted(df["n_cells"].unique()):
            sub = df[(df["metric"] == metric) & (df["n_cells"] == n)]
            methods_list = sorted(sub["method"].unique())
            print(f"\nmetric={metric}, n_cells={n}")
            for m1, m2 in combinations(methods_list, 2):
                d1 = sub[sub["method"] == m1]["dist"]
                d2 = sub[sub["method"] == m2]["dist"]
                if len(d1) > 0 and len(d2) > 0:
                    _, p = mannwhitneyu(d1, d2, alternative="two-sided")
                    print(f"  {m1} vs {m2}: p={p:.6e}")

def plot_time_results(file_path, out_path=None):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv(file_path)
    # apply ggplot style here as well for consistency
    plt.style.use('ggplot')
    sns.set_theme()

    # compute means and std by method and n_cells
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=df, x="n_cells", y="est_time", hue="method", marker='o', err_style='bars', errorbar='sd')
    ax.set_xlabel("n_cells")
    ax.set_ylabel("Estimated time (s)")
    ax.set_title("Time vs n_cells")
    plt.tight_layout()

    if out_path is None:
        out_path = file_path.replace('.csv', '.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved time plot to {out_path}")

def run_experiment(demo=False) -> (str, str):
    # returns the filepath of the stats csv file
    outdir = "./output/"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stats_file_path = outdir + timestamp + "_sparse_nj_quartet_distance_stats.csv"
    times_file_path = outdir + timestamp + "_sparse_rnj_time_estimates.csv"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # parameters  
    if demo:
        n_cells = [20, 50]
        n_repeats = 2
    else:
        n_cells = [20, 50, 100, 200, 500]
        n_repeats = 20

    # programmable lists for easy extension/modification
    methods_order = [
        "nj",
        "dlca_nj",
        "srnj1",        # sparse rnj with k=1 (srnj1)
        "srnj",         # sparse rnj with default k (srnj)
        "srnj_maxlca",  # sparse rnj with max_lca strategy
        "snj",
        "fastme",
        "anj",
    ]
    metrics_order = [
        "rf_distance",
        "quartet_distance",
        "triplet_distance",
        "rootsplit_distance",
        "transfer_distance",
    ]

    # prepare stats file header
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write("n_cells,seed,dist,metric,method\n")

    # prepare times file header
    with open(times_file_path, 'w') as tfile:
        tfile.write("n_cells,seed,method,est_time,num_calls\n")

    for i, n in enumerate(n_cells):
        # replace wiht tqdm
        # for seed in range(n_repeats):
        for seed in tqdm.tqdm(range(n_repeats), desc=f"n={n}", unit="run"):
            # generate random tree and data
            tree, cnp, C, A, tdm = make_random_tree_and_CA(n, seed=seed)
            # print("LCA:\n", C)
            # if seed == 0:
            #     tb_utils.plot_cell_cn_tree(tree, cnp[:n], outfile=outdir + f"tree_n{n}_rep{seed}.png")
            # tree.print_plot(plot_metric='length')
            # copy tree with no unifurcation for later
            tree_collapsed = dpy.Tree(tree)
            tree_collapsed.suppress_unifurcations()

            # compute distance matrix provider
            dist_matrix = FixedDistanceProvider(C, A, track_distinct_calls=True)

            # add a common root to the original tree for fair comparison (done once per run)
            old_root = tree.seed_node
            new_root = dpy.Node(label='root')
            new_root.add_child(old_root)
            tree.seed_node = new_root
            # print("truenewick:", tree.as_string('newick', suppress_edge_lengths=True))

            # containers to accumulate per-method results
            method_trees = {}
            metrics_values = {}
            times_values = {}

            # iterate methods and run them uniformly, storing trees, quartet/triplet scores and timing
            for method in methods_order:
                if method == "srnj1":
                    dist_matrix.reset_call_count()
                    t0 = time.perf_counter()
                    nx_tree = sparse_rnj(dist_matrix, k=1)
                    t_elapsed = time.perf_counter() - t0
                    num_calls = dist_matrix.num_calls
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                    dtree = tree_utils.convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, internal_nodes_label='int')
                elif method == "srnj":
                    dist_matrix.reset_call_count()
                    t0 = time.perf_counter()
                    nx_tree = sparse_rnj(dist_matrix)
                    t_elapsed = time.perf_counter() - t0
                    num_calls = dist_matrix.num_calls
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                    dtree = tree_utils.convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, internal_nodes_label='int')
                elif method == "srnj_maxlca":
                    dist_matrix.reset_call_count()
                    t0 = time.perf_counter()
                    nx_tree = sparse_rnj(dist_matrix, ort_strategy='max_lca')
                    t_elapsed = time.perf_counter() - t0
                    num_calls = dist_matrix.num_calls
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                    dtree = tree_utils.convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, internal_nodes_label='int')
                elif method == "snj":
                    dist_matrix.reset_call_count()
                    t0 = time.perf_counter()
                    nx_tree = sparse_nj(dist_matrix)
                    t_elapsed = time.perf_counter() - t0
                    num_calls = dist_matrix.num_calls
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                    dtree = tree_utils.convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, internal_nodes_label='int')
                elif method == "dlca_nj":
                    # dlca_nj uses full C,A and we emulate O(n^2) calls when estimating time
                    t0 = time.perf_counter()
                    nx_tree = dlca_nj(C, A, taxa=list(range(n)), collapsed_root=False, root_label='root')
                    t_elapsed = time.perf_counter() - t0
                    num_calls = n * (n - 1) // 2
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                    dtree = tree_utils.convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, internal_nodes_label='int')
                elif method == "nj":
                    t0 = time.perf_counter()
                    dtree = std_njx(tdm, taxon_namespace=tree.taxon_namespace, collapsed_root=False)
                    dtree.is_rooted = True
                    dtree.encode_bipartitions(suppress_unifurcations=False)
                    dtree.seed_node.label = 'root'
                    t_elapsed = time.perf_counter() - t0
                    num_calls = n * (n - 1) // 2
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                elif method == "fastme":
                    D, rootdist = dist_matrix.get_dist_matrix_with_root(dist_matrix.taxa)
                    t0 = time.perf_counter()
                    dtree = get_fastme_dtree(D, rootdist, dist_matrix.taxa, tree.taxon_namespace, root_label='root')
                    t_elapsed = time.perf_counter() - t0
                    num_calls = n * (n - 1) // 2
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                elif method == "anj":
                    C, A = dist_matrix.get_dms(taxa=dist_matrix.taxa)
                    t0 = time.perf_counter()
                    dtree = get_anj_dtree(A, C, dist_matrix.taxa, tree.taxon_namespace, root_label='root')
                    t_elapsed = time.perf_counter() - t0
                    num_calls = n * (n - 1) // 2
                    est_time = t_elapsed + num_calls * TIME_PAIRWISE_DIST
                else:
                    print(f"Unknown method {method}, skipping")
                    continue

                # store tree and time info
                method_trees[method] = dtree
                times_values[method] = {"est_time": est_time, "num_calls": num_calls}

                dtree_collapsed = dpy.Tree(dtree)
                dtree_collapsed.suppress_unifurcations()
                for metric in metrics_order:
                    if metric == "rf_distance":
                        rf = tree_utils.normalized_rf_distance(tree_collapsed, dtree_collapsed)
                        metrics_values.setdefault(method, {})[metric] = rf
                        # print(f"method={method}, n={n}, seed={seed}, rf_distance={rf}, time={est_time:.6f}, calls={num_calls}")
                    elif metric == "rootsplit_distance":
                        rsd = tree_utils.normalized_root_split_distance(tree_collapsed, dtree_collapsed)
                        metrics_values.setdefault(method, {})[metric] = rsd
                        # print(f"method={method}, n={n}, seed={seed}, rootsplit_distance={rsd}, time={est_time:.6f}, calls={num_calls}")
                    elif metric == "quartet_distance":
                        qd = tree_utils.normalized_quartet_distance(tree, dtree)
                        metrics_values.setdefault(method, {})[metric] = qd
                        # print(f"method={method}, n={n}, seed={seed}, quartet_distance={qd}, time={est_time:.6f}, calls={num_calls}")
                    elif metric == "triplet_distance":
                        td = tree_utils.normalized_triplet_distance(tree, dtree)
                        metrics_values.setdefault(method, {})[metric] = td
                        # print(f"method={method}, n={n}, seed={seed}, triplet_distance={td}, time={est_time:.6f}, calls={num_calls}")
                    elif metric == "transfer_distance":
                        tfd = tree_utils.transfer_distance(tree_collapsed, dtree_collapsed)
                        metrics_values.setdefault(method, {})[metric] = tfd
                        # print(f"method={method}, n={n}, seed={seed}, transfer_distance={td}, time={est_time:.6f}, calls={num_calls}")

            # write distance metrics to csv
            with open(stats_file_path, 'a') as stats_file:
                for method in methods_order:
                    for metric in metrics_order:
                        val = metrics_values.get(method, {}).get(metric, "")
                        stats_file.write(f"{n},{seed},{val},{metric},{method}\n")

            # save timing estimates to csv
            with open(times_file_path, 'a') as tfile:
                for method in methods_order:
                    tv = times_values.get(method)
                    if tv is None:
                        continue
                    tfile.write(f"{n},{seed},{method},{tv['est_time']:.8f},{tv['num_calls']}\n")

    return stats_file_path, times_file_path


if __name__ == "__main__":
    args = parse_args()
    scores_path, times_path = run_experiment(demo=args.demo)

    # after all runs plot the time results
    plot_time_results(times_path)
    plot_results(scores_path)
