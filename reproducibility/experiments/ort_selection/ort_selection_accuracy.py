"""
Experiment: orienting-leaf selection strategy comparison on CNAsim data.

Generates (or reuses) a CNAsim tumour dataset, builds selection matrices from
observed copy-number profiles, and compares tree-reconstruction accuracy and
oracle-selection accuracy for each strategy.

Requirements
------------
The ``srnj`` conda environment must have CNASim installed (``pip install CNASim``).
See environment.yml.  External binaries triplet_dist, quartet_dist, and
booster_linux64 must be on PATH (same as for sparse_nj_accuracy.py).

Quick demo
----------
    python ort_selection_accuracy.py --demo

Full run (serial)
-----------------
    python ort_selection_accuracy.py --n-cells 50 100 200 --seeds 0 1 2

Full run (32 parallel workers via SLURM)
-----------------------------------------
    sbatch slurm_run.sh
"""

import argparse
import concurrent.futures
import csv
import importlib.util
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import dendropy as dpy
import numpy as np

# ------------------------------------------------------------------
# Path setup: allow running directly (sys.path) or as installed pkg
# ------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent
for _candidate in (_REPO, _REPO.parent, _REPO.parent.parent, _REPO.parent.parent.parent):
    if (_candidate / "src").is_dir():
        sys.path.insert(0, str(_candidate / "src"))
        break

from sparsernj import FixedDistanceProvider, sparse_rnj
from sparsernj.ort_selection import selection_matrices_from_cn, matrix_selection_strategy
from sparsernj.utils import tree_utils
from sparsernj.utils.algorithms.neighbor_joining import dlca_nj, split_tdm
from sparsernj.utils.tree_utils import get_ctr_table

# ------------------------------------------------------------------
# CNAsim defaults (match Snakemake workflow config)
# ------------------------------------------------------------------
NORMAL_FRACTION = 0.2
N_BINS = 1000
N_CLONES = 6
LAMBDA_PARAM = 5
N_CHROM = 10
BIN_LENGTH = 1_000_000
CN_LENGTH_MEAN = 10_000_000

# Strategies that use argmax(LCA depth); oracle for them is gt_max_lca
MAX_LCA_STRATEGIES = frozenset({"gt_max_lca", "hamming_max_lca", "nll_max_lca"})

# Module-level store populated by run_experiment before forking workers.
# Workers inherit it via fork (Linux COW) — no pickling overhead.
_TASK_DATA: dict = {}  # n_cells -> (C, A, gt_newick, taxa, selection_mats, k_eff)


# ------------------------------------------------------------------
# CNAsim helpers
# ------------------------------------------------------------------

def _load_cnasim2adata():
    scripts_dir = _THIS.parent.parent.parent / "workflows" / "srnj_sconce2" / "workflow" / "scripts"
    cnasim2adata_path = scripts_dir / "cnasim2adata.py"
    if not cnasim2adata_path.exists():
        raise FileNotFoundError(f"cnasim2adata.py not found at {cnasim2adata_path}")
    spec = importlib.util.spec_from_file_location("cnasim2adata", cnasim2adata_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_cnasim(out_dir: Path, n_cells: int, cnasim_exec: str = "cnasim") -> Path:
    cnasim_tmp = out_dir / "cnasim_tmp"
    cnasim_tmp.mkdir(parents=True, exist_ok=True)
    n_total = round(n_cells / (1.0 - NORMAL_FRACTION))
    chrom_length = int(N_BINS * BIN_LENGTH / N_CHROM)
    cmd = [
        cnasim_exec, "-m", "1", "--use-uniform-coverage",
        "-n", str(n_total), "-c", str(N_CLONES), "-p1", str(LAMBDA_PARAM),
        "-o", str(cnasim_tmp), "-N", str(N_CHROM), "-L", str(chrom_length),
        "-n1", str(NORMAL_FRACTION), "-B", str(BIN_LENGTH),
        "--cn-length-mean", str(CN_LENGTH_MEAN), "--cn-copy-param", "0.8",
    ]
    subprocess.run(cmd, check=True, cwd=str(cnasim_tmp))
    if not (cnasim_tmp / "readcounts.tsv").is_file():
        raise FileNotFoundError(f"CNAsim did not produce readcounts.tsv in {cnasim_tmp}")
    return cnasim_tmp


def _ground_truth_ca(adata):
    """Extract (C, A, gt_tree) from the CNAsim AnnData."""
    nwk = adata.uns["cell-tree-newick"]
    obs_names = list(adata.obs_names)
    n = len(obs_names)
    name_to_idx = {name: i for i, name in enumerate(obs_names)}
    keep = set(obs_names)

    raw_tree = dpy.Tree.get(data=nwk, schema="newick")
    raw_tree.is_rooted = True
    drop = [lf.taxon for lf in raw_tree.leaf_node_iter() if lf.taxon.label not in keep]
    if drop:
        raw_tree.prune_taxa(drop, suppress_unifurcations=False)
        raw_tree.purge_taxon_namespace()

    for lf in raw_tree.leaf_node_iter():
        new_lbl = str(name_to_idx[lf.taxon.label])
        lf.taxon.label = new_lbl
        lf.label = new_lbl

    tns = dpy.TaxonNamespace([dpy.Taxon(str(i)) for i in range(n)])
    gt_tree = dpy.Tree.get(
        data=raw_tree.as_string(schema="newick"), schema="newick", taxon_namespace=tns
    )
    gt_tree.is_rooted = True
    ctr, _ = get_ctr_table(gt_tree, full=True)
    C, A = split_tdm(ctr)
    return C, A, gt_tree


# ------------------------------------------------------------------
# Tree-metric helper
# ------------------------------------------------------------------

def _tree_metrics(gt_tree: dpy.Tree, est_tree: dpy.Tree) -> dict:
    gt_c = dpy.Tree(gt_tree)
    gt_c.suppress_unifurcations()
    est_c = dpy.Tree(est_tree)
    est_c.suppress_unifurcations()
    gt_qt = dpy.Tree(gt_tree)
    old_root = gt_qt.seed_node
    new_root = dpy.Node(label="root")
    new_root.add_child(old_root)
    gt_qt.seed_node = new_root
    metrics = {
        "rf_distance":      tree_utils.normalized_rf_distance(gt_c, est_c),
        "quartet_distance": tree_utils.normalized_quartet_distance(gt_qt, est_tree),
        "triplet_distance": tree_utils.normalized_triplet_distance(gt_qt, est_tree),
    }
    try:
        metrics["transfer_distance"] = tree_utils.transfer_distance(gt_c, est_c)
    except FileNotFoundError:
        metrics["transfer_distance"] = None
    return metrics


# ------------------------------------------------------------------
# Per-(n_cells, seed) worker  (module-level so fork can see _TASK_DATA)
# ------------------------------------------------------------------

def _run_one_seed(args):
    """Run all strategies for one (n_cells, seed) pair. Returns (metrics_rows, accuracy_rows)."""
    n_cells, seed = args
    C, A, gt_newick, taxa, selection_mats, k_eff = _TASK_DATA[n_cells]

    # Reconstruct gt_tree in worker (dendropy trees aren't fork-safe across re-init)
    n = len(taxa)
    tns = dpy.TaxonNamespace([dpy.Taxon(str(i)) for i in range(n)])
    gt_tree = dpy.Tree.get(data=gt_newick, schema="newick", taxon_namespace=tns)
    gt_tree.is_rooted = True

    random.seed(seed)
    np.random.seed(seed)
    dpy.utility.GLOBAL_RNG.seed(seed)

    metrics_rows = []
    accuracy_rows = []

    def _add_metrics(strategy, tree_m):
        for metric, val in tree_m.items():
            metrics_rows.append({
                "n_cells": n_cells, "seed": seed,
                "strategy": strategy, "metric": metric, "dist": val,
            })

    # --- DLCA-NJ baseline ---
    nx_dlca = dlca_nj(C, A, taxa=taxa, collapsed_root=False, root_label="root")
    d_dlca = tree_utils.convert_networkx_to_dendropy(
        nx_dlca, taxon_namespace=gt_tree.taxon_namespace, internal_nodes_label="int"
    )
    _add_metrics("dlca_nj", _tree_metrics(gt_tree, d_dlca))

    # --- SRNJ built-in strategies ---
    for builtin_name in ("min_D", "max_lca"):
        dp = FixedDistanceProvider(C, A, taxa=taxa, track_distinct_calls=True)
        nx_t = sparse_rnj(dp, ort_selector=builtin_name, k=k_eff)
        d_t = tree_utils.convert_networkx_to_dendropy(
            nx_t, taxon_namespace=gt_tree.taxon_namespace, internal_nodes_label="int"
        )
        _add_metrics(f"srnj_{builtin_name}", _tree_metrics(gt_tree, d_t))

    # --- Cheap-proxy strategies ---
    for strategy in selection_mats:
        oracle_key = "gt_max_lca" if strategy in MAX_LCA_STRATEGIES else "gt_min"
        stats = {
            "total_decisions": 0,
            "correct_A": 0, "correct_B": 0, "correct_pair": 0, "correct_direction": 0,
        }
        selector = matrix_selection_strategy(
            selection_mats[strategy],
            oracle_matrix=selection_mats[oracle_key],
            stats=stats,
            distance_matrices=(C, A),
        )
        dp = FixedDistanceProvider(C, A, taxa=taxa, track_distinct_calls=True)
        nx_t = sparse_rnj(dp, ort_selector=selector, k=k_eff, all_leaves=True)
        d_t = tree_utils.convert_networkx_to_dendropy(
            nx_t, taxon_namespace=gt_tree.taxon_namespace, internal_nodes_label="int"
        )
        _add_metrics(strategy, _tree_metrics(gt_tree, d_t))

        td = stats["total_decisions"]
        accuracy_rows.append({
            "n_cells": n_cells, "seed": seed, "strategy": strategy,
            "correct_A": stats["correct_A"], "correct_B": stats["correct_B"],
            "correct_pair": stats["correct_pair"],
            "correct_direction": stats["correct_direction"],
            "total_decisions": td,
            "pair_accuracy":      stats["correct_pair"] / td if td else None,
            "side_accuracy":      (stats["correct_A"] + stats["correct_B"]) / (2 * td) if td else None,
            "direction_accuracy": stats["correct_direction"] / td if td else None,
        })

    print(f"  [n={n_cells:>3} seed={seed}] done", flush=True)
    return metrics_rows, accuracy_rows


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def run_experiment(
    n_cells_list,
    seeds,
    out_dir: Path,
    skip_cnasim: bool = False,
    k=None,
    cnasim_exec: str = "cnasim",
    n_workers: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_path  = out_dir / f"{timestamp}_ort_metrics.csv"
    accuracy_path = out_dir / f"{timestamp}_ort_selection_accuracy.csv"

    metrics_fields = ["n_cells", "seed", "strategy", "metric", "dist"]
    accuracy_fields = [
        "n_cells", "seed", "strategy",
        "correct_A", "correct_B", "correct_pair", "correct_direction", "total_decisions",
        "pair_accuracy", "side_accuracy", "direction_accuracy",
    ]

    cnasim2adata = _load_cnasim2adata()

    # ── Phase 1: prepare data per n_cells (sequential: cnasim + matrix build) ──
    global _TASK_DATA
    for n_cells in n_cells_list:
        h5ad_path = out_dir / f"input_n{n_cells}.h5ad"

        if skip_cnasim and h5ad_path.exists():
            print(f"[n={n_cells}] Reusing existing AnnData at {h5ad_path}.")
        else:
            print(f"[n={n_cells}] Running CNAsim...")
            cnasim_dir = _run_cnasim(out_dir / f"cnasim_n{n_cells}", n_cells, cnasim_exec=cnasim_exec)
            print(f"[n={n_cells}] Loading AnnData...")
            adata = cnasim2adata.load_cnasim_output_files(str(cnasim_dir), normalize_counts=True)
            adata.write(h5ad_path)
            print(f"[n={n_cells}] Saved to {h5ad_path}.")

        import anndata as ad
        adata = ad.read_h5ad(h5ad_path)
        print(f"[n={n_cells}] {adata.n_obs} cells loaded.")

        normal_mask = np.asarray(adata.obs["normal"], dtype=bool)
        if normal_mask.sum() == 0:
            raise ValueError("No normal cells in AnnData — cannot fit NLL/Hamming baseline.")

        X = np.asarray(adata.layers["copy"]) if "copy" in adata.layers else (
            adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        )
        observed = X.astype(float)
        healthy  = observed[normal_mask]

        C, A, gt_tree = _ground_truth_ca(adata)
        taxa = list(range(adata.n_obs))
        selection_mats = selection_matrices_from_cn(C, A, observed, healthy)
        k_eff = k if k is not None else round(np.log2(len(taxa)))

        print(f"[n={n_cells}] selection_mats: {list(selection_mats)}  k={k_eff}")
        _TASK_DATA[n_cells] = (C, A, gt_tree.as_string("newick"), taxa, selection_mats, k_eff)

    # ── Phase 2: run (n_cells, seed) pairs ────────────────────────────────────
    tasks = [(n_cells, seed) for n_cells in n_cells_list for seed in seeds]
    n_tasks = len(tasks)
    print(f"\nDispatching {n_tasks} tasks with {n_workers} worker(s)...")
    t0 = time.time()

    if n_workers > 1:
        # fork workers so they inherit _TASK_DATA without pickling
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
            all_results = list(exe.map(_run_one_seed, tasks))
    else:
        all_results = [_run_one_seed(t) for t in tasks]

    print(f"All tasks done in {time.time() - t0:.1f}s.")

    # ── Phase 3: write collected results ──────────────────────────────────────
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metrics_fields)
        w.writeheader()
        for metrics_rows, _ in all_results:
            w.writerows(metrics_rows)
    with open(accuracy_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=accuracy_fields)
        w.writeheader()
        for _, accuracy_rows in all_results:
            w.writerows(accuracy_rows)

    print(f"Metrics CSV:  {metrics_path}")
    print(f"Accuracy CSV: {accuracy_path}")
    return metrics_path, accuracy_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true",
                   help="Quick demo: n_cells=[20,50], 3 seeds.")
    p.add_argument("--n-cells", nargs="+", type=int, default=[50, 100, 200],
                   help="Tumour-cell counts to evaluate.")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)),
                   help="Random seeds (controls SRNJ insertion order).")
    p.add_argument("--out-dir", type=Path, default=Path("./output/ort_selection"),
                   help="Output directory for CSVs and intermediate data.")
    p.add_argument("--skip-cnasim", action="store_true",
                   help="Reuse existing <out-dir>/input_n<N>.h5ad instead of running CNAsim.")
    p.add_argument("--k", type=int, default=None,
                   help="Orienting leaves per side for built-in SRNJ strategies (default: log2 n).")
    p.add_argument("--cnasim-exec", type=str, default="cnasim",
                   help="Path to the cnasim binary (default: 'cnasim', found in PATH).")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel workers for (n_cells, seed) tasks (default: 1).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        n_cells_list = [20, 50]
        seeds = [0, 1, 2]
    else:
        n_cells_list = args.n_cells
        seeds = args.seeds
    run_experiment(
        n_cells_list=n_cells_list,
        seeds=seeds,
        out_dir=args.out_dir.resolve(),
        skip_cnasim=args.skip_cnasim,
        cnasim_exec=args.cnasim_exec,
        k=args.k,
        n_workers=args.n_workers,
    )
