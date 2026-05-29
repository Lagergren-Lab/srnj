"""
Dataset inspection plots for the ort_selection experiment.

For each n_cells value present in the results directory (first seed only),
produces a two-page layout in a single PDF:
  page 1 – copy-number heatmap (GT tree sorted, chromosome-annotated)
  page 2 – read-count heatmap  (same ordering)

Usage
-----
    python plot_dataset.py [--results-dir PATH] [--out OUT.pdf]

Default results dir:  ../../../../results  (symlink → srnj_results/ort_test)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import dendropy as dpy
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ── Path setup ─────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
for _candidate in (_THIS.parent, _THIS.parent.parent,
                   _THIS.parent.parent.parent, _THIS.parent.parent.parent.parent):
    if (_candidate / "src").is_dir():
        sys.path.insert(0, str(_candidate / "src"))
        break

from sparsernj.utils.algorithms.neighbor_joining import dlca_nj, split_tdm
from sparsernj.utils.visualization import plot_cn_heatmap, plot_reads_heatmap

# Reuse _ground_truth_ca from the experiment module
_EXP = _THIS.parent / "ort_selection_accuracy.py"
_spec = importlib.util.spec_from_file_location("ort_exp", _EXP)
_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp)
_ground_truth_ca = _exp._ground_truth_ca


def _chromosome_lengths(adata: ad.AnnData):
    """Return list of bin-counts per chromosome, or None."""
    chr_col = next((c for c in ("chrom", "chr", "chromosome") if c in adata.var.columns), None)
    if chr_col is None:
        return None
    return adata.var[chr_col].value_counts(sort=False).tolist()


def _build_gt_nx(adata: ad.AnnData):
    """Return nx.DiGraph of the GT-tree-reconstructed via DLCA-NJ (integer leaf labels)."""
    C, A, _ = _ground_truth_ca(adata)
    taxa = list(range(adata.n_obs))
    return dlca_nj(C, A, taxa=taxa, collapsed_root=False, root_label="root")


def _reads_array(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return arr.astype(float)


def _copy_array(adata: ad.AnnData) -> np.ndarray:
    if "copy" in adata.layers:
        return np.asarray(adata.layers["copy"], dtype=float)
    return _reads_array(adata)


def plot_one_dataset(adata: ad.AnnData, pdf: PdfPages, label: str) -> None:
    """Add two pages to *pdf* for this dataset (CN heatmap + reads heatmap)."""
    print(f"  Building GT tree...")
    try:
        nx_tree = _build_gt_nx(adata)
    except Exception as e:
        print(f"  WARNING: GT tree failed ({e}); plotting without tree.")
        nx_tree = None

    chr_lengths = _chromosome_lengths(adata)
    n_cells = adata.n_obs

    # ── page 1: copy-number heatmap ──────────────────────────────────────
    cn = _copy_array(adata)
    vmax_cn = max(6, int(np.nanpercentile(cn[np.isfinite(cn)], 99)) + 1)
    fig_cn = plot_cn_heatmap(
        cn,
        tree=nx_tree,
        chromosome_lengths=chr_lengths,
        vmin=0, vmax=vmax_cn,
        title=f"Copy-number – {label} ({n_cells} cells, seed 0)",
    )
    pdf.savefig(fig_cn, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig_cn)
    print(f"  CN heatmap done.")

    # ── page 2: read-count heatmap ───────────────────────────────────────
    reads = _reads_array(adata)
    fig_rd = plot_reads_heatmap(
        reads,
        tree=nx_tree,
        chromosome_lengths=chr_lengths,
        title=f"Read counts – {label} ({n_cells} cells, seed 0)",
    )
    pdf.savefig(fig_rd, bbox_inches="tight")
    plt.close(fig_rd)
    print(f"  Reads heatmap done.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-dir", type=Path,
        default=_THIS.parent.parent.parent.parent / "results",
        help="Directory containing input_n*.h5ad files (default: ../../../../results).",
    )
    p.add_argument(
        "--out", type=Path,
        default=None,
        help="Output PDF path (default: <results-dir>/dataset_inspect.pdf).",
    )
    args = p.parse_args()

    results_dir = args.results_dir.resolve()
    out_path = args.out or (results_dir / "dataset_inspect.pdf")

    h5ad_files = sorted(results_dir.glob("input_n*.h5ad"))
    if not h5ad_files:
        print(f"No input_n*.h5ad files found in {results_dir}")
        return

    print(f"Found {len(h5ad_files)} dataset(s): {[f.name for f in h5ad_files]}")
    print(f"Output: {out_path}")

    with PdfPages(out_path) as pdf:
        for h5ad_path in h5ad_files:
            label = h5ad_path.stem.replace("input_", "")
            print(f"\nProcessing {h5ad_path.name} ...")
            adata = ad.read_h5ad(h5ad_path)
            plot_one_dataset(adata, pdf, label)

    print(f"\nAll done → {out_path}")


if __name__ == "__main__":
    main()
