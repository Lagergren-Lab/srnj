#!/usr/bin/env python3
"""
Per-dataset CN heatmap comparison: Ground Truth | SCONCE2 (mode) | HMMcopy.

Helper functions ported from cellmates_dev.utils.visualization_utils
(that file must not be modified).
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


# ---------------------------------------------------------------------------
# Palette – ported from cellmates_dev.utils.visualization_utils
# ---------------------------------------------------------------------------

color_reference = defaultdict(
    lambda: '#D4B9DA',
    {0: '#3182BD', 1: '#9ECAE1', 2: '#CCCCCC', 3: '#FDCC8A',
     4: '#FC8D59', 5: '#E34A33', 6: '#B30000', 7: '#980043',
     8: '#DD1C77', 9: '#DF65B0', 10: '#C994C7', 11: '#D4B9DA'}
)


# ---------------------------------------------------------------------------
# Heatmap helpers – ported from cellmates_dev.utils.visualization_utils
# ---------------------------------------------------------------------------

def _chromosome_boundaries(chromosome_lengths):
    if chromosome_lengths is None or len(chromosome_lengths) <= 1:
        return []
    return list(np.cumsum(chromosome_lengths[:-1]))


def _chromosome_segments(chromosome_lengths):
    if chromosome_lengths is None or len(chromosome_lengths) <= 1:
        return None
    segments, start = [], 0
    for i, length in enumerate(chromosome_lengths):
        segments.append({"mid": start + length / 2.0})
        start += length
    return segments


def _cn_discrete_colormap(vmin=0, vmax=6):
    lo, hi = int(np.floor(vmin)), int(np.ceil(vmax))
    cmap = ListedColormap([color_reference[k] for k in range(lo, hi + 1)])
    norm = BoundaryNorm(np.arange(lo - 0.5, hi + 1.5, 1.0), cmap.N)
    return cmap, norm


def _draw_heatmap_panel(ax, data, cmap, norm, chromosome_lengths,
                        title, ytick_labels=None):
    n_cells, n_sites = data.shape
    ax.pcolormesh(np.ma.masked_invalid(data), cmap=cmap, norm=norm)
    ax.set_ylim(n_cells, 0)
    ax.set_xlim(0, n_sites)

    for x in _chromosome_boundaries(chromosome_lengths):
        ax.axvline(x=x, color='k', linewidth=0.5)

    segs = _chromosome_segments(chromosome_lengths)
    if segs:
        ax.set_xticks([s["mid"] for s in segs])
        ax.set_xticklabels([str(i + 1) for i in range(len(segs))], fontsize=6)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_title(title, pad=22, fontsize=10, fontweight='bold')
    else:
        ax.set_xticks([])
        ax.set_title(title, pad=6, fontsize=10, fontweight='bold')

    if ytick_labels is not None:
        ax.set_yticks(np.arange(n_cells) + 0.5)
        ax.set_yticklabels(ytick_labels, fontsize=5)
    else:
        ax.set_yticks([])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ground_truth(input_ad):
    adata_tot = anndata.read_h5ad(input_ad)
    adata = adata_tot[~adata_tot.obs['normal'].astype(bool)]
    return np.asarray(adata.layers['state'], dtype=float), adata.obs_names.tolist()


def load_sconce2_mode(sconce2_dir, cell_names, k='7'):
    bed_files = glob.glob(os.path.join(sconce2_dir, f"*.bed__k{k}__mode.bed"))
    if not bed_files:
        return None
    with open(bed_files[0]) as fh:
        n_bins = sum(1 for _ in fh)
    pattern = re.compile(r'.__(.+?)\.bed__k' + re.escape(str(k)) + r'__mode\.bed$')
    cn = np.full((len(cell_names), n_bins), np.nan)
    for bed_file in bed_files:
        m = pattern.search(os.path.basename(bed_file))
        if not m:
            continue
        cell_name = m.group(1)
        if cell_name not in cell_names:
            continue
        idx = cell_names.index(cell_name)
        with open(bed_file) as fh:
            for bin_idx, line in enumerate(fh):
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    cn[idx, bin_idx] = float(parts[3])
    return cn


def load_hmmcopy_cn(path):
    return np.loadtxt(path, dtype=float)


def load_chrom_lengths(path):
    with open(path) as fh:
        return [int(x) for x in fh.read().strip().split(',')]


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def plot_comparison(gt, sconce2, hmmcopy, chrom_lengths, output,
                    dataset_name, cell_names, vmin=0, vmax=6):
    cmap, norm = _cn_discrete_colormap(vmin, vmax)
    n_cells = gt.shape[0]
    fig_h = max(3.0, n_cells * 0.12 + 2.0)

    fig = plt.figure(figsize=(19, fig_h))
    if dataset_name:
        fig.suptitle(dataset_name, fontsize=10, y=0.99)

    # 3 heatmap columns + narrow colorbar column
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04],
                           wspace=0.07, figure=fig)

    panels = [
        (gt,      "Ground Truth"),
        (sconce2 if sconce2 is not None else np.full_like(gt, np.nan),
         "SCONCE2 (mode)"),
        (hmmcopy, "HMMcopy"),
    ]

    for col, (data, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        _draw_heatmap_panel(ax, data, cmap, norm, chrom_lengths, title,
                            ytick_labels=cell_names if col == 0 else None)
        if sconce2 is None and col == 1:
            ax.text(0.5, 0.5, "not converged", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray',
                    style='italic')

    cbar_ax = fig.add_subplot(gs[0, 3])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Copy number", fontsize=8)
    cbar.set_ticks(list(range(vmin, vmax + 1)))
    cbar.set_ticklabels([str(i) for i in range(vmin, vmax + 1)], fontsize=7)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Ground Truth | SCONCE2 (mode) | HMMcopy CN heatmaps."
    )
    parser.add_argument('--input-ad', required=True,
                        help='input.h5ad with ground truth in layers["state"]')
    parser.add_argument('--sconce2-dir', required=True,
                        help='Directory containing SCONCE2 mode .bed files')
    parser.add_argument('--hmmcopy-inferred', required=True,
                        help='hmmcopy_inferred.txt (space-delimited cells × bins)')
    parser.add_argument('--chrom-lengths', required=True,
                        help='chrom_lengths.txt (comma-separated bin counts per chr)')
    parser.add_argument('--output', required=True, help='Output PDF path')
    parser.add_argument('--dataset-name', default='')
    parser.add_argument('-k', default='7', help='SCONCE2 k value (default: 7)')
    args = parser.parse_args()

    gt, cell_names = load_ground_truth(args.input_ad)
    sconce2 = load_sconce2_mode(args.sconce2_dir, cell_names, k=args.k)
    hmmcopy = load_hmmcopy_cn(args.hmmcopy_inferred)
    chrom_lengths = load_chrom_lengths(args.chrom_lengths)

    if sconce2 is None:
        print(f"Warning: no SCONCE2 k={args.k} mode bed files in {args.sconce2_dir}")

    dataset_name = args.dataset_name or os.path.basename(os.path.dirname(args.input_ad))
    plot_comparison(gt, sconce2, hmmcopy, chrom_lengths, args.output,
                    dataset_name, cell_names)


if __name__ == '__main__':
    main()
