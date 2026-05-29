"""Heatmap visualization helpers for copy-number and read-count data.

Ported from cellmates_dev.utils.visualization_utils; only the heatmap
functions are included here.  All functions return a ``matplotlib.Figure``
and accept an optional *save_to_folder* path for automatic PDF export.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Total copy-number state palette (from scgenome)
color_reference = defaultdict(
    lambda: '#D4B9DA',
    {0: '#3182BD', 1: '#9ECAE1', 2: '#CCCCCC', 3: '#FDCC8A', 4: '#FC8D59',
     5: '#E34A33', 6: '#B30000', 7: '#980043', 8: '#DD1C77', 9: '#DF65B0',
     10: '#C994C7', 11: '#D4B9DA'},
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, save_to_folder: Optional[str], filename: str) -> None:
    if save_to_folder is not None:
        import os
        os.makedirs(save_to_folder, exist_ok=True)
        fig.savefig(os.path.join(save_to_folder, filename), bbox_inches="tight")


def _get_edge_lengths(tree: nx.DiGraph) -> Dict[tuple, float]:
    return {
        (u, v): d.get("length", d.get("weight", 1.0))
        for u, v, d in tree.edges(data=True)
    }


def _chromosome_boundaries(chromosome_lengths: Optional[List[int]]) -> List[int]:
    if chromosome_lengths is None or len(chromosome_lengths) <= 1:
        return []
    return list(np.cumsum(chromosome_lengths[:-1]))


def _chromosome_segments(
    chromosome_lengths: Optional[List[int]],
) -> Optional[List[Dict[str, Any]]]:
    if chromosome_lengths is None or len(chromosome_lengths) <= 1:
        return None
    segments: List[Dict[str, Any]] = []
    start = 0
    for i, length in enumerate(chromosome_lengths):
        end = start + length
        segments.append({"start": start, "end": end, "mid": start + length / 2.0,
                         "label": f"chr {i + 1}"})
        start = end
    return segments


def _heatmap_leaf_order(
    tree: nx.DiGraph,
    n_cells: int,
    cell_names: Optional[List[str]],
) -> List[int]:
    root = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    dfs_leaves = [
        n for n in nx.dfs_preorder_nodes(tree, root) if tree.out_degree(n) == 0
    ]
    if cell_names is not None:
        name_to_idx = {name: i for i, name in enumerate(cell_names)}
        ordered = [name_to_idx[n] for n in dfs_leaves if n in name_to_idx]
    else:
        ordered = [n for n in dfs_leaves if isinstance(n, int) and 0 <= n < n_cells]
    seen = set(ordered)
    ordered += [i for i in range(n_cells) if i not in seen]
    return ordered


def _cladogram_coords(
    tree: nx.DiGraph,
    leaf_order: List[int],
    cell_names: Optional[List[str]],
) -> tuple:
    root = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    edge_lengths = _get_edge_lengths(tree)

    if cell_names is not None:
        idx_to_leaf = {i: name for i, name in enumerate(cell_names)}
        leaf_nodes = [idx_to_leaf.get(idx, idx) for idx in leaf_order]
    else:
        leaf_nodes = leaf_order

    node_y: dict = {}
    for i, leaf in enumerate(leaf_nodes):
        node_y[leaf] = i + 0.5

    for node in reversed(list(nx.dfs_preorder_nodes(tree, root))):
        if node not in node_y:
            children_y = [node_y[c] for c in tree.successors(node) if c in node_y]
            if children_y:
                node_y[node] = float(np.mean(children_y))

    node_x: dict = {root: 0.0}
    for u, v in nx.bfs_edges(tree, root):
        node_x[v] = node_x.get(u, 0.0) + edge_lengths.get((u, v), 1.0)
    max_x = max(node_x.values()) if node_x else 1.0
    if max_x > 0:
        node_x = {n: x / max_x for n, x in node_x.items()}

    return node_x, node_y


def _draw_cladogram(
    ax: plt.Axes,
    tree: nx.DiGraph,
    leaf_order: List[int],
    cell_names: Optional[List[str]],
    n_cells: int,
) -> None:
    node_x, node_y = _cladogram_coords(tree, leaf_order, cell_names)

    for u, v in tree.edges():
        if u not in node_x or v not in node_x:
            continue
        vx, vy = node_x[v], node_y.get(v, 0)
        ax.plot([node_x[u], vx], [vy, vy], color="k", linewidth=0.7)

    for node in tree.nodes():
        children = list(tree.successors(node))
        if len(children) >= 2 and node in node_x:
            child_ys = [node_y[c] for c in children if c in node_y]
            if child_ys:
                ax.plot([node_x[node], node_x[node]],
                        [min(child_ys), max(child_ys)], color="k", linewidth=0.7)

    root = next(n for n in tree.nodes() if tree.in_degree(n) == 0)
    if root in node_x and root in node_y:
        ax.plot([node_x[root] - 0.05, node_x[root]],
                [node_y[root], node_y[root]], color="k", linewidth=0.7)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(n_cells, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_single_heatmap(
    ax: plt.Axes,
    data_ordered: np.ndarray,
    norm: Any,
    cmap: str,
    vmin: float,
    vmax: float,
    chromosome_lengths: Optional[List[int]],
    ordered_names: Optional[List[str]],
    show_chr_labels: bool,
    panel_title: str,
) -> Any:
    n_cells, n_sites = data_ordered.shape
    mesh = ax.pcolormesh(
        data_ordered, cmap=cmap, norm=norm,
        vmin=(None if norm else vmin),
        vmax=(None if norm else vmax),
        rasterized=True,
    )
    ax.set_ylim(n_cells, 0)
    ax.set_xlim(0, n_sites)

    for x in _chromosome_boundaries(chromosome_lengths):
        ax.axvline(x=x, color="k", linewidth=0.8)

    chr_segs = _chromosome_segments(chromosome_lengths)
    if show_chr_labels and chr_segs is not None:
        ax.set_xticks([seg["mid"] for seg in chr_segs])
        ax.set_xticklabels([str(i + 1) for i in range(len(chr_segs))], fontsize=7)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_title(panel_title, pad=28)
    else:
        ax.set_xticks([])
        if panel_title:
            ax.set_title(panel_title, pad=6)

    if ordered_names is not None:
        ax.set_yticks(np.arange(n_cells) + 0.5)
        ax.set_yticklabels(ordered_names, fontsize=6)
    else:
        ax.set_yticks([])

    return mesh


def _cn_heatmap_discrete_colormap(vmin: float, vmax: float) -> tuple:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lo, hi = int(np.floor(vmin)), int(np.ceil(vmax))
    levels = list(range(lo, hi + 1))
    listed = ListedColormap([color_reference[k] for k in levels])
    norm = BoundaryNorm(np.arange(lo - 0.5, hi + 1.5, 1.0), listed.N)
    return listed, norm


def _plot_heatmap_layout(
    data: np.ndarray,
    *,
    tree: Optional[nx.DiGraph] = None,
    cell_names: Optional[List[str]] = None,
    chromosome_lengths: Optional[List[int]] = None,
    norm: Any,
    cmap: Any,
    vmin: float,
    vmax: float,
    title: str = "",
    figsize: Optional[tuple] = None,
    save_to_folder: Optional[str] = None,
    filename: str = "heatmap.pdf",
    colorbar_label: str = "",
    colorbar_tick_fn: Optional[Callable[[Any], None]] = None,
) -> plt.Figure:
    import matplotlib.gridspec as gridspec

    data = np.asarray(data, dtype=float)
    if data.ndim == 3 and data.shape[2] == 2:
        haplotype_mode = True
        hap_a, hap_b = data[:, :, 0], data[:, :, 1]
        n_cells, n_sites = hap_a.shape
    elif data.ndim == 2:
        haplotype_mode = False
        n_cells, n_sites = data.shape
    else:
        raise ValueError(f"data must be 2-D (N×M) or 3-D (N×M×2), got shape {data.shape}")

    leaf_order = _heatmap_leaf_order(tree, n_cells, cell_names) if tree is not None else list(range(n_cells))
    ordered_names = [cell_names[i] for i in leaf_order] if cell_names is not None else None

    n_rows = 2 if haplotype_mode else 1
    if figsize is None:
        fig_h = max(3.0, n_cells * 0.07 + 1.5) * n_rows
        fig_w = 12.0 + (1.5 if tree is not None else 0.0)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    if tree is not None:
        gs = gridspec.GridSpec(n_rows, 2, width_ratios=[1, 5], wspace=0.02, hspace=0.25)
        axes_tree = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
        axes_heat = [fig.add_subplot(gs[r, 1]) for r in range(n_rows)]
    else:
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.25)
        axes_tree = [None] * n_rows
        axes_heat = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]

    if haplotype_mode:
        panel_configs = [
            (hap_a[leaf_order, :], title, True, "Haplotype A"),
            (hap_b[leaf_order, :], "", False, "Haplotype B"),
        ]
    else:
        panel_configs = [(data[leaf_order, :], title, True, "")]

    meshes = []
    for r, (panel_data, panel_title, show_chr, hap_label) in enumerate(panel_configs):
        mesh = _draw_single_heatmap(
            axes_heat[r], panel_data, norm, cmap, vmin, vmax,
            chromosome_lengths, ordered_names, show_chr, panel_title,
        )
        meshes.append(mesh)
        if hap_label:
            axes_heat[r].set_ylabel(hap_label, fontsize=9, labelpad=4)
        if axes_tree[r] is not None:
            _draw_cladogram(axes_tree[r], tree, leaf_order, cell_names, n_cells)

    cbar = fig.colorbar(meshes[0], ax=axes_heat, fraction=0.02, pad=0.01)
    cbar.set_label(colorbar_label)
    if colorbar_tick_fn is not None:
        colorbar_tick_fn(cbar)

    _save_fig(fig, save_to_folder, filename)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_cn_heatmap(
    data: np.ndarray,
    *,
    tree: Optional[nx.DiGraph] = None,
    cell_names: Optional[List[str]] = None,
    chromosome_lengths: Optional[List[int]] = None,
    vmin: float = 0,
    vmax: float = 6,
    vcenter: Optional[float] = None,
    cmap: str = "RdBu_r",
    title: str = "Copy-number heatmap",
    figsize: Optional[tuple] = None,
    save_to_folder: Optional[str] = None,
    filename: str = "cn_heatmap.pdf",
) -> plt.Figure:
    """Plot an integer copy-number heatmap with optional phylogenetic tree.

    Parameters
    ----------
    data : ndarray, shape (N, M) or (N, M, 2)
        N cells × M bins.  Shape (N, M, 2) is treated as haplotype-specific
        (axis-2: 0=hap A, 1=hap B) and draws two stacked heatmaps.
    tree : nx.DiGraph, optional
        Rooted directed tree.  Rows are sorted by DFS leaf order and a
        cladogram is drawn on the left.  Leaf nodes must be integer row
        indices or strings matching *cell_names*.
    cell_names : list of str, optional
        Row labels (length N).
    chromosome_lengths : list of int, optional
        Number of bins per chromosome for boundary lines and tick labels.
    vmin, vmax : float
        Colormap range (default 0–6).
    title : str
        Figure title.
    figsize : tuple, optional
        ``(width, height)`` in inches; inferred from data shape when omitted.
    save_to_folder : str, optional
        Directory for saving *filename*.
    filename : str
        Output filename (default ``"cn_heatmap.pdf"``).
    """
    if vcenter is None:
        vcenter = (vmin + vmax) / 2.0
    cmap_obj, norm = _cn_heatmap_discrete_colormap(vmin, vmax)

    def _ticks(cbar: Any) -> None:
        lo, hi = int(np.floor(vmin)), int(np.ceil(vmax))
        cbar.set_ticks(list(range(lo, hi + 1)))
        cbar.set_ticklabels([str(t) for t in range(lo, hi + 1)])

    return _plot_heatmap_layout(
        data, tree=tree, cell_names=cell_names,
        chromosome_lengths=chromosome_lengths,
        norm=norm, cmap=cmap_obj, vmin=vmin, vmax=vmax,
        title=title, figsize=figsize,
        save_to_folder=save_to_folder, filename=filename,
        colorbar_label="Copy number", colorbar_tick_fn=_ticks,
    )


def plot_reads_heatmap(
    data: Union[np.ndarray, Mapping[str, Any]],
    *,
    tree: Optional[nx.DiGraph] = None,
    cell_names: Optional[List[str]] = None,
    chromosome_lengths: Optional[List[int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    cmap: str = "viridis",
    title: str = "Read-count heatmap",
    figsize: Optional[tuple] = None,
    save_to_folder: Optional[str] = None,
    filename: str = "reads_heatmap.pdf",
) -> plt.Figure:
    """Plot a read-count heatmap with optional phylogenetic tree.

    Same layout as :func:`plot_cn_heatmap` but uses a continuous colormap.

    Parameters
    ----------
    data : ndarray, shape (N, M)
        Raw or normalised read counts.
    tree : nx.DiGraph, optional
        See :func:`plot_cn_heatmap`.
    cell_names : list of str, optional
        Row labels.
    chromosome_lengths : list of int, optional
        Bins per chromosome.
    vmin, vmax : float, optional
        Colormap range; inferred from data when omitted.
    vcenter : float, optional
        When set uses ``TwoSlopeNorm`` (diverging colormap).
    title : str
        Figure title.
    figsize : tuple, optional
        ``(width, height)`` in inches.
    save_to_folder : str, optional
        Directory for saving *filename*.
    filename : str
        Output filename (default ``"reads_heatmap.pdf"``).
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    norm: Any = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) if vcenter is not None \
        else Normalize(vmin=vmin, vmax=vmax)

    def _ticks(cbar: Any) -> None:
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    return _plot_heatmap_layout(
        arr, tree=tree, cell_names=cell_names,
        chromosome_lengths=chromosome_lengths,
        norm=norm, cmap=cmap, vmin=vmin, vmax=vmax,
        title=title, figsize=figsize,
        save_to_folder=save_to_folder, filename=filename,
        colorbar_label="Reads", colorbar_tick_fn=_ticks,
    )
