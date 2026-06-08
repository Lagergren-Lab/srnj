"""
Helpers for CNASim data processing.
"""
import argparse
import logging
import os
import ast
from pathlib import Path
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import anndata

# Import tree parsing functions from srnj source
sys.path.insert(0, os.path.abspath("../../../../../src"))
from sparsernj.utils.tree_utils import newick_to_nx

# ==================================
# Main functions (read CNAsim files)
# ==================================
def convert_cnasim_output_to_anndata(cnasim_data_dir: str, out_path: str = None, normalize_counts: bool = True, copynumbers_only: bool = False) -> str:
    """
    Convert CNAsim output files to anndata object and save it to out_path
    """
    if copynumbers_only:
        adata = profiles_to_anndata(os.path.join(cnasim_data_dir, 'profiles.tsv'))
        add_sc_tree(adata, cnasim_data_path=cnasim_data_dir)
    else:
        adata = load_cnasim_output_files(cnasim_data_dir, normalize_counts)
    if out_path is None:
        out_path = os.path.join(cnasim_data_dir, 'adata.h5ad')

    if os.path.exists(os.path.join(cnasim_data_dir, 'clean_profiles.tsv')):
        noisy_adata = profiles_to_anndata(os.path.join(cnasim_data_dir, 'profiles.tsv'), 'states')
        adata.layers['noisy-cn'] = noisy_adata.layers['states']

    adata.write(Path(out_path))
    return out_path

def get_sc_tree_leaves_only(cnasim_data_path):
    """
    Parse tree.nwk file that only has labels for leaves (cells).
    e.g. ((leaf1:0.08,((leaf3:0.024,leaf4:0.024):0.055)root;
    """
    tree_file = os.path.join(cnasim_data_path, 'tree.nwk')
    tree_nx = _parse_newick(tree_file)
    return tree_nx

def get_sc_tree(cnasim_data_path: str, cell_types_df: pd.DataFrame) -> nx.DiGraph:
    """
    Parse tree.nwk file to get single cell tree in networkx format
     add the clone assignment id as node attribute 'clone', and 'is_founder' for founder nodes.
    Clones are either 'normal' or 'founder' (progenitor clone) and 'cloneX' for the rest.
    Parameters:
    - cnasim_data_path: path to CNAsim output directory
    - cell_types_df: DataFrame with ancestor to clone assignment (ancestor, clone, is_cell, is_founder, clone_id, clone_int)
    """
    tree_file = os.path.join(cnasim_data_path, 'tree.nwk')
    tree_nx = _parse_newick(tree_file)
    # set nodes attributes to clone founders
    for i, row in cell_types_df.iterrows():
        n = row['ancestor']
        tree_nx.nodes[n]['clone'] = row['clone_id']
        tree_nx.nodes[n]['is_founder'] = row['is_founder']
    return tree_nx

def read_log_params(cnasim_data_path: str) -> dict:
    params = {}
    with open(os.path.join(cnasim_data_path, 'log.txt'), 'r') as f:
        for line in f:
            if line != '\n':
                k, v = line.strip().split(':')
                try:
                    if k == 'out_path':
                        params[k] = v
                    else:
                        params[k] = ast.literal_eval(v)
                except SyntaxError:
                    params[k] = v.strip()
                except ValueError as e:
                    print(f"Error parsing value {v} for key {k}: {e}")
    return params

def read_cell_types(cnasim_data_path: str) -> (np.ndarray, pd.DataFrame):
    """
    Parse cell_types.tsv file to get cell to clone assignment, and ancestor to clone assignment.
    CNAsim generates a normal, a founder clone and a number of clones (clones1, clone2, ...).
    root is the ancestor associated with the normal clone and the founder is the first clone in the list.
    Cell labeling follows the same pattern: cell1, cell2, ... cellN with idx starting from 0.
    Clones will be sorted by their id number but remapped to a range(num_clones) with
    clone 0 being the normal clone and clone 1 the founder clone.

    If CNAsim is run with no clones, then all tumor cells belong to the one clone named 'aneuploid'.
    If only tumor cells are simulated, then all cells belong to the founder clone.

    Returns:
    - cell_assignment: a 1D array with the clone id for each cell
    - ancestor_clone_map: a dictionary mapping ancestor names to clone ids
    """
    clone_id_map = {'normal': 0, 'founder': 1}

    cell_types = pd.read_csv(os.path.join(cnasim_data_path, 'cell_types.tsv'), delimiter='\t', names=['ancestor', 'clone'])
    cell_types['is_cell'] = cell_types['ancestor'].transform(lambda x: 'cell' in x)
    cell_types['is_founder'] = cell_types['clone'].transform(lambda x: 'founder' in x)
    cell_types['clone'] = cell_types['clone'].transform(lambda x: x.split('_')[0])  # remove '_founder' suffix
    cell_types['clone_int'] = cell_types['clone'].transform(lambda x: int(x[5:]) if 'clone' in x else -1)

    # reassign a clone id to make sure they are in range(num_clones)
    # unique will merge normal with founder (which has id -1 for convenience)
    unique_clones = cell_types['clone_int'].sort_values().unique().tolist()
    if 'aneuploid' in cell_types['clone'].values:
        # all cells belong to the same clone
        clone_id_map['aneuploid'] = 2
    else:
        # map clones to new ids starting from 2 (clone0: normal, clone1: founder)
        for i, c in enumerate(unique_clones):
            if c == -1:
                continue
            clone_name = f'clone{c}'
            clone_id_map[clone_name] = i + 2 # normal is 0, founder is 1

    cell_types['clone_id'] = cell_types['clone'].transform(lambda x: clone_id_map[x])
    # map cell to clone
    cell_assignment = np.zeros(cell_types['is_cell'].sum(), dtype=int)
    for i, row in cell_types[cell_types['is_cell']].iterrows():
        cell_id = int(row['ancestor'][4:]) - 1  # cell1 -> 0, cell2 -> 1, ...
        cell_assignment[cell_id] = clone_id_map[row['clone']]

    return cell_assignment, cell_types

def profiles_to_anndata(profiles_path: str, layer_name: str = 'state') -> anndata.AnnData:
    """
    Convert CNAsim profiles.tsv as in the benchmark files into anndata object
    profiles.tsv format:
    CELL	chrom	start	end	CN states
    leaf1    chr_1    0	1000000	1,1
    // or
    cell1    chr1    0 ...
    """
    # read profiles.tsv file and make anndata with obs_names and var_names
    profiles_df = pd.read_csv(profiles_path, delimiter='\t', header=0)
    # mutate copy number
    profiles_df['cnA'] = profiles_df['CN states'].transform(lambda x: int(x.split(',')[0]))
    profiles_df['cnB'] = profiles_df['CN states'].transform(lambda x: int(x.split(',')[1]))
    profiles_df['cn'] = profiles_df['cnA'] + profiles_df['cnB']
    # to wide format
    wide_cn_df = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values=['cn', 'cnA', 'cnB'], columns='CELL',
                                sort=False)  # already sorted by cell number
    # extract cell columns
    cell_names = wide_cn_df['cn'].columns.tolist()
    cn_profiles = wide_cn_df['cn'][cell_names].transpose().to_numpy()
    cn_profilesA = wide_cn_df['cnA'][cell_names].transpose().to_numpy()
    cn_profilesB = wide_cn_df['cnB'][cell_names].transpose().to_numpy()
    adata = anndata.AnnData(cn_profiles)
    adata.layers[layer_name] = cn_profiles
    adata.layers[layer_name + 'A'] = cn_profilesA
    adata.layers[layer_name + 'B'] = cn_profilesB
    adata.obs_names = cell_names
    adata.var_names = wide_cn_df.index.map(lambda x: f"{x[0]}:{x[1]}-{x[2]}")
    # prefix can be 'chr' or 'chr_' and chromosome names can be '1', ... '22', 'X', 'Y'
    pattern = re.compile(r'^chr[_]?([0-9]+|X|Y)$')
    adata.var['chr'] = wide_cn_df.index.map(lambda x: re.findall(pattern, x[0]).pop())
    adata.var['start'] = wide_cn_df.index.map(lambda x: x[1])
    adata.var['end'] = wide_cn_df.index.map(lambda x: x[2])
    return adata

def read_cn_profiles(adata: anndata.AnnData, cnasim_data_path: str, n_cells: int, inplace=True) -> np.ndarray:
    """
    If inplace=True, adds layers ['state', 'Astate', 'Bstate'] to the anndata
    Returns the total copy number state
    """
    profiles_file = 'clean_profiles.tsv' if os.path.exists(os.path.join(cnasim_data_path, 'clean_profiles.tsv')) else 'profiles.tsv'
    profiles_df = pd.read_csv(os.path.join(cnasim_data_path, profiles_file), delimiter='\t', header=0)
    # mutate copy number
    profiles_df[['Acn', 'Bcn']] = profiles_df['CN states'].str.split(',', expand=True).astype(int)
    profiles_df['cn'] = profiles_df['Acn'] + profiles_df['Bcn']
    #profiles_df['cn'] = profiles_df['CN states'].transform(lambda x: sum(list(map(int, x.split(',')))))

    wide_cn_df = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='cn', columns='CELL',
                                sort=False)  # already sorted by cell number
    # extract cell columns sorted by cell number
    cells = [f'cell{i+1}' for i in range(n_cells)]
    cn_profiles = wide_cn_df[cells].transpose().to_numpy()

    if inplace:
        adata.layers['state'] = cn_profiles
        adata.layers['Astate'] = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='Acn', columns='CELL',
                                                sort=False)[cells].transpose().to_numpy()
        adata.layers['Bstate'] = pd.pivot_table(profiles_df, index=['chrom', 'start', 'end'], values='Bcn', columns='CELL',
                                                sort=False)[cells].transpose().to_numpy()
    # add ancestral profiles if available
    # format:
    # CELL	chrom	start	end	CN states
    # ancestor1	chr1	0	1000000	1,1
    anc_prof_file = 'ancestral_profiles.tsv'
    if os.path.exists(os.path.join(cnasim_data_path, anc_prof_file)) and inplace:
        anc_profiles_df = pd.read_csv(os.path.join(cnasim_data_path, anc_prof_file), delimiter='\t', header=0)
        anc_profiles_df[['Acn', 'Bcn']] = anc_profiles_df['CN states'].str.split(',', expand=True).astype(int)
        anc_profiles_df['cn'] = anc_profiles_df['Acn'] + anc_profiles_df['Bcn']
        wide_anc_cn_df = pd.pivot_table(
            anc_profiles_df,
            index=['chrom', 'start', 'end'],
            values=['cn', 'Acn', 'Bcn'],
            columns='CELL',
            sort=False
        )  # already sorted by cell number
        anc_cells = wide_anc_cn_df['cn'].columns.tolist()
        anc_cn_profiles = wide_anc_cn_df['cn'][anc_cells].transpose().to_numpy()
        # concatenate to existing profiles
        adata.uns['ancestral-names'] = anc_cells
        adata.uns['ancestral-cn'] = anc_cn_profiles
        adata.uns['ancestral-cnA'] = wide_anc_cn_df['Acn'][anc_cells].transpose().to_numpy()
        adata.uns['ancestral-cnB'] = wide_anc_cn_df['Bcn'][anc_cells].transpose().to_numpy()
    else:
        raise FileNotFoundError(f"Ancestral profiles file {anc_prof_file} not found in {cnasim_data_path}. It is required to have ancestral profiles for the cell tree.")

    return cn_profiles

def correct_readcounts(adata, min_normal_cells=1, inplace=True) -> np.ndarray:
    """
    Correct read counts, normalizing the counts using normal cells as reference. If normal cells are not
    available, use cnasim parameters to compute mean reads per bin per copy and normalize accordingly.
    Requires the `normal` column in adata.obs to be set.
    """
    normal_cn = 2.
    if not adata.uns['cnasim-params']['use_uniform_coverage']:
        logging.warning("CNAsim data was simulated with non-uniform coverage. Read count correction may be inaccurate.")

    if 'normal' not in adata.obs or adata.obs['normal'].sum() < min_normal_cells:
        # use cnasim params to compute mean read counts per bin per copy
        baseline = adata.uns['cnasim-params']['bin_length'] * adata.uns['cnasim-params']['coverage'] / (adata.uns['cnasim-params']['read_length'] * 2 * normal_cn)
    else:
        baseline = adata[adata.obs['normal']].X.mean(axis=0) / normal_cn # counts per copy

    copy = adata.X / baseline

    if 'Acount' in adata.layers and inplace:
        # normalize also phased data
        adata.layers['Acopy'] = adata.layers['Acount'] / baseline
        adata.layers['Bcopy'] = adata.layers['Bcount'] / baseline

    if inplace:
        adata.layers['copy'] = copy

    return adata.layers['copy']

def load_counts_init_anndata(cnasim_data_path: str):
    # manipulate to extract (chr, start, pos) dataframe - vars for anndata
    readcounts_df = pd.read_csv(os.path.join(cnasim_data_path, 'readcounts.tsv'), delimiter='\t', header=0)
    readcounts_df['chr'] = readcounts_df['chrom'].transform(lambda x: x[3:])
    var_df = readcounts_df[readcounts_df['CELL'] == 'cell1'][['chr', 'start', 'end']].reset_index(drop=True)
    var_df['chr'] = pd.Categorical(var_df['chr'], ordered=True)
    var_df.set_index(var_df[['chr', 'start', 'end']].apply(lambda x: f"{x['chr']}:{x['start']}-{x['end']}", axis=1),
                     inplace=True)

    # sum Acount + Bcount = readcount
    is_phased = 'readcount' not in readcounts_df.columns
    if is_phased:
        readcounts_df['readcount'] = readcounts_df['Acount'] + readcounts_df['Bcount']

    # extract readcounts as a n_cells x n_bins matrix
    # to wide format
    wide_counts_df = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='readcount', columns='CELL',
                                    sort=False)
    # extract cell columns
    cell_names = [c for c in wide_counts_df.columns if 'cell' in c]
    counts = wide_counts_df[cell_names].transpose().to_numpy()

    adata = anndata.AnnData(counts)
    adata.obs_names = cell_names
    adata.var_names = var_df.index
    adata.var = var_df
    adata.uns['cnasim-params'] = read_log_params(cnasim_data_path)

    if is_phased:
        # if phasing is available, then make layers for Acount and Bcount
        adata.layers['Acount'] = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='Acount', columns='CELL',
                                                sort=False)[cell_names].transpose().to_numpy()
        adata.layers['Bcount'] = pd.pivot_table(readcounts_df, index=['chrom', 'start', 'end'], values='Bcount', columns='CELL',
                                                sort=False)[cell_names].transpose().to_numpy()
    return adata


def add_sc_tree(adata, cnasim_data_path: str = None) -> None:
    tree_nx = get_sc_tree_leaves_only(cnasim_data_path)
    cell_tree_newick = _tree_to_newick(tree_nx, root='root', weight='weight') + ";"
    adata.uns['cell-tree-newick'] = cell_tree_newick


def add_tree(adata, cnasim_data_path: str = None, cell_types_df: pd.DataFrame = None) -> None:
    tree_nx = get_sc_tree(cnasim_data_path, cell_types_df)
    clonal_tree_nx = _collapse_equal_clones(tree_nx, clone_attr='clone')
    remapped_clonal_tree_nx = nx.relabel_nodes(clonal_tree_nx, {n: clonal_tree_nx.nodes[n]['clone'] for n in clonal_tree_nx.nodes()})
    # when normal cells are not simulated, set founder as root (tree structure changes)
    root_node = 'root' if np.any(adata.obs['normal']) else 'founder'
    root_node_id = 0 if root_node == 'root' else 1

    cell_tree_newick = _tree_to_newick(tree_nx, root=root_node, weight='weight') + ";"
    clonal_tree_newick = _tree_to_newick(clonal_tree_nx, root=root_node, weight='weight') + ";"
    clone_id_tree_nwk = _tree_to_newick(remapped_clonal_tree_nx, root=root_node_id, weight='weight') + ";"

    adata.uns['cell-tree-newick'] = cell_tree_newick
    adata.uns['clonal-tree-newick'] = clonal_tree_newick
    adata.uns['clone-id-tree-newick'] = clone_id_tree_nwk


def load_cnasim_output_files(cnasim_data_path: str | Path, normalize_counts: bool = True) -> anndata.AnnData:
    """
    Load CNAsim output files into anndata object. The annData object will contain:
    - X: read counts matrix (n_cells x n_bins)
    - layers['copy']: normalized read counts matrix (n_cells x n_bins)
    - layers['state']: copy number state matrix (n_cells x n_bins)
    - obs: cell metadata (normal: bool, clone: categorical)
    - var: bin metadata (chr, start, end)
    - uns: additional data (cnasim-params, cell-tree-newick, clonal-tree-newick, clone-id-tree-newick)
    Parameters:
    ----------
    cnasim_data_path: str, path to CNAsim output directory
    Returns:
    ----------
    anndata.AnnData object
    """
    # 1. load readcounts.tsv and init anndata
    adata = load_counts_init_anndata(cnasim_data_path)
    n_cells = adata.n_obs
    n_bins = adata.n_vars
    params = adata.uns['cnasim-params']  # load simulation params
    # 2. load clean_profiles.tsv / profiles.tsv
    _ = read_cn_profiles(adata, cnasim_data_path, n_cells=n_cells, inplace=True)
    # 3. load cell assignment cell_type.tsv
    cell_assignment, cell_types_df = read_cell_types(cnasim_data_path)
    assert len(cell_assignment) == n_cells, f"Cell assignment length {len(cell_assignment)} does not match n_cells {n_cells}"

    adata.obs_names = [f'cell{i+1}' for i in range(n_cells)]
    adata.obs['normal'] = cell_assignment == 0  # normal is always 0
    adata.obs['clone'] = pd.Categorical(cell_assignment)
    # 3.1 normalize read counts
    if normalize_counts:
        _ = correct_readcounts(adata, inplace=True) # inplace adding layers: ['copy', 'Acopy', 'Bcopy']
    # for each clone, select its founder
    # 4. read and process tree.nwk
    # print(clone_founder)  # ['root', 'ancestor2', 'ancestor19', ...]
    # print(clone_id_map)  # {'normal': 0, 'clone6': 6, 'clone2': 2, ...}
    add_tree(adata, cnasim_data_path, cell_types_df)
    return adata

# =======================================
# Tree manipulation (copied from VICTree)
# =======================================
def _collapse_equal_clones(tree: nx.DiGraph, clone_attr: str = 'clone') -> nx.DiGraph:
    """
    Collapse tree nodes to only contain founder clones and normal clone.
    If a node has the same clone as its parent, then the node is removed and its children are connected to the parent.
    """
    def parent_same_clone(x):
        parent = next(tree.predecessors(x))
        return tree.nodes[parent][clone_attr] == tree.nodes[x][clone_attr]
    collapsed_tree = _collapse_tree_if(tree, parent_same_clone)
    return collapsed_tree

def _collapse_tree_if(tree: nx.DiGraph, condition: callable) -> nx.DiGraph:
    """
    Collapse tree nodes to only contain nodes that satisfy the condition on the node.
    If a node satisfies the condition, then the node is removed and its children are connected to the parent.
    """
    collapsed_tree = tree.copy()
    nodes = list(collapsed_tree.nodes())
    for n in nodes:
        for parent in collapsed_tree.predecessors(n):
            if condition(n):
                children = list(collapsed_tree.successors(n))
                for c in children:
                    collapsed_tree.add_edge(parent, c, weight=collapsed_tree[parent][n]['weight'] + collapsed_tree[n][c]['weight'])
                collapsed_tree.remove_node(n)
    return collapsed_tree

def _tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = _tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)

def _parse_newick(tree_file):
    """
    Parameters
    ----------
    tree_file : filepath or file-like
        If a Newick string is desired, pass StringIO(newick_string) instead.

    Returns
    -------
    nx.DiGraph
        Directed NetworkX tree.
    """
    # Read the newick file content
    if hasattr(tree_file, 'read'):
        # It's a file-like object (StringIO)
        newick_str = tree_file.read()
    else:
        # It's a file path
        with open(tree_file, 'r') as f:
            newick_str = f.read().strip()
    
    tree_nx = newick_to_nx(newick_str, edge_attr='weight')
    
    return tree_nx


def plot_data(ad, title, outfile, layer_name):
    """
    Plot cell CN matrix with tree. Uses scgenome for plotting.
    Falls back gracefully if Bio.Phylo is not available.
    """
    try:
        import io
        import scgenome.plotting as pl
        import matplotlib.pyplot as plt
        
        tree_nwk = ad.uns['cell-tree-newick']
        
        # Try to use Bio.Phylo if available, otherwise skip tree plotting
        try:
            from Bio import Phylo
            biotree = Phylo.read(io.StringIO(tree_nwk), "newick")
            raw = layer_name is None
            g = pl.plot_cell_cn_matrix_fig(ad, tree=biotree, layer_name=layer_name, raw=raw)
        except ImportError:
            logger.warning("Bio.Phylo not available, plotting without tree")
            raw = layer_name is None
            g = pl.plot_cell_cn_matrix_fig(ad, tree=None, layer_name=layer_name, raw=raw)

        g['fig'].suptitle(title)
        g['fig'].savefig(outfile, dpi=300)
        plt.close(g['fig'])
        
    except ImportError as e:
        logger.warning(f"Plotting dependencies not available ({e}), skipping plot generation")
        # Create empty plot file to satisfy workflow expectations
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Plot skipped: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(outfile, dpi=300)
        plt.close(fig)

# =======================
# Command line interface
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    cli = argparse.ArgumentParser(
        description="Convert CNAsim output files to anndata object."
    )
    cli.add_argument('-i', '--input', type=str, required=True, help="CNAsim files directory (where `readcounts.tsv`, `profiles.tsv`, `cell_types.tsv` and `tree.nwk` are located)")
    cli.add_argument('-o', '--output', type=str, default=None, help="output file path e.g. ./datasets/dat.h5ad")
    cli.add_argument('--counts-only', action='store_true', help="Only load read counts without normalization or copy number states")
    cli.add_argument('--copynumbers-only', action='store_true', help="Only load copy number states without read counts")
    cli.add_argument('--plot', action='store_true', help="Plot the dataset with tree (requires scgenome)")
    args = cli.parse_args()

    convert_cnasim_output_to_anndata(args.input, args.output, normalize_counts=not args.counts_only, copynumbers_only=args.copynumbers_only)
    if args.plot:
        logger.info("Plotting dataset with tree (cn and read counts)")
        adata = anndata.read_h5ad(args.output)
        plot_data(adata, args.output, args.output.replace('.h5ad', '_cn.png'), layer_name='state')
        plot_data(adata, args.output, args.output.replace('.h5ad', '_X.png'), layer_name=None)
    logger.info(f"Wrote {args.output}")

if __name__ == "__main__":
    main()

