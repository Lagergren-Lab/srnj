"""Add HMMcopy CN evaluation rows to existing cn_eval_summary.csv.

Run once on completed results to backfill hmmcopy metrics without re-running
the full Snakemake workflow. Future runs will produce these via evaluate_sconce2_cn.py.

Usage:
    python add_hmmcopy_cn_eval.py <results_dir>
"""
import argparse
import os
import sys
import glob

import anndata
import numpy as np
import pandas as pd


def compute_hmmcopy_metrics(hmmcopy_path, input_ad_path, k):
    adata_tot = anndata.read_h5ad(input_ad_path)
    adata = adata_tot[~adata_tot.obs['normal']]
    gt_cn = np.asarray(adata.layers['state'])
    if gt_cn.ndim == 1:
        gt_cn = gt_cn.reshape(adata.shape)

    hmmcopy_cn = np.loadtxt(hmmcopy_path, dtype=int)
    if hmmcopy_cn.shape != gt_cn.shape:
        print(f"  Shape mismatch: hmmcopy {hmmcopy_cn.shape} vs gt {gt_cn.shape} — skipping")
        return None

    euclidean_dist = np.linalg.norm(hmmcopy_cn.astype(float) - gt_cn.astype(float), axis=1)
    hamming_dist   = np.sum(hmmcopy_cn != gt_cn, axis=1)
    s2_dist        = np.sum((hmmcopy_cn.astype(float) - gt_cn.astype(float))**2, axis=1)
    return {
        'k': k, 'cn_type': 'hmmcopy',
        'euclidean_dist_avg': np.mean(euclidean_dist), 'euclidean_dist_std': np.std(euclidean_dist),
        'euclidean_dist_min': np.min(euclidean_dist),  'euclidean_dist_max': np.max(euclidean_dist),
        'hamming_dist_avg':   np.mean(hamming_dist),   'hamming_dist_std':   np.std(hamming_dist),
        'hamming_dist_min':   np.min(hamming_dist),    'hamming_dist_max':   np.max(hamming_dist),
        's2_dist_avg':        np.mean(s2_dist),        's2_dist_std':        np.std(s2_dist),
        's2_dist_min':        np.min(s2_dist),         's2_dist_max':        np.max(s2_dist),
    }


def parse_sample_dirname(dirname):
    # dirname like: R0_N100_M1000_K6_L5_C10_E0.02
    meta = {}
    for token in dirname.split('_'):
        if token.startswith('R'):   meta['seed']      = token[1:]
        elif token.startswith('N'): meta['n_cells']   = 'N' + token[1:]
        elif token.startswith('M'): meta['n_bins']    = 'M' + token[1:]
        elif token.startswith('K'): meta['n_clones']  = 'K' + token[1:]
        elif token.startswith('L'): meta['lamda']     = token[1:]
        elif token.startswith('C'): meta['n_chrom']   = 'C' + token[1:]
        elif token.startswith('E'): meta['seq_error'] = token[1:]
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir', help='Path to results directory (e.g. srnj_sconce2_K7)')
    parser.add_argument('--k', default='7', help='SCONCE2 k value used (default: 7)')
    args = parser.parse_args()

    data_dir = os.path.join(args.results_dir, 'data')
    summary_path = os.path.join(args.results_dir, 'cn_eval_summary.csv')

    sample_dirs = sorted(glob.glob(os.path.join(data_dir, 'R*_N*_M*')))
    print(f"Found {len(sample_dirs)} sample directories")

    existing = pd.read_csv(summary_path)
    if 'hmmcopy' in existing['cn_type'].dropna().values:
        print("HMMcopy rows already present in summary. Remove them first if you want to recompute.")
        sys.exit(0)

    new_rows = []
    for sd in sample_dirs:
        dirname = os.path.basename(sd)
        hmmcopy_path  = os.path.join(sd, 'hmmcopy_output', 'hmmcopy_inferred.txt')
        input_ad_path = os.path.join(sd, 'input.h5ad')

        if not os.path.exists(hmmcopy_path):
            print(f"  Missing hmmcopy_inferred.txt in {dirname} — skipping")
            continue
        if not os.path.exists(input_ad_path):
            print(f"  Missing input.h5ad in {dirname} — skipping")
            continue

        print(f"Processing {dirname}")
        metrics = compute_hmmcopy_metrics(hmmcopy_path, input_ad_path, args.k)
        if metrics is None:
            continue

        meta = parse_sample_dirname(dirname)
        new_rows.append({**meta, **metrics})

    if not new_rows:
        print("No HMMcopy rows to add.")
        return

    new_df = pd.DataFrame(new_rows)
    col_order = list(existing.columns)
    for c in col_order:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[col_order]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(summary_path, index=False)
    print(f"\nAdded {len(new_rows)} HMMcopy rows. Updated: {summary_path}")


if __name__ == '__main__':
    main()
