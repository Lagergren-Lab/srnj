import argparse
import anndata
import numpy as np
import os
import sys

from sparsernj.utils.evaluation.benchmarking import get_sconce2_cn_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCONCE2 CN calls against ground truth CN profiles from .h5ad file. Computes pairwise distance matrices and evaluation metrics.")
    parser.add_argument("--input-ad", type=str, required=True, help="Path to input .h5ad file with ground truth CN profiles in adata.layers['state'] and cell names in adata.obs_names")
    parser.add_argument("--sconce2-done", type=str, required=True, help="Path to SCONCE2 completion file, that is assumed to be in the folder with CN calls in .bed files. ")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file for evaluation results")
    parser.add_argument("-k", type=str, default='5', help="k value used in SCONCE2 CN calling, to match the correct bed files")
    parser.add_argument("--hmmcopy-inferred", type=str, default=None,
                        help="Path to hmmcopy_inferred.txt (space-delimited integer matrix, cells × bins, same cell order as input.h5ad tumor cells)")
    args = parser.parse_args()

    adata_tot = anndata.read_h5ad(args.input_ad)
    adata = adata_tot[~adata_tot.obs['normal']]
    n_cells, n_bins = adata.shape
    print("tumor cells and num bins:", n_cells, n_bins)
    # if sconce2 failed, write NaN values for all metrics and exit
    sconce2_status = "success"
    with open(args.sconce2_done, "r") as f:
        sconce2_status = f.read().strip()
    if sconce2_status != "success":
        print(f"SCONCE2 did not complete successfully. Writing NaN values for all metrics in {args.output} and exiting.")
        with open(args.output, 'w') as f:
            f.write("k,cn_type,euclidean_dist_avg,euclidean_dist_std,euclidean_dist_min,euclidean_dist_max,hamming_dist_avg,hamming_dist_std,hamming_dist_min,hamming_dist_max,s2_dist_avg,s2_dist_std,s2_dist_min,s2_dist_max\n")
            f.write(f"{args.k},mode," + ",".join(["NaN"]*12) + "\n")
            f.write(f"{args.k},median," + ",".join(["NaN"]*12) + "\n")
            f.write(f"{args.k},mean," + ",".join(["NaN"]*12) + "\n")
        return
    sconce2_out_dir = os.path.dirname(args.sconce2_done)
    # make cn matrix with shape (n_cells, n_bins) from sconce2 bed files
    
    # Get ground truth data and ensure it's 2D
    gt_cn = adata.layers['state']
    if gt_cn.ndim == 1:
        # If 1D, reshape to match expected (n_cells, n_bins) shape
        gt_cn = gt_cn.reshape(n_cells, -1)
    print(f"Ground truth CN matrix shape: {gt_cn.shape}")
    
    for cn_type in ['mode', 'median', 'mean']:
        sconce_cn, sconce_cell_names = get_sconce2_cn_matrix(sconce2_out_dir, cn_type, adata.obs_names.tolist(), args.k, n_bins=n_bins)
        print(f"SCONCE2 {cn_type} CN matrix shape: {sconce_cn.shape}")
        
        # Ensure both matrices have the same shape
        if sconce_cn.shape != gt_cn.shape:
            print(f"Shape mismatch: SCONCE2 {sconce_cn.shape} vs ground truth {gt_cn.shape}")
            # Try to reshape ground truth to match SCONCE2
            if sconce_cn.size == gt_cn.size:
                gt_cn = gt_cn.reshape(sconce_cn.shape)
                print(f"Reshaped ground truth to: {gt_cn.shape}")
            else:
                raise ValueError(f"Cannot match shapes: SCONCE2 {sconce_cn.shape} vs ground truth {gt_cn.shape}")
        
        # compute Euclidean and Hamming distance matrices between sconce_cn and gt_cn and sum of squares of differences for each cell
        euclidean_dist = np.linalg.norm(sconce_cn - gt_cn, axis=1)
        hamming_dist = np.sum(sconce_cn != gt_cn, axis=1)
        s2_dist = np.sum((sconce_cn - gt_cn)**2, axis=1)
        # avg, std, min, max of distances
        results = {
            'k': args.k,
            'cn_type': cn_type,
            'euclidean_dist_avg': np.mean(euclidean_dist),
            'euclidean_dist_std': np.std(euclidean_dist),
            'euclidean_dist_min': np.min(euclidean_dist),
            'euclidean_dist_max': np.max(euclidean_dist),
            'hamming_dist_avg': np.mean(hamming_dist),
            'hamming_dist_std': np.std(hamming_dist),
            'hamming_dist_min': np.min(hamming_dist),
            'hamming_dist_max': np.max(hamming_dist),
            's2_dist_avg': np.mean(s2_dist),
            's2_dist_std': np.std(s2_dist),
            's2_dist_min': np.min(s2_dist),
            's2_dist_max': np.max(s2_dist)
        }
        # write results to output csv file
        header = ','.join(results.keys())
        values = ','.join(str(v) for v in results.values())
        if not os.path.exists(args.output):
            with open(args.output, 'w') as f:
                f.write(header + '\n')
        with open(args.output, 'a') as f:
            f.write(values + '\n')

    if args.hmmcopy_inferred and os.path.exists(args.hmmcopy_inferred):
        hmmcopy_cn = np.loadtxt(args.hmmcopy_inferred, dtype=int)
        print(f"HMMcopy CN matrix shape: {hmmcopy_cn.shape}")
        if hmmcopy_cn.shape != gt_cn.shape:
            print(f"HMMcopy shape mismatch: {hmmcopy_cn.shape} vs ground truth {gt_cn.shape}. Skipping HMMcopy eval.")
        else:
            euclidean_dist = np.linalg.norm(hmmcopy_cn.astype(float) - gt_cn.astype(float), axis=1)
            hamming_dist = np.sum(hmmcopy_cn != gt_cn, axis=1)
            s2_dist = np.sum((hmmcopy_cn.astype(float) - gt_cn.astype(float))**2, axis=1)
            results = {
                'k': args.k,
                'cn_type': 'hmmcopy',
                'euclidean_dist_avg': np.mean(euclidean_dist),
                'euclidean_dist_std': np.std(euclidean_dist),
                'euclidean_dist_min': np.min(euclidean_dist),
                'euclidean_dist_max': np.max(euclidean_dist),
                'hamming_dist_avg': np.mean(hamming_dist),
                'hamming_dist_std': np.std(hamming_dist),
                'hamming_dist_min': np.min(hamming_dist),
                'hamming_dist_max': np.max(hamming_dist),
                's2_dist_avg': np.mean(s2_dist),
                's2_dist_std': np.std(s2_dist),
                's2_dist_min': np.min(s2_dist),
                's2_dist_max': np.max(s2_dist)
            }
            values = ','.join(str(v) for v in results.values())
            with open(args.output, 'a') as f:
                f.write(values + '\n')

if __name__ == "__main__":
    main()


