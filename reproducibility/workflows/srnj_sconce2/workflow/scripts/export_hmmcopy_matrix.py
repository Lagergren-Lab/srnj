"""Export per-cell total read-count matrix from input.h5ad for HMMcopy.

Output: headerless CSV (tumor cells × bins), comma-separated chromosome lengths,
and reads-per-copy-per-bin estimate derived from normal cells (l_per_copy.txt).
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input.h5ad")
    parser.add_argument("--output", required=True, help="d_mat.txt (tumor cells x bins)")
    parser.add_argument("--chrom-lengths-output", required=True, help="chrom_lengths.txt")
    parser.add_argument("--l-per-copy-output", required=True,
                        help="l_per_copy.txt: expected reads per copy per bin (for HMMcopy ploidy anchor)")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)

    tumor_mask = ~adata.obs["normal"].astype(bool)
    tumor = adata[tumor_mask]

    counts = np.asarray(tumor.X, dtype=np.float64)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savetxt(args.output, counts, delimiter=",", fmt="%.8f")
    print(f"Wrote {args.output} with shape={counts.shape}")

    # Estimate reads-per-copy-per-bin from normal (diploid) cells.
    # Normal cells have CN=2 everywhere, so mean_reads_per_bin / 2 = l_per_copy.
    # Fall back to median tumor cell mean / 2 if no normal cells are present.
    normal_mask = adata.obs["normal"].astype(bool)
    if normal_mask.sum() > 0:
        normal_counts = np.asarray(adata[normal_mask].X, dtype=np.float64)
        l_per_copy = float(np.mean(normal_counts)) / 2.0
        print(f"l_per_copy estimated from {normal_mask.sum()} normal cells: {l_per_copy:.2f}")
    else:
        l_per_copy = float(np.median(counts.mean(axis=1))) / 2.0
        print(f"No normal cells found; l_per_copy estimated from tumor cell median: {l_per_copy:.2f}")

    os.makedirs(os.path.dirname(args.l_per_copy_output), exist_ok=True)
    with open(args.l_per_copy_output, "w") as f:
        f.write(f"{l_per_copy:.4f}\n")
    print(f"Wrote {args.l_per_copy_output}")

    # Chromosome lengths in genomic order (consecutive runs of same chr in var)
    chroms = list(tumor.var["chr"])
    lengths = []
    current_chr = None
    current_len = 0
    for c in chroms:
        if c != current_chr:
            if current_chr is not None:
                lengths.append(current_len)
            current_chr = c
            current_len = 1
        else:
            current_len += 1
    if current_chr is not None:
        lengths.append(current_len)

    os.makedirs(os.path.dirname(args.chrom_lengths_output), exist_ok=True)
    with open(args.chrom_lengths_output, "w") as f:
        f.write(",".join(str(x) for x in lengths))
    print(f"Wrote chromosome lengths: {lengths}")


if __name__ == "__main__":
    main()
