"""Export per-cell total read-count matrix from input.h5ad for HMMcopy.

Output: headerless CSV (tumor cells × bins) and comma-separated chromosome lengths.
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
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)

    tumor_mask = ~adata.obs["normal"].astype(bool)
    tumor = adata[tumor_mask]

    counts = np.asarray(tumor.X, dtype=np.float64)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savetxt(args.output, counts, delimiter=",", fmt="%.8f")
    print(f"Wrote {args.output} with shape={counts.shape}")

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
