"""Convert HMMcopy inferred CN matrix (cells × bins) to MEDICC2 long-format TSV."""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd


def _chr_name(zero_based_idx: int) -> str:
    return f"chr{zero_based_idx + 1}"


def _medicc2_sample_id(x: str) -> str:
    return f"m{x}" if x.isdigit() else x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inferred", required=True, help="HMMcopy output: integer CN matrix (cells x bins)")
    parser.add_argument("--input-ad", required=True, help="input.h5ad (for cell IDs and chromosome structure)")
    parser.add_argument("--output", required=True, help="MEDICC2 input TSV")
    args = parser.parse_args()

    inferred = np.loadtxt(args.inferred)
    if inferred.ndim == 1:
        inferred = inferred[np.newaxis, :]
    inferred = np.asarray(inferred, dtype=np.int32)

    adata = ad.read_h5ad(args.input_ad)
    tumor_mask = ~adata.obs["normal"].astype(bool)
    tumor = adata[tumor_mask]

    cell_ids = list(tumor.obs_names)
    n_cells, n_bins = inferred.shape

    if n_cells != len(cell_ids):
        raise ValueError(f"Inferred rows ({n_cells}) != number of tumor cells ({len(cell_ids)})")

    # Chromosome index for each bin (by consecutive genomic order in var)
    chroms = list(tumor.var["chr"])
    chr_labels: list[int] = []
    chr_idx = -1
    current_chr = None
    for c in chroms:
        if c != current_chr:
            chr_idx += 1
            current_chr = c
        chr_labels.append(chr_idx)
    chr_names = [_chr_name(i) for i in chr_labels]

    starts = np.arange(n_bins, dtype=np.int64)
    ends = starts + 1

    rows = []
    for i, cell_id in enumerate(cell_ids):
        sid = _medicc2_sample_id(cell_id)
        for j in range(n_bins):
            rows.append({
                "sample_id": sid,
                "chrom": chr_names[j],
                "start": int(starts[j]),
                "end": int(ends[j]),
                "total_cn": int(inferred[i, j]),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {args.output} ({len(df)} rows, {df['sample_id'].nunique()} cells)")


if __name__ == "__main__":
    main()
