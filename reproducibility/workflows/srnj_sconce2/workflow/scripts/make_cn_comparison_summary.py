#!/usr/bin/env python3
"""
Combine per-dataset cn_comparison.pdf files into one summary PDF.
One row per dataset; rasterises each PDF via pdftoppm (poppler) + Pillow.

Usage (discover automatically):
    python make_cn_comparison_summary.py \
        --results-dir results/data \
        --output results/cn_comparison_summary.pdf

Usage (explicit list, e.g. from Snakemake):
    python make_cn_comparison_summary.py \
        --pdfs d1/cn_comparison.pdf d2/cn_comparison.pdf ... \
        --output results/cn_comparison_summary.pdf \
        [--dpi 80] [--rows-per-page 4]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _rasterize_pdf(pdf_path: Path, dpi: int) -> Optional[np.ndarray]:
    if not pdf_path.is_file():
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "p")
            ret = subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-f", "1", "-l", "1",
                 str(pdf_path), prefix],
                capture_output=True, timeout=60,
            )
            if ret.returncode != 0:
                return None
            matches = sorted(Path(tmpdir).glob("p*.ppm"))
            if not matches:
                return None
            return np.asarray(Image.open(matches[0]).convert("RGB"))
    except Exception:
        return None


def _render_page(combos: list[tuple[str, str]], dpi: int,
                 pdf_pages: pdf_backend.PdfPages) -> None:
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, n_rows * 3.5), squeeze=False)
    for row_i, (label, pdf_path) in enumerate(combos):
        ax = axes[row_i][0]
        arr = _rasterize_pdf(Path(pdf_path), dpi=dpi)
        if arr is not None:
            ax.imshow(arr, aspect="auto")
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
        ax.axis("off")
        ax.set_title(label, fontsize=7, loc='left', pad=2)
    plt.tight_layout(pad=0.4)
    pdf_pages.savefig(fig, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results-dir',
                       help='Directory tree containing cn_comparison.pdf files')
    group.add_argument('--pdfs', nargs='+',
                       help='Explicit list of cn_comparison.pdf paths')
    parser.add_argument('--output', required=True, help='Output summary PDF path')
    parser.add_argument('--dpi', type=int, default=80)
    parser.add_argument('--rows-per-page', type=int, default=4)
    args = parser.parse_args()

    if args.pdfs:
        combos = [(os.path.basename(os.path.dirname(p)), p) for p in args.pdfs]
    else:
        pdfs = sorted(Path(args.results_dir).rglob("cn_comparison.pdf"))
        combos = [(p.parent.name, str(p)) for p in pdfs]

    if not combos:
        print("No cn_comparison.pdf files found.")
        return

    print(f"Found {len(combos)} datasets.")
    rows_pp = args.rows_per_page
    n_pages = (len(combos) + rows_pp - 1) // rows_pp
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with pdf_backend.PdfPages(args.output) as pdf:
        for page_i in range(n_pages):
            sl = slice(page_i * rows_pp, (page_i + 1) * rows_pp)
            _render_page(combos[sl], args.dpi, pdf)
            print(f"  page {page_i + 1}/{n_pages}")

    print(f"Wrote {args.output}  ({len(combos)} datasets, {n_pages} pages)")


if __name__ == '__main__':
    main()
