#!/bin/bash
#SBATCH --account=naiss2026-3-168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=srnj_ort_sel
#SBATCH --output=/proj/sc_ml/users/x_vitza/srnj_results/ort_test/slurm_%j.out

set -euo pipefail

# Single-threaded BLAS/OpenMP per worker — we provide parallelism at the process level
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load Miniforge/24.7.1-2-hpc1

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT_DIR=/proj/sc_ml/users/x_vitza/srnj_results/ort_test
CNASIM=/proj/sc_ml/shared/envs/cnasim/bin/cnasim

echo "=== ort_selection start: $(date) ==="
echo "REPO: $REPO_DIR"
echo "OUT:  $OUT_DIR"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# ── Run experiment ─────────────────────────────────────────────────────────────
mamba run -n srnj python "$REPO_DIR/reproducibility/experiments/ort_selection/ort_selection_accuracy.py" \
  --n-cells 20 50 100 200 500 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --out-dir "$OUT_DIR" \
  --skip-cnasim \
  --cnasim-exec "$CNASIM" \
  --n-workers 32

# ── Update canonical CSVs ──────────────────────────────────────────────────────
latest_m=$(ls -t "$OUT_DIR"/*_ort_metrics.csv | head -1)
latest_a=$(ls -t "$OUT_DIR"/*_ort_selection_accuracy.csv | head -1)
cp "$latest_m" "$OUT_DIR/ort_metrics.csv"
cp "$latest_a" "$OUT_DIR/ort_selection_accuracy.csv"
echo "Canonical CSVs: $(basename "$latest_m")  $(basename "$latest_a")"

# ── Regenerate figures ─────────────────────────────────────────────────────────
module load R/4.4
ORT_METRICS_CSV="$OUT_DIR/ort_metrics.csv" \
ORT_ACCURACY_CSV="$OUT_DIR/ort_selection_accuracy.csv" \
ORT_FIG_ROOT="$OUT_DIR" \
Rscript "$REPO_DIR/reproducibility/experiments/ort_selection/plot.R"

# ── Regenerate dataset inspection PDF (fixed page size, includes n=500) ───────
mamba run -n srnj python "$REPO_DIR/reproducibility/experiments/ort_selection/plot_dataset.py" \
  --results-dir "$OUT_DIR" \
  --out "$OUT_DIR/dataset_inspect.pdf"

echo "=== ort_selection end: $(date) ==="
