# Sparse Rooted Neighbor Joining (SparseRNJ)
> code for reproducing the results of the paper "Scalable and robust phylogenetic tree
> reconstruction from copy number data with Sparse Rooted Neighbor Joining" by Zampinetti V. et al.

## Using SparseRNJ as a library

Install the package with pip:
```bash
pip install -e .            # development install from repo root
# or
pip install sparsernj       # once published to PyPI
```

### Providing your own distance estimator

Subclass `DistanceProvider` and implement a single method —
`_compute_triplet(taxon1, taxon2) -> (lca, adm_fwd, adm_rev)` — where
- **lca** is the LCA distance `C[i, j]` (symmetric),
- **adm_fwd** is the asymmetric distance `A[taxon1, taxon2]` (taxon1 → their LCA),
- **adm_rev** is `A[taxon2, taxon1]` (taxon2 → their LCA).

Caching, call-counting, and `get_dms()` are inherited automatically.

```python
from sparsernj import DistanceProvider, sparse_rnj

class CellmatesProvider(DistanceProvider):
    def __init__(self, cells, estimator):
        super().__init__(taxa=list(range(len(cells))))
        self.cells = cells
        self.estimator = estimator

    def _compute_triplet(self, i, j):
        # estimator.triplet_distance returns (lca, adm_fwd, adm_rev)
        return self.estimator.triplet_distance(self.cells[i], self.cells[j])

tree = sparse_rnj(CellmatesProvider(cells, estimator))  # returns nx.DiGraph
```

If your estimator produces pairwise distances + per-cell root distances instead,
use `get_lca_from_pairwise(D, root_dist)` to convert to `(C, A)` and then wrap
them in a `FixedDistanceProvider`.

### Using custom orienting-leaf selection

```python
from sparsernj import sparse_rnj, FixedDistanceProvider
from sparsernj.ort_selection import selection_matrices_from_cn, matrix_selection_strategy

# build selection matrices from observed CN profiles and healthy baseline
mats = selection_matrices_from_cn(C, A, observed_cn, healthy_cn)
# pick a strategy and create a selector callable
selector = matrix_selection_strategy(mats["hamming"])
# run SRNJ with all leaves passed to the selector (cheap proxy — no distance cost)
tree = sparse_rnj(provider, ort_selector=selector, all_leaves=True)
```

Available built-in strategy names: `"min_D"` (default) and `"max_lca"`.
`selection_matrices_from_cn` returns matrices for: `gt_min`, `gt_max_lca`,
`hamming`, `hamming_max_lca`, `nll`, `nll_max_lca`.

---

## Experiment with in-house CN data

This experiment can be run executing the script in `reproducibility/experiments/sparse_nj_accuracy.py`
which requires the Python packages listed in `environment.yml` and
the binaries from the software packages [tqDist](https://www.birc.au.dk/~cstorm/software/tqdist/)
and [Booster](https://github.com/evolbioinfo/booster).
In particular, the script will assume that the following binaries are available in the system PATH (tested version):
```
- triplet_dist
- quartet_dist
- booster_linux64 (v0.1.2)
```
### Installation

First install tqDist building from source as described in the [documentation](https://www.birc.au.dk/~cstorm/software/tqdist/).
Download the binary for Booster v0.1.2 from the [releases page](https://github.com/evolbioinfo/booster/releases).
Add the required binaries to your system PATH.
```bash
# add binaries to PATH
export PATH="/path/to/tqdist/bin/:$PATH"
export PATH="/path/to/booster_linux64:$PATH"
# create and activate environment
conda env create -f environment.yml
conda activate srnj

# verify installation
./scripts/check_deps.sh
```
### Execution
```bash
python reproducibility/experiments/sparse_nj_accuracy.py --demo
```
The `--demo` flag runs a quick validation with reduced parameters,
while omitting it will execute the full evaluation across all parameter ranges (> 45').

## Orienting-leaf selection experiment (CNAsim)

This experiment benchmarks different orienting-leaf selection heuristics on CNAsim
tumour data.  It compares the ground-truth oracle (`gt_min`, `gt_max_lca`) against
cheap copy-number proxies (`hamming`, `hamming_max_lca`, `nll`, `nll_max_lca`).

### Requirements

All dependencies are included in `environment.yml` (CNASim, msprime) plus the same
tqDist / Booster binaries listed above.  Transfer-distance is silently skipped if
`booster_linux64` is not in PATH.

### Execution
```bash
# quick demo (n=20, 50; 3 seeds)
python reproducibility/experiments/ort_selection/ort_selection_accuracy.py --demo

# full run
python reproducibility/experiments/ort_selection/ort_selection_accuracy.py \
    --n-cells 50 100 200 --seeds 0 1 2 3 4 5 6 7 8 9

# generate figures (from the output directory)
ORT_METRICS_CSV=<path>/ort_metrics.csv \
ORT_ACCURACY_CSV=<path>/ort_selection_accuracy.csv \
ORT_FIG_ROOT=<path> \
Rscript reproducibility/experiments/ort_selection/plot.R
```

## Experiment with CNAsim data

This experiment uses [Snakemake](https://snakemake.readthedocs.io/en/stable/)
to automate the workflow from data simulation with CNAsim to tree reconstruction and evaluation.
It also depends on [SCONCE2](https://github.com/NielsenBerkeleyLab/sconce2), [MEDICC2](https://pypi.org/project/medicc2/)
and [CNAsim](https://github.com/samsonweiner/CNAsim).
While SCONCE2 needs to be manually installed, MEDICC2, CNAsim and all other
pipeline dependencies are installed automatically in the Snakemake workflow
in dedicated conda environments.
NOTE: the pipeline also relies on two R scripts from SCONCE2 repository
that are used to prepare the input data, those scripts are downloaded and placed
in the correct location by the `setup_sconce2.sh` script, but SCONCE2 itself is not 
installed as part of the workflow setup, please follow the [original
instructions](https://github.com/NielsenBerkeleyLab/sconce2?tab=readme-ov-file#installation-instructions)
to install it.

### Installation
```bash
# make an empty environment and install snakemake
conda create -n snakemake_env -c conda-forge -c bioconda snakemake
conda activate snakemake_env
# setup SCONCE2 (will download the repository without installing SCONCE2)
./scripts/setup_sconce2.sh
# install SCONCE2 dependencies (BOOST, GSL, ...)
# then install sconce2 e.g. with:
#    cd external/sconce2; make; export PATH="$(pwd):$PATH"; cd ../..
```
### Execution
```bash
# run workflow with demo configuration
cd reproducibility/workflows/srnj_sconce2
snakemake --configfile config/demo.yaml --use-conda --cores 2 # demo configuration with reduced parameters for quick validation
# run workflow with full configuration on HPC with slurm
snakemake --profile workflow/profile --use-conda
```

## Real Data Experiment

Data has been downloaded from the original 10x Genomics website,
binned into 5Mb bins, then merged with CHISEL CN data. We provide
the preprocessed data in `data/merged_5M_N500.h5ad` which contains
the read-counts, CN and cell/bin annotations for the subset of cells
(N=499) chosen for the analysis as described in the Experiments
section of the paper, plus (N=357) normal cells from the same region
that are required by SCONCE2.

To reproduce the analysis, run MEDICC2 on the CN data (extracted as tsv for
convenience and available in the `data` folder)
```bash
medicc2 ./data/merged_5M_N500_E_tumoronly.tsv \
    ./output/breast10x/medicc2_output/ \
    -j 32
# Note: adjust -j parameter according to your system's available cores
```
and SCONCE2 on the read-count data
```bash
# prepare input with the provided script
python ./reproducibility/workflows/srnj_sconce2/workflow/scripts/prepare_sconce2_input.py --input ./data/merged_5M_N500_E.h5ad --output ./output/breast10x/sconce2_input/ --normal-obs-name normal --script-path ./external/sconce2/scripts/ --missing-data remove
sconce2 -d ./output/breast10x/sconce2_input/healthyAvg.bed \
    -t ./output/breast10x/sconce2_input/healthyAvg.bed \
    --meanVarCoefFile ./output/breast10x/sconce2_input/meanVarCoefFile \
    -k 8 \
    -o ./output/breast10x/sconce2_output/model \
    --saveSconce \
    --summarizeAll \
    -j 32 \
    --sconceEstimatesPath ./output/breast10x/sconce2_output/ \
    --pairedEstimatesPath ./output/breast10x/sconce2_output/ \
    > ./output/breast10x/sconce2_output/sconce2.log 2> ./output/breast10x/sconce2_output/sconce2.err
# Note: adjust -j parameter according to your system's available cores
```
Then run the respective analysis scripts for the two methods to obtain
the final trees and evaluation metrics (requires CHISEL analysis data which is
downloaded and setup with the `setup_chisel_data.sh` script)
```bash
./scripts/setup_chisel_data.sh
python ./reproducibility/experiments/breast10x/breast_data_medicc2.py
python ./reproducibility/experiments/breast10x/breast_data_sconce2.py 
```

## Repository Structure

```
src/
└── sparsernj/              # pip-installable package
    ├── distance_provider.py  # DistanceProvider base + FixedDistanceProvider / LazyDistanceProvider
    ├── ort_selection.py      # matrix-based selection strategies (selection_matrices_from_cn, etc.)
    ├── sparse_rnj.py         # SRNJ algorithm (sparse_rnj)
    ├── sparse_nj.py          # SNJ algorithm (unrooted variant)
    ├── treenode.py           # Tree / UTree data structures
    └── utils/                # tree utilities and algorithms (formerly top-level utils/)

reproducibility/
├── experiments/
│   ├── sparse_nj_accuracy.py           # synthetic data experiment
│   ├── ort_selection/                  # orienting-leaf selection experiment
│   │   ├── ort_selection_accuracy.py
│   │   └── plot.R
│   └── breast10x/                      # real 10x data experiment
└── workflows/                          # snakemake pipeline (SCONCE2 + MEDICC2)

scripts/                # setup and utility scripts
tests/                  # unit tests
data/                   # preprocessed data for real data experiment
```

# Contact

For questions or issues regarding the code, 
please open an issue in this repository.

