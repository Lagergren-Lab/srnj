# SparseRNJ: Sparse Neighbor Joining for Phylogenetic Tree Reconstruction

Clean, dependency-managed implementation of Sparse Neighbor Joining algorithms for single-cell phylogenetic tree reconstruction.

## 🚀 Quick Start

### Prerequisites
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate srnj

# Verify installation
./scripts/check_deps.sh
```

### ⚡ Demo 1: Algorithm Accuracy (30 seconds)
```bash
cd reproducibility/experiments
python sparse_nj_accuracy.py --demo
```
Tests SparseRNJ against standard methods with synthetic data. Generates performance plots and statistical analysis.

### 🔬 Demo 2: Complete Workflow (2-5 minutes)  
```bash
# Setup SCONCE2 (downloads repository only)
./scripts/setup_sconce2.sh

# Run workflow with demo configuration
cd reproducibility/workflows/srnj_sconce2
snakemake --configfile config/demo.yaml --use-conda --cores 2
```
Full phylogenetic reconstruction pipeline from simulation to evaluation.

## 📦 Installation

### Python Environment
```bash
conda env create -f environment.yml
conda activate srnj
```

**Included packages:** `dendropy`, `networkx`, `numpy`, `scipy`, `scikit-bio`, `fastme`

### External Binaries

Required for tree distance computation:

| Binary | Purpose | Installation |
|--------|---------|--------------|
| **tqDist** | Quartet/triplet distances | [Download](https://www.birc.au.dk/~cstorm/software/tqdist/) → build → add to PATH |
| **booster** | Transfer distances | [Download](https://github.com/evolbioinfo/booster) → build → rename to `booster_linux64`/`booster_macos64` |
| **SCONCE2** | Copy number analysis (optional) | `./scripts/setup_sconce2.sh` → manual build |

**Verification:**
```bash
./scripts/check_deps.sh  # Shows installation status and provides guidance
```

## 🔬 Experiments

### Synthetic Data Benchmarking
```bash
cd reproducibility/experiments
python sparse_nj_accuracy.py     # Full evaluation across parameter ranges
python sparse_nj_accuracy.py --demo  # Quick validation (reduced parameters)
```

### Real Data Analysis
```bash
cd reproducibility/experiments/breast10x
python breast_data_medicc2.py    # MEDICC2-based analysis
python breast_data_sconce2.py    # SCONCE2-based analysis
```

### Snakemake Pipeline
```bash
cd reproducibility/workflows/srnj_sconce2
snakemake --use-conda --cores 4  # Full workflow (default config)
snakemake --configfile config/demo.yaml --use-conda --cores 2  # Demo mode
```

## 🧬 Core Features

- **Algorithm**: Sparse Neighbor Joining with optimized distance computations
- **Benchmarking**: Comparisons with NJ, DLCA-NJ, ANJ, FastME algorithms  
- **Metrics**: Robinson-Foulds, quartet distance, triplet distance, transfer distance
- **Workflows**: End-to-end pipelines from simulation to phylogenetic evaluation
- **Dependencies**: Clean conda environment with minimal external requirements

## 📊 Output

- **Performance plots**: Algorithm comparison visualizations
- **Distance matrices**: Pairwise phylogenetic distances
- **Statistical analysis**: Significance testing and evaluation metrics
- **Phylogenetic trees**: Reconstructed trees in Newick format

## 🧪 Testing

```bash
python -m pytest tests/ -v  # Run test suite
```

All core functionality is tested with 21 test cases covering tree parsing, algorithms, and data structures.

## 🏗️ Architecture

```
src/
├── sparsernj/          # Core SparseRNJ implementation
├── utils/              # Tree utilities and algorithms
└── utils/evaluation/   # Benchmarking and metrics

reproducibility/
├── experiments/        # Standalone evaluation scripts  
└── workflows/          # Snakemake pipelines

scripts/                # Setup and utility scripts
tests/                  # Comprehensive test suite
```

## 📝 Citation

If you use SparseRNJ in your research, please cite:

```bibtex
@article{sparsernj2024,
  title={SparseRNJ: Efficient Phylogenetic Tree Reconstruction for Single-Cell Analysis},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```