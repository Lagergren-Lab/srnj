# Sparse Rooted Neighbor Joining (SparseRNJ)
> code for reproducing the results of the paper "Scalable and robust phylogenetic tree
> reconstruction from copy number data with Sparse Rooted Neighbor Joining" by Zampinetti V. et al.

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
├── sparsernj/          # core SparseRNJ implementation
├── utils/              # tree utilities and algorithms
└── utils/evaluation/   # benchmarking and metrics

reproducibility/
├── experiments/        # synthetic and real data experiments scripts
└── workflows/          # snakemake pipeline for experiment with SCONCE2 and MEDICC2 (CNAsim data)

scripts/                # setup and utility scripts
tests/                  # unit tests
data/                   # preprocessed data for real data experiment
```

# Contact

For questions or issues regarding the code, 
please open an issue in this repository.

