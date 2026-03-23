import argparse
import subprocess
import os
import numpy as np
import pandas as pd
import scipy.sparse

import anndata


def impute_and_smooth(ad):
    """Linearly interpolate NaNs and apply a small rolling window mean cell-wise."""
    # Convert to dense for processing
    X = ad.X.toarray() if scipy.sparse.issparse(ad.X) else ad.X

    # Process row-wise (cell-wise) across genomic bins (columns)
    df = pd.DataFrame(X)
    # Interpolate along columns (axis=1)
    df = df.interpolate(method='linear', axis=1, limit_direction='both')
    # Apply rolling mean to smooth
    X_processed = df.rolling(window=3, center=True, min_periods=1, axis=1).mean()

    ad.X = X_processed.values
    return ad

def write_bed_files(adata_cell, output_dir, sample_name):
    os.makedirs(output_dir, exist_ok=True)
    bed_file_path = os.path.join(output_dir, f"{sample_name}.bed")
    # add 'chr' prefix to chromosome names if not present
    chr_names = adata_cell.var['chr'].values.tolist() if adata_cell.var['chr'].values[0].startswith('chr') else ('chr' + adata_cell.var['chr'].astype(str)).tolist()
    start = adata_cell.var['start'].values.tolist()
    end = adata_cell.var['end'].values.tolist()
    assert adata_cell.X.shape[0] == 1, "Expected adata.X to be 1D or single row"
    read_counts = adata_cell.X.flatten().tolist()
    # save dataframe to bed file without header
    with open(bed_file_path, 'w') as f:
        for chr_name, s, e, rc in zip(chr_names, start, end, read_counts):
            f.write(f"{chr_name}\t{s}\t{e}\t{int(rc)}\n")

def compute_healthy_avg(diploid_file_list, output_bed_file, scripts_path):
    # Run R script to compute healthy average
    print("Computing healthy average bed file...")
    subprocess.run(["Rscript", os.path.join(scripts_path, "avgDiploid.R"), diploid_file_list, output_bed_file], check=True)
    # touch a dummy file for testing
    # with open(output_bed_file, 'w') as f:
    #     f.write("chr1\t0\t250000\t375\nchr1\t250000\t500000\t400\n")  # dummy content


def compute_mean_var_coef(diploid_file_list, output_mean_var_file, scripts_path):
    # Run R script to compute mean-variance coefficients
    print("Computing mean-variance coefficient file...")
    subprocess.run(["Rscript", os.path.join(scripts_path, "fitMeanVarRlnshp.R"), diploid_file_list, output_mean_var_file], check=True)
    # touch a dummy file for testing
    # with open(output_mean_var_file, 'w') as f:
    #     f.write("""
    #     intercept = 10.79950512968
    #     slope = 1.18499408815
    #     poly2 = 0.01910756218
    #     """
    #     )  # dummy content

def prepare_bed_files(adata_path, normal_bed_dir, tumor_bed_dir, normal_obs_name, missing_data_option):
    adata = anndata.read_h5ad(adata_path)
    assert normal_obs_name in adata.obs.keys(), f"Normal observation name '{normal_obs_name}' not found in adata.obs"
    assert adata.obs[normal_obs_name].sum() > 0, f"No normal cells found with observation name '{normal_obs_name}'"
    # handle missing data if any
    if np.isnan(adata.X).any():
        print("Missing data (NaNs) detected in the input .h5ad file.")
        if missing_data_option == 'remove':
            # remove bins with NaNs across any cell
            bins_with_nans = np.isnan(adata.X).any(axis=0)
            adata = adata[:, ~bins_with_nans]
            print(f"Removed {bins_with_nans.sum()} bins with missing data. Remaining bins: {adata.X.shape[1]}")
            # if ratio of removed bins is high, warn the user
            if bins_with_nans.sum() / adata.X.shape[1] > 0.5:
                print("Warning: More than 50% of bins were removed due to missing data. This may lead to inaccurate results.")
        elif missing_data_option == 'impute':
            adata = impute_and_smooth(adata)
            print("Imputed missing data using linear interpolation and smoothed with a rolling mean.")
        else:
            print(f"Ignoring missing data with option '{missing_data_option}'. This may lead to errors in downstream processing.")
    
    for cell_name in adata.obs_names:
        if adata[cell_name].obs[normal_obs_name].iloc[0]:
            write_bed_files(adata[cell_name], normal_bed_dir, cell_name)
        else:
            write_bed_files(adata[cell_name], tumor_bed_dir, cell_name)



def main():
    parser = argparse.ArgumentParser(description="Prepare input for SCONCE2 from .h5ad file. Three files are needed: `../sconce2 -d test_healthy_avg.bed -t tumorFileList --meanVarCoefFile test.meanVar`")
    parser.add_argument("--input", type=str, required=True, help="Path to the .h5ad input file")
    parser.add_argument("--output", type=str, required=True, help="Path to the SCONCE2 input files. Will generate healthyAvg.bed, meanVarCoefFile, tumorFileList files and a tumor_bed_files/<sample>.bed file for each tumor sample.")
    parser.add_argument("--normal-obs-name", type=str, required=False, help="Observation name for normal cells in the .h5ad file", default="normal")
    parser.add_argument('--script-path', type=str, required=False, help="Path to the scripts directory containing R scripts", default="scripts/")
    parser.add_argument('--missing-data', type=str, required=False, help="How to handle missing data (NaNs) in the input .h5ad file. Options: 'ignore' (default) 'remove' or 'impute'.", default='ignore')
    args = parser.parse_args()
    # paths to files
    healthy_avg_bed = args.output + "/healthyAvg.bed"
    mean_var_coef_file = args.output + "/meanVarCoefFile"
    tumor_file_list = args.output + "/tumorFileList"
    diploid_file_list = args.output + "/diploidFileList"
    tumor_bed_dir = args.output + "/tumor_bed_files/"
    normal_bed_dir = args.output + "/normal_bed_files/"  # will be removed after healthy avg is computed
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(tumor_bed_dir):
        os.makedirs(tumor_bed_dir)
    if not os.path.exists(normal_bed_dir):
        os.makedirs(normal_bed_dir)

    #tumorbed format (tsv with chromosome, start, end, readcount)
    # chr1	0	250000	348
    # chr1	250000	500000	401
    #make normal and tumor bed files
    prepare_bed_files(args.input, normal_bed_dir, tumor_bed_dir, args.normal_obs_name, args.missing_data)
    # create tumor file list
    with open(tumor_file_list, 'w') as f:
        for sample_bed in os.listdir(tumor_bed_dir):
            f.write(f"{tumor_bed_dir}/{sample_bed}\n")
    with open(diploid_file_list, 'w') as f:
        for sample_bed in os.listdir(normal_bed_dir):
            f.write(f"{normal_bed_dir}/{sample_bed}\n")
    # run Rscript scripts/avgDiploid.R test/diploidFileList test/test_healthy_avg.bed
    compute_healthy_avg(diploid_file_list, healthy_avg_bed, args.script_path)
    # run Rscript scripts/fitMeanVarRlnshp.R test/diploidFileList test/ref.meanVar
    compute_mean_var_coef(diploid_file_list, mean_var_coef_file, args.script_path)
    print("SCONCE2 input files prepared successfully.")
    print("Cleaning up temporary normal bed files...")
    # for sample_bed in os.listdir(normal_bed_dir):
    #     os.remove(os.path.join(normal_bed_dir, sample_bed))
    # os.rmdir(normal_bed_dir)
    # print("Temporary files removed.")

if __name__ == "__main__":
    main()
