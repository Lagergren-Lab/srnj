import glob
import os
import re

import numpy as np
import pandas as pd


def parse_hmm(hmmFile):

    out = []
    with open(hmmFile) as f:
        s = f.read().split("#### FINAL HMM ####", 1)[1]

    pattern = re.compile(
        r"HMM\s+\d+\s+\(([^,]+),([^)]+)\).*?"
        r"paramsToEst:\s*([\d.eE+-]+)\s*([\d.eE+-]+)\s*([\d.eE+-]+)",
        re.S
    )

    for m in pattern.finditer(s):
        out.append((m.group(1).replace('.bed',''), m.group(2).replace('.bed',''),
                    float(m.group(3)), float(m.group(4)), float(m.group(5))))

    return out

if __name__ == "__main__":

    hmm_file = "/Users/zemp/phd/scilife/rnj/data/sconce2_test.hmm"
    rows = parse_hmm(hmm_file)

    # Print table
    print("cell0\tcell1\tt1\tt2\tt3")
    for r in rows:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}")


def get_sconce2_split_dist(sconce2_hmm_file) -> tuple[np.ndarray, np.ndarray, list[str]]:
    hmm_data = parse_hmm(sconce2_hmm_file)
    cell_names = []
    cell_id = 0
    # scan cells
    for (c1, c2, l1, l2, l3) in hmm_data:
        if c1 not in cell_names:
            cell_names.append(c1)
            cell_id += 1
        if c2 not in cell_names:
            cell_names.append(c2)
            cell_id += 1
    n = len(cell_names)
    C = np.zeros((n, n))  # LCA distance matrix
    A = np.zeros((n, n))  # Asymmetric distance matrix
    for (c1, c2, l1, l2, l3) in hmm_data:
        c1_id = cell_names.index(c1)
        c2_id = cell_names.index(c2)
        C[c1_id, c2_id] = l1
        C[c2_id, c1_id] = l1
        A[c1_id, c2_id] = l2
        A[c2_id, c1_id] = l3
    return C, A, cell_names

def get_sconce2_cn_matrix(sconce2_out_dir, cn_type, cell_names=None, k=None, n_bins=None):
    if k is None:
        # find k from the first file with pattern *.bed__k{k}__{cn_type}.bed
        bed_file_0 = glob.glob(os.path.join(sconce2_out_dir, f"*.bed__k*__{cn_type}.bed"))[0]
        k = re.search(r'__k(\d+)__', bed_file_0).group(1)
        print(f"Determined k={k} from bed file {bed_file_0}")
    bed_files = glob.glob(os.path.join(sconce2_out_dir, f"*.bed__k{k}__{cn_type}.bed"))
    pattern = re.compile(r'.__(.+?)\.bed__k' + re.escape(k) + '__' + re.escape(cn_type) + r'\.bed$')
    if cell_names is None:
        cell_names = []
        for bed_file in bed_files:
            match = pattern.search(os.path.basename(bed_file))
            if match:
                cell_name = match.group(1)
                if cell_name not in cell_names:
                    cell_names.append(cell_name)
        print(f"Determined cell_names from bed files: {cell_names[:5]}... (total {len(cell_names)} cells)")
    n_cells = len(cell_names)
    if n_bins is None:
        # determine n_bins from first bed file
        with open(bed_files[0], 'r') as f:
            n_bins = sum(1 for line in f)
        print(f"Determined n_bins={n_bins}")
    sconce_cn = np.zeros((n_cells, n_bins))
    for bed_file in bed_files:
        print(f"scanning {bed_file}")
        match = pattern.search(os.path.basename(bed_file))
        if match:
            cell_name = match.group(1)
            assert cell_name in cell_names, f"Cell name {cell_name} from SCONCE2 bed file not found in provided cell_names list"
            cell_idx = cell_names.index(cell_name)
            print(f"- found! cell_name: {cell_name}, cell_idx -> cell_names[{cell_idx}]")
            with open(bed_file, 'r') as f:
                bin_idx = 0
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        cn_value = float(parts[3])
                        if cn_type in ['median', 'mean']:
                            cn_value = round(cn_value)
                        sconce_cn[cell_idx, bin_idx] = cn_value
                        bin_idx += 1
    # check that all cells have been filled
    assert np.all(sconce_cn.sum(axis=1) > 0), f"Some cells in SCONCE2 CN matrix have all zeros for cn_type {cn_type}. Check that bed files are correctly parsed and matched to provided cell_names list."
    return sconce_cn, cell_names

def add_sconce2_cn_to_adata(adata, sconce2_out_dir, cn_types=None, k=None):
    cell_names = adata.obs_names.tolist()
    n_bins = adata.shape[1]
    if cn_types is None:
        cn_types = ['mode', 'median', 'mean']
    for cn_type in cn_types:
        sconce_cn, _ = get_sconce2_cn_matrix(sconce2_out_dir, cn_type, cell_names, k, n_bins=n_bins)
        assert sconce_cn.shape == adata.shape, f"SCONCE2 CN matrix shape {sconce_cn.shape} does not match adata shape {adata.shape}. Check that bed files are correctly parsed and matched to adata.obs_names list."
        adata.layers[f'sconce2_cn_{cn_type}'] = sconce_cn

def read_chisel_mutations(chisel_snv_path) -> pd.DataFrame:
    """
    Reads CHISEL SNV output file and returns an indexed sparse matrix with only the SNVs found in CHISEL region E
    and excluding any SNV that is present in region A.
    The returned DataFrame has cell barcodes (no region prefix) as row index and SNV_IDs as columns, values are 0/1.
    """
    # Read the CHISEL SNV output file
    try:
        snv_df = pd.read_csv(chisel_snv_path, sep="\t", compression="gzip")
    except Exception as e:
        raise ValueError(f"Error reading the file {chisel_snv_path}: {e}")

    # Check for required columns
    required_columns = {'#CHR', 'START', 'END', 'REF', 'ALT', 'HARBORING_CELLS'}
    if not required_columns.issubset(snv_df.columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Create a unique identifier for each SNV
    snv_df['SNV_ID'] = snv_df['#CHR'].astype(str) + ':' + snv_df['START'].astype(str) + ':' + snv_df['REF'] + '>' + \
                       snv_df['ALT']

    # Initialize counters and list to collect kept associations
    records = []
    total_snvs = len(snv_df)
    rows_with_hc = snv_df['HARBORING_CELLS'].notna().sum()
    total_associations_initial = 0
    snvs_filtered_due_to_A = 0
    associations_removed_due_to_A = 0
    associations_kept = 0

    # Iterate over each row to parse HARBORING_CELLS and apply filters
    for _, row in snv_df.iterrows():
        hc = row['HARBORING_CELLS']
        if pd.isna(hc):
            continue
        # split entries, tolerate spaces
        cells = [c.strip() for c in hc.split(',') if c.strip()]
        total_associations_initial += len(cells)

        # extract region letters when possible
        regions = []
        parsed_cells = []
        for c in cells:
            if '-' in c:
                reg, barcode = c.split('-', 1)
                regions.append(reg)
                parsed_cells.append((reg, barcode))
            else:
                # malformed entry, skip it
                continue

        # if SNV is present in region A, discard entire SNV
        if any(r == 'A' for r in regions):
            snvs_filtered_due_to_A += 1
            associations_removed_due_to_A += len(cells)
            continue

        # otherwise keep only region E entries
        e_entries = [p for p in parsed_cells if p[0] == 'E']
        if not e_entries:
            # no E-region cells for this SNV -> skip
            continue

        for reg, barcode in e_entries:
            records.append({'SNV_ID': row['SNV_ID'], 'CELL': barcode})
            associations_kept += 1

    # Build sparse matrix (cells x SNVs)
    if len(records) == 0:
        print(f"Filtering diagnostics: read {total_snvs} SNVs ({rows_with_hc} with HARBORING_CELLS).")
        print("No associations remain after filtering (either all SNVs contained region A or no region E entries).")
        # return empty DataFrame with no rows/cols
        return pd.DataFrame()

    sparse_df = pd.DataFrame(records)

    sparse_matrix = sparse_df.assign(value=1).pivot_table(index='CELL', columns='SNV_ID', values='value',
                                                          fill_value=0)

    # Diagnostics
    unique_cells_kept = sparse_df['CELL'].nunique()
    snvs_kept = sparse_matrix.shape[1]
    associations_dropped_nonE = total_associations_initial - associations_kept - associations_removed_due_to_A

    print(f"Filtering diagnostics:")
    print(f"  Total SNVs read: {total_snvs}")
    print(f"  SNVs with HARBORING_CELLS: {rows_with_hc}")
    print(f"  SNVs removed because present in region A: {snvs_filtered_due_to_A} (removed {associations_removed_due_to_A} associations)")
    print(f"  SNVs kept: {snvs_kept}")
    print(f"  Cell-SNV associations kept (region E only): {associations_kept}")
    print(f"  Unique cell barcodes kept: {unique_cells_kept}")
    print(f"  Associations removed because not in region E (and not due to A): {associations_dropped_nonE}")

    assert sparse_matrix.values.sum() == associations_kept, "Mismatch in number of kept associations, check parsing logic"

    return sparse_matrix

def read_chisel_clones(chisel_mapping_path: str) -> pd.Series:
    """
    #CELL	CLUSTER	CLONE
    AAACCTGCACCAAAGG	199	Clone199
    AAACCTGCAGGACCAA	199	Clone199
    AAACCTGGTAACTTCG	199	Clone199
    AAACCTGGTACCAGTT	270	None
    Args:
        chisel_mapping_path: str, path to the CHISEL mapping file (.tsv.gz / .tsv)
    """
    a = pd.read_csv(chisel_mapping_path, sep='\t', compression='gzip' if chisel_mapping_path.endswith('.gz') else None)
    if '#CELL' not in a.columns or 'CLONE' not in a.columns:
        raise ValueError("Input file must contain 'CELL' and 'CLONE' columns")
    a = a.rename(columns={'#CELL': 'CELL'})
    # print number of unique clones and clusters ('CLUSTER' column is not used in this function but we check it exists for diagnostics)
    num_clones = a['CLONE'].nunique()
    num_clusters = a['CLUSTER'].nunique()
    print(f"CHISEL mapping diagnostics:")
    print(f"  Total cells in mapping: {len(a)}")
    print(f"  Unique clones: {num_clones}")
    print(f"  Unique clusters: {num_clusters}")
    # make cell the index
    a = a.set_index('CELL')
    return a['CLONE']


def get_medicc2_dist(medicc2_dist) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    # medicc2 dist tsv is a n x n matrix with header and index as cell names
    # the last row and column correspond to the diploid cell (root)
    #   OV03-04_Bl_B01	OV03-04_Bl_B03	OV03-04_Om_S01 ...
    # OV03-04_Bl_B01	0.0	77.0	73.0 ...
    # OV03-04_Bl_B03	77.0	0.0	89.0 ...
    # ...
    dist_df = pd.read_csv(medicc2_dist, sep='\t', index_col=0)
    cell_names = dist_df.index.tolist()
    D = dist_df.to_numpy()
    root_dist = D[-1, :-1]  # last row except last element
    root_label = cell_names[-1]
    D = D[:-1, :-1]  # remove last row and last column
    cell_names = cell_names[:-1]
    return D, root_dist, cell_names, root_label
