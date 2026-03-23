# inspect sconce2 input and output relative to success (sconce2.done exists) or failure
import os
import sys

import pandas as pd
import scipy.stats as stats

def get_readcounts_summary(input_dir):
    summary = {}
    # analyse meanVarCoefFile
    meancovfile = os.path.join(input_dir, 'meanVarCoefFile')
    if not os.path.exists(meancovfile):
        return None
    with open(meancovfile) as f:
        for line in f:
            k,v = line.strip().split('=')
            summary[k] = float(v)

    # analyse healthy avg
    with open(os.path.join(input_dir, 'healthyAvg.bed')) as f:
        for line in f:
            chrom, start, end, avg, var = line.strip().split('\t')
            summary['healthy_mean'] = float(avg)
            summary['healthy_var'] = float(var)
    # analyse tumor avg
    tumor_mean, tumor_var, tumor_max, tumor_min = 0, 0, 0, 0
    tumor_reads = []
    it = 0
    for fn in os.listdir(os.path.join(input_dir, 'tumor_bed_files')):
        with open(os.path.join(input_dir, 'tumor_bed_files', fn)) as f:
            for line in f:
                it += 1
                chrom, start, end, reads = line.strip().split('\t')
                tumor_reads.append(float(reads))
                tumor_max = max(tumor_max, float(avg))
                tumor_min = min(tumor_min, float(avg))
    summary['tumor_mean'] = sum(tumor_reads) / it
    summary['tumor_var'] = sum([(x - summary['tumor_mean']) ** 2 for x in tumor_reads]) / it
    summary['tumor_max'] = tumor_max
    summary['tumor_min'] = tumor_min
    return summary


def main():
    # this script is given a list of "sconce2.done" files ...
    summaries = []
    for r in sys.argv[1:]:
        d = os.path.abspath(os.path.join(os.path.dirname(r), os.pardir))  # data dir
        summary = get_readcounts_summary(os.path.join(d, 'sconce2_input'))
        if summary is None:
            continue
        summary['status'] = None
        with open(r, 'r') as f:
            summary['status'] = f.read().strip()
        # get R and N from name of directory, assuming it is in the format "RX_NX_"
        dat_str = os.path.basename(d)
        summary['R'] = int(dat_str.split('_')[0][1:])
        summary['N'] = int(dat_str.split('_')[1][1:])
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    print(df.sort_values(['status','R','N']).to_string())
    # perform t-test for each feature between success and failure
    n = len(df)
    print(f"\nSUCCESS RATIO: {(df['status'] == 'success').mean():.4f} (tot={n})\n")
    features = ['healthy_mean', 'healthy_var', 'tumor_mean', 'tumor_var', 'tumor_max', 'tumor_min', 'N']
    for feature in features:
        success_values = df[df['status'] == 'success'][feature]
        failure_values = df[df['status'] == 'failure'][feature]
        t_stat, p_value = stats.ttest_ind(success_values, failure_values)
        print(f"Feature: {feature}, t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

if __name__ == '__main__':
    main()
