import argparse
import glob
import re
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare medicc2 input from SCONCE2 CN calls.")
    #model__cell9.bed__k5__mode.bed
    parser.add_argument("--input", type=str, required=True, help="Path to SCONCE2 output dir where *.bed files are located")
    parser.add_argument("--output", type=str, required=True, help="Path to medicc2 input TSV file with cn profiles")
    parser.add_argument("--stat", type=str, default='mode', help="Statistic to use from SCONCE2 bed files: mode, median or mean. Median and mean are rounded before writing to medicc2 input.")
    return parser.parse_args()

def main():
    args = parse_args()
    # SCONCE2 outputs bed files for multiple k values (e.g. k7 and k10).
    # Keep only the highest k per cell to avoid duplicate entries in the TSV.
    bed_files = glob.glob(os.path.join(args.input, f"*.bed__k*__{args.stat}.bed"))
    pattern = re.compile(r'.__(.+?)\.bed__k(\d+)__' + re.escape(args.stat) + r'\.bed$')
    best = {}  # cell_name -> (k, path)
    for bed_file in bed_files:
        match = pattern.search(os.path.basename(bed_file))
        if match:
            cell_name, k = match.group(1), int(match.group(2))
            if cell_name not in best or k > best[cell_name][0]:
                best[cell_name] = (k, bed_file)

    with open(args.output, 'w') as of:
        of.write("sample_id\tchrom\tstart\tend\ttotal_cn\n")
        for cell_name, (_, bed_file) in sorted(best.items()):
            with open(bed_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        cn_value = float(parts[3])
                        if args.stat in ['median', 'mean']:
                            cn_value = round(cn_value)
                        of.write(f"{cell_name}\t{parts[0]}\t{parts[1]}\t{parts[2]}\t{cn_value:.0f}\n")
    print(f"MEDICC2 input file prepared successfully at {args.output}")

if __name__ == "__main__":
    main()
