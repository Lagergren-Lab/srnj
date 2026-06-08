args = commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript HMMCopyInference.R <input_d_mat.csv> <output_inferred.txt> [l_per_copy] [chrom_lengths_csv]")
}

input_file    <- args[1]
output_file   <- args[2]
# l_per_copy: expected reads per copy per bin (= simulation param l).
l_per_copy    <- if (length(args) >= 3) as.numeric(args[3]) else 50.0
chrom_len_str <- if (length(args) >= 4) args[4] else NULL

suppressWarnings(library(HMMcopy))

tfile <- system.file("extdata", "tumour.wig", package = "HMMcopy")
gfile <- system.file("extdata", "gc.wig", package = "HMMcopy")
mfile <- system.file("extdata", "map.wig", package = "HMMcopy")
tumour_copy <- correctReadcount(wigsToRangedData(tfile, gfile, mfile))

sim_data <- as.matrix(read.table(input_file, header = FALSE, sep = ","))
sim_data  <- t(t(sim_data))

n_cells <- nrow(sim_data)
n_bins  <- ncol(sim_data)

max_bins <- nrow(tumour_copy)
if (n_bins > max_bins) {
  stop(paste0(
    "Input has ", n_bins, " bins but HMMcopy template provides only ", max_bins,
    " bins. Reduce M in config or provide a longer template."
  ))
}

# Parse chromosome lengths; default = single chromosome covering all bins.
# Running the HMM per chromosome prevents transitions across chr boundaries.
if (!is.null(chrom_len_str) && nchar(trimws(chrom_len_str)) > 0) {
  chrom_lengths <- as.integer(strsplit(trimws(chrom_len_str), ",")[[1]])
} else {
  chrom_lengths <- n_bins
}
chr_bin_ends   <- cumsum(chrom_lengths)
chr_bin_starts <- c(1L, head(chr_bin_ends, -1L) + 1L)

message(sprintf("HMMcopy: %d bins x %d cells, %d chromosome(s)",
                n_bins, n_cells, length(chrom_lengths)))

# Normalise per cell by the per-cell mean.
# E[x | CN=c] = log2(c / mean_cn) where mean_cn = mean_reads / l_per_copy.
normalized_data <- log2(sim_data / rowMeans(sim_data))

mu_log <- matrix(NA, nrow = n_cells, ncol = 6,
                 dimnames = list(NULL, paste0("state", 1:6)))

inferred_states <- matrix(NA, nrow = n_cells, ncol = n_bins)

for (ii in 1:n_cells) {
  message(paste("Cell number", ii, "being processed"))

  tumour_copy_cell         <- tumour_copy[1:n_bins, ]
  tumour_copy_cell$gc      <- 1
  tumour_copy_cell$map     <- 1
  tumour_copy_cell$valid   <- TRUE
  tumour_copy_cell$ideal   <- TRUE
  tumour_copy_cell$cor.gc  <- 1
  tumour_copy_cell$cor.map <- 1
  tumour_copy_cell$copy    <- normalized_data[ii, ]
  # Zero-read bins produce log2(0) = -Inf; replace with a finite floor so
  # HMMsegment doesn't crash when a template-chromosome sub-range is all-zero.
  tumour_copy_cell$copy[!is.finite(tumour_copy_cell$copy)] <- -6

  longseq_param <- HMMsegment(tumour_copy_cell, getparam = TRUE)

  # Per-cell mu: map states 1-6 → CN 0-5 in the mean-normalised space.
  mean_cn <- mean(sim_data[ii, ]) / l_per_copy
  cn_vals  <- c(0, 1, 2, 3, 4, 5)
  mu_vals  <- ifelse(cn_vals == 0, log2(0.05 / mean_cn), log2(cn_vals / mean_cn))
  longseq_param$mu <- mu_vals
  longseq_param$m  <- mu_vals   # anchor m so EM cannot drift mu

  message(sprintf("  Cell %d  mean_cn=%.2f  mu (CN 0-5): %s",
                  ii, mean_cn,
                  paste(round(longseq_param$mu, 3), collapse = ", ")))

  mu_log[ii, ] <- longseq_param$mu

  # e=0.99 → expected run length ~100 bins (appropriate for 200-bin data).
  longseq_param$e        <- 0.99
  longseq_param$nu       <- 4
  longseq_param$strength <- 10e7

  # Run HMM independently per chromosome to prevent cross-boundary transitions.
  # If all bins on a chromosome have zero reads (complete deletion / LOH),
  # HMMsegment cannot normalise the vector — assign CN=0 (state 1) directly.
  cell_states <- integer(0)
  for (chr_idx in seq_along(chrom_lengths)) {
    sb  <- chr_bin_starts[chr_idx]
    eb  <- chr_bin_ends[chr_idx]
    chr_copy <- tumour_copy_cell$copy[sb:eb]
    if (all(!is.finite(chr_copy)) || all(chr_copy == 0)) {
      cell_states <- c(cell_states, rep(1L, eb - sb + 1L))
    } else {
      chr_segs    <- HMMsegment(tumour_copy_cell[sb:eb, ], longseq_param, verbose = FALSE)
      cell_states <- c(cell_states, chr_segs$state)
    }
  }
  inferred_states[ii, ] <- cell_states - 1
}

message("\n=== mu summary across all cells ===")
message("Column = HMMcopy state (1-6) = CN 0-5, Row = cell")
message(paste(capture.output(round(mu_log, 3)), collapse = "\n"))
message("\nPer-state mean mu:")
message(paste(round(colMeans(mu_log), 4), collapse = ", "))

write.table(inferred_states, output_file, col.names = FALSE, row.names = FALSE)
message(paste("\nWrote", output_file))
