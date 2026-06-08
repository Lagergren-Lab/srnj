library(ggplot2)
library(tidyr)
library(ggpubr)
library(dplyr)

if (length(commandArgs(trailingOnly = TRUE)) < 2) {
  stop("Usage: Rscript plot_results.R <input_file.csv> <output_file.png>")
}
input_file  <- commandArgs(trailingOnly = TRUE)[1]
output_file <- commandArgs(trailingOnly = TRUE)[2]
out_dir     <- dirname(output_file)

# ── Read and sort ──────────────────────────────────────────────────────────────
df <- read.csv(input_file) %>%
  arrange(n_cells, n_bins, n_clones, lamda, n_chrom, seq_error, seed, dist_method, tree_method)

df_long <- pivot_longer(df,
  cols = c(quartet_distance, rf_distance, transfer_distance,
           triplet_distance, rootsplit_distance),
  names_to  = "dist_metric",
  values_to = "score"
)

df_long$n_cells   <- as.factor(df_long$n_cells)
df_long$seq_error <- as.factor(df_long$seq_error)
df_long$method_combo <- interaction(df_long$tree_method, df_long$dist_method, sep = "_")

# ── Figure 1: main boxplot (all n_cells, faceted by metric × n_cells) ─────────
my_comparisons <- list(c("rnj_sconce2", "rnj_med2"),
                       c("nj_sconce2",  "rnj_sconce2"))

p1 <- ggplot(df_long, aes(x = method_combo, y = score, fill = dist_method)) +
  geom_boxplot(aes(alpha = tree_method), outlier.shape = NA) +
  geom_jitter(aes(alpha = tree_method), width = 0.2, size = 1) +
  stat_compare_means(comparisons = my_comparisons,
                     method = "wilcox.test", paired = TRUE, label = "p.signif") +
  scale_fill_manual(values = c(sconce2 = "#E69F00", med2 = "#56B4E9")) +
  scale_alpha_manual(values = c(anj = 0.3, fastme = 0.4, nj = 0.5, snj = 0.6,
                                rnj = 0.7, srnj = 0.8, srnj1 = 0.9, srnjmaxlca = 1.0)) +
  facet_grid(dist_metric ~ n_cells, scales = "free_y") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  ) +
  labs(x = "Method", y = "Distance Score",
       fill = "Distance Method", alpha = "Tree Method")

ggsave(output_file, plot = p1, width = 12, height = 12, bg = "white")

# ── Figure 2: seq-error comparison (ε=0.02 vs ε=0.04) ────────────────────────
# Mirrors figure7_full_comparison: facet by metric (rows) × n_cells×ε (columns).
# Focused on n_cells = 50 and 100 for readability.
METRICS <- c("rf_distance", "quartet_distance",
             "transfer_distance", "triplet_distance")
METRIC_LABS <- c(
  rf_distance       = "RF Distance",
  quartet_distance  = "Quartet Distance",
  transfer_distance = "Transfer Distance",
  triplet_distance  = "Triplet Distance"
)
ALL_TREE_LEVELS <- c("nj", "rnj", "srnj", "srnjmaxlca", "anj", "fastme")
TOOL_COLORS     <- c(sconce2 = "#E69F00", med2 = "#56B4E9")

df_long2 <- df_long %>%
  filter(
    dist_metric %in% METRICS,
    n_cells %in% c("50", "100"),
    tree_method %in% ALL_TREE_LEVELS
  ) %>%
  mutate(
    dist_method = factor(dist_method, levels = c("sconce2", "med2"),
                         labels = c("SCONCE2", "MEDICC2")),
    tree_method = factor(tree_method, levels = ALL_TREE_LEVELS),
    metric      = factor(dist_metric, levels = METRICS,
                         labels = unname(METRIC_LABS[METRICS])),
    eps_label   = paste0("ε=", seq_error),
    col_label   = paste0("n=", n_cells, ", ", eps_label)
  )

p2 <- ggplot(df_long2,
             aes(x = tree_method, y = score,
                 fill = tree_method, alpha = dist_method, colour = dist_method)) +
  geom_boxplot(linewidth = 0.3, outlier.shape = NA,
               position = position_dodge2(width = 0.8, preserve = "single")) +
  facet_grid(metric ~ col_label, scales = "free_y") +
  scale_fill_brewer(palette = "Set2", name = "Method") +
  scale_colour_manual(values = c(SCONCE2 = "grey20", MEDICC2 = "grey60"), name = "Tool") +
  scale_alpha_manual(values  = c(SCONCE2 = 0.9, MEDICC2 = 0.4), name = "Tool") +
  labs(x = NULL, y = "Tree Distance") +
  theme_bw(base_size = 9) +
  theme(
    strip.background  = element_rect(fill = "grey92", colour = NA),
    strip.text        = element_text(face = "bold", size = 8),
    axis.text.x       = element_text(angle = 45, hjust = 1, size = 6),
    panel.grid.minor  = element_blank(),
    plot.background   = element_rect(fill = "white", color = NA)
  )

ggsave(file.path(out_dir, "plots_seq_error_comparison.pdf"),
       plot = p2, width = 10, height = 7, device = cairo_pdf)

message("Wrote ", output_file, " and plots_seq_error_comparison.pdf")
