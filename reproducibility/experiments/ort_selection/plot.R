library(tidyverse)
library(patchwork)
library(ggbeeswarm)
library(ggsignif)

# ── Figure root (overridable via env var) ─────────────────────────────────────
fig_root <- Sys.getenv("ORT_FIG_ROOT", unset = ".")

# ── Data ──────────────────────────────────────────────────────────────────────
# Expects two CSV files produced by ort_selection_accuracy.py in the working dir
# or paths passed via env vars ORT_METRICS_CSV / ORT_ACCURACY_CSV.
metrics_csv  <- Sys.getenv("ORT_METRICS_CSV",  unset = "ort_metrics.csv")
accuracy_csv <- Sys.getenv("ORT_ACCURACY_CSV", unset = "ort_selection_accuracy.csv")

metrics  <- read_csv(metrics_csv,  show_col_types = FALSE, na = c("", "NA", "None"))
accuracy <- read_csv(accuracy_csv, show_col_types = FALSE, na = c("", "NA", "None"))

# ── Strategy ordering, readable labels, paired palette ───────────────────────
# Reading order (top → bottom on the flipped y-axis): baseline first, then each
# method family with its two selection rules.  The two rules are:
#   "min d"   – orienting leaves chosen by *minimum* cell–cell distance
#   "max LCA" – orienting leaves chosen by *maximum* root-to-LCA distance
STRATEGY_LEVELS <- c(
  "dlca_nj", "srnj_min_D", "srnj_max_lca",
  "gt_min", "gt_max_lca",
  "hamming", "hamming_max_lca",
  "nll", "nll_max_lca"
)
STRATEGY_LABELS <- c(
  dlca_nj          = "DLCA-NJ",
  srnj_min_D       = "SRNJ (min d)",
  srnj_max_lca     = "SRNJ (max LCA)",
  gt_min           = "GT (min d)",
  gt_max_lca       = "GT (max LCA)",
  hamming          = "Hamming (min d)",
  hamming_max_lca  = "Hamming (max LCA)",
  nll              = "NLL (min d)",
  nll_max_lca      = "NLL (max LCA)"
)
# Paired palette, intentionally *different* from the method-comparison palette
# used elsewhere: one hue per family, darker shade = min-d, lighter = max-LCA.
# DLCA-NJ is an unpaired baseline (neutral grey).
STRATEGY_COLORS <- c(
  dlca_nj          = "#595959",
  srnj_min_D       = "#1F4E79",  # blue (dark)
  srnj_max_lca     = "#7FA8D0",  # blue (light)
  gt_min           = "#B2182B",  # red (dark)
  gt_max_lca       = "#EF8A8A",  # red (light)
  hamming          = "#1B7837",  # green (dark)
  hamming_max_lca  = "#7FBF7B",  # green (light)
  nll              = "#762A83",  # purple (dark)
  nll_max_lca      = "#C2A5CF"   # purple (light)
)

METRIC_LEVELS <- c("rf_distance", "quartet_distance", "triplet_distance", "transfer_distance")
METRIC_LABELS <- c(
  rf_distance       = "RF Distance",
  quartet_distance  = "Quartet Distance",
  triplet_distance  = "Triplet Distance",
  transfer_distance = "Transfer Distance"
)

theme_paper <- function() {
  theme_bw(base_size = 9) +
    theme(
      strip.background  = element_rect(fill = "grey92", colour = NA),
      strip.text        = element_text(face = "bold", size = 8),
      legend.position   = "none",
      panel.grid.minor  = element_blank(),
      axis.text         = element_text(size = 7),
      axis.title        = element_text(size = 8)
    )
}

# ── Tidy up ───────────────────────────────────────────────────────────────────
metrics <- metrics |>
  filter(metric %in% METRIC_LEVELS, !is.na(dist)) |>
  mutate(
    strategy = factor(strategy, levels = STRATEGY_LEVELS),
    metric   = factor(metric,   levels = METRIC_LEVELS, labels = unname(METRIC_LABELS[METRIC_LEVELS])),
    n_cells  = factor(n_cells, levels = sort(unique(as.integer(n_cells))))
  ) |>
  filter(!is.na(strategy))

accuracy <- accuracy |>
  filter(!is.na(pair_accuracy)) |>
  mutate(
    strategy = factor(strategy, levels = STRATEGY_LEVELS),
    n_cells  = factor(n_cells, levels = sort(unique(as.integer(n_cells))))
  ) |>
  filter(!is.na(strategy))

# ── Figure 1: Tree-distance per strategy (horizontal), faceted metric × n_cells
# Methods on the y-axis, distance on the x-axis (boxplots are horizontal).
available_metrics <- levels(droplevels(metrics$metric))
n_ncells <- nlevels(droplevels(metrics$n_cells))

fig1 <- ggplot(metrics, aes(x = strategy, y = dist, fill = strategy)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, linewidth = 0.3) +
  geom_beeswarm(aes(colour = strategy), size = 0.45, alpha = 0.55, cex = 0.5) +
  facet_grid(metric ~ n_cells, scales = "free_x",
             labeller = labeller(n_cells = \(x) paste0("n=", x))) +
  scale_fill_manual(values = STRATEGY_COLORS) +
  scale_colour_manual(values = STRATEGY_COLORS) +
  scale_x_discrete(limits = rev(STRATEGY_LEVELS), labels = STRATEGY_LABELS) +
  coord_flip() +
  labs(x = NULL, y = "Distance") +
  theme_paper()

ggsave(file.path(fig_root, "ort_selection_distances.pdf"), fig1,
       width = max(7, 1.9 * n_ncells),
       height = 1.9 * length(available_metrics) + 0.8, device = cairo_pdf)

# ── Figure 2: Orienting-leaf selection accuracy (horizontal), faceted by n_cells
# Only leaf-pair accuracy is shown: direction accuracy is ~100% everywhere and
# carries no signal.  Significance brackets compare the two selection rules
# (min-d vs max-LCA) within each cheap-proxy family via a two-sided Wilcoxon
# rank-sum test (GT is the oracle and is constant at 100%, so it is not tested).
sig_comparisons <- list(
  c("hamming", "hamming_max_lca"),
  c("nll",     "nll_max_lca")
)

fig2 <- ggplot(accuracy, aes(x = strategy, y = pair_accuracy, fill = strategy)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, linewidth = 0.3) +
  geom_beeswarm(aes(colour = strategy), size = 0.6, alpha = 0.6, cex = 0.5) +
  geom_signif(comparisons = sig_comparisons,
              map_signif_level = TRUE, test = "wilcox.test",
              textsize = 2.4, tip_length = 0.01,
              y_position = c(1.02, 1.02), size = 0.3, vjust = -0.1) +
  facet_grid(. ~ n_cells, labeller = labeller(n_cells = \(x) paste0("n=", x))) +
  scale_fill_manual(values = STRATEGY_COLORS) +
  scale_colour_manual(values = STRATEGY_COLORS) +
  scale_x_discrete(limits = rev(levels(droplevels(accuracy$strategy))),
                   labels = STRATEGY_LABELS) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.12),
                     breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  coord_flip() +
  labs(x = NULL, y = "Leaf-pair accuracy vs oracle") +
  theme_paper()

ggsave(file.path(fig_root, "ort_selection_accuracy.pdf"), fig2,
       width = max(7, 1.9 * nlevels(droplevels(accuracy$n_cells))),
       height = 3.0, device = cairo_pdf)

message("Wrote ort_selection_distances.pdf and ort_selection_accuracy.pdf to ", fig_root)
