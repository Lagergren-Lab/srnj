library(tidyverse)
library(patchwork)
library(ggbeeswarm)

# ── Figure root (overridable via env var) ─────────────────────────────────────
fig_root <- Sys.getenv("ORT_FIG_ROOT", unset = ".")

# ── Data ──────────────────────────────────────────────────────────────────────
# Expects two CSV files produced by ort_selection_accuracy.py in the working dir
# or paths passed via env vars ORT_METRICS_CSV / ORT_ACCURACY_CSV.
metrics_csv  <- Sys.getenv("ORT_METRICS_CSV",  unset = "ort_metrics.csv")
accuracy_csv <- Sys.getenv("ORT_ACCURACY_CSV", unset = "ort_selection_accuracy.csv")

metrics  <- read_csv(metrics_csv,  show_col_types = FALSE, na = c("", "NA", "None"))
accuracy <- read_csv(accuracy_csv, show_col_types = FALSE, na = c("", "NA", "None"))

# ── Strategy ordering and palette ────────────────────────────────────────────
STRATEGY_LEVELS <- c(
  "dlca_nj", "srnj_min_D", "srnj_max_lca",
  "gt_min", "gt_max_lca",
  "hamming", "hamming_max_lca",
  "nll", "nll_max_lca"
)
STRATEGY_LABELS <- c(
  dlca_nj          = "DLCA-NJ",
  srnj_min_D       = "SRNJ (min-D)",
  srnj_max_lca     = "SRNJ (max-LCA)",
  gt_min           = "GT min-D",
  gt_max_lca       = "GT max-LCA",
  hamming          = "Hamming",
  hamming_max_lca  = "Hamming max-LCA",
  nll              = "NLL",
  nll_max_lca      = "NLL max-LCA"
)
STRATEGY_COLORS <- c(
  dlca_nj          = "#2A9D8F",
  srnj_min_D       = "#457B9D",
  srnj_max_lca     = "#1D3557",
  gt_min           = "#E63946",
  gt_max_lca       = "#B5179E",
  hamming          = "#F4A261",
  hamming_max_lca  = "#E76F51",
  nll              = "#6A4C93",
  nll_max_lca      = "#4CC9F0"
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
      legend.position   = "bottom",
      legend.key.size   = unit(0.4, "cm"),
      legend.title      = element_blank(),
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
    n_cells  = factor(n_cells)
  ) |>
  filter(!is.na(strategy))

accuracy <- accuracy |>
  filter(!is.na(pair_accuracy)) |>
  mutate(
    strategy = factor(strategy, levels = STRATEGY_LEVELS),
    n_cells  = factor(n_cells)
  ) |>
  filter(!is.na(strategy))

# ── Figure 1: Tree-distance per strategy, faceted metric × n_cells ─────────
available_metrics <- levels(droplevels(metrics$metric))
fig1_height <- 1.6 * length(available_metrics) + 0.5

fig1 <- ggplot(metrics, aes(x = strategy, y = dist,
                              colour = strategy, fill = strategy)) +
  geom_boxplot(alpha = 0.25, outlier.shape = NA, linewidth = 0.4) +
  geom_beeswarm(size = 0.8, alpha = 0.7, cex = 0.6) +
  facet_grid(metric ~ n_cells, scales = "free_y",
             labeller = labeller(n_cells = \(x) paste0("n=", x))) +
  scale_colour_manual(values = STRATEGY_COLORS, labels = STRATEGY_LABELS) +
  scale_fill_manual(values   = STRATEGY_COLORS, labels = STRATEGY_LABELS) +
  labs(x = NULL, y = "Distance") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        legend.position = "none")

ggsave(file.path(fig_root, "ort_selection_distances.pdf"), fig1,
       width = max(5, 1.5 * nlevels(metrics$n_cells)),
       height = fig1_height, device = cairo_pdf)

# ── Figure 2: Selection accuracy vs n_cells (pair accuracy) ─────────────────
fig2 <- ggplot(accuracy, aes(x = n_cells, y = pair_accuracy,
                              colour = strategy, fill = strategy,
                              group = strategy)) +
  geom_boxplot(alpha = 0.2, outlier.shape = NA, linewidth = 0.4,
               position = position_dodge(0.7)) +
  scale_colour_manual(values = STRATEGY_COLORS, labels = STRATEGY_LABELS, name = "Strategy") +
  scale_fill_manual(values   = STRATEGY_COLORS, labels = STRATEGY_LABELS, name = "Strategy") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  labs(x = "Number of Cells", y = "Correct pair selection (vs oracle)") +
  theme_paper() +
  theme(legend.key.size = unit(0.35, "cm"),
        legend.text     = element_text(size = 6.5))

ggsave(file.path(fig_root, "ort_selection_accuracy.pdf"), fig2,
       width = max(4, 1.5 * nlevels(accuracy$n_cells)),
       height = 3.5, device = cairo_pdf)

message("Wrote ort_selection_distances.pdf and ort_selection_accuracy.pdf to ", fig_root)
