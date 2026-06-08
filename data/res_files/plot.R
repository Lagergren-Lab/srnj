library(tidyverse)
library(patchwork)
library(ggbeeswarm)

# ── Shared aesthetics ──────────────────────────────────────────────────────────
METHOD_LEVELS <- c("NJ", "DLCA-NJ", "SNJ", "SRNJ", "SRNJ1", "SRNJ*",
                   "ANJ", "FastME")
METHOD_COLORS <- c(
  "NJ"      = "#E63946",
  "DLCA-NJ" = "#F4A261",
  "SNJ"     = "#2A9D8F",
  "SRNJ"    = "#457B9D",
  "SRNJ1"   = "#6A4C93",
  "SRNJ*"   = "#B5179E",
  "ANJ"     = "#4CC9F0",
  "FastME"  = "#80B918"
)

METRIC_LABELS <- c(
  rf_distance       = "RF Distance",
  quartet_distance  = "Quartet Distance",
  triplet_distance  = "Triplet Distance",
  rootsplit_distance = "Root-split Distance"
)

theme_paper <- function() {
  theme_bw(base_size = 9) +
    theme(
      strip.background = element_rect(fill = "grey92", colour = NA),
      strip.text       = element_text(face = "bold", size = 8),
      legend.position  = "bottom",
      legend.key.size  = unit(0.4, "cm"),
      legend.title     = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text        = element_text(size = 7),
      axis.title       = element_text(size = 8)
    )
}

# figure folder (override with env var FIG_ROOT)
fig_root <- Sys.getenv("FIG_ROOT", unset = ".")

method_recode <- c(
  nj = "NJ", dlca_nj = "DLCA-NJ", snj = "SNJ",
  srnj = "SRNJ", srnj1 = "SRNJ1", srnj_maxlca = "SRNJ*",
  anj = "ANJ", fastme = "FastME"
)

# ── Data ───────────────────────────────────────────────────────────────────────
inhouse  <- read_csv("inhouse_stats.csv", show_col_types = FALSE, na = c("","NA","None")) |>
  mutate(method = recode(method, !!!method_recode),
         method = factor(method, levels = METHOD_LEVELS),
         n_cells = factor(n_cells))

time_df  <- read_csv("inhouse_time_estimates.csv", show_col_types = FALSE) |>
  mutate(method = recode(method, !!!method_recode))

cn_eval  <- read_csv("cn_eval_summary_k7.csv", show_col_types = FALSE) |>
  mutate(n_cells = str_extract(n_cells, "\\d+") |> as.integer())

cpu      <- read_csv("cputime_k10.csv", show_col_types = FALSE) |>
  mutate(tool = recode(tool, sconce2 = "SCONCE2", medicc2 = "MEDICC2"))



# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – In-house distances
# ── tuneable ──────────────────────────────────────────────────────────────────
F1_METRICS <- c("rf_distance", "quartet_distance", "triplet_distance")
# options: "rf_distance", "quartet_distance", "triplet_distance", "transfer_distance"
# ──────────────────────────────────────────────────────────────────────────────
METRIC_LABELS_ALL <- c(
  rf_distance       = "RF Distance",
  quartet_distance  = "Quartet Distance",
  triplet_distance  = "Triplet Distance",
  transfer_distance = "Transfer Distance"
)

f1_dat <- inhouse |>
  filter(metric %in% F1_METRICS) |>
  mutate(metric = factor(metric, levels = F1_METRICS,
                         labels = unname(METRIC_LABELS_ALL[F1_METRICS])))

fig1_height <- 1.6 * length(F1_METRICS) + 0.5

fig1 <- ggplot(f1_dat, aes(x = method, y = dist, colour = method, fill = method)) +
  geom_boxplot(alpha = 0.25, outlier.shape = NA, linewidth = 0.4) +
  geom_beeswarm(size = 0.8, alpha = 0.7, cex = 0.6) +
  facet_grid(metric ~ n_cells, scales = "free_y",
             labeller = labeller(n_cells = \(x) paste0("N=",x))) +
  scale_colour_manual(values = METHOD_COLORS) +
  scale_fill_manual(values = METHOD_COLORS) +
  labs(x = NULL, y = "Distance") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 6.5),
        legend.position = "none")

ggsave(file.path(fig_root, "figure1_inhouse_distances.pdf"), fig1, width = 7, height = fig1_height, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – In-house: Rootsplit distance (n = 100, 200, 500)
# ══════════════════════════════════════════════════════════════════════════════
f2_dat <- inhouse |>
  filter(metric == "rootsplit_distance",
         n_cells %in% c("100","200","500"))

fig2 <- ggplot(f2_dat, aes(x = method, y = dist, colour = method, fill = method)) +
  geom_boxplot(alpha = 0.25, outlier.shape = NA, linewidth = 0.4) +
  geom_beeswarm(size = 1, alpha = 0.75, cex = 0.7) +
  facet_grid(. ~ n_cells, labeller = labeller(n_cells = \(x) paste0("N=",x))) +
  scale_colour_manual(values = METHOD_COLORS) +
  scale_fill_manual(values = METHOD_COLORS) +
  labs(x = NULL, y = "Root-split Distance") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 7))

ggsave(file.path(fig_root, "figure2_rootsplit.pdf"), fig2, width = 3.5, height = 3.2, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – 3a: SCONCE2 vs MEDICC2 (λ=5) | 3b: Δ(SCONCE2−MEDICC2) by n
# ══════════════════════════════════════════════════════════════════════════════
library(ggpubr)
library(patchwork)

ALL_TREE_LEVELS <- c("NJ","DLCA-NJ","SRNJ","SRNJ1","SRNJ*")
ALL_COLORS      <- METHOD_COLORS[ALL_TREE_LEVELS]

# helper to recode a raw stats csv
recode_stats <- function(df) {
  df |> mutate(
    tree_method = recode(tree_method, nj = "NJ", snj = "SNJ", rnj = "DLCA-NJ",
                          anj = "ANJ", fastme = "FastME", srnj = "SRNJ",
                          srnj1 = "SRNJ1", srnjmaxlca = "SRNJ*"),
    dist_method = recode(dist_method, sconce2 = "SCONCE2", med2 = "MEDICC2"),
    n_cells     = factor(n_cells),
    tree_method = factor(tree_method, levels = ALL_TREE_LEVELS)
  )
}

stats_l5 <- read_csv("stats_k7_l5.csv", show_col_types = FALSE, na = c("","NA","None")) |>
  recode_stats()
stats_l2 <- read_csv("stats_k7_l2.csv", show_col_types = FALSE, na = c("","NA","None")) |>
  recode_stats()

STATS_NEW <- file.path(Sys.getenv("STATS_K7",
  unset = "/proj/sc_ml/users/x_vitza/srnj_results/srnj_sconce2_K7/stats.csv"))

METRICS_3 <- c("rf_distance","quartet_distance","transfer_distance")
METRIC_LABS_3 <- c(rf_distance = "RF Distance", quartet_distance = "Quartet Distance",
                    transfer_distance = "Transfer Distance")
METRIC_ORDER_3 <- c("RF Distance","Quartet Distance","Transfer Distance")

prep_f3 <- function(df) {
  df |>
    filter(n_cells %in% c("50","100"), tree_method %in% ALL_TREE_LEVELS) |>
    pivot_longer(all_of(METRICS_3), names_to = "metric", values_to = "value") |>
    mutate(metric      = factor(METRIC_LABS_3[metric], levels = METRIC_ORDER_3),
           dist_method = factor(dist_method, levels = c("SCONCE2","MEDICC2")))
}



# ── 3a: Δ = SCONCE2 − MEDICC2, n=100, λ = 2 and 5 (violin) ──────────────────
delta_both <- bind_rows(
  prep_f3(stats_l2) |> mutate(lamda = "λ=2"),
  prep_f3(stats_l5) |> mutate(lamda = "λ=5")
) |>
  select(seed, n_cells, metric, tree_method, dist_method, value, lamda) |>
  pivot_wider(names_from = dist_method, values_from = value) |>
  filter(!is.na(SCONCE2), !is.na(MEDICC2),
         n_cells == as.character(max(as.integer(as.character(unique(n_cells))), na.rm = TRUE))) |>
  mutate(delta     = SCONCE2 - MEDICC2,
         col_label = lamda)

fig3a <- ggplot(delta_both, aes(x = tree_method, y = delta,
                                 colour = tree_method, fill = tree_method)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed", colour = "grey50") +
  geom_violin(alpha = 0.3, linewidth = 0.4) +
  geom_beeswarm(size = 0.9, alpha = 0.6, cex = 0.6) +
  facet_grid(metric ~ col_label, scales = "free_y") +
  scale_colour_manual(values = ALL_COLORS, name = "Method") +
  scale_fill_manual(values   = ALL_COLORS, name = "Method") +
  labs(x = NULL, y = "Δ (SCONCE2 − MEDICC2)") +
  guides(colour = guide_legend(nrow = 1),
         fill   = guide_legend(nrow = 1)) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        panel.spacing.x = unit(0.15, "cm"))

# ── 3b: dodged boxplot SCONCE2 vs MEDICC2, n=100, λ=5 ────────────────────────
pval_lbl <- function(p) {
  ifelse(is.na(p), "ns",
    ifelse(p < 1e-10, sprintf("p<%.0e", 1e-10),
    ifelse(p < 0.001, sprintf("p=%.2e", p),
    ifelse(p < 0.05,  sprintf("p=%.3f", p),
                      sprintf("p=%.2f", p)))))
}
f3b_dat <- if (file.exists(STATS_NEW)) {
  read_csv(STATS_NEW, show_col_types = FALSE, na = c("","NA","None")) |>
    recode_stats() |>
    filter(as.character(n_cells) == "100", lamda == 5,
           as.character(seq_error) == "0.02",
           tree_method %in% ALL_TREE_LEVELS) |>
    pivot_longer(all_of(METRICS_3), names_to = "metric", values_to = "value") |>
    mutate(metric      = factor(METRIC_LABS_3[metric], levels = METRIC_ORDER_3),
           dist_method = factor(dist_method, levels = c("SCONCE2","MEDICC2")),
           col_label   = "N=100, λ=5")
} else {
  f3b_n <- as.character(max(as.integer(as.character(unique(stats_l5$n_cells))), na.rm = TRUE))
  prep_f3(stats_l5) |>
    filter(n_cells == f3b_n) |>
    mutate(col_label = paste0("N=", f3b_n, ", λ=5"),
           metric    = factor(metric, levels = METRIC_ORDER_3))
}

# significance on this subset
sig_wide_b <- f3b_dat |>
  select(seed, n_cells, metric, tree_method, dist_method, value) |>
  pivot_wider(names_from = c(tree_method, dist_method), values_from = value, names_sep = "_")

y_max_b <- f3b_dat |> group_by(metric) |>
  summarise(y_max = max(value, na.rm = TRUE), .groups = "drop") |>
  mutate(metric = factor(metric, levels = METRIC_ORDER_3))

groups_b  <- f3b_dat |> distinct(metric)
sig_rows_b <- lapply(seq_len(nrow(groups_b)), function(i) {
  mt <- as.character(groups_b$metric[i])
  d  <- sig_wide_b |> filter(metric == mt)
  p_A <- suppressWarnings(wilcox.test(d[["NJ_SCONCE2"]], d[["DLCA-NJ_SCONCE2"]],
                                       paired = TRUE)$p.value)
  p_B <- suppressWarnings(wilcox.test(d[["DLCA-NJ_SCONCE2"]], d[["DLCA-NJ_MEDICC2"]],
                                       paired = TRUE)$p.value)
  cat(sprintf("3b λ=5 n=100 | %s | NJ_SC vs DLCANJ_SC: p=%.4f | DLCANJ_SC vs DLCANJ_MD: p=%.4f\n",
              mt, p_A, p_B))
  data.frame(metric = mt, p_A = p_A, p_B = p_B, stringsAsFactors = FALSE)
}) |> bind_rows() |>
  mutate(metric = factor(metric, levels = METRIC_ORDER_3))

sig_dat_b  <- sig_rows_b |> left_join(y_max_b, by = "metric") |>
  mutate(lbl_A = pval_lbl(p_A), lbl_B = pval_lbl(p_B))
sig_A_b <- sig_dat_b |> mutate(x1 = 0.8, x2 = 1.8, y = y_max * 0.70, label = lbl_A,
                                metric = factor(metric, levels = METRIC_ORDER_3))
sig_B_b <- sig_dat_b |> mutate(x1 = 1.8, x2 = 2.2, y = y_max * 0.75, label = lbl_B,
                                metric = factor(metric, levels = METRIC_ORDER_3))
sig_segs_b <- bind_rows(sig_A_b, sig_B_b) |>
  mutate(metric = factor(metric, levels = METRIC_ORDER_3))

fig3b <- ggplot(f3b_dat, aes(x = tree_method, y = value,
                              fill = tree_method, alpha = dist_method,
                              colour = dist_method)) +
  geom_boxplot(linewidth = 0.35, outlier.shape = NA,
               position = position_dodge2(width = 0.8, preserve = "single")) +
  geom_segment(data = sig_segs_b,
               aes(x = x1, xend = x2, y = y, yend = y),
               inherit.aes = FALSE, linewidth = 0.35, colour = "grey30") +
  geom_text(data = sig_segs_b,
            aes(x = (x1 + x2) / 2, y = y * 1.06, label = label),
            inherit.aes = FALSE, size = 2.0, colour = "grey20") +
  facet_grid(metric ~ col_label, scales = "free_y") +
  scale_fill_manual(values   = ALL_COLORS, guide = "none") +
  scale_colour_manual(values = c(SCONCE2 = "grey20", MEDICC2 = "grey60"), name = "Tool") +
  scale_alpha_manual(values  = c(SCONCE2 = 0.9, MEDICC2 = 0.4), name = "Tool") +
  labs(x = NULL, y = "Tree Distance") +
  guides(alpha  = guide_legend(nrow = 1),
         colour = guide_legend(nrow = 1)) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6))

ggsave(file.path(fig_root, "figure3a_delta.pdf"), fig3a, width = 5.5, height = 6.5, device = cairo_pdf)
ggsave(file.path(fig_root, "figure3b_tools.pdf"), fig3b, width = 3.0, height = 6.5, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – CN error (appendix, full width) — boxplots per cn_type
# ══════════════════════════════════════════════════════════════════════════════
f4_dat <- cn_eval |>
  pivot_longer(c(hamming_dist_avg, s2_dist_avg),
               names_to = "error_type", values_to = "value") |>
  mutate(error_type = factor(error_type,
           levels = c("hamming_dist_avg","s2_dist_avg"),
           labels = c("Avg. Hamming Distance","Avg. S² Distance")),
         n_cells = factor(n_cells),
         cn_type = str_to_title(cn_type))

fig4 <- ggplot(f4_dat, aes(x = n_cells, y = value,
                            fill = cn_type, colour = cn_type)) +
  geom_boxplot(alpha = 0.35, linewidth = 0.4, outlier.size = 0.7,
               position = position_dodge(0.75)) +
  facet_wrap(~ error_type, scales = "free_y", nrow = 1) +
  scale_fill_manual(values   = c(Mean = "#E76F51", Median = "#2A9D8F", Mode = "#457B9D")) +
  scale_colour_manual(values = c(Mean = "#E76F51", Median = "#2A9D8F", Mode = "#457B9D")) +
  labs(x = "Number of Cells", y = "Error", fill = "CN type", colour = "CN type") +
  theme_paper()

ggsave(file.path(fig_root, "figure4_cn_error.pdf"), fig4, width = 7, height = 3, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Time: num_calls × min SCONCE2 pp time
# ══════════════════════════════════════════════════════════════════════════════
pp_mean <- cpu |> filter(tool == "SCONCE2") |> pull(avg_pp) |> mean()

time_ext <- time_df |> select(n_cells, seed, method, num_calls)

message("Unique n_cells in time_df: ", paste(sort(unique(time_ext$n_cells)), collapse = ", "))
message("Unique methods in time_df: ", paste(sort(unique(time_ext$method)), collapse = ", "))

f5_base <- time_ext |>
  mutate(total_time = num_calls * pp_mean) |>
  group_by(n_cells, method) |>
  summarise(mean_t = mean(total_time, na.rm = TRUE),
            sd_t   = sd(total_time,   na.rm = TRUE), .groups = "drop")

message("n_cells in f5_base: ", paste(sort(unique(f5_base$n_cells)), collapse = ", "))

f5_all <- f5_base |>
  mutate(method  = factor(method, levels = METHOD_LEVELS),
         n_cells = as.integer(n_cells),
         mean_t  = mean_t / 3600,
         sd_t    = replace_na(sd_t / 3600, 0)) |>
  filter(!is.na(method), !method %in% c("ANJ","FastME"))

message("n_cells in f5_all after factor filter: ", paste(sort(unique(f5_all$n_cells)), collapse = ", "))
message("methods in f5_all: ", paste(as.character(unique(f5_all$method)), collapse = ", "))

pd <- position_dodge(width = 20)

fig5 <- ggplot(f5_all, aes(x = n_cells, y = mean_t, colour = method, group = method)) +
  geom_line(linewidth = 0.7, position = pd) +
  geom_point(size = 1.8, position = pd) +
  geom_errorbar(aes(ymin = pmax(mean_t - sd_t, 0), ymax = mean_t + sd_t),
                width = 10, linewidth = 0.35, position = pd) +
  scale_colour_manual(values = METHOD_COLORS, name = "Method") +
  scale_x_continuous(breaks = c(20, 50, 100, 200, 500, 1000),
                     limits = c(NA, 1080)) +
  labs(x = "Number of Cells", y = "Est. Total Time (core-h)") +
  guides(colour = guide_legend(nrow = 3)) +
  theme_paper()

ggsave(file.path(fig_root, "figure5_time.pdf"), fig5, width = 3.5, height = 3.8, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Real 10x: F1 (and optionally Parsimony) by method
# ── tuneable ──────────────────────────────────────────────────────────────────
F6_METHODS       <- c("DLCA-NJ", "NJ", "SRNJ")   # order sets x-axis order
F6_SHOW_PARSIMONY <- FALSE                         # TRUE to add Parsimony facet
# ──────────────────────────────────────────────────────────────────────────────
real10x <- bind_rows(
  read_csv("real_10x_scores_sconce2.csv", show_col_types = FALSE) |> mutate(tool = "SCONCE2"),
  read_csv("real_10x_scores_medicc2.csv", show_col_types = FALSE) |> mutate(tool = "MEDICC2")
) |>
  filter(Method %in% F6_METHODS) |>
  mutate(Method = factor(Method, levels = F6_METHODS),
         tool   = factor(tool, levels = c("SCONCE2","MEDICC2")))

f6_metrics <- c("F1Score")
if (F6_SHOW_PARSIMONY) f6_metrics <- c("ParsimonyScore", f6_metrics)

f6_dat <- real10x |>
  pivot_longer(all_of(f6_metrics), names_to = "metric", values_to = "value") |>
  mutate(metric = factor(metric,
           levels = c("ParsimonyScore","F1Score"),
           labels = c("Parsimony Score","F1 Score")))

# center parsimony by per-sample mean; keep F1 raw
f6_dat <- f6_dat |>
  group_by(Sample, metric, tool) |>
  mutate(value_plot = if_else(metric == "Parsimony Score",
                               value - mean(value, na.rm = TRUE),
                               value)) |>
  ungroup()

# Paired Wilcoxon: DLCA-NJ vs NJ within each tool, per metric
f6_wide <- f6_dat |>
  select(Sample, metric, tool, Method, value_plot) |>
  pivot_wider(names_from = Method, values_from = value_plot)

f6_sig_rows <- lapply(unique(as.character(f6_dat$metric)), function(mt) {
  lapply(c("SCONCE2","MEDICC2"), function(tl) {
    d <- f6_wide |> filter(metric == mt, tool == tl)
    p <- suppressWarnings(wilcox.test(d[["DLCA-NJ"]], d[["NJ"]],
                                       paired = TRUE)$p.value)
    cat(sprintf("Fig6 | %s | %s | DLCA-NJ vs NJ: p=%.4f\n", mt, tl, p))
    data.frame(metric = mt, tool = tl, p = p, stringsAsFactors = FALSE)
  }) |> bind_rows()
}) |> bind_rows()

f6_ymax <- f6_dat |> group_by(metric) |>
  summarise(ymax = max(value_plot, na.rm = TRUE), .groups = "drop")

# x positions: DLCA-NJ=1, NJ=2 (dodged: SCONCE2 left, MEDICC2 right)
# dodge offset ≈ 0.2 within each method position
f6_sig <- f6_sig_rows |>
  left_join(f6_ymax, by = "metric") |>
  mutate(
    lbl  = pval_lbl(p),
    x1   = if_else(tool == "SCONCE2", 0.8, 1.2),   # DLCA-NJ dodged pos
    x2   = if_else(tool == "SCONCE2", 1.8, 2.2),   # NJ dodged pos
    y    = if_else(tool == "SCONCE2", ymax * 1.06, ymax * 1.13),
    metric = factor(metric, levels = levels(f6_dat$metric))
  )


fig6_height <- if (F6_SHOW_PARSIMONY) 5 else 3

fig6 <- ggplot(f6_dat, aes(x = Method, y = value_plot, fill = Method,
                            alpha = tool, colour = tool)) +
  geom_boxplot(linewidth = 0.35, outlier.shape = NA,
               position = position_dodge2(width = 0.8, preserve = "single")) +
  geom_segment(data = f6_sig,
               aes(x = x1, xend = x2, y = y, yend = y),
               inherit.aes = FALSE, linewidth = 0.35, colour = "grey30") +
  geom_text(data = f6_sig,
            aes(x = (x1 + x2) / 2, y = y * 1.02, label = lbl),
            inherit.aes = FALSE, size = 1.8, colour = "grey20") +
  facet_wrap(~ metric, scales = "free_y", ncol = 1,
             labeller = as_labeller(\(x) ifelse(x == "Parsimony Score",
                                                "Parsimony (Δ from mean)", x))) +
  scale_fill_manual(values   = METHOD_COLORS[F6_METHODS], name = "Method") +
  scale_colour_manual(values = c(SCONCE2 = "grey20", MEDICC2 = "grey60"), name = "Tool") +
  scale_alpha_manual(values  = c(SCONCE2 = 0.9, MEDICC2 = 0.4), name = "Tool") +
  labs(x = NULL, y = NULL) +
  guides(fill   = guide_legend(nrow = 3, override.aes = list(alpha = 0.85, colour = "grey30")),
         alpha  = guide_legend(nrow = 2),
         colour = guide_legend(nrow = 2)) +
  theme_paper() +
  theme(legend.key.size = unit(0.3, "cm"),
        legend.text      = element_text(size = 6.5),
        legend.title     = element_text(size = 7),
        legend.box       = "horizontal")

# add zero reference line to parsimony facet only when shown
if (F6_SHOW_PARSIMONY) {
  fig6 <- fig6 + geom_hline(yintercept = 0, linewidth = 0.3,
                              linetype = "dashed", colour = "grey50")
}

ggsave(file.path(fig_root, "figure6_real10x.pdf"), fig6, width = 3, height = fig6_height, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Full comparison: all n, all λ, all distances (appendix)
# ══════════════════════════════════════════════════════════════════════════════
METRICS_7     <- c("rf_distance","quartet_distance","transfer_distance","triplet_distance")
METRIC_LABS_7 <- c(rf_distance       = "RF Distance",
                   quartet_distance  = "Quartet Distance",
                   transfer_distance = "Transfer Distance",
                   triplet_distance  = "Triplet Distance")

f7_dat <- read_csv(STATS_NEW, show_col_types = FALSE, na = c("","NA","None")) |>
  recode_stats() |>
  filter(as.character(n_cells) %in% c("50","100"),
         as.character(seq_error) == "0.02",
         tree_method %in% ALL_TREE_LEVELS) |>
  pivot_longer(all_of(METRICS_7), names_to = "metric", values_to = "value") |>
  mutate(
    metric      = factor(metric, levels = METRICS_7, labels = unname(METRIC_LABS_7)),
    dist_method = factor(dist_method, levels = c("SCONCE2","MEDICC2")),
    lamda       = paste0("λ=", lamda),
    col_label   = factor(paste0("N=",n_cells, ", ", lamda),
                         levels = c("N=50, λ=2","N=50, λ=5",
                                    "N=100, λ=2","N=100, λ=5"))
  )

fig7 <- ggplot(f7_dat, aes(x = tree_method, y = value,
                            fill = tree_method, alpha = dist_method,
                            colour = dist_method)) +
  geom_boxplot(linewidth = 0.3, outlier.shape = NA,
               position = position_dodge2(width = 0.8, preserve = "single")) +
  facet_grid(metric ~ col_label, scales = "free_y") +
  scale_fill_manual(values   = ALL_COLORS, name = "Method") +
  scale_colour_manual(values = c(SCONCE2 = "grey20", MEDICC2 = "grey60"), name = "Tool") +
  scale_alpha_manual(values  = c(SCONCE2 = 0.9, MEDICC2 = 0.4), name = "Tool") +
  labs(x = NULL, y = "Tree Distance") +
  guides(fill   = guide_legend(nrow = 1, override.aes = list(alpha = 0.85, colour = "grey30")),
         alpha  = guide_legend(nrow = 1),
         colour = guide_legend(nrow = 1)) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5.5),
        panel.spacing.x = unit(0.12, "cm"))

n_f7_cols <- length(unique(f7_dat$col_label))
ggsave(file.path(fig_root, "figure7_full_comparison.pdf"), fig7,
       width = max(10, 1.8 * n_f7_cols), height = 7, device = cairo_pdf)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Seq-error comparison: ε=0.02 vs ε=0.04 (appendix)
# ══════════════════════════════════════════════════════════════════════════════
if (file.exists(STATS_NEW)) {
  recode_stats_se <- function(df) {
    df |> mutate(
      tree_method = recode(tree_method, nj = "NJ", snj = "SNJ", rnj = "DLCA-NJ",
                            anj = "ANJ", fastme = "FastME", srnj = "SRNJ",
                            srnj1 = "SRNJ1", srnjmaxlca = "SRNJ*"),
      dist_method = recode(dist_method, sconce2 = "SCONCE2", med2 = "MEDICC2"),
      n_cells     = factor(n_cells),
      seq_error   = factor(seq_error, levels = c("0.02","0.04"),
                            labels = c("ε=0.02 (low)", "ε=0.04 (high)")),
      tree_method = factor(tree_method, levels = ALL_TREE_LEVELS)
    )
  }

  stats_se <- read_csv(STATS_NEW, show_col_types = FALSE, na = c("","NA","None")) |>
    mutate(seq_error = as.character(seq_error)) |>
    recode_stats_se()

  METRICS_8     <- c("rf_distance","quartet_distance","transfer_distance","triplet_distance")
  METRIC_LABS_8 <- c(rf_distance       = "RF Distance",
                     quartet_distance  = "Quartet Distance",
                     transfer_distance = "Transfer Distance",
                     triplet_distance  = "Triplet Distance")

  F8_LAMBDA <- 5   # fix λ=5 for seq-error comparison
  f8_n_max  <- as.character(max(as.integer(as.character(unique(stats_se$n_cells))), na.rm = TRUE))

  f8_dat <- stats_se |>
    filter(lamda == F8_LAMBDA,
           n_cells %in% c("50", f8_n_max),
           tree_method %in% ALL_TREE_LEVELS) |>
    pivot_longer(all_of(METRICS_8), names_to = "metric", values_to = "value") |>
    mutate(metric     = factor(metric, levels = METRICS_8,
                                labels = unname(METRIC_LABS_8)),
           dist_method = factor(dist_method, levels = c("SCONCE2","MEDICC2")),
           col_label  = factor(
             paste0("N=",n_cells, ", λ=", F8_LAMBDA, ", ", seq_error),
             levels = c(
               paste0("N=50, λ=",      F8_LAMBDA, ", ε=0.02 (low)"),
               paste0("N=50, λ=",      F8_LAMBDA, ", ε=0.04 (high)"),
               paste0("N=",f8_n_max, ", λ=", F8_LAMBDA, ", ε=0.02 (low)"),
               paste0("N=",f8_n_max, ", λ=", F8_LAMBDA, ", ε=0.04 (high)")
             )))

  fig8 <- ggplot(f8_dat, aes(x = tree_method, y = value,
                              fill = tree_method, alpha = dist_method,
                              colour = dist_method)) +
    geom_boxplot(linewidth = 0.3, outlier.shape = NA,
                 position = position_dodge2(width = 0.8, preserve = "single")) +
    facet_grid(metric ~ col_label, scales = "free_y") +
    scale_fill_manual(values   = ALL_COLORS, name = "Method") +
    scale_colour_manual(values = c(SCONCE2 = "grey20", MEDICC2 = "grey60"), name = "Tool") +
    scale_alpha_manual(values  = c(SCONCE2 = 0.9, MEDICC2 = 0.4), name = "Tool") +
    labs(x = NULL, y = "Tree Distance") +
    guides(fill   = guide_legend(nrow = 1, override.aes = list(alpha = 0.85, colour = "grey30")),
           alpha  = guide_legend(nrow = 1),
           colour = guide_legend(nrow = 1)) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5.5),
          panel.spacing.x = unit(0.12, "cm"))

  n_f8_cols <- length(unique(f8_dat$col_label))
  ggsave(file.path(fig_root, "figure8_seq_error.pdf"), fig8,
         width = max(10, 1.8 * n_f8_cols), height = 7, device = cairo_pdf)
} else {
  message("Skipping Figure 8: stats file not found at ", STATS_NEW)
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Orienting-leaf selection: tree distances per strategy
# ══════════════════════════════════════════════════════════════════════════════
ORT_METRICS_CSV <- Sys.getenv("ORT_METRICS_CSV", unset = "ort_metrics.csv")

if (file.exists(ORT_METRICS_CSV)) {
  ORT_STRATEGY_LEVELS <- c(
    "dlca_nj", "srnj_min_D", "srnj_max_lca",
    "gt_min", "gt_max_lca",
    "hamming", "hamming_max_lca",
    "nll", "nll_max_lca"
  )
  ORT_STRATEGY_LABELS <- c(
    dlca_nj         = "DLCA-NJ",
    srnj_min_D      = "SRNJ (min d)",
    srnj_max_lca    = "SRNJ (max LCA)",
    gt_min          = "GT (min d)",
    gt_max_lca      = "GT (max LCA)",
    hamming         = "Hamming (min d)",
    hamming_max_lca = "Hamming (max LCA)",
    nll             = "NLL (min d)",
    nll_max_lca     = "NLL (max LCA)"
  )
  ORT_COLORS <- c(
    dlca_nj         = "#595959",
    srnj_min_D      = "#1F4E79",
    srnj_max_lca    = "#7FA8D0",
    gt_min          = "#B2182B",
    gt_max_lca      = "#EF8A8A",
    hamming         = "#1B7837",
    hamming_max_lca = "#7FBF7B",
    nll             = "#762A83",
    nll_max_lca     = "#C2A5CF"
  )
  ORT_METRIC_LEVELS <- c("rf_distance","quartet_distance","triplet_distance","transfer_distance")
  ORT_METRIC_LABELS <- c(
    rf_distance       = "RF Distance",
    quartet_distance  = "Quartet Distance",
    triplet_distance  = "Triplet Distance",
    transfer_distance = "Transfer Distance"
  )

  ort_metrics <- read_csv(ORT_METRICS_CSV, show_col_types = FALSE,
                           na = c("","NA","None")) |>
    filter(metric %in% ORT_METRIC_LEVELS, !is.na(dist)) |>
    mutate(
      strategy = factor(strategy, levels = ORT_STRATEGY_LEVELS),
      metric   = factor(metric, levels = ORT_METRIC_LEVELS,
                        labels = unname(ORT_METRIC_LABELS[ORT_METRIC_LEVELS])),
      n_cells  = factor(n_cells, levels = sort(unique(as.integer(n_cells))))
    ) |>
    filter(!is.na(strategy))

  n_ncells_ort <- nlevels(droplevels(ort_metrics$n_cells))
  n_metrics_ort <- nlevels(droplevels(ort_metrics$metric))

  # ── Significance brackets: NLL(min d) vs GT(min d) and NLL(min d) vs Hamming(min d)
  # Brackets are drawn in the "pre-flip" coordinate system:
  #   x  = strategy position (numeric, on the discrete axis)
  #   y  = dist (will become the horizontal axis after coord_flip)
  # Strategy positions in rev(ORT_STRATEGY_LEVELS): nll=2, hamming=4, gt_min=6
  ort_sig <- ort_metrics |>
    group_by(metric, n_cells) |>
    summarise(
      y_max    = max(dist, na.rm = TRUE),
      p_nll_gt  = suppressWarnings(
        wilcox.test(dist[strategy == "nll"], dist[strategy == "gt_min"],
                    paired = TRUE)$p.value),
      p_nll_ham = suppressWarnings(
        wilcox.test(dist[strategy == "nll"], dist[strategy == "hamming"],
                    paired = TRUE)$p.value),
      .groups = "drop"
    ) |>
    mutate(
      lbl_gt  = ifelse(is.na(p_nll_gt),  "ns",
                ifelse(p_nll_gt  < 0.001, sprintf("p=%.2e", p_nll_gt),
                ifelse(p_nll_gt  < 0.05,  sprintf("p=%.3f", p_nll_gt), "ns"))),
      lbl_ham = ifelse(is.na(p_nll_ham), "ns",
                ifelse(p_nll_ham < 0.001, sprintf("p=%.2e", p_nll_ham),
                ifelse(p_nll_ham < 0.05,  sprintf("p=%.3f", p_nll_ham), "ns"))),
      # bracket heights (on dist axis, extended beyond y_max)
      y1 = y_max * 1.06,
      y2 = y_max * 1.14,
      # strategy x positions (numeric in rev order: 1=nll_max_lca,2=nll,4=hamming,6=gt_min)
      x_nll = 2, x_ham = 4, x_gt = 6
    )

  # build segment data: two brackets per facet cell
  ort_segs <- bind_rows(
    ort_sig |> mutate(x1 = x_nll, x2 = x_gt,  y = y1, label = lbl_gt,  comparison = "nll_vs_gt"),
    ort_sig |> mutate(x1 = x_nll, x2 = x_ham, y = y2, label = lbl_ham, comparison = "nll_vs_ham")
  )

  fig9 <- ggplot(ort_metrics, aes(x = strategy, y = dist,
                                   fill = strategy, colour = strategy)) +
    geom_boxplot(alpha = 0.75, outlier.shape = NA, linewidth = 0.3) +
    geom_beeswarm(size = 0.45, alpha = 0.55, cex = 0.5) +
    geom_segment(data = ort_segs,
                 aes(x = x1, xend = x2, y = y, yend = y),
                 inherit.aes = FALSE, linewidth = 0.3, colour = "grey30") +
    geom_text(data = ort_segs,
              aes(x = (x1 + x2) / 2, y = y * 1.03, label = label),
              inherit.aes = FALSE, size = 1.8, colour = "grey20") +
    facet_grid(metric ~ n_cells, scales = "free_x",
               labeller = labeller(n_cells = \(x) paste0("N=",x))) +
    scale_fill_manual(values = ORT_COLORS) +
    scale_colour_manual(values = ORT_COLORS) +
    scale_x_discrete(limits = rev(ORT_STRATEGY_LEVELS),
                     labels = ORT_STRATEGY_LABELS) +
    coord_flip() +
    labs(x = NULL, y = "Distance") +
    theme_paper() +
    theme(legend.position = "none")

  ggsave(file.path(fig_root, "figure9_ort_selection.pdf"), fig9,
         width = max(7, 1.9 * n_ncells_ort),
         height = 1.9 * n_metrics_ort + 0.8,
         device = cairo_pdf)
} else {
  message("Skipping Figure 9: ", ORT_METRICS_CSV, " not found")
}

message("All figures written.")

