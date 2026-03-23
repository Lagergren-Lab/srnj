library(ggplot2)
library(tidyr)
library(ggpubr)
library(dplyr) # Added for reliable sorting

# Arguments check
if (length(commandArgs(trailingOnly = TRUE)) < 2) {
  stop("Usage: Rscript plot_results.R <input_file.csv> <output_file.png>")
}
input_file <- commandArgs(trailingOnly = TRUE)[1]
output_file <- commandArgs(trailingOnly = TRUE)[2]

# 1. Read and strictly sort data
# We sort by everything EXCEPT the variables being compared (dist_method/tree_method)
# This ensures that for any given subset, the rows for 'sconce2' and 'med2' 
# appear in the exact same seed order for the paired Wilcoxon test.
df <- read.csv(input_file) %>%
  arrange(n_cells, n_bins, n_clones, lamda, n_chrom, seed, dist_method, tree_method)

# 2. Reshape data to long format
df_long <- pivot_longer(df,
                        cols = c(quartet_distance, rf_distance, transfer_distance, triplet_distance, rootsplit_distance),
                        names_to = "dist_metric",
                        values_to = "score")

df_long$n_cells <- as.factor(df_long$n_cells)

# 3. Create interaction variable for x-axis
df_long$method_combo <- interaction(df_long$tree_method, df_long$dist_method, sep = "_")

# Define comparisons for the plot
my_comparisons <- list(c("rnj_sconce2", "rnj_med2"), 
                       c("nj_sconce2", "rnj_sconce2"))

# 4. Create plot
p <- ggplot(df_long, aes(x = method_combo, y = score, fill = dist_method)) +
  # Use outlier.shape = NA because geom_jitter already shows the points
  geom_boxplot(aes(alpha = tree_method), outlier.shape = NA) + 
  geom_jitter(aes(alpha = tree_method), width = 0.2, size = 1) +
  
  # Paled Wilcoxon test
  stat_compare_means(comparisons = my_comparisons,
                     method = "wilcox.test",
                     paired = TRUE,  # Forces the paired test
                     label = "p.signif") +
  
  scale_fill_manual(values = c("sconce2" = "#E69F00", "med2" = "#56B4E9")) +
  scale_alpha_manual(values = c("anj" = 0.3, "fastme" = 0.4, "nj" = 0.5, "snj" = 0.6,
                               "rnj" = 0.7, "srnj" = 0.8, "srnj1" = 0.9, "srnjmaxlca" = 1.0)) +
  
  facet_grid(dist_metric ~ n_cells, scales = "free_y") +
  
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    # Force white background for Linux/Mac consistency in Dark Mode
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  ) +
  labs(x = "Method", y = "Distance Score",
       fill = "Distance Method", alpha = "Tree Method")

# 5. Save plot with explicit white background
ggsave(output_file, plot = p, width = 12, height = 12, bg = "white")
