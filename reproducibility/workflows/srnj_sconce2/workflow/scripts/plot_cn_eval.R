library(ggplot2)
library(tidyr)
library(dplyr)

# first argument is the input file path
# second argument is the output file path
if (length(commandArgs(trailingOnly = TRUE)) < 2) {
  stop("Usage: Rscript plot_cn_eval.R <input_file.csv> <output_file.png>")
}
input_file <- commandArgs(trailingOnly = TRUE)[1]
output_file <- commandArgs(trailingOnly = TRUE)[2]

# Read data
df <- read.csv(input_file)

# Reshape data to long format for distance metrics
df_long <- df %>%
  pivot_longer(cols = c(euclidean_dist_avg, hamming_dist_avg, s2_dist_avg),
               names_to = "dist_metric",
               values_to = "distance") %>%
  mutate(dist_metric = gsub("_dist_avg", "", dist_metric))

# Create plot
p <- ggplot(df_long, aes(x = cn_type, y = distance, fill = cn_type)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, size = 1, alpha = 0.5) +
  scale_fill_manual(values = c("mode" = "#E69F00",
                                "median" = "#56B4E9",
                                "mean" = "#009E73")) +
  facet_grid(dist_metric ~ n_cells, scales = "free_y") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "CN Type", y = "Distance",
       fill = "CN Type",
       title = "Copy Number Distance Comparison")

# Save plot
ggsave(output_file, plot = p, width = 12, height = 8)