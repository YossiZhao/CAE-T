# Load necessary libraries
library(ggplot2)
library(reshape2)

# Create the data
data <- data.frame(
  Classifiers = c("1-layer MLPs", "3-layer MLPs", "Single-head Transformer"),
  Sensitivity = c(81.7, 68.7, 72.8),
  Specificity = c(72.6, 84.9, 84.5),
  Accuracy = c(76.8, 77.4, 79.1)
)

# Reshape data to long format
data_long <- melt(data, id.vars = "Classifiers", variable.name = "Metrics", value.name = "Percentage")

# Plot the bar chart
ggplot(data_long, aes(x = Metrics, y = Percentage, fill = Classifiers)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Comparison of Classifiers on Per-Signal Metrics",
    x = "Metrics",
    y = "Percentage (%)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))