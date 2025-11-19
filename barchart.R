library(ggplot2)
library(dplyr)
library(readr)

# Cargar CSV
df <- read_csv("z3_benchmark_results.csv")

# Convertir Operation a factor
df$Operation <- as.factor(df$Operation)

# Eliminar una operación concreta
df_filtered <- df %>% 
  filter(Operation != "Z3ConfigurationsNumber")

# Gráfico de barras usando Time_Mean (s)
ggplot(df_filtered, aes(x = Operation, y = `Time_Mean (s)`)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 14)
  ) +
  labs(
    title = "Execution Time per Operation (Mean)",
    x = "Operation",
    y = "Time Mean (seconds)"
  )
