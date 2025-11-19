library(ggplot2)
library(readr)
library(dplyr)

# Carga del CSV (cambia 'datos.csv' por tu archivo)
df <- read_csv("z3_benchmark_results.csv")

# Asegurar que Operation es factor
df$Operation <- as.factor(df$Operation)

# Eliminar una operación concreta
df <- df %>% 
  filter(Operation != "Z3ConfigurationsNumber")

# Boxplot con Time_Mean (s)
ggplot(df, aes(x = Operation, y = `Time_Mean (s)`)) +
  geom_boxplot(fill = "skyblue") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 14)
  ) +
  labs(
    title = "Execution Time per Operation",
    x = "Operation",
    y = "Time Mean (seconds)"
  )
