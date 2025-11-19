library(readr)
library(knitr)

# 1) Leer CSV
df <- read_csv("z3_benchmark_metrics_results.csv")

# 2) Seleccionar columnas numéricas
df_num <- df[ , sapply(df, is.numeric)]

# 3) Calcular estadísticas
stats_mat <- sapply(df_num, function(x) {
  c(
    min    = min(x, na.rm = TRUE),
    max    = max(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    mean   = mean(x, na.rm = TRUE),
    sd     = sd(x, na.rm = TRUE)
  )
})

stats_mat <- stats_mat[c("min", "max", "median", "mean", "sd"), , drop = FALSE]

# 4) Convertir a dataframe y formatear
stats_df <- as.data.frame(stats_mat)

# Filas 1:4 como enteros, fila 5 con 3 decimales
stats_df[1:5, ] <- lapply(stats_df[1:5, ], function(x) format(round(x, 0), nsmall = 0))
#stats_df[5, ]   <- lapply(stats_df[5, ], function(x) format(round(x, 3), nsmall = 3))

# 5) Generar tabla LaTeX
kable(stats_df, format = "latex", booktabs = TRUE)
