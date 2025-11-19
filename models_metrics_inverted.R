library(readr)
library(dplyr)
library(tidyr)
library(knitr)

df <- read_csv("z3_benchmark_metrics_results.csv")

df_num <- df %>% 
  select(where(is.numeric))

stats <- df_num %>%
  summarise(
    across(
      everything(),
      list(
        min = ~min(.x, na.rm = TRUE),
        max = ~max(.x, na.rm = TRUE),
        median = ~median(.x, na.rm = TRUE),
        mean = ~mean(.x, na.rm = TRUE),
        sd = ~sd(.x, na.rm = TRUE)
      )
    )
  )

stats_long <- stats %>%
  pivot_longer(
    cols = everything(),
    names_to = c("Variable", "Statistic"),
    names_sep = "_",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Variable,
    values_from = Value
  ) %>%
  arrange(factor(Statistic, levels = c("min", "max", "median", "mean", "sd"))) %>%
  rename(Metric = Statistic)   # <-- renombramos para que aparezca claramente

# Mostrar tabla en LaTeX
kable(stats_long, format = "latex", digits = 3)
