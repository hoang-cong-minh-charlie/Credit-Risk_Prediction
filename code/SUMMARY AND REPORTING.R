# ==============================================================================
# SUMMARY AND REPORTING
# ==============================================================================

# 1. Function for Numeric Summary
summ_numeric <- function(x) {
  c(
    count = sum(!is.na(x)),
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    q1 = quantile(x, 0.25, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    q3 = quantile(x, 0.75, na.rm = TRUE),
    max = max(x, na.rm = TRUE)
  )
}

# 2. Function for Factor Summary
summ_factor <- function(df, var) {
  var_sym <- sym(var)
  
  summary_tbl <- df %>%
    filter(!is.na(!!var_sym)) %>%
    group_by(!!var_sym) %>%
    summarise(Count = n(), .groups = 'drop') %>%
    mutate(
      Total_Valid_N = sum(Count),
      Percentage = (Count / Total_Valid_N) * 100
    ) %>%
    rename(Level = !!var_sym) %>%
    mutate(
      Variable = var,
      Type = "Factor"
    ) %>%
    select(Variable, Type, Level, Count, Percentage, Total_Valid_N)
    
  return(summary_tbl)
}

# ----------------------------------------------------------------------
#  Average Summary of Metrics
# ----------------------------------------------------------------------
metrics_summary <- metrics_runs %>% 
    group_by(Model) %>% 
    summarise(
        N = n(), 
        across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}")
    )

# ----------------------------------------------------------------------
#  Paired T-Test Comparison (Stacking vs Base)
# ----------------------------------------------------------------------
metric_names <- c("Accuracy","Precision","Recall","F1","AUC","Brier")
t_results <- list()

for (m in metric_names) {
    for (pair in list(c("XGBoost","Stacking"), c("GAM (bam)","Stacking"))) {
        vals1 <- metrics_runs %>% filter(Model == pair[1]) %>% pull(!!sym(m)) 
        vals2 <- metrics_runs %>% filter(Model == pair[2]) %>% pull(!!sym(m))
        test <- t.test(vals1, vals2, paired = TRUE) 
        t_results[[length(t_results)+1]] <- tibble(Metric = m, Comparison = paste(pair[1], "vs", pair[2]), p_value = test$p.value)
    }
}
t_results <- bind_rows(t_results)

# ----------------------------------------------------------------------
#  Summary of Mean and SD of Meta-Weights
# ----------------------------------------------------------------------
weights_summary <- weights_runs %>% 
    group_by(term) %>% 
    summarise(
        N = n(), 
        Mean_Estimate = mean(estimate, na.rm = TRUE), 
        SD_Estimate = sd(estimate, na.rm = TRUE)
    ) %>% 
    arrange(desc(abs(Mean_Estimate)))

# ----------------------------------------------------------------------
#  Descriptive Stats 
# ----------------------------------------------------------------------

data_stats <- data_clean %>%
    mutate(lst_num = as.numeric(as.character(lst))) 

#  Numeric Variables Summary 
numeric_vars_list <- data_stats %>% 
    select(where(is.numeric), -c(.row)) %>% 
    names()

numeric_descriptive_stats <- data_stats %>%
    select(all_of(numeric_vars_list)) %>%
    purrr::map(summ_numeric) %>% # Use map() to get a list of named vectors
    tibble::enframe(name = "Variable", value = "Stats") %>%
    tidyr::unnest_wider(Stats) %>% 
    mutate(Type = "Numeric") %>%
    # Đổi tên lst_num lại thành lst trong bảng kết quả
    mutate(Variable = ifelse(Variable == "lst_num", "lst", Variable)) %>% 
    select(Variable, Type, everything())


#  Factor Variables Summary 
factor_vars_list <- data_clean %>% 
    select(where(is.factor), -lst, -c(.row)) %>% 
    names()

factor_stats_list <- purrr::map(factor_vars_list, ~ summ_factor(data_clean, .x))
factor_descriptive_stats <- bind_rows(factor_stats_list)

wb <- createWorkbook() 

# Add Metric Sheets
addWorksheet(wb, "Metrics_Summary")
writeData(wb, "Metrics_Summary", metrics_summary)

addWorksheet(wb, "Metrics_Runs_Raw")
writeData(wb, "Metrics_Runs_Raw", metrics_runs)

# Add Meta-Weights Sheets
addWorksheet(wb, "Meta_Weights_Raw")
writeData(wb, "Meta_Weights_Raw", weights_runs)

addWorksheet(wb, "Meta_Weights_Summary")
writeData(wb, "Meta_Weights_Summary", weights_summary)

# Add T-Test Sheet
addWorksheet(wb, "Ttest")
writeData(wb, "Ttest", t_results)

# Add Descriptive Stats Sheets
addWorksheet(wb, "Numeric_Stats")
writeData(wb, "Numeric_Stats", numeric_descriptive_stats)

addWorksheet(wb, "Factor_Stats")
writeData(wb, "Factor_Stats", factor_descriptive_stats)

# Save the workbook and print final confirmation
saveWorkbook(wb, "results.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results.xlsx\n")
