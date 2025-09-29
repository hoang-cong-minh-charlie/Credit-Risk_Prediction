# 5.1. Average Summary of Metrics
metrics_summary <- metrics_runs %>% group_by(Model) %>% summarise(N = n(), across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}"))

# 5.2. T-Test (Stacking vs Base)
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

# 5.3. Summary of Mean and SD of Meta-Weights
weights_summary <- weights_runs %>% group_by(term) %>% summarise(N = n(), Mean_Estimate = mean(estimate, na.rm = TRUE), SD_Estimate = sd(estimate, na.rm = TRUE)) %>% arrange(desc(abs(Mean_Estimate)))
cat("✅ Meta-Learner Weights Summary calculated.\n")

# 5.4.  Descriptive Stats
descriptive_stats <- data_clean %>% select(-.row, -lst) %>% skim() %>% as_tibble() %>% select(skim_type, skim_variable, n_missing, complete_rate, mean, sd, min, max, everything())
cat("✅ Descriptive Statistics calculated.\n")

# print results 
wb <- createWorkbook() 

addWorksheet(wb, "Metrics_Summary")
writeData(wb, "Metrics_Summary", metrics_summary)

addWorksheet(wb, "Metrics_Runs_Raw")
writeData(wb, "Metrics_Runs_Raw", metrics_runs)

addWorksheet(wb, "Meta_Weights_Raw")
writeData(wb, "Meta_Weights_Raw", weights_runs)

addWorksheet(wb, "Meta_Weights_Summary")
writeData(wb, "Meta_Weights_Summary", weights_summary)

addWorksheet(wb, "Ttest")
writeData(wb, "Ttest", t_results)

addWorksheet(wb, "Descriptive_Stats")
writeData(wb, "Descriptive_Stats", descriptive_stats)

saveWorkbook(wb, "results.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results.xlsx\n"
