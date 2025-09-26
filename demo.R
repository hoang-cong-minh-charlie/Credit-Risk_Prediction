# ==========================
# 1. INSTALL & LOAD LIBRARIES
# ==========================
library(tidymodels)
library(janitor)
library(mgcv)
library(dplyr)
library(tibble)
library(tidyr)
library(pROC)
library(openxlsx)
library(themis)
set.seed(123)

# ==========================
# 2. DATA CLEANING (ONCE)
# ==========================
data <- read.csv("credit_risk_dataset.csv", stringsAsFactors = FALSE) %>%
  clean_names()

data_clean <- data %>%
  filter(
    age >= 18, age <= 100,
    bel >= 0, bel <= age - 18,
    bch <= age - 18,
    api > 0, api <= 150000
  ) %>%
  mutate(.row = row_number())

num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- lapply(data_clean[names(num_vars)], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# ==========================
# 3. FUNCTION TO RUN ONE EXPERIMENT (SPLIT + TRAIN + EVAL)
# ==========================
run_experiment <- function(seed_val = 123) {
  set.seed(seed_val)
  
  # ---- Split ----
  data_split <- initial_split(data_clean, prop = 0.8, strata = lst)
  data_train <- training(data_split)
  data_test  <- testing(data_split)
  
  data_train <- data_train %>% mutate(lst = as.factor(lst)) %>% droplevels()
  data_test  <- data_test %>% mutate(lst = factor(lst, levels = levels(data_train$lst))) %>% droplevels()
  
  # ---- SMOTE (fix seed inside) ----
  smote_rec <- recipe(lst ~ ., data = data_train) %>% step_smote(lst, seed = seed_val)
  data_train <- prep(smote_rec) %>% bake(new_data = NULL)
  
  cat("\n========== RUN", seed_val, "==========\n")
  cat("✅ SMOTE applied. Train distribution:\n")
  print(table(data_train$lst))
  
  # ---- CV folds ----
  folds <- vfold_cv(data_train, v = 10, strata = lst)
  
  base_recipe <- recipe(lst ~ ., data = data_train) %>%
    step_mutate(across(where(is.integer), as.numeric)) %>%
    step_other(all_nominal_predictors(), threshold = 0.01) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_interact(~ api:lam)
  
  # ---- XGBoost ----
  xgb_spec <- boost_tree(
    trees = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = 0,
    sample_size = 0.8
  ) %>%
    set_engine("xgboost") %>%
    set_mode("classification")
  
  xgb_wf <- workflow() %>%
    add_model(xgb_spec) %>%
    add_recipe(base_recipe)
  
  xgb_grid <- grid_latin_hypercube(
    trees(),
    tree_depth(),
    learn_rate(),
    size = 10
  )
  
  xgb_res <- tune_grid(
    xgb_wf,
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(accuracy, roc_auc),
    control = control_grid(save_pred = TRUE)
  )
  
  xgb_best <- select_best(xgb_res, "roc_auc")
  cat("✅ Best hyperparameters for XGBoost:\n")
  print(xgb_best)
  
  xgb_final <- finalize_workflow(xgb_wf, xgb_best) %>%
    fit(data = data_train)
  
  # ---- GAM (tuning k) ----
  vars_gam <- c("lst", "api", "lam", "lgr", "lin")
  
  train_gam <- data_train %>%
    select(all_of(vars_gam)) %>%
    mutate(api = as.numeric(api),
           lam = as.numeric(lam),
           lgr = factor(lgr),
           lin = factor(lin),
           lst_num = ifelse(as.character(lst) == "1", 1, 0)) %>%
    droplevels()
  
  test_gam <- data_test %>%
    select(all_of(vars_gam)) %>%
    mutate(api = as.numeric(api),
           lam = as.numeric(lam),
           lgr = factor(lgr, levels = levels(train_gam$lgr)),
           lin = factor(lin, levels = levels(train_gam$lin))) %>%
    droplevels()
  
  # simple tuning for k values
  k_values <- c(5, 10, 15)
  gam_auc_scores <- c()
  for (k in k_values) {
    gam_cv <- vfold_cv(train_gam, v = 5, strata = lst_num)
    aucs <- map_dbl(gam_cv$splits, function(s) {
      tr <- analysis(s)
      va <- assessment(s)
      m <- bam(lst_num ~ s(api, k = k, select = TRUE) + 
                         s(lam, k = k, select = TRUE) + lgr + lin,
               data = tr, family = binomial(), method = "fREML", discrete = TRUE)
      prob <- predict(m, newdata = va, type = "response")
      roc_obj <- roc(va$lst_num, prob)
      auc(roc_obj)
    })
    gam_auc_scores <- c(gam_auc_scores, mean(aucs))
  }
  best_k <- k_values[which.max(gam_auc_scores)]
  cat("✅ Best k for GAM:", best_k, "\n")
  
  bam_model <- bam(
    lst_num ~ s(api, k = best_k, select = TRUE) +
               s(lam, k = best_k, select = TRUE) +
               lgr + lin,
    data = train_gam, family = binomial(), 
    method = "fREML", discrete = TRUE
  )
  
  cat("✅ Summary of GAM model:\n")
  print(summary(bam_model))
  
  # ---- STACKING ----
  xgb_oof <- collect_predictions(xgb_res) %>%
    select(.row, id, lst = .truth, .pred_1)
  
  gam_oof <- map_dfr(seq_along(folds$splits), function(i) {
    split <- folds$splits[[i]]
    train_fold <- analysis(split)
    valid_fold <- assessment(split)
    
    train_fold_gam <- train_fold %>%
      select(all_of(vars_gam)) %>%
      mutate(api = as.numeric(api),
             lam = as.numeric(lam),
             lgr = factor(lgr),
             lin = factor(lin),
             lst_num = ifelse(as.character(lst) == "1", 1, 0))
    
    gam_fit <- bam(
      lst_num ~ s(api, k = best_k, select = TRUE) +
                 s(lam, k = best_k, select = TRUE) +
                 lgr + lin,
      data = train_fold_gam,
      family = binomial(), method = "fREML", discrete = TRUE
    )
    
    valid_gam <- valid_fold %>%
      select(all_of(vars_gam)) %>%
      mutate(api = as.numeric(api),
             lam = as.numeric(lam),
             lgr = factor(lgr, levels = levels(train_fold_gam$lgr)),
             lin = factor(lin, levels = levels(train_fold_gam$lin)),
             lst = factor(lst))
    
    tibble(
      .row = valid_fold$.row,
      .pred_gam = as.numeric(predict(gam_fit, newdata = valid_gam, type = "response")),
      lst = valid_gam$lst
    )
  })
  
  stack_train <- inner_join(xgb_oof, gam_oof, by = c(".row", "lst")) %>%
    transmute(
      lst = ifelse(as.character(lst) == "1", 1, 0),
      xgb = scale(.pred_1),
      gam = scale(.pred_gam)
    )
  
  meta_model <- glm(lst ~ xgb + gam, data = stack_train, family = binomial())
  cat("✅ Meta-learner logistic regression coefficients:\n")
  print(coef(summary(meta_model)))
  
  # ---- Predictions ----
  xgb_prob <- predict(xgb_final, new_data = data_test, type = "prob")$.pred_1
  gam_prob <- as.numeric(predict(bam_model, newdata = test_gam, type = "response"))
  
  meta_test <- tibble(xgb = scale(xgb_prob), gam = scale(gam_prob))
  stack_prob <- predict(meta_model, newdata = meta_test, type = "response")
  stack_pred <- factor(ifelse(stack_prob > 0.5, "1", "0"), levels = c("0", "1"))
  
  stack_eval <- tibble(lst = data_test$lst, .pred_class = stack_pred, .pred_1 = stack_prob)
  
  # ---- Meta coefficients ----
  coefs <- coef(summary(meta_model))
  coef_xgb <- coefs["xgb", "Estimate"]
  coef_gam <- coefs["gam", "Estimate"]
  
  # ---- Metrics ----
  calc_metrics <- function(eval_tbl, model_name) {
    acc <- accuracy(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
    prec <- precision(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
    rec <- recall(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
    f1 <- f_meas(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
    auc <- roc_auc(eval_tbl, truth = lst, .pred_1, event_level = "second")$.estimate
    brier <- brier_class(
      tibble(.pred_0 = 1 - eval_tbl$.pred_1, .pred_1 = eval_tbl$.pred_1, lst = eval_tbl$lst),
      truth = lst, .pred_0, event_level = "first"
    )$.estimate
    
    tibble(Model = model_name, Accuracy = acc, Precision = prec,
           Recall = rec, F1 = f1, AUC = auc, Brier = brier)
  }
  
  xgb_eval <- tibble(lst = data_test$lst,
                     .pred_class = predict(xgb_final, new_data = data_test, type = "class")$.pred_class,
                     .pred_1 = xgb_prob)
  
  gam_eval <- tibble(lst = data_test$lst,
                     .pred_class = factor(ifelse(gam_prob >= 0.5, "1", "0"), levels = c("0", "1")),
                     .pred_1 = gam_prob)
  
  xgb_metrics <- calc_metrics(xgb_eval, "XGBoost")
  gam_metrics <- calc_metrics(gam_eval, "GAM (bam)")
  stk_metrics <- calc_metrics(stack_eval, "Stacking") %>%
    mutate(Coef_XGB = coef_xgb, Coef_GAM = coef_gam)
  
  res <- bind_rows(xgb_metrics, gam_metrics, stk_metrics) %>%
    mutate(Run = seed_val)
  
  print(res)
  return(res)
}

# ==========================
# 4. REPEATED 30 RUNS
# ==========================
all_results <- map_dfr(1:30, ~ run_experiment(seed_val = .x))

# ==========================
# 5. SUMMARY & STATISTICAL TESTS
# ==========================
# Summary (mean ± sd)
summary_results <- all_results %>%
  group_by(Model) %>%
  summarise(
    across(c(Accuracy, Precision, Recall, F1, AUC, Brier, Coef_XGB, Coef_GAM),
           list(mean = mean, sd = sd), .names = "{.col}_{.fn}")
  )

cat("✅ Summary of 30 runs (mean ± sd):\n")
print(summary_results)

# Friedman test across 3 models
metrics_long <- all_results %>%
  select(Run, Model, Accuracy, Precision, Recall, F1, AUC, Brier) %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1, AUC, Brier),
               names_to = "Metric", values_to = "Value")

cat("✅ Friedman test results (all models, each metric):\n")
friedman_results <- metrics_long %>%
  group_by(Metric) %>%
  do({
    wide <- pivot_wider(., names_from = Model, values_from = Value)
    fr <- friedman.test(as.matrix(select(wide, -Run, -Metric)))
    tibble(statistic = fr$statistic, p.value = fr$p.value)
  })
print(friedman_results)

# ==========================
# 6. SAVE TO EXCEL
# ==========================
wb <- createWorkbook()
addWorksheet(wb, "Results")
writeData(wb, "Results", all_results)
addWorksheet(wb, "Summary")
writeData(wb, "Summary", summary_results)

# save coefficients separately
coef_tbl <- all_results %>%
  filter(Model == "Stacking") %>%
  select(Run, Coef_XGB, Coef_GAM)
addWorksheet(wb, "Coefficients")
writeData(wb, "Coefficients", coef_tbl)

saveWorkbook(wb, "results_30_runs.xlsx", overwrite = TRUE)
cat("✅ Saved 30-run results, summary, and coefficients to results_30_runs.xlsx\n")
