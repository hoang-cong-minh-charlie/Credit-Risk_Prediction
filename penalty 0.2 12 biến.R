# ==========================
# 1. INSTALL & LOAD LIBRARIES
# ==========================
install_if_missing <- function(pkg){
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

packages <- c(
  "tidymodels",  # framework ML
  "janitor",     # clean names
  "skimr",       # thống kê nhanh
  "mgcv",        # GAM (bam)
  "ggplot2",     # vẽ biểu đồ
  "dplyr",       # xử lý dữ liệu
  "tibble",      # data frame
  "tidyr",       # reshape dữ liệu
  "pROC",        # ROC, AUC
  "openxlsx",    # export Excel
  "themis",      # SMOTE
  "broom",       # tidy model output
  "glmnet",      # penalized regression
  "Matrix"       # sparse matrix handling
)

invisible(lapply(packages, install_if_missing))

set.seed(123)

# ==========================
# 2. READ & CLEAN DATA
# ==========================
data <- read.csv("credit_risk_dataset.csv", stringsAsFactors = FALSE) %>%
  clean_names()

# Remove outliers
data_clean <- data %>%
  filter(
    age >= 18, age <= 100,
    bel >= 0, bel <= age - 18,
    bch <= age - 18,
    api > 0, api <= 150000
  ) %>%
  mutate(.row = row_number())

# Replace NA with median (numeric only)
num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- lapply(data_clean[names(num_vars)], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# Prepare workbook
wb <- createWorkbook()
addWorksheet(wb, "DataClean")
writeData(wb, "DataClean", data_clean)
cat("✅ Cleaned data ready\n")

# ==========================
# 3. FUNCTION: CALCULATE METRICS
# ==========================
calc_metrics <- function(eval_tbl, model_name) {
  acc <- accuracy(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
  prec <- precision(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
  rec <- recall(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
  f1 <- f_meas(eval_tbl, truth = lst, estimate = .pred_class)$.estimate
  auc <- roc_auc(eval_tbl, truth = lst, .pred_1, event_level = "second")$.estimate
  brier <- brier_class(
    tibble(
      .pred_0 = 1 - eval_tbl$.pred_1,
      .pred_1 = eval_tbl$.pred_1,
      lst = eval_tbl$lst
    ),
    truth = lst, .pred_0, event_level = "first"
  )$.estimate
  
  tibble(Model = model_name, Accuracy = acc, Precision = prec,
         Recall = rec, F1 = f1, AUC = auc, Brier = brier)
}

# ==========================
# 4. LOOP MULTIPLE RUNS
# ==========================
n_runs <- 30   # 👉 đổi thành 2 để test nhanh, 30 để chạy chính thức

all_metrics <- list()
meta_weights <- list()

for (i in 1:n_runs) {
  cat("🔄 Run:", i, "\n")
  
  # Train/Test split
  data_split <- initial_split(data_clean, prop = 0.8, strata = lst)
  data_train <- training(data_split)
  data_test  <- testing(data_split)
  
  data_train <- data_train %>%
    mutate(across(where(is.integer), as.numeric),
           lst = as.factor(lst)) %>%
    droplevels()
  
  data_test <- data_test %>%
    mutate(across(where(is.integer), as.numeric),
           lst = factor(lst, levels = levels(data_train$lst))) %>%
    droplevels()
  
  # ==========================
  # XGBOOST (with SMOTE)
  # ==========================
  xgb_rec <- recipe(lst ~ ., data = data_train) %>%
    step_other(all_nominal_predictors(), threshold = 0.01) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_smote(lst)
  
  xgb_spec <- boost_tree(
    trees = 1000,
    tree_depth = 6,
    learn_rate = 0.1,
    loss_reduction = 0,
    sample_size = 0.8
  ) %>%
    set_engine("xgboost") %>%
    set_mode("classification")
  
  xgb_wf <- workflow() %>%
    add_model(xgb_spec) %>%
    add_recipe(xgb_rec)
  
  xgb_final <- fit(xgb_wf, data = data_train)
  
  xgb_pred_class_vec <- predict(xgb_final, new_data = data_test, type = "class")$.pred_class
  xgb_prob_vec <- predict(xgb_final, new_data = data_test, type = "prob")$.pred_1
  
  xgb_eval <- tibble(lst = data_test$lst,
                     .pred_class = xgb_pred_class_vec,
                     .pred_1 = xgb_prob_vec)
  
  # ==========================
  # GAM (bam)
  # ==========================
vars_gam <- c("lst","age","api","bel","lam","lir","lpi","bch","pho","lin","lgr","pbd")

train_gam <- data_train %>%
  select(all_of(vars_gam)) %>%
  mutate(
    age = as.numeric(age),
    api = as.numeric(api),
    bel = as.numeric(bel),
    lam = as.numeric(lam),
    lir = as.numeric(lir),
    lpi = as.numeric(lpi),
    bch = as.numeric(bch),
    pho = factor(pho),
    lin = factor(lin),
    lgr = factor(lgr),
    pbd = factor(pbd),
    lst_num = as.numeric(as.character(lst))
  ) %>% droplevels()

test_gam <- data_test %>%
  select(all_of(vars_gam)) %>%
  mutate(
    age = as.numeric(age),
    api = as.numeric(api),
    bel = as.numeric(bel),
    lam = as.numeric(lam),
    lir = as.numeric(lir),
    lpi = as.numeric(lpi),
    bch = as.numeric(bch),
    pho = factor(pho, levels = levels(train_gam$pho)),
    lin = factor(lin, levels = levels(train_gam$lin)),
    lgr = factor(lgr, levels = levels(train_gam$lgr)),
    pbd = factor(pbd, levels = levels(train_gam$pbd)),
    lst_num = as.numeric(as.character(lst))
  ) %>% droplevels()


bam_model <- bam(
  lst_num ~ s(age, k=10) + s(api, k=10) + s(bel, k=10) + s(lam, k=10) +
            s(lir, k=10) + s(lpi, k=10) + s(bch, k=10) +
            pho + lin + lgr + pbd,
  data = train_gam,
  family = binomial(),
  method = "fREML",
  discrete = TRUE
)

gam_prob_vec <- as.numeric(predict(bam_model, newdata = test_gam, type = "response"))
gam_pred_class_vec <- factor(ifelse(gam_prob_vec >= 0.5, "1", "0"), levels = c("0","1"))

gam_eval <- tibble(
  lst = data_test$lst,
  .pred_class = gam_pred_class_vec,
  .pred_1 = gam_prob_vec
)
 # ==========================
  # STACKING (glmnet, ưu tiên GAM)
  # ==========================
  xgb_train_prob_vec <- predict(xgb_final, new_data = data_train, type = "prob")$.pred_1
  gam_train_prob_vec <- as.numeric(predict(bam_model, newdata = train_gam, type = "response"))
  
  X_train <- cbind(xgb = xgb_train_prob_vec, gam = gam_train_prob_vec)
  X_test  <- cbind(xgb = xgb_prob_vec, gam = gam_prob_vec)
  y_train <- ifelse(as.character(data_train$lst) == "1", 1, 0)
  
  cvfit <- cv.glmnet(
    x = X_train,
    y = y_train,
    family = "binomial",
    alpha = 0,  # ridge
    penalty.factor = c(1, 0.2),  # ✅ XGB = 1, GAM = 0.2
    standardize = TRUE,
    nfolds = 5
  )
  
  stacking_prob_vec <- as.numeric(predict(cvfit, newx = X_test, s = "lambda.min", type = "response"))
  stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))
  
  stacking_eval <- tibble(lst = data_test$lst,
                          .pred_class = stacking_pred_class_vec,
                          .pred_1 = stacking_prob_vec)
  
  # ==========================
  # ✅ FIX: tidy coef glmnet (không lỗi zero-length name)
  # ==========================
  coef_mat <- coef(cvfit, s = "lambda.min") %>% as.matrix()
  coef_df <- as.data.frame(coef_mat) %>%
    tibble::rownames_to_column("term") %>%
    rename(estimate = 2) %>%
    mutate(term = ifelse(term == "", "(Intercept)", term),
           Iter = i)
  
  meta_weights[[i]] <- coef_df
  
  # ==========================
  # Metrics
  # ==========================
  xgb_metrics <- calc_metrics(xgb_eval, "XGBoost")
  gam_metrics <- calc_metrics(gam_eval, "GAM (bam)")
  stk_metrics <- calc_metrics(stacking_eval, "Stacking")
  
  all_metrics[[i]] <- bind_rows(
    mutate(xgb_metrics, Iter = i),
    mutate(gam_metrics, Iter = i),
    mutate(stk_metrics, Iter = i)
  )
}

# Gộp kết quả
metrics_runs <- bind_rows(all_metrics)
weights_runs <- bind_rows(meta_weights)

# ==========================
# 5. T-TEST COMPARISON (6 metrics)
# ==========================
metric_names <- c("Accuracy","Precision","Recall","F1","AUC","Brier")
t_results <- list()

for (m in metric_names) {
  for (pair in list(c("XGBoost","Stacking"), c("GAM (bam)","Stacking"))) {
    vals1 <- metrics_runs %>% filter(Model == pair[1]) %>% pull(!!sym(m))
    vals2 <- metrics_runs %>% filter(Model == pair[2]) %>% pull(!!sym(m))
    test <- t.test(vals1, vals2, paired = TRUE)
    t_results[[length(t_results)+1]] <- tibble(
      Metric = m,
      Comparison = paste(pair[1], "vs", pair[2]),
      p_value = test$p.value
    )
  }
}
t_results <- bind_rows(t_results)

# ==========================
# 6. EXPORT RESULTS TO EXCEL
# ==========================
addWorksheet(wb, "Metrics_runs")
writeData(wb, "Metrics_runs", metrics_runs)

addWorksheet(wb, "Meta_Weights")
writeData(wb, "Meta_Weights", weights_runs)

addWorksheet(wb, "Ttest")
writeData(wb, "Ttest", t_results)

saveWorkbook(wb, "results_runs.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results_runs.xlsx\n")
