# ==============================================================================
# QUY TRÌNH NGHIÊN CỨU: 5-FOLD STACKING VỚI SMOTE CHUNG
# ĐIỀU CHỈNH: ÁP DỤNG SMOTE CHO TẬP HUẤN LUYỆN TRƯỚC KHI CHẠY CẢ XGBOOST VÀ GAM
# K=20 (GAM), TREE_DEPTH=8 (XGBoost), N_RUNS=10
# ==============================================================================

# ==========================
# 1. INSTALL & LOAD LIBRARIES
# ==========================
install_if_missing <- function(pkg){
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  require(pkg, character.only = TRUE) 
}

packages <- c(
  "tidymodels", "janitor", "skimr", "mgcv", "ggplot2", "dplyr", "tibble", 
  "tidyr", "pROC", "openxlsx", "themis", "broom", "glmnet", "Matrix",
  "rsample", "yardstick", "purrr"
)
invisible(lapply(packages, install_if_missing))

# Tải các thư viện cốt lõi
library(dplyr)
library(rsample)
library(yardstick)
library(glmnet)
library(mgcv)
library(themis) 
library(openxlsx)
library(janitor)
library(purrr)

set.seed(123)

# ==========================
# 2. DATA PREPROCESSING
# ==========================
# Lưu ý: Thay "credit_risk_dataset.csv" bằng đường dẫn file thực tế của bạn
data <- read.csv("credit_risk_dataset.csv", stringsAsFactors = FALSE) %>% clean_names()

data_clean <- data %>%
  filter(age >= 18, age <= 100, bel >= 0, bel <= age - 18, bch <= age - 18, api > 0, api <= 150000) %>%
  mutate(.row = row_number())

num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- map(data_clean[names(num_vars)], ~
  ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
)

data_clean <- data_clean %>%
  mutate(across(where(is.integer), as.numeric), across(c(pho, lin, lgr, pbd), as.factor), lst = factor(lst, levels = c("0", "1"))) 

wb <- createWorkbook()
cat("✅ Data Cleaning complete\n")

# ==========================
# 3. FUNCTION: CALCULATE METRICS
# ==========================
calc_metrics <- function(eval_tbl, model_name) {
  eval_tbl_clean <- eval_tbl %>%
    mutate(truth = factor(lst, levels = c("0", "1")), .pred_class = factor(.pred_class, levels = c("0", "1"))) 
  
  brier_tbl <- eval_tbl_clean %>% mutate(.pred_0 = 1 - .pred_1)

  acc <- accuracy(eval_tbl_clean, truth = truth, estimate = .pred_class)$.estimate
  prec <- precision(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  rec <- recall(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  f1 <- f_meas(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  auc <- roc_auc(eval_tbl_clean, truth = truth, .pred_1, event_level = "second")$.estimate
  
  brier <- brier_class(brier_tbl, truth = truth, .pred_0, event_level = "first")$.estimate
  
  tibble(Model = model_name, Accuracy = acc, Precision = prec,
                 Recall = rec, F1 = f1, AUC = auc, Brier = brier)
}

# ==========================
# 4. LOOP MULTIPLE RUNS (SỬ DỤNG FOR LOOP TUẦN TỰ)
# ==========================
n_runs <- 10 
K_FOLDS <- 5 
cat("⏳ Running sequential 5-Fold Stacking for", n_runs, "runs with unified SMOTE.\n")

# Định nghĩa công thức GAM (K=20)
gam_formula <- lst_num ~ s(age, k=20) + s(api, k=20) + s(bel, k=20) + s(lam, k=20) + 
                       s(lir, k=20) + s(lpi, k=20) + s(bch, k=20) +
                       pho + lin + lgr + pbd + lgr:pbd + api:lir + age:bch 

metrics_runs <- tibble()
weights_runs <- tibble()

for (i in 1:n_runs) {
  cat(sprintf("   -> Running iteration %d/%d...\n", i, n_runs))
  
  # --------------------------------------------------
  # A. DATA SPLITTING (TRAIN 80% / TEST 20%)
  # --------------------------------------------------
  data_split <- initial_split(data_clean, prop = 0.8, strata = lst)
  data_train_full <- training(data_split) %>% select(-.row)
  data_test <- testing(data_split) %>% select(-.row)
  
  # CHIA data_train_full THÀNH K_FOLDS
  data_folds <- vfold_cv(data_train_full, v = K_FOLDS, strata = lst)
  
  # Chuẩn bị OOF Prediction vectors
  oof_predictions <- data_train_full %>% 
    select(lst) %>% 
    mutate(xgb_oof = NA_real_, gam_oof = NA_real_)
  
  # --------------------------------------------------
  # B. K-FOLD BASE MODEL TRAINING VÀ OOF PREDICTION
  # --------------------------------------------------
  cat("      -> Training K Base Models...\n")
  
  # Định nghĩa Recipe cơ bản chỉ xử lý tiền tố (không SMOTE)
  base_rec <- recipes::recipe(lst ~ ., data = data_train_full) %>% 
    recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01) %>%
    recipes::step_dummy(recipes::all_nominal_predictors()) %>%
    recipes::step_zv(recipes::all_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors())
  
  # Định nghĩa Spec XGBoost
  xgb_spec <- parsnip::boost_tree(trees = 1000, tree_depth = 8, learn_rate = 0.1) %>% 
      parsnip::set_engine("xgboost") %>% 
      parsnip::set_mode("classification")
      
  for (k in 1:K_FOLDS) {
    fold_data <- data_folds$splits[[k]]
    data_train_k <- training(fold_data)
    data_validation_k <- testing(fold_data) 
    
    # LẤY CHỈ MỤC HÀNG CỦA TẬP VALIDATION (OOF)
    all_indices <- 1:nrow(data_train_full)
    validation_indices <- all_indices[!(all_indices %in% fold_data$in_id)]
    
    # === THỰC HIỆN SMOTE CHUNG CHO CẢ 2 BASE LEARNER ===
    
    # 1. Chuẩn bị dữ liệu huấn luyện cân bằng (SMOTE)
    smote_wf <- workflows::workflow() %>% workflows::add_recipe(base_rec %>% themis::step_smote(lst, over_ratio = 1))
    
    # Áp dụng SMOTE (và các bước tiền xử lý khác) cho tập huấn luyện fold
    # SỬ DỤNG recipes::bake() để chỉ áp dụng SMOTE & tiền xử lý
    train_processed <- smote_wf %>% 
        workflows::pull_workflow_preprocessor() %>% 
        recipes::prep(training = data_train_k) %>% 
        recipes::juice()
    
    # 2. XGBOOST K-FOLD Training
    # Tái sử dụng recipe cơ bản cho workflow, vì SMOTE đã được thực hiện bên ngoài
    xgb_wf <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(base_rec)
    xgb_model_k <- parsnip::fit(xgb_wf, data = train_processed) 
    
    # 3. GAM K-FOLD Training (Sử dụng dữ liệu đã SMOTE)
    vars_gam <- all.vars(gam_formula)
    train_gam_k <- train_processed %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam), lst_num)
    
    bam_model_k <- bam(
      gam_formula, data = train_gam_k, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE 
    )
    
    # 4. OOF PREDICTIONS
    
    # XGBoost OOF (Chạy qua recipe gốc để tiền xử lý tập validation)
    xgb_oof_prob <- predict(xgb_model_k, new_data = data_validation_k, type = "prob")$.pred_1
    oof_predictions[validation_indices, "xgb_oof"] <- xgb_oof_prob 
    
    # GAM OOF
    val_gam_k <- data_validation_k %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam))
    gam_oof_prob <- as.numeric(predict(bam_model_k, newdata = val_gam_k, type = "response"))
    oof_predictions[validation_indices, "gam_oof"] <- gam_oof_prob
    
  } # HẾT VÒNG LẶP K-FOLD

  # --------------------------------------------------
  # C. BASE MODEL FINAL TRAINING (TRÊN TOÀN BỘ data_train_full CÂN BẰNG)
  # --------------------------------------------------
  cat("      -> Training Final Base Models for Test Set...\n")
  
  # Final Recipe có SMOTE
  final_rec_smote <- base_rec %>% themis::step_smote(lst, over_ratio = 1)
  
  # 1. XGBOOST FINAL
  xgb_wf_final <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(final_rec_smote)
  xgb_model_final <- parsnip::fit(xgb_wf_final, data = data_train_full)
  
  # 2. GAM FINAL (Lấy dữ liệu đã SMOTE để huấn luyện)
  train_final_processed <- xgb_wf_final %>% 
        workflows::pull_workflow_preprocessor() %>% 
        recipes::prep(training = data_train_full) %>% 
        recipes::juice()
        
  vars_gam <- all.vars(gam_formula)
  train_gam_final <- train_final_processed %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam), lst_num)
  
  bam_model_final <- bam(
    gam_formula, data = train_gam_final, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE
  )
  
  # --------------------------------------------------
  # D. STACKING (META-LEARNER)
  # --------------------------------------------------
  
  # Input Huấn luyện (là OOF predictions)
  X_train_stack <- oof_predictions %>% select(xgb = xgb_oof, gam = gam_oof) %>% as.matrix()
  y_train_stack <- as.numeric(as.character(oof_predictions$lst))
  
  # Lấy Meta-Features từ Test Set (sử dụng mô hình FINAL)
  xgb_prob_vec_test <- predict(xgb_model_final, new_data = data_test, type = "prob")$.pred_1
  
  test_gam_final_data <- data_test %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam))
  gam_prob_vec_test <- as.numeric(predict(bam_model_final, newdata = test_gam_final_data, type = "response"))
  
  X_test_stack <- cbind(xgb = xgb_prob_vec_test, gam = gam_prob_vec_test)
  
  # Chuẩn hóa (sử dụng thống kê từ tập OOF)
  X_train_s <- scale(X_train_stack)
  X_test_s <- scale(X_test_stack, center = attr(X_train_s, "scaled:center"), scale = attr(X_train_s, "scaled:scale"))
  
  cvfit <- cv.glmnet(
    x = X_train_s, y = y_train_stack, family = "binomial", alpha = 0.5,
    penalty.factor = c(1, 0.01), # Ưu tiên GAM
    standardize = FALSE, nfolds = 5
  )
  
  stacking_prob_vec <- as.numeric(predict(cvfit, newx = X_test_s, s = "lambda.min", type = "response"))
  stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))
  
  # --------------------------------------------------
  # E. Metrics & Weights Output
  # --------------------------------------------------
  
  # Base XGBoost Metrics
  xgb_pred_class_vec <- factor(ifelse(xgb_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1"))
  xgb_eval <- tibble(lst = data_test$lst, .pred_class = xgb_pred_class_vec, .pred_1 = xgb_prob_vec_test)
  xgb_metrics <- calc_metrics(xgb_eval, "XGBoost")
  
  # Base GAM Metrics
  gam_pred_class_vec <- factor(ifelse(gam_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1"))
  gam_eval <- tibble(lst = data_test$lst, .pred_class = gam_pred_class_vec, .pred_1 = gam_prob_vec_test)
  gam_metrics <- calc_metrics(gam_eval, "GAM (bam)")
  
  # Stacking Metrics
  stk_eval <- tibble(lst = data_test$lst, .pred_class = stacking_pred_class_vec, .pred_1 = stacking_prob_vec)
  stk_metrics <- calc_metrics(stk_eval, "Stacking")
  
  # Weights
  coef_mat <- coef(cvfit, s = "lambda.min") %>% as.matrix()
  coef_df <- as.data.frame(coef_mat) %>% tibble::rownames_to_column("term") %>%
    rename(estimate = 2) %>% mutate(term = ifelse(term == "", "(Intercept)", term), Iter = i)
  
  # Gộp kết quả vào tibble tổng
  metrics_runs <- bind_rows(metrics_runs, mutate(xgb_metrics, Iter = i), mutate(gam_metrics, Iter = i), mutate(stk_metrics, Iter = i))
  weights_runs <- bind_rows(weights_runs, coef_df)
}

# ==========================
# 5. TỔNG HỢP VÀ PHÂN TÍCH KẾT QUẢ
# ==========================
cat("✅ All runs completed. Summarizing results...\n")

metrics_summary <- metrics_runs %>%
  group_by(Model) %>%
  summarise(
    N = n(),
    across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}")
  )

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

# ==========================
# 6. EXPORT RESULTS TO EXCEL
# ==========================
addWorksheet(wb, "Metrics_Summary")
writeData(wb, "Metrics_Summary", metrics_summary)

addWorksheet(wb, "Metrics_Runs_Raw")
writeData(wb, "Metrics_Runs_Raw", metrics_runs)

addWorksheet(wb, "Meta_Weights")
writeData(wb, "Meta_Weights", weights_runs)

addWorksheet(wb, "Ttest")
writeData(wb, "Ttest", t_results)

saveWorkbook(wb, "results_runs_unified_smote_final.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results_runs_unified_smote_final.xlsx\n")
