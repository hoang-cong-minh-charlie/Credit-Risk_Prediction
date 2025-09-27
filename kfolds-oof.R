# ==============================================================================
# QUY TRÌNH NGHIÊN CỨU: NON-LEAKAGE STACKING (SEQUENTIAL EXECUTION)
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

# Tải các thư viện cốt lõi để code gọn gàng hơn
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
# 2. DATA PREPROCESSING (Xử lý Ngoại lai & Median Imputation)
# ==========================
data <- read.csv("credit_risk_dataset.csv", stringsAsFactors = FALSE) %>%
  clean_names()

# Remove outliers & tạo chỉ số hàng (.row)
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
data_clean[names(num_vars)] <- map(data_clean[names(num_vars)], ~
  ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
)

# Chuyển đổi kiểu dữ liệu
data_clean <- data_clean %>%
  mutate(
    across(where(is.integer), as.numeric),
    across(c(pho, lin, lgr, pbd), as.factor),
    lst = factor(lst, levels = c("0", "1")) 
  ) 

# Prepare workbook
wb <- createWorkbook()
cat("✅ Data Cleaning (Outliers, Median Imputation) complete\n")

# ==========================
# 3. FUNCTION: CALCULATE METRICS (Đã sửa lỗi trùng lặp tên)
# ==========================
calc_metrics <- function(eval_tbl, model_name) {
  # Sửa lỗi trùng tên: Tạo cột 'truth' rõ ràng và đảm bảo kiểu dữ liệu factor
  eval_tbl_clean <- eval_tbl %>%
    mutate(
      truth = factor(lst, levels = c("0", "1")), 
      .pred_class = factor(.pred_class, levels = c("0", "1"))
    ) 
  
  # Tạo cột xác suất cho lớp 0 cho Brier score
  brier_tbl <- eval_tbl_clean %>%
      mutate(.pred_0 = 1 - .pred_1)

  # Tính các chỉ số
  acc <- accuracy(eval_tbl_clean, truth = truth, estimate = .pred_class)$.estimate
  prec <- precision(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  rec <- recall(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  f1 <- f_meas(eval_tbl_clean, truth = truth, estimate = .pred_class, event_level = "second")$.estimate
  auc <- roc_auc(eval_tbl_clean, truth = truth, .pred_1, event_level = "second")$.estimate
  
  brier <- brier_class(
    brier_tbl, truth = truth, .pred_0, event_level = "first"
  )$.estimate
  
  tibble(Model = model_name, Accuracy = acc, Precision = prec,
                 Recall = rec, F1 = f1, AUC = auc, Brier = brier)
}

# ==========================
# 4. LOOP MULTIPLE RUNS (SỬ DỤNG FOR LOOP TUẦN TỰ)
# ==========================
n_runs <- 2 # 👈 Dùng 2 để TEST, 30 để CHẠY CHÍNH THỨC

cat("⏳ Running in sequential mode for", n_runs, "runs.\n")

# Định nghĩa công thức GAM (Bổ sung 3 tương tác)
gam_formula <- lst_num ~ s(age, k=10) + s(api, k=10) + s(bel, k=10) + s(lam, k=10) +
                       s(lir, k=10) + s(lpi, k=10) + s(bch, k=10) +
                       pho + lin + lgr + pbd +
                       lgr:pbd + api:lir + age:bch 

# Khởi tạo danh sách lưu trữ kết quả
metrics_runs <- tibble()
weights_runs <- tibble()

for (i in 1:n_runs) {
  cat(sprintf("   -> Running iteration %d/%d...\n", i, n_runs))
  
  # --------------------------------------------------
  # A. DATA SPLITTING (NON-LEAKAGE L0/L1 SPLIT)
  # --------------------------------------------------
  # 1. Chia Train (80%) / Test (20%)
  data_split <- initial_split(data_clean, prop = 0.8, strata = lst)
  data_train_full <- training(data_split)
  data_test <- testing(data_split) 
  
  # 2. Chia Train Full thành L0 (Base Model) và L1 (Meta-Learner)
  l1_split <- initial_split(data_train_full, prop = 0.8, strata = lst)
  data_train_L0 <- training(l1_split) 
  data_train_L1 <- testing(l1_split) 

  # Loại bỏ cột index và chuẩn hóa kiểu dữ liệu
  data_train_L0 <- data_train_L0 %>% select(-.row)
  data_train_L1 <- data_train_L1 %>% select(-.row)
  data_test <- data_test %>% select(-.row)

  # ==========================
  # B. BASE MODELS TRAINING (ON L0)
  # ==========================
  
  # 1. XGBOOST 
  xgb_rec <- recipes::recipe(lst ~ ., data = data_train_L0) %>% 
    recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01) %>%
    recipes::step_dummy(recipes::all_nominal_predictors()) %>%
    recipes::step_zv(recipes::all_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors()) %>%
    step_smote(lst)
  
  xgb_spec <- parsnip::boost_tree(trees = 1000, tree_depth = 6, learn_rate = 0.1) %>%
    parsnip::set_engine("xgboost") %>% parsnip::set_mode("classification")
  
  xgb_wf <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(xgb_rec)
  xgb_model_L0 <- parsnip::fit(xgb_wf, data = data_train_L0)
  
  # 2. GAM (bam)
  vars_gam <- all.vars(gam_formula)
  
  # FIX: Tạo lst_num trước, sau đó select()
  train_gam_L0 <- data_train_L0 %>% 
      mutate(lst_num = as.numeric(as.character(lst))) %>% 
      select(all_of(vars_gam)) 
  
  bam_model_L0 <- bam(
    gam_formula, data = train_gam_L0,
    family = binomial(),
    method = "fREML",
    discrete = TRUE,
    select = TRUE 
  )
  
  # ==========================
  # C. OOF PREDICTIONS (META-FEATURES)
  # ==========================
  
  # 1. Lấy Meta-Features từ L1 (NON-LEAKAGE cho huấn luyện Stacking)
  xgb_oof_vec_train <- predict(xgb_model_L0, new_data = data_train_L1, type = "prob")$.pred_1
  
  train_gam_L1 <- data_train_L1 %>% 
      mutate(lst_num = as.numeric(as.character(lst))) %>% 
      select(all_of(vars_gam))
      
  gam_oof_vec_train <- as.numeric(predict(bam_model_L0, newdata = train_gam_L1, type = "response"))

  # 2. Lấy Meta-Features từ Test Set (cho đánh giá cuối cùng)
  xgb_prob_vec_test <- predict(xgb_model_L0, new_data = data_test, type = "prob")$.pred_1
  
  test_gam_final <- data_test %>% 
      mutate(lst_num = as.numeric(as.character(lst))) %>% 
      select(all_of(vars_gam))
      
  gam_prob_vec_test <- as.numeric(predict(bam_model_L0, newdata = test_gam_final, type = "response"))
  
  # ==========================
  # D. STACKING (META-LEARNER)
  # ==========================
  
  # Input Huấn luyện (L1) & Input Test (Final Test)
  X_train_stack <- cbind(xgb = xgb_oof_vec_train, gam = gam_oof_vec_train) 
  X_test_stack <- cbind(xgb = xgb_prob_vec_test, gam = gam_prob_vec_test)
  y_train_stack <- as.numeric(as.character(data_train_L1$lst)) 
  
  # Chuẩn hóa (sử dụng thống kê từ L1)
  X_train_s <- scale(X_train_stack)
  X_test_s <- scale(X_test_stack, center = attr(X_train_s, "scaled:center"), scale = attr(X_train_s, "scaled:scale"))
  
  cvfit <- cv.glmnet(
    x = X_train_s, y = y_train_stack, family = "binomial", alpha = 0, 
    penalty.factor = c(1, 0.2), standardize = FALSE, nfolds = 5
  )
  
  stacking_prob_vec <- as.numeric(predict(cvfit, newx = X_test_s, s = "lambda.min", type = "response"))
  stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))
  
  # ==========================
  # E. Metrics & Weights Output
  # ==========================
  
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

# Thống kê mô tả (Trung bình & Độ lệch chuẩn)
metrics_summary <- metrics_runs %>%
  group_by(Model) %>%
  summarise(
    N = n(),
    across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}")
  )

# T-Test (so sánh Stacking với Base Models)
metric_names <- c("Accuracy","Precision","Recall","F1","AUC","Brier")
t_results <- list()

for (m in metric_names) {
  for (pair in list(c("XGBoost","Stacking"), c("GAM (bam)","Stacking"))) {
    vals1 <- metrics_runs %>% filter(Model == pair[1]) %>% pull(!!sym(m)) 
    vals2 <- metrics_runs %>% filter(Model == pair[2]) %>% pull(!!sym(m))
    test <- t.test(vals1, vals2, paired = TRUE)
    t_results[[length(t_results)+1]] <- tibble(
      Metric = m, Comparison = paste(pair[1], "vs", pair[2]), p_value = test$p.value
    )
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

saveWorkbook(wb, "results_runs_sequential_final.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results_runs_sequential_final.xlsx\n")
