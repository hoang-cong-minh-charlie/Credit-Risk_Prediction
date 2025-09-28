# ==============================================================================
# STACKING XGBOOST + GAM (GAM SỬ DỤNG TRỌNG SỐ LỚP)
# Đã FIX lỗi ký tự ẩn và tích hợp Class Weights cho GAM.
# ==============================================================================

# ==========================
# 1. INSTALL & LOAD LIBRARIES
# ==========================
install_if_missing <- function(pkg){
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
    }
    # Đảm bảo không có ký tự ẩn ở cuối dòng
    require(pkg, character.only = TRUE) 
}

packages <- c(
    "tidymodels", "janitor", "skimr", "mgcv", "ggplot2", "dplyr", "tibble", 
    "tidyr", "pROC", "openxlsx", "themis", "broom", "glmnet", "Matrix",
    "rsample", "yardstick", "purrr", "recipes" # Thêm recipes nếu chưa có
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
library(recipes) # Cần thiết cho các bước step_
library(RANN) # Nếu bạn sử dụng SMOTE/NN Imputation cho các mô hình khác (dù đã loại bỏ cho GAM)


set.seed(123)

# ==========================
# 2. DATA PREPROCESSING
# ==========================
# Lưu ý: Thay "data/credit_risk_dataset.csv" bằng đường dẫn file thực tế của bạn
# (Giả định file nằm trong thư mục con 'data' hoặc bạn sửa đường dẫn)
data <- read.csv("data/credit_risk_dataset.csv", stringsAsFactors = FALSE) %>% clean_names()

data_clean <- data %>%
    filter(age >= 18, age <= 100, bel >= 0, bel <= age - 18, bch <= age - 18, api > 0, api <= 150000) %>%
    mutate(.row = row_number())

num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- map(data_clean[names(num_vars)], ~
    ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
)

data_clean <- data_clean %>%
    mutate(across(where(is.integer), as.numeric), 
           across(c(pho, lin, lgr, pbd), as.factor), 
           lst = factor(lst, levels = c("0", "1"))) 

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
# 4. LOOP MULTIPLE RUNS (SEQUENTIAL)
# ==========================
n_runs <- 30    
K_FOLDS <- 10  
cat("⏳ Running sequential Stacking for", n_runs, "runs (GAM with Class Weights).\n")

# Định nghĩa công thức GAM 
gam_formula <- lst_num ~ s(age, k=30) + s(api, k=30) + s(bel, k=30) + s(lam, k=30) + 
                        s(lir, k=30) + s(lpi, k=30) + s(bch, k=30) +
                        pho + lin + lgr + pbd + lgr:pbd + api:lir + age:bch 

# TÍNH SCALE POS WEIGHT CHO XGBOOST (Dùng trên toàn bộ tập dữ liệu)
pos_count <- sum(data_clean$lst == "1")
neg_count <- sum(data_clean$lst == "0")
scale_pos_weight_value <- neg_count / pos_count

metrics_runs <- tibble()
weights_runs <- tibble()

for (i in 1:n_runs) {
    cat(sprintf("    -> Running iteration %d/%d...\n", i, n_runs))
    
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
    for (k in 1:K_FOLDS) {
        fold_data <- data_folds$splits[[k]]
        data_train_k <- training(fold_data)
        data_validation_k <- testing(fold_data) 
        
        # LẤY CHỈ MỤC HÀNG OOF
        all_indices <- 1:nrow(data_train_full)
        validation_indices <- all_indices[!(all_indices %in% fold_data$in_id)]
        
        # 1. XGBOOST K-FOLD Training (Sử dụng scale_pos_weight)
        xgb_rec <- recipes::recipe(lst ~ ., data = data_train_k) %>% 
            recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01, other = "other_cat") %>%
            recipes::step_dummy(recipes::all_nominal_predictors()) %>%
            recipes::step_zv(recipes::all_predictors()) %>%
            recipes::step_normalize(recipes::all_numeric_predictors())
        
        xgb_spec <- parsnip::boost_tree(trees = 1500, tree_depth = 6, learn_rate = 0.1) %>% 
            parsnip::set_engine("xgboost", scale_pos_weight = scale_pos_weight_value) %>% 
            parsnip::set_mode("classification")
        
        xgb_wf <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(xgb_rec)
        xgb_model_k <- parsnip::fit(xgb_wf, data = data_train_k)
        
        # 2. GAM K-FOLD Training (SỬ DỤNG TRỌNG SỐ LỚP)
        vars_gam <- all.vars(gam_formula)
        train_gam_k <- data_train_k %>% 
            mutate(lst_num = as.numeric(as.character(lst))) %>% 
            select(all_of(vars_gam), lst) # Thêm lst để tính trọng số
        
        # === TÍNH TOÁN VÀ ÁP DỤNG TRỌNG SỐ LỚP CHO GAM ===
        class_counts <- table(train_gam_k$lst)
        # Trọng số tỷ lệ nghịch với tần suất: N_total / (N_classes * N_class_i)
        weights_vector <- sum(class_counts) / (length(class_counts) * class_counts)

        # Lấy trọng số cho từng hàng
        sample_weights_k <- weights_vector[as.character(train_gam_k$lst)]
        # ==================================================
        
        bam_model_k <- bam(
            gam_formula, data = train_gam_k, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE,
            weights = sample_weights_k # <-- ÁP DỤNG TRỌNG SỐ
        )
        
        # 3. OOF PREDICTIONS
        xgb_oof_prob <- predict(xgb_model_k, new_data = data_validation_k, type = "prob")$.pred_1
        oof_predictions[validation_indices, "xgb_oof"] <- xgb_oof_prob 
        
        # Validation data cho GAM
        val_gam_k <- data_validation_k %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam))
        gam_oof_prob <- as.numeric(predict(bam_model_k, newdata = val_gam_k, type = "response"))
        oof_predictions[validation_indices, "gam_oof"] <- gam_oof_prob
        
    } # END K-FOLD

    # --------------------------------------------------
    # C. BASE MODEL FINAL TRAINING
    # --------------------------------------------------
    cat("      -> Training Final Base Models for Test Set...\n")
    
    # 1. XGBOOST FINAL
    xgb_wf_final <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% 
        workflows::add_recipe(recipes::recipe(lst ~ ., data = data_train_full) %>% 
            recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01, other = "other_cat") %>%
            recipes::step_dummy(recipes::all_nominal_predictors()) %>%
            recipes::step_zv(recipes::all_predictors()) %>%
            recipes::step_normalize(recipes::all_numeric_predictors())
        )
    xgb_model_final <- parsnip::fit(xgb_wf_final, data = data_train_full)
    
    # 2. GAM FINAL (SỬ DỤNG TRỌNG SỐ LỚP)
    vars_gam <- all.vars(gam_formula)
    train_gam_final <- data_train_full %>% 
        mutate(lst_num = as.numeric(as.character(lst))) %>% 
        select(all_of(vars_gam), lst) # Thêm lst để tính trọng số

    # === TÍNH TOÁN VÀ ÁP DỤNG TRỌNG SỐ LỚP CHO GAM FINAL ===
    class_counts_final <- table(train_gam_final$lst)
    weights_vector_final <- sum(class_counts_final) / (length(class_counts_final) * class_counts_final)
    sample_weights_final <- weights_vector_final[as.character(train_gam_final$lst)]
    # =======================================================
    
    bam_model_final <- bam(
        gam_formula, data = train_gam_final, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE,
        weights = sample_weights_final # <-- ÁP DỤNG TRỌNG SỐ
    )
    
    # --------------------------------------------------
    # D. STACKING (META-LEARNER)
    # --------------------------------------------------
    X_train_stack <- oof_predictions %>% select(xgb = xgb_oof, gam = gam_oof) %>% as.matrix()
    y_train_stack <- as.numeric(as.character(oof_predictions$lst))
    
    xgb_prob_vec_test <- predict(xgb_model_final, new_data = data_test, type = "prob")$.pred_1
    
    # Test data cho GAM
    test_gam_final_data <- data_test %>% mutate(lst_num = as.numeric(as.character(lst))) %>% select(all_of(vars_gam))
    gam_prob_vec_test <- as.numeric(predict(bam_model_final, newdata = test_gam_final_data, type = "response"))
    
    X_test_stack <- cbind(xgb = xgb_prob_vec_test, gam = gam_prob_vec_test)
    
    # Chuẩn hóa
    X_train_s <- scale(X_train_stack)
    X_test_s <- scale(X_test_stack, center = attr(X_train_s, "scaled:center"), scale = attr(X_train_s, "scaled:scale"))
    
    cvfit <- cv.glmnet(
        x = X_train_s, y = y_train_stack, family = "binomial", alpha = 0.1,
        penalty.factor = c(1.4, 0.01), 
        standardize = FALSE, nfolds = 10
    )
    
    stacking_prob_vec <- as.numeric(predict(cvfit, newx = X_test_s, s = "lambda.min", type = "response"))
    stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))
    
    # --------------------------------------------------
    # E. Metrics & Weights
    # --------------------------------------------------
    xgb_pred_class_vec <- factor(ifelse(xgb_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1"))
    xgb_eval <- tibble(lst = data_test$lst, .pred_class = xgb_pred_class_vec, .pred_1 = xgb_prob_vec_test)
    xgb_metrics <- calc_metrics(xgb_eval, "XGBoost")
    
    gam_pred_class_vec <- factor(ifelse(gam_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1"))
    gam_eval <- tibble(lst = data_test$lst, .pred_class = gam_pred_class_vec, .pred_1 = gam_prob_vec_test)
    gam_metrics <- calc_metrics(gam_eval, "GAM (bam)")
    
    stk_eval <- tibble(lst = data_test$lst, .pred_class = stacking_pred_class_vec, .pred_1 = stacking_prob_vec)
    stk_metrics <- calc_metrics(stk_eval, "Stacking")
    
    coef_mat <- coef(cvfit, s = "lambda.min") %>% as.matrix()
    coef_df <- as.data.frame(coef_mat) %>% tibble::rownames_to_column("term") %>%
        rename(estimate = 2) %>% mutate(term = ifelse(term == "", "(Intercept)", term), Iter = i)
    
    metrics_runs <- bind_rows(metrics_runs, mutate(xgb_metrics, Iter = i), mutate(gam_metrics, Iter = i), mutate(stk_metrics, Iter = i))
    weights_runs <- bind_rows(weights_runs, coef_df)
}

# ==========================
# 5. SUMMARY & T-TEST
# ==========================
cat("✅ All runs completed. Summarizing results...\n")

# 5.1. Tóm tắt trung bình các Metrics (Accuracy, AUC,...)
metrics_summary <- metrics_runs %>%
    group_by(Model) %>%
    summarise(
        N = n(),
        across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}")
    )

# 5.2. T-Test so sánh cặp mô hình
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

# 5.3. BỔ SUNG: Tóm tắt Trung bình & SD của Meta-Weights ⭐️
weights_summary <- weights_runs %>%
    group_by(term) %>%
    summarise(
        N = n(),
        Mean_Estimate = mean(estimate, na.rm = TRUE),
        SD_Estimate = sd(estimate, na.rm = TRUE)
    ) %>%
    arrange(desc(abs(Mean_Estimate))) # Sắp xếp theo giá trị tuyệt đối trung bình

cat("✅ Meta-Learner Weights Summary calculated.\n")

# 5.4. BỔ SUNG: Thống kê Mô tả Dữ liệu (Descriptive Stats) ⭐️
# Thống kê trên tập dữ liệu đã làm sạch (data_clean)
library(skimr) 

descriptive_stats <- data_clean %>%
    select(-.row, -lst) %>% # Loại bỏ các cột không cần thiết
    skim() %>%
    as_tibble() %>%
    select(skim_type, skim_variable, n_missing, complete_rate, mean, sd, min, max, everything())
    
cat("✅ Descriptive Statistics calculated.\n")

# ==========================
# ==========================
# 6. EXPORT RESULTS TO EXCEL
# ==========================
addWorksheet(wb, "Metrics_Summary")
writeData(wb, "Metrics_Summary", metrics_summary)

addWorksheet(wb, "Metrics_Runs_Raw")
writeData(wb, "Metrics_Runs_Raw", metrics_runs)

addWorksheet(wb, "Meta_Weights_Raw")
writeData(wb, "Meta_Weights_Raw", weights_runs)

addWorksheet(wb, "Ttest")
writeData(wb, "Ttest", t_results)

# BỔ SUNG: Thêm worksheet cho Meta-Weights Summary ⭐️
addWorksheet(wb, "Meta_Weights_Summary")
writeData(wb, "Meta_Weights_Summary", weights_summary)

# BỔ SUNG: Thêm worksheet cho Descriptive Stats ⭐️
addWorksheet(wb, "Descriptive_Stats")
writeData(wb, "Descriptive_Stats", descriptive_stats)

# Cập nhật tên file để phản ánh kết quả tóm tắt cuối cùng
saveWorkbook(wb, "results_runs_t1500_k20_n30_final_summary.xlsx", overwrite = TRUE)
cat("✅ Exported all results to results_runs_t1500_k20_n30_final_summary.xlsx\n")
