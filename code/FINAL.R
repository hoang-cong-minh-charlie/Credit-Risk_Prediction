# ==============================================================================
# LOAD LIBRARIES
# ==============================================================================

install_if_missing <- function(pkg){
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
    }
    # Load the package into the environment
    require(pkg, character.only = TRUE) 
}

# list of required packages
packages_minimal <- c(
    "dplyr", "rsample", "yardstick", "glmnet", "mgcv", "janitor", 
    "purrr", "recipes", "parsnip", "workflows", "openxlsx", "skimr", "RANN"
)

cat("⏳ Checking, installing, and loading necessary packages...\n")
# Check, install if missing, and load all packages
invisible(lapply(packages_minimal, install_if_missing)) 

# Declare libraries
library(dplyr)
library(rsample)
library(yardstick)
library(glmnet)
library(mgcv)
library(openxlsx)
library(janitor)
library(purrr)
library(recipes)
library(RANN)
library(skimr) 
library(parsnip)
library(workflows)

set.seed(123)

# ==============================================================================
# FUNCTION: CALCULATE METRICS
# ==============================================================================

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

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

# Load data and clean column names
data <- read.csv("data/credit_risk_dataset.csv", stringsAsFactors = FALSE) %>% clean_names()

# Filter outliers and invalid values
data_clean <- data %>%
    filter(age >= 18, age <= 100, bel >= 0, bel <= age - 18, bch <= age - 18, api > 0, api <= 150000) %>%
    mutate(.row = row_number()) # Lưu chỉ mục dòng gốc

# NA Imputation: Replace with Median for numeric variables
num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- map(data_clean[names(num_vars)], ~
    ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
)

# Standardize data types
data_clean <- data_clean %>%
    mutate(across(where(is.integer), as.numeric), 
           across(c(pho, lin, lgr, pbd), as.factor), 
           lst = factor(lst, levels = c("0", "1"))) 

# ==============================================================================
# RUN PARAMETERS AND WEIGHT ADJUSTMENTS
# ==============================================================================

n_runs <- 30    
K_FOLDS <- 10  

# Define the GAM formula (used in the loop)
gam_formula <- lst_num ~ s(age, k=20) + s(api, k=20) + s(bel, k=20) + s(lam, k=20) + 
                        s(lir, k=20) + s(lpi, k=20) + s(bch, k=20) +
                        pho + lin + lgr + pbd + lgr:pbd + api:lir + age:bch 

# CALCULATE SCALE POS WEIGHT FOR XGBOOST
pos_count <- sum(data_clean$lst == "1")
neg_count <- sum(data_clean$lst == "0")
scale_pos_weight_value <- neg_count / pos_count

# Initialize result tibbles
metrics_runs <- tibble()
weights_runs <- tibble()

# ==============================================================================
# REPEATED STACKING PROCEDURE (SEQUENTIAL LOOP)
# ==============================================================================

for (i in 1:n_runs) {
    cat(sprintf("    -> Running iteration %d/%d...\n", i, n_runs))
    
    # A. DATA SPLITTING
    data_split <- initial_split(data_clean, prop = 0.8, strata = lst)
    data_train_full <- training(data_split) %>% select(-.row)
    data_test <- testing(data_split) %>% select(-.row)
    data_folds <- vfold_cv(data_train_full, v = K_FOLDS, strata = lst)
    
    oof_predictions <- data_train_full %>% select(lst) %>% mutate(xgb_oof = NA_real_, gam_oof = NA_real_)
    
    # B. K-FOLD BASE MODEL TRAINING VÀ OOF PREDICTION
    for (k in 1:K_FOLDS) {
        fold_data <- data_folds$splits[[k]]
        data_train_k <- training(fold_data)
        data_validation_k <- testing(fold_data) 
        
        all_indices <- 1:nrow(data_train_full)
        validation_indices <- all_indices[!(all_indices %in% fold_data$in_id)]
        
        # 1. XGBOOST K-FOLD Training (Setup & Fit)
        xgb_rec <- recipes::recipe(lst ~ ., data = data_train_k) %>% recipes::step_other(recipes::all_nominal_predictors(), threshold = 0.01, other = "other_cat") %>% recipes::step_dummy(recipes::all_nominal_predictors()) %>% recipes::step_zv(recipes::all_predictors()) %>% recipes::step_normalize(recipes::all_numeric_predictors())
        xgb_spec <- parsnip::boost_tree(trees = 1500, tree_depth = 6, learn_rate = 0.1) %>% parsnip::set_engine("xgboost", scale_pos_weight = scale_pos_weight_value) %>% parsnip::set_mode("classification")
        xgb_wf <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(xgb_rec)
        xgb_model_k <- parsnip::fit(xgb_wf, data = data_train_k)
        
        # 2. GAM K-FOLD Training (Weighted Fit)
        train_gam_k <- data_train_k %>% mutate(lst_num = as.numeric(as.character(lst))) 
        class_counts <- table(train_gam_k$lst)
        weights_vector <- sum(class_counts) / (length(class_counts) * class_counts)
        sample_weights_k <- weights_vector[as.character(train_gam_k$lst)]
        
        bam_model_k <- bam(gam_formula, data = train_gam_k, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE, weights = sample_weights_k)
        
        # 3. OOF PREDICTIONS
        oof_predictions[validation_indices, "xgb_oof"] <- predict(xgb_model_k, new_data = data_validation_k, type = "prob")$.pred_1
        val_gam_k <- data_validation_k %>% mutate(lst_num = as.numeric(as.character(lst)))
        oof_predictions[validation_indices, "gam_oof"] <- as.numeric(predict(bam_model_k, newdata = val_gam_k, type = "response"))
        
    } # END K-FOLD

    # C. BASE MODEL FINAL TRAINING (On the full Training Set)
    xgb_wf_final <- workflows::workflow() %>% workflows::add_model(xgb_spec) %>% workflows::add_recipe(xgb_rec)
    xgb_model_final <- parsnip::fit(xgb_wf_final, data = data_train_full)
    
    train_gam_final <- data_train_full %>% mutate(lst_num = as.numeric(as.character(lst))) 
    class_counts_final <- table(train_gam_final$lst)
    weights_vector_final <- sum(class_counts_final) / (length(class_counts_final) * class_counts_final)
    sample_weights_final <- weights_vector_final[as.character(train_gam_final$lst)]
    bam_model_final <- bam(gam_formula, data = train_gam_final, family = binomial(), method = "fREML", discrete = TRUE, select = TRUE, weights = sample_weights_final)
    
    # D. STACKING (META-LEARNER)
    X_train_stack <- oof_predictions %>% select(xgb = xgb_oof, gam = gam_oof) %>% as.matrix()
    y_train_stack <- as.numeric(as.character(oof_predictions$lst))
    
    xgb_prob_vec_test <- predict(xgb_model_final, new_data = data_test, type = "prob")$.pred_1
    test_gam_final_data <- data_test %>% mutate(lst_num = as.numeric(as.character(lst)))
    gam_prob_vec_test <- as.numeric(predict(bam_model_final, newdata = test_gam_final_data, type = "response"))
    X_test_stack <- cbind(xgb = xgb_prob_vec_test, gam = gam_prob_vec_test)
    
    # Meta-Learner (Standardized Elastic Net)
    X_train_s <- scale(X_train_stack)
    X_test_s <- scale(X_test_stack, center = attr(X_train_s, "scaled:center"), scale = attr(X_train_s, "scaled:scale"))
    cvfit <- cv.glmnet(x = X_train_s, y = y_train_stack, family = "binomial", alpha = 0.1, penalty.factor = c(1.4, 0.01), standardize = FALSE, nfolds = 10)
    
    stacking_prob_vec <- as.numeric(predict(cvfit, newx = X_test_s, s = "lambda.min", type = "response"))
    stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))
    
    # E. Metrics & Weights Summarized in the loop
    xgb_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = factor(ifelse(xgb_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1")), .pred_1 = xgb_prob_vec_test), "XGBoost")
    gam_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = factor(ifelse(gam_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1")), .pred_1 = gam_prob_vec_test), "GAM (bam)")
    stk_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = stacking_pred_class_vec, .pred_1 = stacking_prob_vec), "Stacking")
    
    coef_mat <- coef(cvfit, s = "lambda.min") %>% as.matrix()
    coef_df <- as.data.frame(coef_mat) %>% tibble::rownames_to_column("term") %>% rename(estimate = 2) %>% mutate(term = ifelse(term == "", "(Intercept)", term), Iter = i)
    
    metrics_runs <- bind_rows(metrics_runs, mutate(xgb_metrics, Iter = i), mutate(gam_metrics, Iter = i), mutate(stk_metrics, Iter = i))
    weights_runs <- bind_rows(weights_runs, coef_df)
}

# ==============================================================================
# SUMMARY AND REPORTING
# ==============================================================================

# Average Summary of Metrics
metrics_summary <- metrics_runs %>% group_by(Model) %>% summarise(N = n(), across(Accuracy:Brier, list(Mean = mean, SD = sd), .names = "{.col}_{.fn}"))

#  T-Test (Stacking vs Base)
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

# Summary of Mean and SD of Meta-Weights
weights_summary <- weights_runs %>% group_by(term) %>% summarise(N = n(), Mean_Estimate = mean(estimate, na.rm = TRUE), SD_Estimate = sd(estimate, na.rm = TRUE)) %>% arrange(desc(abs(Mean_Estimate)))
cat("✅ Meta-Learner Weights Summary calculated.\n")

# Descriptive Stats
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
cat("✅ Exported all results to results.xlsx\n")
