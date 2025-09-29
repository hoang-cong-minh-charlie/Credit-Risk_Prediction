cat("⏳ Running sequential Stacking for", n_runs, "runs.\n")

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
        
        # 2. GAM K-FOLD Training 
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

    # C. BASE MODEL FINAL TRAINING 
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
    
    # E. Metrics & Weights Tóm tắt trong vòng lặp
    xgb_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = factor(ifelse(xgb_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1")), .pred_1 = xgb_prob_vec_test), "XGBoost")
    gam_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = factor(ifelse(gam_prob_vec_test > 0.5, "1", "0"), levels = c("0", "1")), .pred_1 = gam_prob_vec_test), "GAM (bam)")
    stk_metrics <- calc_metrics(tibble(lst = data_test$lst, .pred_class = stacking_pred_class_vec, .pred_1 = stacking_prob_vec), "Stacking")
    
    coef_mat <- coef(cvfit, s = "lambda.min") %>% as.matrix()
    coef_df <- as.data.frame(coef_mat) %>% tibble::rownames_to_column("term") %>% rename(estimate = 2) %>% mutate(term = ifelse(term == "", "(Intercept)", term), Iter = i)
    
    metrics_runs <- bind_rows(metrics_runs, mutate(xgb_metrics, Iter = i), mutate(gam_metrics, Iter = i), mutate(stk_metrics, Iter = i))
    weights_runs <- bind_rows(weights_runs, coef_df)
}
