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
