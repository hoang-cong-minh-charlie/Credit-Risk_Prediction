# ==========================
# 1. INSTALL & LOAD LIBRARIES
# ==========================
install.packages(c("tidymodels", "janitor", "skimr", "mgcv", "ggplot2", "dplyr", 
                   "tibble", "tidyr", "pROC", "openxlsx"))

library(tidymodels)
library(janitor)
library(skimr)
library(mgcv)
library(ggplot2)
library(dplyr)
library(tibble)
library(tidyr)
library(pROC)
library(openxlsx)

set.seed(123)

# ==========================
# 2. READING AND PREPROCESSING DATA
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

# Replace NA values with median
num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- lapply(data_clean[names(num_vars)], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# ==========================
# 3. EXPORT CLEANED DATA TO EXCEL
# ==========================
output_file <- "data_clean.xlsx"
wb <- createWorkbook()
addWorksheet(wb, "DataClean")
writeData(wb, sheet = "DataClean", data_clean)
saveWorkbook(wb, output_file, overwrite = TRUE)
cat("✅ Exported cleaned data to Excel file:", output_file, "\n")

# ==========================
# 4. TRAIN/TEST SPLIT
# ==========================
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
# 5. XGBOOST MODEL
# ==========================
xgb_rec <- recipe(lst ~ ., data = data_train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

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

# Predict
xgb_pred_class_vec <- predict(xgb_final, new_data = data_test, type = "class")$.pred_class
xgb_prob_vec <- predict(xgb_final, new_data = data_test, type = "prob")$.pred_1

xgb_eval <- tibble(
  lst = data_test$lst,
  .pred_class = xgb_pred_class_vec,
  .pred_1 = xgb_prob_vec
)

# ==========================
# 6. GAM MODEL (bam)
# ==========================
vars_gam <- c("lst", "api", "lam", "lgr", "lin")

train_gam <- data_train %>%
  select(all_of(vars_gam)) %>%
  mutate(api = as.numeric(api),
         lam = as.numeric(lam),
         lgr = factor(lgr),
         lin = factor(lin)) %>%
  droplevels()

test_gam <- data_test %>%
  select(all_of(vars_gam)) %>%
  mutate(api = as.numeric(api),
         lam = as.numeric(lam),
         lgr = factor(lgr, levels = levels(train_gam$lgr)),
         lin = factor(lin, levels = levels(train_gam$lin))) %>%
  droplevels()

# Impute NA with median
train_gam <- train_gam %>%
  mutate(api = ifelse(is.na(api), median(api, na.rm = TRUE), api),
         lam = ifelse(is.na(lam), median(lam, na.rm = TRUE), lam))

test_gam <- test_gam %>%
  mutate(api = ifelse(is.na(api), median(train_gam$api, na.rm = TRUE), api),
         lam = ifelse(is.na(lam), median(train_gam$lam, na.rm = TRUE), lam))

# Convert outcome to numeric
train_gam <- train_gam %>%
  mutate(lst_num = ifelse(as.character(lst) == "1", 1, 0))

test_gam <- test_gam %>%
  mutate(lst_num = ifelse(as.character(lst) == "1", 1, 0))

# Fit GAM
bam_model <- bam(
  lst_num ~ s(api, k = 10) + s(lam, k = 10) + lgr + lin,
  data = train_gam, family = binomial(), method = "fREML", discrete = TRUE
)

# Predict
gam_prob_vec <- as.numeric(predict(bam_model, newdata = test_gam, type = "response"))
gam_pred_class_vec <- factor(ifelse(gam_prob_vec >= 0.5, "1", "0"), levels = c("0", "1"))

gam_eval <- tibble(
  lst = data_test$lst,
  .pred_class = gam_pred_class_vec,
  .pred_1 = gam_prob_vec
)

# ==========================
# 7. STACKING (META LOGISTIC)
# ==========================
xgb_train_prob_vec <- predict(xgb_final, new_data = data_train, type = "prob")$.pred_1
gam_train_prob_vec <- as.numeric(predict(bam_model, newdata = train_gam, type = "response"))

meta_train <- tibble(
  xgb = xgb_train_prob_vec,
  gam = gam_train_prob_vec,
  lst = ifelse(as.character(data_train$lst) == "1", 1, 0)
)

meta_model <- glm(lst ~ xgb + gam, data = meta_train, family = binomial())

meta_test <- tibble(
  xgb = xgb_prob_vec,
  gam = gam_prob_vec
)

stacking_prob_vec <- predict(meta_model, newdata = meta_test, type = "response")
stacking_pred_class_vec <- factor(ifelse(stacking_prob_vec > 0.5, "1", "0"), levels = c("0", "1"))

stacking_eval <- tibble(
  lst = data_test$lst,
  .pred_class = stacking_pred_class_vec,
  .pred_1 = stacking_prob_vec
)

# ==========================
# 8. METRIC CALCULATION
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

xgb_metrics <- calc_metrics(xgb_eval, "XGBoost")
gam_metrics <- calc_metrics(gam_eval, "GAM (bam)")
stk_metrics <- calc_metrics(stacking_eval, "Stacking")

results <- bind_rows(xgb_metrics, gam_metrics, stk_metrics)
print(results)

# ==========================
# 9. DESCRIPTIVE STATISTICS
# ==========================
all_vars <- names(data_clean)

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

summ_categorical <- function(x) {
  c(count = sum(!is.na(x)),
    mean = NA, sd = NA, min = NA, q1 = NA,
    median = NA, q3 = NA, max = NA)
}

summary_list <- lapply(all_vars, function(var) {
  x <- data_clean[[var]]
  if (is.numeric(x)) {
    summ_numeric(x)
  } else {
    summ_categorical(x)
  }
})

summary_all <- do.call(cbind, summary_list)
colnames(summary_all) <- all_vars
summary_all <- as.data.frame(summary_all)
summary_all <- tibble::rownames_to_column(summary_all, var = "stat")

output_file <- "summary_table_12cols.xlsx"
wb <- createWorkbook()
addWorksheet(wb, "DescriptiveStats")
writeData(wb, "DescriptiveStats", summary_all)
saveWorkbook(wb, output_file, overwrite = TRUE)
cat("✅ Exported descriptive statistics table with 12 variables = 12 columns:", output_file, "\n")
