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
vars_gam <- c("LST","AGE","API","BEL","LAM","LIR","LPI","BCH","PHO","LIN","LGR","PBD")

train_gam <- data_train %>%
  select(all_of(vars_gam)) %>%
  mutate(
    AGE = as.numeric(AGE),
    API = as.numeric(API),
    BEL = as.numeric(BEL),
    LAM = as.numeric(LAM),
    LIR = as.numeric(LIR),
    LPI = as.numeric(LPI),
    BCH = as.numeric(BCH),
    PHO = factor(PHO),
    LIN = factor(LIN),
    LGR = factor(LGR),
    PBD = factor(PBD),
    LST_num = as.numeric(as.character(LST))
  ) %>% droplevels()

test_gam <- data_test %>%
  select(all_of(vars_gam)) %>%
  mutate(
    AGE = as.numeric(AGE),
    API = as.numeric(API),
    BEL = as.numeric(BEL),
    LAM = as.numeric(LAM),
    LIR = as.numeric(LIR),
    LPI = as.numeric(LPI),
    BCH = as.numeric(BCH),
    PHO = factor(PHO, levels = levels(train_gam$PHO)),
    LIN = factor(LIN, levels = levels(train_gam$LIN)),
    LGR = factor(LGR, levels = levels(train_gam$LGR)),
    PBD = factor(PBD, levels = levels(train_gam$PBD)),
    LST_num = as.numeric(as.character(LST))
  ) %>% droplevels()


bam_model <- bam(
  LST_num ~ s(AGE, k=10) + s(API, k=10) + s(BEL, k=10) + s(LAM, k=10) +
            s(LIR, k=10) + s(LPI, k=10) + s(BCH, k=10) +
            PHO + LIN + LGR + PBD,
  data = train_gam,
  family = binomial(),
  method = "fREML",
  discrete = TRUE
)

gam_prob_vec <- as.numeric(predict(bam_model, newdata = test_gam, type = "response"))
gam_pred_class_vec <- factor(ifelse(gam_prob_vec >= 0.5, "1", "0"), levels = c("0","1"))

gam_eval <- tibble(
  lst = data_test$LST,
  .pred_class = gam_pred_class_vec,
  .pred_1 = gam_prob_vec
)
