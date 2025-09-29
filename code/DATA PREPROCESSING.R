# Read data and clean column names
data <- read.csv("data/credit_risk_dataset.csv", stringsAsFactors = FALSE) %>% clean_names()

# Filter out outliers and invalid values
data_clean <- data %>%
    filter(age >= 18, age <= 100, bel >= 0, bel <= age - 18, bch <= age - 18, api > 0, api <= 150000) %>%
    mutate(.row = row_number())

# Replace with Median for the variable
num_vars <- data_clean %>% select(where(is.numeric))
data_clean[names(num_vars)] <- map(data_clean[names(num_vars)], ~
    ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)
)

# Standardize data types
data_clean <- data_clean %>%
    mutate(across(where(is.integer), as.numeric), 
           across(c(pho, lin, lgr, pbd), as.factor), 
           lst = factor(lst, levels = c("0", "1")))
