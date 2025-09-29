n_runs <- 30    
K_FOLDS <- 10  

# Definition of GAM formula
gam_formula <- lst_num ~ s(age, k=30) + s(api, k=30) + s(bel, k=30) + s(lam, k=30) + 
                        s(lir, k=30) + s(lpi, k=30) + s(bch, k=30) +
                        pho + lin + lgr + pbd + lgr:pbd + api:lir + age:bch 

# CALCULATE SCALE POS WEIGHT FOR XGBOOST (apply to entire dataset)
pos_count <- sum(data_clean$lst == "1")
neg_count <- sum(data_clean$lst == "0")
scale_pos_weight_value <- neg_count / pos_count

# Initialize the result storage frame
metrics_runs <- tibble()
weights_runs <- tibble()
