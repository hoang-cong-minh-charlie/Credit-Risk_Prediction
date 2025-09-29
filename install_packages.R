# ==========================
# INSTALL & LOAD LIBRARIES 
# ==========================
install_if_missing <- function(pkg){
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
    }
    require(pkg, character.only = TRUE) 
}

# List of required packages
packages_minimal <- c(
    "dplyr",      
    "rsample",    
    "yardstick",  
    "glmnet",     
    "mgcv",       
    "janitor",    
    "purrr",     
    "recipes",    
    "parsnip",   
    "workflows",  
    "openxlsx",   
    "skimr",      
    "RANN"        
)

# checking, installing and loading of packages
cat("⏳ Checking, installing, and loading necessary packages...\n")
invisible(lapply(packages_minimal, install_if_missing))
cat("✅ All essential packages loaded successfully.\n")



set.seed(123)


