library(tidyverse)
library(psych)
library(PerformanceAnalytics)

# --- Data Loading ---
load_factor_data <- function(filepath) {
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  df$Date <- as.Date(df$Date)
  df <- df %>% arrange(Date)
  return(df)
}

# --- CAPM Beta Estimation ---
estimate_capm_beta <- function(asset_returns, market_returns) {
  model <- lm(asset_returns ~ market_returns)
  beta <- coef(model)[2]
  return(list(beta = beta, summary = summary(model)))
}

# --- Fama-French 3-Factor Model ---
estimate_ff3 <- function(asset_returns, market, smb, hml) {
  model <- lm(asset_returns ~ market + smb + hml)
  return(summary(model))
}

# --- Carhart 4-Factor Model ---
estimate_carhart <- function(asset_returns, market, smb, hml, momentum) {
  model <- lm(asset_returns ~ market + smb + hml + momentum)
  return(summary(model))
}

# --- Principal Component Analysis (PCA) ---
run_pca <- function(df, cols, n_comp = 2) {
  pca <- prcomp(df[cols], scale. = TRUE)
  return(list(summary = summary(pca), loadings = pca$rotation[, 1:n_comp]))
}

# --- Example Usage ---
# factors <- load_factor_data("factors.csv")
# asset <- load_factor_data("asset_returns.csv")
# merged <- merge(asset, factors, by = "Date")
# capm <- estimate_capm_beta(merged$AssetReturn, merged$Market)
# ff3 <- estimate_ff3(merged$AssetReturn, merged$Market, merged$SMB, merged$HML)
# carhart <- estimate_carhart(merged$AssetReturn, merged$Market, merged$SMB, merged$HML, merged$Momentum)
# pca_result <- run_pca(merged, c("Market", "SMB", "HML", "Momentum"), n_comp = 2)