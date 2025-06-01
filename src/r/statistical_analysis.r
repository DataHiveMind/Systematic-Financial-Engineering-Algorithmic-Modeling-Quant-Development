library(quantmod)
library(tidyverse)
library(moments)
library(psych)

# --- Data Loading ---
load_data <- function(filepath) {
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  df$Date <- as.Date(df$Date)
  df <- df %>% arrange(Date)
  return(df)
}

# --- Return Calculations ---
calc_returns <- function(df, price_col = "Close") {
  df$log_return <- c(NA, diff(log(df[[price_col]])))
  df$simple_return <- c(NA, diff(df[[price_col]]) / head(df[[price_col]], -1))
  return(df)
}

# --- Volatility & Momentum ---
add_volatility_momentum <- function(df, price_col = "Close", window = 21) {
  df$volatility <- zoo::rollapply(df$log_return, window, sd, fill = NA, align = "right")
  df$momentum <- df[[price_col]] / dplyr::lag(df[[price_col]], window) - 1
  return(df)
}

# --- Descriptive Statistics ---
descriptive_stats <- function(df, col) {
  stats <- data.frame(
    mean = mean(df[[col]], na.rm = TRUE),
    sd = sd(df[[col]], na.rm = TRUE),
    skewness = skewness(df[[col]], na.rm = TRUE),
    kurtosis = kurtosis(df[[col]], na.rm = TRUE)
  )
  return(stats)
}

# --- Correlation Matrix ---
correlation_matrix <- function(df, cols) {
  cor_mat <- cor(df[cols], use = "complete.obs")
  return(cor_mat)
}

# --- Regression Analysis ---
regression_model <- function(df, y_col, x_cols) {
  formula <- as.formula(
    paste(y_col, "~", paste(x_cols, collapse = " + "))
  )
  model <- lm(formula, data = df)
  return(summary(model))
}

# --- Factor Analysis ---
factor_analysis <- function(df, cols, n_factors = 2) {
  fa_result <- fa(df[cols], nfactors = n_factors, rotate = "varimax", fm = "ml")
  return(fa_result)
}

# --- Example Usage ---
# df <- load_data("your_timeseries.csv")
# df <- calc_returns(df)
# df <- add_volatility_momentum(df)
# print(descriptive_stats(df, "log_return"))
# print(correlation_matrix(df, c("log_return", "volatility", "momentum")))
# print(regression_model(df, "log_return", c("macro_var1", "macro_var2")))
# print(factor_analysis(df, c("log_return", "volatility", "momentum")))