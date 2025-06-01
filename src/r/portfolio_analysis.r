library(PerformanceAnalytics)
library(tidyverse)
library(quadprog)

# --- Data Loading ---
load_portfolio_data <- function(filepath) {
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  df$Date <- as.Date(df$Date)
  df <- df %>% arrange(Date)
  return(df)
}

# --- Sharpe Ratio ---
compute_sharpe <- function(returns, rf = 0) {
  SharpeRatio(returns, Rf = rf, FUN = "StdDev", annualize = TRUE)
}

# --- Sortino Ratio ---
compute_sortino <- function(returns, rf = 0) {
  SortinoRatio(returns, MAR = rf, annualize = TRUE)
}

# --- Drawdown Analysis ---
compute_drawdowns <- function(returns) {
  drawdowns <- Drawdowns(returns)
  max_dd <- maxDrawdown(returns)
  return(list(drawdowns = drawdowns, max_drawdown = max_dd))
}

# --- Cumulative Returns ---
compute_cumulative_returns <- function(returns) {
  cum_ret <- cumprod(1 + returns) - 1
  return(cum_ret)
}

# --- Monte Carlo Portfolio Optimization ---
monte_carlo_optimization <- function(returns, n_portfolios = 10000, rf = 0) {
  n_assets <- ncol(returns)
  results <- matrix(NA, nrow = n_portfolios, ncol = n_assets + 3)
  colnames(results) <- c(paste0("w", 1:n_assets), "Return", "Risk", "Sharpe")
  
  for (i in 1:n_portfolios) {
    weights <- runif(n_assets)
    weights <- weights / sum(weights)
    port_return <- sum(colMeans(returns) * weights) * 252
    port_risk <- sqrt(t(weights) %*% cov(returns) %*% weights) * sqrt(252)
    sharpe <- (port_return - rf) / port_risk
    results[i, ] <- c(weights, port_return, port_risk, sharpe)
  }
  results_df <- as.data.frame(results)
  best <- results_df[which.max(results_df$Sharpe), ]
  return(list(all_portfolios = results_df, optimal = best))
}

# --- Example Usage ---
# df <- load_portfolio_data("portfolio_returns.csv")
# returns <- as.xts(df[,-1], order.by = df$Date)
# print(compute_sharpe(returns))
# print(compute_sortino(returns))
# print(compute_drawdowns(returns))
# print(compute_cumulative_returns(returns))
# opt <- monte_carlo_optimization(returns)
# print(opt$optimal)