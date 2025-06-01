library(tidyverse)
library(zoo)

# --- Data Loading ---
load_data <- function(filepath) {
  df <- read.csv(filepath, stringsAsFactors = FALSE)
  df$Date <- as.Date(df$Date)
  df <- df %>% arrange(Date)
  return(df)
}

# --- Missing Data Interpolation ---
interpolate_missing <- function(df, cols) {
  for (col in cols) {
    df[[col]] <- na.approx(df[[col]], na.rm = FALSE)
  }
  return(df)
}

# --- Seasonality Adjustment ---
seasonality_adjust <- function(df, col, freq = 252) {
  # Remove seasonality using moving average
  ma <- zoo::rollmean(df[[col]], k = freq, fill = NA, align = "center")
  df[[paste0(col, "_deseason")]] <- df[[col]] - ma
  return(df)
}

# --- Log Normalization ---
log_normalize <- function(df, cols) {
  for (col in cols) {
    df[[paste0(col, "_log")]] <- log(df[[col]] + 1e-8)
  }
  return(df)
}

# --- Standardization ---
standardize <- function(df, cols) {
  for (col in cols) {
    mu <- mean(df[[col]], na.rm = TRUE)
    sigma <- sd(df[[col]], na.rm = TRUE)
    df[[paste0(col, "_std")]] <- (df[[col]] - mu) / sigma
  }
  return(df)
}

# --- Example Usage ---
# df <- load_data("your_timeseries.csv")
# df <- interpolate_missing(df, c("Close", "Volume"))
# df <- seasonality_adjust(df, "Close")
# df <- log_normalize(df, c("Close", "Volume"))
# df <- standardize(df, c("Close", "Volume"))