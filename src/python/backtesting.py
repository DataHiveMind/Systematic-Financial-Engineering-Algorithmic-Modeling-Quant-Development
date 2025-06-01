import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Data Loading ---
def load_data(filepath):
    """
    Loads financial time-series data from a CSV file.
    """
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return df

# --- VaR Models ---
def var_historical(returns, alpha=0.05):
    """
    Historical Simulation VaR.
    """
    return np.percentile(returns, 100 * alpha)

def var_variance_covariance(returns, alpha=0.05):
    """
    Variance-Covariance VaR (parametric, normality assumed).
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    return mu + sigma * norm.ppf(alpha)

def var_monte_carlo(returns, alpha=0.05, n_sim=10000):
    """
    Monte Carlo VaR using normal distribution.
    """
    mu = np.mean(returns)
    sigma = np.std(returns)
    simulated = np.random.normal(mu, sigma, n_sim)
    return np.percentile(simulated, 100 * alpha)

# --- Backtesting (Kupiec Test) ---
def kupiec_test(returns, var_series, alpha=0.05):
    """
    Kupiec Proportion of Failures (POF) Test.
    """
    exceptions = returns < var_series
    n = len(returns)
    x = exceptions.sum()
    p_hat = x / n
    LR_pof = -2 * (np.log(((1 - alpha) ** (n - x) * alpha ** x)) -
                   np.log(((1 - p_hat) ** (n - x) * p_hat ** x)))
    return LR_pof, x

# --- Traffic Light Test (Basel) ---
def traffic_light_test(num_exceptions, n_obs, alpha=0.01):
    """
    Basel Traffic Light Test for VaR exceptions.
    """
    green = norm.ppf(1 - alpha)
    yellow = norm.ppf(1 - alpha / 2)
    if num_exceptions <= green:
        return "Green"
    elif num_exceptions <= yellow:
        return "Yellow"
    else:
        return "Red"

# --- Risk Metrics ---
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return np.mean(excess) / np.std(excess)

def sortino_ratio(returns, risk_free_rate=0.0):
    downside = returns[returns < risk_free_rate]
    downside_std = np.std(downside) if len(downside) > 0 else 1
    return (np.mean(returns) - risk_free_rate) / downside_std

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# --- Example Usage ---
if __name__ == "__main__":
    # Example: Load data and run VaR/backtesting
    df = load_data("your_timeseries.csv")
    returns = df['Close'].pct_change().dropna()

    var_hist = var_historical(returns)
    var_vc = var_variance_covariance(returns)
    var_mc = var_monte_carlo(returns)

    # Backtesting
    var_series = returns.rolling(window=250).apply(var_historical)
    LR_pof, exceptions = kupiec_test(returns[249:], var_series[249:])
    traffic = traffic_light_test(exceptions, len(returns[249:]))

    # Risk metrics
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    drawdown = max_drawdown(returns)

    print(f"VaR (Hist): {var_hist:.4f}, VaR (VC): {var_vc:.4f}, VaR (MC): {var_mc:.4f}")
    print(f"Kupiec LR: {LR_pof:.2f}, Exceptions: {exceptions}, Traffic Light: {traffic}")
    print(f"Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, Max Drawdown: {drawdown:.2%}")