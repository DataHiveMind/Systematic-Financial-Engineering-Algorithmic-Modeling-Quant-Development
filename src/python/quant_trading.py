import pandas as pd
import numpy as np

# --- Technical Indicators ---
def add_indicators(df):
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['std_20'] = df['Close'].rolling(window=20).std()
    df['bollinger_upper'] = df['ma_20'] + 2 * df['std_20']
    df['bollinger_lower'] = df['ma_20'] - 2 * df['std_20']
    df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['signal'] = df['macd'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

# --- Strategy Implementations ---
def mean_reversion_strategy(df):
    signals = []
    for i in range(len(df)):
        if df['Close'].iloc[i] < df['bollinger_lower'].iloc[i]:
            signals.append(1)  # Buy
        elif df['Close'].iloc[i] > df['bollinger_upper'].iloc[i]:
            signals.append(-1) # Sell
        else:
            signals.append(0)  # Hold
    df['signal'] = signals
    return df

def trend_following_strategy(df):
    signals = []
    for i in range(len(df)):
        if df['ma_20'].iloc[i] > df['ma_50'].iloc[i]:
            signals.append(1)  # Buy
        elif df['ma_20'].iloc[i] < df['ma_50'].iloc[i]:
            signals.append(-1) # Sell
        else:
            signals.append(0)  # Hold
    df['signal'] = signals
    return df

def stat_arb_strategy(df):
    # Example: MACD crossover
    signals = []
    for i in range(len(df)):
        if df['macd'].iloc[i] > df['signal'].iloc[i]:
            signals.append(1)  # Buy
        elif df['macd'].iloc[i] < df['signal'].iloc[i]:
            signals.append(-1) # Sell
        else:
            signals.append(0)  # Hold
    df['signal'] = signals
    return df

# --- Position Sizing & Risk Management ---
def position_sizing(df, capital=100000, risk_per_trade=0.01):
    df['position_size'] = 0
    for i in range(len(df)):
        if df['signal'].iloc[i] != 0:
            risk_amount = capital * risk_per_trade
            stop_loss = df['Close'].iloc[i] * 0.98  # 2% stop loss
            size = risk_amount / abs(df['Close'].iloc[i] - stop_loss)
            df.at[df.index[i], 'position_size'] = size * df['signal'].iloc[i]
    return df

# --- Backtesting ---
def backtest(df):
    df['returns'] = df['Close'].pct_change().shift(-1) * df['position_size']
    df['strategy_returns'] = df['returns'].fillna(0)
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

# --- Example Usage ---
if __name__ == "__main__":
    df = pd.read_csv("your_market_data.csv", parse_dates=True, index_col=0)
    df = add_indicators(df)
    # Choose a strategy:
    df = mean_reversion_strategy(df)
    # df = trend_following_strategy(df)
    # df = stat_arb_strategy(df)
    df = position_sizing(df)
    df = backtest(df)
    print(df[['Close', 'signal', 'position_size', 'cumulative_returns']].tail())