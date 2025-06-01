import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Data Loading ---
def load_labeled_data(filepath):
    """
    Loads labeled financial data for supervised learning.
    """
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return df

# --- Feature Engineering ---
def add_features(df):
    """
    Adds volatility indicators, macroeconomic, and sentiment features.
    """
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=21).std()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    df['ma_200'] = df['Close'].rolling(window=200).mean()
    # Placeholder for macro/sentiment features
    # df['macro'] = ...
    # df['sentiment'] = ...
    df = df.dropna()
    return df

# --- Model Training ---
def train_random_forest(X, y, classification=False):
    model = RandomForestClassifier() if classification else RandomForestRegressor()
    model.fit(X, y)
    return model

def train_xgboost(X, y, classification=False):
    model = XGBClassifier() if classification else XGBRegressor()
    model.fit(X, y)
    return model

def train_deep_learning(X, y, classification=False):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid' if classification else 'linear'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if classification else 'mse',
                  metrics=['accuracy'] if classification else ['mse'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

# --- Prediction/Deployment ---
def predict_risk(model, X, classification=False, deep_learning=False):
    if deep_learning:
        preds = model.predict(X)
        return (preds > 0.5).astype(int) if classification else preds
    else:
        return model.predict(X)

# --- Example Usage ---
if __name__ == "__main__":
    df = load_labeled_data("your_labeled_timeseries.csv")
    df = add_features(df)
    features = ['returns', 'volatility', 'ma_50', 'ma_200']
    X = df[features]
    y = df['Target']  # Replace with your label column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train, classification=False)
    rf_preds = predict_risk(rf_model, X_test)
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, classification=False)
    xgb_preds = predict_risk(xgb_model, X_test)
    print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))

    # Deep Learning
    dl_model = train_deep_learning(X_train, y_train, classification=False)
    dl_preds = predict_risk(dl_model, X_test, deep_learning=True)
    print("Deep Learning RMSE:", np.sqrt(mean_squared_error(y_test, dl_preds)))