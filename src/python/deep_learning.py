import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

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

# --- Deep Learning Architectures ---

def build_lstm(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer(input_shape, num_heads=2, ff_dim=32):
    inputs = Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dense(input_shape[-1])(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Reinforcement Learning (Q-learning) Skeleton ---
class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)

# --- Example Usage ---
if __name__ == "__main__":
    df = load_labeled_data("your_labeled_timeseries.csv")
    df = add_features(df)
    features = ['returns', 'volatility', 'ma_50', 'ma_200']
    X = df[features].values
    y = df['Target'].values  # Replace with your label column

    # Prepare data for deep learning (reshape for sequence models)
    # Example: Use past 10 days as sequence length
    seq_len = 10
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # LSTM
    lstm_model = build_lstm((seq_len, X_seq.shape[2]))
    lstm_model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=0)
    lstm_preds = lstm_model.predict(X_seq)
    print("LSTM RMSE:", np.sqrt(mean_squared_error(y_seq, lstm_preds)))

    # CNN
    cnn_model = build_cnn((seq_len, X_seq.shape[2]))
    cnn_model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=0)
    cnn_preds = cnn_model.predict(X_seq)
    print("CNN RMSE:", np.sqrt(mean_squared_error(y_seq, cnn_preds)))

    # Transformer
    transformer_model = build_transformer((seq_len, X_seq.shape[2]))
    transformer_model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=0)
    transformer_preds = transformer_model.predict(X_seq)
    print("Transformer RMSE:", np.sqrt(mean_squared_error(y_seq, transformer_preds)))

    # Q-learning agent skeleton usage (for discrete state/action environments)
    # agent = QLearningAgent(n_actions=3, n_states=100)
    # Example: agent.learn(state, action, reward, next_state)