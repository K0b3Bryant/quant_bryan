import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import LSTM_PARAMS

def rolling_volatility(prices, window=50):
    return prices.pct_change().rolling(window).std()

def detect_anomalies(prices, threshold=2.0):
    vol = rolling_volatility(prices)
    anomalies = (vol > vol.mean() + threshold * vol.std()).astype(int)
    return anomalies

def create_lstm_model():
    model = Sequential([
        LSTM(LSTM_PARAMS["units"], return_sequences=True, input_shape=(LSTM_PARAMS["sequence_length"], 1)),
        Dropout(LSTM_PARAMS["dropout"]),
        LSTM(LSTM_PARAMS["units"], return_sequences=False),
        Dropout(LSTM_PARAMS["dropout"]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
