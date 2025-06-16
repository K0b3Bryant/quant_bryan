import numpy as np

# LPPL Model Initial Parameters
LPPL_PARAMS = {
    "max_iter": 1000,
    "tol": 1e-6,
}

# Anomaly Detection Parameters
ANOMALY_DETECTION = {
    "rolling_window": 50,  # Window for volatility calculation
    "volatility_threshold": 2.0,  # Std deviation multiplier for anomaly detection
}

# Data Settings
DATA_PATH = "data/price_series.csv"

# Model Output Settings
OUTPUT_PATH = "models/bubble_predictions.csv"
PLOT_RESULTS = True

# Machine Learning Parameters
ML_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

# LSTM Parameters
LSTM_PARAMS = {
    "sequence_length": 50,
    "epochs": 20,
    "batch_size": 16,
    "units": 50,
    "dropout": 0.2
}
