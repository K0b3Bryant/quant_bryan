"""Module for creating model features."""
import pandas as pd
import numpy as np
import logging

from config import (
    TARGET_VARIABLE, SPIKE_THRESHOLD, PREDICTION_HORIZON_MINS,
    LAG_PERIODS_MINS, ROLLING_WINDOW_SIZES_MINS, FEATURE_BASE_COLS
)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-series features and the target variable from the raw data.
    """
    logging.info("Starting feature engineering...")
    df_feat = df.copy()

    # Create Target
    horizon_intervals = int(PREDICTION_HORIZON_MINS / 15)
    df_feat['target_spike'] = (df_feat[TARGET_VARIABLE].shift(-horizon_intervals) > SPIKE_THRESHOLD).astype(int)

    # Time-Based Features (with cyclical encoding)
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['dayofweek_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7)
    df_feat['dayofweek_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7)

    # Lag Features (15-min)
    lag_intervals = [int(p / 15) for p in LAG_PERIODS_MINS]
    for col in FEATURE_BASE_COLS:
        for lag in lag_intervals:
            df_feat[f'{col}_lag_{lag*15}m'] = df_feat[col].shift(lag)

    # Rolling Window Statistics (window size: 15-min)
    window_intervals = [int(p / 15) for p in ROLLING_WINDOW_SIZES_MINS]
    for col in FEATURE_BASE_COLS:
        for window in window_intervals:
            df_feat[f'{col}_roll_mean_{window*15}m'] = df_feat[col].rolling(window=window).mean()
            df_feat[f'{col}_roll_std_{window*15}m'] = df_feat[col].rolling(window=window).std()

    # Drop rows with NaNs created by lags/rolling windows and the target shift
    df_feat = df_feat.dropna()
    
    logging.info(f"Feature engineering complete. Shape of feature DataFrame: {df_feat.shape}")
    return df_feat
