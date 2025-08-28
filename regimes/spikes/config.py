"""Central configuration file for the project."""

from pathlib import Path

# --- Directories ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DATA_PATH = DATA_DIR / "processed_market_data.parquet"

# --- Market & Model Parameters ---
TARGET_VARIABLE = 'LMP_HG_NORTH'
SPIKE_THRESHOLD = 200.00  # LMP in $/MWh
PREDICTION_HORIZON_MINS = 60 # Predict spikes 60 minutes ahead
TEST_SET_SIZE_PERCENT = 0.2

# Features to use for training
FEATURE_BASE_COLS = [
    'SYSTEM_DEMAND',
    'WIND_GEN',
    'SOLAR_GEN',
    'NET_LOAD',
    'LMP_HG_NORTH'
]

# Feature Engineering Parameters
LAG_PERIODS_MINS = [15, 30, 60, 120, 1440]
ROLLING_WINDOW_SIZES_MINS = [30, 60, 180]

# --- Model Configuration ---
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}
