"""Module for loading and cleaning data."""
import pandas as pd
import numpy as np
import logging

from config import PROCESSED_DATA_PATH, TARGET_VARIABLE

def load_data() -> pd.DataFrame:
    """
    Loads market data from the processed parquet file. If it doesn't exist,
    it calls the mock data generator.
    """
    if not PROCESSED_DATA_PATH.exists():
        generate_mock_data(PROCESSED_DATA_PATH)
    
    logging.info(f"Loading data from {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataframe index must be a DatetimeIndex.")
        
    return df
