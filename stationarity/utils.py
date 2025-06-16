import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from config import CONFIG

def validate_inputs(time_series_data: pd.DataFrame):
    # Ensure required columns exist
    if 'Date' not in time_series_data.columns:
        raise ValueError("Time series data must have a 'Date' column.")

    # Check date formats and ensure consistency
    time_series_data['Date'] = pd.to_datetime(time_series_data['Date'], format=CONFIG['date_format'])

    # Check for missing values
    if time_series_data.isnull().values.any():
        raise ValueError("Time series data contains missing values.")

    return time_series_data

def test_stationarity(values):
    result = adfuller(values, maxlag=CONFIG['adf_max_lag'])
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'Stationary': result[1] < CONFIG['p_value_threshold']
    }

def test_kpss(values):
    result = kpss(values, regression=CONFIG['kpss_regression'])
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Critical Values': result[3],
        'Stationary': result[1] > CONFIG['p_value_threshold']
    }

def test_multiple_series(data: pd.DataFrame):
    results = {}
    for column in data.columns[1:]:  # Skip the 'Date' column
        values = data[column]
        series_results = {
            'ADF': test_stationarity(values)
        }
        if CONFIG['enable_kpss_test']:
            series_results['KPSS'] = test_kpss(values)
        results[column] = series_results
    return results
