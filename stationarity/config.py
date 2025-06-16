CONFIG = {
    'date_format': '%Y-%m-%d',
    'p_value_threshold': 0.05,
    'enable_kpss_test': True,
    'enable_rolling_test': True,
    'rolling_window_size': 3,
    'adf_max_lag': None,  # None lets the function automatically determine the max lag
    'kpss_regression': 'c'  # Options: 'c' (constant), 'ct' (constant + trend)
}
