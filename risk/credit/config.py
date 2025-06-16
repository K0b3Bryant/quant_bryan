CONFIG = {
    'time_horizon': 1,  # Time horizon in years
    'hull_white_lambda_0': 0.02,  # Initial intensity for Hull-White model
    'hull_white_sigma': 0.1,  # Volatility of intensity in Hull-White model
    'random_seed': 42,  # Random seed for reproducibility in ML model
    'default_probability_model': 'Merton'  # Default probability model: 'Merton', 'Hull-White', 'ML'
}
