import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from config import CONFIG

def validate_inputs(data: pd.DataFrame):
    # Ensure required columns exist
    required_columns = ['Equity_Value', 'Debt_Value', 'Volatility', 'Risk_Free_Rate', 'Feature1', 'Feature2']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain the following columns: {required_columns}")

    # Check for missing values
    if data.isnull().values.any():
        raise ValueError("Data contains missing values.")

    return data

def merton_model(data: pd.DataFrame):
    # Merton model for Probability of Default
    T = CONFIG['time_horizon']
    d1 = (
        np.log(data['Equity_Value'] / data['Debt_Value']) +
        (data['Risk_Free_Rate'] + 0.5 * data['Volatility'] ** 2) * T
    ) / (data['Volatility'] * np.sqrt(T))

    d2 = d1 - data['Volatility'] * np.sqrt(T)

    # PD is the probability that the firm value falls below the debt value
    pd = norm.cdf(-d2)
    return pd

def hull_white_model(data: pd.DataFrame):
    # Hull-White model for Probability of Default
    lambda_0 = CONFIG['hull_white_lambda_0']
    sigma = CONFIG['hull_white_sigma']
    T = CONFIG['time_horizon']

    # Simplified Hull-White default intensity model
    pd = 1 - np.exp(-lambda_0 * T + 0.5 * sigma ** 2 * T ** 2)
    return pd

def ml_model(data: pd.DataFrame):
    # Mock Machine Learning Model for Probability of Default
    # Features used: Feature1, Feature2

    # Training data (for demonstration, replace with actual data)
    train_data = pd.DataFrame({
        'Feature1': [0.4, 0.6, 0.7, 0.3],
        'Feature2': [0.3, 0.2, 0.5, 0.4],
        'Default': [0, 1, 1, 0]  # 1 = Default, 0 = No Default
    })

    # Train a simple classifier
    clf = RandomForestClassifier(random_state=CONFIG['random_seed'])
    clf.fit(train_data[['Feature1', 'Feature2']], train_data['Default'])

    # Predict Probability of Default
    features = data[['Feature1', 'Feature2']]
    pd_ml = clf.predict_proba(features)[:, 1]  # Probability of default
    return pd_ml

def price_risky_bond(face_value, coupon_rate, maturity, risk_free_rate, pd, recovery_rate):
    """
    Calculate the price of a risky bond using default probabilities.
    """
    price = 0
    for t in range(1, maturity + 1):
        coupon = face_value * coupon_rate
        survival_probability = (1 - pd) ** t
        default_probability = 1 - survival_probability

        expected_coupon = coupon * survival_probability
        expected_recovery = face_value * recovery_rate * default_probability

        price += (expected_coupon + expected_recovery) / ((1 + risk_free_rate) ** t)

    # Add the discounted face value at maturity
    price += face_value * survival_probability / ((1 + risk_free_rate) ** maturity)
    return price

def price_cds(notional, pd, recovery_rate, risk_free_rate, maturity):
    """
    Calculate the spread of a credit default swap (CDS).
    """
    expected_loss = 0
    premium_leg = 0

    for t in range(1, maturity + 1):
        survival_probability = (1 - pd) ** t
        default_probability = 1 - survival_probability

        # Calculate expected loss
        expected_loss += notional * (1 - recovery_rate) * default_probability / ((1 + risk_free_rate) ** t)

        # Calculate premium leg (annuity)
        premium_leg += notional * survival_probability / ((1 + risk_free_rate) ** t)

    # CDS spread (annualized, in basis points)
    cds_spread = (expected_loss / premium_leg) * 10000  # Convert to basis points
    return cds_spread
