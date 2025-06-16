import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_alpha(returns, betas):
    """
    Calculate alpha from a series of returns and a matrix of betas (factor exposures).
    
    Parameters:
    - returns: array-like, shape (T,) or (T, 1)
    - betas: array-like, shape (T, K)
    
    Returns:
    - alpha: float
    - model: fitted LinearRegression model (optional for inspection)
    """
    # Ensure inputs are numpy arrays
    y = np.asarray(returns).reshape(-1, 1)
    X = np.asarray(betas)
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_[0]
    return alpha, model

# Example usage
returns = np.array([0.01, 0.02, 0.015, 0.005])
betas = np.array([[0.9, 1.1],
                  [1.0, 1.2],
                  [0.95, 1.05],
                  [1.05, 0.95]])

alpha, model = calculate_alpha(returns, betas)
print(f"Alpha: {alpha:.6f}")
