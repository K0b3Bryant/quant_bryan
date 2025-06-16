import numpy as np
from scipy.optimize import minimize

def lppl(t, A, B, tc, m, C, omega, phi):
    return A + B * (tc - t) ** m * (1 + C * np.cos(omega * np.log(tc - t) + phi))

def lppl_loss(params, t, prices):
    A, B, tc, m, C, omega, phi = params
    predicted = lppl(t, A, B, tc, m, C, omega, phi)
    return np.sum((predicted - prices) ** 2)

def fit_lppl(time_series):
    t = np.arange(len(time_series))
    p0 = [
        np.median(time_series),  # Dynamic baseline
        np.random.uniform(-2, -0.1),  # Randomized B
        max(t) + np.random.uniform(5, 20),  # Varied tc
        np.random.uniform(0.3, 0.7),  # Empirical range for m
        np.random.uniform(-0.5, 0.5),  # Small C values
        np.random.uniform(4, 10),  # Omega varies in log-periodic structures
        np.random.uniform(-np.pi, np.pi)  # Random phase shift
    ]
    bounds = [(None, None), (-np.inf, 0), (max(t), max(t) + 50), (0, 1), (-1, 1), (0, 10), (-np.pi, np.pi)]
    
    result = minimize(lppl_loss, p0, args=(t, time_series), bounds=bounds, options={"maxiter": LPPL_PARAMS["max_iter"]})
    
    if result.success:
        return result.x
    else:
        print("LPPL fitting failed:", result.message)
        return None
