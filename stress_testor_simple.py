import numpy as np
import pandas as pd

# Sample portfolio
assets = ['AAPL', 'GOOG', 'MSFT', 'BND']
weights = np.array([0.3, 0.3, 0.3, 0.1])  # Portfolio weights (must sum to 1)

# Historical return matrix (daily returns)
np.random.seed(0)
returns = pd.DataFrame(np.random.normal(0, 0.01, (252, 4)), columns=assets)
cov_matrix = returns.cov()

# Hypothetical shock: -5% to tech stocks, +1% to bonds
shock_vector = pd.Series({'AAPL': -0.05, 'GOOG': -0.05, 'MSFT': -0.05, 'BND': 0.01})

# Estimate portfolio loss under shock
portfolio_return = np.dot(weights, shock_vector)
print(f"Portfolio return under stress scenario: {portfolio_return:.2%}")

def monte_carlo_stress(returns, weights, n_sim=10000, shock_func=None):
    simulated = []
    for _ in range(n_sim):
        shock = returns.sample(1).values.flatten()  # Draw a 1-day return randomly
        if shock_func:
            shock = shock_func(shock)  # Modify shock, e.g., amplify
        simulated.append(np.dot(weights, shock))
    return simulated

# Amplify negative returns
shock_func = lambda x: x * 3 if np.mean(x) < 0 else x

stressed_returns = monte_carlo_stress(returns, weights, shock_func=shock_func)

# plot
import matplotlib.pyplot as plt

plt.hist(stressed_returns, bins=50, alpha=0.7)
plt.title("Simulated Portfolio Returns under Amplified Stress")
plt.xlabel("Portfolio Return")
plt.ylabel("Frequency")
plt.axvline(np.percentile(stressed_returns, 5), color='red', linestyle='--', label='5% quantile')
plt.legend()
plt.show()
