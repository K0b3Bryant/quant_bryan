import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_stress_test(assets, weights, n_sim=10000, amplify_negative=True, plot=True):
    """
    Performs a historical shock and Monte Carlo stress test on a portfolio.

    Parameters:
    - assets: List of asset tickers
    - weights: NumPy array of portfolio weights (must match asset length)
    - n_sim: Number of Monte Carlo simulations
    - amplify_negative: Whether to apply a shock multiplier to negative return days
    - plot: Whether to show histogram of stressed returns

    Returns:
    - Tuple: (portfolio_return_under_shock, simulated_returns_list)
    """
    assert len(assets) == len(weights), "Weights must match number of assets"

    # Step 1: Generate synthetic returns
    np.random.seed(0)
    returns = pd.DataFrame(np.random.normal(0, 0.01, (252, len(assets))), columns=assets)

    # Step 2: Define shock (example: -5% to tech, +1% to bonds)
    shock_vector = pd.Series({a: -0.05 if a != 'BND' else 0.01 for a in assets})

    # Step 3: Compute portfolio return under direct shock
    portfolio_return = np.dot(weights, shock_vector)
    print(f"Portfolio return under fixed shock: {portfolio_return:.2%}")

    # Step 4: Monte Carlo stress simulation
    def shock_func(x):
        return x * 3 if amplify_negative and np.mean(x) < 0 else x

    def monte_carlo_stress(returns, weights, n_sim, shock_func):
        simulated = []
        for _ in range(n_sim):
            shock = returns.sample(1).values.flatten()
            shock = shock_func(shock)
            simulated.append(np.dot(weights, shock))
        return simulated

    stressed_returns = monte_carlo_stress(returns, weights, n_sim, shock_func)

    # Step 5: Plot
    if plot:
        plt.hist(stressed_returns, bins=50, alpha=0.7)
        plt.title("Simulated Portfolio Returns under Amplified Stress")
        plt.xlabel("Portfolio Return")
        plt.ylabel("Frequency")
        plt.axvline(np.percentile(stressed_returns, 5), color='red', linestyle='--', label='5% quantile')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return portfolio_return, stressed_returns

if __name__ == "__main__":
    assets = ['AAPL', 'GOOG', 'MSFT', 'BND']
    weights = np.array([0.3, 0.3, 0.3, 0.1])
    
    portfolio_return, simulated_returns = monte_carlo_stress_test(
        assets=assets,
        weights=weights,
        n_sim=10000,
        amplify_negative=True,
        plot=True
    )
