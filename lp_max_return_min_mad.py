import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def maximize_return_lp(mean_returns):
    """Maximize expected return using LP."""
    n = len(mean_returns)
    c = -mean_returns  # LP minimizes, so we negate to maximize
    A_eq = [np.ones(n)]
    b_eq = [1]
    bounds = [(0, 1) for _ in range(n)]  # no short selling
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result.x


def minimize_mad_lp(returns):
    """
    Minimize Mean Absolute Deviation (MAD) as a linear risk proxy.
    MAD is linear and convex: a good LP replacement for variance.
    """
    T, n = returns.shape
    mean_returns = returns.mean(axis=0)
    excess = returns - mean_returns

    # Variables: n weights + T absolute deviation terms
    c = np.concatenate([np.zeros(n), np.ones(T)])  # minimize sum of deviations
    A_eq = [np.concatenate([np.ones(n), np.zeros(T)])]
    b_eq = [1]

    # Deviation constraints: abs(excess @ w) ≤ z  =>  excess @ w - z ≤ 0 and -excess @ w - z ≤ 0
    A1 = np.hstack([excess, -np.eye(T)])
    A2 = np.hstack([-excess, -np.eye(T)])
    A_ub = np.vstack([A1, A2])
    b_ub = np.zeros(2 * T)

    bounds = [(0, 1)] * n + [(0, None)] * T  # no shorting; deviations ≥ 0

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result.x[:n]


def efficient_frontier_lp(mean_returns, returns, steps=50):
    """Generate LP-based efficient frontier by blending min risk and max return portfolios."""
    w_max_ret = maximize_return_lp(mean_returns)
    w_min_mad = minimize_mad_lp(returns)
    frontier_weights = []
    frontier_returns = []
    frontier_mad = []

    for alpha in np.linspace(0, 1, steps):
        w = alpha * w_min_mad + (1 - alpha) * w_max_ret
        port_return = np.dot(w, mean_returns)
        mad = np.mean(np.abs((returns @ w) - np.mean(returns @ w)))
        frontier_weights.append(w)
        frontier_returns.append(port_return)
        frontier_mad.append(mad)

    return frontier_weights, frontier_returns, frontier_mad


def optimize_portfolios_lp(mean_returns, returns, plot=False):
    """Return all LP-optimized portfolios and optionally plot."""
    w_max_ret = maximize_return_lp(mean_returns)
    w_min_mad = minimize_mad_lp(returns)
    w_proxy_tangency = 0.5 * w_max_ret + 0.5 * w_min_mad

    result = {
        'max_return': w_max_ret,
        'min_mad': w_min_mad,
        'tangency_proxy': w_proxy_tangency
    }

    if plot:
        _, rets, risks = efficient_frontier_lp(mean_returns, returns)
        plt.figure(figsize=(10, 6))
        plt.plot(risks, rets, label="Efficient Frontier (MAD Proxy)")
        for label, w in result.items():
            r = np.dot(w, mean_returns)
            mad = np.mean(np.abs((returns @ w) - np.mean(returns @ w)))
            plt.scatter(mad, r, label=f"{label.replace('_', ' ').capitalize()}")

        plt.xlabel("Risk Proxy (MAD)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier (Linear Programming)")
        plt.grid(True)
        plt.legend()
        plt.show()

    return result


if __name__ == '__main__':
    np.random.seed(42)

    # Simulate return data (252 trading days)
    assets = ['AAPL', 'MSFT', 'GOOG', 'BND']
    n_assets = len(assets)
    days = 252
    daily_returns = np.random.normal(loc=0.0004, scale=0.01, size=(days, n_assets))
    mean_returns = daily_returns.mean(axis=0) * 252  # annualized

    # Run LP optimization
    portfolios = optimize_portfolios_lp(mean_returns, daily_returns, plot=True)

    print("\nOptimized Portfolios (Weights):")
    for name, weights in portfolios.items():
        print(f"\n{name.replace('_', ' ').capitalize()} Portfolio:")
        for asset, weight in zip(assets, weights):
            print(f"  {asset}: {weight:.2%}")
