import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def get_min_variance_portfolio(mean_returns, cov_matrix):
    """Compute the global minimum variance portfolio using QP."""
    n = len(mean_returns)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return np.array(w.value).flatten()


def get_tangency_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    """Compute the maximum Sharpe ratio (tangency) portfolio using QP."""
    n = len(mean_returns)
    excess_returns = mean_returns - risk_free_rate
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)  # risk aversion parameter (we'll set it to 1)

    # Maximize Sharpe ratio <=> Maximize (excess return) / std.dev => QP version:
    # Maximize: mu^T w  subject to: w.T Σ w <= k  and sum(w) = 1
    objective = cp.Maximize(excess_returns @ w)
    constraints = [cp.sum(w) == 1, w >= 0, cp.quad_form(w, cov_matrix) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return np.array(w.value).flatten()


def efficient_frontier(mean_returns, cov_matrix, points=50):
    """Generate the efficient frontier."""
    n = len(mean_returns)
    target_returns = np.linspace(0.01, max(mean_returns) * 1.1, points)
    results = []

    for r_target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [cp.sum(w) == 1, w >= 0, mean_returns @ w == r_target]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if w.value is not None:
            ret = mean_returns @ w.value
            vol = np.sqrt(w.value.T @ cov_matrix @ w.value)
            results.append((vol, ret))
    return zip(*results)


def optimize_portfolios(mean_returns, cov_matrix, risk_free_rate=0.0, plot=False):
    """Return optimized portfolios and optionally plot."""
    w_min_var = get_min_variance_portfolio(mean_returns, cov_matrix)
    w_tangency = get_tangency_portfolio(mean_returns, cov_matrix, risk_free_rate)
    w_max_slope = w_tangency  # equivalent in this QP setup

    if plot:
        vols, rets = efficient_frontier(mean_returns, cov_matrix)
        plt.figure(figsize=(10, 6))
        plt.plot(vols, rets, label='Efficient Frontier')

        def mark_point(w, label):
            r = mean_returns @ w
            v = np.sqrt(w.T @ cov_matrix @ w)
            plt.scatter(v, r, label=label)
            return r, v

        mark_point(w_min_var, 'Min Variance')
        mark_point(w_tangency, 'Tangency')

        plt.xlabel("Volatility (Std Dev)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier (QP)")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        'min_variance': w_min_var,
        'tangency': w_tangency,
        'max_slope': w_max_slope
    }


if __name__ == '__main__':
    np.random.seed(42)

    # Example data
    assets = ['AAPL', 'MSFT', 'GOOG', 'BND']
    mean_returns = np.array([0.12, 0.10, 0.15, 0.04])
    std_devs = np.array([0.2, 0.18, 0.22, 0.06])
    corr_matrix = np.array([
        [1.0, 0.85, 0.80, 0.1],
        [0.85, 1.0, 0.75, 0.1],
        [0.80, 0.75, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0]
    ])
    cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

    portfolios = optimize_portfolios(mean_returns, cov_matrix, risk_free_rate=0.02, plot=True)

    print("\nOptimal Portfolios (weights):")
    for name, weights in portfolios.items():
        print(f"{name.capitalize()} Portfolio:")
        for asset, weight in zip(assets, weights):
            print(f"  {asset}: {weight:.2%}")
        print()
