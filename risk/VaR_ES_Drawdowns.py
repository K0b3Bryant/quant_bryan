import numpy as np
import pandas as pd


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Historical Value at Risk (VaR) at a given confidence level."""
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    return var


def expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Historical Expected Shortfall (CVaR) at a given confidence level."""
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    es = -returns[returns <= var_threshold].mean()
    return es


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate Maximum Drawdown from cumulative returns."""
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_dd = drawdowns.min()
    return max_dd


def simulate_returns(n_days=252, mu=0.08, sigma=0.15, seed=42) -> pd.Series:
    """Simulate daily returns using a normal distribution."""
    np.random.seed(seed)
    daily_returns = np.random.normal(loc=mu / n_days, scale=sigma / np.sqrt(n_days), size=n_days)
    return pd.Series(daily_returns)


if __name__ == "__main__":
    # Simulate 1-year daily returns
    daily_returns = simulate_returns()

    # Compute cumulative returns for drawdown analysis
    cumulative = (1 + daily_returns).cumprod()

    # Risk metrics
    var_95 = value_at_risk(daily_returns, 0.95)
    es_95 = expected_shortfall(daily_returns, 0.95)
    max_dd = max_drawdown(cumulative)

    print(f"Value at Risk (95%): {var_95:.2%}")
    print(f"Expected Shortfall (95%): {es_95:.2%}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
