import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_stress_test(returns: pd.DataFrame, weights: np.ndarray, pc_index: int = 0, sigma: float = -5.0, plot: bool = True) -> float:
    """
    Stress tests a portfolio by applying a sigma-level shock to a PCA factor.

    Parameters:
    - returns: DataFrame of historical asset returns (assets as columns)
    - weights: Portfolio weights as a NumPy array
    - pc_index: Index of principal component to shock (0-based)
    - sigma: Number of standard deviations to shock the component (e.g., -5 for crash)
    - plot: Whether to plot PCA factor loadings

    Returns:
    - Estimated portfolio return under the shock scenario
    """
    assert returns.shape[1] == len(weights), "Number of assets must match number of weights"

    # Step 1: Center the returns
    returns_centered = returns - returns.mean()

    # Step 2: Fit PCA
    pca = PCA()
    pca.fit(returns_centered)

    components = pd.DataFrame(pca.components_, columns=returns.columns)
    components.index = [f"PC{i+1}" for i in range(len(components))]

    # Step 3: Get std dev of component scores
    pc_scores = pd.DataFrame(pca.transform(returns_centered), columns=components.index)
    std_dev = pc_scores.iloc[:, pc_index].std()
    shock_amount = sigma * std_dev

    # Step 4: Reverse transform the shock to asset returns
    shocked_component = components.iloc[pc_index].values
    asset_shock = shock_amount * shocked_component
    asset_shock_series = pd.Series(asset_shock, index=returns.columns)

    # Step 5: Compute portfolio impact
    portfolio_return = np.dot(weights, asset_shock)

    # Optional: Plot factor loadings
    if plot:
        components.T.plot.bar(figsize=(8, 5))
        plt.title("PCA Factor Loadings")
        plt.xlabel("Assets")
        plt.ylabel("Loading")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"Stress test result (PC{pc_index+1}, {sigma}σ): {portfolio_return:.2%}")
    return portfolio_return

if __name__ == "__main__":
    np.random.seed(42)

    # Simulated returns: 4 assets over 252 days
    assets = ['AAPL', 'GOOG', 'MSFT', 'BND']
    returns = pd.DataFrame(np.random.normal(0, 0.01, size=(252, 4)), columns=assets)

    # Portfolio weights
    weights = np.array([0.3, 0.3, 0.3, 0.1])

    # Run PCA stress test: shock PC1 by -5σ
    pca_stress_test(returns, weights, pc_index=0, sigma=-5.0)
