import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG

def validate_inputs(data: pd.DataFrame):
    # Ensure required columns exist
    required_columns = [col for asset in CONFIG['assets'] for col in [f'{asset}_Price', f'{asset}_Position']]
    required_columns.insert(0, 'Date')

    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain the following columns: {required_columns}")

    # Check date formats and ensure consistency
    data['Date'] = pd.to_datetime(data['Date'], format=CONFIG['date_format'])

    # Check for missing values
    if data.isnull().values.any():
        raise ValueError("Data contains missing values.")

    return data

def calculate_portfolio_metrics(data: pd.DataFrame):
    portfolio_returns = []
    for asset in CONFIG['assets']:
        data[f'{asset}_Returns'] = data[f'{asset}_Price'].pct_change()
        data[f'{asset}_Weighted_Returns'] = data[f'{asset}_Returns'] * data[f'{asset}_Position'].shift(CONFIG['lag'])
        portfolio_returns.append(data[f'{asset}_Weighted_Returns'])

    # Portfolio Returns
    data['Portfolio_Returns'] = sum(portfolio_returns)

    return data[['Date', 'Portfolio_Returns'] + \
                [f'{asset}_Returns' for asset in CONFIG['assets']] + \
                [f'{asset}_Weighted_Returns' for asset in CONFIG['assets']]]

def calculate_risk_measures(data: pd.DataFrame):
    returns = data['Portfolio_Returns'].dropna()

    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 100 * (1 - CONFIG['confidence_level']))

    # Expected Shortfall (ES)
    es_95 = returns[returns <= var_95].mean()

    # Portfolio Volatility
    volatility = returns.std()

    return {
        'VaR (95%)': var_95,
        'Expected Shortfall (95%)': es_95,
        'Volatility': volatility
    }

def calculate_drawdown(data: pd.DataFrame):
    cumulative_returns = (1 + data['Portfolio_Returns']).cumprod()
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()
    duration = (drawdown != 0).astype(int).groupby(drawdown.eq(0).cumsum()).cumsum().max()
    return {
        'Max Drawdown': max_drawdown,
        'Drawdown Duration': duration
    }

def monte_carlo_simulation(returns, simulations=1000, horizon=30):
    simulated_returns = np.random.choice(returns, size=(horizon, simulations), replace=True)
    portfolio_paths = (1 + simulated_returns).cumprod(axis=0)
    return pd.DataFrame(portfolio_paths, columns=[f'Simulation_{i}' for i in range(simulations)])

def stress_test(data: pd.DataFrame, shock=-0.1):
    stressed_returns = data['Portfolio_Returns'] + shock
    var_95 = np.percentile(stressed_returns, 100 * (1 - CONFIG['confidence_level']))
    es_95 = stressed_returns[stressed_returns <= var_95].mean()
    return {
        'Stressed VaR': var_95,
        'Stressed ES': es_95
    }

def compute_correlation_matrix(data: pd.DataFrame):
    returns = pd.concat(
        [data[f'{asset}_Returns'] for asset in CONFIG['assets']],
        axis=1,
        keys=CONFIG['assets']
    )
    return returns.corr()

def plot_risk_visualizations(data: pd.DataFrame, monte_carlo_results, correlation_matrix):
    # Plot cumulative returns
    cumulative_returns = (1 + data['Portfolio_Returns']).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], cumulative_returns, label='Portfolio Cumulative Returns')
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Monte Carlo Simulation
    plt.figure(figsize=(10, 6))
    for col in monte_carlo_results.columns:
        plt.plot(range(len(monte_carlo_results)), monte_carlo_results[col], alpha=0.1, color='blue')
    plt.title('Monte Carlo Simulated Portfolio Paths')
    plt.xlabel('Time Horizon')
    plt.ylabel('Portfolio Value')
    plt.grid()
    plt.show()

    # Plot Correlation Matrix
    plt.figure(figsize=(8, 6))
    plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
    plt.colorbar()
    plt.title('Asset Correlation Matrix', pad=20)
    plt.show()
