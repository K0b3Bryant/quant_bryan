import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG

def validate_backtest_inputs(data: pd.DataFrame):
    # Ensure required columns exist
    required_columns = {'Date', 'Price', 'Position'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data must contain the following columns: {required_columns}")

    # Check date formats and ensure consistency
    data['Date'] = pd.to_datetime(data['Date'], format=CONFIG['date_format'])

    # Check for missing values
    if data.isnull().values.any():
        raise ValueError("Data contains missing values.")

    return data

def calculate_strategy_returns(data: pd.DataFrame):
    # Calculate returns of the underlying prices
    data['Price_Returns'] = data['Price'].pct_change()

    # Lag the positions to avoid lookahead bias
    data['Lagged_Position'] = data['Position'].shift(CONFIG['lag'])

    # Calculate transaction costs
    data['Transaction_Costs'] = CONFIG['transaction_cost'] * abs(data['Position'].diff())

    # Calculate strategy returns
    data['Strategy_Returns'] = (data['Lagged_Position'] * data['Price_Returns']) - data['Transaction_Costs']

    # Drop rows with NaN values caused by shifting or percent change
    data.dropna(inplace=True)

    return data[['Date', 'Price', 'Position', 'Price_Returns', 'Strategy_Returns', 'Transaction_Costs']]

def compute_metrics(data: pd.DataFrame):
    # Cumulative returns
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
    data['Cumulative_Underlying'] = (1 + data['Price_Returns']).cumprod()

    # Whole period metrics
    sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()
    downside_std = data['Strategy_Returns'][data['Strategy_Returns'] < 0].std()
    sortino_ratio = data['Strategy_Returns'].mean() / downside_std
    volatility = data['Strategy_Returns'].std()

    # Rolling metrics
    rolling_sharpe = data['Strategy_Returns'].rolling(CONFIG['rolling_window']).mean() / \
                     data['Strategy_Returns'].rolling(CONFIG['rolling_window']).std()
    rolling_sortino = data['Strategy_Returns'].rolling(CONFIG['rolling_window']).mean() / \
                      data['Strategy_Returns'].rolling(CONFIG['rolling_window']).apply(
                          lambda x: x[x < 0].std(), raw=False)
    rolling_volatility = data['Strategy_Returns'].rolling(CONFIG['rolling_window']).std()

    return {
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Volatility': volatility,
        'Rolling Sharpe': rolling_sharpe,
        'Rolling Sortino': rolling_sortino,
        'Rolling Volatility': rolling_volatility
    }

def plot_cumulative_returns(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Cumulative_Strategy'], label='Cumulative Strategy Returns', linestyle='--')
    plt.plot(data['Date'], data['Cumulative_Underlying'], label='Cumulative Underlying Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()
