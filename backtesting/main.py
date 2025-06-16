# main.py
import pandas as pd
from utils import validate_backtest_inputs, calculate_strategy_returns, compute_metrics, plot_cumulative_returns
from config import CONFIG

def main():
    # Example input data: Replace with actual data or dynamic input handling
    strategy_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Price': [100, 101, 102, 103],
        'Position': [1.0, 0.5, -1.0, 0.0]
    })

    try:
        validated_data = validate_backtest_inputs(strategy_data)

        # Calculate strategy returns
        strategy_returns = calculate_strategy_returns(validated_data)
        print("\nStrategy Returns:")
        print(strategy_returns)

        # Compute metrics
        metrics = compute_metrics(strategy_returns)
        print("\nPerformance Metrics:")
        print(metrics)

        # Plot cumulative returns
        plot_cumulative_returns(strategy_returns)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
