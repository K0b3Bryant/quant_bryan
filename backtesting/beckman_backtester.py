import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    """
    A high-performance, vectorized backtesting engine that consumes price and signal series 
    and computes a wide range of trading performance metrics.
    
    This engine is designed for research and rapid iteration, leveraging
    vectorized operations via pandas and numpy for high performance. It correctly
    handles look-ahead bias and proportional transaction costs.
    """
    def __init__(self,
                 price: pd.Series,
                 signals: pd.Series,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        """
        Initializes the Backtester object.

        Args:
            price (pd.Series): Series of asset close prices, indexed by datetime.
            signals (pd.Series): Series of trading signals (+1 for long, 0 for flat, -1 for short), 
                                 with the same index as the price series. The signal for a given
                                 day T is assumed to be generated based on information available
                                 before the close of day T.
            initial_capital (float): The starting cash for the backtest.
            transaction_cost (float): Proportional cost per trade (e.g., 0.001 for 0.1%).
        """
        # --- Data Preparation and Validation ---
        # Ensure data is sorted by date to prevent temporal errors
        self.price = price.copy().sort_index()
        
        # Align signals to the price index. This is crucial if the signal generation
        # process creates a series with a different index. We fill any missing dates
        # with a flat position (0), assuming no signal means no position.
        self.signals = signals.reindex(self.price.index).fillna(0)
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None # Will store the backtest results DataFrame

    def run(self) -> pd.DataFrame:
        """
        Executes the vectorized backtest.
        
        This method calculates daily PnL by taking the element-wise product of the position
        vector (determined by the previous day's signal) and the current day's return vector.
        Transaction costs are incurred whenever the position changes.
        """
        # --- 1. Calculate Market Returns ---
        # The daily percentage change in price. The first value will be NaN, so we fill it with 0.
        # This vector represents the return earned on day T if a position was held.
        returns = self.price.pct_change().fillna(0)
        
        # --- 2. Determine Positions (Crucial for avoiding look-ahead bias) ---
        # The core of avoiding look-ahead bias is to use the signal from day T-1 to
        # determine the position held throughout day T, which earns the return of day T.
        # Shifting the signal series by one period forward achieves this vectorizedly.
        # The first position will be NaN, so we fill it with 0.
        positions = self.signals.shift(1).fillna(0)

        # --- 3. Vectorized Profit & Loss Computation ---
        # The gross PnL is the element-wise product of the positions vector and the returns vector.
        # This single line calculates the daily PnL for the entire history.
        pnl = positions * returns
        
        # --- 4. Calculate Transaction Costs ---
        # A trade occurs whenever the position changes. The `diff()` method calculates the
        # change from the previous period (e.g., from +1 to -1 is a diff of -2).
        # We take the absolute value as the size of the trade determines the cost, not its direction.
        trades = positions.diff().abs()
        
        # The cost is the value of the trade multiplied by the proportional transaction cost.
        cost = trades * self.transaction_cost
        
        # The net PnL is the gross PnL minus the associated transaction costs.
        net_pnl = pnl - cost

        # --- 5. Equity Curve Calculation ---
        # The equity curve is the cumulative product of (1 + net daily PnL),
        # which represents the compounding growth of the initial capital.
        equity = (1 + net_pnl).cumprod() * self.initial_capital

        # Store all intermediate and final results in a DataFrame for comprehensive analysis.
        self.results = pd.DataFrame({
            'Price': self.price,
            'Signal': self.signals,
            'Position': positions,
            'Market Return': returns,
            'Gross PnL': pnl,
            'Trade Cost': cost,
            'Net PnL': net_pnl,
            'Equity': equity
        })
        return self.results

    def performance_metrics(self, risk_free_rate: float = 0.0) -> dict:
        """
        Computes an extensive set of trading performance metrics from the backtest results.
        
        Args:
            risk_free_rate (float): The annualized risk-free rate for calculating ratios like Sharpe.
                                    This should be expressed as a decimal (e.g., 0.02 for 2%).

        Returns:
            A dictionary containing all key performance indicators (KPIs).
        """
        if self.results is None:
            raise RuntimeError("Backtest has not been run. Call .run() first.")

        r = self.results['Net PnL']
        equity = self.results['Equity']
        
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        n_periods = len(r)
        annual_factor = 252  # Assuming daily data.

        annualized_return = (1 + total_return) ** (annual_factor / n_periods) - 1
        annualized_vol = r.std() * np.sqrt(annual_factor)

        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else np.nan
        
        neg_returns = r[r < 0]
        downside_vol = neg_returns.std() * np.sqrt(annual_factor) if not neg_returns.empty else 0
        sortino = (annualized_return - risk_free_rate) / downside_vol if downside_vol != 0 else np.nan

        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else np.nan

        trade_count = int(self.signals.diff().abs().sum() / 2)
        wins = r[r > 0]
        losses = r[r < 0]
        num_trades = len(wins) + len(losses)
        win_rate = len(wins) / num_trades if num_trades > 0 else np.nan
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Maximum Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Trade Count': trade_count,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Skewness': skew(r, nan_policy='omit'),
            'Kurtosis': kurtosis(r, nan_policy='omit')
        }

    def plot_results(self):
        """Generates a standard performance tear sheet plot."""
        if self.results is None:
            self.run()

        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        fig.suptitle('Strategy performance analysis', fontsize=16)

        # Plot 1: Equity Curve and High Water Mark
        axes[0].plot(self.results['Equity'], label='Equity curve', color='blue', lw=2)
        axes[0].plot(self.results['Equity'].cummax(), label='High water mark', color='green', linestyle='--', lw=1)
        axes[0].set_title('Equity curve')
        axes[0].set_ylabel('Portfolio value ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)

        # Plot 2: Price Series with Positions
        axes[1].plot(self.results['Price'], label='Asset Price', color='black', lw=1)
        axes[1].set_title('Price and trading positions')
        axes[1].set_ylabel('Price')
        ax2_twin = axes[1].twinx()
        ax2_twin.fill_between(self.results.index, 0, self.results['Position'], where=self.results['Position'] > 0, color='green', alpha=0.2, label='Long Position')
        ax2_twin.fill_between(self.results.index, 0, self.results['Position'], where=self.results['Position'] < 0, color='red', alpha=0.2, label='Short Position')
        ax2_twin.set_ylabel('Position')
        ax2_twin.legend(loc='upper right')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # Plot 3: Drawdown Series
        drawdown_series = (self.results['Equity'] - self.results['Equity'].cummax()) / self.results['Equity'].cummax()
        axes[2].fill_between(drawdown_series.index, drawdown_series, 0, color='red', alpha=0.3)
        axes[2].set_title('Drawdown profile')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
Using the Backtester class requires providing price and signal data. The following demonstrates a typical workflow, from generating a simple moving average crossover strategy to evaluating its performance and interpreting the output.

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
price_series = pd.Series(100 + np.random.randn(252 * 5).cumsum(), 
                         index=pd.to_datetime(pd.date_range('2020-01-01', periods=252 * 5)))

# --- 2. Create a Simple Trading Strategy Signal ---
fast_ma = price_series.rolling(window=30).mean()
slow_ma = price_series.rolling(window=90).mean()
signal_series = pd.Series(np.where(fast_ma > slow_ma, 1, -1), index=price_series.index).fillna(0)

# --- 3. Instantiate and Run the Backtester ---
bt = Backtester(price=price_series, 
                signals=signal_series, 
                initial_capital=100000, 
                transaction_cost=0.001)
bt.run()
metrics_dict = bt.performance_metrics()

# --- 4. Analyze Results ---
print("--- Performance Metrics ---")
metrics_series = pd.Series(metrics_dict)
print(metrics_series.to_string(float_format="{:.4f}".format))

bt.plot_results()
