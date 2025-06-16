# system
import warnings

# general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# configuration
data_files = {
    'spx': '/content/SPX.csv',
    'vix': '/content/VIX.csv',
    'features_5d': '/content/features_5d.csv',
    'features_ta': '/content/features_ta.csv'
}

# Global settings
random_state = 42



def generate_features(install_packages=True, data:pd.Dataframe, price_col_name:str, window:int=5, window_min:int=5):
    """Generate time series features for the given price data."""
    
    if install_packages:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import make_forecasting_frame, impute, roll_time_series
        from ta.momentum import RSIIndicator, ROCIndicator
        from ta.volatility import BollingerBands
        from tsfracdiff import FractionalDifferentiator

    # Base feature
    feature_base = pd.DataFrame()
    feature_base['feature'] = price_data['price_scaled']

    # TA features
    features_ta = pd.DataFrame()
    features_ta['RSI'] = RSIIndicator(feature_base['feature'], window=14).rsi()
    features_ta['ROC_6M'] = ROCIndicator(feature_base['feature'], window=125).roc()
    features_ta['ROC_12M'] = ROCIndicator(feature_base['feature'], window=250).roc()
    features_ta['BB'] = BollingerBands(feature_base['feature'], window=20, window_dev=2).bollinger_hband()

    # Custom CMO function
    def cmo(close_prices, window=14):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        sum_gain = gain.rolling(window=window).sum()
        sum_loss = loss.rolling(window=window).sum()
        return ((sum_gain - sum_loss) / (sum_gain + sum_loss)).abs() * 100

    features_ta['CMO'] = cmo(feature_base['feature'], window=14)
    features_ta = features_ta.fillna(method='bfill')

    # tsfresh features
    price_rolled = pd.DataFrame(feature_base['feature'], columns=['feature'])
    price_rolled['date'] = price_rolled.index
    price_rolled['underlying'] = 'spx'

    price_rolled = roll_time_series(
        price_rolled,
        column_id='underlying',
        column_sort='date',
        max_timeshift=window,
        min_timeshift=window_min
    ).drop('underlying', axis=1)

    features_tsfresh = extract_features(
        price_rolled,
        column_id='id',
        column_sort='date',
        column_value='feature',
        impute_function=impute,
        show_warnings=False
    )

    features_tsfresh.index = features_tsfresh.index.map(lambda x: x[1])
    features_tsfresh.index.name = 'last_date'
    features_tsfresh = features_tsfresh.fillna(method='bfill')

    # Merge features
    features_merge = pd.merge(features_tsfresh, features_ta, left_index=True, right_index=True, how='outer').fillna(method='bfill')

    # Apply fractional differencing
    frac_diff = FractionalDifferentiator()
    features_fd = frac_diff.FitTransform(features_merge)

    return features_merge, features_fd

# analysis
import pandas as pd
import numpy as np
from ta.momentum import KAMAIndicator

def load_data(data_files):
    """Load datasets and preprocess them."""
    spx = pd.read_csv(data_files['spx'], parse_dates=['Date'], index_col=0).sort_index().astype(float)
    vix = pd.read_csv(data_files['vix'], parse_dates=['Date'], index_col=0).sort_index().astype(float)
    features_5d = pd.read_csv(data_files['features_5d'], parse_dates=['date'], index_col=0).sort_index()
    features_ta = pd.read_csv(data_files['features_ta'], index_col=0, parse_dates=True).sort_index()

    # Preprocessing
    spx.rename(columns={'Last Price': 'price'}, inplace=True)
    spx.drop(['Volume'], axis=1, inplace=True)
    vix.rename(columns={'VIX': 'vix'}, inplace=True)
    vix.drop(['First', 'Last'], axis=1, inplace=True)
    vix = vix.groupby(vix.index).mean()  # Eliminate duplicates

    features_5d.drop(columns=['date', 'underlying', 'price'], inplace=True)

    raw = pd.concat([spx, vix, features_5d, features_ta], axis=1)
    return raw

def kama_signals(data, fast, slow, signal_window):
    """Calculate KAMA signals."""
    kama = KAMAIndicator(close=data, window_slow=slow, window_fast=fast, window=signal_window).kama()
    return kama

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse}, MAE: {mae}, R^2: {r2}")



def model(install_packages=True):
    
    if install_packages:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])

    return pipeline


def main():
    print("Initiating kama_regime_model")

    # Load data
    raw = load_data(data_files)

    # Generate features
    features, features_fd = generate_features(raw)
    print("Features generated successfully.")

    # Example: Generate KAMA signals
    fast, slow, signal_window = 10, 30, 5
    kama = kama_signals(raw['price'], fast, slow, signal_window)

    # Plot KAMA vs. Price
    plt.figure(figsize=(10, 6))
    plt.plot(raw['price'], label='Price')
    plt.plot(kama, label=f'KAMA ({fast}, {slow}, {signal_window})')
    plt.title('Price and KAMA Indicator')
    plt.legend()
    plt.show()

    # Model pipeline (example)
    model = model()
    
    # Example evaluation
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    evaluate_model(y_true, y_pred)


if __name__ == "__main__":
  main()
