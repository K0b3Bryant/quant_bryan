import numpy as np
import pandas as pd

# --- Volatility Targeting ---
def volatility_targeting(signal, returns, target_vol=0.10, window=20):
    rolling_vol = returns.rolling(window=window).std(ddof=0) * np.sqrt(252)
    vol_scaler = target_vol / rolling_vol
    position = signal * vol_scaler
    return position.fillna(0)

# --- Kelly Criterion (Fractional) ---
def kelly_sizer(signal, returns, fraction=0.25, window=60):
    mu = returns.rolling(window=window).mean()
    sigma2 = returns.rolling(window=window).var(ddof=0)
    kelly_fraction = mu / sigma2
    position = fraction * kelly_fraction * signal
    return position.replace([np.inf, -np.inf], 0).fillna(0)

# --- Optimal f (Empirical via Grid Search) ---
def optimal_f_sizer(signal, returns, capital=1.0, f_range=np.linspace(0, 1, 101)):
    historical_returns = (signal.shift() * returns).dropna()
    best_f = 0
    best_growth = -np.inf

    for f in f_range:
        growth = np.log1p(f * historical_returns).sum()
        if growth > best_growth:
            best_growth = growth
            best_f = f

    position = best_f * signal
    return position.fillna(0), best_f

# --- Sample usage function ---
def generate_position_sizers(signals, returns):
    vt = volatility_targeting(signals, returns)
    kelly = kelly_sizer(signals, returns)
    optf, best_f = optimal_f_sizer(signals, returns)

    df = pd.DataFrame({
        'VolTargeting': vt,
        'KellyFractional': kelly,
        'OptimalF': optf
    })
    return df, best_f
