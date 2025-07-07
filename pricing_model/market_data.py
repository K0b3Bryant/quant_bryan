# market_data.py

import numpy as np
from scipy.interpolate import interp1d
from datetime import date
from dateutil.relativedelta import relativedelta

class YieldCurve:
    """Represents an interest rate yield curve."""
    def __init__(self, dates: list[date], rates: list[float], valuation_date: date):
        self.valuation_date = valuation_date
        
        # Convert dates to years from valuation date for interpolation
        day_counts = np.array([(d - valuation_date).days for d in dates])
        self.years = day_counts / 365.25
        self.rates = np.array(rates)
        
        # Use linear interpolation for rates, fill with edge values for extrapolation
        self._interpolator = interp1d(self.years, self.rates, kind='linear', fill_value="extrapolate")

    def get_rate(self, target_date: date) -> float:
        """Get the interpolated interest rate for a given date."""
        year_frac = (target_date - self.valuation_date).days / 365.25
        return self._interpolator(year_frac).item()

    def get_df(self, target_date: date) -> float:
        """Get the discount factor for a given date."""
        rate = self.get_rate(target_date)
        year_frac = (target_date - self.valuation_date).days / 365.25
        # Continuous compounding for discount factor: exp(-r*t)
        return np.exp(-rate * year_frac)

class MarketDataContext:
    """A container for all market data needed for pricing."""
    def __init__(self, valuation_date: date, yield_curve: YieldCurve, spot_prices: dict, volatilities: dict):
        self.valuation_date = valuation_date
        self.yield_curve = yield_curve
        self.spot_prices = spot_prices # e.g., {'AAPL': 150.0}
        self.volatilities = volatilities # e.g., {'AAPL': 0.20}

    def get_spot(self, ticker: str) -> float:
        return self.spot_prices.get(ticker, 0.0)

    def get_vol(self, ticker: str) -> float:
        return self.volatilities.get(ticker, 0.0)
