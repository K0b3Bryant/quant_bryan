# pricing_engines.py

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from instruments import Instrument, Bond, EuropeanOption
from market_data import MarketDataContext

class PricingEngine(ABC):
    """Abstract Base Class for all pricing engines."""
    @abstractmethod
    def calculate(self, instrument: Instrument, market_data: MarketDataContext) -> float:
        pass

class DiscountingBondEngine(PricingEngine):
    """Prices a bond by discounting its future cash flows."""
    def calculate(self, instrument: Bond, market_data: MarketDataContext) -> float:
        if not isinstance(instrument, Bond):
            raise TypeError("This engine only prices Bonds.")
            
        cashflows = instrument.get_cashflows(market_data.valuation_date)
        
        present_value = 0.0
        for cf_date, amount in cashflows:
            df = market_data.yield_curve.get_df(cf_date)
            present_value += amount * df
            
        return present_value

class BlackScholesEngine(PricingEngine):
    """Prices a European Option using the Black-Scholes formula."""
    def calculate(self, instrument: EuropeanOption, market_data: MarketDataContext) -> float:
        if not isinstance(instrument, EuropeanOption):
            raise TypeError("This engine only prices European Options.")

        S = market_data.get_spot(instrument.underlying_ticker)
        K = instrument.strike_price
        sigma = market_data.get_vol(instrument.underlying_ticker)
        T = (instrument.expiry_date - market_data.valuation_date).days / 365.25
        
        # Risk-free rate from the yield curve for the option's expiry
        r = market_data.yield_curve.get_rate(instrument.expiry_date)

        if T <= 0: # Option has expired
            return max(0, S - K) if instrument.option_type == 'Call' else max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if instrument.option_type == 'Call':
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif instrument.option_type == 'Put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type")
            
        return price
