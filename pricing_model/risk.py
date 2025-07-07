# risk.py

import copy
from market_data import MarketDataContext, YieldCurve
from instruments import Instrument, Bond, EuropeanOption
from pricing_engines import PricingEngine

def calculate_pv01(instrument: Bond, engine: PricingEngine, market_data: MarketDataContext) -> float:
    """Calculates PV01 (Present Value of a Basis Point) for a bond."""
    
    # 1. Price at current market
    base_price = engine.calculate(instrument, market_data)
    
    # 2. Create a bumped market context
    bumped_market_data = copy.deepcopy(market_data)
    
    # Shift the entire yield curve up by 1 basis point (0.0001)
    bumped_rates = market_data.yield_curve.rates + 0.0001
    bumped_market_data.yield_curve = YieldCurve(
        dates=market_data.yield_curve.dates,
        rates=bumped_rates,
        valuation_date=market_data.valuation_date
    )
    
    # 3. Price with the bumped market
    bumped_price = engine.calculate(instrument, bumped_market_data)
    
    # 4. PV01 is the difference in price
    pv01 = base_price - bumped_price
    return pv01

def calculate_greeks(instrument: EuropeanOption, engine: PricingEngine, market_data: MarketDataContext) -> dict:
    """Calculates the primary Greeks for a European Option."""
    greeks = {}
    
    # Small bumps for calculating derivatives
    spot_bump = 0.01
    vol_bump = 0.0001 # 1 basis point of vol
    rate_bump = 0.0001 # 1 basis point of interest rate
    
    base_price = engine.calculate(instrument, market_data)
    
    # --- Delta ---
    bumped_market_delta = copy.deepcopy(market_data)
    bumped_market_delta.spot_prices[instrument.underlying_ticker] += spot_bump
    price_up = engine.calculate(instrument, bumped_market_delta)
    greeks['Delta'] = (price_up - base_price) / spot_bump
    
    # --- Vega ---
    bumped_market_vega = copy.deepcopy(market_data)
    bumped_market_vega.volatilities[instrument.underlying_ticker] += vol_bump
    price_vol_up = engine.calculate(instrument, bumped_market_vega)
    greeks['Vega'] = (price_vol_up - base_price) / (vol_bump * 100) # Per 1% vol change

    # --- Rho ---
    bumped_market_rho = copy.deepcopy(market_data)
    bumped_rates = market_data.yield_curve.rates + rate_bump
    bumped_market_rho.yield_curve = YieldCurve(dates=market_data.yield_curve.dates, rates=bumped_rates, valuation_date=market_data.valuation_date)
    price_rate_up = engine.calculate(instrument, bumped_market_rho)
    greeks['Rho'] = (price_rate_up - base_price) / (rate_bump * 100) # Per 1% rate change

    # Gamma and Theta require more complex calculations or a second-order bump
    # for simplicity, we'll demonstrate the most common ones.
    
    return greeks
