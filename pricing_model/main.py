# main.py

from datetime import date
from dateutil.relativedelta import relativedelta
from market_data import YieldCurve, MarketDataContext
from instruments import Bond, EuropeanOption
from pricing_engines import DiscountingBondEngine, BlackScholesEngine
from risk import calculate_pv01, calculate_greeks

def run_demonstration():
    """Sets up a market scenario and prices various instruments and their risks."""
    
    # --- 1. Setup Market Data ---
    print("--- Setting up Market Environment ---")
    valuation_date = date(2023, 1, 1)
    
    # Yield Curve (a simple upward sloping curve)
    curve_dates = [valuation_date + relativedelta(years=i) for i in [1, 2, 5, 10, 30]]
    curve_rates = [0.03, 0.035, 0.04, 0.042, 0.045]
    yield_curve = YieldCurve(curve_dates, curve_rates, valuation_date)
    
    # Spot and Volatility Data
    spot_prices = {'AAPL': 150.0, 'GOOGL': 2700.0}
    volatilities = {'AAPL': 0.25, 'GOOGL': 0.22}
    
    # Create the market context
    market_context = MarketDataContext(valuation_date, yield_curve, spot_prices, volatilities)
    print(f"Valuation Date: {valuation_date}")
    print(f"AAPL Spot: ${market_context.get_spot('AAPL')}, Vol: {market_context.get_vol('AAPL'):.2%}")
    print("-" * 35 + "\n")

    # --- 2. Setup Instruments ---
    gov_bond = Bond(
        issuer="US Treasury",
        notional=1000.0,
        coupon_rate=0.04,
        maturity_date=date(2033, 1, 1),
        issue_date=date(2023, 1, 1)
    )
    
    aapl_call_option = EuropeanOption(
        underlying_ticker='AAPL',
        option_type='Call',
        strike_price=160.0,
        expiry_date=date(2024, 1, 1)
    )

    aapl_put_option = EuropeanOption(
        underlying_ticker='AAPL',
        option_type='Put',
        strike_price=140.0,
        expiry_date=date(2024, 1, 1)
    )

    # --- 3. Setup Pricing Engines ---
    bond_pricer = DiscountingBondEngine()
    option_pricer = BlackScholesEngine()

    # --- 4. Price Instruments and Calculate Risk ---
    print("--- Pricing and Risk Analysis ---")
    
    # --- Bond ---
    bond_price = bond_pricer.calculate(gov_bond, market_context)
    bond_pv01 = calculate_pv01(gov_bond, bond_pricer, market_context)
    print(f"Instrument: {gov_bond}")
    print(f"  Price: ${bond_price:,.2f}")
    print(f"  Interest Rate Risk (PV01): ${bond_pv01:,.4f}")
    print("-" * 35)

    # --- Call Option ---
    call_price = option_pricer.calculate(aapl_call_option, market_context)
    call_greeks = calculate_greeks(aapl_call_option, option_pricer, market_context)
    print(f"Instrument: {aapl_call_option}")
    print(f"  Price: ${call_price:,.2f}")
    print(f"  Risk (Greeks): " + ', '.join([f"{k}={v:.4f}" for k, v in call_greeks.items()]))
    print("-" * 35)

    # --- Put Option ---
    put_price = option_pricer.calculate(aapl_put_option, market_context)
    put_greeks = calculate_greeks(aapl_put_option, option_pricer, market_context)
    print(f"Instrument: {aapl_put_option}")
    print(f"  Price: ${put_price:,.2f}")
    print(f"  Risk (Greeks): " + ', '.join([f"{k}={v:.4f}" for k, v in put_greeks.items()]))
    print("-" * 35)


if __name__ == "__main__":
    run_demonstration()
