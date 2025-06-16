import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes model for European options.

    Parameters:
    -----------
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the stock (annualized).
    option_type : str, optional
        "call" for a call option, "put" for a put option (default is "call").

    Returns:
    --------
    float
        Option price.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        # Call option price
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        # Put option price
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return option_price


# Example usage
if __name__ == "__main__":
    # Parameters
    S = 100      # Current stock price
    K = 110      # Strike price
    T = 1        # Time to maturity (1 year)
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)

    # Calculate call and put option prices
    call_price = black_scholes(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes(S, K, T, r, sigma, option_type="put")

    print(f"Call Option Price: {call_price:.2f}")
    print(f"Put Option Price: {put_price:.2f}")
