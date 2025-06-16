import numpy as np
from scipy.stats import norm


class MarkovModulatedBlackScholes:
    def __init__(self, regimes, transition_matrix):
        """
        Initialize the Markov-Modulated Black-Scholes model.

        Parameters:
        -----------
        regimes : list of dict
            A list where each element is a dictionary containing the parameters
            for a specific regime. Each dictionary should have the keys:
            'mu', 'sigma', and 'r'.
        transition_matrix : numpy.ndarray
            A square matrix where element [i, j] is the probability of transitioning
            from regime i to regime j. The sum of each row must be 1.
        """
        self.regimes = regimes
        self.transition_matrix = transition_matrix
        self.num_regimes = len(regimes)
        if not np.allclose(np.sum(transition_matrix, axis=1), 1):
            raise ValueError("Each row of the transition matrix must sum to 1.")
    
    def simulate_markov_chain(self, T, dt):
        """
        Simulate a Markov chain over the time period [0, T].

        Parameters:
        -----------
        T : float
            Total time to simulate (in years).
        dt : float
            Time step for the simulation.

        Returns:
        --------
        numpy.ndarray
            Array of regime indices over time.
        """
        num_steps = int(T / dt)
        regimes = np.zeros(num_steps, dtype=int)
        for t in range(1, num_steps):
            current_regime = regimes[t - 1]
            regimes[t] = np.random.choice(
                self.num_regimes, p=self.transition_matrix[current_regime]
            )
        return regimes

    def black_scholes(self, S, K, T, mu, sigma, r, option_type="call"):
        """
        Black-Scholes model for European options in a single regime.

        Parameters:
        -----------
        S : float
            Current stock price.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        mu : float
            Expected return of the stock (not used in pricing but included for consistency).
        sigma : float
            Volatility of the stock (annualized).
        r : float
            Risk-free interest rate (annualized).
        option_type : str
            "call" for a call option, "put" for a put option.

        Returns:
        --------
        float
            Option price.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def price_option(self, S, K, T, dt, option_type="call"):
        """
        Price an option using the Markov-Modulated Black-Scholes model.

        Parameters:
        -----------
        S : float
            Current stock price.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        dt : float
            Time step for regime simulation.
        option_type : str
            "call" for a call option, "put" for a put option.

        Returns:
        --------
        float
            Option price averaged over all simulated regimes.
        """
        regimes = self.simulate_markov_chain(T, dt)
        num_steps = len(regimes)
        step_time = T / num_steps
        option_prices = []

        for t in range(num_steps):
            regime = regimes[t]
            params = self.regimes[regime]
            r = params["r"]
            sigma = params["sigma"]
            mu = params["mu"]
            # Calculate option price in the current regime
            option_price = self.black_scholes(
                S, K, step_time, mu, sigma, r, option_type
            )
            option_prices.append(option_price)

        # Return the average option price across simulated regimes
        return np.mean(option_prices)


# Example usage
if __name__ == "__main__":
    # Define regimes
    regimes = [
        {"mu": 0.08, "sigma": 0.15, "r": 0.03},  # Regime 1: Stable market
        {"mu": 0.05, "sigma": 0.30, "r": 0.02},  # Regime 2: Volatile market
    ]

    # Define transition matrix
    transition_matrix = np.array([
        [0.9, 0.1],  # Transition probabilities from Regime 1
        [0.2, 0.8],  # Transition probabilities from Regime 2
    ])

    # Initialize model
    model = MarkovModulatedBlackScholes(regimes, transition_matrix)

    # Parameters for option pricing
    S = 100      # Current stock price
    K = 110      # Strike price
    T = 1        # Time to maturity (1 year)
    dt = 0.01    # Time step for Markov chain simulation

    # Price call and put options
    call_price = model.price_option(S, K, T, dt, option_type="call")
    put_price = model.price_option(S, K, T, dt, option_type="put")

    print(f"Call Option Price: {call_price:.2f}")
    print(f"Put Option Price: {put_price:.2f}")
