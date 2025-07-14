import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# models.py
# Base ShortRateModel class and implementations
class ShortRateModel:
    def __init__(self, r0, a, b, sigma):
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma = sigma

    def simulate_paths(self, T, n_paths=1000, dt=0.01):
        raise NotImplementedError()

    def zero_coupon_bond_price(self, T, t=0, r=None):
        raise NotImplementedError()

    def monte_carlo_discount_factor(self, T, n_paths=10000, dt=0.01):
        """Simulate discount factor using Monte Carlo: E[exp(-∫r dt)]"""
        paths = self.simulate_paths(T, n_paths=n_paths, dt=dt)
        r_avg = paths.mean(axis=0)  # average over time for each path
        discount_factors = np.exp(-r_avg * T)
        return discount_factors.mean()

# Vasicek Model
class VasicekModel(ShortRateModel):
    def simulate_paths(self, T, n_paths=1000, dt=0.01):
        n_steps = int(T / dt)
        rates = np.zeros((n_steps + 1, n_paths))
        rates[0] = self.r0

        for t in range(1, n_steps + 1):
            dr = self.a * (self.b - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn(n_paths)
            rates[t] = rates[t-1] + dr
        return rates

    def zero_coupon_bond_price(self, T, t=0, r=None):
        if r is None:
            r = self.r0
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        A = np.exp((self.b - (self.sigma**2) / (2 * self.a**2)) * (B - (T - t)) - (self.sigma**2) * B**2 / (4 * self.a))
        return A * np.exp(-B * r)


# CIR Model
class CIRModel(ShortRateModel):
    def simulate_paths(self, T, n_paths=1000, dt=0.01):
        n_steps = int(T / dt)
        rates = np.zeros((n_steps + 1, n_paths))
        rates[0] = self.r0

        for t in range(1, n_steps + 1):
            sqrt_r = np.sqrt(np.maximum(rates[t-1], 0))
            dr = self.a * (self.b - rates[t-1]) * dt + self.sigma * sqrt_r * np.sqrt(dt) * np.random.randn(n_paths)
            rates[t] = np.maximum(rates[t-1] + dr, 0)
        return rates

    def zero_coupon_bond_price(self, T, t=0, r=None):
        if r is None:
            r = self.r0
        gamma = np.sqrt(self.a**2 + 2 * self.sigma**2)
        h1 = (self.a + gamma) / 2
        h2 = 2 * self.a * self.b / self.sigma**2
        exp_gamma_T = np.exp(gamma * (T - t))
        B = 2 * (exp_gamma_T - 1) / ((gamma + self.a) * (exp_gamma_T - 1) + 2 * gamma)
        A = ((2 * gamma * np.exp((self.a + gamma) * (T - t) / 2)) / ((gamma + self.a) * (exp_gamma_T - 1) + 2 * gamma)) ** h2
        return A * np.exp(-B * r)


# Hull-White model (mean-reverting with time-dependent drift approximation)
class HullWhiteModel(ShortRateModel):
    def simulate_paths(self, T, n_paths=1000, dt=0.01):
        n_steps = int(T / dt)
        rates = np.zeros((n_steps + 1, n_paths))
        rates[0] = self.r0

        for t in range(1, n_steps + 1):
            dr = self.a * (self.b - rates[t-1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn(n_paths)
            rates[t] = rates[t-1] + dr
        return rates

    def zero_coupon_bond_price(self, T, t=0, r=None):
        if r is None:
            r = self.r0
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        A = np.exp((self.b - self.sigma**2 / (2 * self.a**2)) * (B - (T - t)) - (self.sigma**2) * B**2 / (4 * self.a))
        return A * np.exp(-B * r)



# prodcuts.py
# Base Product class and implementations - MISSING SOME PRODUCTS!
class Product:
    def __init__(self, model):
        self.model = model

    def price_analytical(self):
        raise NotImplementedError("Override in subclass")

    def price_monte_carlo(self, n_paths=10000, dt=0.01):
        raise NotImplementedError("Override in subclass")

class ZeroCouponBond(Product):
    def __init__(self, model, maturity):
        super().__init__(model)
        self.maturity = maturity

    def price_analytical(self):
        return self.model.zero_coupon_bond_price(self.maturity)

    def price_monte_carlo(self, n_paths=10000, dt=0.01):
        paths = self.model.simulate_paths(self.maturity, n_paths=n_paths, dt=dt)
        # Discount factor = exp(- integral r dt), approximate integral by average rate * maturity
        avg_r = np.mean(paths, axis=0)
        discounts = np.exp(-avg_r * self.maturity)
        return np.mean(discounts)

class FRA(Product):
    def __init__(self, model, start, end, notional, strike):
        super().__init__(model)
        self.start = start
        self.end = end
        self.notional = notional
        self.strike = strike

    def price_analytical(self):
        P_start = self.model.zero_coupon_bond_price(self.start)
        P_end = self.model.zero_coupon_bond_price(self.end)
        forward_rate = (P_start / P_end - 1) / (self.end - self.start)
        price = self.notional * (forward_rate - self.strike) * (self.end - self.start) * P_end
        return price

    def price_monte_carlo(self, n_paths=10000, dt=0.01):
        paths = self.model.simulate_paths(self.end, n_paths=n_paths, dt=dt)
        start_idx = int(self.start / dt)
        end_idx = int(self.end / dt)

        df = np.exp(-dt * np.sum(paths[:end_idx], axis=0))
        r_start = paths[start_idx]
        r_end = paths[end_idx]

        # Approximate forward rate by log return ratio
        forward_rate = (np.exp((r_start - r_end) * (self.end - self.start)) - 1) / (self.end - self.start)
        payoff = (forward_rate - self.strike) * self.notional * (self.end - self.start)
        return np.mean(payoff * df)

class BermudanSwaption(Product):
    def __init__(self, model, notional, strike, n_exercise, swap_length, total_maturity):
        super().__init__(model)
        self.notional = notional
        self.strike = strike
        self.n_exercise = n_exercise
        self.swap_length = swap_length
        self.total_maturity = total_maturity

    def price_monte_carlo(self, n_paths=10000, dt=0.01):
        exercise_times = np.linspace(1, self.total_maturity - self.swap_length, self.n_exercise)
        paths = self.model.simulate_paths(self.total_maturity, n_paths=n_paths, dt=dt)
        idx_grid = [int(t / dt) for t in exercise_times]

        swaption_values = np.zeros((n_paths, len(idx_grid)))
        for j, idx in enumerate(idx_grid):
            t = exercise_times[j]
            payment_times = np.arange(t + 1, t + self.swap_length + 1)
            swap_pv = np.zeros(n_paths)
            for pt in payment_times:
                d_idx = int(pt / dt)
                df = np.exp(-np.cumsum(paths[:d_idx], axis=0)[-1] * dt)
                swap_pv += (self.strike - paths[d_idx]) * df  # receiver swaption
            swaption_values[:, j] = self.notional * swap_pv

        # Longstaff-Schwartz backward induction
        value = swaption_values[:, -1]
        for j in reversed(range(len(idx_grid) - 1)):
            itm = swaption_values[:, j] > 0
            X = swaption_values[itm, j]
            Y = value[itm]
            if len(X) == 0:
                continue
            r_i = paths[idx_grid[j], itm]
            A = np.vstack([np.ones_like(r_i), r_i, r_i**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * r_i + coeffs[2] * r_i**2
            exercise_now = X > continuation
            idxs = np.where(itm)[0][exercise_now]
            value[idxs] = X[exercise_now]

        return np.mean(value)


# engine.py
# Pricing Engine

from models import VasicekModel, CIRModel, HullWhiteModel
from products import ZeroCouponBond, FRA, BermudanSwaption

class PricingEngine:
    def __init__(self, model_name, model_params, product_name, product_params):
        # Instantiate model
        model_classes = {
            "vasicek": VasicekModel,
            "cir": CIRModel,
            "hull-white": HullWhiteModel,
        }
        self.model = model_classes[model_name.lower()](**model_params)

        # Instantiate product
        product_classes = {
            "zero_coupon": ZeroCouponBond,
            "fra": FRA,
            "bermudan_swaption": BermudanSwaption,
        }
        self.product = product_classes[product_name.lower()](self.model, **product_params)

    def price(self, method="analytical", **kwargs):
        if method == "analytical":
            return self.product.price_analytical()
        elif method == "monte_carlo":
            return self.product.price_monte_carlo(**kwargs)
        else:
            raise ValueError("Invalid pricing method")

# new main.py
from engine import PricingEngine

def main():
    # Example parameters
    model_name = "vasicek"
    model_params = {"r0": 0.03, "a": 0.1, "b": 0.05, "sigma": 0.02}

    product_name = "bermudan_swaption"
    product_params = {
        "notional": 1_000_000,
        "strike": 0.03,
        "n_exercise": 5,
        "swap_length": 3,
        "total_maturity": 10,
    }

    engine = PricingEngine(model_name, model_params, product_name, product_params)

    price_analytical = None
    try:
        price_analytical = engine.price(method="analytical")
    except NotImplementedError:
        pass

    price_mc = engine.price(method="monte_carlo", n_paths=5000, dt=0.01)

    print(f"Analytical price: {price_analytical}")
    print(f"Monte Carlo price: {price_mc}")

if __name__ == "__main__":
    main()




# Old Main function
def main():
    model_choice = input("Choose model [vasicek / cir / hull-white]: ").strip().lower()
    product_choice = input("Choose product [zero_coupon / fra / swap / caplet / bermudan_caplet / bermudan_swaption]: ").strip().lower()

    r0 = 0.03
    a = 0.1
    b = 0.05
    sigma = 0.02
    T = 5  # maturity

    if model_choice == "vasicek":
        model = VasicekModel(r0, a, b, sigma)
    elif model_choice == "cir":
        model = CIRModel(r0, a, b, sigma)
    elif model_choice == "hull-white":
        model = HullWhiteModel(r0, a, b, sigma)
    else:
        print("Invalid model.")
        return

    if product_choice == "zero_coupon":
        price = model.zero_coupon_bond_price(T)
        mc_price = model.monte_carlo_discount_factor(T)
        print(f"\nAnalytical ZCB Price at T={T}: {price:.4f}")
        print(f"Monte Carlo ZCB Price (n=10000): {mc_price:.4f}")

        paths = model.simulate_paths(T, n_paths=10)
        plt.plot(paths)
        plt.title(f"Interest Rate Paths - {model_choice.title()}")
        plt.xlabel("Time steps")
        plt.ylabel("Short Rate")
        plt.grid()
        plt.show()

        elif product_choice == "fra":
        T1 = float(input("Start of FRA (T1): "))
        T2 = float(input("End of FRA (T2): "))
        notional = float(input("Notional: "))
        K = float(input("Contract rate (K): "))
        dt = 0.01
        n_paths = 10000

        # Analytical
        P1 = model.zero_coupon_bond_price(T1)
        P2 = model.zero_coupon_bond_price(T2)
        forward_rate = (P1 / P2 - 1) / (T2 - T1)
        price = notional * (forward_rate - K) * (T2 - T1) * P2
        print(f"\nFRA Analytical Price: {price:.4f} (Forward rate: {forward_rate:.4%})")

        # Monte Carlo
        r_paths = model.simulate_paths(T2, n_paths=n_paths, dt=dt)
        t1_idx = int(T1 / dt)
        t2_idx = int(T2 / dt)
        r_t1 = r_paths[t1_idx]
        r_t2 = r_paths[t2_idx]

        df = np.exp(-dt * np.sum(r_paths[:t2_idx], axis=0))
        mc_forward_rate = (np.exp((r_t1 - r_t2) * (T2 - T1)) - 1) / (T2 - T1)
        mc_price = np.mean((mc_forward_rate - K) * notional * (T2 - T1) * df)
        print(f"FRA Monte Carlo Price (n={n_paths}): {mc_price:.4f}")

        # Plot rates
        plt.figure()
        plt.bar(["Analytical Fwd", "Contract K"], [forward_rate, K], color=["green", "red"])
        plt.title("FRA: Forward vs Contract Rate")
        plt.grid(axis="y")
        plt.show()

    elif product_choice == "swap":
        notional = float(input("Notional: "))
        fixed_rate = float(input("Fixed Rate: "))
        n_payments = int(input("Number of payments (e.g. 5 for annual over 5y): "))
        dt = T / n_payments
        pv_fixed = 0
        pv_floating = 1 - model.zero_coupon_bond_price(T)

        fixed_cash_flows = []
        float_cash_flows = []
        times = []

        for i in range(1, n_payments + 1):
            t_i = i * dt
            discount = model.zero_coupon_bond_price(t_i)
            fixed_cf = fixed_rate * dt * discount
            fixed_cash_flows.append(fixed_cf * notional)
            times.append(t_i)

        float_cash_flows = [notional * dt if i == n_payments - 1 else 0 for i in range(n_payments)]
        pv_fixed = sum(fixed_cash_flows)
        price = notional * (pv_floating - pv_fixed / notional)

        print(f"\nInterest Rate Swap PV (Receiver Float, Pay Fixed): {price:.4f}")

        # Plot: Cash flows
        plt.figure()
        plt.bar(times, fixed_cash_flows, width=0.2, label="Fixed", color="orange")
        plt.bar(times, float_cash_flows, width=0.2, label="Floating", color="blue", alpha=0.7)
        plt.title("Interest Rate Swap: Fixed vs Floating Cash Flows")
        plt.xlabel("Time")
        plt.ylabel("Cash Flow")
        plt.legend()
        plt.grid()
        plt.show()

    elif product_choice == "caplet":
        T_start = float(input("Start time of caplet: "))
        T_end = float(input("End time of caplet: "))
        K = float(input("Strike rate (K): "))
        notional = float(input("Notional: "))
        tau = T_end - T_start
        dt = 0.01
        n_paths = 10000

        # Analytical
        P = model.zero_coupon_bond_price(T_end)
        fwd = (model.zero_coupon_bond_price(T_start) / P - 1) / tau
        sigma = model.sigma
        d1 = (np.log(fwd / K) + 0.5 * sigma**2 * T_start) / (sigma * np.sqrt(T_start))
        d2 = d1 - sigma * np.sqrt(T_start)

        caplet_price = notional * tau * P * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
        print(f"\nCaplet Analytical Price: {caplet_price:.4f}")

        # Monte Carlo
        r_paths = model.simulate_paths(T_end, n_paths=n_paths, dt=dt)
        t_idx = int(T_start / dt)
        rt = r_paths[t_idx]
        payoff = np.maximum(rt - K, 0) * notional * tau
        df = np.exp(-dt * np.sum(r_paths[:t_idx], axis=0))
        mc_price = np.mean(payoff * df)
        print(f"Caplet Monte Carlo Price (n={n_paths}): {mc_price:.4f}")

        # Plot payoff
        fwd_rates = np.linspace(0, 2 * K, 100)
        payoffs = [max(fr - K, 0) * notional * tau for fr in fwd_rates]

        plt.figure()
        plt.plot(fwd_rates, payoffs, label="Caplet Payoff", color="purple")
        plt.axvline(fwd, color="green", linestyle="--", label="Fwd Rate")
        plt.axvline(K, color="red", linestyle="--", label="Strike Rate")
        plt.title("Caplet Payoff")
        plt.xlabel("Forward Rate")
        plt.ylabel("Payoff")
        plt.legend()
        plt.grid()
        plt.show()

    elif product_choice == "bermudan_caplet":
        notional = float(input("Notional: "))
        K = float(input("Strike rate (K): "))
        n_exercise = int(input("Number of exercise opportunities (e.g. 5): "))
        T_total = float(input("Final maturity (e.g. 5): "))
        dt = 0.01
        n_paths = 10000
        exercise_times = np.linspace(1, T_total, n_exercise)  # e.g., yearly

        # Simulate short-rate paths
        paths = model.simulate_paths(T_total, n_paths=n_paths, dt=dt)
        time_grid = np.arange(0, T_total + dt, dt)
        payoff_matrix = np.zeros((n_paths, len(exercise_times)))
        discount_matrix = np.ones((n_paths, len(exercise_times)))

        for i, t in enumerate(exercise_times):
            idx = int(t / dt)
            r_t = paths[idx]
            payoff_matrix[:, i] = np.maximum(r_t - K, 0) * notional * (exercise_times[1] - exercise_times[0])
            # Cumulative discount from t=0 to t
            discount_matrix[:, i] = np.exp(-np.cumsum(paths[:idx], axis=0)[-1] * dt)

        # Longstaff-Schwartz Backward Induction
        value = np.zeros(n_paths)
        exercise_idx = len(exercise_times) - 1
        value[:] = payoff_matrix[:, exercise_idx] * discount_matrix[:, exercise_idx]

        for i in reversed(range(exercise_idx)):
            itm = payoff_matrix[:, i] > 0
            X = payoff_matrix[itm, i]
            Y = value[itm]  # continuation value
            if len(X) == 0:
                continue
            # Regression: continuation value ~ r
            r_i = paths[int(exercise_times[i] / dt), itm]
            A = np.vstack([np.ones_like(r_i), r_i, r_i**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * r_i + coeffs[2] * r_i**2
            exercise_now = X > continuation
            # Update value
            idxs = np.where(itm)[0][exercise_now]
            value[idxs] = X[exercise_now] * discount_matrix[idxs, i]

        bermudan_price = value.mean()
        print(f"\nBermudan Caplet Price (Early exercise): {bermudan_price:.4f}")

        # Plot: Average optimal exercise rate by date
        exercised_avg = np.mean((payoff_matrix * discount_matrix) > 0, axis=0)

        plt.figure()
        plt.bar([f"{t:.1f}" for t in exercise_times], exercised_avg * 100)
        plt.title("Bermudan Caplet: Exercise Frequency by Time")
        plt.xlabel("Exercise Time")
        plt.ylabel("Exercise % of Paths")
        plt.grid()
        plt.show()

    elif product_choice == "bermudan_swaption":
        notional = float(input("Notional: "))
        K = float(input("Strike rate (K): "))
        n_ex_dates = int(input("Number of exercise dates (e.g. 5): "))
        swap_length = float(input("Swap length after exercise (e.g. 3 years): "))
        total_maturity = float(input("Total product maturity (e.g. 10 years): "))
        dt = 0.01
        n_paths = 10000

        exercise_times = np.linspace(1, total_maturity - swap_length, n_ex_dates)
        paths = model.simulate_paths(total_maturity, n_paths=n_paths, dt=dt)
        time_grid = np.arange(0, total_maturity + dt, dt)
        idx_grid = [int(t / dt) for t in exercise_times]

        swaption_values = np.zeros((n_paths, len(idx_grid)))

        for j, idx in enumerate(idx_grid):
            t = exercise_times[j]
            payment_times = np.arange(t + 1, t + swap_length + 1)
            swap_pv = np.zeros(n_paths)
            for pt in payment_times:
                d_idx = int(pt / dt)
                df = np.exp(-np.cumsum(paths[:d_idx], axis=0)[-1] * dt)
                swap_pv += (K - paths[d_idx]) * df  # receiver swaption
            swaption_values[:, j] = notional * swap_pv

        # Backward induction for early exercise
        value = swaption_values[:, -1]
        for j in reversed(range(len(idx_grid) - 1)):
            itm = swaption_values[:, j] > 0
            X = swaption_values[itm, j]
            Y = value[itm]
            if len(X) == 0:
                continue
            r_i = paths[idx_grid[j], itm]
            A = np.vstack([np.ones_like(r_i), r_i, r_i**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * r_i + coeffs[2] * r_i**2
            exercise_now = X > continuation
            idxs = np.where(itm)[0][exercise_now]
            value[idxs] = X[exercise_now]

        swaption_price = np.mean(value)
        print(f"\nBermudan Swaption Price (Receiver): {swaption_price:.4f}")

        # Visualization: % exercised per date
        exercise_flags = (swaption_values > 0).astype(int)
        avg_exercise = exercise_flags.mean(axis=0)

        plt.figure()
        plt.bar([f"{t:.1f}" for t in exercise_times], avg_exercise * 100)
        plt.title("Bermudan Swaption: Exercise Frequency by Date")
        plt.xlabel("Exercise Time")
        plt.ylabel("% of In-the-Money Paths")
        plt.grid()
        plt.show()


    else:
        print("Invalid product.")

if __name__ == "__main__":
    main()
