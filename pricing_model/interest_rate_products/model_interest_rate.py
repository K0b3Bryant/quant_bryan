import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Base class for short rate models
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


# Main function
def main():
    model_choice = input("Choose model [vasicek / cir / hull-white]: ").strip().lower()
    product_choice = input("Choose product [zero_coupon]: ").strip().lower()

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

    else:
        print("Invalid product.")

if __name__ == "__main__":
    main()
