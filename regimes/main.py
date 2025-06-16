# Main script: main.py
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from utils import load_time_series, categorize_states, plot_results, extract_transition_probabilities
from config import CONFIG

def main():
    # Load the time series data
    data = load_time_series(CONFIG["time_series_path"])

    # Define the Markov Switching Regression model with Hierarchical Priors
    n = len(data)
    with pm.Model() as model:
        # Transition probabilities (logit parameterization)
        # The prior assumes a normal distribution centered at 0 with standard deviation 2.
        # This reflects a neutral assumption, allowing flexibility in transition probabilities
        # while not overly constraining the model to specific values.
        logit_p_00 = pm.Normal("logit_p_00", mu=0, sigma=2)
        logit_p_11 = pm.Normal("logit_p_11", mu=0, sigma=2)

        p_00 = pm.Deterministic("p_00", pm.math.invlogit(logit_p_00))
        p_11 = pm.Deterministic("p_11", pm.math.invlogit(logit_p_11))
        p_01 = 1 - p_00
        p_10 = 1 - p_11

        # Initial state probabilities
        # The prior assumes a uniform Dirichlet distribution, reflecting equal uncertainty
        # about the probability of starting in either state.
        pi_0 = pm.Dirichlet("pi_0", a=np.ones(2))

        # Hierarchical Priors for Emission Model
        # Hyperpriors for state means
        mean_mu = pm.Normal("mean_mu", mu=0, sigma=10)
        mean_sigma = pm.Exponential("mean_sigma", lam=1)

        # State-specific means drawn from the hyperprior
        state_means = pm.Normal("state_means", mu=mean_mu, sigma=mean_sigma, shape=2)

        # Hyperpriors for state standard deviations
        std_alpha = pm.Gamma("std_alpha", alpha=2, beta=0.5)
        std_beta = pm.Exponential("std_beta", lam=1)

        # State-specific standard deviations drawn from the hyperprior
        state_stds = pm.InverseGamma("state_stds", alpha=std_alpha, beta=std_beta, shape=2)

        # Hidden states (categorical)
        states = pm.Categorical("states", p=pi_0, shape=n)

        # Observations
        likelihood = pm.Normal(
            "y_obs",
            mu=pm.math.switch(states, state_means[1], state_means[0]),
            sigma=pm.math.switch(states, state_stds[1], state_stds[0]),
            observed=data,
        )

        # Sampling
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.95)

    # Extract transition probabilities
    transition_probs = extract_transition_probabilities(trace)
    print("Transition Probabilities:", transition_probs)

    # Categorize states and plot results
    predicted_states = categorize_states(trace, n)
    plot_results(data, predicted_states)

if __name__ == "__main__":
    main()
