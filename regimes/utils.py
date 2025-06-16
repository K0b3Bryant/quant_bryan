import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az

def load_time_series(file_path):
    """Load time series data from a CSV file."""
    return pd.read_csv(file_path).iloc[:, 0].values

def categorize_states(trace, n):
    """Extract and categorize hidden states from the trace."""
    posterior_states = trace.posterior["states"].mean(axis=(0, 1))
    return np.round(posterior_states)

def extract_transition_probabilities(trace):
    """Extract transition probabilities from the trace."""
    p_00 = trace.posterior["p_00"].mean().item()
    p_11 = trace.posterior["p_11"].mean().item()
    p_01 = 1 - p_00
    p_10 = 1 - p_11
    return {
        "p_00": p_00,
        "p_01": p_01,
        "p_10": p_10,
        "p_11": p_11
    }

def plot_results(data, predicted_states):
    """Plot the observed data and predicted states."""
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Observed Data", color="black")
    plt.plot(predicted_states, label="Predicted States", linestyle="--", color="red")
    plt.legend()
    plt.title("Markov Switching Regression: Predicted States")
    plt.show()
