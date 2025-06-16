import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

def plot_results(prices, bubble_probabilities):
    plt.figure(figsize=(12,6))
    plt.plot(prices, label='Price')
    plt.plot(bubble_probabilities * max(prices), label='Bubble Probability', linestyle='dashed')
    plt.legend()
    plt.show()
