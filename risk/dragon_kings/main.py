import numpy as np
import pandas as pd
from lppl import fit_lppl
from anomaly import detect_anomalies, create_lstm_model
from utils import load_data, plot_results
from config import DATA_PATH, OUTPUT_PATH, PLOT_RESULTS, LSTM_PARAMS

def main():
    prices = load_data(DATA_PATH)
    bubble_probabilities = np.zeros(len(prices))
    model = create_lstm_model()
    
    for i in range(50, len(prices)):
        window_prices = prices.iloc[:i]
        lppl_params = fit_lppl(window_prices.values.flatten())
        anomalies = detect_anomalies(window_prices)
        
        if lppl_params is not None:
            bubble_probabilities[i] = anomalies.iloc[-1] * 0.5 + 0.5
    
    results = pd.DataFrame({"Date": prices.index, "Bubble_Probability": bubble_probabilities})
    results.to_csv(OUTPUT_PATH, index=False)
    
    if PLOT_RESULTS:
        plot_results(prices, bubble_probabilities)

if __name__ == "__main__":
    main()
