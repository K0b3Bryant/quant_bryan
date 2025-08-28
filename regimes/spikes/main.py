"""Main script to run the spike prediction pipeline."""

import pandas as pd
import logging

import utils
import data_loader
import feature_engineering
import model
import plotting
from config import OUTPUT_DIR, TARGET_VARIABLE, TEST_SET_SIZE_PERCENT

def main():
    """Executes the full data processing, model training, and evaluation pipeline."""
    utils.setup_logging()
    
    # Setup
    logging.info("--- Starting Spike Prediction Pipeline ---")
    utils.ensure_dir_exists(OUTPUT_DIR)
    
    # Data Loading
    raw_df = data_loader.load_data()
    featured_df = feature_engineering.create_features(raw_df)

    # Time-Series Split
    test_size = int(len(featured_df) * TEST_SET_SIZE_PERCENT)
    train_df = featured_df.iloc[:-test_size]
    test_df = featured_df.iloc[-test_size:]

    y_train = train_df['target_spike']
    X_train = train_df.drop(columns=['target_spike', TARGET_VARIABLE])
    
    y_test = test_df['target_spike']
    X_test = test_df.drop(columns=['target_spike', TARGET_VARIABLE])

    logging.info(f"Data split into training and testing sets:")
    logging.info(f"Train set: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    logging.info(f"Test set:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
    
    # Model Training
    lgbm_model = model.train_model(X_train, y_train)

    # Model Evaluation
    metrics, conf_matrix, y_pred_proba = model.evaluate_model(lgbm_model, X_test, y_test)
    
    print("\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("---------------------------------")

    # Plotting Results
    plotting.plot_confusion_matrix(conf_matrix, OUTPUT_DIR / "confusion_matrix.png")
    plotting.plot_roc_curve(y_test, y_pred_proba, metrics['AUC'], OUTPUT_DIR / "roc_curve.png")
    plotting.plot_feature_importance(lgbm_model, OUTPUT_DIR / "feature_importance.png")
    
    # Backtest plot
    results_df = test_df[[TARGET_VARIABLE, 'target_spike']].copy()
    results_df['spike_probability'] = y_pred_proba
    plotting.plot_backtest(results_df, OUTPUT_DIR / "backtest_sample.png")

    logging.info("--- Pipeline execution finished successfully. Outputs saved to /output folder. ---")

if __name__ == "__main__":
    main()
