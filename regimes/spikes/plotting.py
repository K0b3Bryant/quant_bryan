"""Module for generating result plots."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import lightgbm as lgb
import pandas as pd
import logging

from config import TARGET_VARIABLE, OUTPUT_DIR

def plot_confusion_matrix(conf_matrix, output_path: str):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Spike', 'Spike'], yticklabels=['No Spike', 'Spike'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_test: pd.Series, y_pred_proba: list, auc: float, output_path: str):
    """Plots and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"ROC curve saved to {output_path}")

def plot_feature_importance(model: lgb.LGBMClassifier, output_path: str):
    """Plots and saves the feature importance."""
    plt.figure(figsize=(10, 12))
    lgb.plot_importance(model, max_num_features=30, height=0.8)
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Feature importance plot saved to {output_path}")

def plot_backtest(results_df: pd.DataFrame, output_path: str):
    """Plots the backtest results showing price vs. predicted spike probability."""
    logging.info("Plotting backtest results for a sample period...")

    sample_df = results_df.sample(n=672, random_state=42).sort_index() # 7 days * 96 intervals/day

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot LMP on y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('LMP ($/MWh)', color='tab:blue')
    ax1.plot(sample_df.index, sample_df[TARGET_VARIABLE], color='tab:blue', label='Actual LMP', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=200, color='gray', linestyle='--', label='Spike Threshold ($200)')

    # Create a secondary y-axis for spike probability
    ax2 = ax1.twinx()
    ax2.set_ylabel('Predicted Spike Probability', color='tab:red')
    ax2.plot(sample_df.index, sample_df['spike_probability'], color='tab:red', label='Spike Probability')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1)

    # Highlight actual spikes
    actual_spikes = sample_df[sample_df['target_spike'] == 1]
    ax1.scatter(actual_spikes.index, actual_spikes[TARGET_VARIABLE], color='red', s=100,
                edgecolor='black', zorder=5, label='Actual Spike Occurred')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Backtest: Actual LMP vs. Predicted Spike Probability')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Backtest plot saved to {output_path}")
