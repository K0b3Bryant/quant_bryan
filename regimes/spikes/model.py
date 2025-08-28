"""Module for model training and evaluation."""

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import logging

from config import LGBM_PARAMS

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a LightGBM classification model.
    """
    logging.info("Training LightGBM model...")
    
    # Handle class imbalance - tells the model to pay more attention to the minority class (spikes)
    neg, pos = y_train.value_counts().sort_index()
    scale_pos_weight = neg / pos
    logging.info(f"Class Imbalance: Negative={neg}, Positive={pos}. Using scale_pos_weight={scale_pos_weight:.2f}")

    model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LGBM_PARAMS)
    
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the model on the test set and returns performance metrics.
    """
    logging.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred_proba)
    }
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info("Model evaluation complete.")
    return metrics, conf_matrix, y_pred_proba
