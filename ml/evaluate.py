"""
ml/evaluate.py - Standalone evaluation script on held-out test set
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "model", "chd_pipeline.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "model", "model_metadata.json")
SPLIT_DIR = os.path.join(os.path.dirname(__file__), "split")

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Model or metadata not found. Run ml/train.py first.")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    threshold = meta.get("decision_threshold", 0.12)
    pipeline = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(os.path.join(SPLIT_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(SPLIT_DIR, "y_test.csv")).values.ravel()

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("=" * 60)
    print("      FRAMINGHAM CHD TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Model: {meta.get('model_name')}")
    print(f"Decision Threshold: {threshold:.3f}")
    print(f"Test Set Size: {len(y_test)} (Positives: {y_test.sum()})")
    print("-" * 60)
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print(f"Brier Score:       {brier:.4f}")
    print(f"Sensitivity/Recall:{rec:.4f}")
    print(f"Precision/PPV:     {prec:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"F2 Score:          {f2:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f" TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
    print(f" FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["No CHD", "CHD"]))
    print("=" * 60)

if __name__ == "__main__":
    main()

