"""
Module 5 Week B — Applied Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the Petra
Telecom churn dataset. Handle class imbalance honestly (class_weight as an
operating-point tool at a fixed threshold), evaluate with PR-AUC and
calibration, and demonstrate what tree models capture that linear models
cannot.

Complete the 12 functions below. See the lab guide for task-by-task detail.
Run with:  python lab_trees.py
Tests:     pytest tests/ -v
"""

import os

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-count (quantile) binning.

    Sort samples by predicted probability, split into `n_bins` equal-size
    chunks, and sum the bin-weighted absolute difference between each bin's
    mean predicted probability and its fraction of true positives.

    A perfectly calibrated model has ECE = 0. Higher ECE means predicted
    probabilities don't correspond to empirical rates.

    Args:
        y_true: 1D array-like of true binary labels (0 or 1).
        y_prob: 1D array-like of predicted probabilities for class 1.
        n_bins: Number of equal-count bins.

    Returns:
        ECE as a float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n = len(y_true_sorted)
    bins = np.array_split(np.arange(n), n_bins)

    ece = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue

        mean_prob = y_prob_sorted[bin_idx].mean()
        frac_positive = y_true_sorted[bin_idx].mean()
        bin_weight = len(bin_idx) / n
        ece += bin_weight * abs(mean_prob - frac_positive)

    return ece


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Teaches that pure-leaf trees (unbounded depth) produce extreme
    probabilities → poor calibration; depth-constrained trees smooth
    probabilities → better calibration.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    dt_unbounded = build_decision_tree(X_train, y_train, max_depth=None)
    dt_depth_5 = build_decision_tree(X_train, y_train, max_depth=5)

    prob_unbounded = dt_unbounded.predict_proba(X_test)[:, 1]
    prob_depth_5 = dt_depth_5.predict_proba(X_test)[:, 1]

    ece_unbounded = compute_ece(y_test, prob_unbounded)
    ece_depth_5 = compute_ece(y_test, prob_depth_5)

    return {
        "ece_unbounded": ece_unbounded,
        "ece_depth_5": ece_depth_5,
    }


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=10,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    pairs = zip(feature_names, model.feature_importances_)
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return dict(sorted_pairs)


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice — which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred, zero_division=0)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Threshold-independent: measures the model's ability to rank positives
    above negatives across all thresholds. Unlike recall at a specific
    threshold, PR-AUC does not change when you apply class_weight='balanced'
    in a way that merely shifts predicted probabilities uniformly — the
    ranking is what matters.

    Returns:
        Float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    PrecisionRecallDisplay.from_estimator(
        rf_default,
        X_test,
        y_test,
        ax=ax,
        name="RF default",
    )
    PrecisionRecallDisplay.from_estimator(
        rf_balanced,
        X_test,
        y_test,
        ax=ax,
        name="RF balanced",
    )

    plt.title("Precision-Recall Curves")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    CalibrationDisplay.from_estimator(
        rf_default,
        X_test,
        y_test,
        n_bins=10,
        ax=ax,
        name="RF default",
    )
    CalibrationDisplay.from_estimator(
        rf_balanced,
        X_test,
        y_test,
        n_bins=10,
        ax=ax,
        name="RF balanced",
    )

    plt.title("Calibration Curves")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)
    return model


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test_raw,
                                     X_test_scaled, y_test, feature_names,
                                     min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    The tree-vs-linear capability demonstration. The random forest can
    capture feature interactions, non-monotonic relationships, and threshold
    effects that a linear model cannot express with per-feature coefficients.
    Finding a sample where the two models disagree — and explaining WHY in
    structural terms — is the lab's evidence that trees have capabilities
    linear models don't, regardless of aggregate PR-AUC.

    Args:
        rf_model: Trained RF (takes raw features).
        lr_model: Trained LR (takes scaled features).
        X_test_raw: Unscaled test features (what RF consumes).
        X_test_scaled: Scaled test features (what LR consumes).
        y_test: True labels for the test set.
        feature_names: List of feature name strings.
        min_diff: Minimum probability difference to count as disagreement.

    Returns:
        Dict with keys:
          - sample_idx (int): test-set row index of the selected sample
          - feature_values (dict): {name: value} for the sample's features
          - rf_proba (float): RF's predicted P(churn=1)
          - lr_proba (float): LR's predicted P(churn=1)
          - prob_diff (float): |rf_proba - lr_proba|
          - true_label (int): 0 or 1
    """
    rf_probs = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    prob_diffs = np.abs(rf_probs - lr_probs)
    best_pos = int(np.argmax(prob_diffs))
    best_diff = prob_diffs[best_pos]

    if best_diff < min_diff:
        return None

    sample_idx = X_test_raw.index[best_pos] if hasattr(X_test_raw, "index") else best_pos

    if hasattr(X_test_raw, "iloc"):
        row = X_test_raw.iloc[best_pos]
        feature_values = {name: row[name] for name in feature_names}
    else:
        row = X_test_raw[best_pos]
        feature_values = {name: row[i] for i, name in enumerate(feature_names)}

    true_label = y_test.iloc[best_pos] if hasattr(y_test, "iloc") else y_test[best_pos]

    return {
        "sample_idx": int(sample_idx),
        "feature_values": feature_values,
        "rf_proba": float(rf_probs[best_pos]),
        "lr_proba": float(lr_probs[best_pos]),
        "prob_diff": float(best_diff),
        "true_label": int(true_label),
    }


def main():
    os.makedirs("results", exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # Decision Tree
    dt = build_decision_tree(X_train, y_train)
    print("\n--- Decision Tree (max_depth=5) ---")
    print(classification_report(y_test, dt.predict(X_test), zero_division=0))

    plt.figure(figsize=(14, 8))
    plot_tree(dt, feature_names=NUMERIC_FEATURES, max_depth=3, filled=True)
    plt.savefig("results/decision_tree.png")
    plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
    print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Random Forest
    rf = build_random_forest(X_train, y_train)

    print("\n--- Classification report: RF default ---")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))

    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")

    print("\n--- Classification report: RF balanced ---")
    print(classification_report(y_test, rf_bal.predict(X_test), zero_division=0))

    # Recall + PR-AUC
    r_def = evaluate_recall_at_threshold(rf, X_test, y_test)
    r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test)

    print(f"\nRF default recall@0.5: {r_def:.3f}")
    print(f"RF balanced recall@0.5: {r_bal:.3f}")

    auc_def = compute_pr_auc(rf, X_test, y_test)
    auc_bal = compute_pr_auc(rf_bal, X_test, y_test)

    print(f"\nRF default PR-AUC: {auc_def:.3f}")
    print(f"RF balanced PR-AUC: {auc_bal:.3f}")

    # Plots
    plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
    plot_calibration_curves(rf, rf_bal, X_test, y_test, "results/calibration_curves.png")


if __name__ == "__main__":
    main()