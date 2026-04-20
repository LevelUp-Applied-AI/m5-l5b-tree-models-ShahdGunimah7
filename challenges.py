"""
Optional challenges for Module 5 Week B — Trees & Ensembles

This file adds:
1) Tier 1 — Threshold tuning
2) Tier 2 — Permutation importance
3) Tier 3 — Custom voting ensemble

Run with:
    python challenges.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from lab_trees import (
    NUMERIC_FEATURES,
    build_logistic_regression,
    build_random_forest,
    load_and_split,
)
from sklearn.preprocessing import StandardScaler


# -------------------------
# Tier 1 — Threshold tuning
# -------------------------

def run_threshold_sweep(model, X_test, y_test):
    """
    Sweep thresholds from 0.10 to 0.90 in steps of 0.05.
    Compute precision, recall, and F1 for each threshold.

    Returns:
        thresholds, precisions, recalls, f1s, best_f1_threshold, recall_80_threshold
    """
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)

    precisions = []
    recalls = []
    f1s = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    best_f1_idx = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])

    recall_80_threshold = None
    for threshold, recall_val in zip(thresholds, recalls):
        if recall_val >= 0.80:
            recall_80_threshold = float(threshold)
            break

    return (
        thresholds,
        precisions,
        recalls,
        f1s,
        best_f1_threshold,
        recall_80_threshold,
    )


def plot_threshold_sweep(thresholds, precisions, recalls, f1s, output_path):
    """Plot precision, recall, and F1 vs threshold."""
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="o", label="Recall")
    plt.plot(thresholds, f1s, marker="o", label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep (Balanced RF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def threshold_recommendation_note(best_f1_threshold, recall_80_threshold):
    """Return a short plain-English recommendation."""
    lines = []
    lines.append("Tier 1 note:")
    lines.append(
        f"- The threshold with the best F1 is {best_f1_threshold:.2f}."
    )

    if recall_80_threshold is not None:
        lines.append(
            f"- The first threshold that reaches at least 80% recall is {recall_80_threshold:.2f}."
        )
    else:
        lines.append(
            "- No threshold in this sweep reached 80% recall."
        )

    lines.append(
        "- If Petra Telecom can contact only 200 customers per month, I would choose a lower threshold rather than 0.5."
    )
    lines.append(
        "- A lower threshold catches more likely churn customers, which reduces missed churn cases."
    )
    lines.append(
        "- The tradeoff is more false positives, which means some offers go to customers who would not churn."
    )

    return "\n".join(lines)


# ------------------------------
# Tier 2 — Permutation importance
# ------------------------------

def get_mdi_importance(model, feature_names):
    """Return MDI importances as a sorted list of (feature, importance)."""
    pairs = list(zip(feature_names, model.feature_importances_))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def get_permutation_importance(model, X_test, y_test, feature_names, random_state=42):
    """Return permutation importances as a sorted list of (feature, importance)."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
    )
    pairs = list(zip(feature_names, result.importances_mean))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def plot_permutation_vs_mdi(model, X_test, y_test, feature_names, output_path):
    """
    Create a side-by-side bar chart comparing MDI and permutation importance
    for the top 10 features ranked by MDI.
    """
    mdi_vals = model.feature_importances_
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
    )
    perm_vals = perm_result.importances_mean

    top_idx = np.argsort(mdi_vals)[::-1][:10]

    top_features = [feature_names[i] for i in top_idx]
    top_mdi = mdi_vals[top_idx]
    top_perm = perm_vals[top_idx]

    x = np.arange(len(top_features))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, top_mdi, width=width, label="MDI")
    plt.bar(x + width / 2, top_perm, width=width, label="Permutation")
    plt.xticks(x, top_features, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Permutation Importance vs MDI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def permutation_note(mdi_ranked, perm_ranked):
    """Return a short paragraph explaining possible disagreement."""
    mdi_top = [name for name, _ in mdi_ranked[:5]]
    perm_top = [name for name, _ in perm_ranked[:5]]

    return (
        "Tier 2 note:\n"
        f"- Top 5 by MDI: {mdi_top}\n"
        f"- Top 5 by permutation importance: {perm_top}\n"
        "- The two rankings are usually similar, but not always identical.\n"
        "- MDI can favor features that create many possible splits, while permutation importance measures how much model performance drops when a feature is shuffled.\n"
        "- Because of that, permutation importance often gives a more realistic picture of which features matter most on the test set."
    )


# --------------------------------
# Tier 3 — Custom voting ensemble
# --------------------------------

class CustomVotingEnsemble:
    """
    A simple custom ensemble that:
    - accepts a list of fitted classifiers
    - averages predict_proba outputs
    - predicts by majority vote
    - handles class-order alignment using .classes_
    """

    def __init__(self, models):
        self.models = models
        self.classes_ = None

    def fit(self, X=None, y=None):
        """
        Models are already fitted before being passed in.
        This method just stores a common class order.
        """
        if not self.models:
            raise ValueError("CustomVotingEnsemble needs at least one model.")

        self.classes_ = np.array([0, 1])
        return self

    def _aligned_positive_proba(self, model, X):
        """
        Return probability of class 1, even if model.classes_ has a different order.
        """
        probs = model.predict_proba(X)
        class_to_col = {cls: idx for idx, cls in enumerate(model.classes_)}
        if 1 not in class_to_col:
            raise ValueError("Model does not provide probabilities for class 1.")
        return probs[:, class_to_col[1]]

    def predict_proba(self, X_list):
        """
        Average class-1 probabilities across all models.

        X_list should be a list of inputs matching the models, for example:
            [X_for_lr, X_for_dt, X_for_rf]
        """
        if len(X_list) != len(self.models):
            raise ValueError("X_list must have the same length as models.")

        positive_probs = []
        for model, X in zip(self.models, X_list):
            positive_probs.append(self._aligned_positive_proba(model, X))

        avg_positive = np.mean(positive_probs, axis=0)
        return np.column_stack([1 - avg_positive, avg_positive])

    def predict(self, X_list):
        """
        Majority vote based on each model's class prediction.
        """
        if len(X_list) != len(self.models):
            raise ValueError("X_list must have the same length as models.")

        predictions = []
        for model, X in zip(self.models, X_list):
            predictions.append(model.predict(X))

        predictions = np.asarray(predictions)  # shape: (n_models, n_samples)
        vote_sum = predictions.sum(axis=0)
        majority = (vote_sum >= (len(self.models) / 2)).astype(int)
        return majority


def build_balanced_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """
    Separate helper for Tier 3 because the lab's original build_decision_tree
    did not include class_weight.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_ensemble(lr_model, dt_bal_model, rf_bal_model, X_test_scaled, X_test_raw, y_test):
    """
    Evaluate the custom ensemble using:
    - LR on scaled features
    - balanced DT on raw features
    - balanced RF on raw features
    """
    ensemble = CustomVotingEnsemble([lr_model, dt_bal_model, rf_bal_model]).fit()

    X_list = [X_test_scaled, X_test_raw, X_test_raw]

    y_pred = ensemble.predict(X_list)
    y_prob = ensemble.predict_proba(X_list)[:, 1]

    report = classification_report(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)

    return report, pr_auc


# -------------------------
# Main runner for challenges
# -------------------------

def main():
    os.makedirs("results", exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_and_split()

    # Build scaled data for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Core models
    rf_bal = build_random_forest(
        X_train,
        y_train,
        class_weight="balanced",
        random_state=42,
    )
    lr = build_logistic_regression(X_train_scaled, y_train, random_state=42)
    dt_bal = build_balanced_decision_tree(X_train, y_train, max_depth=5, random_state=42)

    print("\n========== Tier 1 — Threshold tuning ==========")
    (
        thresholds,
        precisions,
        recalls,
        f1s,
        best_f1_threshold,
        recall_80_threshold,
    ) = run_threshold_sweep(rf_bal, X_test, y_test)

    plot_threshold_sweep(
        thresholds,
        precisions,
        recalls,
        f1s,
        "results/threshold_sweep.png",
    )

    print(f"Best F1 threshold: {best_f1_threshold:.2f}")
    if recall_80_threshold is not None:
        print(f"First threshold with recall >= 0.80: {recall_80_threshold:.2f}")
    else:
        print("No threshold in the sweep reached recall >= 0.80.")

    print()
    print(threshold_recommendation_note(best_f1_threshold, recall_80_threshold))

    print("\n========== Tier 2 — Permutation importance ==========")
    mdi_ranked = get_mdi_importance(rf_bal, NUMERIC_FEATURES)
    perm_ranked = get_permutation_importance(rf_bal, X_test, y_test, NUMERIC_FEATURES)

    print("Top 10 MDI importance:")
    for name, value in mdi_ranked[:10]:
        print(f"  {name:<20s} {value:.4f}")

    print("\nTop 10 permutation importance:")
    for name, value in perm_ranked[:10]:
        print(f"  {name:<20s} {value:.4f}")

    plot_permutation_vs_mdi(
        rf_bal,
        X_test,
        y_test,
        NUMERIC_FEATURES,
        "results/permutation_vs_mdi.png",
    )

    print()
    print(permutation_note(mdi_ranked, perm_ranked))

    print("\n========== Tier 3 — Custom voting ensemble ==========")
    report, pr_auc = evaluate_ensemble(
        lr,
        dt_bal,
        rf_bal,
        X_test_scaled,
        X_test,
        y_test,
    )

    print("Ensemble classification report:")
    print(report)
    print(f"Ensemble PR-AUC: {pr_auc:.3f}")

    print("\nTier 3 note:")
    print("- The ensemble averages information from logistic regression, a balanced decision tree, and a balanced random forest.")
    print("- It is most useful when the models make different kinds of mistakes.")
    print("- If the models are too similar, the ensemble may not improve much over the best single model.")


if __name__ == "__main__":
    main()