"""
train_baseline.py
=================
Baseline classification for the LinkGuard phishing URL detector.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

Reads the feature table produced by url_features.py, trains three models, and
reports the metrics that matter for this topic. Saves the best pipeline for the
Assignment 3 web application and writes the charts the report needs.

    python train_baseline.py features.csv

Why recall
----------
The Week 7 lecture sets out three questions for choosing a metric: who pays for
each kind of mistake, whether a human reviews the flags, and how rare the
positive class is. For phishing URL detection the first question settles it. A
missed phishing address costs the user a password; a false alarm costs them one
extra click, and because our warning is advisory rather than blocking they can
still proceed. The two errors are not comparable, so we favour recall on the
malicious class and report F1 to keep precision honest.

This is the same conclusion recorded as NFR-05 in the Assignment 1 plan, now
with the lecture's reasoning behind it.
"""

from __future__ import annotations

import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")          # write files, do not open windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42          # fixed so every run is reproducible
TEST_SIZE = 0.2

# Columns that are not features. url and type are carried through the feature
# table for traceability during error analysis; label is the target.
NON_FEATURES = ["url", "type", "label"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(path: str):
    """Load the feature table and split it into X, y and the traceability columns.

    Returns the feature matrix, the binary target, and the original rows so that
    individual predictions can be traced back to their address and original
    four-class label during error analysis.
    """
    df = pd.read_csv(path)

    missing = [c for c in ("url", "label") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}. Re-run url_features.py.")

    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
    y = df["label"]

    print(f"Loaded {len(df):,} rows and {X.shape[1]} features")
    print(f"Malicious: {y.mean():.1%}  Benign: {1 - y.mean():.1%}")
    return X, y, df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, model, X_test, y_test, results):
    """Score one fitted model and record the result.

    Accuracy is printed alongside the others deliberately. On a dataset that is
    two thirds benign it is the number most likely to mislead, and showing it
    next to recall makes that visible rather than hiding it.
    """
    pred = model.predict(X_test)

    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    acc = (pred == y_test).mean()

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except (AttributeError, ValueError):
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  Accuracy   {acc:.4f}   <- inflated by the benign majority")
    print(f"  Precision  {prec:.4f}")
    print(f"  Recall     {rec:.4f}   <- the metric we optimise for")
    print(f"  F1         {f1:.4f}")
    print(f"  ROC AUC    {auc:.4f}")
    print(f"\n  Missed malicious (false negatives): {fn:,}")
    print(f"  False alarms     (false positives): {fp:,}")

    results.append({"model": name, "accuracy": acc, "precision": prec,
                    "recall": rec, "f1": f1, "roc_auc": auc,
                    "false_negatives": fn, "false_positives": fp})
    return pred


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_confusion(y_test, pred, name, filename):
    """Confusion matrix as a figure for the report."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=["benign", "malicious"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format=",")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_importance(model, feature_names, filename, top=20):
    """Feature importance for the tree model.

    This chart is what tells us which of the features from the research brief
    actually earn their place. Features that rank near zero are candidates to
    report as tested and discarded, which is stronger evidence of research than
    keeping everything.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top)

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(importances) + 1))
    importances.sort_values().plot.barh(ax=ax, color="#2a78d6")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top} features")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print("\nTop 10 features by importance:")
    for n, v in importances.head(10).items():
        print(f"  {v:.4f}  {n}")

    low = pd.Series(model.feature_importances_, index=feature_names)
    low = low.sort_values().head(5)
    print("\nLowest 5 (candidates to test and report as discarded):")
    for n, v in low.items():
        print(f"  {v:.4f}  {n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "features.csv"

    X, y, df = load(path)

    # Stratified so the class balance in the test set matches the whole dataset.
    # Without this the test proportion drifts and the recall figure stops being
    # comparable between runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain {len(X_train):,} | Test {len(X_test):,} "
          f"(malicious {y_test.mean():.1%} in test, {y_train.mean():.1%} in train)")

    results = []

    # --- 0. The accuracy trap, stated explicitly -------------------------
    # A model that predicts "benign" every time. Its accuracy is the benign
    # share of the data. Including it means no reader can mistake a high
    # accuracy for a working detector.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate("0. Always predict benign (baseline for comparison)",
             dummy, X_test, y_test, results)

    # --- 1. Logistic regression ------------------------------------------
    # Wrapped in a Pipeline so scaling is fitted on the training fold only.
    # Scaling outside the pipeline leaks test statistics into training.
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    t = time.time()
    logreg.fit(X_train, y_train)
    print(f"\n[logistic regression trained in {time.time() - t:.1f}s]")
    evaluate("1. Logistic regression", logreg, X_test, y_test, results)

    # --- 2. Logistic regression with balanced class weights ---------------
    # class_weight changes what the model optimises during training, which is a
    # different lever from moving the decision threshold afterwards. Whichever
    # is used must be named in the report; using one silently is the mistake.
    logreg_bal = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    logreg_bal.fit(X_train, y_train)
    evaluate("2. Logistic regression, class_weight='balanced'",
             logreg_bal, X_test, y_test, results)

    # --- 3. Random forest -------------------------------------------------
    # No scaler: trees split on thresholds and are unaffected by feature scale.
    # n_jobs=-1 uses all cores, which matters on 500,000 training rows.
    forest = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced",
    )
    t = time.time()
    forest.fit(X_train, y_train)
    print(f"\n[random forest trained in {time.time() - t:.1f}s]")
    pred_forest = evaluate("3. Random forest, class_weight='balanced'",
                           forest, X_test, y_test, results)

    # --- comparison -------------------------------------------------------
    table = pd.DataFrame(results).set_index("model")
    print(f"\n{'=' * 62}\nComparison\n{'=' * 62}")
    print(table[["accuracy", "precision", "recall", "f1"]].round(4).to_string())
    table.to_csv("results_baseline.csv")

    # --- cross-validation on the winner -----------------------------------
    # A single split gives one number with no sense of its spread. Any later
    # comparison smaller than this standard deviation is not a real difference.
    print("\nCross-validating the random forest (5-fold, recall)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(forest, X, y, cv=cv, scoring="recall", n_jobs=-1)
    print(f"  Recall per fold: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean {scores.mean():.4f}, standard deviation {scores.std():.4f}")
    print(f"  Quote this as {scores.mean():.3f} +/- {scores.std():.3f}")

    # --- charts and model --------------------------------------------------
    plot_confusion(y_test, pred_forest, "Random forest", "fig_confusion.png")
    plot_importance(forest, list(X.columns), "fig_importance.png")

    # Save the whole fitted object, not just the model. For the forest there is
    # no scaler, but saving through the same interface keeps Assignment 3's
    # loading code identical whichever model we end up shipping.
    joblib.dump({"model": forest, "feature_names": list(X.columns)},
                "linkguard_model.joblib")

    print("\nWrote: results_baseline.csv, fig_confusion.png, "
          "fig_importance.png, linkguard_model.joblib")

    # --- error analysis starting point -------------------------------------
    # The rubric asks for false positives and false negatives inspected by hand.
    # This writes them out so that inspection has something to work from.
    test_rows = df.loc[X_test.index].copy()
    test_rows["predicted"] = pred_forest
    missed = test_rows[(test_rows["label"] == 1) & (test_rows["predicted"] == 0)]
    alarms = test_rows[(test_rows["label"] == 0) & (test_rows["predicted"] == 1)]
    missed.to_csv("errors_missed_malicious.csv", index=False)
    alarms.to_csv("errors_false_alarms.csv", index=False)
    print(f"Wrote {len(missed):,} missed and {len(alarms):,} false alarms "
          "for error analysis")

    if "type" in missed.columns and len(missed):
        print("\nWhat we miss, by original class:")
        print(missed["type"].value_counts().to_string())


if __name__ == "__main__":
    main()