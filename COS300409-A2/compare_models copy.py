"""
compare_models.py
=================
Models beyond the unit syllabus, compared against our baseline.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python compare_models.py features.csv

Requires: pip install lightgbm

What this is for
----------------
The assignment asks for at least two models not taught in the unit, implemented
and compared against the baseline on the same test set, "with clear reasoning
for why it suits this data and what it does differently". It also notes that an
unusual model with no justification scores no better than a standard one. The
reasoning below is therefore part of the deliverable, not commentary on it.

The unit covered logistic regression, k-nearest neighbours, decision trees,
random forests, k-means and DBSCAN. Our baseline is a random forest at 0.954
recall. The three models here were chosen to be genuinely different from it,
not merely different in name.

LightGBM - gradient boosting
    A random forest grows many trees independently on bootstrap samples and
    averages them. Each tree sees the whole problem and the errors are reduced
    by averaging out variance. Boosting works the other way: each tree is
    trained specifically on what the previous trees got wrong.

    That difference matters for this data. Our error analysis showed failures
    concentrated in one place - phishing addresses are 91% of all misses, and
    the clustering showed why, because phishing must look ordinary to a human
    reader in a way that defacement and malware need not. A method that
    repeatedly revisits its own mistakes is aimed directly at a concentrated
    error of that kind, where averaging independent trees is not.

    LightGBM in particular grows trees leaf-wise rather than level-wise,
    splitting wherever the loss reduction is largest instead of expanding every
    node at a depth. On a problem where the hard cases are a minority of the
    data, that spends capacity where the difficulty is.

Linear SVM - maximum margin
    Included as a genuine contrast rather than a likely winner. Both our
    baseline and LightGBM are tree ensembles making axis-aligned splits; a
    support vector machine draws a single boundary and places it to maximise
    the distance to the nearest points of each class.

    The expectation is that it will do worse, because our features interact in
    ways a linear boundary cannot express - a long address is unremarkable on a
    popular domain and suspicious on an unranked one. Confirming that expecta-
    tion with a number is a result: it tells us the problem genuinely needs
    non-linear structure rather than assuming it does.

Gaussian Naive Bayes - the independence assumption
    Chosen because we can predict its failure and explain the mechanism. Naive
    Bayes assumes every feature is conditionally independent given the class.
    Ours plainly are not: url_length, path_length and path_depth measure
    overlapping things, and entropy_url moves with several others. Where
    features are correlated, Naive Bayes counts the same evidence repeatedly
    and becomes overconfident.

    It costs seconds to run and sets a floor, which makes the gap to the tree
    models interpretable rather than merely favourable.

On reading the comparison
-------------------------
Five-fold cross-validation on the baseline gave a recall standard deviation of
0.0006. Two standard deviations, about 0.0012, is used as the threshold before
a difference is called real. Any smaller gap is reported as a tie, because that
is what it is.
"""

from __future__ import annotations

import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
TEST_SIZE = 0.2
NON_FEATURES = ["url", "type", "label"]

# From the baseline's five-fold cross-validation. See the module docstring.
CV_SD = 0.0006
NOISE_FLOOR = 2 * CV_SD


def pick_booster():
    """Return a gradient boosting implementation that will actually run here.

    LightGBM is preferred, but its macOS wheel links against Apple's OpenMP
    runtime, which is not bundled and is not present on a default system. The
    import then fails at load time with an OSError rather than an ImportError,
    so both are caught.

    The fallback is scikit-learn's HistGradientBoostingClassifier. This is not a
    compromise for our purposes: it is histogram-based gradient boosting of the
    same family, documented as inspired by LightGBM, and it is no more part of
    the unit syllabus than LightGBM is. The argument for boosting on this data -
    that each tree is trained on what the previous ones got wrong, which suits a
    concentrated error - applies to either implementation.

    Whichever is used must be named in the report.
    """
    try:
        import lightgbm as lgb

        def make(_):
            return lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.1, num_leaves=63,
                is_unbalance=True, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1)

        print("Gradient booster: LightGBM\n")
        return "LightGBM", make

    except (ImportError, OSError) as err:
        print("LightGBM could not be loaded:")
        print(f"  {str(err).splitlines()[0]}")
        print("\nOn macOS this is usually the missing OpenMP runtime. Installing")
        print("it with 'brew install libomp' and re-running will use LightGBM.")
        print("\nFalling back to scikit-learn's HistGradientBoostingClassifier,")
        print("which is histogram-based gradient boosting of the same family and")
        print("equally outside the unit syllabus.\n")

        def make(class_weight):
            return HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                class_weight=class_weight, random_state=RANDOM_STATE)

        return "HistGradientBoosting", make


def booster_importance(name, model, X_test, y_test):
    """Feature importance for whichever booster was used.

    LightGBM exposes feature_importances_ directly as split counts.
    HistGradientBoostingClassifier does not expose any, so permutation
    importance is computed instead: each feature is shuffled in turn and the
    drop in score measured. That is slower, so it runs on a sample, and it
    measures something slightly different - how much the fitted model depends
    on a feature, rather than how often it was split on. Both answer the
    question we are asking.
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=X_test.columns), "split count"

    n = min(20_000, len(X_test))
    sample = X_test.sample(n, random_state=RANDOM_STATE)
    print(f"\n  Computing permutation importance on {n:,} rows...")
    r = permutation_importance(model, sample, y_test.loc[sample.index],
                               n_repeats=3, random_state=RANDOM_STATE,
                               scoring="recall", n_jobs=-1)
    return pd.Series(r.importances_mean, index=X_test.columns), "permutation (recall drop)"


def evaluate(name, model, X_test, y_test, elapsed, results):
    pred = model.predict(X_test)
    rec = recall_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except (AttributeError, ValueError):
        # LinearSVC has no predict_proba. decision_function orders the points
        # the same way, which is all ROC AUC needs.
        try:
            auc = roc_auc_score(y_test, model.decision_function(X_test))
        except (AttributeError, ValueError):
            auc = float("nan")

    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())

    print(f"  {name:<22} recall {rec:.4f}  precision {prec:.4f}  F1 {f1:.4f}"
          f"  AUC {auc:.4f}  [{elapsed:.0f}s]")

    results.append({"model": name, "recall": rec, "precision": prec, "f1": f1,
                    "roc_auc": auc, "missed": fn, "false_alarms": fp,
                    "train_seconds": round(elapsed, 1)})
    return pred


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "features.csv"

    booster_name, make_booster = pick_booster()

    f = pd.read_csv(path)
    X = f.drop(columns=[c for c in NON_FEATURES if c in f.columns])
    y = f["label"]

    # The same seed and the same stratification as train_baseline.py, so every
    # model here is measured on exactly the rows the baseline was measured on.
    # A comparison across different test sets would mean nothing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Train {len(X_train):,} | Test {len(X_test):,} | {X.shape[1]} features")
    print("Same split as train_baseline.py, so the comparison is like for like.\n")

    results, preds = [], {}
    print("=" * 78)
    print("Held-out test set")
    print("=" * 78)

    # --- baseline, retrained here so the comparison is self-contained -----
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight="balanced").fit(X_train, y_train)
    preds["Random forest"] = evaluate("Random forest (base)", rf, X_test, y_test,
                                      time.time() - t, results)

    # --- LightGBM ---------------------------------------------------------
    # is_unbalance plays the same role class_weight='balanced' plays for the
    # forest: it reweights the objective rather than moving the threshold
    # afterwards. Using the same lever on both keeps the comparison fair.
    t = time.time()
    gbm = make_booster("balanced")
    gbm.fit(X_train, y_train)
    preds[booster_name] = evaluate(booster_name, gbm, X_test, y_test,
                                   time.time() - t, results)

    # --- Linear SVM -------------------------------------------------------
    # Scaled inside a Pipeline so the scaler is fitted on training data only.
    # LinearSVC rather than SVC: the kernel version is quadratic in the number
    # of samples and will not finish on half a million rows.
    t = time.time()
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", dual="auto",
                          max_iter=3000, random_state=RANDOM_STATE)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # convergence warnings are expected
        svm.fit(X_train, y_train)
    preds["Linear SVM"] = evaluate("Linear SVM", svm, X_test, y_test,
                                   time.time() - t, results)

    # --- Naive Bayes ------------------------------------------------------
    t = time.time()
    nb = Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
    nb.fit(X_train, y_train)
    preds["Naive Bayes"] = evaluate("Naive Bayes", nb, X_test, y_test,
                                    time.time() - t, results)

    table = pd.DataFrame(results).set_index("model")
    table.to_csv("results_models.csv")

    # --- is the difference real? -----------------------------------------
    print(f"\n{'=' * 78}\nAgainst the baseline\n{'=' * 78}")
    base = table.loc["Random forest (base)", "recall"]
    for name in table.index:
        if name.startswith("Random forest"):
            continue
        d = table.loc[name, "recall"] - base
        verdict = ("a tie - within the noise floor" if abs(d) <= NOISE_FLOOR
                   else ("better" if d > 0 else "worse"))
        print(f"  {name:<22} {d:+.4f}   {verdict}")
    print(f"\n  Noise floor is {NOISE_FLOOR:.4f}, two standard deviations of the")
    print("  baseline's five-fold cross-validated recall. Differences smaller")
    print("  than this cannot be told apart from fold-to-fold variation.")

    # --- cross-validate the winner ---------------------------------------
    best = table.drop(index=[i for i in table.index if i.startswith("Random forest")])
    best_name = best["recall"].idxmax()
    print(f"\nCross-validating {best_name} (5-fold, recall)...")
    model_for_cv = {booster_name: gbm, "Linear SVM": svm, "Naive Bayes": nb}[best_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model_for_cv, X, y, cv=cv, scoring="recall", n_jobs=-1)
    print(f"  Per fold: {[f'{s:.4f}' for s in scores]}")
    print(f"  Mean {scores.mean():.4f}, standard deviation {scores.std():.4f}")
    print(f"  Quote as {scores.mean():.3f} +/- {scores.std():.3f}")

    # --- do the models fail on the same addresses? ------------------------
    # If two models of different design miss the same rows, those rows are hard
    # in the data rather than hard for one algorithm, and no change of model
    # will help. That is a more useful finding than a small difference in
    # recall, and it points the error analysis at the right question.
    print(f"\n{'=' * 78}\nDo the models fail on the same addresses?\n{'=' * 78}")
    rf_miss = (preds["Random forest"] == 0) & (y_test.values == 1)
    gbm_miss = (preds[booster_name] == 0) & (y_test.values == 1)
    both = rf_miss & gbm_miss
    print(f"  Random forest misses : {rf_miss.sum():,}")
    print(f"  {booster_name + ' misses':<21}: {gbm_miss.sum():,}")
    print(f"  Missed by both       : {both.sum():,} "
          f"({both.sum() / max(rf_miss.sum(), 1):.1%} of the forest's misses)")
    print(f"  Caught by {booster_name} only: {(rf_miss & ~gbm_miss).sum():,}")
    print(f"  Caught by forest only  : {(gbm_miss & ~rf_miss).sum():,}")

    if both.sum() / max(rf_miss.sum(), 1) > 0.7:
        print("\n  The overlap is high, so these addresses are hard in the data")
        print("  rather than hard for one algorithm. Changing model will not")
        print("  recover them; better features or more data might.")

    hard = f.loc[X_test.index[both]]
    hard.to_csv("errors_missed_by_both.csv", index=False)
    print(f"  Written to errors_missed_by_both.csv")
    if "type" in hard.columns and len(hard):
        print("\n  What both models miss, by class:")
        print("    " + hard["type"].value_counts().to_string().replace("\n", "\n    "))

    # --- what LightGBM relies on -----------------------------------------
    imp, imp_kind = booster_importance(booster_name, gbm, X_test, y_test)
    imp = imp.sort_values(ascending=False)
    rf_rank = pd.Series(rf.feature_importances_, index=X.columns).rank(ascending=False)
    print(f"\n{'=' * 78}\nTop features, both models\n{'=' * 78}")
    print(f"  {'feature':<26}{booster_name[:12]:>14}{'forest rank':>14}")
    for n in imp.head(10).index:
        print(f"  {n:<26}{imp[n]:>14.4f}{int(rf_rank[n]):>14}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    imp.head(15).sort_values().plot.barh(ax=ax, color="#1baf7a")
    ax.set_xlabel(f"{booster_name} importance ({imp_kind})")
    ax.set_title(f"What {booster_name} relies on")
    fig.tight_layout(); fig.savefig("fig_booster_importance.png", dpi=150); plt.close(fig)

    joblib.dump({"model": gbm, "feature_names": list(X.columns)},
                "linkguard_booster.joblib")
    print("\nWrote results_models.csv, errors_missed_by_both.csv,")
    print("     fig_booster_importance.png, linkguard_booster.joblib")
    print("\nlinkguard_model.joblib is untouched; the clustering and error")
    print("analysis built on it remain valid.")


if __name__ == "__main__":
    main()