"""
fix_scheme_leak.py
==================
Measuring what the is_https artefact was worth.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python fix_scheme_leak.py features.csv --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv --reference tranco.csv

What this tests
---------------
Comparing the two benign populations showed is_https separating them by 15.3
standard deviations: 0.5% of our benign addresses carry an explicit https://
against 100% of PhiUSIIL's. Checking our own data then showed the feature
pointing the wrong way entirely - malicious addresses carry an explicit scheme
fourteen times more often than benign ones, 6.4% against 0.46%.

The feature is therefore not measuring encryption. It is measuring which feed
an address came from: phishing arrives from reporting services that record the
complete address, benign from crawls that store bare domains. That provenance
is present in every cross-validation fold, so no amount of internal validation
would have exposed it.

This script trains a second model with the feature removed and reports what
changes, on our own held-out data and on PhiUSIIL.

What to expect
--------------
Two outcomes are worth distinguishing, and both are reportable.

If recall on our own test set barely moves, the artefact was carrying almost
nothing and the model's performance rests on genuine signal. That is the
stronger result for us.

If recall drops noticeably, the artefact was load-bearing, and we can say how
much of the headline figure it was worth.

The PhiUSIIL false alarm rate should fall, but it will not resolve. Its benign
addresses also differ from ours in path_length, url_length and path_depth,
because they are bare homepages rather than full addresses. That is a
difference in how the datasets were built, not something a feature choice can
repair, and it is worth saying so rather than implying the fix is complete.

This runs as an additional experiment. The model saved by train_baseline.py is
left untouched, so the clustering and error analysis built on it remain valid.
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import reference
from url_features import build_feature_table

RANDOM_STATE = 42
TEST_SIZE = 0.2
NON_FEATURES = ["url", "type", "label"]
LEAKY = ["is_https"]


def train(X_train, y_train):
    return RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced",
    ).fit(X_train, y_train)


def report(name, model, X_test, y_test):
    pred = model.predict(X_test)
    r = recall_score(y_test, pred, zero_division=0)
    p = precision_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    print(f"  {name:<22} recall {r:.4f}   precision {p:.4f}   F1 {f:.4f}"
          f"   missed {fn:,}   false alarms {fp:,}")
    return {"model": name, "recall": r, "precision": p, "f1": f,
            "missed": fn, "false_alarms": fp}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    def opt(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    ref_path = opt("--reference")
    phi_path = opt("--phiusiil")
    path = argv[0] if argv and not argv[0].startswith("--") else "features.csv"

    if ref_path:
        reference.load(ref_path)

    f = pd.read_csv(path)
    X = f.drop(columns=[c for c in NON_FEATURES if c in f.columns])
    y = f["label"]

    present = [c for c in LEAKY if c in X.columns]
    if not present:
        raise SystemExit(f"None of {LEAKY} found in {path}.")
    kept = [c for c in X.columns if c not in present]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Train {len(X_train):,} | Test {len(X_test):,}")
    print(f"Removing: {', '.join(present)}  ({len(kept)} features remain)")

    # --- 1. our own held-out data ---------------------------------------
    print(f"\n{'=' * 78}\n1. On our own test set\n{'=' * 78}")
    full = train(X_train, y_train)
    clean = train(X_train[kept], y_train)

    rows = [report("with is_https", full, X_test, y_test),
            report("without is_https", clean, X_test[kept], y_test)]

    d_recall = rows[1]["recall"] - rows[0]["recall"]

    # Our five-fold cross-validation gave a standard deviation of 0.0006 on
    # recall. A single difference has to clear that spread by a margin before it
    # can be told apart from fold-to-fold variation, so two standard deviations
    # is used as the threshold. An earlier version of this script treated
    # anything above one standard deviation as meaningful and reported a change
    # of 0.0007 as evidence the feature mattered, which it is not.
    CV_SD = 0.0006
    threshold = 2 * CV_SD

    print(f"\n  Change in recall: {d_recall:+.4f}")
    print(f"  Cross-validation standard deviation was {CV_SD}, so a difference")
    print(f"  needs to exceed about {threshold:.4f} before it can be distinguished")
    print("  from ordinary fold-to-fold variation.")

    if abs(d_recall) <= threshold:
        print("\n  This change is at the noise floor. Removing the feature costs")
        print("  essentially nothing, which means the artefact was not")
        print("  load-bearing: it correlated with the label, but the model had")
        print("  better features to split on and relied on those instead.")
        print("  The headline figure therefore rests on genuine signal.")
    elif d_recall < 0:
        print(f"\n  Recall falls by {abs(d_recall):.4f}, beyond the noise floor. The")
        print("  artefact was carrying real weight, and that much of the headline")
        print("  figure came from dataset provenance rather than from phishing")
        print("  detection.")
    else:
        print("\n  Recall improves without it, beyond the noise floor, so the")
        print("  feature was not merely useless but actively misleading.")

    # --- 2. PhiUSIIL ------------------------------------------------------
    if phi_path:
        print(f"\n{'=' * 78}\n2. On PhiUSIIL's legitimate addresses\n{'=' * 78}")
        phi = pd.read_csv(phi_path, encoding="utf-8-sig", low_memory=False)
        legit = phi[phi["label"] == 1]["URL"].drop_duplicates()
        if len(legit) > 40_000:
            legit = legit.sample(40_000, random_state=RANDOM_STATE)
        print(f"  Extracting features for {len(legit):,} addresses...")
        P = build_feature_table(legit)

        fa_full = (full.predict(P[list(X.columns)]) == 1).mean()
        fa_clean = (clean.predict(P[kept]) == 1).mean()
        print(f"\n  with is_https     false alarms {fa_full:.1%}")
        print(f"  without is_https  false alarms {fa_clean:.1%}")
        print(f"  Change: {fa_clean - fa_full:+.1%}")

        if fa_clean < fa_full - 0.05:
            print("\n  A substantial improvement, confirming the feature was the")
            print("  main driver. What remains is explained by the other")
            print("  differences between the two benign populations: PhiUSIIL's")
            print("  are bare homepages, ours are full addresses.")
        else:
            print("\n  Little change, so is_https was not the main driver here.")
            print("  The structural differences - path_length, url_length and")
            print("  path_depth - account for far more than the scheme does.")
            print("\n  The conclusion is about the datasets rather than the model:")
            print("  PhiUSIIL's benign class is drawn entirely from canonical")
            print("  homepage roots, which is neither how benign addresses appear")
            print("  in our data nor how they appear in ordinary browsing. It is")
            print("  therefore not usable as a drop-in test set, and reporting")
            print("  that is more useful than reporting the raw false alarm rate.")

    pd.DataFrame(rows).to_csv("scheme_leak_experiment.csv", index=False)
    joblib.dump({"model": clean, "feature_names": kept},
                "linkguard_model_noscheme.joblib")
    print("\nWrote scheme_leak_experiment.csv and linkguard_model_noscheme.joblib")
    print("linkguard_model.joblib is unchanged, so the clustering and error")
    print("analysis built on it remain valid.")


if __name__ == "__main__":
    main()