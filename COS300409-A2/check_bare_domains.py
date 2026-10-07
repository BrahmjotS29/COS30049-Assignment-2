"""
check_bare_domains.py
=====================
Confirming why the model flags almost every PhiUSIIL legitimate address.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python check_bare_domains.py features.csv

The question
------------
Tested against PhiUSIIL, the detector flagged 134,255 of 134,850 legitimate
addresses as malicious, a false alarm rate of 99.6%. A figure that extreme is
rarely a weak model. A weak model produces 60 or 70 per cent; near-total
failure on one class usually means the test data differs from the training data
in some systematic way.

Profiling PhiUSIIL showed that every one of its legitimate addresses is a bare
homepage root: 100% begin https://www., none has a path, none has a query, and
the median length is 27 characters. Our own benign addresses are ordinary
real-world URLs with paths and parameters.

The hypothesis is therefore that the model has learned, from our training data,
that a short address with no path is unusual for a benign site - and that this
is a property of how our dataset was assembled rather than a property of
phishing.

The test
--------
If the hypothesis holds, the model should also flag bare-domain benign
addresses inside our own data, where no change of source can be blamed. Split
our benign rows by whether they have a path and compare the false alarm rate.

A large gap confirms it. A small gap refutes it, and the PhiUSIIL result would
need a different explanation.

We also report how our training data labels bare domains, since that is the
mechanism: the model can only have learned this if bare domains in our data are
disproportionately malicious.
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_PATH = "linkguard_model.joblib"

# These must match train_baseline.py exactly. The model has seen the training
# rows and predicts them almost perfectly, so measuring false alarms on the
# whole file would hide the very effect we are testing for. Reproducing the
# same split with the same seed recovers the held-out rows.
RANDOM_STATE = 42
TEST_SIZE = 0.2
NON_FEATURES = ["url", "type", "label"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "features.csv"

    f = pd.read_csv(path)
    bundle = joblib.load(MODEL_PATH)
    model, names = bundle["model"], bundle["feature_names"]

    if "path_length" not in f.columns:
        raise SystemExit("features.csv has no path_length column.")

    # A path_length of 0 or 1 means the address is a bare host, possibly with a
    # trailing slash. This is the shape of every PhiUSIIL legitimate address.
    f["bare"] = f["path_length"] <= 1

    # --- 1. How our own data labels bare domains -------------------------
    print("=" * 66)
    print("1. How our training data labels bare domains")
    print("=" * 66)
    rate = f.groupby("bare")["label"].agg(["size", "mean"])
    rate.index = ["has a path", "bare domain"]
    rate.columns = ["addresses", "share malicious"]
    print(rate.assign(**{"share malicious": rate["share malicious"].map("{:.1%}".format)})
              .to_string())

    overall = f["label"].mean()
    bare_rate = f.loc[f["bare"], "label"].mean()
    print(f"\nOverall malicious share: {overall:.1%}")
    print(f"Among bare domains:      {bare_rate:.1%}")
    if bare_rate > overall:
        print(f"\nBare domains are {bare_rate / overall:.1f} times more likely to be")
        print("malicious in our data than the average address. That is what the")
        print("model learned, and it is a fact about this dataset rather than")
        print("about phishing.")
    else:
        print("\nBare domains are not over-represented among malicious addresses")
        print("here, so this is not the mechanism. The PhiUSIIL result needs a")
        print("different explanation.")

    # --- 2. False alarm rate on our own benign rows ----------------------
    print("\n" + "=" * 66)
    print("2. False alarms on our own benign addresses (held-out rows only)")
    print("=" * 66)

    # Rebuild the same split train_baseline.py used, so we score only rows the
    # model never trained on.
    X_all = f.drop(columns=[c for c in NON_FEATURES if c in f.columns] + ["bare"])
    _, X_test, _, _ = train_test_split(
        X_all, f["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=f["label"])
    test = f.loc[X_test.index]
    print(f"  Scoring {len(test):,} held-out rows "
          f"({len(test) / len(f):.0%} of the data)\n")

    benign = test[test["label"] == 0]
    rows = []
    for label, sub in [("Bare domain", benign[benign["bare"]]),
                       ("Has a path", benign[~benign["bare"]])]:
        if not len(sub):
            continue
        fp = (model.predict(sub[names]) == 1).mean()
        rows.append({"group": label, "addresses": len(sub), "false alarm rate": fp})
        print(f"  {label:<14} n={len(sub):>8,}   false alarms {fp:.1%}")

    if len(rows) == 2:
        gap = rows[0]["false alarm rate"] - rows[1]["false alarm rate"]
        print(f"\n  Difference: {gap:+.1%}")
        if gap > 0.10:
            print("\n  Confirmed. The model flags bare-domain benign addresses far")
            print("  more often even within our own data, where the source is")
            print("  identical. The PhiUSIIL result is therefore explained by how")
            print("  its benign class was built, not by a failure to generalise.")
        elif gap > 0.02:
            print("\n  Partly supported: the effect is present but smaller than the")
            print("  PhiUSIIL result alone would suggest. Something else is also")
            print("  contributing.")
        else:
            print("\n  Not supported. The model handles bare domains in our data")
            print("  normally, so the PhiUSIIL result has another cause. Compare")
            print("  the feature distributions of the two benign sets directly.")

    # --- 3. Scale of the comparison --------------------------------------
    print("\n" + "=" * 66)
    print("3. Why the two tests disagreed")
    print("=" * 66)
    share = benign["bare"].mean()
    print(f"  Bare domains are {share:.1%} of our benign addresses.")
    print(f"  They are 100% of PhiUSIIL's.")
    print("\n  Our test set is dominated by the case the model handles well, so")
    print("  the headline figure never exposed this. PhiUSIIL consists entirely")
    print("  of the case it handles badly. Neither number is wrong; they measure")
    print("  different things, which is the argument for testing on more than")
    print("  one source.")

    pd.DataFrame(rows).to_csv("bare_domain_check.csv", index=False)
    print("\nWrote bare_domain_check.csv")


if __name__ == "__main__":
    main()