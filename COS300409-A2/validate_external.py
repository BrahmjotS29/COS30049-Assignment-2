"""
validate_external.py
====================
Testing the trained detector on data it has never seen, from other sources.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python validate_external.py --live phisinglink_livedata.txt
    python validate_external.py --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv
    python validate_external.py --live L.txt --phiusiil P.csv --reference tranco.csv

Why this exists
---------------
A train/test split taken from one dataset answers a narrow question: can the
model separate rows that were collected the same way, at the same time, by the
same people. It cannot tell us whether the model has learned something about
phishing or something about this particular collection.

The assignment asks for an evaluation that "deliberately tests the model's
limits", naming a held-out category or a deliberately harder test set as
examples. These two sources provide both:

  - A list of currently active phishing addresses, gathered separately and
    later. Every row is phishing, so this measures recall on the single class
    our error analysis identified as weakest, with no dilution from the easier
    classes.

  - PhiUSIIL, an independently collected dataset with both classes, which
    tests whether performance survives a change of source entirely.

Nothing here is used for training. The model, the scaler and the feature set
are loaded exactly as saved.

A note on the PhiUSIIL label convention
---------------------------------------
PhiUSIIL encodes label 1 as legitimate and label 0 as phishing, which is the
opposite of our convention. This was confirmed by inspecting the addresses
rather than assumed from the column name: label 0 holds obvious phishing such
as kuerennkayccato-co-jp.* and f0573330.xsph.ru, while label 1 holds ordinary
sites such as peach.ca and gentner.de. The conversion is applied explicitly in
load_phiusiil() and is worth stating in the report, because merging the two
conventions silently would invert the labels on a third of a million rows.

PhiUSIIL also ships 54 pre-extracted features. Those are deliberately ignored.
We take the URL column and run our own extractor, both because the assignment
asks for feature extraction to be our own code and because a comparison is only
meaningful if both sides are measured the same way.
"""

from __future__ import annotations

import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import reference
from url_features import build_feature_table

MODEL_PATH = "linkguard_model.joblib"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: str = MODEL_PATH):
    """Load the saved model and the feature order it was trained on.

    The stored feature list is not decoration. A tree model indexes columns by
    position, so a feature table built in a different order produces confident
    nonsense rather than an error. Every table built here is reindexed to this
    list before prediction.
    """
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found. Run train_baseline.py first.")
    print(f"Loaded model from {path} ({len(bundle['feature_names'])} features)")
    return bundle["model"], bundle["feature_names"]


def load_live(path: str) -> pd.DataFrame:
    """Load a plain list of phishing addresses, one per line.

    No labels are needed: every entry is phishing, so the only measurable
    quantity is recall. That is the point. Precision cannot be computed without
    negatives, and reporting a number we cannot compute would be worse than
    reporting one metric honestly.
    """
    urls = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip()]
    before = len(urls)
    urls = list(dict.fromkeys(urls))          # order-preserving dedupe
    if before != len(urls):
        print(f"  dropped {before - len(urls)} duplicate addresses")
    return pd.DataFrame({"url": urls, "label": 1})


def load_phiusiil(path: str, sample: int | None = None) -> pd.DataFrame:
    """Load PhiUSIIL, taking the URL column only and flipping the label.

    See the module docstring for why the flip is necessary and how it was
    verified.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if "URL" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"{path}: expected URL and label columns, "
                         f"found {list(df.columns)[:6]}...")

    out = pd.DataFrame({
        "url": df["URL"],
        # Their 1 means legitimate; ours means malicious.
        "label": (df["label"] == 0).astype(int),
    })
    out = out.drop_duplicates(subset=["url"]).reset_index(drop=True)

    if sample and sample < len(out):
        out = out.sample(sample, random_state=42).reset_index(drop=True)
        print(f"  sampled {sample:,} rows")

    print(f"  {len(out):,} rows, {out['label'].mean():.1%} malicious after the flip")
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(name, model, feature_names, data, baseline_recall=None):
    """Extract features, predict, and report.

    Reindexing to feature_names is what keeps this honest: any feature the
    extractor no longer produces becomes an explicit error rather than a
    silently shifted column.
    """
    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    print(f"Extracting features for {len(data):,} addresses...")

    X = build_feature_table(data["url"])

    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise SystemExit(
            f"The extractor no longer produces {missing}. The saved model "
            "expects them. Re-run train_baseline.py so model and extractor match."
        )
    X = X[feature_names]

    y = data["label"].values
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    n_pos = int(y.sum())
    recall = (pred[y == 1] == 1).mean() if n_pos else float("nan")

    print(f"\n  Malicious in this set: {n_pos:,}")
    print(f"  Recall:  {recall:.4f}   ({int((pred[y == 1] == 1).sum()):,} caught, "
          f"{int((pred[y == 1] == 0).sum()):,} missed)")

    # Only compute the metrics this data can support. A set with no benign rows
    # cannot give precision, and inventing one would misrepresent the test.
    n_neg = int((y == 0).sum())
    if n_neg:
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  Precision: {precision:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Accuracy:  {(pred == y).mean():.4f}")
        print(f"\n  False alarms: {fp:,} of {n_neg:,} benign")
        print(f"  Missed:       {fn:,} of {n_pos:,} malicious")
    else:
        print("  No benign rows here, so precision and accuracy are not defined.")
        print("  Recall is the only metric this set can support.")

    if baseline_recall is not None and n_pos:
        delta = recall - baseline_recall
        print(f"\n  Against our own test set ({baseline_recall:.4f}): {delta:+.4f}")
        if delta < -0.05:
            print("  Performance drops materially on unseen data. The model has")
            print("  learned something specific to the training collection.")
        elif delta < -0.01:
            print("  A modest drop, consistent with a change of source.")
        else:
            print("  Performance holds, which is evidence the model generalises")
            print("  beyond the dataset it was trained on.")

    return {"set": name, "n": len(data), "n_malicious": n_pos,
            "recall": recall, "pred": pred, "proba": proba, "y": y}


def confidence_report(result, data, filename):
    """Where the model was wrong, and how sure it was.

    A model that misses an address while reporting 49% confidence is behaving
    differently from one that misses it at 3%. The second case means the
    features point the wrong way entirely, which is the more serious failure
    and the one worth inspecting by hand.
    """
    proba = result["proba"]
    if proba is None:
        return

    y, pred = result["y"], result["pred"]
    missed = (y == 1) & (pred == 0)
    if not missed.any():
        return

    conf = proba[missed]
    print(f"\n  Of the {missed.sum():,} missed, the model's malicious score was:")
    print(f"    above 0.40 (close call):     {(conf > 0.40).sum():,}")
    print(f"    0.20 to 0.40:                {((conf > 0.20) & (conf <= 0.40)).sum():,}")
    print(f"    below 0.20 (confidently wrong): {(conf <= 0.20).sum():,}")

    out = data.loc[missed].copy()
    out["malicious_score"] = conf
    out.sort_values("malicious_score").to_csv(filename, index=False)
    print(f"    written to {filename}, lowest score first")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(proba[y == 1], bins=40, color="#2a78d6", alpha=0.85)
    ax.axvline(0.5, ls="--", c="#D64550", lw=1.5, label="decision threshold")
    ax.set_xlabel("Model's malicious score")
    ax.set_ylabel("Addresses")
    ax.set_title("Confidence on known-malicious addresses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename.replace(".csv", ".png"), dpi=150)
    plt.close(fig)


def threshold_table(result):
    """What recall would be at thresholds other than 0.5.

    predict() is predict_proba() compared against 0.5, with one subtlety that
    matters: scikit-learn decides by argmax, so an address scoring exactly 0.5
    is classed benign. With a hundred trees, scores land on exactly 0.5 often
    enough to matter - on one run here, 77 of 772 addresses did. The comparison
    below therefore uses a strict ">" so that the 0.5 row reproduces the recall
    reported above rather than contradicting it.

    Since our topic favours recall it is worth showing what a lower threshold
    would buy. On a set with no benign rows the cost in false alarms cannot be
    measured, so this is presented as a question rather than a recommendation.
    """
    proba = result["proba"]
    if proba is None:
        return
    y = result["y"]
    if y.sum() == 0:
        return

    print("\n  Recall at other decision thresholds:")
    for t in (0.5, 0.4, 0.3, 0.2, 0.1):
        # Strict ">" to match predict()'s argmax tie-breaking; see docstring.
        r = (proba[y == 1] > t).mean()
        marker = "   <- the default, matching the recall above" if t == 0.5 else ""
        print(f"    threshold {t:.1f}  ->  recall {r:.4f}{marker}")
    print("  Lowering the threshold raises recall and costs false alarms. The")
    print("  cost cannot be measured on a set with no benign rows; use the main")
    print("  test set to decide whether the trade is worth making.")


# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    def opt(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    live_path = opt("--live")
    phiusiil_path = opt("--phiusiil")
    ref_path = opt("--reference")
    baseline = float(opt("--baseline") or 0.9539)

    if not (live_path or phiusiil_path):
        raise SystemExit(__doc__.split("Why this exists")[0])

    # The reference list must be loaded before extraction or the rank and TLD
    # features silently come out as zero, and the model was trained with them.
    if ref_path:
        reference.load(ref_path)
    else:
        print("WARNING: no --reference given. If the model was trained with a")
        print("reference list, three features will be zero here and the results")
        print("will understate performance. Pass --reference tranco.csv.\n")

    model, feature_names = load_model()
    results = []

    if live_path:
        print(f"\nLoading live phishing list: {live_path}")
        data = load_live(live_path)
        r = score("Live phishing addresses (all malicious, separate source)",
                  model, feature_names, data, baseline)
        confidence_report(r, data, "errors_live_missed.csv")
        threshold_table(r)
        results.append(r)

    if phiusiil_path:
        print(f"\nLoading PhiUSIIL: {phiusiil_path}")
        data = load_phiusiil(phiusiil_path)
        r = score("PhiUSIIL (independent dataset, both classes)",
                  model, feature_names, data, baseline)
        confidence_report(r, data, "errors_phiusiil_missed.csv")
        results.append(r)

    if len(results) > 1:
        print(f"\n{'=' * 68}\nSummary\n{'=' * 68}")
        print(pd.DataFrame([{k: v for k, v in r.items()
                             if k in ("set", "n", "n_malicious", "recall")}
                            for r in results]).to_string(index=False))

    print("\nNext: read the lowest-scoring missed addresses by hand. The rubric")
    print("asks what the failures have in common, and that is not a number.")


if __name__ == "__main__":
    main()