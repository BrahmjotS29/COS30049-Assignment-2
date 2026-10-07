"""
compare_benign.py
=================
Finding out why the detector rejects PhiUSIIL's legitimate addresses.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python compare_benign.py features.csv PhiUSIIL_Phishing_URL_Dataset.csv --reference tranco.csv

Background
----------
Tested against PhiUSIIL, the model flagged 99.6% of legitimate addresses as
malicious. A first hypothesis - that PhiUSIIL's benign addresses are all bare
domains with no path, and that our model treats bare domains as suspicious -
was tested and rejected: bare domains are 32.3% malicious in our data against
33.4% for addresses with paths, and our model raises fewer false alarms on them,
not more.

So the cause lies elsewhere. This script finds it by measurement rather than by
guessing again.

Method
------
Both benign sets are described in the same feature space, then compared on two
quantities.

The first is how far apart they are. For each feature, the difference in means
is expressed in standard deviations of our own benign distribution, which makes
features with different units comparable.

The second is whether that distance matters. A feature can differ enormously
and change nothing if the model barely uses it. Multiplying the standardised
difference by the model's own feature importance gives a rough ranking of which
differences actually drive the predictions.

The product is a heuristic, not a formal attribution: a forest's importance is
an average over splits and does not decompose cleanly per prediction. It is
enough to point at the two or three features worth examining, which is what we
need.
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
SAMPLE = 40_000          # enough for stable means, small enough to be quick


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    ref_path = None
    if "--reference" in argv:
        i = argv.index("--reference")
        ref_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    ours_path = argv[0] if argv else "features.csv"
    phi_path = argv[1] if len(argv) > 1 else "PhiUSIIL_Phishing_URL_Dataset.csv"

    if ref_path:
        reference.load(ref_path)
    else:
        print("WARNING: no --reference. Rank and TLD features will be zero for")
        print("PhiUSIIL but not for our own rows, which would invent a")
        print("difference that is not there. Pass --reference tranco.csv.\n")

    bundle = joblib.load(MODEL_PATH)
    model, names = bundle["model"], bundle["feature_names"]
    importance = pd.Series(model.feature_importances_, index=names)

    # --- our benign rows ---
    ours = pd.read_csv(ours_path)
    ours = ours[ours["label"] == 0]
    if len(ours) > SAMPLE:
        ours = ours.sample(SAMPLE, random_state=42)
    A = ours[names]
    print(f"Our benign addresses:      {len(A):,}")

    # --- PhiUSIIL legitimate rows, through our own extractor ---
    phi = pd.read_csv(phi_path, encoding="utf-8-sig", low_memory=False)
    # Their 1 means legitimate; see validate_external.py for how this was
    # verified rather than assumed.
    legit = phi[phi["label"] == 1]["URL"].drop_duplicates()
    if len(legit) > SAMPLE:
        legit = legit.sample(SAMPLE, random_state=42)
    print(f"PhiUSIIL legitimate:       {len(legit):,}")
    print("Extracting features...")
    B = build_feature_table(legit)[names]

    # --- compare ---
    mean_a, mean_b = A.mean(), B.mean()

    # A feature with no spread in our benign data cannot be standardised: the
    # division would blow up on floating-point noise rather than report a real
    # difference. An early version produced a "4.6e15 standard deviations" row
    # from a standard deviation of about 1e-18. Anything below this threshold is
    # treated as constant and excluded.
    MIN_SD = 1e-6
    sd_a = A.std()
    unusable = sd_a < MIN_SD
    if unusable.any():
        print(f"\nExcluding {int(unusable.sum())} feature(s) with no spread in our "
              "benign data:")
        print("  " + ", ".join(sd_a.index[unusable]))
    sd_a = sd_a.where(~unusable, np.nan)

    diff = (mean_b - mean_a) / sd_a
    impact = diff.abs() * importance

    table = pd.DataFrame({
        "ours": mean_a,
        "phiusiil": mean_b,
        "diff_sd": diff,
        "importance": importance,
        "impact": impact,
    }).sort_values("impact", ascending=False)

    print(f"\n{'=' * 78}")
    print("Features that differ most, weighted by how much the model uses them")
    print(f"{'=' * 78}")
    print(f"{'feature':<26} {'ours':>11} {'phiusiil':>11} {'diff(sd)':>10} {'impact':>8}")
    print("-" * 78)
    shown = table.dropna(subset=["diff_sd"]).head(12)
    for n, r in shown.iterrows():
        print(f"{n:<26} {r['ours']:>11.3f} {r['phiusiil']:>11.3f} "
              f"{r['diff_sd']:>+10.2f} {r['impact']:>8.4f}")

    print("\nReading this table: diff(sd) is how far PhiUSIIL's legitimate")
    print("addresses sit from ours, in standard deviations of our own benign")
    print("distribution. A value beyond about 1.0 means the two populations")
    print("barely overlap on that feature. impact weights that gap by how much")
    print("the model relies on the feature.")

    top = table.dropna(subset=["diff_sd"]).head(3)
    print(f"\n{'=' * 78}\nLikely explanation\n{'=' * 78}")
    for n, r in top.iterrows():
        direction = "higher" if r["diff_sd"] > 0 else "lower"
        print(f"  {n}: PhiUSIIL legitimate addresses are {abs(r['diff_sd']):.1f} "
              f"sd {direction}")
        print(f"    ours {r['ours']:.3f} vs theirs {r['phiusiil']:.3f}, "
              f"model importance {r['importance']:.4f}")

    table.to_csv("benign_comparison.csv")

    # --- chart ---
    plot = table.dropna(subset=["diff_sd"]).head(12).sort_values("impact")
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(plot) + 1.5))
    colours = ["#D64550" if v > 0 else "#2a78d6" for v in plot["diff_sd"]]
    ax.barh(plot.index, plot["diff_sd"], color=colours)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("Difference in standard deviations (PhiUSIIL minus ours)")
    ax.set_title("Where PhiUSIIL's benign addresses differ from ours")
    fig.tight_layout(); fig.savefig("fig_benign_comparison.png", dpi=150); plt.close(fig)

    print("\nWrote benign_comparison.csv and fig_benign_comparison.png")
    print("\nThe finding to report is not that the model failed, but that the")
    print("two datasets define 'benign' differently, and which measurable")
    print("difference accounts for it.")


if __name__ == "__main__":
    main()