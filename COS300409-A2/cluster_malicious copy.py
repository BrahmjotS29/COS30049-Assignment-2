"""
cluster_malicious.py
====================
Unsupervised structure within the malicious class.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python cluster_malicious.py features.csv

What the assignment requires, and how this script meets it
----------------------------------------------------------
The brief is unusually specific about clustering, so each requirement is
addressed deliberately rather than incidentally.

"Clustering must be applied without using the labels"
    The label column is removed before any fitting. It is read back only in
    Section 5, after every cluster has been assigned, and only to check the
    result. No label touches the scaler, the PCA, or either clustering model.

"and it should be applied within a single class"
    We cluster the malicious rows only. The question is not "what separates
    malicious from benign" - the classifier already answers that. It is "what
    kinds of malicious address are there", which is a question labels cannot
    answer because the dataset's four categories are a labelling decision, not
    a structural one.

"Every cluster will then share the same majority label, which forces you to
describe each group by what its members actually have in common"
    Section 4 profiles each cluster by comparing its feature averages against
    the overall averages and reporting the features that differ most. Because
    the features are standardised before clustering, a cluster centroid
    coordinate is already a z-score: it says directly how many standard
    deviations that cluster sits from the average malicious address.

"Report the label composition of each cluster as a check that the clustering
found real structure. Composition alone is not a description."
    Section 5 reports composition and is explicitly framed as a check. The
    description lives in Section 4.

"the strongest signal is often something missing rather than something present"
    Section 4 reports the largest negative deviations alongside the positive
    ones, and flags when a cluster is defined mainly by absence.
"""

from __future__ import annotations

import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
NON_FEATURES = ["url", "type", "label"]

K_RANGE = range(2, 9)          # candidate cluster counts to evaluate
SILHOUETTE_SAMPLE = 20_000     # silhouette is O(n^2); sample rather than wait
DBSCAN_SAMPLE = 30_000         # DBSCAN is also expensive at full scale
TOP_FEATURES_PER_CLUSTER = 4

# If one cluster swallows this share of the data, the first pass has separated
# an outlier group rather than described the data, and a second pass is run
# inside the dominant cluster. See second_pass() for why this is necessary
# rather than optional.
DOMINANT_SHARE = 0.60

# A split that isolates a handful of points raises the silhouette without
# describing anything. Candidate k values whose smallest cluster falls below
# this share are set aside unless no candidate clears it.
MIN_CLUSTER_SHARE = 0.01


# ---------------------------------------------------------------------------
# 1. Load and restrict to one class
# ---------------------------------------------------------------------------

def load_malicious(path: str):
    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise SystemExit(f"{path} has no label column. Re-run url_features.py.")

    mal = df[df["label"] == 1].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows; clustering the {len(mal):,} malicious rows only")

    # The feature matrix. Dropping url, type and label here is the single most
    # important line in this script: everything downstream is unsupervised
    # because these columns are gone before anything is fitted.
    X = mal.drop(columns=[c for c in NON_FEATURES if c in mal.columns])

    # Constant columns carry no information and make silhouette scores
    # unstable, because a zero-variance column contributes nothing to distance
    # but still counts toward dimensionality.
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    if constant:
        print(f"Dropping {len(constant)} constant column(s): {', '.join(constant)}")
        X = X.drop(columns=constant)

    print(f"Clustering on {X.shape[1]} features")
    return mal, X


# ---------------------------------------------------------------------------
# 2. Scale
# ---------------------------------------------------------------------------

def scale(X):
    """Standardise every feature to mean 0, standard deviation 1.

    Essential before any distance-based clustering. Without it url_length,
    which ranges into the hundreds, would dominate every distance calculation
    and ratio features between 0 and 1 would be invisible. K-means would
    effectively cluster on length alone.

    The side benefit is interpretive: after scaling, a cluster centre's
    coordinate on a feature is that cluster's average in standard deviations
    from the overall malicious average. Section 4 reads centroids directly as
    z-scores because of this.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


# ---------------------------------------------------------------------------
# 3. Choose k, then fit
# ---------------------------------------------------------------------------

def choose_k(Xs, tag=""):
    """Evaluate candidate cluster counts by inertia and silhouette.

    Inertia always falls as k rises, so it cannot pick k by itself; we look for
    the elbow. Silhouette measures how well separated the clusters are and does
    have a meaningful maximum, so it is the tie-breaker.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(SILHOUETTE_SAMPLE, len(Xs)), replace=False)
    sample = Xs[idx]

    inertias, silhouettes, min_shares = [], [], []
    print("\nEvaluating cluster counts:")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(sample, labels[idx])
        smallest = np.bincount(labels, minlength=k).min() / len(labels)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        min_shares.append(smallest)
        flag = "  <- smallest cluster is tiny" if smallest < MIN_CLUSTER_SHARE else ""
        print(f"  k={k}  inertia={km.inertia_:>14,.0f}  silhouette={sil:.4f}"
              f"  smallest={smallest:6.2%}{flag}")

    # Choosing k by the highest silhouette alone tends to over-split: adding a
    # cluster that isolates a handful of outliers raises the score fractionally
    # while producing a group too small to describe or act on. We instead take
    # the smallest k whose silhouette is within a small tolerance of the best,
    # which prefers the simpler structure when the evidence barely separates
    # them. The tolerance and the rule are reported so the choice is auditable.
    tolerance = 0.02

    # Candidates are k values that produce clusters large enough to describe.
    usable = [i for i, m in enumerate(min_shares) if m >= MIN_CLUSTER_SHARE]
    if not usable:
        print(f"\nNo k produces clusters all above {MIN_CLUSTER_SHARE:.0%}. "
              "Every split isolates outliers;")
        print("falling back to the raw silhouette, and treat the result with care.")
        usable = list(range(len(silhouettes)))

    peak = max(silhouettes[i] for i in usable)
    peak_idx = max(usable, key=lambda i: silhouettes[i])

    # Among the usable candidates take the simplest one that is close to the
    # best, since a fractional silhouette gain rarely justifies another cluster.
    chosen_idx = next(i for i in usable if silhouettes[i] >= peak - tolerance)

    best_k = list(K_RANGE)[chosen_idx]
    peak_k = list(K_RANGE)[peak_idx]

    if len(usable) < len(silhouettes):
        rejected = [list(K_RANGE)[i] for i in range(len(silhouettes)) if i not in usable]
        print(f"\nSet aside k={rejected} for producing a cluster under "
              f"{MIN_CLUSTER_SHARE:.0%} of the data.")

    if best_k != peak_k:
        print(f"Highest usable silhouette is k={peak_k} ({peak:.4f}), but k={best_k} "
              f"scores {silhouettes[chosen_idx]:.4f},")
        print(f"within the {tolerance} tolerance. Taking the simpler structure.")
    else:
        print(f"\nChose k={best_k} (silhouette {peak:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(list(K_RANGE), inertias, "o-", color="#2a78d6")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow")
    ax2.plot(list(K_RANGE), silhouettes, "o-", color="#eb6834")
    ax2.axvline(best_k, ls="--", c="grey", lw=1)
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.set_title("Separation")
    fig.tight_layout(); fig.savefig(f"fig_cluster_choose_k{tag}.png", dpi=150); plt.close(fig)

    return best_k, inertias, silhouettes


# ---------------------------------------------------------------------------
# 4. Describe each cluster by what its members share
# ---------------------------------------------------------------------------

def profile(km, feature_names, counts, tag="", reference_label="average malicious address"):
    """Describe clusters by their largest deviations from the average.

    Because the data was standardised, each centroid coordinate is already a
    z-score. A value of +1.8 on nb_subdomains means that cluster averages 1.8
    standard deviations more subdomains than the typical malicious address.

    Both directions are reported. A cluster defined by what it lacks is as real
    as one defined by what it has, and in this data the absences are often the
    clearer signal.
    """
    centres = pd.DataFrame(km.cluster_centers_, columns=feature_names)
    centres.to_csv(f"cluster_profiles{tag}.csv")

    print(f"\n{'=' * 70}\nWhat each cluster has in common\n{'=' * 70}")
    descriptions = {}

    for c in range(len(centres)):
        row = centres.loc[c]
        ordered = row.reindex(row.abs().sort_values(ascending=False).index)
        top = ordered.head(TOP_FEATURES_PER_CLUSTER)

        share = counts[c] / counts.sum()
        print(f"\nCluster {c}  -  {counts[c]:,} addresses ({share:.1%})")

        # A cluster holding well under one per cent of the data is usually a
        # handful of outliers rather than a kind of address. Worth noting
        # rather than describing as though it were a finding.
        if share < 0.01:
            print("    (very small - likely outliers rather than a distinct type)")

        above = [(n, v) for n, v in top.items() if v > 0]
        below = [(n, v) for n, v in top.items() if v < 0]

        if above:
            print(f"  Higher than the {reference_label}:")
            for n, v in above:
                print(f"    {n:<28} {v:+.2f} sd")
        if below:
            print(f"  Lower than the {reference_label}:")
            for n, v in below:
                print(f"    {n:<28} {v:+.2f} sd")

        # Flag clusters defined mainly by absence, since the brief notes these
        # are easy to overlook and often the more informative case.
        if len(below) > len(above):
            print("    -> defined mainly by what is missing")

        descriptions[c] = {"above": above, "below": below}

    # Heatmap of the features that vary most across clusters.
    spread = centres.abs().max().sort_values(ascending=False).head(14).index
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(centres) + 2.5))
    data = centres[spread]
    im = ax.imshow(data.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(data.values).max(), vmax=np.abs(data.values).max())
    ax.set_xticks(range(len(spread)))
    ax.set_xticklabels(spread, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(centres)))
    ax.set_yticklabels([f"Cluster {i}\n({counts[i]:,})" for i in range(len(centres))],
                       fontsize=8)
    ax.set_title(f"Cluster centres, in standard deviations from the {reference_label}")
    fig.colorbar(im, ax=ax, label="sd")
    fig.tight_layout(); fig.savefig(f"fig_cluster_profiles{tag}.png", dpi=150); plt.close(fig)

    return descriptions


def second_pass(mal, X, assignments, counts):
    """Re-cluster inside a cluster that holds most of the data.

    K-means minimises within-cluster distance, so a binary feature with an
    extreme value - has_ip_host sits almost four standard deviations out -
    offers the single largest reduction available and the first split will
    always take it. That split is informative, but it leaves the remaining
    majority undescribed, and "slightly fewer digits than average" is not a
    description of two hundred thousand addresses.

    So we cluster again within the dominant group. Two details matter.

    The subset is re-standardised. The question has changed from "how does this
    differ from all malicious addresses" to "how do these differ from each
    other", and the second question needs the subset's own mean and spread as
    the reference point.

    This remains unsupervised and remains within a single class. It is a
    two-stage analysis of the malicious rows, not a different experiment.
    """
    dominant = int(counts.idxmax())
    share = counts[dominant] / counts.sum()
    if share < DOMINANT_SHARE:
        return None

    print(f"\n{'=' * 70}")
    print(f"Second pass: cluster {dominant} holds {share:.1%} of the data")
    print(f"{'=' * 70}")
    print("The first split separated an extreme group rather than describing")
    print("the data. Re-clustering inside the dominant cluster, re-standardised")
    print("so the comparison is against this subset rather than all malicious.")

    mask = assignments == dominant
    X_sub = X[mask].reset_index(drop=True)
    mal_sub = mal[mask].reset_index(drop=True)

    constant = [c for c in X_sub.columns if X_sub[c].nunique() <= 1]
    if constant:
        print(f"\nDropping {len(constant)} column(s) now constant within the subset")
        X_sub = X_sub.drop(columns=constant)

    Xs_sub, _ = scale(X_sub)
    k_sub, _, _ = choose_k(Xs_sub, tag="_subset")

    km_sub = KMeans(n_clusters=k_sub, random_state=RANDOM_STATE, n_init=10)
    sub_assign = km_sub.fit_predict(Xs_sub)
    sub_counts = pd.Series(sub_assign).value_counts().sort_index()

    sub_share = sub_counts.max() / sub_counts.sum()
    if sub_share >= DOMINANT_SHARE:
        print(f"\nNote: the second pass is also dominated by one cluster "
              f"({sub_share:.1%}).")
        print("That is a result in itself. Beyond the extreme group separated in")
        print("the first pass, this data has no strong natural partition, and")
        print("reporting that is more honest than forcing a split that the")
        print("silhouette scores do not support.")

    print("\nWithin the dominant cluster:")
    profile(km_sub, list(X_sub.columns), sub_counts, tag="_subset",
            reference_label="others in this subset")
    check_against_labels(mal_sub, sub_assign, tag="_subset")
    plot_pca(Xs_sub, sub_assign, sub_counts, tag="_subset")

    return {"parent": dominant, "assignments": sub_assign, "model": km_sub}


# ---------------------------------------------------------------------------
# 5. Check the structure against the labels
# ---------------------------------------------------------------------------

def check_against_labels(mal, assignments, tag=""):
    """Compare clusters to the original four-class labels.

    This is a check, not a description. If the clusters were noise they would
    each mirror the overall class mix; a cluster whose composition departs from
    that mix has found structure the labels alone do not express.

    Run after clustering, never before.
    """
    mal = mal.copy()
    mal["cluster"] = assignments

    overall = mal["type"].value_counts(normalize=True)
    table = pd.crosstab(mal["cluster"], mal["type"], normalize="index")

    print(f"\n{'=' * 70}\nLabel composition - a check, not a description\n{'=' * 70}")
    print("\nOverall malicious mix:")
    for t, v in overall.items():
        print(f"  {t:<14} {v:.1%}")

    print("\nBy cluster:")
    print((table * 100).round(1).to_string())

    # Largest departure from the overall mix tells us which cluster is most
    # distinctive in label terms, which is the evidence that structure is real.
    print("\nMost distinctive cluster per label:")
    for t in table.columns:
        c = table[t].idxmax()
        print(f"  {t:<14} most concentrated in cluster {c} "
              f"({table.loc[c, t]:.1%} vs {overall[t]:.1%} overall)")

    table.to_csv(f"cluster_composition{tag}.csv")
    mal[["url", "type", "cluster"]].to_csv(f"clustered_malicious{tag}.csv", index=False)
    return mal


# ---------------------------------------------------------------------------
# 6. DBSCAN comparison
# ---------------------------------------------------------------------------

def compare_dbscan(Xs):
    """Run DBSCAN on a sample and report honestly how it behaves.

    The Week 7 seminar notes that DBSCAN "shines in two or three dimensions and
    struggles beyond". We have far more than three, so this is run as a
    documented comparison rather than in the expectation it will win. Finding
    that a method does not suit the data is a reportable result, provided the
    reason is given rather than the attempt quietly dropped.

    DBSCAN is also O(n^2) in the worst case, so it runs on a sample.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(DBSCAN_SAMPLE, len(Xs)), replace=False)
    sample = Xs[idx]

    print(f"\n{'=' * 70}\nDBSCAN comparison (sample of {len(sample):,})\n{'=' * 70}")

    for eps in (1.5, 2.5, 3.5, 5.0):
        db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit(sample)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()
        print(f"  eps={eps:<4}  clusters={n_clusters:<4}  noise={noise:.1%}")

    print("\n  Interpretation: in a space of this many dimensions, points are")
    print("  sparse and a single density threshold rarely fits every region.")
    print("  High noise or a collapse to one cluster is the expected outcome,")
    print("  not a configuration error. K-means, which partitions rather than")
    print("  requiring density, is the better fit for this data.")


# ---------------------------------------------------------------------------
# 7. Visualise
# ---------------------------------------------------------------------------

def plot_pca(Xs, assignments, counts, tag=""):
    """Project to two dimensions for a picture of the clustering.

    PCA is used for display only. The clustering itself runs in the full
    feature space; reducing first would discard information. Two components of
    many will not capture everything, so the variance explained is printed to
    keep the picture honest.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(25_000, len(Xs)), replace=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs[idx])
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c in sorted(set(assignments[idx])):
        m = assignments[idx] == c
        ax.scatter(coords[m, 0], coords[m, 1], s=3, alpha=0.4,
                   label=f"Cluster {c} ({counts[c]:,})")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} of variance)")
    ax.set_title("Malicious addresses, clustered")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout(); fig.savefig(f"fig_cluster_pca{tag}.png", dpi=150); plt.close(fig)

    print(f"\nPCA: the two plotted components capture {var.sum():.1%} of total "
          "variance, so the picture understates the separation.")


# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else "features.csv"

    mal, X = load_malicious(path)
    Xs, scaler = scale(X)

    best_k, _, _ = choose_k(Xs)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    assignments = km.fit_predict(Xs)
    counts = pd.Series(assignments).value_counts().sort_index()

    profile(km, list(X.columns), counts)
    check_against_labels(mal, assignments)
    plot_pca(Xs, assignments, counts)

    sub = second_pass(mal, X, assignments, counts)

    compare_dbscan(Xs)

    joblib.dump({"scaler": scaler, "kmeans": km, "features": list(X.columns)},
                "linkguard_clusters.joblib")

    print("\nWrote: cluster_profiles.csv, cluster_composition.csv,")
    print("       clustered_malicious.csv, linkguard_clusters.joblib,")
    print("       fig_cluster_choose_k.png, fig_cluster_profiles.png,")
    print("       fig_cluster_pca.png")
    if sub is not None:
        print("       plus _subset versions from the second pass")
    print("\nNext: open clustered_malicious.csv and read a few addresses from")
    print("each cluster. The report needs real examples, not just statistics.")


if __name__ == "__main__":
    main()