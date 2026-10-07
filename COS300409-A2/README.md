# LinkGuard — Offline Phishing URL Detection

COS30049 Computing Technology Innovation Project, Assignment 2
Brahmjot, Michael, Minh

LinkGuard assesses whether a web address is likely to be malicious, using only
the text of the address. It never contacts the site it describes, so it can be
run safely against addresses that are suspected to be hostile.

The trained model detects **96.6%** of malicious addresses on held-out data and
**97.4%** of live phishing addresses collected from a separate source.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Environment setup](#2-environment-setup)
3. [Getting the data](#3-getting-the-data)
4. [Preparing the features](#4-preparing-the-features)
5. [Training a model](#5-training-a-model)
6. [Making predictions](#6-making-predictions)
7. [Reproducing the analysis](#7-reproducing-the-analysis)
8. [File reference](#8-file-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Before you start

**No data is included in this repository.** The datasets are hundreds of
megabytes and are published elsewhere, so they are downloaded rather than
committed. Section 3 lists what to fetch and from where.

Running everything from scratch takes roughly 20 minutes, most of which is
feature extraction over 641,119 addresses. That step is run once and its output
reused.

Requirements: Python 3.9 or later, about 2 GB of free disk space, and an
internet connection for the initial downloads only. The detector itself never
uses the network.

---

## 2. Environment setup

Either option works. Use one, not both.

### Option A — venv (no extra software needed)

```bash
cd COS300409-A2
python3 -m venv .venv
source .venv/bin/activate          # macOS and Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Option B — conda

```bash
conda create -n linkguard python=3.11 -y
conda activate linkguard

conda install -c conda-forge pandas scikit-learn matplotlib seaborn joblib lightgbm -y
```

### Confirm it worked

```bash
python -c "import pandas, sklearn, matplotlib, joblib; print('ready')"
```

The environment must be activated in **every new terminal**. If you see
`ModuleNotFoundError` for a package you know is installed, check that your
prompt begins with `(.venv)` or `(linkguard)`.

In VS Code, also press `Cmd/Ctrl+Shift+P`, choose **Python: Select Interpreter**,
and pick the one inside your environment. Otherwise notebooks will quietly use a
different Python.

---

## 3. Getting the data

Download these into the project folder, keeping the file names below.

| File | Source | Size | Needed for |
|---|---|---|---|
| `malicious_phish.csv` | [Kaggle — Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | ~75 MB | Training. The base dataset: 651,191 labelled addresses. |
| `tranco.csv` | [tranco-list.eu](https://tranco-list.eu/) — choose the latest daily list | ~25 MB | Three features. Format is `rank,domain`, no header. |
| `PhiUSIIL_Phishing_URL_Dataset.csv` | [UCI repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | ~70 MB | External validation only. Optional. |
| `phisinglink_livedata.txt` | Supplied with the submission | ~50 KB | External validation only. Optional. |

**`tranco.csv` is not optional.** The model is trained with three features
derived from it, so predictions made without it will be wrong in a consistent
direction. Every command below that touches features takes `--reference
tranco.csv`.

---

## 4. Preparing the features

Converts raw addresses into the numeric table the models train on.

```bash
python url_features.py malicious_phish.csv features.csv --reference tranco.csv
```

Takes about 10 minutes. It reports progress as it goes:

```
Reference loaded: 1,000,000 domains, 500 brand names, 1026 TLDs
Read 651,191 rows from malicious_phish.csv
Dropped 10,072 duplicate addresses
Class balance:
  benign 428,080 | defacement 95,308 | phishing 94,086 | malware 23,645
Binary: 33.2% malicious
Wrote 641,119 rows x 45 columns to features.csv
```

**What this does.** Each address is parsed and 42 numeric features extracted:
length and structure, character composition, host and scheme properties, brand
similarity for typosquatting, and randomness measures such as Shannon entropy.
Duplicates are removed **before** extraction, because duplicate addresses split
across training and test sets let a model memorise rows and be scored on the
same rows.

Run it once. Everything afterwards reads `features.csv`.

To check the extractor without processing the whole dataset, run it with no
arguments for a short demonstration on six example addresses.

---

## 5. Training a model

### The baseline

```bash
python train_baseline.py features.csv
```

Trains four models — a deliberately useless always-benign baseline, logistic
regression with and without class balancing, and a random forest — and reports
recall, precision, F1 and a confusion matrix for each. Takes 3 to 5 minutes.

Writes `linkguard_model.joblib`, `results_baseline.csv`, two figures, and the
misclassified addresses for error analysis.

### The better model

```bash
python compare_models.py features.csv
```

Adds three models that were not part of the unit syllabus: LightGBM, a linear
SVM and Gaussian Naive Bayes, all on the identical train/test split so the
comparison is like for like. Takes 5 to 10 minutes.

Writes `linkguard_booster.joblib`, which is the model `predict.py` uses by
default.

Expected results:

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Always predict benign | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | 0.7859 | 0.8931 | 0.8361 |
| Random forest | 0.9539 | 0.9596 | 0.9567 |
| **LightGBM** | **0.9662** | **0.9615** | **0.9638** |
| Linear SVM | 0.8568 | 0.8220 | 0.8390 |
| Gaussian Naive Bayes | 0.1968 | 0.8713 | 0.3211 |

Recall is the headline metric, not accuracy. The always-benign model reaches
66.8% accuracy while detecting nothing, which is why accuracy is reported but
never relied upon.

Parameters are set in the scripts and can be changed there: LightGBM uses 300
estimators, a learning rate of 0.1, 63 leaves and `is_unbalance=True`; the
random forest uses 100 estimators with `class_weight='balanced'`. Both use
`random_state=42` so runs are reproducible.

---

## 6. Making predictions

### A single address

```bash
python predict.py "http://paypa1-secure.verify-account.tk/login" --reference tranco.csv
```

```
  http://paypa1-secure.verify-account.tk/login
  Trust score : 6/100  [#...................]
  Verdict     : DANGEROUS

  Warning signs:
    - Resembles a well-known brand without matching it
    - Top-level domain is one commonly associated with abuse
    - Connection is not encrypted
    - Contains several words common in phishing, such as verify or account
```

Several addresses can be given at once, each in quotes.

### A file of addresses

```bash
python predict.py --file urls.txt --reference tranco.csv --out scored.csv
```

One address per line. Writes a CSV of `url`, `trust_score` and `verdict`.

### From your own code

```python
import reference
from url_features import extract_one
import joblib, pandas as pd

reference.load("tranco.csv")
bundle = joblib.load("linkguard_booster.joblib")
model, names = bundle["model"], bundle["feature_names"]

X = pd.DataFrame([extract_one("http://example.tk/login")])[names]
trust = (1 - model.predict_proba(X)[0, 1]) * 100
print(f"{trust:.0f}/100")
```

Reindexing to `bundle["feature_names"]` matters. The model indexes columns by
position, so a feature table built in a different order produces confident
nonsense rather than an error.

Score bands: **70 and above** is reported as safe, **40 to 69** as suspicious,
**below 40** as dangerous. These are deliberately cautious because a missed
phishing address costs more than a false alarm for this user.

---

## 7. Reproducing the analysis

Optional, but these produce the findings reported in the paper.

```bash
# Unsupervised structure within the malicious class
python cluster_malicious.py features.csv

# Performance on data from entirely different sources
python validate_external.py --live phisinglink_livedata.txt \
    --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv --reference tranco.csv

# Why the model rejects PhiUSIIL's legitimate addresses
python compare_benign.py features.csv PhiUSIIL_Phishing_URL_Dataset.csv \
    --reference tranco.csv

# What a provenance artefact in one feature was worth
python fix_scheme_leak.py features.csv \
    --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv --reference tranco.csv
```

---

## 8. File reference

| File | Purpose |
|---|---|
| `url_features.py` | 42 feature functions, parsing, and the extraction pipeline |
| `reference.py` | Loads the Tranco list; derives domain rank, brand vocabulary and TLD popularity |
| `train_baseline.py` | Baseline models, cross-validation, error export |
| `compare_models.py` | LightGBM, linear SVM and Naive Bayes against the baseline |
| `cluster_malicious.py` | Two-stage k-means within the malicious class, plus a DBSCAN comparison |
| `validate_external.py` | Evaluation on the live feed and on PhiUSIIL |
| `compare_benign.py` | Locates which features separate two benign populations |
| `check_bare_domains.py` | Tests one hypothesis about the PhiUSIIL result |
| `fix_scheme_leak.py` | Measures the cost of removing a provenance-correlated feature |
| `predict.py` | Scores addresses with a trained model |

Generated files are not committed: `features.csv`, the `.joblib` models, the
figures and the `errors_*.csv` files are all produced by the commands above.

---

## 9. Troubleshooting

**`command not found: python`**
The environment is not active in this terminal. Run the activation line from
Section 2. On some systems the command is `python3`.

**`ModuleNotFoundError` for a package you installed**
Same cause. Check your prompt starts with `(.venv)` or `(linkguard)`.

**LightGBM fails with `Library not loaded: @rpath/libomp.dylib`**
macOS does not ship the OpenMP runtime LightGBM needs. Either install it with
`brew install libomp`, or do nothing: `compare_models.py` detects this and falls
back to scikit-learn's `HistGradientBoostingClassifier`, which is gradient
boosting of the same family. The script prints which one it used.

**Scores look wrong, or everything is flagged**
The `--reference tranco.csv` argument was probably omitted. Three of the 42
features come from that list and will be zero without it. The scripts print a
warning when it is missing.

**`zsh: parse error` or `no such file or directory` when pasting**
Python code and file contents cannot be pasted into a shell. Save them to a file
and run `python thefile.py`.

**`git push` fails with HTTP 408 or times out**
The data files are being committed. Check `.gitignore` excludes `*.csv`,
`*.joblib` and `*.png`, then `git rm -r --cached .` and recommit.

---

## Data sources

- Siddhartha, M 2021, *Malicious URLs dataset*, Kaggle.
- Le Pochat, V, Van Goethem, T, Tajalizadehkhoob, S, Korczyński, M & Joosen, W
  2019, 'Tranco: a research-oriented top sites ranking hardened against
  manipulation', *NDSS 2019*.
- Prasad, A & Chandra, S 2024, 'PhiUSIIL: a diverse security profile empowered
  phishing URL detection framework based on similarity index and incremental
  learning', *Computers & Security*, vol. 136, 103545. Released under CC BY 4.0.

Full references are in the project report.
