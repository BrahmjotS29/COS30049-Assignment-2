"""
predict.py
==========
Score a web address with the trained LinkGuard model.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

    python predict.py "http://paypa1-secure.verify-account.tk/login" --reference tranco.csv
    python predict.py --file urls.txt --reference tranco.csv --out scored.csv

This is the entry point a user would actually touch, and it is the piece the
Assignment 3 web application will wrap: the backend loads the same model file
and calls the same functions.

The output follows the prototype designed in Assignment 1 - a trust score from
0 to 100, a plain-language verdict, and the signals that informed it, so the
user can see why rather than being handed a number.

A note on the explanation
-------------------------
The signals listed are the ones that actually fired for the address, described
in plain language. They are not a per-prediction attribution: a gradient
boosted ensemble does not decompose cleanly into "this feature contributed X
per cent of this decision", and claiming otherwise would be misleading. What is
shown is what the extractor observed, which is the honest version of the same
idea and is what the user needs in order to judge the address themselves.

No network request is made at any point. The address is never visited.
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

import reference
from url_features import FEATURE_NAMES, build_feature_table, extract_one

DEFAULT_MODEL = "linkguard_booster.joblib"

# Thresholds for the plain-language verdict. Our topic favours recall, so the
# bands are deliberately cautious: an address the model is unsure about is
# called suspicious rather than safe.
SAFE_ABOVE = 70
SUSPICIOUS_ABOVE = 40

# How to describe each signal when it fires. Only features that mean something
# to a non-specialist are listed; the rest still inform the score but would not
# help a user decide.
SIGNALS = [
    ("has_ip_host", lambda v: v == 1,
     "Host is a numeric IP address rather than a domain name"),
    ("brand_lookalike", lambda v: v == 1,
     "Resembles a well-known brand without matching it"),
    ("is_shortening_service", lambda v: v == 1,
     "Uses a link shortener, so the real destination is hidden"),
    ("has_punycode", lambda v: v == 1,
     "Uses an internationalised domain, a technique behind look-alike attacks"),
    ("suspicious_tld", lambda v: v == 1,
     "Top-level domain is one commonly associated with abuse"),
    ("https_token_in_host", lambda v: v == 1,
     "The word 'https' appears inside the host name, which fakes security"),
    ("nb_at", lambda v: v > 0,
     "Contains an @ symbol, which hides the real host from the reader"),
    ("is_https", lambda v: v == 0,
     "Connection is not encrypted"),
    ("in_top_sites", lambda v: v == 1,
     "Domain appears in a list of the most visited sites worldwide"),
    ("brand_is_registered_domain", lambda v: v == 1,
     "Domain is itself a well-known brand"),
    ("phish_hints", lambda v: v >= 2,
     "Contains several words common in phishing, such as verify or account"),
    ("url_length", lambda v: v > 75,
     "Unusually long address"),
    ("nb_subdomains", lambda v: v >= 3,
     "Unusually deep chain of subdomains"),
]

REASSURING = {"in_top_sites", "brand_is_registered_domain"}


def load_model(path: str = DEFAULT_MODEL):
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(
            f"{path} not found. Train a model first:\n"
            "    python train_baseline.py features.csv       (random forest)\n"
            "    python compare_models.py features.csv       (LightGBM, better)")
    return bundle["model"], bundle["feature_names"]


def trust_score(model, X) -> float:
    """Convert the model's malicious probability into a 0-100 trust score.

    Inverted so that higher is safer, which is the direction a user expects
    from anything presented as a score out of 100.
    """
    # Index [0, 1] rather than [:, 1]: X holds exactly one row here, and
    # predict_proba always returns a 2-D array, so slicing the column would
    # give a one-element array rather than a number.
    return float((1 - model.predict_proba(X)[0, 1]) * 100)


def verdict(score: float) -> str:
    if score >= SAFE_ABOVE:
        return "SAFE"
    if score >= SUSPICIOUS_ABOVE:
        return "SUSPICIOUS"
    return "DANGEROUS"


def explain(features: dict) -> tuple[list, list]:
    """Return the warning signals and the reassuring ones that fired."""
    warnings, reassurances = [], []
    for name, fired, text in SIGNALS:
        if name not in features:
            continue
        if fired(features[name]):
            (reassurances if name in REASSURING else warnings).append(text)
    return warnings, reassurances


def score_one(url: str, model, names) -> dict:
    features = extract_one(url)
    X = pd.DataFrame([features])[names]
    score = trust_score(model, X)
    warnings, reassurances = explain(features)
    return {"url": url, "score": score, "verdict": verdict(score),
            "warnings": warnings, "reassurances": reassurances}


def print_result(r: dict) -> None:
    bar = "#" * int(r["score"] / 5) + "." * (20 - int(r["score"] / 5))
    print(f"\n  {r['url']}")
    print(f"  Trust score : {r['score']:.0f}/100  [{bar}]")
    print(f"  Verdict     : {r['verdict']}")
    if r["warnings"]:
        print("\n  Warning signs:")
        for w in r["warnings"]:
            print(f"    - {w}")
    if r["reassurances"]:
        print("\n  In its favour:")
        for w in r["reassurances"]:
            print(f"    - {w}")
    if not r["warnings"] and not r["reassurances"]:
        print("\n  No individual signal stood out; the score reflects the")
        print("  combination of all 42 measurements.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    def opt(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    ref_path = opt("--reference")
    model_path = opt("--model") or DEFAULT_MODEL
    file_path = opt("--file")
    out_path = opt("--out")

    positional = [a for i, a in enumerate(argv)
                  if not a.startswith("--")
                  and (i == 0 or not argv[i - 1].startswith("--"))]

    if not file_path and not positional:
        print(__doc__.split("This is the entry point")[0])
        raise SystemExit("Give a URL, or --file with one address per line.")

    # The reference list must be loaded before extraction. Without it the three
    # Tranco features come through as zero, and since the model was trained
    # with them, every score would be wrong in the same direction.
    if ref_path:
        reference.load(ref_path)
    else:
        print("WARNING: no --reference given. If the model was trained with a")
        print("reference list, scores here will be unreliable. Pass")
        print("--reference tranco.csv\n")

    model, names = load_model(model_path)

    if file_path:
        urls = [l.strip() for l in open(file_path, encoding="utf-8",
                                        errors="replace") if l.strip()]
        print(f"Scoring {len(urls):,} addresses from {file_path}...")
        X = build_feature_table(urls)[names]
        proba = model.predict_proba(X)[:, 1]
        out = pd.DataFrame({
            "url": urls,
            "trust_score": ((1 - proba) * 100).round(1),
            "verdict": [verdict(s) for s in (1 - proba) * 100],
        })
        print(out["verdict"].value_counts().to_string())
        if out_path:
            out.to_csv(out_path, index=False)
            print(f"\nWrote {out_path}")
        else:
            print("\n" + out.head(20).to_string(index=False))
            if len(out) > 20:
                print(f"... and {len(out) - 20:,} more. Use --out to save them all.")
        return

    for url in positional:
        print_result(score_one(url, model, names))
    print()


if __name__ == "__main__":
    main()