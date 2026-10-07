"""
reference.py
============
External reference data for the LinkGuard feature extractor.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

The base dataset gives us addresses and labels and nothing else. Several of the
signals the literature describes cannot be computed from an address in
isolation: whether a domain is a known legitimate site, whether it resembles a
popular brand, and whether its top-level domain is ordinary or unusual. Each
needs a reference list.

This module loads one external source, the Tranco top-sites list, and derives
three different things from it. That is deliberate. The Assignment 2 rubric
states that "one additional source that is properly integrated scores higher
than several that are simply concatenated", so the aim is depth of integration
rather than a count of files.

What the one list gives us
--------------------------
1. Domain rank         A domain in the top sites list, and its position, is a
                       legitimacy anchor. paypal.com ranks highly; paypa1.com
                       does not appear at all.
2. Brand vocabulary    The top few hundred names are what attackers actually
                       imitate. Edit distance is measured against these only,
                       because nobody typosquats the fifty-thousandth site.
3. TLD popularity      The share of top-sites domains using each TLD. This
                       replaces our hand-written list of "suspicious" TLDs with
                       a measured quantity.

No labels are used anywhere in this module. TLD popularity is computed from the
reference list, not from our training data, so there is no target leakage.

Getting the data
----------------
Download a list from https://tranco-list.eu/ (choose "Download" on the latest
daily list). It arrives as a CSV of rank,domain with no header. Cite it as:

    Le Pochat, V, Van Goethem, T, Tajalizadehkhoob, S, Korczynski, M &
    Joosen, W 2019, 'Tranco: a research-oriented top sites ranking hardened
    against manipulation', in Proceedings of the 26th Annual Network and
    Distributed System Security Symposium, Internet Society.

Tranco is used rather than Alexa or Majestic because it is built for research
use and is designed to be reproducible: each list has a permanent identifier,
so the exact list used can be cited and retrieved again.

Usage
-----
    import reference
    reference.load("tranco.csv")       # once, before extracting features
"""

from __future__ import annotations

import math
from functools import lru_cache

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Module-level because the feature functions are called once per row and must
# not reload the list each time. Empty until load() is called, and every
# consumer checks is_loaded() so the extractor still runs without it.

_ranks: dict[str, int] = {}        # "paypal.com" -> 42
_brand_names: list[str] = []       # ["google", "youtube", "facebook", ...]
_brands_by_length: dict[int, list[tuple[str, int]]] = {}  # (name, letter bitmask)
_tld_share: dict[str, float] = {}  # "com" -> 0.47
_n_domains = 0

# How many of the top names form the typosquatting vocabulary. Beyond a few
# hundred, two things go wrong: the comparison gets slow, and short or generic
# names start sitting within edit distance 2 of ordinary words by accident,
# which produces false lookalike flags.
BRAND_VOCAB_SIZE = 500

# Names too short to compare safely. "bbc" is within distance 2 of hundreds of
# three-letter strings, so including it would flag noise as typosquatting.
MIN_BRAND_LENGTH = 5


def is_loaded() -> bool:
    """True if a reference list has been loaded."""
    return bool(_ranks)


def load(path: str, vocab_size: int = BRAND_VOCAB_SIZE) -> None:
    """Load a Tranco-style CSV of rank,domain.

    Tolerates a header row and either column order being absent, because the
    download format has changed between versions. Builds all three derived
    structures in a single pass over the file.
    """
    global _ranks, _brand_names, _brands_by_length, _tld_share, _n_domains

    ranks: dict[str, int] = {}
    tld_counts: dict[str, int] = {}
    brand_names: list[str] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rank_text, domain = parts[0].strip(), parts[1].strip().lower()
            if not domain or not rank_text.isdigit():
                continue              # skips a header row without special-casing it

            rank = int(rank_text)
            ranks[domain] = rank

            if "." in domain:
                tld = domain.rsplit(".", 1)[1]
                tld_counts[tld] = tld_counts.get(tld, 0) + 1

                # Build the brand vocabulary from the highest-ranked domains.
                if len(brand_names) < vocab_size:
                    name = domain.split(".")[0]
                    if len(name) >= MIN_BRAND_LENGTH and name not in brand_names:
                        brand_names.append(name)

    if not ranks:
        raise ValueError(f"No usable rows in {path}. Expected lines of 'rank,domain'.")

    _ranks = ranks
    _n_domains = len(ranks)
    _brand_names = brand_names

    # Bucket brands by length so edit distance only compares plausible pairs.
    # Edit distance is at least the difference in lengths, so a token of length
    # 8 can never be within 2 of a brand of length 11.
    # Each brand is stored with a bitmask of the distinct letters it contains.
    # The mask drives a cheap pre-filter in _min_distance_cached: if a brand
    # contains three or more distinct characters the token lacks, at least
    # three edits are needed, so the pair cannot be within our cutoff of two
    # and the expensive comparison is skipped entirely.
    _brands_by_length = {}
    for b in brand_names:
        _brands_by_length.setdefault(len(b), []).append((b, _letter_mask(b)))

    _tld_share = {t: c / _n_domains for t, c in tld_counts.items()}

    _min_distance_cached.cache_clear()

    print(f"Reference loaded: {_n_domains:,} domains, "
          f"{len(_brand_names)} brand names, {len(_tld_share)} TLDs")


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def domain_rank(domain: str) -> int | None:
    """Rank of a registered domain, or None if it is not in the list."""
    return _ranks.get(domain)


def rank_score(domain: str) -> float:
    """Rank on a log scale, higher meaning more popular.

    Raw rank is a poor feature: the difference between rank 1 and rank 100
    matters far more than the difference between 900,000 and 900,100. A log
    scale reflects that. Domains absent from the list score 0, which places
    them below every ranked domain.
    """
    r = _ranks.get(domain)
    if r is None:
        return 0.0
    return math.log10(_n_domains / r)


def tld_share(tld: str) -> float:
    """Proportion of reference domains using this TLD.

    A measured quantity replacing a hand-written suspicion list. Ordinary TLDs
    score high; TLDs that barely appear among legitimate popular sites score
    near zero. Note this measures ordinariness, not abuse: a low score means
    unusual, which the model is free to learn is or is not predictive.
    """
    return _tld_share.get(tld, 0.0)


def _letter_mask(text: str) -> int:
    """26-bit mask of which letters appear in a string.

    Used as a lower-bound filter on edit distance. Non-letters are ignored,
    which only makes the filter more permissive and so never discards a pair
    that should have been compared.
    """
    mask = 0
    for c in text:
        i = ord(c) - 97
        if 0 <= i < 26:
            mask |= 1 << i
    return mask


def _levenshtein(a: str, b: str, cutoff: int) -> int:
    """Edit distance with early abandonment.

    Stops as soon as every value in the current row exceeds the cutoff, since
    the final distance can only grow from there. On a vocabulary of several
    hundred brands this avoids completing most comparisons.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        if min(current) > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


@lru_cache(maxsize=300_000)
def _min_distance_cached(token: str) -> int:
    """Smallest edit distance from a token to any brand name, capped at 3.

    Cached because host tokens repeat heavily across a large corpus. Only
    brands within two characters of the token's length are considered, for the
    reason given in load().

    Returns 99 when nothing is close, so the feature stays numeric rather than
    needing a null.
    """
    if not _brand_names or len(token) < MIN_BRAND_LENGTH:
        return 99

    cutoff = 2
    best = 99
    n = len(token)
    tok_mask = _letter_mask(token)

    for length in range(n - cutoff, n + cutoff + 1):
        for b, b_mask in _brands_by_length.get(length, ()):
            # Distinct letters in the brand that the token does not have. Each
            # one costs at least one edit, so three or more rules the pair out
            # before any character-by-character work.
            if bin(b_mask & ~tok_mask).count("1") > cutoff:
                continue
            d = _levenshtein(token, b, cutoff)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def min_brand_distance(token: str) -> int:
    """Public wrapper over the cached lookup."""
    return _min_distance_cached(token)


def is_known_brand(name: str) -> bool:
    """True if a domain name is itself one of the top brand names."""
    return name in _brand_names