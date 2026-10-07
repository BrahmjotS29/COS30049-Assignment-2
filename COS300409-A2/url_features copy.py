"""
url_features.py
===============
Feature extraction for the LinkGuard phishing URL detector.

COS30049 Assignment 2 - Phishing URL Detection
Brahmjot, Michael, Minh

This module turns a raw web address into a table of numeric features. It makes
no network calls: every feature is computed from the address string alone, which
satisfies the offline operation requirement (FR-09) and places the work in the
"static analysis" category described by Sahoo, Liu and Hoi (2017).

Design notes
------------
Each feature is a separate, individually testable function with a docstring that
states what the feature is meant to capture and which published work motivates
it. That structure is deliberate: requirement NFR-07 commits us to signals that
can be added or changed without altering the model interface, and the Assignment
2 rubric awards marks for features "drawn from published work or documented
practice, each justified by what it is meant to capture".

Usage
-----
    import pandas as pd

    from url_features import build_feature_table

    raw = pd.read_csv("malicious_phish.csv")       # columns: url, type
    features = build_feature_table(raw["url"])
    features["label"] = (raw["type"] != "benign").astype(int)

References
----------
Le, H, Pham, Q, Sahoo, D & Hoi, SCH 2018, URLNet: learning a URL representation
    with deep learning for malicious URL detection, arXiv:1802.03162.
Marchal, S, Francois, J, State, R & Engel, T 2014, 'PhishScore: hacking
    phishers' minds', in Proceedings of the 10th International Conference on
    Network and Service Management, IEEE, pp. 46-54.
Mohammad, RM, Thabtah, F & McCluskey, L 2012, 'An assessment of features related
    to phishing websites using an automated technique', in Proceedings of the
    International Conference for Internet Technology and Secured Transactions,
    IEEE, pp. 492-497.
Sahingoz, OK, Buber, E, Demir, O & Diri, B 2019, 'Machine learning based
    phishing detection from URLs', Expert Systems with Applications, vol. 117,
    pp. 345-357.
Sahoo, D, Liu, C & Hoi, SCH 2017, Malicious URL detection using machine
    learning: a survey, arXiv:1701.07179.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlparse

import pandas as pd

import reference

# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------
# These are small and shipped with the module so the detector stays offline.
# The brand list is a placeholder: replace it with a real top-domain list
# (Tranco or Cisco Umbrella) as part of the additional-dataset work, which is
# where the Innovation 1 marks sit.

PHISH_HINT_WORDS = [
    "secure", "account", "login", "signin", "update", "verify", "confirm",
    "banking", "password", "webscr", "ebayisapi", "suspend", "billing",
    "wallet", "recover", "unlock", "validate",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy", "bitly.com",
]

# Placeholder brand vocabulary. Swap for a real top-domain list.
BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "facebook", "netflix",
    "instagram", "whatsapp", "linkedin", "dropbox", "adobe", "chase", "wellsfargo",
    "ebay", "steam", "binance", "coinbase", "outlook", "office365",
]

# TLDs repeatedly reported as over-represented in abuse data. Treat this as a
# starting point to be replaced by a real abuse-rate table.
SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "country", "stream", "download", "racing", "win", "bid", "loan", "zip",
]


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

def parse(url: str) -> dict:
    """Split a raw address into its parts.

    Two quirks of urlparse are handled here, both of which cause silent or
    noisy failures on real data.

    First, many addresses in the base dataset have no scheme (for example
    "www.example.com/path"). urlparse treats such a string as a path with an
    empty netloc, which would silently zero out every host feature. We add a
    default scheme first so the host is found.

    Second, urlparse is lazy: it does not validate the port until .port is
    actually read, and raises ValueError at that point if the text after the
    colon is not a number. Real datasets contain addresses with binary junk in
    the authority, so every attribute read is guarded individually rather than
    wrapping the urlparse call alone.

    Returns a dict rather than a ParseResult so downstream functions can stay
    simple and individually testable.
    """
    url = "" if url is None else str(url).strip()
    working = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) else "http://" + url

    def safe(getter, default):
        """Read one parsed attribute, falling back if it cannot be decoded.

        Malformed addresses do occur in the raw data. Rather than dropping the
        row we return empty parts, so the feature table keeps its shape and the
        row can be found again during error analysis.
        """
        try:
            value = getter()
        except (ValueError, UnicodeError):
            return default
        return default if value is None else value

    try:
        p = urlparse(working)
    except (ValueError, UnicodeError):
        return {"raw": url, "scheme": "", "host": "", "path": "", "query": "",
                "port": None, "malformed": 1}

    host = safe(lambda: p.hostname, "")
    port = safe(lambda: p.port, None)
    scheme = safe(lambda: p.scheme, "")
    path = safe(lambda: p.path, "")
    query = safe(lambda: p.query, "")

    return {
        "raw": url,
        "scheme": scheme,
        "host": host.lower(),
        "path": path,
        "query": query,
        "port": port,
        # Flagged rather than dropped. If a meaningful number of rows are malformed that is itself a finding worth reporting, and the flag lets us measure how many there are.
        "malformed": 0,
    }


def registered_domain(host: str) -> str:
    """Return the registrable part of the host, approximately.

    A correct implementation needs the Public Suffix List, because "co.uk" is a
    suffix but "co.com" is not. We use the last two labels, which is right for
    the large majority of addresses and wrong for second-level country suffixes.
    This limitation is recorded rather than hidden; replacing it with the
    tldextract library is a small, worthwhile improvement.
    """
    if not host:
        return ""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def shannon_entropy(text: str) -> float:
    """Shannon entropy of a string, in bits per character.

    Captures randomness. Algorithmically generated domains draw characters more
    uniformly than human-chosen names, which raises entropy. Used as a phishing
    signal by Marchal et al. (2014).
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings.

    Implemented here rather than imported so the module has no dependency
    beyond pandas. Used for typosquatting detection: "paypa1" is one edit from
    "paypal".
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + (ca != cb)))  # substitution
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Group 1: length and structure
# Captures padding - extra segments used to push the real domain out of view.
# ---------------------------------------------------------------------------

def f_is_malformed(p: dict) -> int:
    """1 if the address could not be fully parsed.

    Kept as a feature rather than a reason to drop the row. An address so
    malformed that a standard parser rejects it is itself unusual, and we would
    rather measure that than quietly discard it.
    """
    return p.get("malformed", 0)


def f_url_length(p: dict) -> int:
    """Total characters. Padding with path and query junk hides the real domain
    (Mohammad et al. 2012; Sahingoz et al. 2019)."""
    return len(p["raw"])


def f_hostname_length(p: dict) -> int:
    """Length of the host alone. Separates padding in the host from padding in
    the path, which are different tactics (Sahingoz et al. 2019)."""
    return len(p["host"])


def f_path_length(p: dict) -> int:
    """Length of the path. Deep or generated paths are common on compromised
    hosts (Sahoo et al. 2017)."""
    return len(p["path"])


def f_query_length(p: dict) -> int:
    """Length of the query string, which may carry victim identifiers
    (Sahoo et al. 2017)."""
    return len(p["query"])


def f_nb_subdomains(p: dict) -> int:
    """Number of subdomain labels. Subdomain stuffing puts a brand where it
    carries no authority, as in paypal.com.login-secure.xyz
    (Mohammad et al. 2012)."""
    if not p["host"]:
        return 0
    return max(0, len(p["host"].split(".")) - 2)


def f_avg_subdomain_length(p: dict) -> float:
    """Mean length of the subdomain labels. Distinguishes many short labels from
    a few long ones (Sahingoz et al. 2019)."""
    if not p["host"]:
        return 0.0
    labels = p["host"].split(".")[:-2]
    return sum(len(x) for x in labels) / len(labels) if labels else 0.0


def f_nb_dots(p: dict) -> int:
    """Dot count across the whole address. A proxy for subdomain depth that also
    catches dots used inside path segments (Mohammad et al. 2012)."""
    return p["raw"].count(".")


def f_path_depth(p: dict) -> int:
    """Number of non-empty path segments (Sahoo et al. 2017)."""
    return len([s for s in p["path"].split("/") if s])


# ---------------------------------------------------------------------------
# Group 2: character composition
# Captures obfuscation and machine generation.
# ---------------------------------------------------------------------------

def f_ratio_digits_url(p: dict) -> float:
    """Proportion of digits. Generated addresses carry more digits than
    human-chosen ones (Sahingoz et al. 2019)."""
    return sum(c.isdigit() for c in p["raw"]) / len(p["raw"]) if p["raw"] else 0.0


def f_ratio_digits_host(p: dict) -> float:
    """Digits in the host specifically, which is more unusual than digits in a
    path (Sahingoz et al. 2019)."""
    return sum(c.isdigit() for c in p["host"]) / len(p["host"]) if p["host"] else 0.0


def f_nb_hyphens(p: dict) -> int:
    """Hyphen count. Used to build brand-like domains such as
    paypal-secure-login.com (Mohammad et al. 2012)."""
    return p["raw"].count("-")


def f_nb_at(p: dict) -> int:
    """Count of @. Browsers ignore everything before an @ in the authority, so
    it hides the real host (Mohammad et al. 2012)."""
    return p["raw"].count("@")


def f_nb_percent(p: dict) -> int:
    """Percent-encoding count. Encoding hides literal text from casual reading
    (Sahoo et al. 2017)."""
    return p["raw"].count("%")


def f_nb_underscore(p: dict) -> int:
    """Underscore count. Not valid in hostnames, so its presence there is itself
    irregular (Sahoo et al. 2017)."""
    return p["raw"].count("_")


def f_nb_double_slash(p: dict) -> int:
    """Count of // after the scheme, which indicates a redirect appended to the
    path (Mohammad et al. 2012)."""
    body = p["raw"].split("://", 1)[-1]
    return body.count("//")


def f_char_repeat(p: dict) -> int:
    """Longest run of one repeated character, a signature of generated strings
    (Sahingoz et al. 2019)."""
    if not p["raw"]:
        return 0
    best = run = 1
    for a, b in zip(p["raw"], p["raw"][1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def f_nb_www(p: dict) -> int:
    """Occurrences of "www" outside its proper leading position, as in
    www-paypal.com (Sahingoz et al. 2019)."""
    return max(0, p["raw"].lower().count("www") - (1 if p["host"].startswith("www.") else 0))


def f_nb_com(p: dict) -> int:
    """Occurrences of "com" beyond the TLD, as in paypal.com.evil.xyz
    (Sahingoz et al. 2019)."""
    return max(0, p["raw"].lower().count("com") - (1 if p["host"].endswith(".com") else 0))


# ---------------------------------------------------------------------------
# Group 3: host and scheme
# ---------------------------------------------------------------------------

def f_has_ip_host(p: dict) -> int:
    """1 if the host is a literal IP address. A numeric host instead of a name
    is unusual for a consumer service (Mohammad et al. 2012).

    Note: at least one recent study found this feature contributed nothing to a
    deployed model. Keep it, measure its importance, and report the result
    either way.
    """
    if not p["host"]:
        return 0
    try:
        ipaddress.ip_address(p["host"])
        return 1
    except ValueError:
        return 0


def f_is_https(p: dict) -> int:
    """1 if the scheme is https. Absence of encryption on a page requesting
    credentials is a warning sign (Mohammad et al. 2012)."""
    return int(p["scheme"].lower() == "https")


def f_https_token_in_host(p: dict) -> int:
    """1 if the literal string "https" appears inside the hostname, used to fake
    security as in https-paypal.com (Sahingoz et al. 2019)."""
    return int("https" in p["host"])


def f_has_port(p: dict) -> int:
    """1 if a non-standard port is given explicitly (Sahoo et al. 2017)."""
    return int(p["port"] is not None and p["port"] not in (80, 443))


def f_has_punycode(p: dict) -> int:
    """1 if the host carries the xn-- prefix, the mechanism behind homograph
    attacks (Sahoo et al. 2017)."""
    return int("xn--" in p["host"])


def f_tld_length(p: dict) -> int:
    """Length of the top-level domain. Kept numeric so the categorical TLD can
    be handled separately."""
    rd = registered_domain(p["host"])
    return len(rd.split(".")[-1]) if "." in rd else 0


def f_suspicious_tld(p: dict) -> int:
    """1 if the TLD appears on our abuse-prone list. Replace the list with a
    real abuse-rate table as part of the additional-dataset work
    (Sahoo et al. 2017)."""
    rd = registered_domain(p["host"])
    tld = rd.split(".")[-1] if "." in rd else ""
    return int(tld in SUSPICIOUS_TLDS)


def f_tld_in_subdomain(p: dict) -> int:
    """1 if a TLD-looking string appears in the subdomain, as in
    paypal.com.evil.xyz (Sahingoz et al. 2019)."""
    if not p["host"]:
        return 0
    labels = p["host"].split(".")[:-2]
    return int(any(x in ("com", "net", "org", "gov", "edu") for x in labels))


def f_tld_in_path(p: dict) -> int:
    """1 if a TLD-looking string appears in the path, which is a sign the real
    brand has been relegated there (Sahingoz et al. 2019)."""
    return int(bool(re.search(r"\.(com|net|org|gov|edu)(/|$)", p["path"].lower())))


# ---------------------------------------------------------------------------
# Group 4: deception and brand
# ---------------------------------------------------------------------------

def f_phish_hints(p: dict) -> int:
    """Count of suspicious words anywhere in the address. PhishMatch found
    tokens such as pay, update and login highly suggestive when present in
    subdomains (Sahingoz et al. 2019)."""
    low = p["raw"].lower()
    return sum(w in low for w in PHISH_HINT_WORDS)


def f_brand_in_subdomain(p: dict) -> int:
    """1 if a known brand appears in the subdomain rather than the registered
    domain. This is the core of subdomain deception
    (Sahingoz et al. 2019)."""
    if not p["host"]:
        return 0
    sub = ".".join(p["host"].split(".")[:-2])
    return int(any(b in sub for b in BRANDS))


def f_brand_in_path(p: dict) -> int:
    """1 if a brand appears in the path, where it carries no authority at all
    (Sahingoz et al. 2019)."""
    return int(any(b in p["path"].lower() for b in BRANDS))


def host_tokens(host: str) -> list:
    """Split a host into comparable words.

    Labels are split on dots, then on hyphens, because attackers build
    brand-like strings both ways: paypal-secure.example.com and
    paypal.secure.example.com. Tokens shorter than four characters are dropped,
    since short strings sit within edit distance 2 of many brands by accident.
    """
    if not host:
        return []
    tokens = []
    for label in host.split("."):
        tokens.extend(t for t in label.split("-") if len(t) >= 4)
    return tokens


@lru_cache(maxsize=200_000)
def _fallback_min_distance(token: str) -> int:
    """Smallest edit distance to any brand in the small built-in list.

    Used only when no external reference list has been loaded, so that the
    extractor still runs standalone. The built-in list holds twenty names,
    which is far too few to be meaningful across a large corpus - see the note
    on f_min_brand_edit_distance.
    """
    best = 99
    n = len(token)
    for b in BRANDS:
        if abs(n - len(b)) >= best:
            continue
        d = levenshtein(token, b)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def f_min_brand_edit_distance(p: dict) -> int:
    """Smallest edit distance from any host token to a known brand name.

    Captures typosquatting: paypa1 is one edit from paypal. Every token in the
    host is compared, not just the registered domain, because in subdomain
    deception the brand-lookalike sits in the subdomain.

    A distance of 0 is an exact brand match and is not in itself suspicious;
    see f_brand_lookalike for the signal that matters.

    The brand vocabulary comes from the external reference list when one is
    loaded, and from the small built-in list otherwise. This matters: with only
    twenty built-in names almost no address in a large corpus matches anything,
    the feature is near-constant, and a tree model correctly ignores it. A low
    importance score under the built-in list is evidence about the list, not
    about the feature.
    """
    tokens = host_tokens(p["host"])
    if not tokens:
        return 99

    lookup = reference.min_brand_distance if reference.is_loaded() else _fallback_min_distance
    best = 99
    for t in tokens:
        d = lookup(t)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def f_brand_lookalike(p: dict) -> int:
    """1 if some host token is close to a brand without matching it exactly.

    Distance 1 or 2 is the typosquatting and homograph band: close enough to
    fool a reader, not an actual brand domain. Distance 0 is excluded because
    an exact match is usually the legitimate site.
    """
    d = f_min_brand_edit_distance(p)
    return int(1 <= d <= 2)


def f_brand_is_registered_domain(p: dict) -> int:
    """1 if the registered domain's name is itself a known brand.

    A legitimacy anchor rather than a suspicion signal. Without it the model
    has no way to tell paypal.com from paypa1.com beyond one character.
    """
    rd = registered_domain(p["host"])
    name = rd.split(".")[0] if rd else ""
    if not name:
        return 0
    if reference.is_loaded():
        return int(reference.is_known_brand(name))
    return int(name in BRANDS)


def f_in_top_sites(p: dict) -> int:
    """1 if the registered domain appears anywhere in the reference top-sites
    list.

    The strongest legitimacy signal available to us offline. A domain that
    hundreds of thousands of people visit is unlikely to be a phishing host,
    and a freshly registered phishing domain cannot be in the list at all.

    Requires the external reference list; returns 0 without it.
    """
    if not reference.is_loaded():
        return 0
    return int(reference.domain_rank(registered_domain(p["host"])) is not None)


def f_domain_rank_score(p: dict) -> float:
    """Popularity of the registered domain on a log scale, 0 if unranked.

    Graded rather than binary, because the gap between rank 1 and rank 100
    carries far more information than the gap between 900,000 and 900,100.
    """
    if not reference.is_loaded():
        return 0.0
    return reference.rank_score(registered_domain(p["host"]))


def f_tld_popularity(p: dict) -> float:
    """Share of reference domains using this TLD.

    A measured replacement for our hand-written suspicious-TLD list. Measures
    ordinariness rather than abuse: a near-zero score means the TLD is rare
    among popular legitimate sites, and the model learns whether that predicts
    anything.

    Falls back to 0 without the reference list, in which case f_suspicious_tld
    carries the signal instead.
    """
    if not reference.is_loaded():
        return 0.0
    rd = registered_domain(p["host"])
    tld = rd.rsplit(".", 1)[1] if "." in rd else ""
    return reference.tld_share(tld)


def f_is_shortening_service(p: dict) -> int:
    """1 if the host is a known URL shortener. The real destination is hidden
    and cannot be assessed offline (Mohammad et al. 2012)."""
    return int(any(p["host"] == s or p["host"].endswith("." + s)
                   for s in SHORTENING_SERVICES))


# ---------------------------------------------------------------------------
# Group 5: randomness and statistical
# Recent work reports these as the most informative group.
# ---------------------------------------------------------------------------

def f_entropy_url(p: dict) -> float:
    """Shannon entropy of the whole address (Marchal et al. 2014)."""
    return shannon_entropy(p["raw"])


def f_entropy_domain(p: dict) -> float:
    """Entropy of the registered domain alone. Catches algorithmically
    generated domains a length feature would miss (Marchal et al. 2014)."""
    return shannon_entropy(registered_domain(p["host"]))


def f_vowel_ratio(p: dict) -> float:
    """Proportion of vowels in the registered domain. A pronounceability proxy:
    random strings have unusual vowel proportions (Marchal et al. 2014)."""
    name = registered_domain(p["host"]).replace(".", "")
    if not name:
        return 0.0
    return sum(c in "aeiou" for c in name) / len(name)


def f_char_continuation_rate(p: dict) -> float:
    """Proportion of adjacent character pairs that stay in the same class
    (letter, digit or symbol). Generated strings switch class more often than
    human-written ones (Le et al. 2018)."""
    s = p["raw"]
    if len(s) < 2:
        return 0.0

    def cls(c):
        return "a" if c.isalpha() else ("d" if c.isdigit() else "s")

    same = sum(cls(a) == cls(b) for a, b in zip(s, s[1:]))
    return same / (len(s) - 1)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

# Order matters. The web application in Assignment 3 must build its input
# DataFrame with these columns, in this order, or predict() will raise.
FEATURES = {
    # group 1
    "is_malformed": f_is_malformed,
    "url_length": f_url_length,
    "hostname_length": f_hostname_length,
    "path_length": f_path_length,
    "query_length": f_query_length,
    "nb_subdomains": f_nb_subdomains,
    "avg_subdomain_length": f_avg_subdomain_length,
    "nb_dots": f_nb_dots,
    "path_depth": f_path_depth,
    # group 2
    "ratio_digits_url": f_ratio_digits_url,
    "ratio_digits_host": f_ratio_digits_host,
    "nb_hyphens": f_nb_hyphens,
    "nb_at": f_nb_at,
    "nb_percent": f_nb_percent,
    "nb_underscore": f_nb_underscore,
    "nb_double_slash": f_nb_double_slash,
    "char_repeat": f_char_repeat,
    "nb_www": f_nb_www,
    "nb_com": f_nb_com,
    # group 3
    "has_ip_host": f_has_ip_host,
    "is_https": f_is_https,
    "https_token_in_host": f_https_token_in_host,
    "has_port": f_has_port,
    "has_punycode": f_has_punycode,
    "tld_length": f_tld_length,
    "suspicious_tld": f_suspicious_tld,
    "tld_in_subdomain": f_tld_in_subdomain,
    "tld_in_path": f_tld_in_path,
    # group 4
    "phish_hints": f_phish_hints,
    "brand_in_subdomain": f_brand_in_subdomain,
    "brand_in_path": f_brand_in_path,
    "min_brand_edit_distance": f_min_brand_edit_distance,
    "brand_lookalike": f_brand_lookalike,
    "brand_is_registered_domain": f_brand_is_registered_domain,
    "is_shortening_service": f_is_shortening_service,
    "in_top_sites": f_in_top_sites,
    "domain_rank_score": f_domain_rank_score,
    "tld_popularity": f_tld_popularity,
    # group 5
    "entropy_url": f_entropy_url,
    "entropy_domain": f_entropy_domain,
    "vowel_ratio": f_vowel_ratio,
    "char_continuation_rate": f_char_continuation_rate,
}

FEATURE_NAMES = list(FEATURES)


def extract_one(url: str) -> dict:
    """Extract every feature for a single address.

    Returned as a plain dict so it can be used directly by the Assignment 3
    backend on one URL, as well as in bulk during training.
    """
    p = parse(url)
    return {name: fn(p) for name, fn in FEATURES.items()}


def build_feature_table(urls) -> pd.DataFrame:
    """Extract features for a Series or list of addresses.

    Returns a DataFrame with one row per address and the columns in
    FEATURE_NAMES order.
    """
    rows = [extract_one(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def main(argv=None):
    """Command line entry point.

    With no arguments, runs a short self-check on hand-picked addresses so you
    can see the extractor behaving before pointing it at real data.

    With a CSV path, extracts features for every row and writes the result. Do
    this once and work from the saved file: extraction over the full base
    dataset takes roughly ten minutes, and there is no reason to pay that cost
    on every experiment.

        python url_features.py malicious_phish.csv features.csv
    """
    import sys
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        samples = [
            "https://www.paypal.com/signin",
            "http://paypa1-secure.verify-account.tk/login",
            "http://192.168.1.1/verify/account",
            "bit.ly/3xR2p",
            "https://xn--pypal-4ve.com/login",
            "www.example.com/a/b/c?id=12345",
        ]
        table = build_feature_table(samples)
        table.insert(0, "url", samples)
        pd.set_option("display.width", 220)
        cols = ["url", "url_length", "has_ip_host", "suspicious_tld",
                "min_brand_edit_distance", "brand_lookalike",
                "brand_is_registered_domain", "phish_hints",
                "is_shortening_service", "has_punycode"]
        print(table[cols].to_string(index=False))
        print(f"\n{len(FEATURE_NAMES)} features extracted per address.")
        print("\nUsage:")
        print("  python url_features.py DATA.csv OUT.csv")
        print("  python url_features.py DATA.csv OUT.csv --reference tranco.csv")
        return

    # Optional reference list. Passing it switches the brand, rank and TLD
    # features from the small built-in lists to the external source.
    ref_path = None
    if "--reference" in argv:
        i = argv.index("--reference")
        ref_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    in_path = argv[0]
    out_path = argv[1] if len(argv) > 1 else "features.csv"

    if ref_path:
        reference.load(ref_path)
    else:
        print("No --reference given: using the small built-in brand and TLD "
              "lists. Brand and TLD features will be weak.")

    raw = pd.read_csv(in_path)
    if "url" not in raw.columns:
        raise SystemExit(f"{in_path} has no 'url' column. Found: {list(raw.columns)}")

    print(f"Read {len(raw):,} rows from {in_path}")

    # Deduplicate before extraction, not after. Duplicate addresses inflate the
    # apparent size of the dataset and leak between the train and test splits if
    # left in, which quietly flatters every number reported afterwards.
    before = len(raw)
    raw = raw.drop_duplicates(subset=["url"]).reset_index(drop=True)
    if before != len(raw):
        print(f"Dropped {before - len(raw):,} duplicate addresses")

    features = build_feature_table(raw["url"])

    # Carry the address through so individual predictions can be traced back to
    # their source row during error analysis.
    features.insert(0, "url", raw["url"].values)

    # The base dataset labels four classes; the detection task is binary.
    # Keeping both columns lets us train on the binary label while still
    # reporting cluster composition against the original four.
    if "type" in raw.columns:
        features["type"] = raw["type"].values
        features["label"] = (raw["type"] != "benign").astype(int).values
        print("\nClass balance:")
        print(raw["type"].value_counts().to_string())
        print(f"\nBinary: {features['label'].mean():.1%} malicious")

    features.to_csv(out_path, index=False)
    print(f"\nWrote {len(features):,} rows x {len(features.columns)} columns to {out_path}")


if __name__ == "__main__":
    main()