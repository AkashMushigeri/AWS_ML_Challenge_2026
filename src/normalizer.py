"""Text Normalization Module for Business Entities.

Member 2 (Data + Blocking) ownership.
Implements deterministic, reproducible string normalization for business names,
addresses, and open-set country attributes.

Key Features:
- Lowercase and Unicode decomposition (NFKD).
- Stripping Latin diacritics/accents (e.g., 'e', 'c') while strictly preserving Indic scripts
  (Devanagari, Tamil, Telugu, etc.) intact.
- Conjunction and symbol normalization (&, + -> 'and').
- Legal suffix standardization and core business brand root extraction.
- Address abbreviation expansion and numeric/postal code preservation.
- Open-set country normalization (never hardcodes country lists; supports US, India, France, etc.).
"""

import re
import unicodedata
from typing import List, Optional, Set
import pandas as pd

# Common legal entity suffixes across US, India, UK, France, and international jurisdictions
LEGAL_SUFFIXES: Set[str] = {
    # English / Global
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "pvt",
    "private",
    "llc",
    "llp",
    "pllc",
    "co",
    "company",
    # German / European
    "gmbh",
    "ag",
    "plc",
    # French
    "sa",
    "sarl",
    "sas",
    "sasu",
    "snc",
    "sci",
    "eurl",
    "gie",
    # Other abbreviations
    "pc",
    "lp",
}

# Standard address abbreviations mapping to canonical words
ADDRESS_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "hwy": "highway",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "dept": "department",
    "dist": "district",
    "opp": "opposite",
    "nr": "near",
    "rte": "route",
    "ngr": "nagar",
    "col": "colony",
    "w": "west",
    "e": "east",
    "n": "north",
    "s": "south",
}


def remove_diacritics(text: str) -> str:
    """Normalize Unicode diacritics and accents.

    - Strips Latin combining diacritics (e.g. 'e', 'c', 'o' -> 'e', 'c', 'o')
    - Expands ligatures ('oe' -> 'oe', 'ae' -> 'ae', 'ss' -> 'ss')
    - Preserves non-Latin scripts (e.g., Devanagari / Hindi vowels and matras) intact.
    """
    if not text or not isinstance(text, str):
        return ""

    # Expand ligatures
    text = text.replace("œ", "oe").replace("æ", "ae").replace("ß", "ss")

    chars: List[str] = []
    for ch in text:
        decomposed = unicodedata.normalize("NFKD", ch)
        # Only strip combining marks if the base character was ASCII (Latin)
        if len(decomposed) > 1 and ord(decomposed[0]) < 128:
            chars.append(decomposed[0])
        else:
            chars.append(ch)
    return "".join(chars)


def normalize_business_name(name: str) -> str:
    """Deterministic normalization for business names.

    - Lowercase and Unicode decomposition
    - Ampersand & plus normalization
    - Punctuation removal (while preserving Indic unicode letters)
    - Whitespace normalization
    - Standard legal suffix handling (trims leading and trailing legal forms)
    """
    if not name or not isinstance(name, str):
        return ""

    text = remove_diacritics(name).lower()

    # Normalize conjunctions
    text = re.sub(r"[\+&]", " and ", text)

    # Remove non-alphanumeric punctuation, keeping spaces, ASCII alphanumerics,
    # and Indic script blocks (\u0900-\u0D7F covers Devanagari, Bengali, Tamil, etc.)
    text = re.sub(r"[^\w\s\u0900-\u0D7F]", " ", text)

    # Tokenize and clean legal suffixes
    tokens = text.split()
    if not tokens:
        return ""

    # Strip leading legal suffixes (e.g. 'llc moncada learning center' -> 'moncada learning center')
    start = 0
    while start < len(tokens) and tokens[start] in LEGAL_SUFFIXES:
        start += 1

    # Strip trailing legal suffixes if there are other tokens
    end = len(tokens)
    while end > start and tokens[end - 1] in LEGAL_SUFFIXES:
        end -= 1

    core = tokens[start:end]
    # If all tokens were legal words, fallback to full token list
    if not core:
        core = tokens

    return " ".join(core)


def extract_core_name_tokens(name: str) -> List[str]:
    """Return core discriminative tokens from a normalized business name."""
    norm = normalize_business_name(name)
    tokens = norm.split()
    # Filter very generic filler words if length > 1
    filler = {"and", "the", "of", "in", "at", "for", "a", "an", "to", "by", "with"}
    core = [t for t in tokens if t not in filler and t not in LEGAL_SUFFIXES]
    return core if core else tokens


def extract_core_name(name: str) -> str:
    """Return the joined core business brand name."""
    tokens = extract_core_name_tokens(name)
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """Deterministic normalization for business addresses.

    - Lowercase and Unicode decomposition
    - Standardize common road/unit abbreviations
    - Punctuation cleanup (preserving Indic script and alphanumeric numbers)
    - Preserve digits (street numbers, unit numbers, postal/PIN codes)
    """
    if not address or not isinstance(address, str):
        return ""

    text = remove_diacritics(address).lower()

    # Normalize conjunctions
    text = re.sub(r"[\+&]", " and ", text)

    # Remove punctuation, keep alphanumeric, Indic range, and spaces
    text = re.sub(r"[^\w\s\u0900-\u0D7F]", " ", text)

    tokens = text.split()
    norm_tokens = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in tokens]

    return " ".join(norm_tokens)


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric components (PIN codes, ZIP codes, street numbers)."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b\d+\b", text)


def extract_postal_code(text: str) -> str:
    """Extract 5-digit or 6-digit postal/PIN/ZIP code if present."""
    if not text or not isinstance(text, str):
        return ""
    matches = re.findall(r"\b\d{5,6}\b", text)
    return matches[0] if matches else ""


def normalize_country(country: str) -> str:
    """Deterministic open-set country normalization.

    - Strip whitespace, uppercase
    - Does NOT restrict to US/India; seamlessly handles France or unseen labels.
    - Fallback to 'UNKNOWN' on missing/empty values.
    """
    if country is None or not isinstance(country, str):
        return "UNKNOWN"
    norm = country.strip().upper()
    return norm if norm else "UNKNOWN"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies deterministic normalization across a DataFrame."""
    df = df.copy()
    if "business_name" in df.columns:
        df["clean_name"] = df["business_name"].fillna("").astype(str).map(normalize_business_name)
        df["core_name"] = df["clean_name"].map(extract_core_name)
    else:
        df["clean_name"] = ""
        df["core_name"] = ""

    if "business_address" in df.columns:
        df["clean_address"] = df["business_address"].fillna("").astype(str).map(normalize_address)
        df["postal_code"] = df["business_address"].fillna("").astype(str).map(extract_postal_code)
        df["numeric_tokens"] = df["business_address"].fillna("").astype(str).map(extract_numeric_tokens)
    else:
        df["clean_address"] = ""
        df["postal_code"] = ""
        df["numeric_tokens"] = [[] for _ in range(len(df))]

    if "country" in df.columns:
        df["clean_country"] = df["country"].fillna("").astype(str).map(normalize_country)
    else:
        df["clean_country"] = "UNKNOWN"

    return df
