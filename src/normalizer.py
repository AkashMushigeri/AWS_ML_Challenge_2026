"""Text Normalization Module for Business Entities.

Member 2 (Data + Blocking) ownership.
Implements deterministic, reproducible string normalization for names, addresses, and countries.
"""

import re
import unicodedata
from typing import Set

# Common legal suffixes to standardize or strip from core business names
LEGAL_SUFFIXES: Set[str] = {
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
    "co",
    "company",
    "sa",
    "sarl",
    "gmbh",
    "plc",
}

# Standard address abbreviations mapping to canonical words
ADDRESS_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "hwy": "highway",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "w": "west",
    "e": "east",
    "n": "north",
    "s": "south",
}


def remove_diacritics(text: str) -> str:
    """Normalize Unicode diacritics and accents (e.g., e, a -> e, a)."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def normalize_business_name(name: str) -> str:
    """Deterministic normalization for business names.

    - Lowercase and Unicode decomposition
    - Ampersand & plus normalization
    - Punctuation removal
    - Whitespace normalization
    - Standard legal suffix handling
    """
    if not name or not isinstance(name, str):
        return ""

    text = remove_diacritics(name).lower()

    # Normalize conjunctions
    text = re.sub(r"[&+]", " and ", text)

    # Remove non-alphanumeric characters except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize and clean legal suffixes
    tokens = text.split()
    if not tokens:
        return ""

    # Strip trailing legal suffixes if there are other tokens
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def extract_core_name_tokens(name: str) -> list[str]:
    """Return core discriminative tokens from a normalized business name."""
    norm = normalize_business_name(name)
    tokens = norm.split()
    # Filter very generic filler words if length > 1
    filler = {"and", "the", "of", "in", "at", "for"}
    core = [t for t in tokens if t not in filler and t not in LEGAL_SUFFIXES]
    return core if core else tokens


def normalize_address(address: str) -> str:
    """Deterministic normalization for business addresses.

    - Lowercase and Unicode decomposition
    - Standardize common road/unit abbreviations
    - Punctuation cleanup
    - Preserve digits (street numbers, unit numbers, postal/PIN codes)
    """
    if not address or not isinstance(address, str):
        return ""

    text = remove_diacritics(address).lower()

    # Remove punctuation, keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()
    norm_tokens = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in tokens]

    return " ".join(norm_tokens)


def extract_numeric_tokens(text: str) -> list[str]:
    """Extract numeric components (PIN codes, ZIP codes, street numbers)."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b\d+\b", text)


def normalize_country(country: str) -> str:
    """Deterministic open-set country normalization.

    - Strip whitespace, uppercase
    - Does NOT restrict to US/India; seamlessly handles France or unseen labels.
    """
    if not country or not isinstance(country, str):
        return "UNKNOWN"
    norm = country.strip().upper()
    return norm if norm else "UNKNOWN"
