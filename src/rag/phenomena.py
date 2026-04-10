"""
Heuristics for detecting informality phenomena in Spanish source sentences,
and lexical overlap scoring between a query and a training example source.

These run on the query only — training examples use their ground-truth
annotation columns (phonetic, informal_lexical_item, indexical_density).
"""

import re

# Informal Spanish and Basque words observed in training data
INFORMAL_WORDLIST = {
    "mogollon", "fatal", "fatall", "bueno", "tia", "tio", "po", "venga",
    "ostia", "jolin", "sobera", "laister", "baina", "ta", "ya", "pelmada",
    "juerga", "guay", "majo", "chaval", "coñazo", "hostia", "joder", "joe",
    "bon", "aufait", "aufaaait", "kriston", "kotxino", "kotxinoo",
}

# Stopwords to ignore in lexical overlap
STOPWORDS = {
    "yo", "tu", "el", "la", "los", "las", "un", "una", "en", "de",
    "que", "y", "a", "no", "me", "se", "le", "lo",
}


def detect_phenomena(text: str) -> dict:
    """
    Run heuristics on a Spanish source sentence (query only).

    Returns
    -------
    dict with keys:
        elongation (int):       1 if 3+ repeated characters found, else 0
        informal_lexical (int): 1 if any token in the informal wordlist, else 0
    """
    elongation = 1 if re.search(r"(.)\1{2,}", text) else 0

    tokens = set(re.findall(r"\w+", text.lower()))
    informal_lexical = 1 if tokens & INFORMAL_WORDLIST else 0

    return {"elongation": elongation, "informal_lexical": informal_lexical}


def lexical_overlap(query: str, example_source: str) -> int:
    """
    Count shared non-stopword tokens (lowercased) between query and example_source.
    """
    query_tokens = set(re.findall(r"\w+", query.lower())) - STOPWORDS
    example_tokens = set(re.findall(r"\w+", example_source.lower())) - STOPWORDS
    return len(query_tokens & example_tokens)
