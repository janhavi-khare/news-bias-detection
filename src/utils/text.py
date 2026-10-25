import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace consistently with the BABE training preprocessing.

    Replaces multiple whitespace characters (including newlines and tabs)
    with a single space and strips leading/trailing spaces.
    Matches: babe_clean['text'].str.replace(r'\\s+', ' ', regex=True).str.strip()
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}.")
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    """Split an article or multi-sentence paragraph into individual sentences.

    Preserves sentence integrity across newlines, quotation marks, and standard punctuation.
    Filters out empty or non-alphanumeric fragments.
    """
    if not text or not text.strip():
        return []

    # First split on explicit line breaks to respect paragraph structure
    raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    sentences: List[str] = []

    for paragraph in raw_paragraphs:
        # Regex split on sentence terminators (. ! ?) followed by whitespace and a boundary character
        # (capital letter, digit, quotation mark, opening bracket)
        raw_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\u201c\u2018\(\[])', paragraph)
        for part in raw_parts:
            cleaned = normalize_whitespace(part)
            # Retain segments with substantial content (at least 2 chars and contains alphanumeric)
            if len(cleaned) >= 2 and any(char.isalnum() for char in cleaned):
                sentences.append(cleaned)
            elif sentences and len(cleaned) > 0:
                # Merge small trailing fragments back into the previous sentence
                sentences[-1] = normalize_whitespace(f"{sentences[-1]} {cleaned}")

    # Fallback if no boundary was matched but text is not empty
    if not sentences and text.strip():
        fallback = normalize_whitespace(text)
        if fallback:
            sentences.append(fallback)

    return sentences
