"""Recall matching — did a response actually state a stored value?

Single source of truth for graduation tests, nap audits, bleed probes and
experiment probes, so every path is scored by the same code.

Rules:
  1. The value must appear on word boundaries ("teal" does not match
     "tealight"; "32" does not match "320"). A possessive or plural suffix
     on the value is tolerated ("Andre's", "pilots").
  2. A hedged response does not count. If a hedge marker ("I don't know",
     "I'm not sure", "no information", ...) appears before the value, the
     value is being guessed at or denied, not recalled.
  3. Multi-word values fall back to content tokens: a 1–2 token value needs
     every token; longer values need 60% of their content tokens.
"""

import re
from typing import Iterable, Optional

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "and", "or", "but", "not", "no", "so", "if",
    "than", "that", "this", "it", "its", "he", "she", "they", "who", "what",
    "my", "your", "user", "user's", "named", "name", "called",
}

HEDGE_RE = re.compile(
    r"(?:i(?: do|'m| am)? ?(?:don't|do not|not|never)\s+(?:know|recall|remember|have|think)|"
    r"i'?m not sure|not sure|no (?:information|idea|record|data)|"
    r"(?:don't|do not|cannot|can't|unable to) (?:have |find |determine |recall |access )|"
    r"not (?:mentioned|specified|provided|stated|clear)|"
    r"i (?:haven't|have not) been told|you (?:haven't|have not|never) (?:told|mentioned)|"
    r"i (?:wasn't|was not) (?:told|given)|wasn't able to)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?;:\"'")


def _boundary_pattern(phrase: str) -> re.Pattern:
    # Allow possessive / plural suffix, require non-word chars around.
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?:'s|s)?(?!\w)")


def _first_match_pos(value: str, response: str) -> Optional[int]:
    """Position of the value in the response (phrase or token fallback), or None."""
    m = _boundary_pattern(value).search(response)
    if m:
        return m.start()

    tokens = [t for t in re.findall(r"\w+", value)
              if t not in STOP_WORDS and len(t) > 1]
    if not tokens:
        return None
    positions = []
    for t in tokens:
        tm = _boundary_pattern(t).search(response)
        if tm:
            positions.append(tm.start())
    if len(tokens) <= 2:
        ok = len(positions) == len(tokens)
    else:
        ok = len(positions) / len(tokens) >= 0.6
    return min(positions) if ok and positions else None


def value_in_response(value: str, response: str) -> bool:
    """True if the response states the value (word-boundary, unhedged)."""
    v = normalize(value)
    r = normalize(response)
    if not v or not r:
        return False
    pos = _first_match_pos(v, r)
    if pos is None:
        return False
    hedge = HEDGE_RE.search(r)
    if hedge and hedge.start() < pos:
        return False
    return True


def first_value_in_response(values: Iterable[str], response: str) -> Optional[str]:
    """Return the first value (in iteration order) that the response states, else None."""
    for v in values:
        if value_in_response(v, response):
            return v
    return None
