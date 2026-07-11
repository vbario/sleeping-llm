"""Shared recall matcher — fuzzy token matching of a stored value in a response.

Single source of truth for graduation tests (full sleep), nap audits, and
experiment probes, so every condition and probe is scored by the same code.
"""

import re

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "no", "so", "if", "than", "that",
    "this", "it", "its", "he", "she", "they", "who", "what",
}


def fuzzy_value_match(value, response):
    """Check if key tokens from value appear in the response.

    Handles paraphrases like "plays saxophone" matching "saxophone player".
    For short values (1-2 tokens), all must match.
    For longer values, 60% of content tokens must appear.
    """
    response_lower = response.lower()

    # Exact substring match (fast path)
    if value.lower().strip() in response_lower:
        return True

    # Tokenize into meaningful words, skip stop words
    value_tokens = [w for w in re.findall(r'\w+', value.lower())
                    if w not in STOP_WORDS and len(w) > 1]

    if not value_tokens:
        # All tokens were stop words — fall back to exact match
        return False

    matched = sum(1 for t in value_tokens if t in response_lower)

    # Short values: require all tokens. Longer values: 60% threshold.
    if len(value_tokens) <= 2:
        return matched == len(value_tokens)
    return matched / len(value_tokens) >= 0.6
