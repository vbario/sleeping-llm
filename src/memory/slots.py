"""Slots — decide whether two fact questions are about the same thing.

"What is the user's favorite color?" and "What color does the user like?"
ask for the same slot; "What is the user's sister's favorite color?" does
not. The decision is lexical and cheap (no model call):

  1. tokenize the question, drop function words, normalize question words
     (which/how -> what) and synonyms (job/profession -> work, like -> favorite,
     based/located -> live, ...), singularize;
  2. split off RELATION words (son, sister, boss, dog, ...). These anchor the
     subject and must match exactly between two questions;
  3. the remaining content tokens (including the question word, which carries
     the answer type: where/when/what) are compared by Jaccard similarity
     against `facts.slot_similarity_threshold`.

`same_value` says whether two values for the same slot agree, so a repeat
mention is skipped while a changed value supersedes the old entry.
"""

import re
from typing import FrozenSet, Set, Tuple

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "has", "have", "had", "of", "to", "for", "in", "on", "at", "by",
    "with", "about", "from", "and", "or", "that", "this", "it", "its", "there",
    "their", "they", "them", "s", "user", "users", "me", "my", "i", "you",
    "your", "we", "our", "he", "she", "his", "her", "him", "know", "tell",
    "remember", "remind", "can", "could", "would", "please", "again", "really",
    "regarding", "think", "say", "said", "mention", "mentioned", "currently",
    "now", "exactly", "specifically", "person", "someone", "something",
}

QUESTION_WORDS = {"what": "what", "which": "what", "how": "what",
                  "where": "where", "when": "when", "who": "who", "whom": "who"}

SYNONYMS = {
    # preference
    "like": "favorite", "likes": "favorite", "favourite": "favorite", "prefer": "favorite",
    "preferred": "favorite", "fond": "favorite", "love": "favorite", "enjoy": "favorite",
    "enjoys": "favorite", "best": "favorite",
    # residence
    "based": "live", "located": "live", "reside": "live", "resides": "live",
    "living": "live", "lives": "live", "residence": "live", "home": "live", "stay": "live",
    # occupation
    "job": "work", "profession": "work", "occupation": "work", "career": "work",
    "works": "work", "employed": "work", "employment": "work", "role": "work",
    # employer
    "company": "employer", "firm": "employer", "organization": "employer",
    # age
    "age": "old", "aged": "old", "birthday": "born", "birthdate": "born", "birth": "born",
    # misc
    "named": "name", "called": "name", "pets": "pet", "hobbies": "hobby",
    "interests": "hobby", "interest": "hobby", "languages": "language", "speaks": "speak",
    "spoken": "speak", "allergies": "allergic", "allergy": "allergic", "studying": "study",
    "studied": "study", "studies": "study", "learning": "learn", "learns": "learn",
    "drive": "car", "drives": "car", "vehicle": "car", "town": "city", "hometown": "born",
}

RELATION_WORDS = {
    "son", "daughter", "wife", "husband", "partner", "spouse", "girlfriend",
    "boyfriend", "brother", "sister", "sibling", "mom", "mother", "dad",
    "father", "parent", "parents", "grandmother", "grandfather", "grandma",
    "grandpa", "uncle", "aunt", "cousin", "child", "kid", "children", "kids",
    "dog", "cat", "pet", "friend", "boss", "colleague", "coworker", "neighbor",
    "neighbour", "roommate", "manager", "teacher", "doctor",
}
_RELATION_CANON = {"mom": "mother", "dad": "father", "grandma": "grandmother",
                   "grandpa": "grandfather", "kid": "child", "kids": "child",
                   "children": "child", "parents": "parent", "neighbour": "neighbor",
                   "coworker": "colleague", "spouse": "partner", "sibling": "sibling"}

_NO_SINGULAR = ("ss", "us", "is", "ous")


def _singular(tok: str) -> str:
    if len(tok) <= 3 or tok.endswith(_NO_SINGULAR):
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith("es") and tok[-3] in "sxz":
        return tok[:-2]
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def tokenize(question: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Return (relation_tokens, content_tokens) for a question."""
    text = question.lower().replace("’", "'")
    text = re.sub(r"'s\b", " ", text)
    raw = re.findall(r"[a-z0-9]+", text)
    relations: Set[str] = set()
    content: Set[str] = set()
    for tok in raw:
        if tok in RELATION_WORDS:
            relations.add(_RELATION_CANON.get(tok, tok))
            continue
        if tok in QUESTION_WORDS:
            content.add(QUESTION_WORDS[tok])
            continue
        if tok in STOP_WORDS:
            continue
        tok = SYNONYMS.get(tok, tok)
        tok = _singular(tok)
        tok = SYNONYMS.get(tok, tok)
        if tok in RELATION_WORDS:
            relations.add(_RELATION_CANON.get(tok, tok))
        elif tok not in STOP_WORDS:
            content.add(tok)
    return frozenset(relations), frozenset(content)


def canonical_slot(question: str) -> str:
    """A stable string key for the slot a question asks about."""
    relations, content = tokenize(question)
    return "|".join(["+".join(sorted(relations)), "+".join(sorted(content))])


def slot_similarity(q1: str, q2: str) -> float:
    """0.0 when the subjects differ; otherwise Jaccard similarity of content tokens."""
    r1, c1 = tokenize(q1)
    r2, c2 = tokenize(q2)
    if r1 != r2:
        return 0.0
    if not c1 and not c2:
        return 1.0
    union = c1 | c2
    return len(c1 & c2) / len(union) if union else 0.0


def same_slot(q1: str, q2: str, threshold: float = 0.6) -> bool:
    return slot_similarity(q1, q2) >= threshold


def _value_tokens(value: str) -> Set[str]:
    toks = re.findall(r"[a-z0-9]+", (value or "").lower().replace("’", "'"))
    return {_singular(t) for t in toks if t not in STOP_WORDS}


def same_value(v1: str, v2: str) -> bool:
    """Two values agree if they normalize equal or one's content tokens are a
    subset of the other's ("Lisbon" vs "Lisbon, Portugal")."""
    a = re.sub(r"\s+", " ", (v1 or "").lower()).strip(" .,!?;:\"'")
    b = re.sub(r"\s+", " ", (v2 or "").lower()).strip(" .,!?;:\"'")
    if a == b:
        return True
    ta, tb = _value_tokens(a), _value_tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta
