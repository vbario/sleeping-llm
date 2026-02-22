"""Fact extractor — extracts structured facts from conversation exchanges.

Runs during the wake phase after each exchange, producing FactTriple objects
that are immediately injected via MEMIT. Uses template-based extraction first
(fast, reliable), with optional model-based fallback.
"""

import re
from typing import List

from src.memory.memit import FactTriple


class FactExtractor:
    """Extracts FactTriple objects from conversation exchanges."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def extract_from_exchange(self, user_message: str, assistant_response: str) -> List[FactTriple]:
        """Primary method — extract facts from a single conversation turn.

        Called after each exchange. Returns a batch of related triples (the "episode").

        Args:
            user_message: What the user said
            assistant_response: What the assistant replied

        Returns:
            List of FactTriple objects
        """
        # Template extraction first (instant, reliable)
        triples = self.extract_template(user_message)

        # If templates found nothing and message has personal-info markers, try model
        if not triples and self._has_personal_markers(user_message):
            triples = self.extract_with_model(user_message)

        # Tag all triples with source
        source = user_message[:100]
        for t in triples:
            t.source_exchange = source

        return triples

    def extract_template(self, text: str) -> List[FactTriple]:
        """Regex-based extraction producing FactTriple format.

        Adapted from curator._extract_facts_template() patterns but outputs
        FactTriple instead of Q&A pairs.
        """
        triples = []
        seen = set()

        # (regex_pattern, subject_fn, relation, object_group_index)
        # subject_fn: callable that returns the subject string, or None for "The user"
        patterns = [
            # Names
            (r"(?:my name is|i'm called|call me|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             lambda m: m.group(1), "is named", lambda m: m.group(1)),
            # Age
            (r"(?:i'm|i am)\s+(\d{1,3})\s+(?:years old|yr|yrs)",
             lambda m: "The user", "is aged", lambda m: m.group(1)),
            # Location
            (r"(?:i live in|i'm from|i'm based in|i am from|i am based in|i live at)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "lives in", lambda m: m.group(1).strip()),
            # Job/profession
            (r"(?:i work as|i'm a|i am a|my job is)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "works as", lambda m: m.group(1).strip()),
            # Works at/for
            (r"(?:i work at|i work for)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "works at", lambda m: m.group(1).strip()),
            # Likes
            (r"(?:i (?:really )?like|i love|i enjoy|i prefer)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "likes", lambda m: m.group(1).strip()),
            # Dislikes
            (r"(?:i (?:don't|do not) like|i hate|i dislike)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "dislikes", lambda m: m.group(1).strip()),
            # Favorites
            (r"my (?:favorite|favourite)\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", lambda m: f"'s favorite {m.group(1)} is", lambda m: m.group(2).strip()),
            # Has/owns
            (r"(?:i have|i've got|i own)\s+(?:a |an )?(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "has", lambda m: m.group(1).strip()),
            # Family members
            (r"my\s+(son|daughter|wife|husband|partner|brother|sister|mom|dad|mother|father)(?:'s name)?\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             lambda m: f"The user's {m.group(1)}", "is named", lambda m: m.group(2)),
            # Uses/works with
            (r"(?:i use|i work with|i'm using|i am using)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "uses", lambda m: m.group(1).strip()),
        ]

        for pattern, subject_fn, relation, object_fn in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject = subject_fn(match)
                    # Handle relation as callable or string
                    rel = relation(match) if callable(relation) else relation
                    obj = object_fn(match)
                except (IndexError, AttributeError):
                    continue

                # Skip very short or very long matches
                if not obj or len(obj) < 2 or len(obj) > 100:
                    continue

                key = (subject.lower(), rel.lower(), obj.lower())
                if key not in seen:
                    seen.add(key)
                    triples.append(FactTriple(
                        subject=subject,
                        relation=rel,
                        object=obj,
                    ))

        return triples

    def extract_with_model(self, user_message: str) -> List[FactTriple]:
        """Model-based extraction fallback.

        Only called when templates find nothing and message has personal-info markers.
        Prompts the model to output structured triples.
        """
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    "Extract specific personal facts from this message as subject-relation-object triples.\n\n"
                    "Message: " + user_message + "\n\n"
                    "Format each fact on its own line as: SUBJECT | RELATION | OBJECT\n"
                    "Example: John | lives in | Portland\n"
                    "Example: The user | works as | software engineer\n\n"
                    "Facts:"
                ),
            }
        ]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=200, temperature=0.1)

        triples = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                subject = parts[0]
                relation = parts[1]
                obj = parts[2]
                if subject and relation and obj and len(obj) >= 2:
                    triples.append(FactTriple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                    ))

        return triples

    def deduplicate(self, new_triples: List[FactTriple], existing_facts: List[FactTriple]) -> List[FactTriple]:
        """Filter out already-known facts by (subject, relation) key.

        Changed objects = updates (not duplicates) and should trigger re-edit.

        Args:
            new_triples: Newly extracted triples
            existing_facts: Facts already in the ledger

        Returns:
            Filtered list of genuinely new or updated triples
        """
        existing_keys = {}
        for fact in existing_facts:
            key = (fact.subject.lower(), fact.relation.lower())
            existing_keys[key] = fact.object.lower()

        result = []
        for triple in new_triples:
            key = (triple.subject.lower(), triple.relation.lower())
            existing_obj = existing_keys.get(key)
            if existing_obj is None:
                # Completely new fact
                result.append(triple)
            elif existing_obj != triple.object.lower():
                # Updated fact (object changed) — should re-edit
                result.append(triple)
            # else: exact duplicate — skip

        return result

    def _has_personal_markers(self, text: str) -> bool:
        """Check if text contains personal information markers."""
        lower = text.lower()
        markers = [
            "my name", "i am", "i'm", "i live", "i work", "i like",
            "i have", "i use", "my favorite", "my favourite",
            "remember", "i prefer", "i want you to",
        ]
        return any(m in lower for m in markers)
