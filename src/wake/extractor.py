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

    def extract_from_exchange(self, user_message: str, assistant_response: str,
                              conversation: list = None) -> List[FactTriple]:
        """Extract facts from a conversation turn.

        If conversation history is provided, reviews the full conversation for
        context (resolves pronouns, accumulates facts). Otherwise falls back
        to single-message extraction.

        Args:
            user_message: What the user said (latest)
            assistant_response: What the assistant replied (latest)
            conversation: Optional list of {role, content} message dicts

        Returns:
            List of FactTriple objects
        """
        # Primary: model reviews conversation (or single exchange as fallback)
        try:
            if conversation and len(conversation) >= 2:
                triples = self._review_conversation(conversation)
            else:
                triples = self._review_exchange(user_message, assistant_response)
        except Exception as e:
            print(f"  [Review] Model review failed: {e}")
            triples = []

        # Supplement: regex catches structured patterns from latest message
        template_triples = self.extract_template(user_message)
        if template_triples:
            seen = {(t.subject.lower(), t.relation.lower()) for t in triples}
            for t in template_triples:
                if (t.subject.lower(), t.relation.lower()) not in seen:
                    triples.append(t)

        # Filter low-quality facts
        triples = self.filter_junk(triples)

        # Tag all triples with source
        source = user_message[:100]
        for t in triples:
            t.source_exchange = source

        return triples

    # AI subject keywords — match when subject describes the AI itself
    _AI_SUBJECTS = [
        "the ai", "the assistant", "the chatbot",
        "the language model", "the conversational ai",
    ]
    # AI identity object patterns — what the AI IS (not what it does/knows)
    _AI_IDENTITY_OBJECTS = [
        "a conversational ai", "a chatbot", "an ai", "a language model",
        "designed to assist", "designed to help", "designed to communicate",
    ]

    def filter_junk(self, triples: List[FactTriple]) -> List[FactTriple]:
        """Remove AI identity facts and tautologies.

        Conservative: only rejects facts matching clear patterns.
        Conversation structure facts are explicitly preserved.
        """
        result = []
        for t in triples:
            subj = t.subject.lower().strip()
            obj = t.object.lower().strip()

            # A: AI identity facts (subject is AI + object describes AI nature)
            if any(kw in subj for kw in self._AI_SUBJECTS):
                if any(pat in obj for pat in self._AI_IDENTITY_OBJECTS):
                    continue

            # B: Tautology (subject == object)
            if subj == obj:
                continue

            result.append(t)

        filtered = len(triples) - len(result)
        if filtered:
            print(f"  [Filter] Removed {filtered} junk fact(s)")
        return result

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
            (r"(?:my name is|i'm called|call me|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
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
            # Family members and pets
            (r"my\s+(son|daughter|wife|husband|partner|brother|sister|mom|dad|mother|father|dog|cat|pet)(?:'s name)?\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             lambda m: f"The user's {m.group(1)}", "is named", lambda m: m.group(2)),
            # Uses/works with
            (r"(?:i use|i work with|i'm using|i am using)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "uses", lambda m: m.group(1).strip()),

            # --- Opinions ---
            # "I think X is Y" — but NOT "I think I should" (false positive guard)
            (r"(?:i think|i believe)\s+(?!i\s)(.+?)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", lambda m: f"thinks {m.group(1)} is", lambda m: m.group(2).strip()),
            # "I prefer X over Y"
            (r"i prefer\s+(.+?)\s+over\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", lambda m: f"prefers {m.group(1).strip()} over", lambda m: m.group(2).strip()),

            # --- Temporal ---
            (r"i graduated (?:from .+ )?in\s+(\d{4})",
             lambda m: "The user", "graduated in", lambda m: m.group(1)),
            (r"i (?:started|began)(?: .+?)? in\s+(\d{4})",
             lambda m: "The user", "started in", lambda m: m.group(1)),
            (r"i was born in\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "was born in", lambda m: m.group(1).strip()),
            (r"i (?:moved|relocated) to\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "moved to", lambda m: m.group(1).strip()),

            # --- Relationships (possessive, for non-name references) ---
            (r"my\s+(sister|brother|partner|wife|husband|mom|dad|mother|father|son|daughter)\s+(?:lives in|is from)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"The user's {m.group(1)}", "lives in", lambda m: m.group(2).strip()),
            (r"my\s+(sister|brother|partner|wife|husband|mom|dad|mother|father|son|daughter)\s+(?:works as|is a)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"The user's {m.group(1)}", "works as", lambda m: m.group(2).strip()),

            # --- Conditions ---
            (r"i(?:'m| am) allergic to\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "is allergic to", lambda m: m.group(1).strip()),
            (r"i speak\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "speaks", lambda m: m.group(1).strip()),
            (r"i(?:'m| am) learning\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "is learning", lambda m: m.group(1).strip()),
            (r"i studied\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "The user", "studied", lambda m: m.group(1).strip()),
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

    def _review_conversation(self, messages: list) -> List[FactTriple]:
        """Review the full conversation and extract all facts.

        Seeing the whole conversation lets the model resolve pronouns
        ("he" → Andre) and accumulate facts across turns.
        """
        # Format conversation, truncate to ~1500 tokens worth (~6000 chars)
        lines = []
        char_budget = 6000
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        convo_text = "\n".join(lines)
        if len(convo_text) > char_budget:
            convo_text = convo_text[-char_budget:]

        prompt_messages = [{
            "role": "user",
            "content": (
                "List only facts explicitly stated in this conversation.\n"
                "Each line: SUBJECT | RELATION | OBJECT\n"
                "Skip anything not directly stated.\n\n"
                f"{convo_text}\n\n"
                "Facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=300, temperature=0.1)
        print(f"  [Review] raw={raw!r}")
        triples = self._parse_triples(raw)
        print(f"  [Review] parsed {len(triples)} triple(s)")
        return triples

    def _review_exchange(self, user_message: str, assistant_response: str) -> List[FactTriple]:
        """Fallback: review a single exchange when no conversation history available."""
        prompt_messages = [{
            "role": "user",
            "content": (
                "List only facts explicitly stated by the user.\n"
                "Each line: SUBJECT | RELATION | OBJECT\n"
                "Skip anything not directly stated.\n\n"
                f'"{user_message}"\n\n'
                "Facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=200, temperature=0.1)
        print(f"  [Review] raw={raw!r}")
        triples = self._parse_triples(raw)
        print(f"  [Review] parsed {len(triples)} triple(s)")
        return triples

    def extract_with_model(self, user_message: str) -> List[FactTriple]:
        """Model-based extraction from a single message (legacy fallback)."""
        prompt_messages = [{
            "role": "user",
            "content": (
                "Extract specific personal facts from this message as subject-relation-object triples.\n"
                "Include: identity, opinions, preferences, temporal events, relationships, conditions.\n\n"
                "Message: " + user_message + "\n\n"
                "Format each fact on its own line as: SUBJECT | RELATION | OBJECT\n"
                "Example: John | lives in | Portland\n"
                "Example: The user | works as | software engineer\n"
                "Example: The user | thinks Python is | better than JavaScript\n\n"
                "Facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=200, temperature=0.1)
        return self._parse_triples(raw)

    # Relations for natural-language parsing, longest first so "is named" matches before "is"
    _RELATIONS = [
        "has a son named", "has a daughter named", "has a dog named", "has a cat named",
        "has a brother named", "has a sister named",
        "is interested in", "is allergic to", "is based in", "is learning",
        "is named", "is called", "goes by",
        "lives in", "is from", "moved to",
        "works as", "works at", "works for",
        "has a", "has an", "has",
        "loves", "likes", "enjoys", "prefers", "dislikes", "hates",
        "produces", "makes", "creates",
        "speaks", "studied", "plays",
        "is",  # Generic — must be last
    ]
    _RELATION_RE = re.compile(
        r"^(.+?)\s+(" + "|".join(re.escape(r) for r in _RELATIONS) + r")\s+(.+)$",
        re.IGNORECASE,
    )

    # Category labels that 3B models use as headers instead of real subjects
    _CATEGORY_LABELS = frozenset([
        "age", "family", "location", "job", "preferences", "name",
        "occupation", "relationship", "relationships", "hobbies",
        "interests", "education", "work", "home", "pets", "children",
        "conversation context", "context", "identity", "personal",
    ])

    # Patterns in objects that signal commentary, not facts
    _COMMENTARY_RE = re.compile(
        r"(?:not (?:explicitly |directly )?mentioned|implied|"
        r"not stated|unknown|unclear|no (?:specific|explicit)|"
        r"RELATION|OBJECT|SUBJECT|N/A|none|n/a)",
        re.IGNORECASE,
    )

    def _parse_triples(self, raw: str) -> List[FactTriple]:
        """Parse model output into triples — strict validation.

        Handles pipe format (preferred) and natural language fallback.
        Rejects placeholders, commentary, category headers, and inferences.
        """
        triples = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("NONE"):
                return []

            # Strip number prefixes, bullets
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            for prefix in ("- ", "* ", "• "):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if not line:
                continue

            # Skip commentary lines
            lower = line.lower()
            if lower.startswith(("note:", "note ", "however", "extracted facts",
                                 "here are", "for the first", "for the second",
                                 "for the third", "there are no", "no additional",
                                 "not explicitly", "not mentioned", "not stated",
                                 "implied", "based on")):
                continue

            # Skip lines with placeholder tokens
            if self._COMMENTARY_RE.search(line):
                continue

            # Strip "Category: ..." prefix (e.g., "Family: The user has a son")
            colon_match = re.match(r"^(\w[\w\s]{0,25}):\s*(.+)$", line)
            if colon_match:
                prefix_word = colon_match.group(1).strip().lower()
                if prefix_word in self._CATEGORY_LABELS:
                    line = colon_match.group(2).strip()
                    if not line:
                        continue

            # Try 1: pipe format (3+ parts)
            triple = None
            if "|" in line:
                parts = [p.strip().rstrip(".") for p in line.split("|")]
                if len(parts) >= 3 and all(parts[:3]):
                    triple = (parts[0], parts[1], parts[2])
                elif len(parts) == 2 and all(parts):
                    # 2-part pipe: "The user | is Vladimir" → parse as natural
                    line = parts[0].strip() + " " + parts[1].strip()

            # Try 2: natural language — "The user is named Vladimir"
            if triple is None:
                sentence = line.rstrip(".")
                m = self._RELATION_RE.match(sentence)
                if m:
                    triple = (m.group(1).strip(), m.group(2).strip(),
                              m.group(3).strip().rstrip("."))

            if triple is None:
                continue

            subject, relation, obj = triple

            # Validate: reject if any part is empty or too short/long
            if not subject or not relation or not obj:
                continue
            if len(obj) < 2 or len(obj) > 80:
                continue

            # Validate: reject category-label subjects
            if subject.lower().strip() in self._CATEGORY_LABELS:
                continue

            # Validate: reject commentary objects
            if self._COMMENTARY_RE.search(obj):
                continue

            # Validate: reject parenthetical commentary in objects
            if "(" in obj and any(kw in obj.lower() for kw in
                                  ("since ", "implied", "because ", "as he", "as she")):
                continue

            # Validate: reject objects that start with negation/uncertainty
            obj_lower = obj.lower()
            if obj_lower.startswith(("not ", "no ", "never ", "unknown", "unclear")):
                continue

            # Validate: reject sentence-like objects (contain ", and" or multiple clauses)
            if ", and " in obj or "; " in obj:
                continue

            # Validate: reject subjects with "context" (model meta-commentary)
            if "context" in subject.lower():
                continue

            triples.append(FactTriple(
                subject=subject, relation=relation, object=obj,
            ))

        return triples

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        """Normalize subject to canonical form for deduplication.

        Maps pronoun-based and descriptive subjects to 'the user'.
        """
        lower = subject.lower().strip()

        if lower in ("his name", "her name", "my name", "their name",
                      "the user's name", "the user name", "user"):
            return "the user"
        if lower in ("he", "she", "him", "her", "his", "they", "them"):
            return "the user"

        return lower

    _NAME_RELATIONS = frozenset(["is", "is named", "is called", "goes by"])

    def _dedup_key(self, triple) -> tuple:
        """Compute a normalized deduplication key for a triple.

        Normalizes subject aliases and collapses name-type facts so that
        'Vladimir | is named | Vladimir', 'His name | is | Vladimir',
        and 'The user\\'s name | is | Vladimir' all map to the same key.
        """
        subj = self._normalize_subject(triple.subject)
        rel = triple.relation.lower().strip()

        # For name-type facts where subject normalizes to 'the user':
        # collapse all identity relations to a single canonical key
        if subj == "the user" and rel in self._NAME_RELATIONS:
            return ("the user", "is named")

        return (subj, rel)

    def deduplicate(self, new_triples: List[FactTriple], existing_facts: List[FactTriple]) -> List[FactTriple]:
        """Filter out already-known facts using normalized dedup keys.

        Changed objects = updates (not duplicates) and should trigger re-edit.

        Args:
            new_triples: Newly extracted triples
            existing_facts: Facts already in the ledger

        Returns:
            Filtered list of genuinely new or updated triples
        """
        existing_keys = {}
        for fact in existing_facts:
            key = self._dedup_key(fact)
            existing_keys[key] = fact.object.lower()

        result = []
        for triple in new_triples:
            key = self._dedup_key(triple)
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
            "i think", "i believe", "i graduated", "i started",
            "i was born", "i moved", "my sister", "my brother",
            "my partner", "my wife", "my husband", "my mom", "my dad",
            "allergic to", "i speak", "i'm learning", "i studied",
        ]
        return any(m in lower for m in markers)
