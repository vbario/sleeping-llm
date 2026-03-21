"""Fact extractor — model-based Q&A fact extraction from conversation.

Runs during the wake phase after each exchange, producing QAPair objects
that are buffered and eventually persisted to the FactLedger. Uses the
model itself to determine what new facts were learned from the latest
user message, given the full conversation context.
"""

import re
from typing import List

from src.memory.facts import QAPair


class FactExtractor:
    """Extracts QAPair objects from conversation exchanges using the model."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def extract_from_exchange(self, user_message: str, assistant_response: str,
                              conversation: list = None) -> List[QAPair]:
        """Extract facts from a conversation turn using the model.

        Feeds the conversation up to this point into the model and asks
        what new concrete facts were learned from the user's last message.

        Returns:
            List of QAPair objects
        """
        try:
            facts = self._extract_new_facts(user_message, conversation)
        except Exception as e:
            print(f"  [Extract] Model extraction failed: {e}")
            facts = []

        # Filter low-quality facts
        facts = self._filter_junk(facts)

        # Tag all facts with source
        source = user_message[:100]
        for f in facts:
            f.source_exchange = source

        return facts

    # --- Model-based extraction ---

    def _extract_new_facts(self, user_message: str,
                           conversation: list = None) -> List[QAPair]:
        """Ask the model what new facts were learned from the last message."""
        # Build conversation context
        if conversation and len(conversation) >= 2:
            lines = []
            char_budget = 4000
            for msg in conversation:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            convo_text = "\n".join(lines)
            if len(convo_text) > char_budget:
                convo_text = convo_text[-char_budget:]
        else:
            convo_text = f"User: {user_message}"

        prompt_messages = [{
            "role": "user",
            "content": (
                "Read the conversation below. List the new facts we learned "
                "from the user's LAST message as short statements.\n\n"
                "Only concrete facts about people, places, jobs, ages, "
                "preferences, relationships, pets, dates.\n"
                "Write each fact as a simple sentence. No questions. "
                "No commentary. If no new facts, write NONE.\n\n"
                "Example:\n"
                "1. Viktor lives in Berlin.\n"
                "2. Viktor is a software engineer.\n\n"
                f"{convo_text}\n\n"
                "New facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=200, temperature=0.1)
        print(f"  [Extract] raw={raw!r}")
        facts = self._parse_statements(raw)
        print(f"  [Extract] parsed {len(facts)} fact(s)")
        return facts

    # --- Parsing ---

    # Patterns in answers that signal commentary, not facts
    _COMMENTARY_RE = re.compile(
        r"(?:not (?:explicitly |directly )?mentioned|implied|"
        r"not stated|unknown|unclear|no (?:specific|explicit)|"
        r"N/A|none|n/a)",
        re.IGNORECASE,
    )

    def _parse_statements(self, raw: str) -> List[QAPair]:
        """Parse numbered/bulleted statements into QAPairs.

        Input like:
            1. Jimmy Jimz is a music producer from Seattle.
            2. Jimmy Jimz makes rap beats.

        Each statement becomes a QAPair where question=statement,
        answer=statement, value=statement (dedup uses the value).
        """
        facts = []

        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("NONE"):
                return []

            # Strip numbering and bullets
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            for prefix in ("- ", "* ", "• "):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Skip commentary
            lower = line.lower()
            if lower.startswith(("note:", "note ", "however", "there are no",
                                 "no additional", "not explicitly", "based on",
                                 "no new", "i did not", "i could not",
                                 "the user", "from the")):
                continue
            if self._COMMENTARY_RE.search(line):
                continue

            # Skip Q:/A: lines (model fell back to Q&A format)
            if re.match(r"^[QqAa][:.]\s", line):
                continue

            # Must be a real sentence (at least 3 words)
            line = line.rstrip(".")
            if len(line.split()) < 3:
                continue
            if len(line) < 5:
                continue

            facts.append(QAPair(
                question=line,
                answer=line,
                value=self._extract_value(line),
            ))

        return facts

    @staticmethod
    def _extract_value(statement: str) -> str:
        """Extract the key distinguishing value from a fact statement.

        "Jazzy Mike is a saxophone player" → "saxophone player"
        "Jazzy Mike is from Toronto, Canada" → "Toronto, Canada"
        "Viktor lives in Berlin" → "Berlin"
        "She has a dog named Rex" → "Rex"
        """
        s = statement.strip().rstrip(".")

        # Order matters: longer patterns first to avoid premature matches
        patterns = [
            # "X has a Y named/called Z" → Z
            (r'^.+?\s+has\s+(?:a|an)\s+.+?\s+(?:named|called)\s+(.+)$', 1),
            # "X has a hit song called Z" → Z
            (r'^.+?\s+has\s+(?:a|an)\s+.+?\s+called\s+(.+)$', 1),
            # "X is from Y" → Y
            (r'^.+?\s+is\s+from\s+(.+)$', 1),
            # "X lives in Y" → Y
            (r'^.+?\s+lives\s+in\s+(.+)$', 1),
            # "X works at/as Y" → Y
            (r'^.+?\s+works\s+(?:at|as)\s+(.+)$', 1),
            # "X is a/an Y" → Y
            (r'^.+?\s+is\s+(?:a|an)\s+(.+)$', 1),
            # "X is aged/allergic to Y" → Y
            (r'^.+?\s+is\s+(?:aged|allergic\s+to|learning|named)\s+(.+)$', 1),
            # "X is Y" (catch-all for copula) → Y
            (r'^.+?\s+is\s+(.+)$', 1),
            # "X has Y" → Y
            (r'^.+?\s+has\s+(.+)$', 1),
            # "X likes/makes/plays Y" → Y
            (r'^.+?\s+(?:likes|makes|plays|speaks|prefers|uses|studied)\s+(.+)$', 1),
        ]

        for pattern, group in patterns:
            m = re.match(pattern, s, re.IGNORECASE)
            if m:
                return m.group(group).strip()

        # Fallback: last 2-3 words
        words = s.split()
        if len(words) > 3:
            return " ".join(words[-3:])
        return s

    # --- Junk filtering ---

    _AI_ANSWER_PATTERNS = [
        "a conversational ai", "a chatbot", "an ai", "a language model",
        "designed to assist", "designed to help", "designed to communicate",
    ]

    _META_QUESTION_RE = re.compile(
        r"(?:is the user (?:responding|making a statement|referring to|asking)|"
        r"is there (?:a user|something)|"
        r"does the user have (?:something specific to say|an opinion)|"
        r"are (?:any (?:details|facts)|there any)|"
        r"does the user (?:recall|remember) things from previous|"
        r"is there something on (?:the user'?s|which))",
        re.IGNORECASE,
    )

    _COMMENTARY_ANSWER_RE = re.compile(
        r"(?:i couldn'?t find|no information|not (?:specified|provided|mentioned|stated|clear)|"
        r"no (?:specific |explicit )?(?:information|details|data)|"
        r"there is no|cannot determine|unable to determine)",
        re.IGNORECASE,
    )

    def _filter_junk(self, facts: List[QAPair]) -> List[QAPair]:
        """Remove low-quality facts."""
        result = []
        for f in facts:
            answer_lower = f.answer.lower().strip()
            value_lower = f.value.lower().strip()
            question_lower = f.question.lower().strip()

            if any(pat in answer_lower for pat in self._AI_ANSWER_PATTERNS):
                continue
            if not value_lower or len(value_lower) < 2:
                continue
            if self._META_QUESTION_RE.search(question_lower):
                continue
            if self._COMMENTARY_ANSWER_RE.search(answer_lower):
                continue

            stripped = answer_lower.rstrip(".,!? ")
            if stripped in ("yes", "no", "true", "false"):
                continue

            result.append(f)

        filtered = len(facts) - len(result)
        if filtered:
            print(f"  [Filter] Removed {filtered} junk fact(s)")
        return result

    # --- Deduplication ---

    _STRIP_WORDS_RE = re.compile(
        r"\b(?:the|user'?s?|your|my|do you|does the user|what is|what are|what does|"
        r"have any|have a)\b",
        re.IGNORECASE,
    )

    _STOP_WORDS = frozenset(
        "a an the is are was were do does did has have had i my me "
        "you your he she it we they at in on of to for and or but "
        "as by with from that this yes no not".split()
    )

    def _dedup_key(self, fact: QAPair) -> str:
        return fact.question.lower().strip()

    def _value_key(self, fact: QAPair) -> str:
        v = fact.value.lower().strip().rstrip(".,!? ")
        v = re.sub(r"^(?:yes,?\s+|no,?\s+|a\s+|an\s+|the\s+)", "", v)
        return re.sub(r"\s+", " ", v)

    def _semantic_question_key(self, fact: QAPair) -> str:
        q = fact.question.lower().strip().rstrip("?.,! ")
        q = self._STRIP_WORDS_RE.sub("", q)
        return re.sub(r"\s+", " ", q).strip()

    @staticmethod
    def _value_contained(val: str, value_set: set) -> bool:
        if len(val) < 5:
            return False
        for existing in value_set:
            if len(existing) < 5:
                continue
            if val in existing or existing in val:
                return True
        return False

    @classmethod
    def _value_overlaps(cls, val: str, value_set: set) -> bool:
        val_words = {w for w in val.split() if w not in cls._STOP_WORDS and len(w) > 1}
        if len(val_words) < 3:
            return False
        for existing in value_set:
            ex_words = {w for w in existing.split() if w not in cls._STOP_WORDS and len(w) > 1}
            if len(ex_words) < 3:
                continue
            shorter, longer = (val_words, ex_words) if len(val_words) <= len(ex_words) else (ex_words, val_words)
            if len(shorter & longer) / len(shorter) >= 0.6:
                return True
        return False

    def deduplicate(self, new_facts: List[QAPair], existing_facts: List[QAPair]) -> List[QAPair]:
        """Filter out already-known facts."""
        existing_q_keys = {}
        existing_sem_keys = set()
        existing_values = set()
        for f in existing_facts:
            existing_q_keys[self._dedup_key(f)] = self._value_key(f)
            existing_sem_keys.add(self._semantic_question_key(f))
            existing_values.add(self._value_key(f))

        result = []
        seen_values = set()
        for f in new_facts:
            q_key = self._dedup_key(f)
            sem_key = self._semantic_question_key(f)
            val_key = self._value_key(f)

            if not val_key or len(val_key) < 2:
                continue
            existing_val = existing_q_keys.get(q_key)
            if existing_val is not None and existing_val == val_key:
                continue
            if sem_key in existing_sem_keys:
                continue
            if val_key in existing_values:
                continue
            if self._value_contained(val_key, existing_values):
                continue
            if self._value_overlaps(val_key, existing_values):
                continue
            if val_key in seen_values:
                continue
            if self._value_contained(val_key, seen_values):
                continue
            if self._value_overlaps(val_key, seen_values):
                continue
            if sem_key in {self._semantic_question_key(r) for r in result}:
                continue

            seen_values.add(val_key)
            result.append(f)

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
