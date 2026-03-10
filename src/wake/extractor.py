"""Fact extractor — extracts Q&A facts from conversation exchanges.

Runs during the wake phase after each exchange, producing QAPair objects
that are buffered and eventually persisted to the FactLedger. Uses
template-based extraction first (fast, reliable), with model-based
extraction for richer context understanding.
"""

import re
from typing import List

from src.memory.facts import QAPair


class FactExtractor:
    """Extracts QAPair objects from conversation exchanges."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def extract_from_exchange(self, user_message: str, assistant_response: str,
                              conversation: list = None) -> List[QAPair]:
        """Extract facts from a conversation turn.

        If conversation history is provided, reviews the full conversation for
        context (resolves pronouns, accumulates facts). Otherwise falls back
        to single-message extraction.

        Returns:
            List of QAPair objects
        """
        # Primary: model reviews conversation (or single exchange as fallback)
        try:
            if conversation and len(conversation) >= 2:
                facts = self._review_conversation(conversation)
            else:
                facts = self._review_exchange(user_message, assistant_response)
        except Exception as e:
            print(f"  [Review] Model review failed: {e}")
            facts = []

        # Supplement: regex catches structured patterns from latest message
        template_facts = self.extract_template(user_message)
        if template_facts:
            seen = {f.question.lower().strip() for f in facts}
            for f in template_facts:
                if f.question.lower().strip() not in seen:
                    facts.append(f)

        # Filter low-quality facts
        facts = self.filter_junk(facts)

        # Tag all facts with source
        source = user_message[:100]
        for f in facts:
            f.source_exchange = source

        return facts

    # AI-related patterns to filter out
    _AI_ANSWER_PATTERNS = [
        "a conversational ai", "a chatbot", "an ai", "a language model",
        "designed to assist", "designed to help", "designed to communicate",
    ]

    def filter_junk(self, facts: List[QAPair]) -> List[QAPair]:
        """Remove AI identity facts and tautologies."""
        result = []
        for f in facts:
            answer_lower = f.answer.lower().strip()
            value_lower = f.value.lower().strip()

            # Skip AI identity facts
            if any(pat in answer_lower for pat in self._AI_ANSWER_PATTERNS):
                continue

            # Skip empty or trivial values
            if not value_lower or len(value_lower) < 2:
                continue

            result.append(f)

        filtered = len(facts) - len(result)
        if filtered:
            print(f"  [Filter] Removed {filtered} junk fact(s)")
        return result

    def extract_template(self, text: str) -> List[QAPair]:
        """Regex-based extraction producing QAPair format.

        Catches structured patterns like "My name is X", "I live in Y", etc.
        """
        facts = []
        seen = set()

        # (regex, question_fn, answer_fn, value_fn)
        patterns = [
            # Names
            (r"(?:my name is|i'm called|call me|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             lambda m: "What is the user's name?",
             lambda m: f"The user's name is {m.group(1)}.",
             lambda m: m.group(1)),
            # Age
            (r"(?:i'm|i am)\s+(\d{1,3})\s+(?:years old|yr|yrs)",
             lambda m: "How old is the user?",
             lambda m: f"The user is {m.group(1)} years old.",
             lambda m: m.group(1)),
            # Location
            (r"(?:i live in|i'm from|i'm based in|i am from|i am based in|i live at)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "Where does the user live?",
             lambda m: f"The user lives in {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Job/profession
            (r"(?:i work as|i'm a|i am a|my job is)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "What does the user do for work?",
             lambda m: f"The user works as {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Works at/for
            (r"(?:i work at|i work for)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "Where does the user work?",
             lambda m: f"The user works at {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Likes
            (r"(?:i (?:really )?like|i love|i enjoy|i prefer)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user like?",
             lambda m: f"The user likes {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Dislikes
            (r"(?:i (?:don't|do not) like|i hate|i dislike)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user dislike?",
             lambda m: f"The user dislikes {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Favorites
            (r"my (?:favorite|favourite)\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What is the user's favorite {m.group(1)}?",
             lambda m: f"The user's favorite {m.group(1)} is {m.group(2).strip()}.",
             lambda m: m.group(2).strip()),
            # Has/owns
            (r"(?:i have|i've got|i own)\s+(?:a |an )?(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user have?",
             lambda m: f"The user has {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Family members and pets
            (r"my\s+(son|daughter|wife|husband|partner|brother|sister|mom|dad|mother|father|dog|cat|pet)(?:'s name)?\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
             lambda m: f"What is the user's {m.group(1)}'s name?",
             lambda m: f"The user's {m.group(1)} is named {m.group(2)}.",
             lambda m: m.group(2)),
            # Uses/works with
            (r"(?:i use|i work with|i'm using|i am using)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user use?",
             lambda m: f"The user uses {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Opinions
            (r"(?:i think|i believe)\s+(?!i\s)(.+?)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user think about {m.group(1).strip()}?",
             lambda m: f"The user thinks {m.group(1).strip()} is {m.group(2).strip()}.",
             lambda m: m.group(2).strip()),
            # Temporal
            (r"i graduated (?:from .+ )?in\s+(\d{4})",
             lambda m: "When did the user graduate?",
             lambda m: f"The user graduated in {m.group(1)}.",
             lambda m: m.group(1)),
            (r"i (?:started|began)(?: .+?)? in\s+(\d{4})",
             lambda m: "When did the user start?",
             lambda m: f"The user started in {m.group(1)}.",
             lambda m: m.group(1)),
            (r"i was born in\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "When was the user born?",
             lambda m: f"The user was born in {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            (r"i (?:moved|relocated) to\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "Where did the user move to?",
             lambda m: f"The user moved to {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            # Family details
            (r"my\s+(sister|brother|partner|wife|husband|mom|dad|mother|father|son|daughter)\s+(?:lives in|is from)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"Where does the user's {m.group(1)} live?",
             lambda m: f"The user's {m.group(1)} lives in {m.group(2).strip()}.",
             lambda m: m.group(2).strip()),
            (r"my\s+(sister|brother|partner|wife|husband|mom|dad|mother|father|son|daughter)\s+(?:works as|is a)\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: f"What does the user's {m.group(1)} do for work?",
             lambda m: f"The user's {m.group(1)} works as {m.group(2).strip()}.",
             lambda m: m.group(2).strip()),
            # Conditions
            (r"i(?:'m| am) allergic to\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "What is the user allergic to?",
             lambda m: f"The user is allergic to {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            (r"i speak\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "What language does the user speak?",
             lambda m: f"The user speaks {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            (r"i(?:'m| am) learning\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "What is the user learning?",
             lambda m: f"The user is learning {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
            (r"i studied\s+(.+?)(?:\.|,|!|\?|$)",
             lambda m: "What did the user study?",
             lambda m: f"The user studied {m.group(1).strip()}.",
             lambda m: m.group(1).strip()),
        ]

        for pattern, question_fn, answer_fn, value_fn in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    question = question_fn(match)
                    answer = answer_fn(match)
                    value = value_fn(match)
                except (IndexError, AttributeError):
                    continue

                if not value or len(value) < 2 or len(value) > 100:
                    continue

                q_key = question.lower().strip()
                if q_key not in seen:
                    seen.add(q_key)
                    facts.append(QAPair(
                        question=question,
                        answer=answer,
                        value=value,
                    ))

        return facts

    def _review_conversation(self, messages: list) -> List[QAPair]:
        """Review the full conversation and extract all facts as Q&A pairs."""
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
                "Extract facts about the user from this conversation.\n"
                "Each fact as a question and concise answer.\n"
                "Format:\n"
                "Q: [question about the user]\n"
                "A: [concise answer]\n\n"
                "Only include facts explicitly stated. Skip inferences.\n\n"
                f"{convo_text}\n\n"
                "Facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=300, temperature=0.1)
        print(f"  [Review] raw={raw!r}")
        facts = self._parse_qa_pairs(raw)
        print(f"  [Review] parsed {len(facts)} fact(s)")
        return facts

    def _review_exchange(self, user_message: str, assistant_response: str) -> List[QAPair]:
        """Fallback: review a single exchange when no conversation history available."""
        prompt_messages = [{
            "role": "user",
            "content": (
                "Extract facts about the user from this message.\n"
                "Each fact as a question and concise answer.\n"
                "Format:\n"
                "Q: [question about the user]\n"
                "A: [concise answer]\n\n"
                "Only include facts explicitly stated.\n\n"
                f'"{user_message}"\n\n'
                "Facts:"
            ),
        }]

        prompt = self.backend.apply_chat_template(prompt_messages)
        raw = self.backend.generate(prompt, max_tokens=200, temperature=0.1)
        print(f"  [Review] raw={raw!r}")
        facts = self._parse_qa_pairs(raw)
        print(f"  [Review] parsed {len(facts)} fact(s)")
        return facts

    # Patterns in answers that signal commentary, not facts
    _COMMENTARY_RE = re.compile(
        r"(?:not (?:explicitly |directly )?mentioned|implied|"
        r"not stated|unknown|unclear|no (?:specific|explicit)|"
        r"N/A|none|n/a)",
        re.IGNORECASE,
    )

    def _parse_qa_pairs(self, raw: str) -> List[QAPair]:
        """Parse model output into Q&A pairs.

        Handles both Q:/A: format and falls back to pipe format for
        backward compatibility with models that produce triples.
        """
        facts = []

        # Try Q/A format first
        current_q = None
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("NONE"):
                return []

            # Strip bullets/numbers
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
            for prefix in ("- ", "* ", "• "):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Skip commentary
            lower = line.lower()
            if lower.startswith(("note:", "note ", "however", "there are no",
                                 "no additional", "not explicitly", "based on")):
                continue
            if self._COMMENTARY_RE.search(line):
                continue

            # Q: line
            q_match = re.match(r"^[Qq][:.]\s*(.+)$", line)
            if q_match:
                current_q = q_match.group(1).strip()
                continue

            # A: line
            a_match = re.match(r"^[Aa][:.]\s*(.+)$", line)
            if a_match and current_q:
                answer = a_match.group(1).strip().rstrip(".")
                if answer and len(answer) >= 2:
                    facts.append(QAPair(
                        question=current_q,
                        answer=answer,
                        value=answer,
                    ))
                current_q = None
                continue

            # Fallback: pipe format (SUBJECT | RELATION | OBJECT)
            if "|" in line:
                parts = [p.strip().rstrip(".") for p in line.split("|")]
                if len(parts) >= 3 and all(parts[:3]):
                    subject, relation, obj = parts[0], parts[1], parts[2]
                    if len(obj) >= 2 and not self._COMMENTARY_RE.search(obj):
                        from src.memory.facts import _triple_to_question
                        question = _triple_to_question(subject, relation, obj)
                        facts.append(QAPair(
                            question=question,
                            answer=f"{subject} {relation} {obj}.",
                            value=obj,
                        ))

        return facts

    def _dedup_key(self, fact: QAPair) -> str:
        """Compute a normalized deduplication key for a QAPair."""
        return fact.question.lower().strip()

    def deduplicate(self, new_facts: List[QAPair], existing_facts: List[QAPair]) -> List[QAPair]:
        """Filter out already-known facts.

        Same question + same value = duplicate (skip).
        Same question + different value = update (include for re-training).
        """
        existing_keys = {}
        for f in existing_facts:
            key = self._dedup_key(f)
            existing_keys[key] = f.value.lower().strip()

        result = []
        for f in new_facts:
            key = self._dedup_key(f)
            existing_val = existing_keys.get(key)
            if existing_val is None:
                result.append(f)
            elif existing_val != f.value.lower().strip():
                result.append(f)
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
