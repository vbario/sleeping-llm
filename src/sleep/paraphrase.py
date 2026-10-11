"""Paraphrase generation — held-out phrasings for honest graduation tests.

Each fact is stored with one canonical question. Training on only that
phrasing and then testing with it proves memorization of a prompt, not
knowledge. This module generates several rephrasings per fact, splits them
into a training set and a held-out set, and never lets the held-out set
touch a training file.

Two sources, combined:
  - templates: deterministic third-person -> first-person rewrites ("What is
    the user's son's name?" -> "What's my son's name?"), which is how users
    actually ask;
  - the model (optional): free rephrasings, filtered so none leaks the value.

The same machinery produces *neighbor* questions for the bleed gate: the
same kind of question about a different subject ("What is my sister's
favorite color?" for a fact about the user's own).
"""

import random
import re
from typing import List, Sequence, Tuple

from src.memory.facts import QAPair
from src.memory.slots import tokenize as slot_tokenize
from src.sleep.recall_match import normalize, value_in_response


# Subject rewrites. Order matters: longer / more specific patterns first.
_FIRST_PERSON_RULES = [
    (re.compile(r"\bdoes the user's\b", re.I), "does my"),
    (re.compile(r"\bis the user's\b", re.I), "is my"),
    (re.compile(r"\bare the user's\b", re.I), "are my"),
    (re.compile(r"\bthe user's\b", re.I), "my"),
    (re.compile(r"\bdoes the user\b", re.I), "do I"),
    (re.compile(r"\bdid the user\b", re.I), "did I"),
    (re.compile(r"\bis the user\b", re.I), "am I"),
    (re.compile(r"\bwas the user\b", re.I), "was I"),
    (re.compile(r"\bhas the user\b", re.I), "have I"),
    (re.compile(r"\bcan the user\b", re.I), "can I"),
    (re.compile(r"\bthe user\b", re.I), "me"),
]

# Second-person rewrites for answers ("The user lives in X." -> "You live in X.")
_SECOND_PERSON_RULES = [
    (re.compile(r"^The user's\b"), "Your"),
    (re.compile(r"\bthe user's\b"), "your"),
    (re.compile(r"^The user\b"), "You"),
    (re.compile(r"\bthe user\b"), "you"),
]
_VERB_AGREEMENT = [
    (re.compile(r"\bYou is\b"), "You are"), (re.compile(r"\byou is\b"), "you are"),
    (re.compile(r"\bYou was\b"), "You were"), (re.compile(r"\byou was\b"), "you were"),
    (re.compile(r"\bYou has\b"), "You have"), (re.compile(r"\byou has\b"), "you have"),
    (re.compile(r"\bYou does\b"), "You do"), (re.compile(r"\byou does\b"), "you do"),
    (re.compile(r"\b(You|you) (\w+?)ies\b"), lambda m: f"{m.group(1)} {m.group(2)}y"),
    (re.compile(r"\b(You|you) (\w+?)(?:sh|ch|ss|x|z)es\b"),
     lambda m: m.group(0)[:-2]),
    (re.compile(r"\b(You|you) (\w+?[^s])s\b"), lambda m: f"{m.group(1)} {m.group(2)}"),
]

# Subjects for neighbor (bleed) questions — same question, different person.
NEIGHBOR_SUBJECTS = [
    "sister", "brother", "mother", "father", "best friend", "boss",
    "neighbor", "colleague", "cousin", "roommate",
]
# Subjects for CONTRAST training rows ("What is my aunt's favorite color?" ->
# "I don't know"). Disjoint from NEIGHBOR_SUBJECTS so the bleed gate never
# asks a question the model was trained to deflect; it has to generalize the
# subject binding, not memorize the gate's subjects.
CONTRAST_SUBJECTS = [
    "aunt", "uncle", "grandmother", "grandfather", "coworker", "landlord",
    "mentor", "teammate", "nephew", "niece", "doctor", "dentist", "professor",
    "manager", "coach", "barber", "accountant", "lawyer", "classmate",
    "stepfather", "stepmother", "godmother", "godfather", "tutor",
]
assert not set(NEIGHBOR_SUBJECTS) & set(CONTRAST_SUBJECTS)
# Relation words that pin a question to a particular person; a neighbor must
# not reuse one already in the question.
RELATION_WORDS = set(NEIGHBOR_SUBJECTS) | set(CONTRAST_SUBJECTS) | {
    "son", "daughter", "wife", "husband", "partner", "mom", "dad", "dog",
    "cat", "pet", "child", "children", "friend", "parents",
}

# Answers for contrast rows: a short denial that names the asked-about
# subject and nothing else. Earlier wording appended "— you've only told me
# about your son's", and the model learned THAT clause as a reflex: on the
# graduation probe it said "I don't know that about your boss — you've only
# told me about their coffee order" for facts a bare prompt recalled (15 of
# 22 failed probes in notes/135). The subject binding comes from the pairing
# of first-party rows with context against these denials, not from the
# denial enumerating what is known. No value appears.
CONTRAST_ANSWERS = [
    "I don't know {topic}.",
    "You haven't told me {topic}.",
]


def to_first_person(question: str) -> str:
    out = question
    for pattern, repl in _FIRST_PERSON_RULES:
        out = pattern.sub(repl, out)
    out = re.sub(r"^What is my\b", "What's my", out)
    return out[0].upper() + out[1:] if out else out


def to_second_person(answer: str) -> str:
    out = answer
    for pattern, repl in _SECOND_PERSON_RULES:
        out = pattern.sub(repl, out)
    for pattern, repl in _VERB_AGREEMENT:
        out = pattern.sub(repl, out)
    return out


def _strip_q(q: str) -> str:
    return q.strip().rstrip("?").strip()


def template_paraphrases(question: str) -> List[str]:
    """Deterministic rephrasings. First-person forms first."""
    fp = to_first_person(question)
    fp_q = fp if fp.endswith("?") else fp + "?"
    core = _strip_q(fp)
    core_lower = core[0].lower() + core[1:] if core else core
    # Each template embeds the direct question unchanged, so it stays grammatical.
    candidates = [
        fp_q,
        f"Remind me, {core_lower}?",
        f"Quick question: {core_lower}?",
        f"Do you remember? {fp_q}",
        f"{core} again?",
        f"I forgot, {core_lower}?",
    ]
    if fp.startswith("What's my"):
        candidates.append(fp_q.replace("What's my", "What is my", 1))
    return _dedupe(candidates, exclude=[question])


def _dedupe(items: Sequence[str], exclude: Sequence[str] = ()) -> List[str]:
    seen = {normalize(e) for e in exclude}
    out = []
    for it in items:
        k = normalize(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


def split_paraphrases(paraphrases: Sequence[str], held_out_count: int,
                      seed_text: str) -> Tuple[List[str], List[str]]:
    """Deterministically split into (train, held_out).

    Held-out gets `held_out_count` phrasings when enough exist; with fewer
    than held_out_count + 1 paraphrases, everything is held out and training
    uses only the canonical question.
    """
    items = list(paraphrases)
    if not items:
        return [], []
    rng = random.Random(f"paraphrase-split:{normalize(seed_text)}")
    rng.shuffle(items)
    if len(items) <= held_out_count:
        return [], items
    return items[held_out_count:], items[:held_out_count]


class ParaphraseGenerator:
    """Generates and persists paraphrases for ledger entries."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        g = config.get("sleep.graduation", {}) or {}
        self.count = int(g.get("paraphrase_count", 6))
        self.held_out_count = int(g.get("held_out_count", 2))
        self.use_model = bool(g.get("use_model", True))

    # ------------------------------------------------------------ questions

    def model_paraphrases(self, qa: QAPair, n: int) -> List[str]:
        """Ask the model for n rephrasings. Filters anything that leaks the value."""
        if n <= 0 or self.backend is None:
            return []
        prompt_messages = [{
            "role": "user",
            "content": (
                "A person is talking to their assistant. Rewrite this question "
                f"in {n} different ways the person might ask it about themselves, "
                "in the first person (use 'my', 'I'). Keep the meaning identical. "
                "Do not answer it. One question per line, no numbering.\n\n"
                f"Question: {qa.question}"
            ),
        }]
        try:
            prompt = self.backend.apply_chat_template(prompt_messages)
            raw = self.backend.generate(prompt, max_tokens=200, temperature=0.7)
        except Exception as e:
            print(f"  [Paraphrase] model generation failed: {e}")
            return []
        out = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
            if not line or "?" not in line or len(line) > 160:
                continue
            line = line[:line.rfind("?") + 1]
            if value_in_response(qa.value, line):
                continue  # leaks the answer
            if ":" in line and line.lower().startswith(("question", "rewrite", "here")):
                continue
            if not self.on_topic(qa.question, line):
                continue  # generic rephrasing, would collide across facts
            out.append(line)
        return out[:n]

    @staticmethod
    def on_topic(question: str, paraphrase: str) -> bool:
        """A paraphrase must keep the question's subject and share a content word.

        Rejects generic rephrasings ("Which one is it for me?") that a small
        model produces for any question — those would collide across facts.
        """
        q_rel, q_content = slot_tokenize(question)
        p_rel, p_content = slot_tokenize(paraphrase)
        if q_rel != p_rel:
            return False
        meaningful = {t for t in q_content if t not in ("what", "where", "when", "who")}
        if not meaningful:
            return True
        return bool(meaningful & p_content)

    def generate_for(self, qa: QAPair, taken: Sequence[str] = ()) -> Tuple[List[str], List[str], List[str]]:
        """Return (train_paraphrases, held_out, answer_variants) for a fact.

        `taken` are phrasings already used by other facts (train or held-out);
        they are never reused, so no held-out line of one fact can be a
        training line of another.
        """
        templates = template_paraphrases(qa.question)
        model = []
        if self.use_model:
            want = max(0, self.count - len(templates))
            # Always ask for a few so model phrasings land in both splits.
            model = self.model_paraphrases(qa, max(want, 3))
        candidates = [c for c in model + templates if self.on_topic(qa.question, c)]
        # Model phrasings first (more diverse), templates fill up to `count`.
        paraphrases = _dedupe(candidates, exclude=[qa.question] + list(taken))[:max(self.count, 1)]
        train, held_out = split_paraphrases(paraphrases, self.held_out_count, qa.question)
        return train, held_out, self.answer_variants(qa)

    # -------------------------------------------------------------- answers

    @staticmethod
    def answer_variants(qa: QAPair) -> List[str]:
        """Slight variations of the answer so training does not fix one string."""
        variants = [qa.answer]
        second = to_second_person(qa.answer)
        if normalize(second) != normalize(qa.answer):
            variants.append(second)
        return _dedupe(variants)

    # ----------------------------------------------------------- persistence

    @staticmethod
    def needs_paraphrases(entry: dict) -> bool:
        return not entry.get("qa", {}).get("held_out")

    def ensure(self, ledger, entries) -> int:
        """Generate paraphrases for entries that have none. Returns count generated."""
        done = 0
        taken: List[str] = []
        for e in ledger.get_active_facts() + ledger.get_graduated_facts():
            q = e.get("qa", {})
            taken.append(q.get("question", ""))
            taken.extend(q.get("paraphrases") or [])
            taken.extend(q.get("held_out") or [])
        for entry in entries:
            if not self.needs_paraphrases(entry):
                continue
            qa = QAPair.from_dict(entry["qa"])
            train, held_out, variants = self.generate_for(qa, taken=taken)
            ledger.set_paraphrases(entry["fact_id"], train, held_out, variants)
            taken.extend(train + held_out)
            done += 1
        if done:
            print(f"        Paraphrases generated for {done} fact(s)")
        return done

    # ------------------------------------------------------------ neighbors

    @staticmethod
    def neighbor_questions(qa: QAPair, n: int, avoid_questions: Sequence[str] = (),
                           subjects: Sequence[str] = NEIGHBOR_SUBJECTS) -> List[str]:
        """Same question about a different person, for the bleed gate.

        'What is the user's favorite color?' -> 'What is the user's sister's favorite color?'
        'Where does the user live?'          -> 'Where does the user's boss live?'
        Subjects already in the question (and questions already in the ledger)
        are skipped so a real fact is never mistaken for bleed.
        """
        q = qa.question
        q_words = set(re.findall(r"\w+", q.lower()))
        avoid = {normalize(a) for a in avoid_questions}
        # A question already about a relative ("...the user's sister...") gets
        # that relative swapped for another, not a second one stacked on top.
        present = [w for w in RELATION_WORDS if re.search(rf"\b{w}\b", q, re.I)]
        present.sort(key=len, reverse=True)
        out = []
        for subj in subjects:
            if len(out) >= n:
                break
            if any(w in q_words for w in subj.split()):
                continue
            if present:
                nq = re.sub(rf"\b{present[0]}\b", subj, q, count=1, flags=re.I)
            elif re.search(r"\bthe user's\b", q, re.I):
                nq = re.sub(r"\bthe user's\b", f"the user's {subj}'s", q, count=1, flags=re.I)
            elif re.search(r"\bthe user\b", q, re.I):
                nq = re.sub(r"\bthe user\b", f"the user's {subj}", q, count=1, flags=re.I)
            elif re.search(r"\bmy\b", q, re.I):
                nq = re.sub(r"\bmy\b", f"my {subj}'s", q, count=1, flags=re.I)
            else:
                nq = f"About the user's {subj}: {q}"
            if normalize(nq) in avoid or normalize(nq) == normalize(q):
                continue
            out.append(nq)
        return out

    # ------------------------------------------------------------- contrast

    @classmethod
    def contrast_examples(cls, qa: QAPair, n: int,
                          avoid_questions: Sequence[str] = ()) -> List[Tuple[str, str]]:
        """(question, answer) rows teaching that the fact belongs to ONE subject.

        For "What is the user's favorite color?" -> "teal", returns rows like
        ("What is the user's aunt's favorite color?", "I don't have any
        information about your aunt's favorite color.") plus the first-person
        form. Subjects come from CONTRAST_SUBJECTS, never from the bleed
        gate's list, and `avoid_questions` (every ledger question and every
        gate probe) is excluded so no real fact is ever trained as unknown.
        """
        if n <= 0 or not qa.value:
            return []
        # A different random draw of subjects per fact. With a fixed handful
        # the model learns "deflect for aunt/uncle, answer for everyone else"
        # and the gate's sister/brother still leak; spread across the whole
        # pool, "only the stated subject" is the cheaper rule to learn.
        pool = list(CONTRAST_SUBJECTS)
        random.Random(normalize(qa.question)).shuffle(pool)
        neighbors = cls.neighbor_questions(qa, n, avoid_questions=avoid_questions, subjects=pool)
        avoid = {normalize(a) for a in avoid_questions}
        rows: List[Tuple[str, str]] = []
        for i, nq in enumerate(neighbors):
            subj = next((s for s in CONTRAST_SUBJECTS if re.search(rf"\b{s}\b", nq, re.I)), None)
            topic = f"that about your {subj}" if subj else "that"
            # Keep the asked-about attribute when the question has the usual shape
            m = re.search(rf"\b{subj}'s\s+(.+?)\?$", nq, re.I) if subj else None
            if m:
                topic = f"your {subj}'s {m.group(1)}"
            answer = CONTRAST_ANSWERS[i % len(CONTRAST_ANSWERS)].format(topic=topic)
            # Raw boundary check, not value_in_response: that one ignores
            # hedged text and every contrast answer is a hedge.
            if re.search(rf"(?<!\w){re.escape(normalize(qa.value))}(?!\w)", normalize(answer)):
                continue
            for question in (nq, to_first_person(nq)):
                if normalize(question) in avoid:
                    continue
                avoid.add(normalize(question))
                rows.append((question, answer))
        return rows
