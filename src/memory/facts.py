"""Fact ledger — Q&A-based fact storage replacing MEMIT triples.

Facts are stored as natural question-answer pairs, used for:
  - System prompt injection (Tier 0: instant recall during wake)
  - LoRA training data (Tier 3: long-term encoding during sleep)
  - Graduation testing (withhold from prompt, check if LoRA recalls)

No weight editing (MEMIT removed). The ledger IS the memory.

Lifecycle of an entry:
  stage 0 (new) -> 1 -> 2 -> 3 (graduated).  Graduated entries move from
  ledger.json to graduated.json and leave the system prompt, but they are
  NOT forgotten by the ledger: every consolidation replays a sample of them
  into the LoRA training set, and every nap / full sleep re-audits a sample.
  A graduated fact that fails audit retreats (back to stage 2, back into
  ledger.json, back into the prompt) and its degrade_count is recorded.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class QAPair:
    """A fact as a natural question-answer pair.

    question: How to ask about this fact ("What is the user's son's name?")
    answer:   The complete answer ("The user's son is named Andre Patandre.")
    value:    The key recall target ("Andre Patandre") — checked during graduation
    """
    question: str
    answer: str
    value: str = ""
    source_exchange: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5
    # Rephrasings of `question` used for TRAINING (see src/sleep/paraphrase.py)
    paraphrases: List[str] = field(default_factory=list)
    # Rephrasings reserved for TESTING. Never written to a training file.
    held_out: List[str] = field(default_factory=list)
    # Slight variations of `answer` so training does not fix one string
    answer_variants: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "value": self.value,
            "source_exchange": self.source_exchange,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "paraphrases": list(self.paraphrases),
            "held_out": list(self.held_out),
            "answer_variants": list(self.answer_variants),
        }

    @staticmethod
    def from_dict(d):
        return QAPair(
            question=d["question"],
            answer=d["answer"],
            value=d.get("value", d.get("answer", "")),
            source_exchange=d.get("source_exchange"),
            timestamp=d.get("timestamp", time.time()),
            priority=d.get("priority", 0.5),
            paraphrases=list(d.get("paraphrases") or []),
            held_out=list(d.get("held_out") or []),
            answer_variants=list(d.get("answer_variants") or []),
        )


# Fields every entry carries. Missing keys in old ledger files are filled
# with these defaults on load, so new fields never break old data.
ENTRY_DEFAULTS = {
    "stage": 0,
    "last_trained": 0.0,
    "train_count": 0,
    "degrade_count": 0,
    "last_verified": 0.0,
    "recall_rate": 0.0,
    "graduated": False,
    "pruned": False,
    # Times this fact was named as the culprit of a failed sleep gate (bleed).
    # At sleep.gates.max_gate_failures the fact is quarantined: it stays in
    # the prompt but is no longer trained into the weights.
    "gate_failures": 0,
    "last_gate_failure": None,
}

GRADUATED_STAGE = 3


class FactLedger:
    """Persistent fact storage — the single source of truth for learned facts.

    Active facts live in ledger.json, graduated facts in graduated.json.
    Graduated facts are carried by LoRA and excluded from the system prompt,
    but remain eligible for replay (training) and audit (recall tests).

    Stages: 0 (new) → 1 (trained) → 2 (strong) → 3 (graduated → moved to graduated.json)
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.graduated_path = self.ledger_path.parent / "graduated.json"
        self._entries: List[dict] = []
        self._graduated: List[dict] = []
        self.load()

    # ------------------------------------------------------------------ add

    def _new_entry(self, qa: QAPair) -> dict:
        entry = {"fact_id": uuid.uuid4().hex[:8], "qa": qa.to_dict()}
        entry.update(ENTRY_DEFAULTS)
        return entry

    def add_fact(self, qa: QAPair) -> str:
        """Add a new fact. Returns the fact_id."""
        entry = self._new_entry(qa)
        self._entries.append(entry)
        self.save()
        return entry["fact_id"]

    # ---------------------------------------------------------------- reads

    def get_entry(self, fact_id: str) -> Optional[dict]:
        """Return the entry dict for a fact_id (active or graduated), or None."""
        for e in self._entries + self._graduated:
            if e["fact_id"] == fact_id:
                return e
        return None

    def get_active_facts(self) -> List[dict]:
        """Return all non-pruned, non-graduated fact entries."""
        return [e for e in self._entries if not e.get("pruned", False)]

    def get_graduated_facts(self) -> List[dict]:
        """Return all graduated fact entries."""
        return [e for e in self._graduated if not e.get("pruned", False)]

    def get_active_qa_pairs(self) -> List[QAPair]:
        """Return QAPairs for active facts (for system prompt).

        Graduated facts live in a separate file — not returned here.
        """
        return [QAPair.from_dict(e["qa"]) for e in self.get_active_facts()]

    def get_all_qa_pairs(self) -> List[QAPair]:
        """Return all QAPairs including graduated (for dedup)."""
        pairs = [QAPair.from_dict(e["qa"]) for e in self.get_active_facts()]
        pairs.extend(QAPair.from_dict(e["qa"]) for e in self.get_graduated_facts())
        return pairs

    def get_active_fact_count(self) -> int:
        return len(self.get_active_facts())

    def get_graduated_count(self) -> int:
        return len(self.get_graduated_facts())

    # ------------------------------------------------- replay / audit samples

    @staticmethod
    def _staleness_key(entry: dict):
        """Sort key: most degraded first, then least recently verified,
        then least recently trained. Used for both replay and audit."""
        return (
            -entry.get("degrade_count", 0),
            entry.get("last_verified", 0.0),
            entry.get("last_trained", 0.0),
        )

    def get_replay_candidates(self, n: int) -> List[dict]:
        """Graduated facts to mix into the next LoRA training set.

        Prefers facts with degrade_count > 0, then the oldest last_verified.
        Returns at most n entries (all graduated facts if n >= count).
        """
        if n <= 0:
            return []
        grads = sorted(self.get_graduated_facts(), key=self._staleness_key)
        return grads[:n]

    def get_audit_candidates(self, n: int) -> List[dict]:
        """Graduated facts to re-test during a nap or full sleep.

        Same ordering as replay: degraded first, then stalest verification.
        """
        return self.get_replay_candidates(n)

    # ------------------------------------------------------------- mutation

    def record_training(self, fact_id: str):
        """Record that a fact was trained via LoRA."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["last_trained"] = time.time()
            e["train_count"] = e.get("train_count", 0) + 1
        self.save()

    def record_degrade(self, fact_id: str):
        """Record that a fact's recall degraded."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["degrade_count"] = e.get("degrade_count", 0) + 1
        self.save()

    def advance_stage(self, fact_id: str) -> int:
        """Advance graduation stage (cap at 3). Returns new stage.

        At stage 3, the fact is moved from _entries to _graduated.
        """
        for e in self._entries:
            if e["fact_id"] == fact_id:
                e["stage"] = min(e.get("stage", 0) + 1, GRADUATED_STAGE)
                if e["stage"] >= GRADUATED_STAGE:
                    e["graduated"] = True
                    self._graduated.append(e)
                    self._entries.remove(e)
                self.save()
                return e["stage"]
        return 0

    def retreat_stage(self, fact_id: str) -> int:
        """Drop one stage. Returns the new stage.

        A graduated fact (stage 3) returns to stage 2: it moves back from
        graduated.json into ledger.json, so it re-enters the system prompt
        and the training set. Partial progress is kept — the fact does not
        fall all the way to stage 0. An active fact drops one stage (min 0).
        """
        for e in self._graduated:
            if e["fact_id"] == fact_id:
                e["stage"] = GRADUATED_STAGE - 1
                e["graduated"] = False
                self._entries.append(e)
                self._graduated.remove(e)
                self.save()
                return e["stage"]
        for e in self._entries:
            if e["fact_id"] == fact_id:
                e["stage"] = max(e.get("stage", 0) - 1, 0)
                e["graduated"] = False
                self.save()
                return e["stage"]
        return 0

    def set_paraphrases(self, fact_id: str, paraphrases, held_out, answer_variants=None):
        """Store generated rephrasings on a fact (training set, held-out test set)."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["qa"]["paraphrases"] = list(paraphrases)
            e["qa"]["held_out"] = list(held_out)
            if answer_variants is not None:
                e["qa"]["answer_variants"] = list(answer_variants)
        self.save()

    def record_gate_failure(self, fact_id: str, gate: str):
        """Record that a fact was identified as the cause of a gate failure."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["gate_failures"] = e.get("gate_failures", 0) + 1
            e["last_gate_failure"] = gate
        self.save()

    def update_verification(self, fact_id: str, recall_rate: float):
        """Update verification after recall test."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["last_verified"] = time.time()
            e["recall_rate"] = recall_rate
        self.save()

    def mark_pruned(self, fact_id: str):
        """Mark a fact as pruned."""
        e = self.get_entry(fact_id)
        if e is not None:
            e["pruned"] = True
        self.save()

    def clear_all(self):
        self._entries = []
        self._graduated = []
        self.save()

    # ---------------------------------------------------------- persistence

    def save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self._entries, f, indent=2)
        with open(self.graduated_path, "w") as f:
            json.dump(self._graduated, f, indent=2)

    def load(self):
        # Load active entries
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                data = []

            if data and isinstance(data, list):
                # Detect old EditLedger format (has "edit_id" key)
                if len(data) > 0 and "edit_id" in data[0]:
                    self._entries = _migrate_from_edit_ledger(data)
                    self.save()
                else:
                    self._entries = data
            else:
                self._entries = []
        else:
            self._entries = []

        # Load graduated entries
        if self.graduated_path.exists():
            try:
                with open(self.graduated_path) as f:
                    data = json.load(f)
                self._graduated = data if isinstance(data, list) else []
            except (json.JSONDecodeError, ValueError):
                self._graduated = []
        else:
            self._graduated = []

        # Backfill defaults so old files never miss a field
        for e in self._entries + self._graduated:
            for key, default in ENTRY_DEFAULTS.items():
                e.setdefault(key, default)

        # Migrate: move any graduated entries from _entries to _graduated
        migrated = []
        for e in self._entries[:]:
            if e.get("graduated", False) and not e.get("pruned", False):
                self._graduated.append(e)
                migrated.append(e)
        if migrated:
            for e in migrated:
                self._entries.remove(e)
            print(f"  [Ledger] Migrated {len(migrated)} graduated fact(s) to graduated.json")
            self.save()


def _migrate_from_edit_ledger(old_edits: list) -> list:
    """Migrate from old EditLedger (triple-based, MEMIT) to FactLedger (QA-based).

    Converts each triple (subject|relation|object) to a QAPair.
    Skips junk entries (placeholders, category labels, etc.).
    """
    new_entries = []
    seen_questions = set()

    for edit in old_edits:
        if edit.get("pruned", False):
            continue
        if edit.get("scale", 1.0) <= 0:
            continue

        facts = edit.get("facts", [])
        stages = edit.get("fact_stages", [0] * len(facts))
        priorities = edit.get("fact_priorities", [0.5] * len(facts))
        last_trained_list = edit.get("fact_last_trained", [0.0] * len(facts))
        train_counts = edit.get("fact_train_count", [0] * len(facts))
        degrade_counts = edit.get("fact_degrade_count", [0] * len(facts))

        for i, fact in enumerate(facts):
            subject = fact.get("subject", "").strip()
            relation = fact.get("relation", "").strip()
            obj = fact.get("object", "").strip()

            # Skip junk entries from 3B model garbled extraction
            if not subject or not obj or len(obj) < 2:
                continue
            if relation in ("RELATION",) or obj in ("OBJECT",):
                continue

            subj_lower = subject.lower().strip()
            rel_lower = relation.lower().strip()
            obj_lower = obj.lower().strip()

            # Skip category-label subjects (model meta-commentary)
            category_labels = {
                "age", "family", "location", "job", "preferences", "name",
                "occupation", "relationship", "relationships", "hobbies",
                "interests", "education", "work", "home", "pets", "children",
                "user", "question", "person", "opinions", "context",
                "conversation context",
            }
            if subj_lower in category_labels:
                continue
            if rel_lower in category_labels:
                continue
            # Skip subjects that START with a category label
            if any(subj_lower.startswith(cat + " ") for cat in category_labels):
                continue

            # Skip pipe/markdown artifacts
            if subject.startswith("|") or obj.startswith("|"):
                continue
            if subject.startswith("*") or subject.startswith("-"):
                continue

            # Skip commentary in any field
            commentary_kw = (
                "not mentioned", "not stated", "unknown", "unclear",
                "implied", "not specified", "not explicitly",
                "no specific", "no explicit",
            )
            if any(kw in subj_lower for kw in commentary_kw):
                continue
            if any(kw in obj_lower for kw in commentary_kw):
                continue

            # Skip objects with parenthetical commentary
            if "(" in obj and any(kw in obj_lower for kw in
                                  ("since ", "implied", "because ", "age ")):
                continue

            # Skip overly long subjects (garbled multi-clause extraction)
            if len(subject) > 60:
                continue

            question = _triple_to_question(subject, relation, obj)
            answer = f"{subject} {relation} {obj}."
            value = obj

            # Dedup by question
            q_key = question.lower().strip()
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)

            stage = stages[i] if i < len(stages) else 0
            entry = {
                "fact_id": uuid.uuid4().hex[:8],
                "qa": {
                    "question": question,
                    "answer": answer,
                    "value": value,
                    "source_exchange": fact.get("source_exchange"),
                    "timestamp": fact.get("timestamp", time.time()),
                    "priority": priorities[i] if i < len(priorities) else 0.5,
                },
                "stage": stage,
                "last_trained": last_trained_list[i] if i < len(last_trained_list) else 0.0,
                "train_count": train_counts[i] if i < len(train_counts) else 0,
                "degrade_count": degrade_counts[i] if i < len(degrade_counts) else 0,
                "last_verified": edit.get("last_verified", 0.0),
                "recall_rate": edit.get("recall_success_rate", 1.0),
                "graduated": (stage >= 3),
                "pruned": False,
            }
            new_entries.append(entry)

    print(f"  [Migration] Converted {len(new_entries)} facts from EditLedger → FactLedger")
    return new_entries


def _triple_to_question(subject, relation, obj):
    """Convert a triple to a natural question (used during migration)."""
    rel_lower = relation.lower().strip()
    subj = subject.strip()

    mapping = [
        ("has a son named", f"What is {subj}'s son's name?"),
        ("has a daughter named", f"What is {subj}'s daughter's name?"),
        ("has a dog named", f"What is {subj}'s dog's name?"),
        ("has a cat named", f"What is {subj}'s cat's name?"),
        ("is allergic to", f"What is {subj} allergic to?"),
        ("is learning", f"What is {subj} learning?"),
        ("is named", "What is the user's name?"),
        ("lives in", f"Where does {subj} live?"),
        ("works as", f"What does {subj} do for work?"),
        ("works at", f"Where does {subj} work?"),
        ("is aged", f"How old is {subj}?"),
        ("likes", f"What does {subj} like?"),
        ("dislikes", f"What does {subj} dislike?"),
        ("has", f"What does {subj} have?"),
        ("uses", f"What does {subj} use?"),
        ("speaks", f"What language does {subj} speak?"),
        ("studied", f"What did {subj} study?"),
        ("graduated in", f"When did {subj} graduate?"),
        ("was born in", f"When was {subj} born?"),
        ("moved to", f"Where did {subj} move to?"),
    ]

    for key, question in mapping:
        if key in rel_lower:
            return question

    return f"What do you know about {subj} regarding {relation}?"
