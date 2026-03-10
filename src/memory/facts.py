"""Fact ledger — Q&A-based fact storage replacing MEMIT triples.

Facts are stored as natural question-answer pairs, used for:
  - System prompt injection (Tier 0: instant recall during wake)
  - LoRA training data (Tier 3: long-term encoding during sleep)
  - Graduation testing (withhold from prompt, check if LoRA recalls)

No weight editing (MEMIT removed). The ledger IS the memory.
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

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "value": self.value,
            "source_exchange": self.source_exchange,
            "timestamp": self.timestamp,
            "priority": self.priority,
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
        )


class FactLedger:
    """Persistent fact storage — the single source of truth for learned facts.

    Active facts live in ledger.json, graduated facts in graduated.json.
    Graduated facts are carried by LoRA and excluded from the system prompt.

    Stages: 0 (new) → 1 (trained) → 2 (strong) → 3 (graduated → moved to graduated.json)
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.graduated_path = self.ledger_path.parent / "graduated.json"
        self._entries: List[dict] = []
        self._graduated: List[dict] = []
        self.load()

    def add_fact(self, qa: QAPair) -> str:
        """Add a new fact. Returns the fact_id."""
        fact_id = uuid.uuid4().hex[:8]
        entry = {
            "fact_id": fact_id,
            "qa": qa.to_dict(),
            "stage": 0,
            "last_trained": 0.0,
            "train_count": 0,
            "degrade_count": 0,
            "last_verified": 0.0,
            "recall_rate": 0.0,
            "graduated": False,
            "pruned": False,
        }
        self._entries.append(entry)
        self.save()
        return fact_id

    def get_active_facts(self) -> List[dict]:
        """Return all non-pruned, non-graduated fact entries."""
        return [e for e in self._entries if not e.get("pruned", False)]

    def get_graduated_facts(self) -> List[dict]:
        """Return all graduated fact entries."""
        return list(self._graduated)

    def get_active_qa_pairs(self) -> List[QAPair]:
        """Return QAPairs for active facts (for system prompt).

        Graduated facts live in a separate file — not returned here.
        """
        return [QAPair.from_dict(e["qa"]) for e in self.get_active_facts()]

    def get_all_qa_pairs(self) -> List[QAPair]:
        """Return all QAPairs including graduated (for dedup)."""
        pairs = [QAPair.from_dict(e["qa"]) for e in self.get_active_facts()]
        pairs.extend(QAPair.from_dict(e["qa"]) for e in self._graduated)
        return pairs

    def get_active_fact_count(self) -> int:
        return len(self.get_active_facts())

    def get_graduated_count(self) -> int:
        return len(self._graduated)

    def record_training(self, fact_id: str):
        """Record that a fact was trained via LoRA."""
        for e in self._entries + self._graduated:
            if e["fact_id"] == fact_id:
                e["last_trained"] = time.time()
                e["train_count"] = e.get("train_count", 0) + 1
                break
        self.save()

    def record_degrade(self, fact_id: str):
        """Record that a fact's recall degraded."""
        for e in self._entries + self._graduated:
            if e["fact_id"] == fact_id:
                e["degrade_count"] = e.get("degrade_count", 0) + 1
                break
        self.save()

    def advance_stage(self, fact_id: str) -> int:
        """Advance graduation stage (cap at 3). Returns new stage.

        At stage 3, the fact is moved from _entries to _graduated.
        """
        for e in self._entries:
            if e["fact_id"] == fact_id:
                e["stage"] = min(e.get("stage", 0) + 1, 3)
                if e["stage"] >= 3:
                    e["graduated"] = True
                    self._graduated.append(e)
                    self._entries.remove(e)
                self.save()
                return e["stage"]
        return 0

    def retreat_stage(self, fact_id: str):
        """Reset stage to 0 (un-graduate). Moves back from graduated if needed."""
        # Check graduated list first
        for e in self._graduated:
            if e["fact_id"] == fact_id:
                e["stage"] = 0
                e["graduated"] = False
                self._entries.append(e)
                self._graduated.remove(e)
                self.save()
                return
        # Also check active entries (might not be graduated yet)
        for e in self._entries:
            if e["fact_id"] == fact_id:
                e["stage"] = 0
                e["graduated"] = False
                break
        self.save()

    def update_verification(self, fact_id: str, recall_rate: float):
        """Update verification after recall test."""
        for e in self._entries + self._graduated:
            if e["fact_id"] == fact_id:
                e["last_verified"] = time.time()
                e["recall_rate"] = recall_rate
                break
        self.save()

    def mark_pruned(self, fact_id: str):
        """Mark a fact as pruned."""
        for e in self._entries:
            if e["fact_id"] == fact_id:
                e["pruned"] = True
                break
        self.save()

    def clear_all(self):
        self._entries = []
        self._graduated = []
        self.save()

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
