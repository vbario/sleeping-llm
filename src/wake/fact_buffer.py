"""Fact buffer — volatile in-memory accumulator for extracted facts.

Facts sit here between extraction and ledger persistence, analogous to
hippocampal short-term memory. Intentionally volatile: crash = amnesia.
No disk persistence by design.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.memory.slots import same_slot, same_value


@dataclass
class BufferedFact:
    """A fact waiting in the buffer for consolidation."""
    qa: object  # QAPair — avoid circular import
    buffered_at: float = field(default_factory=time.time)
    source_turn: int = 0
    surprise_at_extraction: float = 0.0


class FactBuffer:
    """Volatile in-memory fact accumulator with consolidation triggers.

    Design invariants:
      - Never writes to disk (volatile = retrograde amnesia on crash)
      - Consolidation = persist buffered QAPairs to the FactLedger
      - After consolidation, buffer is cleared
    """

    def __init__(self, config, fact_ledger, health_monitor=None):
        cm_config = config.get("consolidation_moment", {}) or {}
        self.max_buffer_size = cm_config.get("max_buffer_size", 20)
        self.overflow_policy = cm_config.get("overflow_policy", "consolidate")
        self.min_buffer_age_seconds = cm_config.get("min_buffer_age_seconds", 0)
        self.min_facts_for_consolidation = cm_config.get("min_facts", 1)
        self.slot_threshold = (config.get("facts", {}) or {}).get("slot_similarity_threshold", 0.6)
        self.last_consolidation_report = {"added": 0, "superseded": 0, "duplicate": 0}

        self._fact_ledger = fact_ledger
        self._health_monitor = health_monitor

        # Volatile state
        self._buffer: List[BufferedFact] = []
        self._consolidation_count = 0
        self._total_facts_consolidated = 0
        self._last_consolidation_time = 0.0

    def add(self, qa, turn: int = 0, surprise: float = 0.0):
        """Add a fact to the buffer. Does NOT touch model weights or disk.

        Same slot already buffered with the same value → skip. Same slot with
        a new value → the newer fact replaces the buffered one (a correction
        made within one session never reaches the ledger as two facts).
        """
        for existing in list(self._buffer):
            if same_slot(existing.qa.question, qa.question, self.slot_threshold):
                if same_value(existing.qa.value, qa.value):
                    return  # Already buffered
                self._buffer.remove(existing)  # superseded within the buffer

        self._buffer.append(BufferedFact(
            qa=qa,
            buffered_at=time.time(),
            source_turn=turn,
            surprise_at_extraction=surprise,
        ))

        # Handle overflow
        if len(self._buffer) > self.max_buffer_size:
            if self.overflow_policy == "consolidate":
                self.consolidate(reason="buffer_overflow")
            else:
                self._buffer.pop(0)

    def consolidate(self, reason: str = "surprise"):
        """Flush the entire buffer into the FactLedger.

        Returns:
            Number of facts persisted, or 0 if buffer was empty
        """
        if not self._buffer:
            return 0

        # Check minimum age (oldest fact must be old enough) — skip for forced flushes
        if (self.min_buffer_age_seconds > 0
                and reason == "surprise"
                and self._buffer):
            oldest_age = time.time() - self._buffer[0].buffered_at
            if oldest_age < self.min_buffer_age_seconds:
                return 0

        # Check minimum count — skip for forced flushes
        if len(self._buffer) < self.min_facts_for_consolidation and reason == "surprise":
            return 0

        qa_pairs = [bf.qa for bf in self._buffer]
        count = len(qa_pairs)

        print(f"  [Consolidation] Flushing {count} fact(s) to ledger — reason: {reason}")

        try:
            report = {"added": 0, "superseded": 0, "duplicate": 0}
            for qa in qa_pairs:
                if hasattr(self._fact_ledger, "upsert"):
                    status = self._fact_ledger.upsert(qa)["status"]
                else:
                    self._fact_ledger.add_fact(qa)
                    status = "added"
                report[status] = report.get(status, 0) + 1
            self.last_consolidation_report = report
            if report["superseded"] or report["duplicate"]:
                print(f"  [Consolidation] added={report['added']} "
                      f"superseded={report['superseded']} duplicate={report['duplicate']}")
            if self._health_monitor:
                self._health_monitor.record_new_facts(report["added"] + report["superseded"])
            self._consolidation_count += 1
            self._total_facts_consolidated += count
            self._last_consolidation_time = time.time()
            self._buffer.clear()
            return count
        except Exception as e:
            print(f"  [Consolidation] Persistence failed: {e}")
            return 0

    def clear(self):
        """Discard all buffered facts (amnesia)."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def oldest_fact_age(self) -> float:
        """Age in seconds of the oldest buffered fact, or 0 if empty."""
        if not self._buffer:
            return 0.0
        return time.time() - self._buffer[0].buffered_at

    def get_qa_pairs(self) -> list:
        """Return all buffered QAPairs (for deduplication checks)."""
        return [bf.qa for bf in self._buffer]

    def to_dict(self) -> dict:
        """Serializable status for API/UI."""
        return {
            "buffer_size": len(self._buffer),
            "max_buffer_size": self.max_buffer_size,
            "buffer_fullness": round(len(self._buffer) / max(1, self.max_buffer_size), 2),
            "oldest_fact_age_seconds": round(self.oldest_fact_age, 1),
            "consolidation_count": self._consolidation_count,
            "total_facts_consolidated": self._total_facts_consolidated,
            "overflow_policy": self.overflow_policy,
            "buffered_facts": [
                {
                    "question": bf.qa.question,
                    "answer": bf.qa.answer,
                    "age_seconds": round(time.time() - bf.buffered_at, 1),
                }
                for bf in self._buffer
            ],
        }
