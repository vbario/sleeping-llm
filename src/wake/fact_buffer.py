"""Fact buffer — volatile in-memory accumulator for extracted facts.

Facts sit here between extraction and MEMIT injection, analogous to
hippocampal short-term memory. Intentionally volatile: crash = amnesia.
No disk persistence by design.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BufferedFact:
    """A fact waiting in the buffer for consolidation."""
    triple: object  # FactTriple — avoid circular import
    buffered_at: float = field(default_factory=time.time)
    source_turn: int = 0
    surprise_at_extraction: float = 0.0


class FactBuffer:
    """Volatile in-memory fact accumulator with consolidation triggers.

    Design invariants:
      - Never writes to disk (volatile = retrograde amnesia on crash)
      - Consolidation = batch MEMIT injection of entire buffer
      - After consolidation, buffer is cleared
    """

    def __init__(self, config, memit_engine, health_monitor=None):
        cm_config = config.get("consolidation_moment", {}) or {}
        self.max_buffer_size = cm_config.get("max_buffer_size", 20)
        self.overflow_policy = cm_config.get("overflow_policy", "consolidate")
        self.min_buffer_age_seconds = cm_config.get("min_buffer_age_seconds", 0)
        self.min_facts_for_consolidation = cm_config.get("min_facts", 1)

        self._memit_engine = memit_engine
        self._health_monitor = health_monitor

        # Volatile state
        self._buffer: List[BufferedFact] = []
        self._consolidation_count = 0
        self._total_facts_consolidated = 0
        self._last_consolidation_time = 0.0

    def add(self, triple, turn: int = 0, surprise: float = 0.0):
        """Add a fact to the buffer. Does NOT touch model weights.

        Skips if a fact with the same (subject, relation) key is already buffered.
        """
        key = (triple.subject.lower().strip(), triple.relation.lower().strip())
        for existing in self._buffer:
            ek = (existing.triple.subject.lower().strip(),
                  existing.triple.relation.lower().strip())
            if key == ek:
                return  # Already buffered

        self._buffer.append(BufferedFact(
            triple=triple,
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
        """Flush the entire buffer into MEMIT as a single batch injection.

        Returns:
            The MemitEdit record, or None if buffer was empty or injection failed
        """
        if not self._buffer:
            return None

        # Check minimum age (oldest fact must be old enough) — skip for forced flushes
        if (self.min_buffer_age_seconds > 0
                and reason == "surprise"
                and self._buffer):
            oldest_age = time.time() - self._buffer[0].buffered_at
            if oldest_age < self.min_buffer_age_seconds:
                return None

        # Check minimum count — skip for forced flushes
        if len(self._buffer) < self.min_facts_for_consolidation and reason == "surprise":
            return None

        triples = [bf.triple for bf in self._buffer]
        count = len(triples)

        print(f"  [Consolidation] Flushing {count} fact(s) — reason: {reason}")

        try:
            edit = self._memit_engine.inject_facts(triples)
            if edit and self._health_monitor:
                self._health_monitor.record_edit(count)
            self._consolidation_count += 1
            self._total_facts_consolidated += count
            self._last_consolidation_time = time.time()
            self._buffer.clear()
            return edit
        except Exception as e:
            print(f"  [Consolidation] Injection failed: {e}")
            # Buffer NOT cleared on failure — facts survive for retry
            return None

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

    def get_triples(self) -> list:
        """Return all buffered triples (for deduplication checks)."""
        return [bf.triple for bf in self._buffer]

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
                    "subject": bf.triple.subject,
                    "relation": bf.triple.relation,
                    "object": bf.triple.object,
                    "age_seconds": round(time.time() - bf.buffered_at, 1),
                }
                for bf in self._buffer
            ],
        }
