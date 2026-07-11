"""Fact buffer — volatile in-memory accumulator for extracted facts.

Facts sit here between extraction and ledger persistence, analogous to
hippocampal short-term memory. Intentionally volatile: crash = amnesia.
No disk persistence by design.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BufferedFact:
    """A fact waiting in the buffer for consolidation."""
    qa: object  # QAPair — avoid circular import
    buffered_at: float = field(default_factory=time.time)
    source_turn: int = 0
    surprise_at_extraction: float = 0.0
    meta: object = None  # per-fact valuation metadata (dict or None)
    bypass_dedup: bool = False  # True = frozen-replay channel


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

        self._fact_ledger = fact_ledger
        self._health_monitor = health_monitor

        # Volatile state
        self._buffer: List[BufferedFact] = []
        self._consolidation_count = 0
        self._total_facts_consolidated = 0
        self._last_consolidation_time = 0.0

        # Optional valuation policy — rescores the batch at consolidation
        self._valuation_policy = None

    def set_valuation_policy(self, policy):
        """Install a ValuationPolicy applied at consolidation time (or None)."""
        self._valuation_policy = policy
        if policy is not None:
            print(f"  [Valuation] Policy set: {getattr(policy, 'name', '?')}")

    @staticmethod
    def _value_key(qa) -> str:
        """Normalized value for content-based dedup."""
        import re
        v = qa.value.lower().strip().rstrip(".,!? ")
        return re.sub(r"\s+", " ", v)

    def add(self, qa, turn: int = 0, surprise: float = 0.0,
            bypass_dedup: bool = False, meta=None):
        """Add a fact to the buffer. Does NOT touch model weights or disk.

        Skips if a fact with the same question key OR same value is already
        buffered. bypass_dedup=True skips all intra-buffer dedup filters
        (frozen-replay streams are deduplicated by construction and
        re-mention events must be delivered). meta is optional per-fact
        valuation metadata passed to the policy at consolidation.
        """
        if not bypass_dedup:
            key = qa.question.lower().strip()
            val_key = self._value_key(qa)
            for existing in self._buffer:
                if existing.qa.question.lower().strip() == key:
                    return  # Same question already buffered
                ex_val = self._value_key(existing.qa)
                if ex_val == val_key:
                    return  # Same value already buffered
                # Containment check (skip very short values to avoid false positives)
                if len(val_key) >= 5 and len(ex_val) >= 5:
                    if val_key in ex_val or ex_val in val_key:
                        return  # Overlapping value already buffered
                # Word-overlap check
                if self._words_overlap(val_key, ex_val):
                    return  # Similar value already buffered

        self._buffer.append(BufferedFact(
            qa=qa,
            buffered_at=time.time(),
            source_turn=turn,
            surprise_at_extraction=surprise,
            meta=meta,
            bypass_dedup=bypass_dedup,
        ))

        # Handle overflow
        if len(self._buffer) > self.max_buffer_size:
            if self.overflow_policy == "consolidate":
                self.consolidate(reason="buffer_overflow")
            else:
                self._buffer.pop(0)

    @staticmethod
    def _words_overlap(a: str, b: str) -> bool:
        """Check if two values share 60%+ content words."""
        stop = frozenset("a an the is are was were do does i my me you your "
                         "he she it we they at in on of to for and or but".split())
        a_words = {w for w in a.split() if w not in stop and len(w) > 1}
        b_words = {w for w in b.split() if w not in stop and len(w) > 1}
        if len(a_words) < 3 or len(b_words) < 3:
            return False
        shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
        return len(shorter & longer) / len(shorter) >= 0.6

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

        # Valuation hook: rescore the batch before persisting (all reasons)
        if self._valuation_policy is not None:
            try:
                metas = [bf.meta for bf in self._buffer]
                ledger_qas = self._fact_ledger.get_all_qa_pairs()
                scores = self._valuation_policy.score_batch(qa_pairs, metas, ledger_qas)
                if scores is not None and len(scores) == count:
                    for qa, score in zip(qa_pairs, scores):
                        qa.priority = score
                else:
                    print("  [Valuation] failed, defaulting")
                # Supersession: retire ledger entries made stale by this batch
                supersessions = getattr(self._valuation_policy, "supersessions", None)
                if callable(supersessions):
                    for stale_id in (supersessions() or []):
                        self._fact_ledger.retreat_stage(stale_id)
                        self._fact_ledger.mark_pruned(stale_id)
                        print(f"  [Valuation] superseded ledger entry {stale_id}")
            except Exception as e:
                print(f"  [Valuation] failed, defaulting ({e})")

        try:
            add_or_refresh = getattr(self._fact_ledger, "add_or_refresh", None)
            get_entry = getattr(self._fact_ledger, "get_entry", None)
            stamped = False
            for bf in self._buffer:
                # Re-mention (add_or_refresh) semantics apply only on the
                # frozen-replay channel (§7.2.2 ledger-merge); production
                # adds (bypass_dedup=False) keep today's add_fact path.
                if bf.bypass_dedup and callable(add_or_refresh):
                    _status, fact_id = add_or_refresh(bf.qa)
                else:
                    fact_id = self._fact_ledger.add_fact(bf.qa)
                # Carry arrival session onto the entry so EgoFullPolicy.rescore
                # can rebuild [t=session n] metadata (replay metas only —
                # production meta is None, so this never fires there).
                if (fact_id and callable(get_entry) and bf.meta
                        and bf.meta.get("session") is not None):
                    entry = get_entry(fact_id)
                    if entry is not None:
                        entry["arrival_session"] = bf.meta["session"]
                        stamped = True
            if stamped:
                self._fact_ledger.save()
            if self._health_monitor:
                self._health_monitor.record_new_facts(count)
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
