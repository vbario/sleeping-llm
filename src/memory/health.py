"""Health monitor — tracks model health and computes sleep pressure.

Sleep pressure is a weighted combination of:
  - fact_pressure: accumulated facts relative to a soft capacity
  - time_pressure: time since last sleep relative to max wake duration
  - perplexity_pressure: model perplexity drift from baseline

When pressure exceeds thresholds, nap or full sleep is triggered.
"""

import time
from dataclasses import dataclass, field


@dataclass
class HealthSnapshot:
    """Point-in-time snapshot of model health metrics."""
    timestamp: float = field(default_factory=time.time)
    fact_count: int = 0
    perplexity: float = 0.0
    coherence_score: float = 1.0
    sleep_pressure: float = 0.0


class HealthMonitor:
    """Monitors model health and determines when sleep is needed.

    Sleep pressure replaces (or augments) the fixed turn counter for
    triggering sleep transitions. Pressure rises with new facts,
    elapsed time, and perplexity drift.
    """

    def __init__(self, config, backend, fact_ledger):
        self.config = config
        self.backend = backend
        self.fact_ledger = fact_ledger

        health_config = config.get("health", {}) or {}
        self.nap_threshold = health_config.get("nap_threshold", 0.4)
        self.sleep_threshold = health_config.get("sleep_threshold", 0.8)
        self.fact_weight = health_config.get("edit_weight", 0.6)  # reuse edit_weight key
        self.time_weight = health_config.get("time_weight", 0.3)
        self.perplexity_weight = health_config.get("perplexity_weight", 0.1)
        self.perplexity_check_interval = health_config.get("perplexity_check_interval", 10)
        self.max_wake_seconds = health_config.get("max_wake_seconds", 7200)

        self.buffer_weight = health_config.get("buffer_weight", 0.0)

        # Soft capacity for facts (used for pressure calculation)
        self.max_facts = health_config.get("max_facts", 50)

        # State
        self._fact_buffer = None
        self._last_sleep_time = time.time()
        self._new_facts_since_sleep = 0
        self._facts_since_perplexity = 0
        self._baseline_perplexity = None
        self._current_perplexity = None

    def set_fact_buffer(self, fact_buffer):
        """Register the fact buffer for pressure calculation."""
        self._fact_buffer = fact_buffer

    def get_sleep_pressure(self) -> float:
        """Compute current sleep pressure (0.0 - 1.0+).

        Weighted combination of fact, time, and perplexity pressures.
        """
        # Fact pressure (based on total active facts relative to capacity)
        total_facts = self.fact_ledger.get_active_fact_count()
        fact_ratio = total_facts / max(1, self.max_facts)
        fact_pressure = min(1.0, fact_ratio ** 1.5)

        # Time pressure
        elapsed = time.time() - self._last_sleep_time
        time_pressure = min(1.0, elapsed / max(1, self.max_wake_seconds))

        # Perplexity pressure
        perplexity_pressure = 0.0
        if self._baseline_perplexity and self._current_perplexity:
            if self._baseline_perplexity > 0:
                ratio = self._current_perplexity / self._baseline_perplexity
                perplexity_pressure = max(0.0, min(1.0, ratio - 1.0))

        # Buffer pressure (how full the fact buffer is)
        buffer_pressure = 0.0
        if self._fact_buffer and self.buffer_weight > 0:
            buffer_pressure = self._fact_buffer.size / max(1, self._fact_buffer.max_buffer_size)

        pressure = (
            self.fact_weight * fact_pressure +
            self.time_weight * time_pressure +
            self.perplexity_weight * perplexity_pressure +
            self.buffer_weight * buffer_pressure
        )

        return pressure

    def should_nap(self) -> bool:
        """Check if pressure exceeds nap threshold."""
        return self.get_sleep_pressure() >= self.nap_threshold

    def should_sleep(self) -> bool:
        """Check if pressure exceeds full sleep threshold."""
        return self.get_sleep_pressure() >= self.sleep_threshold

    def measure_perplexity(self) -> float:
        """Compute perplexity on reference text, update baseline/current."""
        reference_text = (
            "The quick brown fox jumps over the lazy dog. "
            "In machine learning, neural networks process data through layers. "
            "The capital of France is Paris and Berlin is the capital of Germany."
        )

        ppl = self.backend.compute_perplexity(reference_text)

        if self._baseline_perplexity is None:
            self._baseline_perplexity = ppl

        self._current_perplexity = ppl
        self._facts_since_perplexity = 0

        return ppl

    def record_new_facts(self, count=1):
        """Record that new facts were persisted to the ledger."""
        self._new_facts_since_sleep += count
        self._facts_since_perplexity += count

        # Optionally measure perplexity at intervals
        if (self.perplexity_check_interval > 0 and
                self._facts_since_perplexity >= self.perplexity_check_interval):
            try:
                self.measure_perplexity()
            except Exception:
                pass

    def record_sleep(self, sleep_type="nap", facts_refreshed=0, facts_pruned=0):
        """Record that a sleep cycle completed, adjusting pressure."""
        self._last_sleep_time = time.time()
        if sleep_type == "full":
            self._new_facts_since_sleep = 0

    def get_snapshot(self) -> HealthSnapshot:
        """Return a snapshot of current health metrics."""
        return HealthSnapshot(
            timestamp=time.time(),
            fact_count=self.fact_ledger.get_active_fact_count(),
            perplexity=self._current_perplexity or 0.0,
            coherence_score=1.0,
            sleep_pressure=self.get_sleep_pressure(),
        )

    def to_dict(self) -> dict:
        """Return health status as a dict for API responses."""
        snapshot = self.get_snapshot()
        return {
            "sleep_pressure": round(snapshot.sleep_pressure, 3),
            "fact_count": snapshot.fact_count,
            "graduated_count": self.fact_ledger.get_graduated_count(),
            "perplexity": round(snapshot.perplexity, 2) if snapshot.perplexity else None,
            "baseline_perplexity": round(self._baseline_perplexity, 2) if self._baseline_perplexity else None,
            "nap_threshold": self.nap_threshold,
            "sleep_threshold": self.sleep_threshold,
            "should_nap": self.should_nap(),
            "should_sleep": self.should_sleep(),
            "seconds_since_sleep": round(time.time() - self._last_sleep_time, 0),
            "buffer_size": self._fact_buffer.size if self._fact_buffer else 0,
        }
