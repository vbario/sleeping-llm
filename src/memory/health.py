"""Health monitor — tracks model health and computes sleep pressure.

Sleep pressure is a weighted combination of:
  - edit_pressure: accumulated MEMIT edits relative to max capacity
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
    edit_count: int = 0
    perplexity: float = 0.0
    coherence_score: float = 1.0
    sleep_pressure: float = 0.0


class HealthMonitor:
    """Monitors model health and determines when sleep is needed.

    Sleep pressure replaces (or augments) the fixed turn counter for
    triggering sleep transitions. Pressure rises with MEMIT edits,
    elapsed time, and perplexity drift.
    """

    def __init__(self, config, backend, ledger):
        self.config = config
        self.backend = backend
        self.ledger = ledger

        health_config = config.get("health", {}) or {}
        self.nap_threshold = health_config.get("nap_threshold", 0.4)
        self.sleep_threshold = health_config.get("sleep_threshold", 0.8)
        self.edit_weight = health_config.get("edit_weight", 0.6)
        self.time_weight = health_config.get("time_weight", 0.3)
        self.perplexity_weight = health_config.get("perplexity_weight", 0.1)
        self.perplexity_check_interval = health_config.get("perplexity_check_interval", 10)
        self.max_wake_seconds = health_config.get("max_wake_seconds", 7200)

        memit_config = config.get("memit", {}) or {}
        self.max_active_edits = memit_config.get("max_active_edits", 50)

        # State
        self._last_sleep_time = time.time()
        self._edit_count = 0
        self._edits_since_perplexity = 0
        self._baseline_perplexity = None
        self._current_perplexity = None

    def get_sleep_pressure(self) -> float:
        """Compute current sleep pressure (0.0 - 1.0+).

        Weighted combination of edit, time, and perplexity pressures.
        """
        # Edit pressure (non-linear: ramps up faster as capacity fills)
        edit_ratio = self._edit_count / max(1, self.max_active_edits)
        edit_pressure = min(1.0, edit_ratio ** 1.5)

        # Time pressure
        elapsed = time.time() - self._last_sleep_time
        time_pressure = min(1.0, elapsed / max(1, self.max_wake_seconds))

        # Perplexity pressure
        perplexity_pressure = 0.0
        if self._baseline_perplexity and self._current_perplexity:
            if self._baseline_perplexity > 0:
                ratio = self._current_perplexity / self._baseline_perplexity
                # Pressure rises when perplexity increases (ratio > 1)
                perplexity_pressure = max(0.0, min(1.0, ratio - 1.0))

        pressure = (
            self.edit_weight * edit_pressure +
            self.time_weight * time_pressure +
            self.perplexity_weight * perplexity_pressure
        )

        return pressure

    def should_nap(self) -> bool:
        """Check if pressure exceeds nap threshold."""
        return self.get_sleep_pressure() >= self.nap_threshold

    def should_sleep(self) -> bool:
        """Check if pressure exceeds full sleep threshold."""
        return self.get_sleep_pressure() >= self.sleep_threshold

    def measure_perplexity(self) -> float:
        """Compute perplexity on reference text, update baseline/current.

        Returns the measured perplexity value.
        """
        reference_text = (
            "The quick brown fox jumps over the lazy dog. "
            "In machine learning, neural networks process data through layers. "
            "The capital of France is Paris and Berlin is the capital of Germany."
        )

        ppl = self.backend.compute_perplexity(reference_text)

        if self._baseline_perplexity is None:
            self._baseline_perplexity = ppl

        self._current_perplexity = ppl
        self._edits_since_perplexity = 0

        return ppl

    def record_edit(self, count=1):
        """Record that MEMIT edits were applied.

        Args:
            count: Number of facts injected in this batch
        """
        self._edit_count += count
        self._edits_since_perplexity += count

        # Optionally measure perplexity at intervals
        if (self.perplexity_check_interval > 0 and
                self._edits_since_perplexity >= self.perplexity_check_interval):
            try:
                self.measure_perplexity()
            except Exception:
                pass

    def record_sleep(self, sleep_type="nap", facts_refreshed=0, facts_pruned=0):
        """Record that a sleep cycle completed, adjusting pressure.

        Args:
            sleep_type: "full" or "nap"
            facts_refreshed: Number of degraded facts re-injected this cycle.
            facts_pruned: Number of facts pruned (removed) this cycle.
        """
        self._last_sleep_time = time.time()
        if sleep_type == "full":
            # Re-sync edit count from ledger after maintenance
            self._edit_count = max(0, self._edit_count - facts_pruned)
        # Naps are audit-only — no pressure change

    def get_snapshot(self) -> HealthSnapshot:
        """Return a snapshot of current health metrics."""
        return HealthSnapshot(
            timestamp=time.time(),
            edit_count=self._edit_count,
            perplexity=self._current_perplexity or 0.0,
            coherence_score=1.0,
            sleep_pressure=self.get_sleep_pressure(),
        )

    def to_dict(self) -> dict:
        """Return health status as a dict for API responses."""
        snapshot = self.get_snapshot()
        return {
            "sleep_pressure": round(snapshot.sleep_pressure, 3),
            "edit_count": snapshot.edit_count,
            "perplexity": round(snapshot.perplexity, 2) if snapshot.perplexity else None,
            "baseline_perplexity": round(self._baseline_perplexity, 2) if self._baseline_perplexity else None,
            "nap_threshold": self.nap_threshold,
            "sleep_threshold": self.sleep_threshold,
            "should_nap": self.should_nap(),
            "should_sleep": self.should_sleep(),
            "seconds_since_sleep": round(time.time() - self._last_sleep_time, 0),
        }
