"""Micro-sleep — lightweight background LoRA training on high-priority facts.

Models the human Basic Rest-Activity Cycle (BRAC): ~90-minute ultradian
rhythm where the brain alternates between alert encoding and brief
consolidation dips. During the active phase, high-priority facts accumulate.
When the cycle ticks over, a micro-consolidation pass trains LoRA on them.

Two trigger modes:
  - "cycle" (default): Accumulates facts during the active phase, fires
    micro-sleep when the cycle timer expires AND there are pending facts.
  - "immediate": Fires as soon as a high-priority fact arrives (with cooldown).

Does NOT block chat — runs in a daemon thread.
"""

import threading
import time
from pathlib import Path

from src.memory.facts import QAPair


class MicroSleepController:
    """Runs micro-sleep LoRA passes in the background."""

    def __init__(self, config, backend, trainer, fact_ledger, model_lock,
                 paraphraser=None, runner=None):
        self.config = config
        self.backend = backend
        self.trainer = trainer
        self.fact_ledger = fact_ledger
        self.model_lock = model_lock
        self.paraphraser = paraphraser
        # Shared train/fuse/gate/commit path (same code as full sleep)
        self.runner = runner
        self.last_result = None

        micro_cfg = config.get("micro_sleep", {}) or {}
        self.enabled = micro_cfg.get("enabled", False)
        self.priority_threshold = micro_cfg.get("priority_threshold", 0.7)
        self.max_facts = micro_cfg.get("max_facts", 3)
        self.iters = micro_cfg.get("iters", 20)
        self.cooldown_seconds = micro_cfg.get("cooldown_seconds", 120)
        self.min_age_seconds = micro_cfg.get("min_age_seconds", 60)
        self.replay_count = micro_cfg.get("replay_count", 2)

        # Ultradian cycle mode
        self.trigger_mode = micro_cfg.get("trigger_mode", "cycle")
        self.cycle_seconds = micro_cfg.get("cycle_seconds", 5400)  # 90 minutes

        self._last_micro_sleep = 0.0
        self._cycle_start = time.time()  # Start of current active phase
        self._pending_high_priority = False  # High-priority fact arrived this cycle
        self._running = False
        self._lock = threading.Lock()
        self._micro_sleep_count = 0

    def maybe_trigger(self, fact_priority, background_sleep_manager=None):
        """Check if a micro-sleep should fire. Called after fact buffering.

        In "cycle" mode: marks the cycle as having pending facts. The actual
        micro-sleep fires when the cycle timer expires (checked each call).

        In "immediate" mode: fires right away if priority exceeds threshold
        and cooldown has elapsed.

        Passing fact_priority >= 1.0 forces a trigger (used by /microsleep).
        """
        if not self.enabled:
            return False
        if background_sleep_manager and background_sleep_manager.is_sleeping:
            return False

        forced = fact_priority >= 1.0

        if not forced and fact_priority < self.priority_threshold:
            return False

        if self.trigger_mode == "cycle" and not forced:
            return self._maybe_trigger_cycle(background_sleep_manager)
        else:
            return self._maybe_trigger_immediate(background_sleep_manager)

    def _maybe_trigger_cycle(self, background_sleep_manager):
        """Cycle mode: accumulate during active phase, fire at cycle boundary."""
        now = time.time()

        # Mark that a high-priority fact arrived this cycle
        self._pending_high_priority = True

        # Check if the cycle has elapsed
        elapsed_in_cycle = now - self._cycle_start
        if elapsed_in_cycle < self.cycle_seconds:
            remaining = self.cycle_seconds - elapsed_in_cycle
            print(f"  [Micro-sleep] Fact queued for cycle consolidation "
                  f"({remaining:.0f}s remaining)")
            return False

        # Cycle expired and we have pending facts — fire
        return self._fire(background_sleep_manager)

    def _maybe_trigger_immediate(self, background_sleep_manager):
        """Immediate mode: fire now if cooldown elapsed."""
        with self._lock:
            if self._running:
                return False
            elapsed = time.time() - self._last_micro_sleep
            if elapsed < self.cooldown_seconds:
                return False

        return self._fire(background_sleep_manager)

    def check_cycle(self, background_sleep_manager=None):
        """Called periodically (e.g., every turn) to check if the cycle timer
        has expired. Handles the case where no new facts arrive but
        the cycle boundary passes.
        """
        if not self.enabled or self.trigger_mode != "cycle":
            return False
        if not self._pending_high_priority:
            return False

        now = time.time()
        elapsed_in_cycle = now - self._cycle_start
        if elapsed_in_cycle < self.cycle_seconds:
            return False

        return self._fire(background_sleep_manager)

    def _fire(self, background_sleep_manager):
        """Actually start the micro-sleep thread."""
        with self._lock:
            if self._running:
                return False
            self._running = True

        # Reset cycle state
        self._cycle_start = time.time()
        self._pending_high_priority = False

        candidates = self._select_facts()
        if not candidates:
            with self._lock:
                self._running = False
            return False

        thread = threading.Thread(
            target=self._execute,
            args=(candidates,),
            daemon=True,
            name="micro-sleep",
        )
        thread.start()
        return True

    def _select_facts(self):
        """Select up to max_facts high-priority facts for micro-training.

        Prioritizes: high priority, never trained or degraded, old enough.
        """
        candidates = []
        for entry in self.fact_ledger.get_active_facts():
            if entry.get("graduated", False):
                continue

            qa_dict = entry["qa"]
            priority = qa_dict.get("priority", 0.5)
            last_trained = entry.get("last_trained", 0.0)
            degrade_count = entry.get("degrade_count", 0)

            # Skip recently trained (unless degraded)
            age = time.time() - last_trained if last_trained > 0 else float('inf')
            if age < self.min_age_seconds and degrade_count == 0:
                continue

            qa = QAPair.from_dict(qa_dict)

            # Score: priority + boost for degraded facts
            score = priority + (0.2 * degrade_count)
            candidates.append((score, qa, entry["fact_id"]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:self.max_facts]

    def _execute(self, candidates):
        """Run micro-sleep training in background thread.

        Goes through ConsolidationRunner: train, fuse, load candidate, GATES
        (perplexity / bleed / regression), commit or roll back. No graduation
        tests (that is full sleep's job) — micro-sleep only trains.
        """
        try:
            self._micro_sleep_count += 1
            cycle_id = f"micro_{self._micro_sleep_count:04d}"
            entries = [self.fact_ledger.get_entry(c[2]) for c in candidates]
            entries = [e for e in entries if e is not None and not e.get("graduated", False)]
            if not entries:
                return

            print(f"  [Micro-sleep] Training on {len(entries)} fact(s), "
                  f"{self.iters} iters...")

            if self.runner is None:
                from src.sleep.consolidation import ConsolidationRunner
                from src.sleep.probe import RecallProbe
                probe = RecallProbe(self.backend, self.config.context.get("system_prompt", ""))
                self.runner = ConsolidationRunner(
                    self.config, self.backend, self.trainer, self.fact_ledger, probe,
                    paraphraser=self.paraphraser)

            result = self.runner.run(
                cycle_id, entries, iters=self.iters, test_graduation=False,
                replay_count=self.replay_count, weighted=True,
                ref_text=self._ref_text(),
            )
            self.last_result = result
            if result["status"] == "committed":
                print(f"  [Micro-sleep] Complete ({result['trained']} facts trained, "
                      f"{result['replayed']} replayed)")
            else:
                print(f"  [Micro-sleep] {result['status']}"
                      + (f" (gate: {result['failed_gate']})" if result.get("failed_gate") else ""))

        except Exception as e:
            print(f"  [Micro-sleep] Failed: {e}")
        finally:
            with self._lock:
                self._running = False
                self._last_micro_sleep = time.time()

    def _ref_text(self):
        """Perplexity reference text (identity data), or None."""
        try:
            import json
            identity_file = Path(self.config.paths["core_identity"]) / "identity.jsonl"
            if not identity_file.exists():
                return None
            texts = []
            with open(identity_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            texts.append(json.loads(line).get("text", ""))
                        except json.JSONDecodeError:
                            continue
            return " ".join(texts)[:2000] if texts else None
        except Exception:
            return None

    @property
    def is_running(self):
        with self._lock:
            return self._running

    @property
    def cycle_remaining(self):
        """Seconds remaining in the current ultradian cycle."""
        elapsed = time.time() - self._cycle_start
        return max(0, self.cycle_seconds - elapsed)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "micro_sleep_count": self._micro_sleep_count,
            "trigger_mode": self.trigger_mode,
            "priority_threshold": self.priority_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "cycle_seconds": self.cycle_seconds,
            "cycle_remaining": round(self.cycle_remaining),
            "pending_high_priority": self._pending_high_priority,
            "last_micro_sleep": self._last_micro_sleep,
            "last_result": ({k: self.last_result.get(k) for k in
                             ("status", "failed_gate", "trained", "replayed", "cycle_id")}
                            if self.last_result else None),
        }
