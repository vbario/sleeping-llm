"""Orchestrator — the wake/sleep state machine.

Coordinates the full lifecycle:
  wake (chat) → detect sleep trigger → LoRA consolidation → wake

MEMIT removed — facts stored in FactLedger, injected via system prompt,
consolidated via LoRA training during sleep.
"""

import shutil
from pathlib import Path

from src.concurrency.model_lock import ModelLock
from src.memory.facts import FactLedger
from src.memory.health import HealthMonitor
from src.memory.identity import IdentityManager
from src.memory.session_tracker import SessionTracker
from src.sleep.background_sleep import BackgroundSleepManager
from src.sleep.curator import Curator
from src.sleep.full_sleep import FullSleepController
from src.sleep.micro_sleep import MicroSleepController
from src.sleep.nap import NapController
from src.sleep.trainer import SleepTrainer
from src.sleep.validator import SleepValidator
from src.wake.chat import Chat
from src.wake.context import ContextManager
from src.wake.extractor import FactExtractor
from src.wake.fact_buffer import FactBuffer
from src.wake.logger import ConversationLogger
from src.wake.surprise import SurpriseEstimator


class Orchestrator:
    """Central coordinator for the sleeping LLM system."""

    def __init__(self, config, disable_memit=False):
        self.config = config
        self.sleep_cycle_count = 0
        self.nap_cycle_count = 0

        # Initialize backend
        backend_type = config.model.get("backend", "mlx")
        if backend_type == "torch":
            from src.backend.torch_backend import TorchBackend
            self.backend = TorchBackend(config)
        else:
            from src.backend.mlx_backend import MLXBackend
            self.backend = MLXBackend(config)
        print("Loading model...")
        self.backend.load()
        print("Model loaded.")

        # Initialize concurrency components
        self.model_lock = ModelLock()
        self.backend.model_lock = self.model_lock
        self.background_sleep = BackgroundSleepManager(self.model_lock)

        # Initialize wake components
        self.logger = ConversationLogger(config)
        self.context = ContextManager(config, self.backend)
        self.chat = Chat(self.backend, self.context, self.logger, config)
        self.chat.set_sleep_callback(self._on_sleep_trigger)
        self.chat.set_nap_callback(self._on_nap_trigger)

        # Initialize sleep components
        self.curator = Curator(config, self.backend)
        self.validator = SleepValidator(config, self.backend)
        self.identity = IdentityManager(config, self.backend)
        self.session_tracker = SessionTracker(config)

        # Initialize fact ledger (replaces EditLedger + MEMIT)
        ledger_path = config.paths.get("memit_ledger", "data/memit/ledger.json")
        self.fact_ledger = FactLedger(ledger_path)
        self.fact_extractor = FactExtractor(config, self.backend)
        self.health_monitor = HealthMonitor(config, self.backend, self.fact_ledger)

        # Initialize trainer (for LoRA consolidation during sleep)
        lora_enabled = (config.get("lora", {}) or {}).get("enabled", False)
        self.trainer = SleepTrainer(config, self.backend) if lora_enabled else None

        # Initialize micro-sleep (priority-triggered background LoRA)
        self.micro_sleep = None
        if self.trainer and lora_enabled:
            micro_cfg = config.get("micro_sleep", {}) or {}
            if micro_cfg.get("enabled", False):
                self.micro_sleep = MicroSleepController(
                    config, self.backend, self.trainer,
                    self.fact_ledger, self.model_lock,
                )

        # Initialize nap and full sleep controllers
        self.nap_controller = NapController(
            config, self.backend, self.fact_ledger,
        )
        self.full_sleep_controller = FullSleepController(
            config, self.backend, self.fact_ledger,
            self.curator, self.validator, self.session_tracker,
            self.health_monitor, self.fact_extractor,
            trainer=self.trainer,
        )

        # Wire known facts into context (system prompt injection)
        # Only non-graduated facts appear in the prompt
        self.context.set_facts_provider(self.fact_ledger.get_active_qa_pairs)

        # Wire extraction components into chat
        self.chat.set_extraction_components(
            self.fact_extractor, self.fact_ledger, self.health_monitor,
        )
        self.chat._background_sleep = self.background_sleep

        # Initialize consolidation-moment components (surprise-gated buffering)
        cm_config = config.get("consolidation_moment", {}) or {}
        cm_enabled = cm_config.get("enabled", False)
        if cm_enabled:
            self.fact_buffer = FactBuffer(config, self.fact_ledger, self.health_monitor)
            self.surprise_estimator = SurpriseEstimator(config, self.backend)
            self.chat.set_consolidation_components(
                self.fact_buffer, self.surprise_estimator,
            )
            self.health_monitor.set_fact_buffer(self.fact_buffer)
        else:
            self.fact_buffer = None
            self.surprise_estimator = None

        # Wire micro-sleep into chat
        if self.micro_sleep:
            self.chat.set_micro_sleep(self.micro_sleep, self.background_sleep)

        # Seed identity data if first run
        self.identity.seed_defaults()

    def run(self):
        """Main loop — interactive chat with sleep cycles."""
        print("\n=== Sleeping LLM ===")
        print(f"Model: {self.config.model['path']}")
        trigger_mode = self.config.sleep.get("trigger_mode", "turns")
        facts = self.fact_ledger.get_active_fact_count()
        graduated = self.fact_ledger.get_graduated_count()
        print(f"Facts: {facts} ({graduated} graduated) | Trigger: {trigger_mode}")
        print(f"Commands: /sleep, /nap, /consolidate, /status, /compact, /quit")
        print("=" * 40)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye.")
                break

            response = self.chat.process_input(user_input)
            if response is not None:
                print(f"\nAssistant: {response}\n")

    # --- Web-facing methods ---

    def process_message(self, user_input):
        """Process a single message (for web UI). Returns response text or None."""
        if user_input.strip() == "/quit":
            return None
        return self.chat.process_input(user_input)

    def process_message_stream(self, user_input):
        """Process a message with streaming. Yields token strings."""
        if user_input.strip() == "/quit":
            return

        # Swap callbacks to flag-only (don't block during streaming)
        self._auto_sleep_pending = False
        self._auto_nap_pending = False
        original_sleep_cb = self.chat._sleep_callback
        original_nap_cb = self.chat._nap_callback
        self.chat._sleep_callback = lambda t: setattr(self, '_auto_sleep_pending', True)
        self.chat._nap_callback = lambda t: setattr(self, '_auto_nap_pending', True)

        yield from self.chat.process_input_stream(user_input)

        # Restore original callbacks
        self.chat._sleep_callback = original_sleep_cb
        self.chat._nap_callback = original_nap_cb

        # Signal auto-sleep or auto-nap to caller
        if self._auto_sleep_pending:
            self._auto_sleep_pending = False
            yield {"__auto_sleep__": True}
        elif self._auto_nap_pending:
            self._auto_nap_pending = False
            yield {"__auto_nap__": True}

    def trigger_sleep_web(self):
        """Trigger sleep and yield progress dicts for each step."""
        self.sleep_cycle_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        # Flush fact buffer before sleep
        if self.fact_buffer and not self.fact_buffer.is_empty:
            self.fact_buffer.consolidate(reason="pre_sleep")

        try:
            for progress in self.full_sleep_controller.execute_sleep_streaming(
                cycle_id, "full", self._gather_new_messages,
            ):
                yield progress
        except Exception as e:
            ts = self.full_sleep_controller.total_steps
            yield {"step": 0, "total": ts, "label": "Error", "status": "error", "detail": str(e)}
            return

        self.health_monitor.record_sleep("full")

        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        ts = self.full_sleep_controller.total_steps
        yield {"step": ts + 1, "total": ts, "label": "Awake", "status": "done", "detail": "Memories consolidated"}

    def trigger_nap_web(self):
        """Trigger nap and yield progress dicts for each step."""
        self.nap_cycle_count += 1
        cycle_id = f"nap_{self.nap_cycle_count:04d}"

        try:
            yield from self.nap_controller.execute_nap_streaming(cycle_id)
        except Exception as e:
            yield {"step": 0, "total": 2, "label": "Error", "status": "error", "detail": str(e)}
            return

        self.health_monitor.record_sleep("nap")

        yield {"step": 3, "total": 2, "label": "Awake", "status": "done", "detail": "Nap complete"}

    def trigger_sleep_background(self, callback=None):
        """Trigger sleep in a background thread (non-blocking)."""
        if self.background_sleep.is_sleeping:
            return False

        # Wait for micro-sleep to finish before starting full sleep
        if self.micro_sleep and self.micro_sleep.is_running:
            return False

        # Flush fact buffer before sleep
        if self.fact_buffer and not self.fact_buffer.is_empty:
            self.fact_buffer.consolidate(reason="pre_sleep")

        self.sleep_cycle_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        def sleep_generator():
            yield from self.full_sleep_controller.execute_sleep_streaming(
                cycle_id, "full", self._gather_new_messages,
            )

        def on_complete(result):
            self.health_monitor.record_sleep("full")
            if self.context.recent_messages:
                self.context.compact()
            self.chat.reset_turn_count()
            self.context.reset(keep_summary=True)
            self.logger = ConversationLogger(self.config)
            self.chat.logger = self.logger
            if callback:
                callback(result)

        return self.background_sleep.start_sleep(
            sleep_generator, on_complete, sleep_type="sleep",
        )

    def trigger_nap_background(self, callback=None):
        """Trigger nap in a background thread (non-blocking)."""
        if self.background_sleep.is_sleeping:
            return False

        self.nap_cycle_count += 1
        cycle_id = f"nap_{self.nap_cycle_count:04d}"

        def nap_generator():
            yield from self.nap_controller.execute_nap_streaming(cycle_id)

        def on_complete(result):
            self.health_monitor.record_sleep("nap")
            if callback:
                callback(result)

        return self.background_sleep.start_sleep(
            nap_generator, on_complete, sleep_type="nap",
        )

    def get_sleep_state(self):
        """Return current background sleep state."""
        return self.background_sleep.to_dict()

    def get_status(self):
        """Return current system status as a dict."""
        token_count = self.context.get_token_count()
        max_tokens = self.context.max_tokens
        return {
            "session_id": self.logger.session_id,
            "turn_count": self.chat.turn_count,
            "context_tokens": token_count,
            "context_max": max_tokens,
            "context_pct": round((token_count / max_tokens) * 100, 1) if max_tokens else 0,
            "has_summary": self.context.summary is not None,
            "messages_in_context": len(self.context.recent_messages),
            "sleep_cycles": self.sleep_cycle_count,
            "nap_cycles": self.nap_cycle_count,
            "model": self.config.model["path"],
            "consumed_sessions": self.session_tracker.get_consumed_count(),
            "total_sessions": self.session_tracker.get_total_session_count(),
            "total_facts": self.fact_ledger.get_active_fact_count(),
            "graduated_facts": self.fact_ledger.get_graduated_count(),
            "sleep_pressure": round(self.health_monitor.get_sleep_pressure(), 3),
            "health": self.health_monitor.to_dict(),
            "fact_buffer": self.fact_buffer.to_dict() if self.fact_buffer else None,
            "surprise_estimator": self.surprise_estimator.to_dict() if self.surprise_estimator else None,
            "background_sleep": self.background_sleep.to_dict(),
            "micro_sleep": self.micro_sleep.to_dict() if self.micro_sleep else None,
            "model_lock": self.model_lock.stats(),
        }

    def get_current_messages(self):
        """Return current session messages for history display."""
        return self.logger.get_session_messages()

    def reset_weights(self):
        """Reset model to base weights. Clears ledger."""
        # Clear fact ledger
        self.fact_ledger.clear_all()

        # Clear fact buffer (discard unconsolidated facts)
        if self.fact_buffer:
            self.fact_buffer.clear()

        # Reload base model
        self.backend.reload(self.config.model["path"])

        # Reset counters
        self.sleep_cycle_count = 0
        self.nap_cycle_count = 0
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=False)

        return {"status": "ok", "message": "Weights reset to base model"}

    def factory_reset(self):
        """Full reset — weights, conversations, and fact data."""
        # Reset weights first
        self.reset_weights()

        # Delete conversation logs
        conversations_dir = Path(self.config.paths["conversations"])
        if conversations_dir.exists():
            shutil.rmtree(conversations_dir)
            conversations_dir.mkdir(parents=True, exist_ok=True)

        # Delete fact data
        memit_dir = Path(self.config.paths.get("memit_data", "data/memit"))
        if memit_dir.exists():
            shutil.rmtree(memit_dir)
            memit_dir.mkdir(parents=True, exist_ok=True)

        # Start a fresh logger
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        return {"status": "ok", "message": "Factory reset complete. All data cleared."}

    # --- Internal methods ---

    def _on_sleep_trigger(self, trigger_type):
        """Called when a sleep cycle should begin."""
        self.sleep_cycle_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        print(f"\n{'=' * 40}")
        print(f"  Entering sleep (cycle {cycle_id})...")
        print(f"  Trigger: {trigger_type}")
        print(f"{'=' * 40}\n")

        # Flush fact buffer before sleep
        if self.fact_buffer and not self.fact_buffer.is_empty:
            print(f"  Pre-sleep consolidation: {self.fact_buffer.size} buffered fact(s)")
            self.fact_buffer.consolidate(reason="pre_sleep")

        try:
            result = self.full_sleep_controller.execute_sleep(
                cycle_id, "full", self._gather_new_messages,
            )
        except Exception as e:
            print(f"  Sleep cycle failed: {e}")
            print("  Continuing with current model.\n")
            return

        self.health_monitor.record_sleep("full")

        # Compact context before resetting
        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)

        # Start a fresh session file
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        print(f"\n{'=' * 40}")
        print(f"  Awake. Memories consolidated.")
        print(f"{'=' * 40}\n")

    def _on_nap_trigger(self, trigger_type):
        """Called when a nap should begin."""
        self.nap_cycle_count += 1
        cycle_id = f"nap_{self.nap_cycle_count:04d}"

        print(f"\n{'=' * 40}")
        print(f"  Taking a nap (cycle {cycle_id})...")
        print(f"  Trigger: {trigger_type}")
        print(f"{'=' * 40}\n")

        try:
            result = self.nap_controller.execute_nap(cycle_id)
            print(f"  Nap result: {result['status']}")
            if result.get("failed"):
                print(f"  Failed facts: {result['failed']}/{result['audited']}")
        except Exception as e:
            print(f"  Nap failed: {e}")
            print("  Continuing with current model.\n")
            return

        self.health_monitor.record_sleep("nap")

        print(f"\n{'=' * 40}")
        print(f"  Awake. Nap complete.")
        print(f"{'=' * 40}\n")

    def _gather_new_messages(self):
        """Gather messages only from unconsumed sessions."""
        all_messages = []
        unconsumed = self.session_tracker.get_unconsumed_sessions()

        if not unconsumed:
            return self.logger.get_session_messages(), []

        for session_path in unconsumed:
            entries = ConversationLogger.load_session(session_path)
            for entry in entries:
                all_messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })

        return all_messages, unconsumed
