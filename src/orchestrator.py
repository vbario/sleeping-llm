"""Orchestrator — the wake/sleep state machine.

Coordinates the full lifecycle:
  wake (chat) → detect sleep trigger → MEMIT maintenance → wake
"""

import shutil
from pathlib import Path

from src.concurrency.model_lock import ModelLock
from src.memory.health import HealthMonitor
from src.memory.identity import IdentityManager
from src.memory.memit import EditLedger, MemitEngine
from src.memory.session_tracker import SessionTracker
from src.sleep.background_sleep import BackgroundSleepManager
from src.sleep.curator import Curator
from src.sleep.full_sleep import FullSleepController
from src.sleep.nap import NapController
from src.sleep.trainer import SleepTrainer
from src.sleep.validator import SleepValidator
from src.wake.chat import Chat
from src.wake.context import ContextManager
from src.wake.extractor import FactExtractor
from src.wake.logger import ConversationLogger


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
        self.backend.model_lock = self.model_lock  # Enable lock-aware generate/reload
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

        # Initialize MEMIT components
        memit_enabled = config.get("memit.enabled", True) and not disable_memit
        ledger_path = config.paths.get("memit_ledger", "data/memit/ledger.json")
        self.edit_ledger = EditLedger(ledger_path)
        self.memit_engine = MemitEngine(config, self.backend, self.edit_ledger)
        if not memit_enabled:
            self.memit_engine.enabled = False
        self.fact_extractor = FactExtractor(config, self.backend)
        self.health_monitor = HealthMonitor(config, self.backend, self.edit_ledger)

        # Initialize trainer (for LoRA consolidation during sleep)
        lora_enabled = (config.get("lora", {}) or {}).get("enabled", False)
        self.trainer = SleepTrainer(config, self.backend) if lora_enabled else None

        # Initialize nap and full sleep controllers
        self.nap_controller = NapController(
            config, self.backend, self.memit_engine, self.edit_ledger,
        )
        self.full_sleep_controller = FullSleepController(
            config, self.backend, self.memit_engine, self.edit_ledger,
            self.curator, self.validator, self.session_tracker,
            self.health_monitor, self.fact_extractor,
            trainer=self.trainer,
        )

        # Reload persisted MEMIT edits (survives process restarts)
        if self.memit_engine.enabled:
            self.memit_engine.reload_persisted_edits()

        # Wire MEMIT components into chat
        self.chat.set_memit_components(
            self.fact_extractor, self.memit_engine, self.health_monitor,
        )
        self.chat._background_sleep = self.background_sleep

        # Seed identity data if first run
        self.identity.seed_defaults()

    def run(self):
        """Main loop — interactive chat with sleep cycles."""
        print("\n=== Sleeping LLM ===")
        print(f"Model: {self.config.model['path']}")
        memit_status = "ON" if self.memit_engine.enabled else "OFF"
        trigger_mode = self.config.sleep.get("trigger_mode", "turns")
        print(f"MEMIT: {memit_status} | Trigger: {trigger_mode}")
        print(f"Commands: /sleep, /nap, /status, /compact, /quit")
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
        """Process a message with streaming. Yields token strings.

        After all tokens, may yield a dict with sleep/nap trigger signals:
          {"__auto_sleep__": True} or {"__auto_nap__": True}
        """
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

        facts_refreshed = 0
        facts_pruned = 0
        try:
            for progress in self.full_sleep_controller.execute_sleep_streaming(
                cycle_id, "full", self._gather_new_messages,
            ):
                if isinstance(progress, dict):
                    if "facts_refreshed" in progress:
                        facts_refreshed = progress["facts_refreshed"]
                    if "facts_pruned" in progress:
                        facts_pruned = progress["facts_pruned"]
                yield progress
        except Exception as e:
            ts = self.full_sleep_controller.total_steps
            yield {"step": 0, "total": ts, "label": "Error", "status": "error", "detail": str(e)}
            return

        self.health_monitor.record_sleep("full",
                                          facts_refreshed=facts_refreshed,
                                          facts_pruned=facts_pruned)

        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        ts = self.full_sleep_controller.total_steps
        yield {"step": ts + 1, "total": ts, "label": "Awake", "status": "done", "detail": "Memories maintained"}

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
        """Trigger sleep in a background thread (non-blocking).

        Chat continues during sleep. Model access is synchronized via read-write lock.

        Args:
            callback: Optional function called when sleep completes.

        Returns:
            True if sleep started, False if already sleeping.
        """
        if self.background_sleep.is_sleeping:
            return False

        self.sleep_cycle_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        def sleep_generator():
            yield from self.full_sleep_controller.execute_sleep_streaming(
                cycle_id, "full", self._gather_new_messages,
            )

        def on_complete(result):
            refreshed = result.get("facts_refreshed", 0) if result else 0
            pruned = result.get("facts_pruned", 0) if result else 0
            self.health_monitor.record_sleep("full",
                                              facts_refreshed=refreshed,
                                              facts_pruned=pruned)
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
        """Trigger nap in a background thread (non-blocking).

        Args:
            callback: Optional function called when nap completes.

        Returns:
            True if nap started, False if already sleeping.
        """
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
            "memit_enabled": self.memit_engine.enabled,
            "memit_edits": self.memit_engine.get_active_edit_count(),
            "memit_facts": self.memit_engine.get_active_fact_count(),
            "sleep_pressure": round(self.health_monitor.get_sleep_pressure(), 3),
            "health": self.health_monitor.to_dict(),
            "background_sleep": self.background_sleep.to_dict(),
            "model_lock": self.model_lock.stats(),
        }

    def get_current_messages(self):
        """Return current session messages for history display."""
        return self.logger.get_session_messages()

    def reset_weights(self):
        """Reset model to base weights. Reverts all MEMIT edits and clears ledger."""
        # Clear MEMIT edits (reverts weight deltas in memory)
        self.memit_engine.revert_all_active()
        self.edit_ledger.clear_all()

        # Reload base model
        self.backend.reload(self.config.model["path"])

        # Reset counters
        self.sleep_cycle_count = 0
        self.nap_cycle_count = 0
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=False)

        return {"status": "ok", "message": "Weights reset to base model"}

    def factory_reset(self):
        """Full reset — weights, conversations, and MEMIT data."""
        # Reset weights first (also clears MEMIT edits)
        self.reset_weights()

        # Delete conversation logs
        conversations_dir = Path(self.config.paths["conversations"])
        if conversations_dir.exists():
            shutil.rmtree(conversations_dir)
            conversations_dir.mkdir(parents=True, exist_ok=True)

        # Delete MEMIT data
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

        try:
            result = self.full_sleep_controller.execute_sleep(
                cycle_id, "full", self._gather_new_messages,
            )
        except Exception as e:
            print(f"  Sleep cycle failed: {e}")
            print("  Continuing with current model.\n")
            return

        refreshed = result.get("facts_refreshed", 0) if result else 0
        pruned = result.get("facts_pruned", 0) if result else 0
        self.health_monitor.record_sleep("full",
                                          facts_refreshed=refreshed,
                                          facts_pruned=pruned)

        # Compact context before resetting so summary survives into next wake phase
        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)

        # Start a fresh session file so post-sleep messages don't mix with consumed sessions
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        print(f"\n{'=' * 40}")
        print(f"  Awake. Memories maintained.")
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
            if result.get("degraded"):
                print(f"  Degraded facts: {result['degraded']}/{result['audited']}")
        except Exception as e:
            print(f"  Nap failed: {e}")
            print("  Continuing with current model.\n")
            return

        self.health_monitor.record_sleep("nap")

        print(f"\n{'=' * 40}")
        print(f"  Awake. Nap complete.")
        print(f"{'=' * 40}\n")

    def _gather_new_messages(self):
        """Gather messages only from unconsumed sessions.

        Returns:
            Tuple of (messages_list, consumed_session_paths)
        """
        all_messages = []
        unconsumed = self.session_tracker.get_unconsumed_sessions()

        if not unconsumed:
            # No new sessions — use current session messages as fallback
            return self.logger.get_session_messages(), []

        for session_path in unconsumed:
            entries = ConversationLogger.load_session(session_path)
            for entry in entries:
                all_messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })

        return all_messages, unconsumed
