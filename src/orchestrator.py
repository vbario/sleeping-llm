"""Orchestrator — the wake/sleep state machine.

Coordinates the full lifecycle:
  wake (chat) → detect sleep trigger → curate → train → validate → fuse → wake
"""

import json
import shutil
import time
from pathlib import Path

from src.backend.mlx_backend import MLXBackend
from src.memory.checkpoints import CheckpointManager
from src.memory.identity import IdentityManager
from src.memory.replay import ReplayBuffer
from src.memory.session_tracker import SessionTracker
from src.sleep.curator import Curator
from src.sleep.dreamer import Dreamer
from src.sleep.trainer import SleepTrainer
from src.sleep.validator import SleepValidator
from src.wake.chat import Chat
from src.wake.context import ContextManager
from src.wake.logger import ConversationLogger


class Orchestrator:
    """Central coordinator for the sleeping LLM system."""

    def __init__(self, config):
        self.config = config
        self.sleep_cycle_count = 0
        self.light_sleep_count = 0

        # Initialize backend
        self.backend = MLXBackend(config)
        print("Loading model...")
        self.backend.load()
        print("Model loaded.")

        # Initialize wake components
        self.logger = ConversationLogger(config)
        self.context = ContextManager(config, self.backend)
        self.chat = Chat(self.backend, self.context, self.logger, config)
        self.chat.set_sleep_callback(self._on_sleep_trigger)

        # Initialize sleep components
        self.curator = Curator(config, self.backend)
        self.replay_buffer = ReplayBuffer(config)
        self.trainer = SleepTrainer(config, self.backend, self.replay_buffer)
        self.validator = SleepValidator(config, self.backend)
        self.dreamer = Dreamer(config, self.backend)
        self.checkpoints = CheckpointManager(config)
        self.identity = IdentityManager(config, self.backend)
        self.session_tracker = SessionTracker(config)

        # Seed identity data if first run
        self.identity.seed_defaults()

    def run(self):
        """Main loop — interactive chat with sleep cycles."""
        print("\n=== Sleeping LLM ===")
        print(f"Model: {self.config.model['path']}")
        print(f"Sleep after: {self.config.sleep['light_sleep_turns']} turns")
        print(f"Commands: /sleep (manual), /status, /compact, /quit")
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
        yield from self.chat.process_input_stream(user_input)

    def trigger_sleep_web(self):
        """Trigger sleep and yield progress dicts for each step."""
        self.sleep_cycle_count += 1
        self.light_sleep_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        deep_interval = self.config.sleep["deep_sleep_interval"]
        is_deep = self.light_sleep_count >= deep_interval
        sleep_type = "deep" if is_deep else "light"

        progress = []

        def progress_cb(event):
            progress.append(event)

        try:
            yield from self._execute_sleep_streaming(cycle_id, sleep_type)
        except Exception as e:
            yield {"step": 0, "total": 6, "label": "Error", "status": "error", "detail": str(e)}
            return

        if is_deep:
            self.light_sleep_count = 0

        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        yield {"step": 7, "total": 6, "label": "Awake", "status": "done", "detail": "Memories integrated"}

    def get_status(self):
        """Return current system status as a dict."""
        token_count = self.context.get_token_count()
        max_tokens = self.context.max_tokens
        buffer_stats = self.replay_buffer.stats()
        return {
            "session_id": self.logger.session_id,
            "turn_count": self.chat.turn_count,
            "context_tokens": token_count,
            "context_max": max_tokens,
            "context_pct": round((token_count / max_tokens) * 100, 1) if max_tokens else 0,
            "has_summary": self.context.summary is not None,
            "messages_in_context": len(self.context.recent_messages),
            "replay_buffer": buffer_stats,
            "sleep_cycles": self.sleep_cycle_count,
            "model": self.config.model["path"],
            "consumed_sessions": self.session_tracker.get_consumed_count(),
            "total_sessions": self.session_tracker.get_total_session_count(),
        }

    def get_current_messages(self):
        """Return current session messages for history display."""
        return self.logger.get_session_messages()

    def reset_weights(self):
        """Reset model to base weights. Clears current model and checkpoints."""
        # Delete current model
        current_dir = Path(self.config.paths["current_model"])
        if current_dir.exists():
            shutil.rmtree(current_dir)

        # Delete checkpoints
        checkpoints_dir = Path(self.config.paths["checkpoints"])
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Delete adapters
        adapters_dir = Path(self.config.paths["adapters"])
        if adapters_dir.exists():
            shutil.rmtree(adapters_dir)
            adapters_dir.mkdir(parents=True, exist_ok=True)

        # Reload base model
        self.backend.reload(self.config.model["path"])

        # Reset counters
        self.sleep_cycle_count = 0
        self.light_sleep_count = 0
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=False)

        return {"status": "ok", "message": "Weights reset to base model"}

    def factory_reset(self):
        """Full reset — weights, training data, replay buffer, sessions manifest."""
        # Reset weights first
        self.reset_weights()

        # Delete training data
        training_dir = Path(self.config.paths["training"])
        if training_dir.exists():
            shutil.rmtree(training_dir)
            training_dir.mkdir(parents=True, exist_ok=True)

        # Delete replay buffer
        replay_dir = Path(self.config.paths["replay_buffer"])
        if replay_dir.exists():
            shutil.rmtree(replay_dir)
            replay_dir.mkdir(parents=True, exist_ok=True)

        # Delete conversation logs
        conversations_dir = Path(self.config.paths["conversations"])
        if conversations_dir.exists():
            shutil.rmtree(conversations_dir)
            conversations_dir.mkdir(parents=True, exist_ok=True)

        # Start a fresh logger
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        return {"status": "ok", "message": "Factory reset complete. All data cleared."}

    # --- Internal methods ---

    def _on_sleep_trigger(self, trigger_type):
        """Called when a sleep cycle should begin."""
        self.sleep_cycle_count += 1
        self.light_sleep_count += 1
        cycle_id = f"{self.sleep_cycle_count:04d}"

        # Determine sleep depth
        deep_interval = self.config.sleep["deep_sleep_interval"]
        is_deep = self.light_sleep_count >= deep_interval

        sleep_type = "deep" if is_deep else "light"
        print(f"\n{'=' * 40}")
        print(f"  Entering {sleep_type} sleep (cycle {cycle_id})...")
        print(f"  Trigger: {trigger_type}")
        print(f"{'=' * 40}\n")

        try:
            self._execute_sleep(cycle_id, sleep_type)
        except Exception as e:
            print(f"  Sleep cycle failed: {e}")
            print("  Continuing with current model.\n")
            return

        if is_deep:
            self.light_sleep_count = 0

        # Compact context before resetting so summary survives into next wake phase
        if self.context.recent_messages:
            self.context.compact()
        self.chat.reset_turn_count()
        self.context.reset(keep_summary=True)

        # Start a fresh session file so post-sleep messages don't mix with consumed sessions
        self.logger = ConversationLogger(self.config)
        self.chat.logger = self.logger

        print(f"\n{'=' * 40}")
        print(f"  Awake. Memories integrated.")
        print(f"{'=' * 40}\n")

    def _execute_sleep(self, cycle_id, sleep_type):
        """Run the full sleep pipeline."""
        start_time = time.time()

        # 1. Pre-sleep evaluation
        print("  [1/6] Running pre-sleep evaluation...")
        pre_score = self.validator.evaluate()
        print(f"        Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})")

        # 2. Curate training data (only new sessions)
        print("  [2/6] Curating training data...")
        messages, consumed_sessions = self._gather_new_messages()
        print(f"        {len(consumed_sessions)} new session(s) to process")
        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected for training")

        if not curated:
            print("        No training data after curation. Skipping sleep.")
            return

        # 3. Add curated data to replay buffer
        print("  [3/6] Updating replay buffer...")
        self.replay_buffer.add(curated)
        stats = self.replay_buffer.stats()
        print(f"        Buffer: {stats['count']} items, avg priority: {stats.get('avg_priority', 0):.2f}")

        # 4. Dream (deep sleep only)
        if sleep_type == "deep":
            print("  [4/6] Dreaming (REM)...")
            recent = [ex["messages"] for ex in curated[:10]]
            dreams = self.dreamer.dream(recent)
            print(f"        Generated {len(dreams)} dream sequences")
            # Add dreams to training data
            if dreams:
                dream_data = self.dreamer.dream_to_training_data(dreams, self.backend)
                # Append to the cycle's training file
                training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
                train_file = training_dir / "train.jsonl"
                with open(train_file, "a") as f:
                    for item in dream_data:
                        f.write(json.dumps(item) + "\n")
        else:
            print("  [4/6] Skipping dreams (light sleep)")

        # 5. Train
        print(f"  [5/6] Training ({sleep_type} sleep)...")
        adapter_path = self.trainer.train(cycle_id, sleep_type)
        if adapter_path is None:
            print("        No training data available. Skipping.")
            return
        print(f"        Adapter saved: {adapter_path}")

        # 6. Validate and fuse
        print("  [6/6] Validating...")

        # Fuse to a TEMP location first, not current model
        temp_model_dir = Path(self.config.paths["checkpoints"]) / "temp_fused"
        self.backend.fuse_adapter(str(adapter_path), str(temp_model_dir))

        # Load the temp fused model for evaluation
        self.backend.reload(str(temp_model_dir))

        post_score = self.validator.evaluate()
        print(f"        Post-sleep score: {post_score['score']:.2f} ({post_score['correct']}/{post_score['total']})")

        validation = self.validator.validate_sleep(pre_score, post_score)

        if validation["approved"]:
            print(f"        APPROVED: {validation['reason']}")
            # Promote temp model to current
            current_dir = Path(self.config.paths["current_model"])
            if current_dir.exists():
                shutil.rmtree(current_dir)
            shutil.copytree(temp_model_dir, current_dir)
            self.backend.reload(str(current_dir))
            # Save checkpoint
            self.checkpoints.save_checkpoint(cycle_id, metadata={
                "sleep_type": sleep_type,
                "pre_score": pre_score["score"],
                "post_score": post_score["score"],
                "curated_count": len(curated),
            })
            # Mark sessions as consumed (only after successful sleep)
            if consumed_sessions:
                self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
                print(f"        Marked {len(consumed_sessions)} session(s) as consumed")
        else:
            print(f"        REJECTED: {validation['reason']}")
            print("        Rolling back to original model...")
            # Reload the clean model (base or last good checkpoint)
            latest = self.checkpoints.get_latest()
            if latest:
                self.backend.reload(latest["path"])
            else:
                self.backend.reload(self.config.model["path"])
            print("        Rollback complete.")

        # Clean up temp
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        print(f"        Sleep cycle completed in {elapsed:.1f}s")

    def _execute_sleep_streaming(self, cycle_id, sleep_type):
        """Run the sleep pipeline, yielding progress dicts for each step."""
        start_time = time.time()

        # 1. Pre-sleep evaluation
        yield {"step": 1, "total": 6, "label": "Pre-sleep evaluation", "status": "running"}
        pre_score = self.validator.evaluate()
        yield {"step": 1, "total": 6, "label": "Pre-sleep evaluation", "status": "done",
               "detail": f"Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})"}

        # 2. Curate training data
        yield {"step": 2, "total": 6, "label": "Curating training data", "status": "running"}
        messages, consumed_sessions = self._gather_new_messages()
        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)
        yield {"step": 2, "total": 6, "label": "Curating training data", "status": "done",
               "detail": f"{len(curated)} exchanges, {len(consumed_sessions)} session(s)"}

        if not curated:
            yield {"step": 2, "total": 6, "label": "Curating training data", "status": "done",
                   "detail": "No training data. Skipping sleep."}
            return

        # 3. Replay buffer
        yield {"step": 3, "total": 6, "label": "Updating replay buffer", "status": "running"}
        self.replay_buffer.add(curated)
        stats = self.replay_buffer.stats()
        yield {"step": 3, "total": 6, "label": "Updating replay buffer", "status": "done",
               "detail": f"{stats['count']} items, avg priority: {stats.get('avg_priority', 0):.2f}"}

        # 4. Dreaming
        if sleep_type == "deep":
            yield {"step": 4, "total": 6, "label": "Dreaming (REM)", "status": "running"}
            recent = [ex["messages"] for ex in curated[:10]]
            dreams = self.dreamer.dream(recent)
            if dreams:
                dream_data = self.dreamer.dream_to_training_data(dreams, self.backend)
                training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
                train_file = training_dir / "train.jsonl"
                with open(train_file, "a") as f:
                    for item in dream_data:
                        f.write(json.dumps(item) + "\n")
            yield {"step": 4, "total": 6, "label": "Dreaming (REM)", "status": "done",
                   "detail": f"{len(dreams)} dream sequences"}
        else:
            yield {"step": 4, "total": 6, "label": "Dreams", "status": "done",
                   "detail": "Skipped (light sleep)"}

        # 5. Train
        yield {"step": 5, "total": 6, "label": f"Training ({sleep_type})", "status": "running"}
        adapter_path = self.trainer.train(cycle_id, sleep_type)
        if adapter_path is None:
            yield {"step": 5, "total": 6, "label": "Training", "status": "done",
                   "detail": "No data available. Skipped."}
            return
        yield {"step": 5, "total": 6, "label": f"Training ({sleep_type})", "status": "done",
               "detail": "Adapter saved"}

        # 6. Validate and fuse
        yield {"step": 6, "total": 6, "label": "Validating", "status": "running"}
        temp_model_dir = Path(self.config.paths["checkpoints"]) / "temp_fused"
        self.backend.fuse_adapter(str(adapter_path), str(temp_model_dir))
        self.backend.reload(str(temp_model_dir))

        post_score = self.validator.evaluate()
        validation = self.validator.validate_sleep(pre_score, post_score)

        if validation["approved"]:
            current_dir = Path(self.config.paths["current_model"])
            if current_dir.exists():
                shutil.rmtree(current_dir)
            shutil.copytree(temp_model_dir, current_dir)
            self.backend.reload(str(current_dir))
            self.checkpoints.save_checkpoint(cycle_id, metadata={
                "sleep_type": sleep_type,
                "pre_score": pre_score["score"],
                "post_score": post_score["score"],
                "curated_count": len(curated),
            })
            if consumed_sessions:
                self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
            detail = f"APPROVED ({post_score['score']:.2f}). {validation['reason']}"
        else:
            latest = self.checkpoints.get_latest()
            if latest:
                self.backend.reload(latest["path"])
            else:
                self.backend.reload(self.config.model["path"])
            detail = f"REJECTED ({post_score['score']:.2f}). {validation['reason']}. Rolled back."

        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        yield {"step": 6, "total": 6, "label": "Validating", "status": "done",
               "detail": f"{detail} ({elapsed:.1f}s)"}

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
