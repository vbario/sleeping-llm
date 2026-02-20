"""Orchestrator — the wake/sleep state machine.

Coordinates the full lifecycle:
  wake (chat) → detect sleep trigger → curate → train → validate → fuse → wake
"""

import shutil
import time
from pathlib import Path

from src.backend.mlx_backend import MLXBackend
from src.memory.checkpoints import CheckpointManager
from src.memory.identity import IdentityManager
from src.memory.replay import ReplayBuffer
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

        # 2. Curate training data
        print("  [2/6] Curating training data...")
        messages = self._gather_all_messages()
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
                import json
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

    def _gather_all_messages(self):
        """Gather messages from all conversation sessions for training."""
        all_messages = []
        # Load all sessions
        sessions = ConversationLogger.list_sessions(self.config)
        for session_path in sessions:
            entries = ConversationLogger.load_session(session_path)
            for entry in entries:
                all_messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })
        return all_messages
