"""Full sleep controller — multi-stage deep consolidation.

Implements a 4-stage sleep pipeline analogous to human sleep stages:
  1. Triage (NREM 1-2): Score and filter data, pull MEMIT facts
  2. Consolidation (SWS): LoRA training with interleaved replay
  3. Integration (REM): Dreaming for cross-domain associations
  4. Validation: Revert MEMIT, fuse LoRA, verify quality

This replaces the inline sleep logic in orchestrator._execute_sleep()
and _execute_sleep_streaming().
"""

import json
import shutil
import time
from pathlib import Path


class FullSleepController:
    """Executes full sleep cycles with 4 stages + MEMIT integration."""

    def __init__(self, config, backend, memit_engine, ledger, curator,
                 trainer, replay_buffer, dreamer, validator, checkpoints,
                 session_tracker, health_monitor):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger
        self.curator = curator
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.dreamer = dreamer
        self.validator = validator
        self.checkpoints = checkpoints
        self.session_tracker = session_tracker
        self.health_monitor = health_monitor

    def execute_sleep(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute the full 4-stage sleep pipeline.

        Args:
            cycle_id: Unique sleep cycle identifier
            sleep_type: "light" or "deep"
            gather_messages_fn: Callable that returns (messages, consumed_sessions)

        Returns:
            Result dict with status and details
        """
        start_time = time.time()

        # Stage 1: Triage
        print("  [1/6] Running pre-sleep evaluation...")
        pre_score = self.validator.evaluate()
        print(f"        Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})")

        print("  [2/6] Curating training data...")
        messages, consumed_sessions = gather_messages_fn()
        print(f"        {len(consumed_sessions)} new session(s) to process")

        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)
        print(f"        {len(curated)} exchanges selected for training")

        # Pull MEMIT facts from ledger and add to training queue
        memit_facts = self.ledger.get_facts_for_training()
        memit_pairs = []
        if memit_facts:
            memit_pairs = self.curator.triples_to_training_pairs(memit_facts)
            print(f"        + {len(memit_pairs)} MEMIT fact pairs added")

            # Append MEMIT pairs to the cycle's training data
            training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
            training_dir.mkdir(parents=True, exist_ok=True)
            train_file = training_dir / "train.jsonl"
            with open(train_file, "a") as f:
                for pair in memit_pairs:
                    text = self.backend.apply_chat_template(pair, for_training=True)
                    f.write(json.dumps({"text": text}) + "\n")

        if not curated and not memit_facts and self.replay_buffer.stats()["count"] == 0:
            print("        No training data and empty replay buffer. Skipping sleep.")
            return {"status": "skipped", "reason": "No data"}

        # Stage 2: Consolidation (replay buffer + training)
        print("  [3/6] Updating replay buffer...")
        if curated:
            self.replay_buffer.add(curated)
        stats = self.replay_buffer.stats()
        print(f"        Buffer: {stats['count']} items, avg priority: {stats.get('avg_priority', 0):.2f}")

        # Stage 3: Integration (dreams, deep sleep only)
        if sleep_type == "deep":
            print("  [4/6] Dreaming (REM)...")
            recent = [ex["messages"] for ex in curated[:10]]
            dreams = self.dreamer.dream(recent)
            print(f"        Generated {len(dreams)} dream sequences")
            if dreams:
                dream_data = self.dreamer.dream_to_training_data(dreams, self.backend)
                training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
                train_file = training_dir / "train.jsonl"
                with open(train_file, "a") as f:
                    for item in dream_data:
                        f.write(json.dumps(item) + "\n")
        else:
            print("  [4/6] Skipping dreams (light sleep)")

        # Training
        print(f"  [5/6] Training ({sleep_type} sleep)...")
        adapter_path = self.trainer.train(cycle_id, sleep_type)
        if adapter_path is None:
            print("        No training data available. Skipping.")
            return {"status": "skipped", "reason": "No training data"}
        print(f"        Adapter saved: {adapter_path}")

        # Stage 4: Validation
        print("  [6/6] Validating...")

        # CRITICAL ordering: revert MEMIT edits BEFORE fusing new LoRA
        edits_reverted = 0
        if memit_facts:
            edits_reverted = self.memit_engine.revert_all_active()
            print(f"        Reverted {edits_reverted} MEMIT edits before fusion")

        # Fuse to temp
        temp_model_dir = Path(self.config.paths["checkpoints"]) / "temp_fused"
        self.backend.fuse_adapter(str(adapter_path), str(temp_model_dir))
        self.backend.reload(str(temp_model_dir))

        post_score = self.validator.evaluate()
        print(f"        Post-sleep score: {post_score['score']:.2f} ({post_score['correct']}/{post_score['total']})")

        validation = self.validator.validate_sleep(pre_score, post_score)

        if validation["approved"]:
            print(f"        APPROVED: {validation['reason']}")
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
                "memit_facts": len(memit_facts),
            })

            if consumed_sessions:
                self.session_tracker.mark_consumed(consumed_sessions, cycle_id)
                print(f"        Marked {len(consumed_sessions)} session(s) as consumed")

            # Mark MEMIT edits as consolidated
            if memit_facts:
                active_edits = self.ledger.get_active_edits()
                self.ledger.mark_consolidated([e["edit_id"] for e in active_edits])

            status = "approved"
        else:
            print(f"        REJECTED: {validation['reason']}")
            print("        Rolling back to original model...")

            # Re-apply MEMIT edits since we reverted them
            latest = self.checkpoints.get_latest()
            if latest:
                self.backend.reload(latest["path"])
            else:
                self.backend.reload(self.config.model["path"])

            if memit_facts:
                print("        Re-applying MEMIT edits...")
                for fact in memit_facts:
                    self.memit_engine.inject_fact(fact)

            print("        Rollback complete.")
            status = "rejected"

        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        print(f"        Sleep cycle completed in {elapsed:.1f}s")

        return {
            "status": status,
            "pre_score": pre_score["score"],
            "post_score": post_score["score"],
            "curated_count": len(curated),
            "memit_facts": len(memit_facts),
            "edits_reverted": edits_reverted,
            "elapsed_seconds": round(elapsed, 1),
        }

    def execute_sleep_streaming(self, cycle_id, sleep_type, gather_messages_fn):
        """Execute sleep pipeline, yielding progress dicts for each step."""
        start_time = time.time()

        # 1. Pre-sleep evaluation
        yield {"step": 1, "total": 6, "label": "Pre-sleep evaluation", "status": "running"}
        pre_score = self.validator.evaluate()
        yield {"step": 1, "total": 6, "label": "Pre-sleep evaluation", "status": "done",
               "detail": f"Score: {pre_score['score']:.2f} ({pre_score['correct']}/{pre_score['total']})"}

        # 2. Curate training data
        yield {"step": 2, "total": 6, "label": "Curating training data", "status": "running"}
        messages, consumed_sessions = gather_messages_fn()

        if sleep_type == "deep":
            curated = self.curator.curate_with_model(messages, cycle_id)
        else:
            curated = self.curator.curate_session(messages, cycle_id)

        # Pull MEMIT facts
        memit_facts = self.ledger.get_facts_for_training()
        memit_pairs = []
        if memit_facts:
            memit_pairs = self.curator.triples_to_training_pairs(memit_facts)
            training_dir = Path(self.config.paths["training"]) / f"cycle_{cycle_id}"
            training_dir.mkdir(parents=True, exist_ok=True)
            train_file = training_dir / "train.jsonl"
            with open(train_file, "a") as f:
                for pair in memit_pairs:
                    text = self.backend.apply_chat_template(pair, for_training=True)
                    f.write(json.dumps({"text": text}) + "\n")

        detail = f"{len(curated)} exchanges, {len(consumed_sessions)} session(s)"
        if memit_facts:
            detail += f", +{len(memit_facts)} MEMIT facts"
        yield {"step": 2, "total": 6, "label": "Curating training data", "status": "done",
               "detail": detail}

        if not curated and not memit_facts and self.replay_buffer.stats()["count"] == 0:
            yield {"step": 2, "total": 6, "label": "Curating training data", "status": "done",
                   "detail": "No new data and empty replay buffer. Skipping."}
            return

        if not curated and not memit_facts:
            yield {"step": 2, "total": 6, "label": "Curating training data", "status": "done",
                   "detail": "No new data. Consolidating from replay buffer."}

        # 3. Replay buffer
        yield {"step": 3, "total": 6, "label": "Updating replay buffer", "status": "running"}
        if curated:
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

        # 6. Validate (with MEMIT revert)
        yield {"step": 6, "total": 6, "label": "Validating", "status": "running"}

        # CRITICAL: revert MEMIT edits BEFORE fusing new LoRA
        edits_reverted = 0
        if memit_facts:
            edits_reverted = self.memit_engine.revert_all_active()

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
                "memit_facts": len(memit_facts),
            })

            if consumed_sessions:
                self.session_tracker.mark_consumed(consumed_sessions, cycle_id)

            if memit_facts:
                active_edits = self.ledger.get_active_edits()
                self.ledger.mark_consolidated([e["edit_id"] for e in active_edits])

            detail = f"APPROVED ({post_score['score']:.2f}). {validation['reason']}"
        else:
            latest = self.checkpoints.get_latest()
            if latest:
                self.backend.reload(latest["path"])
            else:
                self.backend.reload(self.config.model["path"])

            # Re-apply MEMIT edits
            if memit_facts:
                for fact in memit_facts:
                    self.memit_engine.inject_fact(fact)

            detail = f"REJECTED ({post_score['score']:.2f}). {validation['reason']}. Rolled back."

        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        yield {"step": 6, "total": 6, "label": "Validating", "status": "done",
               "detail": f"{detail} ({elapsed:.1f}s)"}
