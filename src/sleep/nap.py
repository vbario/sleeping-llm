"""Nap controller — quick LoRA consolidation of MEMIT facts.

A nap is a lightweight sleep cycle that:
1. Takes active MEMIT facts from the ledger
2. Converts them to Q&A training pairs
3. Runs quick LoRA training (1 epoch)
4. Fuses to temp, validates recall
5. On success: promotes fused model, reverts MEMIT edits
6. On failure: keeps MEMIT edits, flags for full sleep
"""

import json
import shutil
import time
from pathlib import Path


class NapController:
    """Executes nap cycles — quick LoRA consolidation of MEMIT-injected facts."""

    def __init__(self, config, backend, memit_engine, ledger, replay_buffer, curator):
        self.config = config
        self.backend = backend
        self.memit_engine = memit_engine
        self.ledger = ledger
        self.replay_buffer = replay_buffer
        self.curator = curator

        nap_config = config.get("nap", {}) or {}
        self.epochs = nap_config.get("epochs", 1)
        self.learning_rate = nap_config.get("learning_rate", 1.0e-4)
        self.revert_on_success = nap_config.get("revert_on_success", True)

    def execute_nap(self, cycle_id) -> dict:
        """Execute a nap cycle.

        Args:
            cycle_id: Unique identifier for this nap cycle

        Returns:
            Result dict with status, facts consolidated/remaining, etc.
        """
        start_time = time.time()

        # 1. Get active MEMIT facts
        facts = self.ledger.get_facts_for_training()
        if not facts:
            return {
                "status": "skipped",
                "reason": "No active MEMIT facts to consolidate",
                "elapsed_seconds": 0,
            }

        # 2. Convert to Q&A training pairs
        training_data = self._generate_training_data(facts, cycle_id)
        if not training_data:
            return {
                "status": "skipped",
                "reason": "No training data generated",
                "elapsed_seconds": 0,
            }

        # 3. Quick LoRA training
        adapter_path = Path(self.config.paths["adapters"]) / f"nap_{cycle_id}"
        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"

        self.backend.train_lora(
            data_path=str(data_dir),
            adapter_path=str(adapter_path),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        # 4. Fuse to temp and validate
        temp_model_dir = Path(self.config.paths["checkpoints"]) / "temp_nap_fused"
        self.backend.fuse_adapter(str(adapter_path), str(temp_model_dir))

        # CRITICAL: Revert MEMIT edits BEFORE loading fused model
        # Deltas were computed against old model state
        edits_reverted = 0
        if self.revert_on_success:
            edits_reverted = self.memit_engine.revert_all_active()

        # Load fused model
        self.backend.reload(str(temp_model_dir))

        # 5. Test recall of each fact
        recall_results = self._validate_recall(facts)
        passed = sum(1 for _, ok in recall_results if ok)
        total = len(recall_results)

        if passed >= total * 0.5:  # At least half recalled
            # Success: promote fused model
            current_dir = Path(self.config.paths["current_model"])
            if current_dir.exists():
                shutil.rmtree(current_dir)
            shutil.copytree(temp_model_dir, current_dir)
            self.backend.reload(str(current_dir))

            # Mark edits as consolidated
            active_edits = self.ledger.get_active_edits()
            self.ledger.mark_consolidated([e["edit_id"] for e in active_edits])

            status = "success"
        else:
            # Failure: facts didn't transfer to LoRA
            # Re-apply MEMIT edits if we reverted them
            if edits_reverted > 0:
                # Reload original model and re-inject
                latest_checkpoint = self._get_latest_model_path()
                self.backend.reload(latest_checkpoint)
                for fact in facts:
                    self.memit_engine.inject_fact(fact)

            status = "partial"

        # Clean up temp
        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        return {
            "status": status,
            "facts_consolidated": passed,
            "facts_remaining": total - passed,
            "facts_total": total,
            "edits_reverted": edits_reverted,
            "elapsed_seconds": round(elapsed, 1),
        }

    def execute_nap_streaming(self, cycle_id):
        """Execute nap with streaming progress. Yields progress dicts."""
        start_time = time.time()

        # Step 1: Gather MEMIT facts
        yield {"step": 1, "total": 4, "label": "Gathering MEMIT facts", "status": "running"}
        facts = self.ledger.get_facts_for_training()
        if not facts:
            yield {"step": 1, "total": 4, "label": "Gathering MEMIT facts", "status": "done",
                   "detail": "No active facts. Skipping nap."}
            return
        yield {"step": 1, "total": 4, "label": "Gathering MEMIT facts", "status": "done",
               "detail": f"{len(facts)} facts to consolidate"}

        # Step 2: Training
        yield {"step": 2, "total": 4, "label": "Quick LoRA training", "status": "running"}
        training_data = self._generate_training_data(facts, cycle_id)
        adapter_path = Path(self.config.paths["adapters"]) / f"nap_{cycle_id}"
        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"
        self.backend.train_lora(
            data_path=str(data_dir),
            adapter_path=str(adapter_path),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        yield {"step": 2, "total": 4, "label": "Quick LoRA training", "status": "done",
               "detail": f"Trained on {len(training_data)} examples"}

        # Step 3: Fuse and revert
        yield {"step": 3, "total": 4, "label": "Fusing adapter", "status": "running"}
        temp_model_dir = Path(self.config.paths["checkpoints"]) / "temp_nap_fused"
        self.backend.fuse_adapter(str(adapter_path), str(temp_model_dir))

        edits_reverted = 0
        if self.revert_on_success:
            edits_reverted = self.memit_engine.revert_all_active()

        self.backend.reload(str(temp_model_dir))
        yield {"step": 3, "total": 4, "label": "Fusing adapter", "status": "done",
               "detail": f"Reverted {edits_reverted} MEMIT edits"}

        # Step 4: Validate
        yield {"step": 4, "total": 4, "label": "Validating recall", "status": "running"}
        recall_results = self._validate_recall(facts)
        passed = sum(1 for _, ok in recall_results if ok)
        total = len(recall_results)

        if passed >= total * 0.5:
            current_dir = Path(self.config.paths["current_model"])
            if current_dir.exists():
                shutil.rmtree(current_dir)
            shutil.copytree(temp_model_dir, current_dir)
            self.backend.reload(str(current_dir))

            active_edits = self.ledger.get_active_edits()
            self.ledger.mark_consolidated([e["edit_id"] for e in active_edits])

            detail = f"SUCCESS: {passed}/{total} facts recalled"
        else:
            if edits_reverted > 0:
                latest = self._get_latest_model_path()
                self.backend.reload(latest)
                for fact in facts:
                    self.memit_engine.inject_fact(fact)
            detail = f"PARTIAL: {passed}/{total} facts recalled. MEMIT edits restored."

        if temp_model_dir.exists():
            shutil.rmtree(temp_model_dir)

        elapsed = time.time() - start_time
        yield {"step": 4, "total": 4, "label": "Validating recall", "status": "done",
               "detail": f"{detail} ({elapsed:.1f}s)"}

    def _generate_training_data(self, facts, cycle_id):
        """Convert facts to training JSONL.

        Args:
            facts: List of FactTriple objects
            cycle_id: Nap cycle identifier

        Returns:
            List of training examples
        """
        pairs = self.curator.triples_to_training_pairs(facts)
        if not pairs:
            return []

        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"
        data_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for pair in pairs:
            text = self.backend.apply_chat_template(pair, for_training=True)
            examples.append({"text": text})

        # Write train.jsonl
        with open(data_dir / "train.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        # Write minimal valid.jsonl
        with open(data_dir / "valid.jsonl", "w") as f:
            if examples:
                f.write(json.dumps(examples[0]) + "\n")

        return examples

    def _validate_recall(self, facts):
        """Test recall of each fact.

        Returns:
            List of (fact, passed) tuples
        """
        results = []
        for fact in facts:
            passed, _ = self.memit_engine.test_recall(fact)
            results.append((fact, passed))
        return results

    def _get_latest_model_path(self):
        """Get the path to the latest good model."""
        current_dir = Path(self.config.paths["current_model"])
        if current_dir.exists() and any(current_dir.iterdir()):
            return str(current_dir)
        return self.config.model["path"]
