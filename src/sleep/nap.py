"""Nap controller — NREM-like LoRA reinforcement of MEMIT facts.

A nap replays memories via LoRA training, reinforcing pathways, but does NOT
restructure the MEMIT→LoRA relationship. MEMIT edits are left untouched.

Flow:
1. Gather MEMIT facts → training pairs
2. LoRA training (reinforcement)
3. Update replay buffer
4. Done — no MEMIT revert, no validation, no consolidation
"""

import json
import shutil
import time
from pathlib import Path


class NapController:
    """Executes nap cycles — NREM-like LoRA reinforcement of MEMIT facts.

    Naps train LoRA on MEMIT facts but do NOT revert MEMIT edits or validate
    recall. The ablation data shows that reverting MEMIT after nap LoRA is
    actively harmful — LoRA can't reliably carry the facts alone at 8B+.
    """

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

    def execute_nap(self, cycle_id) -> dict:
        """Execute a nap cycle — LoRA reinforcement only, no MEMIT changes.

        Args:
            cycle_id: Unique identifier for this nap cycle

        Returns:
            Result dict with status and training details
        """
        start_time = time.time()

        # 1. Get active MEMIT facts
        facts = self.ledger.get_facts_for_training()
        if not facts:
            return {
                "status": "skipped",
                "reason": "No active MEMIT facts to reinforce",
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

        # 3. LoRA reinforcement training
        adapter_path = Path(self.config.paths["adapters"]) / f"nap_{cycle_id}"
        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"

        self.backend.train_lora(
            data_path=str(data_dir),
            adapter_path=str(adapter_path),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        # Done. MEMIT edits are untouched — nap just reinforced via LoRA.
        elapsed = time.time() - start_time
        return {
            "status": "success",
            "facts_reinforced": len(facts),
            "training_examples": len(training_data),
            "elapsed_seconds": round(elapsed, 1),
        }

    def execute_nap_streaming(self, cycle_id):
        """Execute nap with streaming progress. Yields progress dicts."""
        start_time = time.time()

        # Step 1: Gather MEMIT facts
        yield {"step": 1, "total": 2, "label": "Gathering MEMIT facts", "status": "running"}
        facts = self.ledger.get_facts_for_training()
        if not facts:
            yield {"step": 1, "total": 2, "label": "Gathering MEMIT facts", "status": "done",
                   "detail": "No active facts. Skipping nap."}
            return
        yield {"step": 1, "total": 2, "label": "Gathering MEMIT facts", "status": "done",
               "detail": f"{len(facts)} facts to reinforce"}

        # Step 2: LoRA training
        yield {"step": 2, "total": 2, "label": "LoRA reinforcement", "status": "running"}
        training_data = self._generate_training_data(facts, cycle_id)
        if not training_data:
            yield {"step": 2, "total": 2, "label": "LoRA reinforcement", "status": "done",
                   "detail": "No training data generated. Skipped."}
            return

        adapter_path = Path(self.config.paths["adapters"]) / f"nap_{cycle_id}"
        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"
        self.backend.train_lora(
            data_path=str(data_dir),
            adapter_path=str(adapter_path),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        elapsed = time.time() - start_time
        yield {"step": 2, "total": 2, "label": "LoRA reinforcement", "status": "done",
               "detail": f"Trained on {len(training_data)} examples. MEMIT intact. ({elapsed:.1f}s)"}

    def _generate_training_data(self, facts, cycle_id):
        """Convert facts to training JSONL (chat pairs + raw completions).

        Args:
            facts: List of FactTriple objects
            cycle_id: Nap cycle identifier

        Returns:
            List of training examples
        """
        result = self.curator.triples_to_training_pairs(facts)
        chat_pairs = result["chat_pairs"]
        raw_texts = result["raw_texts"]
        if not chat_pairs:
            return []

        data_dir = Path(self.config.paths["training"]) / f"nap_{cycle_id}"
        data_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for pair in chat_pairs:
            text = self.backend.apply_chat_template(pair, for_training=True)
            examples.append({"text": text})
        for text in raw_texts:
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

    def _get_latest_model_path(self):
        """Get the path to the latest good model."""
        current_dir = Path(self.config.paths["current_model"])
        if current_dir.exists() and any(current_dir.iterdir()):
            return str(current_dir)
        return self.config.model["path"]
