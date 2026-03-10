"""Sleep trainer — LoRA training orchestrator for consolidation.

Prepares training data from QAPair facts (native chat format) and
delegates LoRA training + fusing to the backend.
"""

import json
import random
from pathlib import Path
from typing import Optional


class SleepTrainer:
    """Orchestrates LoRA training and fusing during sleep consolidation."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

        lora_cfg = config.get("lora", {}) or {}
        self.num_layers = lora_cfg.get("num_layers", 8)
        self.learning_rate = lora_cfg.get("learning_rate", 1e-4)
        self.iters_per_fact = lora_cfg.get("iters_per_fact", 10)
        self.batch_size = lora_cfg.get("batch_size", 1)

    def prepare_training_data(self, qa_pairs, output_dir) -> Path:
        """Write training data from QAPairs to JSONL files.

        Each QAPair becomes a chat training example:
          {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

        mlx_lm requires both train.jsonl and valid.jsonl in the data directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for qa in qa_pairs:
            chat_example = {
                "messages": [
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": qa.answer},
                ]
            }
            examples.append(json.dumps(chat_example))

        data = "\n".join(examples) + "\n"
        (output_dir / "train.jsonl").write_text(data)
        (output_dir / "valid.jsonl").write_text(data)

        print(f"        Training data: {len(qa_pairs)} facts → {len(examples)} examples")
        return output_dir

    def prepare_weighted_training_data(self, qa_pairs, output_dir) -> Path:
        """Write training data with priority-weighted repetition.

        High-priority facts get more training examples (repetitions).
        Priority 1.0 → 3x, 0.5 → 2x, 0.0 → 1x minimum.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for qa in qa_pairs:
            chat_example = {
                "messages": [
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": qa.answer},
                ]
            }
            example_json = json.dumps(chat_example)

            priority = getattr(qa, 'priority', 0.5)
            reps = max(1, round(1 + priority * 2))
            for _ in range(reps):
                examples.append(example_json)

        random.shuffle(examples)

        data = "\n".join(examples) + "\n"
        (output_dir / "train.jsonl").write_text(data)
        (output_dir / "valid.jsonl").write_text(data)

        print(f"        Training data: {len(qa_pairs)} facts → {len(examples)} examples "
              f"(priority-weighted)")
        return output_dir

    def train_and_fuse(self, qa_pairs, cycle_id, save_dir, weighted=False) -> Optional[str]:
        """Train LoRA on QAPairs and fuse into the model.

        Args:
            qa_pairs: List of QAPair objects to train on
            cycle_id: Sleep cycle identifier (for naming)
            save_dir: Directory for fused model output
            weighted: Use priority-weighted repetition

        Returns:
            Path to fused model directory, or None on failure.
        """
        save_dir = Path(save_dir)
        data_dir = save_dir / "training_data" / cycle_id
        adapter_path = save_dir / "adapters" / cycle_id
        fused_path = save_dir / "fused" / cycle_id

        # 1. Prepare training data
        if weighted:
            self.prepare_weighted_training_data(qa_pairs, data_dir)
        else:
            self.prepare_training_data(qa_pairs, data_dir)

        # 2. Compute iterations
        iters = max(len(qa_pairs) * self.iters_per_fact, 20)

        # 3. Train LoRA
        try:
            print(f"        Training LoRA ({iters} iters, {self.num_layers} layers)...")
            self.backend.train_lora(
                data_path=str(data_dir),
                adapter_path=str(adapter_path),
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                iters=iters,
                learning_rate=self.learning_rate,
            )
        except Exception as e:
            print(f"        LoRA training failed: {e}")
            return None

        # 4. Fuse adapter into model
        try:
            print(f"        Fusing adapter...")
            self.backend.fuse_adapter(
                adapter_path=str(adapter_path),
                save_path=str(fused_path),
            )
        except Exception as e:
            print(f"        Fuse failed: {e}")
            return None

        print(f"        Fused model saved to {fused_path}")
        return str(fused_path)
