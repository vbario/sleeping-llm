"""Sleep trainer — LoRA training orchestrator for consolidation.

Prepares training data from MEMIT facts (chat Q&A + raw completion) and
delegates LoRA training + fusing to the backend.
"""

import json
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

    def prepare_training_data(self, facts, output_dir) -> Path:
        """Write training data from facts to JSONL files.

        Each fact becomes a chat Q&A pair:
          {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

        mlx_lm requires both train.jsonl and valid.jsonl in the data directory.
        For small-scale sleep training, we use the same data for both.

        Args:
            facts: List of FactTriple objects
            output_dir: Directory to write train.jsonl + valid.jsonl

        Returns:
            Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for fact in facts:
            chat_example = {
                "messages": [
                    {"role": "user", "content": fact.to_question()},
                    {"role": "assistant", "content": fact.to_answer()},
                ]
            }
            examples.append(json.dumps(chat_example))

        data = "\n".join(examples) + "\n"
        (output_dir / "train.jsonl").write_text(data)
        (output_dir / "valid.jsonl").write_text(data)

        print(f"        Training data: {len(facts)} facts → {len(examples)} examples")
        return output_dir

    def train_and_fuse(self, facts, cycle_id, save_dir) -> Optional[str]:
        """Train LoRA on facts and fuse into the model.

        Args:
            facts: List of FactTriple objects to train on
            cycle_id: Sleep cycle identifier (for naming)
            save_dir: Directory for fused model output

        Returns:
            Path to fused model directory, or None on failure.
        """
        save_dir = Path(save_dir)
        data_dir = save_dir / "training_data" / cycle_id
        adapter_path = save_dir / "adapters" / cycle_id
        fused_path = save_dir / "fused" / cycle_id

        # 1. Prepare training data
        self.prepare_training_data(facts, data_dir)

        # 2. Compute iterations
        iters = max(len(facts) * self.iters_per_fact, 20)

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
