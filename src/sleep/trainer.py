"""Sleep trainer — LoRA training orchestrator for consolidation.

Prepares training data from QAPair facts (native chat format) and
delegates LoRA training + fusing to the backend.
"""

import json
import random
import re
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
        self.augment_paraphrases = lora_cfg.get("augment_paraphrases", 0)

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

    def _generate_paraphrases(self, qa, n=4):
        """Generate question paraphrases for a fact via templates.

        Takes a statement-style fact (e.g. "Jazzy Mike is a saxophone player")
        and produces varied question forms so LoRA learns diverse phrasings.
        Uses templates (no model calls) for speed and reliability.
        """
        statement = qa.answer.strip().rstrip(".")
        # Extract subject (first 1-3 words before a verb/preposition)
        words = statement.split()
        subject = words[0] if words else "this"
        # Try to find a natural subject boundary (2-3 word proper names)
        for i in range(1, min(4, len(words))):
            if words[i].lower() in ("is", "has", "was", "likes", "lives",
                                     "works", "makes", "plays", "from",
                                     "moved", "speaks", "studied", "uses"):
                subject = " ".join(words[:i])
                break
        else:
            subject = " ".join(words[:2]) if len(words) > 1 else words[0]

        templates = [
            f"What do you know about {subject}?",
            f"Tell me about {subject}.",
            f"What can you tell me about {subject}?",
            f"Do you remember anything about {subject}?",
            f"What do you recall about {subject}?",
            f"Who is {subject}?",
            f"What is {subject} known for?",
            f"Can you remind me about {subject}?",
        ]
        random.shuffle(templates)
        return templates[:n]

    def prepare_weighted_training_data(self, qa_pairs, output_dir) -> Path:
        """Write training data with paraphrase augmentation and priority-weighted repetition.

        For each fact:
          1. Keep the original statement-echo example
          2. Generate N question paraphrases via the model (if augment_paraphrases > 0)
          3. Repeat all examples by priority (1.0 → 3x, 0.5 → 2x, 0.0 → 1x)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        total_augmented = 0
        for qa in qa_pairs:
            # Original example (statement echo)
            chat_example = {
                "messages": [
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": qa.answer},
                ]
            }
            fact_examples = [json.dumps(chat_example)]

            # Paraphrase augmentation: generate question variants
            if self.augment_paraphrases > 0:
                try:
                    paraphrases = self._generate_paraphrases(qa, self.augment_paraphrases)
                    for q in paraphrases:
                        aug_example = {
                            "messages": [
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": qa.answer},
                            ]
                        }
                        fact_examples.append(json.dumps(aug_example))
                    total_augmented += len(paraphrases)
                except Exception as e:
                    print(f"        Augmentation failed for '{qa.value[:30]}': {e}")

            # Priority-weighted repetition (applies to ALL examples for this fact)
            priority = getattr(qa, 'priority', 0.5)
            reps = max(1, round(1 + priority * 2))
            for _ in range(reps):
                examples.extend(fact_examples)

        random.shuffle(examples)

        data = "\n".join(examples) + "\n"
        (output_dir / "train.jsonl").write_text(data)
        (output_dir / "valid.jsonl").write_text(data)

        print(f"        Training data: {len(qa_pairs)} facts → {len(examples)} examples "
              f"({total_augmented} paraphrases, priority-weighted)")
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
