"""Trainer — orchestrates LoRA fine-tuning during sleep cycles."""

import json
import os
from pathlib import Path


class SleepTrainer:
    """Runs LoRA fine-tuning on curated conversation data.

    Handles mixing replay buffer data with new training data
    and manages adapter lifecycle.
    """

    def __init__(self, config, backend, replay_buffer):
        self.config = config
        self.backend = backend
        self.replay_buffer = replay_buffer
        self.adapter_dir = Path(config.paths["adapters"])
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir = Path(config.paths["training"])

    def train(self, sleep_cycle_id, sleep_type="light"):
        """Execute a training cycle.

        Args:
            sleep_cycle_id: Unique ID for this sleep cycle
            sleep_type: "light" or "deep" — affects epochs and learning rate

        Returns:
            Path to the saved adapter
        """
        # Prepare combined training data
        data_dir = self._prepare_training_data(sleep_cycle_id, sleep_type=sleep_type)

        if not self._has_training_data(data_dir):
            return None

        # Select hyperparameters based on sleep type
        if sleep_type == "deep":
            epochs = self.config.lora["deep_epochs"]
            lr = self.config.lora["deep_learning_rate"]
        else:
            epochs = self.config.lora["light_epochs"]
            lr = self.config.lora["light_learning_rate"]

        # Run LoRA training
        adapter_path = self.adapter_dir / f"sleep_cycle_{sleep_cycle_id}"
        self.backend.train_lora(
            data_path=str(data_dir),
            adapter_path=str(adapter_path),
            epochs=epochs,
            learning_rate=lr,
        )

        return adapter_path

    def train_rem(self, cycle_id, rem_data_dir):
        """Execute REM integration training with separate hyperparameters.

        Combines REM integration data with identity data (no replay buffer).

        Args:
            cycle_id: Sleep cycle identifier
            rem_data_dir: Directory containing REM training data (train.jsonl)

        Returns:
            Path to the saved adapter, or None if no data
        """
        combined_dir = self.training_dir / f"combined_rem_{cycle_id}"
        combined_dir.mkdir(parents=True, exist_ok=True)

        all_train = []

        # Load REM integration data
        rem_train = Path(rem_data_dir) / "train.jsonl"
        if rem_train.exists():
            with open(rem_train) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_train.append(line)

        # Mix in identity data (no replay buffer for REM)
        identity_data = self._load_identity_data()
        all_train.extend(identity_data)

        print(f"        REM training data: {len(all_train) - len(identity_data)} integration + {len(identity_data)} identity")

        if not all_train:
            return None

        # Write combined training data
        with open(combined_dir / "train.jsonl", "w") as f:
            for line in all_train:
                f.write(line.strip() + "\n")

        # Use first item as validation
        with open(combined_dir / "valid.jsonl", "w") as f:
            f.write(all_train[0].strip() + "\n")

        # REM hyperparameters
        rem_config = self.config.rem
        epochs = rem_config.get("epochs", 1)
        lr = rem_config.get("learning_rate", 5e-5)

        adapter_path = self.adapter_dir / f"rem_cycle_{cycle_id}"
        self.backend.train_lora(
            data_path=str(combined_dir),
            adapter_path=str(adapter_path),
            epochs=epochs,
            learning_rate=lr,
        )

        return adapter_path

    def fuse_and_save(self, adapter_path):
        """Merge adapter into model and save as the new current model."""
        current_model_dir = self.config.paths["current_model"]
        self.backend.fuse_adapter(
            adapter_path=str(adapter_path),
            save_path=current_model_dir,
        )
        return current_model_dir

    def _prepare_training_data(self, sleep_cycle_id, sleep_type="light"):
        """Combine curated data with replay buffer data."""
        cycle_dir = self.training_dir / f"cycle_{sleep_cycle_id}"
        combined_dir = self.training_dir / f"combined_{sleep_cycle_id}"
        combined_dir.mkdir(parents=True, exist_ok=True)

        all_train = []
        all_valid = []

        # Load curated training data from this cycle
        curated_train = cycle_dir / "train.jsonl"
        curated_valid = cycle_dir / "valid.jsonl"

        new_count = 0
        if curated_train.exists():
            with open(curated_train) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_train.append(line)
                        new_count += 1

        if curated_valid.exists():
            with open(curated_valid) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_valid.append(line)

        # Mix in replay buffer data
        replay_data = self.replay_buffer.get_replay_data_for_training(self.backend, sleep_type=sleep_type)
        for item in replay_data:
            all_train.append(json.dumps(item))

        print(f"        Training data: {new_count} new + {len(replay_data)} replay = {len(all_train)} total")

        # Mix in core identity data
        identity_data = self._load_identity_data()
        all_train.extend(identity_data)

        # Write combined training data
        with open(combined_dir / "train.jsonl", "w") as f:
            for line in all_train:
                f.write(line.strip() + "\n")

        with open(combined_dir / "valid.jsonl", "w") as f:
            for line in all_valid:
                f.write(line.strip() + "\n")
            # If no validation data, use a sample from training
            if not all_valid and all_train:
                f.write(all_train[0].strip() + "\n")

        return combined_dir

    def _load_identity_data(self):
        """Load core identity reinforcement data."""
        identity_dir = Path(self.config.paths["core_identity"])
        identity_file = identity_dir / "identity.jsonl"
        if not identity_file.exists():
            return []
        lines = []
        with open(identity_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    def _has_training_data(self, data_dir):
        """Check if there's actual training data to work with."""
        train_file = Path(data_dir) / "train.jsonl"
        if not train_file.exists():
            return False
        with open(train_file) as f:
            content = f.read().strip()
        return len(content) > 0
