"""Replay buffer — implements spaced repetition for training data."""

import json
import os
import time
from pathlib import Path


class ReplayBuffer:
    """Maintains a prioritized buffer of training examples for spaced repetition.

    High-value examples are revisited across multiple sleep cycles
    with decaying frequency, mimicking how the brain replays important
    memories during sleep.
    """

    def __init__(self, config):
        self.config = config
        self.buffer_dir = Path(config.paths["replay_buffer"])
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_file = self.buffer_dir / "buffer.jsonl"
        self.max_items = config.replay["max_items"]
        self.mix_ratio = config.replay["mix_ratio"]
        self.decay_factor = config.replay["decay_factor"]

    def add(self, examples):
        """Add curated examples to the replay buffer.

        Args:
            examples: List of dicts with 'messages', 'scores', 'combined' keys
        """
        existing = self._load_buffer()

        for ex in examples:
            entry = {
                "messages": ex["messages"],
                "combined_score": ex["combined"],
                "replay_count": 0,
                "priority": ex["combined"],  # decays with each replay
                "added_at": time.time(),
                "last_replayed": None,
            }
            existing.append(entry)

        # Sort by priority, keep top max_items
        existing.sort(key=lambda x: x["priority"], reverse=True)
        existing = existing[:self.max_items]

        self._save_buffer(existing)

    def sample(self, count=None):
        """Sample examples for replay during training.

        Args:
            count: Number of examples to sample. If None, uses mix_ratio.

        Returns:
            List of message pairs for training
        """
        buffer = self._load_buffer()
        if not buffer:
            return []

        if count is None:
            count = max(1, int(len(buffer) * self.mix_ratio))

        # Sort by priority, take top N
        buffer.sort(key=lambda x: x["priority"], reverse=True)
        sampled = buffer[:count]

        # Decay priority for sampled items
        now = time.time()
        for entry in sampled:
            entry["replay_count"] += 1
            entry["priority"] *= self.decay_factor
            entry["last_replayed"] = now

        self._save_buffer(buffer)

        return [entry["messages"] for entry in sampled]

    def get_replay_data_for_training(self, backend):
        """Get replay examples formatted as training JSONL strings.

        Args:
            backend: MLX backend for applying chat template

        Returns:
            List of {"text": ...} dicts ready for training
        """
        messages_list = self.sample()
        results = []
        for messages in messages_list:
            text = backend.apply_chat_template(messages, for_training=True)
            results.append({"text": text})
        return results

    def prune(self, min_priority=0.05):
        """Remove items that have decayed below a minimum priority."""
        buffer = self._load_buffer()
        buffer = [entry for entry in buffer if entry["priority"] >= min_priority]
        self._save_buffer(buffer)
        return len(buffer)

    def stats(self):
        """Return buffer statistics."""
        buffer = self._load_buffer()
        if not buffer:
            return {"count": 0}
        priorities = [e["priority"] for e in buffer]
        return {
            "count": len(buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
            "min_priority": min(priorities),
            "avg_replays": sum(e["replay_count"] for e in buffer) / len(buffer),
        }

    def _load_buffer(self):
        if not self.buffer_file.exists():
            return []
        entries = []
        with open(self.buffer_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _save_buffer(self, entries):
        with open(self.buffer_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
