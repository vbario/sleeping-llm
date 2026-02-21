"""Replay buffer — implements spaced repetition for training data."""

import json
import os
import time
from pathlib import Path


class ReplayBuffer:
    """Maintains a prioritized buffer of training examples for spaced repetition.

    Items are never discarded — they decay in priority with each replay.
    Once priority drops below min_priority, they stop being trained on
    but remain in the buffer. High-value items naturally persist longer.

    With decay_factor=0.85 and starting priority ~0.35:
      After  1 replay: 0.30
      After  5 replays: 0.15
      After 10 replays: 0.07
      After 15 replays: 0.03  (below default floor of 0.05)
    """

    def __init__(self, config):
        self.config = config
        self.buffer_dir = Path(config.paths["replay_buffer"])
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_file = self.buffer_dir / "buffer.jsonl"
        self.max_items = config.replay["max_items"]
        self.light_mix_ratio = config.replay.get("light_mix_ratio", config.replay.get("mix_ratio", 0.2))
        self.deep_mix_ratio = config.replay.get("deep_mix_ratio", config.replay.get("mix_ratio", 0.6))
        self.decay_factor = config.replay["decay_factor"]
        self.min_priority = config.replay.get("min_priority", 0.05)

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

    def sample(self, count=None, sleep_type="light"):
        """Sample active examples for replay during training.

        Only items above min_priority are eligible. Items below the floor
        have been replayed enough times and are considered consolidated.

        Args:
            count: Number of examples to sample. If None, uses mix ratio.
            sleep_type: "light" or "deep" — determines how much replay data to mix in.

        Returns:
            List of message pairs for training
        """
        buffer = self._load_buffer()
        if not buffer:
            return []

        # Filter to active items only
        active = [e for e in buffer if e["priority"] >= self.min_priority]
        if not active:
            return []

        if count is None:
            ratio = self.deep_mix_ratio if sleep_type == "deep" else self.light_mix_ratio
            count = max(1, int(len(active) * ratio))

        # Sort by priority, take top N from active items
        active.sort(key=lambda x: x["priority"], reverse=True)
        sampled = active[:count]

        # Decay priority for sampled items (update in the full buffer)
        sampled_ids = {id(e) for e in sampled}
        now = time.time()

        # We need to match by content since id() won't work across lists
        sampled_keys = set()
        for entry in sampled:
            entry["replay_count"] += 1
            entry["priority"] *= self.decay_factor
            entry["last_replayed"] = now
            sampled_keys.add((entry["added_at"], entry["combined_score"]))

        # Apply updates back to the full buffer
        for entry in buffer:
            key = (entry["added_at"], entry["combined_score"])
            if key in sampled_keys:
                matching = [e for e in sampled if (e["added_at"], e["combined_score"]) == key]
                if matching:
                    entry["replay_count"] = matching[0]["replay_count"]
                    entry["priority"] = matching[0]["priority"]
                    entry["last_replayed"] = matching[0]["last_replayed"]

        self._save_buffer(buffer)

        return [entry["messages"] for entry in sampled]

    def get_replay_data_for_training(self, backend, sleep_type="light"):
        """Get replay examples formatted as training JSONL strings.

        Args:
            backend: MLX backend for applying chat template
            sleep_type: "light" or "deep" — determines replay mix ratio

        Returns:
            List of {"text": ...} dicts ready for training
        """
        messages_list = self.sample(sleep_type=sleep_type)
        results = []
        for messages in messages_list:
            text = backend.apply_chat_template(messages, for_training=True)
            results.append({"text": text})
        return results

    def stats(self):
        """Return buffer statistics."""
        buffer = self._load_buffer()
        if not buffer:
            return {"count": 0, "active": 0, "retired": 0}
        priorities = [e["priority"] for e in buffer]
        active = [e for e in buffer if e["priority"] >= self.min_priority]
        retired = len(buffer) - len(active)
        return {
            "count": len(buffer),
            "active": len(active),
            "retired": retired,
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
