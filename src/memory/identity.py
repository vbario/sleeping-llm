"""Identity manager — maintains the core identity dataset.

This is the model's 'narrative identity' — a set of Q&A pairs that define
who it is and how it should behave. These get mixed into every training
run to prevent drift from core behavior.
"""

import json
from pathlib import Path


class IdentityManager:
    """Manages the core identity reinforcement dataset."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.identity_dir = Path(config.paths["core_identity"])
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        self.identity_file = self.identity_dir / "identity.jsonl"

    def get_identity_data(self):
        """Load identity Q&A pairs as training examples."""
        if not self.identity_file.exists():
            return []
        examples = []
        with open(self.identity_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def add_identity(self, question, answer):
        """Add a new identity-defining Q&A pair."""
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        text = self.backend.apply_chat_template(messages, for_training=True)
        entry = {"text": text, "messages": messages}
        with open(self.identity_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def remove_identity(self, index):
        """Remove an identity entry by index."""
        entries = self.get_identity_data()
        if 0 <= index < len(entries):
            entries.pop(index)
            with open(self.identity_file, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            return True
        return False

    def list_identities(self):
        """List all identity entries in human-readable form."""
        entries = self.get_identity_data()
        result = []
        for i, entry in enumerate(entries):
            messages = entry.get("messages", [])
            if len(messages) >= 2:
                result.append({
                    "index": i,
                    "question": messages[0]["content"],
                    "answer": messages[1]["content"],
                })
        return result

    def seed_defaults(self):
        """Create default identity entries if none exist."""
        if self.identity_file.exists():
            with open(self.identity_file) as f:
                if f.read().strip():
                    return  # already has data

        defaults = [
            {
                "question": "Who are you?",
                "answer": "I am a personal AI assistant with persistent memory. I learn from our conversations and remember what matters to you.",
            },
            {
                "question": "What makes you different from other AI assistants?",
                "answer": "I have a sleep-wake cycle. When I sleep, I consolidate what I've learned from our conversations into my long-term memory. This means I genuinely learn and grow from our interactions.",
            },
            {
                "question": "Do you remember our previous conversations?",
                "answer": "Yes. Important information from our conversations gets integrated into my memory through a process similar to human sleep. I may not remember every word, but I retain the key facts, your preferences, and what I've learned.",
            },
        ]

        for d in defaults:
            self.add_identity(d["question"], d["answer"])
