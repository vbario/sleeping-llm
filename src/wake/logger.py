"""Conversation logger — persists all exchanges to JSONL files on disk."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class ConversationLogger:
    """Logs every message to a per-session JSONL file."""

    def __init__(self, config):
        self.log_dir = Path(config.paths["conversations"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{self.session_id}.jsonl"
        self.turn_count = 0

    def log(self, role, content, metadata=None):
        """Append a single message to the session log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn": self.turn_count,
            "role": role,
            "content": content,
        }
        if metadata:
            entry["metadata"] = metadata
        with open(self.session_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if role == "assistant":
            self.turn_count += 1

    def log_exchange(self, user_message, assistant_message):
        """Log a complete user/assistant exchange."""
        self.log("user", user_message)
        self.log("assistant", assistant_message)

    def get_session_log(self):
        """Read back the full session log as a list of message dicts."""
        messages = []
        if not self.session_file.exists():
            return messages
        with open(self.session_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        return messages

    def get_session_messages(self):
        """Return session log as simple role/content pairs (for training)."""
        return [
            {"role": entry["role"], "content": entry["content"]}
            for entry in self.get_session_log()
        ]

    @staticmethod
    def list_sessions(config):
        """List all session files in the conversations directory."""
        log_dir = Path(config.paths["conversations"])
        if not log_dir.exists():
            return []
        return sorted(log_dir.glob("session_*.jsonl"))

    @staticmethod
    def load_session(session_path):
        """Load messages from a specific session file."""
        messages = []
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        return messages
