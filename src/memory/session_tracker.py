"""Session tracker — tracks which conversation sessions have been consumed for training.

Prevents the same session from being re-curated across multiple sleep cycles.
The replay buffer handles revisiting important old material separately.
"""

import json
import time
from pathlib import Path


class SessionTracker:
    """Tracks which conversation session files have been consumed during sleep cycles."""

    def __init__(self, config):
        self.config = config
        self.conversations_dir = Path(config.paths["conversations"])
        self.manifest_file = self.conversations_dir / "consumed_sessions.json"

    def get_unconsumed_sessions(self):
        """Return paths to session files that haven't been consumed yet."""
        from src.wake.logger import ConversationLogger

        all_sessions = ConversationLogger.list_sessions(self.config)
        consumed = self._load_manifest()
        consumed_names = {entry["filename"] for entry in consumed.get("sessions", [])}

        return [s for s in all_sessions if s.name not in consumed_names]

    def mark_consumed(self, session_paths, sleep_cycle_id):
        """Mark session files as consumed after a successful sleep cycle."""
        manifest = self._load_manifest()

        for path in session_paths:
            manifest["sessions"].append({
                "filename": Path(path).name,
                "consumed_by_cycle": sleep_cycle_id,
                "consumed_at": time.time(),
            })

        self._save_manifest(manifest)

    def get_consumed_count(self):
        """Return the number of consumed sessions."""
        manifest = self._load_manifest()
        return len(manifest.get("sessions", []))

    def get_total_session_count(self):
        """Return total number of session files."""
        from src.wake.logger import ConversationLogger
        return len(ConversationLogger.list_sessions(self.config))

    def reset(self):
        """Clear the consumed sessions manifest (reprocess everything)."""
        self._save_manifest({"sessions": []})

    def _load_manifest(self):
        if not self.manifest_file.exists():
            return {"sessions": []}
        with open(self.manifest_file) as f:
            return json.loads(f.read())

    def _save_manifest(self, manifest):
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_file, "w") as f:
            f.write(json.dumps(manifest, indent=2))
