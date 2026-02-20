"""Checkpoint manager — versioning and rollback for model states."""

import json
import os
import shutil
import time
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoints for versioning and rollback."""

    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = Path(config.paths["checkpoints"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_dir = Path(config.paths["current_model"])
        self.manifest_file = self.checkpoint_dir / "manifest.json"

    def save_checkpoint(self, sleep_cycle_id, metadata=None):
        """Save the current model state as a checkpoint.

        Args:
            sleep_cycle_id: Identifier for this sleep cycle
            metadata: Optional dict of extra info (eval scores, etc.)

        Returns:
            Path to the saved checkpoint
        """
        checkpoint_name = f"checkpoint_{sleep_cycle_id}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not self.current_model_dir.exists():
            raise FileNotFoundError(
                f"No current model to checkpoint at {self.current_model_dir}"
            )

        # Copy current model to checkpoint
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        shutil.copytree(self.current_model_dir, checkpoint_path)

        # Update manifest
        manifest = self._load_manifest()
        manifest["checkpoints"].append({
            "id": checkpoint_name,
            "sleep_cycle": sleep_cycle_id,
            "path": str(checkpoint_path),
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        manifest["latest"] = checkpoint_name
        self._save_manifest(manifest)

        return checkpoint_path

    def rollback(self, checkpoint_id=None):
        """Restore a previous checkpoint as the current model.

        Args:
            checkpoint_id: Which checkpoint to restore. If None, uses the previous one.

        Returns:
            Path to the restored checkpoint
        """
        manifest = self._load_manifest()
        checkpoints = manifest.get("checkpoints", [])

        if not checkpoints:
            raise ValueError("No checkpoints available for rollback")

        if checkpoint_id is None:
            # Roll back to the one before latest
            if len(checkpoints) < 2:
                target = checkpoints[0]
            else:
                target = checkpoints[-2]
        else:
            target = None
            for cp in checkpoints:
                if cp["id"] == checkpoint_id:
                    target = cp
                    break
            if target is None:
                raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        source = Path(target["path"])
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint files missing at {source}")

        # Replace current model with checkpoint
        if self.current_model_dir.exists():
            shutil.rmtree(self.current_model_dir)
        shutil.copytree(source, self.current_model_dir)

        manifest["latest"] = target["id"]
        self._save_manifest(manifest)

        return source

    def list_checkpoints(self):
        """List all available checkpoints."""
        manifest = self._load_manifest()
        return manifest.get("checkpoints", [])

    def get_latest(self):
        """Get the latest checkpoint info."""
        manifest = self._load_manifest()
        latest_id = manifest.get("latest")
        if not latest_id:
            return None
        for cp in manifest.get("checkpoints", []):
            if cp["id"] == latest_id:
                return cp
        return None

    def cleanup(self, keep_last=5):
        """Remove old checkpoints, keeping only the most recent N."""
        manifest = self._load_manifest()
        checkpoints = manifest.get("checkpoints", [])

        if len(checkpoints) <= keep_last:
            return 0

        to_remove = checkpoints[:-keep_last]
        to_keep = checkpoints[-keep_last:]

        removed = 0
        for cp in to_remove:
            cp_path = Path(cp["path"])
            if cp_path.exists():
                shutil.rmtree(cp_path)
                removed += 1

        manifest["checkpoints"] = to_keep
        if to_keep:
            manifest["latest"] = to_keep[-1]["id"]
        self._save_manifest(manifest)

        return removed

    def _load_manifest(self):
        if not self.manifest_file.exists():
            return {"checkpoints": [], "latest": None}
        with open(self.manifest_file) as f:
            return json.loads(f.read())

    def _save_manifest(self, manifest):
        with open(self.manifest_file, "w") as f:
            f.write(json.dumps(manifest, indent=2))
