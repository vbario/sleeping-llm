import os
import yaml
from pathlib import Path


class Config:
    """Loads and provides access to sleeping-llm configuration."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        self._root = Path(__file__).parent.parent
        with open(config_path) as f:
            self._data = yaml.safe_load(f)
        self._resolve_paths()

    def _resolve_paths(self):
        """Convert relative paths in config to absolute paths."""
        paths = self._data.get("paths", {})
        for key, value in paths.items():
            if not os.path.isabs(value):
                paths[key] = str(self._root / value)

    @property
    def model(self):
        return self._data["model"]

    @property
    def context(self):
        return self._data["context"]

    @property
    def sleep(self):
        return self._data["sleep"]

    @property
    def lora(self):
        return self._data["lora"]

    @property
    def replay(self):
        return self._data["replay"]

    @property
    def validation(self):
        return self._data["validation"]

    @property
    def dreamer(self):
        return self._data["dreamer"]

    @property
    def memit(self):
        return self._data.get("memit", {})

    @property
    def health(self):
        return self._data.get("health", {})

    @property
    def nap(self):
        return self._data.get("nap", {})

    @property
    def rem(self):
        return self._data.get("rem", {})

    @property
    def paths(self):
        return self._data["paths"]

    def get(self, dotted_key, default=None):
        """Access nested config via dotted key, e.g. 'model.path'."""
        keys = dotted_key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
