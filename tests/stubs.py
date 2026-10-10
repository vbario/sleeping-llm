"""Shared stubs for lifecycle tests — no model, no GPU.

StubBackend answers probe questions through an `oracle(question, system)`
callable so tests can simulate "the fact is only known when it is in the
prompt" (no LoRA) versus "the weights know it" (LoRA absorbed it).
"""

import copy
import re
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config  # noqa: E402


def make_config(tmp_path, overrides=None) -> Config:
    """Load the repo config.yaml, redirect every path into tmp_path, apply overrides."""
    with open(project_root / "config.yaml") as f:
        data = yaml.safe_load(f)
    data["paths"] = {k: str(tmp_path / Path(v).name) for k, v in data["paths"].items()}
    data["paths"]["memit_ledger"] = str(tmp_path / "memit" / "ledger.json")
    data["paths"]["memit_data"] = str(tmp_path / "memit")
    data["paths"]["fused_models"] = str(tmp_path / "fused")
    if overrides:
        _deep_update(data, overrides)
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f)
    return Config(cfg_path)


def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


_USER_RE = re.compile(r"<\|user\|>\n(.*?)(?=\n<\||\Z)", re.S)
_SYSTEM_RE = re.compile(r"<\|system\|>\n(.*?)(?=\n<\||\Z)", re.S)


def make_oracle(facts, learned=()):
    """Build an oracle for StubBackend.

    facts:   {question: (answer_sentence, value)} — what the model could say
    learned: questions the weights "know" (answered even when withheld from prompt)

    Any other question is answered correctly only if its answer sentence is
    present in the system content (prompt bridge). Unknown → "I'm not sure."
    """
    learned = {q.lower().strip() for q in learned}

    def oracle(question, system):
        key = question.lower().strip()
        for q, (answer, value) in facts.items():
            if q.lower().strip() == key:
                if key in learned or answer in system:
                    return f"{answer}"
                return "I'm not sure."
        return "I'm not sure."

    return oracle


class StubBackend:
    """Minimal backend double. Records every call."""

    def __init__(self, oracle=None, ppl=10.0, model_path="base-model"):
        self.oracle = oracle or (lambda q, s: "I'm not sure.")
        self.ppl = ppl
        self._model_path = model_path
        self.model_lock = None
        self.reload_calls = []
        self.train_calls = []
        self.fuse_calls = []
        self.generate_calls = []
        self.fail_train = False
        self.fail_fuse = False

    @property
    def model_path(self):
        return self._model_path

    # --- inference ---
    def apply_chat_template(self, messages, for_training=False):
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if not for_training:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        user = _USER_RE.findall(prompt)
        system = _SYSTEM_RE.findall(prompt)
        question = user[-1].strip() if user else ""
        system_text = system[-1] if system else ""
        self.generate_calls.append((question, system_text))
        return self.oracle(question, system_text)

    def generate_stream(self, prompt, **kw):
        yield self.generate(prompt, **kw)

    def count_tokens(self, text):
        if isinstance(text, list):
            text = self.apply_chat_template(text)
        return len(text.split())

    def compute_perplexity(self, text):
        return self.ppl() if callable(self.ppl) else self.ppl

    # --- training ---
    def train_lora(self, data_path, adapter_path, num_layers=8, batch_size=1,
                   iters=None, learning_rate=1e-4):
        if self.fail_train:
            raise RuntimeError("stub: training failed")
        self.train_calls.append({"data_path": str(data_path), "adapter_path": str(adapter_path),
                                 "iters": iters, "base": self._model_path})
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        (Path(adapter_path) / "adapters.safetensors").write_text("stub")

    def fuse_adapter(self, adapter_path, save_path):
        if self.fail_fuse:
            raise RuntimeError("stub: fuse failed")
        self.fuse_calls.append({"adapter_path": str(adapter_path), "save_path": str(save_path),
                                "base": self._model_path})
        Path(save_path).mkdir(parents=True, exist_ok=True)
        (Path(save_path) / "config.json").write_text("{}")
        (Path(save_path) / "model.safetensors").write_text("stub-weights")

    def reload(self, model_path=None):
        self.reload_calls.append(model_path)
        self._model_path = model_path or self._model_path

    load = reload


def snapshot_stages(ledger):
    """{fact_id: (stage, graduated)} across active + graduated lists."""
    out = {}
    for e in ledger.get_active_facts() + ledger.get_graduated_facts():
        out[e["fact_id"]] = (e.get("stage", 0), e.get("graduated", False))
    return copy.deepcopy(out)
