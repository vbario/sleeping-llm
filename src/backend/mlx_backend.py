"""MLX backend — wraps mlx-lm for inference, tokenization, LoRA training, and adapter fusion."""

import json
import os
import subprocess
import sys
from pathlib import Path


class MLXBackend:
    """Unified interface to MLX model operations."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_path = None

    def load(self, model_path=None):
        """Load model and tokenizer from disk or hub."""
        from mlx_lm import load

        path = model_path or self._resolve_model_path()
        self.model, self.tokenizer = load(path)
        self._model_path = path
        return self

    def _resolve_model_path(self):
        """Determine which model to load: current (post-sleep) or base."""
        current = self.config.paths["current_model"]
        if os.path.exists(current) and os.listdir(current):
            return current
        return self.config.model["path"]

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text from a prompt string."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
        )

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )
        return response

    def apply_chat_template(self, messages, for_training=False):
        """Convert a list of message dicts to a formatted prompt string.

        Args:
            messages: List of {role, content} dicts
            for_training: If True, don't add generation prompt at the end.
                          Training data should end after the last message,
                          not with an empty assistant prompt.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=not for_training,
            )
        # Fallback: simple formatting
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        if not for_training:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def count_tokens(self, text):
        """Count the number of tokens in a text string."""
        if self.tokenizer is None:
            return 0
        if isinstance(text, list):
            text = self.apply_chat_template(text)
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def tokenize(self, text):
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text)

    def train_lora(self, data_path, adapter_path, epochs=None, learning_rate=None):
        """Run LoRA fine-tuning using mlx_lm.lora CLI.

        Args:
            data_path: Directory containing train.jsonl (and optionally valid.jsonl)
            adapter_path: Where to save the LoRA adapter
            epochs: Number of passes over the data
            learning_rate: Override config learning rate
        """
        model_path = self._model_path or self._resolve_model_path()
        epochs = epochs or self.config.lora["light_epochs"]
        learning_rate = learning_rate or self.config.lora["light_learning_rate"]
        lora_layers = self.config.lora["layers"]
        batch_size = self.config.lora["batch_size"]

        # Count training examples to scale iterations properly
        train_file = Path(data_path) / "train.jsonl"
        num_examples = 0
        if train_file.exists():
            with open(train_file) as f:
                num_examples = sum(1 for line in f if line.strip())

        if num_examples == 0:
            return adapter_path

        # iterations = examples × epochs (each example seen `epochs` times)
        iters = max(1, num_examples * epochs)

        os.makedirs(adapter_path, exist_ok=True)

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", str(model_path),
            "--train",
            "--data", str(data_path),
            "--adapter-path", str(adapter_path),
            "--batch-size", str(batch_size),
            "--num-layers", str(lora_layers),
            "--iters", str(iters),
            "--learning-rate", str(learning_rate),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"LoRA training failed:\n{result.stderr}")

        return adapter_path

    def fuse_adapter(self, adapter_path, save_path):
        """Merge a LoRA adapter into the model and save."""
        model_path = self._model_path or self._resolve_model_path()
        os.makedirs(save_path, exist_ok=True)

        cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", str(model_path),
            "--adapter-path", str(adapter_path),
            "--save-path", str(save_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Adapter fusion failed:\n{result.stderr}")

        return save_path

    def generate_stream(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text with streaming. Yields token strings incrementally."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
        )

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            yield response.text

    def reload(self, model_path=None):
        """Reload model (e.g. after fusing a new adapter)."""
        self.model = None
        self.tokenizer = None
        self.load(model_path)
