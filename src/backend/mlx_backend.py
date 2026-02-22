"""MLX backend — wraps mlx-lm for inference, tokenization, LoRA training, and adapter fusion."""

import json
import os
import subprocess
import sys
from pathlib import Path


def nn_create_additive_causal_mask(seq_len, dtype):
    """Create an additive causal mask for self-attention.

    Returns a [seq_len, seq_len] matrix where future positions are -inf.
    """
    import mlx.core as mx
    indices = mx.arange(seq_len)
    mask = indices[:, None] < indices[None, :]
    return mask * mx.array(float('-inf'), dtype=dtype)


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

    # --- Layer-level access for MEMIT ---

    def forward_to_layer(self, input_ids, target_layer):
        """Manual forward through embed_tokens + layers[0:target_layer+1].

        Replicates the Llama forward pass layer-by-layer to extract
        intermediate hidden states needed by MEMIT.

        Args:
            input_ids: mx.array of shape [batch, seq_len]
            target_layer: Which layer to stop at (inclusive)

        Returns:
            Tuple of (hidden_state, mlp_input, mlp_output) at the target layer.
            All shapes: [batch, seq_len, hidden_size]
        """
        import mlx.core as mx

        model = self.model.model  # Access the inner model (e.g., LlamaModel)

        # Embed tokens
        h = model.embed_tokens(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        mask = None
        if seq_len > 1:
            mask = nn_create_additive_causal_mask(seq_len, h.dtype)

        mlp_input_out = None
        mlp_output_out = None

        for i, layer in enumerate(model.layers):
            if i > target_layer:
                break

            if i == target_layer:
                # At target layer, capture MLP input and output separately
                # Run attention first (with residual)
                residual = h
                attn_input = layer.input_layernorm(h)
                attn_output = layer.self_attn(attn_input, mask=mask)
                h = residual + attn_output

                # MLP input = post-attention hidden state after post_attention_layernorm
                mlp_input_out = layer.post_attention_layernorm(h)

                # MLP output
                mlp_raw = layer.mlp(mlp_input_out)
                mlp_output_out = mlp_raw

                # Residual connection
                h = h + mlp_raw
            else:
                # Normal forward through layer
                h = layer(h, mask=mask)

        mx.eval(h)
        if mlp_input_out is not None:
            mx.eval(mlp_input_out)
        if mlp_output_out is not None:
            mx.eval(mlp_output_out)

        return h, mlp_input_out, mlp_output_out

    def forward_layers_range(self, input_ids, start_layer, end_layer):
        """Forward through a range of layers, returning activations at each.

        Args:
            input_ids: mx.array of shape [batch, seq_len]
            start_layer: First layer (inclusive)
            end_layer: Last layer (inclusive)

        Returns:
            Dict mapping layer_idx -> (hidden_state, mlp_input, mlp_output)
        """
        import mlx.core as mx

        model = self.model.model
        h = model.embed_tokens(input_ids)

        seq_len = input_ids.shape[1]
        mask = None
        if seq_len > 1:
            mask = nn_create_additive_causal_mask(seq_len, h.dtype)

        activations = {}

        for i, layer in enumerate(model.layers):
            if i > end_layer:
                break

            if start_layer <= i <= end_layer:
                residual = h
                attn_input = layer.input_layernorm(h)
                attn_output = layer.self_attn(attn_input, mask=mask)
                h = residual + attn_output

                mlp_input = layer.post_attention_layernorm(h)
                mlp_raw = layer.mlp(mlp_input)
                mlp_output = mlp_raw
                h = h + mlp_raw

                mx.eval(h, mlp_input, mlp_output)
                activations[i] = (h, mlp_input, mlp_output)
            else:
                h = layer(h, mask=mask)

        return activations

    def get_layer_mlp_weight(self, layer_idx, proj="down_proj"):
        """Return weight matrix reference for a layer's MLP projection.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"

        Returns:
            mx.array weight matrix, or None if not found
        """
        try:
            layer = self.model.model.layers[layer_idx]
            mlp = layer.mlp
            proj_module = getattr(mlp, proj, None)
            if proj_module is not None:
                return proj_module.weight
        except (IndexError, AttributeError):
            pass
        return None

    def set_layer_mlp_weight(self, layer_idx, proj, new_weight):
        """Set weight matrix for a layer's MLP projection.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"
            new_weight: mx.array new weight matrix
        """
        import mlx.core as mx

        try:
            layer = self.model.model.layers[layer_idx]
            mlp = layer.mlp
            proj_module = getattr(mlp, proj, None)
            if proj_module is not None:
                proj_module.weight = new_weight
                mx.eval(proj_module.weight)
        except (IndexError, AttributeError):
            pass

    def get_num_layers(self):
        """Return total number of transformer layers."""
        try:
            return len(self.model.model.layers)
        except AttributeError:
            return 0

    def compute_perplexity(self, text):
        """Compute perplexity on a text string.

        Args:
            text: Input text to evaluate

        Returns:
            Perplexity value (float). Lower = model is more confident.
        """
        import mlx.core as mx
        import math

        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return float('inf')

        input_ids = mx.array([tokens])
        logits = self.model(input_ids)  # [1, seq_len, vocab_size]

        # Shift: predict token[i+1] from logits[i]
        shift_logits = logits[:, :-1, :]  # [1, seq_len-1, vocab]
        shift_labels = mx.array([tokens[1:]])  # [1, seq_len-1]

        # Log softmax
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs for actual tokens
        seq_len = shift_labels.shape[1]
        token_log_probs = []
        for i in range(seq_len):
            token_id = shift_labels[0, i].item()
            lp = log_probs[0, i, token_id].item()
            token_log_probs.append(lp)

        avg_neg_log_prob = -sum(token_log_probs) / len(token_log_probs)
        perplexity = math.exp(avg_neg_log_prob)

        return perplexity

    def reload(self, model_path=None):
        """Reload model (e.g. after fusing a new adapter)."""
        self.model = None
        self.tokenizer = None
        self.load(model_path)
