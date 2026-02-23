"""MLX backend — wraps mlx-lm for inference, tokenization, LoRA training, and adapter fusion."""

import json
import os
import subprocess
import sys
from pathlib import Path


def nn_create_additive_causal_mask(seq_len, dtype):
    """Create an additive causal mask for self-attention.

    Returns a [seq_len, seq_len] matrix where future positions are -inf
    and past/current positions are 0. Uses mx.where to avoid NaN from
    0 * -inf (IEEE 754: 0 * inf = NaN).
    """
    import mlx.core as mx
    indices = mx.arange(seq_len)
    is_future = indices[:, None] < indices[None, :]
    return mx.where(is_future, mx.array(float('-inf'), dtype=dtype), mx.array(0.0, dtype=dtype))


class MLXBackend:
    """Unified interface to MLX model operations."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_path = None
        self.model_lock = None  # Set by orchestrator for non-blocking sleep

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
            print(f"  Loading from models/current/ (post-sleep model)")
            return current
        base = self.config.model["path"]
        print(f"  Loading from base: {base}")
        return base

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text from a prompt string.

        Acquires a read lock if model_lock is set (non-blocking sleep mode).
        """
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

        def _do_generate():
            return generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            )

        if self.model_lock:
            with self.model_lock.read_lock():
                return _do_generate()
        return _do_generate()

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
        """Generate text with streaming. Yields token strings incrementally.

        Acquires a read lock if model_lock is set (non-blocking sleep mode).
        The lock is held for the entire streaming duration to prevent model
        swaps mid-generation.
        """
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

        def _do_stream():
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                yield response.text

        if self.model_lock:
            with self.model_lock.read_lock():
                yield from _do_stream()
        else:
            yield from _do_stream()

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

    def compute_mlp_intermediate(self, mlp_input, layer_idx):
        """Compute the intermediate MLP activation (input to down_proj).

        For Llama MLP: intermediate = silu(gate_proj(x)) * up_proj(x)
        This is the actual input to the down_proj weight matrix.

        Args:
            mlp_input: mx.array [batch, seq_len, hidden_size] — output of post_attention_layernorm
            layer_idx: Transformer layer index

        Returns:
            mx.array [batch, seq_len, intermediate_size]
        """
        import mlx.core as mx
        import mlx.nn as nn

        layer = self.model.model.layers[layer_idx]
        mlp = layer.mlp
        intermediate = nn.silu(mlp.gate_proj(mlp_input)) * mlp.up_proj(mlp_input)
        mx.eval(intermediate)
        return intermediate

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

    def forward_from_layer(self, hidden_states, start_layer, mask=None):
        """Forward from a given layer to the end, returning logits.

        Runs layers[start_layer:] → final_norm → lm_head (or embed_tokens.as_linear
        for weight-tied models).

        Args:
            hidden_states: mx.array [batch, seq_len, hidden_size] — residual stream
            start_layer: First layer to process (inclusive)
            mask: Attention mask (optional)

        Returns:
            logits: mx.array [batch, seq_len, vocab_size]
        """
        import mlx.core as mx

        model = self.model.model
        h = hidden_states

        for i in range(start_layer, len(model.layers)):
            h = model.layers[i](h, mask=mask)

        h = model.norm(h)

        # Handle weight-tied vs separate lm_head
        if hasattr(self.model, 'args') and getattr(self.model.args, 'tie_word_embeddings', False):
            logits = model.embed_tokens.as_linear(h)
        elif hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(h)
        else:
            # Fallback: try weight-tied embedding
            logits = model.embed_tokens.as_linear(h)

        return logits

    def dequantize_layer(self, layer_idx, proj="down_proj"):
        """Replace a QuantizedLinear with a regular Linear using dequantized weights.

        Required for MEMIT — can't add float deltas to packed 4-bit weights.
        Increases memory by ~48MB per layer (3B model) but enables direct weight editing.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"

        Returns:
            True if dequantized, False if already float or not found
        """
        import mlx.core as mx
        import mlx.nn as nn

        try:
            layer = self.model.model.layers[layer_idx]
            proj_module = getattr(layer.mlp, proj, None)
            if proj_module is None:
                return False

            # Already a regular Linear — nothing to do
            if not isinstance(proj_module, nn.QuantizedLinear):
                return False

            # Dequantize weights
            w_float = mx.dequantize(
                proj_module.weight, proj_module.scales, proj_module.biases,
                proj_module.group_size, proj_module.bits,
            )

            # Create regular Linear replacement
            out_dims, in_dims = w_float.shape
            has_bias = hasattr(proj_module, "bias") and proj_module.get("bias") is not None
            linear = nn.Linear(in_dims, out_dims, bias=has_bias)
            linear.weight = w_float

            if has_bias:
                linear.bias = proj_module.bias

            mx.eval(linear.weight)

            # Replace in the model
            setattr(layer.mlp, proj, linear)
            return True

        except (IndexError, AttributeError) as e:
            print(f"  Warning: dequantize_layer({layer_idx}, {proj}) failed: {e}")
            return False

    def get_layer_mlp_weight(self, layer_idx, proj="down_proj"):
        """Return weight matrix reference for a layer's MLP projection.

        Works with both QuantizedLinear and regular Linear layers.

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
        """Reload model (e.g. after fusing a new adapter).

        Acquires a write lock if model_lock is set (non-blocking sleep mode).
        """
        def _do_reload():
            self.model = None
            self.tokenizer = None
            self.load(model_path)

        if self.model_lock:
            with self.model_lock.write_lock():
                _do_reload()
        else:
            _do_reload()
