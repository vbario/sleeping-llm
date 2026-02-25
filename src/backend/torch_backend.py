"""PyTorch backend — wraps HuggingFace transformers for inference and MEMIT weight editing.

Drop-in replacement for MLXBackend. Uses bfloat16 on CUDA GPUs.
"""

import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig


class TorchBackend:
    """Unified interface to PyTorch model operations on CUDA."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_path = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_lock = None  # Set by orchestrator for non-blocking sleep

    def load(self, model_path=None):
        """Load model and tokenizer from disk or HuggingFace hub."""
        path = model_path or self._resolve_model_path()
        quantize = self.config.model.get("quantize", None)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # Spread model across all available GPUs; reserve headroom for MEMIT
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                load_kwargs["max_memory"] = {i: "70GiB" for i in range(num_gpus)}
                load_kwargs["max_memory"]["cpu"] = "100GiB"
            else:
                load_kwargs["max_memory"] = {0: "70GiB", "cpu": "100GiB"}
            load_kwargs["offload_folder"] = "/tmp/offload"

        self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        self._quantized = quantize == "4bit"
        self.model.eval()
        self._model_path = path
        return self

    def _resolve_model_path(self):
        """Always load from the base model (MEMIT edits are re-applied in memory)."""
        base = self.config.model["path"]
        print(f"  Loading from base: {base}")
        return base

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text from a prompt string.

        Acquires a read lock if model_lock is set (non-blocking sleep mode).
        """
        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

        def _do_generate():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][input_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if self.model_lock:
            with self.model_lock.read_lock():
                return _do_generate()
        return _do_generate()

    def generate_stream(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text with streaming. Yields token strings incrementally.

        Acquires a read lock if model_lock is set (non-blocking sleep mode).
        """
        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

        def _do_stream():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=streamer,
            )

            # Run generation in a background thread so we can yield tokens
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text in streamer:
                if text:
                    yield text

            thread.join()

        if self.model_lock:
            with self.model_lock.read_lock():
                yield from _do_stream()
        else:
            yield from _do_stream()

    def apply_chat_template(self, messages, for_training=False):
        """Convert a list of message dicts to a formatted prompt string."""
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

    # --- Layer-level access for MEMIT ---

    def forward_to_layer(self, input_ids, target_layer):
        """Forward pass capturing hidden states at a specific layer.

        Uses PyTorch forward hooks to intercept activations at the target
        transformer layer, capturing MLP input and output.

        Args:
            input_ids: torch.Tensor of shape [batch, seq_len]
            target_layer: Which layer to capture (0-indexed)

        Returns:
            Tuple of (hidden_state, mlp_input, mlp_output) at the target layer.
            All shapes: [batch, seq_len, hidden_size]
        """
        captured = {}

        def capture_post_attn_norm(module, input, output):
            """Hook on post_attention_layernorm — captures MLP input."""
            captured["mlp_input"] = output.detach()

        def capture_mlp(module, input, output):
            """Hook on the MLP module — captures MLP output."""
            captured["mlp_output"] = output.detach()

        def capture_layer_output(module, input, output):
            """Hook on the full layer — captures residual stream after layer."""
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                captured["hidden"] = output[0].detach()
            else:
                captured["hidden"] = output.detach()

        layer = self.model.model.layers[target_layer]
        hooks = [
            layer.post_attention_layernorm.register_forward_hook(capture_post_attn_norm),
            layer.mlp.register_forward_hook(capture_mlp),
            layer.register_forward_hook(capture_layer_output),
        ]

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return (
            captured.get("hidden"),
            captured.get("mlp_input"),
            captured.get("mlp_output"),
        )

    def forward_layers_range(self, input_ids, start_layer, end_layer):
        """Forward pass capturing activations at a range of layers.

        Args:
            input_ids: torch.Tensor of shape [batch, seq_len]
            start_layer: First layer (inclusive)
            end_layer: Last layer (inclusive)

        Returns:
            Dict mapping layer_idx -> (hidden_state, mlp_input, mlp_output)
        """
        captured = {}
        hooks = []

        for layer_idx in range(start_layer, end_layer + 1):
            layer = self.model.model.layers[layer_idx]
            lid = layer_idx  # capture for closure

            def make_hooks(lid):
                cap = {}
                captured[lid] = cap

                def hook_norm(mod, inp, out, c=cap):
                    c["mlp_input"] = out.detach()
                def hook_mlp(mod, inp, out, c=cap):
                    c["mlp_output"] = out.detach()
                def hook_layer(mod, inp, out, c=cap):
                    c["hidden"] = (out[0] if isinstance(out, tuple) else out).detach()

                return hook_norm, hook_mlp, hook_layer

            hn, hm, hl = make_hooks(lid)
            hooks.append(layer.post_attention_layernorm.register_forward_hook(hn))
            hooks.append(layer.mlp.register_forward_hook(hm))
            hooks.append(layer.register_forward_hook(hl))

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        result = {}
        for lid, cap in captured.items():
            result[lid] = (cap.get("hidden"), cap.get("mlp_input"), cap.get("mlp_output"))
        return result

    def forward_from_layer(self, hidden_states, start_layer, mask=None):
        """Forward from a given layer to the end, returning logits.

        Args:
            hidden_states: torch.Tensor [batch, seq_len, hidden_size]
            start_layer: First layer to process (inclusive)
            mask: Attention mask (optional)

        Returns:
            logits: torch.Tensor [batch, seq_len, vocab_size]
        """
        h = hidden_states
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        # Compute rotary position embeddings (cos, sin) for attention layers
        position_embeddings = self.model.model.rotary_emb(h, position_ids)
        with torch.no_grad():
            for i in range(start_layer, len(self.model.model.layers)):
                layer = self.model.model.layers[i]
                layer_device = next(layer.parameters()).device
                h = h.to(layer_device)
                pos_emb = tuple(p.to(layer_device) for p in position_embeddings)
                layer_out = layer(h, position_embeddings=pos_emb)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            norm_device = next(self.model.model.norm.parameters()).device
            h = self.model.model.norm(h.to(norm_device))
            head_device = next(self.model.lm_head.parameters()).device
            logits = self.model.lm_head(h.to(head_device))
        return logits

    def compute_mlp_intermediate(self, mlp_input, layer_idx):
        """Compute the intermediate MLP activation (input to down_proj).

        For Llama MLP: intermediate = silu(gate_proj(x)) * up_proj(x)

        Args:
            mlp_input: torch.Tensor [batch, seq_len, hidden_size]
            layer_idx: Transformer layer index

        Returns:
            torch.Tensor [batch, seq_len, intermediate_size]
        """
        import torch.nn.functional as F
        layer = self.model.model.layers[layer_idx]
        mlp = layer.mlp
        with torch.no_grad():
            intermediate = F.silu(mlp.gate_proj(mlp_input)) * mlp.up_proj(mlp_input)
        return intermediate

    def dequantize_layer(self, layer_idx, proj="down_proj"):
        """Replace a BitsAndBytes 4-bit Linear with a regular Linear using dequantized weights.

        Required for MEMIT — can't add float deltas to packed 4-bit weights.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"

        Returns:
            True if dequantized, False if already float or not found
        """
        try:
            layer = self.model.model.layers[layer_idx]
            proj_module = getattr(layer.mlp, proj, None)
            if proj_module is None:
                return False

            # Check if it's a BitsAndBytes quantized layer
            weight = proj_module.weight
            if not hasattr(weight, "quant_state"):
                # Already a regular parameter — nothing to do
                return False

            # Dequantize using BitsAndBytes
            from bitsandbytes.functional import dequantize_4bit

            w_float = dequantize_4bit(
                weight.data, weight.quant_state,
            ).to(torch.bfloat16)

            # Create regular Linear replacement
            out_features, in_features = w_float.shape
            has_bias = proj_module.bias is not None
            new_linear = torch.nn.Linear(in_features, out_features, bias=has_bias)

            with torch.no_grad():
                new_linear.weight.copy_(w_float)
                if has_bias:
                    new_linear.bias.copy_(proj_module.bias)

            # Move to same device and dtype
            new_linear = new_linear.to(device=w_float.device, dtype=torch.bfloat16)

            # Replace in the model
            setattr(layer.mlp, proj, new_linear)
            return True

        except (IndexError, AttributeError) as e:
            print(f"  Warning: dequantize_layer({layer_idx}, {proj}) failed: {e}")
            return False

    def get_layer_mlp_weight(self, layer_idx, proj="down_proj"):
        """Return weight tensor for a layer's MLP projection.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"

        Returns:
            torch.Tensor weight matrix, or None if not found
        """
        try:
            layer = self.model.model.layers[layer_idx]
            proj_module = getattr(layer.mlp, proj, None)
            if proj_module is not None:
                return proj_module.weight.data
        except (IndexError, AttributeError):
            pass
        return None

    def set_layer_mlp_weight(self, layer_idx, proj, new_weight):
        """Set weight tensor for a layer's MLP projection.

        Args:
            layer_idx: Transformer layer index
            proj: "down_proj", "up_proj", or "gate_proj"
            new_weight: torch.Tensor new weight matrix
        """
        try:
            layer = self.model.model.layers[layer_idx]
            proj_module = getattr(layer.mlp, proj, None)
            if proj_module is not None:
                with torch.no_grad():
                    proj_module.weight.data.copy_(new_weight)
        except (IndexError, AttributeError):
            pass

    def get_num_layers(self):
        """Return total number of transformer layers."""
        try:
            return len(self.model.model.layers)
        except AttributeError:
            if hasattr(self.model, "config"):
                return getattr(self.model.config, "num_hidden_layers", 0)
            return 0

    def compute_perplexity(self, text):
        """Compute perplexity on a text string.

        Args:
            text: Input text to evaluate

        Returns:
            Perplexity value (float). Lower = model is more confident.
        """
        import math

        tokens = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < 2:
            return float('inf')

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # outputs.loss is the average cross-entropy loss
            neg_log_likelihood = outputs.loss.item()

        return math.exp(neg_log_likelihood)

    def train_lora(self, data_path, adapter_path, num_layers=8,
                   batch_size=1, iters=None, learning_rate=1e-4):
        """Train LoRA adapter using PEFT on the loaded model.

        Args:
            data_path: Directory containing train.jsonl
            adapter_path: Output path for adapter weights
            num_layers: Number of layers to apply LoRA to
            batch_size: Training batch size (unused — trains one example at a time)
            iters: Number of training iterations
            learning_rate: Learning rate
        """
        import json
        import random
        from pathlib import Path
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # 1. Load training data
        train_file = Path(data_path) / "train.jsonl"
        examples = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if not examples:
            raise ValueError(f"No training examples in {train_file}")

        # Tokenize examples with labels masking non-assistant tokens
        tokenized = []
        for ex in examples:
            messages = ex.get("messages", [])
            if not messages:
                continue

            # Full text with assistant response
            full_text = self.apply_chat_template(messages, for_training=True)
            full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"][0]

            # Prompt-only text (everything up to assistant response)
            prompt_messages = messages[:-1]
            prompt_text = self.apply_chat_template(prompt_messages, for_training=False)
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]

            # Labels: mask prompt tokens with -100
            labels = full_ids.clone()
            labels[:len(prompt_ids)] = -100

            tokenized.append({"input_ids": full_ids, "labels": labels})

        if not tokenized:
            raise ValueError("No valid training examples after tokenization")

        if iters is None:
            iters = max(len(tokenized) * 10, 20)

        # 2. Wrap model with PEFT LoRA
        total_layers = len(self.model.model.layers)
        layers_to_transform = list(range(total_layers - num_layers, total_layers))

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=layers_to_transform,
        )

        # Save reference to unwrapped model
        base_model = self.model

        if self._quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        peft_model = get_peft_model(self.model, config)
        peft_model.print_trainable_parameters()

        # 3. Training loop
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        peft_model.train()
        for step in range(iters):
            ex = random.choice(tokenized)
            input_ids = ex["input_ids"].unsqueeze(0).to(peft_model.device)
            labels = ex["labels"].unsqueeze(0).to(peft_model.device)

            outputs = peft_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0 or step == iters - 1:
                print(f"        LoRA step {step}/{iters}  loss={loss.item():.4f}")

        # 4. Save adapter
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(adapter_path)
        print(f"        Adapter saved to {adapter_path}")

        # 5. Unwrap — restore original model without adapter
        self.model = peft_model.unload()
        self.model.eval()
        del peft_model, optimizer, trainable_params
        torch.cuda.empty_cache()

    def fuse_adapter(self, adapter_path, save_path):
        """Fuse LoRA adapter into model weights and save.

        Loads a fresh bfloat16 copy of the base model, merges the adapter,
        and saves the result. The caller should then call reload(save_path)
        to load the fused model back with quantization.

        Args:
            adapter_path: Path to trained adapter weights
            save_path: Output path for fused model
        """
        from pathlib import Path
        from peft import PeftModel

        Path(save_path).mkdir(parents=True, exist_ok=True)

        # 1. Load a fresh bf16 copy (can't merge into 4-bit packed weights)
        print(f"        Loading fresh bf16 base model for fusing...")
        base = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # 2. Load adapter onto the fresh copy
        peft_model = PeftModel.from_pretrained(base, adapter_path)

        # 3. Merge and unload
        merged = peft_model.merge_and_unload()

        # 4. Save fused model + tokenizer
        merged.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"        Fused model saved to {save_path}")

        # 5. Free temporary model
        del merged, peft_model, base
        torch.cuda.empty_cache()

    def reload(self, model_path=None):
        """Reload model (e.g. after fusing a new adapter).

        Acquires a write lock if model_lock is set (non-blocking sleep mode).
        """
        def _do_reload():
            # Free GPU memory
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.load(model_path)

        if self.model_lock:
            with self.model_lock.write_lock():
                _do_reload()
        else:
            _do_reload()

