"""PyTorch backend — wraps HuggingFace transformers + PEFT for inference, training, and adapter fusion.

Drop-in replacement for MLXBackend. Uses bfloat16 on CUDA GPUs.
"""

import json
import os
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


class TorchBackend:
    """Unified interface to PyTorch model operations on CUDA."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_path = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

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
        """Determine which model to load: current (post-sleep) or base."""
        current = self.config.paths["current_model"]
        if os.path.exists(current) and os.listdir(current):
            return current
        return self.config.model["path"]

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text from a prompt string."""
        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

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

    def generate_stream(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """Generate text with streaming. Yields token strings incrementally."""
        max_tokens = max_tokens or self.config.model["max_tokens"]
        temperature = temperature or self.config.model["temperature"]
        top_p = top_p or self.config.model["top_p"]
        repetition_penalty = self.config.model.get("repetition_penalty", 1.1)

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

    def train_lora(self, data_path, adapter_path, epochs=None, learning_rate=None):
        """Run LoRA fine-tuning using PEFT.

        Args:
            data_path: Directory containing train.jsonl (and optionally valid.jsonl)
            adapter_path: Where to save the LoRA adapter
            epochs: Number of passes over the data
            learning_rate: Override config learning rate
        """
        epochs = epochs or self.config.lora["light_epochs"]
        learning_rate = learning_rate or self.config.lora["light_learning_rate"]
        lora_rank = self.config.lora["rank"]
        lora_alpha = self.config.lora["alpha"]
        lora_layers = self.config.lora["layers"]
        batch_size = self.config.lora["batch_size"]

        # Load training data
        train_file = Path(data_path) / "train.jsonl"
        train_texts = []
        if train_file.exists():
            with open(train_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        train_texts.append(item["text"])

        if not train_texts:
            return adapter_path

        # Determine target modules based on model architecture
        target_modules = self._get_target_modules(lora_layers)

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare quantized model for training if needed
        if self._quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        # Wrap model with LoRA
        self.model.train()
        peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"        LoRA params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

        # Training loop
        optimizer = torch.optim.AdamW(
            peft_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(train_texts) * epochs
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for text in train_texts:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=False,
                ).to(peft_model.device)

                # Causal LM: labels = input_ids (shifted internally by the model)
                outputs = peft_model(**tokens, labels=tokens["input_ids"])
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                step += 1

            avg_loss = epoch_loss / max(len(train_texts), 1)
            print(f"        Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({step}/{total_steps} steps)")

        # Save adapter
        os.makedirs(adapter_path, exist_ok=True)
        peft_model.save_pretrained(adapter_path)

        # Unwrap back to base model for inference
        self.model = peft_model.merge_and_unload()
        self.model.eval()

        return adapter_path

    def fuse_adapter(self, adapter_path, save_path):
        """Merge a LoRA adapter into the model and save."""
        os.makedirs(save_path, exist_ok=True)

        # Load adapter onto the current model
        peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        merged = peft_model.merge_and_unload()

        # Save merged model
        merged.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        return save_path

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
        with torch.no_grad():
            for i in range(start_layer, len(self.model.model.layers)):
                layer_out = self.model.model.layers[i](h)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            h = self.model.model.norm(h)
            logits = self.model.lm_head(h)
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

    def reload(self, model_path=None):
        """Reload model (e.g. after fusing a new adapter)."""
        # Free GPU memory
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        self.load(model_path)

    def _get_target_modules(self, num_layers):
        """Determine which layers to apply LoRA to.

        Targets attention projection matrices in the last N transformer layers.
        Works with Llama, Mistral, and similar architectures.
        """
        # Standard target modules for Llama-style models
        # PEFT handles layer selection via layers_to_transform
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Get total number of layers in the model
        if hasattr(self.model, "config"):
            total_layers = getattr(self.model.config, "num_hidden_layers", 32)
        else:
            total_layers = 32

        # We'll target the last N layers by using layers_to_transform in LoraConfig
        # But PEFT's LoraConfig handles this differently — we set target_modules
        # and it applies to all layers. For layer-specific LoRA, we'd need
        # layers_to_transform. For now, target all layers (standard approach on GPU
        # where memory isn't as constrained as on Apple Silicon).
        return target_modules
