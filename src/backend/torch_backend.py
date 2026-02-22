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
        )

        if quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # Reserve GPU headroom for inference/training; offload overflow to CPU
            load_kwargs["max_memory"] = {0: "68GiB", "cpu": "100GiB"}
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
